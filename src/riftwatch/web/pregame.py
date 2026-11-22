"""Pre-game win prediction: each team's chance before the game starts, and why.

Three parts, added in log-odds (logit) and turned back into a chance:

  1. draft -- the win-probability model's chance at minute 0 for these two line-ups: the
     state Swing starts every game from (everyone alive, no gold, kills, objectives or items
     yet), with each team's classes and magic-damage share from the model's catalog and the ten
     champions. The model learned this from stored games, but a solo-queue draft predicts
     little (its AUC in the first two minutes is about 0.55, read from metrics.json), so the
     page says so and the draft usually moves the chance by a point or two.
  2. players -- each player's Swing rating (web.swingview.player_rating: their average Swing
     against their role, pulled towards zero by empirical Bayes, with a margin). A rating is
     percentage points of win chance a player adds per game against an average lobby, so a
     team whose ratings add up to D more than the other's is about D points more likely to
     win: 0.5 + D/100, or 0.04 x D in log-odds near even (the slope of the logit at 50%).
     Players RiftWatch hasn't seen count as average (0) with the spread of true ratings
     (tau) as their margin, and the margins add up in quadrature into "how sure".
  3. comfort -- how much a player's experience on this champion helps or hurts, from the
     measured experience curve (web.draft.comp_tables: stored players' win rate on their
     first, second-third, ... game on a champion), against the overall rate. This partly
     overlaps with the rating (a rating averages a player's games, comfortable or not), so
     each player's comfort is halved before it's added.

Backtest (``backtest``): every stored game is predicted from the players' ratings computed only
from their games *before* it, and scored against the 50% you'd guess without them.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg

from riftwatch.web.gamedata import GameData

PER_POINT = 0.04               # log-odds per percentage point near 50%
COMFORT_WEIGHT = 0.5           # the share of the experience effect not already in the rating
DEFAULT_TAU = 3.0              # spread of true ratings when validation.json isn't there
BACKTEST_FILE = "pregame.json"


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# -- the draft part -------------------------------------------------------------------------

def draft_chance(model, blue: list[int], red: list[int]) -> float | None:
    """Blue's chance from the line-ups alone: the model at minute 0, before anything happens."""
    catalog = getattr(model, "catalog", None)
    if model is None or catalog is None or not getattr(model, "reads_items", False):
        return None
    from riftwatch.winprob.state import BASE_FEATURES, CLASSES

    features = getattr(model, "features", BASE_FEATURES)

    def side(own: list[int], enemy: list[int]) -> dict[str, Any]:
        s: dict[str, Any] = dict.fromkeys(features, 0.0)        # anything the model reads, at zero
        mine, theirs = catalog.comp(own), catalog.comp(enemy)
        values = {"alive_own": 5.0, "alive_enemy": 5.0,          # nobody's dead at the start
                  "dragon_in": 5.0, "elder_in": 6.0, "baron_in": 6.0,   # nothing's up yet
                  "magic_share_own": mine["magic_share"], "magic_share_enemy": theirs["magic_share"]}
        for cls in CLASSES:
            values[f"{cls.lower()}s_own"], values[f"{cls.lower()}s_enemy"] = mine[cls], theirs[cls]
        for key, value in values.items():
            if key in s:
                s[key] = value
        s["_own"] = (list(own) + [0] * 5)[:5]
        s["_enemy"] = (list(enemy) + [0] * 5)[:5]
        return s
    return float(model.blue([side(blue, red)], [side(red, blue)])[0])


def draft_auc(models_dir) -> float | None:
    """How well the chosen model's chances in the first two minutes rank the winners."""
    from riftwatch.winprob import model as wp

    m = wp.metrics(models_dir) or {}
    chosen = next((v for v in m.get("variants", []) if v.get("name") == m.get("chosen")), None)
    return (chosen or {}).get("by_minute", {}).get("0-2")


# -- the players part ------------------------------------------------------------------------

def player_term(conn: psycopg.Connection, model, validation: dict | None, puuid: str | None) -> dict[str, Any]:
    """A player's rating and its margin; unknown players are average with a wide margin."""
    from riftwatch.web import swingview

    tau = (validation or {}).get("rating", {}).get("tau", DEFAULT_TAU)
    found = swingview.player_rating(conn, model, puuid, validation) if puuid and model is not None else None
    if not found:
        return {"value": 0.0, "sd": tau, "games": 0, "known": False, "role": ""}
    return {**found, "known": True}


def comfort_term(comp: dict | None, games: int | None) -> tuple[float, str]:
    """Log-odds from this many earlier games on the champion against the overall rate, halved
    (the rating already holds part of it), and the experience bucket it falls in."""
    if not comp or games is None:
        return 0.0, ""
    from riftwatch.web.draft import _comfort_rate

    rate, label = _comfort_rate(comp, games)
    return COMFORT_WEIGHT * (logit(rate) - logit(comp["overall"])), label


# -- putting it together -----------------------------------------------------------------------

def predict(players: list[dict], draft: float | None, comp: dict | None, fit: dict | None = None,
            sided: bool = True) -> dict[str, Any]:
    """``players``: dicts with team (100/200), name, champion, icon, rating {value, sd, known,
    games}, champion_games. ``fit``: the backtest's calibration (how much the ratings and the
    draft are really worth, and blue side's edge when ``sided``). Returns blue's chance, each
    part and the most and least helpful players."""
    fit = {"blue": 0.0, "players": 1.0, "draft": 1.0, **(fit or {})}
    teams = {100: [p for p in players if p["team"] == 100], 200: [p for p in players if p["team"] == 200]}
    rating_diff = (sum(p["rating"]["value"] for p in teams[100]) - sum(p["rating"]["value"] for p in teams[200]))
    sd = math.sqrt(sum(p["rating"]["sd"] ** 2 for p in players)) if players else 0.0
    comfort = 0.0
    for p in players:
        c, label = comfort_term(comp, p.get("champion_games"))
        p["comfort"], p["comfort_label"] = round(100 * c / 4, 1), label     # points near 50%
        comfort += c if p["team"] == 100 else -c
    draft_logit = (fit["blue"] if sided else 0.0) + (fit["draft"] * logit(draft) if draft is not None else 0.0)
    per_point = fit["players"] * PER_POINT
    total = draft_logit + per_point * rating_diff + comfort
    blue = sigmoid(total)
    low, high = sigmoid(total - per_point * sd), sigmoid(total + per_point * sd)
    ranked = sorted((p for p in players if p["rating"]["known"]), key=lambda p: -p["rating"]["value"])
    known = sum(1 for p in players if p["rating"]["known"])
    return {
        "blue": round(100 * blue, 1), "red": round(100 - 100 * blue, 1),
        "low": round(100 * low, 1), "high": round(100 * high, 1),
        "draft": round(100 * draft, 1) if draft is not None else None,
        "parts": {"draft": round(100 * (sigmoid(draft_logit) - 0.5), 1),
                  "players": round(100 * (sigmoid(per_point * rating_diff) - 0.5), 1),
                  "comfort": round(100 * (sigmoid(comfort) - 0.5), 1)},
        "rating_diff": round(rating_diff, 1), "sd": round(sd, 1), "known": known, "of": len(players),
        "best": ranked[:2], "worst": [p for p in ranked[::-1][:2] if p not in ranked[:2]],
        "sure": ("fairly sure" if known >= 8 and sd < 12 else "rough" if known >= 4 else "a guess: few players known"),
    }


def _context(conn, gd: GameData, model, models_dir) -> tuple[dict | None, dict | None, float | None]:
    from riftwatch.web import draft as draft_mod
    from riftwatch.winprob import validate

    validation = validate.load(models_dir) if models_dir else None
    comp = None
    try:
        comp = draft_mod.comp_tables(conn, draft_mod.champion_classes(gd, model))
    except Exception:          # a missing table only drops the comfort part
        comp = None
    return validation, comp, draft_auc(models_dir) if models_dir else None


def live(conn: psycopg.Connection, gd: GameData, report: dict, model, models_dir,
         me: str | None = None) -> dict[str, Any] | None:
    """The prediction for a scouted live game (the scout report's ten players)."""
    if model is None or not report or not report.get("players"):
        return None
    validation, comp, auc = _context(conn, gd, model, models_dir)
    players, lineups = [], {100: [], 200: []}
    for p in report["players"]:
        champ = gd.champion_by_slug(p.get("champion") or "")
        team = p.get("team_id")
        if team not in (100, 200):
            continue
        if champ:
            lineups[team].append(champ.id)
        players.append({"team": team, "name": p.get("riot_id") or "Unknown", "me": p.get("puuid") == me,
                        "champion": champ.name if champ else (p.get("champion") or ""),
                        "icon": gd.champion_icon(champ.id) if champ else "",
                        "champion_games": p.get("champion_games") or 0,
                        "rating": player_term(conn, model, validation, p.get("puuid"))})
    draft = draft_chance(model, lineups[100], lineups[200]) if lineups[100] and lineups[200] else None
    out = predict(players, draft, comp, (load_backtest(models_dir) or {}).get("fit") if models_dir else None)
    mine = next((p["team"] for p in players if p["me"]), None)
    out.update(kind="live", draft_auc=auc, my_team=mine, tau=(validation or {}).get("rating", {}).get("tau"))
    return out


def lobby(conn: psycopg.Connection, gd: GameData, ids: list[tuple[str, str]], model, models_dir) -> dict[str, Any] | None:
    """A pasted lobby (your own team, no champions yet): how much its players add against an
    average team, from their ratings only."""
    if model is None or not ids:
        return None
    validation, _, _ = _context(conn, gd, model, models_dir)
    players = []
    for name, tag in ids:
        puuid = _resolve(conn, name, tag)
        players.append({"team": 100, "name": f"{name}#{tag}", "me": False, "champion": "", "icon": "",
                        "champion_games": None, "rating": player_term(conn, model, validation, puuid)})
    if not any(p["rating"]["known"] for p in players):
        return None
    out = predict(players, None, None, (load_backtest(models_dir) or {}).get("fit") if models_dir else None, sided=False)
    out.update(kind="lobby", tau=(validation or {}).get("rating", {}).get("tau"))
    return out


def _resolve(conn, name: str, tag: str) -> str | None:
    from riftwatch.db import repo

    account = repo.find_account(conn, name, tag)
    if account is not None:
        return account["puuid"]
    seen = conn.execute("""SELECT puuid FROM participant_builds WHERE lower(riot_name) = lower(%s)
                            AND lower(riot_tag) = lower(%s) ORDER BY game_start DESC LIMIT 1""", (name, tag)).fetchone()
    return seen[0] if seen else None


# -- does it work? -----------------------------------------------------------------------------

def backtest(conn: psycopg.Connection, model, validation: dict, min_prior: int = 5) -> dict[str, Any]:
    """Predict every stored game from ratings built only from each player's *earlier* games
    (role-adjusted, shrunk with the rating's tau and sigma), and score it against 50%.
    Games count when at least one player has ``min_prior`` earlier games."""
    from riftwatch.winprob import swing

    ver = swing.version(model)
    means = validation.get("role_means") or {}
    tau2, sigma2 = validation["rating"]["tau"] ** 2, validation["rating"]["sigma"] ** 2
    rows = conn.execute("""SELECT b.match_id, b.puuid, b.team_id, b.position, b.game_start, b.win, ps.total
                             FROM player_swing ps JOIN participant_builds b USING (match_id, participant_id)
                            WHERE ps.version = %s AND NOT b.remake AND b.queue_id IN (420, 440)
                            ORDER BY b.puuid, b.game_start""", (ver,)).fetchall()
    draft_start = {m: s for m, s in conn.execute(
        "SELECT match_id, (data->>'start')::float FROM match_swing WHERE version = %s", (ver,))}
    games: dict[str, dict] = defaultdict(lambda: {100: 0.0, 200: 0.0, "known": 0, "sd2": 0.0, "blue_win": None})
    last, total, n = None, 0.0, 0
    for match_id, puuid, team, position, _, win, swing_total in rows:
        if puuid != last:
            last, total, n = puuid, 0.0, 0
        g = games[match_id]
        if n >= min_prior:
            mean = total / n
            g[team] += mean * n * tau2 / (n * tau2 + sigma2)
            g["known"] += 1
        if team == 100:
            g["blue_win"] = bool(win)
        total += swing_total - means.get(position, 0.0)
        n += 1
    scored = [(match_id, g) for match_id, g in games.items() if g["known"] and g["blue_win"] is not None]
    if len(scored) < 200:
        return {"games": len(scored), "enough": False}

    def evaluate(preds: list[float], ys: list[float]) -> dict[str, float]:
        loss = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(preds, ys, strict=True)) / len(ys)
        return {"log_loss": round(loss, 4), "auc": round(_auc(preds, ys), 4)}
    ys = [1.0 if g["blue_win"] else 0.0 for _, g in scored]
    players = [min(max(sigmoid(PER_POINT * (g[100] - g[200])), 1e-4), 1 - 1e-4) for _, g in scored]
    with_draft = [min(max(sigmoid(PER_POINT * (g[100] - g[200]) + logit((draft_start.get(m) or 50) / 100)), 1e-4), 1 - 1e-4)
                  for m, g in scored]

    def confident(preds: list[float]) -> dict[str, float]:
        strong = [(p, y) for p, y in zip(preds, ys, strict=True) if abs(p - 0.5) >= 0.05]
        return {"games": len(strong),
                "predicted": round(100 * sum(p if p > 0.5 else 1 - p for p, _ in strong) / max(1, len(strong)), 1),
                "right": round(100 * sum((p > 0.5) == (y == 1.0) for p, y in strong) / max(1, len(strong)), 1)}

    # Calibrated: how much the ratings and the draft are really worth, and blue side's edge,
    # fitted by logistic regression and scored on the half of the games it wasn't fitted on.
    x = [[1.0, PER_POINT * (g[100] - g[200]), logit((draft_start.get(m) or 50) / 100)] for m, g in scored]
    halves = [int(hashlib.md5(m.encode()).hexdigest(), 16) % 2 for m, _ in scored]
    held_out = [0.5] * len(scored)
    for h in (0, 1):
        w = _fit_logistic([r for r, k in zip(x, halves, strict=True) if k != h],
                          [y for y, k in zip(ys, halves, strict=True) if k != h])
        for i, k in enumerate(halves):
            if k == h:
                held_out[i] = min(max(sigmoid(sum(a * b for a, b in zip(w, x[i], strict=True))), 1e-4), 1 - 1e-4)
    w = _fit_logistic(x, ys)
    return {"games": len(scored), "enough": True, "min_prior": min_prior,
            "known_per_game": round(sum(g["known"] for _, g in scored) / len(scored), 2),
            "coin_flip": round(math.log(2), 4), "players": evaluate(players, ys),
            "players_and_draft": evaluate(with_draft, ys), "confident": confident(players),
            "calibrated": {**evaluate(held_out, ys), "confident": confident(held_out)},
            "fit": {"blue": round(w[0], 4), "players": round(w[1], 4), "draft": round(w[2], 4)}}


def _fit_logistic(x: list[list[float]], y: list[float], steps: int = 25) -> list[float]:
    """Logistic regression by Newton's method (a tiny ridge keeps it stable)."""
    import numpy as np

    X, Y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w)))
        hess = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, X.T @ (Y - p))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return [float(v) for v in w]


def _auc(preds: list[float], ys: list[float]) -> float:
    """Probability a random winner got a higher prediction than a random loser (ties half)."""
    order = sorted(range(len(preds)), key=lambda i: preds[i])
    ranks = [0.0] * len(preds)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and preds[order[j + 1]] == preds[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(1 for y in ys if y == 1.0)
    neg = len(ys) - pos
    if not pos or not neg:
        return 0.5
    return (sum(r for r, y in zip(ranks, ys, strict=True) if y == 1.0) - pos * (pos + 1) / 2) / (pos * neg)


def save_backtest(result: dict, models_dir) -> None:
    path = Path(models_dir) / "winprob" / BACKTEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _cache.clear()


_cache: dict[str, Any] = {}


def load_backtest(models_dir) -> dict | None:
    path = Path(models_dir) / "winprob" / BACKTEST_FILE
    key = str(path)
    if key not in _cache:
        _cache[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return _cache[key]


if __name__ == "__main__":       # python -m riftwatch.web.pregame: run and save the backtest
    from dotenv import load_dotenv

    from riftwatch.config import Settings
    from riftwatch.winprob import model as wp
    from riftwatch.winprob import validate

    load_dotenv()
    settings = Settings.from_env()
    models_dir = settings.models_dir
    with psycopg.connect(settings.database_url, autocommit=True) as c:
        result = backtest(c, wp.load(models_dir), validate.load(models_dir))
    save_backtest(result, models_dir)
    print(json.dumps(result, indent=2))
