"""Is Swing a real measure of a player? The checks behind it, and the rating built on them.

A new statistic earns trust the way baseball's did: it should

  * repeat -- a player's average over half their games should match the other half
    (split-half reliability, Spearman-Brown corrected to the full number of games);
  * predict -- a player's Swing over their earlier games should say more about how often they
    win later than the familiar stats do, and add to what their past win rate already says;
  * not hinge on arbitrary choices -- moving each sharing rule (how much of an objective the
    team gets, how economy is split, how much a loss while dead lands on the dead) or swapping
    the win-probability model shouldn't reshuffle who added the most.

Every stat is measured against its role's average first (a support's damage, a jungler's
Swing), so a stat can't look reliable just because players stick to one role.

The **Swing rating** is a player's average Swing against their role's average, pulled towards
zero by how much their games vary next to how much players truly differ (empirical Bayes: the
true spread between players is the covariance of their two halves). It comes with a margin, so
"+3.1 ± 1.8 over 24 games" says how much to trust it.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import psycopg

from riftwatch.winprob import swing as swing_calc

VALIDATION_FILE = "validation.json"
MIN_GAMES = 20                   # players with at least this many games take part in the checks
METRICS = (("swing", "Swing"), ("score", "RiftWatch score (KDA, kill share, damage, gold, farm, vision)"),
           ("kda", "KDA"), ("kp", "Kill participation"), ("dmg", "Damage per minute"),
           ("gold", "Gold per minute"), ("win", "Win rate"))


def _rows(conn: psycopg.Connection, model) -> list[dict]:
    """Every player-game with Swing: the player, when, the role, the result and each stat."""
    from riftwatch.web.views import _scores

    ver = swing_calc.version(model)
    games: dict[str, list[dict]] = defaultdict(list)
    cols = ("match_id", "participant_id", "puuid", "game_start", "position", "win", "kills", "deaths",
            "assists", "team_kills", "damage", "gold", "cs", "vision_score", "duration_s", "total")
    for r in conn.execute("""
            SELECT b.match_id, b.participant_id, b.puuid, b.game_start, b.position, b.win, b.kills, b.deaths,
                   b.assists, b.team_kills, b.damage, b.gold, b.cs, b.vision_score, b.duration_s, ps.total
              FROM player_swing ps JOIN participant_builds b USING (match_id, participant_id)
             WHERE ps.version = %s AND NOT b.remake AND b.queue_id IN (420, 440)""", (ver,)):
        games[r[0]].append(dict(zip(cols, r, strict=True)))
    out = []
    for players in games.values():
        if len(players) != 10:
            continue
        scores = _scores(players)
        for p in players:
            minutes = max(1.0, p["duration_s"] / 60)
            out.append({"puuid": p["puuid"], "match_id": p["match_id"], "t": p["game_start"], "role": p["position"],
                        "swing": float(p["total"]), "score": scores[p["participant_id"]],
                        "kda": min(20.0, (p["kills"] + p["assists"]) / max(1, p["deaths"])),
                        "kp": (p["kills"] + p["assists"]) / max(1, p["team_kills"]),
                        "dmg": p["damage"] / minutes, "gold": p["gold"] / minutes, "win": 1.0 if p["win"] else 0.0})
    return out


def _role_adjust(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Each stat minus its role's average, in place; returns the role averages."""
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for key, _ in METRICS:
            sums[r["role"]][key].append(r[key])
    means = {role: {k: fmean(v) for k, v in d.items()} for role, d in sums.items()}
    for r in rows:
        for key, _ in METRICS:
            r[f"{key}_adj"] = r[key] - (means[r["role"]][key] if key != "win" else 0.0)
    return means


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _ci(r: float, n: int) -> tuple[float, float]:
    """95% interval for a correlation (Fisher z)."""
    if n < 4:
        return (float("nan"), float("nan"))
    z, se = math.atanh(max(-0.999, min(0.999, r))), 1 / math.sqrt(n - 3)
    return (round(math.tanh(z - 1.96 * se), 3), round(math.tanh(z + 1.96 * se), 3))


def _by_player(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["puuid"]].append(r)
    for games in out.values():
        games.sort(key=lambda r: r["t"])
    return {p: g for p, g in out.items() if len(g) >= MIN_GAMES}


def reliability_and_prediction(rows: list[dict], metrics=METRICS) -> dict[str, Any]:
    players = _by_player(rows)
    out: dict[str, Any] = {"players": len(players), "metrics": []}
    later_wins, past_wins, past_swing = [], [], []
    for key, name in metrics:
        col = f"{key}_adj"
        odd, even, early, late_win = [], [], [], []
        for games in players.values():
            values = [g[col] for g in games]
            odd.append(fmean(values[0::2]))
            even.append(fmean(values[1::2]))
            half = len(games) // 2
            early.append(fmean(values[:half]))
            late_win.append(fmean(g["win"] for g in games[half:]))
        r_half = _corr(odd, even)
        rel = 2 * r_half / (1 + r_half) if r_half > -1 else 0.0         # Spearman-Brown, all games
        r_pred = _corr(early, late_win)
        out["metrics"].append({"key": key, "name": name, "split_half": round(r_half, 3),
                               "reliability": round(rel, 3), "reliability_ci": _ci(r_half, len(odd)),
                               "predicts_later_wins": round(r_pred, 3), "predicts_ci": _ci(r_pred, len(early))})
        if key == "win":
            past_wins, later_wins = early, late_win
        elif not past_wins:
            past_wins = [fmean(g["win"] for g in games[:len(games) // 2]) for games in players.values()]
            later_wins = late_win
        if key == "swing":
            past_swing = early
    # What Swing adds once past win rate is known: the partial correlation.
    if past_wins and past_swing:
        r_sw, r_wr, r_both = _corr(past_swing, later_wins), _corr(past_wins, later_wins), _corr(past_swing, past_wins)
        denom = math.sqrt(max(1e-9, (1 - r_wr ** 2) * (1 - r_both ** 2)))
        partial = (r_sw - r_wr * r_both) / denom
        out["swing_beyond_win_rate"] = {"partial_r": round(partial, 3), "ci": _ci(partial, len(past_swing) - 1)}
    # The most any stat could correlate with later win rate: the later half's win rate is
    # itself mostly luck, and a correlation can't exceed the square root of the outcome's
    # reliability (the correlation of two halves of those later games, stepped up).
    later_halves = []
    for games in players.values():
        later = [g["win"] for g in games[len(games) // 2:]]
        later_halves.append((fmean(later[0::2]), fmean(later[1::2])))
    if len(later_halves) > 3:
        r_later = _corr([a for a, _ in later_halves], [b for _, b in later_halves])
        rel_later = max(0.0, 2 * r_later / (1 + r_later)) if r_later > -1 else 0.0
        out["prediction_ceiling"] = round(math.sqrt(rel_later), 3)
    return out


def rating_model(rows: list[dict]) -> dict[str, float]:
    """The spread of true player levels (tau) and of single games (sigma) in role-adjusted
    Swing, for shrinking a player's average: the covariance of two halves of the same players'
    games estimates tau squared (the noise in each half is independent)."""
    players = _by_player(rows)
    odd, even, within = [], [], []
    for games in players.values():
        values = [g["swing_adj"] for g in games]
        odd.append(fmean(values[0::2]))
        even.append(fmean(values[1::2]))
        within.append(float(np.var(values, ddof=1)))
    if len(odd) < 3:                    # too few players to tell: a cautious default (k = 50)
        return {"tau": 2.0, "sigma": 14.14, "k": 50.0}
    tau2 = max(float(np.cov(odd, even)[0, 1]), 0.25)
    sigma2 = fmean(within)
    return {"tau": round(math.sqrt(tau2), 2), "sigma": round(math.sqrt(sigma2), 2), "k": round(sigma2 / tau2, 1)}


def rating(mean_adj: float, games: int, model: dict[str, float]) -> dict[str, float]:
    """A player's Swing rating: their average against their role, pulled towards zero, with a
    one-standard-error margin."""
    tau2, sigma2 = model["tau"] ** 2, model["sigma"] ** 2
    weight = tau2 / (tau2 + sigma2 / max(1, games))
    sd = math.sqrt(1 / (1 / tau2 + games / sigma2))
    return {"value": round(mean_adj * weight, 1), "sd": round(sd, 1), "raw": round(mean_adj, 1), "games": games}


# -- the sharing rules ------------------------------------------------------------------------

# Rule sets compared on every stored game (fixed in advance, not tuned): the fixed shares
# Swing started with, each measured share on its own, and both.
RULE_SETS = (("fixed", swing_calc.FIXED_RULES, "Fixed shares: killer half, assisters half; economy gains by gold earned, losses evenly"),
             ("kills", swing_calc.Rules(economy_split="earned"), "Kills shared by damage dealt to the victim"),
             ("economy", swing_calc.Rules(kill_split="half"), "Economy shared by gold and XP against the lane opponent"),
             ("measured", swing_calc.RULES, "Both measured (used)"))
# Each remaining choice moved a long way either side of the value used.
SENSITIVITY = (("team_share", 0.25, "objectives: a quarter to the whole team (used: half)"),
               ("team_share", 0.75, "objectives: three quarters to the whole team"),
               ("even_economy", 0.25, "economy: a quarter shared evenly (used: half)"),
               ("even_economy", 0.75, "economy: three quarters shared evenly"),
               ("dead_share", 0.25, "losses while dead: a quarter on the dead (used: half)"),
               ("dead_share", 0.75, "losses while dead: three quarters on the dead"))


def rules_and_sensitivity(conn: psycopg.Connection, model, log=print) -> dict[str, Any]:
    """Share every stored game again under each rule set and each sensitivity variant (from
    the facts stored with it, no model needed), then check each rule set the same way as the
    stats, and how far each variant moves Swing from the rules used."""
    from dataclasses import replace

    ver = swing_calc.version(model)
    who: dict[str, dict[int, tuple]] = defaultdict(dict)
    for match_id, pid, puuid, t, role, win in conn.execute(
            """SELECT b.match_id, b.participant_id, b.puuid, b.game_start, b.position, b.win
                 FROM participant_builds b JOIN match_swing s USING (match_id)
                WHERE s.version = %s AND NOT b.remake AND b.queue_id IN (420, 440)""", (ver,)):
        who[match_id][pid] = (puuid, t, role, 1.0 if win else 0.0)
    variants = [(key, rules) for key, rules, _ in RULE_SETS] + [
        (f"{field}={value}", replace(swing_calc.RULES, **{field: value})) for field, value, _ in SENSITIVITY]
    totals: dict[str, list[float]] = defaultdict(list)
    base: list[dict] = []
    with conn.transaction(), conn.cursor(name="swing_facts") as cur:
        cur.itersize = 200
        cur.execute("SELECT match_id, data->'facts' FROM match_swing WHERE version = %s AND data ? 'facts'", (ver,))
        for match_id, facts in cur:
            people = who.get(match_id)
            if not facts or not people or len(people) != 10:
                continue
            for key, rules in variants:
                credit, _, _ = swing_calc.credit_of(facts, rules)
                totals[key].extend(100 * sum(credit[p].values()) for p in range(1, 11))
            for p in range(1, 11):
                puuid, t, role, win = people[p]
                base.append({"puuid": puuid, "t": t, "role": role, "win": win})
    log(f"  {len(base):,} player-games shared under {len(variants)} sets of rules")
    out: dict[str, Any] = {"player_games": len(base), "rule_sets": [], "sensitivity": []}
    if not base:
        return out
    for key, _, label in RULE_SETS:
        rows = [{**b, "swing": v} for b, v in zip(base, totals[key], strict=True)]
        _role_adjust_one(rows, "swing")
        check = reliability_and_prediction(rows, (("swing", label),))
        m = check["metrics"][0]
        out["rule_sets"].append({"key": key, "label": label, "reliability": m["reliability"],
                                 "reliability_ci": m["reliability_ci"], "predicts_later_wins": m["predicts_later_wins"],
                                 "predicts_ci": m["predicts_ci"], "players": check["players"],
                                 "sd_per_game": round(float(np.std(totals[key])), 1)})
        log(f"    {key:<9} reliability {m['reliability']:.2f}  predicts later wins r {m['predicts_later_wins']:.3f}")
    used = np.asarray(totals["measured"])
    for field, value, label in SENSITIVITY:
        alt = np.asarray(totals[f"{field}={value}"])
        out["sensitivity"].append({"rule": label, "r": round(_corr(used, alt), 4),
                                   "moved": round(float(np.mean(np.abs(used - alt))), 2)})
    fixed = np.asarray(totals["fixed"])
    out["measured_vs_fixed"] = {"r": round(_corr(used, fixed), 4), "moved": round(float(np.mean(np.abs(used - fixed))), 2)}
    return out


def _role_adjust_one(rows: list[dict], key: str) -> None:
    sums: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        sums[r["role"]].append(r[key])
    means = {role: fmean(v) for role, v in sums.items()}
    for r in rows:
        r[f"{key}_adj"] = r[key] - means[r["role"]]


def _sample(conn, n: int) -> list[tuple]:
    from riftwatch.winprob.model import GAME_SQL

    ids = [r[0] for r in conn.execute(
        "SELECT match_id FROM match_swing ORDER BY md5(match_id) LIMIT %s", (n,))]
    return conn.execute(GAME_SQL + " WHERE t.match_id = ANY(%s)", (ids,)).fetchall()


def _totals(model, games) -> list[float]:
    from riftwatch.winprob.model import players, positions

    out = []
    for _, frames, events, rows in games:
        champions, purchases = players(rows)
        d = swing_calc.game_swing(model, frames, events, None, champions, purchases, positions(rows))
        out.extend(d["players"][str(p)]["total"] if d else float("nan") for p in range(1, 11))
    return out


def model_robustness(conn, model, other_model, games: int = 200, log=print) -> dict[str, Any]:
    """How much Swing moves with a different win-probability model on the same games."""
    sample = _sample(conn, games)
    base, alt = np.asarray(_totals(model, sample)), np.asarray(_totals(other_model, sample))
    m = ~np.isnan(base) & ~np.isnan(alt)
    out = {"games": len(sample), "r": round(_corr(base[m], alt[m]), 3),
           "moved": round(float(np.mean(np.abs(base[m] - alt[m]))), 2),
           "label": "the first network (game state only, isotonic calibration)"}
    log(f"    other model: r {out['r']:.3f}")
    return out


def build(conn: psycopg.Connection, model, models_dir: Path | str, other_model=None, log=print) -> dict[str, Any]:
    """Run every check and save them (validation.json) with the rating's parameters."""
    started = time.monotonic()
    rows = _rows(conn, model)
    log(f"  {len(rows):,} player-games")
    means = _role_adjust(rows)
    result: dict[str, Any] = {"version": swing_calc.version(model), "player_games": len(rows),
                              "min_games": MIN_GAMES, **reliability_and_prediction(rows)}
    for m in result["metrics"]:
        log(f"  {m['key']:<6} reliability {m['reliability']:.2f}  predicts later wins r {m['predicts_later_wins']:.2f}")
    result["rating"] = rating_model(rows)
    result["role_means"] = {role: round(v["swing"], 2) for role, v in means.items() if role}
    log(f"  rating: true spread ±{result['rating']['tau']}, one game ±{result['rating']['sigma']}, "
        f"pulled as if {result['rating']['k']} more average games")
    log("  the sharing rules:")
    result["rules"] = rules_and_sensitivity(conn, model, log=log)
    if other_model is not None:
        result["other_model"] = model_robustness(conn, model, other_model, log=log)
    result["seconds"] = round(time.monotonic() - started)
    path = Path(models_dir) / "winprob" / VALIDATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _cache.clear()
    return result


_cache: dict[str, Any] = {}


def load(models_dir: Path | str) -> dict[str, Any] | None:
    path = Path(models_dir) / "winprob" / VALIDATION_FILE
    key = str(path)
    if key not in _cache:
        _cache[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return _cache[key]


def _stable_hash(text: str) -> int:
    return hashlib.md5(text.encode()).digest()[0]
