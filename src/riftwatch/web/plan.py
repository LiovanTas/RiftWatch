"""Your plan: one concrete thing to work on, chosen from Swing, with a target and a record of
whether it's getting better.

Every candidate is measured the same way: points of win chance lost per game (Swing), next to
the same figure for players in the same role, game by game (so a player who switches roles is
compared fairly). Candidates are the costly death habits (winprob.habits: deaths that cost an
objective, caught by three or more, ...) and the parts of Swing (kills, deaths, objectives,
structures, economy). The focus is the one costing the most *beyond what's typical*.

The target is realistic: the role's typical level when the gap is small, halfway there when
it's big, and for a habit it's also said in deaths per game. The trend compares the last 10
games with the 20 before, with a rough 95% margin, and says "no clear change yet" rather than
over-claim from a handful of games.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean
from typing import Any

import psycopg

from riftwatch.web.gamedata import POSITIONS, GameData

PARTS = {"kills": "Kills and assists", "deaths": "Deaths", "objectives": "Objectives",
         "structures": "Towers and inhibitors", "economy": "Gold and XP between fights"}
TIPS = {
    "while_dead": "Before you walk forward or take a fight, check what's up next: dragon, Baron, voidgrubs, a "
                  "tower in reach. If dying would hand it over, back off or wait for your team.",
    "outnumbered": "Keep track of where the enemy jungler and missing laners were last seen; when two or more "
                   "are missing, don't stand in fog on your own.",
    "held_gold": "Back when you have gold for an item part before risking a fight: dying with 1,500+ gold "
                 "wastes a whole back.",
    "deep": "Only go into the enemy jungle with vision and when your lanes can move first; ward before you enter.",
    "late": "After about 30 minutes a death keeps you out 40 seconds or more: stay with your team and don't "
            "take side-lane risks without vision.",
    "kills": "Look for fights you're ahead in: being there for picks and skirmishes is where kills and "
             "assists come from.",
    "deaths": "Die less: most costly deaths come from being caught alone, fighting outnumbered or staying "
              "too long.",
    "objectives": "Be at dragons, voidgrubs and Baron early: help set up vision and secure them.",
    "structures": "Turn won fights into towers and plates instead of resetting straight away.",
    "economy": "Out-earn your lane opponent between fights: last hits, plates and safe waves add up.",
}
TREND_LAST, TREND_BEFORE, MIN_TREND = 10, 20, 20
BLOCK = 5


def _habit_meta() -> dict[str, tuple[str, str]]:
    from riftwatch.winprob import habits

    return {key: (title, detail) for key, title, detail, _ in habits.HABITS}


def load_games(conn: psycopg.Connection, model, puuid: str, limit: int = 60) -> list[dict]:
    """The player's most recent games with stored Swing, oldest first: role, champion, each
    part of Swing and their deaths (only the moments where they died, read inside Postgres)."""
    from riftwatch.winprob import habits
    from riftwatch.winprob import swing as swing_calc

    ver = swing_calc.version(model)
    rows = conn.execute("""
        SELECT b.match_id, b.participant_id, b.game_start, b.position, b.champion_id, b.win,
               ps.kills, ps.deaths, ps.objectives, ps.structures, ps.economy,
               jsonb_path_query_array(s.data, '$.moments[*] ? (exists(@.death) && @.lost[*] == $pid)',
                                      jsonb_build_object('pid', b.participant_id))
          FROM participant_builds b
          JOIN player_swing ps USING (match_id, participant_id)
          JOIN match_swing s USING (match_id)
         WHERE b.puuid = %s AND ps.version = %s AND s.version = %s AND NOT b.remake
           AND b.queue_id IN (420, 440) AND b.position <> ''
         ORDER BY b.game_start DESC LIMIT %s""", (puuid, ver, ver, limit)).fetchall()
    out = []
    for mid, pid, t, role, champ, win, kills, deaths, objectives, structures, economy, moments in reversed(rows):
        found = habits.deaths({"moments": moments or []}, pid)
        habit = {key: [0, 0.0] for key in _habit_meta()}
        for _, cost, keys in found:
            for key in keys:
                habit[key][0] += 1
                habit[key][1] += cost
        out.append({"match_id": mid, "t": t, "role": role, "champion": champ, "win": win,
                    "parts": {"kills": kills, "deaths": deaths, "objectives": objectives,
                              "structures": structures, "economy": economy},
                    "habits": habit, "deaths": found})
    return out


def _cost(game: dict, key: str) -> float:
    """Points of win chance this candidate cost in one game (higher is worse)."""
    if key in PARTS:
        return -float(game["parts"][key])
    return float(game["habits"][key][1])


def _typical(game: dict, key: str, habit_norms: dict, part_norms: dict) -> float | None:
    """The same for a typical player in that game's role."""
    if key in PARTS:
        v = (part_norms.get(game["role"]) or {}).get(key)
        return -float(v) if v is not None else None
    h = (habit_norms.get(game["role"]) or {}).get(key)
    return float(h["per_game"]) if h else None


def _typical_count(game: dict, key: str, habit_norms: dict) -> float | None:
    role = habit_norms.get(game["role"]) or {}
    h = role.get(key)
    return float(role["deaths"]) * float(h["share"]) if h and "deaths" in role else None


def candidates(games: list[dict], habit_norms: dict, part_norms: dict) -> list[dict]:
    """Every habit and part, measured over the given games against each game's role, worst
    excess first."""
    meta = _habit_meta()
    out = []
    for key in [*meta, *PARTS]:
        pairs = [(_cost(g, key), _typical(g, key, habit_norms, part_norms)) for g in games]
        pairs = [(c, t) for c, t in pairs if t is not None]
        if len(pairs) < 3:
            continue
        mine, typical = fmean(c for c, _ in pairs), fmean(t for _, t in pairs)
        item = {"key": key, "kind": "habit" if key in meta else "part",
                "name": meta[key][0] if key in meta else PARTS[key],
                "detail": meta[key][1] if key in meta else "",
                "mine": mine, "typical": typical, "excess": mine - typical, "games": len(pairs),
                "tip": TIPS.get(key, "")}
        if key in meta:
            counts = [(g["habits"][key][0], _typical_count(g, key, habit_norms)) for g in games]
            item["count"] = fmean(c for c, _ in counts)
            known = [t for _, t in counts if t is not None]
            item["typical_count"] = fmean(known) if known else None
        out.append(item)
    return sorted(out, key=lambda c: -c["excess"])


def target(c: dict) -> dict:
    """A realistic goal: the role's typical level when the gap is small (2 points or less),
    halfway there when it's bigger; for a habit also in deaths per game."""
    goal = c["typical"] if c["excess"] <= 2 else c["mine"] - c["excess"] / 2
    out = {"cost": goal, "halfway": c["excess"] > 2}
    if c["kind"] == "habit" and c.get("count"):
        per_death = c["mine"] / c["count"] if c["count"] else 0
        out["count"] = goal / per_death if per_death > 0 else None
    return out


def trend(costs: list[float]) -> dict:
    """Last 10 games against the 20 before, with a rough 95% margin (normal approximation)."""
    if len(costs) < MIN_TREND:
        return {"verdict": "early", "games": len(costs)}
    last = costs[-TREND_LAST:]
    before = costs[-(TREND_LAST + TREND_BEFORE):-TREND_LAST]

    def var(xs):
        m = fmean(xs)
        return sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
    better = fmean(before) - fmean(last)                # cost fell = better
    margin = 1.96 * math.sqrt(var(last) / len(last) + var(before) / len(before))
    verdict = "better" if better > margin else "worse" if -better > margin else "steady"
    return {"verdict": verdict, "change": better, "margin": margin, "last": fmean(last),
            "before": fmean(before), "n_last": len(last), "n_before": len(before)}


def blocks(games: list[dict], key: str, habit_norms: dict, part_norms: dict, size: int = BLOCK) -> list[dict]:
    """The cost per game in blocks of ``size`` games, oldest first, with the typical line."""
    out = []
    start = len(games) % size                         # the oldest partial block is dropped
    for i in range(start, len(games), size):
        chunk = games[i:i + size]
        typ = [t for t in (_typical(g, key, habit_norms, part_norms) for g in chunk) if t is not None]
        out.append({"cost": fmean(_cost(g, key) for g in chunk), "typical": fmean(typ) if typ else None,
                    "from": chunk[0]["t"], "to": chunk[-1]["t"],
                    "count": fmean(g["habits"][key][0] for g in chunk) if key not in PARTS else None})
    return out


def chart(points: list[dict], goal: float, width: int = 640, height: int = 150) -> str | None:
    """Inline SVG: the cost per game for each block (solid), the role's typical (dashed) and
    the target (dotted). Higher is worse."""
    if len(points) < 2:
        return None
    values = [p["cost"] for p in points] + [p["typical"] for p in points if p["typical"] is not None] + [goal]
    lo, hi = min(values + [0.0]), max(values + [1.0])
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 40, width - 8, 8, height - 18

    def x(i):
        return left + (right - left) * i / (len(points) - 1)

    def y(v):
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    mine = " ".join(f"{x(i):.1f},{y(p['cost']):.1f}" for i, p in enumerate(points))
    typ = [(i, p["typical"]) for i, p in enumerate(points) if p["typical"] is not None]
    typical = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in typ)
    dots = "".join(f'<circle cx="{x(i):.1f}" cy="{y(p["cost"]):.1f}" r="3"/>' for i, p in enumerate(points))
    return (f'<svg viewBox="0 0 {width} {height}" class="plan-chart" role="img" '
            f'aria-label="Cost per game in blocks of {BLOCK} games, oldest first">'
            f'<line x1="{left}" x2="{right}" y1="{y(goal):.1f}" y2="{y(goal):.1f}" class="goal"/>'
            f'<text x="{left - 6}" y="{y(goal) + 4:.1f}" class="tick" text-anchor="end">{goal:.1f}</text>'
            + (f'<polyline points="{typical}" class="typical"/>' if typical else "")
            + f'<polyline points="{mine}" class="mine"/>{dots}'
            f'<text x="{left}" y="{height - 3}" class="tick">older</text>'
            f'<text x="{right}" y="{height - 3}" class="tick" text-anchor="end">latest</text></svg>')


def examples(games: list[dict], key: str, gd: GameData, as_path: str, n: int = 3) -> list[dict]:
    """The recent moments (habits) or games (parts) where it cost the most."""
    if key in PARTS:
        worst = sorted(games, key=lambda g: g["parts"][key])[:n]
        return [{"match_id": g["match_id"], "url": f"/games/{g['match_id']}?as={as_path}",
                 "champion": gd.champion(g["champion"]).name, "when": g["t"].strftime("%b %d"),
                 "text": f"{PARTS[key]} {g['parts'][key]:+.1f} in this game", "cost": round(-g["parts"][key], 1)}
                for g in worst if g["parts"][key] < 0]
    found = [(cost, m, g) for g in games for m, cost, keys in g["deaths"] if key in keys]
    found.sort(key=lambda x: -x[0])
    return [{"match_id": g["match_id"], "url": f"/games/{g['match_id']}?as={as_path}",
             "champion": gd.champion(g["champion"]).name, "when": g["t"].strftime("%b %d"),
             "clock": m.get("clock"), "text": m.get("text", ""), "cost": round(cost, 1)}
            for cost, m, g in found[:n]]


def build(conn: psycopg.Connection, gd: GameData, puuid: str, model, norms: dict, as_path: str,
          choice: str | None = None, recent: int = 20) -> dict[str, Any] | None:
    """The profile's "Your plan" panel."""
    if model is None:
        return None
    from riftwatch.web import swingview

    games = load_games(conn, model, puuid)
    if len(games) < 5:
        return None
    habit_norms = norms.get("habits") or {}
    part_norms = swingview.role_parts(conn, model) or {}
    ranked = candidates(games[-recent:], habit_norms, part_norms)
    if not ranked:
        return None
    options = [c for c in ranked if c["excess"] > 0.3][:3]
    focus = next((c for c in ranked if c["key"] == choice), None) or (options[0] if options else None)
    role = Counter(g["role"] for g in games[-recent:]).most_common(1)[0][0]
    out: dict[str, Any] = {"games": min(recent, len(games)), "season_games": len(games),
                           "role": POSITIONS.get(role, ""), "options": options, "focus": None}
    if focus is None:
        best = min(ranked, key=lambda c: c["excess"])
        out["clean"] = {"name": best["name"], "excess": round(best["excess"], 1)}
        return out
    goal = target(focus)
    pts = blocks(games, focus["key"], habit_norms, part_norms)
    out["focus"] = {**{k: (round(v, 1) if isinstance(v, float) else v) for k, v in focus.items()},
                    "target": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in goal.items()},
                    "trend": {k: (round(v, 1) if isinstance(v, float) else v)
                              for k, v in trend([_cost(g, focus["key"]) for g in games]).items()},
                    "chart": chart(pts, goal["cost"]), "blocks": len(pts),
                    "picked": bool(choice) and focus["key"] == choice and (not options or focus is not options[0]),
                    "examples": examples(games[-recent:], focus["key"], gd, as_path)}
    return out
