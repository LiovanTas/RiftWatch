"""Were the plays worth the farm? Every play a laner or jungler made, by kind: what it moved
their team's chance (their own share of the play, before farm), the farm it cost them
(farm.py: CS given up against their own pace, as gold, XP and points of win chance) and what
was left, next to the same for players in their role.

Kinds of play:

  * roams and ganks -- before 14:00, a kill you took part in away from your own lane (a
    laner), or in a lane (a jungler)
  * other fights -- every other kill you took part in
  * objectives -- dragons, Baron, heralds, voidgrubs, Atakhan you took part in
  * towers -- towers and inhibitors you took part in
  * deaths -- your deaths: the farm missed while dead and walking back, on top of the death

Shares come from Swing's own sharing rules (swing.credit_of) with the farm charge left out,
so "gained" is exactly the play's share. "Farm" is the farm the play cost beyond what a
typical play like it costs (the part Swing charges it: the model already expects the rest),
so "net" = gained - farm is the play's Swing. "Worth" is the whole farm given up, valued by
the model, for reading alongside the gold and XP.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from riftwatch.winprob import farm as farm_calc
from riftwatch.winprob import swing

KINDS = (("roams", "Roams and ganks", "kills you took part in before 14:00 away from your lane (a jungler: in a lane)"),
         ("fights", "Other fights", "every other kill you took part in"),
         ("objectives", "Objectives", "dragons, Baron, heralds, voidgrubs and Atakhan you were at"),
         ("towers", "Towers", "towers and inhibitors you were at"),
         ("deaths", "Deaths", "the farm missed while dead and walking back, on top of the death itself"))
LANE_NAMES = {"TOP": "top lane", "MIDDLE": "mid lane", "BOTTOM": "bot lane"}
FILE = "trades.json"


def kind_of(m: dict, pid: int, role: str) -> str | None:
    if pid in (m.get("lost") or []):
        return "deaths"
    if pid not in (m.get("players") or []):
        return None
    part = m.get("part")
    if part == "objectives":
        return "objectives"
    if part == "structures":
        return "towers"
    if part != "kills":
        return None
    zone = (m.get("death") or {}).get("zone", "")
    early = m["t"] < farm_calc.LANING_ENDS * 60_000
    if early and role == "JUNGLE" and zone in LANE_NAMES.values():
        return "roams"
    if early and role in LANE_NAMES and zone and zone != LANE_NAMES[role]:
        return "roams"
    return "fights"


def game_trades(data: dict, pid: int, role: str) -> dict[str, dict[str, float]]:
    """One game: per kind, plays, points gained, points of farm, gold and XP given up."""
    facts, moments = data.get("facts") or {}, data.get("moments") or []
    if not facts.get("events") or role not in farm_calc.ROLES:
        return {}
    _, plays, _ = swing.credit_of(facts, replace(swing.RULES, farm_cost=False))
    gained: dict[int, float] = {}
    for share, n in plays[pid]:
        gained[n] = gained.get(n, 0.0) + share
    out: dict[str, dict[str, float]] = {}
    for n, (f, m) in enumerate(zip(facts["events"], moments, strict=False)):
        if f.get("kind") == "items" or n >= len(moments):
            continue
        kind = kind_of(m, pid, role)
        if kind is None:
            continue
        cost = (f.get("farm") or {}).get(str(pid))
        row = out.setdefault(kind, {"n": 0, "gained": 0.0, "farm": 0.0, "worth": 0.0, "gold": 0.0, "xp": 0.0})
        row["n"] += 1
        row["gained"] += 100 * gained.get(n, 0.0)
        if cost:
            row["farm"] += 100 * cost[0]
            row["worth"] += 100 * (cost[5] if len(cost) > 5 else cost[0])
            row["gold"] += cost[2]
            row["xp"] += cost[3]
    return out


def summarise(games: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Per kind over games: plays a game, and gained, farm, net, gold and XP per play."""
    out = {}
    for kind, *_ in KINDS:
        rows = [g[kind] for g in games if kind in g]
        n = sum(r["n"] for r in rows)
        if not n:
            continue
        gained, cost = sum(r["gained"] for r in rows), sum(r["farm"] for r in rows)
        out[kind] = {"plays": n, "per_game": n / max(1, len(games)), "gained": gained / n, "farm": cost / n,
                     "worth": sum(r.get("worth", r["farm"]) for r in rows) / n,
                     "net": (gained - cost) / n, "gold": sum(r["gold"] for r in rows) / n,
                     "xp": sum(r["xp"] for r in rows) / n}
    return out


def build(conn, model, models_dir, sample: int = 3000, log=print) -> dict[str, Any]:
    """The same for every role, over a sample of stored games: trades.json."""
    ver = swing.version(model)
    roles = {}
    for match_id, pid, position in conn.execute("""
            SELECT b.match_id, b.participant_id, b.position
              FROM (SELECT match_id FROM match_swing WHERE version = %s ORDER BY md5(match_id) LIMIT %s) s
              JOIN participant_builds b USING (match_id)
             WHERE NOT b.remake AND b.queue_id IN (420, 440) AND b.position = ANY(%s)""",
                                            (ver, sample, list(farm_calc.ROLES))):
        roles.setdefault(match_id, {})[pid] = position
    by_role: dict[str, list] = {r: [] for r in farm_calc.ROLES}
    with conn.transaction(), conn.cursor(name="trades") as cur:
        cur.itersize = 100
        cur.execute("""SELECT match_id, jsonb_build_object('facts', data->'facts', 'moments', data->'moments')
                         FROM match_swing WHERE version = %s AND match_id = ANY(%s)""", (ver, list(roles)))
        for match_id, data in cur:
            for pid, role in roles[match_id].items():
                by_role[role].append(game_trades(data, pid, role))
    table = {"version": ver, "roles": {role: summarise(games) for role, games in by_role.items() if games},
             "games": {role: len(games) for role, games in by_role.items()}}
    path = Path(models_dir) / "winprob" / FILE
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    _cache.clear()
    log(f"  trades from {len(roles):,} games")
    return table


_cache: dict[str, Any] = {}


def load(models_dir) -> dict | None:
    path = Path(models_dir) / "winprob" / FILE
    key = str(path)
    if key not in _cache:
        _cache[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return _cache[key]


def player(rows: list[tuple[int, str, dict]], table: dict | None, model) -> dict[str, Any] | None:
    """A player's trades over their games ([(participant id, role, Swing data)]), against
    their main role's."""
    from collections import Counter

    rows = [(pid, role, d) for pid, role, d in rows if role in farm_calc.ROLES and (d.get("facts") or {}).get("events")]
    if len(rows) < 3:
        return None
    role = Counter(r for _, r, _ in rows).most_common(1)[0][0]
    mine = summarise([game_trades(d, pid, r) for pid, r, d in rows])
    typical = ((table or {}).get("roles") or {}).get(role, {}) if (table or {}).get("version") == swing.version(model) else {}
    kinds = []
    for kind, name, tip in KINDS:
        if kind not in mine:
            continue
        m, t = mine[kind], typical.get(kind)
        kinds.append({"key": kind, "name": name, "tip": tip, "plays": m["plays"],
                      "per_game": round(m["per_game"], 1), "gained": round(m["gained"], 2), "farm": round(m["farm"], 2),
                      "worth": round(m["worth"], 2),
                      "net": round(m["net"], 2), "gold": round(m["gold"]), "xp": round(m["xp"]),
                      "typical_net": round(t["net"], 2) if t else None,
                      "typical_farm": round(t["farm"], 2) if t else None,
                      "typical_per_game": round(t["per_game"], 1) if t else None})
    if not kinds:
        return None
    farm_total = sum(k["worth"] * k["plays"] for k in kinds) / len(rows)
    extra_total = sum(k["farm"] * k["plays"] for k in kinds) / len(rows)
    gold_total = sum(k["gold"] * k["plays"] for k in kinds) / len(rows)
    xp_total = sum(k["xp"] * k["plays"] for k in kinds) / len(rows)
    rated = [k for k in kinds if k["typical_net"] is not None and k["plays"] >= 5]
    best = max(rated, key=lambda k: k["net"] - k["typical_net"], default=None)
    worst = min(rated, key=lambda k: k["net"] - k["typical_net"], default=None)
    return {"games": len(rows), "role": farm_calc.ROLE_NAMES.get(role, ""), "kinds": kinds,
            "farm_per_game": round(farm_total, 1), "extra_per_game": round(extra_total, 1),
            "gold_per_game": round(gold_total), "xp_per_game": round(xp_total),
            "best": best if best and best["net"] - best["typical_net"] > 0.2 else None,
            "worst": worst if worst and worst["net"] - worst["typical_net"] < -0.2 else None,
            "peak": max([abs(k["net"]) for k in kinds] + [abs(k["typical_net"] or 0) for k in kinds] + [0.5])}
