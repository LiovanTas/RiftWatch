"""Costly habits in a player's deaths, and how often each costs players in the same role.

A death's cost is its own swing plus anything the enemy took before the player respawned
(winprob.swing records both). Each habit is a kind of death; for a player it's measured as
the points those deaths cost per game, beside the same figure for every stored player in that
role, so "you lose 13.7 a game to deaths caught by three or more" comes with "junglers lose
9.8" next to it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

HABITS = (
    ("while_dead", "Deaths that cost an objective",
     "the enemy took a dragon, Baron, herald, tower or inhibitor before you respawned",
     lambda d: bool(d.get("while_dead"))),
    ("outnumbered", "Caught by three or more",
     "three or more enemies took part in the kill",
     lambda d: d.get("enemies", 0) >= 3),
    ("held_gold", "Deaths holding 1,500+ gold",
     "you died with a back's worth of gold unspent (from the minute before)",
     lambda d: d.get("held", 0) >= 1500),
    ("deep", "Deaths deep in enemy territory",
     "in the enemy jungle or base",
     lambda d: d.get("zone") in ("the enemy jungle", "the enemy base")),
    ("late", "Long death timers",
     "deaths with a timer of 40 seconds or more, when a death can lose the game",
     lambda d: d.get("timer", 0) >= 40),
)


def deaths(data: dict, pid: int) -> list[tuple[dict, float, list[str]]]:
    """(moment, cost, habits) for each of a player's deaths in one game's Swing."""
    out = []
    for m in data.get("moments", []):
        d = m.get("death")
        if not d or pid not in m.get("lost", []):
            continue
        cost = m["swing"] + d.get("while_dead_cost", 0.0)
        out.append((m, cost, [key for key, _, _, test in HABITS if test(d)]))
    return out


def role_norms(rows) -> dict[str, Any]:
    """Per role: games, deaths and cost per game overall and for each habit, from
    (position, swing data, participant id) for many stored player-games."""
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for position, data, pid in rows:
        if not position:
            continue
        a = acc[position]
        a["games"] += 1
        for _, cost, keys in deaths(data, pid):
            a["deaths"] += 1
            a["cost"] += cost
            for key in keys:
                a[f"{key}_cost"] += cost
                a[f"{key}_deaths"] += 1
    out = {}
    for position, a in acc.items():
        g = a["games"] or 1
        out[position] = {"games": int(a["games"]), "deaths": round(a["deaths"] / g, 2),
                         "cost": round(a["cost"] / g, 2),
                         **{key: {"per_game": round(a[f"{key}_cost"] / g, 2),
                                  "share": round(a[f"{key}_deaths"] / max(1, a["deaths"]), 3)}
                            for key, *_ in HABITS}}
    return out
