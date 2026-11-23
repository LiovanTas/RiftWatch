"""Do the model's jumps go the right way?

Swing credits each kill, structure and objective with the change it made in the taking team's
chance to win: the state just before against just after, as ``swing.event_states`` builds them.
A take should never lower the taker's chance, so every time the model says one did, Swing
shows a play that helped as one that hurt. This counts how often that happens, by kind of
event, on games the model wasn't trained on (``riftwatch winprob sanity``).

On the model before this check existed (wp2, stored Swing for 19,000 games): kills never went
the wrong way, but a Baron did 9.9% of the time (by more than a point 3.3%), a tower 27% (rarely
by much) and every one of the 8 Atakhans, by 20 points on average.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from riftwatch.winprob.state import Replay, team_of
from riftwatch.winprob.swing import event_states

KINDS = ("kill", "tower", "inhibitor", "Dragon", "Elder", "Baron", "Herald", "Voidgrubs", "Atakhan")
MONSTER_KINDS = {"BARON_NASHOR": "Baron", "RIFTHERALD": "Herald", "HORDE": "Voidgrubs", "ATAKHAN": "Atakhan"}
FILE = "sanity.json"


def kind_of(e: dict) -> tuple[str, int] | None:
    """(kind, taking team) of an event Swing scores, or None (an item, a ward, a herald that
    expired: team 300)."""
    t = e.get("type")
    if t == "CHAMPION_KILL":
        victim = e.get("victimId") or 0
        return ("kill", 300 - team_of(victim)) if 1 <= victim <= 10 else None
    if t == "BUILDING_KILL":
        lost = e.get("teamId")
        if lost not in (100, 200):
            return None
        return ("inhibitor" if e.get("buildingType") == "INHIBITOR_BUILDING" else "tower", 300 - lost)
    if t == "ELITE_MONSTER_KILL":
        side, monster = e.get("killerTeamId"), e.get("monsterType")
        if side not in (100, 200):
            return None
        if monster == "DRAGON":
            return ("Elder" if e.get("monsterSubType") == "ELDER_DRAGON" else "Dragon", side)
        return (MONSTER_KINDS.get(monster, "other"), side)
    return None


def game_jumps(model, replay: Replay) -> list[tuple[str, float]]:
    """(kind, the taking team's change in win chance) for every kill, structure and objective
    in one game, in one call to the model."""
    found, blue, red = [], [], []
    for i, e in enumerate(replay.events):
        k = kind_of(e)
        if k is None:
            continue
        before, after = event_states(replay, i)
        found.append(k)
        blue += [before[0], after[0]]
        red += [before[1], after[1]]
    if not found:
        return []
    p = model.blue(blue, red)
    return [(kind, float(p[2 * n + 1] - p[2 * n]) * (1 if side == 100 else -1)) for n, (kind, side) in enumerate(found)]


def summarise(jumps: list[tuple[str, float]]) -> dict[str, dict[str, Any]]:
    """For each kind: how many, the share that lowered the taker's chance (at all, and by more
    than a point), the average change and the worst, in points."""
    by: dict[str, list[float]] = defaultdict(list)
    for kind, d in jumps:
        by[kind].append(100 * d)
    out = {}
    for kind in sorted(by, key=lambda k: KINDS.index(k) if k in KINDS else len(KINDS)):
        v = np.asarray(by[kind])
        out[kind] = {"events": int(len(v)), "wrong_way": round(float((v < 0).mean()), 4),
                     "wrong_by_1pt": round(float((v < -1).mean()), 4), "mean": round(float(v.mean()), 2),
                     "worst": round(float(v.min()), 1)}
    return out


def load_games(conn, match_ids: list[str]):
    """Replays of these games, with what the model needs (champions, purchases, positions)."""
    from riftwatch.winprob.model import GAME_SQL, players, positions

    with conn.cursor() as cur:
        cur.execute(GAME_SQL + " WHERE t.match_id = ANY(%s)", (list(match_ids),))
        for _, frames, events, rows in cur:
            champions, purchases = players(rows)
            yield frames, events, champions, purchases, positions(rows)


def check(model, games) -> dict[str, dict[str, Any]]:
    """``summarise`` over games given as (frames, events, champions, purchases, positions)."""
    catalog = model.catalog if getattr(model, "reads_items", False) else None
    jumps = []
    for frames, events, champions, purchases, pos in games:
        replay = Replay(frames, events, champions, purchases, catalog, pos)
        if replay.winner in (100, 200) and len(frames) >= 6:
            jumps += game_jumps(model, replay)
    return summarise(jumps)
