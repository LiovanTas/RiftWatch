"""A jungler's early game (the match page's Insights tab): the first 14 minutes, from the
timeline's events and minute frames.

  * ganks by lane: kills the jungler took part in, in each lane, against their own deaths
    there (a lane is where the kill happened, by the map's lane bands);
  * fights in the jungle and river, the same way;
  * invading: minutes the jungler spent in the enemy jungle (by minute frames);
  * objectives before 14:00: each grub, dragon and herald, who took it and whether the
    jungler was in it;
  * first clear: jungle CS at 3:00 and 4:00 against the enemy jungler.
"""

from __future__ import annotations

from typing import Any

from riftwatch.features.map import zone
from riftwatch.winprob.farm import cs as cs_at
from riftwatch.winprob.state import compact, team_of

EARLY_MINUTES = 14
LANES = (("top_lane", "Top"), ("mid_lane", "Mid"), ("bot_lane", "Bot"))
MONSTERS = {"HORDE": "Voidgrub", "DRAGON": "Dragon", "RIFTHERALD": "Herald"}


def _zone(e: dict) -> str | None:
    pos = e.get("position")
    return zone(float(pos.get("x", 0)), float(pos.get("y", 0))) if pos else None


def report(timeline: dict | None, me: int, enemy_jungler: int | None) -> dict[str, Any] | None:
    if not timeline:
        return None
    frames, events = compact(timeline)
    if len(frames) < 6:
        return None
    mine, end = team_of(me), EARLY_MINUTES * 60_000
    early = sorted((e for e in events if e.get("timestamp", 0) < end), key=lambda e: e.get("timestamp", 0))
    places = {name: {"label": label, "kills": 0, "deaths": 0, "first": None} for name, label in LANES}
    places["jungle"] = {"label": "Jungle and river", "kills": 0, "deaths": 0, "first": None}
    for e in early:
        if e.get("type") != "CHAMPION_KILL":
            continue
        z = _zone(e) or ""
        where = places.get(z) or (places["jungle"] if ("jungle" in z or z == "river") else None)
        if where is None:
            continue
        took_part = me == e.get("killerId") or me in (e.get("assistingParticipantIds") or [])
        if took_part and team_of(e.get("victimId", 0)) != mine:
            where["kills"] += 1
            if where["first"] is None:
                s = e.get("timestamp", 0) // 1000
                where["first"] = f"{s // 60}:{s % 60:02d}"
        elif e.get("victimId") == me:
            where["deaths"] += 1
    enemy_jungle = "red_jungle" if mine == 100 else "blue_jungle"
    invading = 0
    for m in range(1, min(EARLY_MINUTES, len(frames) - 1) + 1):
        pos = (frames[m].get(str(me)) or {}).get("position")
        if pos and zone(float(pos.get("x", 0)), float(pos.get("y", 0))) == enemy_jungle:
            invading += 1
    objectives = []
    for e in early:
        if e.get("type") != "ELITE_MONSTER_KILL" or e.get("monsterType") not in MONSTERS:
            continue
        if e.get("killerTeamId") not in (100, 200):
            continue
        s = e.get("timestamp", 0) // 1000
        objectives.append({"name": MONSTERS[e["monsterType"]], "clock": f"{s // 60}:{s % 60:02d}",
                           "ours": e.get("killerTeamId") == mine,
                           "with_me": me == e.get("killerId") or me in (e.get("assistingParticipantIds") or [])})
    clear = []
    for minute in (3, 4):
        if minute < len(frames):
            row = {"minute": minute, "cs": round(cs_at(frames[minute], me))}
            if enemy_jungler:
                row["enemy"] = round(cs_at(frames[minute], enemy_jungler))
            clear.append(row)
    ours = [o for o in objectives if o["ours"]]
    return {"lanes": [places[name] for name, _ in LANES], "jungle": places["jungle"],
            "ganks": sum(p["kills"] for p in places.values()), "deaths": sum(p["deaths"] for p in places.values()),
            "invading": invading, "objectives": objectives, "ours": len(ours),
            "with_me": sum(1 for o in ours if o["with_me"]), "contested": len(objectives), "first_clear": clear}
