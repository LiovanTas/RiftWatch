"""The High-elo map tab: a game replayed minute by minute on the minimap, with what the player
did each minute beside what Grandmaster and Challenger players chose in similar spots.

Positions and events come from the stored timeline (one frame a minute). The comparison comes
from ml.advisor: for minutes 2 to 15 it gives the share of high-elo players in similar
situations who chose each option (farm, gank mid, take an objective, ...). The map draws the
player's actual move to the next minute as a solid arrow and the most common high-elo choice
as a dashed arrow towards where that choice leads (the lane, the pit, their own jungle).
Those targets are illustrative points; the numbers in the side panel are the evidence.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import psycopg

from riftwatch.web.gamedata import POSITIONS, GameData

# Summoner's Rift coordinates (the minimap image covers this square)
X0, Y0, X1, Y1 = -120.0, -120.0, 14870.0, 14980.0

OPTION = {
    "farm": "Farm camps", "gank_top": "Gank top", "gank_mid": "Gank mid", "gank_bot": "Gank bot",
    "objective": "Take an objective", "invade": "Invade", "rotate": "Rotate through a lane",
    "base": "Back to base", "other": "Move between areas", "lane": "Stay in lane",
    "push": "Push for plates or the tower", "fight": "Join a fight", "roam": "Roam",
    "roam_top": "Roam top", "roam_bot": "Roam bot", "roam_mid": "Roam mid",
    "with_adc": "Stay with the ADC", "ward": "Ward",
}
DID = {
    "farm": "farmed camps", "gank_top": "ganked top", "gank_mid": "ganked mid",
    "gank_bot": "ganked bot", "objective": "went to an objective", "invade": "invaded",
    "rotate": "rotated through a lane", "base": "went back to base", "other": "moved between areas",
    "lane": "stayed in lane", "push": "pushed for plates or the tower", "fight": "joined a fight",
    "roam": "roamed", "roam_top": "roamed top", "roam_bot": "roamed bot", "roam_mid": "roamed mid",
    "with_adc": "stayed with the ADC", "ward": "went warding",
}

# Map points (blue side's view; red mirrors through the centre)
LANE_POINT = {"top": (1800, 12900), "mid": (7400, 7400), "bot": (12900, 1800)}
HOME_LANE = {"TOP": "top", "MIDDLE": "mid", "BOTTOM": "bot", "UTILITY": "bot"}
ENEMY_OUTER = {"top": (4318, 13875), "mid": (8955, 8510), "bot": (13866, 4505)}
PITS = ((9866, 4414), (5007, 10471))                     # dragon; baron, herald and grubs
OWN_JUNGLE = ((3900, 7700), (7700, 3900))
RIVER = ((5200, 9300), (9600, 5500))
FOUNTAIN = (400, 420)


def _mirror(p: tuple[float, float], team: int) -> tuple[float, float]:
    return p if team == 100 else (X1 + X0 - p[0], Y1 + Y0 - p[1])


def _pct(x: float, y: float) -> dict[str, float]:
    return {"x": round(100 * (x - X0) / (X1 - X0), 2), "y": round(100 * (1 - (y - Y0) / (Y1 - Y0)), 2)}


def _nearest(points, x: float, y: float):
    return min(points, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)


def target(decision: str, role: str, team: int, x: float, y: float,
           adc: tuple[float, float] | None) -> tuple[float, float] | None:
    """Roughly where a choice leads on the map, for the dashed arrow."""
    if decision in ("gank_top", "roam_top"):
        return LANE_POINT["top"]
    if decision in ("gank_mid", "roam_mid"):
        return LANE_POINT["mid"]
    if decision in ("gank_bot", "roam_bot"):
        return LANE_POINT["bot"]
    if decision == "lane":
        return LANE_POINT.get(HOME_LANE.get(role, ""), None)
    if decision == "push":
        lane = HOME_LANE.get(role)
        return _mirror(ENEMY_OUTER[lane], team) if lane else None
    if decision == "objective":
        return _nearest(PITS, x, y)
    if decision == "farm":
        return _nearest([_mirror(p, team) for p in OWN_JUNGLE], x, y)
    if decision == "invade":
        return _nearest([_mirror(p, 300 - team) for p in OWN_JUNGLE], x, y)
    if decision == "base":
        return _mirror(FOUNTAIN, team)
    if decision == "ward":
        return _nearest(RIVER, x, y)
    if decision == "with_adc":
        return adc
    return None                                         # fight, rotate, roam, other


_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
_lock = threading.Lock()


def map_state(conn: psycopg.Connection, gd: GameData, advisor, match_id: str,
              puuid: str) -> dict[str, Any] | None:
    """Everything the High-elo map tab draws for one player's game, or None when the game,
    its timeline or the player isn't stored."""
    key = (match_id, puuid)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    row = conn.execute("SELECT m.raw, t.raw FROM matches m JOIN match_timelines t USING (match_id) "
                       "WHERE m.match_id = %s", (match_id,)).fetchone()
    if row is None:
        return None
    match, timeline = row
    people = match["info"]["participants"]
    me = next((p for p in people if p["puuid"] == puuid), None)
    if me is None:
        return None
    frames = timeline.get("info", {}).get("frames", [])
    role = me.get("teamPosition") or ""
    team = me["teamId"]
    adc_pid = next((p["participantId"] for p in people
                    if p["teamId"] == team and p.get("teamPosition") == "BOTTOM"), None)

    advice: dict[int, Any] = {}
    agreement = None
    if advisor is not None and role:
        found = advisor.minutes(match, timeline, puuid)
        if found:
            _, minutes = found
            agreement = round(100 * sum(k in ("good", "agree") for _, _, k in minutes)
                              / max(1, sum(k != "neutral" for _, _, k in minutes)))
            for e, options, kind in minutes:
                advice[e.minute] = (e, options, kind)

    players = [{"pid": p["participantId"], "team": p["teamId"], "me": p["puuid"] == puuid,
                "champion": gd.champion(p["championId"]).name,
                "icon": gd.champion_icon(p["championId"]),
                "name": p.get("riotIdGameName") or p.get("summonerName") or "",
                "role": POSITIONS.get(p.get("teamPosition") or "", "")} for p in people]
    out_minutes = []
    for m, frame in enumerate(frames):
        pf = frame.get("participantFrames") or {}
        pos = {int(k): (v.get("position") or {}) for k, v in pf.items()}
        here = pos.get(me["participantId"]) or {}
        nxt = ((frames[m + 1].get("participantFrames") or {}).get(str(me["participantId"])) or {}
               ).get("position") if m + 1 < len(frames) else None
        events = []
        for ev in frame.get("events", []):
            where = ev.get("position")
            if not where:
                continue
            if ev.get("type") == "CHAMPION_KILL":
                victim = ev.get("victimId")
                mine = victim == me["participantId"] or ev.get("killerId") == me["participantId"]
                events.append({"kind": "kill", **_pct(where["x"], where["y"]), "mine": mine,
                               "team": 200 if (victim or 0) <= 5 else 100})
            elif ev.get("type") in ("ELITE_MONSTER_KILL", "BUILDING_KILL"):
                events.append({"kind": "objective" if ev["type"] == "ELITE_MONSTER_KILL" else "tower",
                               **_pct(where["x"], where["y"]),
                               "team": ev.get("killerTeamId") or (300 - ev.get("teamId", 100)),
                               "name": (ev.get("monsterSubType") or ev.get("monsterType")
                                        or ev.get("buildingType") or "").replace("_", " ").title()})
        minute: dict[str, Any] = {
            "minute": m,
            "pos": {str(pid): _pct(p["x"], p["y"]) for pid, p in pos.items() if "x" in p},
            "next": _pct(nxt["x"], nxt["y"]) if nxt else None,
            "events": events,
        }
        if m in advice and here:
            e, options, kind = advice[m]
            ranked = sorted(options, key=lambda o: -o.share)
            did = next((o for o in options if o.decision == e.decision), None)
            best = ranked[0]
            adc = pos.get(adc_pid) if adc_pid else None
            point = target(best.decision, role, team, here["x"], here["y"],
                           (adc["x"], adc["y"]) if adc and "x" in adc else None)
            minute["advice"] = {
                "kind": kind, "did": DID.get(e.decision, e.decision),
                "did_share": round(100 * did.share) if did else None,
                "best": OPTION.get(best.decision, best.decision),
                "best_share": round(100 * best.share),
                "options": [{"label": OPTION.get(o.decision, o.decision), "share": round(100 * o.share),
                             "mine": o.decision == e.decision} for o in ranked[:5]],
                "target": _pct(*point) if point and kind in ("moment", "differ") else None,
            }
        out_minutes.append(minute)

    metrics = {}
    if advisor is not None and role and advisor.available(role):
        metrics = getattr(advisor.model(role), "metrics", {}) or {}
    result = {
        "match_id": match_id, "role": POSITIONS.get(role, role.title()), "team": team,
        "me": me["participantId"], "champion": gd.champion(me["championId"]).name,
        "players": players, "minutes": out_minutes, "agreement": agreement,
        "compared": bool(advice), "first": min(advice) if advice else None,
        "last": max(advice) if advice else None,
        "moments": sum(1 for _, _, k in advice.values() if k == "moment"),
        "games": metrics.get("games"),
        "map": f"https://ddragon.leagueoflegends.com/cdn/{gd.version}/img/map/map11.png",
    }
    with _lock:
        _cache[key] = result
        while len(_cache) > 64:
            _cache.popitem(last=False)
    return result
