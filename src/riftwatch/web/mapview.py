"""The High Elo Comparison tab: a game replayed minute by minute on the minimap, with what the
player did each minute beside what Grandmaster and Challenger players chose in similar spots,
the state of the game at that moment, the facts that fit the high-elo choice, and what
actually happened next.

Positions and events come from the stored timeline (one frame a minute). The comparison comes
from ml.advisor: for every labelled minute (2 to the end of the game) it gives the share of
high-elo players in similar situations who chose each option (farm, gank mid, take an
objective, ...). The map draws the
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


MONSTER = {"DRAGON": "Dragon", "ELDER_DRAGON": "Elder Dragon", "BARON_NASHOR": "Baron",
           "RIFTHERALD": "Rift Herald", "HORDE": "a Voidgrub", "ATAKHAN": "Atakhan"}
DRAKE = {"FIRE_DRAGON": "Infernal", "WATER_DRAGON": "Ocean", "EARTH_DRAGON": "Mountain",
         "AIR_DRAGON": "Cloud", "HEXTECH_DRAGON": "Hextech", "CHEMTECH_DRAGON": "Chemtech"}
LANE_NAME = {"TOP_LANE": "top", "MID_LANE": "mid", "BOT_LANE": "bot"}
NEAR = 2500                               # map units: "near you" for allies and enemies
PLATES_FALL = 14                          # turret plates drop at 14:00


def _clock(ms: int) -> str:
    return f"{ms // 60000}:{(ms // 1000) % 60:02d}"


def _dist(a: dict, b: dict) -> float:
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5


def _describe(ev: dict, champ_of: dict[int, str], me_pid: int, team: int) -> str | None:
    """One line for a kill, objective or tower, from the player's side."""
    who = lambda pid: "you" if pid == me_pid else champ_of.get(pid, "a minion or tower")  # noqa: E731
    t = _clock(ev.get("timestamp", 0))
    kind = ev.get("type")
    if kind == "CHAMPION_KILL":
        killer, victim = ev.get("killerId"), ev.get("victimId")
        helped = me_pid in (ev.get("assistingParticipantIds") or [])
        name = who(killer)
        line = f"{t} {name[:1].upper() + name[1:]} killed {who(victim)}"
        return line + (" (you assisted)" if helped else "")
    if kind == "ELITE_MONSTER_KILL":
        side = "Your team" if ev.get("killerTeamId") == team else "The enemy"
        name = MONSTER.get(ev.get("monsterType"), "an objective")
        if ev.get("monsterType") == "DRAGON" and ev.get("monsterSubType") in DRAKE:
            name = f"{DRAKE[ev['monsterSubType']]} Dragon"
        return f"{t} {side} took {name}"
    if kind == "BUILDING_KILL":
        lost_by = ev.get("teamId")
        lane = LANE_NAME.get(ev.get("laneType"), "")
        what = "inhibitor" if ev.get("buildingType") == "INHIBITOR_BUILDING" else "tower"
        if lost_by == team:
            return f"{t} The enemy destroyed your {lane} {what}".replace("  ", " ")
        return f"{t} Your team destroyed the enemy {lane} {what}".replace("  ", " ")
    return None


def _objectives(events: list[dict], now_min: float) -> dict[str, Any]:
    """Which epic monsters are up (or soon) at ``now_min``, from kills so far (spawn rules as
    in ml.situations)."""
    from riftwatch.ml.situations import (BARON_RESPAWN, BARON_SPAWN, DRAGON_RESPAWN, DRAGON_SPAWN,
                                         GRUBS_SPAWN, HERALD_SPAWN)

    done = [e for e in events if e.get("type") == "ELITE_MONSTER_KILL" and e["timestamp"] / 60000 <= now_min]
    last = lambda mtype: max((e["timestamp"] / 60000 for e in done if e.get("monsterType") == mtype), default=None)  # noqa: E731
    d, b = last("DRAGON"), last("BARON_NASHOR")
    dragon_at = d + DRAGON_RESPAWN if d is not None else DRAGON_SPAWN
    baron_at = b + BARON_RESPAWN if b is not None else BARON_SPAWN
    return {"dragon_in": dragon_at - now_min, "baron_in": baron_at - now_min,
            "grubs_up": now_min >= GRUBS_SPAWN and last("HORDE") is None and now_min < HERALD_SPAWN,
            "herald_up": now_min >= HERALD_SPAWN and last("RIFTHERALD") is None and now_min < BARON_SPAWN}


def _reasons(best: str, did: str, c: dict[str, Any]) -> list[str]:
    """Up to three facts from this moment of the game that fit the most common high-elo
    choice (or warn about the player's own). Facts only: no advice beyond the data."""
    out: list[str] = []
    if c["hp"] == 0:
        return ["You were dead at the start of this minute."]
    obj = c["objectives"]
    lead = c["team_lead"]
    if best == "base" or did == "base":
        if c["unspent"] >= 1100 and best == "base":
            out.append(f"You had {c['unspent']:,} gold unspent, enough to buy a real item component.")
        if c["hp"] is not None and c["hp"] <= 40 and best == "base":
            out.append(f"You were at {c['hp']}% health.")
    if best == "objective":
        if obj["dragon_in"] <= 1:
            out.append("Dragon was up." if obj["dragon_in"] <= 0 else "Dragon was about to spawn.")
        if obj["baron_in"] <= 1 and c["minute"] >= 19:
            out.append("Baron was up." if obj["baron_in"] <= 0 else "Baron was about to spawn.")
        if obj["grubs_up"]:
            out.append("Voidgrubs were up.")
        if obj["herald_up"]:
            out.append("Rift Herald was up.")
    if best in ("objective", "fight", "invade") and c["allies_near"] >= 2:
        out.append(f"{c['allies_near']} teammates were within {NEAR:,} units of you.")
    if did in ("invade", "fight", "push", "roam", "roam_top", "roam_mid", "roam_bot") and c["enemies_near"] >= 3:
        out.append(f"{c['enemies_near']} enemies were within {NEAR:,} units of you (seen on the replay).")
    if best in ("lane", "push", "farm") and c["opp_name"] and c["cs_diff"] <= -15:
        out.append(f"You were {-c['cs_diff']} CS behind {c['opp_name']}, so every wave mattered.")
    if best in ("lane", "push") and c["opp_down"]:
        out.append(f"{c['opp_name']} was {c['opp_down']}, so your lane was free.")
    if best == "push" and c["minute"] < PLATES_FALL:
        out.append(f"Turret plates were still up (they fall at {PLATES_FALL}:00).")
    if best.startswith(("gank_", "roam_")):
        lane = best.split("_", 1)[1]
        foe = c["lane_foes"].get(lane)
        if foe and foe["hp"] is not None and foe["hp"] <= 50:
            out.append(f"The enemy {lane} laner ({foe['name']}) was at {foe['hp']}% health.")
        if foe and foe["level_diff"] >= 1:
            out.append(f"Your {lane} laner was {foe['level_diff']} level{'s' if foe['level_diff'] > 1 else ''} ahead.")
    if best == "farm" and obj["dragon_in"] > 1 and obj["baron_in"] > 1 and not obj["grubs_up"] and not obj["herald_up"]:
        out.append("No epic monster was up, so camps were the best gold available.")
    if did.startswith(("gank_", "roam_")) and best != did:
        lane = did.split("_", 1)[1]
        foe = c["lane_foes"].get(lane)
        if foe and foe["hp"] is not None and foe["hp"] >= 80:
            out.insert(0, f"The enemy {lane} laner ({foe['name']}) was at {foe['hp']}% health, so your "
                          f"{'gank' if did.startswith('gank') else 'roam'} had little to work with.")
    if best == "farm" and lead <= -1500:
        out.append(f"Your team was {-lead:,} gold behind.")
    if lead and abs(lead) >= 1500 and best in ("objective", "fight", "push") :
        out.append(f"Your team was {abs(lead):,} gold {'ahead' if lead > 0 else 'behind'}.")
    return out[:3]


def map_state(conn: psycopg.Connection, gd: GameData, advisor, match_id: str,
              puuid: str) -> dict[str, Any] | None:
    """Everything the High Elo Comparison tab draws for one player's game, or None when the
    game, its timeline or the player isn't stored."""
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
    me_pid = me["participantId"]
    champ_of = {p["participantId"]: gd.champion(p["championId"]).name for p in people}
    team_of = {p["participantId"]: p["teamId"] for p in people}
    role_pid = {(p["teamId"], p.get("teamPosition")): p["participantId"] for p in people}
    opp_pid = role_pid.get((300 - team, role))
    adc_pid = role_pid.get((team, "BOTTOM"))
    all_events = sorted((ev for f in frames for ev in f.get("events", [])), key=lambda e: e.get("timestamp", 0))

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

    def stats(frame: dict, pid: int | None) -> dict:
        return ((frame.get("participantFrames") or {}).get(str(pid)) or {}) if pid else {}

    def hp(frame: dict, pid: int | None) -> int | None:
        cs = stats(frame, pid).get("championStats") or {}
        return round(100 * cs["health"] / cs["healthMax"]) if cs.get("healthMax") else None

    def cs_of(frame: dict, pid: int | None) -> int:
        st = stats(frame, pid)
        return st.get("minionsKilled", 0) + st.get("jungleMinionsKilled", 0)

    def window(a_min: int, b_min: int) -> list[dict]:
        return [e for e in all_events if a_min * 60000 <= e.get("timestamp", 0) < b_min * 60000]

    def team_gold(frame: dict, side: int) -> int:
        return sum(v.get("totalGold", 0) for k, v in (frame.get("participantFrames") or {}).items()
                   if team_of.get(int(k)) == side)

    players = [{"pid": p["participantId"], "team": p["teamId"], "me": p["puuid"] == puuid,
                "champion": champ_of[p["participantId"]], "icon": gd.champion_icon(p["championId"]),
                "name": p.get("riotIdGameName") or p.get("summonerName") or "",
                "role": POSITIONS.get(p.get("teamPosition") or "", "")} for p in people]
    out_minutes = []
    for m, frame in enumerate(frames):
        pf = frame.get("participantFrames") or {}
        pos = {int(k): (v.get("position") or {}) for k, v in pf.items()}
        here = pos.get(me_pid) or {}
        nxt = (stats(frames[m + 1], me_pid).get("position") if m + 1 < len(frames) else None)
        this_minute = window(m, m + 1)
        markers = []
        for ev in this_minute:
            where = ev.get("position")
            if not where:
                continue
            if ev.get("type") == "CHAMPION_KILL":
                victim = ev.get("victimId")
                markers.append({"kind": "kill", **_pct(where["x"], where["y"]),
                                "mine": me_pid in (victim, ev.get("killerId")),
                                "team": 200 if (victim or 0) <= 5 else 100,
                                "text": _describe(ev, champ_of, me_pid, team)})
            elif ev.get("type") in ("ELITE_MONSTER_KILL", "BUILDING_KILL"):
                markers.append({"kind": "objective" if ev["type"] == "ELITE_MONSTER_KILL" else "tower",
                                **_pct(where["x"], where["y"]), "team": ev.get("killerTeamId") or (300 - ev.get("teamId", 100)),
                                "text": _describe(ev, champ_of, me_pid, team)})
        objectives = _objectives(all_events, m)
        lead = team_gold(frame, team) - team_gold(frame, 300 - team)
        dragons = [e for e in all_events if e.get("type") == "ELITE_MONSTER_KILL"
                   and e.get("monsterType") == "DRAGON" and e["timestamp"] <= m * 60000]
        near = [pid for pid, p in pos.items() if pid != me_pid and "x" in p and here and _dist(p, here) <= NEAR]
        opp_down = None
        if opp_pid:
            if any(e.get("type") == "CHAMPION_KILL" and e.get("victimId") == opp_pid
                   and m * 60000 - 45000 <= e["timestamp"] <= m * 60000 for e in all_events):
                opp_down = "dead"
            elif pos.get(opp_pid) and "x" in pos[opp_pid] and own_zone_name(pos[opp_pid], 300 - team) == "own_base":
                opp_down = "in base"
        lane_foes = {}
        for lane, lane_role in (("top", "TOP"), ("mid", "MIDDLE"), ("bot", "BOTTOM")):
            foe, ally = role_pid.get((300 - team, lane_role)), role_pid.get((team, lane_role))
            if foe:
                lane_foes[lane] = {"name": champ_of[foe], "hp": hp(frame, foe),
                                   "level_diff": stats(frame, ally).get("level", 0) - stats(frame, foe).get("level", 0)}
        context = {
            "minute": m, "team_lead": lead, "objectives": objectives,
            "unspent": stats(frame, me_pid).get("currentGold", 0), "hp": hp(frame, me_pid),
            "level": stats(frame, me_pid).get("level"),
            "cs_diff": cs_of(frame, me_pid) - cs_of(frame, opp_pid) if opp_pid else 0,
            "opp_name": champ_of.get(opp_pid, ""), "opp_down": opp_down,
            "allies_near": sum(team_of.get(pid) == team for pid in near),
            "enemies_near": sum(team_of.get(pid) != team for pid in near), "lane_foes": lane_foes,
        }
        up = []
        if objectives["dragon_in"] <= 0:
            up.append("dragon")
        if objectives["baron_in"] <= 0 and m >= 20:
            up.append("baron")
        if objectives["grubs_up"] or objectives["herald_up"]:
            up.append("herald")
        minute: dict[str, Any] = {
            "minute": m,
            "pos": {str(pid): _pct(p["x"], p["y"]) for pid, p in pos.items() if "x" in p},
            "next": _pct(nxt["x"], nxt["y"]) if nxt else None,
            "events": markers,
            "up": up,
            "state": {
                "lead": lead,
                "dragons": [sum(team_of.get(e.get("killerId")) == team for e in dragons),
                            sum(team_of.get(e.get("killerId")) != team for e in dragons)],
                "dragon_in": round(objectives["dragon_in"], 1), "baron_in": round(objectives["baron_in"], 1),
                "level": context["level"], "hp": context["hp"], "dead": context["hp"] == 0,
                "unspent": context["unspent"],
                "cs_diff": context["cs_diff"] if opp_pid else None, "opp": context["opp_name"],
                "allies_near": context["allies_near"], "enemies_near": context["enemies_near"],
            },
            "this_minute": [d for d in (_describe(e, champ_of, me_pid, team) for e in this_minute) if d],
        }
        if m in advice and here:
            e, options, kind = advice[m]
            ranked = sorted(options, key=lambda o: -o.share)
            did = next((o for o in options if o.decision == e.decision), None)
            best = ranked[0]
            adc = pos.get(adc_pid) if adc_pid else None
            point = target(best.decision, role, team, here["x"], here["y"],
                           (adc["x"], adc["y"]) if adc and "x" in adc else None)
            later = min(m + 3, len(frames) - 1)
            lead_later = team_gold(frames[later], team) - team_gold(frames[later], 300 - team)
            followed = [d for d in (_describe(x, champ_of, me_pid, team) for x in window(m + 1, m + 3)) if d]
            minute["advice"] = {
                "kind": kind, "did": DID.get(e.decision, e.decision),
                "did_share": round(100 * did.share) if did else None,
                "best": OPTION.get(best.decision, best.decision),
                "best_share": round(100 * best.share),
                "options": [{"label": OPTION.get(o.decision, o.decision), "share": round(100 * o.share),
                             "mine": o.decision == e.decision} for o in ranked[:5]],
                "target": _pct(*point) if point and kind in ("moment", "differ") else None,
                "why": _reasons(best.decision, e.decision, context),
                "next": followed[:4],
                "lead_change": {"from": lead, "to": lead_later, "minutes": later - m} if later > m else None,
            }
        out_minutes.append(minute)

    metrics = {}
    if advisor is not None and role and advisor.available(role):
        metrics = getattr(advisor.model(role), "metrics", {}) or {}
    result = {
        "match_id": match_id, "role": POSITIONS.get(role, role.title()), "team": team,
        "me": me_pid, "champion": champ_of[me_pid],
        "players": players, "minutes": out_minutes, "agreement": agreement,
        "compared": bool(advice), "first": min(advice) if advice else None,
        "last": max(advice) if advice else None,
        "moments": sum(1 for _, _, k in advice.values() if k == "moment"),
        "games": metrics.get("games"),
        "map": f"https://ddragon.leagueoflegends.com/cdn/{gd.version}/img/map/map11.png",
        "pits": {"dragon": _pct(*PITS[0]), "baron": _pct(*PITS[1])},
    }
    with _lock:
        _cache[key] = result
        while len(_cache) > 64:
            _cache.popitem(last=False)
    return result


def own_zone_name(p: dict, team: int) -> str:
    from riftwatch.ml.situations import own_zone

    return own_zone(p["x"], p["y"], team)


def clear_cache() -> None:
    with _lock:
        _cache.clear()


# -- where a player dies and gets kills ---------------------------------------------------------

ZONE_NAME = {"own_jungle": "your jungle", "enemy_jungle": "the enemy jungle", "river": "the river",
             "top_lane": "top lane", "mid_lane": "mid lane", "bot_lane": "bot lane", "own_base": "your base",
             "enemy_base": "the enemy base"}


def death_map(conn: psycopg.Connection, gd: GameData, puuid: str, games: int = 20) -> dict[str, Any]:
    """The player's deaths and kills over their recent games with timelines, drawn from their
    own side (red-side games are mirrored so their base is always bottom left), and where on
    the map the deaths happened."""
    from collections import Counter

    from riftwatch.ml.situations import own_zone

    mine = conn.execute(
        """SELECT b.match_id, b.participant_id, b.team_id FROM participant_builds b
             JOIN match_timelines t ON t.match_id = b.match_id
            WHERE b.puuid = %s AND b.queue_id IN (420, 440, 400) AND NOT b.remake
            ORDER BY b.game_start DESC LIMIT %s""", (puuid, games)).fetchall()
    who = {m: (pid, team) for m, pid, team in mine}
    points, zones = [], Counter()
    for match_id, ev in conn.execute(
            """SELECT match_id, jsonb_path_query(raw, '$.info.frames[*].events[*] ? (@.type == "CHAMPION_KILL")')
                 FROM match_timelines WHERE match_id = ANY(%s)""", (list(who),)):
        pid, team = who[match_id]
        where = ev.get("position")
        if not where:
            continue
        if ev.get("victimId") == pid:
            kind = "death"
            zones[own_zone(where["x"], where["y"], team)] += 1
        elif ev.get("killerId") == pid:
            kind = "kill"
        else:
            continue
        x, y = (where["x"], where["y"]) if team == 100 else (X1 + X0 - where["x"], Y1 + Y0 - where["y"])
        points.append({**_pct(x, y), "kind": kind, "minute": ev.get("timestamp", 0) // 60000})
    deaths = sum(zones.values())
    return {
        "games": len(who), "map": f"https://ddragon.leagueoflegends.com/cdn/{gd.version}/img/map/map11.png",
        "points": points, "deaths": deaths, "kills": sum(1 for p in points if p["kind"] == "kill"),
        "zones": [{"zone": ZONE_NAME.get(z, z.replace("_", " ")), "share": round(100 * n / deaths), "deaths": n}
                  for z, n in zones.most_common()] if deaths else [],
    }
