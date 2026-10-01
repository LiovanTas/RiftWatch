"""Synthetic match-v5 + timeline JSON with the same shape Riot returns.

Numbers follow simple formulas so tests can compute expected features by hand:

* participant ``p`` (1-10): team 100 for 1-5, 200 for 6-10; roles in the order
  TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY on each team (so 1 faces 6, 2 faces 7, ...)
* at minute ``m``: ``totalGold = 500 + gpm[p] * m``, ``xp = 300 * m * xp_mult[p]``,
  lane CS ``= cs_rate[p] * m`` (laners) and jungle CS ``= 4 * m`` (junglers)
* frames at ``m * 60000 + 13`` ms for m >= 1, plus a final frame at game end
"""

from __future__ import annotations

import copy
from typing import Any

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
CHAMPS = (266, 64, 103, 202, 412, 86, 254, 61, 51, 117)  # Aatrox, Lee, Ahri, Jhin, Thresh ...
NAMES = ("Aatrox", "LeeSin", "Ahri", "Jhin", "Thresh", "Garen", "Vi", "Orianna", "Caitlyn", "Lulu")

BLUE_FOUNTAIN = {"x": 400, "y": 400}

# Per-participant knobs; index 0 unused.
GPM = [0, 380, 330, 400, 420, 250, 360, 390, 340, 400, 240]
CS_RATE = [0, 7, 0, 8, 8, 1, 6, 0, 7, 9, 1]
XP_MULT = [0, 1.0, 0.9, 1.05, 0.95, 0.7, 0.95, 0.9, 1.0, 0.9, 0.7]


def role_of(pid: int) -> str:
    return ROLES[(pid - 1) % 5]


def team_of(pid: int) -> int:
    return 100 if pid <= 5 else 200


def lane_cs(pid: int, minute: int) -> int:
    return CS_RATE[pid] * minute


def jungle_cs(pid: int, minute: int) -> int:
    return 4 * minute if role_of(pid) == "JUNGLE" else 0


def total_gold(pid: int, minute: int) -> int:
    return 500 + GPM[pid] * minute


def xp(pid: int, minute: int) -> int:
    return int(300 * minute * XP_MULT[pid])


def kill(ts, killer, victim, assists=(), x=7400, y=7400):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "killerId": killer, "victimId": victim,
            "assistingParticipantIds": list(assists), "position": {"x": x, "y": y},
            "bounty": 300, "shutdownBounty": 0, "killStreakLength": 0}


def ward(ts, creator, ward_type="YELLOW_TRINKET"):
    return {"type": "WARD_PLACED", "timestamp": ts, "creatorId": creator, "wardType": ward_type}


def ward_kill(ts, killer, ward_type="CONTROL_WARD"):
    return {"type": "WARD_KILL", "timestamp": ts, "killerId": killer, "wardType": ward_type}


def monster(ts, killer, monster_type="DRAGON", sub="FIRE_DRAGON", assists=()):
    e = {"type": "ELITE_MONSTER_KILL", "timestamp": ts, "killerId": killer,
         "killerTeamId": team_of(killer), "monsterType": monster_type,
         "position": {"x": 9866, "y": 4414}}
    if sub:
        e["monsterSubType"] = sub
    if assists:
        e["assistingParticipantIds"] = list(assists)
    return e


def tower(ts, killer, lane="MID_LANE", assists=()):
    return {"type": "BUILDING_KILL", "timestamp": ts, "killerId": killer,
            "teamId": 200 if team_of(killer) == 100 else 100, "buildingType": "TOWER_BUILDING",
            "laneType": lane, "towerType": "OUTER_TURRET",
            "assistingParticipantIds": list(assists), "position": {"x": 5846, "y": 6396}}


def plate(ts, killer, lane="MID_LANE"):
    return {"type": "TURRET_PLATE_DESTROYED", "timestamp": ts, "killerId": killer,
            "laneType": lane, "teamId": 200 if team_of(killer) == 100 else 100,
            "position": {"x": 5846, "y": 6396}}


DEFAULT_EVENTS = [
    ward(90_000, 5), ward(95_000, 10),
    kill(185_000, 2, 6, assists=[1], x=1500, y=10500),          # gank top (top lane)
    kill(410_000, 8, 3, x=7300, y=7500),                         # mid solo kill (mid)
    ward(420_000, 5, "CONTROL_WARD"), ward_kill(430_000, 10),
    monster(560_000, 2, assists=[4, 5]),                         # blue dragon
    plate(600_000, 4, "BOT_LANE"),
    kill(660_000, 4, 9, assists=[5], x=12000, y=2000),           # bot lane
    kill(700_000, 9, 4, x=7500, y=7000),                         # Jhin dies mid... ahead
    tower(800_000, 3, assists=[2]),
    monster(1_000_000, 7, "HORDE", None, assists=[8]),           # red grubs
    monster(1_200_000, 2, "BARON_NASHOR", None, assists=[1, 3, 4, 5]),
    kill(1_250_000, 1, 6, assists=[2, 3], x=4000, y=11000),
    kill(1_300_000, 6, 3, x=3500, y=3500),                       # Ahri dies in own base?! (blue base)
]


def build_game(
    match_id: str = "NA1_5000000001",
    minutes: int = 26,
    extra_seconds: int = 34,
    puuids: list[str] | None = None,
    events: list[dict[str, Any]] | None = None,
    queue: int = 420,
    version: str = "16.19.712.3456",
    start_ms: int = 1_790_000_000_000,
    blue_wins: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    puuids = puuids or [f"puuid-{match_id}-{p}" for p in range(1, 11)]
    events = copy.deepcopy(DEFAULT_EVENTS if events is None else events)
    duration_s = minutes * 60 + extra_seconds
    end_ms = duration_s * 1000

    # Cumulative event counters at time t, for frames and the final match JSON.
    def counts_until(pid: int, t: int) -> dict[str, int]:
        c = {"kills": 0, "deaths": 0, "assists": 0, "wards": 0, "wards_killed": 0}
        for e in events:
            if e["timestamp"] > t:
                continue
            if e["type"] == "CHAMPION_KILL":
                c["kills"] += e["killerId"] == pid
                c["deaths"] += e["victimId"] == pid
                c["assists"] += pid in e.get("assistingParticipantIds", [])
            elif e["type"] == "WARD_PLACED":
                c["wards"] += e["creatorId"] == pid
            elif e["type"] == "WARD_KILL":
                c["wards_killed"] += e["killerId"] == pid
        return c

    def pframe(pid: int, minute_value: float) -> dict[str, Any]:
        m = minute_value
        return {
            "participantId": pid,
            "totalGold": int(total_gold(pid, 1) + GPM[pid] * (m - 1)) if m >= 1 else 500,
            "currentGold": 150,
            "xp": int(300 * m * XP_MULT[pid]),
            "level": min(18, 1 + int(m * 0.7)),
            "minionsKilled": int(CS_RATE[pid] * m),
            "jungleMinionsKilled": int(4 * m) if role_of(pid) == "JUNGLE" else 0,
            "position": {"x": 7000 + pid * 50, "y": 7000 + pid * 40},
            "damageStats": {"totalDamageDoneToChampions": int(500 * m * (1 + pid % 3))},
            "championStats": {},
            "goldPerSecond": 0,
            "timeEnemySpentControlled": 0,
        }

    frame_ts = [0] + [m * 60_000 + 13 for m in range(1, minutes + 1)] + [end_ms]
    frames = []
    for i, ts in enumerate(frame_ts):
        minute_value = ts / 60_000
        frames.append({
            "timestamp": ts,
            "participantFrames": {str(p): pframe(p, minute_value) for p in range(1, 11)},
            "events": [],
        })
    frames[0]["events"].append({"type": "PAUSE_END", "timestamp": 0, "realTimestamp": start_ms})
    for e in sorted(events, key=lambda e: e["timestamp"]):
        idx = next(i for i, ts in enumerate(frame_ts) if ts >= e["timestamp"])
        frames[idx]["events"].append(e)
    frames[-1]["events"].append({"type": "GAME_END", "timestamp": end_ms,
                                 "winningTeam": 100 if blue_wins else 200})

    timeline = {
        "metadata": {"dataVersion": "2", "matchId": match_id, "participants": puuids},
        "info": {
            "frameInterval": 60000,
            "frames": frames,
            "gameId": int(match_id.split("_")[1]),
            "participants": [{"participantId": p, "puuid": puuids[p - 1]} for p in range(1, 11)],
        },
    }

    final = {p: pframe(p, end_ms / 60_000) for p in range(1, 11)}
    participants = []
    for p in range(1, 11):
        c = counts_until(p, end_ms)
        participants.append({
            "participantId": p,
            "puuid": puuids[p - 1],
            "riotIdGameName": f"Player{p}",
            "riotIdTagline": "NA1",
            "teamId": team_of(p),
            "teamPosition": role_of(p),
            "individualPosition": role_of(p),
            "championId": CHAMPS[p - 1],
            "championName": NAMES[p - 1],
            "win": (team_of(p) == 100) == blue_wins,
            "kills": c["kills"],
            "deaths": c["deaths"],
            "assists": c["assists"],
            "totalMinionsKilled": final[p]["minionsKilled"],
            "neutralMinionsKilled": final[p]["jungleMinionsKilled"],
            "goldEarned": final[p]["totalGold"],
            "totalDamageDealtToChampions": final[p]["damageStats"]["totalDamageDoneToChampions"],
            "visionScore": 10 + 3 * p,
            "wardsPlaced": c["wards"],
            "wardsKilled": c["wards_killed"],
            "detectorWardsPlaced": sum(
                1 for e in events if e["type"] == "WARD_PLACED" and e["creatorId"] == p
                and e["wardType"] == "CONTROL_WARD"),
            "champLevel": final[p]["level"],
            "timePlayed": duration_s,
            "gameEndedInEarlySurrender": False,
            "item0": 3031, "item1": 3006, "item2": 0, "item3": 0, "item4": 0, "item5": 0,
            "item6": 3340,
        })

    def team_obj(team_id: int) -> dict[str, Any]:
        def n(mtype: str) -> int:
            return sum(1 for e in events if e["type"] == "ELITE_MONSTER_KILL"
                       and e["killerTeamId"] == team_id and e["monsterType"] == mtype)
        return {
            "teamId": team_id,
            "win": (team_id == 100) == blue_wins,
            "bans": [],
            "objectives": {
                "baron": {"first": False, "kills": n("BARON_NASHOR")},
                "dragon": {"first": False, "kills": n("DRAGON")},
                "horde": {"first": False, "kills": n("HORDE")},
                "riftHerald": {"first": False, "kills": n("RIFTHERALD")},
                "champion": {"first": False, "kills": sum(
                    c["kills"] for p, c in ((p, counts_until(p, end_ms)) for p in range(1, 11))
                    if team_of(p) == team_id)},
                "tower": {"first": False, "kills": sum(
                    1 for e in events if e["type"] == "BUILDING_KILL"
                    and team_of(e["killerId"]) == team_id)},
                "inhibitor": {"first": False, "kills": 0},
            },
        }

    match = {
        "metadata": {"dataVersion": "2", "matchId": match_id, "participants": puuids},
        "info": {
            "gameCreation": start_ms - 60_000,
            "gameStartTimestamp": start_ms,
            "gameEndTimestamp": start_ms + end_ms,
            "gameDuration": duration_s,
            "gameId": int(match_id.split("_")[1]),
            "gameMode": "CLASSIC",
            "gameType": "MATCHED_GAME",
            "gameVersion": version,
            "mapId": 11,
            "platformId": match_id.split("_")[0],
            "queueId": queue,
            "participants": participants,
            "teams": [team_obj(100), team_obj(200)],
            "endOfGameResult": "GameComplete",
        },
    }
    return match, timeline
