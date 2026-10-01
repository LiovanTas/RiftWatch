"""Turn a match-v5 match + timeline into per-minute rows and whole-game metrics.

Pure functions over Riot's JSON -- no database, no network -- so every number the coach
later cites can be traced back to a line here and tested against a fixture.

Timeline shape, briefly: ``info.frames`` holds one snapshot per minute (timestamps a few ms
past each minute mark) plus a final snapshot at the moment the game ended. Each frame's
``events`` are the things that happened since the previous frame. Participant snapshots
are cumulative (total gold, total CS...).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from riftwatch.features.map import readable, zone
from riftwatch.riot.ddragon import patch_of

# Bump when extraction logic changes; stored rows with an older version get re-extracted.
EXTRACTOR_VERSION = 2

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
EPIC_MONSTERS = ("DRAGON", "BARON_NASHOR", "RIFTHERALD", "HORDE", "ATAKHAN", "ELDER_DRAGON")
EARLY_GAME_END_MIN = 14     # when turret plates fall; "early" deaths are before this
AHEAD_GOLD = 500            # gold lead over the lane opponent that counts as "ahead"
# Minute frames land on or just after each minute mark, never before it. The tolerance is
# one-sided: a game ending at 21:58 has a final frame 2 s *before* 22:00, which is not a
# minute-22 snapshot.
MINUTE_TOLERANCE_MS = 5_000


@dataclass
class MinuteRow:
    minute: int
    gold: int
    current_gold: int
    xp: int
    level: int
    cs: int
    lane_cs: int
    jungle_cs: int
    damage_to_champions: int
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    wards_placed: int = 0
    wards_killed: int = 0
    gold_diff: int | None = None
    xp_diff: int | None = None
    cs_diff: int | None = None
    x: int | None = None
    y: int | None = None


@dataclass
class Death:
    timestamp_ms: int
    minute: float
    zone: str
    where: str                  # human-readable, relative to the player's team
    killer_participant_id: int  # 0 = executed by tower/minions/monster
    assisters: int
    gold_diff: int | None       # vs lane opponent at the last full minute before the death
    ahead: bool                 # gold_diff >= AHEAD_GOLD
    early: bool                 # before EARLY_GAME_END_MIN


@dataclass
class ParticipantFeatures:
    participant_id: int
    puuid: str
    team_id: int
    role: str
    champion_id: int
    champion_name: str
    win: bool
    opponent_id: int | None
    minutes: list[MinuteRow] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    deaths: list[Death] = field(default_factory=list)

    def at(self, minute: int) -> MinuteRow | None:
        return self.minutes[minute] if 0 <= minute < len(self.minutes) else None

    def deaths_json(self) -> list[dict[str, Any]]:
        return [asdict(d) for d in self.deaths]


@dataclass
class GameFeatures:
    match_id: str
    patch: str
    queue_id: int
    duration_s: int
    participants: dict[int, ParticipantFeatures]

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60

    def by_puuid(self, puuid: str) -> ParticipantFeatures | None:
        return next((p for p in self.participants.values() if p.puuid == puuid), None)


def _minute_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The frames that sit on a minute mark, indexed by minute. Drops the final
    end-of-game frame (it sits mid-minute) and fills any missing minute with the previous
    frame so ``result[m]`` is always "state at minute m"."""
    by_minute: dict[int, dict[str, Any]] = {}
    for frame in frames:
        ts = frame["timestamp"]
        minute = ts // 60_000
        if ts - minute * 60_000 <= MINUTE_TOLERANCE_MS and minute not in by_minute:
            by_minute[minute] = frame
    if not by_minute:
        return []
    out = []
    for m in range(max(by_minute) + 1):
        out.append(by_minute.get(m, out[-1] if out else frames[0]))
    return out


def _all_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = [e for f in frames for e in f.get("events", [])]
    events.sort(key=lambda e: e["timestamp"])
    return events


def _lane_opponents(participants: list[dict[str, Any]]) -> dict[int, int | None]:
    """participantId -> enemy with the same teamPosition, if exactly one."""
    out: dict[int, int | None] = {}
    for p in participants:
        role = p.get("teamPosition") or ""
        rivals = [q for q in participants
                  if q["teamId"] != p["teamId"] and role and (q.get("teamPosition") or "") == role]
        out[p["participantId"]] = rivals[0]["participantId"] if len(rivals) == 1 else None
    return out


def _cs(pf: dict[str, Any]) -> tuple[int, int]:
    return pf.get("minionsKilled", 0), pf.get("jungleMinionsKilled", 0)


def extract(match: dict[str, Any], timeline: dict[str, Any]) -> GameFeatures:
    info = match["info"]
    match_id = match["metadata"]["matchId"]
    duration_s = int(info["gameDuration"]) if "gameEndTimestamp" in info else int(info["gameDuration"] // 1000)
    frames = timeline["info"]["frames"]
    minute_frames = _minute_frames(frames)
    events = _all_events(frames)
    opponents = _lane_opponents(info["participants"])

    players: dict[int, ParticipantFeatures] = {}
    for p in info["participants"]:
        pid = p["participantId"]
        players[pid] = ParticipantFeatures(
            participant_id=pid,
            puuid=p["puuid"],
            team_id=p["teamId"],
            role=p.get("teamPosition") or "",
            champion_id=p["championId"],
            champion_name=p.get("championName", ""),
            win=bool(p["win"]),
            opponent_id=opponents[pid],
        )

    # -- per-minute snapshots -------------------------------------------------------------
    for minute, frame in enumerate(minute_frames):
        for pid, pf in frame["participantFrames"].items():
            pid = int(pid)
            if pid not in players:
                continue
            lane, jungle = _cs(pf)
            pos = pf.get("position") or {}
            players[pid].minutes.append(MinuteRow(
                minute=minute,
                gold=pf.get("totalGold", 0),
                current_gold=pf.get("currentGold", 0),
                xp=pf.get("xp", 0),
                level=pf.get("level", 1),
                cs=lane + jungle,
                lane_cs=lane,
                jungle_cs=jungle,
                damage_to_champions=(pf.get("damageStats") or {}).get("totalDamageDoneToChampions", 0),
                x=pos.get("x"),
                y=pos.get("y"),
            ))

    # -- cumulative event counters per minute -----------------------------------------------
    # One pass over the (already sorted) events collects each player's timestamps per
    # counter; a minute's count is then a binary search for its frame time. Linear in
    # events, instead of touching every later minute for every event.
    stamps: dict[tuple[int, str], list[int]] = defaultdict(list)
    for e in events:
        ts = e["timestamp"]
        etype = e["type"]
        if etype == "CHAMPION_KILL":
            stamps[e.get("killerId"), "kills"].append(ts)
            stamps[e.get("victimId"), "deaths"].append(ts)
            for a in e.get("assistingParticipantIds") or []:
                stamps[a, "assists"].append(ts)
        elif etype == "WARD_PLACED":
            stamps[e.get("creatorId"), "wards_placed"].append(ts)
        elif etype == "WARD_KILL":
            stamps[e.get("killerId"), "wards_killed"].append(ts)
    frame_times = [f["timestamp"] for f in minute_frames]
    for (pid, attr), times in stamps.items():
        if pid not in players:
            continue
        for row in players[pid].minutes:
            setattr(row, attr, bisect_right(times, frame_times[row.minute]))

    # -- lane-opponent diffs ------------------------------------------------------------------
    for p in players.values():
        opp = players.get(p.opponent_id) if p.opponent_id else None
        if opp is None:
            continue
        for row, orow in zip(p.minutes, opp.minutes, strict=False):
            row.gold_diff = row.gold - orow.gold
            row.xp_diff = row.xp - orow.xp
            row.cs_diff = row.cs - orow.cs

    # -- deaths -----------------------------------------------------------------------------
    for e in events:
        if e["type"] != "CHAMPION_KILL" or e.get("victimId") not in players:
            continue
        victim = players[e["victimId"]]
        pos = e.get("position") or {"x": 0, "y": 0}
        z = zone(pos["x"], pos["y"])
        last_full = victim.at(min(int(e["timestamp"] // 60_000), len(victim.minutes) - 1))
        gold_diff = last_full.gold_diff if last_full else None
        victim.deaths.append(Death(
            timestamp_ms=e["timestamp"],
            minute=round(e["timestamp"] / 60_000, 2),
            zone=z,
            where=readable(z, victim.team_id),
            killer_participant_id=e.get("killerId") or 0,
            assisters=len(e.get("assistingParticipantIds") or []),
            gold_diff=gold_diff,
            ahead=gold_diff is not None and gold_diff >= AHEAD_GOLD,
            early=e["timestamp"] < EARLY_GAME_END_MIN * 60_000,
        ))

    # -- whole-game metrics ---------------------------------------------------------------------
    team_kills = {100: 0, 200: 0}
    team_damage = {100: 0, 200: 0}
    for p in info["participants"]:
        team_kills[p["teamId"]] += p["kills"]
        team_damage[p["teamId"]] += p.get("totalDamageDealtToChampions", 0)

    team_epics = {100: [], 200: []}
    team_towers = {100: [], 200: []}
    for e in events:
        if e["type"] == "ELITE_MONSTER_KILL" and e.get("monsterType") in EPIC_MONSTERS:
            team = e.get("killerTeamId")
            if team in team_epics:
                team_epics[team].append(e)
        elif e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING":
            # teamId is the team that LOST the tower.
            team = 200 if e.get("teamId") == 100 else 100
            team_towers[team].append(e)

    # Riot omits assistingParticipantIds entirely (never sends []) when nobody assisted.
    def took_part(pid: int, e: dict[str, Any]) -> bool:
        return e.get("killerId") == pid or pid in (e.get("assistingParticipantIds") or [])

    minutes_played = max(duration_s / 60, 1e-9)
    for p in info["participants"]:
        pid = p["participantId"]
        f = players[pid]
        cs_total = p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)
        m: dict[str, float] = {
            "kills": p["kills"],
            "deaths": p["deaths"],
            "assists": p["assists"],
            "kda": (p["kills"] + p["assists"]) / max(p["deaths"], 1),
            "cs_per_min": cs_total / minutes_played,
            "gold_per_min": p.get("goldEarned", 0) / minutes_played,
            "damage_per_min": p.get("totalDamageDealtToChampions", 0) / minutes_played,
            "damage_share": (p.get("totalDamageDealtToChampions", 0) / team_damage[f.team_id]
                             if team_damage[f.team_id] else 0.0),
            "kill_participation": ((p["kills"] + p["assists"]) / team_kills[f.team_id]
                                   if team_kills[f.team_id] else 0.0),
            "vision_per_min": p.get("visionScore", 0) / minutes_played,
            "wards_placed_per_min": p.get("wardsPlaced", 0) / minutes_played,
            "wards_killed": p.get("wardsKilled", 0),
            "control_wards": p.get("detectorWardsPlaced", 0),
            "early_deaths": sum(d.early for d in f.deaths),
            "deaths_while_ahead": sum(d.ahead for d in f.deaths),
            "solo_kills": sum(
                1 for e in events if e["type"] == "CHAMPION_KILL" and e.get("killerId") == pid
                and not e.get("assistingParticipantIds")),
            "solo_deaths": sum(1 for d in f.deaths if d.assisters == 0 and d.killer_participant_id),
            "turret_plates": sum(
                1 for e in events if e["type"] == "TURRET_PLATE_DESTROYED" and e.get("killerId") == pid),
        }
        if team_epics[f.team_id]:
            m["objective_participation"] = (
                sum(took_part(pid, e) for e in team_epics[f.team_id]) / len(team_epics[f.team_id]))
        if team_towers[f.team_id]:
            m["tower_participation"] = (
                sum(took_part(pid, e) for e in team_towers[f.team_id]) / len(team_towers[f.team_id]))
        if f.deaths:
            m["first_death_min"] = f.deaths[0].minute
        for checkpoint in (10, 15, 20):
            row = f.at(checkpoint)
            if row is None:
                continue
            m[f"cs_at_{checkpoint}"] = row.cs
            m[f"gold_at_{checkpoint}"] = row.gold
            m[f"xp_at_{checkpoint}"] = row.xp
            if row.gold_diff is not None:
                m[f"gold_diff_at_{checkpoint}"] = row.gold_diff
                m[f"xp_diff_at_{checkpoint}"] = row.xp_diff
                m[f"cs_diff_at_{checkpoint}"] = row.cs_diff
        f.metrics = {k: round(float(v), 4) for k, v in m.items()}

    return GameFeatures(
        match_id=match_id,
        patch=patch_of(info["gameVersion"]),
        queue_id=info["queueId"],
        duration_s=duration_s,
        participants=players,
    )
