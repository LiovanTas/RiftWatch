"""Pull match-history rows out of stored match and timeline JSON.

``participant_builds`` holds, per player per game, everything a match-history row and the
champion statistics need: items, runes, summoner spells, CS, gold, damage, vision, multikills,
and from the timeline the skill order and the order items were bought in. ``match_bans``
holds each game's bans. New games are extracted as they are stored (``repo.insert_match`` and
``insert_timeline``); ``backfill`` fills in games stored before this table existed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import psycopg

SKILLS = "QWER"
_COLUMNS = (
    "match_id", "participant_id", "puuid", "riot_name", "riot_tag", "team_id", "position",
    "champion_id", "opponent_champion_id", "win", "remake", "queue_id", "patch", "game_start",
    "duration_s", "champ_level", "kills", "deaths", "assists", "team_kills", "cs", "gold",
    "damage", "damage_taken", "vision_score", "wards_placed", "wards_killed", "control_wards",
    "largest_multikill", "double_kills", "triple_kills", "quadra_kills", "penta_kills",
    "first_blood", "items", "spells", "keystone", "primary_style", "sub_style", "perks",
    "profile_icon", "summoner_level",
)
# The timeline events extraction needs, filtered inside Postgres so only they are sent.
EVENTS_PATH = ('$.info.frames[*].events[*] ? (@.type == "SKILL_LEVEL_UP" || '
               '@.type == "ITEM_PURCHASED" || @.type == "ITEM_UNDO")')


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _patch(version: str) -> str:
    parts = (version or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version or ""


def _start(info: dict) -> datetime:
    ms = info.get("gameStartTimestamp") or info.get("gameCreation") or 0
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _duration(info: dict) -> int:
    d = _int(info.get("gameDuration"))
    return d if "gameEndTimestamp" in info else d // 1000 if d > 100_000 else d


def participant_rows(raw: dict[str, Any]) -> list[tuple]:
    """One row per player, in ``_COLUMNS`` order."""
    info = raw["info"]
    match_id = raw["metadata"]["matchId"]
    parts = info.get("participants", [])
    team_kills: dict[int, int] = {}
    for p in parts:
        team_kills[p["teamId"]] = team_kills.get(p["teamId"], 0) + _int(p.get("kills"))
    by_slot = {(p["teamId"], p.get("teamPosition") or ""): p["championId"] for p in parts}
    start, duration = _start(info), _duration(info)
    rows = []
    for p in parts:
        position = p.get("teamPosition") or ""
        opponent = (by_slot.get((300 - p["teamId"], position)) if position else None)
        styles = (p.get("perks") or {}).get("styles") or []
        primary = styles[0] if styles else {}
        sub = styles[1] if len(styles) > 1 else {}
        perks = [s.get("perk") for st in styles for s in st.get("selections", [])]
        shards = (p.get("perks") or {}).get("statPerks") or {}
        perks += [shards[k] for k in ("offense", "flex", "defense") if k in shards]
        rows.append((
            match_id, p["participantId"], p["puuid"],
            p.get("riotIdGameName") or p.get("summonerName"), p.get("riotIdTagline"),
            p["teamId"], position, p["championId"], opponent, bool(p.get("win")),
            bool(p.get("gameEndedInEarlySurrender")), info.get("queueId", 0),
            _patch(info.get("gameVersion", "")), start, duration,
            _int(p.get("champLevel")), _int(p.get("kills")), _int(p.get("deaths")),
            _int(p.get("assists")), team_kills.get(p["teamId"], 0),
            _int(p.get("totalMinionsKilled")) + _int(p.get("neutralMinionsKilled")),
            _int(p.get("goldEarned")),
            _int(p.get("totalDamageDealtToChampions")
                 or (p.get("damageStats") or {}).get("totalDamageDoneToChampions")),
            _int(p.get("totalDamageTaken")), _int(p.get("visionScore")),
            _int(p.get("wardsPlaced")), _int(p.get("wardsKilled")),
            _int(p.get("visionWardsBoughtInGame")), _int(p.get("largestMultiKill")),
            _int(p.get("doubleKills")), _int(p.get("tripleKills")), _int(p.get("quadraKills")),
            _int(p.get("pentaKills")), bool(p.get("firstBloodKill")),
            [_int(p.get(f"item{i}")) for i in range(7)],
            [_int(p.get("summoner1Id")), _int(p.get("summoner2Id"))],
            (primary.get("selections") or [{}])[0].get("perk"), primary.get("style"),
            sub.get("style"), [x for x in perks if x is not None] or None,
            p.get("profileIcon"), p.get("summonerLevel"),
        ))
    return rows


def ban_rows(raw: dict[str, Any]) -> list[tuple]:
    match_id = raw["metadata"]["matchId"]
    out = []
    for team in raw["info"].get("teams", []):
        for b in team.get("bans", []):
            if _int(b.get("championId")) > 0:
                out.append((match_id, team["teamId"], _int(b.get("pickTurn")), b["championId"]))
    return out


def timeline_rows(events: list[dict]) -> dict[int, tuple[str, list[int], list[int]]]:
    """participant id -> (skill order, purchases, purchase seconds) from timeline events."""
    skills: dict[int, list[str]] = {}
    bought: dict[int, list[tuple[int, int]]] = {}
    for e in sorted(events, key=lambda e: e.get("timestamp", 0)):
        pid = _int(e.get("participantId"))
        kind = e.get("type")
        if kind == "SKILL_LEVEL_UP" and e.get("levelUpType", "NORMAL") == "NORMAL":
            slot = _int(e.get("skillSlot"))
            if 1 <= slot <= 4:
                skills.setdefault(pid, []).append(SKILLS[slot - 1])
        elif kind == "ITEM_PURCHASED":
            bought.setdefault(pid, []).append((_int(e.get("itemId")), _int(e.get("timestamp")) // 1000))
        elif kind == "ITEM_UNDO":
            mine = bought.get(pid)
            if mine and mine[-1][0] == _int(e.get("beforeId")):
                mine.pop()
    out = {}
    for pid in set(skills) | set(bought):
        b = bought.get(pid, [])
        out[pid] = ("".join(skills.get(pid, [])), [i for i, _ in b], [s for _, s in b])
    return out


def save_match(conn: psycopg.Connection, raw: dict[str, Any]) -> None:
    rows = participant_rows(raw)
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO participant_builds ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join(['%s'] * len(_COLUMNS))}) ON CONFLICT DO NOTHING", rows)
        cur.executemany("INSERT INTO match_bans VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                        ban_rows(raw))


def save_timeline(conn: psycopg.Connection, match_id: str, events: list[dict]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """UPDATE participant_builds SET skill_order = %s, purchases = %s, purchase_s = %s
                WHERE match_id = %s AND participant_id = %s""",
            [(skill, items, secs, match_id, pid)
             for pid, (skill, items, secs) in timeline_rows(events).items()])


def timeline_events(raw_timeline: dict[str, Any]) -> list[dict]:
    return [e for f in raw_timeline.get("info", {}).get("frames", [])
            for e in f.get("events", [])
            if e.get("type") in ("SKILL_LEVEL_UP", "ITEM_PURCHASED", "ITEM_UNDO")]


def backfill(conn: psycopg.Connection, progress: Callable[[str], None] | None = None,
             batch: int = 200) -> tuple[int, int]:
    """Extract every stored game (and timeline) that isn't extracted yet. Returns how many
    games and timelines were added."""
    games = timelines = 0
    while True:
        rows = conn.execute(
            """SELECT m.raw FROM matches m
                WHERE NOT EXISTS (SELECT 1 FROM participant_builds b WHERE b.match_id = m.match_id)
                LIMIT %s""", (batch,)).fetchall()
        if not rows:
            break
        with conn.transaction():
            for (raw,) in rows:
                save_match(conn, raw)
        games += len(rows)
        if progress:
            progress(f"{games} games extracted")
    while True:
        ids = [r[0] for r in conn.execute(
            """SELECT DISTINCT b.match_id FROM participant_builds b
                 JOIN match_timelines t USING (match_id)
                WHERE b.skill_order IS NULL AND b.participant_id = 1 LIMIT %s""",
            (batch,)).fetchall()]
        if not ids:
            break
        found = conn.execute(
            "SELECT match_id, jsonb_path_query_array(raw, %s) FROM match_timelines "
            "WHERE match_id = ANY(%s)", (EVENTS_PATH, ids)).fetchall()
        with conn.transaction():
            for match_id, events in found:
                save_timeline(conn, match_id, events or [])
                # A timeline with no events for participant 1 would be picked again forever.
                conn.execute("UPDATE participant_builds SET skill_order = coalesce(skill_order, '') "
                             "WHERE match_id = %s", (match_id,))
        timelines += len(found)
        if progress:
            progress(f"{timelines} timelines extracted")
    return games, timelines
