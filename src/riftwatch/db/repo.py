"""SQL for the Riot data cache. Plain functions taking a connection; no ORM."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from riftwatch.riot.ddragon import patch_of

APEX = ("MASTER", "GRANDMASTER", "CHALLENGER")


def tier_bucket(tier: str) -> str:
    """Baselines group Master, Grandmaster and Challenger together: too few players apart."""
    tier = tier.upper()
    return "MASTER_PLUS" if tier in APEX else tier


# -- accounts -----------------------------------------------------------------------------

def upsert_account(conn: psycopg.Connection, account: dict[str, Any], platform: str) -> None:
    conn.execute(
        """
        INSERT INTO accounts (puuid, game_name, tag_line, platform)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (puuid) DO UPDATE
           SET game_name = EXCLUDED.game_name, tag_line = EXCLUDED.tag_line,
               platform = EXCLUDED.platform, updated_at = now()
        """,
        (account["puuid"], account["gameName"], account["tagLine"], platform),
    )


def find_account(conn: psycopg.Connection, game_name: str, tag_line: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT puuid, game_name, tag_line, platform FROM accounts
         WHERE lower(game_name) = lower(%s) AND lower(tag_line) = lower(%s)
         ORDER BY updated_at DESC LIMIT 1
        """,
        (game_name, tag_line),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(("puuid", "game_name", "tag_line", "platform"), row, strict=True))


# -- ranks --------------------------------------------------------------------------------

def insert_rank_snapshots(conn: psycopg.Connection, puuid: str, entries: list[dict[str, Any]]) -> None:
    for e in entries:
        conn.execute(
            """
            INSERT INTO rank_snapshots (puuid, queue, tier, division, lp, wins, losses)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                puuid,
                e["queueType"],
                e["tier"],
                None if e["tier"] in APEX else e.get("rank"),
                e.get("leaguePoints", 0),
                e.get("wins", 0),
                e.get("losses", 0),
            ),
        )


def latest_rank(conn: psycopg.Connection, puuid: str, queue: str = "RANKED_SOLO_5x5") -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT tier, division, lp, wins, losses, captured_at FROM rank_snapshots
         WHERE puuid = %s AND queue = %s ORDER BY captured_at DESC LIMIT 1
        """,
        (puuid, queue),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(("tier", "division", "lp", "wins", "losses", "captured_at"), row, strict=True))


# -- matches ------------------------------------------------------------------------------

def known_match_ids(conn: psycopg.Connection, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    rows = conn.execute("SELECT match_id FROM matches WHERE match_id = ANY(%s)", (ids,))
    return {r[0] for r in rows}


def known_timeline_ids(conn: psycopg.Connection, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    rows = conn.execute("SELECT match_id FROM match_timelines WHERE match_id = ANY(%s)", (ids,))
    return {r[0] for r in rows}


def get_match(conn: psycopg.Connection, match_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT raw FROM matches WHERE match_id = %s", (match_id,)).fetchone()
    return row[0] if row else None


def get_timeline(conn: psycopg.Connection, match_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT raw FROM match_timelines WHERE match_id = %s", (match_id,)
    ).fetchone()
    return row[0] if row else None


def _game_start(info: dict[str, Any]) -> datetime:
    ms = info.get("gameStartTimestamp") or info["gameCreation"]
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _duration_s(info: dict[str, Any]) -> int:
    # Since patch 11.20 gameDuration is seconds; before that it was milliseconds and
    # gameEndTimestamp was absent.
    if "gameEndTimestamp" in info:
        return int(info["gameDuration"])
    return int(info["gameDuration"] // 1000)


def insert_match(conn: psycopg.Connection, raw: dict[str, Any]) -> None:
    """Store a match-v5 response and its participant rows. No-op if already stored."""
    info = raw["info"]
    match_id = raw["metadata"]["matchId"]
    with conn.transaction():
        cur = conn.execute(
            """
            INSERT INTO matches (match_id, platform, queue_id, game_version, patch,
                                 game_start, duration_s, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (match_id) DO NOTHING
            """,
            (
                match_id,
                info["platformId"].lower(),
                info["queueId"],
                info["gameVersion"],
                patch_of(info["gameVersion"]),
                _game_start(info),
                _duration_s(info),
                Jsonb(raw),
            ),
        )
        if cur.rowcount == 0:
            return
        with conn.cursor() as c:
            c.executemany(
                """
                INSERT INTO match_participants (match_id, participant_id, puuid, team_id,
                    team_position, champion_id, win, kills, deaths, assists)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        match_id,
                        p["participantId"],
                        p["puuid"],
                        p["teamId"],
                        p.get("teamPosition") or "",
                        p["championId"],
                        p["win"],
                        p["kills"],
                        p["deaths"],
                        p["assists"],
                    )
                    for p in info["participants"]
                ],
            )


def insert_timeline(conn: psycopg.Connection, match_id: str, raw: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO match_timelines (match_id, raw) VALUES (%s, %s)
        ON CONFLICT (match_id) DO NOTHING
        """,
        (match_id, Jsonb(raw)),
    )


def mark_sample(conn: psycopg.Connection, match_id: str, bucket: str, source: str) -> None:
    """Assign a match to a rank bucket. First assignment wins."""
    conn.execute(
        """
        INSERT INTO match_samples (match_id, tier_bucket, source) VALUES (%s, %s, %s)
        ON CONFLICT (match_id) DO NOTHING
        """,
        (match_id, bucket, source),
    )


def player_match_ids(
    conn: psycopg.Connection, puuid: str, *, queue_id: int | None = None, limit: int = 20
) -> list[str]:
    """This player's cached matches, newest first."""
    rows = conn.execute(
        """
        SELECT m.match_id FROM match_participants p JOIN matches m USING (match_id)
         WHERE p.puuid = %s AND (%s::int IS NULL OR m.queue_id = %s)
         ORDER BY m.game_start DESC LIMIT %s
        """,
        (puuid, queue_id, queue_id, limit),
    )
    return [r[0] for r in rows]


# -- cache counters -----------------------------------------------------------------------

def bump_cache(conn: psycopg.Connection, kind: str, hit: bool) -> None:
    bump_cache_many(conn, kind, int(hit), int(not hit))


def bump_cache_many(conn: psycopg.Connection, kind: str, hits: int, misses: int) -> None:
    if not hits and not misses:
        return
    conn.execute(
        """
        INSERT INTO cache_counters (kind, hits, misses) VALUES (%s, %s, %s)
        ON CONFLICT (kind) DO UPDATE
           SET hits = cache_counters.hits + EXCLUDED.hits,
               misses = cache_counters.misses + EXCLUDED.misses
        """,
        (kind, hits, misses),
    )


def cache_stats(conn: psycopg.Connection) -> dict[str, Any]:
    counters = {
        kind: {"hits": hits, "misses": misses}
        for kind, hits, misses in conn.execute("SELECT kind, hits, misses FROM cache_counters")
    }
    totals = conn.execute(
        """
        SELECT (SELECT count(*) FROM matches),
               (SELECT count(*) FROM match_timelines),
               (SELECT count(DISTINCT puuid) FROM match_participants),
               (SELECT pg_total_relation_size('matches') + pg_total_relation_size('match_timelines'))
        """
    ).fetchone()
    return {
        "counters": counters,
        "matches": totals[0],
        "timelines": totals[1],
        "players_seen": totals[2],
        "raw_bytes": totals[3],
    }
