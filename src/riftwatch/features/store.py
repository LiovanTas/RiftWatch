"""Persist extracted features, and extract whatever cached matches still need it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import astuple

import psycopg
from psycopg.types.json import Jsonb

from riftwatch.db import repo
from riftwatch.features.extract import EXTRACTOR_VERSION, GameFeatures, MinuteRow, extract

_MINUTE_COLUMNS = list(MinuteRow.__dataclass_fields__)


def save(conn: psycopg.Connection, game: GameFeatures) -> None:
    """Replace any stored features for this match with ``game``'s."""
    cols = ", ".join(_MINUTE_COLUMNS)
    placeholders = ", ".join(["%s"] * (len(_MINUTE_COLUMNS) + 2))
    with conn.transaction():
        conn.execute("DELETE FROM participant_minute_features WHERE match_id = %s", (game.match_id,))
        conn.execute("DELETE FROM participant_game_summary WHERE match_id = %s", (game.match_id,))
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO participant_minute_features (match_id, participant_id, {cols}) "
                f"VALUES ({placeholders})",
                [
                    (game.match_id, p.participant_id, *astuple(row))
                    for p in game.participants.values()
                    for row in p.minutes
                ],
            )
            cur.executemany(
                """
                INSERT INTO participant_game_summary (match_id, participant_id, role, champion_id,
                    opponent_participant_id, metrics, deaths_detail, extractor_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (game.match_id, p.participant_id, p.role, p.champion_id, p.opponent_id,
                     Jsonb(p.metrics), Jsonb(p.deaths_json()), EXTRACTOR_VERSION)
                    for p in game.participants.values()
                ],
            )


def pending_match_ids(conn: psycopg.Connection, limit: int | None = None) -> list[str]:
    """Cached matches with a timeline whose features are missing or out of date."""
    rows = conn.execute(
        """
        SELECT t.match_id FROM match_timelines t
         WHERE NOT EXISTS (
               SELECT 1 FROM participant_game_summary s
                WHERE s.match_id = t.match_id AND s.extractor_version = %s)
         ORDER BY t.match_id
         LIMIT %s
        """,
        (EXTRACTOR_VERSION, limit),
    )
    return [r[0] for r in rows]


def load(conn: psycopg.Connection, match_id: str) -> GameFeatures | None:
    """Extract from the cached raw JSON (cheap: no API, no stored-feature reads)."""
    match = repo.get_match(conn, match_id)
    timeline = repo.get_timeline(conn, match_id)
    if match is None or timeline is None:
        return None
    return extract(match, timeline)


def extract_pending(
    conn: psycopg.Connection,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[int, list[tuple[str, str]]]:
    """Extract and save features for every pending match.

    Returns (matches extracted, [(match_id, error)]). One malformed match must not stop a
    run over thousands, so failures are collected rather than raised.
    """
    done = 0
    failures: list[tuple[str, str]] = []
    ids = pending_match_ids(conn, limit)
    for i, match_id in enumerate(ids, 1):
        try:
            game = load(conn, match_id)
            if game is not None:
                save(conn, game)
                done += 1
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            failures.append((match_id, f"{type(exc).__name__}: {exc}"))
        if progress and (i % 50 == 0 or i == len(ids)):
            progress(f"features: {i}/{len(ids)}")
    return done, failures
