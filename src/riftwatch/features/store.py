"""Persist extracted features, extract whatever cached matches still need it, and read
features back without touching the raw Riot JSON.

Write path: raw match + timeline -> :func:`extract` -> :func:`save` (once per match).
Read path:  :func:`load_stored` rebuilds :class:`GameFeatures` from the indexed feature
tables -- a few small queries instead of decoding ~200 KB of JSON per game.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import astuple

import psycopg
from psycopg.types.json import Jsonb

from riftwatch.db import repo
from riftwatch.features.extract import (
    EXTRACTOR_VERSION,
    Death,
    GameFeatures,
    MinuteRow,
    ParticipantFeatures,
    extract,
)

_MINUTE_COLUMNS = list(MinuteRow.__dataclass_fields__)


def save(conn: psycopg.Connection, game: GameFeatures) -> None:
    """Replace any stored features for this match with ``game``'s."""
    cols = ", ".join(_MINUTE_COLUMNS)
    with conn.transaction():
        conn.execute("DELETE FROM participant_minute_features WHERE match_id = %s", (game.match_id,))
        conn.execute("DELETE FROM participant_game_summary WHERE match_id = %s", (game.match_id,))
        with conn.cursor() as cur:
            # COPY streams all ~270 minute rows in one operation; executemany would make a
            # round trip per row.
            with cur.copy(
                f"COPY participant_minute_features (match_id, participant_id, {cols}) FROM STDIN"
            ) as copy:
                for p in game.participants.values():
                    for row in p.minutes:
                        copy.write_row((game.match_id, p.participant_id, *astuple(row)))
            cur.executemany(
                """
                INSERT INTO participant_game_summary (match_id, participant_id, role, champion_id,
                    champion_name, opponent_participant_id, metrics, deaths_detail,
                    extractor_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (game.match_id, p.participant_id, p.role, p.champion_id, p.champion_name,
                     p.opponent_id, Jsonb(p.metrics), Jsonb(p.deaths_json()), EXTRACTOR_VERSION)
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
    """Extract from the cached raw JSON. The write path; readers use :func:`load_stored`."""
    match = repo.get_match(conn, match_id)
    timeline = repo.get_timeline(conn, match_id)
    if match is None or timeline is None:
        return None
    return extract(match, timeline)


def load_stored(conn: psycopg.Connection, match_id: str) -> GameFeatures | None:
    """Rebuild one game's features from the feature tables (no raw JSON). None if the
    match hasn't been extracted with the current extractor."""
    return load_stored_many(conn, [match_id]).get(match_id)


def load_stored_many(
    conn: psycopg.Connection, match_ids: list[str], minutes_for: str | None = None,
) -> dict[str, GameFeatures]:
    """Rebuild several games at once: three queries in total, however many games -- the
    per-query round trip, not the data, dominates reading a 20-game history.

    ``minutes_for`` (a PUUID) loads per-minute rows for that player only; the other nine
    keep their summaries but no minute rows. A history view scores one player, and the
    minute rows are ~90% of the data.
    """
    if not match_ids:
        return {}
    heads = {
        mid: (patch, queue_id, duration_s)
        for mid, patch, queue_id, duration_s in conn.execute(
            "SELECT match_id, patch, queue_id, duration_s FROM matches WHERE match_id = ANY(%s)",
            (match_ids,),
        )
    }
    players: dict[str, dict[int, ParticipantFeatures]] = {}
    stale: set[str] = set()
    for (mid, pid, puuid, team, role, champ, champ_name, win, opp, metrics, deaths,
         version) in conn.execute(
        """
        SELECT s.match_id, s.participant_id, p.puuid, p.team_id, s.role, s.champion_id,
               s.champion_name, p.win, s.opponent_participant_id, s.metrics, s.deaths_detail,
               s.extractor_version
          FROM participant_game_summary s
          JOIN match_participants p USING (match_id, participant_id)
         WHERE s.match_id = ANY(%s)
        """,
        (match_ids,),
    ):
        if version != EXTRACTOR_VERSION:
            stale.add(mid)
        players.setdefault(mid, {})[pid] = ParticipantFeatures(
            participant_id=pid, puuid=puuid, team_id=team, role=role, champion_id=champ,
            champion_name=champ_name, win=win, opponent_id=opp, metrics=metrics,
            deaths=[Death(**d) for d in deaths],
        )
    wanted = [m for m in players if m not in stale and m in heads]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT f.match_id, f.participant_id, {", ".join("f." + c for c in _MINUTE_COLUMNS)}
              FROM participant_minute_features f
              JOIN match_participants p USING (match_id, participant_id)
             WHERE f.match_id = ANY(%s) AND (%s::text IS NULL OR p.puuid = %s)
             ORDER BY f.match_id, f.participant_id, f.minute
            """,
            (wanted, minutes_for, minutes_for),
        )
        for row in cur:
            players[row[0]][row[1]].minutes.append(MinuteRow(*row[2:]))
    return {
        mid: GameFeatures(mid, *heads[mid], dict(sorted(players[mid].items())))
        for mid in wanted
    }


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
