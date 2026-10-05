"""Build the training tables: one row per (high-elo player, minute), one table per role.

Games are selected through ``crawl_player_matches`` -- games found via a player on the
Challenger / Grandmaster (optionally Master) ladder. All ten players in a selected game
are used: in a Challenger lobby everyone is high elo.

The table goes to a Parquet file, so training reads it in one call instead of re-parsing
timelines.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import psycopg

from riftwatch.db import repo
from riftwatch.ml.situations import ROLES, examples

DEFAULT_TIERS = ("CHALLENGER", "GRANDMASTER")


@dataclass
class DatasetReport:
    games: int
    skipped: int
    rows: dict[str, int]          # role -> rows
    paths: dict[str, Path]


def high_elo_match_ids(conn: psycopg.Connection, tiers: Iterable[str] = DEFAULT_TIERS,
                       queue_id: int = 420, min_duration_s: int = 15 * 60) -> list[str]:
    """Ranked solo games found through a player in ``tiers``, long enough to have a full
    early game (remakes and fast surrenders teach nothing about pathing)."""
    rows = conn.execute(
        """
        SELECT DISTINCT c.match_id FROM crawl_player_matches c
          JOIN matches m USING (match_id)
          JOIN match_timelines t USING (match_id)
         WHERE c.tier = ANY(%s) AND m.queue_id = %s AND m.duration_s >= %s
         ORDER BY c.match_id
        """,
        ([t.upper() for t in tiers], queue_id, min_duration_s),
    )
    return [r[0] for r in rows]


def build(
    conn: psycopg.Connection,
    out_dir: Path,
    *,
    roles: Iterable[str] = ROLES,
    tiers: Iterable[str] = DEFAULT_TIERS,
    progress: Callable[[str], None] | None = None,
) -> DatasetReport:
    """One pass over the games, writing ``out_dir/<ROLE>.parquet`` for each role."""
    import pandas as pd

    roles = list(roles)
    ids = high_elo_match_ids(conn, tiers)
    rows: dict[str, list[dict]] = {r: [] for r in roles}
    skipped = 0
    for i, match_id in enumerate(ids, 1):
        match, timeline = repo.get_match(conn, match_id), repo.get_timeline(conn, match_id)
        try:
            game_rows = {r: [e.row() for e in examples(match, timeline, r)] for r in roles}
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            skipped += 1        # a malformed game shouldn't sink a build over thousands
            continue
        for r in roles:
            rows[r] += game_rows[r]
        if progress and (i % 200 == 0 or i == len(ids)):
            progress(f"dataset: {i}/{len(ids)} games")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for r in roles:
        paths[r] = out_dir / f"{r}.parquet"
        pd.DataFrame(rows[r]).to_parquet(paths[r], index=False)
    return DatasetReport(len(ids) - skipped, skipped, {r: len(rows[r]) for r in roles}, paths)
