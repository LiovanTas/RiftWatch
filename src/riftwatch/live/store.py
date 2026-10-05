"""Import live recordings into Postgres and link them to Riot matches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from riftwatch.live.analyze import analyze
from riftwatch.live.recorder import read

LINK_WINDOW = timedelta(minutes=15)


@dataclass
class ImportReport:
    imported: int
    linked: int
    skipped: list[tuple[str, str]]


def _game_start(head: dict[str, Any], samples: list[dict[str, Any]]) -> datetime:
    # recorded_at is the wall clock when the first sample was taken, at game time t0.
    t0 = samples[0]["t"] if samples else 0.0
    return datetime.fromtimestamp(head.get("recorded_at", 0) - t0, tz=UTC)


def _hp_series(samples: list[dict[str, Any]], step: float = 2.0) -> list[list[float]]:
    out, next_t = [], -1e9
    for s in samples:
        if s.get("hp_max") and s["t"] >= next_t:
            out.append([round(s["t"], 1), round(s["hp"] / s["hp_max"], 3)])
            next_t = s["t"] + step
    return out


def link(conn: psycopg.Connection, riot_id: str, champion: str, game_start: datetime) -> str | None:
    """The synced match this recording belongs to: same player and champion, starting
    closest in time (within LINK_WINDOW)."""
    name, _, tag = riot_id.partition("#")
    row = conn.execute(
        """
        SELECT m.match_id FROM accounts a
          JOIN match_participants p ON p.puuid = a.puuid
          JOIN matches m USING (match_id)
          LEFT JOIN participant_game_summary s USING (match_id, participant_id)
         WHERE lower(a.game_name) = lower(%s) AND lower(a.tag_line) = lower(%s)
           AND m.game_start BETWEEN %s AND %s
           AND (s.champion_name IS NULL OR lower(s.champion_name) = lower(%s)
                OR lower(replace(s.champion_name, ' ', '')) = lower(replace(%s, ' ', '')))
         ORDER BY abs(extract(epoch FROM m.game_start - %s))
         LIMIT 1
        """,
        (name, tag, game_start - LINK_WINDOW, game_start + LINK_WINDOW, champion, champion,
         game_start),
    ).fetchone()
    return row[0] if row else None


def import_dir(conn: psycopg.Connection, directory: Path) -> ImportReport:
    """Import every recording not imported yet, then try to link any still unlinked (the
    match may not have been synced when the recording was first imported)."""
    imported = linked = 0
    skipped: list[tuple[str, str]] = []
    known = {r[0] for r in conn.execute("SELECT file FROM live_recordings")}
    for path in sorted(directory.glob("*.jsonl.gz")) if directory.exists() else []:
        if path.name in known:
            continue
        head, samples, _events = read(path)
        if not head or len(samples) < 60:
            skipped.append((path.name, "too short or no header"))
            continue
        if head.get("game_mode") not in ("CLASSIC", ""):
            skipped.append((path.name, f"mode {head.get('game_mode')}"))
            continue
        start = _game_start(head, samples)
        summary = analyze(head, samples)
        conn.execute(
            """
            INSERT INTO live_recordings (file, riot_id, champion, position, game_start,
                                         duration_s, match_id, summary, hp_series)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (path.name, head["riot_id"], head.get("champion", ""), head.get("position", ""),
             start, summary.duration_s, link(conn, head["riot_id"], head.get("champion", ""), start),
             Jsonb(summary.to_json()), Jsonb(_hp_series(samples))),
        )
        imported += 1
    for rec_id, riot_id, champion, start in conn.execute(
        "SELECT id, riot_id, champion, game_start FROM live_recordings WHERE match_id IS NULL"
    ).fetchall():
        match_id = link(conn, riot_id, champion, start)
        if match_id:
            conn.execute("UPDATE live_recordings SET match_id = %s WHERE id = %s", (match_id, rec_id))
    linked = conn.execute("SELECT count(*) FROM live_recordings WHERE match_id IS NOT NULL").fetchone()[0]
    return ImportReport(imported, linked, skipped)


def for_match(conn: psycopg.Connection, match_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT summary, hp_series, champion FROM live_recordings WHERE match_id = %s "
        "ORDER BY imported_at DESC LIMIT 1",
        (match_id,),
    ).fetchone()
    if row is None:
        return None
    return {"summary": row[0], "hp_series": row[1], "champion": row[2]}
