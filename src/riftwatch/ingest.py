"""Getting Riot data into Postgres: cache-first fetches, player sync, resumable backfill.

A finished match and its timeline never change, so anything already in Postgres is served
from there and never requested again. Only rank snapshots and match-id lists (which grow as
people play) are re-requested.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import psycopg

from riftwatch.db import repo
from riftwatch.riot.api import RANKED_SOLO, RiotApi
from riftwatch.riot.routing import RiotId, platform_for

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


class NotFound(LookupError):
    pass


@dataclass
class SyncResult:
    puuid: str
    riot_id: str
    platform: str
    rank: dict[str, Any] | None
    new_matches: list[str] = field(default_factory=list)
    already_cached: int = 0


@dataclass
class BackfillResult:
    job_id: int
    ids_seen: int
    fetched: int
    cached: int
    finished: bool


class Ingestor:
    def __init__(self, conn: psycopg.Connection, api: RiotApi, page_size: int = 100) -> None:
        self.conn = conn
        self.api = api
        self.page_size = page_size  # match ids per request; 100 is Riot's maximum

    # -- cache-first single objects -----------------------------------------------------

    def match(self, match_id: str) -> dict[str, Any] | None:
        cached = repo.get_match(self.conn, match_id)
        repo.bump_cache(self.conn, "match", cached is not None)
        if cached is not None:
            return cached
        raw = self.api.match(match_id)
        if raw is not None:
            repo.insert_match(self.conn, raw)
        return raw

    def timeline(self, match_id: str) -> dict[str, Any] | None:
        cached = repo.get_timeline(self.conn, match_id)
        repo.bump_cache(self.conn, "timeline", cached is not None)
        if cached is not None:
            return cached
        if repo.get_match(self.conn, match_id) is None and self.match(match_id) is None:
            return None  # timeline rows reference matches; no match, nothing to attach to
        raw = self.api.timeline(match_id)
        if raw is not None:
            repo.insert_timeline(self.conn, match_id, raw)
        return raw

    def ensure(self, match_id: str, with_timeline: bool = True) -> bool:
        """Make sure a match (and its timeline) is cached. Returns True if anything was
        downloaded, False if it was all cached already."""
        had_match = repo.get_match(self.conn, match_id) is not None
        had_timeline = (not with_timeline) or repo.get_timeline(self.conn, match_id) is not None
        if had_match and had_timeline:
            repo.bump_cache(self.conn, "match", True)
            if with_timeline:
                repo.bump_cache(self.conn, "timeline", True)
            return False
        self.match(match_id)
        if with_timeline:
            self.timeline(match_id)
        return True

    # -- accounts and ranks --------------------------------------------------------------

    def resolve(self, riot_id: RiotId, platform: str) -> dict[str, Any]:
        platform = platform_for(platform)
        account = self.api.account_by_riot_id(platform, riot_id)
        if account is None:
            raise NotFound(f"no Riot account {riot_id} (region {platform})")
        repo.upsert_account(self.conn, account, platform)
        return account

    def refresh_rank(self, puuid: str, platform: str) -> dict[str, Any] | None:
        entries = self.api.league_entries_by_puuid(platform, puuid)
        repo.insert_rank_snapshots(self.conn, puuid, entries)
        return next((e for e in entries if e["queueType"] == RANKED_SOLO), None)

    # -- sync: recent games ----------------------------------------------------------------

    def sync(
        self,
        riot_id: RiotId,
        platform: str,
        *,
        count: int = 20,
        queue: int | None = 420,
        with_timeline: bool = True,
        progress: Progress | None = None,
    ) -> SyncResult:
        """Fetch the player's newest ``count`` games, stopping early at the first one that is
        already cached (everything older was cached by an earlier sync)."""
        platform = platform_for(platform)
        account = self.resolve(riot_id, platform)
        puuid = account["puuid"]
        rank = self.refresh_rank(puuid, platform)
        bucket = repo.tier_bucket(rank["tier"]) if rank else None
        result = SyncResult(puuid, f"{account['gameName']}#{account['tagLine']}", platform, rank)

        ids = self.api.match_ids(platform, puuid, count=min(count, 100), queue=queue)
        for i, match_id in enumerate(ids, 1):
            have_all = repo.get_match(self.conn, match_id) is not None and (
                not with_timeline or repo.get_timeline(self.conn, match_id) is not None
            )
            if have_all:
                result.already_cached = len(ids) - i + 1
                break
            if progress:
                progress(f"[{i}/{len(ids)}] {match_id}")
            self.ensure(match_id, with_timeline)
            result.new_matches.append(match_id)
            if bucket:
                repo.mark_sample(self.conn, match_id, bucket, "player")
        return result

    # -- backfill: whole history -------------------------------------------------------------

    def _open_job(
        self, puuid: str, platform: str, queue: int | None, start_time: datetime | None,
        with_timeline: bool,
    ) -> int:
        row = self.conn.execute(
            """
            SELECT id FROM backfill_jobs
             WHERE puuid = %s AND status = 'running'
               AND queue_id IS NOT DISTINCT FROM %s AND start_time IS NOT DISTINCT FROM %s
             ORDER BY id DESC LIMIT 1
            """,
            (puuid, queue, start_time),
        ).fetchone()
        if row:
            return row[0]
        return self.conn.execute(
            """
            INSERT INTO backfill_jobs (puuid, platform, queue_id, start_time, with_timeline)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            (puuid, platform, queue, start_time, with_timeline),
        ).fetchone()[0]

    def backfill(
        self,
        riot_id: RiotId,
        platform: str,
        *,
        queue: int | None = 420,
        start_time: datetime | None = None,
        with_timeline: bool = True,
        max_pages: int | None = None,
        progress: Progress | None = None,
    ) -> BackfillResult:
        """Walk the player's entire match history a page of ids at a time, caching every match.

        Progress is saved after each page, so an interrupted run resumes where it stopped
        when called again with the same arguments.
        """
        platform = platform_for(platform)
        account = self.resolve(riot_id, platform)
        puuid = account["puuid"]
        rank = self.refresh_rank(puuid, platform)
        bucket = repo.tier_bucket(rank["tier"]) if rank else None
        job = self._open_job(puuid, platform, queue, start_time, with_timeline)
        offset, = self.conn.execute(
            "SELECT next_offset FROM backfill_jobs WHERE id = %s", (job,)
        ).fetchone()

        pages = 0
        finished = False
        while max_pages is None or pages < max_pages:
            ids = self.api.match_ids(
                platform, puuid, start=offset, count=self.page_size, queue=queue,
                start_time=int(start_time.timestamp()) if start_time else None,
            )
            fetched = cached = 0
            for match_id in ids:
                if self.ensure(match_id, with_timeline):
                    fetched += 1
                else:
                    cached += 1
                if bucket:
                    repo.mark_sample(self.conn, match_id, bucket, "player")
            offset += len(ids)
            pages += 1
            finished = len(ids) < self.page_size
            self.conn.execute(
                """
                UPDATE backfill_jobs
                   SET next_offset = %s, ids_seen = ids_seen + %s, fetched = fetched + %s,
                       cached = cached + %s, updated_at = now(),
                       status = CASE WHEN %s THEN 'done' ELSE status END,
                       finished_at = CASE WHEN %s THEN now() ELSE finished_at END
                 WHERE id = %s
                """,
                (offset, len(ids), fetched, cached, finished, finished, job),
            )
            if progress:
                progress(f"page {pages}: {len(ids)} ids, {fetched} downloaded, {cached} cached "
                         f"(offset {offset})")
            if finished:
                break

        ids_seen, fetched_total, cached_total = self.conn.execute(
            "SELECT ids_seen, fetched, cached FROM backfill_jobs WHERE id = %s", (job,)
        ).fetchone()
        return BackfillResult(job, ids_seen, fetched_total, cached_total, finished)
