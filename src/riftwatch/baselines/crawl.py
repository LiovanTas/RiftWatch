"""Collect games from every rank tier so baselines have something to compare against.

For each tier (and division) the crawler reads a ladder page from league-exp-v4, picks
players it hasn't sampled before, and pulls their recent ranked solo games through the
same cache and rate limiter as everything else. Every match is tagged with the tier
bucket it was found in.

With a development key (100 requests / 2 min) a match costs two requests (match +
timeline), so expect roughly 50 new games per two minutes. The crawl is incremental:
rerunning it samples new players and skips cached games.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import psycopg

from riftwatch.db import repo
from riftwatch.ingest import Ingestor
from riftwatch.riot.api import APEX_TIERS, DIVISIONS, RANKED_SOLO_QUEUE_ID, TIERS
from riftwatch.riot.routing import platform_for

Progress = Callable[[str], None]

DEFAULT_TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER")


@dataclass
class CrawlReport:
    players: int = 0
    matches_new: int = 0
    matches_cached: int = 0
    by_bucket: dict[str, int] = field(default_factory=dict)


def _divisions(tier: str) -> tuple[str, ...]:
    return ("I",) if tier in APEX_TIERS else DIVISIONS


def _already_crawled(conn: psycopg.Connection, puuids: Iterable[str]) -> set[str]:
    puuids = list(puuids)
    if not puuids:
        return set()
    rows = conn.execute("SELECT puuid FROM crawl_players WHERE puuid = ANY(%s)", (puuids,))
    return {r[0] for r in rows}


def crawl(
    conn: psycopg.Connection,
    ingestor: Ingestor,
    platform: str,
    *,
    tiers: Iterable[str] = DEFAULT_TIERS,
    players_per_division: int = 2,
    matches_per_player: int = 5,
    with_timeline: bool = True,
    rng: random.Random | None = None,
    progress: Progress | None = None,
) -> CrawlReport:
    platform = platform_for(platform)
    rng = rng or random.Random()
    api = ingestor.api
    report = CrawlReport()

    for tier in (t.upper() for t in tiers):
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        bucket = repo.tier_bucket(tier)
        for division in _divisions(tier):
            # Page 1 is ~205 players; picking at random from it avoids always sampling the
            # same top-of-division players.
            entries = [e for e in api.league_entries(platform, tier, division) if e.get("puuid")]
            seen = _already_crawled(conn, (e["puuid"] for e in entries))
            fresh = [e for e in entries if e["puuid"] not in seen]
            picks = rng.sample(fresh, min(players_per_division, len(fresh)))
            for entry in picks:
                puuid = entry["puuid"]
                ids = api.match_ids(platform, puuid, count=matches_per_player,
                                    queue=RANKED_SOLO_QUEUE_ID)
                found = 0
                for match_id in ids:
                    downloaded = ingestor.ensure(match_id, with_timeline)
                    if downloaded:
                        report.matches_new += 1
                    else:
                        report.matches_cached += 1
                    repo.mark_sample(conn, match_id, bucket, "crawl")
                    found += 1
                conn.execute(
                    """
                    INSERT INTO crawl_players (puuid, platform, tier, division, tier_bucket, matches_found)
                    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (puuid) DO NOTHING
                    """,
                    (puuid, platform, tier, division, bucket, found),
                )
                report.players += 1
                report.by_bucket[bucket] = report.by_bucket.get(bucket, 0) + found
                if progress:
                    progress(f"{tier} {division}: player {report.players}, +{found} games "
                             f"({report.matches_new} downloaded so far)")
    return report


def sample_counts(conn: psycopg.Connection) -> dict[str, int]:
    """Matches available per tier bucket (with timelines, i.e. usable for baselines)."""
    rows = conn.execute(
        """
        SELECT s.tier_bucket, count(*) FROM match_samples s
          JOIN match_timelines t USING (match_id)
         GROUP BY s.tier_bucket
        """
    )
    return dict(rows.fetchall())
