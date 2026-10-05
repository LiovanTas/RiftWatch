"""Collect games from the ranked ladder, for baselines and for the high-elo models.

For each tier (and division) the crawler pages through league-exp-v4, picks players it
hasn't sampled before, and pulls their recent ranked solo games through the same cache and
rate limiter as everything else. Every match is tagged with the tier bucket it was found
in, and every (player, match) pair is recorded with the player's exact tier.

Players are processed in chunks, and each chunk's results are saved before the next starts,
so stopping a long crawl loses at most one chunk. Rerunning samples new players and skips
cached games.

Throughput: a match costs two requests (match + timeline). A development key allows 100
requests per 2 minutes *per region*, so ~25 new games a minute in one region -- and the
regions are independent, so :func:`crawl_regions` crawls several at once.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import psycopg

from riftwatch.db import repo
from riftwatch.ingest import Ingestor
from riftwatch.riot.api import APEX_TIERS, DIVISIONS, RANKED_SOLO_QUEUE_ID, TIERS, RiotApi
from riftwatch.riot.routing import platform_for

Progress = Callable[[str], None]

# Only games this recent are sampled. Many ladder players play rarely, and their "last 5
# games" can reach back months -- patches whose numbers no longer describe the game.
DEFAULT_MAX_AGE_DAYS = 14

DEFAULT_TIERS = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER")
HIGH_ELO_TIERS = ("CHALLENGER", "GRANDMASTER", "MASTER")

CHUNK = 10          # players per saved chunk
MAX_PAGES = 20      # ladder pages to scan per division looking for fresh players


@dataclass
class CrawlReport:
    platform: str = ""
    players: int = 0
    matches_new: int = 0
    matches_cached: int = 0
    by_bucket: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def _divisions(tier: str) -> tuple[str, ...]:
    return ("I",) if tier in APEX_TIERS else DIVISIONS


def _already_crawled(conn: psycopg.Connection, puuids: Iterable[str]) -> set[str]:
    puuids = list(puuids)
    if not puuids:
        return set()
    rows = conn.execute("SELECT puuid FROM crawl_players WHERE puuid = ANY(%s)", (puuids,))
    return {r[0] for r in rows}


def _fresh_players(
    conn: psycopg.Connection, api: RiotApi, platform: str, tier: str, division: str,
    want: int, rng: random.Random,
) -> list[dict]:
    """Up to ``want`` ladder entries not crawled before. Pages forward until it has enough
    candidates (twice ``want``, to sample from) or the ladder runs out."""
    pool: dict[str, dict] = {}
    for page in range(1, MAX_PAGES + 1):
        entries = [e for e in api.league_entries(platform, tier, division, page=page)
                   if e.get("puuid")]
        if not entries:
            break
        seen = _already_crawled(conn, (e["puuid"] for e in entries))
        for e in entries:
            if e["puuid"] not in seen:
                pool.setdefault(e["puuid"], e)
        if len(pool) >= want * 2:
            break
    candidates = list(pool.values())
    return rng.sample(candidates, min(want, len(candidates)))


def _save_chunk(conn, platform, tier, division, bucket, per_player, ok) -> dict[str, int]:
    found: dict[str, int] = {}
    with conn.transaction():
        for puuid, ids in per_player.items():
            kept = [m for m in ids if m in ok]
            for match_id in kept:
                repo.mark_sample(conn, match_id, bucket, "crawl")
            if kept:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO crawl_player_matches (puuid, match_id, platform, tier)
                        VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
                        """,
                        [(puuid, m, platform, tier) for m in kept],
                    )
            conn.execute(
                """
                INSERT INTO crawl_players (puuid, platform, tier, division, tier_bucket, matches_found)
                VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (puuid) DO NOTHING
                """,
                (puuid, platform, tier, division, bucket, len(kept)),
            )
            found[puuid] = len(kept)
    return found


def crawl(
    conn: psycopg.Connection,
    ingestor: Ingestor,
    platform: str,
    *,
    tiers: Iterable[str] = DEFAULT_TIERS,
    players_per_division: int = 2,
    matches_per_player: int = 5,
    with_timeline: bool = True,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    rng: random.Random | None = None,
    progress: Progress | None = None,
    now: float | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> CrawlReport:
    platform = platform_for(platform)
    since = int((now or time.time()) - max_age_days * 86_400)
    rng = rng or random.Random()
    api = ingestor.api
    report = CrawlReport(platform=platform)
    say = progress or (lambda _msg: None)

    tiers = [t.upper() for t in tiers]
    for tier in tiers:
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
    for tier in tiers:
        bucket = repo.tier_bucket(tier)
        for division in _divisions(tier):
            picks = _fresh_players(conn, api, platform, tier, division,
                                   players_per_division, rng)
            for start in range(0, len(picks), CHUNK):
                if should_stop():
                    return report
                chunk = picks[start:start + CHUNK]
                per_player = {
                    e["puuid"]: api.match_ids(platform, e["puuid"], count=matches_per_player,
                                              queue=RANKED_SOLO_QUEUE_ID, start_time=since)
                    for e in chunk
                }
                fetched = ingestor.fetch_many(
                    [m for ids in per_player.values() for m in ids], with_timeline)
                ok = set(fetched.downloaded) | set(fetched.cached)
                found = _save_chunk(conn, platform, tier, division, bucket, per_player, ok)
                report.players += len(chunk)
                report.matches_new += len(fetched.downloaded)
                report.matches_cached += len(fetched.cached)
                report.by_bucket[bucket] = report.by_bucket.get(bucket, 0) + sum(found.values())
                say(f"{platform} {tier} {division}: players {start + len(chunk)}/{len(picks)}, "
                    f"+{len(fetched.downloaded)} new games ({report.matches_new} this run)")
    return report


def crawl_regions(
    database_url: str,
    api: RiotApi,
    platforms: Iterable[str],
    *,
    connect: Callable[[str], psycopg.Connection],
    progress: Progress | None = None,
    **kwargs,
) -> list[CrawlReport]:
    """Crawl several regions at once, one thread and one database connection each. Riot's
    limits are per region, so this multiplies throughput; the shared client's limiter
    already keeps a separate quota per region."""
    lock = threading.Lock()
    reports: list[CrawlReport] = []

    def say(msg: str) -> None:
        if progress:
            with lock:
                progress(msg)

    def run(platform: str) -> None:
        report = CrawlReport(platform=platform_for(platform))
        try:
            with connect(database_url) as conn:
                report = crawl(conn, Ingestor(conn, api), platform, progress=say, **kwargs)
        except Exception as exc:  # one region failing must not stop the others
            report.error = f"{type(exc).__name__}: {exc}"
            say(f"{report.platform}: stopped -- {report.error}")
        with lock:
            reports.append(report)

    threads = [threading.Thread(target=run, args=(p,), name=f"crawl-{p}") for p in platforms]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(reports, key=lambda r: r.platform)


def sample_counts(conn: psycopg.Connection, source: str = "crawl") -> dict[str, int]:
    """Crawled matches per tier bucket that have timelines (i.e. can feed baselines)."""
    rows = conn.execute(
        """
        SELECT s.tier_bucket, count(*) FROM match_samples s
          JOIN match_timelines t USING (match_id)
         WHERE s.source = %s
         GROUP BY s.tier_bucket
        """,
        (source,),
    )
    return dict(rows.fetchall())


def high_elo_counts(conn: psycopg.Connection) -> dict[str, int]:
    """Distinct crawled games per exact apex tier of the player they were found through."""
    rows = conn.execute(
        """
        SELECT tier, count(DISTINCT match_id) FROM crawl_player_matches
         WHERE tier = ANY(%s) GROUP BY tier
        """,
        (list(HIGH_ELO_TIERS),),
    )
    return dict(rows.fetchall())
