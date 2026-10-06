"""Keep baselines on the current patch.

``refresh`` reads the live patch from Data Dragon, counts the crawled games each rank
bucket has on it, tops up the buckets below a target with a recent-games crawl, then
re-extracts features and rebuilds baselines. Run it on a schedule (daily is plenty) and
comparisons move onto a new patch on their own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import psycopg

from riftwatch.baselines import build as baseline_build
from riftwatch.baselines import crawl as baseline_crawl
from riftwatch.riot.ddragon import patch_of

# Tiers crawled to fill each bucket.
BUCKET_TIERS = {
    "IRON": ["IRON"], "BRONZE": ["BRONZE"], "SILVER": ["SILVER"], "GOLD": ["GOLD"],
    "PLATINUM": ["PLATINUM"], "EMERALD": ["EMERALD"], "DIAMOND": ["DIAMOND"],
    "MASTER_PLUS": ["MASTER", "GRANDMASTER", "CHALLENGER"],
}
DEFAULT_TARGET = 300        # crawled games per bucket on the current patch
GAMES_PER_PLAYER = 5        # most recent games requested per sampled player
MAX_PLAYERS = 100           # per division per round
DEFAULT_ROUNDS = 3


@dataclass
class RefreshReport:
    patch: str
    before: dict[str, int]
    crawled: dict[str, int] = field(default_factory=dict)      # bucket -> new games
    after: dict[str, int] = field(default_factory=dict)
    rebuilt: bool = False


def patch_counts(conn: psycopg.Connection, patch: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT s.tier_bucket, count(*) FROM match_samples s JOIN matches m USING (match_id)
         WHERE s.source = 'crawl' AND m.patch = %s AND m.queue_id = 420
         GROUP BY s.tier_bucket
        """,
        (patch,),
    ).fetchall()
    counts = dict.fromkeys(BUCKET_TIERS, 0)
    counts.update(dict(rows))
    return counts


def _divisions(tiers: list[str]) -> int:
    return sum(1 if t in ("MASTER", "GRANDMASTER", "CHALLENGER") else 4 for t in tiers)


def players_needed(missing_games: int, tiers: list[str], games_per_player: float | None = None) -> int:
    """Players to sample per division to find roughly ``missing_games`` new games.

    The first round assumes each player gives GAMES_PER_PLAYER recent games with a third
    lost to overlap or older patches. Later rounds use the yield actually observed -- at the
    bottom of the ladder many accounts haven't played in weeks, so it can be far lower.
    """
    per_player = games_per_player if games_per_player else GAMES_PER_PLAYER / 1.5
    per_player = max(per_player, 0.2)
    return min(MAX_PLAYERS, max(1, math.ceil(missing_games / (per_player * _divisions(tiers)))))


def refresh(
    conn: psycopg.Connection,
    *,
    current_patch: str,
    crawl_fn: Callable[[list[str], int], int],
    extract_fn: Callable[[], object],
    target: int = DEFAULT_TARGET,
    buckets: Iterable[str] | None = None,
    rounds: int = DEFAULT_ROUNDS,
    dry_run: bool = False,
    say: Callable[[str], None] = print,
) -> RefreshReport:
    """``crawl_fn(tiers, players_per_division)`` crawls and returns how many players it
    sampled in total; injected so the CLI can crawl several regions at once and tests can
    stub it. Each bucket gets up to ``rounds`` passes, sized from the yield of the last."""
    report = RefreshReport(current_patch, patch_counts(conn, current_patch))
    for bucket in buckets or BUCKET_TIERS:
        tiers = BUCKET_TIERS[bucket]
        have = report.before.get(bucket, 0)
        observed: float | None = None
        for n in range(1, rounds + 1):
            if have >= target:
                break
            players = players_needed(target - have, tiers, observed)
            say(f"{bucket}: {have}/{target} games on {current_patch}; round {n}, "
                f"{players} player(s) per division")
            if dry_run:
                break
            sampled = crawl_fn(tiers, players)
            now = patch_counts(conn, current_patch).get(bucket, 0)
            report.crawled[bucket] = report.crawled.get(bucket, 0) + now - have
            observed = (now - have) / sampled if sampled else 0.0
            have = now
            if sampled == 0:
                break               # the ladder has no more fresh players
    if report.crawled:
        extract_fn()
        built = baseline_build.build(conn)
        report.rebuilt = True
        say(f"rebuilt {built.rows} baseline rows from {built.games} games ({built.patch_window})")
    report.after = patch_counts(conn, current_patch)
    return report


def live_patch(ddragon) -> str:
    return patch_of(ddragon.versions()[0])


def crawler(api, regions: list[str], connect, database_url: str,
            say: Callable[[str], None] = print) -> Callable[[list[str], int], int]:
    """A crawl_fn that crawls ``regions`` in parallel for recent games."""
    def crawl_fn(tiers: list[str], players: int) -> int:
        reports = baseline_crawl.crawl_regions(
            database_url, api, regions, connect=connect, progress=say, tiers=tiers,
            players_per_division=players, matches_per_player=GAMES_PER_PLAYER)
        for r in reports:
            if r.error:
                say(f"{r.platform}: {r.error}")
        # Yield is per division, so report players per region-division.
        return sum(r.players for r in reports) // max(1, len(reports))
    return crawl_fn

