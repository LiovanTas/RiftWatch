"""Aggregate crawled games into rank-matched baselines, and look them up.

A baseline row is the distribution of one metric for one group of players:
(tier bucket, role, champion or 0 = any champion, minute or NULL = whole game). Postgres
computes n, mean, sd and the 10/25/50/75/90th percentiles with ``percentile_cont``.

Only matches sampled by the crawler feed baselines by default: a player's own synced games
would otherwise be part of the yardstick they are measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from riftwatch.features.metrics import CURVE_METRICS, CURVE_MINUTES

TIER_ORDER = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER_PLUS")
MIN_GAME_S = 600  # drop remakes and other games too short to say anything

_STATS = """
    count(*), avg(v), coalesce(stddev_samp(v), 0),
    percentile_cont(0.10) WITHIN GROUP (ORDER BY v),
    percentile_cont(0.25) WITHIN GROUP (ORDER BY v),
    percentile_cont(0.50) WITHIN GROUP (ORDER BY v),
    percentile_cont(0.75) WITHIN GROUP (ORDER BY v),
    percentile_cont(0.90) WITHIN GROUP (ORDER BY v)
"""

_SOURCE_FILTER = """
    m.queue_id = 420 AND m.patch = ANY(%(patches)s) AND m.duration_s >= %(min_game_s)s
    AND s.role <> '' AND (%(include_player)s OR ms.source = 'crawl')
"""

_GAME_SQL = """
INSERT INTO baselines (tier_bucket, role, champion_id, patch_window, metric, minute,
                       n, mean, sd, p10, p25, p50, p75, p90)
SELECT ms.tier_bucket, s.role, {champion}, %(window)s, kv.key, NULL, {stats}
  FROM participant_game_summary s
  JOIN matches m USING (match_id)
  JOIN match_samples ms USING (match_id)
  CROSS JOIN LATERAL jsonb_each_text(s.metrics) kv
  CROSS JOIN LATERAL (SELECT kv.value::float8 AS v) x
 WHERE {source}
 GROUP BY ms.tier_bucket, s.role, {group_champion} kv.key
HAVING count(*) >= %(min_n)s
"""

_CURVE_VALUES = ", ".join(f"('{c}', f.{c}::float8)" for c in CURVE_METRICS)

_CURVE_SQL = f"""
INSERT INTO baselines (tier_bucket, role, champion_id, patch_window, metric, minute,
                       n, mean, sd, p10, p25, p50, p75, p90)
SELECT ms.tier_bucket, s.role, {{champion}}, %(window)s, c.metric, f.minute, {{stats}}
  FROM participant_minute_features f
  JOIN participant_game_summary s USING (match_id, participant_id)
  JOIN matches m USING (match_id)
  JOIN match_samples ms USING (match_id)
  CROSS JOIN LATERAL (VALUES {_CURVE_VALUES}) AS c(metric, v)
 WHERE {{source}} AND c.v IS NOT NULL AND f.minute BETWEEN %(min_minute)s AND %(max_minute)s
 GROUP BY ms.tier_bucket, s.role, {{group_champion}} c.metric, f.minute
HAVING count(*) >= %(min_n)s
"""


@dataclass
class BuildReport:
    patches: list[str]
    patch_window: str
    rows: int
    games: int


def _patch_key(patch: str) -> tuple[int, int]:
    major, minor = patch.split(".")
    return int(major), int(minor)


def recent_patches(conn: psycopg.Connection, count: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT m.patch FROM matches m JOIN match_samples USING (match_id)"
    ).fetchall()
    patches = sorted((r[0] for r in rows), key=_patch_key, reverse=True)
    return patches[:count]


def build(
    conn: psycopg.Connection,
    *,
    patch_count: int = 3,
    min_n: int = 20,
    include_player_games: bool = False,
) -> BuildReport:
    """Rebuild every baseline from scratch, atomically."""
    patches = recent_patches(conn, patch_count)
    if not patches:
        return BuildReport([], "", 0, 0)
    ordered = sorted(patches, key=_patch_key)
    window = ordered[0] if len(ordered) == 1 else f"{ordered[0]}-{ordered[-1]}"
    params = {
        "patches": patches, "window": window, "min_n": min_n,
        "min_game_s": MIN_GAME_S, "include_player": include_player_games,
        "min_minute": CURVE_MINUTES.start, "max_minute": CURVE_MINUTES.stop - 1,
    }
    with conn.transaction():
        conn.execute("DELETE FROM baselines")
        for template in (_GAME_SQL, _CURVE_SQL):
            for champion, group in (("s.champion_id", "s.champion_id,"), ("0", "")):
                conn.execute(
                    template.format(champion=champion, group_champion=group, stats=_STATS,
                                    source=_SOURCE_FILTER),
                    params,
                )
        rows = conn.execute("SELECT count(*) FROM baselines").fetchone()[0]
        games = conn.execute(
            f"""
            SELECT count(DISTINCT match_id) FROM participant_game_summary s
              JOIN matches m USING (match_id) JOIN match_samples ms USING (match_id)
             WHERE {_SOURCE_FILTER}
            """,
            params,
        ).fetchone()[0]
    return BuildReport(patches, window, rows, games)


# -- lookup -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Baseline:
    metric: str
    minute: int | None
    tier_bucket: str
    role: str
    champion_id: int        # 0 = all champions in the role
    patch_window: str
    n: int
    mean: float
    sd: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float

    @property
    def scope(self) -> str:
        who = "same champion" if self.champion_id else "same role"
        return f"{self.tier_bucket.replace('_', ' ').title()} {self.role.lower()}, {who}"


def neighbor_buckets(bucket: str) -> list[str]:
    """``bucket`` first, then tiers outward by distance (below before above)."""
    if bucket not in TIER_ORDER:
        return [bucket]
    i = TIER_ORDER.index(bucket)
    out = [bucket]
    for d in range(1, len(TIER_ORDER)):
        for j in (i - d, i + d):
            if 0 <= j < len(TIER_ORDER):
                out.append(TIER_ORDER[j])
    return out


class BaselineSet:
    """All baselines for one (tier, role, champion), loaded once, with fallbacks:
    champion at tier -> role at tier -> role at the nearest tier that has data."""

    def __init__(
        self,
        conn: psycopg.Connection,
        tier_bucket: str,
        role: str,
        champion_id: int,
        min_n: int = 20,
        max_tier_distance: int = 1,
    ) -> None:
        self.tier_bucket = tier_bucket
        self.min_n = min_n
        buckets = neighbor_buckets(tier_bucket)[: 1 + 2 * max_tier_distance]
        rows = conn.execute(
            """
            SELECT metric, minute, tier_bucket, role, champion_id, patch_window,
                   n, mean, sd, p10, p25, p50, p75, p90
              FROM baselines
             WHERE role = %s AND tier_bucket = ANY(%s) AND champion_id IN (0, %s)
            """,
            (role, buckets, champion_id),
        ).fetchall()
        self._rows: dict[tuple[str, int | None, str, int], Baseline] = {}
        for r in rows:
            b = Baseline(*r)
            self._rows[(b.metric, b.minute, b.tier_bucket, b.champion_id)] = b
        self._order = [(tier_bucket, champion_id), (tier_bucket, 0)] + [
            (b, 0) for b in buckets[1:]
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, metric: str, minute: int | None = None) -> Baseline | None:
        for bucket, champ in self._order:
            b = self._rows.get((metric, minute, bucket, champ))
            if b is not None and b.n >= self.min_n:
                return b
        return None
