"""Aggregate crawled games into rank-matched baselines, and look them up.

A baseline row is the distribution of one metric for one group of players:
(tier bucket, role, champion or 0 = any champion, minute or NULL = whole game). Postgres
computes n, mean, sd and the 10/25/50/75/90th percentiles with ``percentile_cont``.

Only matches sampled by the crawler feed baselines by default: a player's own synced games
would otherwise be part of the yardstick they are measured against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import psycopg

from riftwatch.features.metrics import CURVE_METRICS, CURVE_MINUTES, LANE_LEAD_METRICS

TIER_ORDER = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER_PLUS")
MIN_GAME_S = 600      # drop remakes and other games too short to say anything
# Games before a champion's own baseline at a tier replaces the champion-adjusted role
# baseline. Below this the median of the champion's own games is noisier (about
# 1.25 / sqrt(n) sd) than the adjusted one.
CHAMPION_MIN_N = 200
# Champion adjustment: how far a champion sits from its role, pooled over every tier in
# standard deviations, then applied to the role baseline at the player's tier. On 16.19 it
# cut the error in predicting a champion's median at a held-out tier by a fifth overall and
# about half for farming, gold, damage share and item timings.
ADJUST_MIN_N = 50     # pooled champion games before an adjustment is considered
ADJUST_SHRINK = 100   # effect * n / (n + this): small samples move the baseline less
ADJUST_GATE = 2.0     # apply only effects beyond this many standard errors
# Lane strength: a champion's mean lead over its lane opponent, sum / (n + this). On 16.19
# held-out games, strength(you) - strength(them) explained twice as much of the CS, gold
# and XP leads as your own champion alone (CS lead at 10: 13% vs 7% of the variance);
# averaging each exact pair on top added nothing.
LANE_SHRINK = 50

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


_LANE_GAME_SQL = """
INSERT INTO lane_strength (role, champion_id, metric, minute, n, strength)
SELECT s.role, s.champion_id, kv.key, 0, count(*), sum(kv.value::float8) / (count(*) + %(shrink)s)
  FROM participant_game_summary s
  JOIN matches m USING (match_id)
  JOIN match_samples ms USING (match_id)
  CROSS JOIN LATERAL jsonb_each_text(s.metrics) kv
 WHERE {source} AND kv.key = ANY(%(lead)s)
 GROUP BY s.role, s.champion_id, kv.key
"""

_LANE_CURVES = [c for c in CURVE_METRICS if c in LANE_LEAD_METRICS]
_LANE_CURVE_SQL = f"""
INSERT INTO lane_strength (role, champion_id, metric, minute, n, strength)
SELECT s.role, s.champion_id, c.metric, f.minute, count(*), sum(c.v) / (count(*) + %(shrink)s)
  FROM participant_minute_features f
  JOIN participant_game_summary s USING (match_id, participant_id)
  JOIN matches m USING (match_id)
  JOIN match_samples ms USING (match_id)
  CROSS JOIN LATERAL (VALUES {", ".join(f"('{c}', f.{c}::float8)" for c in _LANE_CURVES)}) AS c(metric, v)
 WHERE {{source}} AND c.v IS NOT NULL AND f.minute BETWEEN %(min_minute)s AND %(max_minute)s
 GROUP BY s.role, s.champion_id, c.metric, f.minute
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


def recent_patches(conn: psycopg.Connection, count: int, include_player: bool = False) -> list[str]:
    """Newest patches among the games that can feed baselines."""
    rows = conn.execute(
        """
        SELECT DISTINCT m.patch FROM matches m JOIN match_samples ms USING (match_id)
         WHERE %s OR ms.source = 'crawl'
        """,
        (include_player,),
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
    patches = recent_patches(conn, patch_count, include_player_games)
    if not patches:
        return BuildReport([], "", 0, 0)
    ordered = sorted(patches, key=_patch_key)
    window = ordered[0] if len(ordered) == 1 else f"{ordered[0]}-{ordered[-1]}"
    params = {
        "patches": patches, "window": window, "min_n": min_n,
        "min_game_s": MIN_GAME_S, "include_player": include_player_games,
        "min_minute": CURVE_MINUTES.start, "max_minute": CURVE_MINUTES.stop - 1,
        "shrink": LANE_SHRINK, "lead": sorted(LANE_LEAD_METRICS),
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
        conn.execute("DELETE FROM lane_strength")
        for template in (_LANE_GAME_SQL, _LANE_CURVE_SQL):
            conn.execute(template.format(source=_SOURCE_FILTER), params)
        rows = conn.execute("SELECT count(*) FROM baselines").fetchone()[0]
        games = conn.execute(
            f"""
            SELECT count(DISTINCT match_id) FROM participant_game_summary s
              JOIN matches m USING (match_id) JOIN match_samples ms USING (match_id)
             WHERE {_SOURCE_FILTER}
            """,
            params,
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO baseline_builds (patch_window, games, rows) VALUES (%s, %s, %s)",
            (window, games, rows),
        )
    invalidate_cache()  # this process rebuilt: don't wait for the staleness check
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
    adjusted_n: int = 0     # > 0: role baseline shifted using this many games
    adjusted_for: str = ""  # "champion" or "matchup"

    @property
    def scope(self) -> str:
        who = (f"adjusted for {self.adjusted_for}" if self.adjusted_n
               else "same champion" if self.champion_id else "same role")
        return f"{self.tier_bucket.replace('_', ' ').title()} {self.role.lower()}, {who}"

    def shifted(self, champion_id: int, delta: float, pooled_n: int,
                adjusted_for: str = "champion") -> Baseline:
        return replace(self, champion_id=champion_id, adjusted_n=pooled_n,
                       adjusted_for=adjusted_for,
                       mean=self.mean + delta, p10=self.p10 + delta, p25=self.p25 + delta,
                       p50=self.p50 + delta, p75=self.p75 + delta, p90=self.p90 + delta)


def champion_effects(rows: list[Baseline], champion_id: int) -> dict[tuple, tuple[float, int]]:
    """(metric, minute) -> (effect in role standard deviations, pooled champion games).

    For each tier with both a champion and a role baseline, the gap between their medians is
    measured in the role's standard deviations, then averaged over tiers weighted by the
    champion's games. Effects within ADJUST_GATE standard errors of zero are dropped; the
    rest are shrunk toward zero.
    """
    role = {(b.metric, b.minute, b.tier_bucket): b for b in rows if b.champion_id == 0}
    sums: dict[tuple, list[float]] = {}
    for b in rows:
        if b.champion_id != champion_id or champion_id == 0:
            continue
        r = role.get((b.metric, b.minute, b.tier_bucket))
        if r is None or not r.sd:
            continue
        acc = sums.setdefault((b.metric, b.minute), [0.0, 0])
        acc[0] += b.n * (b.p50 - r.p50) / r.sd
        acc[1] += b.n
    out = {}
    for key, (total, n) in sums.items():
        if n < ADJUST_MIN_N:
            continue
        effect = total / n
        if abs(effect) < ADJUST_GATE * 1.25 / n ** 0.5:
            continue
        out[key] = (effect * n / (n + ADJUST_SHRINK), int(n))
    return out


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
    champion at tier (if it has CHAMPION_MIN_N games) -> champion-adjusted role at tier ->
    role at tier -> the same at the nearest tier that has data."""

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
        self.champion_id = champion_id
        buckets = neighbor_buckets(tier_bucket)[: 1 + 2 * max_tier_distance]
        # The champion effect pools every tier, so a champion needs all of them loaded.
        rows = [Baseline(*r) for r in conn.execute(
            """
            SELECT metric, minute, tier_bucket, role, champion_id, patch_window,
                   n, mean, sd, p10, p25, p50, p75, p90
              FROM baselines
             WHERE role = %s AND champion_id IN (0, %s) AND (%s <> 0 OR tier_bucket = ANY(%s))
            """,
            (role, champion_id, champion_id, buckets),
        ).fetchall()]
        self._effects = champion_effects(rows, champion_id)
        self._memo: dict[tuple, Baseline | None] = {}
        self._lane = lane_strengths(conn, role)
        self._rows: dict[tuple[str, int | None, str, int], Baseline] = {}
        for b in rows:
            if b.tier_bucket in buckets:
                self._rows[(b.metric, b.minute, b.tier_bucket, b.champion_id)] = b
        self._order = [(tier_bucket, champion_id), (tier_bucket, 0)] + [
            (b, 0) for b in buckets[1:]
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, metric: str, minute: int | None = None,
            opponent_champion_id: int | None = None) -> Baseline | None:
        """``opponent_champion_id`` (lane-lead metrics only) judges the lead against the
        matchup: the role baseline shifted by strength(you) - strength(them).

        Answers are memoised: sets are shared through the cache, and building an adjusted
        baseline is most of the cost of scoring a game."""
        key = (metric, minute, opponent_champion_id if metric in LANE_LEAD_METRICS else None)
        if key not in self._memo:
            self._memo[key] = self._get(metric, minute, key[2])
        return self._memo[key]

    def _get(self, metric: str, minute: int | None,
             opponent_champion_id: int | None) -> Baseline | None:
        matchup = None
        if opponent_champion_id and metric in LANE_LEAD_METRICS:
            key = minute or 0
            me = self._lane.get((metric, key, self.champion_id))
            them = self._lane.get((metric, key, opponent_champion_id))
            if me or them:
                matchup = ((me[0] if me else 0.0) - (them[0] if them else 0.0),
                           (me[1] if me else 0) + (them[1] if them else 0))
        for bucket, champ in self._order:
            if champ and matchup is not None:
                continue
            b = self._rows.get((metric, minute, bucket, champ))
            # A one-champion group must be big enough to beat the whole-role group as a
            # yardstick: 25 games of one champion is noisier than hundreds of the role.
            needed = max(self.min_n, CHAMPION_MIN_N) if champ else self.min_n
            if b is None or b.n < needed:
                continue
            if matchup is not None:
                return b.shifted(self.champion_id, matchup[0], matchup[1], "matchup")
            effect = self._effects.get((metric, minute)) if not champ else None
            if effect is not None:
                return b.shifted(self.champion_id, effect[0] * b.sd, effect[1])
            return b
        return None


# -- in-process cache ------------------------------------------------------------------------
# Baselines only change when `build` runs, which records a new baseline_builds row. Readers
# keep loaded sets in memory and re-check that one primary-key value per call, so a web
# request scoring a game costs one tiny query instead of a scan of the baselines table.

_cache: dict[tuple, BaselineSet] = {}
_lanes: dict[str, dict[tuple, tuple[float, int]]] = {}
_cache_generation: tuple | None = None
_checked_at = float("-inf")
# A rebuild is picked up within this many seconds; in between, cache hits cost no query.
STALENESS_CHECK_S = 5.0


def current_generation(conn: psycopg.Connection) -> tuple:
    """(id, built_at) of the latest build. The timestamp guards against a recreated
    database reusing build id 1."""
    row = conn.execute(
        "SELECT id, built_at FROM baseline_builds ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return tuple(row) if row else (0, None)


def baselines_for(
    conn: psycopg.Connection, tier_bucket: str, role: str, champion_id: int,
    min_n: int = 20, max_tier_distance: int = 1,
) -> BaselineSet:
    global _cache_generation, _checked_at
    now = time.monotonic()
    if now - _checked_at >= STALENESS_CHECK_S:
        generation = current_generation(conn)
        _checked_at = now
        if generation != _cache_generation:
            _cache.clear()
            _lanes.clear()
            _cache_generation = generation
    key = (tier_bucket, role, champion_id, min_n, max_tier_distance)
    if key not in _cache:
        _cache[key] = BaselineSet(conn, tier_bucket, role, champion_id, min_n, max_tier_distance)
    return _cache[key]


def lane_strengths(conn: psycopg.Connection, role: str) -> dict[tuple, tuple[float, int]]:
    """(metric, minute or 0, champion_id) -> (strength, games) for one role; shared by every
    BaselineSet of the role and dropped with the baseline cache."""
    if role not in _lanes:
        _lanes[role] = {
            (m, minute, champ): (strength, n)
            for m, minute, champ, strength, n in conn.execute(
                "SELECT metric, minute, champion_id, strength, n FROM lane_strength WHERE role = %s",
                (role,))
        }
    return _lanes[role]


def invalidate_cache() -> None:
    global _cache_generation, _checked_at
    _cache.clear()
    _lanes.clear()
    _cache_generation = None
    _checked_at = float("-inf")
