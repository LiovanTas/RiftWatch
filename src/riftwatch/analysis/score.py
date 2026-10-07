"""Score a player's game against rank-matched baselines.

Every comparison becomes a percentile: "your CS at 10 was 62, which is the 23rd percentile
of Platinum top laners (n=412)". Percentiles are read off the baseline's stored quantiles
(10/25/50/75/90th) by linear interpolation -- no normality assumption -- and extrapolated
past the ends using the neighbouring quantile gap.

``goodness`` flips the scale for lower-is-better metrics (deaths), so 90 always means
"better than 90% of comparable players".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from riftwatch.baselines.build import Baseline, BaselineSet
from riftwatch.features.extract import ParticipantFeatures
from riftwatch.features.metrics import (
    CURVE_METRICS,
    CURVE_MINUTES,
    GAME_METRICS,
    LANE_LEAD_METRICS,
    Metric,
)

_QUANTS = (10, 25, 50, 75, 90)


def percentile(value: float, b: Baseline) -> float:
    """Approximate percentile (0-100) of ``value`` within baseline ``b``."""
    qs = list(zip(_QUANTS, (b.p10, b.p25, b.p50, b.p75, b.p90), strict=True))
    tied = [pct for pct, q in qs if q == value]
    if tied:
        # Discrete metrics (deaths, plates) often have several quantiles equal; a value
        # sitting on that plateau is in the middle of it.
        return (min(tied) + max(tied)) / 2
    if value < b.p10:
        gap = b.p25 - b.p10 or b.sd or 1.0
        return max(1.0, 10 - 15 * (b.p10 - value) / gap)
    if value > b.p90:
        gap = b.p90 - b.p75 or b.sd or 1.0
        return min(99.0, 90 + 15 * (value - b.p90) / gap)
    for (lo_pct, lo), (hi_pct, hi) in zip(qs, qs[1:], strict=False):
        if lo < value < hi:
            return lo_pct + (hi_pct - lo_pct) * (value - lo) / (hi - lo)
    return 50.0  # unreachable for well-formed quantiles; keeps the type honest


@dataclass(frozen=True, slots=True)
class Score:
    metric: Metric
    value: float
    baseline: Baseline
    minute: int | None = None
    # Computed once: history views read each score's standing many times (weekly, per area,
    # per champion). Slots, not a per-object dict: a cached player history holds tens of
    # thousands of these.
    percentile: float = field(init=False, repr=False, compare=False)
    goodness: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        p = percentile(self.value, self.baseline)
        object.__setattr__(self, "percentile", p)
        object.__setattr__(self, "goodness", p if self.metric.higher_is_better else 100 - p)

    @property
    def z(self) -> float:
        return (self.value - self.baseline.mean) / self.baseline.sd if self.baseline.sd else 0.0


@dataclass
class GameScore:
    participant: ParticipantFeatures
    tier_bucket: str
    game: dict[str, Score] = field(default_factory=dict)
    curves: dict[str, list[Score]] = field(default_factory=dict)   # metric -> by minute
    missing: list[str] = field(default_factory=list)                # metrics with no baseline
    # Master+ baselines for the same role, as a "where high elo sits" reference. Empty when
    # the player is Master+ themselves, or there's no high-elo data for the role.
    reference: dict[str, Baseline] = field(default_factory=dict)
    reference_label: str = "Master+"
    opponent_champion_id: int | None = None

    def curve_at(self, metric: str, minute: int) -> Score | None:
        return next((s for s in self.curves.get(metric, []) if s.minute == minute), None)


def score_participant(p: ParticipantFeatures, baselines: BaselineSet,
                      opponent_champion_id: int | None = None) -> GameScore:
    """``opponent_champion_id`` is the lane opponent's champion; with it, leads over the
    opponent are compared against the matchup instead of the whole role."""
    gs = GameScore(p, baselines.tier_bucket, opponent_champion_id=opponent_champion_id)
    opp = opponent_champion_id
    for name, value in p.metrics.items():
        metric = GAME_METRICS.get(name)
        if metric is None or not metric.applies_to(p.role):
            continue
        b = baselines.get(name, None, opp if name in LANE_LEAD_METRICS else None)
        if b is None:
            gs.missing.append(name)
            continue
        gs.game[name] = Score(metric, value, b)

    for name, metric in CURVE_METRICS.items():
        if not metric.applies_to(p.role):
            continue
        scores = []
        for row in p.minutes:
            if row.minute not in CURVE_MINUTES:
                continue
            value = getattr(row, name)
            b = baselines.get(name, row.minute, opp if name in LANE_LEAD_METRICS else None)
            if value is None or b is None:
                continue
            scores.append(Score(metric, float(value), b, row.minute))
        if scores:
            gs.curves[name] = scores
    return gs
