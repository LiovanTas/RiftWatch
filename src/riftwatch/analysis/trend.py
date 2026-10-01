"""Many games at once: where a player is consistently strong or weak.

Borrowed from op.gg's "recent 20 vs season" view, but in percentiles: for each metric, the
median goodness over the recent games and over the older ones, and the change.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from riftwatch.analysis.score import GameScore
from riftwatch.features.metrics import GAME_METRICS, Metric


@dataclass(frozen=True)
class MetricTrend:
    metric: Metric
    games: int
    median_goodness: float           # recent games
    median_value: float
    baseline_p50: float              # median of the comparison group (latest game's baseline)
    older_goodness: float | None     # older games, if there are enough
    older_games: int

    @property
    def change(self) -> float | None:
        return None if self.older_goodness is None else self.median_goodness - self.older_goodness


def trends(scores: list[GameScore], recent: int = 20, min_games: int = 5) -> list[MetricTrend]:
    """``scores`` newest first. Metrics seen in fewer than ``min_games`` recent games are
    skipped -- a median of two games isn't a pattern."""
    recent_scores, older_scores = scores[:recent], scores[recent:]
    out = []
    for name, metric in GAME_METRICS.items():
        now = [gs.game[name] for gs in recent_scores if name in gs.game]
        if len(now) < min_games:
            continue
        before = [gs.game[name] for gs in older_scores if name in gs.game]
        out.append(MetricTrend(
            metric=metric,
            games=len(now),
            median_goodness=statistics.median(s.goodness for s in now),
            median_value=statistics.median(s.value for s in now),
            baseline_p50=now[0].baseline.p50,
            older_goodness=(statistics.median(s.goodness for s in before)
                            if len(before) >= min_games else None),
            older_games=len(before),
        ))
    return out
