"""Progress over time: the player's games week by week against one fixed yardstick.

Every game is scored against players at the player's *current* rank (adjusted for champion
and matchup), so the weeks are comparable: a rising line means the games got better, not
that the comparison group moved. Weeks run Monday to Sunday, UTC.

The trend is a least-squares line through the per-game numbers in time order. It is called
improving or declining only when the slope is more than CLEAR_SLOPE_SE standard errors from
flat; otherwise it is steady.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from riftwatch.analysis.pool import game_score
from riftwatch.analysis.score import GameScore
from riftwatch.riot.api import APEX_TIERS, TIERS

CLEAR_SLOPE_SE = 2.0
PER_GAMES = 10            # slope is reported per this many games
_DIVISION = {"IV": 0, "III": 1, "II": 2, "I": 3}


@dataclass
class Week:
    start: date
    games: int
    wins: int
    score: float                                   # mean per-game "better than N%"
    areas: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@dataclass
class RankPoint:
    at: datetime
    tier: str
    division: str | None
    lp: int

    @property
    def label(self) -> str:
        return f"{self.tier.title()} {self.division or ''} {self.lp} LP".replace("  ", " ")

    @property
    def ladder(self) -> int:
        """One number for the whole ladder: 400 per tier below Master, 100 per division,
        plus LP; Master+ share one scale above Diamond I."""
        if self.tier in APEX_TIERS:
            return TIERS.index("MASTER") * 400 + self.lp
        return TIERS.index(self.tier) * 400 + _DIVISION.get(self.division or "IV", 0) * 100 + self.lp


@dataclass
class Trend:
    slope: float          # change in "better than N%" per PER_GAMES games
    se: float
    games: int

    @property
    def verdict(self) -> str:
        if self.games < 10 or self.se == 0:
            return "steady"
        if abs(self.slope) <= CLEAR_SLOPE_SE * self.se:
            return "steady"
        return "improving" if self.slope > 0 else "declining"


@dataclass
class ProgressReport:
    tier_bucket: str
    weeks: list[Week]
    trend: Trend
    ranks: list[RankPoint]
    area_trends: dict[str, Trend] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return sum(w.games for w in self.weeks)


def week_start(at: datetime) -> date:
    d = at.date()
    return d - timedelta(days=d.weekday())


def fit(values: list[float]) -> Trend:
    """Least-squares slope of ``values`` against their index, per PER_GAMES, with its
    standard error."""
    n = len(values)
    if n < 3:
        return Trend(0.0, 0.0, n)
    xs = range(n)
    mx, my = (n - 1) / 2, statistics.fmean(values)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, values, strict=True)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, values, strict=True)]
    se = (sum(r * r for r in resid) / (n - 2) / sxx) ** 0.5
    return Trend(round(slope * PER_GAMES, 2), round(se * PER_GAMES, 2), n)


def summarize(dated: list[tuple[datetime, GameScore]], tier_bucket: str,
              ranks: list[RankPoint]) -> ProgressReport:
    """``dated`` is (game start, score) for each game, in any order."""
    dated = sorted(dated, key=lambda t: t[0])
    by_week: dict[date, list[tuple[GameScore, float]]] = defaultdict(list)
    series: list[float] = []
    area_series: dict[str, list[float]] = defaultdict(list)
    for at, s in dated:
        value = game_score(s)
        if value is None:
            continue
        by_week[week_start(at)].append((s, value))
        series.append(value)
        per_area: dict[str, list[float]] = defaultdict(list)
        for sc in s.game.values():
            if sc.metric.coachable:
                per_area[sc.metric.area].append(sc.goodness)
        for area, v in per_area.items():
            area_series[area].append(statistics.fmean(v))

    weeks = []
    for start in sorted(by_week):
        games = by_week[start]
        areas: dict[str, list[float]] = defaultdict(list)
        for s, _ in games:
            for sc in s.game.values():
                if sc.metric.coachable:
                    areas[sc.metric.area].append(sc.goodness)
        weeks.append(Week(start, len(games), sum(s.participant.win for s, _ in games),
                          round(statistics.fmean(v for _, v in games), 1),
                          {a: round(statistics.fmean(v), 1) for a, v in areas.items()}))
    return ProgressReport(tier_bucket, weeks, fit(series), ranks,
                          {a: fit(v) for a, v in area_series.items()})
