"""Champion pool: how a player performs on each champion they play.

Each game is scored the usual way -- against players at the player's rank in the same role,
adjusted for champion and lane matchup -- and summarised as one number: the mean "better than
N%" over the coachable whole-game stats. Lines are per champion and role, with the spread of
those per-game numbers, so a difference between two champions can be judged against noise.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from riftwatch.analysis.score import GameScore

MIN_GAMES = 5         # fewer, and a line is shown but never called stronger or weaker
CLEAR_GAP_SE = 2.0    # a gap counts once it is this many standard errors


@dataclass
class ChampionLine:
    champion: str
    champion_id: int
    role: str
    games: int
    wins: int
    kills: float
    deaths: float
    assists: float
    score: float                 # mean per-game "better than N%" over coachable stats
    score_se: float              # standard error of that mean
    areas: dict[str, float] = field(default_factory=dict)   # area -> mean goodness
    verdict: str = ""            # "stronger" / "weaker" than the player's other picks in the role
    gap: float = 0.0             # score minus the player's average on their other picks

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def best_area(self) -> str | None:
        return max(self.areas, key=self.areas.get) if self.areas else None

    @property
    def worst_area(self) -> str | None:
        return min(self.areas, key=self.areas.get) if self.areas else None


@dataclass
class PoolReport:
    tier_bucket: str
    games: int
    lines: list[ChampionLine]


def game_score(score: GameScore) -> float | None:
    """One game in one number: mean goodness over the coachable whole-game stats."""
    values = [s.goodness for s in score.game.values() if s.metric.coachable]
    return statistics.fmean(values) if values else None


def summarize(scores: list[GameScore], tier_bucket: str) -> PoolReport:
    by_pick: dict[tuple[int, str], list[tuple[GameScore, float]]] = defaultdict(list)
    for s in scores:
        value = game_score(s)
        if value is not None:
            by_pick[(s.participant.champion_id, s.participant.role)].append((s, value))

    lines = []
    for (champion_id, role), games in by_pick.items():
        values = [v for _, v in games]
        areas: dict[str, list[float]] = defaultdict(list)
        for s, _ in games:
            for sc in s.game.values():
                if sc.metric.coachable:
                    areas[sc.metric.area].append(sc.goodness)
        n = len(games)
        first = games[0][0].participant

        def mean(key: str) -> float:
            return round(statistics.fmean(s.participant.metrics.get(key, 0) for s, _ in games), 1)

        lines.append(ChampionLine(
            champion=first.champion_name, champion_id=champion_id, role=role, games=n,
            wins=sum(s.participant.win for s, _ in games),
            kills=mean("kills"), deaths=mean("deaths"), assists=mean("assists"),
            score=round(statistics.fmean(values), 1),
            score_se=round(statistics.stdev(values) / n ** 0.5, 1) if n > 1 else 0.0,
            areas={a: round(statistics.fmean(v), 1) for a, v in areas.items()},
        ))

    # Each pick against the player's other picks in the same role, game-weighted.
    for line in lines:
        others = [v for (cid, role), games in by_pick.items()
                  if role == line.role and cid != line.champion_id for _, v in games]
        if not others or line.games < MIN_GAMES:
            continue
        line.gap = round(line.score - statistics.fmean(others), 1)
        se_others = statistics.stdev(others) / len(others) ** 0.5 if len(others) > 1 else 0.0
        if abs(line.gap) > CLEAR_GAP_SE * (line.score_se ** 2 + se_others ** 2) ** 0.5:
            line.verdict = "stronger" if line.gap > 0 else "weaker"

    lines.sort(key=lambda l: (-l.games, -l.score))
    return PoolReport(tier_bucket, sum(l.games for l in lines), lines)
