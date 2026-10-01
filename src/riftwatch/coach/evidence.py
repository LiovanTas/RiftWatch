"""Evidence: the only facts the LLM coach is allowed to talk about.

Each item has an id (E1, E2, ...) and a sentence that carries every number it supports.
The coach must cite ids for every point, and the grounding check accepts a number in the
coach's answer only if it appears in the text of an evidence item that point cites.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field

from riftwatch.analysis.score import GameScore, Score
from riftwatch.analysis.trend import MetricTrend
from riftwatch.features.extract import GameFeatures, ParticipantFeatures
from riftwatch.features.metrics import Metric

WEAK = 25.0      # goodness at or below this is a weakness
STRONG = 75.0    # at or above, a strength
MIN_RUN = 4      # minutes a curve must stay weak/strong to count as a pattern


@dataclass
class Evidence:
    id: str
    kind: str          # context | metric | curve | death | trend | pattern
    area: str
    polarity: str      # weakness | strength | neutral
    text: str
    severity: float = 0.0   # how far from typical; used for ordering
    data: dict = field(default_factory=dict)


@dataclass
class EvidenceSet:
    items: list[Evidence]

    def by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.items}

    def fingerprint(self) -> str:
        """Stable hash of the evidence content; a cached coach answer is reused only for
        exactly the same evidence."""
        payload = json.dumps([(e.id, e.text) for e in self.items], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def to_prompt(self) -> str:
        return "\n".join(f"[{e.id}] ({e.polarity}, {e.area}) {e.text}" for e in self.items)

    def to_json(self) -> list[dict]:
        return [asdict(e) for e in self.items]


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def clock(minutes: float) -> str:
    total = int(round(minutes * 60))
    return f"{total // 60}:{total % 60:02d}"


class _Group:
    """The default comparison group, named once in the context item. Lines whose baseline
    matches it say nothing more; the rest name their own group -- that repetition was a
    large share of the prompt."""

    def __init__(self, scores: list[Score]) -> None:
        common = Counter((s.baseline.scope, s.baseline.n) for s in scores).most_common(1)
        self.scope, self.n = common[0][0] if common else ("", 0)

    def suffix(self, s: Score) -> str:
        if (s.baseline.scope, s.baseline.n) == (self.scope, self.n):
            return ""
        if s.baseline.scope == self.scope:
            return f" (n={s.baseline.n})"
        return f" (vs {s.baseline.scope}, n={s.baseline.n})"


def _vs_median(m: Metric, value: float, median: float) -> str:
    diff = value - median
    if m.fmt.endswith("%}"):
        return f"median {m.show(median)}"
    word = "above" if diff > 0 else "below"
    return f"median {m.show(median)}, {m.show(abs(diff)).lstrip('+')} {word} it" if diff else f"median {m.show(median)}"


def _polarity(goodness: float) -> str | None:
    if goodness <= WEAK:
        return "weakness"
    if goodness >= STRONG:
        return "strength"
    return None


class _Builder:
    def __init__(self) -> None:
        self.items: list[Evidence] = []

    def add(self, kind: str, area: str, polarity: str, text: str, severity: float = 0.0,
            **data) -> Evidence:
        e = Evidence(f"E{len(self.items) + 1}", kind, area, polarity, text, severity, data)
        self.items.append(e)
        return e


def _metric_text(s: Score, group: _Group) -> str:
    m = s.metric
    return (f"{m.label}: {m.show(s.value)}; {ordinal(s.percentile)} percentile"
            f"{group.suffix(s)}; {_vs_median(m, s.value, s.baseline.p50)}.")


def _runs(scores: list[Score], test) -> list[list[Score]]:
    runs, cur = [], []
    for s in scores:
        if test(s.goodness):
            cur.append(s)
        else:
            if len(cur) >= MIN_RUN:
                runs.append(cur)
            cur = []
    if len(cur) >= MIN_RUN:
        runs.append(cur)
    return runs


def _death_text(d, p: ParticipantFeatures, game: GameFeatures) -> str:
    killer = game.participants.get(d.killer_participant_id)
    by = f"killed by {killer.champion_name}" if killer else "executed (no champion kill credit)"
    helpers = f" with {d.assisters} helper{'s' if d.assisters != 1 else ''}" if d.assisters else " in a 1v1"
    if d.gold_diff is None:
        state = ""
    elif d.gold_diff >= 0:
        state = f", while {d.gold_diff} gold ahead of the lane opponent"
    else:
        state = f", while {-d.gold_diff} gold behind the lane opponent"
    return f"Death at {clock(d.minute)} in the {d.where}, {by}{helpers}{state}."


def game_evidence(
    game: GameFeatures,
    score: GameScore,
    *,
    max_metric_findings: int = 8,
    max_per_area: int = 2,
    max_deaths: int = 6,
) -> EvidenceSet:
    p = score.participant
    b = _Builder()
    group = _Group(list(score.game.values()) + [s for c in score.curves.values() for s in c])
    any_baseline = next(iter(score.game.values()), None)
    compared = (f" Comparison group unless a line says otherwise: {group.scope}, n={group.n} "
                f"games, patches {any_baseline.baseline.patch_window}." if any_baseline else
                " No rank baseline was available, so no percentile comparisons.")
    b.add("context", "context", "neutral",
          f"Game {game.match_id}: {p.champion_name} {p.role.lower()}, "
          f"{'win' if p.win else 'loss'}, length {clock(game.duration_min)}, patch {game.patch}; "
          f"final score {int(p.metrics.get('kills', 0))}/{int(p.metrics.get('deaths', 0))}/"
          f"{int(p.metrics.get('assists', 0))}.{compared}")

    # Whole-game metrics, most extreme first, capped per area so one area can't crowd out
    # the rest.
    candidates = []
    for s in score.game.values():
        if not s.metric.coachable:
            continue
        pol = _polarity(s.goodness)
        if pol:
            candidates.append((abs(s.goodness - 50), pol, s))
    candidates.sort(key=lambda c: -c[0])
    per_area: Counter[str] = Counter()
    taken = 0
    for severity, pol, s in candidates:
        if taken >= max_metric_findings or per_area[s.metric.area] >= max_per_area:
            continue
        b.add("metric", s.metric.area, pol, _metric_text(s, group), severity, metric=s.metric.name,
              value=s.value, percentile=round(s.percentile, 1), n=s.baseline.n)
        per_area[s.metric.area] += 1
        taken += 1

    # Per-minute curves: sustained stretches below/above the comparison group.
    for name, scores in score.curves.items():
        m = scores[0].metric
        if not m.coachable:
            continue
        for pol, test in (("weakness", lambda g: g <= WEAK), ("strength", lambda g: g >= STRONG)):
            for run in _runs(scores, test):
                last = run[-1]
                b.add("curve", m.area, pol,
                      f"From minute {run[0].minute} to minute {last.minute}, {m.label} stayed "
                      f"{'below the 25th' if pol == 'weakness' else 'above the 75th'} percentile"
                      f"{group.suffix(last)}; at minute {last.minute} it was "
                      f"{m.show(last.value)} ({_vs_median(m, last.value, last.baseline.p50)}).",
                      severity=sum(abs(s.goodness - 50) for s in run) / len(run) + len(run),
                      metric=name, start=run[0].minute, end=last.minute)

    # Deaths: the ones most worth talking about first.
    ranked = sorted(p.deaths, key=lambda d: (not d.ahead, not d.early, d.assisters > 0, d.minute))
    for d in sorted(ranked[:max_deaths], key=lambda d: d.minute):
        pol = "weakness" if (d.ahead or d.early or d.assisters == 0) else "neutral"
        b.add("death", "survival", pol, _death_text(d, p, game),
              severity=10 + 10 * d.ahead + 5 * d.early, minute=d.minute, zone=d.zone)
    if len(p.deaths) > max_deaths:
        b.add("death", "survival", "neutral",
              f"{len(p.deaths) - max_deaths} further deaths are not itemised.")

    return EvidenceSet(b.items)


def trend_evidence(
    trends: list[MetricTrend],
    recent_games: list[tuple[GameFeatures, ParticipantFeatures]],
    tier_label: str,
    *,
    max_findings: int = 10,
) -> EvidenceSet:
    b = _Builder()
    roles = Counter(p.role for _, p in recent_games)
    champs = Counter(p.champion_name for _, p in recent_games)
    wins = sum(p.win for _, p in recent_games)
    b.add("context", "context", "neutral",
          f"Last {len(recent_games)} ranked games: {wins} wins, {len(recent_games) - wins} losses. "
          f"Roles: {', '.join(f'{r.lower()} {n}' for r, n in roles.most_common())}. "
          f"Most played: {', '.join(f'{c} {n}' for c, n in champs.most_common(3))}. "
          f"Each game is compared against {tier_label} players in the role played; "
          f"percentiles below are for that group.")

    found = []
    for t in trends:
        if not t.metric.coachable:
            continue
        pol = _polarity(t.median_goodness)
        if pol:
            found.append((abs(t.median_goodness - 50), pol, t))
    found.sort(key=lambda f: -f[0])
    for severity, pol, t in found[:max_findings]:
        m = t.metric
        change = ""
        if t.change is not None:
            change = (f" In the {t.older_games} games before that it was the "
                      f"{ordinal(t.older_goodness)} percentile.")
        b.add("trend", m.area, pol,
              f"{m.label}: typical {m.show(t.median_value)} over {t.games} games, around the "
              f"{ordinal(t.median_goodness)} percentile.{change}", severity, metric=m.name)

    # Where deaths happen, across games.
    zones: Counter[str] = Counter()
    ahead = early = total = 0
    for _, p in recent_games:
        for d in p.deaths:
            zones[d.where] += 1
            ahead += d.ahead
            early += d.early
            total += 1
    if total:
        top = ", ".join(f"{z} {n}" for z, n in zones.most_common(3))
        b.add("pattern", "survival", "neutral",
              f"{total} deaths across {len(recent_games)} games; most common places: {top}. "
              f"{early} of them came before 14:00 and {ahead} came while 500+ gold ahead of "
              f"the lane opponent.", severity=0)
    return EvidenceSet(b.items)
