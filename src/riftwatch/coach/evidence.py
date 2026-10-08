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
from riftwatch.riot.api import QUEUE_NAMES, RANKED_SOLO_QUEUE_ID

WEAK = 25.0      # goodness at or below this is a weakness
STRONG = 75.0    # at or above, a strength
MIN_RUN = 4      # minutes a curve must stay weak/strong to count as a pattern


@dataclass
class Evidence:
    id: str
    kind: str          # context | metric | curve | death | trend | pattern | live
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


def _metric_text(s: Score, group: _Group, reference=None, reference_label: str = "") -> str:
    m = s.metric
    # "better than N%" reads the same way for every metric: for deaths it already accounts
    # for lower being better, so it can't be misread the way a raw percentile can.
    text = (f"{m.label}: {m.show(s.value)}; better than {round(s.goodness)}% of comparable "
            f"players{group.suffix(s)}; {_vs_median(m, s.value, s.baseline.p50)}.")
    if reference is not None:
        text += f" {reference_label} median: {m.show(reference.p50)}."
    return text


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
    live: dict | None = None,
    review=None,
) -> EvidenceSet:
    p = score.participant
    b = _Builder()
    group = _Group(list(score.game.values()) + [s for c in score.curves.values() for s in c])
    any_baseline = next(iter(score.game.values()), None)
    compared = (f" Comparison group unless a line says otherwise: {group.scope}, n={group.n} "
                f"games, patches {any_baseline.baseline.patch_window}." if any_baseline else
                " No rank baseline was available, so no percentile comparisons.")
    opponent = game.participants.get(p.opponent_id) if p.opponent_id else None
    against = f" against {opponent.champion_name}" if opponent else ""
    mode = QUEUE_NAMES.get(game.queue_id, "other mode")
    if game.queue_id != RANKED_SOLO_QUEUE_ID and any_baseline:
        compared += " Comparisons are with ranked solo/duo players at the same rank."
    b.add("context", "context", "neutral",
          f"Game {game.match_id} ({mode}): {p.champion_name} {p.role.lower()}{against}, "
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
        b.add("metric", s.metric.area, pol,
              _metric_text(s, group, score.reference.get(s.metric.name), score.reference_label),
              severity, metric=s.metric.name,
              value=s.value, better_than=round(s.goodness, 1), n=s.baseline.n)
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
                      f"in the {'bottom' if pol == 'weakness' else 'top'} quarter of comparable players"
                      f"{group.suffix(last)}; at minute {last.minute} it was "
                      f"{m.show(last.value)} ({_vs_median(m, last.value, last.baseline.p50)}).",
                      severity=sum(abs(s.goodness - 50) for s in run) / len(run) + len(run),
                      metric=name, start=run[0].minute, end=last.minute)

    # Deaths: the counts first, so the coach can cite totals instead of counting the itemised
    # deaths itself (the grounding check rejects a number that isn't written in the evidence;
    # on the coach eval, every first-answer violation was such a self-made count).
    if len(p.deaths) >= 2:
        killers = Counter(game.participants[d.killer_participant_id].champion_name
                          for d in p.deaths if d.killer_participant_id in game.participants)
        top = killers.most_common(1)
        b.add("pattern", "survival", "neutral",
              f"{len(p.deaths)} deaths in total: {sum(d.early for d in p.deaths)} before 14:00, "
              f"{sum(d.ahead for d in p.deaths)} while 500+ gold ahead of the lane opponent, "
              f"{sum(d.assisters == 0 and d.killer_participant_id > 0 for d in p.deaths)} with only "
              f"the killer involved"
              + (f"; {top[0][0]} got {top[0][1]} of the kills" if top and top[0][1] >= 2 else "")
              + ".", severity=0)

    # Deaths: the ones most worth talking about first.
    ranked = sorted(p.deaths, key=lambda d: (not d.ahead, not d.early, d.assisters > 0, d.minute))
    for d in sorted(ranked[:max_deaths], key=lambda d: d.minute):
        pol = "weakness" if (d.ahead or d.early or d.assisters == 0) else "neutral"
        b.add("death", "survival", pol, _death_text(d, p, game),
              severity=10 + 10 * d.ahead + 5 * d.early, minute=d.minute, zone=d.zone)
    if len(p.deaths) > max_deaths:
        b.add("death", "survival", "neutral",
              f"{len(p.deaths) - max_deaths} further deaths are not itemised.")

    if live:
        _live_evidence(b, live["summary"], len(p.deaths))
    if review is not None:
        _review_evidence(b, review)
    return EvidenceSet(b.items)


def add_pool_evidence(evidence: EvidenceSet, pool, max_lines: int = 3,
                      min_games: int = 3) -> None:
    """Append one item per champion the player played at least ``min_games`` times in these
    games: record, average standing, best and worst area, and whether it is clearly stronger
    or weaker than their other picks in that role."""
    n = len(evidence.items)
    for line in [l for l in pool.lines if l.games >= min_games][:max_lines]:
        verdict = (f" That is clearly {line.verdict} than their other {line.role.lower()} picks "
                   f"in these games." if line.verdict else "")
        areas = (f" Strongest area {line.best_area}, weakest {line.worst_area}."
                 if line.areas and line.best_area != line.worst_area else "")
        n += 1
        evidence.items.append(Evidence(
            f"E{n}", "pattern", "context",
            {"stronger": "strength", "weaker": "weakness"}.get(line.verdict, "neutral"),
            f"On {line.champion} ({line.role.lower()}, {line.games} of these games): {line.wins} "
            f"wins; on average better than {line.score:.0f}% of comparable players across "
            f"all stats.{areas}{verdict}",
            severity=abs(line.gap), data={"champion": line.champion, "role": line.role}))


def add_video_evidence(evidence: EvidenceSet, mine, library=None, model=None) -> None:
    """Laning from the player's own gameplay video of this game, against the high-elo video
    library and the laning model when there are enough games behind them."""
    s = mine.summary()
    if not s["trades"]:
        return
    items = [("pattern", "laning", "neutral",
              f"From the player's gameplay video of this game, laning until 14:00: {s['trades']} "
              f"trades with the lane opponent -- {s['won']} won, {s['lost']} lost, {s['even']} "
              f"even (won means taking at least 5 health points more than they lost); net "
              f"{s['net']:+.1f} health points per trade; the player started {s['started']} of "
              f"them.", 5.0)]
    if library is not None and library.trades:
        worse = s["net"] < library.net_per_trade - 5
        items.append(("pattern", "laning", "weakness" if worse else "neutral",
                      f"High-elo {(mine.role or 'laners').lower()} players in the video library "
                      f"({library.videos} games): {library.trades_per_10_min:.1f} trades per 10 "
                      f"minutes of laning, won {library.won} and lost {library.lost} of "
                      f"{library.trades}, net {library.net_per_trade:+.1f} health points per "
                      f"trade.", abs(s["net"] - library.net_per_trade)))
    if model is not None and mine.situations and model.get("outcome") is not None:
        from riftwatch.vision.learn import judge

        expected = [net for _, net in judge(model, mine.situations) if net is not None]
        started = [t for t in mine.trades if t[3] in ("you", "both")]
        if expected and started:
            exp = 100 * sum(expected) / len(expected)
            got = 100 * sum(t[2] - t[1] for t in started) / len(started)
            items.append(("pattern", "laning", "weakness" if got < exp - 5 else "neutral",
                          f"From the spots where the player started trades, the laning model "
                          f"(trained on high-elo videos) expected a net of {exp:+.1f} health "
                          f"points per trade; the player's trades from those spots netted "
                          f"{got:+.1f}.", abs(got - exp)))
    n = len(evidence.items)
    for kind, area, polarity, text, severity in items:
        n += 1
        evidence.items.append(Evidence(f"E{n}", kind, area, polarity, text, severity=severity,
                                       data={"source": "video"}))


def add_session_evidence(evidence: EvidenceSet, report) -> None:
    """Append the session patterns that cleared the noise bar (see analysis.sessions)."""
    n = len(evidence.items)
    for f in report.findings:
        n += 1
        evidence.items.append(Evidence(
            f"E{n}", "pattern", "context", "weakness" if f.worse else "neutral", f.text,
            severity=abs(f.gap), data={"session": f.kind}))


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
    modes = Counter(QUEUE_NAMES.get(g.queue_id, "other") for g, _ in recent_games)
    b.add("context", "context", "neutral",
          f"Last {len(recent_games)} games ({', '.join(f'{m} {n}' for m, n in modes.most_common())}): "
          f"{wins} wins, {len(recent_games) - wins} losses. "
          f"Roles: {', '.join(f'{r.lower()} {n}' for r, n in roles.most_common())}. "
          f"Most played: {', '.join(f'{c} {n}' for c, n in champs.most_common(3))}. "
          f"Each game is compared against {tier_label} ranked solo/duo players in the role played; "
          f"comparisons below are with that group.")

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
            change = (f" In the {t.older_games} games before that it was better than about "
                      f"{round(t.older_goodness)}%.")
        b.add("trend", m.area, pol,
              f"{m.label}: typical {m.show(t.median_value)} over {t.games} games, better than about "
              f"{round(t.median_goodness)}% of comparable players.{change}", severity, metric=m.name)

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


def _live_evidence(b: _Builder, summary: dict, total_deaths: int) -> None:
    """Facts from a second-by-second recording of the game (riftwatch record)."""
    m = summary.get("metrics", {})
    drops = summary.get("drops", [])
    early = int(m.get("early_big_hp_losses", 0))
    if early:
        to_recall = int(m.get("early_losses_to_recall", 0))
        to_death = int(m.get("early_losses_to_death", 0))
        b.add("live", "laning", "weakness" if to_recall + to_death >= 2 else "neutral",
              f"Live recording: before 14:00 you lost a fifth or more of your health in a short "
              f"window {early} time{'s' if early != 1 else ''}; {to_recall} of those were followed "
              f"by a recall within a minute and {to_death} by a death within 20 seconds.",
              severity=5 * (to_recall + to_death), drops=len(drops))
        worst = max((d for d in drops if d["start"] < 14 * 60), key=lambda d: d["lost"])
        b.add("live", "laning", "neutral",
              f"Biggest early health loss: {round(worst['lost'] * 100)}% of your health between "
              f"{clock(worst['start'] / 60)} and {clock(worst['end'] / 60)}, leaving you on "
              f"{round(worst['hp_after'] * 100)}%; it was followed by "
              f"{ {'death': 'a death', 'recall': 'a recall', 'stayed': 'you staying in lane'}[worst['led_to']] }.",
              severity=1)
    recalls = int(m.get("recalls", 0))
    if recalls and "avg_recall_hp" in m:
        b.add("live", "economy", "neutral",
              f"Live recording: you recalled {recalls} time{'s' if recalls != 1 else ''}, on average "
              f"with {round(m['avg_recall_hp'] * 100)}% health and {int(m['avg_recall_gold'])} "
              f"unspent gold.", severity=0)
    burst = int(m.get("burst_deaths", 0))
    if burst and total_deaths:
        b.add("live", "survival", "weakness" if burst >= 2 else "neutral",
              f"Live recording: {burst} of your {total_deaths} deaths went from above 60% health "
              f"to dead in 3 seconds or less.", severity=4 * burst)


HIGH_ELO = "Grandmaster/Challenger"
_ROLE_NAME = {"TOP": "top laners", "JUNGLE": "junglers", "MIDDLE": "mid laners",
              "BOTTOM": "ADCs", "UTILITY": "supports"}


def _pct(x: float) -> int:
    return round(x * 100)


def _review_evidence(b: _Builder, review) -> None:
    """Facts from comparing the player's decisions with high-elo play (ml.advisor)."""
    from riftwatch.ml.advisor import DID

    who = f"{HIGH_ELO} {_ROLE_NAME.get(review.role, 'players')}"
    b.add("context", "macro", "neutral",
          f"High-elo comparison: across {review.minutes} minutes of this game, your move matched "
          f"the most common choice of {who} in similar situations {_pct(review.agreement)}% of the "
          "time.")
    for m in review.moments:
        did, best = m.did, m.best
        text = (f"At {m.minute}:00 you {DID[did.decision]}. In similar situations only "
                f"{_pct(did.share)}% of {who} did that; {_pct(best.share)}% {DID[best.decision]}.")
        if m.outcomes_meaningful:
            text += (f" In those games the team took an objective within the next 3 minutes "
                     f"{_pct(best.objective)}% of the time after the common choice versus "
                     f"{_pct(did.objective)}% after yours, and the player died "
                     f"{_pct(best.death)}% versus {_pct(did.death)}% of the time.")
        b.add("decision", "macro", "weakness", text, severity=10 + 20 * m.gain,
              minute=m.minute, decision=did.decision, alternative=best.decision)
    for m in review.good:
        b.add("decision", "macro", "strength",
              f"At {m.minute}:00 you {DID[m.did.decision]}, the choice {_pct(m.did.share)}% of "
              f"{who} made in similar situations.", severity=8, minute=m.minute,
              decision=m.did.decision)
