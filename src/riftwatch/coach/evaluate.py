"""Evaluate the coach on a fixed sample of real games.

The grounding check guarantees every number in a delivered point comes from the evidence the
point cites. This measures how the coach behaves around that guarantee, game after game:

* grounding -- how often the first answer passes, how often a retry is needed, how many
  points are dropped;
* faithfulness the validator doesn't check -- a "weakness" point citing evidence marked as a
  strength (or the reverse), a point filed under a different area than its evidence, advice
  that contains numbers despite the rule;
* usefulness -- whether the most severe weaknesses get addressed, points that cite only the
  game summary, two points built on the same evidence;
* cost and latency per game.

Games are sampled from the crawled ladder games, spread over rank buckets and roles, chosen
by a hash of the match id so the same seed gives the same sample.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import psycopg

from riftwatch.coach.evidence import EvidenceSet
from riftwatch.coach.grounding import CoachOutput, numbers, validate
from riftwatch.coach.llm import CoachRun
from riftwatch.coach.pipeline import game_report, offline_coach

EVAL_MODEL = "claude-sonnet-5-5"     # the model the coach ships with; what the eval measures

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
TOP_WEAKNESSES = 3      # "addressed the biggest problems" looks at this many


@dataclass
class GameEval:
    match_id: str
    tier_bucket: str
    role: str
    champion: str
    evidence_items: int
    attempts: int
    first_pass: bool
    first_violations: int
    points: int
    dropped: int
    kind_mismatches: int        # weakness point citing only strengths, or the reverse
    area_mismatches: int        # point area matches none of its cited metric evidence
    advice_with_numbers: int
    context_only: int           # points citing nothing but the game summary
    shared_evidence: int        # points whose evidence another point already used
    top_weaknesses_addressed: float   # share of the most severe weakness items cited
    cost_usd: float | None
    seconds: float
    error: str | None = None
    served_by: str = ""                                    # model that answered (fallbacks)
    violations: list[str] = field(default_factory=list)   # first answer's, for diagnosis
    answer: dict[str, Any] = field(default_factory=dict)   # the delivered coaching


@dataclass
class EvalReport:
    model: str
    games: list[GameEval] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        ok = [g for g in self.games if g.error is None]
        if not ok:
            return {"games": 0, "errors": len(self.games)}
        points = sum(g.points for g in ok)
        produced = points + sum(g.dropped for g in ok)
        costs = [g.cost_usd for g in ok if g.cost_usd is not None]
        secs = sorted(g.seconds for g in ok)
        return {
            "games": len(ok),
            "errors": len(self.games) - len(ok),
            "first_pass_rate": round(sum(g.first_pass for g in ok) / len(ok), 3),
            "retry_rate": round(sum(g.attempts > 1 for g in ok) / len(ok), 3),
            "points_delivered": points,
            "points_dropped": sum(g.dropped for g in ok),
            "drop_rate": round(sum(g.dropped for g in ok) / produced, 3) if produced else 0.0,
            "kind_mismatch_rate": _rate(sum(g.kind_mismatches for g in ok), points),
            "area_mismatch_rate": _rate(sum(g.area_mismatches for g in ok), points),
            "advice_with_numbers_rate": _rate(sum(g.advice_with_numbers for g in ok), points),
            "context_only_rate": _rate(sum(g.context_only for g in ok), points),
            "shared_evidence_rate": _rate(sum(g.shared_evidence for g in ok), points),
            "top_weaknesses_addressed": round(statistics.mean(g.top_weaknesses_addressed for g in ok), 3),
            "cost_usd_total": round(sum(costs), 4) if costs else None,
            "cost_usd_per_game": round(statistics.mean(costs), 4) if costs else None,
            "seconds_p50": round(statistics.median(secs), 1),
            "seconds_p90": round(secs[min(len(secs) - 1, int(0.9 * len(secs)))], 1),
            "by_bucket": dict(Counter(g.tier_bucket for g in ok)),
            "by_role": dict(Counter(g.role for g in ok)),
        }

    def to_json(self) -> dict[str, Any]:
        return {"model": self.model, "summary": self.summary(),
                "games": [asdict(g) for g in self.games]}


def _rate(n: int, d: int) -> float:
    return round(n / d, 3) if d else 0.0


def score_output(output: CoachOutput, evidence: EvidenceSet) -> dict[str, Any]:
    """The checks that don't need the model: everything here reads the delivered answer and
    the evidence it was given."""
    by_id = evidence.by_id()
    kind = area = advice = context_only = shared = 0
    used: set[str] = set()
    for p in output.points:
        cited = [by_id[e] for e in p.evidence_ids if e in by_id]
        substantive = [e for e in cited if e.kind != "context"]
        if not substantive:
            context_only += 1
        polarities = {e.polarity for e in substantive} - {"neutral"}
        if polarities and p.kind not in polarities:
            kind += 1
        areas = {e.area for e in substantive if e.kind in ("metric", "curve", "trend")}
        if areas and p.area not in areas:
            area += 1
        if numbers(p.advice):
            advice += 1
        ids = {e.id for e in substantive}
        if ids and ids <= used:
            shared += 1
        used |= ids
    weaknesses = sorted((e for e in evidence.items if e.polarity == "weakness"),
                        key=lambda e: -e.severity)[:TOP_WEAKNESSES]
    cited_all = {e for p in output.points for e in p.evidence_ids}
    addressed = (sum(e.id in cited_all for e in weaknesses) / len(weaknesses)) if weaknesses else 1.0
    return {"kind_mismatches": kind, "area_mismatches": area, "advice_with_numbers": advice,
            "context_only": context_only, "shared_evidence": shared,
            "top_weaknesses_addressed": round(addressed, 3)}


def sample_games(conn: psycopg.Connection, n: int, seed: int = 1) -> list[tuple[str, str, str, str]]:
    """(match_id, puuid, tier_bucket, role) for ``n`` crawled games, round-robin over rank
    buckets and roles so every combination shows up, picked by a seeded hash."""
    rows = conn.execute(
        """
        SELECT s.match_id, p.puuid, ms.tier_bucket, s.role
          FROM participant_game_summary s
          JOIN match_participants p USING (match_id, participant_id)
          JOIN match_samples ms USING (match_id)
          JOIN matches m USING (match_id)
         WHERE ms.source = 'crawl' AND m.queue_id = 420 AND m.duration_s >= 900
           AND s.role = ANY(%s)
        """,
        (list(ROLES),),
    ).fetchall()

    def key(r) -> str:
        return hashlib.sha256(f"{seed}:{r[0]}:{r[1]}".encode()).hexdigest()

    groups: dict[tuple[str, str], list] = {}
    for r in sorted(rows, key=key):
        groups.setdefault((r[2], r[3]), []).append(r)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    picked, seen_matches, i = [], set(), 0
    while len(picked) < n and any(groups.values()):
        g = order[i % len(order)]
        i += 1
        while groups[g]:
            r = groups[g].pop(0)
            if r[0] not in seen_matches:
                seen_matches.add(r[0])
                picked.append(tuple(r))
                break
    return picked


class OfflineCoach:
    """The template coach behind the same interface, so the harness runs for free (tests,
    and a floor to compare the LLM against)."""

    label = "offline"

    def write(self, evidence: EvidenceSet, task: str) -> CoachRun:
        output = offline_coach(evidence)
        problems = validate(output, evidence)
        return CoachRun(output, [], problems, problems, 1, "offline", {})


def _tier_for(bucket: str) -> str:
    return "master" if bucket == "MASTER_PLUS" else bucket.lower()


def evaluate(conn: psycopg.Connection, coach, games: list[tuple[str, str, str, str]], *,
             advisor=None, progress: Callable[[str], None] | None = None) -> EvalReport:
    """Run ``coach`` on each game. Nothing is written to the coaching cache: the evidence is
    built without generating, then the coach is called directly."""
    report = EvalReport(getattr(coach, "label", "offline"))
    for n, (match_id, puuid, bucket, role) in enumerate(games, 1):
        try:
            result = game_report(conn, puuid, match_id, tier=_tier_for(bucket), generate=False,
                                 advisor=advisor)
            evidence = result.evidence
            started = time.perf_counter()
            run = coach.write(evidence, "this single game")
            seconds = time.perf_counter() - started
            game, p = result.games[0]
            report.games.append(GameEval(
                match_id=match_id, tier_bucket=bucket, role=role, champion=p.champion_name,
                evidence_items=len(evidence.items), attempts=run.attempts,
                first_pass=not run.first_violations, first_violations=len(run.first_violations),
                points=len(run.output.points), dropped=len(run.dropped),
                cost_usd=run.usage.get("cost_usd"), seconds=round(seconds, 2),
                served_by=run.model, violations=[str(v) for v in run.first_violations],
                answer=run.output.model_dump(),
                **score_output(run.output, evidence),
            ))
        except Exception as exc:  # one bad game is a data point, not the end of the run
            report.games.append(GameEval(match_id, bucket, role, "", 0, 0, False, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0.0, None, 0.0,
                                         error=f"{type(exc).__name__}: {exc}"))
        if progress:
            g = report.games[-1]
            progress(f"{n}/{len(games)} {match_id} {bucket.lower()} {role.lower()}: "
                     + (g.error or f"{g.points} points, {g.attempts} attempt(s), "
                                   f"{g.dropped} dropped, {g.seconds}s"))
    return report
