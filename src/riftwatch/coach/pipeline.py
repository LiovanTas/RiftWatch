"""End to end: stored features -> scores vs baselines -> evidence -> coaching (cached).

Everything here reads the indexed feature tables (never raw Riot JSON), baselines come
from the in-process cache, and LLM answers are stored in ``coach_reports`` keyed on the
evidence fingerprint + model, so only the first view of a given game pays for the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from riftwatch.analysis.score import GameScore, score_participant
from riftwatch.analysis.trend import trends
from riftwatch.baselines.build import baselines_for
from riftwatch.coach.evidence import EvidenceSet, game_evidence, trend_evidence
from riftwatch.coach.grounding import CoachOutput, CoachPoint
from riftwatch.coach.llm import Coach
from riftwatch.db import repo
from riftwatch.features import store
from riftwatch.features.extract import GameFeatures, ParticipantFeatures


class ReportError(LookupError):
    pass


@dataclass
class CoachResult:
    scope: str                       # 'game' | 'recent'
    puuid: str
    tier_bucket: str
    evidence: EvidenceSet
    output: CoachOutput
    model: str                       # 'offline' when no LLM was used
    cached: bool = False
    dropped: list[CoachPoint] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    match_id: str | None = None
    scores: list[GameScore] = field(default_factory=list)
    games: list[tuple[GameFeatures, ParticipantFeatures]] = field(default_factory=list)


# -- offline coach -------------------------------------------------------------------------

_ADVICE = {
    "farming": "Prioritise last-hitting every wave you can reach; between fights, take the "
               "closest wave or camp instead of waiting.",
    "laning": "Trade around your opponent's cooldowns and last hits, and recall on wave "
              "states that don't cost you minions.",
    "fighting": "Group for fights your team is likely to take and stay in range to "
                "contribute damage rather than arriving late.",
    "survival": "Before walking into fog or extending, check where the enemy jungler and "
                "missing laners were last seen; ward or back off if you don't know.",
    "vision": "Buy a control ward every back and use your trinket on cooldown around the "
              "next objective.",
    "objectives": "Track dragon, grubs, herald and baron timers and be near the pit before "
                  "they spawn.",
    "economy": "Keep collecting gold between fights -- waves, camps and plates -- rather "
               "than idling.",
}


# Where a role plays the area differently, its advice replaces the general one.
_ROLE_ADVICE = {
    "JUNGLE": {
        "farming": "Clear camps efficiently and keep farming between ganks; take nearby camps "
                   "on the way to plays instead of walking past them.",
        "laning": "Your lane opponent here is the enemy jungler: track their start and path, "
                  "contest or invade where your lanes have priority, and don't fall behind on "
                  "camps.",
    },
    "UTILITY": {
        "laning": "Look for trades when the enemy ADC goes for last hits, and keep your ADC "
                  "safe through their power spikes.",
    },
}


def offline_coach(evidence: EvidenceSet, limit: int = 6, role: str = "") -> CoachOutput:
    """Template coaching straight from the evidence, for when no LLM is configured. Every
    point quotes its evidence sentence, so it is grounded by construction."""
    ranked = sorted(
        (e for e in evidence.items if e.polarity in ("weakness", "strength")),
        key=lambda e: (e.polarity != "weakness", -e.severity),
    )
    points = []
    for e in ranked[:limit]:
        weak = e.polarity == "weakness"
        points.append(CoachPoint(
            area=e.area if e.area in _ADVICE else "survival",
            kind=e.polarity,
            title=("Work on " if weak else "Keep up ") + e.area,
            explanation=e.text,
            advice=(_ROLE_ADVICE.get(role, {}).get(e.area)
                    or _ADVICE.get(e.area, _ADVICE["survival"])) if weak else
                   "This is a strength; keep doing what produced it.",
            evidence_ids=[e.id],
        ))
    # No counts in the headline: "4 areas to work on" is a number no evidence item states,
    # and the grounding check (rightly) rejects it.
    if not points:
        headline = "Nothing stood out against the comparison group."
    elif points[0].kind == "weakness":
        headline = (f"Biggest gap: {points[0].area}. What to work on and what to keep, "
                    "from the measured evidence.")
    else:
        headline = "No clear weaknesses against the comparison group; strengths below."
    return CoachOutput(headline=headline, points=points)


# -- cache -----------------------------------------------------------------------------------

def _cached(conn, puuid, scope, match_id, fingerprint, model):
    row = conn.execute(
        """
        SELECT output, dropped, usage FROM coach_reports
         WHERE puuid = %s AND scope = %s AND coalesce(match_id, '') = coalesce(%s, '')
           AND evidence_fingerprint = %s AND model = %s
        """,
        (puuid, scope, match_id, fingerprint, model),
    ).fetchone()
    if row is None:
        return None
    output, dropped, usage = row
    return CoachOutput(**output), [CoachPoint(**d) for d in dropped], usage


def _store(conn, puuid, scope, match_id, evidence, model, output, dropped, usage):
    conn.execute(
        """
        INSERT INTO coach_reports (puuid, scope, match_id, evidence_fingerprint, model,
                                   evidence, output, dropped, usage)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """,
        (puuid, scope, match_id, evidence.fingerprint(), model, Jsonb(evidence.to_json()),
         Jsonb(output.model_dump()), Jsonb([d.model_dump() for d in dropped]), Jsonb(usage)),
    )


def _coach(conn, coach: Coach | None, puuid, scope, match_id, evidence, task, refresh,
           role: str = ""):
    if coach is None:
        return offline_coach(evidence, role=role), "offline", False, [], {}
    fingerprint = evidence.fingerprint()
    if not refresh:
        hit = _cached(conn, puuid, scope, match_id, fingerprint, coach.model)
        if hit:
            output, dropped, usage = hit
            return output, coach.model, True, dropped, usage
    run = coach.write(evidence, task)
    _store(conn, puuid, scope, match_id, evidence, coach.model, run.output, run.dropped,
           {**run.usage, "attempts": run.attempts, "served_by": run.model,
            "first_violations": [str(v) for v in run.first_violations]})
    return run.output, coach.model, False, run.dropped, run.usage


# -- reports ---------------------------------------------------------------------------------

def player_bucket(conn: psycopg.Connection, puuid: str, override: str | None = None) -> str:
    if override:
        return repo.tier_bucket(override)
    rank = repo.latest_rank(conn, puuid)
    if rank is None:
        raise ReportError("no solo/duo rank on record for this player -- sync them first, "
                          "or pass --tier (e.g. --tier gold)")
    return repo.tier_bucket(rank["tier"])


def _game(conn: psycopg.Connection, match_id: str) -> GameFeatures:
    game = store.load_stored(conn, match_id)
    if game is None:
        loaded = store.load(conn, match_id)
        if loaded is None:
            raise ReportError(f"{match_id} is not cached with a timeline -- sync it first")
        store.save(conn, loaded)
        game = store.load_stored(conn, match_id)
    return game


def _score(conn, p: ParticipantFeatures, bucket: str, min_n: int) -> GameScore:
    return score_participant(p, baselines_for(conn, bucket, p.role, p.champion_id, min_n))


def game_report(
    conn: psycopg.Connection,
    puuid: str,
    match_id: str,
    *,
    coach: Coach | None = None,
    tier: str | None = None,
    refresh: bool = False,
    min_n: int = 20,
) -> CoachResult:
    game = _game(conn, match_id)
    p = game.by_puuid(puuid)
    if p is None:
        raise ReportError(f"this player is not in {match_id}")
    bucket = player_bucket(conn, puuid, tier)
    score = _score(conn, p, bucket, min_n)
    evidence = game_evidence(game, score)
    output, model, cached, dropped, usage = _coach(
        conn, coach, puuid, "game", match_id, evidence, "this single game", refresh, p.role)
    return CoachResult("game", puuid, bucket, evidence, output, model, cached, dropped, usage,
                       match_id, [score], [(game, p)])


def recent_report(
    conn: psycopg.Connection,
    puuid: str,
    *,
    games: int = 20,
    coach: Coach | None = None,
    tier: str | None = None,
    refresh: bool = False,
    min_n: int = 20,
    queue_id: int = 420,
) -> CoachResult:
    bucket = player_bucket(conn, puuid, tier)
    # Fetch older games too, for the "before" side of each trend.
    ids = repo.player_match_ids(conn, puuid, queue_id=queue_id, limit=games * 2)
    stored = store.load_stored_many(conn, ids, minutes_for=puuid)
    loaded: list[tuple[GameFeatures, ParticipantFeatures]] = []
    scores: list[GameScore] = []
    for match_id in ids:
        game = stored.get(match_id)
        if game is None:
            try:
                game = _game(conn, match_id)    # not extracted yet: extract once, then stored
            except ReportError:
                continue
        p = game.by_puuid(puuid)
        if p is None or not p.role or game.duration_s < 600:
            continue
        loaded.append((game, p))
        scores.append(_score(conn, p, bucket, min_n))
    if not loaded:
        raise ReportError("no analysable games cached for this player -- run sync first")
    tier_label = bucket.replace("_", " ").title()
    evidence = trend_evidence(trends(scores, recent=games), loaded[:games], tier_label)
    output, model, cached, dropped, usage = _coach(
        conn, coach, puuid, "recent", None, evidence, "their recent ranked games", refresh)
    return CoachResult("recent", puuid, bucket, evidence, output, model, cached, dropped,
                       usage, None, scores[:games], loaded[:games])
