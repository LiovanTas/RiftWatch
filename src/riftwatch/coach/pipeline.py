"""End to end: stored features -> scores vs baselines -> evidence -> coaching (cached).

Everything here reads the indexed feature tables (never raw Riot JSON), baselines come
from the in-process cache, and LLM answers are stored in ``coach_reports`` keyed on the
evidence fingerprint + model, so only the first view of a given game pays for the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from riftwatch.analysis import progress as progress_mod
from riftwatch.analysis.pool import PoolReport, summarize
from riftwatch.analysis.score import GameScore, score_participant
from riftwatch.analysis.trend import trends
from riftwatch.baselines import build as baseline_build
from riftwatch.baselines.build import baselines_for
from riftwatch.coach.evidence import EvidenceSet, add_pool_evidence, game_evidence, trend_evidence
from riftwatch.coach.grounding import CoachOutput, CoachPoint
from riftwatch.coach.llm import Coach
from riftwatch.db import repo
from riftwatch.features import store
from riftwatch.features.extract import GameFeatures, ParticipantFeatures
from riftwatch.features.metrics import LANE_LEAD_METRICS
from riftwatch.riot.api import RANKED_FLEX, SUPPORTED_QUEUES


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
    # True when an LLM coach is configured but its answer isn't cached yet and generation
    # wasn't requested; the output is the offline coach meanwhile.
    coach_pending: bool = False
    live: dict[str, Any] | None = None   # second-by-second recording, if one is linked
    review: Any = None                   # comparison with high-elo play (ml.advisor)


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
    "macro": "Before moving, check what's up next on the map -- objective timers, which lanes "
             "are pushed, where the enemy jungler was last seen -- and go where your team is "
             "strongest.",
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
           role: str = "", generate: bool = True):
    """Returns (output, model, cached, dropped, usage, pending)."""
    if coach is None:
        return offline_coach(evidence, role=role), "offline", False, [], {}, False
    fingerprint = evidence.fingerprint()
    if not refresh:
        hit = _cached(conn, puuid, scope, match_id, fingerprint, coach.label)
        if hit:
            output, dropped, usage = hit
            return output, coach.label, True, dropped, usage, False
    if not generate:
        return offline_coach(evidence, role=role), "offline", False, [], {}, True
    run = coach.write(evidence, task)
    usage = {**run.usage, "attempts": run.attempts, "served_by": run.model,
             "first_violations": [str(v) for v in run.first_violations]}
    _store(conn, puuid, scope, match_id, evidence, coach.label, run.output, run.dropped, usage)
    return run.output, coach.label, False, run.dropped, usage, False


# -- reports ---------------------------------------------------------------------------------

def player_bucket(conn: psycopg.Connection, puuid: str, override: str | None = None) -> str:
    if override:
        return repo.tier_bucket(override)
    rank = repo.latest_rank(conn, puuid) or repo.latest_rank(conn, puuid, RANKED_FLEX)
    if rank is None:
        raise ReportError("no solo/duo or flex rank on record for this player -- sync them "
                          "first, or pass --tier (e.g. --tier gold)")
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


HIGH_ELO_BUCKET = "MASTER_PLUS"


def _add_reference(conn, score: GameScore, bucket: str, min_n: int) -> None:
    """Attach Master+ medians for the same role and champion (champion-adjusted unless the
    champion has enough Master+ games of its own) so each stat can show where high elo sits.
    Skipped for Master+ players themselves."""
    if bucket == HIGH_ELO_BUCKET:
        return
    p = score.participant
    ref = baselines_for(conn, HIGH_ELO_BUCKET, p.role, p.champion_id, min_n, max_tier_distance=0)
    for name in score.game:
        b = ref.get(name, None, score.opponent_champion_id if name in LANE_LEAD_METRICS else None)
        if b is not None:
            score.reference[name] = b


# Scored games, shared by the history views (recent games, champion pool, progress), which
# overlap heavily: a player page asks for all three. Keyed on what decides the result -- the
# game, the player, the comparison bucket, the baseline build, and whether per-minute curves
# were scored -- so a rebuild or a rank change simply misses. None marks a game that can't be
# scored (remake, no role), so it isn't reloaded either.
_SCORED: OrderedDict[tuple, tuple[GameFeatures, ParticipantFeatures, GameScore] | None] = OrderedDict()
SCORED_CACHE_MAX = 20_000


def _scored_games(
    conn, puuid: str, ids: list[str], bucket: str, min_n: int, *, minutes: bool,
    extract_missing: bool = False,
) -> list[tuple[GameFeatures, ParticipantFeatures, GameScore]]:
    """(game, player, score) for each scorable game in ``ids``, in order."""
    gen = baseline_build.generation(conn)

    def key(match_id: str) -> tuple:
        return (match_id, puuid, bucket, min_n, minutes, gen)

    missing = [m for m in ids if key(m) not in _SCORED]
    if missing:
        stored = store.load_stored_many(conn, missing, minutes_for=puuid if minutes else None,
                                        minutes=minutes, players_for=puuid)
        for match_id in missing:
            game = stored.get(match_id)
            if game is None and extract_missing:
                try:
                    game = _game(conn, match_id)    # not extracted yet: extract once, then stored
                except ReportError:
                    game = None
            p = game.by_puuid(puuid) if game else None
            if p is None or not p.role or game.duration_s < 600:
                if game is not None:
                    _SCORED[key(match_id)] = None
                continue
            _SCORED[key(match_id)] = (game, p, _score(conn, game, p, bucket, min_n))
        while len(_SCORED) > SCORED_CACHE_MAX:
            _SCORED.popitem(last=False)
    out = []
    for match_id in ids:
        hit = _SCORED.get(key(match_id))
        if hit is not None:
            _SCORED.move_to_end(key(match_id))
            out.append(hit)
    return out


def _score(conn, game: GameFeatures, p: ParticipantFeatures, bucket: str, min_n: int) -> GameScore:
    opponent = game.participants.get(p.opponent_id) if p.opponent_id else None
    return score_participant(p, baselines_for(conn, bucket, p.role, p.champion_id, min_n),
                             opponent.champion_id if opponent else None)


def game_report(
    conn: psycopg.Connection,
    puuid: str,
    match_id: str,
    *,
    coach: Coach | None = None,
    tier: str | None = None,
    refresh: bool = False,
    min_n: int = 20,
    generate: bool = True,
    advisor=None,
) -> CoachResult:
    game = _game(conn, match_id)
    p = game.by_puuid(puuid)
    if p is None:
        raise ReportError(f"this player is not in {match_id}")
    bucket = player_bucket(conn, puuid, tier)
    score = _score(conn, game, p, bucket, min_n)
    _add_reference(conn, score, bucket, min_n)
    from riftwatch.live.store import for_match

    live = for_match(conn, match_id)
    review = None
    if advisor is not None:
        raw_match, raw_timeline = repo.get_match(conn, match_id), repo.get_timeline(conn, match_id)
        if raw_match and raw_timeline:
            review = advisor.review(raw_match, raw_timeline, puuid)
    evidence = game_evidence(game, score, live=live, review=review)
    output, model, cached, dropped, usage, pending = _coach(
        conn, coach, puuid, "game", match_id, evidence, "this single game", refresh, p.role,
        generate)
    return CoachResult("game", puuid, bucket, evidence, output, model, cached, dropped, usage,
                       match_id, [score], [(game, p)], pending, live, review)


def recent_report(
    conn: psycopg.Connection,
    puuid: str,
    *,
    games: int = 20,
    coach: Coach | None = None,
    tier: str | None = None,
    refresh: bool = False,
    min_n: int = 20,
    queue_id: int | tuple[int, ...] = SUPPORTED_QUEUES,
    generate: bool = True,
) -> CoachResult:
    bucket = player_bucket(conn, puuid, tier)
    # Fetch older games too, for the "before" side of each trend.
    ids = repo.player_match_ids(conn, puuid, queue_id=queue_id, limit=games * 2)
    scored = _scored_games(conn, puuid, ids, bucket, min_n, minutes=True, extract_missing=True)
    loaded = [(game, p) for game, p, _ in scored]
    scores = [s for _, _, s in scored]
    if not loaded:
        raise ReportError("no analysable games cached for this player -- run sync first")
    tier_label = bucket.replace("_", " ").title()
    evidence = trend_evidence(trends(scores, recent=games), loaded[:games], tier_label)
    add_pool_evidence(evidence, summarize(scores[:games], bucket))
    output, model, cached, dropped, usage, pending = _coach(
        conn, coach, puuid, "recent", None, evidence, "their recent games", refresh,
        generate=generate)
    return CoachResult("recent", puuid, bucket, evidence, output, model, cached, dropped,
                       usage, None, scores[:games], loaded[:games], pending)


def champion_pool(
    conn: psycopg.Connection,
    puuid: str,
    *,
    games: int = 100,
    tier: str | None = None,
    min_n: int = 20,
    queue_id: int | tuple[int, ...] = SUPPORTED_QUEUES,
) -> PoolReport:
    """The player's last ``games`` games, grouped by champion and role. Reads stored features
    only (no extraction on this path) and skips per-minute rows: whole-game stats are enough."""
    bucket = player_bucket(conn, puuid, tier)
    ids = repo.player_match_ids(conn, puuid, queue_id=queue_id, limit=games)
    scores = [s for _, _, s in _scored_games(conn, puuid, ids, bucket, min_n, minutes=False)]
    if not scores:
        raise ReportError("no analysed games cached for this player -- run sync first")
    return summarize(scores, bucket)


def progress_report(
    conn: psycopg.Connection,
    puuid: str,
    *,
    weeks: int = 12,
    tier: str | None = None,
    min_n: int = 20,
    queue_id: tuple[int, ...] = SUPPORTED_QUEUES,
) -> progress_mod.ProgressReport:
    """The last ``weeks`` weeks of games, all scored against the player's current rank, plus
    the rank snapshots taken in that time (one per sync; Riot keeps no LP history)."""
    bucket = player_bucket(conn, puuid, tier)
    rows = conn.execute(
        """
        SELECT m.match_id, m.game_start FROM match_participants p JOIN matches m USING (match_id)
         WHERE p.puuid = %s AND m.queue_id = ANY(%s)
           AND m.game_start >= now() - make_interval(weeks => %s)
         ORDER BY m.game_start
        """,
        (puuid, list(queue_id), weeks),
    ).fetchall()
    starts = dict(rows)
    dated = [(starts[game.match_id], s) for game, _, s in
             _scored_games(conn, puuid, list(starts), bucket, min_n, minutes=False)]
    if not dated:
        raise ReportError(f"no analysed games in the last {weeks} weeks -- run sync first")
    ranks, last = [], None
    for at, tier_, division, lp in conn.execute(
        """
        SELECT captured_at, tier, division, lp FROM rank_snapshots
         WHERE puuid = %s AND queue = 'RANKED_SOLO_5x5'
           AND captured_at >= now() - make_interval(weeks => %s)
         ORDER BY captured_at
        """,
        (puuid, weeks),
    ):
        if (tier_, division, lp) != last:          # one point per change, not per sync
            ranks.append(progress_mod.RankPoint(at, tier_, division, lp))
            last = (tier_, division, lp)
    return progress_mod.summarize(dated, bucket, ranks)


def stream_coaching(conn: psycopg.Connection, result: CoachResult, coach: Coach,
                    task: str) -> Iterator[dict[str, Any]]:
    """Coaching for an already-built report, streamed. A cached answer comes back as one
    "final" event; otherwise grounded points stream as they complete, the finished answer is
    stored, and "final" carries it."""
    from riftwatch.coach.llm import write_stream

    if not result.coach_pending:            # already cached (or no LLM): nothing to stream
        yield {"type": "final", "output": result.output.model_dump(), "cached": True}
        return
    for event in write_stream(coach, result.evidence, task):
        if event["type"] != "done":
            yield event
            continue
        run = event["run"]
        usage = {**run.usage, "attempts": run.attempts, "served_by": run.model,
                 "first_violations": [str(v) for v in run.first_violations]}
        _store(conn, result.puuid, result.scope, result.match_id, result.evidence, coach.label,
               run.output, run.dropped, usage)
        yield {"type": "final", "output": run.output.model_dump(), "cached": False,
               "dropped": len(run.dropped)}
