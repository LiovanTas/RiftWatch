"""Pipeline results -> plain JSON for the web API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from riftwatch.analysis.score import GameScore
from riftwatch.coach.evidence import clock
from riftwatch.coach.pipeline import CoachResult
from riftwatch.features.extract import GameFeatures, ParticipantFeatures
from riftwatch.report.html import curve_series


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def rank_json(rank: dict[str, Any] | None) -> dict[str, Any] | None:
    if rank is None:
        return None
    return {k: _iso(v) for k, v in rank.items()}


def scorecard(score: GameScore) -> list[dict[str, Any]]:
    rows = []
    for s in sorted(score.game.values(), key=lambda s: (s.metric.area, s.goodness)):
        if not s.metric.coachable:
            continue
        rows.append({
            "metric": s.metric.name, "label": s.metric.label, "area": s.metric.area,
            "value": s.value, "display": s.metric.show(s.value),
            "better_than": round(s.goodness, 1),
            "median": s.baseline.p50, "median_display": s.metric.show(s.baseline.p50),
            "n": s.baseline.n, "scope": s.baseline.scope,
            "flag": "weakness" if s.goodness <= 25 else "strength" if s.goodness >= 75 else None,
        })
    return rows


def coach_json(result: CoachResult) -> dict[str, Any]:
    by_id = result.evidence.by_id()
    return {
        "model": result.model,
        "cached": result.cached,
        "pending": result.coach_pending,
        "headline": result.output.headline,
        "points": [
            {**p.model_dump(),
             "evidence": [{"id": e, "text": by_id[e].text} for e in p.evidence_ids if e in by_id]}
            for p in result.output.points
        ],
        "dropped": len(result.dropped),
    }


def game_summary(game: GameFeatures, p: ParticipantFeatures, game_start: Any = None) -> dict[str, Any]:
    m = p.metrics
    return {
        "match_id": game.match_id, "patch": game.patch, "queue_id": game.queue_id,
        "duration_s": game.duration_s, "duration": clock(game.duration_min),
        "game_start": _iso(game_start), "champion": p.champion_name, "champion_id": p.champion_id,
        "role": p.role, "win": p.win,
        "kills": int(m.get("kills", 0)), "deaths": int(m.get("deaths", 0)),
        "assists": int(m.get("assists", 0)), "cs_per_min": round(m.get("cs_per_min", 0), 2),
        "kill_participation": round(m.get("kill_participation", 0), 3),
    }


def game_review(result: CoachResult) -> dict[str, Any]:
    game, p = result.games[0]
    score = result.scores[0]
    first = next(iter(score.game.values()), None)
    return {
        "game": game_summary(game, p),
        "comparison": {
            "tier": result.tier_bucket,
            "group": first.baseline.scope if first else None,
            "patches": first.baseline.patch_window if first else None,
        },
        "scorecard": scorecard(score),
        "curves": curve_series(score),
        "deaths": [{"minute": d.minute, "clock": clock(d.minute), "where": d.where,
                    "zone": d.zone, "assisters": d.assisters, "gold_diff": d.gold_diff}
                   for d in p.deaths],
        "evidence": result.evidence.to_json(),
        "coach": coach_json(result),
        "live": result.live,
        "high_elo": _review_json(result.review),
    }


def _review_json(review) -> dict[str, Any] | None:
    if review is None:
        return None

    def option(o) -> dict[str, Any]:
        return {"decision": o.decision, "share": round(o.share, 3), "objective": round(o.objective, 3),
                "death": round(o.death, 3), "gold": round(o.gold)}

    return {
        "role": review.role, "minutes": review.minutes, "agreement": round(review.agreement, 3),
        "moments": [{"minute": m.minute, "did": option(m.did), "alternative": option(m.best)}
                    for m in review.moments],
        "matched": [{"minute": m.minute, "did": option(m.did)} for m in review.good],
    }


def recent_review(result: CoachResult) -> dict[str, Any]:
    return {
        "tier": result.tier_bucket,
        "games": [game_summary(g, p) for g, p in result.games],
        "evidence": result.evidence.to_json(),
        "coach": coach_json(result),
    }
