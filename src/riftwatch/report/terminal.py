"""Plain-text rendering of scores and coaching for the CLI."""

from __future__ import annotations

from riftwatch.analysis.score import GameScore
from riftwatch.coach.pipeline import CoachResult
from riftwatch.features.metrics import AREAS

BAR = 20


def bar(goodness: float) -> str:
    filled = round(goodness / 100 * BAR)
    return "[" + "#" * filled + "." * (BAR - filled) + "]"


def scorecard(score: GameScore) -> list[str]:
    """Whole-game metrics grouped by area, each with a percentile bar."""
    lines = []
    by_area: dict[str, list] = {}
    for s in score.game.values():
        if s.metric.coachable:
            by_area.setdefault(s.metric.area, []).append(s)
    if not by_area:
        return ["  (no baselines for this tier/role yet -- run `riftwatch crawl` and "
                "`riftwatch baselines`)"]
    first = next(iter(score.game.values()))
    lines.append(f"  vs {first.baseline.scope}, patches {first.baseline.patch_window}")
    for area in AREAS:
        if area not in by_area:
            continue
        lines.append(f"  {area}")
        for s in sorted(by_area[area], key=lambda s: s.goodness):
            mark = "!" if s.goodness <= 25 else "+" if s.goodness >= 75 else " "
            lines.append(
                f"   {mark} {s.metric.label[:44]:<44} {s.metric.show(s.value):>8}  "
                f"{bar(s.goodness)} better than {round(s.goodness):>3}%  (median {s.metric.show(s.baseline.p50)}, n={s.baseline.n}"
                f"{', M+ ' + s.metric.show(score.reference[s.metric.name].p50) if s.metric.name in score.reference else ''})"
            )
    return lines


def render(result: CoachResult, *, show_scores: bool = True, show_evidence: bool = False) -> str:
    out: list[str] = []
    if result.scope == "game":
        game, p = result.games[0]
        out.append(f"{p.champion_name} {p.role.lower()} -- {'WIN' if p.win else 'LOSS'} -- "
                   f"{game.match_id} (patch {game.patch})")
        if show_scores:
            out.append("")
            out += scorecard(result.scores[0])
    else:
        out.append(f"Last {len(result.games)} ranked games, compared against "
                   f"{result.tier_bucket.replace('_', ' ').title()} players")

    if result.live:
        m = result.live["summary"].get("metrics", {})
        out.append("")
        out.append("  live recording (second by second)")
        out.append(f"   big health losses before 14:00: {int(m.get('early_big_hp_losses', 0))} "
                   f"(-> recall {int(m.get('early_losses_to_recall', 0))}, "
                   f"-> death {int(m.get('early_losses_to_death', 0))})")
        if "avg_recall_hp" in m:
            out.append(f"   recalls: {int(m['recalls'])}, avg {round(m['avg_recall_hp'] * 100)}% health, "
                       f"{int(m['avg_recall_gold'])} gold unspent")
        out.append(f"   burst deaths: {int(m.get('burst_deaths', 0))}")
    out.append("")
    source = ("offline template coach (set ANTHROPIC_API_KEY for the LLM coach)"
              if result.model == "offline" else
              f"{result.model}{' (cached)' if result.cached else ''}")
    out.append(f"COACH  [{source}]")
    out.append(f"  {result.output.headline}")
    by_id = result.evidence.by_id()
    for i, point in enumerate(result.output.points, 1):
        tag = "work on" if point.kind == "weakness" else "strength"
        out.append("")
        out.append(f"  {i}. {point.title}  ({tag}, {point.area})")
        out.append(f"     {point.explanation}")
        out.append(f"     -> {point.advice}")
        cited = "; ".join(by_id[e].text for e in point.evidence_ids if e in by_id)
        out.append(f"     evidence {', '.join(point.evidence_ids)}: {cited}")
    u = result.usage or {}
    if result.model != "offline" and not result.cached and u:
        cost = f", about ${u['cost_usd']:.4f}" if u.get("cost_usd") is not None else ""
        out.append("")
        out.append(f"  tokens: {u.get('input_tokens', 0)} in + "
                   f"{u.get('cache_read_input_tokens', 0)} read from cache + "
                   f"{u.get('cache_creation_input_tokens', 0)} written to cache, "
                   f"{u.get('output_tokens', 0)} out, {u.get('attempts', 1)} attempt(s){cost}")
    if result.dropped:
        out.append("")
        out.append(f"  ({len(result.dropped)} point(s) removed by the grounding check)")
    if show_evidence:
        out.append("")
        out.append("EVIDENCE")
        for e in result.evidence.items:
            out.append(f"  [{e.id}] {e.text}")
    return "\n".join(out)
