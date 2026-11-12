"""HTML sections of the profile's AI Coach tab: progress chart, sessions and champion pool
tables (styled by the site's stylesheet). Pages themselves are Jinja2 templates (web/site.py)."""

from __future__ import annotations

import html
from typing import Any
from urllib.parse import quote

REGIONS = ("na", "euw", "eune", "kr", "br", "lan", "las", "oce", "tr", "jp", "me", "sg", "tw", "vn")


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def path_id(game_name: str, tag_line: str) -> str:
    return quote(f"{game_name}-{tag_line}", safe="")


def _progress_chart(weeks) -> str:
    """Weekly "better than N%" as a line, with the 50% (typical) line for reference.
    Inline SVG in the page's own colour tokens, so it follows light and dark mode."""
    w, h, left, right, top, bottom = 1100, 230, 40, 14, 12, 26
    n = len(weeks)

    def x(i: int) -> float:
        return left + (w - left - right) * (i / (n - 1) if n > 1 else 0.5)

    def y(v: float) -> float:
        return top + (h - top - bottom) * (1 - v / 100)

    pts = " ".join(f"{x(i):.1f},{y(wk.score):.1f}" for i, wk in enumerate(weeks))
    ticks = "".join(
        f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="11" '
        f'fill="var(--muted)">{v}%</text>' for v in (25, 75))
    typical = (f'<line x1="{left}" x2="{w - right}" y1="{y(50):.1f}" y2="{y(50):.1f}" '
               f'stroke="var(--group)" stroke-width="1.5" stroke-dasharray="4 4"/>'
               f'<text x="{left - 6}" y="{y(50) + 4:.1f}" text-anchor="end" font-size="11" '
               f'fill="var(--group)">50%</text>')
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(wk.score):.1f}" r="3.5" fill="var(--you)">'
        f"<title>Week of {wk.start:%b %d}: better than {wk.score:.0f}% "
        f"({wk.games} games, {100 * wk.win_rate:.0f}% wins)</title></circle>"
        for i, wk in enumerate(weeks))
    labels = "".join(
        f'<text x="{x(i):.1f}" y="{h - 8}" text-anchor="middle" font-size="11" '
        f'fill="var(--muted)">{wk.start:%b %d}</text>'
        for i, wk in enumerate(weeks) if i % max(1, n // 6) == 0 or i == n - 1)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" style="width:100%;height:auto" '
            f'aria-label="Weekly average standing against players at your rank">'
            f"{ticks}{typical}"
            f'<polyline points="{pts}" fill="none" stroke="var(--you)" stroke-width="2"/>'
            f"{dots}{labels}</svg>")


def progress_html(progress) -> str:
    if progress is None or not progress.weeks:
        return ""
    t = progress.trend
    trend = {"improving": "improving", "declining": "declining", "steady": "steady"}[t.verdict]
    moving = [f"{a} {tr.verdict}" for a, tr in progress.area_trends.items() if tr.verdict != "steady"]
    ranks = (" · Rank when synced: " + " → ".join(f"{r.at:%b %d} {r.label}" for r in progress.ranks)
             if progress.ranks else "")
    return (f"<h2>Progress</h2><p class=\"sub\">Last {len(progress.weeks)} weeks, "
            f"{progress.games} games, each against players at your current rank in its role. "
            f"Trend: <strong>{trend}</strong> ({t.slope:+.1f} ±{t.se:.1f} points per 10 games)"
            + (f"; {_esc(', '.join(moving))}" if moving else "") + f".{_esc(ranks)}</p>"
            f'<div class="card">{_progress_chart(progress.weeks)}'
            '<p class="small" style="margin:6px 0 0">Blue: your weekly average \u201cbetter than\u201d '
            "across all stats. Dashed: a typical player at your rank (50%).</p></div>")


def sessions_html(report) -> str:
    if report is None or not report.games:
        return ""
    notes = []
    for f in report.findings:
        if f.kind == "session_length":
            notes.append(f"Within a session, your games from the 4th on average {abs(f.gap):.0f} "
                         f"points {'lower' if f.worse else 'higher'} than that session's first.")
        else:
            notes.append(f"Within a session, games right after two or more losses average "
                         f"{abs(f.gap):.0f} points {'lower' if f.worse else 'higher'} than "
                         "games right after a win.")
    if not notes:
        notes.append("Compared within the same session, neither late games nor games after "
                     "losses differ clearly from the rest.")

    def rows(groups):
        return "".join(
            f'<tr><td>{_esc(g.label)}</td><td class="num">{g.games}</td>'
            f'<td class="num">{100 * g.win_rate:.0f}%</td>'
            f'<td class="num">{g.score:.0f}% ±{g.se:.0f}</td></tr>' for g in groups)

    head = ('<tr class="area"><td>{}</td><td>games</td><td>win rate</td>'
            "<td>better than</td></tr>")
    return (f"<h2>Sessions</h2><p class=\"sub\">Last {report.games} games in {report.sessions} "
            f"sessions (an hour's break starts a new one). {_esc(' '.join(notes))}</p>"
            '<div class="card scroll"><table class="score">'
            + head.format("game in session") + rows(report.by_position)
            + head.format("previous game") + rows(report.by_streak) + "</table></div>")


def pool_html(pool) -> str:
    if pool is None or not pool.lines:
        return ""
    rows = []
    for l in pool.lines[:12]:
        verdict = {"stronger": "stronger than your other picks",
                   "weaker": "weaker than your other picks"}.get(l.verdict, "")
        spread = f" ±{l.score_se:.0f}" if l.games > 1 else ""
        rows.append(
            f"<tr><td>{_esc(l.champion)}<div class=\"small\">{_esc(l.role.lower())}</div></td>"
            f'<td class="num">{l.games}</td><td class="num">{100 * l.win_rate:.0f}%</td>'
            f'<td class="num">{l.kills}/{l.deaths}/{l.assists}</td>'
            f'<td class="num">{l.score:.0f}%{spread}<div class="small">{_esc(verdict)}</div></td>'
            f"<td>{_esc(l.best_area or '')}<div class=\"small\">weakest: {_esc(l.worst_area or '')}</div></td></tr>")
    return (f"<h2>Champion pool</h2><p class=\"sub\">Last {pool.games} games. \"Better than\" is "
            "the average share of players at your rank in the same role you beat across all "
            "stats, adjusted for champion and matchup; ± is one standard error.</p>"
            '<div class="card scroll"><table class="score"><tr class="area"><td>champion</td>'
            "<td>games</td><td>win rate</td><td>K/D/A</td><td>better than</td><td>best area</td></tr>"
            + "".join(rows) + "</table></div>")
