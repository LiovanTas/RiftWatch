"""Server-rendered pages. Each page arrives complete in one response (no client-side data
fetching before first paint); JavaScript only runs actions -- Update, Get coaching."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from riftwatch.coach.pipeline import CoachResult
from riftwatch.report.html import coach_html, nav, page

REGIONS = ("na", "euw", "eune", "kr", "br", "lan", "las", "oce", "tr", "jp", "me", "sg", "tw", "vn")

SYNC_JS = r"""
const sync = document.getElementById('sync');
if (sync) sync.addEventListener('click', async () => {
  const status = document.getElementById('sync-status');
  sync.disabled = true; status.textContent = 'Starting...';
  try {
    const r = await fetch(sync.dataset.url, {method: 'POST'});
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || body.error || r.statusText);
    let job = body.job;
    while (job.status === 'queued' || job.status === 'running') {
      status.textContent = job.progress.length ? job.progress[job.progress.length - 1] : 'Waiting for Riot...';
      await new Promise(res => setTimeout(res, 800));
      job = await (await fetch('/api/jobs/' + job.id)).json();
    }
    if (job.status === 'failed') throw new Error(job.error);
    location.reload();
  } catch (e) { sync.disabled = false; status.textContent = 'Update failed: ' + e.message; }
});
"""


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def path_id(game_name: str, tag_line: str) -> str:
    return quote(f"{game_name}-{tag_line}", safe="")


def _ago(iso: str) -> str:
    seconds = (datetime.now(UTC) - datetime.fromisoformat(iso)).total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


def search_page(error: str | None = None) -> str:
    options = "".join(f'<option value="{r}">{r.upper()}</option>' for r in REGIONS)
    body = (
        "<h1>RiftWatch</h1>"
        '<p class="sub">Minute-by-minute coaching against players at your rank, in your role.</p>'
        '<form class="card" action="/search" method="get" style="margin-top:20px;display:flex;'
        'gap:10px;flex-wrap:wrap;align-items:center">'
        f'<select name="region" aria-label="Region" style="font:inherit;padding:8px">{options}</select>'
        '<input name="riot_id" placeholder="Name#TAG" aria-label="Riot ID" required '
        'style="font:inherit;padding:8px;flex:1;min-width:180px">'
        '<button class="action" type="submit">Look up</button></form>'
        + (f'<p class="status">{_esc(error)}</p>' if error else "")
    )
    return page("RiftWatch", "League of Legends coaching against your rank", body)


def _progress_chart(weeks) -> str:
    """Weekly "better than N%" as a line, with the 50% (typical) line for reference.
    Inline SVG in the page's own colour tokens, so it follows light and dark mode."""
    w, h, left, right, top, bottom = 640, 180, 36, 12, 12, 26
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


def _progress_html(progress) -> str:
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


def _sessions_html(report) -> str:
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


def _pool_html(pool) -> str:
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


def player_page(
    region: str,
    riot_id: str,
    player: dict[str, Any] | None,
    recent: CoachResult | None,
    recent_coach_url: str | None,
    pool=None,
    progress=None,
    sessions=None,
) -> str:
    sync_url = f"/api/players/{_esc(region)}/{_esc(riot_id)}/sync"
    update = (f'<p><button class="action" id="sync" data-url="{sync_url}">Update</button>'
              f' <a class="plain" href="/scout/{_esc(region)}/{_esc(riot_id)}">Live game</a>'
              '<span class="status" id="sync-status"></span></p>')
    links = nav([("RiftWatch", "/")])
    if player is None:
        shown = riot_id.rsplit("-", 1)
        body = (links + f"<h1>{_esc('#'.join(shown))}</h1>"
                f'<p class="sub">Not on RiftWatch yet. Update downloads the newest ranked games '
                f"from Riot ({_esc(region.upper())}).</p>" + update)
        return page(f"{'#'.join(shown)} - RiftWatch", "Player not synced yet", body,
                    extra_js=SYNC_JS)

    rank = player["rank"]
    rank_text = (f"{rank['tier'].title()} {rank['division'] or ''} {rank['lp']} LP · "
                 f"{rank['wins']}W {rank['losses']}L" if rank else "Unranked in solo/duo")
    rows = []
    for m in player["recent"]:
        link = f"/players/{_esc(region)}/{_esc(riot_id)}/matches/{_esc(m['match_id'])}"
        rows.append(
            f'<tr><td><a class="plain" href="{link}">{_esc(m["champion"] or "?")}</a>'
            f'<div class="small">{_esc((m["role"] or "").lower())}</div></td>'
            f'<td>{"Win" if m["win"] else "Loss"}<div class="small">{_esc(m["mode"])}</div></td>'
            f'<td class="num">{m["kills"]}/{m["deaths"]}/{m["assists"]}</td>'
            f'<td class="num">{"" if m["cs_per_min"] is None else m["cs_per_min"]}</td>'
            f'<td class="num">{m["duration_s"] // 60}:{m["duration_s"] % 60:02d}</td>'
            f'<td class="num">{_ago(m["game_start"])}</td></tr>'
        )
    games = (f'<div class="card scroll"><table class="score"><tr class="area"><td>champion</td>'
             f"<td>result</td><td>K/D/A</td><td>CS/min</td><td>length</td><td>played</td></tr>"
             f'{"".join(rows)}</table></div>' if rows else '<p class="sub">No games yet.</p>')
    coaching = ""
    if recent is not None:
        coaching = (f"<h2>Recent games coaching</h2><div class=\"card coach\">"
                    f"{coach_html(recent, recent_coach_url)}</div>")
    body = (
        links + f"<h1>{_esc(player['riot_id'])}</h1>"
        f'<p class="sub">{_esc(rank_text)} · {_esc(player["platform"].upper())} · '
        f"{player['games_cached']} games analysed</p>" + update
        + coaching + _progress_html(progress) + _sessions_html(sessions) + _pool_html(pool)
        + "<h2>Recent games</h2>" + games
    )
    return page(f"{player['riot_id']} - RiftWatch", f"RiftWatch coaching for {player['riot_id']}",
                body, extra_js=SYNC_JS)



SCOUT_JS = r"""
const scoutBtn = document.getElementById('scout');
if (scoutBtn) scoutBtn.addEventListener('click', async () => {
  const status = document.getElementById('scout-status');
  scoutBtn.disabled = true; status.textContent = 'Finding the game...';
  try {
    const r = await fetch(scoutBtn.dataset.url, {method: 'POST'});
    const body = await r.json();
    if (!r.ok) throw new Error(body.detail || body.error || r.statusText);
    let job = body.job;
    while (job.status === 'queued' || job.status === 'running') {
      status.textContent = job.progress.length ? job.progress[job.progress.length - 1] : 'Waiting for Riot...';
      await new Promise(res => setTimeout(res, 800));
      job = await (await fetch('/api/jobs/' + job.id)).json();
    }
    if (job.status === 'failed') throw new Error(job.error);
    location.search = '?job=' + job.id;
  } catch (e) { scoutBtn.disabled = false; status.textContent = e.message; }
});
"""


def _scout_row(p: dict[str, Any], me: str, region: str) -> str:
    r = p["rank"]
    rank = (f"{r['tier'].title()} {r['division'] or ''}".strip() + f" · {r['lp']} LP"
            if r else "Unranked")
    season = (f"{round(100 * r['wins'] / max(r['wins'] + r['losses'], 1))}% of "
              f"{r['wins'] + r['losses']}" if r else "")
    recent = (f"{p['wins']}W {p['games'] - p['wins']}L · {p['kills']}/{p['deaths']}/{p['assists']}"
              if p["games"] else "none")
    role = (f"{p['main_role'].lower()} {round(100 * p['main_role_share'])}%"
            if p["main_role"] else "")
    champ = (f"{p['champion_games']} · {round(100 * p['champion_wins'] / p['champion_games'])}%"
             if p["champion_games"] else "none")
    mastery = "" if p["mastery_points"] is None else f"{p['mastery_points']:,}"
    name = _esc(p["riot_id"])
    if p["puuid"] == me:
        name = f"<strong>{name}</strong>"
    elif p["puuid"] and "#" in p["riot_id"]:
        game_name, tag = p["riot_id"].rsplit("#", 1)
        name = f'<a class="plain" href="/players/{_esc(region)}/{path_id(game_name, tag)}">{name}</a>'
    return (f'<tr><td>{name}<div class="small">{_esc(", ".join(p["flags"]))}</div></td>'
            f'<td>{_esc(p["champion"])}</td>'
            f'<td>{_esc(rank)}<div class="small">{_esc(season)}</div></td>'
            f'<td class="num">{_esc(recent)}<div class="small">{_esc(role)}</div></td>'
            f'<td class="num">{_esc(champ)}</td><td class="num">{_esc(mastery)}</td></tr>')


def scout_page(region: str, riot_id: str, report: dict[str, Any] | None,
               error: str | None = None) -> str:
    shown = "#".join(riot_id.rsplit("-", 1))
    url = f"/api/scout/{_esc(region)}/{_esc(riot_id)}"
    button = (f'<p><button class="action" id="scout" data-url="{url}">'
              f'{"Refresh" if report else "Scout live game"}</button>'
              f'<span class="status" id="scout-status">{_esc(error or "")}</span></p>')
    links = nav([("RiftWatch", "/"), (shown, f"/players/{_esc(region)}/{_esc(riot_id)}")])
    if report is None:
        body = (links + "<h1>Live game</h1><p class=\"sub\">Rank, champion experience and recent "
                f"form of everyone in {_esc(shown)}'s current game.</p>" + button)
        return page(f"Live game - {shown}", "RiftWatch live-game scouting", body,
                    extra_js=SCOUT_JS)

    minutes, seconds = divmod(max(report["game_length_s"], 0), 60)
    head = ('<tr class="area"><td>player</td><td>champion</td><td>solo/duo</td>'
            "<td>last games</td><td>ranked games on champ</td><td>mastery</td></tr>")
    teams = []
    for team_id, name in ((100, "Blue team"), (200, "Red team")):
        rows = "".join(_scout_row(p, report["me"], region)
                       for p in report["players"] if p["team_id"] == team_id)
        bans = ", ".join(b["champion"] for b in report["bans"] if b["team_id"] == team_id)
        teams.append(f"<h2>{name}</h2>"
                     + (f'<p class="sub">Bans: {_esc(bans)}</p>' if bans else "")
                     + f'<div class="card scroll"><table class="score">{head}{rows}</table></div>')
    body = (links + "<h1>Live game</h1>"
            f'<p class="sub">{_esc(report["queue"])} · {_esc(report["platform"].upper())} · '
            f"{minutes}:{seconds:02d} in when scouted</p>" + button + "".join(teams))
    return page(f"Live game - {shown}", "RiftWatch live-game scouting", body, extra_js=SCOUT_JS)
