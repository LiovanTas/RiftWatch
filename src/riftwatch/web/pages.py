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


def player_page(
    region: str,
    riot_id: str,
    player: dict[str, Any] | None,
    recent: CoachResult | None,
    recent_coach_url: str | None,
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
        + coaching + "<h2>Recent games</h2>" + games
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
