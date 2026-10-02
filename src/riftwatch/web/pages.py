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
            f'<td>{"Win" if m["win"] else "Loss"}</td>'
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
