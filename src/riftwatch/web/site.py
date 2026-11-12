"""The website's pages: home, player profiles with match history, champion tier list and
champion pages, leaderboard, live-game scouting.

Pages are rendered on the server from Jinja2 templates (``templates/``) with one stylesheet
and one script (``static/``); images come from Riot's Data Dragon CDN. Expanding a game or
loading more games fetches small HTML fragments, so a profile opens fast and stays light.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, PackageLoader, select_autoescape

from riftwatch.coach.pipeline import ReportError, champion_pool, progress_report, recent_report, sessions_report
from riftwatch.db import repo
from riftwatch.riot.routing import UnknownRegion, parse_riot_id, platform_for
from riftwatch.stats import champions as champ_stats
from riftwatch.stats import games as game_stats
from riftwatch.stats import matchups, patches
from riftwatch.web import gamedata, insights, mapview, mastery, moments, pages, recap, views

log = logging.getLogger(__name__)

REGIONS = pages.REGIONS
PLATFORM_REGION = {"na1": "na", "euw1": "euw", "eun1": "eune", "kr": "kr", "br1": "br",
                   "la1": "lan", "la2": "las", "oc1": "oce", "tr1": "tr", "jp1": "jp",
                   "me1": "me", "sg2": "sg", "tw2": "tw", "vn2": "vn", "ru": "ru"}
QUEUE_TABS = (("all", "All"), ("solo", "Ranked Solo"), ("flex", "Ranked Flex"), ("normal", "Normal"))
TIER_BLURBS = (("S+", "best picks"), ("S", "strong"), ("A", "good"), ("B", "fine"),
               ("C", "weak"), ("D", "avoid"))
CHAMP_QUEUES = (("ranked", "Ranked"), ("solo", "Solo/Duo"), ("flex", "Flex"), ("all", "All"))
STATIC = Path(__file__).parent / "static"
LEADERBOARD_S = 1800


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _asset_version() -> str:
    """Changes whenever a static file does, so browsers fetch the new stylesheet and script."""
    stamp = "".join(f"{p.name}{p.stat().st_mtime_ns}" for p in sorted(STATIC.glob("*")))
    return hashlib.sha1(stamp.encode()).hexdigest()[:10]


def register(app: FastAPI, services, *, coach, advisor, brain,
             gd: gamedata.GameData | None = None, guard=None) -> None:
    """Mount the website on ``app`` (the JSON API stays in app.py)."""
    import mimetypes

    mimetypes.add_type("application/manifest+json", ".webmanifest")   # unknown to Windows
    pool = services.pool
    env = Environment(loader=PackageLoader("riftwatch.web", "templates"),
                      autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    env.filters["ordinal"] = _ordinal
    # Riot's game version (16.22) as players know the patch (26.22).
    env.filters["patchname"] = lambda v: (f"{int(v.split('.')[0]) + 10}.{v.split('.')[1]}"
                                          if v and v.split(".")[0].isdigit() and int(v.split(".")[0]) < 20 else v)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    state: dict[str, Any] = {"gd": gd}
    boards: dict[str, tuple[float, dict]] = {}

    def game_data() -> gamedata.GameData:
        if state["gd"] is None:
            state["gd"] = gamedata.shared()
        return state["gd"]

    def warm() -> None:
        try:
            game_data()
        except Exception:
            pass

    threading.Thread(target=warm, name="game-data", daemon=True).start()

    def render(template: str, status: int = 200, **ctx) -> HTMLResponse:
        region = ctx.pop("region", None) or "na"
        ctx.setdefault("positions", list(gamedata.POSITIONS.items()))
        html = env.get_template(template).render(
            regions=REGIONS, region=region, gd=game_data(), asset_version=_asset_version(), **ctx)
        return HTMLResponse(html, status_code=status)

    def error_page(title: str, message: str, status: int = 404, region: str = "na") -> HTMLResponse:
        return render("error.html", status=status, title=title, message=message, region=region)

    def region_or_none(region: str) -> str | None:
        try:
            return platform_for(region)
        except UnknownRegion:
            return None

    def find_account(conn, region: str, riot_id: str):
        platform = platform_for(region)
        name, sep, tag = riot_id.rpartition("-")
        if not sep or not name.strip() or not tag.strip():
            raise HTTPException(400, "expected Name-TAG in the URL")
        account = repo.find_account(conn, name.strip(), tag.strip())
        if account is not None and account["platform"] != platform:
            account = None
        return name.strip(), tag.strip(), account

    # -- home, search, about ----------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        gd = game_data()
        with pool.connection() as conn:
            games = conn.execute("SELECT count(*) FROM matches").fetchone()[0]
            ranked = conn.execute("SELECT count(*) FROM matches WHERE queue_id = 420").fetchone()[0]
            players = conn.execute(
                "SELECT count(DISTINCT puuid) FROM participant_builds").fetchone()[0]
            top = []
            if ranked:
                rows = champ_stats.tier_list(conn, changes=patch_changes(),
                                             systems=item_changes())["rows"]
                for r in [r for r in rows if r["tier"] in ("S+", "S")][:12]:
                    c = gd.champion(r["champion_id"])
                    top.append({"name": c.name, "slug": c.slug, "icon": gd.champion_icon(c.id),
                                "role": gamedata.POSITIONS.get(r["position"], ""),
                                "position": r["position"], "tier": r["tier"],
                                "win_rate": r["win_rate"]})
        stats = {"games": games, "ranked": ranked, "players": players,
                 "champions": len(gd.champions) or 170, "patch": ".".join(gd.version.split(".")[:2])}
        return render("home.html", nav="home", stats=stats, top_champions=top,
                      watch=patch_watch()[:2])

    def search_error(message: str) -> HTMLResponse:
        return render("home.html", status=400, nav="home", error=message,
                      stats={"games": 0, "ranked": 0, "players": 0, "champions": 0,
                             "patch": ""}, top_champions=[])

    @app.get("/search")
    def search(riot_id: str, region: str = "auto"):
        """``region=auto`` (the default) finds the player's region from their Riot ID:
        RiftWatch's own records first, then Riot."""
        try:
            rid = parse_riot_id(riot_id)
            if region.lower() != "auto":
                platform_for(region)
        except ValueError as exc:
            return search_error(str(exc))
        if region.lower() != "auto":
            return RedirectResponse(
                f"/players/{region.lower()}/{views.path_id(rid.game_name, rid.tag_line)}", 303)
        found = locate(rid.game_name, rid.tag_line)
        if found is None:
            return search_error(f"Couldn't find {rid.game_name}#{rid.tag_line} in any region. "
                                "Check the name and #TAG, or pick the region.")
        if isinstance(found, str):
            return search_error(found)
        name, tag, platform = found
        return RedirectResponse(
            f"/players/{PLATFORM_REGION.get(platform, platform)}/{views.path_id(name, tag)}", 303)

    def locate(name: str, tag: str):
        """(name, tag, platform) for a Riot ID in any region, None if it doesn't exist, or a
        message when Riot can't be asked."""
        with pool.connection() as conn:
            known = repo.find_account(conn, name, tag)
            if known is not None:
                return known["game_name"], known["tag_line"], known["platform"]
            seen = conn.execute(
                """SELECT riot_name, riot_tag, lower(split_part(match_id, '_', 1))
                     FROM participant_builds
                    WHERE lower(riot_name) = lower(%s) AND lower(riot_tag) = lower(%s)
                    ORDER BY game_start DESC LIMIT 1""", (name, tag)).fetchone()
            if seen is not None:
                return seen
        if services.api is None:
            return "Pick the player's region: RiftWatch can't look up new players right now."
        from riftwatch.riot.client import RiotApiError
        from riftwatch.riot.routing import RiotId

        try:
            account = services.api.account_by_riot_id("na1", RiotId(name, tag))
            if account is None:
                return None
            platform = services.api.active_platform(account["puuid"]) or "na1"
        except RiotApiError:
            return ("RiftWatch can't reach Riot right now. Pick the player's region and try "
                    "again later.")
        return account.get("gameName", name), account.get("tagLine", tag), platform

    app.state.locate = locate

    @app.get("/about", response_class=HTMLResponse)
    def about() -> HTMLResponse:
        with pool.connection() as conn:
            ranked = conn.execute("SELECT count(*) FROM matches WHERE queue_id = 420").fetchone()[0]
        return render("about.html", nav="about", ranked=ranked)

    @app.get("/api/autocomplete")
    def autocomplete(q: str = Query(..., min_length=1, max_length=40), region: str = "") -> dict[str, Any]:
        """Players RiftWatch has seen whose Riot ID starts with ``q`` (``Name`` or ``Name#TA``)."""
        gd = game_data()
        name, _, tag = q.strip().partition("#")
        if not name:
            return {"players": [], "champions": []}
        like = name.lower().replace("\\", "").replace("%", "").replace("_", "\\_") + "%"
        tag_like = tag.lower().replace("%", "").replace("_", "\\_") + "%"
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT riot_name, riot_tag, platform, icon, games FROM (
                  SELECT DISTINCT ON (puuid) puuid, riot_name, riot_tag,
                         lower(split_part(match_id, '_', 1)) platform, profile_icon icon,
                         count(*) OVER (PARTITION BY puuid) games
                    FROM participant_builds
                   WHERE lower(riot_name) LIKE %s AND lower(coalesce(riot_tag, '')) LIKE %s
                   ORDER BY puuid, game_start DESC) seen
                 ORDER BY (lower(riot_name) = %s) DESC, games DESC LIMIT 8
                """, (like, tag_like, name.lower())).fetchall()
        wanted = region.lower()
        players = [{"name": n, "tag": t, "region": PLATFORM_REGION.get(p, p),
                    "icon": gd.profile_icon(i)} for n, t, p, i, _ in rows if n and t]
        players.sort(key=lambda p: p["region"] != wanted)
        # Champions too, when the text has no tag: "ahr" finds Ahri's page.
        champions = []
        if not tag:
            q_low = name.lower().replace(" ", "").replace("'", "")
            for c in sorted(gd.champions.values(), key=lambda c: c.name):
                key = c.name.lower().replace(" ", "").replace("'", "").replace(".", "")
                if key.startswith(q_low) or (len(q_low) >= 3 and q_low in key):
                    champions.append({"name": c.name, "slug": c.slug, "icon": gd.champion_icon(c.id),
                                      "title": c.title})
            champions.sort(key=lambda c: not c["name"].lower().replace(" ", "").startswith(q_low))
        return {"players": players, "champions": champions[:4]}

    # -- player profile ---------------------------------------------------------------------

    @app.get("/players/{region}/{riot_id}/champions/{slug}", response_class=HTMLResponse)
    def player_champion_page(region: str, riot_id: str, slug: str) -> HTMLResponse:
        """One player on one champion, against the champion at their rank and in Master+."""
        if region_or_none(region) is None:
            return error_page("Unknown region", f"RiftWatch doesn't know the region {region!r}.")
        gd_ = game_data()
        champ = gd_.champion_by_slug(slug)
        if champ is None:
            return error_page("Champion not found", f"There's no champion called {slug!r}.")
        with pool.connection() as conn:
            name, tag, account = find_account(conn, region, riot_id)
            if account is None:
                return RedirectResponse(f"/players/{region}/{riot_id}", 303)
            data = mastery.player_champion(conn, gd_, account["puuid"], champ.id, region.lower(), riot_id,
                                           views.player_tier(conn, account["puuid"]))
            head = views.header(conn, gd_, account)
        return render("player_champion.html", nav="player", region=region.lower(), name=account["game_name"],
                      tag=account["tag_line"], champ=champ, c=data, head=head,
                      champ_icon=gd_.champion_icon(champ.id), splash=gd_.champion_splash(champ.id),
                      profile_url=f"/players/{region.lower()}/{riot_id}")

    @app.get("/players/{region}/{riot_id}/recap", response_class=HTMLResponse)
    def recap_page(region: str, riot_id: str, year: int | None = None) -> HTMLResponse:
        """A player's season in numbers: champions, months, streaks, highlights."""
        if region_or_none(region) is None:
            return error_page("Unknown region", f"RiftWatch doesn't know the region {region!r}.")
        gd_ = game_data()
        with pool.connection() as conn:
            name, tag, account = find_account(conn, region, riot_id)
            if account is None:
                return RedirectResponse(f"/players/{region}/{riot_id}", 303)
            data = recap.season_recap(conn, gd_, account["puuid"], region.lower(), year)
            years = [int(y) for (y,) in conn.execute(
                """SELECT DISTINCT extract(year FROM game_start)::int FROM participant_builds
                    WHERE puuid = %s ORDER BY 1 DESC""", (account["puuid"],))]
            head = views.header(conn, gd_, account)
        return render("recap.html", nav="recap", region=region.lower(), name=account["game_name"],
                      tag=account["tag_line"], r=data, head=head, years=years,
                      profile_url=f"/players/{region.lower()}/{riot_id}",
                      played_json=json.dumps(data["played"]) if data else "[]")

    @app.get("/patches", response_class=HTMLResponse)
    def patches_page() -> HTMLResponse:
        """Recent patch notes, and how each change moved champions on the tier board."""
        gd_ = game_data()
        with pool.connection() as conn:
            data = champ_stats.tier_list(conn, None, "all", patch_changes(), item_changes())
        by_champ: dict[int, list[dict]] = {}
        by_item: dict[str, list[tuple[int, dict]]] = {}
        for r in data["rows"]:
            by_champ.setdefault(r["champion_id"], []).append(r)
            for c in r.get("item_changes", []):
                by_item.setdefault(f"{c['patch']}|{c['name']}", []).append((c["usage"], r))
        name_to = {c.name.lower(): c for c in gd_.champions.values()}
        out = []
        for w in patch_watch():
            champs = []
            for c in w["changes"]:
                champ = name_to.get(c["champion"].lower())
                roles = [{"role": gamedata.POSITIONS.get(r["position"], ""), "tier": r["tier"],
                          "tier_raw": r["tier_raw"], "win_rate": r["win_rate"], "games": r["games"],
                          "position": r["position"]}
                         for r in by_champ.get(champ.id if champ else -1, [])]
                champs.append({**c, "roles": roles})
            systems = []
            for c in w.get("systems", []):
                hit = sorted(by_item.get(f"{w['patch']}|{c['name']}", []), key=lambda x: -x[0])[:6]
                systems.append({**c, "users": [{"name": gd_.champion(r["champion_id"]).name,
                                                "slug": gd_.champion(r["champion_id"]).slug,
                                                "icon": gd_.champion_icon(r["champion_id"]),
                                                "role": gamedata.POSITIONS.get(r["position"], ""),
                                                "position": r["position"], "usage": usage,
                                                "tier": r["tier"], "tier_raw": r["tier_raw"]} for usage, r in hit]})
            out.append({**w, "champs": champs, "systems": systems})
        return render("patches.html", nav="patches", patches=out, shift=patches.SHIFT,
                      preview_weight=patches.PREVIEW_WEIGHT, item_shift=patches.ITEM_SHIFT,
                      updated=patches.notes().get("updated"))

    @app.get("/stats", response_class=HTMLResponse)
    def stats_page(tab: str = "objectives", rank: str = "all") -> HTMLResponse:
        """Game-wide numbers: what objectives and gold leads are worth, and item win rates."""
        gd_ = game_data()
        rank = _rank(rank)
        tab = "items" if tab == "items" else "objectives"
        ctx: dict[str, Any] = {"tab": tab, "rank": rank,
                               "ranks": [(k, v[0]) for k, v in champ_stats.RANKS.items()]}
        with pool.connection() as conn:
            if tab == "objectives":
                ctx["o"] = game_stats.objectives(conn, rank)
            else:
                data = game_stats.items(conn, rank)
                changes = {}
                for c in item_changes():
                    for i in c["ids"]:
                        changes.setdefault(i, []).append(c)
                rows = []
                for r in data["rows"]:
                    it = gd_.item(r["item_id"])
                    if it is None or not (it.is_legendary or it.is_boots) or it.is_starter:
                        continue
                    rows.append({**r, "name": it.name, "icon": gd_.item_icon(r["item_id"]),
                                 "text": it.text, "gold": it.gold, "kind": "Boots" if it.is_boots else "Legendary",
                                 "changes": changes.get(r["item_id"], []),
                                 "champions": [{**c, "name": gd_.champion(c["champion_id"]).name,
                                                "slug": gd_.champion(c["champion_id"]).slug,
                                                "icon": gd_.champion_icon(c["champion_id"])}
                                               for c in r["champions"]]})
                ctx.update(items=rows, players=data["players"])
        return render("stats.html", nav="stats", **ctx)

    @app.get("/compare", response_class=HTMLResponse)
    def compare_page(a: str = "", b: str = "") -> HTMLResponse:
        """Two players side by side, and their games together or against each other."""
        ids = [views.parse_lobby(x)[:1] for x in (a, b)]
        data = None
        if all(ids):
            with pool.connection() as conn:
                data = recap.compare(conn, game_data(), [i[0] for i in ids], PLATFORM_REGION)
        return render("compare.html", nav="compare", a=a, b=b, data=data,
                      unparsed=bool(a.strip() or b.strip()) and not all(ids))

    @app.get("/counters", response_class=HTMLResponse)
    def counters_page(vs: str = "", role: str = "", me: str = "") -> HTMLResponse:
        """Draft helper: who does best against a champion in its lane, and how you've done."""
        gd_ = game_data()
        champs = sorted(gd_.champions.values(), key=lambda c: c.name)
        enemy = gd_.champion_by_slug(vs) if vs else None
        ctx: dict[str, Any] = {"vs": vs, "role": role.upper(), "me": me, "enemy": None,
                               "all": [{"name": c.name, "slug": c.slug, "icon": gd_.champion_icon(c.id)}
                                       for c in champs]}
        if enemy is not None:
            with pool.connection() as conn:
                position = role.upper() if role.upper() in gamedata.POSITIONS else champ_stats.main_position(conn, enemy.id)
                rows = champ_stats.counter_picks(conn, enemy.id, position) if position else []
                mine: dict[int, dict] = {}
                found = recap.locate(conn, *views.parse_lobby(me)[0]) if views.parse_lobby(me) else None
                if found and position:
                    for cid, n, w, vs_n, vs_w in conn.execute(
                            """SELECT champion_id, count(*), sum(win::int),
                                      count(*) FILTER (WHERE opponent_champion_id = %s),
                                      coalesce(sum(win::int) FILTER (WHERE opponent_champion_id = %s), 0)
                                 FROM participant_builds WHERE puuid = %s AND position = %s AND NOT remake
                                  AND queue_id IN (420, 440, 400) GROUP BY 1""",
                            (enemy.id, enemy.id, found[0], position)):
                        mine[cid] = {"games": n, "wins": w, "vs_games": vs_n, "vs_wins": vs_w}
            ctx.update(enemy={"name": enemy.name, "slug": enemy.slug, "icon": gd_.champion_icon(enemy.id),
                              "title": enemy.title},
                       role=position or "", role_name=gamedata.POSITIONS.get(position or "", ""),
                       me_found=found is not None, me_name=f"{found[2]}#{found[3]}" if found else "",
                       rows=[{**r, "name": gd_.champion(r["champion_id"]).name,
                              "slug": gd_.champion(r["champion_id"]).slug,
                              "icon": gd_.champion_icon(r["champion_id"]),
                              "mine": mine.get(r["champion_id"])} for r in rows],
                       pool=sorted(({"name": gd_.champion(c).name, "slug": gd_.champion(c).slug,
                                     "icon": gd_.champion_icon(c), **v} for c, v in mine.items()),
                                   key=lambda x: -x["games"])[:8])
        return render("counters.html", nav="counters", **ctx)

    @app.get("/players/{region}/{riot_id}", response_class=HTMLResponse)
    def player_page(request: Request, region: str, riot_id: str, tab: str = "overview",
                    queue: str = "all", cq: str = "ranked", job: str | None = None, champ: str = ""):
        if region_or_none(region) is None:
            return error_page("Unknown region", f"RiftWatch doesn't know the region {region!r}.")
        region = region.lower()
        gd = game_data()
        tab = tab if tab in ("overview", "champions", "coach", "live") else "overview"
        queue = queue if queue in views.QUEUE_FILTERS else "all"
        cq = cq if cq in dict(CHAMP_QUEUES) else "ranked"
        try:
            with pool.connection() as conn:
                name, tag, account = find_account(conn, region, riot_id)
                if account is None:
                    elsewhere = repo.find_account(conn, name, tag)
                    if elsewhere is not None:            # stored under another region
                        right = PLATFORM_REGION.get(elsewhere["platform"], elsewhere["platform"])
                        query = f"?{request.url.query}" if request.url.query else ""
                        target = views.path_id(elsewhere["game_name"], elsewhere["tag_line"])
                        return RedirectResponse(f"/players/{right}/{target}{query}", 303)
                base = {"region": region, "name": name, "tag": tag, "shown": f"{name}#{tag}",
                        "sync_url": f"/api/players/{region}/{riot_id}/sync",
                        "scout_url": f"/api/scout/{region}/{riot_id}", "tab": tab,
                        "queue": queue, "queues": QUEUE_TABS, "cq": cq,
                        "champ_queues": CHAMP_QUEUES, "year": datetime.now(UTC).year}
                if account is None:
                    base["profile_json"] = json.dumps({"region": region, "name": name, "tag": tag,
                                                       "synced": False})
                    live = _live_tab(job, region) if tab == "live" else {}
                    return render("profile.html", synced=False, head=None, solo=None, **base, **live)
                puuid = account["puuid"]
                head = views.header(conn, gd, account)
                name, tag = account["game_name"], account["tag_line"]
                base.update(name=name, tag=tag, shown=f"{name}#{tag}",
                            recap_url=f"/players/{region}/{riot_id}/recap",
                            profile_base=f"/players/{region}/{riot_id}",
                            compare_url=f"/compare?a={quote(name + '#' + tag)}")
                ctx: dict[str, Any] = {
                    "synced": True, "head": head,
                    "solo": views.rank_card(repo.latest_rank(conn, puuid), gd),
                    "flex": views.rank_card(repo.latest_rank(conn, puuid, "RANKED_FLEX_SR"), gd),
                    "games_cached": conn.execute(
                        "SELECT count(*) FROM participant_builds WHERE puuid = %s",
                        (puuid,)).fetchone()[0],
                    "profile_json": json.dumps({"region": region, "name": name, "tag": tag,
                                                "icon": head["icon"], "synced": True}),
                }
                if tab == "overview":
                    picked = gd.champion_by_slug(champ) if champ else None
                    rows = views.match_rows(conn, gd, puuid, region, queue,
                                            champion_id=picked.id if picked else None)
                    solo_history = views.rank_history(conn, puuid)
                    if solo_history["points"]:
                        ctx["rank_json"] = json.dumps({
                            "solo": solo_history,
                            "flex": views.rank_history(conn, puuid, "RANKED_FLEX_SR")})
                    ctx.update(rows=rows, groups=views.day_groups(rows),
                               summary=views.summary(rows),
                               season=views.champion_stats(conn, gd, puuid, "ranked"),
                               friends=views.played_with(conn, puuid, region),
                               detail_base=f"/fragments/players/{region}/{riot_id}/matches",
                               more_url=(f"/fragments/players/{region}/{riot_id}/matches?queue={queue}"
                                         + (f"&champ={picked.slug}" if picked else "")),
                               champ=picked.slug if picked else "")
                    ctx["badges"] = recap.achievements(conn, gd, puuid)
                    ctx["tilt"] = views.tilt(conn, puuid)
                    if ctx["summary"]:
                        ctx["focus"] = views.focus(rows, ctx["summary"]["patterns"], insights.baselines(conn),
                                                   views.player_tier(conn, puuid))
                else:                                   # the header's stat band and form
                    ctx["summary"] = views.summary(views.match_rows(conn, gd, puuid, region))
                if tab == "champions":
                    ctx["season"] = views.champion_stats(conn, gd, puuid, cq)
                    ctx["matchups"] = views.personal_matchups(conn, gd, puuid)
                elif tab == "coach":
                    ctx.update(_coach_tab(conn, puuid, region, riot_id))
                elif tab == "live":
                    ctx.update(_live_tab(job, region))
        except HTTPException as exc:
            return error_page("Not a Riot ID", str(exc.detail), 400, region)
        return render("profile.html", **base, **ctx)

    def _coach_tab(conn, puuid: str, region: str, riot_id: str) -> dict[str, Any]:
        from riftwatch.report.html import ACTION_JS, coach_html

        coach_url = (f"/api/players/{region}/{riot_id}/recent/coach/stream"
                     if coach is not None else None)
        out: dict[str, Any] = {"action_js": ACTION_JS}
        try:
            out["coaching"] = coach_html(recent_report(conn, puuid, coach=coach, generate=False),
                                         coach_url)
        except ReportError:
            out["coaching"] = None
        for key, build, html in (("progress_html", progress_report, pages.progress_html),
                                 ("sessions_html", sessions_report, pages.sessions_html),
                                 ("pool_html", champion_pool, pages.pool_html)):
            try:
                out[key] = html(build(conn, puuid))
            except ReportError:
                out[key] = ""
        return out

    def _live_tab(job_id: str | None, region: str) -> dict[str, Any]:
        found = services.jobs.get(job_id) if job_id else None
        if found is None or found.kind != "scout":
            return {"live": None, "live_error": None}
        if found.status == "failed":
            return {"live": None, "live_error": found.error}
        if found.status != "done":
            return {"live": None, "live_error": "Still scouting -- refresh in a moment."}
        return {"live": _live_view(found.result, region)}

    def _live_view(report: dict, region: str) -> dict:
        gd = game_data()
        teams = []
        # The scouted player, so each enemy can get a "why" against the player's own champion.
        mine = next((p for p in report["players"] if p["puuid"] == report.get("me")), None)
        my_champ = gd.champion_by_slug((mine or {}).get("champion") or "")
        me = ({"slug": my_champ.slug, "name": my_champ.name, "icon": gd.champion_icon(my_champ.id),
               "team": mine["team_id"], "role": mine["main_role"] or "",
               "role_name": gamedata.POSITIONS.get(mine["main_role"] or "", "")}
              if mine and my_champ else None)
        for team_id, label in ((100, "Blue team"), (200, "Red team")):
            players = []
            for p in report["players"]:
                if p["team_id"] != team_id:
                    continue
                champ = gd.champion_by_slug(p["champion"] or "")
                r = p["rank"]
                url = None
                if p["puuid"] and "#" in p["riot_id"]:
                    n, t = p["riot_id"].rsplit("#", 1)
                    url = f"/players/{region}/{views.path_id(n, t)}"
                players.append({
                    "name": p["riot_id"], "url": url, "flags": p["flags"],
                    "me": p["puuid"] == report.get("me"),
                    "champion": p["champion"], "icon": gd.champion_icon(champ.id) if champ else "",
                    "slug": champ.slug if champ else None,
                    "lane_opponent": bool(me and p["team_id"] != me["team"] and me["role"]
                                          and p["main_role"] == me["role"]),
                    "rank": (f"{r['tier'].title()} {r['division'] or ''}".strip() if r else None),
                    "rank_key": r["tier"].lower() if r else "", "lp": r["lp"] if r else None,
                    "season": (f"{round(100 * r['wins'] / max(r['wins'] + r['losses'], 1))}% of "
                               f"{r['wins'] + r['losses']}" if r else ""),
                    "recent": (f"{p['wins']}W {p['games'] - p['wins']}L · "
                               f"{p['kills']}/{p['deaths']}/{p['assists']}" if p["games"] else "none"),
                    "role": (f"{p['main_role'].title()} {round(100 * p['main_role_share'])}%"
                             if p["main_role"] else ""),
                    "champ_games": (f"{p['champion_games']} games · "
                                    f"{round(100 * p['champion_wins'] / p['champion_games'])}%"
                                    if p["champion_games"] else "first time"),
                    "mastery": "" if p["mastery_points"] is None else f"{p['mastery_points']:,}"})
            bans = [{"name": b["champion"],
                     "icon": gd.champion_icon((gd.champion_by_slug(b["champion"] or "") or gd.champion(0)).id)}
                    for b in report["bans"] if b["team_id"] == team_id]
            teams.append({"id": team_id, "label": label, "players": players, "bans": bans})
        minutes, seconds = divmod(max(report["game_length_s"], 0), 60)
        return {"queue": report["queue"], "platform": report["platform"],
                "clock": f"{minutes}:{seconds:02d}", "teams": teams, "me": me}

    @app.get("/fragments/players/{region}/{riot_id}/matches", response_class=HTMLResponse)
    def more_matches(region: str, riot_id: str, queue: str = "all",
                     offset: int = Query(0, ge=0), after_day: str = "", champ: str = "") -> HTMLResponse:
        with pool.connection() as conn:
            _, _, account = find_account(conn, region, riot_id)
            if account is None:
                raise HTTPException(404, "player not synced")
            picked = game_data().champion_by_slug(champ) if champ else None
            rows = views.match_rows(conn, game_data(), account["puuid"], region.lower(),
                                    queue if queue in views.QUEUE_FILTERS else "all", offset=offset,
                                    champion_id=picked.id if picked else None)
        html = env.get_template("_match_rows.html").render(
            groups=views.day_groups(rows, after_day or None),
            detail_base=f"/fragments/players/{region}/{riot_id}/matches")
        return HTMLResponse(html)

    @app.get("/fragments/players/{region}/{riot_id}/matches/{match_id}", response_class=HTMLResponse)
    def match_fragment(region: str, riot_id: str, match_id: str) -> HTMLResponse:
        ctx = detail_context(region, riot_id, match_id)
        if ctx is None:
            raise HTTPException(404, "game not stored")
        return HTMLResponse(env.get_template("_match_detail.html").render(**ctx))

    def detail_context(region: str | None, riot_id: str | None, match_id: str) -> dict | None:
        """What _match_detail.html needs for one game, seen from a player if one is given."""
        with pool.connection() as conn:
            account = find_account(conn, region, riot_id)[2] if region and riot_id else None
            detail = views.match_detail(conn, game_data(), match_id,
                                        account["puuid"] if account else None,
                                        (region or PLATFORM_REGION.get(match_id.split("_")[0].lower(), "na")).lower())
        if detail is None:
            return None
        analysis = (f"/players/{region}/{riot_id}/matches/{match_id}"
                    if account is not None and detail["me"] else None)
        map_url = (f"/api/players/{region}/{riot_id}/matches/{match_id}/map"
                   if analysis and detail.get("has_timeline") else None)
        game_url = f"/games/{match_id}" + (f"?as={region}/{riot_id}" if account is not None else "")
        return {"d": detail, "analysis_url": analysis, "map_url": map_url, "game_url": game_url,
                "map_compared": bool(advisor is not None and detail.get("role")
                                     and advisor.available(detail["role"]))}

    @app.get("/games/{match_id}", response_class=HTMLResponse)
    def game_page(match_id: str, as_: str = Query("", alias="as")) -> HTMLResponse:
        """One game on its own page, to share: scoreboards, insights, build, gold graph and the
        High Elo Comparison when seen from a player (``?as=<region>/<Name-TAG>``)."""
        region, _, riot_id = as_.partition("/")
        if region and region_or_none(region) is None:
            region, riot_id = "", ""
        ctx = detail_context(region.lower() or None, riot_id or None, match_id)
        if ctx is None:
            return error_page("Game not found", "RiftWatch hasn't stored this game.", 404)
        me = next((p for t in ctx["d"]["teams"] for p in t["players"] if p.get("me")), None)
        return render("game.html", nav="game", me=me, player_url=(f"/players/{region}/{riot_id}" if me else None),
                      on_game_page=True, **ctx)

    @app.post("/api/players/{region}/{riot_id}/matches/{match_id}/map/{minute}/explain")
    def explain_moment(request: Request, region: str, riot_id: str, match_id: str, minute: int) -> dict:
        """A short AI explanation of one minute of the High Elo Comparison (web.moments)."""
        if coach is None:
            raise HTTPException(503, "AI explanations are switched off on this server")
        with pool.connection() as conn:
            _, _, account = find_account(conn, region, riot_id)
            if account is None:
                raise HTTPException(404, "player not synced")
            game = mapview.map_state(conn, game_data(), advisor, match_id, account["puuid"])
            if game is None or not 0 <= minute < len(game["minutes"]) or not game["minutes"][minute].get("advice"):
                raise HTTPException(404, "no high-elo comparison for that minute")
            stored = conn.execute("SELECT 1 FROM moment_notes WHERE match_id = %s AND puuid = %s AND minute = %s "
                                  "AND version = %s", (match_id, account["puuid"], minute,
                                                       moments.PROMPT_VERSION)).fetchone()
        if stored is None and guard is not None:
            guard(request, "moment")
        try:
            with pool.connection() as conn:
                return moments.explain(conn, coach, game, minute, account["puuid"])
        except Exception as exc:                              # noqa: BLE001 -- shown to the visitor
            log.warning("moment note %s %s failed: %s", match_id, minute, exc)
            raise HTTPException(502, "the coach couldn't write this right now") from exc

    @app.get("/api/players/{region}/{riot_id}/matches/{match_id}/map")
    def match_map(region: str, riot_id: str, match_id: str) -> dict:
        """The High-elo map tab's data: positions, kills and objectives each minute, and for
        the laning minutes what the player did beside what high-elo players chose."""
        with pool.connection() as conn:
            _, _, account = find_account(conn, region, riot_id)
            if account is None:
                raise HTTPException(404, "player not synced")
            state = mapview.map_state(conn, game_data(), advisor, match_id, account["puuid"])
        if state is None:
            raise HTTPException(404, "this game or its timeline isn't stored")
        return state

    @app.get("/scout/{region}/{riot_id}")
    def scout_page(region: str, riot_id: str, job: str | None = None):
        suffix = f"&job={job}" if job else ""
        return RedirectResponse(f"/players/{region}/{riot_id}?tab=live{suffix}", 303)

    # -- champions --------------------------------------------------------------------------

    def _query_string(request: Request):
        def qs(**changes) -> str:
            params = dict(request.query_params)
            for k, v in changes.items():
                if v is None:
                    params.pop(k, None)
                else:
                    params[k] = v
            return urlencode(params)
        return qs

    def _rank(rank: str) -> str:
        return rank if rank in champ_stats.RANKS else "all"

    def patch_changes() -> dict[int, list[dict]]:
        """Champion id -> recent patch-note changes (stats.patches)."""
        gd = game_data()
        return patches.by_champion({c.name.lower(): c.id for c in gd.champions.values()})

    def item_changes() -> list[dict]:
        """Item changes from the patch notes, with Data Dragon item ids (stats.patches)."""
        gd = game_data()
        ids: dict[str, list[int]] = {}
        for iid, it in gd.items.items():
            ids.setdefault(it.name.lower(), []).append(iid)
        return patches.systems(ids)

    def patch_watch() -> list[dict]:
        """The patches in patch_notes.json, newest first, with champion icons for the board."""
        gd = game_data()
        by_name = {c.name.lower(): c for c in gd.champions.values()}
        out = []
        for p in patches.notes()["patches"]:
            items = []
            for c in p["changes"]:
                champ = by_name.get(c["champion"].lower())
                if champ:
                    items.append({**c, "slug": champ.slug, "icon": gd.champion_icon(champ.id)})
            systems = [{**c, "icon": gd.item_icon(min(c["ids"])) if c["ids"] else None}
                       for c in item_changes() if c["patch"] == p["patch"]]
            out.append({**p, "changes": sorted(items, key=lambda c: ("buff", "nerf", "adjust").index(c["kind"])),
                        "systems": systems})
        return out

    @app.get("/champions", response_class=HTMLResponse)
    def champions_page(request: Request, role: str | None = None, rank: str = "all",
                       patch: str | None = None, view: str = "board",
                       notes: str = "on") -> HTMLResponse:
        gd = game_data()
        role = role.upper() if role and role.upper() in gamedata.POSITIONS else None
        rank = _rank(rank)
        adjusted = notes != "raw"
        with pool.connection() as conn:
            known = dict(champ_stats.patches(conn))
            patch = patch if patch in known else None
            data = champ_stats.tier_list(conn, patch, rank, patch_changes() if adjusted else None,
                                         item_changes() if adjusted else None)
        rows = []
        for r in data["rows"]:
            if role and r["position"] != role:
                continue
            c = gd.champion(r["champion_id"])
            rows.append({**r, "name": c.name, "slug": c.slug, "icon": gd.champion_icon(c.id),
                         "role": gamedata.POSITIONS.get(r["position"], ""),
                         "counters": [{"name": gd.champion(x["champion_id"]).name,
                                       "slug": gd.champion(x["champion_id"]).slug,
                                       "icon": gd.champion_icon(x["champion_id"]),
                                       "win_rate": x["win_rate"], "games": x["games"]}
                                      for x in r["counters"]]})
        return render("champions.html", nav="champions", rows=rows, games=data["games"],
                      role=role, role_name=gamedata.POSITIONS.get(role or "", ""), rank=rank,
                      ranks=[(k, v[0]) for k, v in champ_stats.RANKS.items()], patch=patch,
                      patches=list(known.items())[:12], qs=_query_string(request),
                      view="table" if view == "table" else "board", tiers=TIER_BLURBS,
                      adjusted=adjusted, watch=patch_watch(), shift=patches.SHIFT,
                      preview_weight=patches.PREVIEW_WEIGHT,
                      data_patch=max(known.items(), key=lambda kv: kv[1])[0] if known else None)

    @app.get("/champions/{slug}", response_class=HTMLResponse)
    def champion_page(request: Request, slug: str, role: str | None = None, rank: str = "all",
                      patch: str | None = None) -> HTMLResponse:
        gd = game_data()
        champ = gd.champion_by_slug(slug)
        if champ is None:
            return error_page("Champion not found", f"There's no champion called {slug!r}.")
        rank = _rank(rank)

        def item_check(attr):
            return lambda i: bool(gd.item(i) and getattr(gd.item(i), attr))

        def is_starter(i):
            it = gd.item(i)
            return bool(it and it.is_starter and "Trinket" not in it.tags)

        with pool.connection() as conn:
            known = dict(champ_stats.patches(conn))
            patch = patch if patch in known else None
            s = champ_stats.champion(conn, champ.id, role.upper() if role else None, patch, rank,
                                     is_legendary=item_check("is_legendary"),
                                     is_boots=item_check("is_boots"), is_starter=is_starter)
            mains = views.champion_players(conn, champ.id, s.get("position"), PLATFORM_REGION)
            duo = champ_stats.partners(conn, champ.id, s.get("position"))
            curves = champ_stats.win_rate_curves(conn, champ.id, s.get("position"))
            table = champ_stats.matchup_table(conn, champ.id, s.get("position"))
            systems = item_changes()
            usage = champ_stats.item_usage(conn, champ.id, s.get("position"),
                                           sorted({i for c in systems for i in c["ids"]}))
            _, item_hits = patches.system_shift(systems, usage, [("0.0", 1)])

        def item_view(i):
            it = gd.item(i)
            return {"id": i, "icon": gd.item_icon(i), "name": it.name if it else str(i),
                    "text": it.text if it else ""}

        def stat(e, **extra):
            return {"games": e["games"], "win_rate": e["win_rate"], "pick_rate": e["pick_rate"], **extra}

        ctx: dict[str, Any] = {"champ": champ, "s": s, "champ_icon": gd.champion_icon(champ.id),
                               "splash": gd.champion_splash(champ.id), "rank": rank,
                               "ranks": [(k, v[0]) for k, v in champ_stats.RANKS.items()],
                               "patch": patch, "patches": list(known.items())[:12],
                               "positions": gamedata.POSITIONS,
                               "role_name": gamedata.POSITIONS.get(s.get("position", ""), ""),
                               "qs": _query_string(request),
                               "patch_changes": patch_changes().get(champ.id, []),
                               "item_hits": item_hits,
                               "mains": mains, "curves": curves,
                               "matchup_rows": [{**m, "name": gd.champion(m["champion_id"]).name,
                                                 "slug": gd.champion(m["champion_id"]).slug,
                                                 "icon": gd.champion_icon(m["champion_id"])}
                                                for m in table],
                               "duo_role": gamedata.POSITIONS.get(duo["role"] or "", ""),
                               "duo": [{"name": gd.champion(d["champion_id"]).name,
                                        "slug": gd.champion(d["champion_id"]).slug,
                                        "icon": gd.champion_icon(d["champion_id"]),
                                        "games": d["games"], "win_rate": d["win_rate"]}
                                       for d in duo["rows"]]}
        if s.get("games"):
            ctx["runes"] = [stat(e, keystone=gd.rune_name(e["key"][0]), keystone_icon=gd.rune_icon(e["key"][0]),
                                 primary=gd.rune_name(e["key"][1]), primary_icon=gd.rune_icon(e["key"][1]),
                                 sub=gd.rune_name(e["key"][2]), sub_icon=gd.rune_icon(e["key"][2]))
                            for e in s["runes"]]
            page = s["rune_pages"][0]["key"] if s["rune_pages"] else None
            best = s["runes"][0]["key"] if s["runes"] else (None, None, None)
            ctx["page"] = (views._rune_page({"perks": list(page), "primary_style": best[1],
                                             "sub_style": best[2]}, gd) if page else None)
            ctx["spells"] = [stat(e, spells=[{"icon": gd.spell_icon(x), "name": gd.spell_name(x),
                                              "text": gd.spells.get(x, {}).get("text", "")}
                                             for x in e["key"]]) for e in s["spells"]]
            ctx["skill_max"] = [stat(e, order=list(e["key"])) for e in s["skill_max"]]
            ctx["skill_grid"] = (views._skill_grid(s["skill_order"]) if s["skill_order"] else None)
            ctx["core"] = [stat(e, items=[item_view(i) for i in e["key"]]) for e in s["core"]]
            ctx["starting"] = [stat(e, items=[item_view(i) for i in e["key"]]) for e in s["starting"]]
            ctx["boots"] = [stat(e, item=item_view(e["key"])) for e in s["boots"]]
            ctx["late"] = [{"label": label, "items": [stat(e, item=item_view(e["key"])) for e in s[key]]}
                           for key, label in (("fourth", "Fourth item"), ("fifth", "Fifth item"),
                                              ("sixth", "Sixth item")) if s[key]]

            def vs(v):
                c = gd.champion(v["champion_id"])
                return {"name": c.name, "slug": c.slug, "icon": gd.champion_icon(c.id),
                        "games": v["games"], "win_rate": v["win_rate"]}
            ctx["counters"] = [vs(v) for v in s["counters"]]
            ctx["best"] = [vs(v) for v in s["best_against"]]
        detail = gd.champion_detail(champ)
        if detail:
            abil = [{"key": "P", "name": detail["passive"]["name"],
                     "text": gamedata.plain(detail["passive"]["description"]),
                     "icon": gd.passive_icon(detail["passive"]["image"]["full"]), "cooldown": None}]
            for key, sp in zip("QWER", detail["spells"], strict=False):
                abil.append({"key": key, "name": sp["name"], "text": gamedata.plain(sp["description"]),
                             "icon": gd.ability_icon(sp["image"]["full"]),
                             "cooldown": sp.get("cooldownBurn")})
            ctx["abilities"] = abil
        return render("champion.html", nav="champions", **ctx)

    @app.post("/api/champions/{slug}/matchups/{opponent}")
    def matchup_note(request: Request, slug: str, opponent: str, role: str) -> dict:
        """Why ``slug`` does well or badly against ``opponent`` in ``role``: stored, or written
        now by the LLM from measured lane numbers and both kits (see stats.matchups)."""
        gd_ = game_data()
        a, b = gd_.champion_by_slug(slug), gd_.champion_by_slug(opponent)
        role = role.upper()
        if a is None or b is None or role not in gamedata.POSITIONS:
            raise HTTPException(404, "unknown champion or role")
        with pool.connection() as conn:
            pack = matchups.build(conn, gd_, a.id, b.id, role)
            base = {"champion": a.name, "opponent": b.name, "glance": pack["glance"]}
            hit = matchups.cached(conn, a.id, b.id, role, matchups.cache_version(gd_))
            if hit:
                return {**hit, **base}
            if coach is None:
                return {"output": None, "evidence": pack["evidence"], **base,
                        "note": "AI explanations are switched off on this server; here is the evidence."}
        if guard is not None:
            guard(request, "matchup")
        with pool.connection() as conn:
            try:
                note = matchups.explain(conn, gd_, coach, a.id, b.id, role, pack["evidence"])
            except Exception as exc:                          # noqa: BLE001 -- the evidence still helps
                log.warning("matchup note %s vs %s failed: %s", slug, opponent, exc)
                reason = ("the server's Anthropic API key was rejected"
                          if "401" in str(exc) or "api-key" in str(exc) else "the AI call failed")
                return {"output": None, "evidence": pack["evidence"], **base,
                        "note": f"The written explanation isn't available right now ({reason}). "
                                "Here is the evidence it would be written from."}
        return {**note, **base}

    # -- multi-search -----------------------------------------------------------------------

    @app.get("/multi", response_class=HTMLResponse)
    def multi_search(q: str = "") -> HTMLResponse:
        """Everyone in a lobby at once: paste champ-select chat or a list of Riot IDs."""
        gd_ = game_data()
        ids = views.parse_lobby(q)
        with pool.connection() as conn:
            cards = [views.multi_card(conn, gd_, n, t, PLATFORM_REGION) for n, t in ids]
        return render("multi.html", nav="multi", q=q, cards=cards,
                      unparsed=bool(q.strip()) and not ids)

    # -- leaderboard ------------------------------------------------------------------------

    @app.get("/leaderboard", include_in_schema=False)
    def leaderboard_root():
        return RedirectResponse("/leaderboard/na", 303)

    @app.get("/leaderboard/{region}", response_class=HTMLResponse)
    def leaderboard(region: str) -> HTMLResponse:
        platform = region_or_none(region)
        if platform is None:
            return error_page("Unknown region", f"RiftWatch doesn't know the region {region!r}.")
        region = region.lower()
        hit = boards.get(platform)
        if hit is None or time.monotonic() - hit[0] > LEADERBOARD_S:
            board = _leaderboard(platform, region)
            boards[platform] = (time.monotonic(), board)
        else:
            board = hit[1]
        return render("leaderboard.html", nav="leaderboard", region=region,
                      board_regions=("na", "euw", "kr", "eune", "br", "oce", "jp", "vn"), **board)

    def _leaderboard(platform: str, region: str) -> dict:
        gd = game_data()
        error = None
        players: list[dict] = []
        source = ""
        if services.api is not None:
            try:
                entries = []
                for tier in ("CHALLENGER", "GRANDMASTER"):
                    entries += [dict(e, tier=tier) for e in
                                services.api.league_entries(platform, tier, "I")]
                    if len(entries) >= 100:
                        break
                entries.sort(key=lambda e: (e["tier"] != "CHALLENGER", -e.get("leaguePoints", 0)))
                entries = entries[:100]
                with pool.connection() as conn:
                    names = champ_stats.leaderboard_names(conn, [e["puuid"] for e in entries if e.get("puuid")])
                for i, e in enumerate(entries, 1):
                    n, t = names.get(e.get("puuid"), (None, None))
                    games = e.get("wins", 0) + e.get("losses", 0)
                    players.append({"rank": i, "name": n, "tag": t, "tier": e["tier"],
                                    "crest": gd.tier_crest(e["tier"]), "lp": e.get("leaguePoints"),
                                    "wins": e.get("wins"), "losses": e.get("losses"),
                                    "win_rate": round(100 * e.get("wins", 0) / games) if games else None,
                                    "puuid": e.get("puuid"),
                                    "url": f"/players/{region}/{views.path_id(n, t)}" if n and t else None})
                source = (f"Top {len(players)} ranked solo/duo players on {region.upper()}, live "
                          "from Riot (refreshed every 30 minutes).")
            except Exception as exc:          # an expired key shouldn't take the page down
                error = f"Riot didn't answer ({exc})."
                players = []
        if not players:
            with pool.connection() as conn:
                rows = conn.execute(
                    """SELECT puuid, tier FROM crawl_players WHERE platform = %s
                         AND tier IN ('CHALLENGER', 'GRANDMASTER', 'MASTER')
                       ORDER BY array_position(ARRAY['CHALLENGER','GRANDMASTER','MASTER'], tier),
                                matches_found DESC LIMIT 100""", (platform,)).fetchall()
                names = champ_stats.leaderboard_names(conn, [r[0] for r in rows])
            for i, (puuid, tier) in enumerate(rows, 1):
                n, t = names.get(puuid, (None, None))
                players.append({"rank": i, "name": n, "tag": t, "tier": tier,
                                "crest": gd.tier_crest(tier), "lp": None, "wins": None,
                                "losses": None, "win_rate": None, "puuid": puuid,
                                "url": f"/players/{region}/{views.path_id(n, t)}" if n and t else None})
            source = ("Apex-tier players RiftWatch has analysed games from (live LP needs the Riot "
                      "API).") if players else ""
        # Their mains, from games RiftWatch has stored (crawled high-elo games cover most of them).
        with pool.connection() as conn:
            found = champ_stats.mains(conn, [p["puuid"] for p in players if p.get("puuid")])
        for p in players:
            m = found.get(p.pop("puuid", None) or "")
            p["mains"] = [{"name": gd.champion(c["champion_id"]).name, "slug": gd.champion(c["champion_id"]).slug,
                           "icon": gd.champion_icon(c["champion_id"]), "games": c["games"]}
                          for c in (m or {}).get("champions", [])]
            p["role"] = gamedata.POSITIONS.get((m or {}).get("role") or "", "")
            p["role_key"] = (m or {}).get("role") or ""
            p["stored"] = (m or {}).get("games", 0)
        return {"players": players, "source": source, "error": error}

    # -- the match analysis page, inside the site's header -----------------------------------

    def site_chrome(page_html: str, region: str) -> str:
        asset_version = _asset_version()
        head = (f'<link rel="stylesheet" href="/static/site.css?v={asset_version}">'
                '<script>try{var t=JSON.parse(localStorage.getItem("rw-theme"));'
                'document.documentElement.dataset.theme=t||"dark"}catch(e){'
                'document.documentElement.dataset.theme="dark"}</script>')
        header = env.get_template("_header.html").render(regions=REGIONS, region=region,
                                                         nav="match")
        script = f'<script src="/static/site.js?v={asset_version}" defer></script>'
        page_html = page_html.replace("</head>", head + "</head>", 1)
        page_html = page_html.replace("<body><main>", f'<body class="report">{header}<main>', 1)
        return page_html.replace("</body>", script + "</body>", 1)

    app.state.site_chrome = site_chrome

    @app.exception_handler(404)
    async def not_found(request: Request, exc) -> Any:
        if request.url.path.startswith(("/api/", "/fragments/")):
            detail = getattr(exc, "detail", "not found")
            return JSONResponse({"detail": detail}, status_code=404)
        return error_page("Page not found", "That page doesn't exist. Search for a player instead.")
