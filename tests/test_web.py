import os

import pytest

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="RIFTWATCH_TEST_DATABASE_URL not set")


@pytest.fixture
def web(request, tmp_path):
    from fastapi.testclient import TestClient
    from psycopg_pool import ConnectionPool

    from riftwatch.baselines import build as bl
    from riftwatch.config import Settings
    from riftwatch.db import migrate, repo
    from riftwatch.db.connection import connect
    from riftwatch.features import store
    from riftwatch.web.app import create_app
    from riftwatch.web.jobs import JobQueue
    from tests.fixtures import build_game
    from riftwatch.coach.grounding import CoachOutput
    from tests.test_coach import fake_coach, point
    from tests.test_ingest import FakeApi

    with connect(TEST_DB) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(conn)
        # Crawled games to build GOLD... baselines from; the player is EMERALD in FakeApi,
        # so tag these EMERALD.
        for i in range(25):
            mid = f"NA1_{3000 + i}"
            match, timeline = build_game(mid, cs_bonus=i * 0.1)
            repo.insert_match(conn, match)
            repo.insert_timeline(conn, mid, timeline)
            repo.mark_sample(conn, mid, "EMERALD", "crawl")
        store.extract_pending(conn)
        bl.build(conn)
        bl.invalidate_cache()

    pool = ConnectionPool(TEST_DB, min_size=1, max_size=6, open=True, kwargs={"autocommit": True})
    jobs = JobQueue(pool, workers=2, cooldown_s=60)
    api = FakeApi(5)
    # Cites only the context item and states no numbers, so it passes grounding for any game.
    ok = CoachOutput(headline="Review.", points=[point(evidence_ids=["E1"], explanation="A game.")])
    coach, coach_calls = fake_coach(ok, ok, ok)
    # Tests can pass extra settings with @pytest.mark.parametrize("web", [{...}], indirect=True).
    # An empty models folder, so the tests don't load whatever models are trained locally.
    settings = Settings.from_env({"RIFTWATCH_DATABASE_URL": TEST_DB,
                                  "RIFTWATCH_MODELS_DIR": str(tmp_path / "models"),
                                  **getattr(request, "param", {})})
    from riftwatch.web.gamedata import GameData

    from riftwatch.web import insights, mapview

    from riftwatch.stats import games as game_stats

    mapview.clear_cache()         # module-level caches would carry one test's games into the next
    insights.clear_cache()
    game_stats.clear_cache()
    gd = GameData()
    gd.loaded = True              # no Data Dragon in tests: names fall back, icons are empty
    app = create_app(settings, api=api, coach=coach, pool=pool, jobs=jobs, gamedata=gd)
    with TestClient(app) as client:
        yield client, api, coach_calls, jobs
    pool.close()


def sync(client, jobs):
    r = client.post("/api/players/na/Me-NA1/sync")
    assert r.status_code == 202
    job = jobs.wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    return r.json(), job


def test_health(web):
    client, *_ = web
    assert client.get("/api/health").json()["ok"] is True


def test_unknown_player_404_points_at_sync(web):
    client, *_ = web
    r = client.get("/api/players/na/Me-NA1")
    assert r.status_code == 404 and "sync" in r.json()["detail"]


def test_bad_inputs(web):
    client, *_ = web
    assert client.get("/api/players/na/NoTag").status_code == 400
    assert client.get("/api/players/atlantis/Me-NA1").status_code == 400


def test_sync_then_player_page(web):
    client, api, _, jobs = web
    body, job = sync(client, jobs)
    assert job.result == {"riot_id": "Me#NA1", "new_matches": 5, "already_cached": 0, "failed": 0}
    player = client.get("/api/players/na/me-na1").json()       # case-insensitive
    assert player["rank"]["tier"] == "EMERALD" and player["games_cached"] == 5
    recent = player["recent"]
    assert [m["match_id"] for m in recent] == api.ids        # newest first
    assert recent[0]["analysed"] and recent[0]["cs_per_min"] > 0
    page = client.get("/api/players/na/Me-NA1/matches?limit=2&offset=1").json()
    assert [m["match_id"] for m in page["matches"]] == api.ids[1:3]


def test_repeat_sync_reuses_the_job(web):
    client, api, _, jobs = web
    first, _ = sync(client, jobs)
    calls = len(api.calls)
    again = client.post("/api/players/na/Me-NA1/sync").json()
    assert again["created"] is False and again["job"]["id"] == first["job"]["id"]
    assert len(api.calls) == calls                            # cooldown: no Riot traffic


def test_match_review_never_calls_the_llm_on_get(web):
    client, api, coach_calls, jobs = web
    sync(client, jobs)
    mid = api.ids[0]
    review = client.get(f"/api/players/na/Me-NA1/matches/{mid}").json()
    assert review["comparison"]["tier"] == "EMERALD"
    assert review["coach"]["pending"] is True and review["coach"]["model"] == "offline"
    assert coach_calls.calls == []
    cs = next(c for c in review["curves"] if c["name"] == "cs")
    assert cs["points"][9]["p50"] is not None
    assert review["scorecard"] and review["evidence"][0]["kind"] == "context"

    coached = client.post(f"/api/players/na/Me-NA1/matches/{mid}/coach").json()
    assert coached["coach"]["model"].startswith("claude-sonnet-5-5")
    assert len(coach_calls.calls) == 1

    again = client.get(f"/api/players/na/Me-NA1/matches/{mid}").json()
    assert again["coach"]["cached"] is True and again["coach"]["pending"] is False
    assert len(coach_calls.calls) == 1                        # served from the cache


def test_recent_review_and_html_page(web):
    client, api, _, jobs = web
    sync(client, jobs)
    recent = client.get("/api/players/na/Me-NA1/recent?games=5").json()
    assert len(recent["games"]) == 5 and recent["coach"]["pending"] is True
    page = client.get(f"/players/na/Me-NA1/matches/{api.ids[0]}")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "Minute by minute" in page.text


def test_review_of_a_game_the_player_wasnt_in(web):
    client, api, _, jobs = web
    sync(client, jobs)
    r = client.get("/api/players/na/Me-NA1/matches/NA1_3000")
    assert r.status_code == 404


# -- pages ------------------------------------------------------------------------------------

def test_search_redirects_to_player_page(web):
    client, *_ = web
    home = client.get("/").text
    assert 'action="/search"' in home and 'name="riot_id"' in home
    r = client.get("/search?region=NA&riot_id=Some%20One%23NA1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/players/na/Some%20One-NA1"
    assert client.get("/search?region=na&riot_id=nohash").status_code == 400


def test_unsynced_player_page_loads_the_player(web):
    client, *_ = web
    page = client.get("/players/na/Me-NA1").text
    assert 'data-autosync="/api/players/na/Me-NA1/sync"' in page   # starts the sync itself
    assert "Loading Me#NA1 from Riot" in page


def test_player_and_match_pages(web):
    client, api, _, jobs = web
    sync(client, jobs)
    page = client.get("/players/na/Me-NA1").text
    assert "Me#NA1" in page and "Emerald" in page
    assert page.count('class="match ') == 5                        # every synced game
    assert f'data-detail="/fragments/players/na/Me-NA1/matches/{api.ids[0]}"' in page
    assert "Victory" in page or "Defeat" in page
    detail = client.get(f"/fragments/players/na/Me-NA1/matches/{api.ids[0]}")
    assert detail.status_code == 200 and "Blue team" in detail.text and "Red team" in detail.text
    assert f'href="/players/na/Me-NA1/matches/{api.ids[0]}"' in detail.text   # analysis link
    more = client.get("/fragments/players/na/Me-NA1/matches?queue=all&offset=3").text
    assert more.count('class="match ') == 2
    coach_tab = client.get("/players/na/Me-NA1?tab=coach").text
    assert "/api/players/na/Me-NA1/recent/coach" in coach_tab      # coaching not cached yet
    assert client.get("/players/na/Me-NA1?tab=champions").status_code == 200
    match = client.get(f"/players/na/Me-NA1/matches/{api.ids[0]}").text
    assert f"/api/players/na/Me-NA1/matches/{api.ids[0]}/coach" in match
    assert 'href="/players/na/Me-NA1"' in match                    # back to the player
    assert 'class="site-header"' in match                          # inside the site's header


def test_scout_job_and_page(web):
    client, api, _, jobs = web
    page = client.get("/scout/na/Me-NA1")                          # old URL: now the Live tab
    assert page.status_code == 200 and "Scout live game" in page.text
    assert 'data-scout="/api/scout/na/Me-NA1"' in page.text
    r = client.post("/api/scout/na/Me-NA1")
    assert r.status_code == 202
    job = jobs.wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert len(job.result["players"]) == 10
    html = client.get(f"/players/na/Me-NA1?tab=live&job={job.token}").text
    assert "Blue team" in html and "Red team" in html and "<strong>Me#NA1</strong>" in html
    assert 'href="/players/na/Player2-NA1"' in html

    api.in_game = False
    jobs.cooldown_s = 0
    failed = jobs.wait(client.post("/api/scout/na/Me-NA1").json()["job"]["id"])
    assert failed.status == "failed" and "isn't in a game" in failed.error
    assert "in a game" in client.get(f"/players/na/Me-NA1?tab=live&job={failed.token}").text


def test_champion_pool_api_and_page_section(web):
    client, api, _, jobs = web
    sync(client, jobs)
    body = client.get("/api/players/na/Me-NA1/champions").json()
    assert body["tier"] == "EMERALD" and body["champions"][0]["champion"] == "Aatrox"
    assert body["champions"][0]["games"] == 5
    assert "Champion pool" in client.get("/players/na/Me-NA1?tab=coach").text


def test_progress_api(web):
    client, api, _, jobs = web
    sync(client, jobs)
    # FakeApi games are from late September 2026; 52 weeks reaches them.
    body = client.get("/api/players/na/Me-NA1/progress", params={"weeks": 52}).json()
    assert body["games"] == 5 and body["tier"] == "EMERALD"
    assert body["trend"]["verdict"] in ("steady", "improving", "declining")
    assert sum(w["games"] for w in body["weeks"]) == 5


def test_progress_section_renders_chart_and_trend():
    from datetime import date

    from riftwatch.analysis.progress import ProgressReport, Trend, Week
    from riftwatch.web.pages import progress_html

    weeks = [Week(date(2026, 9, 7), 10, 6, 44.0, {"farming": 40.0}),
             Week(date(2026, 9, 14), 8, 3, 39.5, {"farming": 35.0})]
    html = progress_html(ProgressReport("PLATINUM", weeks, Trend(-0.4, 0.1, 18), [],
                                         {"farming": Trend(-0.6, 0.2, 18)}))
    assert "<h2>Progress</h2>" in html and "<polyline" in html and "declining" in html
    assert "farming declining" in html and "Week of Sep 07: better than 44%" in html


def test_recent_coaching_streams_on_the_player_page(web):
    import json

    client, api, coach_calls, jobs = web
    sync(client, jobs)
    page = client.get("/players/na/Me-NA1?tab=coach").text
    assert 'data-stream="/api/players/na/Me-NA1/recent/coach/stream"' in page
    r = client.post("/api/players/na/Me-NA1/recent/coach/stream")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]
    assert [e["type"] for e in events if e["type"] in ("point", "final")][-1] == "final"
    assert any(e["type"] == "point" for e in events) and len(coach_calls.calls) == 1
    # Cached now: the page shows it, and streaming again is a single "final" event.
    again = [json.loads(line[6:]) for line in
             client.post("/api/players/na/Me-NA1/recent/coach/stream").text.split("\n\n")
             if line.startswith("data: ")]
    assert [e["type"] for e in again] == ["final"] and again[0]["cached"]
    assert len(coach_calls.calls) == 1


def test_jobs_are_only_reachable_by_their_token(web):
    client, api, _, jobs = web
    body = client.post("/api/players/na/Me-NA1/sync").json()
    token = body["job"]["id"]
    assert len(token) == 32 and not token.isdigit()
    job = jobs.wait(token)
    assert client.get(f"/api/jobs/{token}").json()["status"] == "done"
    # The sequential id is no way in.
    assert client.get(f"/api/jobs/{job.id}").status_code == 404
    assert client.get("/api/jobs/1").status_code == 404


def test_page_escapes_riot_ids(web):
    client, *_ = web
    page = client.get("/players/na/%3Cscript%3Ealert(1)%3C%2Fscript%3E-NA1").text
    assert "<script>alert(1)</script>" not in page


# -- public-site guards -------------------------------------------------------------------------

@pytest.mark.parametrize("web", [{"RIFTWATCH_RATE_LIMITS": "sync=2"}], indirect=True)
def test_visitor_rate_limit_answers_429_with_retry_after(web):
    client, *_ = web
    for _ in range(2):
        assert client.post("/api/players/na/Me-NA1/sync").status_code == 202
    blocked = client.post("/api/players/na/Me-NA1/sync")
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["retry-after"]) <= 3600
    assert "too many sync requests" in blocked.json()["detail"]
    # Other actions keep their own allowance.
    assert client.post("/api/scout/na/Me-NA1").status_code == 202


@pytest.mark.parametrize("web", [{"RIFTWATCH_COACH_DAILY_BUDGET_USD": "0"}], indirect=True)
def test_coaching_stops_at_the_daily_budget(web):
    client, api, coach_calls, jobs = web
    sync(client, jobs)
    r = client.post(f"/api/players/na/Me-NA1/matches/{api.ids[0]}/coach")
    assert r.status_code == 503 and "budget" in r.json()["detail"]
    assert coach_calls.calls == []
    # Reading pages and cached data still works.
    assert client.get(f"/api/players/na/Me-NA1/matches/{api.ids[0]}").status_code == 200


def test_pages_carry_the_legal_notice_headers_and_compression(web):
    client, *_ = web
    r = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert "isn't endorsed by Riot Games" in r.text
    assert r.headers["content-encoding"] == "gzip"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert client.get("/api/health").headers["referrer-policy"] == "same-origin"


def test_parse_limits():
    from riftwatch.web.limits import DEFAULT_LIMITS, parse_limits

    assert parse_limits(None) == DEFAULT_LIMITS
    assert parse_limits("coach=3, sync=100") == {**DEFAULT_LIMITS, "coach": 3, "sync": 100}
    with pytest.raises(ValueError):
        parse_limits("coach=lots")
    with pytest.raises(ValueError):
        parse_limits("download=5")


# -- the rest of the site ---------------------------------------------------------------------

def test_champion_pages_and_tier_list(web):
    client, *_ = web
    from riftwatch.stats import champions

    champions.clear_cache()
    tiers = client.get("/champions")
    assert tiers.status_code == 200 and "Champion tier list" in tiers.text
    assert client.get("/champions?role=top&rank=master_plus&patch=99.1").status_code == 200
    assert client.get("/champions/not-a-champion").status_code == 404


def test_autocomplete_finds_players_seen_in_games(web):
    client, api, _, jobs = web
    sync(client, jobs)
    found = client.get("/api/autocomplete?q=pla").json()["players"]
    assert found and all(p["name"].lower().startswith("pla") for p in found)
    assert {"name", "tag", "region", "icon"} <= set(found[0])
    assert client.get("/api/autocomplete?q=zzzz").json()["players"] == []


def test_leaderboard_about_and_missing_pages(web):
    client, *_ = web
    assert client.get("/leaderboard/na").status_code == 200
    assert client.get("/leaderboard/atlantis").status_code == 404
    assert "How RiftWatch works" in client.get("/about").text
    missing = client.get("/no/such/page")
    assert missing.status_code == 404 and "Page not found" in missing.text
    assert client.get("/api/nope").json()["detail"]
    assert client.get("/static/site.css").status_code == 200


def test_players_are_found_in_any_region(web):
    client, api, _, jobs = web
    # Not stored yet: Riot says where they play.
    r = client.get("/search?riot_id=Me%23NA1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/players/na/Me-NA1"
    # Opened under the wrong region: the sync job doesn't sync there, it points the way.
    r = client.post("/api/players/euw/Me-NA1/sync")
    job = jobs.wait(r.json()["job"]["id"])
    assert job.status == "done" and job.result == {"redirect": "/players/na/Me-NA1"}
    sync(client, jobs)
    # Stored: found without asking Riot, and a wrong-region URL redirects.
    calls = len(api.calls)
    r = client.get("/search?region=auto&riot_id=me%23na1", follow_redirects=False)
    assert r.headers["location"] == "/players/na/Me-NA1" and len(api.calls) == calls
    r = client.get("/players/euw/Me-NA1?tab=champions", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/players/na/Me-NA1?tab=champions"
    missing = client.get("/search?riot_id=Nobody%23NA1")
    assert missing.status_code == 400 and "in any region" in missing.text


def test_game_insights_and_map_tabs(web):
    client, api, _, jobs = web
    sync(client, jobs)
    page = client.get("/players/na/Me-NA1").text
    assert 'class="m-notes"' in page or 'class="note ' not in page   # notes only when something stood out
    detail = client.get(f"/fragments/players/na/Me-NA1/matches/{api.ids[0]}").text
    assert 'data-pane="insights"' in detail and "What went well" in detail and "What to work on" in detail
    assert f'data-map-url="/api/players/na/Me-NA1/matches/{api.ids[0]}/map"' in detail
    assert "Map replay" in detail                                      # no high-elo model in tests
    state = client.get(f"/api/players/na/Me-NA1/matches/{api.ids[0]}/map").json()
    assert state["compared"] is False and state["me"] and len(state["minutes"]) > 10
    first = state["minutes"][5]
    assert str(state["me"]) in first["pos"] and 0 <= first["pos"][str(state["me"])]["x"] <= 100
    assert client.get("/api/players/na/Me-NA1/matches/NA1_999/map").status_code == 404


def test_matchup_notes_need_known_champions(web):
    client, *_ = web
    r = client.post("/api/champions/nobody/matchups/noone?role=top")
    assert r.status_code == 404


def test_lobby_lookup_page(web):
    client, api, _, jobs = web
    sync(client, jobs)
    page = client.get("/multi", params={"q": "Me #NA1 joined the lobby\nStranger#EUW"}).text
    assert "Open profile" in page and 'href="/players/na/Me-NA1"' in page
    assert "Load this player" in page and "Stranger" in page
    assert "No Riot IDs found" in client.get("/multi", params={"q": "hello"}).text


def test_tier_board_can_show_raw_data(web):
    client, *_ = web
    assert "Patch watch" in client.get("/champions").text
    raw = client.get("/champions?notes=raw").text
    assert "Showing the raw data" in raw


def test_rank_progression_and_comparison_state(web):
    client, api, _, jobs = web
    sync(client, jobs)
    page = client.get("/players/na/Me-NA1").text
    assert 'class="panel rankprog"' in page and "Rank progression" in page
    state = client.get(f"/api/players/na/Me-NA1/matches/{api.ids[0]}/map").json()
    minute = state["minutes"][10]
    assert {"lead", "dragons", "level", "hp", "unspent"} <= set(minute["state"])
    assert "dragon" in state["pits"] and isinstance(minute["this_minute"], list)
    # No high-elo model in tests, so there's no minute for the coach to explain.
    r = client.post(f"/api/players/na/Me-NA1/matches/{api.ids[0]}/map/10/explain")
    assert r.status_code == 404


def test_recap_compare_and_counter_pages(web):
    client, api, _, jobs = web
    sync(client, jobs)
    recap = client.get("/players/na/Me-NA1/recap")
    assert recap.status_code == 200 and "Season recap" in recap.text and "Most played" in recap.text
    assert 'class="heatmap" data-played="[[' in recap.text
    profile = client.get("/players/na/Me-NA1").text
    assert 'href="/players/na/Me-NA1/recap"' in profile and 'href="/compare?a=Me%23NA1"' in profile
    cmp = client.get("/compare", params={"a": "Me#NA1", "b": "Stranger#EUW"}).text
    assert "Load this player" in cmp and 'href="/players/na/Me-NA1"' in cmp
    assert "needs a Riot ID" in client.get("/compare", params={"a": "nohash", "b": "x"}).text
    counters = client.get("/counters")
    assert counters.status_code == 200 and "Who are you up against?" in counters.text
    assert client.get("/players/na/Nobody-NA1/recap", follow_redirects=False).status_code == 303


def test_game_page_stats_page_and_history_filter(web):
    client, api, _, jobs = web
    sync(client, jobs)
    game = client.get(f"/games/{api.ids[0]}?as=na/Me-NA1")
    assert game.status_code == 200 and 'class="res ' in game.text and 'data-pane="insights"' in game.text
    assert 'href="/players/na/Me-NA1"' in game.text          # back to the player it is seen from
    assert "Game page" not in game.text                        # no link to the page it's already on
    assert client.get(f"/games/{api.ids[0]}").status_code == 200   # no player: scoreboards only
    assert client.get("/games/NA1_404").status_code == 404
    detail = client.get(f"/fragments/players/na/Me-NA1/matches/{api.ids[0]}").text
    assert f'href="/games/{api.ids[0]}?as=na/Me-NA1"' in detail
    filtered = client.get("/players/na/Me-NA1?champ=nobody")         # unknown champion: no filter
    assert filtered.status_code == 200 and 'class="match ' in filtered.text
    stats = client.get("/stats")
    assert stats.status_code == 200 and "What wins games" in stats.text and "Gold lead at 15 minutes" in stats.text
    assert client.get("/stats?tab=items").status_code == 200


def test_player_champion_page_and_manifest(web):
    client, api, _, jobs = web
    sync(client, jobs)
    page = client.get("/players/na/Me-NA1")
    assert 'href="/players/na/Me-NA1/champions/' in page.text or "Season" in page.text
    assert client.get("/players/na/Me-NA1/champions/nobody").status_code in (400, 404)
    assert client.get("/players/na/Nobody-NA1/champions/zed", follow_redirects=False).status_code in (303, 404)
    manifest = client.get("/static/manifest.webmanifest")
    assert manifest.status_code == 200 and manifest.json()["name"] == "RiftWatch"
    assert 'rel="manifest"' in page.text


def test_patches_page_and_champion_search(web):
    client, *_ = web
    page = client.get("/patches")
    assert page.status_code == 200 and "Patch notes, and what they moved" in page.text and "Patch 26.22" in page.text
    found = client.get("/api/autocomplete?q=ah").json()
    assert "players" in found and "champions" in found


def test_benchmarks_page(web):
    client, *_ = web
    page = client.get("/benchmarks?role=jungle")
    assert page.status_code == 200 and "What jungle players do at each rank" in page.text
    assert client.get("/benchmarks?role=nonsense").status_code == 200
