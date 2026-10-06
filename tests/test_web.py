import os

import pytest

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="RIFTWATCH_TEST_DATABASE_URL not set")


@pytest.fixture
def web(request):
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
    settings = Settings.from_env({"RIFTWATCH_DATABASE_URL": TEST_DB, **getattr(request, "param", {})})
    app = create_app(settings, api=api, coach=coach, pool=pool, jobs=jobs)
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
    assert "Name#TAG" in client.get("/").text
    r = client.get("/search?region=NA&riot_id=Some%20One%23NA1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/players/na/Some%20One-NA1"
    assert client.get("/search?region=na&riot_id=nohash").status_code == 400


def test_unsynced_player_page_offers_update(web):
    client, *_ = web
    page = client.get("/players/na/Me-NA1").text
    assert "Not on RiftWatch yet" in page and 'id="sync"' in page
    assert "/api/players/na/Me-NA1/sync" in page


def test_player_and_match_pages(web):
    client, api, _, jobs = web
    sync(client, jobs)
    page = client.get("/players/na/Me-NA1").text
    assert "Me#NA1" in page and "Emerald" in page
    assert f"/players/na/Me-NA1/matches/{api.ids[0]}" in page
    assert "/api/players/na/Me-NA1/recent/coach" in page           # coaching not cached yet
    match = client.get(f"/players/na/Me-NA1/matches/{api.ids[0]}").text
    assert f"/api/players/na/Me-NA1/matches/{api.ids[0]}/coach" in match
    assert 'href="/players/na/Me-NA1"' in match                    # back to the player


def test_scout_job_and_page(web):
    client, api, _, jobs = web
    assert 'href="/scout/na/Me-NA1">Live game' in client.get("/players/na/Me-NA1").text
    page = client.get("/scout/na/Me-NA1")
    assert page.status_code == 200 and "Scout live game" in page.text
    r = client.post("/api/scout/na/Me-NA1")
    assert r.status_code == 202
    job = jobs.wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert len(job.result["players"]) == 10
    html = client.get(f"/scout/na/Me-NA1?job={job.id}").text
    assert "Blue team" in html and "Red team" in html and "<strong>Me#NA1</strong>" in html
    assert 'href="/players/na/Player2-NA1"' in html

    api.in_game = False
    jobs.cooldown_s = 0
    failed = jobs.wait(client.post("/api/scout/na/Me-NA1").json()["job"]["id"])
    assert failed.status == "failed" and "isn't in a game" in failed.error
    assert "isn&#x27;t in a game" in client.get(f"/scout/na/Me-NA1?job={failed.id}").text


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
