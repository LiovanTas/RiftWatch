"""The champ-select draft assistant: the score's parts, and the page and API."""

import os

import pytest

from riftwatch.web import draft
from riftwatch.web.draft import Draft, Evidence

COMP = {"magic": [(0.2, 0.44, 1000), (0.42, 0.505, 9000), (0.7, 0.49, 2000)],
        "tanks": {0: (1000, 494), 1: (1000, 504), 2: (1000, 513)},
        "comfort": {"0": (1000, 497), "1-2": (1000, 500), "3-9": (1000, 513), "10-29": (1000, 525), "30+": (1000, 523)},
        "overall": 0.5, "avg_magic": 0.42, "teams": 13000}
CLASSES = {1: ("Mage", 0.9), 2: ("Fighter", 0.1), 3: ("Marksman", 0.05), 4: ("Tank", 0.3), 5: ("Assassin", 0.2),
           6: ("Fighter", 0.05), 7: ("Mage", 0.85), 8: ("Marksman", 0.05)}


def evidence(**kw):
    base = dict(role_table={1: (2000, 1000), 2: (2000, 1000), 4: (2000, 1000), 5: (2000, 1000), 7: (40, 34)},
                classes=CLASSES, comp=COMP)
    base.update(kw)
    return Evidence(**base)


def by_id(result):
    return {r["champion_id"]: r for r in result["picks"]}


def test_banned_and_picked_champions_are_never_suggested():
    d = Draft(role="MIDDLE", bans=[1], allies=[(2, "TOP")], enemies=[(4, None)])
    picks = by_id(draft.score(d, evidence()))
    assert set(picks) == {5, 7}


def test_small_samples_are_pulled_towards_even():
    # 34-6 over 40 games is 85%, but 40 games prove little: it's pulled most of the way back.
    picks = by_id(draft.score(Draft(role="MIDDLE"), evidence()))
    strength = next(p for p in picks[7]["parts"] if p["key"] == "strength")
    assert 50 < 50 + strength["pts"] < 62
    assert picks[1]["win"] == pytest.approx(50.0, abs=0.5)    # an even record, an even chance


def test_the_lane_matchup_counts_only_beyond_the_champions_own_strength():
    ev = evidence(lane_opponent=8, lane={1: (200, 120), 2: (2, 2)})
    picks = by_id(draft.score(Draft(role="MIDDLE", enemies=[(8, "MIDDLE")]), ev))
    lane = {c: next(p for p in picks[c]["parts"] if p["key"] == "lane")["pts"] for c in (1, 2, 4)}
    assert lane[1] > 5                       # 60% over 200 games against this laner
    assert 0 < lane[2] < 2                   # 2-0 is almost nothing
    assert lane[4] == 0                      # never met: no effect


def test_a_physical_team_wants_magic_damage():
    d = Draft(role="MIDDLE", allies=[(2, "TOP"), (3, "BOTTOM"), (6, "JUNGLE")], enemies=[(7, None)])
    result = draft.score(d, evidence())
    comp = {r["champion_id"]: next(p for p in r["parts"] if p["key"] == "comp")["pts"] for r in result["picks"]}
    assert comp[1] > 0 > comp[5]             # the mage helps an all-physical team; another AD pick doesn't
    texts = " ".join(n["text"] for n in result["needs"])
    assert "magic-damage pick helps" in texts and "magic resist will matter" in texts and "No tank yet" in texts


def test_pairs_are_capped_and_comfort_rewards_experience():
    pairs = {(5, 2, True): (500, 400), (5, 3, True): (500, 400), (5, 6, True): (500, 400)}
    player = {"champs": {5: {"games": 40, "wins": 26, "role_games": 40, "role_wins": 26}},
              "classes": {"Assassin": (40, 26), "Mage": (100, 45)}, "overall": (140, 71), "swing": {}}
    d = Draft(role="MIDDLE", allies=[(2, "TOP"), (3, "BOTTOM"), (6, "JUNGLE")])
    picks = by_id(draft.score(d, evidence(pairs=pairs, player=player)))
    allies = next(p for p in picks[5]["parts"] if p["key"] == "allies")
    assert allies["pts"] <= draft.pts(draft.PAIR_CAP) + 0.01
    comfort = {c: next(p for p in picks[c]["parts"] if p["key"] == "comfort")["pts"] for c in (5, 1)}
    assert comfort[5] > 0 > comfort[1]       # 40 games on it, a winning record; never played the other
    style = {c: next(p for p in picks[c]["parts"] if p["key"] == "style")["pts"] for c in (5, 1)}
    assert style[5] > 0 > style[1]           # wins more on assassins than mages
    result = draft.score(d, evidence(pairs=pairs, player=player))
    assert [r["champion_id"] for r in result["yours"]] == [5]


def test_picks_parse_with_roles_limits_and_no_repeats():
    assert draft.parse_ids("103:middle,64,103,x,7:nope", 5) == [(103, "MIDDLE"), (64, None), (7, None)]
    assert len(draft.parse_ids(",".join(str(i) for i in range(1, 30)), 10)) == 10


# -- the page and the API, on a database of its own ----------------------------------------

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


@pytest.fixture
def web(tmp_path):
    if not TEST_DB:
        pytest.skip("RIFTWATCH_TEST_DATABASE_URL not set")
    from fastapi.testclient import TestClient
    from psycopg_pool import ConnectionPool

    from riftwatch.config import Settings
    from riftwatch.db import migrate, repo
    from riftwatch.db.connection import connect
    from riftwatch.stats import champions as champion_stats
    from riftwatch.web.app import create_app
    from riftwatch.web.gamedata import GameData
    from riftwatch.web.jobs import JobQueue
    from tests.fixtures import build_game
    from tests.test_ingest import FakeApi

    with connect(TEST_DB) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(conn)
        for i in range(40):                  # enough games for each champion to be suggested
            mid = f"NA1_{7000 + i}"
            match, timeline = build_game(mid, cs_bonus=i * 0.1, blue_wins=i % 3 != 0)
            repo.insert_match(conn, match)
            repo.insert_timeline(conn, mid, timeline)
            repo.mark_sample(conn, mid, "GOLD", "crawl")
        from riftwatch.stats import extract

        extract.backfill(conn)
    champion_stats.clear_cache()
    pool = ConnectionPool(TEST_DB, min_size=1, max_size=4, open=True, kwargs={"autocommit": True})
    jobs = JobQueue(pool, workers=1, cooldown_s=60)
    settings = Settings.from_env({"RIFTWATCH_DATABASE_URL": TEST_DB, "RIFTWATCH_MODELS_DIR": str(tmp_path / "models")})
    gd = GameData()
    gd.loaded = True
    app = create_app(settings, api=FakeApi(5), pool=pool, jobs=jobs, gamedata=gd)
    with TestClient(app) as client:
        with connect(TEST_DB) as conn:
            role, champ = conn.execute("SELECT position, champion_id FROM participant_builds "
                                       "WHERE position <> '' LIMIT 1").fetchone()
        yield client, role, champ
    pool.close()


def test_the_draft_page_and_its_suggestions(web):
    client, role, champ = web
    page = client.get("/draft?as=na/Me-NA1")
    assert page.status_code == 200 and "Draft assistant" in page.text and "/static/draft.js" in page.text
    assert 'value="Me#NA1"' in page.text and " onclick=" not in page.text and " onsubmit=" not in page.text
    r = client.get(f"/api/draft/suggest?role={role}")
    assert r.status_code == 200
    data = r.json()
    ids = [p["id"] for p in data["picks"]]
    assert champ in ids and data["picks"][0]["parts"][0]["key"] == "strength"
    assert all(0 < p["win"] < 100 for p in data["picks"])
    banned = client.get(f"/api/draft/suggest?role={role}&bans={champ}").json()
    assert champ not in [p["id"] for p in banned["picks"]]
    picked = client.get(f"/api/draft/suggest?role={role}&enemies={champ}:{role}").json()
    assert champ not in [p["id"] for p in picked["picks"]] and picked["lane_opponent"]
    assert client.get("/api/draft/suggest?role=SUPPORTISH").status_code == 400
    missing = client.get(f"/api/draft/suggest?role={role}&me=Nobody%23NA1").json()
    assert missing["me_missing"] and missing["me"] is None
