import copy
import os

import pytest

from riftwatch.features import store
from riftwatch.features.extract import EXTRACTOR_VERSION, extract
from riftwatch.features.map import readable, zone
from riftwatch.features.metrics import ALL_METRICS, AREAS, CURVE_METRICS
from tests.fixtures import build_game

MATCH, TIMELINE = build_game()
GAME = extract(MATCH, TIMELINE)
P = GAME.participants


# -- map ------------------------------------------------------------------------------------

@pytest.mark.parametrize("x, y, expected", [
    (400, 400, "blue_base"), (14400, 14500, "red_base"),
    (1500, 10500, "top_lane"), (4000, 14000, "top_lane"),
    (12000, 2000, "bot_lane"), (14000, 6000, "bot_lane"),
    (7400, 7400, "mid_lane"),
    (4900, 10000, "river"), (9800, 4400, "river"),
    (3800, 7900, "blue_jungle"), (11000, 7000, "red_jungle"),
])
def test_zones(x, y, expected):
    assert zone(x, y) == expected


def test_readable_zone_is_relative_to_team():
    assert readable("red_jungle", 100) == "enemy jungle"
    assert readable("red_jungle", 200) == "own jungle"
    assert readable("blue_base", 100) == "own base"
    assert readable("mid_lane", 200) == "mid lane"


# -- per-minute rows ------------------------------------------------------------------------

def test_one_row_per_minute_and_final_partial_frame_dropped():
    # 26:34 game -> minutes 0..26; the 26:34 end-of-game frame is not a minute.
    assert [r.minute for r in P[1].minutes] == list(range(27))
    assert GAME.duration_s == 26 * 60 + 34
    assert GAME.patch == "16.19"


def test_snapshot_values_at_ten():
    row = P[1].at(10)
    assert (row.gold, row.cs, row.lane_cs, row.jungle_cs) == (4300, 70, 70, 0)
    jungler = P[2].at(10)
    assert (jungler.cs, jungler.lane_cs, jungler.jungle_cs) == (40, 0, 40)


def test_lane_opponent_diffs():
    assert P[1].opponent_id == 6 and P[5].opponent_id == 10
    row = P[1].at(10)
    assert (row.gold_diff, row.cs_diff) == (4300 - 4100, 70 - 60)
    assert P[6].at(10).gold_diff == -200


def test_event_counters_are_cumulative_by_minute():
    # Lee Sin's kill at 3:05 shows from the minute-4 snapshot on.
    assert P[2].at(3).kills == 0 and P[2].at(4).kills == 1
    assert P[1].at(4).assists == 1
    assert P[5].at(1).wards_placed == 0
    assert P[5].at(2).wards_placed == 1
    assert P[5].at(26).wards_placed == 2
    assert P[10].at(26).wards_killed == 1


def test_final_frame_just_before_a_minute_mark_is_not_that_minute():
    # Real case: a 21:58 game's end frame sits 2 s before 22:00. It must not become minute 22.
    match, timeline = build_game("NA1_9", minutes=21, extra_seconds=58)
    game = extract(match, timeline)
    assert len(game.participants[1].minutes) == 22  # minutes 0..21


def test_missing_frame_is_filled_from_previous_minute():
    tl = copy.deepcopy(TIMELINE)
    del tl["info"]["frames"][5]
    game = extract(MATCH, tl)
    assert len(game.participants[1].minutes) == 27
    assert game.participants[1].at(5).gold == game.participants[1].at(4).gold


def test_no_opponent_when_role_unknown():
    match = copy.deepcopy(MATCH)
    match["info"]["participants"][0]["teamPosition"] = ""
    game = extract(match, TIMELINE)
    assert game.participants[1].opponent_id is None
    assert game.participants[6].opponent_id is None
    assert game.participants[1].at(10).gold_diff is None
    assert "gold_diff_at_10" not in game.participants[1].metrics


# -- deaths ---------------------------------------------------------------------------------

def test_death_details():
    top_death = P[6].deaths[0]
    assert top_death.zone == "top_lane"
    assert top_death.killer_participant_id == 2 and top_death.assisters == 1
    assert top_death.early

    first, second = P[3].deaths
    assert first.zone == "mid_lane" and first.gold_diff == 60 * 6 and not first.ahead
    # Ahri (+60 gold/min on her opponent) is 1260 ahead at 21 min and dies in her own base.
    assert second.where == "own base"
    assert second.gold_diff == 60 * 21 and second.ahead and not second.early


# -- game metrics ---------------------------------------------------------------------------

def test_core_metrics():
    m = P[1].metrics
    assert m["kills"] == 1 and m["assists"] == 1
    assert m["kill_participation"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["cs_per_min"] == pytest.approx(185 / (1594 / 60), abs=1e-3)
    assert m["cs_at_10"] == 70
    assert m["gold_diff_at_15"] == 20 * 15
    assert "cs_at_20" in m


def test_solo_kills_and_deaths():
    assert P[8].metrics["solo_kills"] == 1      # mid solo kill
    assert P[2].metrics["solo_kills"] == 0      # had an assist
    assert P[4].metrics["solo_deaths"] == 1


def test_objective_and_tower_participation():
    assert P[2].metrics["objective_participation"] == 1.0   # dragon + baron
    assert P[1].metrics["objective_participation"] == 0.5   # baron only
    assert P[7].metrics["objective_participation"] == 1.0   # red's only epic: grubs
    assert P[6].metrics["objective_participation"] == 0.0
    assert P[3].metrics["tower_participation"] == 1.0
    assert "tower_participation" not in P[6].metrics       # red took no towers
    assert P[4].metrics["turret_plates"] == 1


def test_survival_metrics():
    assert P[3].metrics["early_deaths"] == 1
    assert P[3].metrics["deaths_while_ahead"] == 1
    assert P[3].metrics["first_death_min"] == pytest.approx(410 / 60, abs=0.01)


def test_every_extracted_metric_is_catalogued():
    for p in P.values():
        assert set(p.metrics) <= set(ALL_METRICS), set(p.metrics) - set(ALL_METRICS)


def test_catalogue_is_consistent():
    assert {m.area for m in ALL_METRICS.values()} <= set(AREAS)
    row_fields = set(P[1].minutes[0].__dataclass_fields__)
    assert set(CURVE_METRICS) <= row_fields


# -- storage --------------------------------------------------------------------------------

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


@pytest.fixture
def conn():
    if not TEST_DB:
        pytest.skip("RIFTWATCH_TEST_DATABASE_URL not set")
    from riftwatch.db import migrate
    from riftwatch.db.connection import connect

    with connect(TEST_DB) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(c)
        yield c


def test_extract_pending_saves_and_is_idempotent(conn):
    from riftwatch.db import repo

    for i in range(3):
        match, timeline = build_game(f"NA1_{700 + i}")
        repo.insert_match(conn, match)
        if i < 2:
            repo.insert_timeline(conn, f"NA1_{700 + i}", timeline)

    assert store.pending_match_ids(conn) == ["NA1_700", "NA1_701"]  # no timeline, no features
    done, failures = store.extract_pending(conn)
    assert (done, failures) == (2, [])
    assert store.pending_match_ids(conn) == []

    rows = conn.execute(
        "SELECT count(*) FROM participant_minute_features WHERE match_id = 'NA1_700'"
    ).fetchone()[0]
    assert rows == 10 * 27
    metrics, version = conn.execute(
        "SELECT metrics, extractor_version FROM participant_game_summary "
        "WHERE match_id = 'NA1_700' AND participant_id = 1"
    ).fetchone()
    assert metrics["cs_at_10"] == 70 and version == EXTRACTOR_VERSION

    # Re-saving replaces rather than duplicates.
    store.save(conn, store.load(conn, "NA1_700"))
    assert conn.execute(
        "SELECT count(*) FROM participant_game_summary WHERE match_id = 'NA1_700'"
    ).fetchone()[0] == 10


def test_bad_match_is_reported_not_fatal(conn):
    from riftwatch.db import repo

    match, timeline = build_game("NA1_800")
    repo.insert_match(conn, match)
    del timeline["info"]["frames"]
    repo.insert_timeline(conn, "NA1_800", timeline)
    done, failures = store.extract_pending(conn)
    assert done == 0 and failures[0][0] == "NA1_800"
