import os

import pytest

from riftwatch.analysis.pool import MIN_GAMES, game_score, summarize
from riftwatch.analysis.score import GameScore, Score
from riftwatch.baselines.build import Baseline
from riftwatch.features.extract import ParticipantFeatures
from riftwatch.features.metrics import GAME_METRICS

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


def B():
    return Baseline("m", None, "GOLD", "JUNGLE", 0, "16.19", 100, 50, 10, 30, 40, 50, 60, 70)


def scored(champ_id, champ, role, goodness_values, win=True, kills=5):
    """A game whose coachable stats sit at the given percentiles."""
    p = ParticipantFeatures(1, "me", 100, role, champ_id, champ, win, None,
                            metrics={"kills": kills, "deaths": 3, "assists": 6})
    gs = GameScore(p, "GOLD")
    # kill_participation (fighting) and vision_per_min (vision): higher is better, so the
    # value at quantile q of B() is q itself.
    for name, g in zip(("kill_participation", "vision_per_min"), goodness_values, strict=True):
        value = {10: 30, 25: 40, 50: 50, 75: 60, 90: 70}[g]
        gs.game[name] = Score(GAME_METRICS[name], value, B())
    gs.game["kda"] = Score(GAME_METRICS["kda"], 70, B())          # not coachable: ignored
    return gs


def test_game_score_averages_only_coachable_stats():
    assert game_score(scored(1, "A", "JUNGLE", (90, 10))) == 50


def test_summarize_groups_by_champion_and_role_and_calls_clear_gaps():
    games = ([scored(19, "Warwick", "JUNGLE", (90, 75), win=w) for w in (1, 1, 1, 0, 1, 1)]
             + [scored(234, "Viego", "JUNGLE", (25, 10), win=w) for w in (0, 0, 1, 0, 0, 0)]
             + [scored(19, "Warwick", "TOP", (50, 50))]
             + [scored(64, "Lee Sin", "JUNGLE", (50, 75)) for _ in range(MIN_GAMES - 1)])
    report = summarize(games, "GOLD")
    assert report.games == len(games)
    lines = {(l.champion, l.role): l for l in report.lines}
    ww = lines[("Warwick", "JUNGLE")]
    assert ww.games == 6 and ww.wins == 5 and ww.score == 82.5 and ww.verdict == "stronger"
    assert ww.areas == {"fighting": 90.0, "vision": 75.0}
    assert ww.best_area == "fighting" and ww.worst_area == "vision"
    assert lines[("Viego", "JUNGLE")].verdict == "weaker"
    # Under MIN_GAMES: shown, never judged. A different role is its own line.
    assert lines[("Lee Sin", "JUNGLE")].verdict == ""
    assert lines[("Warwick", "TOP")].games == 1 and lines[("Warwick", "TOP")].verdict == ""
    assert report.lines[0].games >= report.lines[-1].games            # most played first


def test_no_verdict_when_the_gap_is_within_noise():
    noisy = ([scored(19, "Warwick", "JUNGLE", g) for g in ((90, 10), (10, 90), (75, 50),
                                                           (25, 50), (90, 25), (50, 50))]
             + [scored(234, "Viego", "JUNGLE", g) for g in ((50, 25), (75, 10), (25, 90),
                                                            (10, 50), (90, 50), (50, 50))])
    assert all(l.verdict == "" for l in summarize(noisy, "GOLD").lines)


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


def test_champion_pool_end_to_end_and_narrow_loading(conn):
    from riftwatch.baselines import build as bl
    from riftwatch.coach.pipeline import champion_pool
    from riftwatch.db import repo
    from riftwatch.features import store
    from tests.fixtures import build_game

    for i in range(25):
        mid = f"NA1_{7000 + i}"
        puuids = ["me"] + [f"o{i}-{p}" for p in range(2, 11)]
        match, timeline = build_game(mid, puuids=puuids, cs_bonus=i * 0.1,
                                     start_ms=1_790_000_000_000 + i * 3_600_000)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, "GOLD", "crawl")
    store.extract_pending(conn)
    bl.build(conn)

    # The narrow load: just the player (Aatrox top, participant 1) and the lane opponent.
    ids = repo.player_match_ids(conn, "me", limit=3)
    narrow = store.load_stored_many(conn, ids, minutes=False, players_for="me")
    game = narrow[ids[0]]
    assert sorted(game.participants) == [1, 6] and game.participants[1].minutes == []

    report = champion_pool(conn, "me", tier="gold")
    assert report.games == 25 and len(report.lines) == 1
    line = report.lines[0]
    assert (line.champion, line.role, line.games) == ("Aatrox", "TOP", 25)
    assert 0 < line.score < 100 and line.score_se > 0
