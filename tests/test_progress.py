import os
from datetime import UTC, date, datetime, timedelta

import pytest

from riftwatch.analysis.progress import RankPoint, fit, summarize, week_start
from tests.test_pool import scored

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
MONDAY = datetime(2026, 9, 7, 18, tzinfo=UTC)


def test_week_start_is_monday():
    assert week_start(datetime(2026, 9, 13, 23, tzinfo=UTC)) == date(2026, 9, 7)   # Sunday
    assert week_start(MONDAY) == date(2026, 9, 7)


def test_fit_slope_and_verdicts():
    rising = fit([40 + i for i in range(20)])
    assert rising.slope == pytest.approx(10.0) and rising.se == 0 and rising.games == 20
    noisy = fit([50, 30, 70, 40, 60, 35, 65, 45, 55, 50, 30, 70])
    assert noisy.verdict == "steady"
    falling = fit([60 - 0.5 * i + (1 if i % 2 else -1) for i in range(40)])
    assert falling.verdict == "declining" and falling.slope == pytest.approx(-5.0, abs=0.5)
    assert fit([1, 2]).verdict == "steady"                       # too few games to say


def test_weekly_summary_and_area_trends():
    # Three weeks, getting better: fighting/vision percentiles climb week by week.
    weekly = [((25, 25), False), ((50, 50), True), ((75, 75), True)]
    dated = []
    for w, (goodness, win) in enumerate(weekly):
        for g in range(4):
            dated.append((MONDAY + timedelta(weeks=w, hours=g), scored(19, "Warwick", "JUNGLE",
                                                                         goodness, win=win)))
    report = summarize(list(reversed(dated)), "GOLD", [])
    assert [w.start for w in report.weeks] == [date(2026, 9, 7), date(2026, 9, 14), date(2026, 9, 21)]
    assert [w.score for w in report.weeks] == [25.0, 50.0, 75.0]
    assert [w.wins for w in report.weeks] == [0, 4, 4] and report.games == 12
    assert report.weeks[0].areas == {"fighting": 25.0, "vision": 25.0}
    assert report.trend.verdict == "improving"
    assert report.area_trends["fighting"].verdict == "improving"


def test_rank_points_on_one_ladder():
    plat2 = RankPoint(MONDAY, "PLATINUM", "II", 20)
    plat3 = RankPoint(MONDAY, "PLATINUM", "III", 90)
    master = RankPoint(MONDAY, "MASTER", None, 50)
    diamond1 = RankPoint(MONDAY, "DIAMOND", "I", 99)
    assert plat3.ladder < plat2.ladder < diamond1.ladder < master.ladder
    assert plat2.label == "Platinum II 20 LP" and master.label == "Master 50 LP"


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


def test_progress_report_end_to_end(conn):
    import time

    from riftwatch.baselines import build as bl
    from riftwatch.coach import pipeline
    from riftwatch.db import repo
    from riftwatch.features import store
    from tests.fixtures import build_game

    now_ms = int(time.time() * 1000)
    for i in range(24):
        mid = f"NA1_{8000 + i}"
        puuids = ["me"] + [f"p{i}-{p}" for p in range(2, 11)]
        # One game every two days, the newest yesterday; better CS as they go.
        start = now_ms - (24 - i) * 2 * 86_400_000
        match, timeline = build_game(mid, puuids=puuids, cs_bonus=i * 0.2, start_ms=start)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, "GOLD", "crawl")
    store.extract_pending(conn)
    bl.build(conn)
    for tier, lp in (("GOLD", 10), ("GOLD", 10), ("GOLD", 55)):
        repo.insert_rank_snapshots(conn, "me", [{"queueType": "RANKED_SOLO_5x5", "tier": tier,
                                                 "rank": "I", "leaguePoints": lp,
                                                 "wins": 30, "losses": 28}])

    report = pipeline.progress_report(conn, "me", weeks=4, tier="gold")
    assert 13 <= report.games <= 15                     # games from the last four weeks
    assert len(report.weeks) in (4, 5)
    assert [r.lp for r in report.ranks] == [10, 55]     # unchanged snapshots collapse

    # The scored-game cache: a second report scores nothing new.
    calls = []
    real = pipeline._score
    pipeline._score = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        pipeline.progress_report(conn, "me", weeks=4, tier="gold")
        pipeline.champion_pool(conn, "me", tier="gold", games=10)
    finally:
        pipeline._score = real
    assert calls == []
