import os

import pytest

from riftwatch.baselines.refresh import players_needed

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


def test_players_needed_uses_the_observed_yield():
    # First round: 5 games a player, a third lost -> ~3.3 per player, 4 divisions.
    assert players_needed(300, ["GOLD"]) == 23
    assert players_needed(300, ["MASTER", "GRANDMASTER", "CHALLENGER"]) == 30
    assert players_needed(1, ["GOLD"]) == 1
    # Later rounds: a quarter of a game per player (inactive Iron accounts) -> capped.
    assert players_needed(300, ["IRON"], games_per_player=0.25) == 100
    assert players_needed(40, ["IRON"], games_per_player=1.0) == 10


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


def add_games(conn, bucket, n, start, version="16.19.1.1"):
    from riftwatch.db import repo
    from tests.fixtures import build_game

    for i in range(start, start + n):
        mid = f"NA1_{6000 + i}"
        match, timeline = build_game(mid, cs_bonus=i * 0.05, version=version)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, bucket, "crawl")


def test_refresh_tops_up_thin_buckets_and_rebuilds(conn):
    from riftwatch.baselines.refresh import refresh
    from riftwatch.features import store

    add_games(conn, "GOLD", 25, 0)
    add_games(conn, "PLATINUM", 5, 100)
    add_games(conn, "PLATINUM", 30, 200, version="16.18.1.1")    # old patch: doesn't count
    crawled = []

    def crawl_fn(tiers, players):
        # Each round finds only 8 games from 10 players: low yield, like the low ladder.
        crawled.append((tiers, players))
        add_games(conn, "PLATINUM", 8, 300 + 10 * len(crawled))
        return 10

    report = refresh(conn, current_patch="16.19", target=25, buckets=["GOLD", "PLATINUM"],
                     crawl_fn=crawl_fn, extract_fn=lambda: store.extract_pending(conn),
                     say=lambda s: None)
    # GOLD already has 25. PLATINUM: 5 -> 13 -> 21 -> 29 over three rounds, the later
    # rounds sized from the 0.8 games-per-player actually seen.
    assert [c[0] for c in crawled] == [["PLATINUM"]] * 3
    assert crawled[1][1] == players_needed(12, ["PLATINUM"], 0.8)
    assert report.before["PLATINUM"] == 5 and report.after["PLATINUM"] == 29
    assert report.crawled["PLATINUM"] == 24
    assert report.rebuilt
    assert conn.execute("SELECT count(*) FROM baselines").fetchone()[0] > 0


def test_dry_run_crawls_nothing(conn):
    from riftwatch.baselines.refresh import refresh

    calls = []
    report = refresh(conn, current_patch="16.19", target=10, crawl_fn=lambda t, p: calls.append(t) or 0,
                     extract_fn=lambda: None, dry_run=True, say=lambda s: None)
    assert calls == [] and not report.rebuilt and report.after["IRON"] == 0
