import os
import random
import statistics

import pytest

from riftwatch.analysis.score import Score, percentile, score_participant
from riftwatch.analysis.trend import trends
from riftwatch.baselines import build as bl
from riftwatch.baselines.crawl import crawl, sample_counts
from riftwatch.features.extract import extract
from riftwatch.features.metrics import GAME_METRICS
from tests.fixtures import build_game


def B(p10, p25, p50, p75, p90, mean=None, sd=1.0, n=100):
    return bl.Baseline("m", None, "GOLD", "TOP", 0, "16.19", n,
                       p50 if mean is None else mean, sd, p10, p25, p50, p75, p90)


# -- percentile maths ---------------------------------------------------------------------

def test_percentile_interpolates_between_quantiles():
    b = B(10, 20, 30, 40, 50)
    assert percentile(30, b) == 50
    assert percentile(25, b) == pytest.approx(37.5)
    assert percentile(45, b) == pytest.approx(82.5)


def test_percentile_tails_extrapolate_and_clamp():
    b = B(10, 20, 30, 40, 50)
    assert percentile(5, b) == pytest.approx(10 - 15 * 5 / 10)
    assert percentile(-1000, b) == 1.0
    assert percentile(1000, b) == 99.0


def test_percentile_on_tied_quantiles_takes_the_plateau_middle():
    deaths = B(2, 4, 4, 4, 7)   # p25 == p50 == p75
    assert percentile(4, deaths) == 50
    assert percentile(2, deaths) == 10


def test_goodness_flips_for_lower_is_better():
    b = B(2, 3, 5, 7, 9)
    deaths = Score(GAME_METRICS["deaths"], 3, b)
    cs = Score(GAME_METRICS["cs_per_min"], 3, b)
    assert cs.goodness == 25
    assert deaths.goodness == 75


def test_neighbor_buckets():
    assert bl.neighbor_buckets("PLATINUM")[:3] == ["PLATINUM", "GOLD", "EMERALD"]
    assert bl.neighbor_buckets("IRON")[:2] == ["IRON", "BRONZE"]


# -- building from Postgres ------------------------------------------------------------------

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


def load_games(conn, n, bucket="GOLD", source="crawl", start=0, version="16.19.1.1"):
    from riftwatch.db import repo
    from riftwatch.features import store

    games = []
    for i in range(start, start + n):
        mid = f"NA1_{1000 + i}"
        match, timeline = build_game(mid, cs_bonus=i * 0.1, version=version)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, bucket, source)
        games.append(extract(match, timeline))
    store.extract_pending(conn)
    return games


def stored(conn, **where):
    clause = " AND ".join(f"{k} IS NULL" if v is None else f"{k} = %({k})s" for k, v in where.items())
    return conn.execute(f"SELECT n, p10, p50, p90 FROM baselines WHERE {clause}", where).fetchone()


def test_build_matches_python_quantiles(conn):
    games = load_games(conn, 25)
    report = bl.build(conn, min_n=20)
    assert report.games == 25 and report.patch_window == "16.19"

    values = [g.participants[1].metrics["cs_at_10"] for g in games]   # Aatrox top
    n, p10, p50, p90 = stored(conn, tier_bucket="GOLD", role="TOP", champion_id=266,
                              metric="cs_at_10", minute=None)
    q = statistics.quantiles(values, n=10, method="inclusive")   # same as percentile_cont
    assert n == 25
    assert p50 == pytest.approx(statistics.median(values))
    assert p10 == pytest.approx(q[0]) and p90 == pytest.approx(q[8])

    # Role-level rows pool both teams' top laners.
    assert stored(conn, tier_bucket="GOLD", role="TOP", champion_id=0,
                  metric="cs_at_10", minute=None)[0] == 50
    # Curves get one row per minute.
    assert stored(conn, tier_bucket="GOLD", role="TOP", champion_id=0, metric="cs", minute=10)[0] == 50


def test_build_respects_min_n_and_excludes_player_games(conn):
    load_games(conn, 15)
    load_games(conn, 10, source="player", start=15)
    assert bl.build(conn, min_n=20).rows == 0 or stored(
        conn, tier_bucket="GOLD", role="TOP", champion_id=266, metric="cs_at_10", minute=None) is None
    # Role level: 2 tops x 15 crawled games = 30 >= 20, player games still excluded.
    assert stored(conn, tier_bucket="GOLD", role="TOP", champion_id=0,
                  metric="cs_at_10", minute=None)[0] == 30
    bl.build(conn, min_n=20, include_player_games=True)
    assert stored(conn, tier_bucket="GOLD", role="TOP", champion_id=266,
                  metric="cs_at_10", minute=None)[0] == 25


def test_patch_window_takes_newest_patches(conn):
    load_games(conn, 21, version="16.17.1.1")
    load_games(conn, 21, start=21, version="16.19.1.1")
    load_games(conn, 21, start=42, version="16.9.1.1")   # numerically older than 16.17
    report = bl.build(conn, patch_count=2)
    assert report.patches == ["16.19", "16.17"] and report.patch_window == "16.17-16.19"
    assert report.games == 42


def test_baseline_set_fallbacks(conn):
    load_games(conn, 25)
    bl.build(conn, min_n=20)
    # Champion-level exists for Aatrox (266) in GOLD.
    gold_aatrox = bl.BaselineSet(conn, "GOLD", "TOP", 266)
    assert gold_aatrox.get("cs_at_10").champion_id == 266
    # Unknown champion -> role level.
    assert bl.BaselineSet(conn, "GOLD", "TOP", 999).get("cs_at_10").champion_id == 0
    # No PLATINUM data -> nearest tier (GOLD).
    plat = bl.BaselineSet(conn, "PLATINUM", "TOP", 266)
    assert plat.get("cs_at_10").tier_bucket == "GOLD"
    # Too far away -> nothing.
    assert bl.BaselineSet(conn, "DIAMOND", "TOP", 266).get("cs_at_10") is None


def test_score_and_trend_end_to_end(conn):
    games = load_games(conn, 25)
    bl.build(conn, min_n=20)
    baselines = bl.BaselineSet(conn, "GOLD", "TOP", 266)
    best = score_participant(games[-1].participants[1], baselines)    # most CS
    worst = score_participant(games[0].participants[1], baselines)
    assert best.game["cs_at_10"].goodness > 85
    assert worst.game["cs_at_10"].goodness < 15
    assert best.curve_at("cs", 10).value == best.game["cs_at_10"].value
    assert "cs" in best.curves and len(best.curves["cs"]) == 26   # minutes 1..26

    scored = [score_participant(g.participants[1], baselines) for g in reversed(games)]
    t = {tr.metric.name: tr for tr in trends(scored, recent=10)}
    assert t["cs_at_10"].median_goodness > t["cs_at_10"].older_goodness
    assert t["cs_at_10"].change > 0


# -- crawler ----------------------------------------------------------------------------------

class CrawlApi:
    def __init__(self):
        self.ladder_calls = []
        self.games = {}

    def league_entries(self, platform, tier, division, queue="RANKED_SOLO_5x5", page=1):
        self.ladder_calls.append((tier, division))
        return [{"puuid": f"{tier}-{division}-{i}", "tier": tier, "rank": division}
                for i in range(5)]

    def match_ids(self, platform, puuid, *, count=20, queue=None, **_):
        ids = [f"NA1_{abs(hash((puuid, k))) % 10**9}" for k in range(count)]
        for mid in ids:
            self.games.setdefault(mid, build_game(mid))
        return ids

    def match(self, mid):
        return self.games[mid][0]

    def timeline(self, mid):
        return self.games[mid][1]


def test_crawl_samples_each_division_and_tags_buckets(conn):
    from riftwatch.ingest import Ingestor

    api = CrawlApi()
    ing = Ingestor(conn, api)
    report = crawl(conn, ing, "na", tiers=["GOLD", "GRANDMASTER"], players_per_division=2,
                   matches_per_player=3, rng=random.Random(1))
    assert api.ladder_calls == [("GOLD", d) for d in ("I", "II", "III", "IV")] + [("GRANDMASTER", "I")]
    assert report.players == 10
    assert set(sample_counts(conn)) == {"GOLD", "MASTER_PLUS"}

    # A rerun picks players not sampled before.
    crawl(conn, ing, "na", tiers=["GOLD"], players_per_division=2, matches_per_player=1,
          rng=random.Random(1))
    assert conn.execute("SELECT count(*) FROM crawl_players").fetchone()[0] == 18


def test_crawl_rejects_unknown_tier(conn):
    from riftwatch.ingest import Ingestor

    with pytest.raises(ValueError):
        crawl(conn, Ingestor(conn, CrawlApi()), "na", tiers=["WOOD"])
