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


def test_patch_window_ignores_player_only_patches(conn):
    load_games(conn, 21, version="16.18.1.1")
    load_games(conn, 5, start=21, source="player", version="16.19.1.1")
    assert bl.build(conn).patches == ["16.18"]


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
    # Aatrox (266) has 25 GOLD games: real, but under CHAMPION_MIN_N, so the role baseline
    # (50 players) is the better yardstick...
    gold_aatrox = bl.BaselineSet(conn, "GOLD", "TOP", 266)
    assert gold_aatrox.get("cs_at_10").champion_id == 0
    # ...until the champion threshold is met.
    bl.CHAMPION_MIN_N, saved = 20, bl.CHAMPION_MIN_N
    try:
        assert bl.BaselineSet(conn, "GOLD", "TOP", 266).get("cs_at_10").champion_id == 266
    finally:
        bl.CHAMPION_MIN_N = saved
    # Unknown champion -> role level.
    assert bl.BaselineSet(conn, "GOLD", "TOP", 999).get("cs_at_10").champion_id == 0
    # No PLATINUM data -> nearest tier (GOLD).
    plat = bl.BaselineSet(conn, "PLATINUM", "TOP", 266)
    assert plat.get("cs_at_10").tier_bucket == "GOLD"
    # Too far away -> nothing.
    assert bl.BaselineSet(conn, "DIAMOND", "TOP", 266).get("cs_at_10") is None


def row(tier, champ, p50, n=100, sd=10.0, metric="cs_per_min", minute=None):
    return bl.Baseline(metric, minute, tier, "JUNGLE", champ, "16.19", n, p50, sd,
                       p50 - 12, p50 - 6, p50, p50 + 6, p50 + 12)


def test_champion_effect_pools_tiers_and_gates_noise():
    rows = [row("GOLD", 0, 50), row("GOLD", 64, 55, n=30),            # +0.5 sd on 30 games
            row("DIAMOND", 0, 60), row("DIAMOND", 64, 65, n=90),      # +0.5 sd on 90 games
            row("GOLD", 0, 5, metric="deaths"), row("GOLD", 64, 5.2, n=120, metric="deaths")]
    effects = bl.champion_effects(rows, 64)
    effect, n = effects[("cs_per_min", None)]
    assert n == 120 and effect == pytest.approx(0.5 * 120 / (120 + bl.ADJUST_SHRINK))
    # +0.02 sd is well inside the noise of 120 games: no adjustment.
    assert ("deaths", None) not in effects
    # Too few pooled games: nothing.
    assert bl.champion_effects([row("GOLD", 0, 50), row("GOLD", 64, 80, n=40)], 64) == {}


def test_adjusted_baseline_shifts_the_role_at_the_players_tier(conn):
    rows = [row("GOLD", 0, 50, n=400), row("GOLD", 64, 56, n=30),
            row("MASTER_PLUS", 0, 70, n=900), row("MASTER_PLUS", 64, 76, n=170)]
    with conn.cursor() as cur:
        for b in rows:
            cur.execute(
                """INSERT INTO baselines (tier_bucket, role, champion_id, patch_window, metric,
                       minute, n, mean, sd, p10, p25, p50, p75, p90)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (b.tier_bucket, b.role, b.champion_id, b.patch_window, b.metric, b.minute,
                 b.n, b.mean, b.sd, b.p10, b.p25, b.p50, b.p75, b.p90))
    got = bl.BaselineSet(conn, "GOLD", "JUNGLE", 64).get("cs_per_min")
    shift = 0.6 * 200 / (200 + bl.ADJUST_SHRINK) * 10
    assert got.tier_bucket == "GOLD" and got.n == 400 and got.adjusted_n == 200
    assert got.p50 == pytest.approx(50 + shift) and got.p90 == pytest.approx(62 + shift)
    assert got.scope == "Gold jungle, adjusted for champion"
    # Another champion, or the role itself, gets the plain role baseline.
    assert bl.BaselineSet(conn, "GOLD", "JUNGLE", 0).get("cs_per_min").p50 == 50
    plain = bl.BaselineSet(conn, "GOLD", "JUNGLE", 11).get("cs_per_min")
    assert plain.adjusted_n == 0 and plain.scope == "Gold jungle, same role"
    # Enough games of its own at the tier: the champion's real baseline wins.
    assert bl.BaselineSet(conn, "MASTER_PLUS", "JUNGLE", 64).get("cs_per_min").adjusted_n
    bl.CHAMPION_MIN_N, saved = 150, bl.CHAMPION_MIN_N
    try:
        own = bl.BaselineSet(conn, "MASTER_PLUS", "JUNGLE", 64).get("cs_per_min")
        assert own.champion_id == 64 and own.adjusted_n == 0 and own.p50 == 76
    finally:
        bl.CHAMPION_MIN_N = saved


def test_score_and_trend_end_to_end(conn):
    games = load_games(conn, 25)
    bl.build(conn, min_n=20)
    baselines = bl.BaselineSet(conn, "GOLD", "TOP", 266)
    best = score_participant(games[-1].participants[1], baselines)    # most CS
    worst = score_participant(games[0].participants[1], baselines)
    # Role-level baseline (both teams' top laners): the best game far above the worst.
    assert best.game["cs_at_10"].goodness > 85
    assert worst.game["cs_at_10"].goodness < 30 < best.game["cs_at_10"].goodness - 50
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
        self.start_times = []
        self.games = {}

    def league_entries(self, platform, tier, division, queue="RANKED_SOLO_5x5", page=1):
        self.ladder_calls.append((tier, division))
        return [{"puuid": f"{tier}-{division}-{i}", "tier": tier, "rank": division}
                for i in range(5)]

    def match_ids(self, platform, puuid, *, count=20, queue=None, start_time=None, **_):
        self.start_times.append(start_time)
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
                   matches_per_player=3, rng=random.Random(1), now=1_800_000_000)
    assert api.ladder_calls == [("GOLD", d) for d in ("I", "II", "III", "IV")] + [("GRANDMASTER", "I")]
    assert report.players == 10
    assert set(api.start_times) == {1_800_000_000 - 14 * 86_400}   # recent games only
    assert set(sample_counts(conn)) == {"GOLD", "MASTER_PLUS"}

    # A rerun picks players not sampled before.
    crawl(conn, ing, "na", tiers=["GOLD"], players_per_division=2, matches_per_player=1,
          rng=random.Random(1))
    assert conn.execute("SELECT count(*) FROM crawl_players").fetchone()[0] == 18


def test_crawl_rejects_unknown_tier(conn):
    from riftwatch.ingest import Ingestor

    with pytest.raises(ValueError):
        crawl(conn, Ingestor(conn, CrawlApi()), "na", tiers=["WOOD"])


class PagedApi(CrawlApi):
    """A ladder with distinct players on each page, ending after `pages` pages."""

    def __init__(self, per_page=5, pages=3):
        super().__init__()
        self.per_page, self.pages = per_page, pages

    def league_entries(self, platform, tier, division, queue="RANKED_SOLO_5x5", page=1):
        self.ladder_calls.append((tier, division, page))
        if page > self.pages:
            return []
        return [{"puuid": f"{platform}-{tier}-{page}-{i}"} for i in range(self.per_page)]


def test_crawl_pages_the_ladder_and_records_exact_tier(conn):
    from riftwatch.baselines import crawl as crawl_mod
    from riftwatch.ingest import Ingestor

    api = PagedApi(per_page=5, pages=3)
    report = crawl(conn, Ingestor(conn, api), "kr", tiers=["CHALLENGER"],
                   players_per_division=12, matches_per_player=2, rng=random.Random(3))
    # Wants 24 candidates, so it reads every page (15 players) and samples 12 of them.
    assert [c[2] for c in api.ladder_calls] == [1, 2, 3, 4]
    assert report.players == 12
    tiers = {r[0] for r in conn.execute("SELECT DISTINCT tier FROM crawl_player_matches")}
    assert tiers == {"CHALLENGER"}
    assert crawl_mod.high_elo_counts(conn)["CHALLENGER"] == report.matches_new + report.matches_cached
    assert sample_counts(conn) == {"MASTER_PLUS": report.matches_new + report.matches_cached}


def test_crawl_saves_each_chunk_before_the_next(conn):
    from riftwatch.ingest import Ingestor

    api = PagedApi(per_page=25, pages=1)
    calls = {"n": 0}

    def stop_after_first_chunk():
        calls["n"] += 1
        return calls["n"] > 1

    crawl(conn, Ingestor(conn, api), "na", tiers=["GRANDMASTER"], players_per_division=25,
          matches_per_player=1, rng=random.Random(1), should_stop=stop_after_first_chunk)
    saved = conn.execute("SELECT count(*) FROM crawl_players").fetchone()[0]
    assert saved == 10       # one chunk, kept even though the crawl stopped


def test_crawl_regions_isolates_failures(conn):
    from riftwatch.baselines.crawl import crawl_regions
    from riftwatch.db.connection import connect

    class Flaky(PagedApi):
        def league_entries(self, platform, *a, **k):
            if platform == "euw1":
                raise RuntimeError("euw is down")
            return super().league_entries(platform, *a, **k)

    reports = crawl_regions(TEST_DB, Flaky(per_page=3, pages=1), ["na", "euw"], connect=connect,
                            tiers=["CHALLENGER"], players_per_division=2, matches_per_player=1)
    by = {r.platform: r for r in reports}
    assert by["na1"].error is None and by["na1"].players == 2
    assert by["euw1"].error == "RuntimeError: euw is down"
