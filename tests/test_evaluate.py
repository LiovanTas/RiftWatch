import os

import pytest

from riftwatch.coach.evaluate import EvalReport, GameEval, score_output
from riftwatch.coach.evidence import Evidence, EvidenceSet
from riftwatch.coach.grounding import CoachOutput, CoachPoint

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")

EV = EvidenceSet([
    Evidence("E1", "context", "context", "neutral", "Game NA1_1: Ahri middle, loss."),
    Evidence("E2", "metric", "farming", "weakness", "CS at 10: 52, median 64.", severity=30),
    Evidence("E3", "metric", "vision", "weakness", "wards per minute: 0.2.", severity=20),
    Evidence("E4", "metric", "fighting", "strength", "kill participation: 70%.", severity=25),
    Evidence("E5", "death", "survival", "weakness", "Died at 7:10 in the river.", severity=10),
])


def pt(kind, area, ids, advice="Last-hit under tower."):
    return CoachPoint(area=area, kind=kind, title="t", explanation="e", advice=advice,
                      evidence_ids=ids)


def test_score_output_flags_what_the_validator_does_not():
    out = CoachOutput(headline="h", points=[
        pt("weakness", "farming", ["E1", "E2"]),                 # fine
        pt("weakness", "fighting", ["E4"]),                      # calls a strength a weakness
        pt("weakness", "economy", ["E3"]),                       # filed under the wrong area
        pt("weakness", "farming", ["E2"], advice="Get 70 CS."),  # numbers in advice; reuses E2
        pt("strength", "macro", ["E1"]),                         # cites only the summary
    ])
    s = score_output(out, EV)
    assert s["kind_mismatches"] == 1 and s["area_mismatches"] == 1
    assert s["advice_with_numbers"] == 1 and s["shared_evidence"] == 1
    assert s["context_only"] == 1
    # Top three weaknesses by severity: E2, E3, E5 -- two of them cited.
    assert s["top_weaknesses_addressed"] == pytest.approx(2 / 3, abs=1e-3)


def test_summary_rates():
    def g(**kw):
        base = dict(match_id="m", tier_bucket="GOLD", role="TOP", champion="Aatrox",
                    evidence_items=10, attempts=1, first_pass=True, first_violations=0, points=4,
                    dropped=0, kind_mismatches=0, area_mismatches=0, advice_with_numbers=0,
                    context_only=0, shared_evidence=0, top_weaknesses_addressed=1.0,
                    cost_usd=0.01, seconds=8.0)
        return GameEval(**(base | kw))

    r = EvalReport("m", [g(), g(attempts=2, first_pass=False, first_violations=1, dropped=1,
                           points=3, cost_usd=0.02, seconds=12.0),
                         g(error="boom", points=0)])
    s = r.summary()
    assert s["games"] == 2 and s["errors"] == 1
    assert s["first_pass_rate"] == 0.5 and s["retry_rate"] == 0.5
    assert s["drop_rate"] == pytest.approx(1 / 8, abs=1e-3)
    assert s["cost_usd_total"] == 0.03 and s["seconds_p50"] == 10.0


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


def test_offline_eval_end_to_end_writes_nothing_to_the_cache(conn):
    from riftwatch.baselines import build as bl
    from riftwatch.coach.evaluate import OfflineCoach, evaluate, sample_games
    from riftwatch.db import repo
    from riftwatch.features import store
    from tests.fixtures import build_game

    for i in range(25):
        mid = f"NA1_{4000 + i}"
        match, timeline = build_game(mid, cs_bonus=i * 0.1)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, "GOLD", "crawl")
    store.extract_pending(conn)
    bl.build(conn)

    games = sample_games(conn, 6, seed=7)
    assert games == sample_games(conn, 6, seed=7)                  # reproducible
    assert len({g[0] for g in games}) == 6                         # one player per match
    assert {g[3] for g in games} == {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}
    report = evaluate(conn, OfflineCoach(), games)
    s = report.summary()
    assert s["games"] == 6 and s["errors"] == 0
    assert s["first_pass_rate"] == 1.0          # the template coach quotes its evidence
    assert s["kind_mismatch_rate"] == 0.0 and s["advice_with_numbers_rate"] == 0.0
    assert conn.execute("SELECT count(*) FROM coach_reports").fetchone()[0] == 0
