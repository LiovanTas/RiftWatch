import json
import os
from types import SimpleNamespace

import pytest

from riftwatch.analysis.score import GameScore, Score
from riftwatch.baselines.build import Baseline
from riftwatch.coach import grounding
from riftwatch.coach.evidence import Evidence, EvidenceSet, clock, game_evidence, ordinal
from riftwatch.coach.grounding import CoachOutput, CoachPoint, numbers, validate
from riftwatch.coach.llm import Coach
from riftwatch.coach.pipeline import offline_coach
from riftwatch.features.extract import extract
from riftwatch.features.metrics import CURVE_METRICS, GAME_METRICS
from tests.fixtures import build_game

EV = EvidenceSet([
    Evidence("E1", "context", "context", "neutral", "Game NA1_1: Jhin bottom, loss, length 26:34, patch 16.19."),
    Evidence("E2", "metric", "farming", "weakness",
             "CS at 10 min: 52, which is the 18th percentile of Platinum bottom, same role (n=240); "
             "median 64, 12 below it."),
    Evidence("E3", "metric", "vision", "strength",
             "vision score per minute: 1.84, which is the 81st percentile of Platinum bottom (n=240); "
             "median 1.21, 0.63 above it."),
    Evidence("E4", "death", "survival", "weakness",
             "Death at 13:20 in the river, killed by Lee Sin with 2 helpers, while 650 gold ahead "
             "of the lane opponent."),
])


def point(**kw):
    base = dict(area="farming", kind="weakness", title="CS", explanation="", advice="",
                evidence_ids=["E2"])
    return CoachPoint(**{**base, **kw})


# -- number extraction ---------------------------------------------------------------------

def test_numbers_extraction():
    vals = [(v, d) for v, d, _ in numbers("62nd percentile, +340 gold, 13:20, 45%, 1,250 and 6.94")]
    assert vals == [(62, 0), (340, 0), (13, 0), (20, 0), (45, 0), (1250, 0), (6.94, 2)]


def test_ids_and_matchup_shorthand_are_not_numbers():
    assert numbers("as E12 shows, a 2v2 went badly") == []


# -- validation ----------------------------------------------------------------------------

def test_grounded_point_passes():
    out = CoachOutput(headline="Farming held you back.", points=[
        point(explanation="You had 52 CS at 10, the 18th percentile; the median is 64.")])
    assert validate(out, EV) == []


def test_rounding_to_coarser_precision_is_allowed():
    out = CoachOutput(headline="x", points=[point(
        area="vision", kind="strength", evidence_ids=["E3"],
        explanation="About 1.8 vision per minute, roughly 2 a minute.")])
    assert validate(out, EV) == []


def test_invented_number_is_caught():
    out = CoachOutput(headline="x", points=[point(explanation="You were 15 CS behind.")])
    problems = validate(out, EV)
    assert len(problems) == 1 and "'15'" in str(problems[0])


def test_number_from_uncited_evidence_is_caught():
    # 650 is real, but it's in E4 and the point only cites E2.
    out = CoachOutput(headline="x", points=[point(explanation="You died 650 gold ahead.")])
    assert "'650'" in str(validate(out, EV)[0])


def test_missing_and_unknown_ids():
    out = CoachOutput(headline="x", points=[point(evidence_ids=[]), point(evidence_ids=["E9"])])
    problems = [str(p) for p in validate(out, EV)]
    assert problems == ["point 1: cites no evidence", "point 2: cites unknown evidence E9"]


def test_headline_numbers_checked_against_all_evidence():
    assert validate(CoachOutput(headline="Your 52 CS at 10 hurt", points=[]), EV) == []
    assert validate(CoachOutput(headline="Your 99 CS", points=[]), EV)


def test_without_violations_drops_bad_points():
    out = CoachOutput(headline="Your 99 CS", points=[point(), point(explanation="7 deaths")])
    clean, dropped = grounding.without_violations(out, validate(out, EV))
    assert len(clean.points) == 1 and len(dropped) == 1
    assert "99" not in clean.headline


# -- evidence building ----------------------------------------------------------------------

def B(p10, p25, p50, p75, p90, n=200):
    return Baseline("m", None, "PLATINUM", "TOP", 0, "16.17-16.19", n, p50, 10, p10, p25, p50, p75, p90)


def test_helpers():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 100)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "100th"]
    assert clock(13.3333) == "13:20"


def make_score():
    match, timeline = build_game()
    game = extract(match, timeline)
    p = game.participants[3]   # Ahri mid, dies twice
    gs = GameScore(p, "PLATINUM")
    gs.game["cs_at_10"] = Score(GAME_METRICS["cs_at_10"], 80, B(60, 70, 80, 90, 100))       # 50th: neutral
    gs.game["deaths"] = Score(GAME_METRICS["deaths"], 2, B(2, 3, 5, 7, 9))                  # few deaths: strength
    gs.game["kill_participation"] = Score(GAME_METRICS["kill_participation"], 0.2,
                                          B(0.3, 0.4, 0.5, 0.6, 0.7))                        # weak
    gs.curves["cs"] = [Score(CURVE_METRICS["cs"], 1, B(5, 6, 7, 8, 9), minute=m) for m in range(1, 7)]
    return game, gs


def test_game_evidence_contents():
    game, gs = make_score()
    ev = game_evidence(game, gs)
    kinds = [e.kind for e in ev.items]
    assert kinds[0] == "context" and "Ahri" in ev.items[0].text
    metrics = {e.data.get("metric"): e for e in ev.items if e.kind == "metric"}
    assert set(metrics) == {"deaths", "kill_participation"}       # 50th-percentile CS left out
    assert metrics["kill_participation"].polarity == "weakness"
    assert "20%" in metrics["kill_participation"].text
    curve = next(e for e in ev.items if e.kind == "curve")
    assert curve.data["start"] == 1 and curve.data["end"] == 6
    deaths = [e for e in ev.items if e.kind == "death"]
    assert len(deaths) == 2 and "own base" in deaths[1].text and "ahead" in deaths[1].text
    assert [e.id for e in ev.items] == [f"E{i}" for i in range(1, len(ev.items) + 1)]


def test_fingerprint_tracks_content():
    game, gs = make_score()
    a, b = game_evidence(game, gs), game_evidence(game, gs)
    assert a.fingerprint() == b.fingerprint()
    b.items[1].text += " changed"
    assert a.fingerprint() != b.fingerprint()


def test_offline_coach_is_grounded_by_construction():
    game, gs = make_score()
    ev = game_evidence(game, gs)
    out = offline_coach(ev)
    assert out.points and out.points[0].kind == "weakness"
    assert validate(out, ev) == []


# -- LLM loop with a scripted client ----------------------------------------------------------

class FakeMessages:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        return SimpleNamespace(
            stop_reason="end_turn", parsed_output=out, model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=out.model_dump_json())],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )

    def stream(self, **kwargs):
        """Same scripted answer as parse(), delivered in 17-character chunks."""
        resp = self.parse(**kwargs)
        text = resp.content[0].text

        class _Stream:
            text_stream = (text[i:i + 17] for i in range(0, len(text), 17))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return resp

        return _Stream()


def fake_coach(*outputs):
    messages = FakeMessages(outputs)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return Coach("claude-sonnet-5-5", client=client), messages


GOOD = CoachOutput(headline="CS at 10 was low.", points=[
    point(explanation="52 CS at 10 against a median of 64.")])
BAD = CoachOutput(headline="CS at 10 was low.", points=[
    point(explanation="You lost 30 CS to your opponent.")])


def test_clean_answer_needs_one_call():
    coach, messages = fake_coach(GOOD)
    run = coach.write(EV, "this game")
    assert run.attempts == 1 and run.output == GOOD and run.dropped == []
    assert run.usage["cost_usd"] == pytest.approx((100 * 2 + 50 * 10) / 1e6)
    call = messages.calls[0]
    assert call["fallbacks"] == "default" and call["output_format"] is CoachOutput
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in call   # no per-game breakpoint; see llm.py
    assert call["thinking"] == {"type": "adaptive"} and call["output_config"] == {"effort": "low"}
    assert "[E2]" in call["messages"][0]["content"]


def test_violation_triggers_one_retry_with_feedback():
    coach, messages = fake_coach(BAD, GOOD)
    run = coach.write(EV, "this game")
    assert run.attempts == 2 and run.output == GOOD
    assert len(run.first_violations) == 1 and run.final_violations == []
    retry = messages.calls[1]["messages"]
    assert retry[1]["role"] == "assistant"
    assert "'30'" in retry[2]["content"]
    assert run.usage["input_tokens"] == 200


def test_persistent_violation_is_dropped():
    coach, _ = fake_coach(BAD, BAD)
    run = coach.write(EV, "this game")
    assert run.output.points == [] and len(run.dropped) == 1


def test_point_cap_is_enforced():
    many = CoachOutput(headline="x", points=[point(explanation="52 CS") for _ in range(7)])
    coach, _ = fake_coach(many)
    assert len(coach.write(EV, "this game").output.points) == 5


def test_between_tools_only_on_sonnet_5_5():
    assert Coach("claude-opus-5-5", thinking="between_tools", client=object()).thinking == "adaptive"
    assert Coach("claude-sonnet-5-5", thinking="between_tools", client=object()).thinking == "between_tools"


def test_refusal_is_an_error():
    coach, messages = fake_coach(GOOD)
    messages.parse = lambda **kw: SimpleNamespace(stop_reason="refusal", parsed_output=None)
    from riftwatch.coach.llm import CoachError
    with pytest.raises(CoachError, match="declined"):
        coach.write(EV, "this game")


# -- pipeline with Postgres ----------------------------------------------------------------------

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


def test_game_report_caches_llm_answer(conn):
    from riftwatch.baselines import build as bl
    from riftwatch.coach.pipeline import game_report, recent_report
    from riftwatch.db import repo
    from riftwatch.features import store

    for i in range(25):
        mid = f"NA1_{2000 + i}"
        match, timeline = build_game(mid, cs_bonus=i * 0.1, start_ms=1_790_000_000_000 + i * 3_600_000)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, "GOLD", "crawl")
    store.extract_pending(conn)
    bl.build(conn)

    puuid = "puuid-NA1_2000-1"    # Aatrox in the lowest-CS game
    # Answers that only cite the context item, so they pass grounding whatever it says.
    ok = CoachOutput(headline="Review.", points=[point(evidence_ids=["E1"], explanation="Aatrox top.")])
    coach, messages = fake_coach(ok, ok)

    first = game_report(conn, puuid, "NA1_2000", coach=coach, tier="gold")
    second = game_report(conn, puuid, "NA1_2000", coach=coach, tier="gold")
    assert not first.cached and second.cached and len(messages.calls) == 1
    assert second.output == first.output
    assert first.scores[0].game["cs_at_10"].goodness < 15

    offline = game_report(conn, puuid, "NA1_2000", tier="gold")
    assert offline.model == "offline" and offline.output.points

    with pytest.raises(LookupError, match="rank"):
        game_report(conn, puuid, "NA1_2000")        # no rank on record and no --tier

    recent = recent_report(conn, puuid, tier="gold")
    assert recent.scope == "recent" and len(recent.games) == 1


def test_master_plus_reference(conn):
    from riftwatch.baselines import build as bl
    from riftwatch.coach.pipeline import game_report
    from riftwatch.db import repo
    from riftwatch.features import store

    for i, bucket in enumerate(["GOLD"] * 25 + ["MASTER_PLUS"] * 25):
        mid = f"NA1_{4000 + i}"
        match, timeline = build_game(mid, cs_bonus=i * 0.1)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, bucket, "crawl")
    store.extract_pending(conn)
    bl.build(conn)
    bl.invalidate_cache()

    gold = game_report(conn, "puuid-NA1_4000-1", "NA1_4000", tier="gold")
    ref = gold.scores[0].reference["cs_at_10"]
    assert ref.tier_bucket == "MASTER_PLUS" and ref.p50 > gold.scores[0].game["cs_at_10"].baseline.p50
    assert any("Master+ median:" in e.text for e in gold.evidence.items if e.kind == "metric")

    master = game_report(conn, "puuid-NA1_4030-1", "NA1_4030", tier="master")
    assert master.scores[0].reference == {}



# -- streaming ------------------------------------------------------------------------------------

def test_completed_points_only_returns_finished_objects():
    from riftwatch.coach.llm import completed_points

    full = GOOD.model_dump_json()
    assert completed_points(full) == [GOOD.points[0].model_dump()]
    cut = full[: full.index('"evidence_ids"')]          # mid-point
    assert completed_points(cut) == []
    tricky = '{"headline": "x", "points": [{"title": "a } b", "explanation": "q\\"}", '              '"advice": "", "area": "farming", "kind": "weakness", "evidence_ids": ["E2"]}'
    assert completed_points(tricky)[0]["title"] == "a } b"


def test_stream_emits_only_grounded_points_then_final():
    from riftwatch.coach.llm import write_stream

    mixed = CoachOutput(headline="x", points=[
        point(explanation="52 CS at 10 against a median of 64."),     # grounded
        point(explanation="You lost 30 CS to your opponent.")])       # invented number
    coach, messages = fake_coach(mixed, GOOD)
    events = list(write_stream(coach, EV, "this game"))
    kinds = [e["type"] for e in events]
    assert kinds == ["point", "retrying", "done"]
    assert "52 CS" in events[0]["point"]["explanation"]
    run = events[-1]["run"]
    assert run.attempts == 2 and run.output == GOOD


def test_stream_endpoint_streams_then_caches(conn):
    from fastapi.testclient import TestClient
    from psycopg_pool import ConnectionPool

    from riftwatch.baselines import build as bl
    from riftwatch.config import Settings
    from riftwatch.db import repo
    from riftwatch.features import store
    from riftwatch.web.app import create_app
    from riftwatch.web.jobs import JobQueue

    for i in range(25):
        mid = f"NA1_{5000 + i}"
        match, timeline = build_game(mid, cs_bonus=i * 0.1)
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, mid, timeline)
        repo.mark_sample(conn, mid, "GOLD", "crawl")
    store.extract_pending(conn)
    bl.build(conn)
    bl.invalidate_cache()
    repo.upsert_account(conn, {"puuid": "puuid-NA1_5000-1", "gameName": "Me", "tagLine": "NA1"}, "na1")
    repo.insert_rank_snapshots(conn, "puuid-NA1_5000-1", [{"queueType": "RANKED_SOLO_5x5", "tier": "GOLD",
                                                          "rank": "II", "leaguePoints": 1, "wins": 1, "losses": 1}])
    ok = CoachOutput(headline="Review.", points=[point(evidence_ids=["E1"], explanation="A game.")])
    coach, messages = fake_coach(ok)
    pool = ConnectionPool(TEST_DB, min_size=1, max_size=3, open=True, kwargs={"autocommit": True})
    app = create_app(Settings.from_env({"RIFTWATCH_DATABASE_URL": TEST_DB}), api=None, coach=coach,
                     pool=pool, jobs=JobQueue(pool))
    with TestClient(app) as client:
        r = client.post("/api/players/na/Me-NA1/matches/NA1_5000/coach/stream")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]
        assert [e["type"] for e in events] == ["point", "final"]
        again = client.post("/api/players/na/Me-NA1/matches/NA1_5000/coach/stream")
        events = [json.loads(line[6:]) for line in again.text.split("\n\n") if line.startswith("data: ")]
        assert events == [{"type": "final", "output": ok.model_dump(), "cached": True}]
        assert len(messages.calls) == 1
    pool.close()
