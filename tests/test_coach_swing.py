"""Swing as coaching evidence (coach.swing_evidence): grounded, and absent without a model."""

import os

import pytest

from riftwatch.coach import swing_evidence
from riftwatch.coach.grounding import CoachOutput, numbers, validate
from tests.fixtures import build_game
from tests.test_coach import point
from tests.test_winprob import LeadModel

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="RIFTWATCH_TEST_DATABASE_URL not set")
PUUIDS = [f"swing-p{i}" for i in range(1, 11)]


@pytest.fixture
def conn(tmp_path):
    from riftwatch.baselines import build as bl
    from riftwatch.db import migrate, repo
    from riftwatch.db.connection import connect
    from riftwatch.features import store
    from riftwatch.stats import champions as champ_stats

    with connect(TEST_DB) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(c)
        # The same ten players in six games (for recent-games coaching), plus crawled games for
        # the rank baselines.
        for i in range(6):
            mid = f"NA1_{7000 + i}"
            match, timeline = build_game(mid, puuids=PUUIDS, blue_wins=i % 2 == 0, cs_bonus=i * 0.2,
                                         start_ms=1_790_000_000_000 + i * 3_600_000)
            repo.insert_match(c, match)
            repo.insert_timeline(c, mid, timeline)
        for i in range(20):
            mid = f"NA1_{7100 + i}"
            match, timeline = build_game(mid, cs_bonus=i * 0.1, start_ms=1_780_000_000_000 + i * 3_600_000)
            repo.insert_match(c, match)
            repo.insert_timeline(c, mid, timeline)
            repo.mark_sample(c, mid, "GOLD", "crawl")
        store.extract_pending(c)
        bl.build(c)
        bl.invalidate_cache()
        champ_stats.clear_cache()
        swing_evidence.use(LeadModel(), tmp_path)
        yield c
    swing_evidence.use(None, None)
    champ_stats.clear_cache()


def _swing_items(result):
    return [e for e in result.evidence.items if e.data.get("source") == "swing"]


def test_game_coaching_gets_the_players_swing_and_moments(conn):
    from riftwatch.coach.pipeline import game_report
    from riftwatch.winprob import swing

    puuid = PUUIDS[2]                                   # blue mid: dies twice, kills once
    result = game_report(conn, puuid, "NA1_7000", tier="gold")
    items = _swing_items(result)
    assert items, [e.text for e in result.evidence.items]
    head = items[0]
    data = swing.stored(conn, ["NA1_7000"], LeadModel())["NA1_7000"]
    mine = data["players"]["3"]
    assert "percentage points of win chance" in head.text and f"{mine['total']:+.1f}" in head.text
    assert "deaths" in head.text and head.area == "macro"
    moments = [e for e in items if e.kind == "swing_moment"]
    assert moments and any("your team's chance to win went from" in e.text for e in moments)
    assert any("You died at level" in e.text and "out for" in e.text for e in moments)
    assert any(e.polarity == "weakness" for e in moments) and any(e.polarity == "strength" for e in moments)
    # Ids stay consecutive after the Swing items are appended.
    assert [e.id for e in result.evidence.items] == [f"E{i + 1}" for i in range(len(result.evidence.items))]

    # A coach point quoting the Swing items' numbers passes the grounding check, and the
    # offline coach (built from the evidence) is grounded by construction.
    worst = next(e for e in moments if e.polarity == "weakness")
    quoted = " ".join(raw for _, _, raw in numbers(worst.text)[:3])
    answer = CoachOutput(headline="Review.", points=[point(evidence_ids=[head.id, worst.id],
                                                           explanation=f"Swing {mine['total']:+.1f}; {quoted}.")])
    assert validate(answer, result.evidence) == []
    assert validate(result.output, result.evidence) == []


def test_recent_coaching_compares_the_parts_with_the_role(conn):
    from riftwatch.coach.pipeline import recent_report

    result = recent_report(conn, PUUIDS[2], tier="gold")
    items = _swing_items(result)
    text = " ".join(e.text for e in items)
    assert "Swing per game over the last 6 games, by part, against the average mid laner" in text
    assert text.count("Swing is RiftWatch's measure") == 1          # defined once
    assert "(typical" in text
    assert validate(result.output, result.evidence) == []


def test_without_a_model_coaching_is_exactly_as_before(conn):
    from riftwatch.coach.pipeline import game_report, recent_report

    with_swing = game_report(conn, PUUIDS[2], "NA1_7000", tier="gold")
    swing_evidence.use(None, None)
    plain = game_report(conn, PUUIDS[2], "NA1_7000", tier="gold")
    assert not _swing_items(plain)
    assert [e.text for e in plain.evidence.items] == [e.text for e in with_swing.evidence.items
                                                      if e.data.get("source") != "swing"]
    assert not _swing_items(recent_report(conn, PUUIDS[2], tier="gold"))


def test_a_failure_in_swing_evidence_leaves_coaching_untouched(conn, monkeypatch):
    from riftwatch.coach import pipeline

    def broken(*_args):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(pipeline, "add_game_swing", broken)
    result = pipeline.game_report(conn, PUUIDS[2], "NA1_7000", tier="gold")
    assert not _swing_items(result) and result.output.points


def test_the_prompt_version_is_part_of_the_cache_key():
    from riftwatch.coach.llm import PROMPT_VERSION, SYSTEM_PROMPT, Coach

    assert Coach("claude-sonnet-5-5", client=object()).label.endswith(f"/{PROMPT_VERSION}")
    assert "Swing items" in SYSTEM_PROMPT
