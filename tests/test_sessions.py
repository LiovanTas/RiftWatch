import os
from datetime import UTC, datetime, timedelta

import pytest

from riftwatch.analysis.sessions import summarize

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
DAY = datetime(2026, 9, 1, 18, tzinfo=UTC)
GAME_S = 30 * 60


def session(day: int, scores: list[float], wins: list[bool] | None = None):
    """One evening's games, back to back (5 minutes between them)."""
    wins = wins or [True] * len(scores)
    start = DAY + timedelta(days=day)
    return [(start + timedelta(minutes=35 * i), GAME_S, w, s)
            for i, (s, w) in enumerate(zip(scores, wins, strict=True))]


def test_sessions_split_at_an_hours_break_and_group_by_position():
    games = session(0, [50, 50, 50, 50, 50]) + session(1, [40, 40])
    # A 59-minute gap stays in the same session; an hour starts a new one.
    end = games[-1][0] + timedelta(seconds=GAME_S)
    games.append((end + timedelta(minutes=59), GAME_S, True, 40))
    games.append((end + timedelta(minutes=59 + 30 + 60), GAME_S, True, 40))
    report = summarize(games)
    assert report.sessions == 3 and report.games == 9
    # 1st: all three; 2nd and 3rd: the first two sessions; 4th+: games 4-5 of the first.
    assert [g.games for g in report.by_position] == [3, 2, 2, 2]


def test_a_real_within_session_drop_is_found():
    games = []
    for day in range(12):
        base = 40 + (day % 3) * 5
        games += session(day, [base + 6, base + 3, base, base - 6, base - 7])
    report = summarize(games)
    assert report.late_vs_first.sessions == 12 and report.late_vs_first.clear
    (finding,) = [f for f in report.findings if f.kind == "session_length"]
    assert finding.worse and finding.gap == pytest.approx(-12.5, abs=0.1)
    assert f"{12.5:.0f} points lower" in finding.text and "12 sessions" in finding.text


def test_long_sessions_on_bad_days_are_not_mistaken_for_fatigue():
    # Within any one evening, every game is the same. But long evenings are the bad ones,
    # so pooled "first game" vs "4th+ game" averages differ a lot. No finding.
    games = []
    for day in range(12):
        games += session(day, [55, 55])                   # good days: short sessions
        games += session(day + 100, [35] * 6)             # bad days: long sessions
    report = summarize(games)
    first, late = report.by_position[0], report.by_position[-1]
    assert first.score - late.score > 8                   # the pooled trap
    assert report.late_vs_first.gap == 0 and not report.late_vs_first.clear
    assert report.findings == []


def test_after_losses_pattern_within_sessions():
    games = []
    for day in range(12):
        # win, win, loss, loss, (game after 2 losses), loss-recovery
        games += session(day, [50, 52, 48, 47, 38, 40], [True, True, False, False, False, True])
    report = summarize(games)
    lw = report.losses_vs_win
    assert lw.sessions == 12 and lw.clear and lw.gap < 0
    assert any(f.kind == "after_losses" and f.worse for f in report.findings)


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


def test_sessions_report_end_to_end(conn):
    from riftwatch.coach import pipeline
    from tests.test_pool import _load_player

    _load_player(conn)          # 25 games an hour apart: back-to-back 26-minute games
    report = pipeline.sessions_report(conn, "me", tier="gold")
    assert report.games == 25 and report.sessions == 1
    assert report.by_position[-1].games == 22


def test_findings_become_coach_evidence():
    from riftwatch.coach.evidence import Evidence, EvidenceSet, add_session_evidence

    games = []
    for day in range(12):
        games += session(day, [50, 49, 47, 38, 37])
    ev = EvidenceSet([Evidence("E1", "context", "context", "neutral", "Last 20 games.")])
    add_session_evidence(ev, summarize(games))
    (item,) = ev.items[1:]
    assert item.id == "E2" and item.polarity == "weakness" and "fourth game on" in item.text
