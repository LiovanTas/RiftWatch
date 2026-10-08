import pytest

from riftwatch.vision.healthbars import Bar
from riftwatch.vision.lane import Sample, samples, trades


def seq(rows, step=0.25, start=180.0):
    """rows: (me, opponent, distance, others) at 4 per second."""
    return [Sample(start + i * step, *r) for i, r in enumerate(rows)]


def test_a_trade_is_measured_from_before_it_to_its_lowest_point():
    rows = [(0.9, 0.8, 0.2, 0)] * 8                       # standing in range, nothing happens
    rows += [(0.9, 0.7, 0.2, 0), (0.85, 0.6, 0.2, 0), (0.8, 0.55, 0.2, 0), (0.8, 0.55, 0.2, 0)]
    rows += [(0.8, 0.55, 0.6, 0)] * 12                    # back off, quiet
    report = trades(seq(rows))
    (t,) = report.trades
    assert t.started_by == "you" and t.result == "won" and not t.skirmish
    assert t.me_lost == pytest.approx(0.1) and t.opponent_lost == pytest.approx(0.25)
    assert report.summary()["won"] == 1 and report.summary()["you_started"] == 1


def test_nothing_counts_out_of_range_and_one_misread_frame_is_not_a_hit():
    rows = [(0.9, 0.8, 0.8, 0), (0.6, 0.5, 0.8, 0), (0.6, 0.5, 0.8, 0)] * 4    # far apart
    rows += [(0.9, 0.8, 0.2, 0), (0.9, 0.3, 0.2, 0), (0.9, 0.8, 0.2, 0)] * 2   # one-frame blips
    assert trades(seq(rows)).trades == []


def test_a_fight_stays_one_trade_while_the_opponent_flickers_out_of_view():
    rows = [(1.0, 1.0, 0.2, 0)] * 4
    rows += [(0.9, 0.9, 0.2, 0), (0.8, None, None, 0), (0.7, None, None, 0),
             (0.6, 0.6, 0.2, 0), (0.5, 0.5, 0.25, 0), (0.5, 0.5, 0.25, 0)]
    rows += [(0.5, 0.5, 0.7, 0)] * 12
    (t,) = trades(seq(rows)).trades
    assert t.me_lost == pytest.approx(0.5) and t.opponent_lost == pytest.approx(0.5)
    assert t.result == "even" and t.started_by == "both"


def test_another_enemy_close_by_makes_it_a_skirmish():
    rows = [(0.9, 0.9, 0.2, 1)] * 4 + [(0.7, 0.85, 0.2, 1), (0.6, 0.85, 0.2, 1),
                                       (0.6, 0.85, 0.2, 1)] + [(0.6, 0.85, 0.7, 0)] * 12
    report = trades(seq(rows))
    assert report.trades[0].skirmish and report.summary()["skirmishes"] == 1
    assert report.summary()["trades"] == 0


def test_laning_ends_at_fourteen_minutes():
    rows = [(0.9, 0.9, 0.2, 0)] * 4 + [(0.7, 0.9, 0.2, 0), (0.6, 0.9, 0.2, 0)] + \
           [(0.6, 0.9, 0.7, 0)] * 12
    assert trades(seq(rows, start=14 * 60 + 5)).trades == []


def test_samples_follow_the_replay_champion_and_track_its_opponent():
    def frame(t, me_frac, opp_x, opp_frac, extra=None):
        bars = [Bar("enemy", 605, 245, round(70 * me_frac), 70, 7),          # followed (red)
                Bar("ally", opp_x, 260, round(70 * opp_frac), 70, 7)]          # lane opponent
        if extra:
            bars.append(extra)
        return (t, 1280, 720, bars)

    rows = [frame(0, 1.0, 700, 1.0), frame(0.25, 0.9, 690, 0.8),
            # The jungler walks up on the far side: still the same opponent tracked.
            frame(0.5, 0.8, 695, 0.7, Bar("ally", 450, 250, 70, 70, 7))]
    out = samples(rows, game_offset=60)
    assert [s.t for s in out] == [60, 60.25, 60.5]
    assert [round(s.me, 2) for s in out] == [1.0, 0.9, 0.8]
    assert [round(s.opponent, 2) for s in out] == [1.0, 0.8, 0.7]
    assert out[-1].others_in_range == 1 and out[0].distance < 0.35
