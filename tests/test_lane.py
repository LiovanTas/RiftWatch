import pytest

from riftwatch.vision.healthbars import Bar
from riftwatch.vision.lane import FrameExtra, Sample, samples, trades
from riftwatch.vision.panel import PanelReading


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


def with_hud(rows, mana=0.7, ready=0b001111, level=6, depth=0.1, **kw):
    """seq(rows) as a replay reads it: the panel and the minimap on every moment."""
    out = seq(rows, **kw)
    for s in out:
        s.mana, s.ready, s.level, s.depth, s.in_base = mana, ready, level, depth, False
    return out


def test_a_trade_records_its_start_state_and_what_followed():
    rows = [(0.9, 0.5, 0.2, 0)] * 8
    rows += [(0.9, 0.35, 0.2, 0), (0.85, 0.2, 0.2, 0), (0.8, 0.15, 0.2, 0), (0.8, 0.15, 0.2, 0)]
    rows += [(0.8, 0.15, 0.3, 0)] * 4                     # a moment, then off they go, low
    rows += [(0.8, None, None, 0)] * 116
    s = with_hud(rows)
    for x in s[92:]:                                      # 23 s later: back in base
        x.in_base, x.depth = True, -0.95
    report = trades(s)
    (t,) = report.trades
    assert (t.mana, t.ready, t.level, t.depth) == (0.7, 0b001111, 6, 0.1)
    assert t.died is False and t.opponent_left_low and 20 < t.back_after_s < 25
    summary = report.summary()
    assert summary["with_ultimate_ready"]["trades"] == 1 and summary["with_ultimate_down"]["trades"] == 0
    assert (summary["died_after"], summary["back_after"], summary["opponent_left_low"]) == (0, 1, 1)


def test_a_death_after_a_trade_counts_only_where_the_panel_shows_it():
    rows = [(0.5, 0.9, 0.2, 0)] * 8
    rows += [(0.35, 0.9, 0.2, 0), (0.2, 0.9, 0.2, 0), (0.1, 0.9, 0.2, 0), (0.1, 0.9, 0.2, 0)]
    rows += [(0.1, 0.9, 0.6, 0)] * 8 + [(None, 0.9, None, 0)] * 40
    s = with_hud(rows, ready=0b000111, level=7)          # R on cooldown at 7
    for x in s[20:]:                                     # killed two seconds after
        x.dead, x.mana, x.ready, x.in_base, x.depth = True, None, None, None, None
    (t,) = trades(s).trades
    assert t.result == "lost" and t.started_by == "opponent" and t.died is True
    assert trades(s).summary()["with_ultimate_down"]["trades"] == 1
    # The same fight without the panel (your own recordings): a death can't be seen.
    (t,) = trades(seq(rows)).trades
    assert t.died is None and t.back_after_s is None and t.mana is None


def test_samples_add_the_panel_and_the_map():
    def frame(t, cam, xp, dark_r=False, dead=False):
        bars = [] if dead else [Bar("enemy", 605, 245, 60, 70, 7), Bar("ally", 690, 260, 50, 70, 7)]
        slots = {k: 200.0 for k in "QWERDF"}
        if dark_r:
            slots["R"] = 40.0
        reading = (PanelReading(None, None, None, slots) if dead
                   else PanelReading(0.8, 0.6, xp, slots))
        return (t, 1280, 720, bars, FrameExtra([], reading, cam))

    rows = [frame(0, (0.92, 0.9), 0.2, dark_r=True),     # in base (red side)
            frame(0.25, None, 0.6, dark_r=True),          # minimap missed: last position held
            frame(0.5, (0.6, 0.6), 0.1),                  # out on the map, level 2, R up
            frame(0.75, (0.6, 0.6), 0.15),
            frame(1.0, (0.92, 0.9), None, dead=True)]
    out = samples(rows, game_offset=60)
    assert [s.level for s in out] == [1, 1, 2, 2, 2]
    assert out[0].in_base and out[1].map_x == out[0].map_x and out[1].in_base
    assert not out[2].in_base and out[2].depth > out[0].depth
    assert not out[0].is_ready("R") and out[0].is_ready("Q") and out[2].is_ready("R")
    assert out[4].dead and out[4].ready is None and out[4].map_x is None
