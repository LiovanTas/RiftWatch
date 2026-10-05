import pytest

from riftwatch.features.map import MAP_X, MAP_Y
from riftwatch.ml.situations import JUNGLE_DECISIONS, jungle_examples, own_side, own_zone
from tests.fixtures import build_game, kill, monster

# -- perspective ------------------------------------------------------------------------------


@pytest.mark.parametrize("x, y, blue, red", [
    (400, 400, "own_base", "enemy_base"),
    (14400, 14500, "enemy_base", "own_base"),
    (1500, 10500, "top_lane", "top_lane"),        # top stays top for both teams
    (12000, 2000, "bot_lane", "bot_lane"),
    (7400, 7400, "mid_lane", "mid_lane"),
    (3800, 7900, "own_jungle", "enemy_jungle"),   # blue's jungle is red's enemy jungle
    (11000, 7000, "enemy_jungle", "own_jungle"),
    (4900, 10000, "river", "river"),
])
def test_zones_from_each_side(x, y, blue, red):
    assert own_zone(x, y, 100) == blue
    assert own_zone(x, y, 200) == red


@pytest.mark.parametrize("x, y, lane", [
    (1000, 10400, "top"), (4000, 13800, "top"), (7400, 7400, "mid"), (10500, 1000, "bot"),
    (13800, 5000, "bot"),
    (2288, 8448, None),    # blue gromp: next to top lane, but a camp, not the lane
    (12703, 6444, None),   # red gromp
    (3871, 7901, None),    # blue buff
    (400, 400, None),      # fountain
])
def test_lane_at(x, y, lane):
    from riftwatch.ml.situations import lane_at

    assert lane_at(x, y) == lane


def test_reflection_is_an_involution():
    u, v = own_side(3000, 9000, 200)
    back = own_side(u * MAP_X, v * MAP_Y, 200)   # reflecting twice is the identity...
    assert back == pytest.approx((3000 / MAP_X, 9000 / MAP_Y), abs=0.01)   # ...up to map shape


# -- examples from a fixture game ---------------------------------------------------------------

def examples(events=None):
    match, timeline = build_game(events=events)
    return jungle_examples(match, timeline)


def test_one_example_per_jungler_per_minute():
    ex = examples()
    assert {e.participant_id for e in ex} == {2, 7}           # the two junglers
    assert sorted({e.minute for e in ex}) == list(range(2, 16))
    assert all(e.decision in JUNGLE_DECISIONS for e in ex)


def test_no_enemy_positions_leak_into_features():
    ex = examples()[0]
    assert not any(k.startswith(("enemy_u", "enemy_v", "foe")) for k in ex.features)
    # Only the jungler's own position and teammates' distances/pushes are positional.
    positional = {k for k in ex.features if k in ("u", "v") or k.endswith(("_push", "_dist"))}
    assert positional == {"u", "v", "top_ally_push", "top_ally_dist", "middle_ally_push",
                          "middle_ally_dist", "bottom_ally_push", "bottom_ally_dist"}


def test_gank_is_read_from_a_kill_in_lane():
    # Blue jungler (2) kills red top (6) in top lane during minute 3 -> 4.
    ex = examples([kill(190_000, 2, 6, x=1500, y=10500)])
    m3 = next(e for e in ex if e.participant_id == 2 and e.minute == 3)
    assert m3.decision == "gank_top"
    m2 = next(e for e in ex if e.participant_id == 2 and e.minute == 2)
    assert m2.outcome["team_kills"] >= 1 and m3.outcome["team_kills"] == 0


# Realistic starting spots: each player in their own lane (blue), mirrored for red.
LANE_SPOTS = {1: (1100, 9000), 2: (3800, 7900), 3: (6000, 6000), 4: (9000, 1200), 5: (9400, 1400),
              6: (5900, 13880), 7: (11070, 6970), 8: (8870, 8980), 9: (13770, 5980), 10: (13570, 5580)}


def placed_game(moves=None, events=None):
    """A fixture game with everyone in lane every minute, then ``moves``:
    {(pid, minute): (x, y)} overriding positions in that minute's frame."""
    from riftwatch.ml.situations import _minute_frames

    # No scripted fights unless a test adds them: the default fixture has a top gank at 3:05.
    match, timeline = build_game(events=events if events is not None else [])
    frames = _minute_frames(timeline)
    for f in timeline["info"]["frames"]:
        for pid, (x, y) in LANE_SPOTS.items():
            f["participantFrames"][str(pid)]["position"] = {"x": x, "y": y}
    for (pid, minute), (x, y) in (moves or {}).items():
        frames[minute]["participantFrames"][str(pid)]["position"] = {"x": x, "y": y}
    return match, timeline


def decision(role, pid, minute, moves=None, events=None):
    from riftwatch.ml.situations import examples as role_examples

    match, timeline = placed_game(moves, events)
    return next(e for e in role_examples(match, timeline, role)
                if e.participant_id == pid and e.minute == minute)


def test_lane_without_a_target_is_a_rotation():
    # Blue jungler steps into mid lane at 4:00 with red mid (8) far away and no camp taken.
    from riftwatch.ml.situations import _minute_frames

    match, timeline = placed_game({(2, 4): (5000, 5000), (8, 4): (11000, 11000)})
    for f in _minute_frames(timeline)[3:5]:
        f["participantFrames"]["2"]["jungleMinionsKilled"] = 20
    ex = next(e for e in jungle_examples(match, timeline) if e.participant_id == 2 and e.minute == 3)
    assert ex.decision == "rotate"


def test_jungler_next_to_the_enemy_laner_is_a_gank():
    ex = decision("JUNGLE", 2, 3, {(2, 4): (8000, 8000)})        # beside red mid (8)
    assert ex.decision == "gank_mid"


def test_mid_laner_decisions():
    assert decision("MIDDLE", 3, 5).decision == "lane"
    assert decision("MIDDLE", 3, 5, {(3, 6): (1100, 10000)}).decision == "roam_top"
    assert decision("MIDDLE", 3, 5, {(3, 6): (11000, 1100)}).decision == "roam_bot"
    assert decision("MIDDLE", 3, 5, {(3, 6): (400, 400)}).decision == "base"


def test_support_decisions():
    assert decision("UTILITY", 5, 5).decision == "with_adc"
    assert decision("UTILITY", 5, 5, {(5, 6): (6200, 6100)}).decision == "roam_mid"
    from tests.fixtures import ward
    warded = decision("UTILITY", 5, 5, {(5, 6): (9800, 4400)}, events=[ward(330_000, 5)])
    assert warded.decision == "ward"


def test_top_laner_push_and_roam():
    from tests.fixtures import plate
    assert decision("TOP", 1, 5, events=[plate(320_000, 1, "TOP_LANE")]).decision == "push"
    assert decision("TOP", 1, 5, {(1, 6): (6000, 6000)}).decision == "roam"


def test_duo_features_for_bot_lane():
    ex = decision("BOTTOM", 4, 5)
    assert ex.features["duo_dist"] < 1.0                         # thousands of units
    assert "wards_placed" in decision("UTILITY", 5, 5).features
    assert "duo_dist" not in decision("MIDDLE", 3, 5).features


def test_every_role_produces_valid_examples():
    from riftwatch.ml.situations import DECISIONS, ROLES
    from riftwatch.ml.situations import examples as role_examples

    match, timeline = placed_game()
    for role in ROLES:
        ex = role_examples(match, timeline, role)
        assert len(ex) == 2 * 14 and all(e.decision in DECISIONS[role] for e in ex)
        assert all(e.role == role for e in ex)


def test_objective_beats_everything_else():
    ex = examples([monster(310_000, 2, assists=[4])])         # dragon at 5:10
    m5 = next(e for e in ex if e.participant_id == 2 and e.minute == 5)
    assert m5.decision == "objective"
    # Outcomes start after the decision minute: the dragon is minute 3's outcome (it falls
    # within 4:00-7:00) but not minute 5's -- taking it *was* minute 5's decision.
    m3 = next(e for e in ex if e.participant_id == 2 and e.minute == 3)
    assert m3.outcome["team_objective"] == 1.0
    assert m5.outcome["team_objective"] == 0.0


def test_scoreboard_features():
    ex = examples([kill(130_000, 1, 6, x=1500, y=10500)])      # blue top kills red top at 2:10
    blue_m3 = next(e for e in ex if e.participant_id == 2 and e.minute == 3)
    red_m3 = next(e for e in ex if e.participant_id == 7 and e.minute == 3)
    assert blue_m3.features["top_kill_diff"] == 1 and red_m3.features["top_kill_diff"] == -1
    assert blue_m3.features["team_kill_diff"] == 1


def test_rows_flatten_for_training():
    row = examples()[0].row()
    assert row["decision"] in JUNGLE_DECISIONS
    assert "f_minute" in row and "o_gold_swing" in row


def test_previous_decisions_match_earlier_labels():
    ex = [e for e in examples() if e.participant_id == 2]
    by_minute = {e.minute: e for e in ex}
    for m in range(4, 16):
        prev = by_minute[m - 1].decision
        assert by_minute[m].features[f"prev1_{prev}"] == 1.0
        assert sum(v for k, v in by_minute[m].features.items() if k.startswith("prev1_")) == 1.0
    assert by_minute[2].features["prev1_farm"] == 0.0     # nothing before the first minute


def test_enemy_jungler_last_seen_comes_only_from_visible_fights():
    # Red jungler (7) is never in a fight in this game -> "not seen" defaults.
    quiet = [e for e in examples([]) if e.participant_id == 2 and e.minute == 10][0]
    assert quiet.features["enemy_jg_seen_minutes_ago"] == 15.0
    # Red jungler kills blue mid at 4:00 in blue's jungle, top side.
    seen = [e for e in examples([kill(240_000, 7, 3, x=3800, y=7900)])
            if e.participant_id == 2 and e.minute == 6][0]
    assert seen.features["enemy_jg_seen_minutes_ago"] == pytest.approx(2.0, abs=0.01)
    assert seen.features["enemy_jg_seen_on_our_half"] == 1.0
    assert seen.features["enemy_jg_seen_topside"] == 1.0


def test_objective_timers():
    ex = {e.minute: e for e in examples([monster(310_000, 2, assists=[4])]) if e.participant_id == 2}
    assert ex[3].features["dragon_up_in"] == pytest.approx(2.0, abs=0.01)    # first spawn 5:00
    assert ex[5].features["dragon_up"] == 1.0
    # Taken at 5:10 -> back at 10:10.
    assert ex[6].features["dragon_up"] == 0.0
    assert ex[6].features["dragon_up_in"] == pytest.approx(10.1667 - 6.0, abs=0.01)
    assert ex[7].features["grubs_up_in"] == pytest.approx(1.0, abs=0.01)
    assert ex[9].features["grubs_up"] == 1.0
    assert ex[14].features["herald_up_in"] == pytest.approx(1.0, abs=0.01)
