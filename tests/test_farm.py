"""Farm given up for plays (winprob.farm), how Swing charges it, and the lane-gap inputs."""

import pytest

import riftwatch.winprob.farm as farm
from riftwatch.winprob import swing
from riftwatch.winprob.state import LANE_FEATURES, Replay
from tests.test_winprob import NAMES, LeadModel

TOP, MID, BOT = (1000, 10000), (7400, 7400), (12000, 1500)
ROLES = {1: "TOP", 2: "JUNGLE", 3: "MIDDLE", 4: "BOTTOM", 5: "UTILITY",
         6: "TOP", 7: "JUNGLE", 8: "MIDDLE", 9: "BOTTOM", 10: "UTILITY"}
HOME = {"TOP": TOP, "JUNGLE": (4000, 7000), "MIDDLE": MID, "BOTTOM": BOT, "UTILITY": BOT}


def roaming_game(minutes=20, away=(10, 11), rate=8):
    """Everyone farms their own area; blue's top laner (1) walks mid for frames 10 and 11,
    taking no CS from frame 9 to 12, and kills red's mid laner (8) at 10:30."""
    frames = []
    cs = dict.fromkeys(range(1, 11), 0)
    for m in range(minutes + 1):
        frame = {}
        for pid in range(1, 11):
            if m and not (pid == 1 and 9 <= m - 1 <= 11):
                cs[pid] += rate
            x, y = MID if pid == 1 and m in away else HOME[ROLES[pid]]
            frame[str(pid)] = {"minionsKilled": cs[pid], "jungleMinionsKilled": 0, "position": {"x": x, "y": y},
                               "totalGold": 500 + 300 * m + 20 * cs[pid], "xp": 400 * m + 50 * cs[pid],
                               "currentGold": 300, "level": min(18, 1 + m // 2)}
        frames.append(frame)
    events = [{"type": "CHAMPION_KILL", "timestamp": 630_000, "killerId": 1, "victimId": 8,
               "assistingParticipantIds": [], "bounty": 300, "position": {"x": 7400, "y": 7400}},
              {"type": "GAME_END", "timestamp": minutes * 60_000, "winningTeam": 100}]
    return frames, events


def test_farming_minutes_leave_out_time_away_dead_and_in_plays():
    frames, events = roaming_game()
    replay = Replay(frames, events)
    plays, deaths = farm.involvement(replay.events), farm.death_spans(replay)
    top = farm.farming(frames, 1, "TOP", deaths.get(1, []), plays.get(1, []))
    assert top[8] and not top[9] and not top[10] and not top[11] and top[12]
    mid = farm.farming(frames, 8, "MIDDLE", deaths.get(8, []), plays.get(8, []))
    assert not mid[10] and mid[9] and mid[15]                    # dead at 10:30, back in lane later
    assert plays[1] == [630_000] and plays[8] == [630_000]


def test_a_roam_gives_up_the_waves_missed_at_the_players_own_pace():
    frames, events = roaming_game()
    replay = Replay(frames, events)
    moments = [{"t": 630_000, "part": "kills", "players": [1], "lost": [8]}]
    rows = farm.given_up(replay, frames, moments, {1: "TOP", 8: "MIDDLE"}, farm.default_table())
    top = next(r for r in rows if r["pid"] == 1)
    assert top["moments"] == [0] and (top["from"], top["to"]) == (9, 11) and top["away"] == 3
    # Three minutes with no CS at a pace a little under typical (8 a minute against 9).
    assert 0.85 < top["pace"] < 1 and 18 < top["cs"] < 30
    assert top["typical_cs"] == 0 and not top["died"]           # no typical table: all of it counts
    assert top["gold"] == pytest.approx(top["cs"] * farm.DEFAULT_GOLD["TOP"], abs=2)
    assert top["xp"] == pytest.approx(top["cs"] * farm.DEFAULT_XP["TOP"], abs=4)
    assert top["frame"] == 14                                    # two farming minutes after it count
    # Supports aren't measured, and a play in your own area at your pace costs nothing.
    assert not farm.given_up(replay, frames, moments, {1: "UTILITY"}, farm.default_table())


def test_swing_charges_the_play_its_farm_without_changing_anyones_total():
    frames, events = roaming_game()
    with_farm = swing.game_swing(LeadModel(), frames, events, NAMES, positions=ROLES)
    without = swing.game_swing(LeadModel(), frames, events, NAMES, positions=ROLES,
                               rules=swing.Rules(farm_cost=False))
    kill = next(m for m in with_farm["moments"] if m["clock"] == "10:30")
    charged = {f["pid"]: f for f in kill["farm"]}
    assert 1 in charged and charged[1]["cost"] > 0 and charged[1]["gold"] > 0
    a, b = with_farm["players"]["1"], without["players"]["1"]
    assert a["total"] == pytest.approx(b["total"], abs=0.11)
    assert a["kills"] == pytest.approx(b["kills"] - charged[1]["cost"], abs=0.11)
    assert a["economy"] == pytest.approx(b["economy"] + charged[1]["cost"], abs=0.11)
    assert a["farm"]["plays"] == 1 and a["farm"]["gold"] == charged[1]["gold"]
    blue = sum(with_farm["players"][str(p)]["total"] for p in range(1, 6))
    assert blue == pytest.approx(50, abs=0.3)
    i = with_farm["moments"].index(kill)
    net = next(x["swing"] for x in a["best"] if x["moment"] == i)
    gross = next(x["swing"] for x in b["best"] if x["moment"] == i)
    assert net == pytest.approx(gross - charged[1]["cost"], abs=0.11)


def test_only_farm_beyond_a_typical_play_is_charged_and_less_is_credited():
    frames, events = roaming_game()
    model = LeadModel()
    worth = None
    for typical, expect in ((0.0, "full"), (5.0, "less"), (60.0, "credit")):
        table = farm.default_table()
        table["typical_cs"] = {"TOP": {"early": {"alive": typical}}}
        model.farm = table
        d = swing.game_swing(model, frames, events, NAMES, positions=ROLES)
        f = next(x for x in next(m for m in d["moments"] if m["clock"] == "10:30")["farm"] if x["pid"] == 1)
        assert f["typical_cs"] == typical and f["worth"] > 0
        worth = worth or f["worth"]
        assert f["worth"] == pytest.approx(worth)                 # what the farm was worth doesn't move
        if expect == "full":
            assert f["cost"] == pytest.approx(f["worth"])
        elif expect == "less":
            assert 0 < f["cost"] < f["worth"]
        else:
            assert f["cost"] < 0                                  # cheaper than a typical roam: credited
        blue = sum(d["players"][str(p)]["total"] for p in range(1, 6))
        assert blue == pytest.approx(50, abs=0.3)


def test_lane_gaps_read_each_position_from_both_sides():
    frames, events = roaming_game()
    replay = Replay(frames, events, positions=ROLES)
    blue, red = replay.pair(5 * 60_000)
    assert set(LANE_FEATURES) <= set(blue)
    assert blue["lane_gold_top"] == pytest.approx(0.0) and blue["lane_gold_top"] == -red["lane_gold_top"]
    later_blue, later_red = replay.pair(12 * 60_000)
    assert later_blue["lane_gold_top"] < 0 and later_red["lane_gold_top"] == pytest.approx(-later_blue["lane_gold_top"])
    bonus = replay.pair(12 * 60_000, player_bonus={1: 300})[0]
    assert bonus["lane_gold_top"] == pytest.approx(later_blue["lane_gold_top"] + 0.3)
    assert Replay(frames, events).pair(5 * 60_000)[0]["lane_gold_top"] == 0.0     # no positions known


def test_the_table_is_measured_from_farming_minutes():
    shifted = farm.shifted(({"gold_diff": 1.0, "xp_diff": 0.0, "cs_diff": 0.0, "lane_gold_top": 0.0, "lane_xp_top": 0.0},
                            {"gold_diff": -1.0, "xp_diff": 0.0, "cs_diff": 0.0, "lane_gold_top": 0.0, "lane_xp_top": 0.0}),
                           200, 500, 600, 20, "TOP")
    assert shifted[0]["gold_diff"] == pytest.approx(0.5) and shifted[1]["gold_diff"] == pytest.approx(-0.5)
    assert shifted[1]["lane_gold_top"] == pytest.approx(0.5) and shifted[1]["cs_diff"] == pytest.approx(2.0)
    assert farm.stage_of(5) == "early" and farm.stage_of(20) == "mid" and farm.stage_of(40) == "late"


def test_plays_are_sorted_by_kind_and_netted_against_their_farm():
    from riftwatch.winprob import trades

    frames, events = roaming_game()
    data = swing.game_swing(LeadModel(), frames, events, NAMES, positions=ROLES)
    kill = next(m for m in data["moments"] if m["clock"] == "10:30")
    assert trades.kind_of(kill, 1, "TOP") == "roams"            # a top laner's kill in mid lane before 14:00
    assert trades.kind_of(kill, 1, "MIDDLE") == "fights"        # in their own lane it's a fight
    assert trades.kind_of(kill, 8, "MIDDLE") == "deaths"
    assert trades.kind_of({**kill, "t": 20 * 60_000}, 1, "TOP") == "fights"
    assert trades.kind_of(kill, 3, "MIDDLE") is None            # not in it
    mine = trades.game_trades(data, 1, "TOP")["roams"]
    charged = next(f for f in kill["farm"] if f["pid"] == 1)
    assert mine["n"] == 1 and mine["gained"] > 0 and mine["farm"] == pytest.approx(charged["cost"], abs=0.01)
    assert mine["gold"] == charged["gold"]
    summary = trades.summarise([trades.game_trades(data, 1, "TOP")] * 4)
    assert summary["roams"]["per_game"] == 1 and summary["roams"]["net"] == pytest.approx(mine["gained"] - mine["farm"])
    table = {"version": swing.version(LeadModel()), "roles": {"TOP": summary}}
    out = trades.player([(1, "TOP", data)] * 4, table, LeadModel())
    assert out["role"] == "top laner" and out["kinds"][0]["key"] == "roams"
    assert out["kinds"][0]["typical_net"] == pytest.approx(out["kinds"][0]["net"], abs=0.01)


def test_a_cs_is_weighted_by_what_the_model_says_it_was_worth():
    frames, events = roaming_game()
    replay = Replay(frames, events, positions=ROLES)
    weights = farm._sensitivity(LeadModel(), replay, len(frames), ROLES)
    assert set(weights) == {1, 2, 3, 4, 6, 7, 8, 9}                # supports aren't measured
    assert all(w > 0 for w in weights[1]) and len(weights[1]) == len(frames)
    assert weights[1][5] == pytest.approx(weights[6][5], rel=0.05)  # an even game: worth the same to both
