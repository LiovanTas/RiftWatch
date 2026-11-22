"""The match page's laning phase card and the game's turning point."""

import riftwatch.winprob.farm as farm
from riftwatch.web import laning, swingview
from riftwatch.winprob import swing
from tests.test_farm import ROLES, roaming_game
from tests.test_winprob import NAMES, LeadModel


def timeline(frames, events):
    return {"info": {"frames": [{"participantFrames": f, "events": events if i == 0 else []}
                                for i, f in enumerate(frames)]}}


def test_the_laning_card_reads_leads_minutes_pace_and_plates():
    frames, events = roaming_game()
    events = [*events, {"type": "TURRET_PLATE_DESTROYED", "timestamp": 500_000, "killerId": 1, "laneType": "TOP_LANE",
                        "teamId": 200},
              {"type": "TURRET_PLATE_DESTROYED", "timestamp": 520_000, "killerId": 6, "laneType": "TOP_LANE",
               "teamId": 100}]
    sw = swing.game_swing(LeadModel(), frames, events, NAMES, positions=ROLES)
    out = laning.report(timeline(frames, events), 1, 6, "TOP", sw, farm.default_table())
    assert [c["minute"] for c in out["checks"]] == [5, 10, 14]
    # The top laner stopped farming from 9:00 to 12:00 to roam: behind on CS at 10 and 14.
    assert out["checks"][0]["cs_diff"] == 0 and out["checks"][1]["cs_diff"] < 0 and out["checks"][2]["cs_diff"] < 0
    assert out["away_minutes"] == 3 and out["farm_minutes"] == 10 and out["base_minutes"] == 0
    assert out["pace"] is not None and 80 < out["pace"] < 100      # 8 a minute against 9
    assert out["given"]["plays"] == 1 and out["given"]["cs"] > 15
    assert out["plates_took"] == 1 and out["plates_lost"] == 1 and (out["kills"], out["deaths"]) == (1, 0)
    assert out["verdict"].startswith(("Even", "Slightly", "Won", "Lost"))
    # A support has leads and plates but no farming pace.
    sup = laning.report(timeline(frames, events), 5, 10, "UTILITY", sw, None)
    assert "pace" not in sup and sup["plates_lost"] is not None
    assert laning.report(None, 1, 6, "TOP", sw, None) is None


def test_the_turning_point_is_the_last_lead_change_that_held():
    moments = [{"t": 300_000, "clock": "5:00", "text": "Red took a kill", "part": "kills", "before": 50.0, "after": 40.0, "swing": 10.0},
               {"t": 600_000, "clock": "10:00", "text": "Blue took a kill", "part": "kills", "before": 40.0, "after": 55.0, "swing": 15.0},
               {"t": 700_000, "clock": "11:40", "text": "Red took dragon", "part": "objectives", "before": 55.0, "after": 30.0, "swing": 25.0},
               {"t": 900_000, "clock": "15:00", "text": "Blue took Baron", "part": "objectives", "before": 30.0, "after": 62.0, "swing": 32.0},
               {"t": 950_000, "clock": "15:50", "text": "Blue finished Edge", "part": "items", "before": 62.0, "after": 70.0, "swing": 8.0}]
    series = [[0, 50.0], [300, 40.0], [600, 55.0], [700, 30.0], [800, 28.0], [900, 62.0], [1200, 90.0]]
    tp = swingview.turning_point({"winner": 100, "moments": moments, "series": series})
    assert tp["text"] == "Blue took Baron" and tp["kind"] == "comeback" and (tp["before"], tp["after"]) == (30, 62)
    assert tp["low"] == 28 and tp["low_clock"] == "13:20"
    # Red winning: the same game seen from red's side; they led after 11:40 until Baron, so no
    # lead change held and the biggest swing their way is the dragon.
    red = swingview.turning_point({"winner": 200, "moments": moments, "series": series})
    assert red["text"] == "Red took dragon" and red["kind"] == "swing" and red["before"] == 45
    assert swingview.turning_point({"winner": 100, "moments": [], "series": series}) is None


def test_clutch_counts_the_games_a_player_decided_or_died_in():
    class Champ:
        name = "Ahri"

    class GD:
        def champion(self, _):
            return Champ()

    turn = {"t": 900_000, "clock": "15:00", "text": "Blue took Baron", "part": "objectives", "before": 30.0,
            "after": 62.0, "swing": 32.0, "team": 100, "players": [2, 3], "lost": []}
    death = {"t": 900_000, "clock": "15:00", "text": "C2 killed C7", "part": "kills", "before": 30.0,
             "after": 62.0, "swing": 32.0, "team": 100, "players": [2], "lost": [7]}
    series = [[0, 50.0], [600, 30.0], [900, 62.0], [1200, 90.0]]
    rows = [(("G1", 2, "JUNGLE", 100, 1), {"winner": 100, "moments": [turn], "series": series}),
            (("G2", 7, "MIDDLE", 200, 1), {"winner": 100, "moments": [death], "series": series}),
            (("G3", 4, "BOTTOM", 100, 1), {"winner": 100, "moments": [turn], "series": series})]
    out = swingview._clutch(rows, GD())
    assert (out["games"], out["made"], out["died"]) == (3, 1, 1)
    assert [e["kind"] for e in out["examples"]] == ["made", "died"]
    assert swingview._clutch(rows[2:], GD()) is None


def test_the_jungle_card_counts_ganks_by_lane_objectives_and_invading():
    from riftwatch.web import jungle

    frames, events = roaming_game()
    events = [*events,
              {"type": "CHAMPION_KILL", "timestamp": 300_000, "killerId": 2, "victimId": 6,
               "assistingParticipantIds": [1], "position": {"x": 1000, "y": 10000}},
              {"type": "CHAMPION_KILL", "timestamp": 500_000, "killerId": 9, "victimId": 2,
               "assistingParticipantIds": [], "position": {"x": 12000, "y": 1500}},
              {"type": "ELITE_MONSTER_KILL", "timestamp": 360_000, "killerTeamId": 100, "killerId": 2,
               "monsterType": "HORDE"},
              {"type": "ELITE_MONSTER_KILL", "timestamp": 420_000, "killerTeamId": 200, "killerId": 7,
               "monsterType": "DRAGON"}]
    out = jungle.report(timeline(frames, events), 2, 7)
    lanes = {l["label"]: l for l in out["lanes"]}
    assert lanes["Top"]["kills"] == 1 and lanes["Top"]["first"] == "5:00" and lanes["Bot"]["deaths"] == 1
    assert out["ganks"] == 1 and out["deaths"] == 1
    assert (out["ours"], out["with_me"], out["contested"]) == (1, 1, 2)
    assert out["invading"] == 0 and [c["minute"] for c in out["first_clear"]] == [3, 4]
    assert jungle.report(None, 2, 7) is None
