"""Match insights (did well / could improve), the high-elo map, and matchup grounding."""

from types import SimpleNamespace

from riftwatch.ml import advisor as advisor_mod
from riftwatch.ml.advisor import Advisor, Option
from riftwatch.stats import matchups
from riftwatch.web import insights, mapview


def row(pid, team, position, champ, **extra):
    base = {"participant_id": pid, "team_id": team, "position": position, "champion_id": champ,
            "remake": False, "duration_s": 1800, "kills": 3, "deaths": 4, "assists": 5,
            "team_kills": 20, "cs": 200, "gold": 10000, "damage": 15000, "vision_score": 20,
            "control_wards": 2, "first_blood": False, "largest_multikill": 1, "win": team == 100}
    base.update(extra)
    return base


def game(**me):
    players = [row(1, 100, "TOP", 10), row(2, 100, "JUNGLE", 11), row(3, 100, "MIDDLE", 12, **me),
               row(4, 100, "BOTTOM", 13), row(5, 100, "UTILITY", 14),
               row(6, 200, "TOP", 20), row(7, 200, "JUNGLE", 21), row(8, 200, "MIDDLE", 22),
               row(9, 200, "BOTTOM", 23), row(10, 200, "UTILITY", 24)]
    return players[2], players


def names(cid):
    return {22: "Ahri"}.get(cid, f"Champ{cid}")


# (20th, 50th, 80th percentile, games) for Gold mid laners
TABLE = {("GOLD", "MIDDLE"): {"cs_min": (5.5, 6.5, 7.5, 400), "vision_min": (0.5, 0.7, 0.9, 400),
                              "dmg_min": (500, 650, 800, 400), "deaths_10": (1.0, 1.8, 2.6, 400),
                              "kp": (0.4, 0.5, 0.6, 400), "control_wards": (0, 2, 4, 400)}}


def labels(notes, kind):
    return [n.label for n in notes if n.kind == kind]


def test_a_carry_game_reads_as_strengths():
    me, players = game(damage=40000, cs=290, kills=12, assists=8, first_blood=True, deaths=1)
    notes = insights.game_notes(me, players, TABLE, "GOLD", names)
    good = labels(notes, "good")
    assert "Most damage in the game" in good
    assert "Out-farmed Ahri by 90 CS" in good
    assert "Farm: 9.7 CS/min" in good             # 290 / 30 min is above Gold's 80th percentile
    assert "First blood" in good
    assert not labels(notes, "bad")


def test_a_rough_game_reads_as_things_to_work_on():
    me, players = game(damage=6000, cs=120, deaths=11, kills=1, assists=2, control_wards=0)
    bad = labels(insights.game_notes(me, players, TABLE, "GOLD", names), "bad")
    assert "80 CS behind Ahri" in bad
    assert "Low farm: 4.0 CS/min" in bad
    assert "Died 11 times" in bad
    assert "No control wards" in bad
    assert "Missed fights: 15% KP" in bad


def test_rank_comparisons_fall_back_to_all_ranks_when_a_tier_is_thin():
    thin = {("DIAMOND", "MIDDLE"): {k: (*v[:3], 10) for k, v in TABLE[("GOLD", "MIDDLE")].items()},
            ("ALL", "MIDDLE"): TABLE[("GOLD", "MIDDLE")]}
    me, players = game(cs=290)
    notes = insights.game_notes(me, players, thin, "DIAMOND", names)
    farm = next(n for n in notes if n.label.startswith("Farm"))
    assert "ranked mid laners" in farm.why          # not "Diamond": that tier had 10 games


def test_lane_leads_come_from_the_timeline():
    def frame(gold_me, gold_opp):
        return {"participantFrames": {"3": {"totalGold": gold_me, "minionsKilled": 80, "xp": 5000},
                                      "8": {"totalGold": gold_opp, "minionsKilled": 70, "xp": 4800}}}
    timeline = {"info": {"frames": [frame(500, 500)] * 10 + [frame(4200, 3000)] * 6}}
    leads = insights.lane_leads(timeline, 3, 8)
    assert leads[10]["gold"] == 1200 and leads[10]["cs"] == 10 and 15 in leads
    me, players = game()
    notes = insights.game_notes(me, players, {}, "GOLD", names, lane=leads)
    assert "+1.2k gold on Ahri at 15" in labels(notes, "good")


def test_card_shows_two_of_each():
    notes = [insights.Note("good", f"g{i}", "", 3 - i) for i in range(3)] + \
            [insights.Note("bad", f"b{i}", "", 3 - i) for i in range(3)]
    assert [n["label"] for n in insights.card(notes)] == ["g0", "g1", "b0", "b1"]


def test_map_targets_mirror_for_the_red_side():
    assert mapview.target("base", "JUNGLE", 100, 7000, 7000, None) == mapview.FOUNTAIN
    red = mapview.target("base", "JUNGLE", 200, 7000, 7000, None)
    assert red[0] > 14000 and red[1] > 14000
    assert mapview.target("gank_mid", "JUNGLE", 200, 0, 0, None) == mapview.LANE_POINT["mid"]
    # Farming heads for the nearer camp quadrant of the player's own jungle
    assert mapview.target("farm", "JUNGLE", 100, 8000, 3000, None) == (7700, 3900)
    assert mapview.target("with_adc", "UTILITY", 100, 0, 0, (12000, 2000)) == (12000, 2000)
    assert mapview.target("fight", "TOP", 100, 0, 0, None) is None
    pct = mapview._pct(mapview.X0, mapview.Y0)
    assert pct == {"x": 0.0, "y": 100.0}           # bottom-left corner of the minimap


def test_advisor_minutes_label_each_choice(monkeypatch, tmp_path):
    (tmp_path / "JUNGLE.joblib").write_bytes(b"")
    picks = {3: "farm", 4: "invade", 5: "gank_mid", 6: "other"}
    fake = [SimpleNamespace(minute=m, decision=d, participant_id=2) for m, d in picks.items()]
    monkeypatch.setattr(advisor_mod, "examples", lambda match, timeline, role: fake)

    def options(trained, minutes):
        table = {3: {"farm": .7, "invade": .1, "gank_mid": .2},          # matched a consensus
                 4: {"farm": .2, "invade": .05, "objective": .75},       # against a consensus
                 5: {"farm": .45, "gank_mid": .35, "invade": .2},        # less common
                 6: {"farm": .6, "other": .4}}
        return [[Option(d, s, 0, 0, 0) for d, s in table[e.minute].items()] for e in minutes]

    adv = Advisor(tmp_path)
    monkeypatch.setattr(adv, "model", lambda role: None)
    monkeypatch.setattr(Advisor, "_options", staticmethod(options))
    match = {"info": {"participants": [{"puuid": "me", "teamPosition": "JUNGLE", "participantId": 2}]}}
    role, minutes = adv.minutes(match, {}, "me")
    assert role == "JUNGLE"
    assert [k for _, _, k in minutes] == ["good", "moment", "differ", "neutral"]


EVIDENCE = [{"id": "E1", "text": "Record: Garen against Darius: 40 games, Garen won 15 (38%)."},
            {"id": "E2", "text": "Darius Q Decimate: cooldown 9/8/7/6/5 s (ranks 1 up), range 425."}]


def test_matchup_check_catches_invented_numbers_and_bad_citations():
    ok = matchups.MatchupOutput(
        verdict="Darius is favoured: Garen won 38% of 40 games.",
        points=[matchups.MatchupPoint(title="Respect Decimate", explanation="It comes back every 9 s at rank 1.",
                                      evidence_ids=["E2"])],
        tips=["Trade when Decimate is down."])
    assert matchups.check(ok, EVIDENCE) == []
    bad = matchups.MatchupOutput(
        verdict="Garen loses 70% of the time.",
        points=[matchups.MatchupPoint(title="Range", explanation="Decimate reaches 600.", evidence_ids=["E2"]),
                matchups.MatchupPoint(title="Stats", explanation="Garen won 38%.", evidence_ids=["E9"])],
        tips=["Wait for level 11."])
    problems = matchups.check(bad, EVIDENCE)
    assert any("verdict: 70" in p for p in problems)
    assert any("point 1: 600" in p for p in problems)
    assert any("point 2: cites ['E9']" in p for p in problems)
    assert any("tip 1: 11" in p for p in problems)


def test_matchup_write_retries_then_drops_what_is_still_ungrounded():
    first = matchups.MatchupOutput(verdict="Garen wins 90%.", points=[], tips=[])
    second = matchups.MatchupOutput(
        verdict="Darius is favoured.",
        points=[matchups.MatchupPoint(title="Good", explanation="Garen won 38%.", evidence_ids=["E1"]),
                matchups.MatchupPoint(title="Bad", explanation="Garen won 55%.", evidence_ids=["E1"])],
        tips=["Back off at 3 stacks."])
    calls = []

    def parse(**request):
        calls.append(request)
        out = first if len(calls) == 1 else second
        return SimpleNamespace(parsed_output=out, content=[],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50))

    coach = SimpleNamespace(model="claude-sonnet-5-5", max_tokens=4000, thinking="adaptive", effort="low",
                            client=SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse))))
    output, dropped, usage = matchups.write(coach, EVIDENCE, "Garen", "Darius", "Top")
    assert len(calls) == 2 and "grounding check" in calls[1]["messages"][-1]["content"]
    assert [p.title for p in output.points] == ["Good"] and dropped[0]["title"] == "Bad"
    assert output.tips == []                        # "3" isn't in the evidence
    assert usage["requests"] == 2


def test_patch_names_and_game_versions_compare():
    from riftwatch.stats import patches
    assert patches.game_patch("26.22") == patches.game_patch("16.22") == (26, 22)
    assert patches.game_patch("16.9") < patches.game_patch("26.19") < patches.game_patch("16.22")


def test_patch_shift_counts_only_games_from_before_the_change():
    from riftwatch.stats import patches
    counts = [("16.19", 75), ("16.22", 25)]
    buff = [{"kind": "buff", "status": "live", "patch": "26.22"}]
    assert patches.shift(buff, counts) == 0.75 * patches.SHIFT       # a quarter already has it
    nerf_preview = [{"kind": "nerf", "status": "preview", "patch": "26.23"}]
    assert patches.shift(nerf_preview, counts) == -patches.PREVIEW_WEIGHT * patches.SHIFT
    assert patches.shift([{"kind": "adjust", "status": "live", "patch": "26.22"}], counts) == 0
    assert patches.mark(buff + nerf_preview) == "adjust" and patches.mark(buff) == "buff"


def test_patch_notes_file_names_real_kinds_and_sources():
    from riftwatch.stats import patches
    data = patches.notes()
    assert data["patches"] and all(p["source"].startswith("https://") for p in data["patches"])
    kinds = {c["kind"] for p in data["patches"] for c in p["changes"]}
    assert kinds <= {"buff", "nerf", "adjust"}
    found = patches.by_champion({"syndra": 134, "lee sin": 64})
    assert found[134][0]["status"] == "preview" and found[64][0]["patch"] == "26.21"


def test_lobby_text_gives_riot_ids_in_order_without_repeats():
    from riftwatch.web.views import parse_lobby
    chat = ("Faker #KR1 joined the lobby\nCaps #EUW joined the lobby\n"
            "faker#kr1 left the lobby\nno tag here\nDouble  Space #NA1, Short#ab")
    assert parse_lobby(chat) == [("Faker", "KR1"), ("Caps", "EUW"), ("Double  Space", "NA1"),
                                 ("Short", "ab")]
    assert parse_lobby("") == [] and len(parse_lobby("\n".join(f"P{i}#NA1" for i in range(15)))) == 10


def test_patterns_count_recurring_notes():
    def n(label, kind="bad"):
        return insights.Note(kind, label, "", 1)
    games = [[n("Low farm: 4.9 CS/min"), n("Most damage in the game", "good")],
             [n("Low farm: 5.1 CS/min")], [n("Low farm: 5.0 CS/min"), n("No control wards")], []]
    found = insights.patterns(games)
    assert found["bad"] == [{"key": "farm_low", "phrase": "bottom-20% farm for the rank", "games": 3, "of": 4}]
    assert found["good"] == []                                      # one game isn't a pattern
    assert all(n(label).pattern for label in ("Out-farmed Ahri by 40 CS", "40 CS behind Ahri",
                                              "+1.0k gold on Ahri at 15", "Died 9 times", "Quadra kill"))
