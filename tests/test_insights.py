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


def test_item_changes_move_champions_by_how_often_they_buy_the_item():
    from riftwatch.stats import patches
    counts = [("16.19", 100)]
    nerf = [{"name": "Voltaic Cyclosword", "kind": "nerf", "text": "", "patch": "26.23",
             "status": "preview", "ids": [6699, 226699]}]
    shift, applied = patches.system_shift(nerf, {6699: 0.5}, counts)
    assert shift == -patches.ITEM_SHIFT * patches.PREVIEW_WEIGHT * 0.5 and applied[0]["usage"] == 50
    assert patches.system_shift(nerf, {6699: 0.05}, counts) == (0.0, [])      # rarely bought: ignored
    assert patches.mark([], applied) == "nerf"
    ids = patches.systems({"voltaic cyclosword": [6699]})
    assert any(c["name"] == "Voltaic Cyclosword" and c["ids"] == [6699] for c in ids)


def test_ladder_scores_and_labels_round_trip():
    from riftwatch.web.views import ladder_label, ladder_score
    assert ladder_score("IRON", "IV", 0) == 0
    assert ladder_score("PLATINUM", "IV", 6) == 4 * 400 + 6
    assert ladder_label(ladder_score("GOLD", "II", 55)) == "Gold II 55 LP"
    assert ladder_label(ladder_score("MASTER", None, 120)) == "Master+ 120 LP"


def test_gold_series_has_team_and_player_gold():
    from riftwatch.web.views import _gold_series

    def frame(blue_me, red_opp):
        pfs = {str(p): {"totalGold": 500} for p in range(1, 11)}
        pfs["3"]["totalGold"], pfs["8"]["totalGold"] = blue_me, red_opp
        return {"participantFrames": pfs}
    series = _gold_series({"info": {"frames": [frame(500, 500), frame(1500, 900)]}}, 3, 8)
    assert series[1] == {"minute": 1, "diff": 600, "blue": 3500, "red": 2900, "me": 1500, "opp": 900}


def test_map_moment_lines_describe_events_from_the_players_side():
    from riftwatch.web import mapview
    names = {1: "Garen", 3: "Ahri", 8: "Zed"}
    kill = {"type": "CHAMPION_KILL", "timestamp": 501_000, "killerId": 8, "victimId": 3}
    assert mapview._describe(kill, names, 3, 100) == "8:21 Zed killed you"
    assist = {"type": "CHAMPION_KILL", "timestamp": 61_000, "killerId": 1, "victimId": 8,
              "assistingParticipantIds": [3]}
    assert mapview._describe(assist, names, 3, 100) == "1:01 Garen killed Zed (you assisted)"
    drake = {"type": "ELITE_MONSTER_KILL", "timestamp": 600_000, "killerTeamId": 200,
             "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON"}
    assert mapview._describe(drake, names, 3, 100) == "10:00 The enemy took Infernal Dragon"
    tower = {"type": "BUILDING_KILL", "timestamp": 840_000, "teamId": 200, "laneType": "MID_LANE",
             "buildingType": "TOWER_BUILDING"}
    assert mapview._describe(tower, names, 3, 100) == "14:00 Your team destroyed the enemy mid tower"


def test_map_reasons_are_facts_of_the_moment():
    from riftwatch.web import mapview
    calm = {"dragon_in": 3.0, "baron_in": 8.0, "grubs_up": False, "herald_up": False}
    ctx = {"minute": 9, "team_lead": -2000, "objectives": calm, "unspent": 1400, "hp": 30, "level": 8,
           "cs_diff": -20, "opp_name": "Zed", "opp_down": None, "allies_near": 0, "enemies_near": 0,
           "lane_foes": {"mid": {"name": "Zed", "hp": 95, "level_diff": 0}}}
    assert mapview._reasons("base", "lane", ctx)[:2] == [
        "You had 1,400 gold unspent, enough to buy a real item component.", "You were at 30% health."]
    farm = mapview._reasons("farm", "gank_mid", ctx)
    assert "No epic monster was up, so camps were the best gold available." in farm
    assert any("Zed) was at 95% health" in r for r in farm)
    dragon = mapview._reasons("objective", "farm", {**ctx, "objectives": {**calm, "dragon_in": 0}})
    assert dragon[0] == "Dragon was up."
    assert mapview._reasons("farm", "lane", {**ctx, "hp": 0}) == ["You were dead at the start of this minute."]


MINUTE = {"minute": 12, "state": {"lead": -1500, "dragons": [0, 2], "dragon_in": 0.5, "baron_in": 8,
                                 "level": 9, "hp": 60, "dead": False, "unspent": 900, "cs_diff": -12,
                                 "opp": "Zed", "allies_near": 1, "enemies_near": 2},
          "advice": {"kind": "moment", "did": "invaded", "did_share": 4, "best": "Take an objective",
                     "best_share": 62, "options": [{"label": "Take an objective", "share": 62},
                                                    {"label": "Invade", "share": 4}],
                     "why": ["Dragon was about to spawn."], "next": ["12:40 The enemy took Ocean Dragon"],
                     "lead_change": {"from": -1500, "to": -2600, "minutes": 3}}}


def test_moment_evidence_carries_the_numbers_the_coach_may_use():
    from riftwatch.web import moments
    items = moments.evidence(MINUTE, {"champion": "Ahri", "role": "Mid"})
    text = " ".join(i["text"] for i in items)
    for fact in ("Minute 12", "1,500 gold behind", "60% health", "-12 CS against Zed", "4% of high-elo",
                 "take an objective (62%)", "12:40 The enemy took Ocean Dragon", "-1,500 to -2,600"):
        assert fact in text, fact


def test_moment_notes_retry_and_drop_invented_numbers():
    from riftwatch.web import moments
    first = moments.MomentNote(explanation="Most took dragon (75%).", tip="Ward at 2 minutes.", evidence_ids=["E5"])
    second = moments.MomentNote(explanation="Most took the objective (62%).", tip="Ward river early.", evidence_ids=["E5"])
    calls = []

    def parse(**request):
        calls.append(request)
        return SimpleNamespace(parsed_output=first if len(calls) == 1 else second, content=[],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    coach = SimpleNamespace(model="claude-sonnet-5-5", max_tokens=2000, thinking="adaptive", effort="low",
                            client=SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse))))
    note, usage = moments.write(coach, moments.evidence(MINUTE, {"champion": "Ahri", "role": "Mid"}))
    assert len(calls) == 2 and "75" in calls[1]["messages"][-1]["content"]
    assert note.explanation == "Most took the objective (62%)." and usage["requests"] == 2


def test_recap_streaks():
    from riftwatch.web.recap import _streaks
    assert _streaks([True, True, False, True, True, True, False, False]) == (3, 2)
    assert _streaks([]) == (0, 0)


def test_focus_picks_the_most_frequent_problem_and_its_trend():
    from riftwatch.web.views import focus
    rows = [{"result_key": "win", "position": "MIDDLE", "cs_min": v, "minutes": 30, "damage": 0, "vision": 0,
             "deaths": 0, "kp": 50, "control_wards": 1} for v in (7.0, 6.8, 6.4, 6.0, 5.6, 5.2)]   # newest first
    patterns = {"bad": [{"key": "farm_low", "phrase": "bottom-20% farm for the rank", "games": 5, "of": 6}]}
    f = focus(rows, patterns, TABLE, "GOLD")
    assert f["title"] == "Farm" and f["typical"] == "6.5 CS/min" and f["who"] == "Gold mid laners"
    assert f["trend"] == "better" and f["recent"] == "6.7 CS/min"         # newer half: 7.0, 6.8, 6.4
    assert f["spark"].startswith("<svg") and f["spark"].count("<circle") == 6
    assert focus(rows, {"bad": []}, TABLE, "GOLD") is None


def test_comebacks_and_thrown_leads():
    me, players = game(win=True)
    good = labels(insights.game_notes(me, players, {}, "GOLD", names, team15=-3200), "good")
    assert "Came back from 3.2k behind" in good
    me, players = game(win=False)
    bad = labels(insights.game_notes(me, players, {}, "GOLD", names, team15=4100), "bad")
    assert "Lost a 4.1k lead" in bad
    assert insights.Note("bad", "Lost a 4.1k lead", "", 1).pattern == "throw"
    assert insights.Note("good", "Came back from 3.2k behind", "", 1).pattern == "comeback"
    me, players = game(win=True)
    assert not [n for n in insights.game_notes(me, players, {}, "GOLD", names, team15=-1000)
                if n.label.startswith("Came back")]


def test_tilt_splits_sessions_and_finds_losing_streak_effects():
    from datetime import UTC, datetime, timedelta

    from riftwatch.web.views import tilt
    start = datetime(datetime.now(UTC).year, 1, 5, 18, tzinfo=UTC)
    rows = []
    # 12 sessions of 4 games, an hour apart within a session and a day apart between them:
    # lose, lose, then lose again, then win -- so after two losses in a row you win 1 of 2.
    for day in range(12):
        t = start + timedelta(days=day)
        for i, win in enumerate((False, False, False, True)):
            rows.append((t + timedelta(minutes=40 * i), 30 * 60, win))

    class Conn:
        def execute(self, sql, params):
            return SimpleNamespace(fetchall=lambda: rows)
    t = tilt(Conn(), "p")
    assert t["sessions"] == 12 and t["per_session"] == 4.0 and t["overall"] == 25
    assert [b["win_rate"] for b in t["by_index"]] == [0, 0, 0, 100]
    assert t["after_two"] == {"games": 24, "win_rate": 50}
    assert tilt(Conn(), "p", min_games=100) is None


def test_benchmarks_place_each_number_at_a_rank():
    from riftwatch.web.views import benchmarks
    def tier(cs, deaths):
        return {"cs_min": (0, cs, 0, 500), "dmg_min": (0, 600, 0, 500), "vision_min": (0, 0.9, 0, 500),
                "kp": (0, 0.5, 0, 500), "deaths_10": (0, deaths, 0, 500), "control_wards": (0, 1, 0, 500)}
    table = {("SILVER", "MIDDLE"): tier(6.0, 2.2), ("GOLD", "MIDDLE"): tier(6.5, 2.0),
             ("PLATINUM", "MIDDLE"): tier(7.0, 1.9)}
    mine = {"cs_min": 6.6, "dmg_min": 700, "vision_min": 0.5, "kp": 0.5, "deaths_10": 2.1, "control_wards": 1}
    b = benchmarks(table, "MIDDLE", mine, "GOLD")
    assert b["tiers"] == ["Silver", "Gold", "Platinum"]
    rows = {r["key"]: r for r in b["rows"]}
    assert rows["cs_min"]["like"] == "Gold" and rows["cs_min"]["met"] == [True, True, False]
    assert rows["deaths_10"]["like"] == "Silver"                    # lower is better
    assert rows["vision_min"]["like"] == "below Iron"
    assert b["step"]["next"] == "Platinum" and b["step"]["label"] == "Vision per minute"


def test_build_paths_follow_purchase_order():
    from riftwatch.stats import champions as cs
    rows = [([1001, 3031, 3036], None, True)] * 12 + [([3031, 6672], None, False)] * 6 + [([6672, 3031], None, True)] * 3

    class Conn:
        def execute(self, sql, params):
            return iter(rows)
    cs.clear_cache()
    paths = cs._build_paths(Conn(), 1, "BOTTOM", lambda i: i > 3000, 4, 3)
    assert paths[0]["item"] == 3031 and paths[0]["games"] == 18
    assert {n["item"]: n["games"] for n in paths[0]["next"]} == {3036: 12, 6672: 6}
    assert all(p["games"] >= 10 for p in paths)                     # 6672 first only 3 times


def test_event_log_lists_fights_objectives_and_my_items_in_order():
    from riftwatch.web.gamedata import GameData, Item
    from riftwatch.web.views import event_log
    gd = GameData()
    gd.loaded = True
    gd.items[3031] = Item(3031, "Infinity Edge", "", 3400, ["Damage"], [], 3)
    players = [{"participant_id": i, "champion_id": 0} for i in range(1, 11)]
    timeline = {"info": {"frames": [{"events": [
        {"type": "ITEM_PURCHASED", "timestamp": 600_000, "participantId": 3, "itemId": 3031},
        {"type": "CHAMPION_KILL", "timestamp": 90_000, "killerId": 3, "victimId": 8, "position": {"x": 1, "y": 1}},
        {"type": "LEVEL_UP", "timestamp": 400_000, "participantId": 3, "level": 6},
        {"type": "LEVEL_UP", "timestamp": 410_000, "participantId": 3, "level": 7},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 500_000, "killerTeamId": 200, "monsterType": "DRAGON"},
    ]}]}}
    log = event_log(timeline, players, gd, 3, 100)
    assert [e["icon"] for e in log] == ["kill", "level", "objective theirs", "item"]
    assert log[0]["text"].startswith("1:30 You killed") and log[-1]["text"] == "10:00 You finished Infinity Edge"
    assert log[0]["mine"] and not log[2]["mine"]
