"""The win-probability state, network and the Swing metric built on it."""

import math
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from riftwatch.winprob import choices
from riftwatch.winprob import model as wp
from riftwatch.winprob import swing
from riftwatch.winprob.state import (BASE_FEATURES, FULL_FEATURES, Catalog, Replay, compact,
                                     respawn_seconds)
from tests.fixtures import build_game

NAMES = {i: f"C{i}" for i in range(1, 11)}


class LeadModel:
    """Stands in for the network: blue's chance follows gold, kills and towers."""

    stamp = "test"

    def blue(self, blue_states, red_states):
        def side(s):
            z = 0.4 * s["gold_diff"] + 0.3 * (s["kills_own"] - s["kills_enemy"]) + 0.5 * (s["towers_own"] - s["towers_enemy"])
            return 1 / (1 + math.exp(-z))
        return np.array([(side(b) + 1 - side(r)) / 2 for b, r in zip(blue_states, red_states, strict=True)])


class ItemModel(LeadModel):
    """Also reads items: armor is worth more against a physical-damage team; and drafts with
    more marksmen are favoured, so the game doesn't start at 50%."""

    reads_items = True

    def __init__(self, catalog):
        self.catalog = catalog

    def blue(self, blue_states, red_states):
        def side(s):
            z = (0.4 * s["gold_diff"] + 0.3 * (s["kills_own"] - s["kills_enemy"])
                 + 0.2 * (s["items_own"] - s["items_enemy"]) + 0.3 * s["armor_own"] * (1 - s["magic_share_enemy"])
                 + 0.2 * (s["marksmans_own"] - s["marksmans_enemy"]))
            return 1 / (1 + math.exp(-z))
        return np.array([(side(b) + 1 - side(r)) / 2 for b, r in zip(blue_states, red_states, strict=True)])


def catalog():
    cat = Catalog(items={1001: ("items", "ad", "crit"), 1002: ("items", "armor", "health"), 1003: ("items", "ap")},
                  item_names={1001: "Edge", 1002: "Plate", 1003: "Staff"})
    cat.champions = {c: ("Marksman" if c in (1, 2) else "Fighter", 0.2) for c in range(1, 11)}
    cat.index = {c: c - 1 for c in range(1, 11)}
    cat.popular = {c: [1001, 1002, 1003] for c in range(1, 11)}
    return cat


def game(blue_wins=True):
    _, timeline = build_game("NA1_1", blue_wins=blue_wins)
    return compact(timeline)


def test_state_counts_events_from_each_side():
    frames, events = game()
    r = Replay(frames, events)
    assert r.winner == 100
    blue, red = r.state(14 * 60_000, 100), r.state(14 * 60_000, 200)
    assert blue["kills_own"] == red["kills_enemy"] == 2 and blue["kills_enemy"] == 2
    assert blue["towers_own"] == 1 and red["towers_enemy"] == 1
    assert blue["dragons_own"] == 1 and red["dragons_enemy"] == 1
    assert blue["gold_diff"] == pytest.approx(-red["gold_diff"])
    # Baron at 20:00: all five of blue are alive and get the buff; blue's 3 dies at 21:40 and
    # loses it; it's gone for everyone by 24:00, and the take still counts.
    assert r.state(21 * 60_000, 100)["baron_own"] == 5 and r.state(21 * 60_000, 200)["baron_enemy"] == 5
    assert r.state(22 * 60_000, 100)["baron_own"] == 4
    late = r.state(24 * 60_000, 100)
    assert late["baron_own"] == 0 and late["barons_own"] == 1


def test_dead_players_come_back_after_their_timer():
    frames, events = game()
    r = Replay(frames, events)
    # Red's 6 dies at 3:05 at a low level: dead a few seconds later, alive again by 3:30.
    assert r.state(187_000, 200)["alive_own"] == 4
    assert r.state(210_000, 200)["alive_own"] == 5
    assert r.dead_at(187_000) == {6}
    assert respawn_seconds(18, 40) > respawn_seconds(18, 10) > respawn_seconds(1, 10)


def test_the_richer_state_reads_towers_plates_timers_and_objectives():
    frames, events = game()
    r = Replay(frames, events)
    # Blue's 2 kills red's 6 at 3:05: red has one player out, for the rest of the death timer.
    blue, red = r.pair(186_000)
    assert red["dead_time_own"] == pytest.approx(blue["dead_time_enemy"]) and 0 < red["dead_time_own"] < 0.3
    # Blue takes the mid outer tower at 13:20 and a bot plate at 10:00.
    later = r.pair(14 * 60_000)[0]
    assert later["towers_outer_own"] == 1 and later["towers_inner_own"] == 0 and later["plates_own"] == 1
    # Dragon is taken at 9:20: down at 10:00, back up five minutes after the kill.
    assert r.pair(10 * 60_000)[0]["dragon_up"] == 0 and r.pair(14 * 60_000 + 21_000)[0]["dragon_up"] == 1
    # Baron spawns at 20:00, is taken at 20:00 and is back six minutes later.
    assert r.pair(19 * 60_000)[0]["baron_up"] == 0 and r.pair(21 * 60_000)[0]["baron_up"] == 0
    assert r.pair(26 * 60_000 + 1_000)[0]["baron_up"] == 1
    assert later["lvl6_own"] == 5 and r.pair(4 * 60_000)[0]["lvl6_own"] == 0
    assert later["held_gold_own"] > 0


def test_items_and_composition_are_read_from_purchases():
    frames, events = game()
    champions = {p: p for p in range(1, 11)}
    purchases = {4: [(600, 1001), (900, 9999), (1200, 1002)], 9: [(700, 1003)]}
    r = Replay(frames, events, champions, purchases, catalog())
    early, later = r.pair(12 * 60_000), r.pair(21 * 60_000)
    assert early[0]["items_own"] == 1 and early[0]["crit_own"] == 1 and early[0]["ap_enemy"] == 1
    assert later[0]["items_own"] == 2 and later[0]["armor_own"] == 1 and later[1]["armor_enemy"] == 1
    assert early[0]["marksmans_own"] == 2 and early[1]["marksmans_enemy"] == 2
    assert early[0]["_own"] == [1, 2, 3, 4, 5] and early[1]["_own"] == [6, 7, 8, 9, 10]
    # An item someone could have bought instead.
    extra = r.pair(12 * 60_000, item_extra={100: Counter(("items", "armor"))})
    assert extra[0]["items_own"] == 2 and extra[0]["armor_own"] == 1
    assert set(FULL_FEATURES) <= set(early[0])


def test_team_swing_adds_up_to_the_result():
    for blue_wins in (True, False):
        frames, events = game(blue_wins)
        d = swing.game_swing(LeadModel(), frames, events, NAMES)
        blue = sum(d["players"][str(p)]["total"] for p in range(1, 6))
        red = sum(d["players"][str(p)]["total"] for p in range(6, 11))
        assert d["curve"][0] == 50.0                       # an even start is a coin flip
        assert blue == pytest.approx(50 if blue_wins else -50, abs=0.3)   # five totals rounded to 0.1
        assert red == pytest.approx(-blue, abs=0.5)
        for v in d["players"].values():                    # the parts make up the total
            assert sum(v[k] for k in swing.PARTS) == pytest.approx(v["total"], abs=0.3)
            assert v["line"][-1] == pytest.approx(v["total"], abs=0.3)   # and the running line ends there


def test_a_herald_nobody_took_is_nobodys_and_the_sums_still_hold():
    frames, events = game()
    # Riot records a summoned herald expiring as killed by team 300.
    events = [*events, {"type": "ELITE_MONSTER_KILL", "timestamp": 15 * 60_000, "killerTeamId": 300,
                        "killerId": 0, "monsterType": "RIFTHERALD"}]
    r = Replay(frames, events)
    assert r.state(16 * 60_000, 100)["heralds_enemy"] == r.state(14 * 60_000, 100)["heralds_enemy"]
    d = swing.game_swing(LeadModel(), frames, events, NAMES)
    assert not any("Herald" in m["text"] for m in d["moments"])
    assert sum(d["players"][str(p)]["total"] for p in range(1, 6)) == pytest.approx(50, abs=0.3)


def test_kills_credit_the_killer_assisters_and_the_dead():
    frames, events = game()
    d = swing.game_swing(LeadModel(), frames, events, NAMES)
    first = next(m for m in d["moments"] if m["clock"] == "3:05")
    assert first["text"] == "C2 killed C6" and first["players"] == [1, 2] and first["lost"] == [6]
    assert first["by"] == 2 and first["after"] > first["before"]
    # Why: the jump is split exactly among what changed.
    assert set(first["why"]) == {"alive", "gold", "kills"}
    assert sum(first["why"].values()) == pytest.approx(first["after"] - first["before"], abs=0.1)
    death = first["death"]
    assert death["timer"] > 0 and death["enemies"] >= 2 and death["zone"]
    i = d["moments"].index(first)
    assert any(b["moment"] == i for b in d["players"]["2"]["best"])
    assert any(w["moment"] == i for w in d["players"]["6"]["worst"])
    assert d["players"]["6"]["deaths"] < 0 and d["players"]["2"]["kills"] > 0
    two, one = (next(b["swing"] for b in d["players"][p]["best"] if b["moment"] == i) for p in ("2", "1"))
    assert two == pytest.approx(one, abs=0.1)              # half to the killer, half to the one assist
    top = swing.top_moments(d, 3)
    assert len(top) == 3 and top[0]["swing"] >= top[1]["swing"] >= top[2]["swing"]
    assert [s[0] for s in d["series"]] == sorted(s[0] for s in d["series"])


def test_losses_while_dead_land_on_the_dead():
    frames, events = game()
    # Blue's 3 dies at 12:00; red takes the mid tower 10 s later, while 3 is still dead.
    events = events + [
        {"type": "CHAMPION_KILL", "timestamp": 720_000, "killerId": 8, "victimId": 3, "assistingParticipantIds": [],
         "position": {"x": 7000, "y": 7000}, "bounty": 300},
        {"type": "BUILDING_KILL", "timestamp": 730_000, "teamId": 100, "killerId": 8, "assistingParticipantIds": [],
         "buildingType": "TOWER_BUILDING", "laneType": "MID_LANE"}]
    d = swing.game_swing(LeadModel(), frames, events, NAMES)
    tower = next(m for m in d["moments"] if m["clock"] == "12:10")
    assert tower["dead"] == [3] and tower["team"] == 200
    death = next(m for m in d["moments"] if m["clock"] == "12:00")
    assert d["moments"].index(tower) in death["death"]["while_dead"] and death["death"]["while_dead_cost"] > 0
    # The same death without the tower: player 3's deaths cost less.
    alone = swing.game_swing(LeadModel(), frames, events[:-1], NAMES)
    assert d["players"]["3"]["deaths"] < alone["players"]["3"]["deaths"]


def test_finished_items_and_the_draft_move_the_chance():
    frames, events = game()
    champions = {p: p for p in range(1, 11)}
    purchases = {4: [(600, 1002)], 9: [(700, 1003)]}
    d = swing.game_swing(ItemModel(catalog()), frames, events, NAMES, champions, purchases)
    assert d["reads_items"] and d["start"] > 50                 # blue drafted the marksmen
    assert d["players"]["1"]["draft"] > 0 > d["players"]["6"]["draft"]
    plate = next(m for m in d["moments"] if m["part"] == "items" and m["by"] == 4)
    assert plate["text"] == "C4 finished Plate" and plate["slot"] == 1 and plate["gain"] > 0
    assert {a["name"] for a in plate["alternatives"]} == {"Edge", "Staff"}
    # Armor against a physical team beat the alternatives in this model.
    assert all(plate["gain"] >= a["swing"] for a in plate["alternatives"])
    assert d["players"]["4"]["items"] > 0
    blue = sum(d["players"][str(p)]["total"] for p in range(1, 6))
    assert blue == pytest.approx(50, abs=0.2)


def test_games_without_a_result_have_no_swing():
    frames, events = game()
    assert swing.game_swing(LeadModel(), frames, [e for e in events if e["type"] != "GAME_END"]) is None


def test_leverage_compares_with_the_typical_moment():
    norms = {"kills": {"0": 2.0, "1": 4.0}}
    assert swing.leverage({"part": "kills", "t": 300_000, "swing": 5.0}, norms) == 2.5
    assert swing.leverage({"part": "kills", "t": 900_000, "swing": 2.0}, norms) == 0.5
    assert swing.leverage({"part": "towers", "t": 0, "swing": 1.0}, norms) is None


def test_first_max_and_enemy_context():
    assert choices.first_max("QWEQQRQQ") == "Q" and choices.first_max("QEWEEREE") == "E"
    assert choices.first_max("QWE") is None
    cat = catalog()
    cat.champions.update({11: ("Tank", 0.3), 12: ("Tank", 0.3), 13: ("Mage", 0.9)})
    assert choices.enemy_context(cat, [11, 12, 3, 4, 5]) == "tanks"
    assert choices.enemy_context(cat, [13, 13, 13, 4, 5]) == "ap_heavy"
    assert choices.enemy_context(cat, [3, 4, 5, 6, 7]) == "ad_heavy"
    assert choices.enemy_context(None, [1]) == "all"


def tiny_dataset(games=400, seed=3, full=False):
    """Made-up games where the team ahead in gold usually wins."""
    rng = np.random.default_rng(seed)
    feats = FULL_FEATURES if full else BASE_FEATURES
    xs, ys, gs, ms, own, enemy = [], [], [], [], [], []
    for g in range(games):
        drift = rng.normal(0, 1.5)
        won = float(rng.random() < 1 / (1 + math.exp(-drift)))
        champs = rng.choice(np.arange(1, 11), 10, replace=False)
        for minute in range(1, 21):
            for side, sign in ((1.0, 1), (0.0, -1)):
                row = dict.fromkeys(feats, 0.0)
                row["minute"] = minute
                row["gold_diff"] = sign * (drift * minute / 4 + rng.normal(0, 0.5))
                row["alive_own"] = row["alive_enemy"] = 5
                xs.append([row[k] for k in feats])
                ys.append(won if side else 1 - won)
                gs.append(g)
                ms.append(minute)
                own.append(champs[:5] if side else champs[5:])
                enemy.append(champs[5:] if side else champs[:5])
    data = {"X": np.asarray(xs, dtype=np.float32), "y": np.asarray(ys, dtype=np.float32), "games": np.asarray(gs),
            "minutes": np.asarray(ms, dtype=np.float32)}
    if full:
        data["own"], data["enemy"] = np.asarray(own, dtype=np.int16), np.asarray(enemy, dtype=np.int16)
    return data


def tiny_model(models_dir, full=False):
    model, metrics = wp.train(tiny_dataset(full=full), catalog() if full else None, log=lambda *_: None, layers=(8,))
    wp.save(model, metrics, models_dir)
    return model, metrics


def test_the_network_trains_saves_and_reads_both_sides(tmp_path):
    model, metrics = tiny_model(tmp_path)
    assert metrics["network"]["auc"] > 0.6 and metrics["baseline"]["auc"] > 0.6
    assert metrics["calibration"] and metrics["network"]["layers"] == [8] and metrics["chosen"] == "state"
    loaded = wp.load(tmp_path)
    assert loaded is not None and loaded.stamp and wp.metrics(tmp_path)["games"] == 400
    even = dict.fromkeys(BASE_FEATURES, 0.0)
    assert float(loaded.blue([even], [even])[0]) == pytest.approx(0.5)
    ahead, behind = dict(even, gold_diff=5.0, minute=20), dict(even, gold_diff=-5.0, minute=20)
    assert float(loaded.blue([ahead], [behind])[0]) > 0.6
    assert wp.load(tmp_path / "nothing") is None and wp.metrics(tmp_path / "nothing") is None


def test_all_three_versions_train_and_the_best_is_kept(tmp_path):
    model, metrics = tiny_model(tmp_path, full=True)
    assert [v["name"] for v in metrics["variants"]] == ["state", "state_plus", "items", "champions", "stacked",
                                                        "stacked_plus"]
    chosen = next(v for v in metrics["variants"] if v["name"] == metrics["chosen"])
    assert chosen["selection_log_loss"] == min(v["selection_log_loss"] for v in metrics["variants"])
    with_champs = next(v for v in metrics["variants"] if v["name"] == "champions")
    assert with_champs["inputs"] == len(FULL_FEATURES) + 20
    # A champion-aware model reads states with the champions in them.
    if metrics["chosen"] in ("champions", "stacked"):
        assert model.reads_items
    frames, events = game()
    loaded = wp.load(tmp_path)
    d = swing.game_swing(loaded, frames, events, NAMES, {p: p for p in range(1, 11)}, {4: [(600, 1002)]})
    assert sum(d["players"][str(p)]["total"] for p in range(1, 6)) == pytest.approx(50, abs=0.2)


def test_habits_count_costly_deaths_and_role_norms():
    from riftwatch.winprob import habits

    frames, events = game()
    d = swing.game_swing(LeadModel(), frames, events, NAMES)
    found = habits.deaths(d, 3)                      # blue's 3 dies twice in the fixture game
    assert len(found) == 2 and all(cost > 0 for _, cost, _ in found)
    norms = habits.role_norms([("MIDDLE", d, 3), ("MIDDLE", d, 3), ("TOP", d, 1)])
    mid = norms["MIDDLE"]
    assert mid["games"] == 2 and mid["deaths"] == 2.0
    assert mid["cost"] == pytest.approx(sum(c for _, c, _ in found), abs=0.01)
    assert set(mid) >= {key for key, *_ in habits.HABITS}
    assert swing.group_name("item:armor") == "armor (vs an average item)" and swing.group_name("alive") == "a player down"


def test_shares_add_up_exactly_and_are_zero_sum():
    # Efficiency and zero-sum hold exactly (before rounding), under every set of rules.
    frames, events = game()
    positions = {1: "TOP", 2: "JUNGLE", 3: "MIDDLE", 4: "BOTTOM", 5: "UTILITY",
                 6: "TOP", 7: "JUNGLE", 8: "MIDDLE", 9: "BOTTOM", 10: "UTILITY"}
    data = swing.evaluate(LeadModel(), frames, events, NAMES, positions=positions)
    for rules in (swing.RULES, swing.FIXED_RULES, swing.Rules(team_share=0.9, dead_share=0.1)):
        credit, _, gains = swing.credit_of(data["facts"], rules)
        blue = sum(sum(credit[p].values()) for p in range(1, 6))
        red = sum(sum(credit[p].values()) for p in range(6, 11))
        assert blue == pytest.approx(0.5, abs=1e-9) and red == pytest.approx(-0.5, abs=1e-9)
        for p in range(1, 11):                             # the running line ends at the total
            assert sum(gains[p]) == pytest.approx(sum(credit[p].values()), abs=1e-9)


def test_an_objectives_team_part_goes_to_the_wards_that_set_it_up():
    wards = swing.ward_times([
        {"type": "WARD_PLACED", "timestamp": 500_000, "creatorId": 5, "wardType": "YELLOW_TRINKET"},
        {"type": "WARD_PLACED", "timestamp": 550_000, "creatorId": 5, "wardType": "CONTROL_WARD"},
        {"type": "WARD_KILL", "timestamp": 560_000, "killerId": 4, "wardType": "YELLOW_TRINKET"},
        {"type": "WARD_PLACED", "timestamp": 570_000, "creatorId": 5, "wardType": "UNDEFINED"},      # not a ward
        {"type": "WARD_PLACED", "timestamp": 400_000, "creatorId": 3, "wardType": "YELLOW_TRINKET"},  # too early
        {"type": "WARD_PLACED", "timestamp": 575_000, "creatorId": 7, "wardType": "YELLOW_TRINKET"}])  # the enemy's
    assert swing.vision_before(wards, 600_000, 100) == {"4": 1, "5": 2}
    assert swing.vision_before(wards, 600_000, 200) == {"7": 1}

    facts = {"winner": 100, "times": [0, 60_000], "curve": [0.5, 0.6], "earned": {str(p): [0] for p in range(1, 11)},
             "opponent": {}, "events": [{"t": 30_000, "d": 0.1, "side": 100, "kind": "objectives", "got": [2],
                                         "dead": [], "vision": {"4": 1, "5": 3}}]}
    seen, plays, _ = swing.credit_of(facts, swing.RULES)
    even, _, _ = swing.credit_of(facts, replace(swing.RULES, objective_team="even"))
    # half to the jungler who took it; the team's half by the wards, 3 : 1, and it's their play
    assert seen[2]["objectives"] == pytest.approx(0.05) and seen[5]["objectives"] == pytest.approx(0.0375)
    assert seen[4]["objectives"] == pytest.approx(0.0125) and seen[1]["objectives"] == 0.0
    assert (pytest.approx(0.0375), 0) in plays[5]
    assert even[5]["objectives"] == pytest.approx(0.01) and even[2]["objectives"] == pytest.approx(0.06)
    for credit in (seen, even):                              # still exactly +50 and -50
        assert sum(sum(credit[p].values()) for p in range(1, 6)) == pytest.approx(0.5, abs=1e-12)
        assert sum(sum(credit[p].values()) for p in range(6, 11)) == pytest.approx(-0.5, abs=1e-12)
    nobody = {**facts, "events": [{**facts["events"][0], "vision": {}}]}   # nobody warded: evenly
    assert swing.credit_of(nobody, swing.RULES)[0][5]["objectives"] == pytest.approx(0.01)

    # evaluate records who warded an objective up, from the taking team only
    frames, events = game()
    events = [e for e in events if e["type"] not in ("WARD_PLACED", "WARD_KILL")]
    t = 9 * 60_000 + 30_000
    events = sorted([*events,
                     {"type": "WARD_PLACED", "timestamp": t - 30_000, "creatorId": 5, "wardType": "YELLOW_TRINKET"},
                     {"type": "WARD_KILL", "timestamp": t - 10_000, "killerId": 4, "wardType": "YELLOW_TRINKET"},
                     {"type": "WARD_PLACED", "timestamp": t - 5_000, "creatorId": 8, "wardType": "CONTROL_WARD"},
                     {"type": "ELITE_MONSTER_KILL", "timestamp": t, "killerId": 2, "killerTeamId": 100,
                      "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON"}], key=lambda e: e.get("timestamp", 0))
    data = swing.evaluate(LeadModel(), frames, events, NAMES)
    fact = next(f for f in data["facts"]["events"] if f["kind"] == "objectives" and f["t"] == t)
    assert fact["vision"] == {"4": 1, "5": 1}
    assert next(m for m in data["moments"] if m["t"] == t and m["part"] == "objectives")["vision"] == {"4": 1, "5": 1}


def test_a_calibrator_keeps_the_bends_it_was_fitted_with(monkeypatch):
    rng = np.random.default_rng(0)
    raw, minute = rng.uniform(0.05, 0.95, 4000), rng.uniform(0, 40, 4000)
    y = (rng.random(4000) < raw).astype(float)
    new = wp.PhasePlatt().fit(raw, y, minute)
    assert new.knots == wp.KNOTS and 2.0 in new.knots      # the opening minutes get their own curve
    monkeypatch.setattr(wp, "KNOTS", wp.WP3_KNOTS)
    old = wp.PhasePlatt().fit(raw, y, minute)
    del old.knots                                          # as a calibrator saved before wp4 is
    monkeypatch.undo()
    p = old.predict(raw[:50], minute[:50])                 # still reads with the bends it was fitted with
    assert p.shape == (50,) and np.all((p > 0) & (p < 1))


def test_momentum_is_how_far_the_gaps_moved_lately():
    frames, events = game()
    r = Replay(frames, events)
    t = 12 * 60_000 + 30_000
    blue, red = r.pair(t)
    three_ago, _ = r.pair(t - 180_000)
    two_ago, _ = r.pair(t - 120_000)
    assert blue["kills_moved"] == ((blue["kills_own"] - blue["kills_enemy"])
                                   - (three_ago["kills_own"] - three_ago["kills_enemy"]))
    assert blue["towers_moved"] == ((blue["towers_own"] - blue["towers_enemy"])
                                    - (three_ago["towers_own"] - three_ago["towers_enemy"]))
    assert blue["gold_moved"] == pytest.approx(blue["gold_diff"] - two_ago["gold_diff"])
    assert blue["xp_moved"] == pytest.approx(blue["xp_diff"] - two_ago["xp_diff"])
    for key in ("gold_moved", "xp_moved", "kills_moved", "towers_moved"):     # the same seen from red
        assert red[key] == pytest.approx(-blue[key])
    early, _ = r.pair(60_000)                                   # nothing to look back on yet
    assert early["gold_moved"] == pytest.approx(early["gold_diff"] - r.pair(0)[0]["gold_diff"])


def test_the_swing_page_lists_momentum_only_for_models_that_read_it():
    from riftwatch.web import swingview

    weights = {"items_diff": 0.03, "armor vs physical": 0.12, "tanks_diff": 0.08}
    before = swingview.context_view(weights)
    assert not any(g["title"].startswith("Momentum") for g in before)
    after = swingview.context_view({**weights, "gold_moved (late)": 0.1033, "towers_moved (late)": -0.0183})
    rows = next(g for g in after if g["title"].startswith("Momentum"))["rows"]
    assert [(r["value"], r["pts"]) for r in rows] == [(0.103, 2.6), (-0.018, -0.5)]


def test_resampled_networks_draw_whole_games_and_their_spread_is_a_moments_uncertainty():
    games_ = np.array([0, 0, 1, 1, 1, 2, 3, 3])
    rows = wp._resample(games_, 4)
    drawn = Counter(games_[rows].tolist())
    for g, size in Counter(games_.tolist()).items():
        assert drawn.get(g, 0) % size == 0                     # a game is drawn whole or not at all
    assert wp._resample(games_, 4).tolist() == rows.tolist() and len(rows) > 0     # the same seed, the same draw

    class TwoNetworks(LeadModel):
        """Two networks that agree on towers and gold but not on what a kill is worth."""

        def blue_members(self, blue_states, red_states):
            def side(s, k):
                z = 0.4 * s["gold_diff"] + k * (s["kills_own"] - s["kills_enemy"]) + 0.5 * (s["towers_own"] - s["towers_enemy"])
                return 1 / (1 + math.exp(-z))
            return np.array([[(side(b, k) + 1 - side(r, k)) / 2 for b, r in zip(blue_states, red_states, strict=True)]
                             for k in (0.2, 0.4)])

    frames, events = game()
    with_spread = swing.evaluate(TwoNetworks(), frames, events, NAMES)
    kills = [m for m in with_spread["moments"] if m["part"] == "kills"]
    towers = [m for m in with_spread["moments"] if m["part"] == "structures"]
    # they disagree about kills, and about towers only through where the game already stood
    assert kills and towers and all(m["spread"] > 0 for m in kills)
    assert np.mean([m["spread"] for m in kills]) > 2 * np.mean([m["spread"] for m in towers])
    assert all("spread" not in m for m in swing.evaluate(LeadModel(), frames, events, NAMES)["moments"])


def test_kills_are_shared_by_damage_and_economy_by_the_lane():
    frames, events = game()
    first = next(e for e in events if e.get("type") == "CHAMPION_KILL")
    # The assister (1) did most of the damage, the killer (2) the last hit.
    first["victimDamageReceived"] = [{"participantId": 1, "physicalDamage": 900}, {"participantId": 2, "physicalDamage": 100}]
    positions = {1: "TOP", 2: "JUNGLE", 3: "MIDDLE", 4: "BOTTOM", 5: "UTILITY",
                 6: "TOP", 7: "JUNGLE", 8: "MIDDLE", 9: "BOTTOM", 10: "UTILITY"}
    data = swing.evaluate(LeadModel(), frames, events, NAMES, positions=positions)
    i = next(n for n, f in enumerate(data["facts"]["events"]) if f["kind"] == "kills")
    one = {**data["facts"], "events": [data["facts"]["events"][i]]}
    # the kill's own shares, before any farm it cost is charged (tests/test_farm.py)
    measured, _, _ = swing.credit_of(one, replace(swing.RULES, farm_cost=False))
    fixed, _, _ = swing.credit_of(one, replace(swing.FIXED_RULES, farm_cost=False))
    # half by damage (0.9 / 0.1), half equally: 0.7 / 0.3 of the kill; fixed rules: 0.5 / 0.5
    assert measured[1]["kills"] / measured[2]["kills"] == pytest.approx(0.7 / 0.3, rel=1e-6)
    assert fixed[1]["kills"] == pytest.approx(fixed[2]["kills"])
    assert data["facts"]["opponent"]["1"] == 6 and data["facts"]["opponent"]["9"] == 4


def test_the_rating_shrinks_towards_the_role_by_the_measured_spread():
    from riftwatch.winprob import validate

    model = {"tau": 2.0, "sigma": 14.0}                    # k = 49 games
    few, many = validate.rating(10.0, 5, model), validate.rating(10.0, 500, model)
    assert few["value"] == pytest.approx(10 * 5 / 54, abs=0.05) and many["value"] == pytest.approx(10 * 500 / 549, abs=0.05)
    assert few["sd"] > many["sd"] and few["raw"] == 10.0


def test_reliability_finds_real_differences_and_not_noise():
    import random
    from datetime import datetime, timedelta

    from riftwatch.winprob import validate

    rng = random.Random(4)
    rows = []
    for p in range(120):
        level = rng.gauss(0, 3)                            # true differences between players
        for j in range(30):
            rows.append({"puuid": f"p{p}", "t": datetime(2026, 1, 1) + timedelta(hours=j), "role": "MIDDLE",
                         "swing": level + rng.gauss(0, 10), "noise": rng.gauss(0, 10), "win": float(rng.random() < 0.5)})
    validate._role_adjust_one(rows, "swing")
    validate._role_adjust_one(rows, "noise")
    for r in rows:
        r["win_adj"] = r["win"]
    out = validate.reliability_and_prediction(rows, (("swing", "Swing"), ("noise", "Noise"), ("win", "Win")))
    by = {m["key"]: m for m in out["metrics"]}
    # Spearman-Brown on 30 games: 30 * 9 / (30 * 9 + 100) = 0.73 for the real signal
    assert by["swing"]["reliability"] > 0.55 and abs(by["noise"]["reliability"]) < 0.3
    assert out["players"] == 120 and "prediction_ceiling" in out
    fit = validate.rating_model(rows)
    assert 1.5 < fit["tau"] < 4.5 and 9 < fit["sigma"] < 11


def test_points_never_come_out_as_negative_zero():
    # Postgres stores -0.0 as 0, so a fresh result must say +0.0 too, or cached coaching
    # (keyed by its evidence text) misses on the second request.
    import math

    assert str(swing._pts(-0.0004)) == "0.0" and math.copysign(1, swing._pts(-0.0004)) == 1
    assert swing._pts(0.01234) == 1.2 and swing._pts(-0.01234) == -1.2


def test_vision_and_objective_timers_are_read_from_each_side():
    frames, events = game()
    events = [e for e in events if e["type"] not in ("WARD_PLACED", "WARD_KILL")]    # only the wards below
    wards = [{"type": "WARD_PLACED", "timestamp": 60_000 * m, "creatorId": 1, "wardType": "YELLOW_TRINKET"}
             for m in (5, 6, 7)]
    wards += [{"type": "WARD_PLACED", "timestamp": 400_000, "creatorId": 7, "wardType": "CONTROL_WARD"},
              {"type": "WARD_KILL", "timestamp": 410_000, "killerId": 7, "wardType": "YELLOW_TRINKET"},
              {"type": "WARD_PLACED", "timestamp": 420_000, "creatorId": 8, "wardType": "UNDEFINED"}]
    r = Replay(frames, [*events, *wards])
    blue, red = r.pair(7.5 * 60_000)
    assert blue["wards_own"] == 3 and red["wards_enemy"] == 3       # placed in the last 3 minutes
    assert red["control_wards_own"] == 1 and blue["control_wards_enemy"] == 1
    assert red["wards_cleared_own"] == 1 and red["wards_own"] == 0  # an undefined ward isn't counted
    assert r.pair(8.5 * 60_000)[0]["wards_own"] == 2                # the 5:00 ward is out of the window
    # Dragon spawns at 5:00 and is taken at 9:20: 3 minutes away at 2:00, up at 6:00, 4:20 away at 10:00.
    assert r.pair(2 * 60_000)[0]["dragon_in"] == pytest.approx(3.0)
    assert r.pair(6 * 60_000)[0]["dragon_in"] == 0.0
    assert r.pair(10 * 60_000)[0]["dragon_in"] == pytest.approx(4 + 20 / 60, abs=1e-6)
    assert r.pair(10 * 60_000)[0]["baron_in"] == 6.0 and r.pair(10 * 60_000)[0]["elder_in"] == 6.0
    # Swing ignores wards: the same moments, and the soul still comes with its dragon.
    d = swing.game_swing(LeadModel(), frames, [*events, *wards], NAMES)
    assert [m["text"] for m in d["moments"]] == [m["text"] for m in swing.game_swing(LeadModel(), frames, events, NAMES)["moments"]]


def test_the_elder_is_read_from_its_dragon_subtype_and_buffs_go_to_the_living():
    from tests.fixtures import kill, monster

    frames, events = game()
    # Red's 9 dies at 24:50, red takes the Elder at 25:00 (the timeline files it as a dragon
    # with an ELDER_DRAGON subtype), red's 10 dies at 25:30.
    extra = [kill(1_490_000, 1, 9), monster(1_500_000, 7, "DRAGON", "ELDER_DRAGON", assists=[8]),
             kill(1_530_000, 2, 10)]
    r = Replay(frames, [*events, *extra])
    blue, red = r.pair(1_510_000)
    assert red["elder_own"] == 4 and blue["elder_enemy"] == 4          # the dead 9 didn't get it
    assert red["dragons_own"] == r.pair(1_480_000)[1]["dragons_own"]   # the Elder isn't one of the dragons
    assert r.pair(1_540_000)[1]["elder_own"] == 3                      # 10 died and lost it
    assert r.pair(1_500_000 + 151_000)[1]["elder_own"] == 0            # over after 150 seconds
    d = swing.game_swing(LeadModel(), frames, [*events, *extra], NAMES)
    assert any(m["text"] == "Red took Elder Dragon" for m in d["moments"])


def test_the_richer_network_leaves_out_vision_and_atakhan():
    feats = {name: f for name, f, _ in wp.VARIANTS}["state_plus"]
    assert not set(feats) & {"wards_own", "control_wards_enemy", "wards_cleared_own", "atakhan_own", "atakhan_enemy"}
    assert {"baron_own", "baron_in", "lane_gold_mid", "towers_outer_own"} <= set(feats)


def _calibration_data(n=40_000, seed=0):
    """A network whose output is right before 30 minutes and overconfident after."""
    rng = np.random.default_rng(seed)
    minute = rng.uniform(1, 40, n)
    z = rng.normal(0, 2, n)
    slope = np.where(minute < 30, 1.0, 0.6)
    y = (rng.random(n) < 1 / (1 + np.exp(-slope * z))).astype(float)
    return 1 / (1 + np.exp(-z)), y, minute


def _loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def test_the_calibration_can_change_with_the_clock_and_always_increases():
    raw, y, minute = _calibration_data()
    tr, te = slice(0, 20_000), slice(20_000, None)
    platt = wp.Platt().fit(raw[tr], y[tr])
    phase = wp.PhasePlatt().fit(raw[tr], y[tr], minute[tr])
    assert phase.fallback is None
    assert _loss(phase.predict(raw[te], minute[te]), y[te]) < _loss(platt.predict(raw[te]), y[te]) - 0.002
    grid = np.linspace(0.01, 0.99, 50)
    for m in (5, 15, 25, 35, 60):
        assert (np.diff(phase.predict(grid, np.full(50, m))) > 0).all()
    beta = wp.Beta().fit(raw[tr], y[tr])
    assert (np.diff(beta.predict(grid)) > 0).all()
    # By group: one group's network is overconfident all game, the other's isn't.
    group = np.where(np.arange(len(y)) % 2 == 0, "A", "B")
    rng = np.random.default_rng(1)
    z = np.log(raw / (1 - raw))
    yg = (rng.random(len(y)) < 1 / (1 + np.exp(-np.where(group == "A", 1.0, 0.5) * z))).astype(float)
    by_group = wp.GroupPhasePlatt().fit(raw[tr], yg[tr], minute[tr], group[tr])
    plain = wp.PhasePlatt().fit(raw[tr], yg[tr], minute[tr])
    assert (_loss(by_group.predict(raw[te], minute[te], group[te]), yg[te])
            < _loss(plain.predict(raw[te], minute[te]), yg[te]) - 0.005)


def test_a_model_calibrated_by_the_clock_reads_the_minute(tmp_path):
    model, metrics = wp.train(tiny_dataset(), None, log=lambda *_: None, layers=(8,), calibrator=wp.PhasePlatt)
    assert metrics["calibrator"] == "PhasePlatt"
    wp.save(model, metrics, tmp_path)
    loaded = wp.load(tmp_path)
    even = dict.fromkeys(BASE_FEATURES, 0.0)
    ahead, behind = dict(even, gold_diff=5.0, minute=20), dict(even, gold_diff=-5.0, minute=20)
    assert float(loaded.blue([ahead], [behind])[0]) > 0.6
    assert float(loaded.blue([dict(even, minute=10)], [dict(even, minute=10)])[0]) == pytest.approx(0.5)


def test_cross_validation_tests_every_game_once_and_compares_calibrations():
    data = tiny_dataset(games=240)
    folds = wp._folds(data["games"], 3)
    for g in np.unique(data["games"]):
        assert len(set(folds[data["games"] == g].tolist())) == 1                  # a game stays together
    out = wp.cross_validate(data, None, [("state", BASE_FEATURES, False)], folds=3,
                            calibrations={"platt": wp.Platt, "phase": wp.PhasePlatt}, layers=(8,),
                            log=lambda *_: None)
    assert out["games"] == 240 and out["folds"] == 3 and out["first"] == ["state", "platt"]
    assert [(r["version"], r["calibration"]) for r in out["results"]] == [("state", "platt"), ("state", "phase")]
    assert out["results"][0]["auc"] > 0.6 and "vs_first" in out["results"][1]
    assert set(out["results"][0]["ece_by_minute"]) == {"0-10", "10-20"}


def test_the_sanity_check_counts_takes_that_lower_the_takers_chance():
    from riftwatch.winprob import sanity

    frames, events = game()
    r = Replay(frames, events)
    summary = sanity.summarise(sanity.game_jumps(LeadModel(), r))
    assert {"kill", "tower", "Dragon", "Baron", "Voidgrubs"} <= set(summary)
    assert summary["kill"]["wrong_way"] == 0 and summary["tower"]["wrong_way"] == 0 and summary["kill"]["mean"] > 0

    class TowersHurt(LeadModel):
        def blue(self, blue_states, red_states):
            def flip(s):
                return dict(s, towers_own=-s["towers_own"], towers_enemy=-s["towers_enemy"])
            return super().blue([flip(s) for s in blue_states], [flip(s) for s in red_states])
    assert sanity.check(TowersHurt(), [(frames, events, None, None, None)])["tower"]["wrong_way"] == 1.0
