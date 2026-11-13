"""The win-probability state, network and the Swing metric built on it."""

import math
from collections import Counter

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
    # Baron at 20:00: the buff is up at 21:00 and gone by 24:00; the take still counts.
    assert r.state(21 * 60_000, 100)["baron_own"] == 1
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
        assert blue == pytest.approx(50 if blue_wins else -50, abs=0.2)
        assert red == pytest.approx(-blue, abs=0.2)
        for v in d["players"].values():                    # the parts make up the total
            assert sum(v[k] for k in swing.PARTS) == pytest.approx(v["total"], abs=0.3)
            assert v["line"][-1] == pytest.approx(v["total"], abs=0.3)   # and the running line ends there


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
    assert [v["name"] for v in metrics["variants"]] == ["state", "items", "champions", "stacked"]
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
