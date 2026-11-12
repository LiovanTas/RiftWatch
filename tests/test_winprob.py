"""The win-probability state, network and the Swing metric built on it."""

import math

import numpy as np
import pytest

from riftwatch.winprob import model as wp
from riftwatch.winprob import swing
from riftwatch.winprob.state import FEATURES, Replay, compact, respawn_seconds
from tests.fixtures import build_game


class LeadModel:
    """Stands in for the network: blue's chance follows gold, kills and towers."""

    stamp = "test"

    def blue(self, blue_states, red_states):
        def side(s):
            z = 0.4 * s["gold_diff"] + 0.3 * (s["kills_own"] - s["kills_enemy"]) + 0.5 * (s["towers_own"] - s["towers_enemy"])
            return 1 / (1 + math.exp(-z))
        return np.array([(side(b) + 1 - side(r)) / 2 for b, r in zip(blue_states, red_states, strict=True)])


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
    assert respawn_seconds(18, 40) > respawn_seconds(18, 10) > respawn_seconds(1, 10)


def test_team_swing_adds_up_to_the_result():
    for blue_wins in (True, False):
        frames, events = game(blue_wins)
        d = swing.game_swing(LeadModel(), frames, events, {i: f"C{i}" for i in range(1, 11)})
        blue = sum(d["players"][str(p)]["total"] for p in range(1, 6))
        red = sum(d["players"][str(p)]["total"] for p in range(6, 11))
        assert d["curve"][0] == 50.0                       # an even start is a coin flip
        assert blue == pytest.approx(50 if blue_wins else -50, abs=0.2)
        assert red == pytest.approx(-blue, abs=0.2)
        for v in d["players"].values():                    # the parts make up the total
            assert sum(v[k] for k in swing.PARTS) == pytest.approx(v["total"], abs=0.3)


def test_kills_credit_the_killer_assisters_and_the_dead():
    frames, events = game()
    d = swing.game_swing(LeadModel(), frames, events, {i: f"C{i}" for i in range(1, 11)})
    first = next(m for m in d["moments"] if m["clock"] == "3:05")
    assert first["text"] == "C2 killed C6" and first["players"] == [1, 2] and first["lost"] == [6]
    assert first["after"] > first["before"]                # good for blue
    # The killer's best play is that kill; the victim's worst moment is the same one.
    i = d["moments"].index(first)
    assert any(b["moment"] == i for b in d["players"]["2"]["best"])
    assert any(w["moment"] == i for w in d["players"]["6"]["worst"])
    two, one = (next(b["swing"] for b in d["players"][p]["best"] if b["moment"] == i) for p in ("2", "1"))
    assert two == pytest.approx(one, abs=0.1)              # half to the killer, half to the one assist
    top = swing.top_moments(d, 3)
    assert len(top) == 3 and top[0]["swing"] >= top[1]["swing"] >= top[2]["swing"]
    assert [s[0] for s in d["series"]] == sorted(s[0] for s in d["series"])


def test_games_without_a_result_have_no_swing():
    frames, events = game()
    assert swing.game_swing(LeadModel(), frames, [e for e in events if e["type"] != "GAME_END"]) is None


def tiny_dataset(games=400, seed=3):
    """Made-up games where the team ahead in gold usually wins."""
    rng = np.random.default_rng(seed)
    xs, ys, gs, ms = [], [], [], []
    for g in range(games):
        drift = rng.normal(0, 1.5)
        won = float(rng.random() < 1 / (1 + math.exp(-drift)))
        for minute in range(1, 21):
            for side, sign in ((1.0, 1), (0.0, -1)):
                row = dict.fromkeys(FEATURES, 0.0)
                row["minute"] = minute
                row["gold_diff"] = sign * (drift * minute / 4 + rng.normal(0, 0.5))
                row["alive_own"] = row["alive_enemy"] = 5
                xs.append([row[k] for k in FEATURES])
                ys.append(won if side else 1 - won)
                gs.append(g)
                ms.append(minute)
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            np.asarray(gs), np.asarray(ms, dtype=np.float32))


def tiny_model(models_dir):
    X, y, g, m = tiny_dataset()
    model, metrics = wp.train(X, y, g, m, log=lambda *_: None, layers=(8,))
    wp.save(model, metrics, models_dir)
    return model, metrics


def test_the_network_trains_saves_and_reads_both_sides(tmp_path):
    model, metrics = tiny_model(tmp_path)
    assert metrics["network"]["auc"] > 0.6 and metrics["baseline"]["auc"] > 0.6
    assert metrics["calibration"] and metrics["network"]["layers"] == [8]
    loaded = wp.load(tmp_path)
    assert loaded is not None and loaded.stamp and wp.metrics(tmp_path)["games"] == 400
    even = dict.fromkeys(FEATURES, 0.0)
    assert float(loaded.blue([even], [even])[0]) == pytest.approx(0.5)
    ahead, behind = dict(even, gold_diff=5.0, minute=20), dict(even, gold_diff=-5.0, minute=20)
    assert float(loaded.blue([ahead], [behind])[0]) > 0.6
    assert wp.load(tmp_path / "nothing") is None and wp.metrics(tmp_path / "nothing") is None
