"""Pre-game win prediction: the draft from the model at minute 0, players' Swing ratings and
comfort, combined in log-odds."""

import math
import os

import numpy as np
import pytest

from riftwatch.web import pregame

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


def player(team, value, sd=1.0, known=True, games=0, name="P"):
    return {"team": team, "name": name, "champion": "", "icon": "", "champion_games": games,
            "rating": {"value": value, "sd": sd, "known": known, "games": 20 if known else 0, "role": "Mid"}}


def test_equal_teams_are_even_and_ratings_add_up_in_points():
    even = pregame.predict([player(100, 2.0), player(200, 2.0)], None, None)
    assert even["blue"] == 50.0 and even["parts"]["players"] == 0.0
    # Blue's ratings add up to 5 more points: about 5 points more likely to win.
    ahead = pregame.predict([player(100, 3.0), player(100, 2.0), player(200, 0.0)], None, None)
    assert ahead["blue"] == pytest.approx(54.9, abs=0.3) and ahead["rating_diff"] == 5.0
    assert ahead["low"] < ahead["blue"] < ahead["high"]
    assert [p["rating"]["value"] for p in ahead["best"]] == [3.0, 2.0]


def test_the_draft_and_comfort_move_the_chance():
    with_draft = pregame.predict([player(100, 0.0), player(200, 0.0)], 0.55, None)
    assert with_draft["blue"] == pytest.approx(55.0, abs=0.05) and with_draft["parts"]["draft"] == pytest.approx(5.0)
    comp = {"comfort": {"0": (1000, 470), "30+": (1000, 540)}, "overall": 0.5}
    first_time = pregame.predict([player(100, 0.0, games=0), player(200, 0.0, games=50)], None, comp)
    assert first_time["blue"] < 50 and first_time["parts"]["comfort"] < 0
    # Half of the experience effect counts (the rating already holds the rest).
    full = pregame.logit(0.54) - pregame.logit(0.47)
    assert pregame.logit(first_time["blue"] / 100) == pytest.approx(-pregame.COMFORT_WEIGHT * full, abs=1e-3)


def test_unknown_players_are_average_with_a_wide_margin():
    sure = pregame.predict([player(t, 0.0, sd=1.0) for t in (100,) * 5 + (200,) * 5], None, None)
    guess = pregame.predict([player(t, 0.0, sd=3.0, known=False) for t in (100,) * 5 + (200,) * 5], None, None)
    assert sure["sd"] < guess["sd"] and guess["known"] == 0 and guess["sure"].startswith("a guess")
    assert guess["high"] - guess["low"] > sure["high"] - sure["low"]


class CompModel:
    """Reads only the comp inputs, like the stacked model's context layer: marksmen help."""

    reads_items = True

    def __init__(self):
        from riftwatch.winprob.state import FULL_FEATURES, Catalog

        self.features = FULL_FEATURES + ("some_future_input",)
        self.catalog = Catalog()
        self.catalog.champions = {1: ("Marksman", 0.1), 2: ("Mage", 0.9), 3: ("Tank", 0.2)}
        self.seen = []

    def blue(self, blue_states, red_states):
        self.seen.extend(blue_states + red_states)

        def f(s):
            return 1 / (1 + math.exp(-(0.3 * (s["marksmans_own"] - s["marksmans_enemy"]))))
        return np.array([(f(b) + 1 - f(r)) / 2 for b, r in zip(blue_states, red_states, strict=True)])


def test_the_draft_is_the_model_at_minute_zero():
    model = CompModel()
    p = pregame.draft_chance(model, [1, 1, 3], [2, 2, 3])
    assert p == pytest.approx(1 / (1 + math.exp(-0.6)), abs=1e-9)
    state = model.seen[0]
    assert state["minute"] == 0 and state["alive_own"] == 5 and state["gold_diff"] == 0
    assert state["some_future_input"] == 0.0 and state["_own"] == [1, 1, 3, 0, 0]
    assert pregame.draft_chance(None, [1], [2]) is None


def test_auc_counts_winners_ranked_above_losers():
    assert pregame._auc([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0]) == 1.0
    assert pregame._auc([0.1, 0.2, 0.8, 0.9], [1.0, 1.0, 0.0, 0.0]) == 0.0
    assert pregame._auc([0.5, 0.5], [1.0, 0.0]) == 0.5


@pytest.mark.skipif(not TEST_DB, reason="RIFTWATCH_TEST_DATABASE_URL not set")
def test_live_and_lobby_predictions_from_the_database(tmp_path):
    from riftwatch.db import migrate
    from riftwatch.db.connection import connect
    from riftwatch.web.gamedata import GameData

    with connect(TEST_DB) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(conn)
        gd = GameData()
        gd.loaded = True
        report = {"me": "p1", "players": [
            {"puuid": f"p{i}", "team_id": 100 if i <= 5 else 200, "riot_id": f"Player{i}#NA1",
             "champion": "", "champion_games": i} for i in range(1, 11)]}
        out = pregame.live(conn, gd, report, CompModel(), tmp_path / "models", me="p1")
        # Nobody's rated (no stored games): an even game, said to be a guess.
        assert out["blue"] == 50.0 and out["known"] == 0 and out["my_team"] == 100
        assert pregame.lobby(conn, gd, [("Nobody", "NA1")], CompModel(), tmp_path / "models") is None
        result = pregame.backtest(conn, CompModel(), {"rating": {"tau": 3.0, "sigma": 15.0}, "role_means": {}})
        assert result["enough"] is False
