"""The profile's "Your plan": focus, target, trend and examples, built on Swing."""

import os
from datetime import datetime, timedelta

import pytest

from riftwatch.web import plan

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
HABIT_NORMS = {"JUNGLE": {"games": 100, "deaths": 5.0, "cost": 20.0,
                          **{k: {"per_game": 4.0, "share": 0.2} for k in ("while_dead", "outnumbered", "held_gold",
                                                                           "deep", "late")}},
               "MIDDLE": {"games": 100, "deaths": 4.0, "cost": 16.0,
                          **{k: {"per_game": 2.0, "share": 0.1} for k in ("while_dead", "outnumbered", "held_gold",
                                                                           "deep", "late")}}}
PART_NORMS = {"JUNGLE": {"kills": 8.0, "deaths": -10.0, "objectives": 2.0, "structures": 0.5, "economy": -1.0},
              "MIDDLE": {"kills": 9.0, "deaths": -9.0, "objectives": 0.5, "structures": 0.5, "economy": 0.0}}


def game(i, role="JUNGLE", while_dead=(0, 0.0), kills=8.0, deaths=-10.0):
    habits = {k: [0, 0.0] for k in ("while_dead", "outnumbered", "held_gold", "deep", "late")}
    habits["while_dead"] = list(while_dead)
    return {"match_id": f"NA1_{i}", "t": datetime(2026, 11, 1) + timedelta(hours=i), "role": role, "champion": 1,
            "win": i % 2 == 0, "parts": {"kills": kills, "deaths": deaths, "objectives": 2.0, "structures": 0.5,
                                         "economy": -1.0},
            "habits": habits, "deaths": []}


def test_the_focus_is_what_costs_most_beyond_the_role():
    # Two deaths a game that cost an objective, 6 points each: 12 a game against 4 typical.
    games = [game(i, while_dead=(2, 12.0)) for i in range(20)]
    ranked = plan.candidates(games, HABIT_NORMS, PART_NORMS)
    top = ranked[0]
    assert top["key"] == "while_dead" and top["kind"] == "habit"
    assert top["mine"] == pytest.approx(12.0) and top["typical"] == pytest.approx(4.0)
    assert top["count"] == pytest.approx(2.0) and top["typical_count"] == pytest.approx(1.0)
    # Each game is compared with its own role: the same 2 points in mid (typical 2) is no gap.
    mid = [game(i, role="MIDDLE", while_dead=(1, 2.0), kills=9.0, deaths=-9.0) for i in range(20)]
    by_key = {c["key"]: c for c in plan.candidates(mid, HABIT_NORMS, PART_NORMS)}
    assert by_key["while_dead"]["excess"] == pytest.approx(0.0)
    # A part: kills and assists below the role's average show as a cost.
    low = [game(i, kills=3.0) for i in range(20)]
    by_key = {c["key"]: c for c in plan.candidates(low, HABIT_NORMS, PART_NORMS)}
    assert by_key["kills"]["excess"] == pytest.approx(5.0) and by_key["kills"]["kind"] == "part"


def test_targets_are_realistic():
    big = {"kind": "habit", "mine": 12.0, "typical": 4.0, "excess": 8.0, "count": 2.0}
    t = plan.target(big)
    assert t["halfway"] and t["cost"] == pytest.approx(8.0) and t["count"] == pytest.approx(8.0 / 6.0)
    small = {"kind": "habit", "mine": 5.5, "typical": 4.0, "excess": 1.5, "count": 1.0}
    assert plan.target(small)["cost"] == pytest.approx(4.0) and not plan.target(small)["halfway"]
    part = {"kind": "part", "mine": -3.0, "typical": -8.0, "excess": 5.0}
    assert plan.target(part)["cost"] == pytest.approx(-5.5) and "count" not in plan.target(part)


def test_the_trend_needs_games_and_a_real_change():
    assert plan.trend([5.0] * 12)["verdict"] == "early"
    better = plan.trend([10.0, 12.0] * 10 + [2.0, 3.0] * 5)        # 20 games around 11, then 10 around 2.5
    assert better["verdict"] == "better" and better["change"] > better["margin"]
    worse = plan.trend([2.0, 3.0] * 10 + [10.0, 12.0] * 5)
    assert worse["verdict"] == "worse"
    noisy = plan.trend([0.0, 20.0] * 15)                           # the same spread throughout
    assert noisy["verdict"] == "steady"


def test_blocks_and_chart():
    games = [game(i, while_dead=(1, 6.0 if i < 10 else 2.0)) for i in range(17)]
    pts = plan.blocks(games, "while_dead", HABIT_NORMS, PART_NORMS)
    assert len(pts) == 3                                            # 17 games: the 2 oldest are dropped
    assert pts[0]["cost"] > pts[-1]["cost"] and pts[0]["typical"] == pytest.approx(4.0)
    svg = plan.chart(pts, 4.0)
    assert svg.startswith("<svg") and 'class="mine"' in svg and 'class="typical"' in svg and 'class="goal"' in svg
    assert plan.chart(pts[:1], 4.0) is None


@pytest.mark.skipif(not TEST_DB, reason="RIFTWATCH_TEST_DATABASE_URL not set")
def test_a_plan_from_stored_swing():
    """Stored Swing written directly (as the backfill stores it), so this tests the plan's
    reading of it, not the win-probability model."""
    import json

    from jinja2 import Environment, PackageLoader, select_autoescape

    from riftwatch.db import migrate, repo
    from riftwatch.db.connection import connect
    from riftwatch.stats import extract
    from riftwatch.web.gamedata import GameData
    from riftwatch.winprob import swing
    from tests.fixtures import build_game
    from tests.test_winprob import LeadModel

    model = LeadModel()
    ver = swing.version(model)
    with connect(TEST_DB) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(conn)
        for i in range(30):                                         # the same player is blue's 3 every game
            mid = f"NA1_{7000 + i}"
            puuids = [f"p-{mid}-{p}" for p in range(1, 11)]
            puuids[2] = "me"
            match, timeline = build_game(mid, puuids=puuids, start_ms=1_790_000_000_000 + i * 3_600_000,
                                         blue_wins=i % 2 == 0)
            repo.insert_match(conn, match)
            repo.insert_timeline(conn, mid, timeline)
            # Two deaths a game that cost an objective early on, none in the last 10 games.
            costly = i < 20
            moments = [{"t": 600_000, "clock": "10:00", "text": "Enemy killed Me", "swing": 3.0, "lost": [3],
                        "death": {"enemies": 2, "held": 300, "zone": "the river", "timer": 25,
                                  "while_dead": [1] if costly else [], "while_dead_cost": 4.0 if costly else 0.0}},
                       {"t": 900_000, "clock": "15:00", "text": "Enemy killed Me again", "swing": 2.0, "lost": [3],
                        "death": {"enemies": 3, "held": 200, "zone": "the river", "timer": 30,
                                  "while_dead": [1] if costly else [], "while_dead_cost": 3.0 if costly else 0.0}},
                       {"t": 950_000, "clock": "15:50", "text": "Red took Baron", "swing": 5.0, "lost": []}]
            conn.execute("INSERT INTO match_swing (match_id, version, data) VALUES (%s, %s, %s)",
                         (mid, ver, json.dumps({"moments": moments})))
            for pid in range(1, 11):
                conn.execute("""INSERT INTO player_swing (match_id, participant_id, version, total, fights, objectives,
                                                          structures, economy, draft, kills, deaths, items)
                                VALUES (%s, %s, %s, 0, 0, 0, 0, 0, 0, 0, %s, 0)""",
                             (mid, pid, ver, -12.0 if pid == 3 and costly else -4.0))
        extract.backfill(conn)
        role = conn.execute("SELECT position FROM participant_builds WHERE puuid = 'me' LIMIT 1").fetchone()[0]
        norms = {"habits": {role: {"games": 100, "deaths": 2.0, "cost": 6.0,
                                   **{k: {"per_game": 3.0, "share": 0.25}
                                      for k in ("while_dead", "outnumbered", "held_gold", "deep", "late")}}}}
        gd = GameData()
        gd.loaded = True
        games = plan.load_games(conn, model, "me")
        assert len(games) == 30 and all(len(g["deaths"]) == 2 for g in games)     # only the player's deaths
        assert games[0]["habits"]["while_dead"] == [2, 12.0] and games[-1]["habits"]["while_dead"] == [0, 0.0]
        out = plan.build(conn, gd, "me", model, norms, "na/Me-NA1")
        assert out["games"] == 20 and out["season_games"] == 30
        f = out["focus"]
        # Over the last 20 games: 10 costly (12 a game) and 10 clean: 6 a game against 3 typical.
        assert f["key"] == "while_dead" and f["mine"] == pytest.approx(6.0) and f["typical"] == pytest.approx(3.0)
        assert f["target"]["cost"] == pytest.approx(4.5) and f["target"]["halfway"]
        assert f["trend"]["verdict"] == "better" and f["blocks"] == 6
        assert f["chart"].startswith("<svg")
        assert f["examples"] and all(e["url"].endswith("?as=na/Me-NA1") and e["clock"] for e in f["examples"])
        # Another candidate on request.
        chosen = plan.build(conn, gd, "me", model, norms, "na/Me-NA1", "deaths")
        assert chosen["focus"]["key"] == "deaths" and chosen["focus"]["picked"] and not f["picked"]
        # The panel renders, with every value escaped.
        env = Environment(loader=PackageLoader("riftwatch.web", "templates"), autoescape=select_autoescape(["html"]))
        html = env.get_template("_plan.html").render(plan=out)
        assert "Your plan" in html and "Deaths that cost an objective" in html and "Getting better" in html
        assert "You chose this one" in env.get_template("_plan.html").render(plan=chosen)
        assert plan.build(conn, gd, "nobody", model, norms, "na/x") is None
