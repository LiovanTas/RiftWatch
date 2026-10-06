import os

import pytest

from riftwatch.scout import PlayerScout, ScoutReport, flags, summarize

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


def player(**kw):
    base = {"puuid": "p", "riot_id": "A#B", "team_id": 100, "champion_id": 103, "champion": "Ahri"}
    return PlayerScout(**(base | kw))


ENTRY = {"queueType": "RANKED_SOLO_5x5", "tier": "GOLD", "rank": "II", "leaguePoints": 40,
         "wins": 60, "losses": 50}


def test_summarize_counts_form_role_and_streak():
    # (champion_id, team_position, win, kills, deaths, assists), newest first
    recent = [(103, "MIDDLE", False, 2, 6, 3), (103, "MIDDLE", False, 4, 4, 4),
              (103, "MIDDLE", False, 1, 8, 2), (7, "MIDDLE", True, 9, 1, 5),
              (103, "TOP", True, 4, 1, 1)]
    p = summarize(player(), recent, recent[:3] + recent[4:], ENTRY, {"championPoints": 80_000})
    assert (p.games, p.wins, p.streak) == (5, 2, -3)
    assert (p.kills, p.deaths, p.assists) == (4.0, 4.0, 3.0)
    assert p.main_role == "MIDDLE" and p.main_role_share == 0.8
    assert p.champion_share == 0.8 and (p.champion_games, p.champion_wins) == (4, 1)
    assert p.rank == {"tier": "GOLD", "division": "II", "lp": 40, "wins": 60, "losses": 50}
    assert p.flags == ["plays mostly Ahri", "lost last 3"]
    assert round(p.kda, 2) == 1.75


def test_flags_for_new_unranked_and_bot_players():
    p = summarize(player(), [], [], None, None)        # mastery 404 = never played it
    assert p.mastery_points == 0
    assert p.flags == ["new to Ahri", "unranked in solo/duo"]
    fresh = summarize(player(), [], [], ENTRY | {"wins": 5, "losses": 6}, {"championPoints": 50_000})
    assert fresh.flags == ["few ranked games this season"]
    assert flags(player(puuid=None)) == ["bot"]


@pytest.fixture
def conn():
    if not TEST_DB:
        pytest.skip("RIFTWATCH_TEST_DATABASE_URL not set")
    from riftwatch.db import migrate
    from riftwatch.db.connection import connect

    with connect(TEST_DB) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(c)
        yield c


def test_scout_live_game_end_to_end(conn):
    from riftwatch.report.terminal import scout_text
    from riftwatch.riot.routing import RiotId
    from riftwatch.scout import NotInGame, scout
    from tests.test_ingest import FakeApi

    api = FakeApi(5)
    report = scout(conn, api, RiotId("Me", "NA1"), "na", games=5, names=lambda c: f"C{c}")
    assert isinstance(report, ScoutReport) and report.queue == "Ranked Solo/Duo"
    assert len(report.team(100)) == 5 and len(report.team(200)) == 5
    me = report.players[0]
    # In all five fixture games, always on Aatrox (266), blue side, blue always wins.
    assert me.riot_id == "Me#NA1" and me.champion == "C266"
    assert (me.games, me.wins, me.streak, me.champion_games) == (5, 5, 5, 5)
    assert me.main_role == "TOP" and me.mastery_points == 250_000
    assert me.flags == ["plays mostly C266", "won last 5"]
    other = report.players[1]                # only in the oldest game
    assert other.games == 1 and other.mastery_points == 0 and "new to C64" in other.flags
    assert report.players[-1].puuid is None and report.players[-1].flags == ["bot"]
    assert report.bans == [{"team_id": 100, "champion": "C157"}]
    # Matches only, never timelines, and every game downloaded once.
    assert api.calls.count("match") == 5 and "timeline" not in api.calls

    api.calls.clear()
    scout(conn, api, RiotId("Me", "NA1"), "na", games=5)
    assert "match" not in api.calls           # second scout: all cached

    text = scout_text(report)
    assert "Blue team  (bans: C157)" in text and "*Me#NA1" in text and "won last 5" in text
    assert report.to_json()["players"][0]["kda"] > 0

    api.in_game = False
    with pytest.raises(NotInGame):
        scout(conn, api, RiotId("Me", "NA1"), "na")
