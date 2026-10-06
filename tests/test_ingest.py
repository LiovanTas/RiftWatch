"""Cache and ingest tests against a real Postgres (see test_migrate.py for the env var)."""

import os
from datetime import UTC, datetime

import pytest

from riftwatch.db import migrate, repo
from riftwatch.ingest import Ingestor, NotFound
from riftwatch.riot.routing import RiotId
from tests.fixtures import build_game

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


@pytest.fixture
def conn():
    if not TEST_DB:
        pytest.skip("RIFTWATCH_TEST_DATABASE_URL not set")
    from riftwatch.db.connection import connect

    with connect(TEST_DB) as c:
        c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(c)
        yield c


class FakeApi:
    """Stands in for RiotApi: serves fixture games and counts every call."""

    def __init__(self, n_games: int = 5, puuid: str = "me") -> None:
        self.puuid = puuid
        self.games = {}
        for i in range(n_games):
            mid = f"NA1_{5000000000 + i}"
            puuids = [puuid] + [f"other-{i}-{p}" for p in range(2, 11)]
            self.games[mid] = build_game(mid, puuids=puuids, start_ms=1_790_000_000_000 + i * 3_600_000)
        # newest first, like Riot
        self.ids = sorted(self.games, reverse=True)
        self.calls = []
        self.tier = "EMERALD"

    def account_by_riot_id(self, platform, riot_id):
        self.calls.append("account")
        if riot_id.game_name == "Nobody":
            return None
        return {"puuid": self.puuid, "gameName": riot_id.game_name, "tagLine": riot_id.tag_line}

    def league_entries_by_puuid(self, platform, puuid):
        self.calls.append("league")
        return [{"queueType": "RANKED_SOLO_5x5", "tier": self.tier, "rank": "II",
                 "leaguePoints": 40, "wins": 50, "losses": 45}]

    def match_ids(self, platform, puuid, *, start=0, count=20, queue=None, start_time=None,
                  end_time=None):
        self.calls.append("ids")
        return self.ids[start:start + count]

    def match(self, match_id):
        self.calls.append("match")
        return self.games[match_id][0]

    def timeline(self, match_id):
        self.calls.append("timeline")
        return self.games[match_id][1]

    in_game = True

    def active_game(self, platform, puuid):
        """A live game: the player on Aatrox with the nine others from their oldest game,
        plus one bot in place of the last red player."""
        self.calls.append("spectator")
        if not self.in_game:
            return None
        oldest = self.games[self.ids[-1]][0]["info"]["participants"]
        players = [{"puuid": p["puuid"], "riotId": f"{p['riotIdGameName']}#NA1",
                    "teamId": p["teamId"], "championId": p["championId"], "bot": False}
                   for p in oldest[:9]]
        players[0]["riotId"] = "Me#NA1"
        players.append({"puuid": None, "riotId": "", "teamId": 200, "championId": 117,
                        "bot": True})
        return {"gameId": 99, "gameQueueConfigId": 420, "gameStartTime": 1_790_100_000_000,
                "gameLength": 125, "participants": players,
                "bannedChampions": [{"teamId": 100, "championId": 157, "pickTurn": 1},
                                    {"teamId": 200, "championId": -1, "pickTurn": 6}]}

    def champion_mastery(self, platform, puuid, champion_id):
        self.calls.append("mastery")
        if puuid == self.puuid:
            return {"championId": champion_id, "championPoints": 250_000}
        return None


def test_insert_match_extracts_participants(conn):
    match, _ = build_game("NA1_42")
    repo.insert_match(conn, match)
    repo.insert_match(conn, match)  # idempotent
    row = conn.execute(
        "SELECT platform, queue_id, patch, duration_s FROM matches WHERE match_id = 'NA1_42'"
    ).fetchone()
    assert row == ("na1", 420, "16.19", 26 * 60 + 34)
    parts = conn.execute(
        "SELECT participant_id, team_position, champion_id, win FROM match_participants "
        "WHERE match_id = 'NA1_42' ORDER BY participant_id"
    ).fetchall()
    assert len(parts) == 10
    assert parts[0] == (1, "TOP", 266, True)
    assert parts[9] == (10, "UTILITY", 117, False)


def test_cache_first_never_refetches(conn):
    api = FakeApi(1)
    ing = Ingestor(conn, api)
    mid = api.ids[0]
    ing.match(mid)
    ing.timeline(mid)
    ing.match(mid)
    ing.timeline(mid)
    assert api.calls.count("match") == 1 and api.calls.count("timeline") == 1
    stats = repo.cache_stats(conn)["counters"]
    assert stats["match"] == {"hits": 1, "misses": 1}
    assert stats["timeline"] == {"hits": 1, "misses": 1}


def test_timeline_fetches_its_match_first(conn):
    api = FakeApi(1)
    Ingestor(conn, api).timeline(api.ids[0])
    assert api.calls == ["match", "timeline"]
    assert repo.get_match(conn, api.ids[0]) is not None


def test_sync_then_resync_stops_at_first_cached(conn):
    api = FakeApi(5)
    ing = Ingestor(conn, api)
    first = ing.sync(RiotId("Me", "NA1"), "na", count=3)
    assert first.new_matches == api.ids[:3]
    assert first.rank["tier"] == "EMERALD"
    assert repo.find_account(conn, "me", "na1")["puuid"] == "me"

    api.calls.clear()
    second = ing.sync(RiotId("Me", "NA1"), "na", count=5)
    assert second.new_matches == []
    assert second.already_cached == 5
    assert "match" not in api.calls

    buckets = {r[0] for r in conn.execute("SELECT tier_bucket FROM match_samples")}
    assert buckets == {"EMERALD"}
    assert repo.latest_rank(conn, "me")["division"] == "II"


def test_sync_unknown_player(conn):
    with pytest.raises(NotFound):
        Ingestor(conn, FakeApi(1)).sync(RiotId("Nobody", "NA1"), "na")


def test_backfill_pages_and_resumes(conn):
    api = FakeApi(5)
    ing = Ingestor(conn, api, page_size=2)  # 5 games -> pages of 2, 2, 1

    partial = ing.backfill(RiotId("Me", "NA1"), "na", max_pages=1)
    assert not partial.finished and partial.ids_seen == 2

    # "Crash" and rerun: resumes from offset 2 in the same job.
    done = ing.backfill(RiotId("Me", "NA1"), "na")
    assert done.job_id == partial.job_id
    assert done.finished
    assert done.ids_seen == 5 and done.fetched == 5 and done.cached == 0
    assert api.calls.count("match") == 5


def test_backfill_counts_cached_games(conn):
    api = FakeApi(3)
    ing = Ingestor(conn, api)
    ing.sync(RiotId("Me", "NA1"), "na", count=2)
    result = ing.backfill(RiotId("Me", "NA1"), "na",
                          start_time=datetime(2026, 1, 1, tzinfo=UTC))
    assert (result.fetched, result.cached) == (1, 2)


def test_player_match_ids_newest_first(conn):
    api = FakeApi(4)
    Ingestor(conn, api).sync(RiotId("Me", "NA1"), "na", count=4)
    assert repo.player_match_ids(conn, "me", limit=2) == api.ids[:2]


class ModesApi(FakeApi):
    """Five solo, two flex and two draft games, interleaved in time, plus ARAM ids that must
    never be fetched; the player has only a flex rank."""

    def __init__(self):
        super().__init__(9)
        newest_first = self.ids
        self.by_queue = {420: newest_first[0::2], 440: newest_first[1:4:2],
                         400: newest_first[5:9:2], 450: ["NA1_9999999999"]}
        self.asked = []

    def match_ids(self, platform, puuid, *, start=0, count=20, queue=None, start_time=None,
                  end_time=None):
        self.calls.append("ids")
        self.asked.append(queue)
        ids = self.by_queue.get(queue, []) if queue is not None else self.ids
        return ids[start:start + count]

    def league_entries_by_puuid(self, platform, puuid):
        self.calls.append("league")
        return [{"queueType": "RANKED_FLEX_SR", "tier": "SILVER", "rank": "I",
                 "leaguePoints": 10, "wins": 20, "losses": 18}]


def test_sync_merges_draft_and_ranked_modes_newest_first(conn):
    api = ModesApi()
    result = Ingestor(conn, api).sync(RiotId("Me", "NA1"), "na", count=6)
    assert sorted(a for a in api.asked if a) == [400, 420, 440]
    # The six newest across all three modes, in time order, and never the ARAM game.
    assert result.new_matches == api.ids[:6]
    assert "NA1_9999999999" not in result.new_matches
    # No solo/duo rank: the flex rank stands in.
    assert result.rank["tier"] == "SILVER" and result.rank["queueType"] == "RANKED_FLEX_SR"


def test_sync_one_mode_only(conn):
    api = ModesApi()
    result = Ingestor(conn, api).sync(RiotId("Me", "NA1"), "na", count=10, queues=(440,))
    assert result.new_matches == api.by_queue[440]


def test_player_match_ids_filters_by_queues(conn):
    match, timeline = build_game("NA1_61", puuids=["me"] + [f"x{p}" for p in range(2, 11)], queue=440)
    repo.insert_match(conn, match)
    match, _ = build_game("NA1_62", puuids=["me"] + [f"x{p}" for p in range(2, 11)], queue=450,
                          start_ms=1_790_100_000_000)
    repo.insert_match(conn, match)
    assert repo.player_match_ids(conn, "me", queue_id=(420, 440, 400)) == ["NA1_61"]
    assert repo.player_match_ids(conn, "me", queue_id=450) == ["NA1_62"]
    assert repo.player_match_ids(conn, "me") == ["NA1_62", "NA1_61"]


def test_parse_queues():
    from riftwatch.riot.api import parse_queues

    assert parse_queues("solo,flex,draft") == (420, 440, 400)
    assert parse_queues(" Draft , solo,draft") == (400, 420)
    with pytest.raises(ValueError, match="aram"):
        parse_queues("aram")


def test_tier_bucket():
    assert repo.tier_bucket("grandmaster") == "MASTER_PLUS"
    assert repo.tier_bucket("GOLD") == "GOLD"


def test_fetch_many_parallel_records_failures(conn):
    from riftwatch.riot.client import RiotApiError

    api = FakeApi(6)
    bad = api.ids[2]
    real_match = api.match

    def flaky(match_id):
        if match_id == bad:
            raise RiotApiError("gave up after 5 attempts; last: 503")
        return real_match(match_id)

    api.match = flaky
    ing = Ingestor(conn, api, workers=4)
    result = ing.fetch_many(api.ids)
    assert sorted(result.downloaded) == sorted(m for m in api.ids if m != bad)
    assert result.failed == [(bad, "gave up after 5 attempts; last: 503")]
    assert repo.known_timeline_ids(conn, api.ids) == set(api.ids) - {bad}

    again = ing.fetch_many(api.ids)
    assert again.downloaded == [] and len(again.cached) == 5 and len(again.failed) == 1
