import gzip
import os
from datetime import UTC, datetime

import httpx
import pytest

from riftwatch.live.analyze import analyze
from riftwatch.live.recorder import LiveClient, Recorder, read, sample

ME = "Liovan#G2EU"


class FakeGame:
    """Riot's local game API, scripted second by second.

    0-199 s full health; 200-205 s loses 40% (a lost trade); recalls -- purchase at 240 s
    with 1,200 gold banked; 300-302 s burst from 90% to dead; dead until 330 s; shops on
    respawn at 335 s (not a recall); game ends at 400 s, then the API goes away.
    """

    def __init__(self):
        self.t = None          # None = no game running

    def hp(self, t):
        if 200 <= t < 206:
            return 1000 - (t - 199) * 66.7
        if 206 <= t < 240:
            return 600
        if 300 <= t < 302:
            return 900 - (t - 299) * 400
        if 302 <= t < 330:
            return 0
        return 1000

    def items(self, t):
        items = [1055, 2003]
        if t >= 240:
            items.append(1036)
        if t >= 335:
            items.append(1037)
        return items

    def payload(self):
        t = self.t
        dead = 302 <= t < 330
        players = [
            {"riotId": ME, "championName": "Warwick", "team": "ORDER", "position": "JUNGLE",
             "isDead": dead, "respawnTimer": max(0.0, 330 - t) if dead else 0.0, "level": 3,
             "scores": {"kills": 0, "deaths": 1 if t >= 302 else 0, "assists": 0,
                        "creepScore": t // 15, "wardScore": 0.0},
             "items": [{"itemID": i} for i in self.items(t)]},
            {"riotId": "Enemy#NA1", "championName": "Lee Sin", "team": "CHAOS", "position": "JUNGLE",
             "isDead": False, "respawnTimer": 0.0, "level": 3,
             "scores": {"kills": 1 if t >= 302 else 0, "deaths": 0, "assists": 0,
                        "creepScore": 0, "wardScore": 0.0}, "items": []},
        ]
        events = [{"EventID": 0, "EventName": "GameStart", "EventTime": 0.0}]
        if t >= 302:
            events.append({"EventID": 1, "EventName": "ChampionKill", "EventTime": 302.0,
                           "KillerName": "Enemy", "VictimName": "Liovan", "Assisters": []})
        if t >= 400:
            events.append({"EventID": 2, "EventName": "GameEnd", "EventTime": 400.0, "Result": "Lose"})
        gold = 1200.0 if 220 <= t < 240 else 300.0
        return {
            "activePlayer": {"riotId": ME, "currentGold": gold, "level": 3,
                             "championStats": {"currentHealth": self.hp(t), "maxHealth": 1000.0,
                                               "resourceValue": 300.0, "resourceMax": 300.0},
                             "abilities": {"Q": {"abilityLevel": 1}, "W": {"abilityLevel": 1},
                                           "E": {"abilityLevel": 1}, "R": {"abilityLevel": 0}}},
            "allPlayers": players,
            "events": {"Events": events},
            "gameData": {"gameMode": "CLASSIC", "gameTime": float(t)},
        }

    def transport(self):
        def handler(request):
            if self.t is None:
                raise httpx.ConnectError("no game", request=request)
            return httpx.Response(200, json=self.payload())
        return httpx.MockTransport(handler)


def record_game(tmp_path, game=None):
    game = game or FakeGame()
    said = []
    rec = Recorder(tmp_path, LiveClient(transport=game.transport()), say=said.append)
    rec.tick()                                  # no game yet
    for t in range(0, 401):
        game.t = t
        rec.tick()
    game.t = None
    rec.tick()                                  # game closed
    return list(tmp_path.glob("*.jsonl.gz")), said


def test_recorder_writes_one_file_per_game(tmp_path):
    files, said = record_game(tmp_path)
    assert len(files) == 1
    head, samples, events = read(files[0])
    assert head["riot_id"] == ME and head["champion"] == "Warwick" and head["position"] == "JUNGLE"
    assert len(samples) == 401 and samples[0]["t"] == 0.0
    assert [e["EventName"] for e in events] == ["GameStart", "ChampionKill", "GameEnd"]   # no repeats
    assert any("recording stopped" in s for s in said) and any("game over" in s for s in said)


def test_no_game_means_no_file(tmp_path):
    rec = Recorder(tmp_path, LiveClient(transport=FakeGame().transport()), say=lambda s: None)
    rec.tick()
    assert list(tmp_path.glob("*")) == []


def test_truncated_recording_is_still_readable(tmp_path):
    files, _ = record_game(tmp_path)
    data = files[0].read_bytes()
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(data[: len(data) - 40])     # killed mid-write
    head, samples, _ = read(cut)
    assert head["riot_id"] == ME and 300 < len(samples) <= 401


def test_analysis_finds_trade_recall_and_burst_death(tmp_path):
    files, _ = record_game(tmp_path)
    head, samples, _ = read(files[0])
    s = analyze(head, samples)

    first = s.drops[0]
    assert first.start == 199 and first.lost == pytest.approx(0.40, abs=0.01)
    assert first.led_to == "recall"

    assert [r.t for r in s.recalls] == [240]               # respawn shopping at 335 s excluded
    assert s.recalls[0].hp == pytest.approx(0.6, abs=0.01)
    assert s.recalls[0].gold == 1200

    [death] = s.deaths
    assert death.t == 302 and death.burst and death.seconds_from_60pct <= 3
    burst_drop = next(d for d in s.drops if d.start >= 299)
    assert burst_drop.led_to == "death"

    m = s.metrics()
    assert m["early_big_hp_losses"] == 2 and m["early_losses_to_recall"] == 1
    assert m["burst_deaths"] == 1 and m["avg_recall_gold"] == 1200


def test_sample_is_compact():
    game = FakeGame()
    game.t = 10
    rec = sample(game.payload())
    assert set(rec) == {"t", "hp", "hp_max", "res", "res_max", "gold", "level", "abilities", "players"}
    assert rec["players"][0]["items"] == [1055, 2003]


# -- import and link (Postgres) -------------------------------------------------------------------

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="RIFTWATCH_TEST_DATABASE_URL not set")
def test_import_links_to_the_synced_match(tmp_path):
    from riftwatch.db import migrate, repo
    from riftwatch.db.connection import connect
    from riftwatch.features import store as feature_store
    from riftwatch.live.store import for_match, import_dir
    from tests.fixtures import build_game

    files, _ = record_game(tmp_path)
    head, samples, _ = read(files[0])
    start_ms = int((head["recorded_at"] - samples[0]["t"]) * 1000) + 30_000   # 30 s off

    with connect(TEST_DB) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(conn)
        puuids = ["me-puuid"] + [f"p{i}" for i in range(2, 11)]
        match, timeline = build_game("NA1_777", puuids=puuids, start_ms=start_ms)
        match["info"]["participants"][0]["championName"] = "Warwick"
        repo.insert_match(conn, match)
        repo.insert_timeline(conn, "NA1_777", timeline)
        repo.upsert_account(conn, {"puuid": "me-puuid", "gameName": "Liovan", "tagLine": "G2EU"}, "na1")
        feature_store.extract_pending(conn)

        report = import_dir(conn, tmp_path)
        assert (report.imported, report.linked) == (1, 1)
        assert import_dir(conn, tmp_path).imported == 0         # already imported
        rec = for_match(conn, "NA1_777")
        assert rec["summary"]["metrics"]["burst_deaths"] == 1
        assert rec["hp_series"][0] == [0.0, 1.0] and len(rec["hp_series"]) > 150


def live_result(tmp_path):
    from riftwatch.live.store import _hp_series
    from tests.test_report import result

    files, _ = record_game(tmp_path)
    head, samples, _ = read(files[0])
    r = result()
    r.live = {"summary": analyze(head, samples).to_json(), "hp_series": _hp_series(samples),
              "champion": "Warwick"}
    return r


def test_live_evidence_is_grounded(tmp_path):
    from riftwatch.coach.evidence import game_evidence
    from riftwatch.coach.grounding import validate
    from riftwatch.coach.pipeline import offline_coach

    r = live_result(tmp_path)
    game, p = r.games[0]
    ev = game_evidence(game, r.scores[0], live=r.live)
    live = [e for e in ev.items if e.kind == "live"]
    texts = " ".join(e.text for e in live)
    assert "lost a fifth or more of your health in a short window 2 times" in texts
    assert "1 of those were followed by a recall" in texts
    assert "recalled 1 time, on average with 60% health and 1200 unspent gold" in texts
    assert validate(offline_coach(ev), ev) == []


def test_html_report_shows_health_section(tmp_path):
    from riftwatch.report.html import game_html
    from tests.test_report import embedded_data

    page = game_html(live_result(tmp_path))
    assert "Health (live recording)" in page and "recalled at 60% health with 1200 gold" in page
    data = embedded_data(page)
    assert data["health"]["series"][0] == [0.0, 1.0]
    assert {m["kind"] for m in data["health"]["marks"]} == {"loss", "recall", "death"}
