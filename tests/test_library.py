import os
import random

import pytest

from riftwatch.vision.library import parse_title

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


@pytest.mark.parametrize("name, expect", [
    ("IRELIA vs RENEKTON (TOP) NA Grandmaster 26.1_720p.mp4",
     ("Irelia", "Renekton", "TOP", "NA", "GRANDMASTER", "26.1")),
    ("KAI'SA vs EZREAL (ADC) KR Challenger 15.24.mp4",
     ("Kai'sa", "Ezreal", "BOTTOM", "KR", "CHALLENGER", "15.24")),
    ("LEE SIN vs VIEGO (JUNGLE) EUW Master 26.3_1080p.mkv",
     ("Lee Sin", "Viego", "JUNGLE", "EUW", "MASTER", "26.3")),
    ("Thresh vs Nautilus (Support).mp4", ("Thresh", "Nautilus", "UTILITY", None, None, None)),
    ("my ranked game.mp4", (None, None, None, None, None, None)),
])
def test_parse_title(name, expect):
    m = parse_title(name)
    assert (m.champion, m.opponent, m.role, m.region, m.tier, m.patch) == expect


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


def test_add_registers_once_and_follows_moved_files(conn, tmp_path):
    from riftwatch.vision import library

    folder = tmp_path / "videos"
    (folder / "sub").mkdir(parents=True)
    a = folder / "IRELIA vs RENEKTON (TOP) NA Grandmaster 26.1_720p.mp4"
    a.write_bytes(b"a" * 5000)
    (folder / "sub" / "AHRI vs ZED (MID) KR Challenger 26.2.mp4").write_bytes(b"b" * 3000)
    (folder / "notes.txt").write_text("not a video")
    assert library.add(conn, [folder]) == (2, 0)
    assert library.add(conn, [folder]) == (0, 2)
    moved = tmp_path / "elsewhere.mp4"
    a.rename(moved)
    assert library.add(conn, [moved]) == (0, 1)
    path, champ, role = conn.execute(
        "SELECT path, champion, role FROM videos WHERE champion = 'Irelia'").fetchone()
    assert path.endswith("elsewhere.mp4") and role == "TOP"
    assert [p for _, p in library.pending(conn)] and len(library.pending(conn)) == 2


def fake_video(conn, vid, role="TOP", view="spectator", match_id=None, seed=0):
    """A processed video whose player trades when ahead on health and wins those trades,
    and loses the few taken while behind -- a pattern the models should find."""
    from riftwatch.vision import library

    rng = random.Random(seed)
    conn.execute(
        """INSERT INTO videos (id, path, fingerprint, title, champion, role, tier, view, match_id,
                               status, analyzer_version, game_offset_s)
           VALUES (%s, %s, %s, %s, 'Irelia', %s, 'GRANDMASTER', %s, %s, 'done', %s, 60)""",
        (vid, f"v{vid}.mp4", f"fp{vid}", f"video {vid}", role, view, match_id,
         library.ANALYZER_VERSION))
    t, samples, trades = 120.0, [], []
    while t < 840:
        # A second of steady state (four readings), then the player decides.
        me, opp = rng.uniform(0.3, 1), rng.uniform(0.3, 1)
        mine, theirs = rng.randint(0, 6), rng.randint(0, 6)
        for k in range(4):
            samples.append((vid, t + 0.25 * k, me, opp, 0.2, 0, mine, theirs))
        t += 1.0
        ahead = me - opp > 0.15
        if rng.random() < (0.5 if ahead else 0.15):
            win = ahead and rng.random() < 0.85
            me_lost, opp_lost = (0.08, 0.25) if win else (0.25, 0.08)
            trades.append((vid, t, t + 2, me_lost, opp_lost, "you", False, mine - theirs,
                           "won" if win else "lost"))
            t += 3
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO video_samples VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", samples)
        cur.executemany("INSERT INTO video_trades VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", trades)


def test_stats_situations_and_training_find_the_pattern(conn, tmp_path):
    from riftwatch.vision import learn, library

    for vid in range(1, 9):
        fake_video(conn, vid, seed=vid)
    st = library.stats(conn, role="TOP")
    assert st.videos == 8 and st.trades > 50 and st.started_share == 1.0
    assert st.won + st.lost == st.trades and 5 < st.trades_per_10_min < 400

    rows = learn.situations(conn)
    assert {r.video_id for r in rows} == set(range(1, 9))
    taken = [r for r in rows if r.started]
    assert taken and all(r.net is not None for r in taken)
    ahead_rate = sum(r.started for r in rows if r.features["hp_diff"] > 0.15) / \
        max(1, sum(r.features["hp_diff"] > 0.15 for r in rows))
    assert ahead_rate > 0.3

    metrics = learn.train(conn, tmp_path)
    assert metrics["videos"] == 8 and metrics["test_videos"] >= 1
    assert metrics["decision_logloss"] < metrics["decision_logloss_base"]   # beats the base rate
    model = learn.load(tmp_path)
    (p_ahead, net_ahead), (p_behind, net_behind) = learn.judge(model, [
        learn._features(300, 0.9, 0.4, 0.2, 3, 3, 0, "TOP"),
        learn._features(300, 0.4, 0.9, 0.2, 3, 3, 0, "TOP")])
    assert p_ahead > p_behind and net_ahead > net_behind


def test_hud_fields_reach_the_stats_and_the_models(conn, tmp_path):
    from riftwatch.vision import learn, library

    for vid in range(1, 7):
        fake_video(conn, vid, seed=vid)
    # Ultimate up early in each game, down later; trades lost were followed by a recall.
    conn.execute("UPDATE video_samples SET mana = 0.6, level = 7, depth = 0.1, "
                 "ready = CASE WHEN t < 480 THEN 15 ELSE 7 END")
    conn.execute("UPDATE video_trades SET level = 7, died = false, "
                 "ready = CASE WHEN start_s < 480 THEN 15 ELSE 7 END")
    conn.execute("UPDATE video_trades SET back_after_s = 25 WHERE result = 'lost'")
    st = library.stats(conn, role="TOP")
    assert st.won_ultimate_ready[1] + st.won_ultimate_down[1] == st.trades
    assert st.won_ultimate_ready[1] > 0 and st.won_ultimate_down[1] > 0
    assert st.back_after == st.lost and st.died_after == 0

    rows = learn.situations(conn)
    early = next(r for r in rows if r.t < 480)
    assert early.features["mana"] == 0.6 and early.features["level"] == 7
    assert early.features["r_ready"] == 1.0 and early.features["d_ready"] == 0.0
    taken = [r for r in rows if r.started]
    assert any(r.back for r in taken) and not any(r.died for r in taken)
    metrics = learn.train(conn, tmp_path)
    model = learn.load(tmp_path)
    assert "r_ready" in model["features"] and metrics["videos"] == 6
    # A situation from a video without the panel still gets judged.
    (p, _), = learn.judge(model, [learn._features(300, 0.9, 0.4, 0.2, 3, 3, 0, "TOP")])
    assert 0 <= p <= 1


def test_features_never_seen_are_left_out_of_training(conn, tmp_path):
    from riftwatch.vision import learn

    for vid in range(1, 7):
        fake_video(conn, vid, seed=vid)        # no panel or map anywhere (all unknown)
    learn.train(conn, tmp_path)
    model = learn.load(tmp_path)
    assert "mana" not in model["features"] and "hp_diff" in model["features"]


def test_training_refuses_with_too_few_videos(conn, tmp_path):
    from riftwatch.vision import learn

    fake_video(conn, 1)
    with pytest.raises(ValueError, match="need at least"):
        learn.train(conn, tmp_path)


def test_coach_gets_the_players_video_against_the_library(conn, tmp_path):
    from riftwatch.coach.evidence import Evidence, EvidenceSet
    from riftwatch.coach.pipeline import _video_evidence
    from riftwatch.vision import learn

    for vid in range(1, 7):
        fake_video(conn, vid, seed=vid)
    learn.train(conn, tmp_path)
    fake_video(conn, 99, view="player", match_id="NA1_777", seed=42)
    ev = EvidenceSet([Evidence("E1", "context", "context", "neutral", "Game NA1_777.")])
    _video_evidence(conn, ev, "NA1_777", learn.load(tmp_path))
    texts = [e.text for e in ev.items[1:]]
    assert len(texts) == 3 and [e.id for e in ev.items] == ["E1", "E2", "E3", "E4"]
    assert texts[0].startswith("From the player's gameplay video of this game")
    assert "High-elo top players in the video library (6 games)" in texts[1]
    assert "laning model" in texts[2]
    # No video for a match: no evidence, nothing breaks.
    before = len(ev.items)
    _video_evidence(conn, ev, "NA1_000", None)
    assert len(ev.items) == before
