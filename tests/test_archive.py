import os

import numpy as np
import pytest

from riftwatch.vision.healthbars import Bar
from riftwatch.vision.lane import FrameExtra
from riftwatch.vision.panel import PanelReading

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")


def rows(seconds=120.0, step=0.25):
    """Scanned frames as the scanner returns them: the followed champion (red) and its lane
    opponent trading now and then, with the HUD panel, camera, clock and side columns."""
    out = []
    for i in range(int(seconds / step)):
        t = i * step
        me = 1.0 - 0.3 * ((i // 40) % 2) * (i % 40) / 40
        bars = [Bar("enemy", 605, 245, round(70 * me), 70, 7), Bar("ally", 690, 262, 50, 70, 7)]
        extra = FrameExtra([Bar("ally", 650, 300, 30, 41, 3)],
                           PanelReading(me, 0.6, (t % 60) / 60, {k: 200.0 for k in "QWERDF"}),
                           (0.2, 0.8), int(t + 60), [[0.9, 0.5, 120.0, 60.0, 150.0]] * 10)
        out.append((t, 1280, 720, bars, extra))
    return out


def test_every_reading_survives_the_archive():
    from riftwatch.vision import archive

    original = rows(10)
    back = archive.unpack(archive.pack(original))
    assert len(back) == len(original)
    a, b = original[7], back[7]
    assert a[:3] == b[:3] and a[3] == b[3]
    assert b[4].minions == a[4].minions and b[4].camera == a[4].camera
    assert b[4].clock == a[4].clock and b[4].sides == a[4].sides
    assert b[4].panel.hp == pytest.approx(a[4].panel.hp, abs=1e-4)
    assert b[4].panel.slots == a[4].panel.slots


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


def synthetic_video(path, seconds=20, fps=10):
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    rng = np.random.default_rng(0)
    for _ in range(seconds * fps):
        w.write(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))
    w.release()


def test_a_deleted_video_is_reanalysed_from_its_archive(conn, tmp_path):
    pytest.importorskip("cv2")
    from riftwatch.vision import archive, library

    video = tmp_path / "game.mp4"
    synthetic_video(video)
    conn.execute(
        """INSERT INTO videos (id, path, fingerprint, title, role, view, status, game_offset_s,
                               duration_s)
           VALUES (1, %s, 'fp1', 'game', 'TOP', 'spectator', 'pending', 60, 20)""", (str(video),))
    archive.save(conn, 1, rows(20), duration_s=20)
    assert archive.snapshot(conn, 1, str(video)) == 10 + 2 + 2       # minimap 2 s, boards 10 s
    minimaps = archive.snapshots(conn, 1, "minimap")
    assert len(minimaps) == 10 and minimaps[0][1].shape[:2] == (185, 185)

    # Not processed yet: keep the file. Processed: safe to delete.
    assert not archive.verify(conn, 1)[0]
    library.process(conn)
    ok, why = archive.verify(conn, 1)
    assert ok, why
    states = {vid: (on_disk, safe) for vid, _, on_disk, safe, _ in library.archive_state(conn)}
    assert states[1] == (True, True)

    # Delete the file, change the analysis version: rebuilt from the archive alone.
    video.unlink()
    conn.execute("UPDATE videos SET analyzer_version = 0 WHERE id = 1")
    conn.execute("DELETE FROM video_samples WHERE video_id = 1")
    done, failed = library.process(conn)
    assert done == 1 and not failed
    n, status = conn.execute("SELECT (SELECT count(*) FROM video_samples WHERE video_id = 1), "
                             "status FROM videos WHERE id = 1").fetchone()
    assert n > 50 and status == "done"


def test_an_unarchived_video_is_not_safe_to_delete(conn, tmp_path):
    from riftwatch.vision import archive

    conn.execute("INSERT INTO videos (id, path, fingerprint, title, view, status) "
                 "VALUES (2, 'x.mp4', 'fp2', 'x', 'spectator', 'done')")
    ok, why = archive.verify(conn, 2)
    assert not ok and "not archived" in why
    archive.save(conn, 2, rows(20), duration_s=120)              # readings stop early
    assert "stop at" in archive.verify(conn, 2)[1]


def test_team_finds_the_followed_champion_and_reads_its_opponent():
    import pandas as pd

    from riftwatch.brain import team

    t = np.arange(0, 200, 0.25)
    panel = 0.5 + 0.4 * np.sin(t / 10)
    d = {"t": t + 60, "panel_hp": panel}
    for k in range(10):
        d[f"s{k}_hp"] = np.full(len(t), 0.8)
        for icon in ("res", "r", "d", "f"):
            d[f"s{k}_{icon}"] = np.full(len(t), 150.0)
    d["s7_hp"] = panel + 0.01                                # the followed champion: red mid
    d["s2_hp"] = np.where(t > 100, np.nan, 0.6)              # blue mid: dies 100 s in
    d["s2_d"] = np.where(t > 50, 40.0, 150.0)                # ... with a summoner down from 50 s
    f = pd.DataFrame(d)
    assert team.followed_slot(f) == 7
    out = team.features(f, np.array([80.0, 130.0, 200.0]))   # game times
    assert list(out["opp_hp_side"].iloc[:2]) == [0.6, 0.6] and np.isnan(out["opp_hp_side"].iloc[2])
    assert list(out["opp_summoners_ready"].iloc[:2]) == [2.0, 1.0]
    assert out["enemies_down"].iloc[2] == 1 and out["allies_down"].iloc[0] == 0
    # No slot tracks the panel: nothing is guessed.
    f["s7_hp"] = 0.2
    assert team.followed_slot(f) is None
    assert team.features(f, np.array([80.0])).isna().all(axis=None)
