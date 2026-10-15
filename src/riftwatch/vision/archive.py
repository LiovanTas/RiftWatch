"""Everything a video gives RiftWatch, kept in the database so the video can be deleted.

Training videos are large and the user doesn't keep them, so processing a video squeezes it
once, over the *whole* game, and stores the result with the rest of the data:

* **Readings** (``video_readings``): what every reader saw, four times a second -- every
  champion and minion health bar with its geometry, the followed champion's HUD panel
  (health, mana, experience, each ability and summoner slot's brightness), the minimap
  camera, the clock, and all ten champions from the side columns (health, resource, the
  ultimate's and summoner spells' brightness). Gzipped JSON lines, about 1 MB a game.
  Everything downstream -- trades, situations, features, labels, the brain -- is rebuilt from
  this alone, so improving the analysis or the model never needs the video again.
* **Snapshots** (``video_snapshots``): small JPEG crops of what no reader understands yet --
  the minimap every MINIMAP_EVERY_S (where all ten champions are, wards, camps), the
  scoreboard and the top bar every BOARD_EVERY_S (CS, KDA, items, gold, kills, towers) -- so
  readers written later can run on games whose video is long gone. About 10 MB a game.

``verify`` says whether a video's archive is complete enough to delete the file.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import psycopg

from riftwatch.vision.healthbars import Bar

DETECTOR_VERSION = 1      # bump when a reader changes what it reads (videos still on disk are re-read)
W, H = 1280, 720
MINIMAP_EVERY_S = 2.0
BOARD_EVERY_S = 10.0
JPEG_QUALITY = 85
SNAPSHOTS = {             # region: (x0, y0, x1, y1) on a 1280x720 replay frame, every s
    "minimap": ((1095, 535, 1280, 720), MINIMAP_EVERY_S),
    "scoreboard": ((400, 565, 890, 720), BOARD_EVERY_S),
    "topbar": ((260, 0, 1030, 72), BOARD_EVERY_S),
}


# -- readings -----------------------------------------------------------------------------

def _bar(b: Bar) -> list:
    return [b.team, b.x, b.y, b.fill, b.total, b.height]


def _num(v):
    return None if v is None else round(float(v), 4)


def encode(row: tuple) -> dict:
    """One scanned frame as plain JSON."""
    t, w, h, bars, extra = row
    d: dict = {"t": round(float(t), 4), "w": w, "h": h, "bars": [_bar(b) for b in bars]}
    if extra is not None:
        d["minions"] = [_bar(b) for b in extra.minions]
        if extra.panel is not None:
            p = extra.panel
            d["panel"] = {"hp": _num(p.hp), "mana": _num(p.mana), "xp": _num(p.xp),
                          "slots": {k: round(float(v), 2) for k, v in p.slots.items()}}
        if extra.camera is not None:
            d["camera"] = [_num(extra.camera[0]), _num(extra.camera[1])]
        if extra.clock is not None:
            d["clock"] = int(extra.clock)
        if extra.sides is not None:
            d["sides"] = [[_num(v) for v in champ] for champ in extra.sides]
    return d


def decode(d: dict) -> tuple:
    from riftwatch.vision.lane import FrameExtra
    from riftwatch.vision.panel import PanelReading

    extra = None
    if "minions" in d:
        p = d.get("panel")
        extra = FrameExtra([Bar(*b) for b in d["minions"]],
                           PanelReading(p["hp"], p["mana"], p["xp"], p["slots"]) if p else None,
                           tuple(d["camera"]) if "camera" in d else None, d.get("clock"),
                           d.get("sides"))
    return (d["t"], d["w"], d["h"], [Bar(*b) for b in d["bars"]], extra)


def pack(rows: list[tuple]) -> bytes:
    lines = "\n".join(json.dumps(encode(r), separators=(",", ":")) for r in rows)
    return gzip.compress(lines.encode("utf-8"), compresslevel=6)


def unpack(blob: bytes) -> list[tuple]:
    return [decode(json.loads(line)) for line in gzip.decompress(blob).decode("utf-8").split("\n")
            if line.strip()]


def save(conn: psycopg.Connection, video_id: int, rows: list[tuple], duration_s: float) -> None:
    conn.execute(
        """
        INSERT INTO video_readings (video_id, detector_version, frames, first_s, last_s,
                                    duration_s, readings)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (video_id) DO UPDATE SET detector_version = EXCLUDED.detector_version,
            frames = EXCLUDED.frames, first_s = EXCLUDED.first_s, last_s = EXCLUDED.last_s,
            duration_s = EXCLUDED.duration_s, readings = EXCLUDED.readings, saved_at = now()
        """,
        (video_id, DETECTOR_VERSION, len(rows), rows[0][0] if rows else None,
         rows[-1][0] if rows else None, duration_s, pack(rows)))


def load(conn: psycopg.Connection, video_id: int) -> list[tuple] | None:
    row = conn.execute("SELECT readings FROM video_readings WHERE video_id = %s",
                       (video_id,)).fetchone()
    return unpack(bytes(row[0])) if row else None


def stored_version(conn: psycopg.Connection, video_id: int) -> int | None:
    row = conn.execute("SELECT detector_version FROM video_readings WHERE video_id = %s",
                       (video_id,)).fetchone()
    return row[0] if row else None


# -- snapshots ----------------------------------------------------------------------------

def snapshot(conn: psycopg.Connection, video_id: int, path: str,
             progress: Callable[[str], None] | None = None) -> int:
    """Crop and store the snapshot regions through the whole video. Returns how many."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"can't open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    step = min(every for _, every in SNAPSHOTS.values())
    rows = []
    try:
        t = 0.0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if ok:
                h, w = img.shape[:2]
                for name, ((x0, y0, x1, y1), every) in SNAPSHOTS.items():
                    if abs(t / every - round(t / every)) > 1e-6:
                        continue
                    crop = img[round(y0 / H * h):round(y1 / H * h), round(x0 / W * w):round(x1 / W * w)]
                    ok2, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok2:
                        rows.append((video_id, round(t, 2), name, jpg.tobytes()))
            t = round(t + step, 6)
    finally:
        cap.release()
    with conn.transaction():
        conn.execute("DELETE FROM video_snapshots WHERE video_id = %s", (video_id,))
        with conn.cursor() as cur:
            with cur.copy("COPY video_snapshots (video_id, t, region, image) FROM STDIN") as copy:
                for r in rows:
                    copy.write_row(r)
    if progress:
        progress(f"  {len(rows)} snapshots")
    return len(rows)


def snapshots(conn: psycopg.Connection, video_id: int, region: str) -> list[tuple[float, np.ndarray]]:
    """(video seconds, image) of every stored snapshot of ``region``, in order."""
    import cv2

    out = []
    for t, blob in conn.execute(
            "SELECT t, image FROM video_snapshots WHERE video_id = %s AND region = %s ORDER BY t",
            (video_id, region)):
        out.append((t, cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)))
    return out


# -- is it safe to delete the video? ------------------------------------------------------

def verify(conn: psycopg.Connection, video_id: int) -> tuple[bool, str]:
    row = conn.execute(
        """SELECT r.detector_version, r.frames, r.first_s, r.last_s, r.duration_s,
                  (SELECT count(*) FROM video_snapshots s WHERE s.video_id = r.video_id
                     AND s.region = 'minimap'),
                  v.status, length(r.readings)
             FROM video_readings r JOIN videos v ON v.id = r.video_id WHERE r.video_id = %s""",
        (video_id,)).fetchone()
    if row is None:
        return False, "not archived yet"
    version, frames, first, last, duration, minimaps, status, size = row
    if status != "done":
        return False, f"processing {status}"
    if version != DETECTOR_VERSION:
        return False, f"archived by an older reader (v{version}); re-process before deleting"
    if not frames or last is None:
        return False, "no readings"
    if duration and last < duration - 5:
        return False, f"readings stop at {last:.0f} s of {duration:.0f} s"
    if duration and minimaps < 0.9 * duration / MINIMAP_EVERY_S:
        return False, f"only {minimaps} minimap snapshots"
    rows = load(conn, video_id)
    gaps = np.diff([r[0] for r in rows]) if rows else np.array([])
    if len(gaps) and gaps.max() > 5:
        return False, f"a {gaps.max():.0f} s gap in the readings"
    return True, f"{frames} readings over {last / 60:.0f} min, {size / 1e6:.1f} MB"
