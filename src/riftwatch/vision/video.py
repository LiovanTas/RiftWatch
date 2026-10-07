"""Health bars over a whole recorded game, lined up with the live recording's game clock.

A video file and the game clock don't start together, and the video says nothing about
game time. But the recorder logged the player's own health every second, and the player's
own bar is in the video. So the two are lined up by sliding one health curve along the
other and taking the offset where they agree best. With the offset known, the player's own
bar doubles as an accuracy check (the detector against the recorder's exact numbers), and
every enemy bar gets a game time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median

import numpy as np

from riftwatch.vision.healthbars import Bar, Calibration, find_bars

MIN_OVERLAP_S = 60       # seconds of shared own-health readings an alignment must rest on


@dataclass
class Frame:
    t: float                 # seconds into the video
    width: int
    height: int
    bars: list[Bar]

    def own(self) -> Bar | None:
        """The player's bar: the "self" bar nearest the screen centre (the camera follows
        the player by default)."""
        mine = [b for b in self.bars if b.team == "self"]
        if not mine:
            return None
        cx, cy = self.width / 2, self.height / 2
        return min(mine, key=lambda b: (b.center[0] - cx) ** 2 + (b.center[1] - cy) ** 2)


def _segment(path: str, first: int, last: int, step: int, calibration: Calibration | None,
             extra: Callable | None) -> list[tuple]:
    """Scan frames [first, last) of the video, keeping every ``step``-th. Runs in a worker
    process; each opens the file itself and seeks to its first frame."""
    import cv2

    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out, index = [], first
    try:
        while index < last and cap.grab():
            if index % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    out.append((index / native, img.shape[1], img.shape[0],
                                find_bars(img, calibration), extra(img) if extra else None))
            index += 1
    finally:
        cap.release()
    return out


def scan(path: str | Path, fps: float = 2.0, calibration: Calibration | None = None,
         extra: Callable | None = None, workers: int | None = None,
         progress: Callable[[str], None] | None = None) -> list[tuple]:
    """(t, width, height, bars, extra(frame)) for ``fps`` frames a second of the video.

    Decoding is the slow part (every frame is decoded, kept or not), so the video is split
    into segments decoded in parallel processes. ``extra`` must be a module-level function
    (it is sent to the workers)."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"can't open video {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    step = max(1, round(native / fps))
    if workers is None:
        # Spawning processes costs about a second each: not worth it for short clips.
        workers = 1 if total < native * 120 else min(8, os.cpu_count() or 1)
    if workers <= 1 or total <= 0:
        return _segment(str(path), 0, total if total > 0 else 1 << 62, step, calibration, extra)
    # Segment boundaries on multiples of ``step``, so the kept frames are the same as a
    # single pass would keep.
    parts = workers * 3
    size = max(step, -(-total // parts // step) * step)
    bounds = [(a, min(total, a + size)) for a in range(0, total, size)]
    results: dict[int, list[tuple]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_segment, str(path), a, b, step, calibration, extra): a
                   for a, b in bounds}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if progress:
                progress(f"video {100 * done / len(bounds):.0f}%")
    return [row for a, _ in bounds for row in results[a]]


def frames(path: str | Path, fps: float = 2.0, calibration: Calibration | None = None,
           progress: Callable[[str], None] | None = None,
           workers: int | None = None) -> list[Frame]:
    """Bars in ``fps`` frames per second of the video."""
    return [Frame(t, w, h, bars)
            for t, w, h, bars, _ in scan(path, fps, calibration, None, workers, progress)]


def per_second(values: list[tuple[float, float]]) -> dict[int, float]:
    """Median of the readings in each whole second (robust to one bad frame)."""
    bins: dict[int, list[float]] = {}
    for t, v in values:
        bins.setdefault(int(t), []).append(v)
    return {s: median(v) for s, v in bins.items()}


def own_series(scanned: list[Frame]) -> dict[int, float]:
    return per_second([(f.t, b.fraction) for f in scanned if (b := f.own()) is not None])


def recorder_series(samples: list[dict]) -> dict[int, float]:
    """The recorder's own health fraction per second of game time, while alive."""
    return per_second([(s["t"], s["hp"] / s["hp_max"]) for s in samples
                       if s.get("hp_max") and s.get("hp", 0) > 0])


@dataclass
class Alignment:
    offset: int              # game time = video time + offset (seconds)
    overlap: int             # seconds where both had a reading
    mae: float               # mean absolute difference in health fraction at that offset


def align(video: dict[int, float], game: dict[int, float],
          min_overlap: int = MIN_OVERLAP_S) -> Alignment | None:
    """The offset at which the video's own-health curve best matches the recorder's."""
    if not video or not game:
        return None
    v0, v1 = min(video), max(video)
    g0, g1 = min(game), max(game)
    vs = np.full(v1 - v0 + 1, np.nan)
    gs = np.full(g1 - g0 + 1, np.nan)
    for s, x in video.items():
        vs[s - v0] = x
    for s, x in game.items():
        gs[s - g0] = x
    best: Alignment | None = None
    # game second = video second + offset; slide over every offset with enough overlap.
    for offset in range(g0 - v1, g1 - v0 + 1):
        lo, hi = max(v0, g0 - offset), min(v1, g1 - offset)
        if hi - lo + 1 < min_overlap:
            continue
        a = vs[lo - v0:hi - v0 + 1]
        b = gs[lo + offset - g0:hi + offset - g0 + 1]
        ok = ~np.isnan(a) & ~np.isnan(b)
        n = int(ok.sum())
        if n < min_overlap:
            continue
        mae = float(np.abs(a[ok] - b[ok]).mean())
        if best is None or mae < best.mae:
            best = Alignment(offset, n, mae)
    return best


@dataclass
class Second:
    t: int                                   # game time
    own: float | None
    enemies: list[tuple[float, float]] = field(default_factory=list)   # (health, px from you)


@dataclass
class VisionReport:
    alignment: Alignment | None
    own_coverage: float                      # alive seconds in the video with your bar read
    own_error: float | None                  # mean |detected - recorded| health, percent
    seconds: list[Second]

    def to_json(self) -> dict:
        a = self.alignment
        return {
            "alignment": None if a is None else {"offset_s": a.offset, "overlap_s": a.overlap,
                                                 "mae": round(a.mae, 4)},
            "own_bar": {"coverage": round(self.own_coverage, 3),
                        "error_pct": None if self.own_error is None else round(self.own_error, 2)},
            "seconds": [{"t": s.t, "own": s.own,
                         "enemies": [[round(h, 3), round(d)] for h, d in s.enemies]}
                        for s in self.seconds],
        }


def analyse(scanned: list[Frame], samples: list[dict] | None) -> VisionReport:
    """Line the video up with the recording (if given) and summarise every second."""
    video = own_series(scanned)
    game = recorder_series(samples) if samples else {}
    alignment = align(video, game) if game else None
    offset = alignment.offset if alignment else 0

    coverage, error = 0.0, None
    if alignment:
        span = range(min(video) + offset, max(video) + offset + 1)
        alive = [s for s in span if s in game]
        read = [s for s in alive if s - offset in video]
        coverage = len(read) / len(alive) if alive else 0.0
        error = 100 * fmean(abs(video[s - offset] - game[s]) for s in read) if read else None

    by_second: dict[int, Second] = {}
    for f in scanned:
        t = int(f.t) + offset
        if t in by_second:          # one frame per second is enough, and keeps counts honest
            continue
        own = f.own()
        sec = by_second[t] = Second(t, None if own is None else round(own.fraction, 3))
        ox, oy = own.center if own else (f.width / 2, f.height / 2)
        for b in f.bars:
            if b.team == "enemy":
                bx, by = b.center
                sec.enemies.append((b.fraction, ((bx - ox) ** 2 + (by - oy) ** 2) ** 0.5))
    return VisionReport(alignment, coverage, error, [by_second[t] for t in sorted(by_second)])
