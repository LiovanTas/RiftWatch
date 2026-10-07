"""Measure the bar reader on replay footage, where the HUD shows the truth.

In a replay or spectator video the camera follows one champion, and the HUD's bottom-left
panel shows that champion's health bar (with the exact numbers on it). The panel bar is big
and plain, so its fill is a second, independent reading of the same health. Comparing the
overhead bar the reader finds for the followed champion against the panel, second by second,
measures the reader on real footage without a live recording.

The followed champion's bar is the one of its team nearest the top-centre of the screen;
its team is the colour most often found there, learned from the video itself -- never from
the panel, so the comparison stays honest. Panel geometry is the default spectator HUD,
scaled to the frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import fmean, median

import numpy as np

from riftwatch.vision.healthbars import Bar, Calibration, find_bars

PANEL_ROW = 574 / 720          # health row of the bottom-left panel, as a share of height
PANEL_X = (44 / 1280, 130 / 1280)
FULL_PERCENTILE = 97           # the panel's full width: the long end of the fills seen


def panel_fill(frame_bgr: np.ndarray) -> int | None:
    """Length in px of the green fill on the HUD panel's health bar; None if not shown."""
    import cv2

    h, w = frame_bgr.shape[:2]
    y = round(PANEL_ROW * h)
    x0, x1 = round(PANEL_X[0] * w), round(PANEL_X[1] * w)
    row = cv2.cvtColor(frame_bgr[y:y + 1, x0:x1], cv2.COLOR_BGR2HSV)[0]
    green = (row[:, 0] >= 40) & (row[:, 0] <= 85) & (row[:, 1] >= 90) & (row[:, 2] >= 110)
    found = np.flatnonzero(green)
    if len(found) == 0:
        return None
    start = pos = int(found[0])
    while pos < len(row) and row[pos, 2] > 80:      # bright: the fill, or the digits on it
        pos += 1
    return pos - start


def _nearest(bars: list[Bar], target: tuple[float, float]) -> Bar | None:
    return min(bars, key=lambda b: (b.center[0] - target[0]) ** 2 + (b.center[1] - target[1]) ** 2,
               default=None)


@dataclass
class PanelCheck:
    seconds: int
    with_panel: int                    # seconds the panel showed health
    matched: int                       # ...and the followed champion's bar was found
    errors: list[float] = field(default_factory=list)   # |overhead - panel|, health share

    @property
    def coverage(self) -> float:
        return self.matched / self.with_panel if self.with_panel else 0.0

    def summary(self) -> dict:
        e = sorted(self.errors)
        pct = (lambda q: round(100 * e[min(len(e) - 1, int(q * len(e)))], 1)) if e else (lambda q: None)
        return {"seconds": self.seconds, "coverage": round(self.coverage, 3),
                "error_mean": round(100 * fmean(e), 1) if e else None,
                "error_median": round(100 * median(e), 1) if e else None,
                "error_p90": pct(0.9)}


def check(path: str, calibration: Calibration | None = None, every_s: float = 1.0,
          progress: Callable[[str], None] | None = None) -> PanelCheck:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"can't open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps * every_s))
    votes: dict[str, int] = {}
    readings: list[tuple[float | None, int | None]] = []
    index = 0
    try:
        while cap.grab():
            if index % step == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                h, w = img.shape[:2]
                bars = find_bars(img, calibration)
                target = (w / 2, h / 2 - 0.12 * h)
                near = _nearest(bars, target)
                if near is not None:
                    votes[near.team] = votes.get(near.team, 0) + 1
                team = max(votes, key=votes.get) if votes else None
                followed = _nearest([b for b in bars if b.team == team], target)
                if followed is not None and ((followed.center[0] - target[0]) ** 2
                                             + (followed.center[1] - target[1]) ** 2) ** 0.5 > 0.2 * h:
                    followed = None
                readings.append((None if followed is None else followed.fraction, panel_fill(img)))
                if progress and total and index % (step * 300) == 0:
                    progress(f"video {100 * index / total:.0f}%")
            index += 1
    finally:
        cap.release()
    fills = [p for _, p in readings if p]
    full = float(np.percentile(fills, FULL_PERCENTILE)) if fills else 0.0
    result = PanelCheck(len(readings), 0, 0)
    for overhead, panel in readings:
        if not panel or not full:
            continue
        result.with_panel += 1
        if overhead is not None:
            result.matched += 1
            result.errors.append(abs(overhead - min(1.0, panel / full)))
    return result
