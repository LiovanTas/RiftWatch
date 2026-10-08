"""Read the game clock off the screen, to put video on the game's timeline.

A replay's clock sits top-centre under the score as MM:SS in white on a dark panel. Each
character is cut out by its columns of bright pixels, scaled to a fixed size (so any
resolution works) and matched against digit templates. The templates are learned from video
whose game time is already known (``learn``) and shipped with the package.

For a whole video only the offset matters (game time = video time + offset, constant unless
the replay was paused or cut), so ``offset`` reads the clock at many moments and takes the
value most of them agree on: a few misreads -- a flash over the clock, a frame mid-change --
are simply outvoted.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

# Spectator / replay HUD: the clock under the score, as shares of the frame.
REGION = (0.46, 0.065, 0.54, 0.092)          # x0, y0, x1, y1
GLYPH = (12, 16)                             # normalised glyph size (w, h)
TEMPLATES = Path(__file__).parent / "data" / "clock_digits.npz"
MIN_MATCH = 0.6                              # correlation below this: not confident


def _crop(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = REGION
    return frame_bgr[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]


def _characters(frame_bgr: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """(glyph scaled to GLYPH, is_colon) for each character of the clock, left to right.

    Ink is white-ish pixels inside the clock's text band; marks off to the sides (bits of the
    scene showing through) are ignored. A colon is told from a narrow "1" by shape: it is two
    dots, so it has ink on under half the text's rows, where a digit has ink on nearly all."""
    import cv2

    crop = _crop(frame_bgr)
    if crop.size == 0:
        return []
    ch, cw = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ink = (hsv[..., 2] >= 150) & (hsv[..., 1] <= 90)
    ink[:round(0.15 * ch)] = False               # the panel's gold edge
    ink[round(0.85 * ch):] = False
    rows = np.flatnonzero(ink.any(axis=1))
    if len(rows) == 0:
        return []
    top, bottom = rows[0], rows[-1] + 1
    cols = np.flatnonzero(ink.any(axis=0))
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    out = []
    for g in groups:
        centre = (g[0] + g[-1]) / 2 / cw
        if not 0.2 <= centre <= 0.85:
            continue
        band = ink[top:bottom, g[0]:g[-1] + 1]
        colon = band.any(axis=1).sum() < 0.5 * (bottom - top)
        piece = gray[top:bottom, g[0]:g[-1] + 1]
        out.append((cv2.resize(piece, GLYPH, interpolation=cv2.INTER_AREA), colon))
    return out


def _digits(frame_bgr: np.ndarray) -> list[np.ndarray] | None:
    """The four digit glyphs of MM:SS, or None if the clock isn't readable as that."""
    chars = _characters(frame_bgr)
    if [c for _, c in chars] != [False, False, True, False, False]:
        return None
    return [g for g, colon in chars if not colon]


def _normalise(g: np.ndarray) -> np.ndarray:
    v = g.astype(np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def load_templates(path: Path = TEMPLATES) -> np.ndarray | None:
    """(10, GLYPH h*w) normalised templates for digits 0-9, or None if not learned yet."""
    if not path.exists():
        return None
    return np.load(path)["digits"]


def read(frame_bgr: np.ndarray, templates: np.ndarray | None = None) -> int | None:
    """Game time on the clock, in seconds; None if it can't be read confidently."""
    templates = load_templates() if templates is None else templates
    if templates is None:
        return None
    digits = _digits(frame_bgr)
    if digits is None:
        return None
    values = []
    for g in digits:
        scores = templates @ _normalise(g)
        best = int(np.argmax(scores))
        if scores[best] < MIN_MATCH:
            return None
        values.append(best)
    m1, m2, s1, s2 = values
    if s1 > 5:
        return None
    return (m1 * 10 + m2) * 60 + s1 * 10 + s2


def learn(labelled: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """Templates from (frame, true game seconds) pairs: the mean glyph of each digit."""
    sums = np.zeros((10, GLYPH[0] * GLYPH[1]), np.float64)
    counts = np.zeros(10, int)
    for frame, seconds in labelled:
        digits = _digits(frame)
        if digits is None:
            continue
        m, s = divmod(int(seconds), 60)
        for g, d in zip(digits, (m // 10, m % 10, s // 10, s % 10), strict=True):
            sums[d] += _normalise(g)
            counts[d] += 1
    missing = [d for d in range(10) if counts[d] == 0]
    if missing:
        raise ValueError(f"no examples of digit(s) {missing}")
    means = sums / counts[:, None]
    return np.stack([m / np.linalg.norm(m) for m in means])


def offset(readings: list[tuple[float, int | None]], tolerance: float = 1.5) -> tuple[float, float] | None:
    """(game time - video time, share of readings agreeing within ``tolerance`` s) from
    (video seconds, clock seconds) readings; None if nothing was read. The clock shows whole
    seconds (rounded down), so on average it reads half a second behind: that is added back."""
    diffs = [clock + 0.5 - t for t, clock in readings if clock is not None]
    if not diffs:
        return None
    mode = Counter(round(d) for d in diffs).most_common(1)[0][0]
    agree = [d for d in diffs if abs(d - mode) <= tolerance]
    return float(np.median(agree)), len(agree) / len(readings)


def video_offset(path: str, samples: int = 41) -> tuple[float, float] | None:
    """Read the clock at ``samples`` moments spread over the video and return ``offset``."""
    import cv2

    templates = load_templates()
    if templates is None:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"can't open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    readings = []
    try:
        for k in range(samples):
            t = duration * (k + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if ok:
                readings.append((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - 1 / fps, read(img, templates)))
    finally:
        cap.release()
    return offset(readings) if readings else None
