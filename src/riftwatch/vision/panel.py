"""The followed champion's HUD panel in replays: resources, ability and summoner cooldowns,
experience.

Replays show the followed champion's panel at the bottom left: health, mana (or energy) and
experience bars, then a row of ability icons (Q W E R) and the two summoner spells. An icon on
cooldown is drawn dark with the seconds left in white; a ready one is bright. How bright
"ready" is depends on the champion's icon, so a frame only records each slot's brightness and
the video decides: a slot counts as ready when it is at least READY_SHARE of its own bright
level over the video (``readiness``). An ability not learned yet is dark too, so reads as not
ready -- which it isn't.

Experience fills as the champion earns it and empties at each level-up, so counting the
resets from the start of the game gives the level (``levels``), as long as the video starts
before the first level-up.

Geometry is the default spectator HUD, as shares of the frame, measured at 1280x720.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

W, H = 1280, 720
SLOTS = {"Q": (37, 613, 56, 630), "W": (62, 613, 82, 630), "E": (88, 613, 108, 630),
         "R": (114, 613, 133, 630), "D": (148, 613, 166, 630), "F": (172, 613, 190, 630)}
BARS = {"hp": ((40, 85), 569, 578), "mana": ((85, 125), 583, 592), "xp": ((130, 165), 597, 601)}
BAR_X = (51, 116)                 # all three bars span these columns
READY_SHARE = 0.6
DIGITS_MAX = 0.5                  # countdown digits never cover more of an icon's middle
LEVEL_FROM, LEVEL_TO = 0.5, 0.35  # a level-up: experience from at least this to at most this
MAX_LEVEL = 18


@dataclass
class PanelReading:
    hp: float | None              # share of the bar, None if not shown
    mana: float | None
    xp: float | None
    slots: dict[str, float]       # mean brightness of each slot's icon


def _box(frame, box):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    return frame[round(y0 / H * h):round(y1 / H * h) + 1, round(x0 / W * w):round(x1 / W * w) + 1]


def _fill(hsv, hue: tuple[int, int], y0: int, y1: int, frame_w: int, frame_h: int) -> float | None:
    """Share of a panel bar filled with its colour: from the bar's start to its last pixel of
    that colour (white numbers printed over the bar don't count, and don't stop it)."""
    x0 = round(BAR_X[0] / W * frame_w)
    x1 = round(BAR_X[1] / W * frame_w)
    rows = hsv[round(y0 / H * frame_h):round(y1 / H * frame_h) + 1, x0:x1 + 1]
    lo, hi = hue
    coloured = ((rows[..., 0] >= lo) & (rows[..., 0] <= hi) & (rows[..., 1] >= 90)
                & (rows[..., 2] >= 90))
    columns = np.flatnonzero(coloured.any(axis=0))
    if len(columns) == 0:
        return None
    if columns[0] > 0.15 * (x1 - x0):            # doesn't start at the bar's start: not a bar
        return None
    return min(1.0, (columns[-1] + 1) / (x1 - x0 + 1))


def read(frame_bgr: np.ndarray) -> PanelReading:
    import cv2

    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    bars = {name: _fill(hsv, hue, y0, y1, w, h) for name, (hue, y0, y1) in BARS.items()}
    slots = {}
    for name, box in SLOTS.items():
        icon = _box(hsv, box)
        # The middle of the icon (the border and the level pips under it don't change),
        # leaving out the white countdown digits drawn over a cooling-down icon: they're
        # bright, and would make a dark icon look lit. Digits cover a small part of the
        # icon; a mostly pale one is a bright icon (Flash's core), counted whole.
        ih, iw = icon.shape[:2]
        core = icon[ih // 5:ih - ih // 5, iw // 5:iw - iw // 5]
        digits = (core[..., 1] < 50) & (core[..., 2] > 170)
        rest = core[..., 2][~digits] if digits.mean() < DIGITS_MAX else core[..., 2]
        slots[name] = float(rest.mean()) if rest.size else 0.0
    return PanelReading(bars["hp"], bars["mana"], bars["xp"], slots)


def readiness(readings: list[PanelReading]) -> list[dict[str, bool | None]]:
    """Ready or not for every slot in every reading, judged against that slot's own bright
    level over the video (its 90th percentile)."""
    out: list[dict[str, bool | None]] = [{} for _ in readings]
    for name in SLOTS:
        values = [r.slots.get(name) for r in readings]
        seen = [v for v in values if v is not None]
        if not seen:
            continue
        bright = float(np.percentile(seen, 90))
        for i, v in enumerate(values):
            out[i][name] = None if v is None or bright <= 0 else v >= READY_SHARE * bright
    return out


def levels(times: list[float], xp: list[float | None], start_level: int = 1) -> list[int | None]:
    """Level at each reading, counting experience resets as level-ups from ``start_level``.
    A reset is a fall from at least LEVEL_FROM to at most LEVEL_TO that the next reading
    confirms (one misread frame isn't a level-up); the level stops at MAX_LEVEL. Readings
    without an experience bar keep the last level."""
    seen = [(i, v) for i, v in enumerate(xp) if v is not None]
    ups = set()
    for k in range(1, len(seen)):
        prev, (i, value) = seen[k - 1][1], seen[k]
        nxt = seen[k + 1][1] if k + 1 < len(seen) else value
        if prev >= LEVEL_FROM and value <= LEVEL_TO and nxt <= LEVEL_TO + 0.15:
            ups.add(i)
    out, level, started = [], start_level, False
    for i, value in enumerate(xp):
        if i in ups:
            level = min(MAX_LEVEL, level + 1)
        started = started or value is not None
        out.append(level if started else None)
    return out
