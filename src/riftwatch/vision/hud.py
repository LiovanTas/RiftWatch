"""The rest of the replay HUD: all ten champions, not just the followed one.

* **Side columns.** Replays list each team's five champions down the screen's sides (blue
  team left, red right, in role order top to support), each with a health bar and a
  resource bar under the portrait, the ultimate's icon beside it and the two summoner spells.
  ``sides`` reads, for all ten, both bars' fill and the brightness of the ultimate and
  summoner icons (dark: on cooldown; judged per video against each icon's bright level,
  as with the followed champion's panel). This is every champion's health every frame,
  on screen or not: the jungler's health before a gank, the enemy laner's after a trade
  even when the camera has moved.
* Champion icons on the minimap are too small and blurred at 720p to find by colour
  reliably, so minimap snapshots are archived instead (``archive``) for a later reader.

Geometry is the default spectator HUD at 1280x720, scaled to the frame.
"""

from __future__ import annotations

import numpy as np

W, H = 1280, 720
ROWS = 5
ROW_STEP = 68.5                       # px between champions in a side column
HP_Y, RES_Y = 138, 144                # bar rows of the first champion (4 px tall)
BAR_H = 4
SIDE_BARS = {"blue": (21, 48), "red": (1232, 1259)}
# Icons of the first champion, (x0, y0, x1, y1): the ultimate's circle, summoner spells.
SIDE_ICONS = {"blue": {"r": (44, 102, 54, 112), "d": (6, 105, 16, 115), "f": (6, 119, 16, 129)},
              "red": {"r": (1226, 102, 1236, 112), "d": (1264, 105, 1274, 115),
                      "f": (1264, 119, 1274, 129)}}


def _y(v: float, h: int) -> int:
    return round(v / H * h)


def _x(v: float, w: int) -> int:
    return round(v / W * w)


def _fill(hsv: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float | None:
    """Share of a side bar filled: from its start to the end of the unbroken run of coloured
    pixels (any hue -- resources are blue, yellow, red or white-ish depending on the
    champion; a portrait or icon past the bar's end is cut off by the bar's span)."""
    rows = hsv[y0:y1, x0:x1 + 1]
    if rows.size == 0:
        return None
    coloured = ((rows[..., 1] >= 80) & (rows[..., 2] >= 90)).any(axis=0)
    if not coloured[:3].any():
        return None if not coloured.any() else 0.0
    gaps = np.flatnonzero(~coloured)
    end = gaps[0] if len(gaps) and gaps[0] > 2 else len(coloured)
    if len(gaps) and gaps[0] <= 2:                # a speck at the start: take the run after
        run = np.flatnonzero(coloured)
        end = run[-1] + 1 if len(run) else 0
    return round(min(1.0, end / (x1 - x0 + 1)), 3)


def sides(frame_bgr: np.ndarray) -> list[list] | None:
    """For each champion, blue team top to support then red: [health, resource, ultimate
    brightness, summoner D brightness, summoner F brightness] (None where unreadable)."""
    import cv2

    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out = []
    seen = False
    for team in ("blue", "red"):
        bx0, bx1 = (_x(v, w) for v in SIDE_BARS[team])
        for i in range(ROWS):
            dy = i * ROW_STEP
            hp = _fill(hsv, bx0, bx1, _y(HP_Y + dy, h), _y(HP_Y + dy + BAR_H, h))
            res = _fill(hsv, bx0, bx1, _y(RES_Y + dy, h), _y(RES_Y + dy + BAR_H, h))
            icons = []
            for key in ("r", "d", "f"):
                x0, y0, x1, y1 = SIDE_ICONS[team][key]
                patch = hsv[_y(y0 + dy, h):_y(y1 + dy, h), _x(x0, w):_x(x1, w), 2]
                icons.append(round(float(patch.mean()), 1) if patch.size else None)
            seen |= hp is not None
            out.append([hp, res, *icons])
    return out if seen else None
