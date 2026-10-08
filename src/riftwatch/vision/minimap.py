"""Where the followed champion is on the map, from the minimap's camera rectangle.

Replays follow one champion, and the minimap draws the camera's view as a white rectangle --
so its centre is that champion's position on the map, every frame, without identifying any
icon. Positions are map coordinates from 0 to 1 with the blue base at the bottom left
(0, 0) and the red base at the top right (1, 1), as Riot's timeline positions are.

From position: how deep into the lane the champion stands (towards the enemy base or back
towards their own), and whether they are in their own base (recalled or dead).

Geometry is the default spectator HUD at 1280x720, scaled to the frame.
"""

from __future__ import annotations

import numpy as np

W, H = 1280, 720
MAP = (1106, 546, 1269, 709)           # the map inside the minimap's frame
# The camera rectangle, as shares of the map: about 55 x 32 px on a 720p minimap.
BOX = (0.34, 0.20)
LINE = 0.12                            # a white run this long (share of the map) is an edge
GAP = 0.09                             # an icon over an edge breaks it by up to this much
BASES = {"blue": (0.06, 0.06), "red": (0.94, 0.94)}
IN_BASE = 0.22                         # within this of a base corner: in base


def _map_crop(frame_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = MAP
    X0, Y0, X1, Y1 = round(x0 / W * w), round(y0 / H * h), round(x1 / W * w), round(y1 / H * h)
    return frame_bgr[Y0:Y1, X0:X1], X1 - X0, Y1 - Y0


def _runs(row: np.ndarray, gap: int = 0) -> list[tuple[int, int, int]]:
    """(start, end, white pixels) of each run of white, joining runs split by at most
    ``gap`` pixels (an icon drawn over the line)."""
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
            merged[-1][2] += b - a
        else:
            merged.append([a, b, b - a])
    return [tuple(m) for m in merged]


def camera(frame_bgr: np.ndarray) -> tuple[float, float] | None:
    """Centre of the camera rectangle in map coordinates, or None if it isn't found.

    The rectangle is found by its long white edges, not as one shape: champion icons drawn
    over it break its lines, and near the map's edge part of it is off the minimap. Its
    centre comes from whichever edges are visible plus its known size (BOX)."""
    import cv2

    crop, mw, mh = _map_crop(frame_bgr)
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] >= 190) & (hsv[..., 1] <= 60)
    bw, bh = BOX[0] * mw, BOX[1] * mh
    # Horizontal edges: rows with a long white run; keep that run's extent.
    lines = []
    for y in range(mh):
        runs = [r for r in _runs(white[y], round(GAP * mw)) if r[2] >= LINE * mw]
        if runs:
            x0, x1, _ = max(runs, key=lambda r: r[2])
            lines.append((y, x0, x1))
    if not lines:
        return None
    # The best pair of edges one rectangle-height apart; else a single edge at the map's
    # top or bottom, the rest of the rectangle being off the minimap.
    pair = None
    for i, (ya, a0, a1) in enumerate(lines):
        for yb, b0, b1 in lines[i + 1:]:
            if abs((yb - ya) - bh) <= 0.3 * bh and min(a1, b1) - max(a0, b0) >= LINE * mw:
                score = (a1 - a0) + (b1 - b0)
                if pair is None or score > pair[0]:
                    pair = (score, ya, yb, min(a0, b0), max(a1, b1))
    if pair is not None:
        _, ya, yb, x0, x1 = pair
        cy = (ya + yb) / 2
    else:
        # One edge: which one it is shows in the rectangle's sides, running up from a bottom
        # edge or down from a top edge.
        y, x0, x1 = max(lines, key=lambda l: l[2] - l[1])
        # (A side clipped by the map's top or bottom is short: judge each direction by the
        # share of the room it has.)
        span = round(bh * 0.8)
        cols = [c for c in (x0, x1 - 1) if 0 <= c <= mw - 1]
        room_up, room_down = min(span, y), min(span, mh - y - 1)
        up = (sum(int(white[y - room_up:y, c].sum()) for c in cols) / (room_up * len(cols))
              if room_up >= 2 else 0.0)
        down = (sum(int(white[y + 1:y + 1 + room_down, c].sum()) for c in cols) / (room_down * len(cols))
                if room_down >= 2 else 0.0)
        if max(up, down) < 0.4:
            return None                      # no sides either way: not the rectangle
        cy = y - bh / 2 if up > down else y + bh / 2
    # The camera's centre is on the map: an edge that puts it well off is something else.
    if not -0.1 * bh <= cy <= mh + 0.1 * bh:
        return None
    # Across: the run's middle, unless it reaches the map's side (clipped there).
    if x0 <= 1 and x1 < mw - 2:
        cx = x1 - bw / 2
    elif x1 >= mw - 2 and x0 > 1:
        cx = x0 + bw / 2
    else:
        cx = (x0 + x1) / 2
    return (min(1.0, max(0.0, cx / mw)), min(1.0, max(0.0, 1 - cy / mh)))


def depth(position: tuple[float, float], team: str) -> float:
    """How far towards the enemy base: -1 at your own base, +1 at theirs."""
    own = np.array(BASES[team])
    enemy = np.array(BASES["red" if team == "blue" else "blue"])
    p = np.array(position)
    d_own, d_enemy = np.linalg.norm(p - own), np.linalg.norm(p - enemy)
    return float((d_own - d_enemy) / (d_own + d_enemy)) if d_own + d_enemy else 0.0


def in_base(position: tuple[float, float], team: str) -> bool:
    return float(np.linalg.norm(np.array(position) - np.array(BASES[team]))) <= IN_BASE
