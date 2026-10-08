"""Find champion health bars in a frame of recorded gameplay and read how full they are.

League draws a bar over every champion: a coloured fill for current health (your own
champion, allies and enemies each in their own colour), a dark remainder for missing
health, thin dark ticks across the fill every so many hit points, all inside a black
outline. So a bar is a short, wide run of one team colour that continues as a dark run,
and fill / (fill + dark) is the health fraction.

The colour ranges and bar size are starting values for the default colour scheme at
1080p; ``Calibration`` holds them so real footage can tune them (see `riftwatch vision
--frame`). Everything scales with the frame height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TEAMS = ("self", "ally", "enemy")
# Frame around the fill, as a share of the fill's own brightness. In real footage the line
# under the health is clearly dark (~0.3 of the fill) while the top outline and level box are
# dim (up to ~0.6), so the bottom is the stricter test.
FRAME_RATIO = 0.65
BELOW_RATIO = 0.45
# Missing health: at most this share of the fill's brightness (or below dark_max). Relative,
# because around a minion's thin bar the grass sits right at any fixed cut.
MISSING_RATIO = 0.5
# Every champion bar has a level box on its left: dark, with the level in white. Turret and
# minion bars and spell effects don't. Measured on a real replay: real boxes were 66-84% dark
# with 5-16 white digit pixels; the false bars failed one or the other.
LEVEL_BOX_SHARE = 0.23     # box width as a share of the bar width
LEVEL_DARK_RATIO = 0.55    # "dark" in the box: at most this share of the fill's brightness
LEVEL_MIN_DARK = 0.55
LEVEL_MIN_WHITE = 0.02


@dataclass(frozen=True)
class HsvRange:
    """OpenCV HSV: hue 0-179, saturation and value 0-255. ``hue`` may wrap (red)."""
    hue: tuple[int, int]
    sat_min: int = 110
    val_min: int = 110

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = self.hue
        hue_ok = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        return hue_ok & (s >= self.sat_min) & (v >= self.val_min)


@dataclass(frozen=True)
class Calibration:
    colors: dict[str, HsvRange] = field(default_factory=lambda: {
        "self": HsvRange((40, 85)),          # green
        "ally": HsvRange((95, 125)),         # blue
        "enemy": HsvRange((172, 8)),         # red, wrapping round hue 0
    })
    # Measured on real footage (a 720p replay): champion bars are 69-72 px wide and 7 px
    # tall there, the same for every champion. Turret, minion and HUD bars differ. The width
    # is a constant, not measured per bar: at full health the "missing" dark run can't be
    # told from dark scenery beyond the bar.
    bar_height: tuple[int, int] = (8, 13)    # health fill height in px at 1080p
    bar_px: int = 105                        # whole bar width in px at 1080p
    bar_tolerance: int = 5                   # px at 1080p: blur and edge pixels
    min_fill: int = 4                        # px at 1080p; thinner fills are noise
    # Bars touching the screen's outer edge are skipped: the HUD lives there (scoreboard,
    # and in replays a column of champion portraits down each side, whose little bars even
    # have level boxes), and a champion bar that far out is clipped anyway.
    edge_margin: float = 0.03                # top and bottom, share of frame height
    side_margin: float = 0.05                # left and right, share of frame width
    level_box: bool = True                   # champion bars have one; minion bars don't
    # Fixed parts of the HUD, as (x0, y0, x1, y1) shares of the frame: the minimap, the
    # scoreboard and the stats panel. Nothing in them is a health bar over the game world.
    hud_boxes: tuple[tuple[float, float, float, float], ...] = (
        (0.86, 0.75, 1.0, 1.0), (0.31, 0.78, 0.69, 1.0), (0.0, 0.77, 0.16, 1.0))
    dark_max: int = 80                       # V at or below this is "missing health"
    # Dark lines across the fill: thin ticks every 100 health and a thicker one every 1000
    # (about 3 px at 720p). The bar-width cap keeps this from jumping the end outline.
    tick_px: int = 5
    outline_px: int = 2                      # black outline after the bar's last pixel

    def scaled(self, frame_height: int) -> tuple[tuple[int, int], tuple[int, int], int, int]:
        """(height range, (bar width, tolerance), tick, outline) in px for this frame."""
        k = frame_height / 1080
        return ((max(2, round(self.bar_height[0] * k)), max(3, round(self.bar_height[1] * k))),
                (round(self.bar_px * k), max(2, round(self.bar_tolerance * k))),
                max(1, round(self.tick_px * k)), max(0, round(self.outline_px * k)))

    def min_fill_px(self, frame_height: int) -> int:
        return max(2, round(self.min_fill * frame_height / 1080))


@dataclass(frozen=True)
class Bar:
    team: str
    x: int                  # left edge of the bar
    y: int                  # top of the fill
    fill: int               # px of health
    total: int              # px of the whole bar
    height: int

    @property
    def fraction(self) -> float:
        return self.fill / self.total if self.total else 0.0

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.total / 2, self.y + self.height / 2


def _hsv(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)


def _walk_fill(dark: np.ndarray, framed: np.ndarray, start: int, tick: int,
               limit: int | None = None) -> int:
    """End (exclusive) of the fill that starts at ``start`` on one row. Health is told from
    missing health by brightness, not colour: video compression blurs colour at that edge,
    but not brightness. Dark gaps of at most ``tick`` px are crossed only when health
    continues after them *inside the bar* -- where ``framed`` (the outline row above) is still
    dark -- which tells a tick across the fill from the outline at the bar's end. Health
    can't run past the bar, so a walk longer than ``limit`` (the widest bar) went through
    the end outline into something bright beyond it; it ends at the last gap within reach."""
    pos, n = start, len(dark)
    gaps: list[int] = []
    while pos < n:
        if not dark[pos]:
            pos += 1
            continue
        gap = pos
        while gap < n and dark[gap] and gap - pos < tick:
            gap += 1
        if gap < n and gap > pos and not dark[gap] and framed[gap]:
            gaps.append(pos)
            pos = gap
            continue
        break
    if limit is not None and pos - start > limit:
        within = [g for g in gaps if g - start <= limit]
        pos = max(within) if within else start + limit
    return pos


def _dark_run(dark: np.ndarray, start: int, limit: int) -> int:
    end = start
    while end < min(len(dark), start + limit) and dark[end]:
        end += 1
    return end - start


def find_bars(frame_bgr: np.ndarray, calibration: Calibration | None = None,
              rejected: list | None = None) -> list[Bar]:
    """Every health bar in the frame, with its team and fill. Pass a list as ``rejected`` to
    get (team, x, y, w, h, reason) for every candidate turned down -- for tuning."""

    def reject(reason: str) -> None:
        if rejected is not None:
            rejected.append((team, int(x), int(y), int(w), int(h), reason))
    import cv2

    cal = calibration or Calibration()
    (h_min, h_max), (width, tol), tick, outline = cal.scaled(frame_bgr.shape[0])
    hsv = _hsv(frame_bgr)
    value = hsv[..., 2]
    bars: list[Bar] = []
    for team, color in cal.colors.items():
        raw = color.mask(hsv)
        # Candidates: bridge the ticks so one bar is one component. Measuring happens on the
        # raw mask below, because bridging can also jump the thin outline to stray pixels.
        closed = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((1, 2 * tick + 1), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
        for i in range(1, count):
            x, y, w, h, _area = stats[i]
            if not h_min - 2 <= h <= h_max:     # loose: colour can be a row or two short
                reject('height')
                continue
            if w > width + tol + 4 * tick:      # wider than any bar: an effect, not health
                reject('too wide')
                continue
            mid = y + h // 2
            row_c = raw[mid]
            # "Frame dark": clearly darker than this bar's own fill. The outline and level box
            # are dim rather than black in real footage (and shading varies), so they're
            # judged against the fill, not an absolute cut.
            # (Thresholds are applied to the rows each check needs, never the whole frame:
            # a frame has dozens of candidates.)
            fill_v = float(np.median(value[mid, x:x + w]))
            frame_dark = max(cal.dark_max, FRAME_RATIO * fill_v)
            row_d = value[mid] <= max(cal.dark_max, MISSING_RATIO * fill_v)
            # The fill starts right after the level box / outline on its left.
            # (Within a few pixels: blur leaves a half-tone pixel between edge and fill.)
            start = next((p for p in range(x, x + w)
                          if row_c[p] and p > 0 and value[mid, max(0, p - 4):p].min() <= frame_dark),
                         None)
            if start is None:
                reject('no fill start')
                continue
            # A speck of the colour just outside the outline isn't the fill: if the run starts
            # with one or two pixels, then a dark gap, then colour, the fill starts after it.
            lead = start
            while lead < start + 3 and lead < len(row_c) and row_c[lead]:
                lead += 1
            if lead - start <= 2:
                gap_end = lead
                while gap_end < len(row_d) and row_d[gap_end] and gap_end - lead < tick + 2:
                    gap_end += 1
                if gap_end > lead and gap_end < len(row_c) and row_c[gap_end]:
                    start = gap_end
            # The outline row: just above the fill, judged over every health-coloured pixel on
            # the middle row from the start onwards -- the colour blob's own top can be off by
            # a row when stray pixels touch it, and the fill's first stretch can be a sliver
            # when a gold or damage number covers it. A few strays can't outvote the bar.
            cols = start + np.flatnonzero(row_c[start:start + width + tol])
            top = mid
            while (top - 1 >= max(0, y - 3)
                   and np.median(value[top - 1, cols]) > FRAME_RATIO * fill_v):
                top -= 1
            framed = ((value[max(0, top - 2):top] <= frame_dark).any(axis=0) if top > 0
                      else np.zeros_like(row_d))
            end = _walk_fill(row_d, framed, start, tick, limit=width + tol)
            fill = end - start
            # Mostly the team's colour, or it's something else bright next to a dark edge.
            if fill < cal.min_fill_px(frame_bgr.shape[0]) or row_c[start:end].mean() < 0.6:
                reject('short fill or not team colour')
                continue
            # The fill's real height is its bright band: compression drains colour from the
            # edge rows but leaves their brightness.
            bright = FRAME_RATIO * fill_v
            top, bottom = mid, mid
            while top - 1 >= max(0, y - 3) and np.median(hsv[top - 1, start:end, 2]) > bright:
                top -= 1
            while bottom + 1 < min(hsv.shape[0], y + h + 3) and np.median(hsv[bottom + 1, start:end, 2]) > bright:
                bottom += 1
            y, h = top, bottom - top + 1
            if not h_min <= h <= h_max:
                reject('bright band height')
                continue
            # Whatever the fill doesn't cover is missing health, and has to look it: dark,
            # straight after the fill (less a little blur).
            expected_missing = width - fill - tol
            if expected_missing > 0 and _dark_run(row_d, end, width) < expected_missing:
                reject('missing part not dark')
                continue
            total = max(width, fill)
            fh, fw = frame_bgr.shape[:2]
            mx, my = cal.side_margin * fw, cal.edge_margin * fh
            if start < mx or start + width > fw - mx or y < my or y + h > fh - my:
                reject('screen edge')
                continue
            cx, cy = (start + width / 2) / fw, (y + h / 2) / fh
            if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in cal.hud_boxes):
                reject('HUD box')
                continue
            # The level box on the left: mostly dark, with a white number in it.
            box_w = max(4, round(LEVEL_BOX_SHARE * width))
            if not cal.level_box:
                box_w = 0
            if box_w:
                box = hsv[max(0, y - 1):y + h + 1, max(0, start - box_w - 2):max(0, start - 2)]
                if box.size == 0:
                    reject('level box (no box)')
                    continue
                box_v, box_s = box[..., 2].astype(int), box[..., 1].astype(int)
                if ((box_v <= LEVEL_DARK_RATIO * fill_v).mean() < LEVEL_MIN_DARK
                        or ((box_v >= 160) & (box_s <= 100)).mean() < LEVEL_MIN_WHITE):
                    reject('level box')
                    continue
            # A real bar has a dark outline just above and below the fill. Look a few rows
            # out: compression drains colour from the fill's edge rows, so the coloured part
            # can be a row short of the bright part.
            above = value[max(0, y - 3):y, start:end] <= frame_dark
            below = (hsv[y + h:y + h + 3, start:end, 2]
                     <= max(cal.dark_max, BELOW_RATIO * fill_v))
            if (above.size == 0 or below.size == 0 or above.mean(axis=1).max() < 0.6
                    or below.mean(axis=1).max() < 0.6):
                reject('outline')
                continue
            bars.append(Bar(team, int(start), int(y), int(fill), int(total), int(h)))
    return bars


def draw(frame_bgr: np.ndarray, bars: list[Bar]) -> np.ndarray:
    """A copy of the frame with each detected bar boxed and labelled, for calibration."""
    import cv2

    out = frame_bgr.copy()
    colors = {"self": (0, 255, 0), "ally": (255, 160, 0), "enemy": (0, 0, 255)}
    for b in bars:
        cv2.rectangle(out, (b.x - 2, b.y - 2), (b.x + b.total + 1, b.y + b.height + 1),
                      colors[b.team], 1)
        cv2.putText(out, f"{b.team} {100 * b.fraction:.0f}%", (b.x, b.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[b.team], 1, cv2.LINE_AA)
    return out


# Minion bars: the same build as champion bars, smaller, with no level box and no thick
# 1000-health line. Measured on a real 720p replay: about 41 px wide and 2-4 px tall.
MINIONS = Calibration(
    colors={"ally": HsvRange((95, 125), sat_min=90), "enemy": HsvRange((172, 8), sat_min=90)},
    bar_height=(3, 6), bar_px=62, bar_tolerance=3, min_fill=3, tick_px=2, level_box=False,
)


def find_minion_bars(frame_bgr: np.ndarray, calibration: Calibration = MINIONS,
                     champions: list[Bar] | None = None) -> list[Bar]:
    """Every minion health bar in the frame ("ally" = blue side, "enemy" = red side, as for
    champion bars in replays). A thin bar just under a champion's health bar, within its
    width, is that champion's mana or energy bar, not a minion: those are dropped."""
    champions = find_bars(frame_bgr) if champions is None else champions
    out = []
    for b in find_bars(frame_bgr, calibration):
        under = any(c.x - 4 <= b.x <= c.x + c.total and c.y < b.y <= c.y + 3 * c.height
                    for c in champions)
        if not under:
            out.append(b)
    return out
