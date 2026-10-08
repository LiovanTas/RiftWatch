import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from riftwatch.vision.healthbars import Bar, find_bars  # noqa: E402

# BGR fills close to the game's default bar colours.
FILL = {"self": (60, 170, 70), "ally": (200, 130, 48), "enemy": (40, 40, 200)}


def draw_bar(frame, x, y, fraction, team, width=105, height=10, tick_every=10):
    """A health bar the way the game draws one: black outline, level box on the left,
    coloured fill, dark missing part, dark ticks across the fill."""
    cv2.rectangle(frame, (x - 22, y - 2), (x + width + 1, y + height + 1), (8, 8, 8), -1)
    cv2.rectangle(frame, (x - 20, y), (x - 4, y + height - 1), (40, 40, 40), -1)    # level box
    cv2.putText(frame, "9", (x - 16, y + height - 2), cv2.FONT_HERSHEY_PLAIN,
                height / 14, (235, 235, 235), 1)                                    # level
    fill = round(width * fraction)
    if fill:
        cv2.rectangle(frame, (x, y), (x + fill - 1, y + height - 1), FILL[team], -1)
    cv2.rectangle(frame, (x + fill, y), (x + width - 1, y + height - 1), (25, 25, 25), -1)
    for tx in range(x + tick_every, x + fill, tick_every):
        cv2.line(frame, (tx, y), (tx, y + height - 1), (20, 20, 20), 1)
    return fill


def in_hud(x, y, w=1920, h=1080, bar=105):
    from riftwatch.vision.healthbars import Calibration

    cx, cy = (x + bar / 2) / w, (y + 5) / h
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in Calibration().hud_boxes)


def background(seed=0, h=1080, w=1920):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 140, size=(h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)


def test_reads_each_bar_and_its_fill():
    frame = background()
    truth = [("self", 900, 500, 0.73), ("ally", 300, 200, 1.0), ("enemy", 1200, 420, 0.31),
             ("enemy", 1500, 700, 0.05)]
    for team, x, y, frac in truth:
        draw_bar(frame, x, y, frac, team)
    bars = sorted(find_bars(frame), key=lambda b: b.x)
    assert [(b.team, b.x, b.y) for b in bars] == sorted(
        [(t, x, y) for t, x, y, _ in truth], key=lambda t: t[1])
    for b in bars:
        frac = next(f for t, x, y, f in truth if x == b.x)
        assert b.fraction == pytest.approx(frac, abs=0.015) and b.total == 105


def test_ignores_red_things_that_are_not_bars():
    frame = background(1)
    # A bar-shaped red strip with outline but no level box: a turret or minion bar.
    cv2.rectangle(frame, (1198, 198), (1306, 211), (8, 8, 8), -1)
    cv2.rectangle(frame, (1200, 200), (1270, 209), FILL["enemy"], -1)
    cv2.rectangle(frame, (1271, 200), (1304, 209), (25, 25, 25), -1)
    cv2.circle(frame, (400, 400), 30, FILL["enemy"], -1)                     # a spell effect
    cv2.rectangle(frame, (800, 800), (905, 809), FILL["enemy"], -1)          # no outline
    cv2.rectangle(frame, (100, 900), (600, 940), FILL["enemy"], -1)          # too tall
    assert find_bars(frame) == []


def test_scales_with_resolution():
    frame = background(2, h=1440, w=2560)
    draw_bar(frame, 1000, 600, 0.5, "enemy", width=140, height=13, tick_every=13)
    (bar,) = find_bars(frame)
    assert bar.team == "enemy" and bar.fraction == pytest.approx(0.5, abs=0.015)


def test_bar_geometry():
    b = Bar("enemy", 100, 50, 30, 120, 10)
    assert b.fraction == 0.25 and b.center == (160.0, 55.0)


def test_worst_case_error_over_many_random_bars():
    rng = np.random.default_rng(7)
    errors = []
    for k in range(30):
        frame = background(k)
        placed = []
        for _ in range(10):
            # Inside the edge margins (5% of width, 3% of height), where bars are skipped.
            x, y = int(rng.integers(120, 1690)), int(rng.integers(40, 1020))
            if any(abs(x - px) < 160 and abs(y - py) < 30 for px, py, _ in placed):
                continue
            if in_hud(x, y):
                continue
            frac = float(rng.uniform(0.05, 1.0))     # under ~4%: see the test below
            draw_bar(frame, x, y, frac, str(rng.choice(["self", "ally", "enemy"])))
            placed.append((x, y, frac))
        found = {(b.x, b.y): b for b in find_bars(frame)}
        for x, y, frac in placed:
            assert (x, y) in found, (k, x, y)
            errors.append(abs(found[(x, y)].fraction - frac))
    # Pixel rounding: half a pixel of fill plus the outline, on a 105 px bar.
    assert len(errors) > 200 and max(errors) < 0.015 and np.mean(errors) < 0.006


# -- video and alignment ------------------------------------------------------------------------

def test_alignment_recovers_the_offset_between_video_and_game_time():
    from riftwatch.vision.video import align

    rng = np.random.default_rng(3)
    game = {}
    hp = 1.0
    for t in range(90, 1500):                     # recorder: game seconds 90..1499
        hp = float(np.clip(hp + rng.normal(0, 0.06), 0.05, 1.0))
        game[t] = hp
    offset = 212                                  # video started 212 s into the game
    video = {t - offset: v + float(rng.normal(0, 0.01)) for t, v in game.items()
             if rng.random() > 0.3 and t - offset >= 0}   # gaps: camera away, deaths
    a = align(video, game)
    assert a.offset == offset and a.mae < 0.02 and a.overlap > 500


def test_scans_a_video_file_and_checks_itself_against_the_recording(tmp_path):
    from riftwatch.vision.video import analyse, frames

    path = tmp_path / "game.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (1280, 720))
    assert writer.isOpened()
    own_hp = []
    for i in range(4 * 150):                      # 150 s at 4 fps
        t = i / 4
        frac = 0.55 + 0.4 * np.sin(t / 9)
        own_hp.append((t, frac))
        frame = background(0, h=720, w=1280)
        draw_bar(frame, 600, 300, frac, "self", width=70, height=7, tick_every=7)
        draw_bar(frame, 800, 250, 0.4, "enemy", width=70, height=7, tick_every=7)
        writer.write(frame)
    writer.release()

    scanned = list(frames(path, fps=2))
    assert len(scanned) == 300
    # The recorder started 40 s before the video: game time = video time + 40.
    samples = [{"t": t + 40, "hp": 1000 * f, "hp_max": 1000} for t, f in own_hp]
    report = analyse(scanned, samples)
    assert report.alignment.offset == 40
    assert report.own_coverage > 0.95 and report.own_error < 2.0
    second = report.seconds[10]
    assert second.t == 50 and len(second.enemies) == 1
    health, distance = second.enemies[0]
    assert health == pytest.approx(0.4, abs=0.03) and 150 < distance < 250


def test_panel_reading_crosses_the_digits_drawn_on_it():
    from riftwatch.vision.spectator import panel_fill

    frame = background(4, h=720, w=1280)
    cv2.rectangle(frame, (48, 568), (118, 579), (20, 20, 20), -1)             # panel frame
    cv2.rectangle(frame, (50, 569), (96, 578), (40, 200, 60), -1)             # 47 px of green
    cv2.putText(frame, "1107/1209", (60, 577), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)
    assert panel_fill(frame) == 47                 # 50..96 inclusive


def test_slivers_under_the_minimum_fill_are_skipped():
    # Below about 4% health the fill is a few pixels -- in real footage mostly noise.
    frame = background(6)
    draw_bar(frame, 700, 500, 0.03, "enemy")
    draw_bar(frame, 700, 700, 0.06, "enemy")
    assert [round(b.fraction, 2) for b in find_bars(frame)] == [0.06]


def test_parallel_scan_matches_a_single_pass(tmp_path):
    from riftwatch.vision.video import scan

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720))
    for i in range(10 * 40):                     # 40 s at 10 fps, health falling steadily
        frame = background(1, h=720, w=1280)
        draw_bar(frame, 500, 300, 1 - i / 450, "enemy", width=70, height=7, tick_every=7)
        writer.write(frame)
    writer.release()
    single = scan(path, fps=2, workers=1)
    parallel = scan(path, fps=2, workers=3)
    assert len(single) == 80
    assert [(t, [b.fill for b in bars]) for t, _, _, bars, _ in parallel] == \
           [(t, [b.fill for b in bars]) for t, _, _, bars, _ in single]


# -- game clock -----------------------------------------------------------------------------------

def clock_frame(seconds, seed=0, h=720, w=1280):
    """A frame with the replay clock drawn where the HUD puts it."""
    frame = background(seed, h=h, w=w)
    k = h / 720
    cv2.rectangle(frame, (round(600 * k), round(48 * k)), (round(690 * k), round(64 * k)), (30, 28, 24), -1)
    m, s = divmod(seconds, 60)
    cv2.putText(frame, f"{m:02d}:{s:02d}", (round(626 * k), round(61 * k)), cv2.FONT_HERSHEY_PLAIN,
                0.85 * k, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def test_clock_learns_digits_and_reads_unseen_times():
    from riftwatch.vision import clock

    templates = clock.learn([(clock_frame(t, seed=t % 7), t) for t in range(60, 700, 7)])
    for t in (75, 389, 1203, 2059, 3599):
        assert clock.read(clock_frame(t, seed=3), templates) == t


def test_clock_offset_outvotes_misreads():
    from riftwatch.vision import clock

    readings = [(float(t), t + 60) for t in range(0, 600, 15)]
    readings[3] = (45.0, 9999)                 # a misread
    readings[7] = (105.0, None)                # unreadable
    off, agree = clock.offset(readings)
    assert off == pytest.approx(60.5) and agree == pytest.approx(38 / 40)


def test_shipped_clock_templates_exist():
    from riftwatch.vision import clock

    templates = clock.load_templates()
    assert templates is not None and templates.shape == (10, clock.GLYPH[0] * clock.GLYPH[1])


# -- minions --------------------------------------------------------------------------------------

def draw_minion_bar(frame, x, y, fraction, team, width=62, height=4):
    """A minion's bar: outline, fill, dark remainder; no level box."""
    cv2.rectangle(frame, (x - 2, y - 2), (x + width + 1, y + height + 1), (8, 8, 8), -1)
    fill = round(width * fraction)
    if fill:
        cv2.rectangle(frame, (x, y), (x + fill - 1, y + height - 1), FILL[team], -1)
    cv2.rectangle(frame, (x + fill, y), (x + width - 1, y + height - 1), (25, 25, 25), -1)


def test_minion_bars_are_found_separately_from_champion_bars():
    from riftwatch.vision.healthbars import find_minion_bars

    frame = background(8)
    draw_bar(frame, 900, 400, 0.7, "enemy")                          # a champion
    cv2.rectangle(frame, (900, 412), (1004, 415), (200, 160, 40), -1)  # its mana bar, just under
    for i, frac in enumerate((1.0, 0.6, 0.25)):
        draw_minion_bar(frame, 500 + 90 * i, 650, frac, "ally")
    draw_minion_bar(frame, 1200, 300, 0.8, "enemy")
    minions = sorted(find_minion_bars(frame), key=lambda b: b.x)
    assert [(b.team, b.x) for b in minions] == [("ally", 500), ("ally", 590), ("ally", 680),
                                                ("enemy", 1200)]
    assert [b.fraction for b in minions] == pytest.approx([1.0, 0.6, 0.25, 0.8], abs=0.02)
    # Minion bars aren't champion bars (no level box), and the champion still is one.
    assert [(b.team, b.x) for b in find_bars(frame)] == [("enemy", 900)]


# The replay HUD panel: bar colours (BGR) inside each bar's hue range, and icon brightness.
PANEL = {"hp": (60, 180, 40), "mana": (220, 120, 40), "xp": (200, 60, 170)}


def panel_frame(hp=1.0, mana=1.0, xp=0.0, cooling=(), seed=0):
    """A 720p frame with the spectator panel drawn: bars filled to the given shares, ability
    and summoner icons bright, except those in ``cooling`` (dark with a white countdown)."""
    from riftwatch.vision import panel

    frame = np.full((720, 1280, 3), 25, np.uint8)
    x0, x1 = panel.BAR_X
    for name, share in (("hp", hp), ("mana", mana), ("xp", xp)):
        _, y0, y1 = panel.BARS[name]
        if share:
            frame[y0:y1 + 1, x0:x0 + round(share * (x1 - x0 + 1))] = PANEL[name]
    rng = np.random.default_rng(seed)
    for slot, (a, b, c, d) in panel.SLOTS.items():
        lit = slot not in cooling
        base = rng.integers(150, 210) if lit else rng.integers(35, 60)
        frame[b:d + 1, a:c + 1] = (base, base - 15, base - 30)
        if not lit:                                   # the seconds left, in white
            cv2.putText(frame, "8", (a + 5, d - 3), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
    return frame


def test_panel_reads_resources_and_which_abilities_are_ready():
    from riftwatch.vision import panel

    frames = [panel_frame(0.8, 0.5, 0.25, cooling=("R", "F"), seed=1),
              panel_frame(0.6, 0.3, 0.4, cooling=("Q",), seed=2),
              panel_frame(1.0, 1.0, 0.6, seed=3)]
    readings = [panel.read(f) for f in frames]
    first = readings[0]
    assert first.hp == pytest.approx(0.8, abs=0.03) and first.mana == pytest.approx(0.5, abs=0.03)
    assert first.xp == pytest.approx(0.25, abs=0.03)
    ready = panel.readiness(readings)
    assert ready[0] == {"Q": True, "W": True, "E": True, "R": False, "D": True, "F": False}
    assert ready[1]["Q"] is False and ready[1]["R"] and all(ready[2].values())
    # No panel (a dead champion's is greyed out): no bars.
    empty = panel.read(np.full((720, 1280, 3), 25, np.uint8))
    assert empty.hp is None and empty.mana is None and empty.xp is None


def test_panel_scales_with_resolution():
    from riftwatch.vision import panel

    big = cv2.resize(panel_frame(0.5, 0.75, 0.3), (1920, 1080), interpolation=cv2.INTER_NEAREST)
    r = panel.read(big)
    assert r.hp == pytest.approx(0.5, abs=0.03) and r.mana == pytest.approx(0.75, abs=0.03)


def test_levels_count_experience_resets_and_ignore_one_misread():
    from riftwatch.vision.panel import levels

    t = list(range(9))
    xp = [None, 0.2, 0.6, 0.1, 0.2, 0.7, 0.05, 0.65, 0.9]   # up at 3; 6 is a one-frame blip
    assert levels(t, xp) == [None, 1, 1, 2, 2, 2, 2, 2, 2]
    assert levels([0, 1, 2], [0.9, 0.1, 0.2], start_level=18) == [18, 18, 18]


def minimap_frame(cx, cy, icon=None, seed=0):
    """A 720p frame whose minimap shows the camera rectangle centred at map pixel (cx, cy)
    (from the map's top left), 1 px white lines, clipped by the map's edges; ``icon`` draws
    a champion icon over that point."""
    from riftwatch.vision import minimap

    frame = np.full((720, 1280, 3), 30, np.uint8)
    x0, y0, x1, y1 = minimap.MAP
    rng = np.random.default_rng(seed)
    terrain = rng.integers(40, 90, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    terrain[..., 1] += 25                                 # greenish, never white
    frame[y0:y1, x0:x1] = terrain
    mw, mh = x1 - x0, y1 - y0
    bw, bh = minimap.BOX[0] * mw, minimap.BOX[1] * mh
    crop = frame[y0:y1, x0:x1]
    cv2.rectangle(crop, (round(cx - bw / 2), round(cy - bh / 2)),
                  (round(cx + bw / 2), round(cy + bh / 2)), (235, 235, 235), 1)
    if icon is not None:
        cv2.circle(crop, icon, 5, (40, 40, 200), -1)
    return frame, mw, mh


@pytest.mark.parametrize("cx,cy,icon", [
    (82, 82, None),                  # mid lane
    (82, 82, (70, 66)),              # an icon breaks the top edge
    (40, 120, None),                 # bot side
    (82, 5, None),                   # top of the map: only the bottom edge shows
    (160, 60, None),                 # right side clipped
])
def test_minimap_camera_is_where_the_followed_champion_is(cx, cy, icon):
    from riftwatch.vision import minimap

    frame, mw, mh = minimap_frame(cx, cy, icon)
    pos = minimap.camera(frame)
    assert pos is not None
    assert pos[0] == pytest.approx(min(1, cx / mw), abs=0.04)
    assert pos[1] == pytest.approx(1 - cy / mh, abs=0.04)


def test_minimap_without_a_camera_rectangle_reads_nothing():
    from riftwatch.vision import minimap

    frame, _, _ = minimap_frame(-500, -500)
    assert minimap.camera(frame) is None


def test_lane_depth_and_base_from_map_position():
    from riftwatch.vision import minimap

    assert minimap.depth((0.06, 0.06), "blue") == pytest.approx(-1)
    assert minimap.depth((0.5, 0.5), "red") == pytest.approx(0)
    assert minimap.depth((0.7, 0.7), "blue") > 0 > minimap.depth((0.7, 0.7), "red")
    assert minimap.in_base((0.95, 0.9), "red") and not minimap.in_base((0.95, 0.9), "blue")
