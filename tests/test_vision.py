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
    fill = round(width * fraction)
    if fill:
        cv2.rectangle(frame, (x, y), (x + fill - 1, y + height - 1), FILL[team], -1)
    cv2.rectangle(frame, (x + fill, y), (x + width - 1, y + height - 1), (25, 25, 25), -1)
    for tx in range(x + tick_every, x + fill, tick_every):
        cv2.line(frame, (tx, y), (tx, y + height - 1), (20, 20, 20), 1)
    return fill


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
            # Inside the 3% edge margin, where the HUD is and bars are skipped.
            x, y = int(rng.integers(80, 1730)), int(rng.integers(40, 1020))
            if any(abs(x - px) < 160 and abs(y - py) < 30 for px, py, _ in placed):
                continue
            frac = float(rng.uniform(0.05, 1.0))     # under ~4%: see the test below
            draw_bar(frame, x, y, frac, str(rng.choice(["self", "ally", "enemy"])))
            placed.append((x, y, frac))
        found = {(b.x, b.y): b for b in find_bars(frame)}
        for x, y, frac in placed:
            assert (x, y) in found, (k, x, y)
            errors.append(abs(found[(x, y)].fraction - frac))
    # Pixel rounding: half a pixel of fill plus the outline, on a 105 px bar.
    assert len(errors) > 250 and max(errors) < 0.015 and np.mean(errors) < 0.006


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
