import subprocess
import sys
import time

import pytest

from riftwatch.watchdog import win32
from riftwatch.watchdog.core import HangDetector, HotkeyError, Observation, parse_hotkey

# -- hotkey parsing --------------------------------------------------------------------------


def test_parse_hotkey():
    spec = parse_hotkey("Ctrl+Alt+K")
    assert spec.modifiers == win32.MOD_CONTROL | win32.MOD_ALT and spec.vk == ord("K")
    assert parse_hotkey("ctrl+shift+f12").vk == 0x7B
    assert parse_hotkey("win+end").vk == 0x23
    assert parse_hotkey("scrolllock").modifiers == 0


@pytest.mark.parametrize("bad", ["", "k", "f5", "hyper+k", "ctrl+nope", "alt+f4", "ctrl+f25"])
def test_parse_hotkey_rejects(bad):
    with pytest.raises(HotkeyError):
        parse_hotkey(bad)


# -- hang detection ---------------------------------------------------------------------------

def stuck(t):
    return Observation(t, running=True, foreground=True, fullscreen=True, responding=False)


def test_hang_needs_the_full_threshold():
    d = HangDetector(threshold_s=5)
    assert [d.update(stuck(t)) for t in (0, 2, 4.9)] == [None, None, None]
    assert d.update(stuck(5)) == "hang"
    assert d.update(stuck(9)) is None            # reported once
    assert d.update(Observation(10, running=True, foreground=True, fullscreen=True)) == "recovered"


def test_brief_stall_does_not_count():
    d = HangDetector(threshold_s=5)
    d.update(stuck(0))
    d.update(Observation(3, running=True, foreground=True, fullscreen=True))   # responded
    assert d.update(stuck(4)) is None
    assert d.update(stuck(8.9)) is None
    assert d.update(stuck(9)) == "hang"          # clock restarted at 4


@pytest.mark.parametrize("obs", [
    Observation(10, running=True, foreground=False, fullscreen=True, responding=False),  # alt-tabbed away
    Observation(10, running=True, foreground=True, fullscreen=False, responding=False),  # windowed
    Observation(10, running=False),
])
def test_not_a_black_screen_hang(obs):
    d = HangDetector(threshold_s=5)
    d.update(Observation(0, running=obs.running, foreground=obs.foreground,
                         fullscreen=obs.fullscreen, responding=obs.responding))
    assert d.update(obs) is None


def test_rect_covers():
    monitor = win32.Rect(0, 0, 1920, 1080)
    assert win32.Rect(0, 0, 1920, 1080).covers(monitor)
    assert win32.Rect(-1, -1, 1921, 1081).covers(monitor)
    assert not win32.Rect(0, 0, 1280, 720).covers(monitor)


# -- real Win32 calls -------------------------------------------------------------------------

windows_only = pytest.mark.skipif(not win32.IS_WINDOWS, reason="Win32 only")

# A Tk window that freezes its own message loop after `freeze_after` seconds -- the same
# symptom as the game's black-screen hang, as far as Windows can tell.
TK_APP = """
import sys, time, tkinter
root = tkinter.Tk(); root.title("riftwatch-test")
freeze_after = float(sys.argv[1])
if freeze_after >= 0:
    root.after(int(freeze_after * 1000), lambda: time.sleep(60))
root.mainloop()
"""


def spawn(freeze_after: float) -> subprocess.Popen:
    # The real interpreter, not sys.executable: in a venv on Windows that is a launcher that
    # starts python as a child, so the window would belong to a different pid than ours.
    python = getattr(sys, "_base_executable", sys.executable)
    return subprocess.Popen([python, "-c", TK_APP, str(freeze_after)])


def window_for(pid: int, timeout: float = 10.0) -> win32.Window:
    exe = win32.process_image(pid).rsplit("\\", 1)[-1].lower()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for w in win32.windows_of({exe}):
            if w.pid == pid:
                return w
        time.sleep(0.2)
    raise AssertionError("test window never appeared")


@windows_only
def test_live_window_is_seen_and_responding():
    proc = spawn(-1)
    try:
        w = window_for(proc.pid)
        assert win32.is_responding(w.hwnd)
        r = win32.window_rect(w.hwnd)
        assert r.right > r.left
    finally:
        proc.kill()


@windows_only
def test_frozen_window_is_detected_and_killed():
    proc = spawn(1.0)
    try:
        w = window_for(proc.pid)
        deadline = time.monotonic() + 15
        while win32.is_responding(w.hwnd, timeout_ms=500):
            assert time.monotonic() < deadline, "freeze never detected"
            time.sleep(0.5)
        assert win32.terminate(proc.pid)
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()


@windows_only
def test_hotkey_registers_and_releases():
    spec = parse_hotkey("ctrl+alt+shift+f24")
    key = win32.Hotkey(spec.modifiers, spec.vk, ident=0xB0C)
    try:
        assert key.pressed() is False
        with pytest.raises(OSError):
            win32.Hotkey(spec.modifiers, spec.vk, ident=0xB0D)   # already taken
    finally:
        key.close()
    win32.Hotkey(spec.modifiers, spec.vk, ident=0xB0E).close()   # free again


# -- the whole loop ---------------------------------------------------------------------------

def run_in_thread(**kwargs):
    """Start the watchdog loop on its own thread (hotkeys belong to the registering
    thread). Returns (thread, native thread id, messages said, stop flag)."""
    import threading

    said, stop, ready = [], threading.Event(), threading.Event()
    box = {}

    def target():
        box["tid"] = threading.get_native_id()
        ready.set()
        from riftwatch.watchdog.run import run
        run(parse_hotkey("ctrl+alt+shift+f23"), say=said.append, should_stop=stop.is_set,
            hotkey_id=0xB10, **kwargs)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    ready.wait(5)
    return t, box["tid"], said, stop


def wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


@windows_only
def test_hotkey_kills_the_game_window():
    import ctypes

    proc = spawn(-1)
    try:
        window_for(proc.pid)
        killed = []

        def kill_only(windows):  # the real kill minus restore_desktop, which would steal focus
            pids = sorted({w.pid for w in windows if win32.terminate(w.pid)})
            killed.extend(pids)
            return pids

        t, tid, said, stop = run_in_thread(match=lambda w: w.pid == proc.pid, kill_fn=kill_only)
        wait_for(lambda: "game detected" in said)
        time.sleep(0.2)
        # Exactly what Windows delivers when the registered combination is pressed.
        assert ctypes.windll.user32.PostThreadMessageW(tid, win32.WM_HOTKEY, 0xB10, 0)
        wait_for(lambda: any(s.startswith("kill switch: terminated") for s in said))
        assert killed == [proc.pid]
        assert proc.wait(timeout=5) is not None
        stop.set()
        t.join(5)
        assert not t.is_alive()
    finally:
        if proc.poll() is None:
            proc.kill()


@windows_only
def test_confirmed_hang_is_auto_killed():
    fake = [win32.Window(hwnd=1, pid=424242, exe="league of legends.exe")]
    kills = []
    t, _, said, stop = run_in_thread(
        observe_fn=lambda now: (stuck(now), fake),
        kill_fn=lambda windows: kills.append(windows) or [w.pid for w in windows],
        hang_threshold=0.0, auto_kill_after=0.0,
    )
    try:
        wait_for(lambda: kills)
        assert kills[0] == fake
        assert any("not responded" in s for s in said)
        assert any(s.startswith("auto-killed hung game") for s in said)
    finally:
        stop.set()
        t.join(5)
