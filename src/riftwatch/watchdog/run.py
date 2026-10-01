"""The watchdog loop: poll the game window, listen for the kill-switch hotkey.

Polls window state once a second (cheap: a window enumeration and two OS checks) and the
hotkey queue every 50 ms, so the kill switch reacts within a frame or two of the press.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from riftwatch.watchdog import win32
from riftwatch.watchdog.core import GAME_EXES, HangDetector, HotkeySpec, Observation

log = logging.getLogger("riftwatch.watchdog")

Match = Callable[[win32.Window], bool]


def game_windows(match: Match | None = None) -> list[win32.Window]:
    """League's game windows, or -- for tests -- exactly the windows ``match`` selects."""
    if match is None:
        return win32.windows_of(set(GAME_EXES))
    return [w for w in win32.windows_of(None) if match(w)]


def observe(now: float, match: Match | None = None) -> tuple[Observation, list[win32.Window]]:
    windows = game_windows(match)
    if not windows:
        return Observation(now, running=False), []
    fg = win32.foreground_window()
    main = next((w for w in windows if w.hwnd == fg), windows[0])
    return Observation(
        now,
        running=True,
        foreground=main.hwnd == fg,
        fullscreen=win32.window_rect(main.hwnd).covers(win32.monitor_rect(main.hwnd)),
        responding=win32.is_responding(main.hwnd),
    ), windows


def kill_game(windows: list[win32.Window]) -> list[int]:
    killed = sorted({w.pid for w in windows if win32.terminate(w.pid)})
    win32.restore_desktop()
    return killed


def run(
    hotkey: HotkeySpec,
    *,
    auto_kill_after: float | None = None,
    hang_threshold: float = 5.0,
    log_file: Path | None = None,
    say: Callable[[str], None] = print,
    should_stop: Callable[[], bool] = lambda: False,
    match: Match | None = None,
    observe_fn: Callable[[float], tuple[Observation, list[win32.Window]]] | None = None,
    kill_fn: Callable[[list[win32.Window]], list[int]] = kill_game,
    hotkey_id: int = 0xB0B,
) -> None:
    """Run until ``should_stop`` or Ctrl+C. ``auto_kill_after`` (seconds of confirmed hang)
    kills the game without waiting for the hotkey; off by default.

    ``match``, ``observe_fn`` and ``kill_fn`` exist so tests can aim the loop at their own
    window instead of a real game.
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format="%(asctime)s %(message)s")
    look = observe_fn or (lambda now: observe(now, match))
    key = win32.Hotkey(hotkey.modifiers, hotkey.vk, ident=hotkey_id)
    detector = HangDetector(hang_threshold)
    say(f"watchdog armed: press {hotkey.text} to kill League and get your desktop back"
        + (f"; auto-kill after {auto_kill_after:.0f}s of a confirmed hang" if auto_kill_after else ""))
    windows: list[win32.Window] = []
    was_running = False
    hang_at: float | None = None
    next_poll = 0.0
    try:
        while not should_stop():
            now = time.monotonic()
            if now >= next_poll:
                obs, windows = look(now)
                next_poll = now + 1.0
                if obs.running != was_running:
                    say("game detected" if obs.running else "game closed")
                    was_running = obs.running
                event = detector.update(obs)
                if event == "hang":
                    hang_at = now
                    say(f"game has not responded for {hang_threshold:.0f}s while fullscreen -- "
                        f"press {hotkey.text} to kill it")
                    log.info("hang detected")
                elif event == "recovered":
                    hang_at = None
                    say("game is responding again")
                    log.info("recovered")
                if hang_at is not None and auto_kill_after is not None \
                        and now - hang_at >= auto_kill_after:
                    killed = kill_fn(windows)
                    say(f"auto-killed hung game (pid {killed})")
                    log.info("auto-kill pids=%s", killed)
                    windows, hang_at = [], None
            if key.pressed():
                if not windows:
                    windows = look(time.monotonic())[1]
                if windows:
                    killed = kill_fn(windows)
                    say(f"kill switch: terminated League (pid {killed}), desktop restored")
                    log.info("hotkey kill pids=%s", killed)
                else:
                    win32.restore_desktop()
                    say("kill switch pressed, but League isn't running; desktop restored")
                windows, hang_at = [], None
            time.sleep(0.05)
    except KeyboardInterrupt:
        say("watchdog stopped")
    finally:
        key.close()
