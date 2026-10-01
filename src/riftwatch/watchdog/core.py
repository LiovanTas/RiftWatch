"""Watchdog logic that doesn't touch Win32: hotkey parsing and hang detection.

The bug being guarded against: the League game client sometimes stops presenting frames
while still holding fullscreen focus, so the screen goes black and Alt-Tab / Ctrl-Alt-Del
appear to do nothing. The watchdog notices the game window has stopped responding while
it owns the screen, and a user-chosen global hotkey kills the game and restores the
desktop whether or not a hang was detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from riftwatch.watchdog.win32 import MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN

GAME_EXES = frozenset({"league of legends.exe"})

_MODIFIERS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT,
              "shift": MOD_SHIFT, "win": MOD_WIN}
_NAMED_KEYS = {
    "end": 0x23, "home": 0x24, "insert": 0x2D, "delete": 0x2E, "pause": 0x13,
    "scrolllock": 0x91, "backspace": 0x08, "escape": 0x1B, "space": 0x20,
}
# Keys distinctive enough to use on their own; anything else needs a modifier so a
# normal keypress in game can never kill it.
_SOLO_OK = {"pause", "scrolllock"} | {f"f{n}" for n in range(13, 25)}


class HotkeyError(ValueError):
    pass


@dataclass(frozen=True)
class HotkeySpec:
    text: str
    modifiers: int
    vk: int


def parse_hotkey(text: str) -> HotkeySpec:
    """``"ctrl+alt+k"`` -> modifiers + virtual-key code."""
    parts = [p.strip().lower() for p in text.split("+") if p.strip()]
    if not parts:
        raise HotkeyError("empty hotkey")
    *mods, key = parts
    modifiers = 0
    for m in mods:
        if m not in _MODIFIERS:
            raise HotkeyError(f"unknown modifier {m!r} (use ctrl, alt, shift, win)")
        modifiers |= _MODIFIERS[m]
    if len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif key in _NAMED_KEYS:
        vk = _NAMED_KEYS[key]
    else:
        raise HotkeyError(f"unknown key {key!r}")
    if not modifiers and key not in _SOLO_OK:
        raise HotkeyError(f"{text!r} needs a modifier (e.g. ctrl+alt+{key}) so a normal "
                          "keypress in game can't trigger it")
    if modifiers == MOD_ALT and key == "f4":
        raise HotkeyError("alt+f4 is already the system close shortcut")
    return HotkeySpec(text, modifiers, vk)


@dataclass(frozen=True)
class Observation:
    """One poll of the game's window state."""
    t: float
    running: bool
    foreground: bool = False   # the game window has focus
    fullscreen: bool = False   # it covers its whole monitor
    responding: bool = True    # Windows' hung-window checks pass


@dataclass
class HangDetector:
    """Flags a hang once the game has owned the screen without responding for
    ``threshold_s`` seconds, and re-arms when it responds again or exits."""

    threshold_s: float = 5.0
    _since: float | None = field(default=None, init=False)
    _flagged: bool = field(default=False, init=False)

    def update(self, obs: Observation) -> str | None:
        """Returns "hang" the moment a hang is confirmed, "recovered" when a flagged hang
        clears, else None."""
        stuck = obs.running and obs.foreground and obs.fullscreen and not obs.responding
        if not stuck:
            was_flagged = self._flagged
            self._since, self._flagged = None, False
            return "recovered" if was_flagged else None
        if self._since is None:
            self._since = obs.t
        if not self._flagged and obs.t - self._since >= self.threshold_s:
            self._flagged = True
            return "hang"
        return None
