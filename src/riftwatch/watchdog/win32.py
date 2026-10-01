"""Thin ctypes bindings to the Win32 calls the watchdog needs.

Scope, deliberately narrow so it never trips anti-cheat (Vanguard): window enumeration,
window and monitor geometry, the OS's own "is this window responding" checks, a global
hotkey, and TerminateProcess. Nothing here reads or writes another process's memory,
injects input, or hooks the game.

Importable on any OS (so tests and CI run); every function raises on non-Windows.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                    ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]

    class DEVMODEW(ctypes.Structure):
        # Display variant of DEVMODEW; the printer/display union is kept as raw DWORDs.
        _fields_ = [
            ("dmDeviceName", wintypes.WCHAR * 32), ("dmSpecVersion", wintypes.WORD),
            ("dmDriverVersion", wintypes.WORD), ("dmSize", wintypes.WORD),
            ("dmDriverExtra", wintypes.WORD), ("dmFields", wintypes.DWORD),
            ("dmPositionX", wintypes.LONG), ("dmPositionY", wintypes.LONG),
            ("dmDisplayOrientation", wintypes.DWORD), ("dmDisplayFixedOutput", wintypes.DWORD),
            ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short),
            ("dmYResolution", ctypes.c_short), ("dmTTOption", ctypes.c_short),
            ("dmCollate", ctypes.c_short), ("dmFormName", wintypes.WCHAR * 32),
            ("dmLogPixels", wintypes.WORD), ("dmBitsPerPel", wintypes.DWORD),
            ("dmPelsWidth", wintypes.DWORD), ("dmPelsHeight", wintypes.DWORD),
            ("dmDisplayFlags", wintypes.DWORD), ("dmDisplayFrequency", wintypes.DWORD),
            ("dmICMMethod", wintypes.DWORD), ("dmICMIntent", wintypes.DWORD),
            ("dmMediaType", wintypes.DWORD), ("dmDitherType", wintypes.DWORD),
            ("dmReserved1", wintypes.DWORD), ("dmReserved2", wintypes.DWORD),
            ("dmPanningWidth", wintypes.DWORD), ("dmPanningHeight", wintypes.DWORD),
        ]

    class MSG(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT),
                    ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
                    ("time", wintypes.DWORD), ("pt", wintypes.POINT)]

    user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
    user32.IsHungAppWindow.argtypes = [wintypes.HWND]
    user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                           wintypes.LPARAM, wintypes.UINT, wintypes.UINT,
                                           ctypes.POINTER(ctypes.c_size_t)]
    user32.SendMessageTimeoutW.restype = ctypes.c_size_t
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT,
                                    wintypes.UINT, wintypes.UINT]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetShellWindow.restype = wintypes.HWND
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.ChangeDisplaySettingsW.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    user32.ChangeDisplaySettingsW.restype = ctypes.c_long
    user32.EnumDisplaySettingsW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(DEVMODEW)]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                    wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MONITOR_DEFAULTTONEAREST = 2
WM_NULL = 0x0000
WM_HOTKEY = 0x0312
SMTO_ABORTIFHUNG = 0x0002
PM_REMOVE = 0x0001
SW_MINIMIZE = 6
ENUM_CURRENT_SETTINGS = 0xFFFFFFFF
ENUM_REGISTRY_SETTINGS = 0xFFFFFFFE

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000


class NotWindows(RuntimeError):
    pass


def _require() -> None:
    if not IS_WINDOWS:
        raise NotWindows("the watchdog only runs on Windows")


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    def covers(self, other: Rect, slack: int = 2) -> bool:
        return (self.left <= other.left + slack and self.top <= other.top + slack
                and self.right >= other.right - slack and self.bottom >= other.bottom - slack)


@dataclass(frozen=True)
class Window:
    hwnd: int
    pid: int
    exe: str


def process_image(pid: int) -> str:
    """Full path of a process's executable ('' if it can't be opened)."""
    _require()
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        kernel32.CloseHandle(handle)


def windows_of(exe_names: set[str] | None) -> list[Window]:
    """Visible top-level windows whose process image name is in ``exe_names`` (lowercase);
    ``None`` returns every visible top-level window."""
    _require()
    found: list[Window] = []
    names: dict[int, str] = {}

    @WNDENUMPROC
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value not in names:
            names[pid.value] = process_image(pid.value).rsplit("\\", 1)[-1].lower()
        if exe_names is None or names[pid.value] in exe_names:
            found.append(Window(int(hwnd), pid.value, names[pid.value]))
        return True

    user32.EnumWindows(callback, 0)
    return found


def foreground_window() -> int:
    _require()
    return int(user32.GetForegroundWindow() or 0)


def window_rect(hwnd: int) -> Rect:
    _require()
    r = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return Rect(r.left, r.top, r.right, r.bottom)


def monitor_rect(hwnd: int) -> Rect:
    _require()
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), ctypes.byref(info))
    m = info.rcMonitor
    return Rect(m.left, m.top, m.right, m.bottom)


def is_responding(hwnd: int, timeout_ms: int = 1000) -> bool:
    """Windows' own test: the OS hung-window flag, then a no-op message that must be
    processed within ``timeout_ms``. Neither touches the game's memory."""
    _require()
    if user32.IsHungAppWindow(hwnd):
        return False
    result = ctypes.c_size_t()
    ok = user32.SendMessageTimeoutW(hwnd, WM_NULL, 0, 0, SMTO_ABORTIFHUNG, timeout_ms,
                                    ctypes.byref(result))
    return bool(ok)


def terminate(pid: int) -> bool:
    _require()
    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
    if not handle:
        return False
    try:
        return bool(kernel32.TerminateProcess(handle, 1))
    finally:
        kernel32.CloseHandle(handle)


def _display_mode(which: int) -> tuple[int, int, int, int]:
    mode = DEVMODEW()
    mode.dmSize = ctypes.sizeof(DEVMODEW)
    if not user32.EnumDisplaySettingsW(None, which, ctypes.byref(mode)):
        return (0, 0, 0, 0)
    return (mode.dmPelsWidth, mode.dmPelsHeight, mode.dmBitsPerPel, mode.dmDisplayFrequency)


def display_mode_changed() -> bool:
    """True if the primary display isn't in its saved mode -- what an exclusive-fullscreen
    game leaves behind when it dies mid-game."""
    _require()
    return _display_mode(ENUM_CURRENT_SETTINGS) != _display_mode(ENUM_REGISTRY_SETTINGS)


def restore_desktop() -> None:
    """After the game is gone: put the display back in its saved mode if the game changed
    it, and hand focus back to the desktop shell. The mode reset is conditional because it
    makes the screen flicker even when nothing changed."""
    _require()
    if display_mode_changed():
        user32.ChangeDisplaySettingsW(None, 0)
    shell = user32.GetShellWindow()
    if shell:
        user32.SetForegroundWindow(shell)


class Hotkey:
    """A system-wide hotkey. Works while a fullscreen game has focus, because Windows
    delivers WM_HOTKEY to the registering thread regardless of the foreground window."""

    def __init__(self, modifiers: int, vk: int, ident: int = 0xB0B) -> None:
        _require()
        self.ident = ident
        if not user32.RegisterHotKey(None, ident, modifiers | MOD_NOREPEAT, vk):
            raise OSError(ctypes.get_last_error(),
                          "could not register the hotkey (another program may already own it)")

    def pressed(self) -> bool:
        """Non-blocking: True if the hotkey fired since the last call."""
        msg = MSG()
        fired = False
        while user32.PeekMessageW(ctypes.byref(msg), None, WM_HOTKEY, WM_HOTKEY, PM_REMOVE):
            if msg.wParam == self.ident:
                fired = True
        return fired

    def close(self) -> None:
        user32.UnregisterHotKey(None, self.ident)
