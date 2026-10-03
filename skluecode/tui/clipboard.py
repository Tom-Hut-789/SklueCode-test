from __future__ import annotations

import sys

__all__ = ["read_clipboard_text"]


def read_clipboard_text() -> str | None:
    """Read text from the OS clipboard.

    Returns the clipboard text, or ``None`` when the clipboard is empty or the
    text cannot be read on the current platform. This bypasses the terminal so
    non-ASCII input (e.g. Chinese) works even when the terminal's raw mode has
    no IME support.
    """
    if sys.platform == "win32":
        text = _read_windows_clipboard()
        if text is not None:
            return text
    return _read_tkinter_clipboard()


def _read_windows_clipboard() -> str | None:
    import ctypes
    from ctypes import wintypes

    cf_unicodetext = 13
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.restype = wintypes.BOOL

    if not user32.IsClipboardFormatAvailable(cf_unicodetext):
        return None
    if not user32.OpenClipboard(None):
        return None
    try:
        handle = user32.GetClipboardData(cf_unicodetext)
        if not handle:
            return None
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.c_wchar_p(pointer).value
        finally:
            kernel32.GlobalUnlock(handle)
    except OSError:
        return None
    finally:
        user32.CloseClipboard()


def _read_tkinter_clipboard() -> str | None:
    try:
        import tkinter
    except ImportError:
        return None
    try:
        root = tkinter.Tk()
    except Exception:
        return None
    try:
        root.withdraw()
        try:
            return root.clipboard_get()
        except Exception:
            return None
    finally:
        try:
            root.destroy()
        except Exception:
            pass
