"""Toast notifications.

Toasts are borderless ``Toplevel`` windows stacked in a screen corner, each
fading in, resting, then fading out.  They are deliberately *not* modal and
never take focus — a notification that steals focus mid-gesture would be
worse than no notification at all.

Thread safety
-------------
Tk is not thread-safe and must only be touched from the thread that created
the root window.  Gesture events arrive on the processing thread, so
:meth:`ToastManager.notify` is safe to call from anywhere: it marshals onto
the Tk thread via ``after(0, ...)`` rather than touching widgets directly.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from logger import get_logger
from themes import Theme
from utils import clamp

log = get_logger(__name__)


class Level(str, Enum):
    """Notification severity, which selects the accent colour and icon."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


#: Level -> (theme token, glyph).
_LEVEL_STYLE: Dict[str, tuple] = {
    Level.INFO.value: ("info", "i"),
    Level.SUCCESS.value: ("success", "✓"),
    Level.WARNING.value: ("warning", "!"),
    Level.ERROR.value: ("error", "×"),
}


@dataclass
class Toast:
    """A queued notification."""

    title: str
    message: str = ""
    level: str = Level.INFO.value
    duration: float = 2.8
    created_at: float = field(default_factory=time.time)
    #: Optional de-duplication key; a repeat replaces rather than stacks.
    key: str = ""


class ToastManager:
    """Renders and animates toast windows."""

    MAX_VISIBLE = 4
    WIDTH = 320
    MARGIN = 18
    SPACING = 10
    FADE_STEPS = 12
    FADE_INTERVAL = 16  # ms

    def __init__(self, root: object, theme: Theme, enabled: bool = True,
                 corner: str = "bottom-right") -> None:
        self.root = root
        self.theme = theme
        self.enabled = enabled
        self.corner = corner

        self._windows: List[object] = []
        self._pending: "queue.Queue[Toast]" = queue.Queue(maxsize=64)
        self._by_key: Dict[str, object] = {}
        self._lock = threading.Lock()
        self.history: List[Toast] = []

    def set_theme(self, theme: Theme) -> None:
        """Swap the palette used by future toasts."""
        self.theme = theme

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable notifications."""
        self.enabled = enabled

    # -- public API ------------------------------------------------------- #

    def notify(self, title: str, message: str = "", level: str = "info",
               duration: float = 2.8, key: str = "") -> None:
        """Queue a notification.  Safe to call from any thread."""
        if not self.enabled:
            return

        toast = Toast(title=title, message=message, level=level,
                      duration=duration, key=key)
        with self._lock:
            self.history.append(toast)
            if len(self.history) > 200:
                self.history = self.history[-200:]

        # Marshal onto the Tk thread; never touch widgets from here.
        try:
            self.root.after(0, lambda: self._show(toast))  # type: ignore[attr-defined]
        except Exception as exc:
            log.debug("could not schedule toast: %s", exc)

    def clear(self) -> None:
        """Dismiss every visible toast."""
        for window in list(self._windows):
            self._destroy(window)

    # -- rendering -------------------------------------------------------- #

    def _show(self, toast: Toast) -> None:
        """Build and display a toast window.  Must run on the Tk thread."""
        try:
            import tkinter as tk
        except ImportError:  # pragma: no cover - no GUI available
            return

        try:
            # De-duplicate: a repeated key replaces the existing toast rather
            # than stacking identical messages (e.g. rapid volume changes).
            if toast.key and toast.key in self._by_key:
                self._destroy(self._by_key[toast.key])

            while len(self._windows) >= self.MAX_VISIBLE:
                self._destroy(self._windows[0])

            theme = self.theme
            token, glyph = _LEVEL_STYLE.get(toast.level, _LEVEL_STYLE["info"])
            accent = theme.token(token)

            window = tk.Toplevel(self.root)  # type: ignore[arg-type]
            window.overrideredirect(True)
            window.attributes("-topmost", True)
            try:
                window.attributes("-alpha", 0.0)
            except tk.TclError:
                pass
            window.configure(bg=theme.border)

            height = 74 if toast.message else 56

            outer = tk.Frame(window, bg=theme.border)
            outer.pack(fill="both", expand=True, padx=1, pady=1)
            inner = tk.Frame(outer, bg=theme.surface)
            inner.pack(fill="both", expand=True)

            # Accent stripe down the left edge.
            tk.Frame(inner, bg=accent, width=4).pack(side="left", fill="y")

            body = tk.Frame(inner, bg=theme.surface)
            body.pack(side="left", fill="both", expand=True, padx=12, pady=10)

            header = tk.Frame(body, bg=theme.surface)
            header.pack(fill="x", anchor="w")
            tk.Label(header, text=glyph, bg=theme.surface, fg=accent,
                     font=(theme.font_family, 13, "bold")).pack(side="left")
            tk.Label(header, text=toast.title, bg=theme.surface, fg=theme.text,
                     font=(theme.font_family, 12, "bold"),
                     anchor="w").pack(side="left", padx=(8, 0))

            if toast.message:
                tk.Label(body, text=toast.message, bg=theme.surface,
                         fg=theme.text_muted, font=(theme.font_family, 10),
                         anchor="w", justify="left",
                         wraplength=self.WIDTH - 60).pack(fill="x", anchor="w",
                                                          pady=(3, 0))

            window.geometry(f"{self.WIDTH}x{height}")
            window.bind("<Button-1>", lambda _e: self._destroy(window))

            self._windows.append(window)
            if toast.key:
                self._by_key[toast.key] = window
            setattr(window, "_toast_key", toast.key)

            self._reposition()
            self._fade(window, 0.0, 0.96, on_done=lambda: self._schedule_dismiss(
                window, toast.duration))
        except Exception as exc:
            log.debug("toast display failed: %s", exc)

    def _reposition(self) -> None:
        """Stack visible toasts in the configured corner."""
        try:
            screen_w = self.root.winfo_screenwidth()   # type: ignore[attr-defined]
            screen_h = self.root.winfo_screenheight()  # type: ignore[attr-defined]
        except Exception:
            return

        y = self.MARGIN
        offsets: List[int] = []
        for window in self._windows:
            try:
                offsets.append(window.winfo_height() or 60)  # type: ignore[attr-defined]
            except Exception:
                offsets.append(60)

        bottom = "bottom" in self.corner
        right = "right" in self.corner

        if bottom:
            y = screen_h - self.MARGIN
            for window, height in zip(reversed(self._windows), reversed(offsets)):
                y -= height
                x = (screen_w - self.WIDTH - self.MARGIN) if right else self.MARGIN
                self._place(window, x, y)
                y -= self.SPACING
        else:
            for window, height in zip(self._windows, offsets):
                x = (screen_w - self.WIDTH - self.MARGIN) if right else self.MARGIN
                self._place(window, x, y)
                y += height + self.SPACING

    @staticmethod
    def _place(window: object, x: int, y: int) -> None:
        """Move a toast window."""
        try:
            window.geometry(f"+{int(x)}+{int(y)}")  # type: ignore[attr-defined]
        except Exception:
            pass

    def _fade(self, window: object, start: float, end: float,
              step: int = 0, on_done: Optional[Callable[[], None]] = None) -> None:
        """Animate a window's alpha from ``start`` to ``end``."""
        if window not in self._windows and end > start:
            return
        try:
            progress = step / self.FADE_STEPS
            # Ease-out so the fade feels like it settles rather than stops.
            eased = 1.0 - (1.0 - progress) ** 2
            alpha = start + (end - start) * clamp(eased, 0.0, 1.0)
            window.attributes("-alpha", alpha)  # type: ignore[attr-defined]

            if step < self.FADE_STEPS:
                window.after(  # type: ignore[attr-defined]
                    self.FADE_INTERVAL,
                    lambda: self._fade(window, start, end, step + 1, on_done),
                )
            elif on_done is not None:
                on_done()
        except Exception:
            # The window was destroyed mid-animation; that is normal.
            if on_done is not None:
                try:
                    on_done()
                except Exception:
                    pass

    def _schedule_dismiss(self, window: object, duration: float) -> None:
        """Queue the fade-out after the rest period."""
        try:
            window.after(  # type: ignore[attr-defined]
                int(duration * 1000),
                lambda: self._fade(window, 0.96, 0.0,
                                   on_done=lambda: self._destroy(window)),
            )
        except Exception:
            self._destroy(window)

    def _destroy(self, window: object) -> None:
        """Remove a toast window and restack the rest."""
        if window in self._windows:
            self._windows.remove(window)
        key = getattr(window, "_toast_key", "")
        if key and self._by_key.get(key) is window:
            del self._by_key[key]
        try:
            window.destroy()  # type: ignore[attr-defined]
        except Exception:
            pass
        self._reposition()


class NullNotifier:
    """No-op notifier used in headless runs and tests."""

    def __init__(self) -> None:
        self.messages: List[tuple] = []

    def notify(self, title: str, message: str = "", level: str = "info",
               duration: float = 2.8, key: str = "") -> None:
        """Record the notification instead of displaying it."""
        self.messages.append((title, message, level))
        log.info("[notify:%s] %s%s", level, title, f" - {message}" if message else "")
