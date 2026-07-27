"""Cursor mapping, smoothing and mouse output.

Turning a fingertip position into a cursor position that *feels* right is the
hardest part of a gesture mouse, and it is almost entirely a signal-processing
problem.  Four stages are applied in order:

1. **Active region mapping.**  The hand never reaches the edge of the camera
   frame comfortably, so only a central sub-rectangle is mapped to the screen.
   This is what lets a ~25 cm hand movement cover a 27" display.
2. **Dead zone.**  Sub-threshold motion is discarded outright, so a hand
   holding still leaves the cursor perfectly still — critical for clicking on
   small targets.
3. **One Euro filtering.**  Speed-adaptive smoothing: heavy at rest to kill
   jitter, light when moving fast to stay responsive.
4. **Velocity prediction.**  Extrapolating a few tens of milliseconds ahead
   hides the camera + inference latency that otherwise makes the cursor feel
   like it is dragging behind the hand.

Mouse output goes through ``pynput`` in preference to ``pyautogui``: pyautogui
adds a per-call sleep and a corner "failsafe" that raises an exception, both
of which are actively harmful when driving the cursor at 60 Hz.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from config import CursorConfig
from logger import get_logger
from utils import (
    OneEuroFilter2D, VelocityEstimator, clamp, distance_2d, remap,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Monitor geometry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Monitor:
    """A physical display in virtual-desktop coordinates."""

    index: int
    x: int
    y: int
    width: int
    height: int
    name: str = ""
    is_primary: bool = False

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """``(left, top, right, bottom)`` in virtual-desktop pixels."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def centre(self) -> Tuple[int, int]:
        """Centre point of the monitor."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, x: int, y: int) -> bool:
        """Whether a virtual-desktop point falls inside this monitor."""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def __str__(self) -> str:
        tag = " (primary)" if self.is_primary else ""
        return f"{self.width}x{self.height} @ {self.x},{self.y}{tag}"


class MonitorLayout:
    """Detected display topology and the virtual desktop that spans it."""

    def __init__(self) -> None:
        self.monitors: List[Monitor] = self._detect()
        self.virtual_bounds = self._compute_virtual_bounds()

    @staticmethod
    def _detect() -> List[Monitor]:
        """Enumerate displays, falling back to a single 1920x1080 desktop."""
        try:
            from screeninfo import get_monitors  # type: ignore[import-not-found]

            found = get_monitors()
            if found:
                monitors = [
                    Monitor(
                        index=i, x=m.x, y=m.y, width=m.width, height=m.height,
                        name=getattr(m, "name", None) or f"Display {i + 1}",
                        is_primary=bool(getattr(m, "is_primary", i == 0)),
                    )
                    for i, m in enumerate(found)
                ]
                log.info("detected %d monitor(s): %s",
                         len(monitors), "; ".join(str(m) for m in monitors))
                return monitors
        except Exception as exc:
            log.warning("monitor detection failed (%s); using fallback", exc)

        try:
            import pyautogui  # type: ignore[import-not-found]

            width, height = pyautogui.size()
        except Exception:
            width, height = 1920, 1080
        return [Monitor(0, 0, 0, int(width), int(height), "Display 1", True)]

    def _compute_virtual_bounds(self) -> Tuple[int, int, int, int]:
        """Bounding box spanning every monitor."""
        left = min(m.x for m in self.monitors)
        top = min(m.y for m in self.monitors)
        right = max(m.x + m.width for m in self.monitors)
        bottom = max(m.y + m.height for m in self.monitors)
        return (left, top, right, bottom)

    @property
    def primary(self) -> Monitor:
        """The primary display, or the first one detected."""
        return next((m for m in self.monitors if m.is_primary), self.monitors[0])

    def get(self, index: int) -> Monitor:
        """Return monitor ``index``; ``-1`` or out-of-range gives the primary."""
        if 0 <= index < len(self.monitors):
            return self.monitors[index]
        return self.primary

    def target_bounds(self, index: int) -> Tuple[int, int, int, int]:
        """Bounds to map the hand onto: one monitor, or the whole desktop."""
        if index < 0:
            return self.virtual_bounds
        return self.get(index).bounds

    def monitor_at(self, x: int, y: int) -> Monitor:
        """Which monitor contains a virtual-desktop point."""
        for monitor in self.monitors:
            if monitor.contains(x, y):
                return monitor
        return self.primary

    def refresh(self) -> None:
        """Re-detect displays (hot-plug support)."""
        self.monitors = self._detect()
        self.virtual_bounds = self._compute_virtual_bounds()


# --------------------------------------------------------------------------- #
# Mouse backends
# --------------------------------------------------------------------------- #

class MouseBackend:
    """Thin wrapper over the OS pointer, preferring ``pynput``.

    Falls back to ``pyautogui`` when pynput is unavailable.  All methods are
    no-ops (returning ``False``) if neither backend loads, so the rest of the
    application runs unchanged in a headless test environment.
    """

    def __init__(self) -> None:
        self._impl = "none"
        self._mouse = None
        self._button = None
        self._pyautogui = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Resolve the best available pointer backend."""
        try:
            from pynput.mouse import Button, Controller  # type: ignore[import-not-found]

            self._mouse = Controller()
            self._button = Button
            self._impl = "pynput"
            log.info("mouse backend: pynput")
            return
        except Exception as exc:
            log.debug("pynput unavailable: %s", exc)

        try:
            import pyautogui  # type: ignore[import-not-found]

            # The corner failsafe raises mid-gesture and the per-call pause
            # caps us at a fraction of our target frame rate.
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.0
            self._pyautogui = pyautogui
            self._impl = "pyautogui"
            log.info("mouse backend: pyautogui")
            return
        except Exception as exc:
            log.warning("no mouse backend available: %s", exc)

    @property
    def available(self) -> bool:
        """Whether pointer control is possible."""
        return self._impl != "none"

    @property
    def name(self) -> str:
        """Active backend name."""
        return self._impl

    def position(self) -> Tuple[int, int]:
        """Current cursor position in virtual-desktop pixels."""
        try:
            if self._impl == "pynput":
                x, y = self._mouse.position  # type: ignore[union-attr]
                return (int(x), int(y))
            if self._impl == "pyautogui":
                pos = self._pyautogui.position()  # type: ignore[union-attr]
                return (int(pos.x), int(pos.y))
        except Exception as exc:
            log.debug("position() failed: %s", exc)
        return (0, 0)

    def move_to(self, x: int, y: int) -> bool:
        """Warp the cursor to an absolute position."""
        with self._lock:
            try:
                if self._impl == "pynput":
                    self._mouse.position = (int(x), int(y))  # type: ignore[union-attr]
                    return True
                if self._impl == "pyautogui":
                    self._pyautogui.moveTo(int(x), int(y), _pause=False)  # type: ignore[union-attr]
                    return True
            except Exception as exc:
                log.debug("move_to failed: %s", exc)
            return False

    def _resolve_button(self, button: str):  # type: ignore[no-untyped-def]
        """Map a button name onto the backend's button constant."""
        mapping = {
            "left": getattr(self._button, "left", None),
            "right": getattr(self._button, "right", None),
            "middle": getattr(self._button, "middle", None),
        }
        return mapping.get(button, mapping["left"])

    def click(self, button: str = "left", count: int = 1) -> bool:
        """Press and release a mouse button ``count`` times."""
        with self._lock:
            try:
                if self._impl == "pynput":
                    self._mouse.click(self._resolve_button(button), count)  # type: ignore[union-attr]
                    return True
                if self._impl == "pyautogui":
                    self._pyautogui.click(button=button, clicks=count, _pause=False)  # type: ignore[union-attr]
                    return True
            except Exception as exc:
                log.debug("click failed: %s", exc)
            return False

    def press(self, button: str = "left") -> bool:
        """Hold a mouse button down."""
        with self._lock:
            try:
                if self._impl == "pynput":
                    self._mouse.press(self._resolve_button(button))  # type: ignore[union-attr]
                    return True
                if self._impl == "pyautogui":
                    self._pyautogui.mouseDown(button=button, _pause=False)  # type: ignore[union-attr]
                    return True
            except Exception as exc:
                log.debug("press failed: %s", exc)
            return False

    def release(self, button: str = "left") -> bool:
        """Release a held mouse button."""
        with self._lock:
            try:
                if self._impl == "pynput":
                    self._mouse.release(self._resolve_button(button))  # type: ignore[union-attr]
                    return True
                if self._impl == "pyautogui":
                    self._pyautogui.mouseUp(button=button, _pause=False)  # type: ignore[union-attr]
                    return True
            except Exception as exc:
                log.debug("release failed: %s", exc)
            return False

    def scroll(self, dx: int, dy: int) -> bool:
        """Scroll by discrete ticks."""
        with self._lock:
            try:
                if self._impl == "pynput":
                    self._mouse.scroll(dx, dy)  # type: ignore[union-attr]
                    return True
                if self._impl == "pyautogui":
                    if dy:
                        self._pyautogui.scroll(dy, _pause=False)  # type: ignore[union-attr]
                    if dx:
                        self._pyautogui.hscroll(dx, _pause=False)  # type: ignore[union-attr]
                    return True
            except Exception as exc:
                log.debug("scroll failed: %s", exc)
            return False


# --------------------------------------------------------------------------- #
# Cursor controller
# --------------------------------------------------------------------------- #

@dataclass
class CursorState:
    """Observable cursor state, surfaced on the dashboard."""

    x: int = 0
    y: int = 0
    speed: float = 0.0
    monitor_index: int = 0
    precision: bool = False
    dragging: bool = False
    moved: bool = False


class CursorController:
    """Maps normalized hand coordinates onto smoothed screen positions."""

    def __init__(self, cfg: CursorConfig,
                 layout: Optional[MonitorLayout] = None,
                 backend: Optional[MouseBackend] = None) -> None:
        self.cfg = cfg
        self.layout = layout or MonitorLayout()
        self.mouse = backend or MouseBackend()

        self._filter = OneEuroFilter2D(
            freq=60.0,
            min_cutoff=cfg.one_euro_min_cutoff,
            beta=cfg.one_euro_beta,
        )
        self._velocity = VelocityEstimator(window=5)
        self._last_raw: Optional[Tuple[float, float]] = None
        self._last_screen: Optional[Tuple[float, float]] = None
        self._last_time: float = 0.0

        self.state = CursorState()
        self.dragging = False
        self._drag_button = "left"
        self._scroll_remainder = 0.0
        self.enabled = True

        # Movement counters for the dashboard.
        self.total_distance = 0.0
        self.click_count = 0
        self.scroll_count = 0

    # -- configuration ---------------------------------------------------- #

    def apply_config(self, cfg: CursorConfig) -> None:
        """Hot-apply cursor settings."""
        self.cfg = cfg
        self._filter.tune(cfg.one_euro_min_cutoff, cfg.one_euro_beta)

    def reset(self) -> None:
        """Clear filter and velocity state (call when tracking re-acquires)."""
        self._filter.reset()
        self._velocity.reset()
        self._last_raw = None
        self._last_screen = None

    # -- mapping ---------------------------------------------------------- #

    def _active_region(self) -> Tuple[float, float, float, float]:
        """Normalized ``(x0, y0, x1, y1)`` sub-rectangle mapped to the screen."""
        margin = clamp(self.cfg.active_region_margin, 0.0, 0.45)
        return (margin, margin, 1.0 - margin, 1.0 - margin)

    def map_to_screen(self, point: Sequence[float],
                      monitor_index: Optional[int] = None) -> Tuple[float, float]:
        """Map a normalized hand point onto virtual-desktop pixels.

        Sensitivity expands or contracts the active region around its centre:
        higher sensitivity means a smaller region, so less hand travel covers
        the same screen distance.
        """
        x0, y0, x1, y1 = self._active_region()

        sensitivity = max(self.cfg.sensitivity, 0.05)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half_w = (x1 - x0) / 2.0 / sensitivity
        half_h = (y1 - y0) / 2.0 / sensitivity
        x0, x1 = cx - half_w, cx + half_w
        y0, y1 = cy - half_h, cy + half_h

        index = self.cfg.target_monitor if monitor_index is None else monitor_index
        left, top, right, bottom = self.layout.target_bounds(index)

        screen_x = remap(point[0], x0, x1, left, right)
        screen_y = remap(point[1], y0, y1, top, bottom)
        return (screen_x, screen_y)

    def update(self, point: Optional[Sequence[float]], timestamp: float,
               precision: bool = False) -> CursorState:
        """Process one frame's cursor source point.

        Args:
            point: Normalized ``(x, y)``, or ``None`` to leave the cursor put.
            timestamp: Monotonic seconds for this frame.
            precision: Whether precision mode is engaged.

        Returns:
            The updated :class:`CursorState`.
        """
        self.state.moved = False
        self.state.precision = precision
        self.state.dragging = self.dragging

        if point is None or not self.enabled:
            self._last_raw = None
            return self.state

        # 1. Dead zone — reject sub-threshold motion before it enters the filter.
        if self._last_raw is not None:
            travel = distance_2d(point, self._last_raw)
            if travel < self.cfg.dead_zone:
                return self.state
        self._last_raw = (float(point[0]), float(point[1]))

        # 2. Adaptive smoothing.
        smoothing = clamp(self.cfg.smoothing, 0.0, 0.98)
        # Map the 0-1 UI slider onto a One Euro cutoff: more smoothing means a
        # lower cutoff frequency.
        self._filter.tune(
            min_cutoff=max(0.05, self.cfg.one_euro_min_cutoff * (1.0 - smoothing) * 2.0),
            beta=self.cfg.one_euro_beta,
        )
        filtered = self._filter(self._last_raw, timestamp)

        # 3. Velocity-based prediction to hide pipeline latency.
        vx, vy = self._velocity.update(filtered, timestamp)
        lead = clamp(self.cfg.prediction_time, 0.0, 0.15)
        predicted = (filtered[0] + vx * lead, filtered[1] + vy * lead)

        # 4. Map into screen space.
        target_x, target_y = self.map_to_screen(predicted)

        # Precision mode: scale movement down *relative to the current cursor
        # position* rather than remapping, so fine adjustment stays local.
        gain = self.cfg.speed * (self.cfg.precision_factor if precision else 1.0)
        if self._last_screen is not None and gain != 1.0:
            prev_x, prev_y = self._last_screen
            target_x = prev_x + (target_x - prev_x) * gain
            target_y = prev_y + (target_y - prev_y) * gain

        left, top, right, bottom = self.layout.target_bounds(self.cfg.target_monitor)
        target_x = clamp(target_x, left, right - 1)
        target_y = clamp(target_y, top, bottom - 1)

        if self._last_screen is not None:
            moved = distance_2d((target_x, target_y), self._last_screen)
            self.total_distance += moved
            elapsed = max(timestamp - self._last_time, 1e-6)
            self.state.speed = moved / elapsed
        self._last_screen = (target_x, target_y)
        self._last_time = timestamp

        self.mouse.move_to(int(round(target_x)), int(round(target_y)))
        self.state.x = int(round(target_x))
        self.state.y = int(round(target_y))
        self.state.moved = True
        self.state.monitor_index = self.layout.monitor_at(
            self.state.x, self.state.y).index
        return self.state

    def recenter(self) -> None:
        """Snap the cursor to the centre of the active monitor."""
        monitor = self.layout.get(self.cfg.target_monitor)
        x, y = monitor.centre
        self.mouse.move_to(x, y)
        self._last_screen = (float(x), float(y))
        self.reset()
        log.info("cursor recentred on %s", monitor.name)

    # -- buttons ---------------------------------------------------------- #

    def click(self, button: str = "left", count: int = 1) -> bool:
        """Click a mouse button."""
        if not self.enabled:
            return False
        if self.mouse.click(button, count):
            self.click_count += count
            return True
        return False

    def start_drag(self, button: str = "left") -> bool:
        """Press and hold a button.  Idempotent."""
        if not self.enabled or self.dragging:
            return False
        if self.mouse.press(button):
            self.dragging = True
            self._drag_button = button
            log.debug("drag started (%s)", button)
            return True
        return False

    def end_drag(self) -> bool:
        """Release a held button.  Idempotent, and safe to call defensively."""
        if not self.dragging:
            return False
        self.dragging = False
        result = self.mouse.release(self._drag_button)
        log.debug("drag ended (%s)", self._drag_button)
        return result

    def scroll(self, delta: float) -> bool:
        """Scroll vertically.

        ``delta`` is normalized hand travel; positive means the hand moved
        down.  Fractional ticks are accumulated rather than truncated, so slow
        movement still scrolls instead of being rounded away to nothing.
        """
        if not self.enabled:
            return False

        direction = 1.0 if self.cfg.natural_scroll else -1.0
        self._scroll_remainder += delta * self.cfg.scroll_speed * 12.0 * direction

        ticks = int(self._scroll_remainder)
        if ticks == 0:
            return False
        self._scroll_remainder -= ticks
        self.scroll_count += abs(ticks)
        return self.mouse.scroll(0, ticks)

    def emergency_release(self) -> None:
        """Release every button we might be holding.

        Called by the emergency stop and on shutdown.  Buttons are released
        unconditionally rather than only when we believe they are held: if our
        state ever disagrees with the OS, the safe direction to be wrong is
        releasing a button that was already up.
        """
        self.dragging = False
        for button in ("left", "right", "middle"):
            self.mouse.release(button)
        log.info("emergency release: all mouse buttons cleared")
