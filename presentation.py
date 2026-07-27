"""Air presentation mode.

Presenting is the one context where a gesture mouse beats a real mouse
outright: you are standing away from the desk with no surface to click on.
This mode swaps the normal gesture vocabulary for a small, deliberately
coarse one — big, unambiguous motions that survive being performed from three
metres away while you are also talking.

It also intentionally *raises* cooldowns and confidence thresholds.  A missed
slide advance costs a second; an accidental one during a talk is far worse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from logger import get_logger
from utils import Cooldown, clamp

log = get_logger(__name__)

Point = Tuple[float, float]


class PointerStyle(str, Enum):
    """Laser pointer rendering styles."""

    DOT = "Dot"
    RING = "Ring"
    GLOW = "Glow"
    SPOTLIGHT = "Spotlight"


@dataclass
class PresentationConfig:
    """Tuning for presentation mode."""

    #: Seconds between accepted slide changes.
    slide_cooldown: float = 0.9
    #: Confidence floor, higher than normal — misfires are costly on stage.
    min_confidence: float = 0.80
    #: Minimum horizontal travel for a slide swipe.
    swipe_distance: float = 0.26
    pointer_style: PointerStyle = PointerStyle.GLOW
    pointer_colour: str = "#FF3B30"
    pointer_size: int = 18
    #: Fraction the screen dims outside the spotlight.
    spotlight_dim: float = 0.72
    show_timer: bool = True
    #: Warn the presenter at this elapsed time (seconds); 0 disables.
    time_limit: float = 0.0


@dataclass
class PresentationStats:
    """Session statistics shown on the presenter overlay."""

    started_at: float = field(default_factory=time.monotonic)
    slides_forward: int = 0
    slides_back: int = 0
    pointer_seconds: float = 0.0
    blackouts: int = 0

    @property
    def elapsed(self) -> float:
        """Seconds since the presentation started."""
        return time.monotonic() - self.started_at

    @property
    def net_slide(self) -> int:
        """Estimated current slide offset from the start."""
        return self.slides_forward - self.slides_back


class PresentationMode:
    """Slide control, laser pointer and screen blanking."""

    def __init__(self, config: Optional[PresentationConfig] = None,
                 on_action: Optional[Callable[[str], None]] = None) -> None:
        self.config = config or PresentationConfig()
        self.on_action = on_action

        self.active = False
        self.pointer_visible = False
        self.blackout = False
        self.pointer_position: Optional[Point] = None

        self.stats = PresentationStats()
        self._slide_cooldown = Cooldown(self.config.slide_cooldown)
        self._blackout_cooldown = Cooldown(1.2)
        self._pointer_since: Optional[float] = None

    # -- lifecycle -------------------------------------------------------- #

    def toggle(self) -> bool:
        """Enter or leave presentation mode.  Returns the new state."""
        self.active = not self.active
        if self.active:
            self.stats = PresentationStats()
            log.info("presentation mode started")
        else:
            self.pointer_visible = False
            self.blackout = False
            log.info("presentation mode ended after %.0fs (%d slides)",
                     self.stats.elapsed, self.stats.net_slide)
        return self.active

    def _emit(self, action: str) -> None:
        """Notify the application that a presentation action fired."""
        if self.on_action:
            try:
                self.on_action(action)
            except Exception as exc:
                log.debug("presentation callback failed: %s", exc)

    # -- slide control ---------------------------------------------------- #

    def next_slide(self) -> bool:
        """Advance one slide, respecting the cooldown."""
        if not self._slide_cooldown.ready():
            return False
        self.stats.slides_forward += 1
        self._emit("next_slide")
        log.debug("next slide (%d)", self.stats.net_slide)
        return True

    def previous_slide(self) -> bool:
        """Go back one slide, respecting the cooldown."""
        if not self._slide_cooldown.ready():
            return False
        self.stats.slides_back += 1
        self._emit("prev_slide")
        log.debug("previous slide (%d)", self.stats.net_slide)
        return True

    def toggle_blackout(self) -> bool:
        """Blank or restore the projected screen."""
        if not self._blackout_cooldown.ready():
            return self.blackout
        self.blackout = not self.blackout
        if self.blackout:
            self.stats.blackouts += 1
        self._emit("black_screen")
        return self.blackout

    # -- pointer ---------------------------------------------------------- #

    def update_pointer(self, point: Optional[Point], visible: bool,
                       timestamp: float) -> None:
        """Update the laser pointer position and visibility."""
        if not self.active:
            return

        if visible and point is not None:
            if self._pointer_since is None:
                self._pointer_since = timestamp
            self.pointer_position = point
            self.pointer_visible = True
        else:
            if self._pointer_since is not None:
                self.stats.pointer_seconds += timestamp - self._pointer_since
                self._pointer_since = None
            self.pointer_visible = False

    def cycle_pointer_style(self) -> PointerStyle:
        """Move to the next pointer style."""
        styles = list(PointerStyle)
        index = styles.index(self.config.pointer_style)
        self.config.pointer_style = styles[(index + 1) % len(styles)]
        return self.config.pointer_style

    # -- rendering -------------------------------------------------------- #

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pointer and blackout onto a frame."""
        if not self.active:
            return frame

        import cv2

        height, width = frame.shape[:2]

        if self.blackout:
            frame = (frame * 0.06).astype(np.uint8)
            cv2.putText(frame, "SCREEN BLANKED", (width // 2 - 150, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2, cv2.LINE_AA)
            return frame

        if not (self.pointer_visible and self.pointer_position):
            return frame

        x = int(clamp(self.pointer_position[0], 0.0, 1.0) * width)
        y = int(clamp(self.pointer_position[1], 0.0, 1.0) * height)
        colour = _bgr(self.config.pointer_colour)
        size = self.config.pointer_size
        style = self.config.pointer_style

        if style == PointerStyle.DOT:
            cv2.circle(frame, (x, y), size, colour, -1, cv2.LINE_AA)

        elif style == PointerStyle.RING:
            cv2.circle(frame, (x, y), size, colour, 3, cv2.LINE_AA)
            cv2.circle(frame, (x, y), max(2, size // 4), colour, -1, cv2.LINE_AA)

        elif style == PointerStyle.GLOW:
            # Concentric translucent rings approximate a soft glow far more
            # cheaply than a real Gaussian blur, which is too slow per frame.
            glow = frame.copy()
            for radius, alpha in ((size * 3, 0.10), (size * 2, 0.18), (size, 0.85)):
                layer = glow.copy()
                cv2.circle(layer, (x, y), int(radius), colour, -1, cv2.LINE_AA)
                cv2.addWeighted(layer, alpha, glow, 1 - alpha, 0, glow)
            frame = glow
            cv2.circle(frame, (x, y), max(2, size // 3), (255, 255, 255), -1,
                       cv2.LINE_AA)

        elif style == PointerStyle.SPOTLIGHT:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (x, y), size * 6, 255, -1, cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (61, 61), 0)
            dimmed = (frame * (1.0 - self.config.spotlight_dim)).astype(np.uint8)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            frame = (frame * alpha + dimmed * (1 - alpha)).astype(np.uint8)

        return frame

    def overlay_lines(self) -> List[str]:
        """Status lines for the presenter's own overlay."""
        lines = [f"Slide {self.stats.net_slide:+d}"]
        if self.config.show_timer:
            minutes, seconds = divmod(int(self.stats.elapsed), 60)
            marker = ""
            if self.config.time_limit and self.stats.elapsed > self.config.time_limit:
                marker = "  OVER TIME"
            lines.append(f"{minutes:02d}:{seconds:02d}{marker}")
        lines.append(f"Pointer: {self.config.pointer_style.value}")
        return lines

    def apply_to_gesture_config(self, gestures: object) -> Dict[str, float]:
        """Return the gesture-config overrides presentation mode wants.

        Returned rather than applied so the caller can restore the previous
        values on exit — a mode should never silently mutate the user's
        saved profile.
        """
        return {
            "global_cooldown": self.config.slide_cooldown,
            "min_confidence": self.config.min_confidence,
        }


def _bgr(colour: str) -> Tuple[int, int, int]:
    """Convert ``#rrggbb`` to an OpenCV BGR tuple."""
    colour = colour.lstrip("#")
    if len(colour) == 3:
        colour = "".join(c * 2 for c in colour)
    r, g, b = (int(colour[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)
