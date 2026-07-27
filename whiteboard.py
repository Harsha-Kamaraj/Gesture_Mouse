"""Virtual whiteboard — draw in the air with your fingertip.

The canvas is a full-resolution RGBA layer composited over the camera feed.
Strokes are stored as vector paths rather than baked pixels, which is what
makes undo/redo, colour changes and resolution-independent export possible;
the raster layer is a cache rebuilt from the paths whenever they change.

Drawing state is driven by the same pose vocabulary as the rest of the app:
point to draw, open palm to erase, and gestures for undo/redo/clear.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from config import SCREENSHOT_DIR
from logger import get_logger

log = get_logger(__name__)

Point = Tuple[float, float]

#: Palette offered by the colour picker, as ``(name, #rrggbb)``.
PALETTE: List[Tuple[str, str]] = [
    ("White", "#FFFFFF"), ("Violet", "#7C5CFF"), ("Cyan", "#22D3EE"),
    ("Green", "#34D399"), ("Yellow", "#FBBF24"), ("Orange", "#FB923C"),
    ("Red", "#F87171"), ("Pink", "#F472B6"), ("Black", "#111111"),
]


class Tool(str, Enum):
    """Available whiteboard tools."""

    PEN = "Pen"
    HIGHLIGHTER = "Highlighter"
    ERASER = "Eraser"


@dataclass
class Stroke:
    """One continuous drawn path in normalized coordinates."""

    points: List[Point] = field(default_factory=list)
    colour: str = "#FFFFFF"
    width: int = 4
    tool: Tool = Tool.PEN
    created_at: float = field(default_factory=time.time)

    @property
    def is_drawable(self) -> bool:
        """Whether the stroke has enough points to render."""
        return len(self.points) >= 2

    def to_dict(self) -> dict:
        """Serialisable form."""
        return {
            "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
            "colour": self.colour,
            "width": self.width,
            "tool": self.tool.value,
        }


class Whiteboard:
    """Vector canvas with undo/redo, rendered onto camera frames."""

    MAX_HISTORY = 100

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.width = width
        self.height = height
        self.strokes: List[Stroke] = []
        self._redo_stack: List[Stroke] = []
        self._current: Optional[Stroke] = None

        self.active = False
        self.tool = Tool.PEN
        self.colour = "#7C5CFF"
        self.width_px = 5
        self.eraser_radius = 0.045

        self._cache: Optional[np.ndarray] = None
        self._cache_dirty = True

    # -- lifecycle -------------------------------------------------------- #

    def toggle(self) -> bool:
        """Show or hide the whiteboard.  Returns the new state."""
        self.active = not self.active
        if not self.active:
            self.end_stroke()
        log.info("whiteboard %s", "opened" if self.active else "closed")
        return self.active

    def resize(self, width: int, height: int) -> None:
        """Update the raster size; strokes are normalized so they survive."""
        if (width, height) != (self.width, self.height):
            self.width, self.height = width, height
            self._cache_dirty = True

    # -- drawing ---------------------------------------------------------- #

    def update(self, point: Optional[Point], drawing: bool) -> None:
        """Feed one frame of fingertip position.

        Args:
            point: Normalized fingertip position, or ``None`` when untracked.
            drawing: Whether the user is in the drawing pose.
        """
        if not self.active:
            return

        if not drawing or point is None:
            self.end_stroke()
            return

        if self.tool == Tool.ERASER:
            self.erase_at(point)
            return

        if self._current is None:
            self.begin_stroke(point)
        else:
            self._append(point)

    def begin_stroke(self, point: Point) -> None:
        """Start a new stroke at ``point``."""
        width = self.width_px * (3 if self.tool == Tool.HIGHLIGHTER else 1)
        self._current = Stroke(points=[point], colour=self.colour,
                               width=width, tool=self.tool)
        # A new stroke invalidates the redo branch, as in every editor.
        self._redo_stack.clear()

    def _append(self, point: Point) -> None:
        """Add a point to the in-progress stroke, skipping duplicates."""
        assert self._current is not None
        last = self._current.points[-1]
        if abs(point[0] - last[0]) < 0.001 and abs(point[1] - last[1]) < 0.001:
            return
        self._current.points.append(point)
        self._cache_dirty = True

    def end_stroke(self) -> Optional[Stroke]:
        """Commit the in-progress stroke."""
        stroke = self._current
        self._current = None
        if stroke is None or not stroke.is_drawable:
            return None

        self.strokes.append(stroke)
        if len(self.strokes) > self.MAX_HISTORY:
            self.strokes.pop(0)
        self._cache_dirty = True
        return stroke

    def erase_at(self, point: Point) -> int:
        """Delete strokes passing near ``point``.  Returns how many went."""
        before = len(self.strokes)
        radius_sq = self.eraser_radius ** 2
        self.strokes = [
            stroke for stroke in self.strokes
            if not any((px - point[0]) ** 2 + (py - point[1]) ** 2 <= radius_sq
                       for px, py in stroke.points)
        ]
        removed = before - len(self.strokes)
        if removed:
            self._cache_dirty = True
        return removed

    # -- history ---------------------------------------------------------- #

    def undo(self) -> bool:
        """Undo the most recent stroke."""
        self.end_stroke()
        if not self.strokes:
            return False
        self._redo_stack.append(self.strokes.pop())
        self._cache_dirty = True
        return True

    def redo(self) -> bool:
        """Redo the most recently undone stroke."""
        if not self._redo_stack:
            return False
        self.strokes.append(self._redo_stack.pop())
        self._cache_dirty = True
        return True

    def clear(self) -> int:
        """Erase everything.  Returns how many strokes were removed."""
        count = len(self.strokes)
        self.end_stroke()
        self.strokes.clear()
        self._redo_stack.clear()
        self._cache_dirty = True
        log.info("whiteboard cleared (%d strokes)", count)
        return count

    @property
    def can_undo(self) -> bool:
        """Whether there is anything to undo."""
        return bool(self.strokes)

    @property
    def can_redo(self) -> bool:
        """Whether there is anything to redo."""
        return bool(self._redo_stack)

    @property
    def stroke_count(self) -> int:
        """Number of committed strokes."""
        return len(self.strokes)

    # -- tools ------------------------------------------------------------ #

    def set_tool(self, tool: Tool) -> None:
        """Select the active tool."""
        self.end_stroke()
        self.tool = tool

    def set_colour(self, colour: str) -> None:
        """Select the pen colour."""
        self.colour = colour
        if self.tool == Tool.ERASER:
            self.tool = Tool.PEN

    def cycle_colour(self) -> str:
        """Advance to the next palette colour (gesture-driven picker)."""
        colours = [hex_code for _, hex_code in PALETTE]
        try:
            index = colours.index(self.colour)
        except ValueError:
            index = -1
        self.colour = colours[(index + 1) % len(colours)]
        return self.colour

    def set_width(self, width: int) -> None:
        """Set the brush width in pixels."""
        self.width_px = max(1, min(int(width), 40))

    # -- rendering -------------------------------------------------------- #

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Composite the canvas over ``frame`` and return it.

        Highlighter strokes are alpha-blended while pen strokes are drawn
        opaque, which is the visual difference users expect between the two.
        """
        if not self.active:
            return frame

        import cv2

        height, width = frame.shape[:2]
        self.resize(width, height)

        overlay = frame.copy()
        strokes = list(self.strokes)
        if self._current is not None and self._current.is_drawable:
            strokes.append(self._current)

        for stroke in strokes:
            colour = _bgr(stroke.colour)
            pixels = np.array(
                [[int(x * width), int(y * height)] for x, y in stroke.points],
                dtype=np.int32,
            )
            if stroke.tool == Tool.HIGHLIGHTER:
                layer = overlay.copy()
                cv2.polylines(layer, [pixels], False, colour, stroke.width,
                              cv2.LINE_AA)
                cv2.addWeighted(layer, 0.45, overlay, 0.55, 0, overlay)
            else:
                cv2.polylines(overlay, [pixels], False, colour, stroke.width,
                              cv2.LINE_AA)

        return overlay

    def to_image(self, width: Optional[int] = None, height: Optional[int] = None,
                 background: str = "#111318") -> np.ndarray:
        """Render the drawing alone onto a solid background."""
        import cv2

        width = width or self.width
        height = height or self.height
        canvas = np.full((height, width, 3), _bgr(background), dtype=np.uint8)

        for stroke in self.strokes:
            pixels = np.array(
                [[int(x * width), int(y * height)] for x, y in stroke.points],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pixels], False, _bgr(stroke.colour),
                          stroke.width, cv2.LINE_AA)
        return canvas

    def save(self, directory: Path = SCREENSHOT_DIR) -> Optional[Path]:
        """Write the drawing to a PNG.  Returns the path, or ``None``."""
        if not self.strokes:
            log.info("nothing to save: whiteboard is empty")
            return None
        try:
            import cv2

            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"whiteboard_{datetime.now():%Y%m%d_%H%M%S}.png"
            if cv2.imwrite(str(path), self.to_image()):
                log.info("whiteboard saved: %s", path.name)
                return path
        except Exception as exc:
            log.error("whiteboard save failed: %s", exc)
        return None


def _bgr(colour: str) -> Tuple[int, int, int]:
    """Convert ``#rrggbb`` to an OpenCV BGR tuple."""
    colour = colour.lstrip("#")
    if len(colour) == 3:
        colour = "".join(c * 2 for c in colour)
    r, g, b = (int(colour[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)
