"""Camera-feed overlay renderer.

Everything drawn *on top of* the video lives here: hand skeletons, the status
panel, motion trails, mode indicators and the volume/brightness meters.

Rendering runs once per frame, so it is written for speed rather than
elegance in a few places:

* Panels are drawn onto a copy and alpha-blended in a single
  ``addWeighted`` call rather than per-shape, because blending is the
  expensive part and doing it once amortises it.
* Rounded rectangles are composed from two rects and four ellipses instead
  of a mask, which is roughly an order of magnitude cheaper.
* Text is measured once and cached by string, since ``getTextSize`` shows up
  surprisingly high in a profile when called for every label every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detector import HAND_CONNECTIONS, FINGER_TIPS, HandLandmarks
from gesture_engine import EngineOutput, Mode
from themes import Theme
from utils import clamp

Colour = Tuple[int, int, int]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

#: Cache for text measurements, keyed by (text, font, scale, thickness).
_text_cache: Dict[Tuple[str, int, float, int], Tuple[int, int]] = {}


def measure_text(text: str, font: int = _FONT, scale: float = 0.5,
                 thickness: int = 1) -> Tuple[int, int]:
    """Return ``(width, height)`` for ``text``, memoised."""
    key = (text, font, scale, thickness)
    cached = _text_cache.get(key)
    if cached is None:
        (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
        cached = (width, height)
        # Bound the cache; gesture names and metrics are a small fixed set,
        # but FPS strings alone would grow it without limit.
        if len(_text_cache) > 512:
            _text_cache.clear()
        _text_cache[key] = cached
    return cached


def rounded_rect(image: np.ndarray, top_left: Tuple[int, int],
                 bottom_right: Tuple[int, int], colour: Colour,
                 radius: int = 10, thickness: int = -1) -> None:
    """Draw a rounded rectangle in place."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if radius == 0:
        cv2.rectangle(image, top_left, bottom_right, colour, thickness, cv2.LINE_AA)
        return

    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), colour, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), colour, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(image, (cx, cy), radius, colour, -1, cv2.LINE_AA)
        return

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), colour, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), colour, thickness, cv2.LINE_AA)
    for (cx, cy), angle in (((x1 + radius, y1 + radius), 180),
                            ((x2 - radius, y1 + radius), 270),
                            ((x2 - radius, y2 - radius), 0),
                            ((x1 + radius, y2 - radius), 90)):
        cv2.ellipse(image, (cx, cy), (radius, radius), angle, 0, 90,
                    colour, thickness, cv2.LINE_AA)


def glass_panel(frame: np.ndarray, top_left: Tuple[int, int],
                bottom_right: Tuple[int, int], theme: Theme,
                opacity: Optional[float] = None, radius: int = 12,
                border: bool = True) -> None:
    """Draw a translucent 'glass' panel in place.

    Blends a single filled rectangle rather than compositing a blurred crop:
    a real frosted-glass blur costs several milliseconds per panel, which is
    an unacceptable share of a 16 ms frame budget for a purely cosmetic
    effect.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return

    alpha = theme.overlay_opacity if opacity is None else opacity
    panel = frame.copy()
    rounded_rect(panel, (x1, y1), (x2, y2), theme.bgr("overlay_bg"), radius, -1)
    cv2.addWeighted(panel, alpha, frame, 1.0 - alpha, 0, frame)

    if border:
        rounded_rect(frame, (x1, y1), (x2, y2), theme.bgr("border"), radius, 1)


def draw_text(frame: np.ndarray, text: str, origin: Tuple[int, int],
              colour: Colour, scale: float = 0.5, thickness: int = 1,
              font: int = _FONT, shadow: bool = True) -> None:
    """Draw text with an optional shadow for legibility over video."""
    if shadow:
        cv2.putText(frame, text, (origin[0] + 1, origin[1] + 1), font, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, colour, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Hand rendering
# --------------------------------------------------------------------------- #

def draw_hand(frame: np.ndarray, hand: HandLandmarks, theme: Theme,
              show_skeleton: bool = True, show_landmarks: bool = True,
              highlight: Optional[Sequence[int]] = None) -> None:
    """Render one hand's skeleton and landmarks."""
    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in hand.points]

    if show_skeleton:
        skeleton = theme.bgr("skeleton")
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], skeleton, 2, cv2.LINE_AA)

    if show_landmarks:
        landmark = theme.bgr("landmark")
        accent = theme.bgr("accent")
        highlight_set = set(highlight or ())
        for index, point in enumerate(points):
            if index in highlight_set:
                cv2.circle(frame, point, 8, accent, -1, cv2.LINE_AA)
                cv2.circle(frame, point, 8, (255, 255, 255), 1, cv2.LINE_AA)
            elif index in FINGER_TIPS:
                cv2.circle(frame, point, 5, landmark, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, point, 3, landmark, -1, cv2.LINE_AA)


def draw_hand_label(frame: np.ndarray, hand: HandLandmarks, theme: Theme,
                    label: str) -> None:
    """Draw a small label above a hand's bounding box."""
    height, width = frame.shape[:2]
    x_min, y_min, _, _ = hand.bounding_box
    x = int(x_min * width)
    y = max(int(y_min * height) - 10, 18)

    text_w, text_h = measure_text(label, _FONT, 0.45, 1)
    glass_panel(frame, (x - 6, y - text_h - 8), (x + text_w + 8, y + 6),
                theme, opacity=0.65, radius=6, border=False)
    draw_text(frame, label, (x, y), theme.bgr("text"), 0.45, 1, shadow=False)


def draw_trail(frame: np.ndarray, trail: Sequence[Tuple[float, float]],
               theme: Theme) -> None:
    """Draw the motion trail, fading toward the oldest sample."""
    if len(trail) < 2:
        return

    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y in trail]
    accent = theme.bgr("secondary")
    count = len(points)

    for i in range(1, count):
        fade = i / count
        colour = tuple(int(c * fade) for c in accent)
        cv2.line(frame, points[i - 1], points[i], colour,
                 max(1, int(1 + 3 * fade)), cv2.LINE_AA)

    cv2.circle(frame, points[-1], 6, theme.bgr("accent"), -1, cv2.LINE_AA)


def draw_active_region(frame: np.ndarray, margin: float, theme: Theme) -> None:
    """Outline the region of the frame mapped to the screen.

    Showing this is genuinely useful rather than decorative: users otherwise
    have no idea why the cursor stops moving before their hand reaches the
    frame edge.
    """
    height, width = frame.shape[:2]
    x1, y1 = int(margin * width), int(margin * height)
    x2, y2 = int((1 - margin) * width), int((1 - margin) * height)

    colour = theme.bgr("border")
    dash = 14
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), colour, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), colour, 1, cv2.LINE_AA)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), colour, 1, cv2.LINE_AA)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), colour, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# HUD elements
# --------------------------------------------------------------------------- #

@dataclass
class HUDState:
    """Everything the status panel needs, gathered by the caller."""

    fps: float = 0.0
    mode: str = "Navigate"
    pose: str = "None"
    confidence: float = 0.0
    profile: str = "Default"
    tracking: bool = False
    hand_count: int = 0
    resolution: str = ""
    backend: str = ""
    paused: bool = False
    recording: bool = False
    precision: bool = False
    latency_ms: float = 0.0


def draw_status_panel(frame: np.ndarray, state: HUDState, theme: Theme) -> None:
    """Draw the top-left status panel."""
    padding = 14
    line_height = 22
    rows: List[Tuple[str, str, Colour]] = [
        ("FPS", f"{state.fps:.0f}", _fps_colour(state.fps, theme)),
        ("Mode", state.mode, theme.bgr("accent")),
        ("Gesture", state.pose, theme.bgr("text")),
        ("Confidence", f"{state.confidence:.0%}",
         _confidence_colour(state.confidence, theme)),
        ("Profile", state.profile, theme.bgr("text_muted")),
    ]
    if state.latency_ms:
        rows.append(("Latency", f"{state.latency_ms:.0f} ms", theme.bgr("text_muted")))

    label_width = max(measure_text(f"{label}", _FONT, 0.45, 1)[0]
                      for label, _, _ in rows)
    value_width = max(measure_text(value, _FONT_BOLD, 0.48, 1)[0]
                      for _, value, _ in rows)
    panel_width = label_width + value_width + padding * 3 + 12
    panel_height = len(rows) * line_height + padding * 2

    glass_panel(frame, (12, 12), (12 + panel_width, 12 + panel_height), theme)

    y = 12 + padding + 14
    for label, value, colour in rows:
        draw_text(frame, label, (12 + padding, y), theme.bgr("text_muted"), 0.45, 1)
        draw_text(frame, value, (12 + padding + label_width + 12, y), colour,
                  0.48, 1, font=_FONT_BOLD)
        y += line_height


def draw_status_badges(frame: np.ndarray, state: HUDState, theme: Theme) -> None:
    """Draw state badges along the top-right edge."""
    badges: List[Tuple[str, Colour]] = []

    if state.paused:
        badges.append(("PAUSED", theme.bgr("warning")))
    if not state.tracking and not state.paused:
        badges.append(("NO HAND", theme.bgr("error")))
    if state.recording:
        badges.append(("REC", theme.bgr("error")))
    if state.precision:
        badges.append(("PRECISION", theme.bgr("info")))
    if state.hand_count > 1:
        badges.append((f"{state.hand_count} HANDS", theme.bgr("secondary")))

    if not badges:
        return

    width = frame.shape[1]
    x = width - 12
    for text, colour in badges:
        text_w, text_h = measure_text(text, _FONT_BOLD, 0.44, 1)
        box_w = text_w + 20
        x -= box_w
        glass_panel(frame, (x, 12), (x + box_w, 12 + text_h + 16), theme,
                    opacity=0.78, radius=8, border=False)
        rounded_rect(frame, (x, 12), (x + box_w, 12 + text_h + 16), colour, 8, 1)
        draw_text(frame, text, (x + 10, 12 + text_h + 5), colour, 0.44, 1,
                  font=_FONT_BOLD, shadow=False)
        x -= 8


def draw_level_meter(frame: np.ndarray, level: float, label: str,
                     theme: Theme, icon: str = "") -> None:
    """Draw the vertical volume/brightness meter on the right edge."""
    height, width = frame.shape[:2]
    bar_w, bar_h = 26, min(240, int(height * 0.45))
    x = width - bar_w - 34
    y = (height - bar_h) // 2

    glass_panel(frame, (x - 16, y - 46), (x + bar_w + 16, y + bar_h + 40), theme,
                opacity=0.80, radius=14)

    # Track.
    rounded_rect(frame, (x, y), (x + bar_w, y + bar_h),
                 theme.bgr("surface_alt"), bar_w // 2, -1)

    # Fill, from the bottom up.
    level = clamp(level, 0.0, 1.0)
    fill_h = int(bar_h * level)
    if fill_h > 4:
        fill_y = y + bar_h - fill_h
        colour = theme.bgr("accent") if label != "Brightness" else theme.bgr("warning")
        rounded_rect(frame, (x, fill_y), (x + bar_w, y + bar_h), colour,
                     bar_w // 2, -1)

    percent = f"{level:.0%}"
    text_w, _ = measure_text(percent, _FONT_BOLD, 0.55, 1)
    draw_text(frame, percent, (x + bar_w // 2 - text_w // 2, y - 16),
              theme.bgr("text"), 0.55, 1, font=_FONT_BOLD)

    label_w, _ = measure_text(label, _FONT, 0.42, 1)
    draw_text(frame, label, (x + bar_w // 2 - label_w // 2, y + bar_h + 26),
              theme.bgr("text_muted"), 0.42, 1)


def draw_mode_banner(frame: np.ndarray, mode: Mode, theme: Theme) -> None:
    """Draw a centred banner naming the active non-default mode."""
    if mode in (Mode.NAVIGATE,):
        return

    text = mode.value.upper()
    text_w, text_h = measure_text(text, _FONT_BOLD, 0.7, 2)
    width = frame.shape[1]
    x1 = width // 2 - text_w // 2 - 22
    x2 = width // 2 + text_w // 2 + 22

    colour = {
        Mode.SLEEPING: theme.bgr("text_muted"),
        Mode.DRAG: theme.bgr("warning"),
        Mode.SCROLL: theme.bgr("secondary"),
        Mode.VOLUME: theme.bgr("accent"),
        Mode.BRIGHTNESS: theme.bgr("warning"),
        Mode.ZOOM: theme.bgr("info"),
        Mode.DRAW: theme.bgr("success"),
        Mode.PRESENT: theme.bgr("accent"),
    }.get(mode, theme.bgr("accent"))

    glass_panel(frame, (x1, 12), (x2, 12 + text_h + 18), theme,
                opacity=0.82, radius=10, border=False)
    rounded_rect(frame, (x1, 12), (x2, 12 + text_h + 18), colour, 10, 1)
    draw_text(frame, text, (x1 + 22, 12 + text_h + 8), colour, 0.7, 2,
              font=_FONT_BOLD, shadow=False)


def draw_confidence_bar(frame: np.ndarray, confidence: float, threshold: float,
                        theme: Theme) -> None:
    """Draw the bottom confidence bar with the threshold marked.

    Showing the threshold alongside the value is what makes a rejected
    gesture explicable: the user can see they were just under the line
    instead of concluding the app is broken.
    """
    height, width = frame.shape[:2]
    bar_w = min(360, width - 40)
    x = (width - bar_w) // 2
    y = height - 34

    glass_panel(frame, (x - 12, y - 16), (x + bar_w + 12, y + 20), theme,
                opacity=0.72, radius=10, border=False)
    rounded_rect(frame, (x, y), (x + bar_w, y + 10), theme.bgr("surface_alt"), 5, -1)

    fill = int(bar_w * clamp(confidence, 0.0, 1.0))
    if fill > 2:
        rounded_rect(frame, (x, y), (x + fill, y + 10),
                     _confidence_colour(confidence, theme), 5, -1)

    marker_x = x + int(bar_w * clamp(threshold, 0.0, 1.0))
    cv2.line(frame, (marker_x, y - 4), (marker_x, y + 14),
             theme.bgr("text"), 1, cv2.LINE_AA)


def draw_tracking_lost(frame: np.ndarray, theme: Theme) -> None:
    """Draw the centred 'show your hand' prompt."""
    height, width = frame.shape[:2]
    text = "Show your hand to the camera"
    text_w, text_h = measure_text(text, _FONT, 0.62, 1)
    x = width // 2 - text_w // 2
    y = height // 2

    glass_panel(frame, (x - 26, y - text_h - 20), (x + text_w + 26, y + 22),
                theme, opacity=0.75, radius=14)
    draw_text(frame, text, (x, y), theme.bgr("text_muted"), 0.62, 1)


def draw_calibration(frame: np.ndarray, instruction: str, progress: float,
                     overall: float, theme: Theme) -> None:
    """Draw the calibration wizard's on-camera guidance."""
    height, width = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    panel_h = 132
    y1 = height - panel_h - 24
    glass_panel(frame, (24, y1), (width - 24, height - 24), theme,
                opacity=0.88, radius=16)

    # Word-wrap the instruction to the panel width.
    words = instruction.split()
    lines: List[str] = []
    current = ""
    max_width = width - 96
    for word in words:
        candidate = f"{current} {word}".strip()
        if measure_text(candidate, _FONT, 0.56, 1)[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    y = y1 + 34
    for line in lines[:2]:
        draw_text(frame, line, (48, y), theme.bgr("text"), 0.56, 1)
        y += 26

    bar_x, bar_w = 48, width - 96
    bar_y = height - 62
    rounded_rect(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                 theme.bgr("surface_alt"), 4, -1)
    fill = int(bar_w * clamp(progress, 0.0, 1.0))
    if fill > 2:
        rounded_rect(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 8),
                     theme.bgr("accent"), 4, -1)

    label = f"Step {overall:.0%}"
    draw_text(frame, label, (bar_x, bar_y + 30), theme.bgr("text_muted"), 0.44, 1)


def _fps_colour(fps: float, theme: Theme) -> Colour:
    """Traffic-light colour for a frame rate."""
    if fps >= 25:
        return theme.bgr("success")
    if fps >= 15:
        return theme.bgr("warning")
    return theme.bgr("error")


def _confidence_colour(confidence: float, theme: Theme) -> Colour:
    """Traffic-light colour for a confidence value."""
    if confidence >= 0.8:
        return theme.bgr("success")
    if confidence >= 0.6:
        return theme.bgr("warning")
    return theme.bgr("error")


# --------------------------------------------------------------------------- #
# Composite renderer
# --------------------------------------------------------------------------- #

class OverlayRenderer:
    """Composes every overlay element onto a camera frame."""

    def __init__(self, theme: Theme) -> None:
        self.theme = theme
        self.show_landmarks = True
        self.show_skeleton = True
        self.show_panel = True
        self.show_trail = True
        self.show_active_region = True
        self.show_confidence_bar = True

    def set_theme(self, theme: Theme) -> None:
        """Swap the active palette."""
        self.theme = theme

    def render(self, frame: np.ndarray, hands: Sequence[HandLandmarks],
               output: Optional[EngineOutput], hud: HUDState,
               active_margin: float = 0.16,
               confidence_threshold: float = 0.72,
               level: Optional[Tuple[str, float]] = None) -> np.ndarray:
        """Draw the full overlay onto ``frame`` and return it.

        The frame is modified in place and also returned, so callers can
        chain without needing to know which happens.
        """
        theme = self.theme

        if self.show_active_region and not hud.paused:
            draw_active_region(frame, active_margin, theme)

        for hand in hands:
            draw_hand(frame, hand, theme, self.show_skeleton, self.show_landmarks)
            if len(hands) > 1:
                draw_hand_label(frame, hand, theme,
                                f"{hand.handedness} {hand.score:.0%}")

        if self.show_trail and output is not None and output.trail:
            draw_trail(frame, output.trail, theme)

        if output is not None:
            draw_mode_banner(frame, output.mode, theme)

        if self.show_panel:
            draw_status_panel(frame, hud, theme)
        draw_status_badges(frame, hud, theme)

        if level is not None:
            draw_level_meter(frame, level[1], level[0], theme)

        if not hud.tracking and not hud.paused:
            draw_tracking_lost(frame, theme)
        elif self.show_confidence_bar and hud.confidence > 0:
            draw_confidence_bar(frame, hud.confidence, confidence_threshold, theme)

        return frame
