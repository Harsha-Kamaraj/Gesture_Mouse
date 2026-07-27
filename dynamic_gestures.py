"""Dynamic (motion-path) gesture recognition.

Static poses answer *"what shape is the hand making?"*.  This module answers
*"what did the hand draw in the air?"* — circles, waves, letters, polygons.

Why the $1 Unistroke Recognizer
-------------------------------
The obvious modern answer is an LSTM or a small Transformer over the landmark
sequence.  For this problem that is the wrong trade:

* it needs a labelled training set the user does not have,
* the user must be able to *record a new gesture once* and have it work
  immediately (see :mod:`gesture_recorder`), which is a one-shot learning
  problem, and
* a misfire that opens the wrong application is expensive, so the scoring
  must be explainable and tunable rather than a black-box logit.

The $1 Recognizer (Wobbrock, Wilson & Li, UIST 2007) is a template matcher
that solves exactly this: resample the stroke to a fixed point count,
rotation-normalise, scale to a reference square, translate to the origin, then
score against each stored template by mean point-wise distance under a
golden-section search over rotations.  It learns from a single example, runs
in well under a millisecond, and produces a bounded ``[0, 1]`` score that maps
naturally onto our confidence threshold.

A ``Protractor``-style closed-form angular distance is used for the final
scoring pass because it is both faster and more stable than the original
golden-section search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import path_length

#: Points every stroke is resampled to.  64 is the value the original paper
#: found optimal; higher costs time without improving accuracy.
RESAMPLE_POINTS = 64

#: Edge length of the reference square strokes are scaled into.
SQUARE_SIZE = 250.0

#: Half-diagonal of the reference square — the maximum possible mean distance,
#: used to normalise raw distances into a ``[0, 1]`` score.
HALF_DIAGONAL = 0.5 * math.sqrt(SQUARE_SIZE ** 2 + SQUARE_SIZE ** 2)

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Geometry pipeline
# --------------------------------------------------------------------------- #

def _resample(points: Sequence[Point], n: int = RESAMPLE_POINTS) -> List[Point]:
    """Resample a stroke to exactly ``n`` equidistant points.

    Removes the recogniser's dependence on drawing *speed*: a slowly drawn
    circle and a fast one become the same point sequence.
    """
    if len(points) < 2:
        return [tuple(points[0]) for _ in range(n)] if points else [(0.0, 0.0)] * n

    interval = path_length(points) / (n - 1)
    if interval <= 0:
        return [tuple(points[0]) for _ in range(n)]

    resampled: List[Point] = [tuple(points[0])]  # type: ignore[list-item]
    accumulated = 0.0
    src = [tuple(p) for p in points]

    i = 1
    while i < len(src):
        prev, curr = src[i - 1], src[i]
        segment = math.dist(prev, curr)
        if accumulated + segment >= interval:
            t = (interval - accumulated) / segment if segment > 0 else 0.0
            new_point = (prev[0] + t * (curr[0] - prev[0]),
                         prev[1] + t * (curr[1] - prev[1]))
            resampled.append(new_point)
            # Continue interpolating from the point we just inserted.
            src.insert(i, new_point)
            accumulated = 0.0
        else:
            accumulated += segment
        i += 1

    # Floating point drift can leave us one point short.
    while len(resampled) < n:
        resampled.append(tuple(src[-1]))  # type: ignore[arg-type]
    return resampled[:n]


def _centroid(points: Sequence[Point]) -> Point:
    """Arithmetic mean of the stroke."""
    arr = np.asarray(points, dtype=np.float64)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def _indicative_angle(points: Sequence[Point]) -> float:
    """Angle from the centroid to the first point, in radians."""
    cx, cy = _centroid(points)
    return math.atan2(cy - points[0][1], cx - points[0][0])


def _rotate_by(points: Sequence[Point], radians: float) -> List[Point]:
    """Rotate the stroke about its centroid."""
    cx, cy = _centroid(points)
    cos_r, sin_r = math.cos(radians), math.sin(radians)
    return [
        ((p[0] - cx) * cos_r - (p[1] - cy) * sin_r + cx,
         (p[0] - cx) * sin_r + (p[1] - cy) * cos_r + cy)
        for p in points
    ]


#: Below this width/height (or height/width) ratio a stroke is treated as
#: essentially one-dimensional and scaled uniformly instead of to the square.
_DEGENERATE_ASPECT = 0.10


def _scale_to_square(points: Sequence[Point], size: float = SQUARE_SIZE) -> List[Point]:
    """Scale the stroke into a ``size`` × ``size`` box.

    Non-uniform scaling is what makes a tall skinny circle match a wide flat
    one.  It is also why $1 cannot distinguish gestures differing *only* in
    aspect ratio — an acceptable trade for our shape set.

    Near-1D strokes are the dangerous case: dividing by a near-zero extent
    amplifies sensor noise into a full-scale garbage shape that can outscore
    the real match.  Those are scaled uniformly instead, which keeps them
    recognisably linear so they simply fail to match any 2D template.
    """
    arr = np.asarray(points, dtype=np.float64)
    width = float(arr[:, 0].max() - arr[:, 0].min())
    height = float(arr[:, 1].max() - arr[:, 1].min())
    longest = max(width, height)
    if longest < 1e-9:
        return [(0.0, 0.0) for _ in points]

    if min(width, height) / longest < _DEGENERATE_ASPECT:
        factor = size / longest
        return [(p[0] * factor, p[1] * factor) for p in points]

    return [(p[0] * size / width, p[1] * size / height) for p in points]


def _translate_to_origin(points: Sequence[Point]) -> List[Point]:
    """Move the stroke's centroid to ``(0, 0)``."""
    cx, cy = _centroid(points)
    return [(p[0] - cx, p[1] - cy) for p in points]


def _vectorize(points: Sequence[Point]) -> np.ndarray:
    """Flatten and L2-normalise a stroke into a unit vector (Protractor)."""
    flat = np.asarray(points, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 1e-12 else flat


def normalize_stroke(points: Sequence[Point]) -> Tuple[List[Point], np.ndarray]:
    """Run the full $1 normalisation pipeline on a raw stroke.

    Returns:
        A ``(points, vector)`` pair: the geometric form used for debugging and
        rendering, and the unit vector used for scoring.
    """
    resampled = _resample(points)
    rotated = _rotate_by(resampled, -_indicative_angle(resampled))
    scaled = _scale_to_square(rotated)
    translated = _translate_to_origin(scaled)
    return translated, _vectorize(translated)


def _optimal_cosine_distance(template: np.ndarray, candidate: np.ndarray) -> float:
    """Closed-form minimum angular distance between two stroke vectors.

    This is the Protractor scoring step: instead of iteratively searching for
    the best rotation, it solves for it analytically.
    """
    if template.shape != candidate.shape or template.size == 0:
        return math.pi

    a = float(np.dot(template, candidate))
    # Cross-term over alternating x/y components.
    tx, ty = template[0::2], template[1::2]
    cx, cy = candidate[0::2], candidate[1::2]
    b = float(np.dot(tx, cy) - np.dot(ty, cx))

    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return math.pi
    angle = math.atan(b / a) if abs(a) > 1e-12 else math.pi / 2
    value = a * math.cos(angle) + b * math.sin(angle)
    return math.acos(max(-1.0, min(1.0, value)))


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

@dataclass
class GestureTemplate:
    """A single stored stroke exemplar."""

    name: str
    points: List[Point]
    #: Set to False to keep the template on disk but exclude it from matching.
    enabled: bool = True
    #: Cached unit vector; rebuilt on load.
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def __post_init__(self) -> None:
        if self.vector.size == 0 and self.points:
            _, self.vector = normalize_stroke(self.points)

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable form (the cached vector is not persisted)."""
        return {
            "name": self.name,
            "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GestureTemplate":
        """Rebuild a template from its serialised form."""
        raw = data.get("points") or []
        points = [(float(p[0]), float(p[1])) for p in raw]  # type: ignore[index]
        return cls(
            name=str(data.get("name", "unnamed")),
            points=points,
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class DynamicMatch:
    """Result of matching a stroke against the template library."""

    name: str
    score: float
    #: Every candidate considered, best first — useful for the debug overlay.
    ranked: List[Tuple[str, float]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Built-in stroke definitions
# --------------------------------------------------------------------------- #

def _circle(samples: int = 48, clockwise: bool = False) -> List[Point]:
    """Unit circle stroke."""
    sign = 1.0 if clockwise else -1.0
    return [
        (math.cos(sign * 2 * math.pi * i / samples),
         math.sin(sign * 2 * math.pi * i / samples))
        for i in range(samples + 1)
    ]


def _polygon(sides: int, rotation: float = -math.pi / 2) -> List[Point]:
    """Regular polygon stroke with densely sampled edges."""
    corners = [
        (math.cos(rotation + 2 * math.pi * i / sides),
         math.sin(rotation + 2 * math.pi * i / sides))
        for i in range(sides + 1)
    ]
    points: List[Point] = []
    for i in range(len(corners) - 1):
        a, b = corners[i], corners[i + 1]
        for t in np.linspace(0.0, 1.0, 12, endpoint=False):
            points.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    points.append(corners[-1])
    return points


def _polyline(vertices: Sequence[Point], per_edge: int = 14) -> List[Point]:
    """Densely sample a polyline through ``vertices``."""
    points: List[Point] = []
    for i in range(len(vertices) - 1):
        a, b = vertices[i], vertices[i + 1]
        for t in np.linspace(0.0, 1.0, per_edge, endpoint=False):
            points.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    points.append(tuple(vertices[-1]))  # type: ignore[arg-type]
    return points


def _wave(cycles: int = 2, samples: int = 60) -> List[Point]:
    """Horizontal sinusoid — the "wave" gesture."""
    return [
        (-1.0 + 2.0 * i / samples, 0.45 * math.sin(2 * math.pi * cycles * i / samples))
        for i in range(samples + 1)
    ]


def _s_curve(samples: int = 60) -> List[Point]:
    """Letter S drawn as two opposing arcs."""
    points: List[Point] = []
    for i in range(samples + 1):
        t = i / samples
        angle = math.pi * (1.0 - 2.0 * t)
        if t < 0.5:
            points.append((math.cos(angle) * 0.6, 0.5 + math.sin(angle) * 0.5))
        else:
            points.append((-math.cos(angle) * 0.6, -0.5 + math.sin(angle) * 0.5))
    return points


def builtin_templates() -> List[GestureTemplate]:
    """The stroke library shipped with the application."""
    return [
        GestureTemplate("circle", _circle()),
        GestureTemplate("circle_cw", _circle(clockwise=True)),
        GestureTemplate("triangle", _polygon(3)),
        GestureTemplate("square", _polygon(4, rotation=math.pi / 4)),
        GestureTemplate("wave", _wave()),
        GestureTemplate("z", _polyline([(-1, 1), (1, 1), (-1, -1), (1, -1)])),
        GestureTemplate("s", _s_curve()),
        GestureTemplate("v_check", _polyline([(-1, 0.6), (0, -1), (1, 1)])),
        GestureTemplate("caret", _polyline([(-1, -1), (0, 1), (1, -1)])),
    ]


# --------------------------------------------------------------------------- #
# Directional swipes
# --------------------------------------------------------------------------- #

#: Net displacement must be at least this fraction of the path length for a
#: stroke to count as a straight swipe rather than a wandering doodle.
_SWIPE_STRAIGHTNESS = 0.82


@dataclass
class SwipeResult:
    """A detected directional swipe."""

    direction: str          # "left" | "right" | "up" | "down"
    confidence: float       # [0, 1]
    displacement: float     # net travel in normalized units
    speed: float            # normalized units per second


def detect_swipe(points: Sequence[Point], duration: float,
                 min_displacement: float = 0.22,
                 min_speed: float = 0.45) -> Optional[SwipeResult]:
    """Detect a straight directional swipe in a motion trail.

    Swipes are handled *outside* the $1 recogniser on purpose.  $1 normalises
    away both rotation and direction, so a left swipe and a right swipe reduce
    to the identical template — it structurally cannot tell them apart.  A
    direct test on net displacement is both correct and cheaper.

    Args:
        points: Raw trail samples in normalized image coordinates.
        duration: Seconds spanned by the trail.
        min_displacement: Minimum straight-line travel required.
        min_speed: Minimum average speed required, to reject slow drifting.

    Returns:
        A :class:`SwipeResult`, or ``None`` when the motion is not a swipe.
    """
    if len(points) < 5 or duration <= 1e-6:
        return None

    start, end = points[0], points[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    displacement = math.hypot(dx, dy)
    if displacement < min_displacement:
        return None

    travelled = path_length(points)
    if travelled < 1e-9:
        return None

    # A straight swipe travels almost exactly its net displacement; a circle
    # travels far further than it displaces.
    straightness = displacement / travelled
    if straightness < _SWIPE_STRAIGHTNESS:
        return None

    speed = displacement / duration
    if speed < min_speed:
        return None

    # Dominant axis wins; y grows downward in image coordinates.
    if abs(dx) >= abs(dy):
        direction = "right" if dx > 0 else "left"
        dominance = abs(dx) / (abs(dx) + abs(dy) + 1e-9)
    else:
        direction = "down" if dy > 0 else "up"
        dominance = abs(dy) / (abs(dx) + abs(dy) + 1e-9)

    confidence = min(1.0, straightness * dominance * min(1.0, speed / min_speed))
    return SwipeResult(direction, confidence, displacement, speed)


# --------------------------------------------------------------------------- #
# Recognizer
# --------------------------------------------------------------------------- #

class DollarOneRecognizer:
    """Template-matching recogniser for air-drawn strokes.

    Example:
        >>> rec = DollarOneRecognizer()
        >>> match = rec.recognize(_circle(30))
        >>> match.name
        'circle'
    """

    def __init__(self, templates: Optional[Sequence[GestureTemplate]] = None) -> None:
        self._templates: List[GestureTemplate] = list(
            templates if templates is not None else builtin_templates()
        )

    # -- library management ---------------------------------------------- #

    @property
    def templates(self) -> List[GestureTemplate]:
        """All templates, including disabled ones."""
        return list(self._templates)

    @property
    def names(self) -> List[str]:
        """Unique template names, preserving insertion order."""
        seen: Dict[str, None] = {}
        for template in self._templates:
            seen.setdefault(template.name, None)
        return list(seen)

    def add(self, template: GestureTemplate) -> None:
        """Add a template (multiple exemplars per name are supported)."""
        self._templates.append(template)

    def remove(self, name: str) -> int:
        """Remove every template called ``name``; returns how many were removed."""
        before = len(self._templates)
        self._templates = [t for t in self._templates if t.name != name]
        return before - len(self._templates)

    def rename(self, old: str, new: str) -> int:
        """Rename every template called ``old``; returns how many changed."""
        count = 0
        for template in self._templates:
            if template.name == old:
                template.name = new
                count += 1
        return count

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable every template called ``name``."""
        for template in self._templates:
            if template.name == name:
                template.enabled = enabled

    def clear(self) -> None:
        """Drop every template."""
        self._templates.clear()

    # -- matching --------------------------------------------------------- #

    def recognize(self, points: Sequence[Point],
                  min_score: float = 0.0) -> Optional[DynamicMatch]:
        """Match a raw stroke against the library.

        Args:
            points: Raw ``(x, y)`` samples in any coordinate system.
            min_score: Reject matches scoring below this value.

        Returns:
            The best :class:`DynamicMatch`, or ``None`` if the stroke was too
            short or nothing scored above ``min_score``.
        """
        if len(points) < 8:
            return None

        active = [t for t in self._templates if t.enabled and t.vector.size]
        if not active:
            return None

        _, vector = normalize_stroke(points)

        scored: Dict[str, float] = {}
        for template in active:
            if template.vector.shape != vector.shape:
                continue
            angle = _optimal_cosine_distance(template.vector, vector)
            # Map angular distance onto a bounded, monotonic score.
            score = 1.0 / (1.0 + angle)
            # Keep the best exemplar per gesture name.
            if score > scored.get(template.name, 0.0):
                scored[template.name] = score

        if not scored:
            return None

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ranked[0]
        if best_score < min_score:
            return None
        return DynamicMatch(name=best_name, score=best_score, ranked=ranked[:5])


class MotionTrail:
    """Rolling buffer of hand positions used as the stroke source.

    Tracks the drawing point over time and reports when enough travel has
    accumulated to be worth attempting recognition.
    """

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self._points: List[Point] = []
        self._times: List[float] = []

    def add(self, point: Point, timestamp: float) -> None:
        """Append a sample, evicting the oldest when at capacity."""
        self._points.append((float(point[0]), float(point[1])))
        self._times.append(timestamp)
        if len(self._points) > self.capacity:
            self._points.pop(0)
            self._times.pop(0)

    def clear(self) -> None:
        """Discard the trail."""
        self._points.clear()
        self._times.clear()

    @property
    def points(self) -> List[Point]:
        """Samples, oldest first."""
        return list(self._points)

    @property
    def length(self) -> float:
        """Total travelled distance in normalized units."""
        return path_length(self._points) if len(self._points) > 1 else 0.0

    @property
    def duration(self) -> float:
        """Seconds spanned by the trail."""
        return (self._times[-1] - self._times[0]) if len(self._times) > 1 else 0.0

    @property
    def is_stationary(self) -> bool:
        """True when the last few samples barely moved (stroke has ended)."""
        if len(self._points) < 5:
            return False
        return path_length(self._points[-5:]) < 0.012

    def __len__(self) -> int:
        return len(self._points)
