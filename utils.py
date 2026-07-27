"""Reusable math, filtering and timing primitives.

Everything here is deliberately dependency-light (NumPy only) and free of
side effects so it can be unit-tested without a camera, a screen or a
MediaPipe install.  The signal-processing helpers are the reason the cursor
feels smooth rather than twitchy, so they carry the most documentation.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Sequence, Tuple

import numpy as np

Vector2 = Tuple[float, float]
Vector3 = Tuple[float, float, float]


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #

def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the inclusive range ``[low, high]``."""
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate from ``a`` to ``b`` by factor ``t``."""
    return a + (b - a) * t


def remap(value: float, in_lo: float, in_hi: float,
          out_lo: float, out_hi: float, clamped: bool = True) -> float:
    """Map ``value`` from one range onto another.

    Used to convert the hand's normalized position inside the *active region*
    into screen coordinates.
    """
    if math.isclose(in_hi, in_lo):
        return out_lo
    t = (value - in_lo) / (in_hi - in_lo)
    if clamped:
        t = clamp(t, 0.0, 1.0)
    return out_lo + (out_hi - out_lo) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation — smooth 0→1 ramp with zero end derivatives."""
    t = clamp((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #

def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two points of equal dimensionality."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance using only the first two components."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Sequence[float], b: Sequence[float]) -> Vector2:
    """Midpoint of two 2D points."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_between(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Return the interior angle ABC in degrees.

    ``b`` is the vertex.  This is the core primitive for deciding whether a
    finger is extended: a straight finger has near-180° joint angles while a
    curled one drops well below 140°.
    """
    v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def direction_angle(start: Sequence[float], end: Sequence[float]) -> float:
    """Bearing from ``start`` to ``end`` in degrees, 0° = +x, CCW positive."""
    return math.degrees(math.atan2(-(end[1] - start[1]), end[0] - start[0]))


def path_length(points: Sequence[Sequence[float]]) -> float:
    """Total polyline length of ``points``."""
    return sum(distance_2d(points[i], points[i + 1]) for i in range(len(points) - 1))


def centroid(points: Sequence[Sequence[float]]) -> Vector2:
    """Arithmetic mean of a point cloud."""
    if not points:
        return (0.0, 0.0)
    xs = sum(p[0] for p in points) / len(points)
    ys = sum(p[1] for p in points) / len(points)
    return (xs, ys)


def bounding_box(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` for a point cloud."""
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

class LowPassFilter:
    """First-order exponential low-pass filter with dynamic alpha."""

    __slots__ = ("_value", "_initialised")

    def __init__(self) -> None:
        self._value: float = 0.0
        self._initialised = False

    def __call__(self, value: float, alpha: float) -> float:
        """Filter ``value`` with smoothing factor ``alpha`` in ``(0, 1]``."""
        if not self._initialised:
            self._value = value
            self._initialised = True
        else:
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        """Last filtered value."""
        return self._value

    def reset(self) -> None:
        """Forget history so the next sample is passed through unchanged."""
        self._initialised = False
        self._value = 0.0


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez, Roussel & Vogel, CHI 2012).

    A plain exponential filter forces a single trade-off: smooth but laggy, or
    responsive but jittery.  The 1€ filter resolves this by making the cutoff
    frequency a function of the signal's speed — it filters aggressively when
    the hand is nearly still (killing sensor jitter, which is what you notice
    when trying to hold the cursor on a button) and relaxes as the hand moves
    fast (preserving responsiveness during large sweeps).

    Args:
        freq: Nominal sampling frequency in Hz; only used before the first
            timestamped sample arrives.
        min_cutoff: Cutoff frequency at zero speed.  Lower = smoother.
        beta: Speed coefficient.  Higher = more responsive when moving fast.
        d_cutoff: Cutoff for the derivative estimate itself.
    """

    def __init__(self, freq: float = 60.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self.freq = max(freq, 1e-6)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPassFilter()
        self._dx = LowPassFilter()
        self._last_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """Convert a cutoff frequency into an exponential smoothing factor."""
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        te = 1.0 / max(freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, value: float, timestamp: float | None = None) -> float:
        """Filter one sample, optionally with an explicit ``timestamp``."""
        if timestamp is not None:
            if self._last_time is not None and timestamp > self._last_time:
                self.freq = 1.0 / (timestamp - self._last_time)
            self._last_time = timestamp

        prev = self._x.value if self._x._initialised else value
        # Derivative of the *raw* signal, itself smoothed.
        derivative = (value - prev) * self.freq
        edx = self._dx(derivative, self._alpha(self.d_cutoff, self.freq))

        # Speed-adaptive cutoff: the heart of the algorithm.
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(value, self._alpha(cutoff, self.freq))

    def reset(self) -> None:
        """Clear filter state (call when tracking is re-acquired)."""
        self._x.reset()
        self._dx.reset()
        self._last_time = None


class OneEuroFilter2D:
    """Convenience wrapper applying :class:`OneEuroFilter` to x and y."""

    def __init__(self, freq: float = 60.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self.x = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)
        self.y = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)

    def __call__(self, point: Sequence[float], timestamp: float | None = None) -> Vector2:
        """Filter a 2D point."""
        return (self.x(point[0], timestamp), self.y(point[1], timestamp))

    def tune(self, min_cutoff: float, beta: float) -> None:
        """Update tuning parameters live (settings slider support)."""
        for f in (self.x, self.y):
            f.min_cutoff = min_cutoff
            f.beta = beta

    def reset(self) -> None:
        """Clear both axes."""
        self.x.reset()
        self.y.reset()


class VelocityEstimator:
    """Estimates 2D velocity from timestamped positions via linear regression.

    A finite-difference estimate over two frames is far too noisy to drive
    motion prediction, so we least-squares fit position against time over a
    short window instead.
    """

    def __init__(self, window: int = 5) -> None:
        self._samples: Deque[Tuple[float, float, float]] = deque(maxlen=max(window, 2))

    def update(self, point: Sequence[float], timestamp: float) -> Vector2:
        """Add a sample and return the current velocity in units/second."""
        self._samples.append((timestamp, point[0], point[1]))
        if len(self._samples) < 2:
            return (0.0, 0.0)

        times = np.array([s[0] for s in self._samples], dtype=np.float64)
        times -= times[0]
        if float(times[-1]) < 1e-6:
            return (0.0, 0.0)

        xs = np.array([s[1] for s in self._samples], dtype=np.float64)
        ys = np.array([s[2] for s in self._samples], dtype=np.float64)
        vx = float(np.polyfit(times, xs, 1)[0])
        vy = float(np.polyfit(times, ys, 1)[0])
        return (vx, vy)

    def reset(self) -> None:
        """Discard all samples."""
        self._samples.clear()


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

class FPSMeter:
    """Rolling frame-rate meter with peak tracking."""

    def __init__(self, window: int = 60) -> None:
        self._times: Deque[float] = deque(maxlen=window)
        self._last = time.perf_counter()
        self.peak = 0.0

    def tick(self) -> float:
        """Record a frame boundary and return the smoothed FPS."""
        now = time.perf_counter()
        delta = now - self._last
        self._last = now
        if delta > 0:
            self._times.append(delta)
        fps = self.fps
        self.peak = max(self.peak, fps)
        return fps

    @property
    def fps(self) -> float:
        """Mean FPS over the current window."""
        if not self._times:
            return 0.0
        mean = sum(self._times) / len(self._times)
        return 1.0 / mean if mean > 0 else 0.0

    def reset(self) -> None:
        """Clear the window and the peak."""
        self._times.clear()
        self.peak = 0.0


class Cooldown:
    """Rate limiter — ``ready()`` is True at most once per ``interval``."""

    __slots__ = ("interval", "_last")

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._last = 0.0

    def ready(self, now: float | None = None) -> bool:
        """Return True and arm the cooldown, else False."""
        now = time.monotonic() if now is None else now
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False

    def remaining(self, now: float | None = None) -> float:
        """Seconds until the next :meth:`ready` can succeed."""
        now = time.monotonic() if now is None else now
        return max(0.0, self.interval - (now - self._last))

    def reset(self) -> None:
        """Make the next :meth:`ready` succeed immediately."""
        self._last = 0.0


class Stopwatch:
    """Context manager measuring wall-clock duration in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


@dataclass
class RingBuffer:
    """Fixed-capacity numeric buffer with cheap statistics.

    Backs every sparkline in the dashboard and every metric in the
    performance monitor.
    """

    capacity: int = 120
    _data: Deque[float] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self._data = deque(maxlen=self.capacity)

    def push(self, value: float) -> None:
        """Append a sample, evicting the oldest when full."""
        self._data.append(float(value))

    def extend(self, values: Iterable[float]) -> None:
        """Append many samples."""
        for v in values:
            self.push(v)

    @property
    def values(self) -> List[float]:
        """Samples as a list, oldest first."""
        return list(self._data)

    @property
    def mean(self) -> float:
        """Arithmetic mean, or 0.0 when empty."""
        return sum(self._data) / len(self._data) if self._data else 0.0

    @property
    def latest(self) -> float:
        """Most recent sample, or 0.0 when empty."""
        return self._data[-1] if self._data else 0.0

    @property
    def minimum(self) -> float:
        """Smallest sample, or 0.0 when empty."""
        return min(self._data) if self._data else 0.0

    @property
    def maximum(self) -> float:
        """Largest sample, or 0.0 when empty."""
        return max(self._data) if self._data else 0.0

    def percentile(self, pct: float) -> float:
        """Return the ``pct``-th percentile (0-100) of the buffer."""
        if not self._data:
            return 0.0
        return float(np.percentile(np.array(self._data, dtype=np.float64), pct))

    def clear(self) -> None:
        """Drop all samples."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def format_duration(seconds: float) -> str:
    """Render a duration as ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_bytes(num_bytes: float) -> str:
    """Render a byte count with a binary unit suffix."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def hex_to_rgb(colour: str) -> Tuple[int, int, int]:
    """Convert ``#rrggbb`` to an ``(r, g, b)`` tuple."""
    colour = colour.lstrip("#")
    if len(colour) == 3:
        colour = "".join(ch * 2 for ch in colour)
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Sequence[int]) -> str:
    """Convert an ``(r, g, b)`` tuple to ``#rrggbb``."""
    return "#{:02x}{:02x}{:02x}".format(*(int(clamp(c, 0, 255)) for c in rgb))


def mix_colours(a: str, b: str, t: float) -> str:
    """Blend two hex colours; ``t=0`` returns ``a``, ``t=1`` returns ``b``."""
    ra, ga, ba = hex_to_rgb(a)
    rb, gb, bb = hex_to_rgb(b)
    return rgb_to_hex((lerp(ra, rb, t), lerp(ga, gb, t), lerp(ba, bb, t)))
