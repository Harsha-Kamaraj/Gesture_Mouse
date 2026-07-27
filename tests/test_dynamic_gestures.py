"""Tests for the $1 stroke recogniser and the swipe detector.

Strokes are synthesised with jitter, random scale, random translation and a
random number of samples so the tests exercise the same invariances the real
recogniser must provide when a human draws in the air.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_gestures import (  # noqa: E402
    DollarOneRecognizer, GestureTemplate, MotionTrail, detect_swipe,
    normalize_stroke,
)

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Stroke synthesis
# --------------------------------------------------------------------------- #

def jitter(points: List[Point], noise: float, scale: float,
           offset: Point, rng: random.Random) -> List[Point]:
    """Apply noise, scaling and translation to a clean stroke."""
    return [
        (p[0] * scale + offset[0] + rng.gauss(0, noise),
         p[1] * scale + offset[1] + rng.gauss(0, noise))
        for p in points
    ]


def draw_circle(n: int, rng: random.Random) -> List[Point]:
    """A hand-drawn-ish circle."""
    start = rng.uniform(0, 2 * math.pi)
    return [(math.cos(start + 2 * math.pi * i / n),
             math.sin(start + 2 * math.pi * i / n)) for i in range(n + 1)]


def draw_polyline(vertices: List[Point], per_edge: int) -> List[Point]:
    """Sample a polyline."""
    out: List[Point] = []
    for i in range(len(vertices) - 1):
        a, b = vertices[i], vertices[i + 1]
        for k in range(per_edge):
            t = k / per_edge
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    out.append(vertices[-1])
    return out


def draw_triangle(n: int) -> List[Point]:
    """A closed triangle."""
    corners = [(math.cos(-math.pi / 2 + 2 * math.pi * i / 3),
                math.sin(-math.pi / 2 + 2 * math.pi * i / 3)) for i in range(4)]
    return draw_polyline(corners, n)


def draw_square(n: int) -> List[Point]:
    """A closed square."""
    corners: List[Point] = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    return draw_polyline(corners, n)


def draw_z(n: int) -> List[Point]:
    """The letter Z."""
    return draw_polyline([(-1, 1), (1, 1), (-1, -1), (1, -1)], n)


def draw_wave(n: int) -> List[Point]:
    """A two-cycle horizontal wave."""
    return [(-1 + 2 * i / n, 0.45 * math.sin(4 * math.pi * i / n)) for i in range(n + 1)]


SHAPES = {
    "circle": lambda rng: draw_circle(rng.randint(24, 60), rng),
    "triangle": lambda rng: draw_triangle(rng.randint(8, 20)),
    "square": lambda rng: draw_square(rng.randint(8, 20)),
    "z": lambda rng: draw_z(rng.randint(10, 22)),
    "wave": lambda rng: draw_wave(rng.randint(40, 80)),
}


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_normalisation_is_invariant() -> None:
    """Scale and translation must not change the normalised vector."""
    rng = random.Random(7)
    base = draw_circle(40, rng)
    moved = [(p[0] * 3.7 + 12.0, p[1] * 3.7 - 5.0) for p in base]

    _, v1 = normalize_stroke(base)
    _, v2 = normalize_stroke(moved)
    drift = max(abs(a - b) for a, b in zip(v1, v2))
    assert drift < 1e-6, f"normalisation not invariant (drift={drift})"


def test_resample_produces_fixed_length() -> None:
    """Every stroke normalises to the same vector length regardless of input."""
    rng = random.Random(3)
    lengths = set()
    for n in (9, 17, 40, 133, 400):
        _, vec = normalize_stroke(draw_circle(n, rng))
        lengths.add(vec.size)
    assert lengths == {128}, f"inconsistent vector sizes: {lengths}"


def test_recognises_noisy_shapes() -> None:
    """Each built-in shape must be identified across many noisy renditions."""
    rec = DollarOneRecognizer()
    rng = random.Random(42)
    trials = 40
    results = {}

    for name, factory in SHAPES.items():
        hits = 0
        for _ in range(trials):
            stroke = jitter(
                factory(rng),
                noise=rng.uniform(0.01, 0.06),
                scale=rng.uniform(0.4, 3.0),
                offset=(rng.uniform(-4, 4), rng.uniform(-4, 4)),
                rng=rng,
            )
            match = rec.recognize(stroke)
            if match and match.name.startswith(name):
                hits += 1
        accuracy = hits / trials
        results[name] = accuracy

    for name, accuracy in results.items():
        assert accuracy >= 0.80, f"{name} accuracy {accuracy:.0%} below 80% ({results})"


def test_custom_template_one_shot() -> None:
    """A single recorded exemplar must be enough to recognise a new gesture."""
    rec = DollarOneRecognizer()
    rng = random.Random(11)
    exemplar = draw_polyline([(-1, -1), (0, 1), (1, -1), (0, 0)], 12)
    rec.add(GestureTemplate("lightning", exemplar))

    hits = 0
    for _ in range(20):
        stroke = jitter(exemplar, 0.03, rng.uniform(0.5, 2.5),
                        (rng.uniform(-2, 2), rng.uniform(-2, 2)), rng)
        match = rec.recognize(stroke)
        if match and match.name == "lightning":
            hits += 1
    assert hits >= 16, f"one-shot learning only {hits}/20"


def test_disabled_templates_are_skipped() -> None:
    """A disabled template must never be returned."""
    rec = DollarOneRecognizer()
    rec.set_enabled("circle", False)
    rec.set_enabled("circle_cw", False)
    rng = random.Random(5)
    match = rec.recognize(draw_circle(40, rng))
    assert match is None or not match.name.startswith("circle")


def test_library_management() -> None:
    """Add / rename / remove must all behave."""
    rec = DollarOneRecognizer()
    original = len(rec.templates)
    rec.add(GestureTemplate("temp", draw_square(10)))
    assert len(rec.templates) == original + 1
    assert rec.rename("temp", "renamed") == 1
    assert "renamed" in rec.names
    assert rec.remove("renamed") == 1
    assert len(rec.templates) == original


def test_short_stroke_rejected() -> None:
    """Strokes with too few samples must not produce a match."""
    rec = DollarOneRecognizer()
    assert rec.recognize([(0, 0), (1, 1), (2, 2)]) is None


def test_swipe_directions() -> None:
    """All four swipe directions must be detected with correct labels."""
    cases = {
        "right": [(0.1 + 0.02 * i, 0.5) for i in range(20)],
        "left": [(0.9 - 0.02 * i, 0.5) for i in range(20)],
        "down": [(0.5, 0.1 + 0.02 * i) for i in range(20)],
        "up": [(0.5, 0.9 - 0.02 * i) for i in range(20)],
    }
    for expected, points in cases.items():
        result = detect_swipe(points, duration=0.35)
        assert result is not None, f"{expected} swipe not detected"
        assert result.direction == expected, f"got {result.direction}, want {expected}"
        assert result.confidence > 0.5


def test_circle_is_not_a_swipe() -> None:
    """A circular path displaces little relative to its length."""
    rng = random.Random(1)
    circle = [(0.5 + 0.2 * math.cos(t / 10), 0.5 + 0.2 * math.sin(t / 10))
              for t in range(63)]
    assert detect_swipe(circle, duration=0.6) is None


def test_slow_drift_is_not_a_swipe() -> None:
    """Movement below the speed floor must be ignored."""
    points = [(0.1 + 0.02 * i, 0.5) for i in range(20)]
    assert detect_swipe(points, duration=5.0) is None


def test_motion_trail() -> None:
    """The trail must bound its capacity and report travel correctly."""
    trail = MotionTrail(capacity=10)
    for i in range(25):
        trail.add((i * 0.01, 0.0), timestamp=i * 0.03)
    assert len(trail) == 10
    assert trail.length > 0
    assert trail.duration > 0

    stationary = MotionTrail(capacity=10)
    for i in range(8):
        stationary.add((0.5, 0.5), timestamp=i * 0.03)
    assert stationary.is_stationary


def _run_all() -> int:
    """Minimal runner so the file works without pytest installed."""
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
