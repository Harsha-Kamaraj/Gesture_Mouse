"""Tests for feature extraction, pose classification and the engine's
state machines.

All hands are synthesised (see :mod:`synthetic_hand`), so the whole
recognition path is exercised without a camera or a desktop session.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from config import GestureConfig  # noqa: E402
from detector import HandLandmarks  # noqa: E402
from gesture_engine import (  # noqa: E402
    GestureEngine, Mode, Pose, StaticClassifier, TemporalStabilizer,
    extract_features,
)
from synthetic_hand import (  # noqa: E402
    POSE_PRESETS, HandPose, build_landmarks, make_pinch, touch_fingertip,
)

PINCH_THRESHOLD = GestureConfig().pinch_threshold


def hand_from(points: np.ndarray, side: str = "Right",
              score: float = 0.95) -> HandLandmarks:
    """Wrap raw landmarks in a :class:`HandLandmarks`."""
    return HandLandmarks(points, points.copy(), side, score)


def features_for(pose_name: str, **kwargs: object):
    """Extract features for a named preset pose."""
    preset = POSE_PRESETS[pose_name]
    pose = HandPose(**{**preset.__dict__, **kwargs})  # type: ignore[arg-type]
    return extract_features(hand_from(build_landmarks(pose)))


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def test_finger_extension_separates_open_from_closed() -> None:
    """Open and closed hands must sit at opposite ends of the score range."""
    open_features = features_for("Open Palm")
    fist_features = features_for("Fist")

    assert all(e > 0.85 for e in open_features.extensions[1:]), \
        f"open hand fingers not extended: {open_features.extensions}"
    assert all(e < 0.15 for e in fist_features.extensions[1:]), \
        f"fist fingers not curled: {fist_features.extensions}"
    assert open_features.extensions[0] > 0.85, "open hand thumb not extended"
    assert fist_features.extensions[0] < 0.35, "fist thumb not curled"


def test_palm_size_normalisation_is_scale_invariant() -> None:
    """Doubling hand size must not change any normalised measurement."""
    small = extract_features(hand_from(build_landmarks(
        HandPose(scale=0.10, centre=(0.3, 0.6)))))
    large = extract_features(hand_from(build_landmarks(
        HandPose(scale=0.28, centre=(0.7, 0.4)))))

    for a, b in zip(small.extensions, large.extensions):
        assert abs(a - b) < 0.02, f"extension drift {a:.3f} vs {b:.3f}"
    for a, b in zip(small.pinch_distances, large.pinch_distances):
        assert abs(a - b) < 0.02, f"pinch drift {a:.3f} vs {b:.3f}"


def test_rotation_invariance() -> None:
    """Pose classification must survive a rolled hand."""
    classifier = StaticClassifier()
    for rotation in (-40, -20, 0, 20, 40):
        features = features_for("Peace", rotation=rotation)
        match = classifier.classify(features, PINCH_THRESHOLD)
        assert match.pose == Pose.PEACE, \
            f"rotation {rotation}° misread as {match.pose.value}"


def test_pinch_distance_responds_to_contact() -> None:
    """A closed pinch must fall below threshold; an open one above."""
    closed = extract_features(hand_from(make_pinch(POSE_PRESETS["Point"], True)))
    opened = extract_features(hand_from(make_pinch(POSE_PRESETS["Point"], False)))
    assert closed.pinch_index < PINCH_THRESHOLD, f"closed pinch {closed.pinch_index:.3f}"
    assert opened.pinch_index > PINCH_THRESHOLD * 3, f"open pinch {opened.pinch_index:.3f}"


# --------------------------------------------------------------------------- #
# Pose classification
# --------------------------------------------------------------------------- #

def test_all_presets_classify_correctly() -> None:
    """Every pose in the library must be recognised from its canonical form."""
    classifier = StaticClassifier()
    wrong: List[str] = []
    for name, pose in POSE_PRESETS.items():
        match = classifier.classify(
            extract_features(hand_from(build_landmarks(pose))), PINCH_THRESHOLD)
        if match.pose.value != name:
            wrong.append(f"{name} -> {match.pose.value} ({match.confidence:.2f})")
    assert not wrong, "misclassified: " + ", ".join(wrong)


def test_poses_survive_landmark_noise() -> None:
    """Classification must stay accurate with realistic detector jitter."""
    classifier = StaticClassifier()
    rng = random.Random(2024)
    trials = 30
    failures = {}

    for name, pose in POSE_PRESETS.items():
        hits = 0
        for _ in range(trials):
            points = build_landmarks(pose, noise=0.035, rng=rng)
            match = classifier.classify(
                extract_features(hand_from(points)), PINCH_THRESHOLD)
            if match.pose.value == name:
                hits += 1
        accuracy = hits / trials
        if accuracy < 0.80:
            failures[name] = accuracy

    assert not failures, f"poses below 80% under noise: {failures}"


def test_ambiguous_hand_scores_low() -> None:
    """A half-curled hand must not produce a confident classification."""
    classifier = StaticClassifier()
    features = extract_features(hand_from(build_landmarks(
        HandPose(thumb=0.5, index=0.5, middle=0.5, ring=0.5, pinky=0.5))))
    match = classifier.classify(features, PINCH_THRESHOLD)
    assert match.confidence < 0.72, \
        f"ambiguous pose was confident: {match.pose.value} {match.confidence:.2f}"


def test_l_shape_distinct_from_point() -> None:
    """The thumb must be enough to separate these two otherwise-identical poses."""
    classifier = StaticClassifier()
    point = classifier.classify(features_for("Point"), PINCH_THRESHOLD)
    l_shape = classifier.classify(features_for("L Shape"), PINCH_THRESHOLD)
    assert point.pose == Pose.POINT
    assert l_shape.pose == Pose.GUN
    assert l_shape.margin > 0.15, f"L-shape margin too thin: {l_shape.margin:.2f}"


# --------------------------------------------------------------------------- #
# Temporal stabilisation
# --------------------------------------------------------------------------- #

def test_stabiliser_requires_consecutive_frames() -> None:
    """A pose must persist before it is accepted."""
    stabiliser = TemporalStabilizer(required_frames=3)
    assert stabiliser.update(Pose.POINT) == Pose.UNKNOWN
    assert stabiliser.update(Pose.POINT) == Pose.UNKNOWN
    assert stabiliser.update(Pose.POINT) == Pose.POINT


def test_stabiliser_rejects_single_frame_glitch() -> None:
    """One bad frame must not change the accepted pose."""
    stabiliser = TemporalStabilizer(required_frames=3)
    for _ in range(4):
        stabiliser.update(Pose.PEACE)
    assert stabiliser.stable_pose == Pose.PEACE

    stabiliser.update(Pose.FIST)       # glitch frame
    assert stabiliser.stable_pose == Pose.PEACE, "glitch changed the stable pose"
    stabiliser.update(Pose.PEACE)
    assert stabiliser.stable_pose == Pose.PEACE


# --------------------------------------------------------------------------- #
# Engine state machines
# --------------------------------------------------------------------------- #

def make_engine(**overrides: object) -> GestureEngine:
    """Build an engine with a permissive config for deterministic testing."""
    cfg = GestureConfig(**overrides)  # type: ignore[arg-type]
    cfg.stability_frames = 1
    cfg.global_cooldown = 0.0
    return GestureEngine(cfg)


def feed(engine: GestureEngine, points: np.ndarray, t: float,
         frames: int = 1, side: str = "Right") -> List:
    """Push ``frames`` copies of a hand through the engine, collecting events."""
    events: List = []
    for i in range(frames):
        out = engine.update([hand_from(points, side)], t + i * 0.033)
        events.extend(out.events)
    return events


def test_short_pinch_emits_left_click() -> None:
    """Pinch and release inside the drag threshold must be a click."""
    engine = make_engine()
    open_hand = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)

    feed(engine, open_hand, 0.0, frames=3)
    feed(engine, closed, 0.10, frames=2)
    events = feed(engine, open_hand, 0.25, frames=2)

    names = [e.name for e in events]
    assert "pinch_tap" in names, f"no click emitted, got {names}"
    assert engine.stats["clicks"] == 1


def test_long_pinch_becomes_drag_then_drop() -> None:
    """Holding a pinch past the threshold must drag, and releasing must drop."""
    engine = make_engine()
    open_hand = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)

    feed(engine, open_hand, 0.0, frames=3)
    feed(engine, closed, 0.10)
    drag_events = feed(engine, closed, 0.10 + engine.cfg.drag_hold_time + 0.05)

    assert any(e.name == "drag_start" for e in drag_events), "drag never started"
    assert engine.mode == Mode.DRAG

    drop_events = feed(engine, open_hand, 1.5)
    assert any(e.name == "drag_end" for e in drop_events), "drag never ended"
    assert engine.mode != Mode.DRAG


def test_double_click_detected() -> None:
    """Two pinches inside the interval must produce a double click."""
    engine = make_engine()
    open_hand = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)

    feed(engine, open_hand, 0.0, frames=3)
    feed(engine, closed, 0.10)
    feed(engine, open_hand, 0.18)
    feed(engine, closed, 0.26)
    events = feed(engine, open_hand, 0.34)

    assert any(e.name == "pinch_double" for e in events), \
        f"no double click: {[e.name for e in events]}"


def test_pinch_hysteresis_prevents_chatter() -> None:
    """Hovering between the two thresholds must not emit repeated clicks."""
    engine = make_engine()
    points = make_pinch(POSE_PRESETS["Point"], closed=True)
    # Park the thumb between close and release thresholds.
    palm = float(np.linalg.norm(points[9][:2] - points[0][:2]))
    between = (engine.cfg.pinch_threshold + engine.cfg.pinch_release_threshold) / 2
    points[4] = (points[8][0] + between * palm, points[8][1], 0.0)

    feed(engine, points, 0.0, frames=3)
    events = feed(engine, points, 0.2, frames=30)
    clicks = [e for e in events if e.name in ("pinch_tap", "pinch_double")]
    assert not clicks, f"hysteresis band produced {len(clicks)} phantom clicks"


def test_tracking_loss_releases_drag() -> None:
    """Losing the hand mid-drag must release the button, not leave it down."""
    engine = make_engine()
    open_hand = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)

    feed(engine, open_hand, 0.0, frames=3)
    feed(engine, closed, 0.10)
    feed(engine, closed, 0.10 + engine.cfg.drag_hold_time + 0.05)
    assert engine.mode == Mode.DRAG

    # Hand vanishes for longer than the tracking timeout.
    engine.update([], 1.0)
    out = engine.update([], 1.0 + engine.cfg.tracking_lost_timeout + 0.1)

    assert any(e.name == "drag_end" for e in out.events), \
        "drag was not released on tracking loss"
    assert engine.mode != Mode.DRAG


def test_mode_switching_via_pose() -> None:
    """Scroll / volume / brightness modes must follow their poses."""
    engine = make_engine()
    expectations = {
        "Peace": Mode.SCROLL,
        "Rock": Mode.VOLUME,
        "Three Fingers": Mode.BRIGHTNESS,
        "Point": Mode.NAVIGATE,
    }
    t = 0.0
    for name, expected in expectations.items():
        points = build_landmarks(POSE_PRESETS[name])
        for i in range(4):
            engine.update([hand_from(points)], t + i * 0.033)
        t += 0.5
        assert engine.mode == expected, \
            f"{name} gave {engine.mode.value}, expected {expected.value}"


def test_scroll_emits_directional_events() -> None:
    """Vertical hand travel in scroll mode must emit scroll events."""
    engine = make_engine()
    preset = POSE_PRESETS["Peace"]

    engine.update([hand_from(build_landmarks(preset))], 0.0)
    events: List = []
    for i in range(1, 8):
        pose = HandPose(**{**preset.__dict__, "centre": (0.5, 0.75 - i * 0.03)})
        out = engine.update([hand_from(build_landmarks(pose))], i * 0.033)
        events.extend(out.events)

    scrolls = [e for e in events if e.action == "scroll"]
    assert scrolls, "no scroll events emitted"
    assert all(e.data["direction"] == "up" for e in scrolls), \
        f"wrong scroll direction: {[e.data['direction'] for e in scrolls]}"


def test_right_click_via_middle_pinch() -> None:
    """Thumb touching the middle fingertip must emit a right click."""
    engine = make_engine()
    base = build_landmarks(POSE_PRESETS["Open Palm"])
    apart = touch_fingertip(base, 12, closed=False)
    together = touch_fingertip(base, 12, closed=True)

    feed(engine, apart, 0.0, frames=3)
    events = feed(engine, together, 0.2, frames=2)
    assert any(e.name == "pinch_middle" for e in events), \
        f"no right click: {[e.name for e in events]}"


def test_low_confidence_events_are_suppressed() -> None:
    """Events below the confidence floor must never fire."""
    engine = make_engine()
    engine.cfg.min_confidence = 0.99

    open_hand = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)
    feed(engine, open_hand, 0.0, frames=3)
    feed(engine, closed, 0.10)
    events = feed(engine, open_hand, 0.25, frames=2)

    assert not [e for e in events if e.name == "pinch_tap"], \
        "click fired despite failing the confidence gate"


def test_disabled_gesture_does_not_fire() -> None:
    """A gesture disabled in the profile must be inert."""
    engine = make_engine()
    engine.cfg.overrides["pinch_tap"] = {"enabled": False}

    open_hand = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)
    feed(engine, open_hand, 0.0, frames=3)
    feed(engine, closed, 0.10)
    events = feed(engine, open_hand, 0.25, frames=2)
    assert not [e for e in events if e.name == "pinch_tap"]


def test_paused_engine_emits_nothing() -> None:
    """Pausing must silence the engine while it keeps processing frames."""
    engine = make_engine()
    engine.pause()

    open_hand = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)
    feed(engine, open_hand, 0.0, frames=3)
    feed(engine, closed, 0.10)
    events = feed(engine, open_hand, 0.25, frames=2)
    assert not events, f"paused engine emitted {[e.name for e in events]}"


def test_sleep_and_wake_cycle() -> None:
    """Holding an open palm sleeps; a fresh palm hold wakes.

    Waking deliberately requires the pose to change away from the open palm
    first.  Without that, the same continuous hold that put the engine to
    sleep would immediately wake it again one hold-duration later.
    """
    engine = make_engine()
    palm = build_landmarks(POSE_PRESETS["Open Palm"])
    fist = build_landmarks(POSE_PRESETS["Fist"])

    engine.update([hand_from(palm)], 0.0)
    out = engine.update([hand_from(palm)], engine.HOLD_DURATION + 0.1)
    assert any(e.name == "open_palm_hold" for e in out.events), "sleep never fired"
    assert engine.sleeping

    # Holding the same palm must NOT wake it back up.
    out = engine.update([hand_from(palm)], 3.0)
    assert not any(e.action == "wake_tracking" for e in out.events), \
        "continuous hold woke the engine immediately"
    assert engine.sleeping

    # Change pose, then present a fresh palm hold.
    engine.update([hand_from(fist)], 4.0)
    engine.update([hand_from(palm)], 5.0)
    out = engine.update([hand_from(palm)], 5.0 + engine.HOLD_DURATION + 0.1)
    assert any(e.action == "wake_tracking" for e in out.events), "wake never fired"
    assert not engine.sleeping


def test_cursor_source_follows_mode() -> None:
    """The cursor must track the fingertip, and stop moving in scroll mode."""
    engine = make_engine()
    point = build_landmarks(POSE_PRESETS["Point"])
    out = engine.update([hand_from(point)], 0.0)
    assert out.cursor_point is not None

    peace = build_landmarks(POSE_PRESETS["Peace"])
    for i in range(4):
        out = engine.update([hand_from(peace)], 1.0 + i * 0.033)
    assert out.cursor_point is None, "cursor moved while scrolling"


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
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
