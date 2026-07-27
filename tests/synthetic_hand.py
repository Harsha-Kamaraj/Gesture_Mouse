"""Kinematic hand model that synthesises MediaPipe-compatible landmarks.

The gesture engine's accuracy is the project's core claim, and it must be
verifiable in CI where there is no webcam, no operator and no hand.  This
module builds anatomically plausible 21-landmark hands from a small parameter
vector (per-finger curl + thumb abduction), so every pose in the library can
be generated, perturbed with noise, and asserted on.

Landmark indices and topology match MediaPipe Hands exactly, so the output
drops straight into :func:`gesture_engine.extract_features`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Segment lengths as a fraction of palm size, roughly anatomical.
_PHALANX: Dict[str, Tuple[float, float, float]] = {
    #        proximal, middle, distal
    "index":  (0.46, 0.28, 0.21),
    "middle": (0.50, 0.31, 0.22),
    "ring":   (0.45, 0.28, 0.21),
    "pinky":  (0.35, 0.21, 0.18),
}

# MCP knuckle positions relative to the wrist, in palm-size units.
_MCP_OFFSETS: Dict[str, Tuple[float, float]] = {
    "index":  (-0.26, -0.95),
    "middle": (-0.02, -1.00),
    "ring":   (0.20, -0.95),
    "pinky":  (0.40, -0.84),
}

_FINGER_ORDER = ("index", "middle", "ring", "pinky")

# Landmark slots for each finger: MCP, PIP, DIP, TIP.
_FINGER_SLOTS: Dict[str, Tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass
class HandPose:
    """Parameterised hand configuration.

    Attributes:
        thumb: Thumb abduction, 0 = tucked across the palm, 1 = fully out.
        index, middle, ring, pinky: Curl, 0 = fully closed, 1 = straight.
        rotation: Whole-hand roll in degrees (0 = fingers pointing up).
        scale: Palm size in normalized image units.
        centre: Wrist position in normalized image coordinates.
        handedness: ``"Left"`` or ``"Right"``.
    """

    thumb: float = 1.0
    index: float = 1.0
    middle: float = 1.0
    ring: float = 1.0
    pinky: float = 1.0
    rotation: float = 0.0
    scale: float = 0.16
    centre: Tuple[float, float] = (0.5, 0.75)
    handedness: str = "Right"

    @property
    def curls(self) -> Tuple[float, float, float, float]:
        """Curl values for the four non-thumb fingers."""
        return (self.index, self.middle, self.ring, self.pinky)


def _rotate(point: Tuple[float, float], degrees: float) -> Tuple[float, float]:
    """Rotate a point about the origin."""
    radians = math.radians(degrees)
    cos_r, sin_r = math.cos(radians), math.sin(radians)
    return (point[0] * cos_r - point[1] * sin_r,
            point[0] * sin_r + point[1] * cos_r)


def _build_finger(name: str, curl: float) -> List[Tuple[float, float]]:
    """Return the four joint positions of one finger in palm-local space.

    Curl is modelled the way a real finger flexes: each joint contributes a
    progressively larger bend, so a closed finger folds its tip back toward
    the palm rather than simply shrinking.
    """
    base = _MCP_OFFSETS[name]
    lengths = _PHALANX[name]

    # Joint flexion in degrees at MCP, PIP, DIP for a fully closed finger.
    max_flex = (85.0, 95.0, 60.0)
    flex = [(1.0 - curl) * angle for angle in max_flex]

    points = [base]
    direction = (0.0, -1.0)          # straight "up" in image space
    heading = 0.0
    for length, bend in zip(lengths, flex):
        heading += bend
        direction = _rotate((0.0, -1.0), heading)
        points.append((points[-1][0] + direction[0] * length,
                       points[-1][1] + direction[1] * length))
    return points


def _build_thumb(abduction: float, handedness: str) -> List[Tuple[float, float]]:
    """Return the four thumb joint positions in palm-local space.

    The thumb is modelled as abduction (swinging away from the palm) rather
    than flexion, which is what actually distinguishes a thumbs-up from a
    fist and is exactly what the engine's thumb rule measures.
    """
    side = -1.0 if handedness == "Right" else 1.0

    # Swing angle from tucked-across-palm to fully-out-to-the-side.
    tucked, extended = 12.0, 62.0
    angle = tucked + (extended - tucked) * abduction

    cmc = (side * 0.20, -0.18)
    lengths = (0.34, 0.28, 0.24)
    points = [cmc]
    for i, length in enumerate(lengths):
        # Successive segments open up slightly further.
        theta = math.radians(angle + i * 8.0 * abduction)
        direction = (side * math.sin(theta), -math.cos(theta))
        points.append((points[-1][0] + direction[0] * length,
                       points[-1][1] + direction[1] * length))
    return points


def build_landmarks(pose: HandPose, noise: float = 0.0,
                    rng: random.Random | None = None) -> np.ndarray:
    """Synthesise a ``(21, 3)`` landmark array for ``pose``.

    Args:
        pose: The hand configuration to render.
        noise: Standard deviation of Gaussian jitter, in palm-size units.
            ``0.02`` approximates a good MediaPipe detection; ``0.06`` is a
            poor one.
        rng: Optional seeded random source for reproducibility.

    Returns:
        Landmarks in normalized image coordinates, matching MediaPipe layout.
    """
    rng = rng or random.Random()
    points: List[Tuple[float, float]] = [(0.0, 0.0)] * 21

    points[0] = (0.0, 0.0)  # wrist at the local origin

    thumb = _build_thumb(pose.thumb, pose.handedness)
    for slot, position in zip((1, 2, 3, 4), thumb):
        points[slot] = position

    for name, curl in zip(_FINGER_ORDER, pose.curls):
        joints = _build_finger(name, curl)
        for slot, position in zip(_FINGER_SLOTS[name], joints):
            points[slot] = position

    out = np.zeros((21, 3), dtype=np.float32)
    for i, point in enumerate(points):
        x, y = point
        if noise > 0:
            x += rng.gauss(0.0, noise)
            y += rng.gauss(0.0, noise)
        x, y = _rotate((x, y), pose.rotation)
        out[i] = (
            pose.centre[0] + x * pose.scale,
            pose.centre[1] + y * pose.scale,
            0.0,
        )
    return out


#: Canonical parameter sets for every pose the engine claims to recognise.
POSE_PRESETS: Dict[str, HandPose] = {
    "Open Palm":      HandPose(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
    "Fist":           HandPose(thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0),
    "Point":          HandPose(thumb=0.15, index=1.0, middle=0.0, ring=0.0, pinky=0.0),
    "Peace":          HandPose(thumb=0.1, index=1.0, middle=1.0, ring=0.0, pinky=0.0),
    "Three Fingers":  HandPose(thumb=0.1, index=1.0, middle=1.0, ring=1.0, pinky=0.0),
    "Four Fingers":   HandPose(thumb=0.05, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
    "Thumb Up":       HandPose(thumb=1.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0),
    "Rock":           HandPose(thumb=0.1, index=1.0, middle=0.0, ring=0.0, pinky=1.0),
    "Call Me":        HandPose(thumb=1.0, index=0.0, middle=0.0, ring=0.0, pinky=1.0),
    "L Shape":        HandPose(thumb=1.0, index=1.0, middle=0.0, ring=0.0, pinky=0.0),
    "Pinky":          HandPose(thumb=0.1, index=0.0, middle=0.0, ring=0.0, pinky=1.0),
}


def make_pinch(base: HandPose, closed: bool = True) -> np.ndarray:
    """Build landmarks where the thumb tip touches (or clears) the index tip.

    The kinematic model cannot produce a true opposed pinch — real thumb
    opposition rotates out of the palm plane — so the thumb tip is placed
    directly for pinch tests.  Everything else stays model-generated.
    """
    points = build_landmarks(base)
    index_tip = points[8].copy()
    palm_size = float(np.linalg.norm(points[9][:2] - points[0][:2]))

    if closed:
        offset = 0.02 * palm_size
    else:
        offset = 0.55 * palm_size

    points[4] = (index_tip[0] + offset, index_tip[1] + offset, 0.0)
    # Drag the IP joint along so the thumb stays anatomically continuous.
    points[3] = (
        (points[2][0] + points[4][0]) / 2.0,
        (points[2][1] + points[4][1]) / 2.0,
        0.0,
    )
    return points


def touch_fingertip(points: np.ndarray, tip_index: int,
                    closed: bool = True) -> np.ndarray:
    """Move the thumb tip onto (or away from) an arbitrary fingertip."""
    result = points.copy()
    target = result[tip_index].copy()
    palm_size = float(np.linalg.norm(result[9][:2] - result[0][:2]))
    offset = 0.02 * palm_size if closed else 0.55 * palm_size
    result[4] = (target[0] + offset, target[1] + offset, 0.0)
    return result
