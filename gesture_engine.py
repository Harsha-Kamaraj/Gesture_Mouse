"""Gesture recognition engine.

Pipeline
--------
``HandLandmarks`` → :class:`HandFeatures` → :class:`StaticClassifier`
→ :class:`TemporalStabilizer` → mode state machine → :class:`GestureEvent`

Three ideas carry most of the accuracy:

**Scale invariance.**  Every distance is divided by the palm size
(wrist→middle-MCP), so a threshold tuned for one user works for a child's
hand, an adult's hand, and for someone sitting twice as far from the camera.

**Continuous scores, not booleans.**  A finger is not "extended" or "curled";
it has an extension score in ``[0, 1]``.  Gesture confidence then falls out of
the pose match naturally instead of being invented after the fact, and near-
miss poses score low rather than flapping between two hard classifications.

**Temporal stabilisation.**  A pose must hold for N consecutive frames before
it can fire.  This is the single most effective defence against accidental
clicks: MediaPipe occasionally emits one bad frame during fast motion, and
without stabilisation that frame becomes a click in the user's document.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import GestureConfig
from detector import (
    FINGER_NAMES, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP,
    MIDDLE_PIP, MIDDLE_TIP, PINKY_MCP, PINKY_PIP, PINKY_TIP, RING_MCP,
    RING_PIP, RING_TIP, THUMB_IP, THUMB_MCP, THUMB_TIP, WRIST,
    HandLandmarks,
)
from dynamic_gestures import DollarOneRecognizer, MotionTrail, detect_swipe
from logger import get_logger
from utils import (
    Cooldown, angle_between, clamp, distance_2d, smoothstep,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #

class Mode(str, Enum):
    """Top-level engine state.

    Modes make otherwise-conflicting gestures coexist: a pinch means "click"
    in :attr:`NAVIGATE` but "set the volume level" in :attr:`VOLUME`.  Without
    modes, every continuous control would have to steal a distinct pose, and
    we would run out of distinguishable hand shapes very quickly.
    """

    SLEEPING = "Sleeping"
    NAVIGATE = "Navigate"
    SCROLL = "Scroll"
    VOLUME = "Volume"
    BRIGHTNESS = "Brightness"
    DRAG = "Drag"
    DRAW = "Draw"
    PRESENT = "Present"
    ZOOM = "Zoom"


class Pose(str, Enum):
    """Named static hand shapes."""

    UNKNOWN = "Unknown"
    OPEN_PALM = "Open Palm"
    FIST = "Fist"
    POINT = "Point"
    PEACE = "Peace"
    THREE = "Three Fingers"
    FOUR = "Four Fingers"
    THUMB_UP = "Thumb Up"
    THUMB_DOWN = "Thumb Down"
    ROCK = "Rock"
    CALL = "Call Me"
    GUN = "L Shape"
    PINKY = "Pinky"
    OK_SIGN = "OK Sign"
    PINCH_INDEX = "Pinch"
    PINCH_MIDDLE = "Middle Pinch"
    PINCH_RING = "Ring Pinch"


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

#: PIP-joint angle (degrees) below which a finger reads as fully curled.
_CURLED_ANGLE = 95.0
#: PIP-joint angle at or above which a finger reads as fully extended.
_EXTENDED_ANGLE = 158.0

#: Palm-normalised thumb spread bounds (tip → index MCP).  A tucked thumb
#: rests near the index knuckle (~0.15-0.45); an abducted one swings out to
#: roughly 0.8-1.0 palm units.
_THUMB_CURLED_SPREAD = 0.45
_THUMB_EXTENDED_SPREAD = 0.82

_FINGER_JOINTS: Tuple[Tuple[int, int, int], ...] = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)


@dataclass
class HandFeatures:
    """Scale-invariant measurements derived from one hand's landmarks.

    Everything a gesture rule could need is computed once per frame here, so
    classification itself is pure comparison with no geometry in it.
    """

    #: Per-finger extension in ``[0, 1]``, ordered thumb→pinky.
    extensions: Tuple[float, float, float, float, float]
    #: Palm-normalised thumb-tip distance to each fingertip, index→pinky.
    pinch_distances: Tuple[float, float, float, float]
    #: Normalized ``(x, y)`` of the index fingertip — the cursor source.
    index_tip: Tuple[float, float]
    #: Normalized ``(x, y)`` of the palm centre — a steadier anchor.
    palm_centre: Tuple[float, float]
    #: Hand roll in degrees; 0° = fingers pointing up.
    orientation: float
    #: True when the palm faces the camera (used to reject back-of-hand poses).
    palm_facing: bool
    #: Wrist→middle-MCP distance in normalized units; proxy for camera distance.
    palm_size: float
    #: How spread the four fingers are, ``[0, 1]``.
    spread: float
    handedness: str
    detector_score: float

    @property
    def fingers_up(self) -> Tuple[bool, bool, bool, bool, bool]:
        """Boolean view of :attr:`extensions`, thresholded at 0.5."""
        return tuple(e >= 0.5 for e in self.extensions)  # type: ignore[return-value]

    @property
    def extended_count(self) -> int:
        """Number of fingers currently reading as extended."""
        return sum(self.fingers_up)

    @property
    def pinch_index(self) -> float:
        """Palm-normalised thumb↔index-tip distance."""
        return self.pinch_distances[0]

    def describe(self) -> str:
        """Human-readable finger summary for the debug overlay."""
        return " ".join(
            f"{n[0].upper()}{'+' if e >= 0.5 else '-'}"
            for n, e in zip(FINGER_NAMES, self.extensions)
        )


def _finger_extension(hand: HandLandmarks, mcp: int, pip: int, tip: int) -> float:
    """Continuous extension score for one non-thumb finger.

    Combines the PIP joint angle (primary — reliable at any hand orientation)
    with a tip-versus-PIP reach test (secondary — disambiguates the case where
    a finger is straight but folded down at the knuckle).
    """
    angle = angle_between(hand.points[mcp], hand.points[pip], hand.points[tip])
    angular = smoothstep(_CURLED_ANGLE, _EXTENDED_ANGLE, angle)

    # A straight-but-folded finger has a large PIP angle yet its tip sits no
    # further from the wrist than its own PIP joint.
    scale = hand.palm_size
    tip_reach = distance_2d(hand.points[WRIST], hand.points[tip]) / scale
    pip_reach = distance_2d(hand.points[WRIST], hand.points[pip]) / scale
    reach = smoothstep(0.98, 1.22, tip_reach / max(pip_reach, 1e-6))

    # Weighted toward the angle; reach only vetoes clear folds.
    return clamp(0.72 * angular + 0.28 * reach, 0.0, 1.0)


def _thumb_extension(hand: HandLandmarks) -> float:
    """Continuous extension score for the thumb.

    The thumb needs its own rule: it has one fewer phalanx and its motion is
    mostly abduction (sideways) rather than flexion, so a PIP-style angle test
    barely moves between a tucked and an extended thumb.  Lateral spread from
    the index knuckle is the discriminating measurement.
    """
    scale = hand.palm_size
    spread = distance_2d(hand.points[THUMB_TIP], hand.points[INDEX_MCP]) / scale
    spread_score = smoothstep(_THUMB_CURLED_SPREAD, _THUMB_EXTENDED_SPREAD, spread)

    straightness = angle_between(
        hand.points[THUMB_MCP], hand.points[THUMB_IP], hand.points[THUMB_TIP]
    )
    straight_score = smoothstep(120.0, 165.0, straightness)

    # Spread dominates: the thumb stays fairly straight even when tucked
    # across the palm, so straightness alone barely discriminates.
    return clamp(0.8 * spread_score + 0.2 * straight_score, 0.0, 1.0)


def extract_features(hand: HandLandmarks) -> HandFeatures:
    """Compute the full :class:`HandFeatures` set for one hand."""
    extensions = (
        _thumb_extension(hand),
        *(_finger_extension(hand, *joints) for joints in _FINGER_JOINTS),
    )

    scale = hand.palm_size
    pinches = tuple(
        distance_2d(hand.points[THUMB_TIP], hand.points[tip]) / scale
        for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    )

    # Orientation: wrist→middle-MCP bearing, 0° when fingers point up.
    wrist = hand.points[WRIST]
    middle = hand.points[MIDDLE_MCP]
    orientation = float(np.degrees(np.arctan2(middle[0] - wrist[0],
                                              wrist[1] - middle[1])))

    # Palm facing: the sign of the cross product of two palm edge vectors
    # flips when the hand turns over.  Handedness inverts the expected sign.
    v1 = hand.points[INDEX_MCP][:2] - wrist[:2]
    v2 = hand.points[PINKY_MCP][:2] - wrist[:2]
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    palm_facing = cross < 0 if hand.handedness == "Right" else cross > 0

    spread_raw = distance_2d(hand.points[INDEX_TIP], hand.points[PINKY_TIP]) / scale
    spread = clamp(smoothstep(0.5, 1.8, spread_raw), 0.0, 1.0)

    return HandFeatures(
        extensions=extensions,  # type: ignore[arg-type]
        pinch_distances=pinches,  # type: ignore[arg-type]
        index_tip=hand.xy(INDEX_TIP),
        palm_centre=hand.palm_centre,
        orientation=orientation,
        palm_facing=palm_facing,
        palm_size=scale,
        spread=spread,
        handedness=hand.handedness,
        detector_score=hand.score,
    )


# --------------------------------------------------------------------------- #
# Static pose classification
# --------------------------------------------------------------------------- #

@dataclass
class PoseDefinition:
    """Declarative description of a static hand pose.

    ``pattern`` holds five entries (thumb→pinky): ``1`` requires extended,
    ``0`` requires curled, ``None`` means "don't care".  Keeping poses as data
    rather than code means the gesture library screen can list, edit and
    extend them without touching the classifier.
    """

    pose: Pose
    pattern: Tuple[Optional[int], ...]
    #: Extra confidence multiplier, lets specific poses outrank generic ones.
    priority: float = 1.0
    #: Optional requirement that the thumb+index are touching.
    requires_pinch: bool = False
    #: Optional requirement that the thumb+index are apart.
    requires_no_pinch: bool = False
    description: str = ""

    def match(self, features: HandFeatures, pinch_threshold: float) -> float:
        """Score how well ``features`` fit this pose, in ``[0, 1]``."""
        scores: List[float] = []
        for expected, actual in zip(self.pattern, features.extensions):
            if expected is None:
                continue
            scores.append(actual if expected == 1 else 1.0 - actual)

        if not scores:
            return 0.0

        # The weakest finger dominates: one clearly wrong finger should sink
        # the whole match rather than being averaged away by four right ones.
        mean_score = sum(scores) / len(scores)
        weakest = min(scores)
        score = 0.55 * mean_score + 0.45 * weakest

        pinching = features.pinch_index <= pinch_threshold
        if self.requires_pinch and not pinching:
            return 0.0
        if self.requires_no_pinch and pinching:
            score *= 0.35

        return clamp(score * self.priority, 0.0, 1.0)


#: The built-in pose library.  Order is irrelevant; scoring decides the winner.
POSE_LIBRARY: Tuple[PoseDefinition, ...] = (
    PoseDefinition(Pose.OPEN_PALM, (1, 1, 1, 1, 1),
                   description="All five fingers extended — neutral / wake."),
    PoseDefinition(Pose.FIST, (0, 0, 0, 0, 0),
                   description="Closed hand — grab and hold."),
    # The thumb is *not* a wildcard here: with it unconstrained, POINT also
    # matches the L-shape perfectly and swallows it, since the two poses
    # differ by nothing else.
    PoseDefinition(Pose.POINT, (0, 1, 0, 0, 0), priority=1.05,
                   description="Index only — moves the cursor."),
    PoseDefinition(Pose.PEACE, (0, 1, 1, 0, 0), priority=1.05,
                   description="Index + middle — scroll mode."),
    PoseDefinition(Pose.THREE, (0, 1, 1, 1, 0),
                   description="Index + middle + ring — brightness mode."),
    PoseDefinition(Pose.FOUR, (0, 1, 1, 1, 1),
                   description="Four fingers, thumb tucked — presentation mode."),
    PoseDefinition(Pose.THUMB_UP, (1, 0, 0, 0, 0),
                   description="Thumb only — context dependent."),
    PoseDefinition(Pose.ROCK, (0, 1, 0, 0, 1), priority=1.1,
                   description="Index + pinky — volume mode."),
    PoseDefinition(Pose.CALL, (1, 0, 0, 0, 1), priority=1.1,
                   description="Thumb + pinky — screen recording."),
    PoseDefinition(Pose.GUN, (1, 1, 0, 0, 0), priority=1.05,
                   requires_no_pinch=True,
                   description="Thumb + index at 90° — precision mode."),
    PoseDefinition(Pose.PINKY, (0, 0, 0, 0, 1),
                   description="Pinky only — whiteboard toggle."),
    PoseDefinition(Pose.OK_SIGN, (None, None, 1, 1, 1), priority=1.15,
                   requires_pinch=True,
                   description="Thumb+index touching, others up — screenshot."),
)


@dataclass
class PoseMatch:
    """Winning pose plus the runner-up, for debugging and hysteresis."""

    pose: Pose
    confidence: float
    runner_up: Pose = Pose.UNKNOWN
    runner_up_confidence: float = 0.0

    @property
    def margin(self) -> float:
        """Gap to the runner-up — low margin means an ambiguous hand shape."""
        return self.confidence - self.runner_up_confidence


class StaticClassifier:
    """Scores a feature set against the pose library."""

    def __init__(self, library: Sequence[PoseDefinition] = POSE_LIBRARY) -> None:
        self.library = list(library)

    def classify(self, features: HandFeatures, pinch_threshold: float) -> PoseMatch:
        """Return the best-matching pose for ``features``."""
        scored = sorted(
            ((definition.pose, definition.match(features, pinch_threshold))
             for definition in self.library),
            key=lambda item: item[1],
            reverse=True,
        )
        if not scored or scored[0][1] <= 0.0:
            return PoseMatch(Pose.UNKNOWN, 0.0)

        best_pose, best_score = scored[0]
        runner_pose, runner_score = scored[1] if len(scored) > 1 else (Pose.UNKNOWN, 0.0)

        # Fold in the detector's own confidence — a shaky detection should not
        # produce a certain-looking gesture.
        confidence = clamp(best_score * (0.65 + 0.35 * features.detector_score), 0.0, 1.0)
        return PoseMatch(best_pose, confidence, runner_pose, runner_score)


class TemporalStabilizer:
    """Requires a pose to persist before accepting it.

    Also implements *release* hysteresis: once a pose is accepted it survives
    a couple of contradicting frames, which stops a mode from dropping out
    during a momentary tracking glitch.
    """

    def __init__(self, required_frames: int = 3, release_frames: int = 2) -> None:
        self.required_frames = max(1, required_frames)
        self.release_frames = max(1, release_frames)
        self._candidate: Pose = Pose.UNKNOWN
        self._candidate_count = 0
        self._stable: Pose = Pose.UNKNOWN
        self._miss_count = 0

    def update(self, pose: Pose) -> Pose:
        """Feed one frame's pose; return the current *stable* pose."""
        if pose == self._stable:
            self._miss_count = 0
            self._candidate = pose
            self._candidate_count = self.required_frames
            return self._stable

        if pose == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = pose
            self._candidate_count = 1

        if self._candidate_count >= self.required_frames:
            if self._stable != self._candidate:
                self._stable = self._candidate
                self._miss_count = 0
            return self._stable

        # Candidate not yet confirmed — hold the previous stable pose briefly.
        self._miss_count += 1
        if self._miss_count > self.release_frames:
            self._stable = Pose.UNKNOWN
        return self._stable

    @property
    def stable_pose(self) -> Pose:
        """The currently accepted pose."""
        return self._stable

    def reset(self) -> None:
        """Forget all state (call when tracking is lost)."""
        self._candidate = Pose.UNKNOWN
        self._candidate_count = 0
        self._stable = Pose.UNKNOWN
        self._miss_count = 0


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #

@dataclass
class GestureEvent:
    """A discrete, actionable gesture occurrence."""

    name: str
    action: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    hand: str = "Right"
    #: Action-specific payload (scroll delta, volume level, swipe direction...).
    data: Dict[str, object] = field(default_factory=dict)
    duration: float = 0.0

    def __str__(self) -> str:
        return f"{self.name} -> {self.action} ({self.confidence:.0%})"


@dataclass
class EngineOutput:
    """Everything the engine produces from one frame."""

    mode: Mode = Mode.NAVIGATE
    pose: Pose = Pose.UNKNOWN
    confidence: float = 0.0
    events: List[GestureEvent] = field(default_factory=list)
    #: Normalized cursor source point, or ``None`` when the cursor shouldn't move.
    cursor_point: Optional[Tuple[float, float]] = None
    features: Optional[HandFeatures] = None
    tracking: bool = False
    #: Continuous control value for the active mode, ``[0, 1]``.
    control_value: Optional[float] = None
    #: Live motion trail in normalized coordinates, for the overlay.
    trail: List[Tuple[float, float]] = field(default_factory=list)
    hand_count: int = 0


# --------------------------------------------------------------------------- #
# Default bindings
# --------------------------------------------------------------------------- #

#: Gestures that *release* a held control.  These bypass the confidence and
#: cooldown gates entirely: requiring confidence to let go of a mouse button
#: is how you end up with a stuck drag when tracking degrades.  You must be
#: confident to engage a control, never to release one.
RELEASE_GESTURES: frozenset = frozenset({"drag_end"})

#: gesture name -> action id.  Users override these in the gesture library.
DEFAULT_BINDINGS: Dict[str, str] = {
    # Discrete pose gestures
    "pinch_tap": "left_click",
    "pinch_double": "double_click",
    "pinch_middle": "right_click",
    "pinch_ring": "middle_click",
    "drag_start": "drag_start",
    "drag_end": "drag_end",
    # Deliberately unbound.  A relaxed or transitioning hand reads as a fist
    # more than any other pose, and "hold_click" *latches* the left button, so
    # an accidental fire leaves the button down until something toggles it
    # back — selecting text and dragging files in the meantime.  Session logs
    # showed this firing 6 times in 2 minutes of ordinary use.
    #
    # Nothing is lost: pinch-and-hold already gives press-and-hold, and it is
    # self-limiting because releasing the pinch releases the button. Users who
    # want the latching behaviour can bind it in the Gestures view.
    "fist_hold": "none",
    "ok_sign": "screenshot",
    "call_hold": "toggle_recording",
    "pinky_hold": "toggle_whiteboard",
    "four_hold": "toggle_presentation",
    "open_palm_hold": "toggle_sleep",
    "thumb_up": "volume_up",
    "thumb_down": "volume_down",
    # Swipes
    "swipe_left": "browser_back",
    "swipe_right": "browser_forward",
    "swipe_up": "next_slide",
    "swipe_down": "prev_slide",
    # Air-drawn shapes
    "circle": "open_browser",
    "circle_cw": "open_browser",
    "triangle": "open_vscode",
    "square": "open_terminal",
    "wave": "toggle_mute",
    "z": "open_spotify",
    "s": "media_play_pause",
    "v_check": "lock_screen",
    "caret": "app_launcher",
}


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class GestureEngine:
    """Turns per-frame hand landmarks into modes, cursor points and events.

    The engine is deliberately free of side effects — it never moves the
    mouse or changes the volume itself.  It reports *what happened*; the
    action layer decides what to do about it.  That separation is what makes
    the whole recognition path unit-testable without a desktop session.
    """

    #: Fallback hold duration when the config predates ``hold_duration``.
    HOLD_DURATION = 1.2

    def __init__(self, cfg: GestureConfig,
                 recognizer: Optional[DollarOneRecognizer] = None) -> None:
        self.cfg = cfg
        self.classifier = StaticClassifier()
        self.stabilizer = TemporalStabilizer(cfg.stability_frames)
        self.recognizer = recognizer or DollarOneRecognizer()
        self.trail = MotionTrail(cfg.motion_history_length)

        self.mode: Mode = Mode.NAVIGATE
        self._previous_mode: Mode = Mode.NAVIGATE

        # Pinch / click state machine
        self._pinch_active = False
        self._pinch_start_time = 0.0
        self._pinch_start_point: Tuple[float, float] = (0.0, 0.0)
        self._drag_active = False
        # Sentinel far in the past: starting at 0.0 would make the very first
        # click register as a double, since engine timestamps start near zero.
        self._last_click_time = float("-inf")
        self._pinch_confidence = 0.0

        # Secondary pinch edges (right / middle click)
        self._middle_pinch_active = False
        self._ring_pinch_active = False

        # Hold tracking
        self._pose_entered_at = 0.0
        self._hold_fired = False
        self._previous_pose_for_hold: Pose = Pose.UNKNOWN

        # Scroll / control anchors
        self._scroll_anchor: Optional[float] = None
        self._control_anchor: Optional[float] = None

        # Two-hand zoom
        self._zoom_anchor: Optional[float] = None

        self._global_cooldown = Cooldown(cfg.global_cooldown)
        self._per_gesture_cooldowns: Dict[str, Cooldown] = {}
        self._last_seen = 0.0
        self._tracking = False
        self.enabled = True
        self.sleeping = False
        self.precision_mode = False

        self.stats: Dict[str, int] = {"clicks": 0, "scrolls": 0, "gestures": 0}

    # -- configuration ---------------------------------------------------- #

    def apply_config(self, cfg: GestureConfig) -> None:
        """Hot-apply new gesture settings without dropping engine state."""
        self.cfg = cfg
        self.stabilizer.required_frames = max(1, cfg.stability_frames)
        self._global_cooldown.interval = cfg.global_cooldown
        self.trail.capacity = cfg.motion_history_length

    @property
    def _hold_duration(self) -> float:
        """Seconds a pose must be held before a hold gesture fires.

        Read from the live config rather than cached, so the Settings slider
        takes effect immediately.  Falls back to the class constant for
        configs saved before the setting existed.
        """
        return float(getattr(self.cfg, "hold_duration", self.HOLD_DURATION))

    def binding_for(self, gesture: str) -> str:
        """Resolve a gesture name to its action id, honouring user overrides."""
        return self.cfg.bindings.get(gesture) or DEFAULT_BINDINGS.get(gesture, "none")

    def is_enabled(self, gesture: str) -> bool:
        """Whether ``gesture`` is currently enabled in the active profile."""
        override = self.cfg.overrides.get(gesture, {})
        return bool(override.get("enabled", True))

    def _cooldown_for(self, gesture: str) -> Cooldown:
        """Per-gesture rate limiter, created lazily from the override table."""
        if gesture not in self._per_gesture_cooldowns:
            override = self.cfg.overrides.get(gesture, {})
            interval = float(override.get("cooldown", self.cfg.global_cooldown))
            self._per_gesture_cooldowns[gesture] = Cooldown(interval)
        return self._per_gesture_cooldowns[gesture]

    # -- event emission --------------------------------------------------- #

    def _emit(self, output: EngineOutput, gesture: str, confidence: float,
              hand: str = "Right", **data: object) -> bool:
        """Queue a gesture event if it passes every gate.

        Gates, in order: gesture enabled → confidence floor → per-gesture
        cooldown → global cooldown.  The global cooldown is checked last so a
        rejected gesture does not consume the shared budget.
        """
        if not self.enabled or not self.is_enabled(gesture):
            return False

        if gesture not in RELEASE_GESTURES:
            override = self.cfg.overrides.get(gesture, {})
            threshold = float(override.get("threshold", self.cfg.min_confidence))
            if confidence < threshold:
                log.debug("suppressed %s: confidence %.2f < %.2f",
                          gesture, confidence, threshold)
                return False

            if not self._cooldown_for(gesture).ready():
                return False
            if not self._global_cooldown.ready():
                return False

        action = self.binding_for(gesture)
        if action == "none":
            return False

        event = GestureEvent(
            name=gesture, action=action, confidence=confidence,
            hand=hand, data=dict(data),
        )
        output.events.append(event)
        self.stats["gestures"] += 1
        log.debug("gesture %s", event)
        return True

    # -- main entry point ------------------------------------------------- #

    def update(self, hands: Sequence[HandLandmarks], timestamp: float) -> EngineOutput:
        """Process one frame's worth of hands.

        Args:
            hands: Detected hands, may be empty.
            timestamp: Monotonic seconds for this frame.

        Returns:
            An :class:`EngineOutput` describing the frame.
        """
        output = EngineOutput(mode=self.mode, hand_count=len(hands))

        if not hands:
            self._handle_tracking_lost(output, timestamp)
            return output

        primary = self._select_primary(hands)
        if primary is None:
            self._handle_tracking_lost(output, timestamp)
            return output

        self._last_seen = timestamp
        if not self._tracking:
            self._tracking = True
            log.debug("tracking acquired")
        output.tracking = True

        features = extract_features(primary)
        output.features = features

        match = self.classifier.classify(features, self.cfg.pinch_threshold)
        stable = self.stabilizer.update(match.pose)
        output.pose = stable
        output.confidence = match.confidence

        if stable != self._previous_pose_for_hold:
            self._pose_entered_at = timestamp
            self._hold_fired = False
            self._previous_pose_for_hold = stable

        # Two-hand zoom takes precedence over everything else.
        if len(hands) >= 2 and self._update_zoom(hands, output, timestamp):
            return output
        self._zoom_anchor = None

        if self.sleeping:
            self._update_sleeping(features, stable, match, output, timestamp)
            return output

        self._update_mode(features, stable, output, timestamp)
        self._update_cursor_source(features, output)
        self._update_pinch_state(features, match, output, timestamp)
        self._update_motion_gestures(features, stable, output, timestamp)
        self._update_holds(features, stable, match, output, timestamp)

        output.mode = self.mode
        return output

    # -- sub-steps -------------------------------------------------------- #

    def _select_primary(self, hands: Sequence[HandLandmarks]) -> Optional[HandLandmarks]:
        """Choose which hand drives the cursor."""
        preferred = self.cfg.overrides.get("_primary_hand", {}).get("value", "Any")
        if preferred in ("Left", "Right"):
            matches = [h for h in hands if h.handedness == preferred]
            if matches:
                return max(matches, key=lambda h: h.score)
        return max(hands, key=lambda h: h.score)

    def _handle_tracking_lost(self, output: EngineOutput, timestamp: float) -> None:
        """Release held state when the hand disappears.

        Releasing a drag on tracking loss is a safety requirement, not a
        nicety: leaving the button down when the user drops their hand out of
        frame would let the desktop keep dragging whatever was grabbed.
        """
        output.tracking = False
        output.mode = self.mode

        if self._tracking and (timestamp - self._last_seen) > self.cfg.tracking_lost_timeout:
            self._tracking = False
            self.stabilizer.reset()
            self.trail.clear()
            log.info("tracking lost")

            if self._drag_active:
                self._drag_active = False
                self.mode = Mode.NAVIGATE
                output.events.append(GestureEvent(
                    name="drag_end", action="drag_end", confidence=1.0,
                    data={"reason": "tracking_lost"},
                ))
            self._pinch_active = False
            self._scroll_anchor = None
            self._control_anchor = None
            if self.mode in (Mode.SCROLL, Mode.VOLUME, Mode.BRIGHTNESS, Mode.ZOOM):
                self.mode = Mode.NAVIGATE

    def _update_sleeping(self, features: HandFeatures, pose: Pose,
                         match: PoseMatch, output: EngineOutput,
                         timestamp: float) -> None:
        """While asleep, only the wake gesture is honoured."""
        output.mode = Mode.SLEEPING
        output.pose = pose
        if pose == Pose.OPEN_PALM and (timestamp - self._pose_entered_at) >= self._hold_duration:
            if not self._hold_fired:
                self._hold_fired = True
                self.sleeping = False
                self.mode = Mode.NAVIGATE
                output.mode = Mode.NAVIGATE
                output.events.append(GestureEvent(
                    name="wake", action="wake_tracking",
                    confidence=match.confidence, hand=features.handedness,
                ))
                log.info("tracking resumed by wake gesture")

    def _update_mode(self, features: HandFeatures, pose: Pose,
                     output: EngineOutput, timestamp: float) -> None:
        """Map the stable pose onto the engine mode."""
        # A drag in progress owns the engine until the pinch is released.
        if self._drag_active:
            self.mode = Mode.DRAG
            return

        self.precision_mode = pose == Pose.GUN

        pose_modes = {
            Pose.PEACE: Mode.SCROLL,
            Pose.ROCK: Mode.VOLUME,
            Pose.THREE: Mode.BRIGHTNESS,
        }
        target = pose_modes.get(pose)

        if target is not None:
            if self.mode != target:
                self.mode = target
                self._scroll_anchor = None
                self._control_anchor = None
                log.debug("mode -> %s", target.value)
        elif self.mode in (Mode.SCROLL, Mode.VOLUME, Mode.BRIGHTNESS):
            self.mode = Mode.NAVIGATE
            self._scroll_anchor = None
            self._control_anchor = None

    def _update_cursor_source(self, features: HandFeatures,
                              output: EngineOutput) -> None:
        """Decide whether the cursor should move this frame, and from where.

        The index fingertip is the natural pointer but it is also the noisiest
        landmark.  While dragging we switch to the palm centre, which is far
        steadier — during a drag the fingers are curled and the fingertip
        estimate degrades badly.
        """
        if self.mode in (Mode.SCROLL, Mode.VOLUME, Mode.BRIGHTNESS):
            output.cursor_point = None
            return

        if self.mode == Mode.DRAG or output.pose == Pose.FIST:
            output.cursor_point = features.palm_centre
        elif output.pose in (Pose.POINT, Pose.GUN, Pose.PINCH_INDEX,
                             Pose.OPEN_PALM, Pose.UNKNOWN):
            output.cursor_point = features.index_tip
        else:
            output.cursor_point = None

    def pinch_confidence(self, features: HandFeatures, finger: int = 0) -> float:
        """Confidence that a deliberate pinch is happening, in ``[0, 1]``.

        Pinch events must *not* be scored by the static pose classifier.  A
        pinching hand matches the L-shape pattern, which the library
        deliberately penalises via ``requires_no_pinch`` — so pose confidence
        collapses at exactly the moment the pinch fires.  Scoring the pinch by
        its own measurement (how firmly closed, weighted by detector
        confidence) is both correct and independent of pose.
        """
        distance = features.pinch_distances[finger]
        span = max(self.cfg.pinch_release_threshold, 1e-6)
        closeness = 1.0 - clamp(distance / span, 0.0, 1.0)
        return clamp(features.detector_score * (0.55 + 0.45 * closeness), 0.0, 1.0)

    def _update_pinch_state(self, features: HandFeatures, match: PoseMatch,
                            output: EngineOutput, timestamp: float) -> None:
        """Click / double-click / drag / drop state machine.

        Uses separate close and release thresholds (Schmitt-trigger style).  A
        single threshold makes the pinch chatter open/closed while the user
        holds it right at the boundary, producing a burst of phantom clicks.
        """
        if self.mode in (Mode.VOLUME, Mode.BRIGHTNESS):
            return  # thumb↔index distance is the control input in these modes

        distance = features.pinch_index
        closed = distance <= self.cfg.pinch_threshold
        released = distance >= self.cfg.pinch_release_threshold

        if closed and not self._pinch_active:
            self._pinch_active = True
            self._pinch_start_time = timestamp
            self._pinch_start_point = features.index_tip
            # Capture confidence at closure, while the pinch is firmest.
            self._pinch_confidence = self.pinch_confidence(features)

        elif self._pinch_active:
            held = timestamp - self._pinch_start_time
            if closed:
                self._pinch_confidence = max(self._pinch_confidence,
                                             self.pinch_confidence(features))

            if not self._drag_active and held >= self.cfg.drag_hold_time:
                self._drag_active = True
                self.mode = Mode.DRAG
                self._emit(output, "drag_start", self._pinch_confidence,
                           features.handedness, point=features.index_tip)

            if released:
                self._pinch_active = False
                if self._drag_active:
                    self._drag_active = False
                    self.mode = Mode.NAVIGATE
                    self._emit(output, "drag_end", features.detector_score,
                               features.handedness, point=features.index_tip)
                elif held < self.cfg.drag_hold_time:
                    self._fire_click(features, match, output, timestamp)

        # Secondary pinches: rising edge only, and never mid-drag.
        if not self._drag_active:
            self._update_secondary_pinch(features, match, output, index=1,
                                         gesture="pinch_middle")
            self._update_secondary_pinch(features, match, output, index=2,
                                         gesture="pinch_ring")

    def _update_secondary_pinch(self, features: HandFeatures, match: PoseMatch,
                                output: EngineOutput, index: int,
                                gesture: str) -> None:
        """Rising-edge detection for the middle/ring pinches."""
        attribute = "_middle_pinch_active" if index == 1 else "_ring_pinch_active"
        active = getattr(self, attribute)
        distance = features.pinch_distances[index]

        if distance <= self.cfg.pinch_threshold and not active:
            setattr(self, attribute, True)
            confidence = self.pinch_confidence(features, finger=index)
            if self._emit(output, gesture, confidence, features.handedness):
                self.stats["clicks"] += 1
        elif distance >= self.cfg.pinch_release_threshold and active:
            setattr(self, attribute, False)

    def _fire_click(self, features: HandFeatures, match: PoseMatch,
                    output: EngineOutput, timestamp: float) -> None:
        """Emit a single or double click depending on inter-click timing."""
        is_double = (timestamp - self._last_click_time) <= self.cfg.double_click_interval
        gesture = "pinch_double" if is_double else "pinch_tap"

        if self._emit(output, gesture, self._pinch_confidence, features.handedness,
                      point=features.index_tip):
            self.stats["clicks"] += 1
            # Reset the timer after a double so a third pinch starts fresh
            # rather than chaining into a triple.
            self._last_click_time = float("-inf") if is_double else timestamp

    def _update_motion_gestures(self, features: HandFeatures, pose: Pose,
                                output: EngineOutput, timestamp: float) -> None:
        """Scroll deltas, continuous controls, swipes and air-drawn shapes."""
        point = features.index_tip

        if self.mode == Mode.SCROLL:
            self._update_scroll(point, output, features)
            self.trail.clear()
            return

        if self.mode in (Mode.VOLUME, Mode.BRIGHTNESS):
            self._update_continuous_control(features, output)
            self.trail.clear()
            return

        # Shapes and swipes are drawn with a single pointing finger, so the
        # trail only accumulates in that pose — this stops ordinary cursor
        # movement from being interpreted as a drawn gesture.
        if pose != Pose.POINT or self._drag_active:
            self.trail.clear()
            return

        self.trail.add(point, timestamp)
        output.trail = self.trail.points

        if len(self.trail) < 12 or self.trail.length < self.cfg.dynamic_min_path:
            return

        swipe = detect_swipe(self.trail.points, self.trail.duration)
        if swipe is not None and swipe.confidence >= self.cfg.min_confidence:
            if self._emit(output, f"swipe_{swipe.direction}", swipe.confidence,
                          features.handedness, direction=swipe.direction,
                          speed=swipe.speed):
                self.trail.clear()
            return

        # Only attempt shape matching once the stroke has settled, otherwise
        # a half-drawn circle matches "caret" and fires early.
        if not self.trail.is_stationary:
            return

        result = self.recognizer.recognize(self.trail.points,
                                           self.cfg.dynamic_min_score)
        if result is not None:
            if self._emit(output, result.name, result.score, features.handedness,
                          shape=result.name, ranked=result.ranked):
                self.trail.clear()

    def _update_scroll(self, point: Tuple[float, float], output: EngineOutput,
                       features: HandFeatures) -> None:
        """Convert vertical hand travel into scroll ticks."""
        if self._scroll_anchor is None:
            self._scroll_anchor = point[1]
            return

        delta = point[1] - self._scroll_anchor
        # Dead zone stops a perfectly still hand from creeping.
        if abs(delta) < 0.012:
            return

        direction = "down" if delta > 0 else "up"
        self._scroll_anchor = point[1]
        self.stats["scrolls"] += 1
        output.events.append(GestureEvent(
            name=f"scroll_{direction}", action="scroll",
            confidence=features.detector_score, hand=features.handedness,
            data={"delta": float(delta), "direction": direction},
        ))

    def _update_continuous_control(self, features: HandFeatures,
                                   output: EngineOutput) -> None:
        """Map thumb↔index distance onto a ``[0, 1]`` control value.

        The mapping range is expressed in palm-normalised units so the full
        travel is the same physical gesture regardless of hand size.
        """
        level = clamp((features.pinch_index - 0.15) / (1.15 - 0.15), 0.0, 1.0)
        output.control_value = level

        action = "set_volume" if self.mode == Mode.VOLUME else "set_brightness"
        output.events.append(GestureEvent(
            name=f"{self.mode.value.lower()}_set", action=action,
            confidence=features.detector_score, hand=features.handedness,
            data={"level": level},
        ))

    def _update_holds(self, features: HandFeatures, pose: Pose, match: PoseMatch,
                      output: EngineOutput, timestamp: float) -> None:
        """Fire the 'hold this pose' gestures once per hold."""
        if self._hold_fired or self._drag_active:
            return

        held = timestamp - self._pose_entered_at
        if held < self._hold_duration:
            return

        hold_gestures = {
            Pose.OK_SIGN: "ok_sign",
            Pose.CALL: "call_hold",
            Pose.PINKY: "pinky_hold",
            Pose.FOUR: "four_hold",
            Pose.FIST: "fist_hold",
            Pose.OPEN_PALM: "open_palm_hold",
        }
        gesture = hold_gestures.get(pose)
        if gesture is None:
            return

        if self._emit(output, gesture, match.confidence, features.handedness,
                      held=held):
            self._hold_fired = True
            if gesture == "open_palm_hold":
                self.sleeping = True
                self.mode = Mode.SLEEPING
                output.mode = Mode.SLEEPING
                log.info("tracking paused by sleep gesture")

    def _update_zoom(self, hands: Sequence[HandLandmarks], output: EngineOutput,
                     timestamp: float) -> bool:
        """Two-handed pinch zoom; returns True when zoom consumed the frame."""
        left = next((h for h in hands if h.handedness == "Left"), None)
        right = next((h for h in hands if h.handedness == "Right"), None)
        if left is None or right is None:
            return False

        left_features = extract_features(left)
        right_features = extract_features(right)

        # Both hands must be pinching for zoom to engage — that makes the
        # gesture deliberate and impossible to trigger while typing-adjacent
        # hand motion happens in frame.
        if (left_features.pinch_index > self.cfg.pinch_threshold * 1.6
                or right_features.pinch_index > self.cfg.pinch_threshold * 1.6):
            self._zoom_anchor = None
            return False

        separation = distance_2d(left_features.index_tip, right_features.index_tip)
        self.mode = Mode.ZOOM
        output.mode = Mode.ZOOM
        output.pose = Pose.PINCH_INDEX
        output.tracking = True
        output.features = right_features

        if self._zoom_anchor is None:
            self._zoom_anchor = separation
            return True

        delta = separation - self._zoom_anchor
        if abs(delta) >= 0.04:
            self._zoom_anchor = separation
            output.events.append(GestureEvent(
                name="pinch_zoom", action="zoom",
                confidence=min(left_features.detector_score,
                               right_features.detector_score),
                data={"delta": float(delta),
                      "direction": "in" if delta > 0 else "out"},
            ))
        return True

    # -- control ---------------------------------------------------------- #

    def pause(self) -> None:
        """Stop emitting events but keep processing frames."""
        self.enabled = False
        log.info("gesture engine paused")

    def resume(self) -> None:
        """Resume emitting events."""
        self.enabled = True
        self.sleeping = False
        self.mode = Mode.NAVIGATE
        log.info("gesture engine resumed")

    def reset(self) -> None:
        """Return the engine to a clean state."""
        self.stabilizer.reset()
        self.trail.clear()
        self.mode = Mode.NAVIGATE
        self._pinch_active = False
        self._drag_active = False
        self._middle_pinch_active = False
        self._ring_pinch_active = False
        self._scroll_anchor = None
        self._control_anchor = None
        self._zoom_anchor = None
        self._hold_fired = False
