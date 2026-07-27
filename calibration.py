"""Calibration wizard.

Gesture thresholds that are hard-coded for "an average hand at an average
distance in average lighting" work badly for everyone else.  Calibration
measures the actual user in their actual environment and derives settings
from those measurements.

The wizard is a state machine with four measurement stages:

============  ==========================================================
Stage         What it measures and why
============  ==========================================================
HAND_SIZE     Apparent palm size, which is a proxy for camera distance.
              Sets how far the hand must travel per screen pixel.
PINCH_RANGE   The user's own open and closed pinch distances.  People's
              pinch geometry varies enough that a fixed threshold either
              misses deliberate pinches or fires on a relaxed hand.
MOVEMENT      The rectangle the hand comfortably reaches, which becomes
              the active region.  This is the single biggest ergonomic
              win: no more reaching for screen corners.
STABILITY     Landmark jitter while holding still, which sets the dead
              zone and smoothing strength.
============  ==========================================================

Each stage collects samples for a fixed duration, then derives settings from
robust statistics (percentiles, not means) so one bad frame cannot skew the
result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import AppConfig
from detector import HandLandmarks
from gesture_engine import HandFeatures, extract_features
from logger import get_logger
from utils import clamp, distance_2d

log = get_logger(__name__)


class Stage(str, Enum):
    """Calibration wizard stages, in order."""

    IDLE = "Idle"
    HAND_SIZE = "Hand Size"
    PINCH_RANGE = "Pinch Range"
    MOVEMENT = "Movement Range"
    STABILITY = "Stability"
    COMPLETE = "Complete"


#: Stage -> (instruction, seconds to collect).
STAGE_SCRIPT: Dict[Stage, Tuple[str, float]] = {
    Stage.HAND_SIZE: (
        "Hold your open hand comfortably in front of the camera",
        4.0,
    ),
    Stage.PINCH_RANGE: (
        "Slowly pinch your thumb and index finger together, then apart. Repeat.",
        6.0,
    ),
    Stage.MOVEMENT: (
        "Point with your index finger and trace the edges of your "
        "comfortable reach",
        7.0,
    ),
    Stage.STABILITY: (
        "Point at the centre of the screen and hold as still as you can",
        4.0,
    ),
}


@dataclass
class CalibrationResult:
    """Measurements and the settings derived from them."""

    palm_size: float = 0.0
    camera_distance: str = "unknown"
    pinch_closed: float = 0.0
    pinch_open: float = 0.0
    reach: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    jitter: float = 0.0
    brightness: float = 0.0
    lighting: str = "unknown"
    sample_count: int = 0
    duration: float = 0.0
    #: Derived, ready-to-apply settings.
    settings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        """Human-readable report for the wizard's final screen."""
        left, top, right, bottom = self.reach
        return [
            f"Hand size: {self.palm_size:.3f} ({self.camera_distance})",
            f"Pinch range: {self.pinch_closed:.3f} - {self.pinch_open:.3f}",
            f"Reach: {(right - left):.0%} x {(bottom - top):.0%} of frame",
            f"Jitter: {self.jitter:.4f}",
            f"Lighting: {self.lighting}",
            f"Samples: {self.sample_count}",
        ]

    def to_dict(self) -> Dict[str, object]:
        """Serialisable form, stored on the profile."""
        return {
            "palm_size": round(self.palm_size, 5),
            "camera_distance": self.camera_distance,
            "pinch_closed": round(self.pinch_closed, 5),
            "pinch_open": round(self.pinch_open, 5),
            "reach": [round(v, 4) for v in self.reach],
            "jitter": round(self.jitter, 6),
            "lighting": self.lighting,
            "brightness": round(self.brightness, 2),
            "sample_count": self.sample_count,
            "settings": {k: round(v, 5) for k, v in self.settings.items()},
            "calibrated_at": time.time(),
        }


class CalibrationWizard:
    """Drives the calibration sequence, one frame at a time.

    The wizard never blocks and never owns a loop: the application feeds it
    frames, and it reports progress.  That keeps the UI responsive and lets
    the same wizard drive a CLI, a GUI or a test.
    """

    def __init__(self, on_stage_change: Optional[Callable[[Stage, str], None]] = None,
                 on_complete: Optional[Callable[[CalibrationResult], None]] = None) -> None:
        self.stage = Stage.IDLE
        self.on_stage_change = on_stage_change
        self.on_complete = on_complete

        self._stage_started = 0.0
        self._samples: Dict[str, List[float]] = {}
        self._positions: List[Tuple[float, float]] = []
        self._brightness_samples: List[float] = []
        self._started = 0.0
        self.result = CalibrationResult()
        self._order = [Stage.HAND_SIZE, Stage.PINCH_RANGE,
                       Stage.MOVEMENT, Stage.STABILITY]

    # -- lifecycle -------------------------------------------------------- #

    @property
    def is_running(self) -> bool:
        """Whether calibration is in progress."""
        return self.stage not in (Stage.IDLE, Stage.COMPLETE)

    @property
    def instruction(self) -> str:
        """Current on-screen instruction."""
        return STAGE_SCRIPT.get(self.stage, ("", 0.0))[0]

    @property
    def progress(self) -> float:
        """Progress through the *current* stage, ``[0, 1]``."""
        if not self.is_running:
            return 1.0 if self.stage == Stage.COMPLETE else 0.0
        duration = STAGE_SCRIPT.get(self.stage, ("", 1.0))[1]
        return clamp((time.monotonic() - self._stage_started) / duration, 0.0, 1.0)

    @property
    def overall_progress(self) -> float:
        """Progress through the whole wizard, ``[0, 1]``."""
        if self.stage == Stage.COMPLETE:
            return 1.0
        if self.stage == Stage.IDLE:
            return 0.0
        index = self._order.index(self.stage)
        return (index + self.progress) / len(self._order)

    def start(self) -> None:
        """Begin calibration from the first stage."""
        self._samples = {}
        self._positions = []
        self._brightness_samples = []
        self.result = CalibrationResult()
        self._started = time.monotonic()
        self._enter(self._order[0])
        log.info("calibration started")

    def cancel(self) -> None:
        """Abort calibration, discarding measurements."""
        self.stage = Stage.IDLE
        log.info("calibration cancelled")

    def _enter(self, stage: Stage) -> None:
        """Transition into a stage and reset its timer."""
        self.stage = stage
        self._stage_started = time.monotonic()
        if self.on_stage_change:
            try:
                self.on_stage_change(stage, self.instruction)
            except Exception as exc:
                log.debug("stage callback failed: %s", exc)

    def _advance(self) -> None:
        """Move to the next stage, or finish."""
        index = self._order.index(self.stage)
        if index + 1 < len(self._order):
            self._enter(self._order[index + 1])
        else:
            self._finish()

    # -- sampling --------------------------------------------------------- #

    def update(self, hands: Sequence[HandLandmarks],
               frame_brightness: Optional[float] = None) -> Stage:
        """Feed one frame into the wizard.  Returns the current stage."""
        if not self.is_running:
            return self.stage

        if frame_brightness is not None:
            self._brightness_samples.append(frame_brightness)

        if hands:
            features = extract_features(max(hands, key=lambda h: h.score))
            self._collect(features)

        if self.progress >= 1.0:
            self._advance()
        return self.stage

    def _collect(self, features: HandFeatures) -> None:
        """Record the measurements relevant to the current stage."""
        def push(key: str, value: float) -> None:
            self._samples.setdefault(key, []).append(value)

        if self.stage == Stage.HAND_SIZE:
            push("palm_size", features.palm_size)

        elif self.stage == Stage.PINCH_RANGE:
            push("pinch", features.pinch_index)

        elif self.stage == Stage.MOVEMENT:
            self._positions.append(features.index_tip)

        elif self.stage == Stage.STABILITY:
            self._positions.append(features.index_tip)
            if len(self._positions) >= 2:
                push("jitter", distance_2d(self._positions[-1], self._positions[-2]))

    # -- derivation ------------------------------------------------------- #

    def _finish(self) -> None:
        """Compute the result and derived settings."""
        result = CalibrationResult()
        result.duration = time.monotonic() - self._started
        result.sample_count = sum(len(v) for v in self._samples.values())

        palm = self._samples.get("palm_size", [])
        if palm:
            result.palm_size = float(np.median(palm))
            result.camera_distance = _classify_distance(result.palm_size)
        else:
            result.warnings.append("No hand detected during sizing")

        pinch = self._samples.get("pinch", [])
        if len(pinch) >= 10:
            # Percentiles, not min/max: a single bad landmark frame would
            # otherwise define the entire pinch range.
            result.pinch_closed = float(np.percentile(pinch, 8))
            result.pinch_open = float(np.percentile(pinch, 92))
            if result.pinch_open - result.pinch_closed < 0.15:
                result.warnings.append(
                    "Pinch range is narrow — try opening your fingers wider")
        else:
            result.warnings.append("Not enough pinch samples")

        if len(self._positions) >= 20:
            xs = [p[0] for p in self._positions]
            ys = [p[1] for p in self._positions]
            result.reach = (
                float(np.percentile(xs, 3)), float(np.percentile(ys, 3)),
                float(np.percentile(xs, 97)), float(np.percentile(ys, 97)),
            )
        else:
            result.warnings.append("Not enough movement samples")

        jitter = self._samples.get("jitter", [])
        if jitter:
            result.jitter = float(np.percentile(jitter, 75))

        if self._brightness_samples:
            result.brightness = float(np.mean(self._brightness_samples))
            result.lighting = _classify_lighting(result.brightness)
            if result.lighting == "dark":
                result.warnings.append(
                    "Lighting is dim — detection accuracy will suffer")

        result.settings = derive_settings(result)
        self.result = result
        self.stage = Stage.COMPLETE

        log.info("calibration complete: %s", "; ".join(result.summary_lines()))
        if result.warnings:
            log.warning("calibration warnings: %s", "; ".join(result.warnings))
        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception as exc:
                log.debug("completion callback failed: %s", exc)


def _classify_distance(palm_size: float) -> str:
    """Describe camera distance from the apparent palm size."""
    if palm_size < 0.10:
        return "far"
    if palm_size < 0.18:
        return "comfortable"
    return "close"


def _classify_lighting(brightness: float) -> str:
    """Describe scene lighting from mean frame luminance (0-255)."""
    if brightness < 55:
        return "dark"
    if brightness < 110:
        return "dim"
    if brightness < 200:
        return "good"
    return "bright"


def derive_settings(result: CalibrationResult) -> Dict[str, float]:
    """Turn raw measurements into concrete configuration values.

    Every derivation is clamped to a sane band: calibration should adapt the
    defaults, never produce a configuration that makes the app unusable if a
    measurement went wrong.
    """
    settings: Dict[str, float] = {}

    # Pinch thresholds are anchored near the *closed* end of the measured
    # range, not its midpoint.  A click must require a deliberate pinch;
    # placing the threshold a large fraction of the way toward an open hand
    # would fire whenever the fingers merely drifted inward.  The absolute
    # floors keep the band wider than landmark noise.
    if result.pinch_open > result.pinch_closed >= 0:
        span = result.pinch_open - result.pinch_closed
        close_point = result.pinch_closed + max(0.035, span * 0.05)
        release_point = result.pinch_closed + max(0.065, span * 0.10)

        threshold = clamp(close_point, 0.03, 0.16)
        release = clamp(release_point, 0.05, 0.24)
        # Hysteresis is the whole point of having two thresholds, so never
        # let a degenerate measurement collapse them onto each other.
        release = max(release, threshold * 1.4)

        settings["pinch_threshold"] = threshold
        settings["pinch_release_threshold"] = release

    # Dead zone tracks measured jitter, so a steady hand gets a fine cursor
    # and a shaky one gets a stable one.
    if result.jitter > 0:
        settings["dead_zone"] = clamp(result.jitter * 1.5, 0.001, 0.03)
        # More jitter also warrants more smoothing.
        settings["smoothing"] = clamp(0.40 + result.jitter * 22.0, 0.30, 0.88)

    # The active region becomes the measured comfortable reach, expressed as
    # the margin trimmed from each edge.
    left, top, right, bottom = result.reach
    width, height = right - left, bottom - top
    if width > 0.15 and height > 0.15:
        margin = clamp((1.0 - min(width, height)) / 2.0, 0.02, 0.40)
        settings["active_region_margin"] = margin

    # Compensate for camera distance: a hand that appears small has less
    # pixel travel available, so it needs more gain.
    if result.palm_size > 0:
        settings["sensitivity"] = clamp(0.16 / result.palm_size, 0.6, 2.2)

    return settings


def apply_calibration(config: AppConfig, result: CalibrationResult) -> AppConfig:
    """Write derived settings into a config and stash the raw measurements."""
    settings = result.settings

    for key in ("pinch_threshold", "pinch_release_threshold"):
        if key in settings:
            setattr(config.gestures, key, settings[key])

    for key in ("dead_zone", "smoothing", "active_region_margin", "sensitivity"):
        if key in settings:
            setattr(config.cursor, key, settings[key])

    config.calibration = result.to_dict()
    log.info("applied calibration to profile %r", config.profile_name)
    return config
