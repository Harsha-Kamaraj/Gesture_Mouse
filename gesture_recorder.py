"""Custom gesture recording and the gesture library.

Users record a new air-drawn shape by performing it two or three times.  The
recorder captures the fingertip trail for each repetition, validates that the
repetitions actually agree with each other, and stores them as $1 templates.

Validating *self-consistency* before saving is what stops the library filling
with unusable gestures: if a user's three attempts at "spiral" do not resemble
each other, that gesture will never be recognised reliably in use, and telling
them at record time is far better than letting it silently misfire later.

The library also checks a new gesture against existing ones and refuses names
that collide geometrically — two templates that score highly against each
other would make both unreliable.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from config import GESTURE_LIBRARY_FILE
from dynamic_gestures import (
    DollarOneRecognizer, GestureTemplate, _optimal_cosine_distance,
    builtin_templates, normalize_stroke,
)
from logger import get_logger
from utils import path_length

log = get_logger(__name__)

Point = Tuple[float, float]


class RecordState(str, Enum):
    """Recorder state machine."""

    IDLE = "Idle"
    COUNTDOWN = "Get ready"
    RECORDING = "Recording"
    BETWEEN = "Repeat the gesture"
    REVIEW = "Review"


@dataclass
class RecordingSession:
    """One in-progress custom gesture recording."""

    name: str
    required_takes: int = 3
    takes: List[List[Point]] = field(default_factory=list)
    current: List[Point] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)

    @property
    def complete(self) -> bool:
        """Whether enough repetitions have been captured."""
        return len(self.takes) >= self.required_takes

    @property
    def progress(self) -> float:
        """Fraction of required takes captured."""
        return min(1.0, len(self.takes) / max(self.required_takes, 1))


@dataclass
class ValidationReport:
    """Result of checking a set of recorded takes."""

    valid: bool
    consistency: float = 0.0
    conflicts: List[Tuple[str, float]] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """One-line verdict for the UI."""
        if self.valid:
            return f"Looks good — {self.consistency:.0%} consistent across takes"
        return "; ".join(self.messages) or "Recording rejected"


class GestureRecorder:
    """Captures repeated fingertip strokes and turns them into templates."""

    #: Minimum normalized path length for a stroke to count as a gesture.
    MIN_PATH = 0.35
    #: Minimum samples per take.
    MIN_SAMPLES = 12
    #: Takes must agree at least this well to be accepted.
    MIN_CONSISTENCY = 0.72
    #: A new gesture scoring above this against an existing one is a conflict.
    MAX_SIMILARITY = 0.93
    #: Seconds of stillness that ends a take.
    STILLNESS_TIMEOUT = 0.55

    def __init__(self, library: Optional["GestureLibrary"] = None,
                 on_state_change: Optional[Callable[[RecordState, str], None]] = None) -> None:
        self.library = library
        self.on_state_change = on_state_change
        self.state = RecordState.IDLE
        self.session: Optional[RecordingSession] = None
        self._last_movement = 0.0
        self._countdown = 0.0
        self._countdown_until: Optional[float] = None
        self.report: Optional[ValidationReport] = None

    # -- lifecycle -------------------------------------------------------- #

    def start(self, name: str, takes: int = 3, countdown: float = 1.5) -> None:
        """Begin recording a gesture called ``name``.

        The countdown deadline is resolved on the first :meth:`update` rather
        than here, so the recorder uses exactly one clock — the caller's.
        Anchoring it to ``time.monotonic()`` here would silently break any
        caller feeding timestamps from a different source (a test harness, or
        a frame-relative clock).
        """
        self.session = RecordingSession(name=name, required_takes=max(1, takes))
        self.report = None
        self._countdown = max(0.0, countdown)
        self._countdown_until = None
        self._set_state(RecordState.COUNTDOWN,
                        f"Get ready to draw {name!r} {takes} times")
        log.info("recording gesture %r (%d takes)", name, takes)

    def cancel(self) -> None:
        """Abort the current recording."""
        self.session = None
        self._set_state(RecordState.IDLE, "Cancelled")

    def _set_state(self, state: RecordState, message: str = "") -> None:
        """Transition state and notify the UI."""
        self.state = state
        if self.on_state_change:
            try:
                self.on_state_change(state, message)
            except Exception as exc:
                log.debug("recorder callback failed: %s", exc)

    @property
    def is_active(self) -> bool:
        """Whether a recording session is in progress."""
        return self.state != RecordState.IDLE and self.session is not None

    # -- capture ---------------------------------------------------------- #

    def update(self, point: Optional[Point], drawing: bool,
               timestamp: float) -> RecordState:
        """Feed one frame of fingertip position into the recorder.

        Args:
            point: Normalized fingertip position, or ``None`` if not tracked.
            drawing: Whether the user is currently in the drawing pose.
            timestamp: Monotonic seconds.

        Returns:
            The current recorder state.
        """
        if self.session is None:
            return self.state

        if self.state == RecordState.COUNTDOWN:
            if self._countdown_until is None:
                self._countdown_until = timestamp + self._countdown
            if timestamp >= self._countdown_until:
                self._set_state(RecordState.RECORDING, "Draw now")
            return self.state

        if self.state == RecordState.BETWEEN:
            if drawing and point is not None:
                self._set_state(RecordState.RECORDING, "Draw now")
            else:
                return self.state

        if self.state != RecordState.RECORDING:
            return self.state

        if drawing and point is not None:
            self.session.current.append(point)
            self._last_movement = timestamp
            return self.state

        # Not drawing: end the take once the stroke has been still long enough.
        if self.session.current:
            if (timestamp - self._last_movement) >= self.STILLNESS_TIMEOUT:
                self._end_take()
        return self.state

    def _end_take(self) -> None:
        """Finish the current stroke and either store or discard it."""
        assert self.session is not None
        stroke = self.session.current
        self.session.current = []

        if len(stroke) < self.MIN_SAMPLES or path_length(stroke) < self.MIN_PATH:
            self._set_state(RecordState.BETWEEN,
                            "Too short — draw a larger shape")
            return

        self.session.takes.append(stroke)
        if self.session.complete:
            self._review()
        else:
            remaining = self.session.required_takes - len(self.session.takes)
            self._set_state(RecordState.BETWEEN,
                            f"Good — {remaining} more to go")

    def _review(self) -> None:
        """Validate the captured takes and move to review."""
        assert self.session is not None
        self.report = self.validate(self.session.takes, self.session.name)
        self._set_state(RecordState.REVIEW, self.report.summary)

    # -- validation ------------------------------------------------------- #

    def validate(self, takes: Sequence[Sequence[Point]],
                 name: str = "") -> ValidationReport:
        """Check that takes agree with each other and don't clash with the library."""
        report = ValidationReport(valid=True)

        if len(takes) < 1:
            return ValidationReport(False, messages=["No strokes recorded"])

        vectors = [normalize_stroke(list(take))[1] for take in takes]

        if len(vectors) > 1:
            scores: List[float] = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    angle = _optimal_cosine_distance(vectors[i], vectors[j])
                    scores.append(1.0 / (1.0 + angle))
            report.consistency = sum(scores) / len(scores)

            if report.consistency < self.MIN_CONSISTENCY:
                report.valid = False
                report.messages.append(
                    f"Your takes differ too much ({report.consistency:.0%} alike). "
                    "Try to draw the shape the same way each time."
                )
        else:
            report.consistency = 1.0

        # Geometric collision with the existing library.
        if self.library is not None:
            for template in self.library.recognizer.templates:
                if not template.enabled or template.name == name:
                    continue
                if template.vector.shape != vectors[0].shape:
                    continue
                angle = _optimal_cosine_distance(template.vector, vectors[0])
                similarity = 1.0 / (1.0 + angle)
                if similarity >= self.MAX_SIMILARITY:
                    report.conflicts.append((template.name, similarity))

            if report.conflicts:
                report.valid = False
                worst = max(report.conflicts, key=lambda c: c[1])
                report.messages.append(
                    f"Too similar to {worst[0]!r} ({worst[1]:.0%}). "
                    "Both gestures would be unreliable."
                )

        return report

    # -- commit ----------------------------------------------------------- #

    def save(self, action: str = "none", force: bool = False) -> bool:
        """Store the recorded takes in the library.

        Args:
            action: Action id to bind the new gesture to.
            force: Save even if validation failed.

        Returns:
            True when the gesture was stored.
        """
        if self.session is None or not self.session.takes:
            return False
        if self.report is not None and not self.report.valid and not force:
            log.warning("refusing to save %r: %s",
                        self.session.name, self.report.summary)
            return False
        if self.library is None:
            log.error("no library attached to recorder")
            return False

        self.library.add_custom(
            name=self.session.name,
            takes=self.session.takes,
            action=action,
        )
        log.info("saved custom gesture %r with %d exemplars",
                 self.session.name, len(self.session.takes))
        self.session = None
        self._set_state(RecordState.IDLE, "Saved")
        return True


@dataclass
class CustomGesture:
    """A user-defined gesture and its metadata."""

    name: str
    action: str = "none"
    enabled: bool = True
    threshold: float = 0.82
    cooldown: float = 0.5
    takes: List[List[Point]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable form."""
        return {
            "name": self.name,
            "action": self.action,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "cooldown": self.cooldown,
            "created_at": self.created_at,
            "description": self.description,
            "takes": [[[round(x, 5), round(y, 5)] for x, y in take]
                      for take in self.takes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CustomGesture":
        """Rebuild from a serialised gesture."""
        raw_takes = data.get("takes") or []
        takes = [
            [(float(p[0]), float(p[1])) for p in take]   # type: ignore[index]
            for take in raw_takes                        # type: ignore[union-attr]
        ]
        return cls(
            name=str(data.get("name", "unnamed")),
            action=str(data.get("action", "none")),
            enabled=bool(data.get("enabled", True)),
            threshold=float(data.get("threshold", 0.82)),   # type: ignore[arg-type]
            cooldown=float(data.get("cooldown", 0.5)),      # type: ignore[arg-type]
            takes=takes,
            created_at=float(data.get("created_at", time.time())),  # type: ignore[arg-type]
            description=str(data.get("description", "")),
        )


class GestureLibrary:
    """Owns the recogniser plus the custom-gesture metadata and persistence."""

    def __init__(self, path: Path = GESTURE_LIBRARY_FILE) -> None:
        self.path = path
        self.recognizer = DollarOneRecognizer(builtin_templates())
        self.custom: Dict[str, CustomGesture] = {}
        self._lock = threading.RLock()
        self.load()

    # -- persistence ------------------------------------------------------ #

    def load(self) -> int:
        """Load custom gestures from disk and register their templates."""
        if not self.path.exists():
            return 0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read gesture library: %s", exc)
            return 0

        entries = data.get("gestures", []) if isinstance(data, dict) else data
        loaded = 0
        with self._lock:
            for entry in entries:
                try:
                    gesture = CustomGesture.from_dict(entry)
                except (TypeError, ValueError, KeyError):
                    continue
                self.custom[gesture.name] = gesture
                self._register_templates(gesture)
                loaded += 1
        log.info("loaded %d custom gestures", loaded)
        return loaded

    def save(self) -> bool:
        """Write custom gestures to disk atomically."""
        with self._lock:
            payload = {
                "version": 1,
                "gestures": [g.to_dict() for g in self.custom.values()],
            }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            return True
        except OSError as exc:
            log.error("could not save gesture library: %s", exc)
            return False

    def _register_templates(self, gesture: CustomGesture) -> None:
        """Add a custom gesture's exemplars to the recogniser."""
        self.recognizer.remove(gesture.name)
        if not gesture.enabled:
            return
        for take in gesture.takes:
            if len(take) >= 8:
                self.recognizer.add(GestureTemplate(gesture.name, list(take)))

    # -- CRUD ------------------------------------------------------------- #

    def add_custom(self, name: str, takes: Sequence[Sequence[Point]],
                   action: str = "none", description: str = "") -> CustomGesture:
        """Create or replace a custom gesture."""
        with self._lock:
            gesture = CustomGesture(
                name=name, action=action, description=description,
                takes=[list(take) for take in takes],
            )
            self.custom[name] = gesture
            self._register_templates(gesture)
        self.save()
        return gesture

    def rename(self, old: str, new: str) -> bool:
        """Rename a custom gesture."""
        with self._lock:
            if old not in self.custom or new in self.custom:
                return False
            gesture = self.custom.pop(old)
            gesture.name = new
            self.custom[new] = gesture
            self.recognizer.remove(old)
            self._register_templates(gesture)
        self.save()
        log.info("renamed gesture %r -> %r", old, new)
        return True

    def delete(self, name: str) -> bool:
        """Remove a custom gesture and its templates."""
        with self._lock:
            if name not in self.custom:
                return False
            del self.custom[name]
            self.recognizer.remove(name)
        self.save()
        log.info("deleted gesture %r", name)
        return True

    def duplicate(self, name: str) -> Optional[CustomGesture]:
        """Copy a custom gesture under a free name."""
        with self._lock:
            source = self.custom.get(name)
            if source is None:
                return None
            new_name = f"{name} copy"
            index = 2
            while new_name in self.custom:
                new_name = f"{name} copy {index}"
                index += 1
        return self.add_custom(new_name, source.takes, source.action,
                               source.description)

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a gesture (custom or built-in)."""
        with self._lock:
            gesture = self.custom.get(name)
            if gesture is not None:
                gesture.enabled = enabled
                self._register_templates(gesture)
                self.save()
                return True
            # Built-in shape templates can be toggled too.
            if name in self.recognizer.names:
                self.recognizer.set_enabled(name, enabled)
                return True
        return False

    def set_action(self, name: str, action: str) -> bool:
        """Rebind a custom gesture to a different action."""
        with self._lock:
            gesture = self.custom.get(name)
            if gesture is None:
                return False
            gesture.action = action
        self.save()
        return True

    def get(self, name: str) -> Optional[CustomGesture]:
        """Look up a custom gesture."""
        return self.custom.get(name)

    @property
    def names(self) -> List[str]:
        """Every recognisable shape name, built-in and custom."""
        return self.recognizer.names

    @property
    def custom_names(self) -> List[str]:
        """Names of user-recorded gestures only."""
        return sorted(self.custom)

    # -- import / export -------------------------------------------------- #

    def export(self, destination: Path, names: Optional[Sequence[str]] = None) -> int:
        """Export selected (or all) custom gestures to a file."""
        with self._lock:
            selected = [g for name, g in self.custom.items()
                        if names is None or name in names]
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps({"version": 1,
                            "gestures": [g.to_dict() for g in selected]}, indent=2),
                encoding="utf-8",
            )
            log.info("exported %d gestures to %s", len(selected), destination)
            return len(selected)
        except OSError as exc:
            log.error("gesture export failed: %s", exc)
            return 0

    def import_gestures(self, source: Path, overwrite: bool = False) -> int:
        """Import gestures from a file, skipping name clashes unless overwriting."""
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("gesture import failed: %s", exc)
            return 0

        entries = data.get("gestures", []) if isinstance(data, dict) else data
        imported = 0
        with self._lock:
            for entry in entries:
                try:
                    gesture = CustomGesture.from_dict(entry)
                except (TypeError, ValueError, KeyError):
                    continue
                if gesture.name in self.custom and not overwrite:
                    log.debug("skipping existing gesture %r", gesture.name)
                    continue
                self.custom[gesture.name] = gesture
                self._register_templates(gesture)
                imported += 1
        if imported:
            self.save()
        log.info("imported %d gestures", imported)
        return imported
