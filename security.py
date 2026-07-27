"""Presence detection and auto-lock.

A gesture mouse watches the user through a camera the whole time it runs, so
it is uniquely placed to notice when they walk away — and uniquely dangerous
if it keeps accepting gestures after they do.

This module provides two safety features:

* **Presence detection** — pauses tracking when nobody is in front of the
  camera, so a pet or a passer-by cannot drive the mouse.
* **Auto-lock** — locks the desktop session after a configurable absence.

Backends
--------
Like the hand detector, face detection is backend-agnostic:

* **MediaPipe BlazeFace** (preferred) — MediaPipe is already a core
  dependency, the model is ~224 KB, and it is markedly more robust to pose
  and lighting than a cascade.
* **OpenCV Haar cascade** (fallback) — needs no download, but *OpenCV 5.0
  removed the bundled cascade XML files entirely*, so this path only works
  on older OpenCV builds. Discovering that at runtime rather than assuming
  is exactly why the backend is pluggable.

Detection runs at a low rate on a background thread.  Running it inline would
roughly halve the gesture frame rate for a signal that changes on the scale
of seconds.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from logger import get_logger

log = get_logger(__name__)


class PresenceState(str, Enum):
    """Whether a user appears to be present."""

    UNKNOWN = "Unknown"
    PRESENT = "Present"
    ABSENT = "Absent"
    LOCKED = "Locked"


@dataclass
class SecurityConfig:
    """Tuning for presence detection."""

    enabled: bool = False
    #: Seconds without a face before presence flips to ABSENT.
    absence_threshold: float = 8.0
    #: Seconds absent before the session is locked; 0 disables auto-lock.
    auto_lock_seconds: float = 0.0
    #: How often face detection runs, in seconds.
    detection_interval: float = 1.0
    #: Pause gesture tracking while absent.
    pause_on_absence: bool = True
    #: Minimum face size as a fraction of frame height.
    min_face_fraction: float = 0.08


@dataclass
class PresenceStatus:
    """Current presence assessment."""

    state: PresenceState = PresenceState.UNKNOWN
    face_count: int = 0
    last_seen: float = 0.0
    absent_for: float = 0.0
    faces: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def is_present(self) -> bool:
        """Whether a user is considered present."""
        return self.state == PresenceState.PRESENT


FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _ensure_face_model() -> Optional[Path]:
    """Return the BlazeFace model path, downloading it once if needed."""
    from detector import MODEL_DIR

    path = MODEL_DIR / "blaze_face_short_range.tflite"
    if path.exists() and path.stat().st_size > 1024:
        return path

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log.info("downloading face detection model (~224 KB), one time only...")
        temp = path.with_suffix(".part")
        with urllib.request.urlopen(FACE_MODEL_URL, timeout=30) as response:
            temp.write_bytes(response.read())
        temp.replace(path)
        return path
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.warning("face model download failed: %s", exc)
        return None


class FaceBackend(ABC):
    """Interface every face detection backend implements."""

    name = "abstract"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return ``(x, y, w, h)`` boxes in ``frame`` coordinates."""


class MediaPipeFaceBackend(FaceBackend):
    """BlazeFace short-range detector via the MediaPipe Tasks API."""

    name = "mediapipe"

    def __init__(self, min_confidence: float = 0.5) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model = _ensure_face_model()
        if model is None:
            raise RuntimeError("face model unavailable")

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:  # noqa: D102
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        result = self._detector.detect(image)

        boxes: List[Tuple[int, int, int, int]] = []
        for detection in result.detections or []:
            box = detection.bounding_box
            boxes.append((int(box.origin_x), int(box.origin_y),
                          int(box.width), int(box.height)))
        return boxes


class HaarFaceBackend(FaceBackend):
    """Haar cascade detector — only available on OpenCV builds that ship it."""

    name = "haar"

    def __init__(self, min_face_fraction: float = 0.08) -> None:
        import cv2

        self.min_face_fraction = min_face_fraction
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.exists():
            # OpenCV 5.0 removed the bundled cascades; the directory still
            # exists but is empty, so CascadeClassifier would fail opaquely.
            raise RuntimeError(f"cascade file not bundled with OpenCV ({path})")

        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError("cascade failed to load")
        self._cascade = cascade

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:  # noqa: D102
        import cv2

        # Detection is slow at full resolution; half-size greyscale is ample.
        small = cv2.resize(frame, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        grey = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

        min_size = max(int(small.shape[0] * self.min_face_fraction), 24)
        faces = self._cascade.detectMultiScale(
            grey, scaleFactor=1.15, minNeighbors=5,
            minSize=(min_size, min_size),
        )
        return [(int(x * 2), int(y * 2), int(w * 2), int(h * 2))
                for x, y, w, h in faces]


class FaceDetector:
    """Backend-agnostic face detector used for presence sensing."""

    def __init__(self, min_face_fraction: float = 0.08) -> None:
        self.min_face_fraction = min_face_fraction
        self._backend: Optional[FaceBackend] = None
        self._load()

    def _load(self) -> None:
        """Resolve the best available face backend."""
        for factory in (
            lambda: MediaPipeFaceBackend(),
            lambda: HaarFaceBackend(self.min_face_fraction),
        ):
            try:
                self._backend = factory()
                log.info("face detector backend: %s", self._backend.name)
                return
            except Exception as exc:
                log.debug("face backend unavailable: %s", exc)
        log.info("face detection unavailable; presence features disabled")

    @property
    def available(self) -> bool:
        """Whether face detection can run."""
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend.name if self._backend else "none"

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return ``(x, y, w, h)`` boxes for faces in ``frame``."""
        if self._backend is None:
            return []
        try:
            return self._backend.detect(frame)
        except Exception as exc:
            log.debug("face detection failed: %s", exc)
            return []


class PresenceMonitor:
    """Tracks user presence and triggers pause / lock actions."""

    def __init__(self, config: Optional[SecurityConfig] = None,
                 on_absent: Optional[Callable[[], None]] = None,
                 on_present: Optional[Callable[[], None]] = None,
                 on_lock: Optional[Callable[[], None]] = None) -> None:
        self.config = config or SecurityConfig()
        self.on_absent = on_absent
        self.on_present = on_present
        self.on_lock = on_lock

        self.detector = FaceDetector(self.config.min_face_fraction)
        self.status = PresenceStatus()

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._locked_at = 0.0

    @property
    def available(self) -> bool:
        """Whether presence detection can run on this machine."""
        return self.detector.available

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> bool:
        """Start the background detection thread."""
        if not self.config.enabled or not self.available:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True

        self._stop.clear()
        self.status = PresenceStatus(last_seen=time.monotonic())
        self._thread = threading.Thread(target=self._loop, name="presence",
                                        daemon=True)
        self._thread.start()
        log.info("presence monitoring started")
        return True

    def stop(self) -> None:
        """Stop the detection thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("presence monitoring stopped")

    def submit_frame(self, frame: np.ndarray) -> None:
        """Hand the monitor the newest camera frame.

        Only a reference to the latest frame is kept; the worker samples it on
        its own schedule rather than processing every frame.
        """
        if not self.config.enabled:
            return
        with self._lock:
            self._frame = frame

    def _loop(self) -> None:
        """Worker: sample the latest frame at the configured interval."""
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()

            if frame is not None:
                faces = self.detector.detect(frame)
                self._update_state(faces)

            self._stop.wait(max(self.config.detection_interval, 0.25))

    def _update_state(self, faces: List[Tuple[int, int, int, int]]) -> None:
        """Fold a detection result into the presence state machine."""
        now = time.monotonic()
        previous = self.status.state

        self.status.faces = faces
        self.status.face_count = len(faces)

        if faces:
            self.status.last_seen = now
            self.status.absent_for = 0.0
            self.status.state = PresenceState.PRESENT
            if previous in (PresenceState.ABSENT, PresenceState.LOCKED):
                log.info("user returned")
                if self.on_present:
                    self._safe_call(self.on_present)
            return

        self.status.absent_for = now - self.status.last_seen

        if self.status.absent_for >= self.config.absence_threshold:
            if previous != PresenceState.ABSENT and previous != PresenceState.LOCKED:
                self.status.state = PresenceState.ABSENT
                log.info("user absent for %.0fs", self.status.absent_for)
                if self.config.pause_on_absence and self.on_absent:
                    self._safe_call(self.on_absent)

            lock_after = self.config.auto_lock_seconds
            if (lock_after > 0 and self.status.absent_for >= lock_after
                    and self.status.state != PresenceState.LOCKED
                    and now - self._locked_at > 30.0):
                self.status.state = PresenceState.LOCKED
                self._locked_at = now
                log.warning("auto-locking after %.0fs absence", self.status.absent_for)
                if self.on_lock:
                    self._safe_call(self.on_lock)

    @staticmethod
    def _safe_call(callback: Callable[[], None]) -> None:
        """Invoke a callback without letting it kill the monitor thread."""
        try:
            callback()
        except Exception as exc:
            log.warning("presence callback failed: %s", exc)

    # -- rendering -------------------------------------------------------- #

    def draw(self, frame: np.ndarray, colour: Tuple[int, int, int] = (80, 200, 120)) -> np.ndarray:
        """Draw detected face boxes onto a frame (debug view)."""
        if not self.status.faces:
            return frame
        import cv2

        for x, y, w, h in self.status.faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        return frame
