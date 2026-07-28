"""Camera capture and MediaPipe hand landmark detection.

Two concerns live here, deliberately separated:

``CameraStream``
    A threaded frame grabber implementing *latest-frame-wins* semantics.  The
    naive ``cap.read()`` in the render loop couples your frame rate to the
    camera's blocking I/O and builds a latency backlog: OpenCV buffers frames
    internally, so a slow consumer ends up processing images from seconds ago.
    Reading on a dedicated thread and keeping only the newest frame bounds
    end-to-end latency to a single frame interval.

``HandDetector``
    A backend-agnostic wrapper over MediaPipe Hands.  Google ships two
    incompatible APIs and which one you get depends on your Python version:

    * the **legacy Solutions API** (``mediapipe.solutions.hands``) — bundled
      model, available on Python ≤3.12;
    * the **Tasks API** (``mediapipe.tasks.python.vision.HandLandmarker``) —
      the current generation, requires an external ``.task`` model file, and
      is the *only* API present on newer Python builds.

    Rather than pinning users to one interpreter, we detect what's installed
    and adapt.  Both backends emit the same :class:`HandLandmarks` value
    object, so nothing downstream knows or cares which one is active.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import ASSETS_DIR, CameraConfig, DetectionConfig
from logger import get_logger
from utils import distance_2d

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Landmark index constants (MediaPipe hand topology)
# --------------------------------------------------------------------------- #

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

#: Landmark pairs forming the drawn hand skeleton.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm base
)

MODEL_DIR = ASSETS_DIR / "models"
MODEL_FILE = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #

@dataclass
class HandLandmarks:
    """One detected hand, normalized to the frame.

    Attributes:
        points: ``(21, 3)`` array of ``(x, y, z)`` in ``[0, 1]`` image space
            (``z`` is relative depth, negative = closer to the camera).
        world_points: ``(21, 3)`` metric coordinates in metres relative to the
            hand's geometric centre.  Scale-invariant, so it is what we use
            for size-independent gesture measurements.
        handedness: ``"Left"`` or ``"Right"`` from the camera's point of view
            *after* mirroring has been accounted for.
        score: Detector confidence in ``[0, 1]``.
    """

    points: np.ndarray
    world_points: np.ndarray
    handedness: str
    score: float

    def point(self, index: int) -> np.ndarray:
        """Return landmark ``index`` as an ``(x, y, z)`` array."""
        return self.points[index]

    def xy(self, index: int) -> Tuple[float, float]:
        """Return landmark ``index`` as a normalized ``(x, y)`` tuple."""
        return (float(self.points[index][0]), float(self.points[index][1]))

    def pixel(self, index: int, width: int, height: int) -> Tuple[int, int]:
        """Return landmark ``index`` in pixel coordinates."""
        return (int(self.points[index][0] * width),
                int(self.points[index][1] * height))

    @property
    def wrist(self) -> np.ndarray:
        """Wrist landmark."""
        return self.points[WRIST]

    @property
    def palm_centre(self) -> Tuple[float, float]:
        """Approximate palm centre from the wrist and the four MCP joints."""
        idx = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        return (float(np.mean(self.points[idx, 0])),
                float(np.mean(self.points[idx, 1])))

    @property
    def palm_size(self) -> float:
        """Wrist→middle-MCP distance — the natural per-hand scale unit.

        Every distance threshold in the gesture engine is expressed as a
        fraction of this value so that recognition is invariant to hand size
        and to how far the user sits from the camera.
        """
        return max(distance_2d(self.points[WRIST], self.points[MIDDLE_MCP]), 1e-6)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Normalized ``(x_min, y_min, x_max, y_max)`` around the hand."""
        xs, ys = self.points[:, 0], self.points[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    def normalized_distance(self, a: int, b: int) -> float:
        """Distance between two landmarks, scaled by :attr:`palm_size`.

        This is *the* measurement used for pinch detection.
        """
        return distance_2d(self.points[a], self.points[b]) / self.palm_size


@dataclass
class DetectionResult:
    """Everything produced from a single camera frame."""

    frame: np.ndarray
    hands: List[HandLandmarks] = field(default_factory=list)
    timestamp: float = 0.0
    #: Wall-clock milliseconds spent inside the detector.
    inference_ms: float = 0.0
    frame_index: int = 0

    @property
    def has_hands(self) -> bool:
        """True when at least one hand was detected."""
        return bool(self.hands)

    def hand_by_side(self, side: str) -> Optional[HandLandmarks]:
        """Return the highest-scoring hand matching ``side``.

        ``side`` may be ``"Left"``, ``"Right"`` or ``"Any"``.
        """
        if side == "Any":
            return max(self.hands, key=lambda h: h.score) if self.hands else None
        matches = [h for h in self.hands if h.handedness == side]
        return max(matches, key=lambda h: h.score) if matches else None


# --------------------------------------------------------------------------- #
# Detection backends
# --------------------------------------------------------------------------- #

class DetectorBackend(ABC):
    """Interface every MediaPipe backend implements."""

    name: str = "abstract"

    @abstractmethod
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:
        """Run inference on an RGB image and return the detected hands."""

    def close(self) -> None:
        """Release native resources.  Safe to call more than once."""


def _to_arrays(landmarks: Sequence[object]) -> np.ndarray:
    """Convert a MediaPipe landmark list into an ``(N, 3)`` float array."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],  # type: ignore[attr-defined]
        dtype=np.float32,
    )


class LegacySolutionsBackend(DetectorBackend):
    """Adapter for ``mediapipe.solutions.hands`` (Python ≤ 3.12)."""

    name = "solutions"

    def __init__(self, cfg: DetectionConfig) -> None:
        import mediapipe as mp  # local import: keeps module import cheap

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        log.info("detector backend: legacy solutions API")

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:  # noqa: D102
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        hands: List[HandLandmarks] = []
        world = result.multi_hand_world_landmarks or [None] * len(result.multi_hand_landmarks)
        handedness = result.multi_handedness or []

        for i, landmarks in enumerate(result.multi_hand_landmarks):
            points = _to_arrays(landmarks.landmark)
            world_points = (
                _to_arrays(world[i].landmark) if i < len(world) and world[i] is not None
                else points.copy()
            )
            label, score = "Right", 1.0
            if i < len(handedness) and handedness[i].classification:
                label = handedness[i].classification[0].label
                score = float(handedness[i].classification[0].score)
            hands.append(HandLandmarks(points, world_points, label, score))
        return hands

    def close(self) -> None:  # noqa: D102
        try:
            self._hands.close()
        except Exception:  # pragma: no cover - native teardown
            pass


class TasksBackend(DetectorBackend):
    """Adapter for ``mediapipe.tasks.python.vision.HandLandmarker``.

    The Tasks API needs an external model bundle.  We download it once into
    ``assets/models/`` and cache it; subsequent runs are offline.
    """

    name = "tasks"

    def __init__(self, cfg: DetectionConfig) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = ensure_model_downloaded()
        if model_path is None:
            raise RuntimeError("hand_landmarker.task model unavailable")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        # The VIDEO running mode demands strictly increasing timestamps.
        self._last_ts = -1
        log.info("detector backend: tasks API (%s)", model_path.name)

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:  # noqa: D102
        import mediapipe as mp

        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return []

        hands: List[HandLandmarks] = []
        for i, landmarks in enumerate(result.hand_landmarks):
            points = _to_arrays(landmarks)
            world_points = (
                _to_arrays(result.hand_world_landmarks[i])
                if result.hand_world_landmarks and i < len(result.hand_world_landmarks)
                else points.copy()
            )
            label, score = "Right", 1.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                category = result.handedness[i][0]
                label = category.category_name
                score = float(category.score)
            hands.append(HandLandmarks(points, world_points, label, score))
        return hands

    def close(self) -> None:  # noqa: D102
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover - native teardown
            pass


def ensure_model_downloaded(timeout: float = 30.0) -> Optional[Path]:
    """Return the path to the hand landmarker model, downloading if needed."""
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 1024:
        return MODEL_FILE

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmark model (~7.5 MB), one time only...")
    try:
        temp = MODEL_FILE.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as response:
            temp.write_bytes(response.read())
        temp.replace(MODEL_FILE)
        log.info("model saved to %s", MODEL_FILE)
        return MODEL_FILE
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.error("model download failed: %s", exc)
        return None


def create_backend(cfg: DetectionConfig) -> DetectorBackend:
    """Instantiate the best MediaPipe backend available in this environment.

    Preference order is legacy-first because the Solutions API bundles its
    model and therefore works offline out of the box; the Tasks API is used
    when Solutions is absent (newer Python builds ship Tasks only).
    """
    errors: Dict[str, str] = {}

    for factory in (LegacySolutionsBackend, TasksBackend):
        try:
            return factory(cfg)
        except Exception as exc:
            errors[factory.name] = str(exc)
            log.debug("backend %s unavailable: %s", factory.name, exc)

    raise RuntimeError(
        "No usable MediaPipe backend. Tried: "
        + "; ".join(f"{k} ({v})" for k, v in errors.items())
    )


# --------------------------------------------------------------------------- #
# Threaded camera
# --------------------------------------------------------------------------- #

class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or has died."""


class CameraStream:
    """Background webcam reader with latest-frame-wins semantics."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_time: float = 0.0
        self._frame_index: int = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.read_failures: int = 0
        #: Seconds to pace the reader at; 0 means free-run at the device rate.
        #: Raised by the pipeline while no hand is being tracked, so an idle
        #: application stops decoding frames nothing will look at.
        self.idle_interval: float = 0.0

    # -- lifecycle -------------------------------------------------------- #

    def open(self) -> None:
        """Open the device and start the reader thread.

        Raises:
            CameraError: if the device cannot be opened or yields no frames.
        """
        if self._running.is_set():
            return

        capture = cv2.VideoCapture(self.cfg.device_index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Cannot open camera index {self.cfg.device_index}. "
                "Is another application using it, or is camera permission denied?"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        capture.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)
        # A 1-frame driver buffer is the other half of the latency fix.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:  # pragma: no cover - unsupported on some backends
            pass

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise CameraError("Camera opened but returned no frames.")

        self.actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
        self.actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        self._capture = capture
        self._frame = frame
        self._frame_time = time.monotonic()
        self.read_failures = 0

        self._running.set()
        self._thread = threading.Thread(target=self._reader, name="camera", daemon=True)
        self._thread.start()
        log.info(
            "camera %d opened at %dx%d",
            self.cfg.device_index, self.actual_width, self.actual_height,
        )

    def _reader(self) -> None:
        """Reader loop: pull frames as fast as the device allows.

        When :attr:`idle_interval` is set the loop paces itself instead.
        Decoding a frame and converting its colour space is not free — it
        measured as the single most expensive stage of the whole pipeline —
        so capturing at 30 fps to feed a consumer that is only sampling at 8
        is most of a CPU core spent on frames nobody looks at.
        """
        while self._running.is_set() and self._capture is not None:
            started = time.monotonic()
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self.read_failures += 1
                if self.read_failures > 30:
                    log.error("camera stalled after %d failed reads", self.read_failures)
                    self._running.clear()
                    break
                time.sleep(0.01)
                continue

            self.read_failures = 0
            with self._new_frame:
                self._frame = frame
                self._frame_time = time.monotonic()
                self._frame_index += 1
                self._new_frame.notify_all()

            interval = self.idle_interval
            if interval > 0.0:
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)

    def read(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Return ``(frame_copy, timestamp, index)`` for the newest frame."""
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame.copy(), self._frame_time, self._frame_index

    def wait_for_frame(self, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], float, int]:
        """Block until a frame newer than the last one read arrives."""
        with self._new_frame:
            last = self._frame_index
            if not self._new_frame.wait_for(
                lambda: self._frame_index != last or not self._running.is_set(),
                timeout=timeout,
            ):
                return None, 0.0, last
            if self._frame is None:
                return None, 0.0, self._frame_index
            return self._frame.copy(), self._frame_time, self._frame_index

    @property
    def is_running(self) -> bool:
        """True while the reader thread is alive and healthy."""
        return self._running.is_set()

    @property
    def is_stalled(self) -> bool:
        """True when no frame has arrived within the configured timeout."""
        if not self._running.is_set():
            return True
        return (time.monotonic() - self._frame_time) > self.cfg.stall_timeout

    def close(self) -> None:
        """Stop the reader thread and release the device."""
        self._running.clear()
        with self._new_frame:
            self._new_frame.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        log.info("camera closed")

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def enumerate_devices(max_index: int = 5) -> List[int]:
        """Probe for usable camera indices.

        Note this genuinely opens each device, so it is slow (~100 ms each)
        and should only be called from the settings screen, never per-frame.
        """
        found: List[int] = []
        for index in range(max_index):
            capture = cv2.VideoCapture(index)
            try:
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        found.append(index)
            finally:
                capture.release()
        return found


# --------------------------------------------------------------------------- #
# Detector facade
# --------------------------------------------------------------------------- #

class HandDetector:
    """Owns the camera and the MediaPipe backend; produces detection results."""

    def __init__(self, camera_cfg: CameraConfig, detection_cfg: DetectionConfig) -> None:
        self.camera_cfg = camera_cfg
        self.detection_cfg = detection_cfg
        self.stream = CameraStream(camera_cfg)
        self._backend: DetectorBackend | None = None
        self._start_time = time.monotonic()
        self._handedness_counts: Dict[str, int] = {"Left": 0, "Right": 0}

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        """Open the camera and initialise the inference backend."""
        self.stream.open()
        if self._backend is None:
            self._backend = create_backend(self.detection_cfg)

    def stop(self) -> None:
        """Tear down camera and backend."""
        self.stream.close()
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    @property
    def backend_name(self) -> str:
        """Name of the active backend (``solutions``/``tasks``/``none``)."""
        return self._backend.name if self._backend else "none"

    def reconfigure(self, detection_cfg: DetectionConfig) -> None:
        """Rebuild the backend after detection settings change."""
        self.detection_cfg = detection_cfg
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._backend = create_backend(detection_cfg)

    # -- inference -------------------------------------------------------- #

    def process(self, block: bool = True) -> Optional[DetectionResult]:
        """Grab the newest frame and run hand detection on it.

        Returns ``None`` when no frame is currently available (camera warming
        up or momentarily stalled) — callers should treat that as "skip this
        iteration", not as an error.
        """
        if self._backend is None:
            return None

        frame, timestamp, index = (
            self.stream.wait_for_frame(timeout=0.5) if block else self.stream.read()
        )
        if frame is None:
            return None

        if self.camera_cfg.mirror:
            frame = cv2.flip(frame, 1)

        start = time.perf_counter()
        hands = self._infer(frame)
        inference_ms = (time.perf_counter() - start) * 1000.0

        for hand in hands:
            self._handedness_counts[hand.handedness] = (
                self._handedness_counts.get(hand.handedness, 0) + 1
            )

        return DetectionResult(
            frame=frame,
            hands=hands,
            timestamp=timestamp,
            inference_ms=inference_ms,
            frame_index=index,
        )

    def _infer(self, frame: np.ndarray) -> List[HandLandmarks]:
        """Run the backend on ``frame``, downscaling first if configured."""
        assert self._backend is not None

        scale = self.camera_cfg.inference_scale
        if 0.2 <= scale < 0.999:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)

        try:
            hands = self._backend.detect(rgb, timestamp_ms)
        except Exception as exc:
            log.warning("inference failed: %s", exc)
            return []

        # Landmarks come back normalized, so downscaling needs no correction.
        if self.camera_cfg.mirror:
            # The frame was flipped before inference, so MediaPipe's left/right
            # labels are inverted relative to the user's actual hands.
            for hand in hands:
                hand.handedness = "Left" if hand.handedness == "Right" else "Right"
        return hands

    @property
    def dominant_hand(self) -> str:
        """The hand seen most often so far — used for auto hand-dominance."""
        if not any(self._handedness_counts.values()):
            return self.detection_cfg.primary_hand
        return max(self._handedness_counts, key=lambda k: self._handedness_counts[k])

    def reset_dominance(self) -> None:
        """Clear accumulated handedness statistics."""
        self._handedness_counts = {"Left": 0, "Right": 0}
