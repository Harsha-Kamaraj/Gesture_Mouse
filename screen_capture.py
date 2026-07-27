"""Screenshot capture and screen recording.

Both services are intentionally decoupled from the gesture layer: they expose
plain ``capture()`` / ``start()`` / ``stop()`` methods that the action
registry calls, and they know nothing about hands.

Screen recording runs on a worker thread with a bounded frame budget.  Encoding
video is far too slow to do inline — a single 1080p frame encode can cost more
than a whole gesture frame's time budget — so the recorder grabs on its own
clock and never blocks the recognition loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from config import RECORDING_DIR, SCREENSHOT_DIR
from logger import get_logger

log = get_logger(__name__)


def _timestamp() -> str:
    """Filename-safe timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _grab_screen() -> Optional[np.ndarray]:
    """Capture the whole desktop as a BGR array, or ``None`` on failure."""
    try:
        import pyautogui  # type: ignore[import-not-found]
        import cv2

        shot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        log.warning("screen grab failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Screenshots
# --------------------------------------------------------------------------- #

@dataclass
class CaptureResult:
    """Outcome of a screenshot request."""

    success: bool
    path: Optional[Path] = None
    error: str = ""


class ScreenshotService:
    """Saves timestamped screenshots to ``screenshots/``."""

    def __init__(self, directory: Path = SCREENSHOT_DIR) -> None:
        self.directory = directory
        self.count = 0

    def capture(self, region: Optional[Tuple[int, int, int, int]] = None,
                prefix: str = "shot") -> CaptureResult:
        """Capture the screen (or a region) and write it to disk.

        Args:
            region: Optional ``(left, top, width, height)`` crop.
            prefix: Filename prefix.

        Returns:
            A :class:`CaptureResult` with the saved path on success.
        """
        try:
            import cv2
            import pyautogui  # type: ignore[import-not-found]

            self.directory.mkdir(parents=True, exist_ok=True)
            shot = pyautogui.screenshot(region=region)
            frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

            path = self.directory / f"{prefix}_{_timestamp()}.png"
            if not cv2.imwrite(str(path), frame):
                return CaptureResult(False, error="imwrite failed")

            self.count += 1
            log.info("screenshot saved: %s", path.name)
            return CaptureResult(True, path=path)
        except Exception as exc:
            log.error("screenshot failed: %s", exc)
            return CaptureResult(False, error=str(exc))

    def list_recent(self, limit: int = 20) -> List[Path]:
        """Return the most recently saved screenshots, newest first."""
        if not self.directory.exists():
            return []
        files = sorted(
            self.directory.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files[:limit]


# --------------------------------------------------------------------------- #
# Screen recording
# --------------------------------------------------------------------------- #

class ScreenRecorder:
    """Threaded desktop recorder writing MP4 (falling back to AVI)."""

    def __init__(self, directory: Path = RECORDING_DIR, fps: int = 15) -> None:
        self.directory = directory
        self.fps = fps
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._writer = None
        self._path: Optional[Path] = None
        self._started_at = 0.0
        self._frames = 0
        self._lock = threading.Lock()
        self.on_state_change: Optional[Callable[[bool], None]] = None

    @property
    def is_recording(self) -> bool:
        """Whether a recording is currently in progress."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def duration(self) -> float:
        """Seconds elapsed in the current recording."""
        return time.monotonic() - self._started_at if self.is_recording else 0.0

    @property
    def output_path(self) -> Optional[Path]:
        """Path of the current or most recent recording."""
        return self._path

    def start(self) -> bool:
        """Begin recording.  Returns False if already running or unavailable."""
        with self._lock:
            if self.is_recording:
                return False

            frame = _grab_screen()
            if frame is None:
                log.error("cannot start recording: screen capture unavailable")
                return False

            try:
                import cv2

                self.directory.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                # Even dimensions are required by most H.264 encoders.
                width -= width % 2
                height -= height % 2

                path = self.directory / f"recording_{_timestamp()}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps, (width, height),
                )
                if not writer.isOpened():
                    path = path.with_suffix(".avi")
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                        self.fps, (width, height),
                    )
                if not writer.isOpened():
                    log.error("no usable video encoder found")
                    return False

                self._writer = writer
                self._path = path
                self._size = (width, height)
                self._frames = 0
                self._started_at = time.monotonic()
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._record_loop, name="recorder", daemon=True)
                self._thread.start()
                log.info("recording started: %s", path.name)
                if self.on_state_change:
                    self.on_state_change(True)
                return True
            except Exception as exc:
                log.error("failed to start recording: %s", exc)
                return False

    def _record_loop(self) -> None:
        """Worker: grab and encode at a fixed cadence."""
        import cv2

        interval = 1.0 / max(self.fps, 1)
        next_frame = time.monotonic()

        while not self._stop.is_set():
            frame = _grab_screen()
            if frame is not None and self._writer is not None:
                try:
                    if (frame.shape[1], frame.shape[0]) != self._size:
                        frame = cv2.resize(frame, self._size)
                    self._writer.write(frame)
                    self._frames += 1
                except Exception as exc:
                    log.warning("frame write failed: %s", exc)

            # Fixed-cadence sleep that absorbs encode jitter rather than
            # letting it accumulate into drift.
            next_frame += interval
            delay = next_frame - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_frame = time.monotonic()

    def stop(self) -> Optional[Path]:
        """Stop recording and finalise the file."""
        with self._lock:
            if not self.is_recording:
                return None

            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._thread = None

            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception as exc:
                    log.warning("writer release failed: %s", exc)
                self._writer = None

            log.info("recording stopped: %s (%d frames)",
                     self._path.name if self._path else "?", self._frames)
            if self.on_state_change:
                self.on_state_change(False)
            return self._path

    def toggle(self) -> bool:
        """Start if stopped, stop if started.  Returns the new state."""
        if self.is_recording:
            self.stop()
            return False
        return self.start()
