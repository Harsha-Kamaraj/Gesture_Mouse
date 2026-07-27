"""End-to-end pipeline test with a simulated camera and detector.

Exercises the real :class:`~app.GestureMouseApp` wiring — engine, cursor
controller, action registry, history, overlay and shared state — while
substituting the two things that cannot run in CI:

* the **camera + MediaPipe backend**, replaced by a scripted sequence of
  synthetic hands, and
* the **mouse backend**, replaced by a recorder, so the test never actually
  moves the pointer or clicks on the developer's desktop.

Everything between those two boundaries is the production code path.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import config as config_module  # noqa: E402
from detector import DetectionResult, HandLandmarks  # noqa: E402
from synthetic_hand import (  # noqa: E402
    POSE_PRESETS, HandPose, build_landmarks, make_pinch,
)


class RecordingMouse:
    """Stand-in for :class:`~cursor_controller.MouseBackend`.

    Records every call rather than touching the real pointer.  Without this,
    running the test would click on whatever the developer had focused.
    """

    def __init__(self) -> None:
        self.available = True
        self.name = "recording"
        self.moves: List[Tuple[int, int]] = []
        self.clicks: List[Tuple[str, int]] = []
        self.presses: List[str] = []
        self.releases: List[str] = []
        self.scrolls: List[Tuple[int, int]] = []

    def position(self) -> Tuple[int, int]:
        """Last recorded position."""
        return self.moves[-1] if self.moves else (0, 0)

    def move_to(self, x: int, y: int) -> bool:
        """Record a move."""
        self.moves.append((int(x), int(y)))
        return True

    def click(self, button: str = "left", count: int = 1) -> bool:
        """Record a click."""
        self.clicks.append((button, count))
        return True

    def press(self, button: str = "left") -> bool:
        """Record a button press."""
        self.presses.append(button)
        return True

    def release(self, button: str = "left") -> bool:
        """Record a button release."""
        self.releases.append(button)
        return True

    def scroll(self, dx: int, dy: int) -> bool:
        """Record a scroll."""
        self.scrolls.append((int(dx), int(dy)))
        return True


class ScriptedDetector:
    """Replaces :class:`~detector.HandDetector` with a scripted hand sequence."""

    def __init__(self, script: Sequence[Optional[np.ndarray]],
                 width: int = 640, height: int = 480,
                 interval: float = 0.012) -> None:
        self.script = list(script)
        self.width = width
        self.height = height
        # Frames are paced rather than returned instantly. The engine's
        # click/drag/hold logic is all time-based, so a detector that emits a
        # hundred frames in one millisecond would make those thresholds
        # untestable — and would not resemble a real camera.
        self.interval = interval
        self.index = 0
        self.backend_name = "scripted"
        self.exhausted = False

        class _Stream:
            is_running = True
            is_stalled = False
            actual_width = width
            actual_height = height

            def close(self) -> None:
                """No-op."""

        self.stream = _Stream()

    def start(self) -> None:
        """No-op; nothing to open."""

    def stop(self) -> None:
        """No-op; nothing to release."""

    def process(self, block: bool = True) -> Optional[DetectionResult]:
        """Return the next scripted frame's detection result."""
        if self.index >= len(self.script):
            self.exhausted = True
            time.sleep(0.005)
            return None

        landmarks = self.script[self.index]
        self.index += 1
        time.sleep(self.interval)

        frame = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        hands = ([HandLandmarks(landmarks, landmarks.copy(), "Right", 0.95)]
                 if landmarks is not None else [])
        return DetectionResult(frame=frame, hands=hands,
                               timestamp=time.monotonic(), inference_ms=6.0,
                               frame_index=self.index)


def build_app(script: Sequence[Optional[np.ndarray]], tmp: Path):
    """Construct a fully wired app with simulated I/O boundaries."""
    # Redirect every filesystem path into a temp dir before importing app, so
    # the test never touches the developer's real profiles or history.
    config_module.PROFILES_DIR = tmp / "profiles"
    config_module.DATA_DIR = tmp / "data"
    config_module.HISTORY_FILE = tmp / "data" / "history.jsonl"
    config_module.GESTURE_LIBRARY_FILE = tmp / "data" / "gestures.json"
    config_module.APP_STATE_FILE = tmp / "data" / "state.json"
    config_module.SCREENSHOT_DIR = tmp / "screenshots"
    config_module.RECORDING_DIR = tmp / "recordings"
    config_module.PLUGIN_DIR = tmp / "plugins"
    for directory in (config_module.PROFILES_DIR, config_module.DATA_DIR,
                      config_module.SCREENSHOT_DIR, config_module.PLUGIN_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    from app import GestureMouseApp
    from settings import ProfileManager

    args = argparse.Namespace(
        headless=True, stats=False, profile=None, camera=None,
        list_cameras=False, no_cursor=False, debug=False,
    )

    # Point the profile manager at the temp directory.
    original_init = ProfileManager.__init__

    def patched_init(self, directory=config_module.PROFILES_DIR):  # type: ignore[no-untyped-def]
        original_init(self, directory)

    ProfileManager.__init__ = patched_init  # type: ignore[method-assign]
    try:
        app = GestureMouseApp(args)
    finally:
        ProfileManager.__init__ = original_init  # type: ignore[method-assign]

    app.detector = ScriptedDetector(script)
    mouse = RecordingMouse()
    app.cursor.mouse = mouse
    app.sounds.enabled = False
    app.config.features.plugins_enabled = False
    app.config.features.face_unlock = False
    app.config.features.voice_commands = False
    # Deterministic gating for a short scripted run.
    app.config.gestures.stability_frames = 1
    app.config.gestures.global_cooldown = 0.0
    app.engine.apply_config(app.config.gestures)
    return app, mouse


def run_app(app, mouse, max_seconds: float = 6.0) -> None:
    """Start the app and let it drain the script."""
    assert app.start(), "app failed to start"
    deadline = time.monotonic() + max_seconds
    while not app.detector.exhausted and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.15)  # let the last frames finish processing
    app.stop()


# --------------------------------------------------------------------------- #
# Scripts
# --------------------------------------------------------------------------- #

def script_cursor_movement(frames: int = 40) -> List[np.ndarray]:
    """A pointing hand sweeping across the frame."""
    preset = POSE_PRESETS["Point"]
    script = []
    for i in range(frames):
        pose = HandPose(**{**preset.__dict__,
                           "centre": (0.30 + 0.010 * i, 0.55)})
        script.append(build_landmarks(pose))
    return script


def script_click() -> List[np.ndarray]:
    """Point, pinch closed, release — one click."""
    apart = make_pinch(POSE_PRESETS["Point"], closed=False)
    closed = make_pinch(POSE_PRESETS["Point"], closed=True)
    return [apart] * 6 + [closed] * 3 + [apart] * 6


def script_scroll(frames: int = 14) -> List[np.ndarray]:
    """A peace sign moving upward — scroll."""
    preset = POSE_PRESETS["Peace"]
    return [build_landmarks(HandPose(**{**preset.__dict__,
                                        "centre": (0.5, 0.70 - 0.012 * i)}))
            for i in range(frames)]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_pipeline_moves_cursor() -> None:
    """A pointing hand must drive real cursor output through the full stack."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_cursor_movement(), Path(tmpdir))
        run_app(app, mouse)

        assert len(mouse.moves) >= 10, \
            f"expected cursor movement, got {len(mouse.moves)} moves"

        xs = [x for x, _ in mouse.moves]
        assert xs[-1] > xs[0], "cursor did not travel in the hand's direction"

        # Movement should be monotonic-ish, not jumping around.
        backtracks = sum(1 for a, b in zip(xs, xs[1:]) if b < a - 5)
        assert backtracks <= len(xs) * 0.2, \
            f"cursor path unstable: {backtracks} backtracks in {len(xs)} moves"


def test_pipeline_click_reaches_mouse() -> None:
    """A pinch must travel gesture -> engine -> action -> mouse backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_click(), Path(tmpdir))
        run_app(app, mouse)

        assert mouse.clicks, "no click reached the mouse backend"
        assert mouse.clicks[0][0] == "left", f"wrong button: {mouse.clicks}"


def test_pipeline_records_history() -> None:
    """Executed gestures must land in the history log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_click(), Path(tmpdir))
        run_app(app, mouse)

        entries = app.history.entries
        assert entries, "history recorded nothing"
        assert any(e.gesture.startswith("pinch") for e in entries), \
            f"no pinch in history: {[e.gesture for e in entries]}"

        summary = app.history.summary()
        assert summary["total"] >= 1
        assert 0.0 < float(summary["mean_confidence"]) <= 1.0


def test_pipeline_scroll() -> None:
    """A scroll pose with vertical travel must reach the mouse backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_scroll(), Path(tmpdir))
        run_app(app, mouse)

        assert mouse.scrolls, "no scroll reached the mouse backend"


def test_pipeline_renders_overlay() -> None:
    """The shared state must expose an annotated frame for the UI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_cursor_movement(20), Path(tmpdir))
        run_app(app, mouse)

        frame = app.state.frame
        assert frame is not None, "no frame published to shared state"
        assert frame.shape == (480, 640, 3)
        # A blank 32-grey frame plus an overlay must contain brighter pixels.
        assert int(frame.max()) > 100, "overlay does not appear to have drawn"


def test_engine_state_bundle() -> None:
    """The UI-facing state bundle must be populated and well formed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_cursor_movement(20), Path(tmpdir))
        run_app(app, mouse)

        state = app.get_engine_state()
        for key in ("tracking", "paused", "mode", "pose", "confidence",
                    "clicks", "gestures", "session", "latency", "backend"):
            assert key in state, f"missing key {key!r} in engine state"
        assert isinstance(state["latency"], dict)


def test_performance_metrics_collected() -> None:
    """Frame timings must be recorded during a run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_cursor_movement(30), Path(tmpdir))
        run_app(app, mouse)

        snapshot = app.performance.snapshot()
        assert app.performance.total_frames >= 10, \
            f"only {app.performance.total_frames} frames counted"
        assert snapshot.frame_ms > 0, "no frame time recorded"


def test_no_hand_releases_drag() -> None:
    """Losing the hand mid-drag must release the button on the real backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        closed = make_pinch(POSE_PRESETS["Point"], closed=True)
        apart = make_pinch(POSE_PRESETS["Point"], closed=False)
        # Hold the pinch long enough to become a drag, then drop the hand.
        script = [apart] * 3 + [closed] * 40 + [None] * 40

        app, mouse = build_app(script, Path(tmpdir))
        app.config.gestures.drag_hold_time = 0.05
        app.config.gestures.tracking_lost_timeout = 0.05
        app.engine.apply_config(app.config.gestures)
        run_app(app, mouse)

        assert mouse.presses, "drag never pressed the button"
        assert mouse.releases, "drag was never released — button left held"


def test_shutdown_is_clean() -> None:
    """Stopping the app must release input and leave no live threads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app, mouse = build_app(script_cursor_movement(15), Path(tmpdir))
        run_app(app, mouse)

        assert app._thread is None or not app._thread.is_alive(), \
            "pipeline thread still running after stop()"
        # emergency_release fires on shutdown, clearing all three buttons.
        assert len(mouse.releases) >= 3, \
            f"shutdown did not release all buttons: {mouse.releases}"


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
                import traceback
                print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
