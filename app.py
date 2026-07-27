"""AI Gesture Mouse Pro — application entry point and orchestrator.

Threading model
---------------
Three threads, with a deliberately narrow contract between them::

    [camera thread]  ---latest frame--->  [processing thread]  ---queue--->  [Tk main thread]
     detector.py                           this module                        ui.py

* The **camera thread** (inside :class:`~detector.CameraStream`) only decodes
  frames and keeps the newest one.  It never blocks on a consumer.
* The **processing thread** owns the whole recognition pipeline: inference,
  gesture recognition, cursor movement and action dispatch.  Keeping actions
  here rather than on the UI thread means a slow action (launching an app,
  taking a screenshot) delays the next gesture, never the interface.
* The **Tk main thread** owns every widget.  It pulls the latest annotated
  frame and state; it never reaches into the pipeline.

Shared state is exchanged through a single mutex-guarded snapshot object
rather than by passing widget references around, which is what keeps the UI
optional — ``--headless`` runs the identical pipeline with no Tk at all.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import compat  # noqa: F401  # must precede any customtkinter import
from actions import ActionContext, ActionRegistry, MacroRecorder
from calibration import CalibrationWizard, apply_calibration
from config import (
    APP_NAME, APP_VERSION, AppConfig, PLATFORM_NAME, ensure_directories,
)
from cursor_controller import CursorController, MonitorLayout
from detector import CameraError, HandDetector
from gesture_engine import GestureEngine, GestureEvent, Mode, Pose
from gesture_recorder import GestureLibrary, GestureRecorder
from history import GestureHistory
from logger import get_logger, install_excepthook, setup_logging
from notifications import NullNotifier
from overlay import HUDState, OverlayRenderer
from performance import PerformanceMonitor
from platform_bridge import PlatformBridge
from plugins import PluginManager
from presentation import PresentationMode
from screen_capture import ScreenRecorder, ScreenshotService
from security import PresenceMonitor, SecurityConfig
from settings import ProfileManager
from sounds import SoundPlayer
from themes import ThemeManager
from utils import format_duration
from voice import VoiceEngine
from whiteboard import Whiteboard

log = get_logger(__name__)


@dataclass
class SharedState:
    """Snapshot of pipeline state, read by the UI under a lock."""

    frame: Optional[np.ndarray] = None
    tracking: bool = False
    paused: bool = False
    mode: str = Mode.NAVIGATE.value
    pose: str = Pose.UNKNOWN.value
    confidence: float = 0.0
    hand_count: int = 0
    clicks: int = 0
    scrolls: int = 0
    gestures: int = 0
    session_start: float = field(default_factory=time.time)
    camera_error: str = ""
    level: Optional[Tuple[str, float]] = None
    level_until: float = 0.0


class GestureMouseApp:
    """Owns every subsystem and the processing loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        ensure_directories()

        # -- configuration ------------------------------------------------ #
        self.profiles = ProfileManager()
        self.config: AppConfig = self.profiles.load_startup_profile()
        if args.profile:
            try:
                self.config = self.profiles.load(args.profile)
            except Exception as exc:
                log.warning("could not load profile %r: %s", args.profile, exc)

        self.themes = ThemeManager()
        self.themes.set_theme(self.config.ui.theme)
        self.themes.set_high_contrast(self.config.ui.high_contrast)

        # -- pipeline ----------------------------------------------------- #
        self.detector = HandDetector(self.config.camera, self.config.detection)
        self.library = GestureLibrary()
        self.engine = GestureEngine(self.config.gestures, self.library.recognizer)
        self.monitors = MonitorLayout()
        self.cursor = CursorController(self.config.cursor, self.monitors)
        self.performance = PerformanceMonitor()
        self.history = GestureHistory()
        self.overlay = OverlayRenderer(self.themes.theme)

        # -- services ----------------------------------------------------- #
        self.platform = PlatformBridge()
        self.sounds = SoundPlayer(enabled=self.config.features.sound_effects)
        self.screenshots = ScreenshotService()
        self.recorder = ScreenRecorder()
        self.whiteboard = Whiteboard()
        self.presentation = PresentationMode(on_action=self._presentation_action)
        self.recorder_wizard = GestureRecorder(library=self.library)
        self.calibration = CalibrationWizard(on_complete=self._calibration_complete)
        self.voice = VoiceEngine(on_command=self._voice_command)

        self.presence = PresenceMonitor(
            SecurityConfig(
                enabled=self.config.features.face_unlock,
                auto_lock_seconds=self.config.features.auto_lock_seconds,
            ),
            on_absent=self._on_user_absent,
            on_present=self._on_user_present,
            on_lock=lambda: self.platform.system.lock_screen(),
        )

        self.notifier: Any = NullNotifier()
        self.context = ActionContext(
            cursor=self.cursor, platform=self.platform, sounds=self.sounds,
            screenshots=self.screenshots, recorder=self.recorder,
            notifier=self.notifier, hooks=self._build_hooks(),
        )
        self.actions = ActionRegistry(self.context)
        self.macros = MacroRecorder(self.actions)
        self.plugins = PluginManager(self.actions)

        # -- runtime state ------------------------------------------------ #
        self.state = SharedState()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.window: Any = None
        self._hotkey_listener: Any = None
        self._saved_gesture_cfg: Dict[str, float] = {}

        self.profiles.subscribe(self._on_config_changed)

    # -- hooks ------------------------------------------------------------ #

    def _build_hooks(self) -> Dict[str, Any]:
        """Late-bound callbacks the action layer invokes."""
        return {
            "toggle_whiteboard": self._toggle_whiteboard,
            "toggle_presentation": self._toggle_presentation,
            "toggle_sleep": self._toggle_sleep,
            "wake_tracking": self._wake,
            "toggle_precision": self._toggle_precision,
            "app_launcher": lambda: self.platform.system.launch_app("browser"),
            "show_volume": lambda level: self._show_level("Volume", level),
            "show_brightness": lambda level: self._show_level("Brightness", level),
        }

    def _show_level(self, label: str, level: float) -> None:
        """Display the on-camera level meter for a couple of seconds."""
        with self._state_lock:
            self.state.level = (label, float(level))
            self.state.level_until = time.monotonic() + 1.6

    def _toggle_whiteboard(self) -> str:
        """Show or hide the whiteboard."""
        active = self.whiteboard.toggle()
        return "Opened" if active else "Closed"

    def _toggle_presentation(self) -> str:
        """Enter or leave presentation mode.

        Presentation mode temporarily raises cooldowns and the confidence
        floor.  The previous values are saved and restored on exit so the
        user's profile is never silently rewritten by entering a mode.
        """
        active = self.presentation.toggle()
        if active:
            self._saved_gesture_cfg = {
                "global_cooldown": self.config.gestures.global_cooldown,
                "min_confidence": self.config.gestures.min_confidence,
            }
            for key, value in self.presentation.apply_to_gesture_config(
                    self.config.gestures).items():
                setattr(self.config.gestures, key, value)
        elif self._saved_gesture_cfg:
            for key, value in self._saved_gesture_cfg.items():
                setattr(self.config.gestures, key, value)
            self._saved_gesture_cfg = {}

        self.engine.apply_config(self.config.gestures)
        return "Started" if active else "Ended"

    def _toggle_sleep(self) -> str:
        """Pause gesture tracking."""
        self.engine.sleeping = True
        self.cursor.end_drag()
        return "Tracking paused"

    def _wake(self) -> str:
        """Resume gesture tracking."""
        self.engine.sleeping = False
        return "Tracking resumed"

    def _toggle_precision(self) -> str:
        """Toggle precision cursor mode."""
        self.engine.precision_mode = not self.engine.precision_mode
        return "On" if self.engine.precision_mode else "Off"

    def _presentation_action(self, action: str) -> None:
        """Route a presentation-mode action through the registry."""
        self.actions.execute(GestureEvent(name="presentation", action=action,
                                          confidence=1.0))

    def _voice_command(self, action: str, phrase: str) -> None:
        """Execute a recognised voice command."""
        log.info("voice: %r -> %s", phrase, action)
        self.actions.execute(GestureEvent(name=f"voice:{phrase}", action=action,
                                          confidence=1.0))

    def _on_user_absent(self) -> None:
        """Pause tracking when the user leaves the camera's view."""
        self.engine.pause()
        self.cursor.emergency_release()
        self.notify("User absent", "Tracking paused", "warning")

    def _on_user_present(self) -> None:
        """Resume tracking when the user comes back."""
        self.engine.resume()
        self.notify("Welcome back", "Tracking resumed", "success")

    def _calibration_complete(self, result: Any) -> None:
        """Apply calibration results to the active profile."""
        apply_calibration(self.config, result)
        self.profiles.save(self.config)
        self.engine.apply_config(self.config.gestures)
        self.cursor.apply_config(self.config.cursor)
        self.sounds.play("calibration")
        self.notify("Calibration complete",
                    " · ".join(result.summary_lines()[:2]), "success")

    def _on_config_changed(self, config: AppConfig) -> None:
        """Re-read settings across every subsystem after a profile change."""
        self.config = config
        self.engine.apply_config(config.gestures)
        self.cursor.apply_config(config.cursor)
        self.sounds.set_enabled(config.features.sound_effects)
        self.overlay.show_landmarks = config.ui.show_landmarks
        self.overlay.show_skeleton = config.ui.show_skeleton
        self.overlay.show_panel = config.ui.show_overlay_panel

        theme = self.themes.set_theme(config.ui.theme)
        theme = self.themes.set_high_contrast(config.ui.high_contrast)
        self.overlay.set_theme(theme)

    # -- notifications ---------------------------------------------------- #

    def notify(self, title: str, message: str = "", level: str = "info") -> None:
        """Show a notification through whichever notifier is active."""
        try:
            self.notifier.notify(title, message, level)
        except Exception as exc:
            log.debug("notify failed: %s", exc)

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> bool:
        """Start every subsystem and the processing thread."""
        log.info("%s %s starting on %s", APP_NAME, APP_VERSION, PLATFORM_NAME)

        self.sounds.initialise()
        self.performance.start()
        self.history.load(limit=500)

        if self.config.features.plugins_enabled:
            self.plugins.write_example()
            self.plugins.load_all()

        try:
            self.detector.start()
        except Exception as exc:
            # Deliberately broad: CameraError covers a missing or busy device,
            # but backend construction raises RuntimeError (no usable MediaPipe
            # install), and native loaders can raise almost anything. Any of
            # these means "no tracking", and none of them should be an
            # unhandled traceback in the user's terminal.
            reason = ("camera unavailable" if isinstance(exc, CameraError)
                      else "hand tracking unavailable")
            log.error("%s: %s", reason, exc)
            with self._state_lock:
                self.state.camera_error = str(exc)
            self.notify(reason.title(), str(exc), "error")
            if self.args.headless:
                return False
            # With a UI we keep running so the user can fix the problem and
            # retry from Settings rather than being dumped back to a shell.

        if self.config.features.voice_commands:
            self.voice.start()
        if self.config.features.face_unlock:
            self.presence.start()

        self._install_hotkeys()

        self._stop.clear()
        self._thread = threading.Thread(target=self._process_loop,
                                        name="pipeline", daemon=True)
        self._thread.start()

        self.sounds.play("start")
        log.info("pipeline running (backend=%s, mouse=%s)",
                 self.detector.backend_name, self.cursor.mouse.name)
        return True

    def stop(self) -> None:
        """Shut everything down in dependency order."""
        if self._stop.is_set():
            return
        log.info("shutting down")
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        # Release input first: a stuck mouse button outlives the process.
        self.cursor.emergency_release()

        if self.recorder.is_recording:
            self.recorder.stop()
        self.voice.stop()
        self.presence.stop()
        self.performance.stop()
        self.detector.stop()
        self.history.flush()

        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass

        self.profiles.save(self.config)
        self.sounds.play("stop")
        log.info("shutdown complete")

    # -- hotkeys ---------------------------------------------------------- #

    def _install_hotkeys(self) -> None:
        """Register global keyboard shortcuts.

        Hotkeys are the emergency escape hatch — if gesture recognition
        misbehaves and the cursor runs away, the keyboard must still work.
        """
        try:
            from pynput import keyboard  # type: ignore[import-not-found]

            hotkeys = self.config.hotkeys
            mapping = {
                hotkeys.emergency_stop: self.emergency_stop,
                hotkeys.toggle_tracking: self.toggle_tracking,
                hotkeys.toggle_precision: lambda: self._toggle_precision(),
                hotkeys.screenshot: lambda: self.actions.execute(
                    GestureEvent("hotkey", "screenshot", 1.0)),
                hotkeys.toggle_whiteboard: self._toggle_whiteboard,
                hotkeys.toggle_presentation: self._toggle_presentation,
                hotkeys.recenter: self.cursor.recenter,
            }
            self._hotkey_listener = keyboard.GlobalHotKeys(
                {combo: handler for combo, handler in mapping.items() if combo})
            self._hotkey_listener.start()
            log.info("global hotkeys registered")
        except Exception as exc:
            # On macOS this needs Accessibility permission, which the user may
            # not have granted; the app is fully usable without it.
            log.warning("could not register global hotkeys: %s", exc)

    def emergency_stop(self) -> None:
        """Release all input and pause immediately."""
        log.warning("EMERGENCY STOP")
        self.engine.pause()
        self.cursor.emergency_release()
        self.cursor.enabled = False
        self.sounds.play("error")
        self.notify("Emergency Stop", "All input released — press again to resume",
                    "warning")

    def toggle_tracking(self) -> bool:
        """Pause or resume tracking.  Returns True when running."""
        if self.engine.enabled:
            self.engine.pause()
            self.cursor.end_drag()
            self.cursor.enabled = False
            self.notify("Tracking paused", "", "info")
        else:
            self.engine.resume()
            self.cursor.enabled = True
            self.engine.reset()
            self.cursor.reset()
            self.notify("Tracking resumed", "", "success")
        self.sounds.play("mode_change")
        return self.engine.enabled

    # -- processing loop -------------------------------------------------- #

    def _process_loop(self) -> None:
        """Main pipeline: capture → detect → recognise → act → render."""
        log.info("processing thread started")
        consecutive_failures = 0

        while not self._stop.is_set():
            frame_start = self.performance.begin_frame()

            try:
                result = self.detector.process(block=True)
            except Exception as exc:
                log.error("detector error: %s", exc)
                result = None

            if result is None:
                consecutive_failures += 1
                self.performance.record_drop()
                if consecutive_failures == 60:
                    log.error("no frames for ~2s; camera may have disconnected")
                    self.notify("Camera stalled", "No frames received", "error")
                if consecutive_failures > 300:
                    time.sleep(0.05)
                continue

            consecutive_failures = 0
            timestamp = time.monotonic()

            try:
                self._process_frame(result, timestamp, frame_start)
            except Exception as exc:
                log.error("frame processing failed: %s", exc, exc_info=True)

        log.info("processing thread stopped")

    def _process_frame(self, result: Any, timestamp: float,
                       frame_start: float) -> None:
        """Handle one detection result end to end."""
        frame = result.frame

        # -- calibration takes over the whole pipeline -------------------- #
        if self.calibration.is_running:
            brightness = float(np.mean(frame)) if frame is not None else None
            self.calibration.update(result.hands, brightness)
            self._render(frame, result, None, calibrating=True)
            self.performance.end_frame(frame_start, result.inference_ms)
            return

        if self.config.features.face_unlock:
            self.presence.submit_frame(frame)

        # -- recognition --------------------------------------------------- #
        output = self.engine.update(result.hands, timestamp)

        # -- cursor -------------------------------------------------------- #
        if output.tracking and not self.engine.sleeping and self.cursor.enabled:
            self.cursor.update(output.cursor_point, timestamp,
                               precision=self.engine.precision_mode)

        # -- gesture recording -------------------------------------------- #
        if self.recorder_wizard.is_active:
            drawing = output.pose == Pose.POINT
            point = output.features.index_tip if output.features else None
            self.recorder_wizard.update(point, drawing, timestamp)

        # -- whiteboard ----------------------------------------------------- #
        if self.whiteboard.active:
            drawing = output.pose == Pose.POINT
            point = output.features.index_tip if output.features else None
            self.whiteboard.update(point, drawing)

        # -- presentation --------------------------------------------------- #
        if self.presentation.active:
            visible = output.pose in (Pose.POINT, Pose.GUN)
            point = output.features.index_tip if output.features else None
            self.presentation.update_pointer(point, visible, timestamp)

        # -- actions -------------------------------------------------------- #
        for event in output.events:
            self._dispatch(event, output)

        # -- voice ---------------------------------------------------------- #
        if self.voice.is_listening:
            for action, phrase in self.voice.drain():
                self._voice_command(action, phrase)

        self._publish(frame, result, output)
        self._render(frame, result, output)
        self.performance.end_frame(frame_start, result.inference_ms)

    def _dispatch(self, event: GestureEvent, output: Any) -> None:
        """Execute a gesture event and record it."""
        executed = self.actions.execute(event)
        self.macros.capture(event)

        position = None
        if output.features is not None:
            position = output.features.index_tip

        # Continuous controls fire every frame; logging them would bury the
        # discrete gestures the history view exists to show.
        if event.action not in ("set_volume", "set_brightness", "scroll"):
            self.history.record(
                gesture=event.name, action=event.action,
                confidence=event.confidence, executed=executed,
                hand=event.hand, mode=output.mode.value,
                profile=self.config.profile_name, position=position,
            )

    def _publish(self, frame: np.ndarray, result: Any, output: Any) -> None:
        """Copy pipeline state into the shared snapshot for the UI."""
        with self._state_lock:
            self.state.tracking = output.tracking
            self.state.paused = not self.engine.enabled or self.engine.sleeping
            self.state.mode = output.mode.value
            self.state.pose = output.pose.value
            self.state.confidence = output.confidence
            self.state.hand_count = len(result.hands)
            self.state.clicks = self.cursor.click_count
            self.state.scrolls = self.cursor.scroll_count
            self.state.gestures = self.engine.stats["gestures"]

            if output.control_value is not None:
                label = "Volume" if output.mode == Mode.VOLUME else "Brightness"
                self.state.level = (label, output.control_value)
                self.state.level_until = time.monotonic() + 1.6
            elif self.state.level and time.monotonic() > self.state.level_until:
                self.state.level = None

    def _render(self, frame: np.ndarray, result: Any, output: Any,
                calibrating: bool = False) -> None:
        """Draw the overlay and store the frame for the UI."""
        if frame is None:
            return

        if self.whiteboard.active:
            frame = self.whiteboard.render(frame)
        if self.presentation.active:
            frame = self.presentation.render(frame)

        if calibrating:
            from overlay import draw_calibration

            draw_calibration(frame, self.calibration.instruction,
                             self.calibration.progress,
                             self.calibration.overall_progress,
                             self.themes.theme)
        else:
            snapshot = self.performance.snapshot()
            with self._state_lock:
                level = self.state.level
            hud = HUDState(
                fps=snapshot.fps,
                mode=output.mode.value if output else Mode.NAVIGATE.value,
                pose=output.pose.value if output else Pose.UNKNOWN.value,
                confidence=output.confidence if output else 0.0,
                profile=self.config.profile_name,
                tracking=output.tracking if output else False,
                hand_count=len(result.hands),
                paused=not self.engine.enabled or self.engine.sleeping,
                recording=self.recorder.is_recording,
                precision=self.engine.precision_mode,
                latency_ms=snapshot.frame_ms,
                backend=self.detector.backend_name,
            )
            self.overlay.render(
                frame, result.hands, output, hud,
                active_margin=self.config.cursor.active_region_margin,
                confidence_threshold=self.config.gestures.min_confidence,
                level=level,
            )

        with self._state_lock:
            self.state.frame = frame

    # -- UI accessors ------------------------------------------------------ #

    def get_display_frame(self) -> Optional[Any]:
        """Return the latest frame as a CTkImage for the UI."""
        with self._state_lock:
            frame = None if self.state.frame is None else self.state.frame.copy()
        if frame is None:
            return None

        try:
            import cv2
            import customtkinter as ctk
            from PIL import Image

            height, width = frame.shape[:2]
            # Fit within the panel while preserving aspect ratio.
            max_w, max_h = 940, 560
            scale = min(max_w / width, max_h / height, 1.0)
            size = (int(width * scale), int(height * scale))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            return ctk.CTkImage(light_image=image, dark_image=image, size=size)
        except Exception as exc:
            log.debug("frame conversion failed: %s", exc)
            return None

    def get_engine_state(self) -> Dict[str, object]:
        """State bundle for the dashboard and header."""
        with self._state_lock:
            state = self.state
            elapsed = time.time() - state.session_start
            return {
                "tracking": state.tracking,
                "paused": state.paused,
                "mode": state.mode,
                "pose": state.pose,
                "confidence": state.confidence,
                "hands": state.hand_count,
                "clicks": state.clicks,
                "scrolls": state.scrolls,
                "gestures": state.gestures,
                "session": format_duration(elapsed),
                "latency": self.performance.latency_percentiles(),
                "backend": self.detector.backend_name,
                "camera_error": state.camera_error,
            }

    def list_gesture_rows(self) -> List[Dict[str, object]]:
        """Rows for the gesture library view."""
        from gesture_engine import DEFAULT_BINDINGS, POSE_LIBRARY

        rows: List[Dict[str, object]] = []
        described = {d.pose.value: d.description for d in POSE_LIBRARY}

        for name, default_action in sorted(DEFAULT_BINDINGS.items()):
            override = self.config.gestures.overrides.get(name, {})
            rows.append({
                "name": name,
                "label": name.replace("_", " ").title(),
                "description": described.get(name, _gesture_help(name)),
                "action": self.config.gestures.bindings.get(name, default_action),
                "enabled": bool(override.get("enabled", True)),
                "custom": False,
            })

        for name, gesture in sorted(self.library.custom.items()):
            rows.append({
                "name": name,
                "label": name.title(),
                "description": gesture.description or "Custom air-drawn gesture",
                "action": gesture.action,
                "enabled": gesture.enabled,
                "custom": True,
            })
        return rows

    def update_setting(self, section: str, key: str, value: object) -> None:
        """Apply a settings change from the UI."""
        try:
            self.profiles.update_section(section, **{key: value})
        except Exception as exc:
            log.error("could not update %s.%s: %s", section, key, exc)

    def set_theme(self, name: str) -> None:
        """Change the theme and restyle both UI and overlay."""
        self.config.ui.theme = name
        theme = self.themes.set_theme(name)
        self.overlay.set_theme(theme)
        self.profiles.save(self.config)
        if self.window is not None:
            self.window.apply_theme(theme)

    def bind_gesture(self, gesture: str, action_id: str) -> None:
        """Rebind a gesture to a different action."""
        if gesture in self.library.custom:
            self.library.set_action(gesture, action_id)
        else:
            self.config.gestures.bindings[gesture] = action_id
            self.profiles.save(self.config)
        log.info("bound %s -> %s", gesture, action_id)

    def toggle_gesture(self, gesture: str, enabled: bool) -> None:
        """Enable or disable a gesture."""
        if gesture in self.library.custom:
            self.library.set_enabled(gesture, enabled)
        else:
            self.config.gestures.overrides.setdefault(gesture, {})["enabled"] = enabled
            self.profiles.save(self.config)

    def start_gesture_recording(self, name: str) -> None:
        """Begin recording a custom gesture."""
        self.recorder_wizard.start(name, takes=3)
        self.notify("Recording gesture",
                    f"Draw {name!r} three times with your index finger", "info")

    def delete_gesture(self, name: str) -> None:
        """Delete a custom gesture."""
        if self.library.delete(name):
            self.notify("Gesture deleted", name, "info")

    def start_calibration(self) -> None:
        """Launch the calibration wizard."""
        self.calibration.start()
        self.notify("Calibration started", "Follow the on-screen prompts", "info")

    # -- run modes --------------------------------------------------------- #

    def run_headless(self) -> int:
        """Run the pipeline with no UI until interrupted."""
        if not self.start():
            return 1

        log.info("running headless — press Ctrl+C to stop")

        def handle_signal(signum: int, _frame: object) -> None:
            log.info("received signal %d", signum)
            self._stop.set()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        try:
            while not self._stop.is_set():
                time.sleep(0.5)
                if self.args.stats:
                    snapshot = self.performance.snapshot()
                    state = self.get_engine_state()
                    print(
                        f"\rFPS {snapshot.fps:5.1f} | {state['mode']:<10} | "
                        f"{state['pose']:<14} | conf {state['confidence']:.0%} | "
                        f"gestures {state['gestures']:<4}",
                        end="", flush=True,
                    )
        except KeyboardInterrupt:
            pass
        finally:
            print()
            self.stop()
        return 0

    def run_gui(self) -> int:
        """Run with the full desktop interface."""
        from ui import AppServices, MainWindow

        services = AppServices(
            get_frame=self.get_display_frame,
            get_performance=self.performance.snapshot,
            get_engine_state=self.get_engine_state,
            get_history=lambda: self.history,
            get_config=lambda: self.config,
            get_capabilities=self.platform.capabilities,
            get_system_info=self.performance.system_info,
            toggle_tracking=self.toggle_tracking,
            start_calibration=self.start_calibration,
            emergency_stop=self.emergency_stop,
            update_setting=self.update_setting,
            set_theme=self.set_theme,
            set_profile=lambda name: self.profiles.load(name),
            list_profiles=self.profiles.list_profiles,
            create_profile=lambda name: self.profiles.create(name, self.config),
            delete_profile=lambda name: self.profiles.delete(name),
            duplicate_profile=lambda name: self.profiles.duplicate(name),
            export_profile=lambda name, path: self.profiles.export(name, path),
            import_profile=lambda path: self.profiles.import_profile(path),
            list_gestures=self.list_gesture_rows,
            list_actions=self.actions.labels,
            bind_gesture=self.bind_gesture,
            toggle_gesture=self.toggle_gesture,
            record_gesture=self.start_gesture_recording,
            delete_gesture=self.delete_gesture,
            duplicate_gesture=lambda name: self.library.duplicate(name),
            export_gestures=lambda path: self.library.export(path),
            import_gestures=lambda path: self.library.import_gestures(path),
            export_history=lambda path: self.history.export_csv(path),
            clear_history=self.history.clear,
            on_close=self.stop,
        )

        self.window = MainWindow(services, self.themes)
        # Swap the null notifier for real toasts now the window exists.
        self.notifier = self.window
        self.context.notifier = self.window

        if not self.start():
            self.window.notify("Startup failed", "See the log for details", "error")

        with self._state_lock:
            error = self.state.camera_error
        if error:
            self.window.notify("Camera unavailable", error, "error")

        try:
            self.window.mainloop()
        finally:
            self.stop()
        return 0


def _gesture_help(name: str) -> str:
    """One-line help text for a built-in gesture."""
    help_text = {
        "pinch_tap": "Touch thumb and index together",
        "pinch_double": "Two quick pinches",
        "pinch_middle": "Touch thumb to middle finger",
        "pinch_ring": "Touch thumb to ring finger",
        "drag_start": "Hold a pinch to begin dragging",
        "drag_end": "Release the pinch to drop",
        "fist_hold": "Hold a closed fist",
        "ok_sign": "OK sign held for one second",
        "call_hold": "Thumb and pinky extended",
        "pinky_hold": "Pinky only, held",
        "four_hold": "Four fingers, thumb tucked",
        "open_palm_hold": "Open palm held for one second",
        "swipe_left": "Swipe your pointing finger left",
        "swipe_right": "Swipe your pointing finger right",
        "swipe_up": "Swipe your pointing finger up",
        "swipe_down": "Swipe your pointing finger down",
        "circle": "Draw a circle in the air",
        "circle_cw": "Draw a clockwise circle",
        "triangle": "Draw a triangle in the air",
        "square": "Draw a square in the air",
        "wave": "Draw a wave in the air",
        "z": "Draw the letter Z",
        "s": "Draw the letter S",
        "v_check": "Draw a check mark",
        "caret": "Draw a caret (^)",
        "thumb_up": "Thumbs up",
        "thumb_down": "Thumbs down",
    }
    return help_text.get(name, "Gesture")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=f"{APP_NAME} — control your computer with hand gestures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python app.py                     launch the desktop app\n"
            "  python app.py --headless --stats  run without a UI\n"
            "  python app.py --profile Gaming    start with a named profile\n"
            "  python app.py --list-cameras      show available camera indices\n"
        ),
    )
    parser.add_argument("--headless", action="store_true",
                        help="run the pipeline without the desktop interface")
    parser.add_argument("--stats", action="store_true",
                        help="print live statistics in headless mode")
    parser.add_argument("--profile", metavar="NAME",
                        help="start with a specific profile")
    parser.add_argument("--camera", type=int, metavar="INDEX",
                        help="override the camera device index")
    parser.add_argument("--list-cameras", action="store_true",
                        help="probe for available cameras and exit")
    parser.add_argument("--no-cursor", action="store_true",
                        help="recognise gestures but never move the mouse")
    parser.add_argument("--debug", action="store_true",
                        help="enable debug logging")
    parser.add_argument("--version", action="version",
                        version=f"{APP_NAME} {APP_VERSION}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Application entry point."""
    args = build_parser().parse_args(argv)

    import logging

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    install_excepthook()

    if args.list_cameras:
        from detector import CameraStream

        print("Probing for cameras…")
        found = CameraStream.enumerate_devices()
        if found:
            for index in found:
                print(f"  [{index}] available")
        else:
            print("  No cameras found.")
        return 0

    app = GestureMouseApp(args)

    if args.camera is not None:
        app.config.camera.device_index = args.camera
    if args.no_cursor:
        app.cursor.enabled = False
        log.info("cursor movement disabled by --no-cursor")

    try:
        return app.run_headless() if args.headless else app.run_gui()
    except Exception as exc:
        log.critical("fatal error: %s", exc, exc_info=True)
        app.stop()
        return 1


if __name__ == "__main__":
    sys.exit(main())
