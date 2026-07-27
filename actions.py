"""Action registry — the layer that actually *does* things.

The gesture engine is deliberately side-effect free; it reports that a
gesture happened.  This module owns the consequences.  The split matters for
three reasons: recognition stays unit-testable without a desktop session,
users can rebind any gesture to any action without touching recognition
logic, and plugins can register new actions without modifying either side.

Actions are registered in a dictionary keyed by a stable string id.  That id
is what profiles persist, so renaming an action's *label* never breaks an
existing user's bindings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from cursor_controller import CursorController
from gesture_engine import GestureEvent
from logger import get_logger
from platform_bridge import PlatformBridge
from screen_capture import ScreenRecorder, ScreenshotService
from sounds import SoundPlayer
from utils import clamp

log = get_logger(__name__)


class Notifier(Protocol):
    """Anything that can show the user a transient message."""

    def notify(self, title: str, message: str = "",
               level: str = "info") -> None:  # pragma: no cover - protocol
        """Display a notification."""
        ...


@dataclass
class ActionContext:
    """Everything an action handler may need, injected once at construction.

    Passing a context object rather than wiring globals keeps handlers pure
    functions of their inputs, which is what makes them straightforward to
    test and to swap out in a headless run.
    """

    cursor: CursorController
    platform: PlatformBridge
    sounds: SoundPlayer
    screenshots: ScreenshotService
    recorder: ScreenRecorder
    notifier: Optional[Notifier] = None
    #: Late-bound hooks the application fills in (whiteboard, presentation...).
    hooks: Dict[str, Callable[..., object]] = field(default_factory=dict)

    def notify(self, title: str, message: str = "", level: str = "info") -> None:
        """Show a notification if a notifier is attached."""
        if self.notifier is not None:
            try:
                self.notifier.notify(title, message, level)
            except Exception as exc:  # pragma: no cover - UI must not break actions
                log.debug("notifier failed: %s", exc)

    def call_hook(self, name: str, *args: object) -> object:
        """Invoke an optional late-bound hook by name."""
        hook = self.hooks.get(name)
        if hook is None:
            log.debug("hook %r not registered", name)
            return None
        try:
            return hook(*args)
        except Exception as exc:
            log.warning("hook %s failed: %s", name, exc)
            return None


@dataclass
class ActionSpec:
    """Metadata describing a bindable action, used to build the UI."""

    action_id: str
    label: str
    category: str
    handler: Callable[[ActionContext, GestureEvent], bool]
    description: str = ""
    #: Whether this action produces a sound by default.
    sound: str = ""


class ActionRegistry:
    """Holds every bindable action and dispatches gesture events to them."""

    def __init__(self, context: ActionContext) -> None:
        self.context = context
        self._actions: Dict[str, ActionSpec] = {}
        self.execution_count: Dict[str, int] = {}
        self.last_error: str = ""
        self._register_builtins()

    # -- registration ----------------------------------------------------- #

    def register(self, spec: ActionSpec, overwrite: bool = False) -> bool:
        """Add an action.  Returns False if the id is taken and not overwriting."""
        if spec.action_id in self._actions and not overwrite:
            log.warning("action id %r already registered", spec.action_id)
            return False
        self._actions[spec.action_id] = spec
        return True

    def unregister(self, action_id: str) -> bool:
        """Remove an action by id."""
        return self._actions.pop(action_id, None) is not None

    @property
    def actions(self) -> List[ActionSpec]:
        """Every registered action, sorted by category then label."""
        return sorted(self._actions.values(), key=lambda a: (a.category, a.label))

    def by_category(self) -> Dict[str, List[ActionSpec]]:
        """Actions grouped by category, for the bindings editor."""
        grouped: Dict[str, List[ActionSpec]] = {}
        for spec in self.actions:
            grouped.setdefault(spec.category, []).append(spec)
        return grouped

    def get(self, action_id: str) -> Optional[ActionSpec]:
        """Look up an action by id."""
        return self._actions.get(action_id)

    def labels(self) -> Dict[str, str]:
        """``action_id -> label`` map for dropdowns."""
        return {spec.action_id: spec.label for spec in self.actions}

    # -- dispatch --------------------------------------------------------- #

    def execute(self, event: GestureEvent) -> bool:
        """Run the action bound to ``event``.

        Handler exceptions are caught and logged rather than propagated: a
        failing action (a missing application, a revoked permission) must not
        take down the recognition loop that is still processing frames.
        """
        spec = self._actions.get(event.action)
        if spec is None:
            if event.action != "none":
                log.warning("unknown action %r for gesture %r",
                            event.action, event.name)
            return False

        try:
            result = spec.handler(self.context, event)
        except Exception as exc:
            self.last_error = f"{event.action}: {exc}"
            log.error("action %s failed: %s", event.action, exc, exc_info=True)
            self.context.sounds.play("error")
            return False

        if result:
            self.execution_count[event.action] = \
                self.execution_count.get(event.action, 0) + 1
            if spec.sound:
                self.context.sounds.play(spec.sound)
        return bool(result)

    # -- built-in actions ------------------------------------------------- #

    def _register_builtins(self) -> None:
        """Register the shipped action set."""
        for spec in _builtin_actions():
            self.register(spec)
        log.info("registered %d actions", len(self._actions))


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #

def _left_click(ctx: ActionContext, event: GestureEvent) -> bool:
    """Single left click."""
    return ctx.cursor.click("left")


def _double_click(ctx: ActionContext, event: GestureEvent) -> bool:
    """Double left click."""
    return ctx.cursor.click("left", count=2)


def _right_click(ctx: ActionContext, event: GestureEvent) -> bool:
    """Single right click."""
    return ctx.cursor.click("right")


def _middle_click(ctx: ActionContext, event: GestureEvent) -> bool:
    """Middle (wheel) click."""
    return ctx.cursor.click("middle")


def _drag_start(ctx: ActionContext, event: GestureEvent) -> bool:
    """Press and hold the left button."""
    if ctx.cursor.start_drag("left"):
        ctx.notify("Drag started", "Move your hand to drag", "info")
        return True
    return False


def _drag_end(ctx: ActionContext, event: GestureEvent) -> bool:
    """Release the held button."""
    released = ctx.cursor.end_drag()
    if released and event.data.get("reason") == "tracking_lost":
        ctx.notify("Drag released", "Hand tracking was lost", "warning")
    return released


def _hold_click(ctx: ActionContext, event: GestureEvent) -> bool:
    """Toggle a held left button (grab / release)."""
    if ctx.cursor.dragging:
        return ctx.cursor.end_drag()
    return ctx.cursor.start_drag("left")


def _scroll(ctx: ActionContext, event: GestureEvent) -> bool:
    """Scroll by the event's normalized delta."""
    delta = float(event.data.get("delta", 0.0))  # type: ignore[arg-type]
    return ctx.cursor.scroll(delta)


def _zoom(ctx: ActionContext, event: GestureEvent) -> bool:
    """Ctrl+scroll zoom driven by two-handed pinch separation."""
    delta = float(event.data.get("delta", 0.0))  # type: ignore[arg-type]
    ticks = 1 if delta > 0 else -1
    try:
        from pynput.keyboard import Controller, Key  # type: ignore[import-not-found]

        keyboard = Controller()
        with keyboard.pressed(Key.ctrl):
            ctx.cursor.mouse.scroll(0, ticks)
        return True
    except Exception as exc:
        log.debug("zoom failed: %s", exc)
        return False


def _set_volume(ctx: ActionContext, event: GestureEvent) -> bool:
    """Set the master volume from the event's ``level``."""
    level = clamp(float(event.data.get("level", 0.0)), 0.0, 1.0)  # type: ignore[arg-type]
    if not ctx.platform.volume.available:
        return False
    ctx.platform.volume.set_volume(level)
    ctx.call_hook("show_volume", level)
    return True


def _volume_up(ctx: ActionContext, event: GestureEvent) -> bool:
    """Step the volume up."""
    if not ctx.platform.volume.available:
        return False
    level = ctx.platform.volume.step(+0.08)
    ctx.notify("Volume", f"{level:.0%}", "info")
    return True


def _volume_down(ctx: ActionContext, event: GestureEvent) -> bool:
    """Step the volume down."""
    if not ctx.platform.volume.available:
        return False
    level = ctx.platform.volume.step(-0.08)
    ctx.notify("Volume", f"{level:.0%}", "info")
    return True


def _toggle_mute(ctx: ActionContext, event: GestureEvent) -> bool:
    """Mute or unmute the system output."""
    if not ctx.platform.volume.available:
        return False
    ctx.platform.volume.toggle_mute()
    ctx.notify("Audio", "Mute toggled", "info")
    return True


def _set_brightness(ctx: ActionContext, event: GestureEvent) -> bool:
    """Set display brightness from the event's ``level``."""
    level = clamp(float(event.data.get("level", 0.0)), 0.0, 1.0)  # type: ignore[arg-type]
    if not ctx.platform.brightness.available:
        return False
    ctx.platform.brightness.set_brightness(level)
    ctx.call_hook("show_brightness", level)
    return True


def _screenshot(ctx: ActionContext, event: GestureEvent) -> bool:
    """Capture and save a screenshot."""
    result = ctx.screenshots.capture()
    if result.success and result.path:
        ctx.notify("Screenshot saved", result.path.name, "success")
        return True
    ctx.notify("Screenshot failed", result.error, "error")
    return False


def _toggle_recording(ctx: ActionContext, event: GestureEvent) -> bool:
    """Start or stop screen recording."""
    if ctx.recorder.is_recording:
        path = ctx.recorder.stop()
        ctx.notify("Recording stopped", path.name if path else "", "success")
        ctx.sounds.play("stop")
    else:
        if not ctx.recorder.start():
            ctx.notify("Recording failed", "Could not start capture", "error")
            return False
        ctx.notify("Recording started", "Repeat the gesture to stop", "info")
        ctx.sounds.play("start")
    return True


def _lock_screen(ctx: ActionContext, event: GestureEvent) -> bool:
    """Lock the desktop session."""
    ctx.cursor.emergency_release()
    return ctx.platform.system.lock_screen()


def _media(action: str) -> Callable[[ActionContext, GestureEvent], bool]:
    """Build a media-key handler."""

    def handler(ctx: ActionContext, event: GestureEvent) -> bool:
        method = getattr(ctx.platform.system, action)
        result = bool(method())
        if result:
            ctx.notify("Media", action.replace("media_", "").replace("_", " ").title())
        return result

    return handler


def _launch(app_key: str) -> Callable[[ActionContext, GestureEvent], bool]:
    """Build an application-launch handler."""

    def handler(ctx: ActionContext, event: GestureEvent) -> bool:
        result = ctx.platform.system.launch_app(app_key)
        ctx.notify("Launching" if result else "Launch failed", app_key.title(),
                   "success" if result else "error")
        return result

    return handler


def _key_action(*keys: str) -> Callable[[ActionContext, GestureEvent], bool]:
    """Build a handler that taps a key combination.

    ``keys`` are ``pynput`` key names; a single-character entry is typed
    literally.  Used for browser navigation and slide control.
    """

    def handler(ctx: ActionContext, event: GestureEvent) -> bool:
        try:
            from pynput.keyboard import Controller, Key  # type: ignore[import-not-found]

            keyboard = Controller()
            resolved = [getattr(Key, k, k) for k in keys]
            modifiers, final = resolved[:-1], resolved[-1]

            if modifiers:
                pressed = []
                try:
                    for modifier in modifiers:
                        keyboard.press(modifier)
                        pressed.append(modifier)
                    keyboard.tap(final)
                finally:
                    for modifier in reversed(pressed):
                        keyboard.release(modifier)
            else:
                keyboard.tap(final)
            return True
        except Exception as exc:
            log.debug("key action %s failed: %s", keys, exc)
            return False

    return handler


def _hook_action(hook: str, title: str) -> Callable[[ActionContext, GestureEvent], bool]:
    """Build a handler that defers to a late-bound application hook."""

    def handler(ctx: ActionContext, event: GestureEvent) -> bool:
        result = ctx.call_hook(hook)
        if result is not None:
            ctx.notify(title, str(result) if not isinstance(result, bool) else "",
                       "info")
        return result is not None

    return handler


def _no_op(ctx: ActionContext, event: GestureEvent) -> bool:
    """Explicitly do nothing — lets a user unbind a gesture."""
    return False


def _builtin_actions() -> List[ActionSpec]:
    """Every action shipped with the application."""
    return [
        # -- mouse -------------------------------------------------------- #
        ActionSpec("left_click", "Left Click", "Mouse", _left_click,
                   "Primary click.", sound="click"),
        ActionSpec("double_click", "Double Click", "Mouse", _double_click,
                   "Two rapid primary clicks.", sound="double_click"),
        ActionSpec("right_click", "Right Click", "Mouse", _right_click,
                   "Opens the context menu.", sound="click"),
        ActionSpec("middle_click", "Middle Click", "Mouse", _middle_click,
                   "Wheel click.", sound="click"),
        ActionSpec("drag_start", "Start Drag", "Mouse", _drag_start,
                   "Presses and holds the left button."),
        ActionSpec("drag_end", "Drop", "Mouse", _drag_end,
                   "Releases the held button."),
        ActionSpec("hold_click", "Hold / Release Click", "Mouse", _hold_click,
                   "Toggles a held left button.", sound="click"),
        ActionSpec("scroll", "Scroll", "Mouse", _scroll,
                   "Scrolls by the gesture's vertical travel."),
        ActionSpec("zoom", "Pinch Zoom", "Mouse", _zoom,
                   "Ctrl+scroll zoom from two-handed pinch."),

        # -- navigation --------------------------------------------------- #
        ActionSpec("browser_back", "Browser Back", "Navigation",
                   _key_action("alt", "left"), "Navigates back."),
        ActionSpec("browser_forward", "Browser Forward", "Navigation",
                   _key_action("alt", "right"), "Navigates forward."),
        ActionSpec("next_slide", "Next Slide", "Navigation",
                   _key_action("right"), "Advances a presentation."),
        ActionSpec("prev_slide", "Previous Slide", "Navigation",
                   _key_action("left"), "Goes back a slide."),
        ActionSpec("black_screen", "Blank Slide", "Navigation",
                   _key_action("b"), "Blanks the projector (PowerPoint 'B')."),
        ActionSpec("escape", "Escape", "Navigation",
                   _key_action("esc"), "Sends the Escape key."),
        ActionSpec("switch_window", "Switch Window", "Navigation",
                   _key_action("alt", "tab"), "Alt-Tab to the next window."),

        # -- system ------------------------------------------------------- #
        ActionSpec("set_volume", "Set Volume", "System", _set_volume,
                   "Maps pinch distance onto the master volume."),
        ActionSpec("volume_up", "Volume Up", "System", _volume_up,
                   "Steps the volume up."),
        ActionSpec("volume_down", "Volume Down", "System", _volume_down,
                   "Steps the volume down."),
        ActionSpec("toggle_mute", "Mute / Unmute", "System", _toggle_mute,
                   "Toggles system mute.", sound="notification"),
        ActionSpec("set_brightness", "Set Brightness", "System", _set_brightness,
                   "Maps pinch distance onto display brightness."),
        ActionSpec("screenshot", "Screenshot", "System", _screenshot,
                   "Saves a timestamped screenshot.", sound="screenshot"),
        ActionSpec("toggle_recording", "Screen Recording", "System",
                   _toggle_recording, "Starts or stops screen recording."),
        ActionSpec("lock_screen", "Lock Screen", "System", _lock_screen,
                   "Locks the desktop session.", sound="notification"),
        ActionSpec("media_play_pause", "Play / Pause", "System",
                   _media("media_play_pause"), "Toggles media playback."),
        ActionSpec("media_next", "Next Track", "System",
                   _media("media_next"), "Skips to the next track."),
        ActionSpec("media_previous", "Previous Track", "System",
                   _media("media_previous"), "Returns to the previous track."),

        # -- applications ------------------------------------------------- #
        ActionSpec("open_browser", "Open Browser", "Applications",
                   _launch("browser"), "Launches the default browser."),
        ActionSpec("open_vscode", "Open VS Code", "Applications",
                   _launch("vscode"), "Launches Visual Studio Code."),
        ActionSpec("open_spotify", "Open Spotify", "Applications",
                   _launch("spotify"), "Launches Spotify."),
        ActionSpec("open_terminal", "Open Terminal", "Applications",
                   _launch("terminal"), "Launches the terminal."),
        ActionSpec("open_files", "Open File Manager", "Applications",
                   _launch("files"), "Opens the file manager."),
        ActionSpec("open_calculator", "Open Calculator", "Applications",
                   _launch("calculator"), "Opens the calculator."),

        # -- application modes -------------------------------------------- #
        ActionSpec("toggle_whiteboard", "Toggle Whiteboard", "Modes",
                   _hook_action("toggle_whiteboard", "Whiteboard"),
                   "Shows or hides the air-drawing canvas.", sound="mode_change"),
        ActionSpec("toggle_presentation", "Presentation Mode", "Modes",
                   _hook_action("toggle_presentation", "Presentation Mode"),
                   "Enters or leaves slide-control mode.", sound="mode_change"),
        ActionSpec("toggle_sleep", "Pause Tracking", "Modes",
                   _hook_action("toggle_sleep", "Tracking paused"),
                   "Suspends gesture detection.", sound="mode_change"),
        ActionSpec("wake_tracking", "Resume Tracking", "Modes",
                   _hook_action("wake_tracking", "Tracking resumed"),
                   "Resumes gesture detection.", sound="mode_change"),
        ActionSpec("toggle_precision", "Precision Mode", "Modes",
                   _hook_action("toggle_precision", "Precision Mode"),
                   "Slows the cursor for fine positioning."),
        ActionSpec("app_launcher", "App Launcher", "Modes",
                   _hook_action("app_launcher", "App Launcher"),
                   "Opens the gesture-driven application launcher."),
        ActionSpec("recenter_cursor", "Recentre Cursor", "Modes",
                   lambda ctx, e: (ctx.cursor.recenter(), True)[1],
                   "Snaps the cursor to the screen centre."),

        # -- misc --------------------------------------------------------- #
        ActionSpec("none", "Do Nothing", "Other", _no_op,
                   "Unbinds the gesture."),
    ]


class MacroRecorder:
    """Records a sequence of actions and replays them as one gesture.

    Macros are stored as ordered ``(action_id, delay)`` pairs.  Replay honours
    the recorded inter-action delays so a macro that types into a dialog does
    not outrun the dialog appearing.
    """

    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry
        self._recording = False
        self._steps: List[Dict[str, object]] = []
        self._last_time = 0.0
        self.macros: Dict[str, List[Dict[str, object]]] = {}

    @property
    def is_recording(self) -> bool:
        """Whether a macro is currently being captured."""
        return self._recording

    def start(self) -> None:
        """Begin capturing executed actions."""
        self._recording = True
        self._steps = []
        self._last_time = time.monotonic()
        log.info("macro recording started")

    def capture(self, event: GestureEvent) -> None:
        """Record one executed action; called by the dispatcher."""
        if not self._recording:
            return
        now = time.monotonic()
        self._steps.append({
            "action": event.action,
            "delay": round(min(now - self._last_time, 5.0), 3),
            "data": dict(event.data),
        })
        self._last_time = now

    def stop(self, name: str) -> int:
        """Finish recording and store the macro under ``name``."""
        self._recording = False
        if not self._steps:
            return 0
        self.macros[name] = self._steps
        count = len(self._steps)
        self._steps = []
        log.info("macro %r saved with %d steps", name, count)
        return count

    def play(self, name: str) -> bool:
        """Replay a stored macro."""
        steps = self.macros.get(name)
        if not steps:
            return False

        for step in steps:
            delay = float(step.get("delay", 0.0))  # type: ignore[arg-type]
            if delay > 0:
                time.sleep(min(delay, 5.0))
            self.registry.execute(GestureEvent(
                name=f"macro:{name}",
                action=str(step.get("action", "none")),
                confidence=1.0,
                data=dict(step.get("data", {})),  # type: ignore[arg-type]
            ))
        return True

    def to_dict(self) -> Dict[str, List[Dict[str, object]]]:
        """Serialisable macro library."""
        return dict(self.macros)

    def from_dict(self, data: Dict[str, List[Dict[str, object]]]) -> None:
        """Load a macro library."""
        self.macros = dict(data or {})
