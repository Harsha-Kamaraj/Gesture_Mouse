"""Configuration schema and persistence for AI Gesture Mouse Pro.

This module is the single source of truth for every tunable in the
application.  Configuration is modelled with :mod:`dataclasses` rather than
raw dictionaries so that:

* every option has a documented type and a sane default,
* the IDE / type-checker can catch typos at author time,
* profiles round-trip losslessly through JSON, and
* new options can be added without breaking already-saved profiles
  (unknown keys are ignored, missing keys fall back to the default).

The configuration tree is intentionally flat-ish and grouped by concern so
that the settings UI can be generated almost mechanically from it.
"""

from __future__ import annotations

# Environment shims must run before MediaPipe, matplotlib or CustomTkinter are
# imported anywhere.  ``config`` is the module every other module depends on,
# which makes it the one place that reliably comes first.  See compat.py.
import compat  # noqa: F401

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, get_type_hints

APP_NAME = "AI Gesture Mouse Pro"
APP_SLUG = "gesture-mouse-pro"
APP_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #

#: Root of the source tree.  All bundled resources are resolved relative to it.
ROOT_DIR = Path(__file__).resolve().parent

ASSETS_DIR = ROOT_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
THEMES_DIR = ASSETS_DIR / "themes"
SOUNDS_DIR = ASSETS_DIR / "sounds"
PROFILES_DIR = ROOT_DIR / "profiles"
SCREENSHOT_DIR = ROOT_DIR / "screenshots"
RECORDING_DIR = ROOT_DIR / "recordings"
LOG_DIR = ROOT_DIR / "logs"
PLUGIN_DIR = ROOT_DIR / "plugins"
DOCS_DIR = ROOT_DIR / "docs"

#: User-writable state that is *not* part of a profile (history, macros, ...).
DATA_DIR = ROOT_DIR / "data"

GESTURE_LIBRARY_FILE = DATA_DIR / "custom_gestures.json"
HISTORY_FILE = DATA_DIR / "gesture_history.jsonl"
MACRO_FILE = DATA_DIR / "macros.json"
APP_STATE_FILE = DATA_DIR / "app_state.json"

_ALL_DIRS = (
    ASSETS_DIR, ICONS_DIR, THEMES_DIR, SOUNDS_DIR, PROFILES_DIR,
    SCREENSHOT_DIR, RECORDING_DIR, LOG_DIR, PLUGIN_DIR, DATA_DIR,
)


def ensure_directories() -> None:
    """Create every directory the application expects to exist.

    Safe to call repeatedly; used at startup before any I/O happens.
    """
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Platform detection
# --------------------------------------------------------------------------- #

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

PLATFORM_NAME = "Windows" if IS_WINDOWS else "macOS" if IS_MACOS else "Linux"


# --------------------------------------------------------------------------- #
# Configuration sections
# --------------------------------------------------------------------------- #

@dataclass
class CameraConfig:
    """Video capture parameters."""

    device_index: int = 0
    # 640x480 rather than 720p: capture plus the BGR->RGB conversion measured
    # 45% of a CPU core at 1280x720 against 25% here, which made raw capture
    # more expensive than the hand tracking it feeds. MediaPipe downscales for
    # inference regardless, so the larger frame bought nothing but heat.
    width: int = 640
    height: int = 480
    target_fps: int = 30
    #: Mirror the frame so that moving your hand right moves the cursor right.
    mirror: bool = True
    #: Downscale factor applied *before* inference.  Values below 1.0 trade a
    #: little accuracy for a large latency win on CPU-only machines.
    inference_scale: float = 0.75
    #: Seconds without a decoded frame before the camera is considered lost.
    stall_timeout: float = 3.0
    #: Backend hint; "auto" lets OpenCV choose.
    backend: str = "auto"


@dataclass
class DetectionConfig:
    """MediaPipe Hands parameters."""

    max_num_hands: int = 2
    #: 0 = lite/fast, 1 = full/accurate.
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    #: Prefer this hand for cursor control ("Right", "Left" or "Any").
    primary_hand: str = "Right"
    #: When True the primary hand is chosen automatically from usage stats.
    auto_hand_dominance: bool = True
    #: Frames per second to run inference at once no hand has been seen for
    #: ``idle_timeout`` seconds.  Hand tracking is by far the most expensive
    #: stage, and running it at full rate against an empty frame burns a CPU
    #: core to learn nothing.  A hand entering the frame is picked up within
    #: one idle frame, after which full rate resumes immediately.
    idle_fps: float = 8.0
    #: Seconds without a detected hand before dropping to ``idle_fps``.
    idle_timeout: float = 2.5


@dataclass
class CursorConfig:
    """Cursor mapping, smoothing and prediction."""

    #: Multiplies raw hand travel.  >1 amplifies, <1 dampens.
    sensitivity: float = 1.0
    #: Extra gain applied after smoothing; useful for accessibility.
    speed: float = 1.0
    #: 0 = no smoothing (jittery), 1 = maximum smoothing (laggy).
    smoothing: float = 0.55
    #: Radius in normalized units below which motion is ignored entirely.
    dead_zone: float = 0.004
    #: How far ahead (seconds) velocity is extrapolated to hide latency.
    prediction_time: float = 0.035
    #: Fraction of the frame trimmed from each edge to form the active region.
    #: A smaller active region means less arm travel to cross the screen.
    active_region_margin: float = 0.16
    #: Cursor gain while precision mode is engaged.
    precision_factor: float = 0.28
    #: Pixels per scroll gesture tick.
    scroll_speed: float = 3.0
    #: Invert scroll direction ("natural" scrolling).
    natural_scroll: bool = False
    #: Move the cursor to the monitor the hand points at.
    multi_monitor: bool = True
    #: Index into the detected monitor list; -1 spans the full virtual desktop.
    target_monitor: int = -1
    #: One Euro filter tuning.  Lower min_cutoff = smoother but laggier.
    one_euro_min_cutoff: float = 1.2
    one_euro_beta: float = 0.012


@dataclass
class GestureConfig:
    """Recognition thresholds shared by the whole gesture engine."""

    #: Actions below this confidence are dropped outright.
    min_confidence: float = 0.72
    #: A gesture must be observed this many consecutive frames to fire.
    stability_frames: int = 3
    #: Global minimum seconds between two triggered actions.
    global_cooldown: float = 0.28
    #: Normalized pinch distance below which thumb+index count as touching.
    pinch_threshold: float = 0.055
    #: Hysteresis applied when releasing a pinch, prevents click chatter.
    pinch_release_threshold: float = 0.085
    #: Seconds a pinch must be held before it becomes a drag rather than a click.
    drag_hold_time: float = 0.45
    #: Seconds a pose must be held before a "hold" gesture (whiteboard,
    #: presentation, click-hold, sleep) fires.  A resting hand often happens to
    #: match a pose like Fist or Four Fingers, so too low a value makes those
    #: modes toggle on their own while the user is doing something else.
    hold_duration: float = 1.2
    #: Maximum seconds between two clicks for a double click.
    double_click_interval: float = 0.42
    #: Number of frames kept for dynamic (motion) gesture recognition.
    motion_history_length: int = 64
    #: Minimum $1-recognizer score for a dynamic gesture to be accepted.
    dynamic_min_score: float = 0.80
    #: Minimum normalized path length before a dynamic gesture is considered.
    dynamic_min_path: float = 0.65
    #: Seconds of no hand before tracking is reported as lost.
    tracking_lost_timeout: float = 0.8
    #: Per-gesture enable flags and overrides, keyed by gesture name.
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: gesture name -> action id.  Empty means "use the built-in default".
    bindings: Dict[str, str] = field(default_factory=dict)


@dataclass
class UIConfig:
    """Appearance and window behaviour."""

    theme: str = "dark"
    accent: str = "#7C5CFF"
    #: Global widget scale; raise for accessibility.
    ui_scale: float = 1.0
    show_landmarks: bool = True
    show_skeleton: bool = True
    show_fps: bool = True
    show_confidence: bool = True
    show_overlay_panel: bool = True
    high_contrast: bool = False
    large_text: bool = False
    start_minimized: bool = False
    always_on_top: bool = False
    window_width: int = 1360
    window_height: int = 820


@dataclass
class FeatureConfig:
    """Feature toggles for the optional subsystems."""

    voice_commands: bool = False
    sound_effects: bool = True
    toast_notifications: bool = True
    voice_feedback: bool = False
    left_handed_mode: bool = False
    face_unlock: bool = False
    #: Seconds with no face visible before the screen is locked (0 disables).
    auto_lock_seconds: int = 0
    #: Seconds of inactivity before tracking auto-pauses (0 disables).
    idle_timeout: int = 300
    plugins_enabled: bool = True
    analytics: bool = True
    whiteboard_enabled: bool = True
    presentation_enabled: bool = True


@dataclass
class HotkeyConfig:
    """Global keyboard shortcuts.  Values use ``pynput``-style notation."""

    emergency_stop: str = "<ctrl>+<alt>+q"
    # Named keys must be bracketed; a bare "space" fails to parse.
    toggle_tracking: str = "<ctrl>+<alt>+<space>"
    toggle_precision: str = "<ctrl>+<alt>+p"
    screenshot: str = "<ctrl>+<alt>+s"
    toggle_whiteboard: str = "<ctrl>+<alt>+w"
    toggle_presentation: str = "<ctrl>+<alt>+d"
    recenter: str = "<ctrl>+<alt>+c"


@dataclass
class AppConfig:
    """Root configuration object — one instance per active profile."""

    profile_name: str = "Default"
    description: str = "Balanced defaults suitable for everyday desktop use."
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    #: Populated by the calibration wizard; ``None`` until first calibration.
    calibration: Dict[str, Any] | None = None
    schema_version: int = 1

    # -- serialisation ---------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-JSON representation of the whole config tree."""
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a pretty JSON document."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Build a config from ``data``, tolerating missing/unknown keys."""
        return _from_dict(cls, data or {})

    def save(self, path: os.PathLike[str] | str) -> None:
        """Atomically write the config to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "AppConfig":
        """Read a config from ``path``, falling back to defaults on error."""
        path = Path(path)
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return cls()

    def clone(self, new_name: str | None = None) -> "AppConfig":
        """Deep-copy the config, optionally renaming the profile."""
        copy = AppConfig.from_dict(self.to_dict())
        if new_name:
            copy.profile_name = new_name
        return copy


# --------------------------------------------------------------------------- #
# Generic dataclass <- dict hydration
# --------------------------------------------------------------------------- #

T = TypeVar("T")


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively hydrate dataclass ``cls`` from ``data``.

    Unknown keys are ignored and missing keys keep their defaults, which is
    what makes old profiles forward-compatible with new releases.  Because
    ``from __future__ import annotations`` turns every annotation into a
    string, nested dataclass types are resolved through :func:`get_type_hints`
    rather than read straight off ``Field.type``.
    """
    if not is_dataclass(cls):  # pragma: no cover - guarded by call sites
        return data  # type: ignore[return-value]

    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints.get(f.name)
        if isinstance(hint, type) and is_dataclass(hint) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(hint, value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Built-in profile presets
# --------------------------------------------------------------------------- #

def _preset(name: str, description: str, **overrides: Any) -> AppConfig:
    """Create a preset profile by applying dotted-path ``overrides``."""
    cfg = AppConfig(profile_name=name, description=description)
    for dotted, value in overrides.items():
        section, _, attr = dotted.partition("__")
        target = getattr(cfg, section) if attr else cfg
        setattr(target, attr or section, value)
    return cfg


def builtin_profiles() -> List[AppConfig]:
    """Return the four shipped profiles used to seed a fresh install."""
    return [
        _preset(
            "Default",
            "Balanced defaults suitable for everyday desktop use.",
        ),
        _preset(
            "Gaming",
            "Low latency: minimal smoothing, high gain, fast cooldowns.",
            cursor__sensitivity=1.45,
            cursor__smoothing=0.25,
            cursor__prediction_time=0.055,
            cursor__one_euro_min_cutoff=2.4,
            gestures__global_cooldown=0.16,
            gestures__stability_frames=2,
            detection__model_complexity=0,
            camera__target_fps=60,
        ),
        _preset(
            "Office",
            "Precise and calm: heavy smoothing, conservative thresholds.",
            cursor__sensitivity=0.85,
            cursor__smoothing=0.72,
            cursor__precision_factor=0.22,
            gestures__min_confidence=0.80,
            gestures__stability_frames=4,
        ),
        _preset(
            "Presentation",
            "Tuned for air-slide control and laser pointing on a big screen.",
            cursor__sensitivity=1.2,
            cursor__smoothing=0.65,
            gestures__global_cooldown=0.55,
            features__toast_notifications=True,
        ),
        _preset(
            "Accessibility",
            "Large UI, amplified cursor, forgiving thresholds, voice feedback.",
            cursor__sensitivity=1.8,
            cursor__speed=1.35,
            cursor__smoothing=0.78,
            cursor__dead_zone=0.010,
            gestures__min_confidence=0.62,
            gestures__stability_frames=5,
            gestures__global_cooldown=0.45,
            ui__ui_scale=1.25,
            ui__large_text=True,
            ui__high_contrast=True,
            features__voice_feedback=True,
        ),
    ]
