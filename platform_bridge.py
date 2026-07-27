"""Operating-system abstraction layer.

Every OS-specific capability the application needs — system volume, display
brightness, media keys, launching applications, locking the session — is
funnelled through this module behind a uniform interface.

Design notes
------------
The rest of the codebase must never branch on ``sys.platform``.  It asks the
bridge for a capability and receives either a working implementation or a
no-op stub whose :attr:`available` flag is ``False``.  That keeps the feature
code linear and means a missing optional dependency (``pycaw`` on a Linux
box, say) degrades one feature instead of crashing startup.

Backends are resolved lazily on first use so that importing this module is
always fast and always safe.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import webbrowser
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

from config import IS_LINUX, IS_MACOS, IS_WINDOWS
from logger import get_logger
from utils import clamp

log = get_logger(__name__)


def _run(command: Sequence[str], timeout: float = 5.0) -> Optional[str]:
    """Run ``command`` and return stdout, or ``None`` if it fails.

    Never raises: a failed shell-out degrades a feature, it does not take the
    application down.
    """
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            log.debug("command failed (%s): %s", result.returncode, " ".join(command))
            return None
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("command error %s: %s", " ".join(command), exc)
        return None


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #

class VolumeBackend(ABC):
    """Abstract system-volume controller."""

    #: Whether this backend can actually talk to the OS mixer.
    available: bool = False
    name: str = "none"

    @abstractmethod
    def get_volume(self) -> float:
        """Return the master volume in ``[0, 1]``."""

    @abstractmethod
    def set_volume(self, level: float) -> None:
        """Set the master volume; ``level`` is clamped to ``[0, 1]``."""

    def step(self, delta: float) -> float:
        """Adjust volume by ``delta`` and return the new level."""
        level = clamp(self.get_volume() + delta, 0.0, 1.0)
        self.set_volume(level)
        return level

    def toggle_mute(self) -> None:
        """Mute or unmute; default implementation zeroes/restores volume."""
        current = self.get_volume()
        if current > 0.0:
            self._muted_level = current
            self.set_volume(0.0)
        else:
            self.set_volume(getattr(self, "_muted_level", 0.5))


class _NullVolume(VolumeBackend):
    """Fallback used when no mixer backend is reachable."""

    available = False
    name = "unavailable"

    def get_volume(self) -> float:  # noqa: D102
        return 0.0

    def set_volume(self, level: float) -> None:  # noqa: D102
        log.debug("volume control unavailable; ignoring set_volume(%.2f)", level)


class _WindowsVolume(VolumeBackend):
    """Windows master volume via the Core Audio API (``pycaw``)."""

    name = "pycaw"

    def __init__(self) -> None:
        from comtypes import CLSCTX_ALL  # type: ignore[import-not-found]
        from pycaw.pycaw import (  # type: ignore[import-not-found]
            AudioUtilities, IAudioEndpointVolume,
        )
        from ctypes import cast, POINTER

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self._volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.available = True

    def get_volume(self) -> float:  # noqa: D102
        return float(self._volume.GetMasterVolumeLevelScalar())

    def set_volume(self, level: float) -> None:  # noqa: D102
        self._volume.SetMasterVolumeLevelScalar(clamp(level, 0.0, 1.0), None)

    def toggle_mute(self) -> None:  # noqa: D102
        self._volume.SetMute(not self._volume.GetMute(), None)


class _MacVolume(VolumeBackend):
    """macOS master volume via AppleScript (``osascript``)."""

    name = "osascript"

    def __init__(self) -> None:
        if not shutil.which("osascript"):
            raise RuntimeError("osascript not found")
        self.available = True

    def get_volume(self) -> float:  # noqa: D102
        out = _run(["osascript", "-e", "output volume of (get volume settings)"])
        try:
            return clamp(float(out) / 100.0, 0.0, 1.0) if out else 0.0
        except ValueError:
            return 0.0

    def set_volume(self, level: float) -> None:  # noqa: D102
        percent = int(round(clamp(level, 0.0, 1.0) * 100))
        _run(["osascript", "-e", f"set volume output volume {percent}"])

    def toggle_mute(self) -> None:  # noqa: D102
        out = _run(["osascript", "-e", "output muted of (get volume settings)"])
        muted = (out or "false").lower() == "true"
        _run(["osascript", "-e", f"set volume output muted {str(not muted).lower()}"])


class _LinuxVolume(VolumeBackend):
    """Linux master volume via ``pactl`` (PulseAudio / PipeWire)."""

    name = "pactl"
    _SINK = "@DEFAULT_SINK@"

    def __init__(self) -> None:
        if not shutil.which("pactl"):
            raise RuntimeError("pactl not found")
        self.available = True

    def get_volume(self) -> float:  # noqa: D102
        out = _run(["pactl", "get-sink-volume", self._SINK])
        if not out:
            return 0.0
        for token in out.split():
            if token.endswith("%"):
                try:
                    return clamp(int(token.rstrip("%")) / 100.0, 0.0, 1.0)
                except ValueError:
                    continue
        return 0.0

    def set_volume(self, level: float) -> None:  # noqa: D102
        percent = int(round(clamp(level, 0.0, 1.0) * 100))
        _run(["pactl", "set-sink-volume", self._SINK, f"{percent}%"])

    def toggle_mute(self) -> None:  # noqa: D102
        _run(["pactl", "set-sink-mute", self._SINK, "toggle"])


# --------------------------------------------------------------------------- #
# Brightness
# --------------------------------------------------------------------------- #

class BrightnessBackend(ABC):
    """Abstract display-brightness controller."""

    available: bool = False
    name: str = "none"

    @abstractmethod
    def get_brightness(self) -> float:
        """Return brightness in ``[0, 1]``."""

    @abstractmethod
    def set_brightness(self, level: float) -> None:
        """Set brightness; ``level`` is clamped to ``[0, 1]``."""

    def step(self, delta: float) -> float:
        """Adjust brightness by ``delta`` and return the new level."""
        level = clamp(self.get_brightness() + delta, 0.0, 1.0)
        self.set_brightness(level)
        return level


class _NullBrightness(BrightnessBackend):
    """Fallback when brightness cannot be controlled."""

    available = False
    name = "unavailable"

    def get_brightness(self) -> float:  # noqa: D102
        return 0.0

    def set_brightness(self, level: float) -> None:  # noqa: D102
        log.debug("brightness control unavailable; ignoring %.2f", level)


class _SBCBrightness(BrightnessBackend):
    """Cross-platform brightness via ``screen-brightness-control``.

    Works on Windows and most Linux laptops; on macOS the library has no
    backend, so construction fails and we fall through to the null stub.
    """

    name = "screen-brightness-control"

    def __init__(self) -> None:
        import screen_brightness_control as sbc  # type: ignore[import-not-found]

        self._sbc = sbc
        # Probe once — raises if no display backend exists on this machine.
        values = sbc.get_brightness()
        if not values:
            raise RuntimeError("no brightness-capable display")
        self.available = True

    def get_brightness(self) -> float:  # noqa: D102
        try:
            values = self._sbc.get_brightness()
            return clamp(float(values[0]) / 100.0, 0.0, 1.0) if values else 0.0
        except Exception:  # pragma: no cover - hardware dependent
            return 0.0

    def set_brightness(self, level: float) -> None:  # noqa: D102
        try:
            self._sbc.set_brightness(int(round(clamp(level, 0.0, 1.0) * 100)))
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.debug("set_brightness failed: %s", exc)


class _MacBrightness(BrightnessBackend):
    """macOS brightness via the optional ``brightness`` CLI (Homebrew)."""

    name = "brightness-cli"

    def __init__(self) -> None:
        if not shutil.which("brightness"):
            raise RuntimeError("brightness CLI not installed")
        self.available = True

    def get_brightness(self) -> float:  # noqa: D102
        out = _run(["brightness", "-l"])
        if not out:
            return 0.0
        for line in out.splitlines():
            if "brightness" in line:
                try:
                    return clamp(float(line.split()[-1]), 0.0, 1.0)
                except ValueError:
                    continue
        return 0.0

    def set_brightness(self, level: float) -> None:  # noqa: D102
        _run(["brightness", f"{clamp(level, 0.0, 1.0):.2f}"])


# --------------------------------------------------------------------------- #
# System actions (media keys, lock, app launching)
# --------------------------------------------------------------------------- #

class SystemControl:
    """Media playback, session locking and application launching."""

    #: Friendly name -> per-platform launch command.
    APP_COMMANDS: Dict[str, Dict[str, List[str]]] = {
        "browser": {
            "Windows": ["cmd", "/c", "start", "", "chrome"],
            "macOS": ["open", "-a", "Google Chrome"],
            "Linux": ["xdg-open", "https://www.google.com"],
        },
        "vscode": {
            "Windows": ["cmd", "/c", "start", "", "code"],
            "macOS": ["open", "-a", "Visual Studio Code"],
            "Linux": ["code"],
        },
        "spotify": {
            "Windows": ["cmd", "/c", "start", "", "spotify"],
            "macOS": ["open", "-a", "Spotify"],
            "Linux": ["spotify"],
        },
        "terminal": {
            "Windows": ["cmd", "/c", "start", "", "wt"],
            "macOS": ["open", "-a", "Terminal"],
            "Linux": ["x-terminal-emulator"],
        },
        "files": {
            "Windows": ["explorer"],
            "macOS": ["open", "."],
            "Linux": ["xdg-open", "."],
        },
        "calculator": {
            "Windows": ["calc"],
            "macOS": ["open", "-a", "Calculator"],
            "Linux": ["gnome-calculator"],
        },
    }

    def __init__(self) -> None:
        self._keyboard = None
        self._lock = threading.Lock()

    # -- media ------------------------------------------------------------ #

    def _tap_media_key(self, key_name: str) -> bool:
        """Send a media key via pynput; returns True when delivered."""
        try:
            from pynput.keyboard import Controller, Key  # type: ignore[import-not-found]

            with self._lock:
                if self._keyboard is None:
                    self._keyboard = Controller()
                key = getattr(Key, key_name, None)
                if key is None:
                    return False
                self._keyboard.tap(key)
            return True
        except Exception as exc:
            log.debug("media key %s failed: %s", key_name, exc)
            return False

    def media_play_pause(self) -> bool:
        """Toggle media playback."""
        return self._tap_media_key("media_play_pause")

    def media_next(self) -> bool:
        """Skip to the next track."""
        return self._tap_media_key("media_next")

    def media_previous(self) -> bool:
        """Go to the previous track."""
        return self._tap_media_key("media_previous")

    # -- session ---------------------------------------------------------- #

    def lock_screen(self) -> bool:
        """Lock the desktop session.  Returns True on success."""
        if IS_WINDOWS:
            return _run(["rundll32.exe", "user32.dll,LockWorkStation"]) is not None
        if IS_MACOS:
            return _run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "q" '
                "using {control down, command down}",
            ]) is not None
        for cmd in (["loginctl", "lock-session"], ["xdg-screensaver", "lock"],
                    ["gnome-screensaver-command", "-l"]):
            if shutil.which(cmd[0]) and _run(cmd) is not None:
                return True
        return False

    # -- applications ----------------------------------------------------- #

    def launch_app(self, app_key: str) -> bool:
        """Launch a known application by key (see :attr:`APP_COMMANDS`)."""
        platform = "Windows" if IS_WINDOWS else "macOS" if IS_MACOS else "Linux"
        command = self.APP_COMMANDS.get(app_key, {}).get(platform)
        if not command:
            log.warning("no launch command for %r on %s", app_key, platform)
            return False
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=not IS_WINDOWS,
            )
            log.info("launched %s", app_key)
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("failed to launch %s: %s", app_key, exc)
            return False

    def open_url(self, url: str) -> bool:
        """Open ``url`` in the default browser."""
        try:
            return webbrowser.open(url)
        except Exception as exc:  # pragma: no cover - platform dependent
            log.warning("failed to open %s: %s", url, exc)
            return False

    def run_command(self, command: Sequence[str]) -> bool:
        """Run an arbitrary user-configured command (plugin/macro support)."""
        try:
            subprocess.Popen(
                list(command),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=not IS_WINDOWS,
            )
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("command failed %s: %s", command, exc)
            return False


# --------------------------------------------------------------------------- #
# Backend resolution
# --------------------------------------------------------------------------- #

def _build_volume_backend() -> VolumeBackend:
    """Pick the best available volume backend for this platform."""
    candidates: List[Callable[[], VolumeBackend]] = []
    if IS_WINDOWS:
        candidates.append(_WindowsVolume)
    elif IS_MACOS:
        candidates.append(_MacVolume)
    elif IS_LINUX:
        candidates.append(_LinuxVolume)

    for factory in candidates:
        try:
            backend = factory()
            log.info("volume backend: %s", backend.name)
            return backend
        except Exception as exc:
            log.debug("volume backend %s unavailable: %s", factory.__name__, exc)
    log.warning("no volume backend available on this system")
    return _NullVolume()


def _build_brightness_backend() -> BrightnessBackend:
    """Pick the best available brightness backend for this platform."""
    candidates: List[Callable[[], BrightnessBackend]] = []
    if IS_MACOS:
        candidates.append(_MacBrightness)
    candidates.append(_SBCBrightness)

    for factory in candidates:
        try:
            backend = factory()
            log.info("brightness backend: %s", backend.name)
            return backend
        except Exception as exc:
            log.debug("brightness backend %s unavailable: %s", factory.__name__, exc)
    log.warning("no brightness backend available on this system")
    return _NullBrightness()


class PlatformBridge:
    """Lazily-initialised facade over every OS capability.

    A single instance is created by the application and injected wherever
    system control is needed, which keeps the expensive COM/AppleScript probes
    to exactly one per process.
    """

    def __init__(self) -> None:
        self._volume: VolumeBackend | None = None
        self._brightness: BrightnessBackend | None = None
        self.system = SystemControl()

    @property
    def volume(self) -> VolumeBackend:
        """The resolved volume backend (probed on first access)."""
        if self._volume is None:
            self._volume = _build_volume_backend()
        return self._volume

    @property
    def brightness(self) -> BrightnessBackend:
        """The resolved brightness backend (probed on first access)."""
        if self._brightness is None:
            self._brightness = _build_brightness_backend()
        return self._brightness

    def capabilities(self) -> Dict[str, bool]:
        """Report which optional capabilities resolved on this machine.

        Surfaced in the dashboard so a user can see at a glance why, for
        example, brightness gestures are doing nothing.
        """
        return {
            "volume": self.volume.available,
            "brightness": self.brightness.available,
            "media_keys": True,
            "app_launch": True,
        }
