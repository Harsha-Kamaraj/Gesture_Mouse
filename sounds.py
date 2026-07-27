"""Procedurally generated UI sound effects.

Rather than shipping binary audio assets — which bloat the repository, carry
their own licences and cannot be tweaked — the small set of UI sounds is
synthesised as WAV files on first run using only the standard library.

Each cue is a short shaped tone or chirp with an exponential envelope, which
is enough to read as a distinct "click", "shutter" or "error" without sounding
like a beep from 1994.  Playback uses whatever the platform provides
(``winsound``, ``afplay``, ``paplay``) and silently degrades to nothing if
none is available.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import threading
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from config import IS_LINUX, IS_MACOS, IS_WINDOWS, SOUNDS_DIR
from logger import get_logger
from utils import clamp

log = get_logger(__name__)

SAMPLE_RATE = 44_100
AMPLITUDE = 0.32


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #

def _envelope(index: int, total: int, attack: float = 0.02,
              decay: float = 0.6) -> float:
    """Percussive amplitude envelope in ``[0, 1]``."""
    position = index / max(total - 1, 1)
    if position < attack:
        return position / attack
    fade = (position - attack) / max(1.0 - attack, 1e-6)
    return math.exp(-fade / max(decay, 1e-6))


def _tone(frequency: float, duration: float, harmonics: Sequence[float] = (1.0,),
          sweep: float = 0.0, decay: float = 0.6) -> List[float]:
    """Render a shaped tone, optionally sweeping in frequency.

    Args:
        frequency: Base frequency in Hz.
        duration: Length in seconds.
        harmonics: Relative amplitudes of successive harmonics.
        sweep: Frequency multiplier applied linearly across the tone; ``0``
            keeps the pitch constant, ``0.5`` sweeps up 50%.
        decay: Envelope decay constant; smaller is snappier.
    """
    count = int(SAMPLE_RATE * duration)
    samples: List[float] = []
    phase = 0.0

    for i in range(count):
        position = i / max(count - 1, 1)
        current = frequency * (1.0 + sweep * position)
        phase += 2.0 * math.pi * current / SAMPLE_RATE

        value = sum(
            amplitude * math.sin(phase * (h + 1))
            for h, amplitude in enumerate(harmonics)
        )
        normaliser = sum(harmonics) or 1.0
        samples.append((value / normaliser) * _envelope(i, count, decay=decay))
    return samples


def _noise_burst(duration: float, decay: float = 0.25) -> List[float]:
    """Filtered pseudo-noise, used for the camera-shutter cue."""
    import random

    rng = random.Random(1337)  # deterministic so the file is reproducible
    count = int(SAMPLE_RATE * duration)
    previous = 0.0
    samples: List[float] = []
    for i in range(count):
        white = rng.uniform(-1.0, 1.0)
        # One-pole low-pass keeps it from sounding like pure hiss.
        previous = 0.35 * white + 0.65 * previous
        samples.append(previous * _envelope(i, count, attack=0.005, decay=decay))
    return samples


def _mix(*layers: Sequence[float]) -> List[float]:
    """Sum layers of differing length, padding with silence."""
    if not layers:
        return []
    length = max(len(layer) for layer in layers)
    out = [0.0] * length
    for layer in layers:
        for i, value in enumerate(layer):
            out[i] += value
    return out


def _write_wav(path: Path, samples: Sequence[float]) -> None:
    """Write mono 16-bit PCM, with clipping protection."""
    peak = max((abs(s) for s in samples), default=1.0) or 1.0
    scale = AMPLITUDE / peak

    frames = b"".join(
        struct.pack("<h", int(clamp(sample * scale, -1.0, 1.0) * 32767))
        for sample in samples
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames)


#: Cue name -> generator.  Adding a sound is one entry here.
SOUND_RECIPES: Dict[str, Callable[[], List[float]]] = {
    "click": lambda: _tone(1180, 0.045, (1.0, 0.25), decay=0.18),
    "double_click": lambda: _mix(
        _tone(1180, 0.04, (1.0, 0.25), decay=0.15),
        [0.0] * int(SAMPLE_RATE * 0.07) + _tone(1180, 0.04, (1.0, 0.25), decay=0.15),
    ),
    "screenshot": lambda: _mix(
        _noise_burst(0.10, decay=0.15),
        _tone(2300, 0.05, (1.0, 0.4), decay=0.10),
    ),
    "notification": lambda: _mix(
        _tone(784, 0.12, (1.0, 0.3), decay=0.45),
        [0.0] * int(SAMPLE_RATE * 0.08) + _tone(1046, 0.16, (1.0, 0.3), decay=0.45),
    ),
    "error": lambda: _tone(196, 0.28, (1.0, 0.6, 0.3), sweep=-0.25, decay=0.5),
    "calibration": lambda: _mix(
        _tone(523, 0.14, (1.0, 0.3), decay=0.4),
        [0.0] * int(SAMPLE_RATE * 0.10) + _tone(659, 0.14, (1.0, 0.3), decay=0.4),
        [0.0] * int(SAMPLE_RATE * 0.20) + _tone(784, 0.26, (1.0, 0.35), decay=0.5),
    ),
    "mode_change": lambda: _tone(660, 0.09, (1.0, 0.2), sweep=0.35, decay=0.3),
    "start": lambda: _tone(440, 0.22, (1.0, 0.45, 0.2), sweep=0.6, decay=0.5),
    "stop": lambda: _tone(660, 0.22, (1.0, 0.45, 0.2), sweep=-0.4, decay=0.5),
}


def generate_sound_assets(force: bool = False) -> Dict[str, Path]:
    """Render every cue to ``assets/sounds/``, skipping existing files."""
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    for name, recipe in SOUND_RECIPES.items():
        path = SOUNDS_DIR / f"{name}.wav"
        if force or not path.exists():
            try:
                _write_wav(path, recipe())
                log.debug("generated sound %s", path.name)
            except Exception as exc:
                log.warning("could not generate sound %s: %s", name, exc)
                continue
        generated[name] = path
    return generated


# --------------------------------------------------------------------------- #
# Playback
# --------------------------------------------------------------------------- #

class SoundPlayer:
    """Fire-and-forget, non-blocking playback of the generated cues.

    Playback always happens on a worker thread: on every platform the
    available playback call blocks for the duration of the sound, and a 250 ms
    stall in the gesture loop is an eternity at 60 FPS.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._paths: Dict[str, Path] = {}
        self._player: Optional[str] = None
        self._lock = threading.Lock()
        self._ready = False

    def initialise(self) -> None:
        """Generate assets and resolve a playback command."""
        with self._lock:
            if self._ready:
                return
            self._paths = generate_sound_assets()
            self._player = self._resolve_player()
            self._ready = True
            log.info("sound player: %s (%d cues)",
                     self._player or "unavailable", len(self._paths))

    @staticmethod
    def _resolve_player() -> Optional[str]:
        """Pick a platform playback mechanism."""
        if IS_WINDOWS:
            try:
                import winsound  # type: ignore[import-not-found]  # noqa: F401

                return "winsound"
            except ImportError:
                return None
        if IS_MACOS:
            return "afplay" if shutil.which("afplay") else None
        if IS_LINUX:
            for candidate in ("paplay", "aplay", "ffplay"):
                if shutil.which(candidate):
                    return candidate
        return None

    def play(self, name: str) -> None:
        """Play a cue by name.  Never blocks, never raises."""
        if not self.enabled:
            return
        if not self._ready:
            self.initialise()

        path = self._paths.get(name)
        if path is None or self._player is None or not path.exists():
            return

        threading.Thread(
            target=self._play_blocking, args=(path,),
            name=f"sound-{name}", daemon=True,
        ).start()

    def _play_blocking(self, path: Path) -> None:
        """Worker body — runs the platform playback call."""
        try:
            if self._player == "winsound":
                import winsound  # type: ignore[import-not-found]

                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            elif self._player == "ffplay":
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
                    check=False, timeout=5,
                )
            elif self._player:
                subprocess.run([self._player, str(path)], check=False, timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            log.debug("sound playback failed: %s", exc)

    def set_enabled(self, enabled: bool) -> None:
        """Toggle sound output."""
        self.enabled = enabled
