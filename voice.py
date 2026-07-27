"""Optional voice command engine.

Voice is a genuinely useful complement to gestures because the two modalities
fail differently: gestures are precise about *where* but clumsy about *what*
(you have to remember a pose), while speech is the reverse.  Saying "click"
while pointing is faster and more reliable than contorting your hand.

Speech recognition is entirely optional.  ``SpeechRecognition`` and a working
microphone may well be absent, so the engine reports
:attr:`VoiceEngine.available` and the rest of the application simply skips the
feature rather than failing.

Recognition runs on a worker thread and pushes commands onto a queue the main
loop drains, so a slow network round-trip to a recognition backend can never
stall the gesture pipeline.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from logger import get_logger

log = get_logger(__name__)


@dataclass
class VoiceCommand:
    """A recognised phrase mapped to an action."""

    phrase: str
    action: str
    #: Alternative phrasings that trigger the same action.
    aliases: List[str] = field(default_factory=list)
    description: str = ""

    def matches(self, text: str) -> bool:
        """Whether ``text`` triggers this command."""
        text = text.lower().strip()
        if self.phrase in text:
            return True
        return any(alias in text for alias in self.aliases)


#: Default vocabulary.  Kept small deliberately — a compact grammar is
#: dramatically more accurate than an open one, and these are the commands
#: that genuinely beat a gesture.
DEFAULT_COMMANDS: List[VoiceCommand] = [
    VoiceCommand("click", "left_click", ["select", "tap"], "Left click"),
    VoiceCommand("double click", "double_click", ["double"], "Double click"),
    VoiceCommand("right click", "right_click", ["menu", "context"], "Right click"),
    VoiceCommand("scroll up", "scroll_up_voice", ["page up"], "Scroll up"),
    VoiceCommand("scroll down", "scroll_down_voice", ["page down"], "Scroll down"),
    VoiceCommand("copy", "copy", [], "Copy selection"),
    VoiceCommand("paste", "paste", [], "Paste clipboard"),
    VoiceCommand("screenshot", "screenshot", ["capture", "snapshot"], "Screenshot"),
    VoiceCommand("mute", "toggle_mute", ["silence"], "Toggle mute"),
    VoiceCommand("volume up", "volume_up", ["louder"], "Increase volume"),
    VoiceCommand("volume down", "volume_down", ["quieter"], "Decrease volume"),
    VoiceCommand("open browser", "open_browser", ["browser", "chrome"], "Open browser"),
    VoiceCommand("open code", "open_vscode", ["vs code", "editor"], "Open VS Code"),
    VoiceCommand("open terminal", "open_terminal", ["terminal", "console"], "Open terminal"),
    VoiceCommand("play", "media_play_pause", ["pause"], "Play / pause media"),
    VoiceCommand("next slide", "next_slide", ["next"], "Next slide"),
    VoiceCommand("previous slide", "prev_slide", ["previous", "go back"], "Previous slide"),
    VoiceCommand("lock screen", "lock_screen", ["lock"], "Lock the session"),
    VoiceCommand("stop listening", "_stop_voice", ["stop voice"], "Disable voice"),
    VoiceCommand("pause tracking", "toggle_sleep", ["freeze"], "Pause gestures"),
    VoiceCommand("resume tracking", "wake_tracking", ["unfreeze"], "Resume gestures"),
]


class VoiceEngine:
    """Background speech recogniser producing action ids."""

    def __init__(self, commands: Optional[Sequence[VoiceCommand]] = None,
                 on_command: Optional[Callable[[str, str], None]] = None) -> None:
        self.commands = list(commands or DEFAULT_COMMANDS)
        self.on_command = on_command
        self.queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=32)

        self._recognizer = None
        self._microphone = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._available = False
        self._error = ""
        self.last_heard = ""
        self.recognised_count = 0

    # -- availability ----------------------------------------------------- #

    def probe(self) -> bool:
        """Check whether speech recognition can run here."""
        try:
            import speech_recognition as sr  # type: ignore[import-not-found]

            self._recognizer = sr.Recognizer()
            # Bias toward ignoring background noise over catching quiet speech;
            # a false "click" is far more disruptive than a missed one.
            self._recognizer.energy_threshold = 4000
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.6

            self._microphone = sr.Microphone()
            self._available = True
            log.info("voice engine available")
            return True
        except ImportError:
            self._error = ("SpeechRecognition not installed "
                           "(pip install SpeechRecognition pyaudio)")
        except Exception as exc:
            self._error = f"No usable microphone: {exc}"

        log.info("voice engine unavailable: %s", self._error)
        self._available = False
        return False

    @property
    def available(self) -> bool:
        """Whether the engine can listen."""
        return self._available

    @property
    def error(self) -> str:
        """Why the engine is unavailable, if it is."""
        return self._error

    @property
    def is_listening(self) -> bool:
        """Whether the listener thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> bool:
        """Begin listening on a background thread."""
        if self.is_listening:
            return True
        if not self._available and not self.probe():
            return False

        self._stop.clear()
        self._thread = threading.Thread(target=self._listen_loop,
                                        name="voice", daemon=True)
        self._thread.start()
        log.info("voice recognition started")
        return True

    def stop(self) -> None:
        """Stop listening."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("voice recognition stopped")

    def _listen_loop(self) -> None:
        """Worker: calibrate once, then recognise phrases continuously."""
        import speech_recognition as sr  # type: ignore[import-not-found]

        try:
            with self._microphone as source:  # type: ignore[union-attr]
                self._recognizer.adjust_for_ambient_noise(source, duration=1.0)  # type: ignore[union-attr]
                log.info("microphone calibrated (threshold %.0f)",
                         self._recognizer.energy_threshold)  # type: ignore[union-attr]
        except Exception as exc:
            log.error("microphone calibration failed: %s", exc)
            self._available = False
            return

        while not self._stop.is_set():
            try:
                with self._microphone as source:  # type: ignore[union-attr]
                    audio = self._recognizer.listen(  # type: ignore[union-attr]
                        source, timeout=1.0, phrase_time_limit=3.0)
            except sr.WaitTimeoutError:
                continue
            except Exception as exc:
                log.debug("listen error: %s", exc)
                time.sleep(0.3)
                continue

            if self._stop.is_set():
                break

            try:
                text = self._recognizer.recognize_google(audio).lower()  # type: ignore[union-attr]
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                log.warning("speech backend unreachable: %s", exc)
                time.sleep(2.0)
                continue
            except Exception as exc:
                log.debug("recognition error: %s", exc)
                continue

            self.last_heard = text
            self._dispatch(text)

    def _dispatch(self, text: str) -> None:
        """Match recognised text against the vocabulary and enqueue."""
        # Longest phrase first, so "double click" wins over "click".
        for command in sorted(self.commands,
                              key=lambda c: len(c.phrase), reverse=True):
            if command.matches(text):
                self.recognised_count += 1
                log.info("voice command: %r -> %s", text, command.action)

                if command.action == "_stop_voice":
                    self.stop()
                    return
                try:
                    self.queue.put_nowait((command.action, text))
                except queue.Full:
                    log.debug("voice queue full; dropping command")
                if self.on_command:
                    try:
                        self.on_command(command.action, text)
                    except Exception as exc:
                        log.debug("voice callback failed: %s", exc)
                return

        log.debug("unmatched speech: %r", text)

    # -- consumption ------------------------------------------------------ #

    def drain(self, limit: int = 8) -> List[tuple[str, str]]:
        """Pop pending commands.  Called from the main loop each frame."""
        out: List[tuple[str, str]] = []
        for _ in range(limit):
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return out

    def vocabulary(self) -> Dict[str, str]:
        """``phrase -> description`` for the settings screen."""
        return {command.phrase: command.description for command in self.commands}
