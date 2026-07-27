"""Centralised logging for AI Gesture Mouse Pro.

Provides a rotating file log (full detail, machine-greppable) plus a coloured
console stream (human friendly), and a small in-memory ring buffer that the
UI subscribes to so the Logs panel can render without touching disk.

Usage::

    from logger import get_logger, setup_logging

    setup_logging()                 # once, at startup
    log = get_logger(__name__)
    log.info("camera opened", extra={"event": "camera.open"})
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Deque, List

from config import APP_SLUG, LOG_DIR

#: Log records kept in memory for the UI panel.
_RING_CAPACITY = 500

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_ANSI = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;197m",
}
_RESET = "\033[0m"

_configured = False
_lock = threading.Lock()


class _ColourFormatter(logging.Formatter):
    """Console formatter that tints the level name when a TTY is attached."""

    def __init__(self, colour: bool) -> None:
        super().__init__(_LOG_FORMAT, _DATE_FORMAT)
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        text = super().format(record)
        if not self.colour:
            return text
        return f"{_ANSI.get(record.levelname, '')}{text}{_RESET}"


class RingBufferHandler(logging.Handler):
    """Keeps the last *N* formatted records in memory for live UI display.

    Subscribers registered through :meth:`subscribe` are invoked on every
    record.  Callbacks must be cheap and must never raise — exceptions are
    swallowed so a broken UI widget can't take the logging system down.
    """

    def __init__(self, capacity: int = _RING_CAPACITY) -> None:
        super().__init__()
        self.records: Deque[str] = deque(maxlen=capacity)
        self._subscribers: List[Callable[[str, str], None]] = []
        self._guard = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - formatting must never crash
            return
        with self._guard:
            self.records.append(message)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(record.levelname, message)
            except Exception:  # pragma: no cover - UI errors are non-fatal
                pass

    def subscribe(self, callback: Callable[[str, str], None]) -> None:
        """Register ``callback(levelname, formatted_message)``."""
        with self._guard:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, str], None]) -> None:
        """Remove a previously registered callback (no-op if absent)."""
        with self._guard:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def snapshot(self) -> List[str]:
        """Return a copy of the buffered records, oldest first."""
        with self._guard:
            return list(self.records)


#: Module-level singleton so the UI can attach without plumbing.
ring_handler = RingBufferHandler()


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None) -> None:
    """Configure the root logger.  Idempotent — repeat calls are ignored."""
    global _configured
    with _lock:
        if _configured:
            return

        directory = Path(log_dir or LOG_DIR)
        directory.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        # Drop anything a library installed before us.
        for handler in list(root.handlers):
            root.removeHandler(handler)

        file_handler = logging.handlers.RotatingFileHandler(
            directory / f"{APP_SLUG}.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(_ColourFormatter(colour=sys.stdout.isatty()))
        root.addHandler(console)

        ring_handler.setLevel(logging.DEBUG)
        ring_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S"))
        root.addHandler(ring_handler)

        # MediaPipe and absl are extremely chatty at INFO.
        for noisy in ("mediapipe", "absl", "PIL", "matplotlib", "comtypes"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    if not _configured:
        setup_logging()
    # Strip the package prefix so console lines stay narrow.
    return logging.getLogger(name.rsplit(".", 1)[-1])


def install_excepthook() -> None:
    """Route uncaught exceptions (main thread and workers) into the log."""
    log = get_logger("crash")

    def _hook(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    def _thread_hook(args):  # type: ignore[no-untyped-def]
        log.critical(
            "Uncaught exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _hook
    threading.excepthook = _thread_hook
