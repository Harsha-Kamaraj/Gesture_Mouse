"""Gesture history, analytics and CSV export.

Every executed gesture is appended to an in-memory ring *and* to a JSON Lines
file on disk.  JSONL is chosen over a single JSON document because it is
append-only — no read-modify-write cycle, so a crash mid-session costs at most
the last line rather than the whole history.

The analytics layer computes the numbers the dashboard renders: per-gesture
counts, mean confidence, accept/reject ratio, activity over time, and a
spatial heatmap of where gestures are performed within the camera frame.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import HISTORY_FILE
from logger import get_logger

log = get_logger(__name__)


@dataclass
class HistoryEntry:
    """One recorded gesture occurrence."""

    timestamp: float
    gesture: str
    action: str
    confidence: float
    duration: float = 0.0
    hand: str = "Right"
    mode: str = "Navigate"
    profile: str = "Default"
    executed: bool = True
    #: Normalized frame position where the gesture happened, for the heatmap.
    position: Optional[Tuple[float, float]] = None

    @property
    def datetime(self) -> datetime:
        """Wall-clock time of the entry."""
        return datetime.fromtimestamp(self.timestamp)

    @property
    def time_string(self) -> str:
        """``HH:MM:SS`` for display."""
        return self.datetime.strftime("%H:%M:%S")

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable form."""
        data = asdict(self)
        if self.position is not None:
            data["position"] = [round(self.position[0], 4), round(self.position[1], 4)]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "HistoryEntry":
        """Rebuild from a serialised entry, tolerating missing fields."""
        position = data.get("position")
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),          # type: ignore[arg-type]
            gesture=str(data.get("gesture", "")),
            action=str(data.get("action", "")),
            confidence=float(data.get("confidence", 0.0)),        # type: ignore[arg-type]
            duration=float(data.get("duration", 0.0)),            # type: ignore[arg-type]
            hand=str(data.get("hand", "Right")),
            mode=str(data.get("mode", "Navigate")),
            profile=str(data.get("profile", "Default")),
            executed=bool(data.get("executed", True)),
            position=(float(position[0]), float(position[1]))     # type: ignore[index]
            if isinstance(position, (list, tuple)) and len(position) == 2 else None,
        )


@dataclass
class SessionStats:
    """Aggregate statistics for the current session."""

    started_at: float = field(default_factory=time.time)
    total: int = 0
    executed: int = 0
    rejected: int = 0
    by_gesture: Counter = field(default_factory=Counter)
    by_action: Counter = field(default_factory=Counter)
    confidence_sum: float = 0.0

    @property
    def duration(self) -> float:
        """Session length in seconds."""
        return time.time() - self.started_at

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across all recorded gestures."""
        return self.confidence_sum / self.total if self.total else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of gestures that were executed rather than rejected."""
        return self.executed / self.total if self.total else 0.0

    @property
    def gestures_per_minute(self) -> float:
        """Throughput, useful for spotting a misconfigured cooldown."""
        minutes = self.duration / 60.0
        return self.total / minutes if minutes > 0.01 else 0.0


class GestureHistory:
    """Thread-safe gesture log with analytics and export.

    Writes are buffered and flushed in batches: appending to disk on every
    single gesture would put file I/O in the recognition loop's path.
    """

    #: Heatmap resolution (cells per axis).
    HEATMAP_BINS = 24

    def __init__(self, path: Path = HISTORY_FILE, capacity: int = 2000,
                 flush_every: int = 10) -> None:
        self.path = path
        self.capacity = capacity
        self.flush_every = max(1, flush_every)

        self._entries: Deque[HistoryEntry] = deque(maxlen=capacity)
        self._pending: List[HistoryEntry] = []
        self._lock = threading.RLock()
        self.stats = SessionStats()
        self._heatmap = np.zeros((self.HEATMAP_BINS, self.HEATMAP_BINS),
                                 dtype=np.float32)

    # -- recording -------------------------------------------------------- #

    def record(self, gesture: str, action: str, confidence: float,
               executed: bool = True, **extra: object) -> HistoryEntry:
        """Append a gesture occurrence and update the running statistics."""
        entry = HistoryEntry(
            timestamp=time.time(), gesture=gesture, action=action,
            confidence=float(confidence), executed=executed,
            duration=float(extra.get("duration", 0.0)),   # type: ignore[arg-type]
            hand=str(extra.get("hand", "Right")),
            mode=str(extra.get("mode", "Navigate")),
            profile=str(extra.get("profile", "Default")),
            position=extra.get("position"),               # type: ignore[arg-type]
        )

        with self._lock:
            self._entries.append(entry)
            self._pending.append(entry)

            self.stats.total += 1
            self.stats.confidence_sum += entry.confidence
            self.stats.by_gesture[gesture] += 1
            self.stats.by_action[action] += 1
            if executed:
                self.stats.executed += 1
            else:
                self.stats.rejected += 1

            if entry.position is not None:
                self._add_to_heatmap(entry.position)

            should_flush = len(self._pending) >= self.flush_every

        if should_flush:
            self.flush()
        return entry

    def _add_to_heatmap(self, position: Tuple[float, float]) -> None:
        """Accumulate a normalized position into the spatial heatmap."""
        bins = self.HEATMAP_BINS
        col = int(min(max(position[0], 0.0), 0.9999) * bins)
        row = int(min(max(position[1], 0.0), 0.9999) * bins)
        self._heatmap[row, col] += 1.0

    # -- persistence ------------------------------------------------------ #

    def flush(self) -> int:
        """Append buffered entries to disk.  Returns how many were written."""
        with self._lock:
            if not self._pending:
                return 0
            batch = self._pending
            self._pending = []

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                for entry in batch:
                    handle.write(json.dumps(entry.to_dict()) + "\n")
            return len(batch)
        except OSError as exc:
            log.warning("history flush failed: %s", exc)
            with self._lock:
                # Put them back so nothing is silently lost.
                self._pending = batch + self._pending
            return 0

    def load(self, limit: int = 2000) -> int:
        """Load the most recent entries from disk into memory."""
        if not self.path.exists():
            return 0

        loaded = 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                lines = deque(handle, maxlen=limit)
            with self._lock:
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._entries.append(HistoryEntry.from_dict(json.loads(line)))
                        loaded += 1
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
            log.info("loaded %d history entries", loaded)
        except OSError as exc:
            log.warning("history load failed: %s", exc)
        return loaded

    def clear(self) -> None:
        """Wipe in-memory history, statistics and the on-disk log."""
        with self._lock:
            self._entries.clear()
            self._pending.clear()
            self.stats = SessionStats()
            self._heatmap.fill(0.0)
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError as exc:
            log.warning("could not delete history file: %s", exc)
        log.info("gesture history cleared")

    # -- querying --------------------------------------------------------- #

    @property
    def entries(self) -> List[HistoryEntry]:
        """All in-memory entries, oldest first."""
        with self._lock:
            return list(self._entries)

    def recent(self, count: int = 50) -> List[HistoryEntry]:
        """The ``count`` most recent entries, newest first."""
        with self._lock:
            return list(self._entries)[-count:][::-1]

    def filter(self, gesture: Optional[str] = None,
               min_confidence: float = 0.0,
               since: Optional[float] = None) -> List[HistoryEntry]:
        """Return entries matching every supplied criterion."""
        with self._lock:
            results = list(self._entries)
        if gesture:
            results = [e for e in results if e.gesture == gesture]
        if min_confidence > 0:
            results = [e for e in results if e.confidence >= min_confidence]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return results

    # -- analytics -------------------------------------------------------- #

    def top_gestures(self, limit: int = 8) -> List[Tuple[str, int]]:
        """Most frequently used gestures."""
        with self._lock:
            return self.stats.by_gesture.most_common(limit)

    def confidence_by_gesture(self) -> Dict[str, float]:
        """Mean confidence per gesture — reveals which need threshold tuning."""
        totals: Dict[str, List[float]] = defaultdict(list)
        for entry in self.entries:
            totals[entry.gesture].append(entry.confidence)
        return {name: sum(values) / len(values) for name, values in totals.items()}

    def activity_series(self, buckets: int = 30,
                        window: float = 300.0) -> List[int]:
        """Gesture counts bucketed over the last ``window`` seconds."""
        now = time.time()
        series = [0] * buckets
        width = window / buckets
        for entry in self.entries:
            age = now - entry.timestamp
            if 0 <= age < window:
                index = buckets - 1 - int(age / width)
                if 0 <= index < buckets:
                    series[index] += 1
        return series

    def heatmap(self, normalise: bool = True) -> np.ndarray:
        """Spatial gesture heatmap over the camera frame.

        Reveals ergonomic problems directly: a heatmap clustered at one edge
        means the active region is mapped badly and the user is reaching.
        """
        with self._lock:
            data = self._heatmap.copy()
        if normalise:
            peak = float(data.max())
            if peak > 0:
                data /= peak
        return data

    def summary(self) -> Dict[str, object]:
        """Compact statistics block for the dashboard."""
        with self._lock:
            stats = self.stats
            return {
                "total": stats.total,
                "executed": stats.executed,
                "rejected": stats.rejected,
                "mean_confidence": round(stats.mean_confidence, 3),
                "acceptance_rate": round(stats.acceptance_rate, 3),
                "gestures_per_minute": round(stats.gestures_per_minute, 1),
                "duration": round(stats.duration, 1),
                "unique_gestures": len(stats.by_gesture),
            }

    # -- export ----------------------------------------------------------- #

    #: Column order for CSV export.
    CSV_FIELDS = ("timestamp", "time", "gesture", "action", "confidence",
                  "duration", "hand", "mode", "profile", "executed")

    def export_csv(self, path: Path,
                   entries: Optional[Iterable[HistoryEntry]] = None) -> int:
        """Write history to a CSV file.  Returns the number of rows."""
        rows = list(entries) if entries is not None else self.entries
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
                for entry in rows:
                    writer.writerow({
                        "timestamp": f"{entry.timestamp:.3f}",
                        "time": entry.datetime.isoformat(timespec="seconds"),
                        "gesture": entry.gesture,
                        "action": entry.action,
                        "confidence": f"{entry.confidence:.4f}",
                        "duration": f"{entry.duration:.3f}",
                        "hand": entry.hand,
                        "mode": entry.mode,
                        "profile": entry.profile,
                        "executed": entry.executed,
                    })
            log.info("exported %d history rows to %s", len(rows), path)
            return len(rows)
        except OSError as exc:
            log.error("CSV export failed: %s", exc)
            return 0
