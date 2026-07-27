"""Runtime performance monitoring.

Tracks the numbers that actually determine whether the app feels good:
end-to-end frame time, inference latency, achieved FPS, and process resource
use.  Everything is stored in fixed-capacity ring buffers so memory use is
constant regardless of session length.

Process metrics are sampled on a background thread at a low rate.  Calling
``psutil.cpu_percent`` inline would be self-defeating: the first call blocks
and every call costs more than a frame's budget at 60 FPS.
"""

from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from logger import get_logger
from utils import FPSMeter, RingBuffer, format_bytes

log = get_logger(__name__)

try:
    import psutil  # type: ignore[import-not-found]

    _PSUTIL = True
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]
    _PSUTIL = False


@dataclass
class PerformanceSnapshot:
    """Immutable view of the metrics at one instant."""

    fps: float = 0.0
    peak_fps: float = 0.0
    frame_ms: float = 0.0
    inference_ms: float = 0.0
    render_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    thread_count: int = 0
    gpu_name: str = ""
    dropped_frames: int = 0
    uptime: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        """Dashboard-friendly, pre-rounded representation."""
        return {
            "FPS": f"{self.fps:.1f}",
            "Peak FPS": f"{self.peak_fps:.1f}",
            "Frame Time": f"{self.frame_ms:.1f} ms",
            "Inference": f"{self.inference_ms:.1f} ms",
            "Render": f"{self.render_ms:.1f} ms",
            "CPU": f"{self.cpu_percent:.0f}%",
            "Memory": f"{self.memory_mb:.0f} MB",
            "Threads": str(self.thread_count),
            "Dropped": str(self.dropped_frames),
        }


class PerformanceMonitor:
    """Collects frame timings and process resource usage."""

    #: How often the background sampler polls process metrics.
    SAMPLE_INTERVAL = 1.0

    def __init__(self, window: int = 120) -> None:
        self.fps_meter = FPSMeter(window=60)
        self.frame_times = RingBuffer(window)
        self.inference_times = RingBuffer(window)
        self.render_times = RingBuffer(window)
        self.fps_history = RingBuffer(window)
        self.cpu_history = RingBuffer(window)
        self.memory_history = RingBuffer(window)

        self.dropped_frames = 0
        self.total_frames = 0
        self._started = time.monotonic()

        self._process = psutil.Process(os.getpid()) if _PSUTIL else None
        self._cpu_percent = 0.0
        self._memory_mb = 0.0
        self._memory_percent = 0.0
        self._thread_count = 0
        self._gpu_name = _detect_gpu()

        self._stop = threading.Event()
        self._sampler: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if self._process is not None:
            # Prime the CPU counter; the first reading is always meaningless.
            try:
                self._process.cpu_percent(interval=None)
            except Exception:  # pragma: no cover - platform dependent
                pass

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        """Begin background resource sampling."""
        if self._sampler is not None or self._process is None:
            return
        self._stop.clear()
        self._sampler = threading.Thread(
            target=self._sample_loop, name="perf-sampler", daemon=True)
        self._sampler.start()

    def stop(self) -> None:
        """Stop the sampler thread."""
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2.0)
            self._sampler = None

    def _sample_loop(self) -> None:
        """Worker: poll process metrics at a low, fixed rate."""
        while not self._stop.is_set():
            try:
                assert self._process is not None
                cpu = self._process.cpu_percent(interval=None)
                memory = self._process.memory_info().rss / (1024 * 1024)
                percent = self._process.memory_percent()
                threads = self._process.num_threads()

                with self._lock:
                    self._cpu_percent = cpu
                    self._memory_mb = memory
                    self._memory_percent = percent
                    self._thread_count = threads
                    self.cpu_history.push(cpu)
                    self.memory_history.push(memory)
            except Exception as exc:  # pragma: no cover - platform dependent
                log.debug("resource sampling failed: %s", exc)

            self._stop.wait(self.SAMPLE_INTERVAL)

    # -- frame accounting ------------------------------------------------- #

    def begin_frame(self) -> float:
        """Mark the start of a frame; returns the timestamp to pass back."""
        return time.perf_counter()

    def end_frame(self, start: float, inference_ms: float = 0.0,
                  render_ms: float = 0.0) -> float:
        """Record a completed frame and return the smoothed FPS."""
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        fps = self.fps_meter.tick()

        self.frame_times.push(elapsed_ms)
        self.inference_times.push(inference_ms)
        self.render_times.push(render_ms)
        self.fps_history.push(fps)
        self.total_frames += 1
        return fps

    def record_drop(self) -> None:
        """Note a frame that was requested but never arrived."""
        self.dropped_frames += 1

    # -- reporting -------------------------------------------------------- #

    def snapshot(self) -> PerformanceSnapshot:
        """Capture the current metrics."""
        with self._lock:
            cpu, memory = self._cpu_percent, self._memory_mb
            percent, threads = self._memory_percent, self._thread_count

        return PerformanceSnapshot(
            fps=self.fps_meter.fps,
            peak_fps=self.fps_meter.peak,
            frame_ms=self.frame_times.mean,
            inference_ms=self.inference_times.mean,
            render_ms=self.render_times.mean,
            cpu_percent=cpu,
            memory_mb=memory,
            memory_percent=percent,
            thread_count=threads,
            gpu_name=self._gpu_name,
            dropped_frames=self.dropped_frames,
            uptime=time.monotonic() - self._started,
        )

    def latency_percentiles(self) -> Dict[str, float]:
        """p50/p95/p99 frame time — the numbers that expose stutter.

        Mean frame time hides hitching almost completely; a 60 FPS average
        with a 90 ms p99 feels broken to the user, and only the tail shows it.
        """
        return {
            "p50": self.frame_times.percentile(50),
            "p95": self.frame_times.percentile(95),
            "p99": self.frame_times.percentile(99),
            "max": self.frame_times.maximum,
        }

    def health(self) -> Dict[str, object]:
        """Traffic-light assessment used to warn the user proactively."""
        snapshot = self.snapshot()
        warnings: List[str] = []

        if snapshot.fps and snapshot.fps < 15:
            warnings.append(f"Low frame rate ({snapshot.fps:.0f} FPS)")
        if snapshot.cpu_percent > 85:
            warnings.append(f"High CPU usage ({snapshot.cpu_percent:.0f}%)")
        if snapshot.inference_ms > 45:
            warnings.append(f"Slow inference ({snapshot.inference_ms:.0f} ms)")
        if self.total_frames > 100 and self.dropped_frames / self.total_frames > 0.1:
            warnings.append("Frequent dropped frames")

        return {
            "ok": not warnings,
            "warnings": warnings,
            "fps": snapshot.fps,
            "cpu": snapshot.cpu_percent,
        }

    def reset(self) -> None:
        """Clear all buffers and counters."""
        self.fps_meter.reset()
        for buffer in (self.frame_times, self.inference_times, self.render_times,
                       self.fps_history, self.cpu_history, self.memory_history):
            buffer.clear()
        self.dropped_frames = 0
        self.total_frames = 0
        self._started = time.monotonic()

    def system_info(self) -> Dict[str, str]:
        """Static machine description, shown on the performance page."""
        info = {
            "Platform": f"{platform.system()} {platform.release()}",
            "Python": platform.python_version(),
            "Machine": platform.machine(),
            "GPU": self._gpu_name or "Not detected",
        }
        if _PSUTIL:
            try:
                info["CPU Cores"] = f"{psutil.cpu_count(logical=False)} " \
                                    f"({psutil.cpu_count()} logical)"
                info["Total RAM"] = format_bytes(psutil.virtual_memory().total)
            except Exception:  # pragma: no cover - platform dependent
                pass
        return info


def _detect_gpu() -> str:
    """Best-effort GPU name detection; empty string when unknown."""
    try:
        import subprocess

        if platform.system() == "Darwin":
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            for line in out.splitlines():
                if "Chipset Model:" in line:
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Windows":
            out = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            if len(lines) > 1:
                return lines[1]
        else:
            out = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5, check=False,
            ).stdout
            for line in out.splitlines():
                if "VGA compatible controller" in line:
                    return line.split(":")[-1].strip()
    except Exception:
        pass
    return ""
