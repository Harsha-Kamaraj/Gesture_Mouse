"""Dashboard widgets — stat tiles, sparklines, gauges and the heatmap.

These are custom-drawn on Tk ``Canvas`` widgets rather than assembled from
stock controls.  Tk has no chart primitives, and hand-drawing gives exact
control over the palette so the dashboard matches the camera overlay.

Every widget follows the same contract: construct once, then call ``update``
with new data.  Rebuilding widgets each refresh is what makes naive Tk
dashboards flicker and leak; these mutate existing canvas items in place.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from logger import get_logger
from themes import Theme
from utils import clamp, mix_colours

log = get_logger(__name__)

try:
    import customtkinter as ctk  # type: ignore[import-not-found]

    _CTK = True
except ImportError:  # pragma: no cover - optional at import time
    ctk = None  # type: ignore[assignment]
    _CTK = False


def _font(theme: Theme, size: int, weight: str = "normal") -> tuple:
    """Build a Tk font tuple from the theme."""
    return (theme.font_family, size, weight)


class StatTile:
    """A single metric card: label, big value, optional delta and sparkline."""

    def __init__(self, parent: object, theme: Theme, label: str,
                 value: str = "—", unit: str = "", accent: Optional[str] = None,
                 sparkline: bool = False, width: int = 190, height: int = 96) -> None:
        self.theme = theme
        self.label = label
        self.unit = unit
        self.accent = accent or theme.accent
        self.has_sparkline = sparkline
        self._history: List[float] = []

        self.frame = ctk.CTkFrame(
            parent, width=width, height=height,
            corner_radius=theme.corner_radius, fg_color=theme.surface,
            border_width=1, border_color=theme.border,
        )
        self.frame.pack_propagate(False)

        self._label = ctk.CTkLabel(
            self.frame, text=label.upper(), font=_font(theme, 10, "bold"),
            text_color=theme.text_muted, anchor="w",
        )
        self._label.pack(fill="x", padx=14, pady=(12, 0))

        row = ctk.CTkFrame(self.frame, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(2, 0))

        self._value = ctk.CTkLabel(
            row, text=value, font=_font(theme, 24, "bold"),
            text_color=theme.text, anchor="w",
        )
        self._value.pack(side="left")

        self._unit = ctk.CTkLabel(
            row, text=unit, font=_font(theme, 11),
            text_color=theme.text_muted, anchor="w",
        )
        self._unit.pack(side="left", padx=(4, 0), pady=(8, 0))

        if sparkline:
            self._canvas = ctk.CTkCanvas(
                self.frame, height=26, bg=theme.surface,
                highlightthickness=0, bd=0,
            )
            self._canvas.pack(fill="x", padx=12, pady=(2, 10))
        else:
            self._canvas = None

    def update(self, value: str, sample: Optional[float] = None,
               colour: Optional[str] = None) -> None:
        """Set the displayed value and optionally push a sparkline sample."""
        self._value.configure(text=value, text_color=colour or self.theme.text)

        if sample is not None and self._canvas is not None:
            self._history.append(float(sample))
            if len(self._history) > 60:
                self._history.pop(0)
            self._draw_sparkline()

    def _draw_sparkline(self) -> None:
        """Redraw the mini trend line."""
        canvas = self._canvas
        if canvas is None or len(self._history) < 2:
            return

        canvas.delete("all")
        width = canvas.winfo_width() or 160
        height = canvas.winfo_height() or 26

        low = min(self._history)
        high = max(self._history)
        span = (high - low) or 1.0
        step = width / max(len(self._history) - 1, 1)

        points: List[float] = []
        for i, value in enumerate(self._history):
            x = i * step
            y = height - 3 - ((value - low) / span) * (height - 6)
            points.extend((x, y))

        # Filled area under the curve, then the curve itself.
        area = points + [width, height, 0.0, height]
        canvas.create_polygon(area, fill=mix_colours(self.theme.surface,
                                                     self.accent, 0.22),
                              outline="")
        canvas.create_line(points, fill=self.accent, width=2, smooth=True)

    def grid(self, **kwargs: object) -> "StatTile":
        """Grid the underlying frame and return self for chaining."""
        self.frame.grid(**kwargs)
        return self

    def pack(self, **kwargs: object) -> "StatTile":
        """Pack the underlying frame and return self for chaining."""
        self.frame.pack(**kwargs)
        return self


class Sparkline:
    """A standalone line chart with optional threshold band."""

    def __init__(self, parent: object, theme: Theme, title: str = "",
                 height: int = 120, capacity: int = 120,
                 colour: Optional[str] = None, y_max: Optional[float] = None) -> None:
        self.theme = theme
        self.capacity = capacity
        self.colour = colour or theme.accent
        self.y_max = y_max
        self._values: List[float] = []

        self.frame = ctk.CTkFrame(
            parent, corner_radius=theme.corner_radius, fg_color=theme.surface,
            border_width=1, border_color=theme.border,
        )
        if title:
            ctk.CTkLabel(self.frame, text=title.upper(),
                         font=_font(theme, 10, "bold"),
                         text_color=theme.text_muted, anchor="w"
                         ).pack(fill="x", padx=14, pady=(10, 2))

        self.canvas = ctk.CTkCanvas(self.frame, height=height, bg=theme.surface,
                                    highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def push(self, value: float) -> None:
        """Append a sample and redraw."""
        self._values.append(float(value))
        if len(self._values) > self.capacity:
            self._values.pop(0)
        self.draw()

    def set_values(self, values: Sequence[float]) -> None:
        """Replace the whole series."""
        self._values = list(values)[-self.capacity:]
        self.draw()

    def draw(self) -> None:
        """Redraw the chart."""
        canvas = self.canvas
        canvas.delete("all")
        width = canvas.winfo_width() or 400
        height = canvas.winfo_height() or 120

        if len(self._values) < 2:
            canvas.create_text(width / 2, height / 2, text="Collecting data…",
                               fill=self.theme.text_muted,
                               font=_font(self.theme, 10))
            return

        high = self.y_max if self.y_max is not None else max(self._values)
        high = high or 1.0
        low = 0.0

        # Horizontal guide lines at quarter intervals.
        for fraction in (0.25, 0.5, 0.75):
            y = height - fraction * height
            canvas.create_line(0, y, width, y, fill=self.theme.border, dash=(2, 4))

        step = width / max(len(self._values) - 1, 1)
        points: List[float] = []
        for i, value in enumerate(self._values):
            x = i * step
            y = height - clamp((value - low) / (high - low), 0.0, 1.0) * (height - 6) - 3
            points.extend((x, y))

        canvas.create_polygon(points + [width, height, 0.0, height],
                              fill=mix_colours(self.theme.surface, self.colour, 0.18),
                              outline="")
        canvas.create_line(points, fill=self.colour, width=2, smooth=True)

        latest = self._values[-1]
        canvas.create_text(width - 6, 10, text=f"{latest:.0f}", anchor="e",
                           fill=self.colour, font=_font(self.theme, 10, "bold"))

    def pack(self, **kwargs: object) -> "Sparkline":
        """Pack the frame and return self."""
        self.frame.pack(**kwargs)
        return self

    def grid(self, **kwargs: object) -> "Sparkline":
        """Grid the frame and return self."""
        self.frame.grid(**kwargs)
        return self


class Gauge:
    """A semicircular gauge for bounded values like CPU or confidence."""

    def __init__(self, parent: object, theme: Theme, title: str,
                 size: int = 130, colour: Optional[str] = None) -> None:
        self.theme = theme
        self.size = size
        self.colour = colour or theme.accent
        self.title = title

        self.frame = ctk.CTkFrame(
            parent, corner_radius=theme.corner_radius, fg_color=theme.surface,
            border_width=1, border_color=theme.border,
        )
        self.canvas = ctk.CTkCanvas(self.frame, width=size, height=size * 0.72,
                                    bg=theme.surface, highlightthickness=0, bd=0)
        self.canvas.pack(padx=12, pady=(12, 4))
        ctk.CTkLabel(self.frame, text=title.upper(), font=_font(theme, 10, "bold"),
                     text_color=theme.text_muted).pack(pady=(0, 10))
        self.set(0.0)

    def set(self, fraction: float, label: Optional[str] = None) -> None:
        """Set the gauge to ``fraction`` in ``[0, 1]``."""
        canvas = self.canvas
        canvas.delete("all")
        fraction = clamp(fraction, 0.0, 1.0)

        size = self.size
        pad = 12
        box = (pad, pad, size - pad, size - pad)

        canvas.create_arc(*box, start=180, extent=-180, style="arc",
                          outline=self.theme.surface_alt, width=12)
        if fraction > 0.005:
            canvas.create_arc(*box, start=180, extent=-180 * fraction, style="arc",
                              outline=self.colour, width=12)

        canvas.create_text(size / 2, size / 2 - 6,
                           text=label if label is not None else f"{fraction:.0%}",
                           fill=self.theme.text, font=_font(self.theme, 16, "bold"))

    def pack(self, **kwargs: object) -> "Gauge":
        """Pack the frame and return self."""
        self.frame.pack(**kwargs)
        return self

    def grid(self, **kwargs: object) -> "Gauge":
        """Grid the frame and return self."""
        self.frame.grid(**kwargs)
        return self


class BarChart:
    """Horizontal bar chart for categorical counts (top gestures)."""

    def __init__(self, parent: object, theme: Theme, title: str = "",
                 rows: int = 6) -> None:
        self.theme = theme
        self.rows = rows

        self.frame = ctk.CTkFrame(
            parent, corner_radius=theme.corner_radius, fg_color=theme.surface,
            border_width=1, border_color=theme.border,
        )
        if title:
            ctk.CTkLabel(self.frame, text=title.upper(),
                         font=_font(theme, 10, "bold"),
                         text_color=theme.text_muted, anchor="w"
                         ).pack(fill="x", padx=14, pady=(10, 4))

        self.canvas = ctk.CTkCanvas(self.frame, height=rows * 26 + 10,
                                    bg=theme.surface, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def set_data(self, data: Sequence[Tuple[str, int]]) -> None:
        """Render ``(label, count)`` pairs, highest first."""
        canvas = self.canvas
        canvas.delete("all")
        width = canvas.winfo_width() or 380

        if not data:
            canvas.create_text(width / 2, 30, text="No gestures yet",
                               fill=self.theme.text_muted,
                               font=_font(self.theme, 10))
            return

        peak = max(count for _, count in data) or 1
        label_w = 118
        bar_area = max(width - label_w - 46, 40)

        for index, (label, count) in enumerate(data[:self.rows]):
            y = 14 + index * 26
            display = label if len(label) <= 16 else label[:15] + "…"
            canvas.create_text(6, y, text=display, anchor="w",
                               fill=self.theme.text_muted,
                               font=_font(self.theme, 10))

            bar_w = max(3, int(bar_area * count / peak))
            canvas.create_rectangle(label_w, y - 7, label_w + bar_w, y + 7,
                                    fill=self.theme.accent, outline="")
            canvas.create_text(label_w + bar_w + 8, y, text=str(count), anchor="w",
                               fill=self.theme.text, font=_font(self.theme, 10, "bold"))

    def pack(self, **kwargs: object) -> "BarChart":
        """Pack the frame and return self."""
        self.frame.pack(**kwargs)
        return self

    def grid(self, **kwargs: object) -> "BarChart":
        """Grid the frame and return self."""
        self.frame.grid(**kwargs)
        return self


class HeatmapWidget:
    """Renders the spatial gesture heatmap as a coloured grid."""

    def __init__(self, parent: object, theme: Theme, title: str = "Gesture Heatmap",
                 size: int = 200) -> None:
        self.theme = theme
        self.size = size

        self.frame = ctk.CTkFrame(
            parent, corner_radius=theme.corner_radius, fg_color=theme.surface,
            border_width=1, border_color=theme.border,
        )
        ctk.CTkLabel(self.frame, text=title.upper(), font=_font(theme, 10, "bold"),
                     text_color=theme.text_muted, anchor="w"
                     ).pack(fill="x", padx=14, pady=(10, 4))
        self.canvas = ctk.CTkCanvas(self.frame, width=size, height=int(size * 0.62),
                                    bg=theme.surface, highlightthickness=0, bd=0)
        self.canvas.pack(padx=12, pady=(0, 12))

    def set_data(self, grid: Sequence[Sequence[float]]) -> None:
        """Render a normalized 2D intensity grid."""
        canvas = self.canvas
        canvas.delete("all")

        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if not rows or not cols:
            return

        width = self.size
        height = int(self.size * 0.62)
        cell_w = width / cols
        cell_h = height / rows

        cold = self.theme.surface_alt
        warm = self.theme.accent
        hot = self.theme.error

        for r in range(rows):
            for c in range(cols):
                value = clamp(float(grid[r][c]), 0.0, 1.0)
                if value <= 0.001:
                    continue
                colour = (mix_colours(cold, warm, value * 2) if value < 0.5
                          else mix_colours(warm, hot, (value - 0.5) * 2))
                canvas.create_rectangle(
                    c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h,
                    fill=colour, outline="",
                )

    def pack(self, **kwargs: object) -> "HeatmapWidget":
        """Pack the frame and return self."""
        self.frame.pack(**kwargs)
        return self

    def grid(self, **kwargs: object) -> "HeatmapWidget":
        """Grid the frame and return self."""
        self.frame.grid(**kwargs)
        return self


class DashboardPanel:
    """Assembles the full dashboard view and refreshes it from app state."""

    def __init__(self, parent: object, theme: Theme) -> None:
        self.theme = theme
        self.root = ctk.CTkScrollableFrame(parent, fg_color="transparent")

        # -- headline tiles ------------------------------------------------ #
        tiles = ctk.CTkFrame(self.root, fg_color="transparent")
        tiles.pack(fill="x", pady=(0, 14))
        for column in range(4):
            tiles.grid_columnconfigure(column, weight=1, uniform="tile")

        self.tiles: Dict[str, StatTile] = {}
        specs = [
            ("fps", "Frame Rate", "fps", theme.success, True),
            ("confidence", "Confidence", "", theme.accent, True),
            ("cpu", "CPU", "%", theme.warning, True),
            ("memory", "Memory", "MB", theme.info, True),
            ("gesture", "Current Gesture", "", theme.secondary, False),
            ("mode", "Mode", "", theme.accent, False),
            ("clicks", "Clicks", "", theme.success, False),
            ("session", "Session", "", theme.text_muted, False),
        ]
        for index, (key, label, unit, colour, spark) in enumerate(specs):
            tile = StatTile(tiles, theme, label, unit=unit, accent=colour,
                            sparkline=spark)
            tile.grid(row=index // 4, column=index % 4, padx=5, pady=5, sticky="nsew")
            self.tiles[key] = tile

        # -- charts -------------------------------------------------------- #
        charts = ctk.CTkFrame(self.root, fg_color="transparent")
        charts.pack(fill="both", expand=True)
        charts.grid_columnconfigure(0, weight=3)
        charts.grid_columnconfigure(1, weight=2)

        self.fps_chart = Sparkline(charts, theme, "Frame Rate (last 2 min)",
                                   height=140, colour=theme.success, y_max=70)
        self.fps_chart.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.top_gestures = BarChart(charts, theme, "Most Used Gestures")
        self.top_gestures.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.latency_chart = Sparkline(charts, theme, "Frame Time (ms)",
                                       height=140, colour=theme.warning)
        self.latency_chart.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        side = ctk.CTkFrame(charts, fg_color="transparent")
        side.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.heatmap = HeatmapWidget(side, theme, "Gesture Heatmap", size=230)
        self.heatmap.pack(fill="both", expand=True)

        # -- system info --------------------------------------------------- #
        self.info_frame = ctk.CTkFrame(
            self.root, corner_radius=theme.corner_radius,
            fg_color=theme.surface, border_width=1, border_color=theme.border,
        )
        self.info_frame.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(self.info_frame, text="SYSTEM", font=_font(theme, 10, "bold"),
                     text_color=theme.text_muted, anchor="w"
                     ).pack(fill="x", padx=14, pady=(10, 4))
        self._info_label = ctk.CTkLabel(
            self.info_frame, text="", font=_font(theme, 11),
            text_color=theme.text_muted, anchor="w", justify="left",
        )
        self._info_label.pack(fill="x", padx=14, pady=(0, 12))

    def pack(self, **kwargs: object) -> "DashboardPanel":
        """Pack the root frame."""
        self.root.pack(**kwargs)
        return self

    def pack_forget(self) -> None:
        """Hide the panel."""
        self.root.pack_forget()

    def set_system_info(self, info: Dict[str, str]) -> None:
        """Populate the static system description."""
        self._info_label.configure(
            text="   ".join(f"{k}: {v}" for k, v in info.items()))

    def refresh(self, snapshot: object, engine_state: Dict[str, object],
                history_summary: Dict[str, object],
                top_gestures: Sequence[Tuple[str, int]],
                heatmap: Optional[Sequence[Sequence[float]]] = None) -> None:
        """Update every widget from the latest application state."""
        fps = float(getattr(snapshot, "fps", 0.0))
        cpu = float(getattr(snapshot, "cpu_percent", 0.0))
        memory = float(getattr(snapshot, "memory_mb", 0.0))
        frame_ms = float(getattr(snapshot, "frame_ms", 0.0))

        self.tiles["fps"].update(f"{fps:.0f}", fps,
                                 colour=_traffic(fps, 25, 15, self.theme))
        self.tiles["cpu"].update(f"{cpu:.0f}", cpu,
                                 colour=_traffic(100 - cpu, 40, 15, self.theme))
        self.tiles["memory"].update(f"{memory:.0f}", memory)

        confidence = float(engine_state.get("confidence", 0.0))  # type: ignore[arg-type]
        self.tiles["confidence"].update(
            f"{confidence:.0%}", confidence * 100,
            colour=_traffic(confidence, 0.8, 0.6, self.theme))

        self.tiles["gesture"].update(str(engine_state.get("pose", "—")))
        self.tiles["mode"].update(str(engine_state.get("mode", "—")))
        self.tiles["clicks"].update(str(engine_state.get("clicks", 0)))
        self.tiles["session"].update(str(engine_state.get("session", "0:00")))

        self.fps_chart.push(fps)
        self.latency_chart.push(frame_ms)
        self.top_gestures.set_data(top_gestures)
        if heatmap is not None:
            self.heatmap.set_data(heatmap)


def _traffic(value: float, good: float, warn: float, theme: Theme) -> str:
    """Green/amber/red colour for a value against two thresholds."""
    if value >= good:
        return theme.success
    if value >= warn:
        return theme.warning
    return theme.error
