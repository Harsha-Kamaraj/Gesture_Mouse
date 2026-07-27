"""Main application window.

A sidebar-plus-content shell hosting eight views: Live camera, Dashboard,
Gesture Library, Performance, Profiles, History, Logs and Settings.

Design notes
------------
The window owns *no* application logic.  It receives an ``AppServices`` bundle
of callbacks and state accessors, and everything it does goes back through
that interface.  Keeping the boundary strict is what allows the whole
recognition stack to run headless (``app.py --headless``) with no UI at all,
and it stops view code from quietly becoming the place business rules live.

Views are built lazily on first navigation.  Constructing all eight up front
costs noticeable startup time for screens the user may never open.
"""

from __future__ import annotations

import compat  # noqa: F401  # must precede customtkinter; see module docstring

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import customtkinter as ctk

from config import APP_NAME, APP_VERSION
from dashboard import DashboardPanel, Gauge, Sparkline
from logger import get_logger, ring_handler
from notifications import ToastManager
from themes import Theme, ThemeManager

log = get_logger(__name__)


def _font(theme: Theme, size: int, weight: str = "normal") -> tuple:
    """Font tuple from the theme, honouring the large-text setting."""
    return (theme.font_family, size, weight)


@dataclass
class AppServices:
    """Callbacks and accessors the UI uses to talk to the application.

    Every field is optional so the window can be constructed and inspected in
    isolation (which is how the UI is smoke-tested).
    """

    # State accessors
    get_frame: Optional[Callable[[], object]] = None
    get_performance: Optional[Callable[[], object]] = None
    get_engine_state: Optional[Callable[[], Dict[str, object]]] = None
    get_history: Optional[Callable[[], object]] = None
    get_config: Optional[Callable[[], object]] = None
    get_capabilities: Optional[Callable[[], Dict[str, bool]]] = None
    get_system_info: Optional[Callable[[], Dict[str, str]]] = None

    # Commands
    toggle_tracking: Optional[Callable[[], bool]] = None
    start_calibration: Optional[Callable[[], None]] = None
    emergency_stop: Optional[Callable[[], None]] = None
    update_setting: Optional[Callable[[str, str, object], None]] = None
    set_theme: Optional[Callable[[str], None]] = None
    set_profile: Optional[Callable[[str], None]] = None
    list_profiles: Optional[Callable[[], List[str]]] = None
    create_profile: Optional[Callable[[str], None]] = None
    delete_profile: Optional[Callable[[str], None]] = None
    duplicate_profile: Optional[Callable[[str], None]] = None
    export_profile: Optional[Callable[[str, Path], None]] = None
    import_profile: Optional[Callable[[Path], None]] = None
    list_gestures: Optional[Callable[[], List[Dict[str, object]]]] = None
    list_actions: Optional[Callable[[], Dict[str, str]]] = None
    bind_gesture: Optional[Callable[[str, str], None]] = None
    toggle_gesture: Optional[Callable[[str, bool], None]] = None
    record_gesture: Optional[Callable[[str], None]] = None
    delete_gesture: Optional[Callable[[str], None]] = None
    duplicate_gesture: Optional[Callable[[str], None]] = None
    export_gestures: Optional[Callable[[Path], int]] = None
    import_gestures: Optional[Callable[[Path], int]] = None
    export_history: Optional[Callable[[Path], int]] = None
    clear_history: Optional[Callable[[], None]] = None
    on_close: Optional[Callable[[], None]] = None


NAV_ITEMS: List[Tuple[str, str, str]] = [
    ("live", "Live", "▶"),
    ("dashboard", "Dashboard", "▦"),
    ("gestures", "Gestures", "✋"),
    ("performance", "Performance", "◱"),
    ("profiles", "Profiles", "◍"),
    ("history", "History", "≡"),
    ("logs", "Logs", "◧"),
    ("settings", "Settings", "⚙"),
]

#: Log level -> theme token, used to colour rows in the Logs view.
_LOG_LEVEL_TOKENS: Dict[str, str] = {
    "DEBUG": "text_muted",
    "INFO": "text",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "error",
}


class MainWindow(ctk.CTk):
    """The application shell."""

    #: UI refresh interval in milliseconds (video is refreshed separately).
    REFRESH_MS = 500
    VIDEO_MS = 33

    def __init__(self, services: AppServices, theme_manager: ThemeManager) -> None:
        super().__init__()
        self.services = services
        self.theme_manager = theme_manager
        self.theme = theme_manager.theme

        ctk.set_appearance_mode(self.theme.appearance)
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1360x820")
        self.minsize(1080, 680)
        self.configure(fg_color=self.theme.background)

        self._views: Dict[str, ctk.CTkFrame] = {}
        self._nav_buttons: Dict[str, ctk.CTkButton] = {}
        self._active_view = ""
        self._video_label: Optional[ctk.CTkLabel] = None
        self._photo_ref: Optional[object] = None
        self._running = True
        self._widgets: Dict[str, object] = {}
        self._log_filter = "INFO"
        self._log_paused = False

        self.toasts = ToastManager(self, self.theme)

        self._build_shell()
        self.show_view("live")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.REFRESH_MS, self._refresh_loop)
        self.after(self.VIDEO_MS, self._video_loop)

    # -- shell ------------------------------------------------------------ #

    def _build_shell(self) -> None:
        """Construct the sidebar, header and content area."""
        theme = self.theme

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- sidebar ------------------------------------------------------- #
        sidebar = ctk.CTkFrame(self, width=216, corner_radius=0,
                               fg_color=theme.surface)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=18, pady=(22, 26))
        ctk.CTkLabel(brand, text="◈", font=_font(theme, 26, "bold"),
                     text_color=theme.accent).pack(side="left")
        title_box = ctk.CTkFrame(brand, fg_color="transparent")
        title_box.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(title_box, text="Gesture Mouse", anchor="w",
                     font=_font(theme, 14, "bold"),
                     text_color=theme.text).pack(anchor="w")
        ctk.CTkLabel(title_box, text=f"PRO  v{APP_VERSION}", anchor="w",
                     font=_font(theme, 9, "bold"),
                     text_color=theme.accent).pack(anchor="w")

        for key, label, icon in NAV_ITEMS:
            button = ctk.CTkButton(
                sidebar, text=f"  {icon}   {label}", anchor="w", height=42,
                corner_radius=theme.corner_radius - 2,
                fg_color="transparent", hover_color=theme.surface_alt,
                text_color=theme.text_muted, font=_font(theme, 13),
                command=lambda k=key: self.show_view(k),
            )
            button.pack(fill="x", padx=12, pady=2)
            self._nav_buttons[key] = button

        # Status block pinned to the bottom of the sidebar.
        status = ctk.CTkFrame(sidebar, fg_color=theme.surface_alt,
                              corner_radius=theme.corner_radius)
        status.pack(side="bottom", fill="x", padx=12, pady=12)

        self._widgets["status_dot"] = ctk.CTkLabel(
            status, text="●", font=_font(theme, 14),
            text_color=theme.text_muted)
        self._widgets["status_dot"].pack(side="left", padx=(12, 6), pady=10)
        self._widgets["status_text"] = ctk.CTkLabel(
            status, text="Starting…", font=_font(theme, 11),
            text_color=theme.text_muted, anchor="w")
        self._widgets["status_text"].pack(side="left", pady=10)

        self._widgets["stop_button"] = ctk.CTkButton(
            sidebar, text="Emergency Stop", height=36,
            fg_color=theme.error, hover_color=theme.error,
            font=_font(theme, 12, "bold"),
            corner_radius=theme.corner_radius - 2,
            command=self._emergency_stop,
        )
        self._widgets["stop_button"].pack(side="bottom", fill="x", padx=12, pady=(0, 4))

        # -- content ------------------------------------------------------- #
        self.content = ctk.CTkFrame(self, fg_color=theme.background,
                                    corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.content, height=64, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))

        self._widgets["view_title"] = ctk.CTkLabel(
            header, text="Live", font=_font(theme, 22, "bold"),
            text_color=theme.text, anchor="w")
        self._widgets["view_title"].pack(side="left")

        self._widgets["profile_chip"] = ctk.CTkLabel(
            header, text="Default", font=_font(theme, 11, "bold"),
            text_color=theme.accent, fg_color=theme.surface,
            corner_radius=10, padx=12, pady=5)
        self._widgets["profile_chip"].pack(side="right")

        self._widgets["fps_chip"] = ctk.CTkLabel(
            header, text="— FPS", font=_font(theme, 11, "bold"),
            text_color=theme.text_muted, fg_color=theme.surface,
            corner_radius=10, padx=12, pady=5)
        self._widgets["fps_chip"].pack(side="right", padx=(0, 8))

        self.view_container = ctk.CTkFrame(self.content, fg_color="transparent")
        self.view_container.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 20))

    # -- navigation ------------------------------------------------------- #

    def show_view(self, key: str) -> None:
        """Switch to a view, building it lazily on first use."""
        if key == self._active_view:
            return

        for view in self._views.values():
            view.pack_forget()

        if key not in self._views:
            builder = getattr(self, f"_build_{key}", None)
            if builder is None:
                log.warning("no builder for view %r", key)
                return
            self._views[key] = builder()

        self._views[key].pack(fill="both", expand=True)
        self._active_view = key

        theme = self.theme
        for nav_key, button in self._nav_buttons.items():
            active = nav_key == key
            button.configure(
                fg_color=theme.accent if active else "transparent",
                text_color="#FFFFFF" if active else theme.text_muted,
            )

        label = next((l for k, l, _ in NAV_ITEMS if k == key), key.title())
        self._widgets["view_title"].configure(text=label)  # type: ignore[attr-defined]

    # -- views ------------------------------------------------------------ #

    def _build_live(self) -> ctk.CTkFrame:
        """Camera feed with the primary controls."""
        theme = self.theme
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")

        video_card = ctk.CTkFrame(frame, fg_color=theme.surface,
                                  corner_radius=theme.corner_radius,
                                  border_width=1, border_color=theme.border)
        video_card.pack(fill="both", expand=True)

        self._video_label = ctk.CTkLabel(video_card, text="Starting camera…",
                                         font=_font(theme, 13),
                                         text_color=theme.text_muted)
        self._video_label.pack(fill="both", expand=True, padx=10, pady=10)

        controls = ctk.CTkFrame(frame, fg_color="transparent")
        controls.pack(fill="x", pady=(14, 0))

        self._widgets["track_button"] = ctk.CTkButton(
            controls, text="Pause Tracking", width=160, height=40,
            font=_font(theme, 13, "bold"), fg_color=theme.accent,
            hover_color=theme.accent_hover,
            corner_radius=theme.corner_radius - 2,
            command=self._toggle_tracking,
        )
        self._widgets["track_button"].pack(side="left")

        ctk.CTkButton(
            controls, text="Calibrate", width=130, height=40,
            font=_font(theme, 13), fg_color=theme.surface_alt,
            hover_color=theme.border, text_color=theme.text,
            corner_radius=theme.corner_radius - 2,
            command=self._start_calibration,
        ).pack(side="left", padx=10)

        self._widgets["live_hint"] = ctk.CTkLabel(
            controls,
            text="Point to move  ·  Pinch to click  ·  Hold pinch to drag  "
                 "·  Open palm 1s to pause",
            font=_font(theme, 11), text_color=theme.text_muted)
        self._widgets["live_hint"].pack(side="right")
        return frame

    def _build_dashboard(self) -> ctk.CTkFrame:
        """Metrics dashboard."""
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")
        panel = DashboardPanel(frame, self.theme)
        panel.pack(fill="both", expand=True)
        self._widgets["dashboard"] = panel

        if self.services.get_system_info:
            try:
                panel.set_system_info(self.services.get_system_info())
            except Exception as exc:
                log.debug("system info failed: %s", exc)
        return frame

    def _build_gestures(self) -> ctk.CTkFrame:
        """Gesture library: bindings, enable/disable and recording."""
        theme = self.theme
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")

        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 12))

        self._widgets["record_name"] = ctk.CTkEntry(
            toolbar, placeholder_text="New gesture name…", width=220, height=36,
            corner_radius=theme.corner_radius - 4)
        self._widgets["record_name"].pack(side="left")

        ctk.CTkButton(
            toolbar, text="Record Gesture", height=36, width=150,
            font=_font(theme, 12, "bold"), fg_color=theme.accent,
            hover_color=theme.accent_hover,
            corner_radius=theme.corner_radius - 4,
            command=self._record_gesture,
        ).pack(side="left", padx=10)

        ctk.CTkLabel(
            toolbar,
            text="Draw the shape in the air 3 times with your index finger",
            font=_font(theme, 11), text_color=theme.text_muted,
        ).pack(side="left")

        ctk.CTkButton(
            toolbar, text="Import", height=36, width=90, font=_font(theme, 12),
            fg_color=theme.surface_alt, hover_color=theme.border,
            text_color=theme.text, corner_radius=theme.corner_radius - 4,
            command=self._import_gestures,
        ).pack(side="right")

        ctk.CTkButton(
            toolbar, text="Export", height=36, width=90, font=_font(theme, 12),
            fg_color=theme.surface_alt, hover_color=theme.border,
            text_color=theme.text, corner_radius=theme.corner_radius - 4,
            command=self._export_gestures,
        ).pack(side="right", padx=8)

        self._widgets["gesture_list"] = ctk.CTkScrollableFrame(
            frame, fg_color=theme.surface, corner_radius=theme.corner_radius,
            border_width=1, border_color=theme.border)
        self._widgets["gesture_list"].pack(fill="both", expand=True)

        self.refresh_gesture_list()
        return frame

    def _build_performance(self) -> ctk.CTkFrame:
        """Performance charts and gauges."""
        theme = self.theme
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")

        gauges = ctk.CTkFrame(frame, fg_color="transparent")
        gauges.pack(fill="x", pady=(0, 12))
        for i in range(4):
            gauges.grid_columnconfigure(i, weight=1, uniform="g")

        for index, (key, label, colour) in enumerate([
            ("cpu", "CPU", theme.warning),
            ("memory", "Memory", theme.info),
            ("fps", "Frame Rate", theme.success),
            ("confidence", "Confidence", theme.accent),
        ]):
            gauge = Gauge(gauges, theme, label, colour=colour)
            gauge.grid(row=0, column=index, padx=5, sticky="nsew")
            self._widgets[f"gauge_{key}"] = gauge

        charts = ctk.CTkFrame(frame, fg_color="transparent")
        charts.pack(fill="both", expand=True)
        charts.grid_columnconfigure(0, weight=1)
        charts.grid_columnconfigure(1, weight=1)

        self._widgets["perf_frame_chart"] = Sparkline(
            charts, theme, "Frame Time (ms)", height=150,
            colour=theme.warning).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self._widgets["perf_infer_chart"] = Sparkline(
            charts, theme, "Inference Time (ms)", height=150,
            colour=theme.secondary).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        detail = ctk.CTkFrame(charts, fg_color=theme.surface,
                              corner_radius=theme.corner_radius,
                              border_width=1, border_color=theme.border)
        detail.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(detail, text="LATENCY DISTRIBUTION",
                     font=_font(theme, 10, "bold"), text_color=theme.text_muted,
                     anchor="w").pack(fill="x", padx=14, pady=(10, 4))
        self._widgets["perf_detail"] = ctk.CTkLabel(
            detail, text="Collecting…", font=(theme.font_family_mono, 11),
            text_color=theme.text, anchor="w", justify="left")
        self._widgets["perf_detail"].pack(fill="x", padx=14, pady=(0, 12))
        return frame

    def _build_profiles(self) -> ctk.CTkFrame:
        """Profile switching and management."""
        theme = self.theme
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")

        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 12))

        self._widgets["profile_name"] = ctk.CTkEntry(
            toolbar, placeholder_text="New profile name…", width=220, height=36,
            corner_radius=theme.corner_radius - 4)
        self._widgets["profile_name"].pack(side="left")

        ctk.CTkButton(toolbar, text="Create", height=36, width=110,
                      font=_font(theme, 12, "bold"), fg_color=theme.accent,
                      hover_color=theme.accent_hover,
                      corner_radius=theme.corner_radius - 4,
                      command=self._create_profile).pack(side="left", padx=10)

        ctk.CTkButton(toolbar, text="Import", height=36, width=90,
                      font=_font(theme, 12), fg_color=theme.surface_alt,
                      hover_color=theme.border, text_color=theme.text,
                      corner_radius=theme.corner_radius - 4,
                      command=self._import_profile).pack(side="right")

        self._widgets["profile_list"] = ctk.CTkScrollableFrame(
            frame, fg_color=theme.surface, corner_radius=theme.corner_radius,
            border_width=1, border_color=theme.border)
        self._widgets["profile_list"].pack(fill="both", expand=True)

        self.refresh_profile_list()
        return frame

    def _build_history(self) -> ctk.CTkFrame:
        """Gesture history table with export."""
        theme = self.theme
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")

        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 12))
        ctk.CTkButton(toolbar, text="Export CSV", height=36, width=130,
                      font=_font(theme, 12, "bold"), fg_color=theme.accent,
                      hover_color=theme.accent_hover,
                      corner_radius=theme.corner_radius - 4,
                      command=self._export_history).pack(side="left")
        ctk.CTkButton(toolbar, text="Clear", height=36, width=100,
                      font=_font(theme, 12), fg_color=theme.surface_alt,
                      hover_color=theme.border, text_color=theme.text,
                      corner_radius=theme.corner_radius - 4,
                      command=self._clear_history).pack(side="left", padx=10)
        self._widgets["history_summary"] = ctk.CTkLabel(
            toolbar, text="", font=_font(theme, 11),
            text_color=theme.text_muted)
        self._widgets["history_summary"].pack(side="right")

        header = ctk.CTkFrame(frame, fg_color=theme.surface_alt,
                              corner_radius=theme.corner_radius - 4, height=32)
        header.pack(fill="x")
        for text, width in (("TIME", 90), ("GESTURE", 170), ("ACTION", 190),
                            ("CONF", 70), ("MODE", 110)):
            ctk.CTkLabel(header, text=text, width=width, anchor="w",
                         font=_font(theme, 9, "bold"),
                         text_color=theme.text_muted).pack(side="left", padx=(12, 0),
                                                           pady=7)

        self._widgets["history_list"] = ctk.CTkScrollableFrame(
            frame, fg_color=theme.surface, corner_radius=theme.corner_radius,
            border_width=1, border_color=theme.border)
        self._widgets["history_list"].pack(fill="both", expand=True, pady=(4, 0))
        return frame

    def _build_logs(self) -> ctk.CTkFrame:
        """Live log viewer fed by the in-memory ring buffer.

        Reads from :data:`logger.ring_handler` rather than tailing the log
        file, so opening this view costs no disk I/O and shows records the
        instant they are emitted — including ones from the pipeline thread.
        """
        theme = self.theme
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")

        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 12))

        self._log_paused = False

        def toggle_pause() -> None:
            """Freeze or resume the live tail."""
            self._log_paused = not self._log_paused
            pause_button.configure(
                text="Resume" if self._log_paused else "Pause",
                fg_color=theme.warning if self._log_paused else theme.surface_alt,
                text_color="#FFFFFF" if self._log_paused else theme.text,
            )

        pause_button = ctk.CTkButton(
            toolbar, text="Pause", height=36, width=100, font=_font(theme, 12),
            fg_color=theme.surface_alt, hover_color=theme.border,
            text_color=theme.text, corner_radius=theme.corner_radius - 4,
            command=toggle_pause,
        )
        pause_button.pack(side="left")

        ctk.CTkButton(
            toolbar, text="Clear", height=36, width=100, font=_font(theme, 12),
            fg_color=theme.surface_alt, hover_color=theme.border,
            text_color=theme.text, corner_radius=theme.corner_radius - 4,
            command=self._clear_logs,
        ).pack(side="left", padx=10)

        level_menu = ctk.CTkOptionMenu(
            toolbar, values=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
            width=130, height=36, fg_color=theme.surface_alt,
            button_color=theme.accent, font=_font(theme, 12),
            command=self._set_log_level_filter,
        )
        level_menu.set("INFO")
        level_menu.pack(side="left")
        self._log_filter = "INFO"

        ctk.CTkLabel(
            toolbar, text=f"Log file: logs/{APP_NAME.lower().replace(' ', '-')}.log",
            font=_font(theme, 11), text_color=theme.text_muted,
        ).pack(side="right")

        self._widgets["log_box"] = ctk.CTkTextbox(
            frame, fg_color=theme.surface, corner_radius=theme.corner_radius,
            border_width=1, border_color=theme.border,
            font=(theme.font_family_mono, 11), text_color=theme.text,
            wrap="none",
        )
        self._widgets["log_box"].pack(fill="both", expand=True)

        # Colour tags, one per level.
        box = self._widgets["log_box"]
        for level, token in _LOG_LEVEL_TOKENS.items():
            try:
                box.tag_config(level, foreground=theme.token(token))  # type: ignore[attr-defined]
            except Exception:
                pass

        self._render_log_snapshot()
        return frame

    def _log_passes_filter(self, line: str) -> bool:
        """Whether a formatted record clears the level filter."""
        if self._log_filter == "ALL":
            return True
        order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        try:
            minimum = order.index(self._log_filter)
        except ValueError:
            return True
        for index, level in enumerate(order):
            if f"| {level:<8}|" in line or f"| {level} " in line:
                return index >= minimum
        return True

    @staticmethod
    def _level_of(line: str) -> str:
        """Extract the level name from a formatted record."""
        for level in ("CRITICAL", "ERROR", "WARNING", "DEBUG", "INFO"):
            if level in line:
                return level
        return "INFO"

    def _render_log_snapshot(self) -> None:
        """Fill the log box from the ring buffer."""
        box = self._widgets.get("log_box")
        if box is None:
            return
        try:
            box.configure(state="normal")            # type: ignore[attr-defined]
            box.delete("1.0", "end")                 # type: ignore[attr-defined]
            for line in ring_handler.snapshot():
                if self._log_passes_filter(line):
                    box.insert("end", line + "\n", self._level_of(line))  # type: ignore[attr-defined]
            box.see("end")                           # type: ignore[attr-defined]
            box.configure(state="disabled")          # type: ignore[attr-defined]
        except Exception as exc:
            log.debug("log render failed: %s", exc)

    def _set_log_level_filter(self, level: str) -> None:
        """Change the minimum displayed level."""
        self._log_filter = level
        self._render_log_snapshot()

    def _clear_logs(self) -> None:
        """Empty the in-memory buffer and the view."""
        ring_handler.records.clear()
        self._render_log_snapshot()

    def _build_settings(self) -> ctk.CTkFrame:
        """Settings across camera, cursor, gestures, appearance and features."""
        theme = self.theme
        frame = ctk.CTkFrame(self.view_container, fg_color="transparent")
        scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        config = self.services.get_config() if self.services.get_config else None

        def section(title: str) -> ctk.CTkFrame:
            """Create a titled settings card."""
            card = ctk.CTkFrame(scroll, fg_color=theme.surface,
                                corner_radius=theme.corner_radius,
                                border_width=1, border_color=theme.border)
            card.pack(fill="x", pady=(0, 12))
            ctk.CTkLabel(card, text=title.upper(), font=_font(theme, 10, "bold"),
                         text_color=theme.text_muted, anchor="w"
                         ).pack(fill="x", padx=18, pady=(14, 8))
            return card

        def slider(parent: ctk.CTkFrame, sect: str, key: str, label: str,
                   low: float, high: float, value: float,
                   fmt: str = "{:.2f}") -> None:
            """Add a labelled slider bound to a config field."""
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=5)
            ctk.CTkLabel(row, text=label, width=190, anchor="w",
                         font=_font(theme, 12),
                         text_color=theme.text).pack(side="left")
            readout = ctk.CTkLabel(row, text=fmt.format(value), width=64,
                                   font=(theme.font_family_mono, 11),
                                   text_color=theme.accent)
            readout.pack(side="right")

            def on_change(new_value: float) -> None:
                readout.configure(text=fmt.format(new_value))
                self._update_setting(sect, key, float(new_value))

            control = ctk.CTkSlider(row, from_=low, to=high, command=on_change,
                                    button_color=theme.accent,
                                    progress_color=theme.accent, height=16)
            control.set(value)
            control.pack(side="right", fill="x", expand=True, padx=12)

        def switch(parent: ctk.CTkFrame, sect: str, key: str, label: str,
                   value: bool) -> None:
            """Add a labelled toggle bound to a config field."""
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=5)
            ctk.CTkLabel(row, text=label, anchor="w", font=_font(theme, 12),
                         text_color=theme.text).pack(side="left")
            control = ctk.CTkSwitch(
                row, text="", progress_color=theme.accent, width=44,
                command=lambda: self._update_setting(sect, key,
                                                     bool(control.get())))
            control.select() if value else control.deselect()
            control.pack(side="right")

        cursor = section("Cursor")
        cur = getattr(config, "cursor", None)
        slider(cursor, "cursor", "sensitivity", "Sensitivity", 0.3, 2.5,
               getattr(cur, "sensitivity", 1.0))
        slider(cursor, "cursor", "speed", "Speed", 0.3, 2.5,
               getattr(cur, "speed", 1.0))
        slider(cursor, "cursor", "smoothing", "Smoothing", 0.0, 0.95,
               getattr(cur, "smoothing", 0.55))
        slider(cursor, "cursor", "dead_zone", "Dead Zone", 0.0, 0.03,
               getattr(cur, "dead_zone", 0.004), "{:.4f}")
        slider(cursor, "cursor", "prediction_time", "Motion Prediction", 0.0, 0.12,
               getattr(cur, "prediction_time", 0.035), "{:.3f}")
        slider(cursor, "cursor", "active_region_margin", "Active Region", 0.0, 0.40,
               getattr(cur, "active_region_margin", 0.16))
        slider(cursor, "cursor", "scroll_speed", "Scroll Speed", 0.5, 8.0,
               getattr(cur, "scroll_speed", 3.0), "{:.1f}")
        switch(cursor, "cursor", "natural_scroll", "Natural Scrolling",
               getattr(cur, "natural_scroll", False))

        gestures = section("Gesture Recognition")
        ges = getattr(config, "gestures", None)
        slider(gestures, "gestures", "min_confidence", "Minimum Confidence",
               0.4, 0.98, getattr(ges, "min_confidence", 0.72))
        slider(gestures, "gestures", "pinch_threshold", "Pinch Threshold",
               0.02, 0.20, getattr(ges, "pinch_threshold", 0.055), "{:.3f}")
        slider(gestures, "gestures", "global_cooldown", "Cooldown (s)",
               0.05, 1.5, getattr(ges, "global_cooldown", 0.28))
        slider(gestures, "gestures", "drag_hold_time", "Drag Hold (s)",
               0.15, 1.2, getattr(ges, "drag_hold_time", 0.45))
        slider(gestures, "gestures", "stability_frames", "Stability Frames",
               1, 8, getattr(ges, "stability_frames", 3), "{:.0f}")

        camera = section("Camera")
        cam = getattr(config, "camera", None)
        slider(camera, "camera", "target_fps", "Target FPS", 15, 60,
               getattr(cam, "target_fps", 30), "{:.0f}")
        slider(camera, "camera", "inference_scale", "Inference Scale", 0.3, 1.0,
               getattr(cam, "inference_scale", 0.6))
        switch(camera, "camera", "mirror", "Mirror Image",
               getattr(cam, "mirror", True))

        appearance = section("Appearance")
        row = ctk.CTkFrame(appearance, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=5)
        ctk.CTkLabel(row, text="Theme", anchor="w", font=_font(theme, 12),
                     text_color=theme.text).pack(side="left")
        names = self.theme_manager.names
        selector = ctk.CTkOptionMenu(
            row, values=[self.theme_manager.display_names().get(n, n) for n in names],
            width=190, fg_color=theme.surface_alt, button_color=theme.accent,
            command=lambda label: self._set_theme_by_label(label),
        )
        selector.set(self.theme.display_name)
        selector.pack(side="right")

        ui = getattr(config, "ui", None)
        switch(appearance, "ui", "show_landmarks", "Show Landmarks",
               getattr(ui, "show_landmarks", True))
        switch(appearance, "ui", "show_skeleton", "Show Skeleton",
               getattr(ui, "show_skeleton", True))
        switch(appearance, "ui", "high_contrast", "High Contrast",
               getattr(ui, "high_contrast", False))
        switch(appearance, "ui", "large_text", "Large Text",
               getattr(ui, "large_text", False))

        features = section("Features")
        feat = getattr(config, "features", None)
        switch(features, "features", "sound_effects", "Sound Effects",
               getattr(feat, "sound_effects", True))
        switch(features, "features", "toast_notifications", "Notifications",
               getattr(feat, "toast_notifications", True))
        switch(features, "features", "voice_commands", "Voice Commands",
               getattr(feat, "voice_commands", False))
        switch(features, "features", "left_handed_mode", "Left-Handed Mode",
               getattr(feat, "left_handed_mode", False))
        switch(features, "features", "face_unlock", "Presence Detection",
               getattr(feat, "face_unlock", False))
        switch(features, "features", "plugins_enabled", "Plugins",
               getattr(feat, "plugins_enabled", True))

        capabilities = section("System Capabilities")
        caps = self.services.get_capabilities() if self.services.get_capabilities else {}
        for name, available in (caps or {}).items():
            row = ctk.CTkFrame(capabilities, fg_color="transparent")
            row.pack(fill="x", padx=18, pady=3)
            ctk.CTkLabel(row, text=name.replace("_", " ").title(), anchor="w",
                         font=_font(theme, 12),
                         text_color=theme.text).pack(side="left")
            ctk.CTkLabel(row, text="Available" if available else "Unavailable",
                         font=_font(theme, 11, "bold"),
                         text_color=theme.success if available else theme.text_muted
                         ).pack(side="right")
        ctk.CTkFrame(capabilities, fg_color="transparent", height=8).pack()
        return frame

    # -- list refreshers -------------------------------------------------- #

    def refresh_gesture_list(self) -> None:
        """Rebuild the gesture library rows."""
        container = self._widgets.get("gesture_list")
        if container is None or not self.services.list_gestures:
            return

        for child in container.winfo_children():  # type: ignore[attr-defined]
            child.destroy()

        theme = self.theme
        try:
            gestures = self.services.list_gestures()
        except Exception as exc:
            log.warning("could not list gestures: %s", exc)
            return

        actions = self.services.list_actions() if self.services.list_actions else {}
        action_labels = list(actions.values()) or ["Do Nothing"]
        label_to_id = {v: k for k, v in actions.items()}

        for gesture in gestures:
            name = str(gesture.get("name", ""))
            row = ctk.CTkFrame(container, fg_color=theme.surface_alt,  # type: ignore[arg-type]
                               corner_radius=theme.corner_radius - 4)
            row.pack(fill="x", padx=8, pady=4)

            toggle = ctk.CTkSwitch(
                row, text="", width=42, progress_color=theme.accent,
                command=lambda n=name, t=None: self._toggle_gesture(n, t))
            toggle.select() if gesture.get("enabled", True) else toggle.deselect()
            toggle.configure(command=lambda n=name, w=toggle:
                             self._toggle_gesture(n, bool(w.get())))
            toggle.pack(side="left", padx=(12, 6), pady=10)

            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(info, text=str(gesture.get("label", name)), anchor="w",
                         font=_font(theme, 13, "bold"),
                         text_color=theme.text).pack(anchor="w")
            ctk.CTkLabel(info, text=str(gesture.get("description", "")), anchor="w",
                         font=_font(theme, 10),
                         text_color=theme.text_muted).pack(anchor="w")

            if gesture.get("custom"):
                ctk.CTkLabel(row, text="CUSTOM", font=_font(theme, 9, "bold"),
                             text_color=theme.secondary).pack(side="left", padx=8)
                ctk.CTkButton(
                    row, text="Delete", width=72, height=28,
                    font=_font(theme, 11), fg_color="transparent",
                    hover_color=theme.error, text_color=theme.text_muted,
                    border_width=1, border_color=theme.border,
                    command=lambda n=name: self._delete_gesture(n),
                ).pack(side="right", padx=(6, 12))
                ctk.CTkButton(
                    row, text="Duplicate", width=84, height=28,
                    font=_font(theme, 11), fg_color="transparent",
                    hover_color=theme.surface, text_color=theme.text_muted,
                    border_width=1, border_color=theme.border,
                    command=lambda n=name: self._duplicate_gesture(n),
                ).pack(side="right")

            current = actions.get(str(gesture.get("action", "none")), "Do Nothing")
            menu = ctk.CTkOptionMenu(
                row, values=action_labels, width=190, height=30,
                fg_color=theme.surface, button_color=theme.accent,
                font=_font(theme, 11),
                command=lambda label, n=name: self._bind_gesture(
                    n, label_to_id.get(label, "none")),
            )
            menu.set(current)
            menu.pack(side="right", padx=8, pady=10)

    def refresh_profile_list(self) -> None:
        """Rebuild the profile rows."""
        container = self._widgets.get("profile_list")
        if container is None or not self.services.list_profiles:
            return

        for child in container.winfo_children():  # type: ignore[attr-defined]
            child.destroy()

        theme = self.theme
        config = self.services.get_config() if self.services.get_config else None
        active = getattr(config, "profile_name", "")

        for name in self.services.list_profiles():
            is_active = name == active
            row = ctk.CTkFrame(
                container,  # type: ignore[arg-type]
                fg_color=theme.surface_alt,
                corner_radius=theme.corner_radius - 4,
                border_width=2 if is_active else 0,
                border_color=theme.accent,
            )
            row.pack(fill="x", padx=8, pady=4)

            ctk.CTkLabel(row, text=name, anchor="w",
                         font=_font(theme, 13, "bold"),
                         text_color=theme.accent if is_active else theme.text
                         ).pack(side="left", padx=16, pady=12)

            if is_active:
                ctk.CTkLabel(row, text="ACTIVE", font=_font(theme, 9, "bold"),
                             text_color=theme.success).pack(side="left")
            else:
                ctk.CTkButton(
                    row, text="Activate", width=90, height=30,
                    font=_font(theme, 11, "bold"), fg_color=theme.accent,
                    hover_color=theme.accent_hover,
                    command=lambda n=name: self._set_profile(n),
                ).pack(side="right", padx=(6, 12), pady=10)
                ctk.CTkButton(
                    row, text="Delete", width=72, height=30,
                    font=_font(theme, 11), fg_color="transparent",
                    hover_color=theme.error, text_color=theme.text_muted,
                    border_width=1, border_color=theme.border,
                    command=lambda n=name: self._delete_profile(n),
                ).pack(side="right", pady=10)

            # Duplicate and export stay available for the active profile too:
            # duplicating the profile you are tuning is the common case.
            ctk.CTkButton(
                row, text="Export", width=76, height=30,
                font=_font(theme, 11), fg_color="transparent",
                hover_color=theme.surface, text_color=theme.text_muted,
                border_width=1, border_color=theme.border,
                command=lambda n=name: self._export_profile(n),
            ).pack(side="right", padx=6, pady=10)
            ctk.CTkButton(
                row, text="Duplicate", width=88, height=30,
                font=_font(theme, 11), fg_color="transparent",
                hover_color=theme.surface, text_color=theme.text_muted,
                border_width=1, border_color=theme.border,
                command=lambda n=name: self._duplicate_profile(n),
            ).pack(side="right", pady=10)

    def refresh_history_list(self) -> None:
        """Rebuild the history rows from the newest entries."""
        container = self._widgets.get("history_list")
        if container is None or not self.services.get_history:
            return

        try:
            history = self.services.get_history()
            entries = history.recent(60)  # type: ignore[attr-defined]
            summary = history.summary()   # type: ignore[attr-defined]
        except Exception as exc:
            log.debug("history refresh failed: %s", exc)
            return

        self._widgets["history_summary"].configure(  # type: ignore[attr-defined]
            text=f"{summary.get('total', 0)} gestures  ·  "
                 f"{summary.get('mean_confidence', 0):.0%} mean confidence  ·  "
                 f"{summary.get('gestures_per_minute', 0)}/min")

        for child in container.winfo_children():  # type: ignore[attr-defined]
            child.destroy()

        theme = self.theme
        for entry in entries:
            row = ctk.CTkFrame(container, fg_color="transparent")  # type: ignore[arg-type]
            row.pack(fill="x", padx=4)
            cells = (
                (entry.time_string, 90, theme.text_muted),
                (entry.gesture, 170, theme.text),
                (entry.action, 190, theme.text_muted),
                (f"{entry.confidence:.0%}", 70,
                 theme.success if entry.confidence >= 0.8 else theme.warning),
                (entry.mode, 110, theme.text_muted),
            )
            for text, width, colour in cells:
                ctk.CTkLabel(row, text=text, width=width, anchor="w",
                             font=_font(theme, 11), text_color=colour
                             ).pack(side="left", padx=(12, 0), pady=3)

    # -- periodic refresh ------------------------------------------------- #

    def _video_loop(self) -> None:
        """Pull the latest annotated frame and display it."""
        if not self._running:
            return
        try:
            if self._video_label is not None and self.services.get_frame:
                image = self.services.get_frame()
                if image is not None:
                    self._photo_ref = image  # keep a reference; Tk will not
                    self._video_label.configure(image=image, text="")
        except Exception as exc:
            log.debug("video refresh failed: %s", exc)

        self.after(self.VIDEO_MS, self._video_loop)

    def _refresh_loop(self) -> None:
        """Refresh whichever view is visible, plus the shared chrome."""
        if not self._running:
            return
        try:
            self._refresh_chrome()
            if self._active_view == "dashboard":
                self._refresh_dashboard()
            elif self._active_view == "performance":
                self._refresh_performance()
            elif self._active_view == "history":
                self.refresh_history_list()
            elif self._active_view == "logs" and not self._log_paused:
                self._render_log_snapshot()
        except Exception as exc:
            log.debug("refresh failed: %s", exc)

        self.after(self.REFRESH_MS, self._refresh_loop)

    def _refresh_chrome(self) -> None:
        """Update the header chips and sidebar status."""
        theme = self.theme
        state = self.services.get_engine_state() if self.services.get_engine_state else {}
        snapshot = self.services.get_performance() if self.services.get_performance else None

        if snapshot is not None:
            fps = float(getattr(snapshot, "fps", 0.0))
            self._widgets["fps_chip"].configure(  # type: ignore[attr-defined]
                text=f"{fps:.0f} FPS",
                text_color=theme.success if fps >= 25 else
                theme.warning if fps >= 15 else theme.error)

        config = self.services.get_config() if self.services.get_config else None
        if config is not None:
            self._widgets["profile_chip"].configure(  # type: ignore[attr-defined]
                text=getattr(config, "profile_name", "Default"))

        paused = bool(state.get("paused", False))
        tracking = bool(state.get("tracking", False))
        if paused:
            text, colour = "Paused", theme.warning
        elif tracking:
            text, colour = "Tracking", theme.success
        else:
            text, colour = "No hand detected", theme.text_muted

        self._widgets["status_dot"].configure(text_color=colour)   # type: ignore[attr-defined]
        self._widgets["status_text"].configure(text=text)          # type: ignore[attr-defined]

        button = self._widgets.get("track_button")
        if button is not None:
            button.configure(text="Resume Tracking" if paused else "Pause Tracking")  # type: ignore[attr-defined]

    def _refresh_dashboard(self) -> None:
        """Push new data into the dashboard panel."""
        panel = self._widgets.get("dashboard")
        if panel is None:
            return

        snapshot = self.services.get_performance() if self.services.get_performance else None
        state = self.services.get_engine_state() if self.services.get_engine_state else {}
        history = self.services.get_history() if self.services.get_history else None

        summary: Dict[str, object] = {}
        top: List[Tuple[str, int]] = []
        heatmap = None
        if history is not None:
            try:
                summary = history.summary()          # type: ignore[attr-defined]
                top = history.top_gestures(6)        # type: ignore[attr-defined]
                heatmap = history.heatmap().tolist() # type: ignore[attr-defined]
            except Exception as exc:
                log.debug("history stats failed: %s", exc)

        if snapshot is not None:
            panel.refresh(snapshot, state, summary, top, heatmap)  # type: ignore[attr-defined]

    def _refresh_performance(self) -> None:
        """Update the performance gauges and charts."""
        snapshot = self.services.get_performance() if self.services.get_performance else None
        state = self.services.get_engine_state() if self.services.get_engine_state else {}
        if snapshot is None:
            return

        fps = float(getattr(snapshot, "fps", 0.0))
        cpu = float(getattr(snapshot, "cpu_percent", 0.0))
        memory_pct = float(getattr(snapshot, "memory_percent", 0.0))
        frame_ms = float(getattr(snapshot, "frame_ms", 0.0))
        infer_ms = float(getattr(snapshot, "inference_ms", 0.0))

        self._widgets["gauge_cpu"].set(cpu / 100.0, f"{cpu:.0f}%")            # type: ignore[attr-defined]
        self._widgets["gauge_memory"].set(memory_pct / 100.0,                  # type: ignore[attr-defined]
                                          f"{getattr(snapshot, 'memory_mb', 0):.0f}MB")
        self._widgets["gauge_fps"].set(min(fps / 60.0, 1.0), f"{fps:.0f}")     # type: ignore[attr-defined]
        confidence = float(state.get("confidence", 0.0))                       # type: ignore[arg-type]
        self._widgets["gauge_confidence"].set(confidence, f"{confidence:.0%}")  # type: ignore[attr-defined]

        self._widgets["perf_frame_chart"].push(frame_ms)   # type: ignore[attr-defined]
        self._widgets["perf_infer_chart"].push(infer_ms)   # type: ignore[attr-defined]

        percentiles = state.get("latency", {})
        if isinstance(percentiles, dict) and percentiles:
            text = "\n".join(
                f"{key:>4}   {value:6.1f} ms" for key, value in percentiles.items())
            self._widgets["perf_detail"].configure(text=text)  # type: ignore[attr-defined]

    # -- command handlers ------------------------------------------------- #

    def _call(self, name: str, *args: object) -> object:
        """Invoke a service callback, logging rather than raising on failure."""
        callback = getattr(self.services, name, None)
        if callback is None:
            log.debug("service %r not wired", name)
            return None
        try:
            return callback(*args)
        except Exception as exc:
            log.error("service %s failed: %s", name, exc)
            self.toasts.notify("Action failed", str(exc), "error")
            return None

    def _toggle_tracking(self) -> None:
        """Pause or resume tracking."""
        self._call("toggle_tracking")

    def _start_calibration(self) -> None:
        """Kick off the calibration wizard."""
        self._call("start_calibration")
        self.show_view("live")

    def _emergency_stop(self) -> None:
        """Trigger the emergency stop."""
        self._call("emergency_stop")
        self.toasts.notify("Emergency Stop", "All input released", "warning")

    def _update_setting(self, section: str, key: str, value: object) -> None:
        """Push a settings change back to the application."""
        self._call("update_setting", section, key, value)

    def _set_theme_by_label(self, label: str) -> None:
        """Change theme from its display name."""
        for name, display in self.theme_manager.display_names().items():
            if display == label:
                self._call("set_theme", name)
                self.toasts.notify("Theme changed", display, "info")
                return

    def _set_profile(self, name: str) -> None:
        """Activate a profile and refresh dependent views."""
        self._call("set_profile", name)
        self.refresh_profile_list()
        self.toasts.notify("Profile activated", name, "success")

    def _create_profile(self) -> None:
        """Create a profile from the name entry."""
        entry = self._widgets.get("profile_name")
        name = entry.get().strip() if entry else ""  # type: ignore[attr-defined]
        if not name:
            self.toasts.notify("Name required", "Enter a profile name", "warning")
            return
        self._call("create_profile", name)
        entry.delete(0, "end")  # type: ignore[attr-defined]
        self.refresh_profile_list()

    def _delete_profile(self, name: str) -> None:
        """Delete a profile."""
        self._call("delete_profile", name)
        self.refresh_profile_list()

    def _duplicate_profile(self, name: str) -> None:
        """Copy a profile under a new name."""
        self._call("duplicate_profile", name)
        self.refresh_profile_list()
        self.toasts.notify("Profile duplicated", name, "success")

    def _export_profile(self, name: str) -> None:
        """Write a profile to a JSON file via a save dialog."""
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile=f"{name.lower().replace(' ', '-')}.json")
        if not path:
            return
        self._call("export_profile", name, Path(path))
        self.toasts.notify("Profile exported", Path(path).name, "success")

    def _import_profile(self) -> None:
        """Load a profile from a JSON file via an open dialog."""
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        self._call("import_profile", Path(path))
        self.refresh_profile_list()
        self.toasts.notify("Profile imported", Path(path).name, "success")

    def _record_gesture(self) -> None:
        """Start recording a custom gesture."""
        entry = self._widgets.get("record_name")
        name = entry.get().strip() if entry else ""  # type: ignore[attr-defined]
        if not name:
            self.toasts.notify("Name required", "Enter a gesture name", "warning")
            return
        self._call("record_gesture", name)
        entry.delete(0, "end")  # type: ignore[attr-defined]
        self.show_view("live")

    def _delete_gesture(self, name: str) -> None:
        """Delete a custom gesture."""
        self._call("delete_gesture", name)
        self.refresh_gesture_list()

    def _duplicate_gesture(self, name: str) -> None:
        """Copy a custom gesture, so it can be retuned without losing the original."""
        self._call("duplicate_gesture", name)
        self.refresh_gesture_list()
        self.toasts.notify("Gesture duplicated", name, "success")

    def _export_gestures(self) -> None:
        """Write the custom gesture library to a shareable JSON file."""
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="custom_gestures.json")
        if not path:
            return
        count = self._call("export_gestures", Path(path))
        self.toasts.notify("Gestures exported", f"{count or 0} gestures", "success")

    def _import_gestures(self) -> None:
        """Merge a gesture library file into the current one."""
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        count = self._call("import_gestures", Path(path))
        self.refresh_gesture_list()
        self.toasts.notify("Gestures imported", f"{count or 0} gestures", "success")

    def _bind_gesture(self, name: str, action_id: str) -> None:
        """Rebind a gesture to a different action."""
        self._call("bind_gesture", name, action_id)

    def _toggle_gesture(self, name: str, enabled: Optional[bool]) -> None:
        """Enable or disable a gesture."""
        if enabled is None:
            return
        self._call("toggle_gesture", name, enabled)

    def _export_history(self) -> None:
        """Export history to CSV via a save dialog."""
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="gesture_history.csv")
        if not path:
            return
        count = self._call("export_history", Path(path))
        self.toasts.notify("History exported", f"{count} rows", "success")

    def _clear_history(self) -> None:
        """Clear the gesture history."""
        self._call("clear_history")
        self.refresh_history_list()
        self.toasts.notify("History cleared", "", "info")

    # -- theming ---------------------------------------------------------- #

    def apply_theme(self, theme: Theme) -> None:
        """Rebuild the window with a new palette.

        Tk cannot restyle a live widget tree wholesale, so views are dropped
        and rebuilt lazily.  Doing it any other way means chasing every widget
        reference by hand and missing some.
        """
        self.theme = theme
        self.toasts.set_theme(theme)
        ctk.set_appearance_mode(theme.appearance)
        self.configure(fg_color=theme.background)

        active = self._active_view
        for view in self._views.values():
            view.destroy()
        self._views.clear()
        self._nav_buttons.clear()
        self._widgets.clear()
        self._video_label = None

        for child in self.winfo_children():
            if isinstance(child, (ctk.CTkFrame,)):
                child.destroy()

        self._active_view = ""
        self._build_shell()
        self.show_view(active or "live")

    # -- lifecycle -------------------------------------------------------- #

    def notify(self, title: str, message: str = "", level: str = "info") -> None:
        """Show a toast.  Safe to call from any thread."""
        self.toasts.notify(title, message, level)

    def _on_close(self) -> None:
        """Handle the window close button."""
        self._running = False
        try:
            if self.services.on_close:
                self.services.on_close()
        except Exception as exc:
            log.error("shutdown handler failed: %s", exc)
        self.destroy()
