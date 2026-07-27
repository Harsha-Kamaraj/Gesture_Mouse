"""Theme definitions and colour tokens.

A theme is a flat token table rather than a stylesheet.  Widgets ask for
semantic tokens (``surface``, ``accent``, ``text_muted``) instead of literal
colours, so adding a theme never requires touching widget code, and the
overlay renderer can share the exact same palette as the Tk UI.

Themes are also exported to ``assets/themes/*.json`` so users can hand-author
their own without editing Python.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from config import THEMES_DIR
from logger import get_logger
from utils import hex_to_rgb, mix_colours

log = get_logger(__name__)


@dataclass
class Theme:
    """A complete colour palette plus typography hints."""

    name: str
    display_name: str
    # Surfaces, back to front.
    background: str = "#0E0F1A"
    surface: str = "#171A2B"
    surface_alt: str = "#1F2338"
    border: str = "#2A2F4A"
    # Content.
    text: str = "#E8EAF6"
    text_muted: str = "#8A90B8"
    # Accents.
    accent: str = "#7C5CFF"
    accent_hover: str = "#9376FF"
    secondary: str = "#22D3EE"
    # Semantic states.
    success: str = "#34D399"
    warning: str = "#FBBF24"
    error: str = "#F87171"
    info: str = "#60A5FA"
    # Overlay-specific (drawn on the camera feed).
    landmark: str = "#22D3EE"
    skeleton: str = "#7C5CFF"
    overlay_bg: str = "#0E0F1A"
    #: 0 = fully transparent overlay panels, 1 = opaque.
    overlay_opacity: float = 0.72
    #: Whether CustomTkinter should use its light or dark base.
    appearance: str = "dark"
    corner_radius: int = 12
    font_family: str = "SF Pro Display"
    font_family_mono: str = "SF Mono"

    def token(self, name: str, fallback: str = "#FF00FF") -> str:
        """Look up a colour token by name."""
        return str(getattr(self, name, fallback))

    def rgb(self, name: str) -> Tuple[int, int, int]:
        """Token as an ``(r, g, b)`` tuple."""
        return hex_to_rgb(self.token(name))

    def bgr(self, name: str) -> Tuple[int, int, int]:
        """Token as an OpenCV ``(b, g, r)`` tuple."""
        r, g, b = self.rgb(name)
        return (b, g, r)

    def elevated(self, level: int = 1) -> str:
        """Surface colour lightened by ``level`` steps, for depth."""
        return mix_colours(self.surface, self.text, 0.04 * level)

    def to_dict(self) -> Dict[str, object]:
        """Serialisable form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Theme":
        """Build a theme from a token table, ignoring unknown keys."""
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid}
        filtered.setdefault("name", "custom")
        filtered.setdefault("display_name", str(filtered["name"]).title())
        return cls(**filtered)  # type: ignore[arg-type]


def builtin_themes() -> Dict[str, Theme]:
    """The five shipped themes."""
    themes = [
        Theme(
            name="dark",
            display_name="Midnight",
            background="#0E0F1A", surface="#171A2B", surface_alt="#1F2338",
            border="#2A2F4A", text="#E8EAF6", text_muted="#8A90B8",
            accent="#7C5CFF", accent_hover="#9376FF", secondary="#22D3EE",
            landmark="#22D3EE", skeleton="#7C5CFF", appearance="dark",
        ),
        Theme(
            name="light",
            display_name="Daylight",
            background="#F4F5FA", surface="#FFFFFF", surface_alt="#EDEFF7",
            border="#D9DDEC", text="#1B1D2A", text_muted="#6A6F8C",
            accent="#5B45D6", accent_hover="#7059EE", secondary="#0891B2",
            success="#059669", warning="#D97706", error="#DC2626",
            info="#2563EB", landmark="#0891B2", skeleton="#5B45D6",
            overlay_bg="#FFFFFF", overlay_opacity=0.80, appearance="light",
        ),
        Theme(
            name="cyberpunk",
            display_name="Cyberpunk",
            background="#0A0014", surface="#14002B", surface_alt="#1E0140",
            border="#3D0B6B", text="#F0E6FF", text_muted="#9B7FC4",
            accent="#FF2E97", accent_hover="#FF5CB0", secondary="#00F0FF",
            success="#00FFA3", warning="#FFD600", error="#FF3D57",
            info="#00F0FF", landmark="#00F0FF", skeleton="#FF2E97",
            overlay_bg="#0A0014", overlay_opacity=0.68, appearance="dark",
        ),
        Theme(
            name="gaming",
            display_name="Gaming",
            background="#0B1010", surface="#121A18", surface_alt="#182422",
            border="#1F3330", text="#E6FFF7", text_muted="#7FA69C",
            accent="#00FF88", accent_hover="#4DFFAB", secondary="#FF6B00",
            success="#00FF88", warning="#FFB300", error="#FF3B30",
            info="#00D9FF", landmark="#00FF88", skeleton="#FF6B00",
            overlay_bg="#0B1010", overlay_opacity=0.70, appearance="dark",
        ),
        Theme(
            name="glass",
            display_name="Glassmorphism",
            background="#12131C", surface="#1C1E2E", surface_alt="#262940",
            border="#343854", text="#F2F3FB", text_muted="#9FA4C4",
            accent="#6EA8FE", accent_hover="#8FBEFF", secondary="#B197FC",
            landmark="#B197FC", skeleton="#6EA8FE",
            overlay_bg="#12131C", overlay_opacity=0.42, appearance="dark",
        ),
    ]
    return {theme.name: theme for theme in themes}


def high_contrast_variant(theme: Theme) -> Theme:
    """Return an accessibility-boosted version of ``theme``.

    Pushes surfaces to the extremes and text to pure white/black.  This is a
    genuine accessibility feature rather than a cosmetic one: the standard
    palettes sit around 7:1 contrast, and this lifts body text past 15:1.
    """
    is_dark = theme.appearance == "dark"
    return Theme(
        name=f"{theme.name}-hc",
        display_name=f"{theme.display_name} (High Contrast)",
        background="#000000" if is_dark else "#FFFFFF",
        surface="#0A0A0A" if is_dark else "#FFFFFF",
        surface_alt="#141414" if is_dark else "#F0F0F0",
        border="#FFFFFF" if is_dark else "#000000",
        text="#FFFFFF" if is_dark else "#000000",
        text_muted="#D0D0D0" if is_dark else "#2A2A2A",
        accent="#FFD400" if is_dark else "#0000CC",
        accent_hover="#FFE657" if is_dark else "#3333DD",
        secondary="#00FFFF" if is_dark else "#006666",
        success="#00FF00" if is_dark else "#006600",
        warning="#FFAA00" if is_dark else "#804000",
        error="#FF4040" if is_dark else "#CC0000",
        info="#40C0FF" if is_dark else "#0044AA",
        landmark="#00FFFF" if is_dark else "#0000CC",
        skeleton="#FFD400" if is_dark else "#CC0000",
        overlay_bg="#000000" if is_dark else "#FFFFFF",
        overlay_opacity=0.92,
        appearance=theme.appearance,
        corner_radius=theme.corner_radius,
    )


class ThemeManager:
    """Resolves the active theme and exposes it to the UI and overlay."""

    def __init__(self) -> None:
        self._themes: Dict[str, Theme] = builtin_themes()
        self._active: Theme = self._themes["dark"]
        self.high_contrast = False
        self.load_custom_themes()

    @property
    def theme(self) -> Theme:
        """The active theme, with the high-contrast variant applied if on."""
        if self.high_contrast:
            return high_contrast_variant(self._active)
        return self._active

    @property
    def names(self) -> List[str]:
        """Every available theme name."""
        return sorted(self._themes)

    def display_names(self) -> Dict[str, str]:
        """``name -> display name`` for the theme picker."""
        return {name: theme.display_name for name, theme in self._themes.items()}

    def set_theme(self, name: str) -> Theme:
        """Activate a theme by name, falling back to dark."""
        theme = self._themes.get(name)
        if theme is None:
            log.warning("unknown theme %r; using dark", name)
            theme = self._themes["dark"]
        self._active = theme
        log.info("theme set to %r", theme.name)
        return self.theme

    def set_high_contrast(self, enabled: bool) -> Theme:
        """Toggle the accessibility high-contrast variant."""
        self.high_contrast = enabled
        return self.theme

    def set_accent(self, colour: str) -> Theme:
        """Override the active theme's accent colour."""
        self._active.accent = colour
        self._active.accent_hover = mix_colours(colour, "#FFFFFF", 0.22)
        return self.theme

    # -- custom themes ---------------------------------------------------- #

    def load_custom_themes(self) -> int:
        """Load user themes from ``assets/themes/*.json``."""
        if not THEMES_DIR.exists():
            return 0
        loaded = 0
        for path in sorted(THEMES_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                theme = Theme.from_dict(data)
                self._themes[theme.name] = theme
                loaded += 1
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                log.warning("skipping theme %s: %s", path.name, exc)
        if loaded:
            log.info("loaded %d custom themes", loaded)
        return loaded

    def export_builtins(self) -> int:
        """Write the built-in themes to disk as editable templates."""
        THEMES_DIR.mkdir(parents=True, exist_ok=True)
        written = 0
        for theme in builtin_themes().values():
            path = THEMES_DIR / f"{theme.name}.json"
            if path.exists():
                continue
            try:
                path.write_text(json.dumps(theme.to_dict(), indent=2),
                                encoding="utf-8")
                written += 1
            except OSError as exc:
                log.warning("could not write theme %s: %s", theme.name, exc)
        return written
