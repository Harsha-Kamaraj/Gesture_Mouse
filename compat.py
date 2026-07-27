"""Environment compatibility shims.

Import this **before** CustomTkinter anywhere in the application.

Why this exists
---------------
``customtkinter`` imports ``darkdetect``, which at module import time calls
``platform.mac_ver()`` and does ``int(version.split('.')[0])`` on the result.
On macOS 26 ("Tahoe") with some CPython builds ``platform.mac_ver()`` returns
``('', ('', '', ''), '')`` — an empty version string — so that call raises
``ValueError: invalid literal for int() with base 10: ''`` and the *entire UI
becomes unimportable*.

The failure happens during a third-party module's import, so it cannot be
caught and handled at our call sites; the only place to fix it is before the
import chain starts.  :func:`patch_macos_version` therefore repairs
``platform.mac_ver`` itself, sourcing the real version from ``sw_vers``.

Each shim is narrowly scoped, checks whether it is actually needed, and is a
no-op everywhere else.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Optional, Tuple

_applied = False


def _query_macos_version() -> Optional[str]:
    """Return the macOS product version via ``sw_vers``, or ``None``."""
    try:
        result = subprocess.run(
            ["sw_vers", "-productVersion"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        version = result.stdout.strip()
        # Must start with a parsable major number to be useful to callers.
        if version and version.split(".")[0].isdigit():
            return version
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _derive_from_darwin() -> Optional[str]:
    """Infer the macOS version from the Darwin kernel release.

    Darwin majors have tracked macOS majors with a constant offset since
    Big Sur: Darwin 20 = macOS 11 … Darwin 25 = macOS 26.  Used only if
    ``sw_vers`` is unavailable.
    """
    try:
        darwin_major = int(platform.release().split(".")[0])
    except (ValueError, IndexError):
        return None

    if darwin_major >= 25:      # Darwin 25 -> macOS 26 (Tahoe) and onward
        return f"{darwin_major + 1}.0"
    if darwin_major >= 20:      # Darwin 20-24 -> macOS 11-15
        return f"{darwin_major - 9}.0"
    return None


def patch_macos_version() -> bool:
    """Repair ``platform.mac_ver`` when it reports an empty version.

    Returns:
        True if the patch was applied, False if it was unnecessary.
    """
    if sys.platform != "darwin":
        return False

    current = platform.mac_ver()
    if current[0]:
        return False  # Already reporting a usable version.

    version = _query_macos_version() or _derive_from_darwin()
    if version is None:
        # Last resort: a plausible modern version.  Anything parsable beats
        # letting a third-party import crash the application outright.
        version = "26.0"

    machine = platform.machine()
    original = platform.mac_ver

    def mac_ver(release: str = "", versioninfo: Tuple[str, str, str] = ("", "", ""),
                machine_: str = "") -> Tuple[str, Tuple[str, str, str], str]:
        """Patched ``platform.mac_ver`` returning a non-empty version."""
        result = original(release, versioninfo, machine_)
        if result[0]:
            return result
        return (version, ("", "", ""), machine)

    platform.mac_ver = mac_ver  # type: ignore[assignment]
    return True


def apply_all() -> None:
    """Apply every shim.  Idempotent and safe to call repeatedly."""
    global _applied
    if _applied:
        return
    patch_macos_version()
    _applied = True


# Applied on import so that a bare ``import compat`` before ``import
# customtkinter`` is enough.
apply_all()
