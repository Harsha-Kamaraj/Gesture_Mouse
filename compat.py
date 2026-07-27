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

import os
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

    ``platform.mac_ver()`` reads ``SystemVersion.plist`` through ``plistlib``,
    which needs expat.  So on a Python with a broken ``pyexpat`` the parse
    fails silently and the function returns ``('', ('', '', ''), '')``.

    ``darkdetect`` then does ``int(version.split('.')[0])`` on that empty
    string at import time and raises ``ValueError``, which takes CustomTkinter
    — and therefore the whole UI — down with it.

    The replacement resolves the version once via ``sw_vers`` and returns it
    directly.  It deliberately does **not** delegate to the original: that
    call is exactly what fails, and re-attempting it on every invocation would
    either raise through the expat stub or re-read and re-parse a file whose
    answer cannot change.

    Returns:
        True if the patch was applied, False if it was unnecessary.
    """
    if sys.platform != "darwin":
        return False

    try:
        current = platform.mac_ver()
    except Exception:
        # A stubbed expat makes the underlying plist parse raise outright.
        current = ("", ("", "", ""), "")

    if current[0]:
        return False  # Already reporting a usable version.

    version = _query_macos_version() or _derive_from_darwin()
    if version is None:
        # Last resort: a plausible modern version.  Anything parsable beats
        # letting a third-party import crash the application outright.
        version = "26.0"

    machine = platform.machine()
    resolved: Tuple[str, Tuple[str, str, str], str] = (version, ("", "", ""), machine)

    def mac_ver(release: str = "", versioninfo: Tuple[str, str, str] = ("", "", ""),
                machine_: str = "") -> Tuple[str, Tuple[str, str, str], str]:
        """Patched ``platform.mac_ver`` returning a non-empty version."""
        return resolved

    platform.mac_ver = mac_ver  # type: ignore[assignment]
    return True


def _pyexpat_is_broken() -> bool:
    """Return True if ``pyexpat`` exists but fails to load."""
    if "pyexpat" in sys.modules:
        return False
    try:
        import pyexpat  # noqa: F401

        return False
    except ImportError:
        # A missing-symbol dlopen failure surfaces as ImportError, same as a
        # genuinely absent module; either way the stub is the right response.
        return True
    except Exception:
        return True


class BrokenPyexpatError(RuntimeError):
    """Raised when this interpreter's ``pyexpat`` extension cannot load."""


_EXPAT_HELP = """\
This Python installation has a broken 'pyexpat' extension.

  Cause:  pyexpat is linked against /usr/lib/libexpat.1.dylib, but that
          system library does not export the symbol it needs
          (_XML_SetAllocTrackerActivationThreshold). This affects every
          Homebrew Python on recent macOS, not just this project.

  Impact: MediaPipe cannot be imported at all, because its import chain
          reaches XML parsing:
            mediapipe -> tasks.vision -> drawing_utils -> matplotlib
                      -> font_manager -> plistlib -> pyexpat
          CustomTkinter is affected too, via darkdetect reading the macOS
          version from a plist.

  Fix:    Run the bundled repair script, then start the app again:

            brew install expat
            python scripts/fix_macos_expat.py

          It builds a corrected copy of the extension inside your virtual
          environment. It does not modify Homebrew or system files.
"""


def activate_expat_fix() -> bool:
    """Put a repaired ``pyexpat`` on ``sys.path`` if one has been built.

    :mod:`scripts.fix_macos_expat` installs a corrected extension into
    ``site-packages/_expat_fix/`` plus a ``.pth`` file that adds that directory
    at interpreter startup.  The ``.pth`` route is fragile in one specific way:
    ``site.py`` silently skips any ``.pth`` carrying macOS's ``UF_HIDDEN``
    flag, and files under a virtualenv can acquire that flag without the user
    doing anything.  When that happens the repair looks installed but has no
    effect.

    Since this module is imported before anything else in the application, it
    can simply do the job itself.  This is belt-and-braces: when the ``.pth``
    works, this is a no-op.

    Returns:
        True if a repaired extension was found and activated here.
    """
    if sys.platform != "darwin" or not _pyexpat_is_broken():
        return False

    import sysconfig

    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return False

    fix_dir = os.path.join(purelib, "_expat_fix")
    if not os.path.isdir(fix_dir):
        return False

    if fix_dir not in sys.path:
        sys.path.insert(0, fix_dir)

    # Only report success if the extension now actually loads.
    return not _pyexpat_is_broken()


def check_pyexpat(raise_on_error: bool = False) -> bool:
    """Report whether XML parsing works in this interpreter.

    An earlier version of this module tried to *stub out* expat, on the
    reasoning that this project never parses XML.  That was wrong: the C
    extension ``_elementtree`` requires the real ``pyexpat.expat_CAPI``
    capsule, which a Python stub cannot provide, so pyobjc and anything else
    importing ``xml.etree`` still failed.  A partial workaround that moves the
    crash somewhere less obvious is worse than none, so the shim was removed
    in favour of detecting the problem and explaining the fix.

    Args:
        raise_on_error: Raise :class:`BrokenPyexpatError` instead of
            returning ``False``.

    Returns:
        True if expat is usable.
    """
    if not _pyexpat_is_broken():
        return True
    if raise_on_error:
        raise BrokenPyexpatError(_EXPAT_HELP)
    return False


def apply_all() -> None:
    """Apply every shim.  Idempotent and safe to call repeatedly.

    Order matters: a repaired expat has to be on ``sys.path`` before the macOS
    version lookup runs, because that lookup parses a plist and is therefore
    one of the things a broken expat breaks.
    """
    global _applied
    if _applied:
        return
    activate_expat_fix()
    patch_macos_version()
    _applied = True


# Applied on import so that a bare ``import compat`` before ``import
# customtkinter`` is enough.
apply_all()
