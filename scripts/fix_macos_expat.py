#!/usr/bin/env python3
"""Repair a broken ``pyexpat`` extension on macOS.

The problem
-----------
Homebrew's CPython builds link ``pyexpat`` against ``/usr/lib/libexpat.1.dylib``
but compile it against a newer expat.  On recent macOS the system library no
longer exports ``_XML_SetAllocTrackerActivationThreshold``, so importing
``pyexpat`` fails with a ``Symbol not found`` dlopen error.

That breaks far more than XML: MediaPipe imports matplotlib (for drawing
helpers), matplotlib imports ``plistlib``, and ``plistlib`` imports expat — so
hand tracking will not import at all.  CustomTkinter is affected too, since
``darkdetect`` reads the macOS version from a plist.

The fix
-------
Copy the extension into the active virtual environment, repoint its library
dependency at Homebrew's expat (which does export the symbol), re-sign it, and
add a ``.pth`` file so the corrected copy is found first.

Nothing outside the virtual environment is modified, so the change is
completely reversible: delete ``_expat_fix/`` and ``_expat_fix.pth`` from
``site-packages``.

Usage::

    brew install expat
    python scripts/fix_macos_expat.py           # repair
    python scripts/fix_macos_expat.py --check   # report status only
    python scripts/fix_macos_expat.py --undo    # remove the repair
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Optional

FIX_DIR_NAME = "_expat_fix"
PTH_NAME = "_expat_fix.pth"
SYSTEM_LIB = "/usr/lib/libexpat.1.dylib"

BREW_CANDIDATES = (
    "/opt/homebrew/opt/expat/lib/libexpat.1.dylib",   # Apple Silicon
    "/usr/local/opt/expat/lib/libexpat.1.dylib",      # Intel
)


def expat_works() -> bool:
    """Return True if ``pyexpat`` imports in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", "import pyexpat, xml.etree.ElementTree"],
        capture_output=True, check=False,
    )
    return result.returncode == 0


def find_site_packages() -> Path:
    """Return the active environment's site-packages directory."""
    path = sysconfig.get_paths().get("purelib")
    if not path:
        raise SystemExit("Could not locate site-packages for this interpreter.")
    return Path(path)


def find_stdlib_pyexpat() -> Optional[Path]:
    """Locate the interpreter's ``pyexpat`` shared object."""
    dynload = Path(sysconfig.get_config_var("DESTSHARED") or "")
    if dynload.is_dir():
        matches = sorted(dynload.glob("pyexpat*.so"))
        if matches:
            return matches[0]

    stdlib = Path(sysconfig.get_paths()["stdlib"]) / "lib-dynload"
    matches = sorted(stdlib.glob("pyexpat*.so"))
    return matches[0] if matches else None


def find_brew_expat() -> Optional[Path]:
    """Locate a Homebrew libexpat that exports the required symbol."""
    for candidate in BREW_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    result = subprocess.run(["brew", "--prefix", "expat"],
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        path = Path(result.stdout.strip()) / "lib" / "libexpat.1.dylib"
        if path.exists():
            return path
    return None


def undo(site_packages: Path) -> int:
    """Remove a previously applied repair."""
    removed = False
    fix_dir = site_packages / FIX_DIR_NAME
    pth = site_packages / PTH_NAME

    if fix_dir.exists():
        shutil.rmtree(fix_dir)
        removed = True
    if pth.exists():
        pth.unlink()
        removed = True

    print("Repair removed." if removed else "No repair was installed.")
    return 0


def apply(site_packages: Path) -> int:
    """Build and install the corrected extension."""
    source = find_stdlib_pyexpat()
    if source is None:
        print("error: could not find this interpreter's pyexpat extension.")
        return 1

    brew_expat = find_brew_expat()
    if brew_expat is None:
        print("error: Homebrew's expat not found. Install it first:\n"
              "         brew install expat")
        return 1

    fix_dir = site_packages / FIX_DIR_NAME
    fix_dir.mkdir(parents=True, exist_ok=True)
    target = fix_dir / source.name

    print(f"  source   {source}")
    print(f"  expat    {brew_expat}")
    print(f"  target   {target}")

    shutil.copy2(source, target)
    target.chmod(0o755)

    rename = subprocess.run(
        ["install_name_tool", "-change", SYSTEM_LIB, str(brew_expat), str(target)],
        capture_output=True, text=True, check=False,
    )
    if rename.returncode != 0:
        print(f"error: install_name_tool failed: {rename.stderr.strip()}")
        return 1

    # Rewriting the load command invalidates the signature; on Apple Silicon
    # an unsigned extension will not load at all, so re-sign ad hoc.
    sign = subprocess.run(["codesign", "-f", "-s", "-", str(target)],
                          capture_output=True, text=True, check=False)
    if sign.returncode != 0:
        print(f"warning: codesign failed: {sign.stderr.strip()}")

    # A .pth file runs at interpreter startup, which is early enough to shadow
    # the stdlib copy before anything imports pyexpat.
    #
    # Note ``__file__`` is NOT defined while a .pth line executes, so the
    # directory has to be located by scanning the paths site has already set
    # up rather than derived from this file's location.
    pth = site_packages / PTH_NAME
    pth.write_text(
        "import sys, os; "
        f"_n = {FIX_DIR_NAME!r}; "
        "_c = [os.path.join(d, _n) for d in list(sys.path) "
        "if os.path.isdir(os.path.join(d, _n))]; "
        "_c and sys.path.insert(0, _c[0])\n",
        encoding="utf-8",
    )

    if expat_works():
        print("\nSuccess — pyexpat now loads correctly.")
        return 0

    print("\nerror: pyexpat still fails to import after the repair.")
    print("Run with --undo to revert, then please report the issue.")
    return 1


def main(argv: Optional[list] = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Repair a broken pyexpat extension on macOS.")
    parser.add_argument("--check", action="store_true",
                        help="report status without changing anything")
    parser.add_argument("--undo", action="store_true",
                        help="remove a previously applied repair")
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        print("This script is only needed on macOS.")
        return 0

    site_packages = find_site_packages()

    if args.undo:
        return undo(site_packages)

    if expat_works():
        print("pyexpat works correctly — no repair needed.")
        return 0

    print("pyexpat is broken in this interpreter.\n")
    if args.check:
        print("Re-run without --check to repair it.")
        return 1

    return apply(site_packages)


if __name__ == "__main__":
    sys.exit(main())
