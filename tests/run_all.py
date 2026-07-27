"""Run every test module and report a combined result.

Works with or without pytest installed — each test file exposes a ``_run_all``
runner, so ``python tests/run_all.py`` is always sufficient.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODULES = (
    "test_dynamic_gestures",
    "test_gesture_engine",
    "test_integration",
)


def main() -> int:
    """Run all suites; return a non-zero exit code on any failure."""
    total_failures = 0
    started = time.perf_counter()

    for name in MODULES:
        print(f"\n{name}")
        print("-" * 60)
        module = importlib.import_module(name)
        failures = module._run_all()  # noqa: SLF001
        total_failures += failures

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 60)
    if total_failures:
        print(f"FAILED — {total_failures} failing test(s) in {elapsed:.1f}s")
    else:
        print(f"All tests passed in {elapsed:.1f}s")
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
