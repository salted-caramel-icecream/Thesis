"""Run the full CPU test suite: ``python tests/run_all.py``.

No pytest dependency — plain functions named ``test_*`` in ``test_*.py``
files. Exits non-zero on any failure. These tests are the gate before ANY
GPU run: if they fail, do not upload / launch training.

Version guards: the suite runs on torch >= 2.3 (the RMSNorm fused kernel is
exercised only when the local torch has it; the fallback is exercised
otherwise — both paths are correct).
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import time
import traceback

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Make `pvt_moe` and `helpers` importable regardless of invocation directory.
for p in (str(REPO_ROOT), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    import torch

    print(f"torch {torch.__version__} | python {sys.version.split()[0]}")

    modules = sorted(p.stem for p in TESTS_DIR.glob("test_*.py"))
    passed, failed, errors = 0, 0, []

    for mod_name in modules:
        module = importlib.import_module(mod_name)
        tests = [
            (name, fn)
            for name, fn in vars(module).items()
            if name.startswith("test_") and callable(fn)
        ]
        print(f"\n== {mod_name} ({len(tests)} tests) ==")
        for name, fn in tests:
            t0 = time.time()
            try:
                fn()
                passed += 1
                print(f"  PASS {name} ({time.time() - t0:.2f}s)")
            except BaseException:      # noqa: BLE001
                # BaseException, not Exception: argparse's error() raises
                # SystemExit, and a test that lets one escape used to kill the
                # runner mid-suite — 76 of 315 tests ran and no summary printed.
                failed += 1
                errors.append((mod_name, name, traceback.format_exc()))
                print(f"  FAIL {name}")

    print(f"\n{'=' * 60}\n{passed} passed, {failed} failed")
    for mod_name, name, tb in errors:
        print(f"\n--- {mod_name}.{name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
