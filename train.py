#!/usr/bin/env python3
"""Terminal entry point — see `python train.py --help`.

A thin shim so the repo can be trained from a shell without installing it;
all logic lives in ``pvt_moe.cli`` (the package is the single source of truth).
Installed as the ``pvt-moe-train`` console script by `pip install -e .`.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pvt_moe.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
