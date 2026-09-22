#!/usr/bin/env python3
"""Assert the built wheel ships every package that exists in the tree.

    python tools/check_packaging.py [REPO_ROOT]

Exits non-zero, and names the packages, when the two disagree. Needs pip and
setuptools only -- no torch, no dataset, no GPU.

Nothing imports pyproject.toml, so a wrong packages list cannot fail a test --
it fails at `pip install` time, for someone else. During the 2026 cleanup that
happened three times: pvt_moe.ssl named after deletion, pvt_moe.config omitted
after the split created it, and pvt_moe.eval omitted on the ssl branch, where
pvt_moe/ssl/__init__.py imports it at module level, so `import pvt_moe.ssl`
raised ModuleNotFoundError from any wheel.

Builds from a COPY of the tracked tree: a stale build/ or *.egg-info in a
working checkout makes pip reuse the previous build's file list, and the check
then silently passes on a broken pyproject (observed while writing this).
"""
import glob, pathlib, shutil, subprocess, sys, tempfile, zipfile

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
# -z: a tracked path may contain a space (this repo has one).
tracked = [f for f in subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                     capture_output=True, text=True,
                                     check=True).stdout.split("\0") if f]

with tempfile.TemporaryDirectory() as clean, tempfile.TemporaryDirectory() as out:
    for rel in tracked:
        dst = pathlib.Path(clean) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dst)
    on_disk = {str(p.parent.relative_to(clean)).replace("/", ".")
               for p in pathlib.Path(clean).glob("pvt_moe/**/__init__.py")}
    subprocess.run([sys.executable, "-m", "pip", "wheel", clean, "--no-deps", "-w", out, "-q"],
                   check=True)
    names = zipfile.ZipFile(glob.glob(f"{out}/*.whl")[0]).namelist()

in_wheel = {n.rsplit("/", 1)[0].replace("/", ".") for n in names if n.endswith("__init__.py")}
in_wheel = {p for p in in_wheel if p == "pvt_moe" or p.startswith("pvt_moe.")}
missing, extra = sorted(on_disk - in_wheel), sorted(in_wheel - on_disk)

for p in sorted(on_disk | in_wheel):
    print(f"  {'ok  ' if p in on_disk and p in in_wheel else 'BAD '} {p}")
if missing:
    print(f"\nNOT SHIPPED: {missing}\nThese exist in the tree but not in the wheel, so importing them "
          "fails for anyone who pip-installed. Fix pyproject.toml.")
if extra:
    print(f"\nIN WHEEL BUT NOT IN TREE: {extra}")
sys.exit(1 if (missing or extra) else 0)
