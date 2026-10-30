#!/usr/bin/env python3
"""Install the local CUDA/CPU workbench into .venv and open its settings page.

Run this only from a trusted clone: installing any Python project executes its
packaging code. Downloads packages from official PyTorch/PyPI sources, never
model checkpoints. Existing environments are not silently repurposed.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def plan(root, executable, backend="cuda", *, platform=None):
    platform = sys.platform if platform is None else platform
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    if backend == "cuda" and platform not in ("linux", "win32"):
        raise ValueError("CUDA installer requires Linux or Windows; use --backend cpu on this platform")
    flavor = "cu132" if backend == "cuda" else "cpu"
    version = "2.15.0" if platform == "darwin" else "2.15.0+" + flavor
    return [
        [str(executable), "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/" + flavor, "torch==" + version],
        [str(executable), "-m", "pip", "install", "--index-url", "https://pypi.org/simple", "-e", str(root) + "[app]"],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda" if sys.platform in ("linux", "win32") else "cpu", help="PyTorch build: CUDA 13.2 (Linux/Windows default; NVIDIA driver required) or CPU")
    parser.add_argument("--no-launch", action="store_true", help="Install without opening the local app")
    parser.add_argument("--reuse-venv", action="store_true", help="Explicitly reuse this checkout's existing .venv")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without installing or launching")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11): parser.error("Python 3.11 or newer is required")
    env = ROOT / ".venv"
    executable = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try: commands = plan(ROOT, executable, args.backend)
    except ValueError as exc: parser.error(str(exc))
    if args.dry_run:
        print("Create isolated environment:", env)
        for command in commands: print(subprocess.list2cmdline(command))
        print("Launch:", executable, "-m looking_glass.app")
        return 0
    if env.is_symlink(): parser.error("Refusing a symlinked .venv")
    if env.exists() and not args.reuse_venv:
        parser.error(".venv already exists. Use its python -m looking_glass.app, or explicitly --reuse-venv")
    if not env.exists(): venv.EnvBuilder(with_pip=True).create(env)
    if not (env / "pyvenv.cfg").is_file() or not executable.is_file(): parser.error("Existing .venv is not a usable Python environment")
    print(f"Installing {args.backend.upper()} dependencies; no drivers or model checkpoints will be downloaded.", flush=True)
    for command in commands: subprocess.run(command, cwd=ROOT, check=True)
    if not args.no_launch: return subprocess.call([str(executable), "-m", "looking_glass.app"], cwd=ROOT)
    print("Installed. Run:", executable, "-m looking_glass.app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
