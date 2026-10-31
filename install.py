#!/usr/bin/env python3
"""Install the local CUDA/CPU workbench into .venv and open its settings page.

Run this only from a trusted clone: installing any Python project executes its
packaging code. Downloads packages from official PyTorch/PyPI sources, never
model checkpoints. Existing environments are not silently repurposed.
"""
from __future__ import annotations
import argparse
import json
import shlex
import threading
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


class InstallError(RuntimeError):
    pass


def command_text(command):
    return subprocess.list2cmdline([str(x) for x in command]) if os.name == "nt" else shlex.join(str(x) for x in command)


def _probe_child(executable):
    # The selected environment is explicitly trusted. -I ignores PYTHONPATH and
    # user site; unlike -S on Python 3.11/12 it preserves actual venv prefix.
    code = ("import sys,json,importlib.util; print(json.dumps({"
            "'version':list(sys.version_info[:3]),'prefix':sys.prefix,"
            "'base_prefix':sys.base_prefix,'pip':importlib.util.find_spec('pip') is not None}))")
    process = subprocess.Popen([str(executable), "-I", "-B", "-c", code],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = bytearray()
    excessive = threading.Event()
    read_errors = []
    def read():
        try:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk: break
                if len(output) + len(chunk) > 65536:
                    excessive.set()
                    process.kill()
                    break
                output.extend(chunk)
        except (OSError, ValueError) as error:
            read_errors.append(error)
            if process.poll() is None:
                try: process.kill()
                except OSError: pass
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        try:
            status = process.wait(timeout=15)
        except subprocess.TimeoutExpired as error:
            raise InstallError("Environment interpreter probe exceeded 15 seconds") from error
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=5)
        reader.join(timeout=5)
        if not reader.is_alive(): process.stdout.close()
    if reader.is_alive():
        raise InstallError("Environment interpreter probe output did not close within the cleanup timeout")
    if read_errors:
        raise InstallError("Environment interpreter probe output could not be read") from read_errors[0]
    if excessive.is_set():
        raise InstallError("Environment interpreter probe exceeded its output limit")
    if status != 0:
        raise InstallError(f"Environment interpreter probe failed with exit code {status}")
    try:
        report = json.loads(output)
    except (ValueError, UnicodeError) as error:
        raise InstallError("Environment interpreter probe returned invalid JSON") from error
    return report


def probe_environment(env, executable):
    if env.is_symlink() or getattr(env, "is_junction", lambda: False)() or env.resolve() != env.parent.resolve() / env.name:
        raise InstallError("Refusing a symlinked or redirected .venv")
    config = env / "pyvenv.cfg"
    if not config.is_file() or not executable.is_file():
        raise InstallError("Environment is incomplete: expected pyvenv.cfg and its Python executable")
    if config.stat().st_size > 65536:
        raise InstallError("Environment pyvenv.cfg is unexpectedly large")
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise InstallError("Environment pyvenv.cfg is not valid UTF-8") from error
    flags = [line.split("=", 1)[1].strip().lower() for line in lines
             if "=" in line and line.split("=", 1)[0].strip().lower() == "include-system-site-packages"]
    if flags != ["false"]:
        raise InstallError("Environment must explicitly disable system-site packages in pyvenv.cfg")
    report = _probe_child(executable)
    if type(report) is not dict or set(report) != {"version", "prefix", "base_prefix", "pip"}:
        raise InstallError("Environment interpreter probe returned an invalid schema")
    version = report["version"]
    if type(version) is not list or len(version) != 3 or any(type(x) is not int or x < 0 for x in version) or tuple(version) < (3, 11):
        raise InstallError("The environment's Python must be 3.11 or newer, regardless of the installer Python")
    if type(report["prefix"]) is not str or not Path(report["prefix"]).is_absolute() or Path(report["prefix"]).resolve() != env.resolve():
        raise InstallError("Environment interpreter sys.prefix does not match the chosen .venv")
    if type(report["base_prefix"]) is not str or not Path(report["base_prefix"]).is_absolute() or Path(report["base_prefix"]).resolve() == env.resolve():
        raise InstallError("Environment interpreter is not an isolated virtual environment")
    if report["pip"] is not True:
        raise InstallError("Environment pip is unavailable; repair it manually before retrying")
    return report


def pip_environment():
    """Per-child config isolation, not a sandbox for trusted package code."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("PIP_", "PYTHON"))}
    environment["PIP_CONFIG_FILE"] = os.devnull
    return environment


def plan(root, executable, backend="cuda", *, platform=None):
    platform = sys.platform if platform is None else platform
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    if backend == "cuda" and platform not in ("linux", "win32"):
        raise ValueError("CUDA installer requires Linux or Windows; use --backend cpu on this platform")
    flavor = "cu132" if backend == "cuda" else "cpu"
    version = "2.15.0" if platform == "darwin" else "2.15.0+" + flavor
    return [
        [str(executable), "-I", "-m", "pip", "--isolated", "--require-virtualenv", "install", "--index-url", "https://download.pytorch.org/whl/" + flavor, "torch==" + version],
        [str(executable), "-I", "-m", "pip", "--isolated", "--require-virtualenv", "install", "--index-url", "https://pypi.org/simple", "-e", str(root) + "[app]"],
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
        print("Validate/reuse existing environment:" if args.reuse_venv else "Create isolated environment (existing .venv requires explicit --reuse-venv):", env)
        print("Pip child policy: ignore inherited PIP_/PYTHON settings and all pip config files; no configuration files are edited.")
        for command in commands: print(command_text(command))
        print("Launch:", executable, "-m looking_glass.app")
        print("Optional GPU validation:", executable, "-m looking_glass.app_selftest")
        return 0
    if env.is_symlink() or getattr(env, "is_junction", lambda: False)() or env.resolve() != ROOT.resolve() / ".venv":
        parser.error("Refusing a symlinked or redirected .venv")
    if env.exists() and not args.reuse_venv:
        parser.error(".venv already exists. Use its python -m looking_glass.app, or explicitly --reuse-venv")
    usable = False
    stage = "environment creation"
    try:
        if not env.exists(): venv.EnvBuilder(with_pip=True).create(env)
        stage = "environment validation"
        probe_environment(env, executable)
        usable = True
        print(f"Installing {args.backend.upper()} dependencies; no drivers or model checkpoints will be downloaded.", flush=True)
        for number, command in enumerate(commands, 1):
            stage = f"{'PyTorch' if number == 1 else 'app'} dependency installation stage {number}/{len(commands)}"
            subprocess.run(command, cwd=ROOT, check=True, env=pip_environment())
        print("Installed. Restart:", command_text([executable, "-m", "looking_glass.app"]), flush=True)
        print("Optional GPU validation (local synthetic models only):", command_text([executable, "-m", "looking_glass.app_selftest"]), flush=True)
        if not args.no_launch:
            stage = "application launch"
            status = subprocess.call([str(executable), "-m", "looking_glass.app"], cwd=ROOT)
            if status:
                print(f"Installation finished, but the application exited with code {status}. Restart:",
                      command_text([executable, "-m", "looking_glass.app"]), file=sys.stderr)
            return status if status >= 0 else 1
        return 0
    except (InstallError, subprocess.SubprocessError, OSError, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            code, detail = 130, "interrupted (exit code 130)"
        elif isinstance(error, subprocess.CalledProcessError):
            code = error.returncode if 0 < error.returncode <= 255 else 1
            detail = f"subprocess exited with code {error.returncode}"
        else:
            code, detail = 1, str(error)
        print(f"Installer stopped during {stage}: {detail}. Environment preserved; no automatic repair or retry.", file=sys.stderr)
        if stage == "application launch":
            print("Installation finished; restart the app directly:", command_text([executable, "-m", "looking_glass.app"]), file=sys.stderr)
        elif usable:
            resume = [sys.executable, str(ROOT / "install.py"), "--reuse-venv", "--backend", args.backend]
            resume.append("--no-launch")
            print("After resolving the error, resume explicitly:", command_text(resume), file=sys.stderr)
        else:
            print("Inspect or manually repair the incomplete/incompatible .venv first; no app/PyTorch dependency installation or launch was attempted. No automatic deletion is performed.", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
