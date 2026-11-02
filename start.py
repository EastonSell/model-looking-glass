#!/usr/bin/env python3
"""Start this trusted checkout's already-installed local workbench, without installing.

First-time setup still uses install.py. Pass app options after --, for example:
python start.py -- --no-browser --port 8765
"""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROBE_TIMEOUT = 60

# Executed only by the validated environment, isolated from PYTHONPATH/user site.
# Do not alter sys.path: the installer's editable installation must resolve this
# checkout through that environment's trusted site configuration.
APPLICATION_PROBE = r'''
import importlib, importlib.util, pathlib, sys
try:
    expected = pathlib.Path(sys.argv[1]).resolve() / 'looking_glass' / 'app.py'
    spec = importlib.util.find_spec('looking_glass.app')
    if spec is None or not isinstance(spec.origin, str) or pathlib.Path(spec.origin).resolve() != expected or not expected.is_file():
        raise RuntimeError('installed app does not resolve to this checkout; its editable installation may be missing, moved or mismatched')
    required = ('torch', 'transformers', 'safetensors', 'numpy', 'PIL')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('missing app dependencies: ' + ', '.join(missing))
    for name in required:
        importlib.import_module(name)
    app = importlib.import_module('looking_glass.app')
    importlib.import_module('looking_glass.app_runtime')
    if not callable(getattr(app, 'main', None)):
        raise RuntimeError('installed app has no callable main entry point')
except Exception as error:
    print('App readiness check failed (' + type(error).__name__ + '): ' + str(error), file=sys.stderr)
    sys.exit(1)
'''


def command_text(command):
    values = [str(value) for value in command]
    return subprocess.list2cmdline(values) if os.name == 'nt' else shlex.join(values)


def _installer_support():
    """Load only our sibling validation helpers, never installer.main()."""
    path = ROOT / 'install.py'
    spec = importlib.util.spec_from_file_location('_mlg_launcher_install_support', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot read this checkout\'s install.py validation helpers')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print the launch command without validating, installing or starting anything')
    parser.add_argument('app_args', nargs=argparse.REMAINDER, help='App arguments after --')
    args = parser.parse_args(argv)
    app_args = args.app_args[1:] if args.app_args[:1] == ['--'] else args.app_args
    environment = ROOT / '.venv'
    executable = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    command = [str(executable), '-I', '-m', 'looking_glass.app', *app_args]
    if args.dry_run:
        print('Existing environment only; no validation, creation, installation or launch performed.')
        print('Working directory:', ROOT)
        print('Launch:', command_text(command))
        return 0
    if sys.version_info < (3, 11):
        print('Use Python 3.11 or newer to run start.py; no environment was changed.', file=sys.stderr)
        return 1
    stage = 'environment validation'
    usable = False
    try:
        if not environment.exists() and not environment.is_symlink():
            print('This checkout has no .venv. Run install.py explicitly for first-time setup; start.py never installs anything.', file=sys.stderr)
            return 1
        support = _installer_support()
        support.probe_environment(environment, executable)
        usable = True
        stage = 'app/dependency readiness check'
        readiness = subprocess.run([str(executable), '-I', '-B', '-c', APPLICATION_PROBE, str(ROOT)],
                                   cwd=ROOT, timeout=PROBE_TIMEOUT, check=False)
        if readiness.returncode:
            print('Nothing was installed or repaired. Review the readiness error above. To repair intentionally, use install.py --reuse-venv --backend <your same chosen cpu or cuda backend> --no-launch; replace the placeholder, do not switch builds just to relaunch.', file=sys.stderr)
            return readiness.returncode if 0 < readiness.returncode <= 255 else 1
        stage = 'application launch'
        print('Starting existing installation:', command_text(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            print(f'Application exited with code {result.returncode}; no installation or repair was attempted. Restart when ready: {command_text([sys.executable, ROOT / "start.py", "--", *app_args])}', file=sys.stderr)
        return result.returncode if 0 <= result.returncode <= 255 else 1
    except KeyboardInterrupt:
        print(f'Launcher interrupted during {stage} (exit code 130); environment preserved.', file=sys.stderr)
        return 130
    except (ImportError, RuntimeError, subprocess.SubprocessError, OSError, SyntaxError) as error:
        # All operations above are local validation or a foreground subprocess.
        # Do not retry, rebuild the venv, invoke pip, or claim a successful start.
        if isinstance(error, subprocess.TimeoutExpired):
            detail = f'readiness check exceeded {PROBE_TIMEOUT} seconds'
        else:
            detail = f'{type(error).__name__}: {error}'
        print(f'Launcher stopped during {stage}: {detail}. Existing environment preserved.', file=sys.stderr)
        if usable:
            print('No packages were installed. Inspect the error, then retry start.py or explicitly repair using install.py --reuse-venv with your chosen --backend.', file=sys.stderr)
        else:
            print('Inspect or manually repair the incomplete/incompatible .venv, or use install.py explicitly for setup. No automatic repair or deletion is performed.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
