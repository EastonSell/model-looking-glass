"""Run one selector stage and record its same-interpreter package versions.

Invoke the direct base interpreter with -I -S -B and a fresh empty
-X pycache_prefix. The surrounding
custodian observes this process's actual exit and retains partial records.
This wrapper neither forms a pool nor loads a pretrained model.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading


SOURCE_PATHS = ('tools/run_flan_sentiment_selector_stage_v1.py',
                'tools/run_flan_sentiment_selector_v1.py',
                'looking_glass/__init__.py', 'looking_glass/flan_t5_sentiment_selector_v1.py',
                'looking_glass/flan_t5_sentiment_screen.py', 'looking_glass/flan_t5_feasibility.py',
                'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md', 'docs/FLAN_T5_SENTIMENT_SELECTOR_V1.md')


def _clear_git_environment():
    for name in tuple(os.environ):
        if name.upper().startswith('GIT_'):
            os.environ.pop(name, None)


def _raise_with_priority(primary, cleanup_error):
    if primary is None:
        raise cleanup_error
    if not isinstance(primary, Exception) or isinstance(cleanup_error, Exception):
        raise primary from cleanup_error
    raise cleanup_error from primary


def _empty_prefix():
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()):
        raise ValueError('run the direct base interpreter with -I -S -B and a fresh -X pycache_prefix')
    path = Path(sys.pycache_prefix)
    if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
        raise ValueError('bytecode prefix is not fresh and empty')


def _dependency_site(venv):
    """Validate the chosen venv without executing site startup or .pth files."""
    venv = Path(venv)
    if (not venv.is_absolute() or not venv.is_dir() or venv.is_symlink()
            or getattr(venv.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ValueError('venv must be an absolute ordinary directory')
    config = venv / 'pyvenv.cfg'
    if not config.is_file() or config.is_symlink() or config.stat().st_size > 16384:
        raise ValueError('venv configuration is absent or too large')
    raw = config.read_bytes()
    entries = {}
    for line in raw.decode('utf8').splitlines():
        if '=' in line:
            name, value = (part.strip() for part in line.split('=', 1))
            name = name.lower()
            if name in entries:
                raise ValueError('duplicate venv configuration entry')
            entries[name] = value
    home = entries.get('home', '')
    if (not home or not Path(home).is_absolute()
            or Path(home).resolve() != Path(sys.base_prefix).resolve()
            or entries.get('implementation') != 'CPython'
            or entries.get('version_info') != '.'.join(map(str, sys.version_info[:3]))
            or entries.get('include-system-site-packages', '').lower() != 'false'
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()):
        raise ValueError('venv does not match the direct base interpreter')
    site = venv / ('Lib/site-packages' if os.name == 'nt' else
                   'lib/python%d.%d/site-packages' % sys.version_info[:2])
    if (not site.is_dir() or site.is_symlink()
            or getattr(site.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ValueError('venv site-packages is unavailable or aliased')
    return site.resolve(), hashlib.sha256(raw).hexdigest()


def _process_identity(venv, pid, parent_pid):
    site, config_sha = _dependency_site(venv)
    def path_sha(path):
        return hashlib.sha256(str(Path(path).resolve()).encode('utf8')).hexdigest()
    return {'schema': 'flan-t5-sentiment-direct-child-v1', 'pid': pid, 'parent_pid': parent_pid,
            'executable_path_sha256': path_sha(sys.executable),
            'prefix_path_sha256': path_sha(sys.prefix),
            'dependency_site_path_sha256': path_sha(site), 'venv_config_sha256': config_sha}


def _check_child_identity(log_path, venv, pid):
    try:
        with Path(log_path).open('rb') as stream:
            line = stream.readline(4097)
        if not line.endswith(b'\n') or len(line) > 4096:
            return False
        return json.loads(line) == _process_identity(venv, pid, os.getpid())
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def _receive_grant(expected):
    if type(expected) is not str or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('selector child grant digest is invalid')
    done = threading.Event()
    def expire():
        if not done.wait(30):
            os._exit(70)
    threading.Thread(target=expire, name='flan-selector-grant-deadline', daemon=True).start()
    try:
        grant = sys.stdin.buffer.read(33)
        if len(grant) != 32 or hashlib.sha256(grant).hexdigest() != expected:
            raise ValueError('selector child grant is absent or invalid')
    finally:
        done.set()


def _snapshot(root, revision):
    _clear_git_environment()
    git = ['git', '--no-replace-objects']
    head = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=root).strip().decode('ascii')
    if (head != revision or len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision)
            or subprocess.check_output(git + ['cat-file', '-t', revision], cwd=root).strip() != b'commit'):
        raise ValueError('selector stage source revision differs')
    result = {}
    for name in SOURCE_PATHS:
        data = (root / name).read_bytes()
        if data != subprocess.check_output(git + ['show', revision + ':' + name], cwd=root):
            raise ValueError('selector stage source differs from its commit')
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def _write_new(path, record):
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short selector-stage receipt write')
        stream.flush()
        os.fsync(stream.fileno())


def _argument(arguments, option):
    positions = [i for i, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 == len(arguments):
        raise ValueError('selector stage argument is missing or duplicated')
    return arguments[positions[0] + 1]


def main(argv=None):
    _empty_prefix()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--stage-receipt', required=True, type=Path)
    parser.add_argument('--venv', required=True, type=Path)
    parser.add_argument('--grant-sha256', required=True)
    parser.add_argument('selector_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    _receive_grant(args.grant_sha256)
    _clear_git_environment()
    site, _ = _dependency_site(args.venv)
    if str(site) in sys.path:
        raise ValueError('venv site-packages was already on the search path')
    sys.path.append(str(site))
    print(json.dumps(_process_identity(args.venv, os.getpid(), os.getppid()), sort_keys=True), flush=True)
    rest = args.selector_args
    if rest and rest[0] == '--':
        rest = rest[1:]
    if not rest or rest[0] not in ('commit-pool', 'select') or not args.stage_receipt.is_absolute():
        raise ValueError('expected a selector stage and an absolute new receipt path')
    stage = rest[0]
    root = Path(__file__).resolve().parents[1]
    if Path(_argument(rest, '--repo')).resolve(strict=True) != root:
        raise ValueError('selector checkout differs from this source-bound wrapper')
    source = _snapshot(root, args.revision)
    versions = {'python': platform.python_version(),
                'transformers': importlib.metadata.version('transformers'),
                'tokenizers': importlib.metadata.version('tokenizers')}
    original_argv = sys.argv
    try:
        launcher = root / 'tools/run_flan_sentiment_selector_v1.py'
        spec = importlib.util.spec_from_file_location('_flan_sentiment_selector_launcher', launcher)
        if spec is None or spec.loader is None:
            raise ValueError('cannot resolve selector launcher')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [str(launcher), *rest]
        module.main()
        _empty_prefix()
        if _snapshot(root, args.revision) != source:
            raise ValueError('selector stage source changed')
        if stage == 'select':
            import transformers
            import tokenizers
            if (transformers.__version__ != versions['transformers']
                    or tokenizers.__version__ != versions['tokenizers']
                    or not Path(transformers.__file__).resolve().is_relative_to(site)
                    or not Path(tokenizers.__file__).resolve().is_relative_to(site)):
                raise ValueError('loaded tokenizer libraries differ from the selected venv')
        pool_card = json.loads(Path(_argument(rest, '--pool-card')).read_bytes())
        selection_card = (json.loads(Path(_argument(rest, '--selection-card')).read_bytes())
                          if stage == 'select' else None)
        _write_new(args.stage_receipt, {'schema': 'flan-t5-sentiment-selector-stage-runtime-v1',
                   'stage': stage, 'status': 'completed', 'selector_revision': args.revision,
                   'source_sha256': source, 'versions': versions,
                   'pool_commitment': pool_card['commitment'],
                   'selection_commitment': selection_card['commitment'] if selection_card else None,
                   'scope': 'Same-process version observation; external direct-exit and custody checks remain required.'})
    except BaseException as error:
        if not args.stage_receipt.exists():
            try:
                _write_new(args.stage_receipt, {'schema': 'flan-t5-sentiment-selector-stage-runtime-v1',
                           'stage': stage, 'status': 'incomplete', 'selector_revision': args.revision,
                           'error_type': type(error).__name__})
            except BaseException as receipt_error:
                _raise_with_priority(error, receipt_error)
        raise
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
