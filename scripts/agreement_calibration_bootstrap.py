"""Load only the clean committed agreement calibration sources in isolation.

This bootstrap does not authorize device work. The dedicated controller and
owner retain their independent readiness gates and supervisor contract.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time


_DOCS = (
    'docs/AGREEMENT_GPU_CALIBRATION_V1.md',
    'docs/AGREEMENT_CUDA_ADAPTER.md',
    'docs/AGREEMENT_CALIBRATION_SAMPLING_DRAFT.md',
    'docs/TRUSTED_SERVICE_PROFILE_V1.md',
)
_SCRIPTS = ('scripts/agreement_calibration_bootstrap.py', 'scripts/v2_source_guards.py')


def _plain_file(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink() or
            getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise RuntimeError('source must be a regular non-link file')


def _current_python(repo):
    package = repo / 'looking_glass'
    package_info = package.lstat()
    if (not stat.S_ISDIR(package_info.st_mode) or package.is_symlink() or
            getattr(package_info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise RuntimeError('project source package is not a plain directory')
    current = set()
    def fail_walk(error):
        raise RuntimeError('cannot inventory the project source') from error
    for base, dirs, names in os.walk(package, followlinks=False, onerror=fail_walk):
        dirs[:] = sorted(name for name in dirs if name != '__pycache__')
        for name in dirs:
            directory = Path(base) / name
            info = directory.lstat()
            if (not stat.S_ISDIR(info.st_mode) or directory.is_symlink() or
                    getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
                raise RuntimeError('project source directory is not a plain directory')
        for name in names:
            if name.lower().endswith('.py'):
                path = Path(base) / name
                _plain_file(path)
                current.add(path.relative_to(repo).as_posix())
    return current


def capture_source(repo, git):
    """Compare all package Python bytes and protocol sources to one commit."""
    repo = Path(repo).resolve(strict=True)
    git = Path(git).resolve(strict=True)
    git_env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    git_env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                    'GIT_CONFIG_SYSTEM': os.devnull, 'GIT_TERMINAL_PROMPT': '0',
                    'GIT_OPTIONAL_LOCKS': '0', 'GIT_NO_REPLACE_OBJECTS': '1'})
    commit = subprocess.run([str(git), '--no-replace-objects', 'rev-parse', 'HEAD'], cwd=repo,
                            env=git_env, capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    dirty = subprocess.run([str(git), '--no-replace-objects', 'status', '--porcelain', '--untracked-files=no'],
                           cwd=repo, env=git_env, capture_output=True, text=True, check=True, timeout=20).stdout
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None or dirty:
        raise RuntimeError('calibration requires a clean committed checkout')
    result = subprocess.run([str(git), '--no-replace-objects', 'archive', '--format=tar', commit, '--',
                             'looking_glass', *_DOCS, *_SCRIPTS], cwd=repo,
                            env=git_env, capture_output=True, check=True, timeout=30)
    if len(result.stdout) > 32 * 1024 * 1024:
        raise RuntimeError('committed source archive exceeds the bound')
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        for item in archive:
            name = item.name
            selected = ((name.startswith('looking_glass/') and name.lower().endswith('.py')) or
                        name in _DOCS or name in _SCRIPTS)
            if not selected:
                continue
            if (not item.isfile() or name in sources or item.size > 4 * 1024 * 1024 or
                    '..' in Path(name).parts or '\\' in name):
                raise RuntimeError('unsupported committed source entry')
            stream = archive.extractfile(item)
            if stream is None:
                raise RuntimeError('committed source is unavailable')
            raw = stream.read()
            path = repo / name
            _plain_file(path)
            if path.read_bytes() != raw:
                raise RuntimeError('checkout differs from committed source')
            sources[name] = raw
    if not {'looking_glass/__init__.py', *_DOCS, *_SCRIPTS} <= sources.keys():
        raise RuntimeError('required calibration source is missing')
    committed_python = {name for name in sources if name.startswith('looking_glass/')}
    if _current_python(repo) != committed_python:
        raise RuntimeError('untracked or missing project Python source')
    manifest = {'schema': 'agreement-controller-source-v1', 'commit': commit,
                'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(sources.items())}}
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return manifest, hashlib.sha256(canonical).hexdigest(), sources


def capture_parent_frozen_source(repo, run_dir, expected_digest):
    """Verify the parent's frozen inventory without spawning child processes."""
    repo, run_dir = Path(repo).resolve(strict=True), Path(run_dir)
    if (not run_dir.is_absolute() or run_dir.resolve(strict=True) != run_dir or run_dir.is_symlink() or
            getattr(run_dir.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise RuntimeError('private run directory is not canonical')
    path = run_dir / 'controller-source.json'
    _plain_file(path)
    if path.stat().st_size > 1024 * 1024:
        raise RuntimeError('parent source inventory is too large')
    raw = path.read_bytes()
    def unique(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError('duplicate source inventory key')
            result[name] = value
        return result
    manifest = json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite inventory')))
    if (type(manifest) is not dict or set(manifest) != {'schema', 'commit', 'files'} or
            manifest['schema'] != 'agreement-controller-source-v1' or
            type(manifest['commit']) is not str or re.fullmatch(r'[0-9a-f]{40}', manifest['commit']) is None or
            type(manifest['files']) is not dict or len(manifest['files']) > 4096):
        raise RuntimeError('parent source inventory is invalid')
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != expected_digest or raw != canonical + b'\n':
        raise RuntimeError('parent source inventory does not match its commitment')
    sources = {}
    for name, expected in manifest['files'].items():
        if (type(name) is not str or '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts or
                not ((name.startswith('looking_glass/') and name.lower().endswith('.py')) or
                     name in _DOCS or name in _SCRIPTS) or type(expected) is not str or
                re.fullmatch(r'[0-9a-f]{64}', expected) is None):
            raise RuntimeError('parent source inventory contains an invalid entry')
        source_path = repo / name
        _plain_file(source_path)
        if source_path.stat().st_size > 4 * 1024 * 1024:
            raise RuntimeError('project source exceeds its bound')
        content = source_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError('source differs from the parent freeze')
        sources[name] = content
    if not {'looking_glass/__init__.py', *_DOCS, *_SCRIPTS} <= sources.keys():
        raise RuntimeError('parent source inventory is incomplete')
    if _current_python(repo) != {name for name in sources if name.startswith('looking_glass/')}:
        raise RuntimeError('untracked or missing project Python source')
    return manifest, digest, sources


class _FrozenLoader(importlib.abc.Loader):
    def __init__(self, path, source):
        self.path, self.source = path, source

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path)
        exec(compile(self.source, str(self.path), 'exec', dont_inherit=True), module.__dict__)


class _ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, repo, sources):
        self.repo, self.sources = repo, sources

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('looking_glass.'):
            return None
        suffix = fullname.removeprefix('looking_glass.').replace('.', '/')
        name = 'looking_glass/' + suffix + '.py'
        source = self.sources.get(name)
        if source is None:
            raise ImportError('project import is outside the frozen source inventory')
        loader = _FrozenLoader(self.repo / name, source)
        return importlib.util.spec_from_loader(fullname, loader, origin=str(self.repo / name))


def _check_meta_path(project_finder=None):
    """Disallow finders that can bypass the selected dependency site."""
    expected = ([project_finder] if project_finder is not None else []) + [
        importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    ]
    if len(sys.meta_path) != len(expected) or any(
            actual is not wanted for actual, wanted in zip(sys.meta_path, expected)):
        raise RuntimeError('unexpected Python import finder')


def _check_dependency_origins(site):
    """Check top-level package resolution without importing the packages."""
    site = Path(site).resolve(strict=True)
    if sum(Path(entry).resolve() == site for entry in sys.path if entry) != 1:
        raise RuntimeError('dependency directory must occur exactly once')
    for name in ('torch', 'transformers', 'huggingface_hub', 'numpy', 'safetensors', 'tokenizers'):
        actual = importlib.machinery.PathFinder.find_spec(name, sys.path)
        expected = importlib.machinery.PathFinder.find_spec(name, [str(site)])
        if (actual is None or expected is None or actual.origin is None or
                actual.origin != expected.origin):
            raise ImportError('dependency does not resolve from the configured site')
        origin = Path(actual.origin).resolve(strict=True)
        if not origin.is_relative_to(site):
            raise ImportError('dependency origin is outside the configured site')


def _wait_for_supervisor_grant():
    """Observe, but do not consume, the supervisor's actual launch token.

    The owner session later consumes that same token and rechecks its parent.
    The precheck is before Git or any other child process is started, so the
    supervisor can attach its process Job before permitting descendants.
    """
    if os.name != 'nt':
        raise RuntimeError('this calibration bootstrap requires Windows')
    import ctypes
    from ctypes import wintypes
    endpoint = int(os.environ['MLG_OWNER_LAUNCH_HANDLE'])
    parent = int(os.environ['MLG_SUPERVISOR_PID'])
    if endpoint < 0 or parent <= 0 or os.getppid() != parent:
        raise RuntimeError('supervisor launch identity is invalid')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.PeekNamedPipe.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                     wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD)
    kernel.PeekNamedPipe.restype = wintypes.BOOL
    deadline = time.monotonic() + 30
    while True:
        data = ctypes.create_string_buffer(1)
        received, available = wintypes.DWORD(), wintypes.DWORD()
        if not kernel.PeekNamedPipe(wintypes.HANDLE(endpoint), data, 1, ctypes.byref(received),
                                    ctypes.byref(available), None):
            raise RuntimeError('supervisor launch pipe is unavailable')
        if available.value:
            if received.value != 1 or data.raw != b'\x01':
                raise RuntimeError('supervisor launch token is invalid')
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('supervisor launch token timed out')
        time.sleep(.01)


def main():
    prefix = sys.pycache_prefix
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or
            not isinstance(prefix, str) or sys._xoptions.get('pycache_prefix') != prefix):
        raise RuntimeError('start with -I -S -B and a fresh absolute pycache prefix')
    directory = Path(prefix)
    if not directory.is_absolute():
        raise RuntimeError('pycache prefix must be absolute')
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or directory.is_symlink() or
            getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400) or
            next(directory.iterdir(), None) is not None):
        raise RuntimeError('pycache prefix must be a plain empty directory')
    if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
        raise RuntimeError('project modules were imported before bootstrap')
    _check_meta_path()
    if '--owner' in sys.argv[1:]:
        _wait_for_supervisor_grant()
    if len(sys.argv) < 3 or sys.argv[1] != '--bootstrap-site':
        raise ValueError('an explicit dependency directory is required')
    site = Path(sys.argv[2]).resolve(strict=True)
    if not site.is_dir():
        raise ValueError('dependency directory must exist')
    del sys.argv[1:3]
    if len(sys.argv) >= 3 and sys.argv[1] == '--git-executable':
        git = Path(sys.argv[2]).resolve(strict=True)
        del sys.argv[1:3]
    else:
        git = shutil.which('git')
        if git is None:
            raise RuntimeError('Git is required for source verification')
        git = Path(git).resolve(strict=True)
    repo = Path(__file__).resolve(strict=True).parent.parent
    if '--owner' in sys.argv[1:]:
        index = sys.argv.index('--owner')
        if len(sys.argv) != index + 7:
            raise ValueError('invalid supervised owner arguments')
        manifest, digest, sources = capture_parent_frozen_source(repo, sys.argv[index + 1], sys.argv[index + 4])
    else:
        manifest, digest, sources = capture_source(repo, git)
    package_path = repo / 'looking_glass' / '__init__.py'
    loader = _FrozenLoader(package_path, sources['looking_glass/__init__.py'])
    spec = importlib.util.spec_from_loader('looking_glass', loader, origin=str(package_path), is_package=True)
    spec.submodule_search_locations = [str(package_path.parent)]
    module = importlib.util.module_from_spec(spec)
    module.__agreement_source_manifest__ = manifest
    module.__agreement_source_sha256__ = digest
    module.__agreement_git__ = str(git)
    module.__agreement_recheck__ = capture_source
    finder = _ProjectFinder(repo, sources)
    sys.meta_path.insert(0, finder)
    module.__agreement_validate_import_finders__ = lambda: _check_meta_path(finder)
    sys.modules['looking_glass'] = module
    spec.loader.exec_module(module)
    guard_path = repo / 'scripts' / 'v2_source_guards.py'
    guard_loader = _FrozenLoader(guard_path, sources['scripts/v2_source_guards.py'])
    guard_spec = importlib.util.spec_from_loader('_agreement_source_guards', guard_loader, origin=str(guard_path))
    guard_module = importlib.util.module_from_spec(guard_spec)
    sys.modules['_agreement_source_guards'] = guard_module
    guard_spec.loader.exec_module(guard_module)
    sys.path.append(str(site))
    if repo in (Path(entry).resolve() for entry in sys.path if entry):
        raise RuntimeError('checkout must not be a generic import directory')
    _check_dependency_origins(site)
    module.__agreement_validate_import_finders__()
    os.environ['LOOKING_GLASS_AGREEMENT_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_AGREEMENT_BOOTSTRAP_SITE'] = str(site)
    os.environ['MLG_AGREEMENT_SITE'] = str(site)
    from looking_glass.agreement_calibration_controller import main as controller_main
    is_owner = '--owner' in sys.argv[1:]
    status = controller_main(sys.argv[1:])
    if is_owner:
        # Keep the device lease and all process roots until the OS reaps us.
        os._exit(status)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
