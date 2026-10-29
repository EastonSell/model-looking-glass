"""Isolated, source-bound entry point for the bounded worker failure-validation probe."""

import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import stat
import subprocess
import sys

_OWNER = 'looking_glass/worker_failure_gpu.py'
_BOOTSTRAP = 'scripts/worker_failure_gpu_bootstrap.py'
_INIT = 'looking_glass/__init__.py'
_MANIFEST = 'evidence/gpu-tap-latency-repeated-v1/schedule.json'
_MANIFEST_SHA256 = 'ce2f990a6937da86565a21c52f13ad54f11b1e683421a8a2060e544694fe6bf9'
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_CHILD_MANIFEST = 'protected-manifest/source-manifest.json'
_MANIFEST_ENV = 'LOOKING_GLASS_FAILURE_VALIDATION_MANIFEST_SHA256'
_UNAME_ENV = 'LOOKING_GLASS_FAILURE_VALIDATION_PLATFORM_UNAME'


def _checked_path(path):
    lexical = Path(os.path.abspath(path))
    for item in (lexical, *lexical.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise RuntimeError('symlinks and reparse points are unsupported')
    return lexical.resolve(strict=True)


def _relative_source(value):
    if type(value) is not str or not value or '\\' in value:
        raise ValueError('invalid project source path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in path.parts) or str(path) != value:
        raise ValueError('invalid project source path')
    return value


def _git(git, checkout, *args):
    environment = {key: os.environ[key] for key in ('SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT')
                   if key in os.environ}
    environment.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                        'GIT_CONFIG_COUNT': '0', 'GIT_NO_REPLACE_OBJECTS': '1',
                        'GIT_TERMINAL_PROMPT': '0'})
    return subprocess.run([str(git), '--no-replace-objects', *args], cwd=checkout, env=environment,
                          capture_output=True, check=True, timeout=15).stdout


def _verify_one(git, checkout, commit, relative):
    _relative_source(relative)
    # On Windows with core.symlinks=false, a Git symlink can be represented
    # by a regular worktree file with identical bytes and a clean status.
    # Bind both the tree entry type and its bytes to the declared path.
    entry = _git(git, checkout, 'ls-tree', '-z', '--full-tree', commit, '--', ':(literal)' + relative)
    if not entry.endswith(b'\0') or entry.count(b'\0') != 1:
        raise RuntimeError('project source has an invalid committed tree entry')
    metadata, separator, recorded_path = entry[:-1].partition(b'\t')
    fields = metadata.split(b' ')
    if (separator != b'\t' or recorded_path != relative.encode('utf-8') or len(fields) != 3
            or fields[0] not in (b'100644', b'100755') or fields[1] != b'blob'
            or re.fullmatch(b'[0-9a-f]{40}', fields[2]) is None):
        raise RuntimeError('project source is not a committed regular file')
    path = _checked_path(checkout / relative)
    if not path.is_file() or not path.is_relative_to(checkout):
        raise RuntimeError('project source is outside the checkout')
    with path.open('rb') as stream:
        actual = stream.read(_MAX_SOURCE_BYTES + 1)
    if len(actual) > _MAX_SOURCE_BYTES:
        raise RuntimeError('project source exceeds its byte limit')
    stored = _git(git, checkout, 'show', commit + ':' + relative)
    if actual != stored:
        raise RuntimeError('project source differs from the committed tree')
    return actual


def _literal_source_paths(owner_bytes):
    tree = ast.parse(owner_bytes, filename=_OWNER)
    bindings = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
                and node.id == '_SOURCE_PATHS' and isinstance(node.ctx, ast.Store)]
    values = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '_SOURCE_PATHS'):
            values.append(ast.literal_eval(node.value))
    if len(bindings) != 1 or len(values) != 1 or type(values[0]) is not tuple or not values[0]:
        raise RuntimeError('owner must declare one literal source tuple')
    paths = tuple(_relative_source(path) for path in values[0])
    if len(paths) != len(set(paths)) or not {_BOOTSTRAP, _OWNER, _INIT, _MANIFEST}.issubset(paths):
        raise RuntimeError('owner source tuple is incomplete or duplicated')
    return paths


def _verify_source(checkout, commit, git):
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('a complete source commit is required')
    if _git(git, checkout, 'rev-parse', 'HEAD').decode('ascii').strip() != commit:
        raise RuntimeError('checkout is not at the supplied source commit')
    if _git(git, checkout, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('checkout has tracked source changes')
    _verify_one(git, checkout, commit, _BOOTSTRAP)
    owner = _verify_one(git, checkout, commit, _OWNER)
    paths = _literal_source_paths(owner)
    for relative in paths:
        data = _verify_one(git, checkout, commit, relative)
        if relative == _MANIFEST and hashlib.sha256(data).hexdigest() != _MANIFEST_SHA256:
            raise RuntimeError('fixed timing schedule does not match the pinned digest')
    return paths


def _verify_child_manifest(checkout, commit, run_dir, run_id):
    """Verify parent-bound source bytes without spawning a Job descendant.

    The parent must first verify Git blob bytes and modes and create this
    digest-bound manifest. The manifest is a cooperative source binding, not
    an attestation against a hostile local process or later source mutation.
    """
    digest = os.environ.get(_MANIFEST_ENV, '')
    directory = Path(run_dir)
    if (re.fullmatch('[0-9a-f]{40}', commit) is None
            or re.fullmatch('[0-9a-f]{64}', digest) is None
            or not directory.is_absolute() or _checked_path(directory) != directory
            or not directory.is_dir() or type(run_id) is not str or not run_id):
        raise ValueError('invalid child source-manifest binding')
    path = _checked_path(directory / _CHILD_MANIFEST)
    if not path.is_file() or not path.is_relative_to(directory):
        raise ValueError('child source manifest is unavailable')
    with path.open('rb') as stream:
        raw = stream.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
        raise RuntimeError('child source-manifest digest differs')
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate source-manifest field')
            value[key] = item
        return value
    def reject_constant(_value):
        raise ValueError('nonfinite source-manifest value')
    manifest = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject_constant)
    canonical = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if (canonical != raw or type(manifest) is not dict
            or set(manifest) != {'schema', 'run_id', 'source_commit', 'source_files'}
            or manifest['schema'] != 'worker-failure-source-manifest-v1'
            or manifest['run_id'] != run_id or manifest['source_commit'] != commit
            or type(manifest['source_files']) is not dict):
        raise ValueError('child source manifest is not the fixed schema')
    files = manifest['source_files']
    def verify(relative):
        _relative_source(relative)
        entry = files.get(relative)
        if (type(entry) is not dict or set(entry) != {'mode', 'sha256'}
                or entry['mode'] not in ('100644', '100755')
                or type(entry['sha256']) is not str or re.fullmatch('[0-9a-f]{64}', entry['sha256']) is None):
            raise ValueError('child source entry is invalid')
        source = _checked_path(checkout / relative)
        if not source.is_file() or not source.is_relative_to(checkout):
            raise RuntimeError('child source is outside the checkout')
        with source.open('rb') as stream:
            data = stream.read(_MAX_SOURCE_BYTES + 1)
        if len(data) > _MAX_SOURCE_BYTES or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise RuntimeError('child source bytes differ from the parent manifest')
        return data
    owner = verify(_OWNER)
    paths = _literal_source_paths(owner)
    if set(files) != set(paths):
        raise ValueError('child source tuple differs from the parent manifest')
    for relative in paths:
        data = verify(relative)
        if relative == _MANIFEST and hashlib.sha256(data).hexdigest() != _MANIFEST_SHA256:
            raise RuntimeError('fixed timing schedule does not match the pinned digest')
    return paths


def _prime_windows_platform():
    """Reuse the parent's real uname observation without a Job subprocess.

    CPython 3.11's Windows ``platform.uname`` otherwise invokes ``ver``
    during Torch import. The parent obtains these fields before spawning and
    retains the node privately; this does not authenticate the host.
    """
    raw = os.environ.get(_UNAME_ENV, '')
    if (sys.platform != 'win32' or sys.implementation.name != 'cpython'
            or sys.version_info[:3] != (3, 11, 15) or not 1 <= len(raw) <= 4096):
        raise RuntimeError('the pinned Windows platform handoff is required')
    value = json.loads(raw)
    keys = ('system', 'node', 'release', 'version', 'machine', 'processor')
    if (type(value) is not dict or set(value) != set(keys)
            or json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) != raw
            or any(type(value[key]) is not str or len(value[key]) > 255
                   or any(not char.isprintable() for char in value[key]) for key in keys)
            or value['system'] != 'Windows' or value['machine'] != 'AMD64'
            or re.fullmatch('[0-9]+\\.[0-9]+\\.[0-9]+', value['version']) is None):
        raise ValueError('parent Windows platform observation is invalid')
    import platform
    if getattr(platform, '_uname_cache', None) is not None:
        raise RuntimeError('platform identity was initialized before the handoff')
    cached = platform.uname_result(*(value[key] for key in keys[:5]))
    cached.processor = value['processor']
    platform._uname_cache = cached


class _RejectSourceless(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
            raise ImportError('sourceless dependency bytecode is unsupported')
        return None


class _ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package, source_paths):
        self.package = _checked_path(package)
        self.modules = frozenset(path.removeprefix('looking_glass/').removesuffix('.py')
                                 for path in source_paths if path.startswith('looking_glass/')
                                 and path.endswith('.py') and '/' not in path.removeprefix('looking_glass/'))

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('looking_glass.'):
            return None
        name = fullname.removeprefix('looking_glass.')
        if name not in self.modules:
            raise ImportError('project module is outside the committed source tuple')
        source = _checked_path(self.package / (name + '.py'))
        if source.parent != self.package or not source.is_file():
            raise ImportError('project source is unavailable')
        return importlib.util.spec_from_file_location(fullname, source)


def main():
    prefix = sys.pycache_prefix
    if (not isinstance(prefix, str) or sys._xoptions.get('pycache_prefix') != prefix
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not Path(prefix).is_absolute()):
        raise RuntimeError('start with -I -S -B and an absolute empty -X pycache_prefix')
    directory = _checked_path(prefix)
    if not directory.is_dir() or next(directory.iterdir(), None) is not None:
        raise RuntimeError('the bytecode prefix must be a plain empty directory')
    if len(sys.argv) < 7 or (sys.argv[1], sys.argv[3], sys.argv[5]) != (
            '--bootstrap-site', '--source-commit', '--git-executable'):
        raise ValueError('site, source commit and Git executable must be explicit')
    site_arg, commit, git_arg = sys.argv[2], sys.argv[4], sys.argv[6]
    if not Path(site_arg).is_absolute() or not Path(git_arg).is_absolute():
        raise ValueError('site and Git executable must be absolute')
    site, git = _checked_path(site_arg), _checked_path(git_arg)
    checkout = _checked_path(__file__).parent.parent
    if (not site.is_dir() or not git.is_file() or site == checkout or site.is_relative_to(checkout)
            or checkout.is_relative_to(site) or directory == checkout or directory.is_relative_to(checkout)
            or directory == site or directory.is_relative_to(site)):
        raise ValueError('site, checkout and bytecode prefix must be separate')
    if any(name == root or name.startswith(root + '.') for name in sys.modules
           for root in ('looking_glass', 'torch', 'transformers', 'numpy', 'tokenizers',
                        'safetensors', 'huggingface_hub')):
        raise RuntimeError('project or model dependencies were imported before preflight')
    for entry in sys.path:
        location = Path(entry or os.getcwd()).resolve()
        if location == checkout or location.is_relative_to(checkout):
            raise RuntimeError('checkout paths must not appear on the import search path')
    args = sys.argv[7:]
    if '--owner' in args:
        if (len(args) != 6 or args[0] != '--owner' or args[4] != commit
                or not Path(args[5]).is_absolute() or _checked_path(args[5]) != git):
            raise ValueError('owner source arguments do not match the bootstrap preflight')
    source_paths = (_verify_child_manifest(checkout, commit, args[1], args[3])
                    if args and args[0] == '--owner' else _verify_source(checkout, commit, git))
    if args and args[0] == '--owner':
        _prime_windows_platform()
    del sys.argv[1:7]
    sys.path.append(str(site))
    sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), _RejectSourceless())
    package = _checked_path(checkout / 'looking_glass')
    source = _checked_path(package / '__init__.py')
    if source.parent != package or not source.is_file():
        raise ImportError('the project package is unavailable')
    spec = importlib.util.spec_from_file_location('looking_glass', source,
                                                  submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.meta_path.insert(0, _ProjectFinder(package, source_paths))
    sys.modules['looking_glass'] = module
    spec.loader.exec_module(module)
    os.environ['LOOKING_GLASS_FAILURE_VALIDATION_SOURCE_COMMIT'] = commit
    os.environ['LOOKING_GLASS_FAILURE_VALIDATION_GIT_EXE'] = str(git)
    os.environ['LOOKING_GLASS_FAILURE_VALIDATION_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_FAILURE_VALIDATION_BOOTSTRAP_SITE'] = str(site)
    runpy.run_module('looking_glass.worker_failure_gpu', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    if '--owner' in sys.argv:
        # The CUDA owner must not run Python shutdown hooks or finalizers after
        # an uncertain drain. Its own failure and attestation records are
        # written before it returns; the supervisor treats every other exit as
        # an unclean attempt. The owner keeps the physical lease rooted until
        # this hard process exit.
        try:
            main()
        except SystemExit as error:
            status = 0 if error.code is None or type(error.code) is int and error.code == 0 else 1
            os._exit(status)
        except BaseException:
            os._exit(1)
        os._exit(0)
    main()
