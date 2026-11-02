"""Isolated, source-bound entry point for the exposed-pair CLS CUDA exploration."""

import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import stat
import subprocess
import sys

_OWNER = 'looking_glass/vision_coco_cls_exposed_gpu.py'
_BOOTSTRAP = 'scripts/vision_coco_cls_exposed_gpu_bootstrap.py'
_INIT = 'looking_glass/__init__.py'
_MANIFEST = 'experiments/vision_coco_recognized_v4.json'
_MANIFEST_SHA256 = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'
_MAX_SOURCE_BYTES = 4 * 1024 * 1024


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
            raise RuntimeError('public case manifest does not match the pinned digest')
    return paths


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
    if any(name == 'looking_glass' or name.startswith('looking_glass.')
           or name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub')
           for name in sys.modules):
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
    source_paths = _verify_source(checkout, commit, git)
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
    os.environ['LOOKING_GLASS_VISION_SOURCE_COMMIT'] = commit
    os.environ['LOOKING_GLASS_VISION_GIT_EXE'] = str(git)
    os.environ['LOOKING_GLASS_VISION_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_VISION_BOOTSTRAP_SITE'] = str(site)
    runpy.run_module('looking_glass.vision_coco_cls_exposed_gpu', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
