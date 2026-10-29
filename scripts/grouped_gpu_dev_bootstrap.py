"""Isolated source-only entrypoint for the grouped GPU development owner."""

import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import stat
import sys


_MODULES = frozenset((
    'grouped_gpu_dev', 'bert_gpu_evidence', 'gpu_oneshot', 'device_identity',
    'device_lease', 'supervisor', 'cuda_cleanup', 'pretrained', 'resident_worker',
    'memory_budget', 'budgeted_worker', 'host_budget', 'fair_queue', 'microbatch',
    'grouped_worker', 'grouped_broker_state', 'grouped_broker', 'batch_validation',
    'certified_batch_policy', 'grouped_observation', 'grouped_owner_worker', 'owner_completion',
    'agreement_calibration_workspace',
))


def _plain(path):
    lexical = Path(os.path.abspath(path))
    for part in (lexical, *lexical.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise RuntimeError('symlink or reparse path is unsupported')
    return lexical.resolve(strict=True)


class _RejectSourceless(importlib.abc.MetaPathFinder):
    _grouped_sourceless_guard = True

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
            raise ImportError('sourceless dependency bytecode is unsupported')
        return None


class _ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = _plain(package)
        self._grouped_project = str(self.package)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('looking_glass.'):
            return None
        name = fullname.removeprefix('looking_glass.')
        if name not in _MODULES:
            raise ImportError('project module is outside the grouped pilot allowlist')
        source = _plain(self.package / (name + '.py'))
        if source.parent != self.package or not source.is_file():
            raise ImportError('project source is unavailable')
        return importlib.util.spec_from_file_location(fullname, source)


def main():
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.flags.ignore_environment
            or not sys.dont_write_bytecode or not sys.pycache_prefix
            or sys._xoptions.get('pycache_prefix') != sys.pycache_prefix):
        raise RuntimeError('use -I -S -B and an absolute fresh -X pycache_prefix')
    prefix = Path(sys.pycache_prefix)
    if not prefix.is_absolute() or not _plain(prefix).is_dir() or any(prefix.iterdir()):
        raise RuntimeError('the bytecode prefix must be an empty plain directory')
    checkout = _plain(Path(__file__).parent.parent)
    for entry in sys.path:
        location = Path(entry or os.getcwd()).resolve()
        if location == checkout or location.is_relative_to(checkout):
            raise RuntimeError('checkout paths must not appear in the import search path')
    if (any(name == root or name.startswith(root + '.') for name in sys.modules
            for root in ('torch', 'transformers', 'huggingface_hub', 'looking_glass'))):
        raise RuntimeError('project or model dependencies were imported too early')
    if len(sys.argv) < 3 or sys.argv[1] != '--site-packages':
        raise ValueError('an explicit dependency directory is required')
    site = Path(sys.argv[2])
    if not site.is_absolute() or not _plain(site).is_dir():
        raise ValueError('the dependency directory must be an absolute plain directory')
    site = _plain(site)
    if site == checkout or site.is_relative_to(checkout) or checkout.is_relative_to(site):
        raise ValueError('the dependency directory must be separate from the checkout')
    del sys.argv[1:3]
    sys.path.append(str(site))
    guard = _RejectSourceless()
    sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), guard)
    if sys.argv[1:] == ['--check-only']:
        spec = importlib.util.find_spec('torch')
        print(json.dumps({'status': 'ok', 'torch_origin': None if spec is None else spec.origin}))
        return 0
    package = _plain(checkout / 'looking_glass')
    source = _plain(package / '__init__.py')
    spec = importlib.util.spec_from_file_location('looking_glass', source,
                                                submodule_search_locations=[str(package)])
    if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
        raise ImportError('project package must load from source')
    module = importlib.util.module_from_spec(spec)
    sys.meta_path.insert(0, _ProjectFinder(package))
    sys.modules['looking_glass'] = module
    spec.loader.exec_module(module)
    os.environ['LOOKING_GLASS_GROUPED_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_GROUPED_SITE'] = str(site)
    runpy.run_module('looking_glass.grouped_gpu_dev', run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
