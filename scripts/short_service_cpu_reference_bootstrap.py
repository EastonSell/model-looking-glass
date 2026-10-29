"""Isolated, source-only startup for the short-service CPU reference."""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


_MODULES = frozenset(('short_service_cpu_reference', 'short_service_pilot_core', 'pretrained',
                      'broker', 'owner_completion', 'resident_worker', 'budgeted_worker',
                      'memory_budget', 'grouped_observation', 'fair_queue'))


class _SourceOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
            raise ImportError('sourceless dependency bytecode is unsupported')
        return None


class _ProjectSource(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('looking_glass.'):
            return None
        name = fullname.removeprefix('looking_glass.')
        if name not in _MODULES:
            raise ImportError('module is outside the fixed CPU reference source set')
        source = (self.package / (name + '.py')).resolve(strict=True)
        if source.parent != self.package or not source.is_file():
            raise ImportError('project source is unavailable')
        return importlib.util.spec_from_file_location(fullname, source)


def main():
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
            or not sys.pycache_prefix or sys._xoptions.get('pycache_prefix') != sys.pycache_prefix):
        raise RuntimeError('use -I -S -B and a fresh absolute -X pycache_prefix')
    prefix = Path(sys.pycache_prefix)
    if not prefix.is_absolute() or not prefix.resolve(strict=True).is_dir() or any(prefix.iterdir()):
        raise RuntimeError('the bytecode prefix must be an empty directory')
    repo = Path(__file__).resolve(strict=True).parents[1]
    if any((p := Path(entry or os.getcwd()).resolve()) == repo or p.is_relative_to(repo) for entry in sys.path):
        raise RuntimeError('checkout paths must not appear in the initial import path')
    if any(name == root or name.startswith(root + '.') for name in sys.modules
           for root in ('torch', 'transformers', 'huggingface_hub', 'looking_glass')):
        raise RuntimeError('model or project dependencies were imported too early')
    if len(sys.argv) < 3 or sys.argv[1] != '--site-packages':
        raise ValueError('an explicit dependency directory is required')
    site = Path(sys.argv[2])
    if not site.is_absolute() or not site.resolve(strict=True).is_dir():
        raise ValueError('the dependency directory must be absolute')
    site = site.resolve(strict=True)
    if site == repo or site.is_relative_to(repo) or repo.is_relative_to(site):
        raise ValueError('the dependency directory must be separate from the checkout')
    del sys.argv[1:3]
    sys.path.append(str(site))
    sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), _SourceOnly())
    if sys.argv[1:] == ['--check-only']:
        spec = importlib.util.find_spec('torch')
        print(json.dumps({'status': 'ok', 'torch_available': spec is not None}))
        return 0
    package = (repo / 'looking_glass').resolve(strict=True)
    spec = importlib.util.spec_from_file_location('looking_glass', package / '__init__.py',
                                                 submodule_search_locations=[str(package)])
    if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
        raise ImportError('project package must load from source')
    module = importlib.util.module_from_spec(spec)
    sys.meta_path.insert(0, _ProjectSource(package))
    sys.modules['looking_glass'] = module
    spec.loader.exec_module(module)
    runpy.run_module('looking_glass.short_service_cpu_reference', run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
