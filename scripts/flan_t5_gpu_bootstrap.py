"""Isolated source-only bootstrap for the supervised FLAN-T5 GPU smoke owner."""
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import runpy
import stat
import sys

_MODULES = frozenset(('flan_t5_device_smoke', 'flan_t5_device_owner',
    'flan_t5_device_observation', 'flan_t5_feasibility', 'flan_t5_equal_dose',
    'intervention_plan', 'interventions', 'model_ownership', 'device_identity',
    'device_lease', 'memory_budget', 'supervisor', 'gpu_oneshot',
    'bert_gpu_calibration', 'bert_gpu_evidence', 'bert_gpu_family_owner'))


def _checked_path(path):
    lexical = Path(os.path.abspath(path))
    for item in (lexical, *lexical.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise RuntimeError('symlinks and reparse points are unsupported')
    return lexical.resolve(strict=True)


class _RejectSourceless(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
            raise ImportError('sourceless dependency bytecode is unsupported')
        return None


class _ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = _checked_path(package)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('looking_glass.'):
            return None
        name = fullname.removeprefix('looking_glass.')
        if name not in _MODULES:
            raise ImportError('project module is outside the FLAN-T5 source allowlist')
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
        raise RuntimeError('the bytecode prefix must be an empty plain directory')
    if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
        raise RuntimeError('project modules were imported before the bootstrap')
    if len(sys.argv) < 4 or sys.argv[1] != '--bootstrap-site' or sys.argv[3] != '--owner':
        raise ValueError('an explicit dependency directory and owner role are required')
    site_argument = Path(sys.argv[2])
    if not site_argument.is_absolute():
        raise ValueError('dependency directory must be absolute')
    site = _checked_path(site_argument)
    checkout = _checked_path(__file__).parent.parent
    if (not site.is_dir() or site == checkout or site.is_relative_to(checkout)
            or checkout.is_relative_to(site)):
        raise ValueError('dependency directory must be separate from the checkout')
    for entry in sys.path:
        location = Path(entry or os.getcwd()).resolve()
        if location == checkout or location.is_relative_to(checkout):
            raise RuntimeError('checkout paths must not appear on the import search path')
    del sys.argv[1:4]
    sys.path.append(str(site))
    sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), _RejectSourceless())
    package = _checked_path(checkout / 'looking_glass')
    source = _checked_path(package / '__init__.py')
    if source.parent != package or not source.is_file():
        raise ImportError('project package is unavailable')
    spec = importlib.util.spec_from_file_location('looking_glass', source,
                                                  submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.meta_path.insert(0, _ProjectFinder(package))
    sys.modules['looking_glass'] = module
    spec.loader.exec_module(module)
    os.environ['LOOKING_GLASS_FLAN_T5_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_FLAN_T5_BOOTSTRAP_SITE'] = str(site)
    runpy.run_module('looking_glass.flan_t5_device_smoke', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
