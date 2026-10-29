"""Isolated, CPU-masked source bootstrap for the BERT agreement study."""

import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import stat
import sys


_MODULES = frozenset(('bert_agreement_study', 'bert_agreement_cases', 'bert_agreement_exposures_v2', 'bert_agreement_runner',
                      'bert_agreement_audit', 'pretrained', 'intervention_plan',
                      'interventions', 'model_ownership', 'cpu_study_supervision',
                      'supervisor', 'device_lease', 'device_identity'))


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


def _install_bytecode_guard():
    index = sys.meta_path.index(importlib.machinery.PathFinder)
    sys.meta_path.insert(index, _RejectSourceless())


class _ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = _checked_path(package)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('looking_glass.'):
            return None
        name = fullname.removeprefix('looking_glass.')
        if name not in _MODULES:
            raise ImportError('project module is outside the BERT source allowlist')
        source = _checked_path(self.package / (name + '.py'))
        if source.parent != self.package or not source.is_file():
            raise ImportError('project source is unavailable')
        return importlib.util.spec_from_file_location(fullname, source)


def _check_startup(checkout):
    prefix = sys.pycache_prefix
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.flags.ignore_environment
            or not sys.dont_write_bytecode or type(prefix) is not str
            or sys._xoptions.get('pycache_prefix') != prefix or not Path(prefix).is_absolute()):
        raise RuntimeError('use -I -S -B and an absolute fresh -X pycache_prefix')
    directory = _checked_path(prefix)
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or directory.is_symlink()
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
            or any(directory.iterdir())):
        raise RuntimeError('the bytecode prefix must be an empty plain directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('-1', ''):
        raise RuntimeError('the CPU study requires CUDA visibility to be masked')
    if ('torch' in sys.modules or 'transformers' in sys.modules
            or any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules)):
        raise RuntimeError('project or model dependencies were imported too early')
    for entry in sys.path:
        location = Path(entry or os.getcwd()).resolve()
        if location == checkout or location.is_relative_to(checkout):
            raise RuntimeError('checkout paths must not appear on the import search path')


def _load_package(package):
    package = _checked_path(package)
    source = _checked_path(package / '__init__.py')
    if source.parent != package or not source.is_file():
        raise ImportError('project package source is unavailable')
    spec = importlib.util.spec_from_file_location('looking_glass', source,
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    finder = _ProjectFinder(package)
    sys.meta_path.insert(0, finder)
    sys.modules['looking_glass'] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules['looking_glass']
        sys.meta_path.remove(finder)
        raise


def main():
    checkout = _checked_path(__file__).parent.parent
    _check_startup(checkout)
    if len(sys.argv) < 3 or sys.argv[1] != '--bootstrap-site':
        raise ValueError('an absolute dependency directory is required')
    site = Path(sys.argv[2])
    if not site.is_absolute():
        raise ValueError('the dependency directory must be an absolute plain directory')
    site = _checked_path(site)
    if not site.is_dir():
        raise ValueError('the dependency directory must be an absolute plain directory')
    del sys.argv[1:3]
    if sys.argv[1:] == ['--check-only']:
        print(json.dumps({'status': 'ok', 'torch_imported': 'torch' in sys.modules}, sort_keys=True))
        return
    sys.path.append(str(site))  # Standard-library paths retain precedence.
    _install_bytecode_guard()
    _load_package(checkout / 'looking_glass')
    os.environ['LOOKING_GLASS_BERT_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_BERT_ROOT'] = str(checkout)
    os.environ['LOOKING_GLASS_BERT_SITE'] = str(site)
    runpy.run_module('looking_glass.bert_agreement_study', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
