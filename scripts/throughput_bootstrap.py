"""Fresh-bytecode bootstrap for supervised throughput experiments.

Execute this file directly. Python reads the entry script from source, then
the script checks the isolated cache before importing the project package.
"""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import stat
import sys


_MODULES = frozenset(('throughput_benchmark', 'pretrained', 'resident_worker', 'broker',
                      'fair_queue', 'memory_budget', 'budgeted_worker', 'device_lease',
                      'device_identity', 'supervisor'))


class _ProjectFinder(importlib.abc.MetaPathFinder):
    """Load only explicitly named package modules from their source files."""
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('looking_glass.'):
            return None
        name = fullname.removeprefix('looking_glass.')
        if name not in _MODULES:
            raise ImportError(f'project module is outside the throughput source allowlist: {fullname}')
        source = self.package / (name + '.py')
        if not source.is_file() or source.is_symlink():
            raise ImportError(f'project module source is unavailable: {fullname}')
        return importlib.util.spec_from_file_location(fullname, source)


def _load_project_package(package):
    source = package / '__init__.py'
    if not source.is_file() or source.is_symlink():
        raise ImportError('project package source is unavailable')
    spec = importlib.util.spec_from_file_location(
        'looking_glass', source, submodule_search_locations=[str(package)])
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


def _check_startup():
    prefix = sys.pycache_prefix
    option = sys._xoptions.get('pycache_prefix')
    if (type(prefix) is not str or not prefix or option != prefix
            or sys.flags.dont_write_bytecode != 1 or sys.dont_write_bytecode is not True
            or sys.flags.no_site != 1 or sys.flags.ignore_environment != 1
            or sys.flags.isolated != 1):
        raise RuntimeError('start with -I -S -B and an absolute empty -X pycache_prefix')
    path = Path(prefix)
    if not path.is_absolute():
        raise RuntimeError('the bytecode prefix must be absolute')
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise RuntimeError('the bytecode prefix must be a plain directory')
    if next(path.iterdir(), None) is not None:
        raise RuntimeError('the bytecode prefix must be empty')
    if any(name == 'looking_glass' or name.startswith('looking_glass.') for name in sys.modules):
        raise RuntimeError('project modules were imported before bootstrap')


def main():
    _check_startup()
    checkout = Path(__file__).resolve().parent.parent
    module = 'looking_glass.throughput_benchmark'
    project = str(checkout)
    site_paths = []
    if len(sys.argv) >= 3 and sys.argv[1] == '--bootstrap-internal':
        spec = json.loads(sys.argv[2])
        if (type(spec) is not dict or set(spec) != {'module', 'project', 'site_paths'}
                or type(spec['module']) is not str or not spec['module']
                or type(spec['project']) is not str or not Path(spec['project']).is_absolute()
                or type(spec['site_paths']) is not list
                or any(type(path) is not str or not Path(path).is_absolute() for path in spec['site_paths'])):
            raise ValueError('invalid internal bootstrap specification')
        module, project, site_paths = spec['module'], spec['project'], spec['site_paths']
        del sys.argv[1:3]
    else:
        while len(sys.argv) >= 3 and sys.argv[1] == '--bootstrap-site':
            site_paths.append(sys.argv[2])
            del sys.argv[1:3]
    if (not site_paths or any(type(path) is not str or not Path(path).is_absolute()
                              or not Path(path).is_dir() for path in site_paths)):
        raise ValueError('provide at least one absolute dependency directory using --bootstrap-site')
    sys.path.extend(site_paths)
    os.environ['LOOKING_GLASS_THROUGHPUT_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_THROUGHPUT_SITE_PATHS'] = json.dumps(site_paths, separators=(',', ':'))
    package = Path(project) / 'looking_glass'
    if not package.is_dir():
        package = checkout / 'looking_glass'
    if package.is_dir():
        _load_project_package(package)
    if module == 'looking_glass.throughput_benchmark':
        if not package.is_dir():
            raise ImportError('project package is unavailable')
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    else:
        # This path exists only for bounded CPU bootstrap/ownership probes.
        if not module.isidentifier():
            raise ValueError('probe module must be a single identifier')
        probe = Path(project) / (module + '.py')
        if not probe.is_file() or probe.is_symlink():
            raise ValueError('probe source is unavailable')
        runpy.run_path(str(probe), run_name='__main__')


if __name__ == '__main__':
    main()
