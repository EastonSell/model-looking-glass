"""Source-only isolated bootstrap for the versioned CPU batch experiment."""
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import runpy
import sys


if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode
        or not sys.pycache_prefix or not os.path.isabs(sys.pycache_prefix)
        or sys._xoptions.get('pycache_prefix') != sys.pycache_prefix
        or os.path.lexists(sys.pycache_prefix)):
    raise SystemExit('use -I -S -B and a nonexistent -X pycache_prefix path')

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'looking_glass'
DEPENDENCIES = Path(sys.executable).absolute().parent.parent / 'Lib' / 'site-packages'
if not DEPENDENCIES.is_dir():
    raise SystemExit('the selected virtual environment has no dependency directory')


class SourceOnlyFinder(importlib.abc.MetaPathFinder):
    """Bind project imports to named sources and reject sourceless dependencies."""
    project_modules = {
        'looking_glass': PACKAGE / '__init__.py',
        'looking_glass.distilgpt2_batch_cpu_v1': PACKAGE / 'distilgpt2_batch_cpu_v1.py',
        'looking_glass.batch_validation': PACKAGE / 'batch_validation.py',
        'looking_glass.microbatch': PACKAGE / 'microbatch.py',
        'looking_glass.pretrained': PACKAGE / 'pretrained.py',
        'looking_glass.resident_worker': PACKAGE / 'resident_worker.py',
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'looking_glass' or fullname.startswith('looking_glass.'):
            source = self.project_modules.get(fullname)
            if source is None or not source.is_file():
                raise ImportError(f'project module is not in the source allowlist: {fullname}')
            if fullname == 'looking_glass':
                return importlib.util.spec_from_file_location(fullname, source,
                    submodule_search_locations=[str(PACKAGE)])
            return importlib.util.spec_from_file_location(fullname, source)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and (isinstance(spec.loader, importlib.machinery.SourcelessFileLoader)
                                 or (isinstance(spec.origin, str)
                                     and spec.origin.lower().endswith(('.pyc', '.pyo')))):
            raise ImportError(f'sourceless dependency is not permitted: {fullname}')
        return spec


path_finder = sys.meta_path.index(importlib.machinery.PathFinder)
sys.meta_path.insert(path_finder, SourceOnlyFinder())
sys.path.append(str(DEPENDENCIES))
runpy.run_module('looking_glass.distilgpt2_batch_cpu_v1', run_name='__main__')
