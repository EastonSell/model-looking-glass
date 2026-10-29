"""Source-first entrypoint for the two-token tap latency controller.

Run this file directly with Python 3.11+ and -I -B -S plus a fresh,
nonexistent -X pycache_prefix.  No project import happens before those flags
and an explicit trusted virtualenv site-packages path have been checked.
"""

import sys

if (not sys.flags.isolated or not sys.flags.ignore_environment or not getattr(sys.flags, "safe_path", False)
        or not sys.flags.no_site or not sys.dont_write_bytecode or not sys.pycache_prefix):
    sys.stderr.write("Source-first benchmark startup requires -I -B -S and a fresh -X pycache_prefix.\n")
    raise SystemExit(2)

import os

if (not os.path.isabs(sys.pycache_prefix)
        or sys._xoptions.get("pycache_prefix") != sys.pycache_prefix
        or os.path.lexists(sys.pycache_prefix)):
    sys.stderr.write("Benchmark bytecode-cache prefix must not exist.\n")
    raise SystemExit(2)
if len(sys.argv) < 3 or sys.argv[1] != "--site-packages":
    sys.stderr.write("Provide --site-packages followed by the absolute trusted dependency directory.\n")
    raise SystemExit(2)

site = sys.argv[2]
if not os.path.isabs(site) or not os.path.isdir(site):
    sys.stderr.write("The site-packages directory must exist and be absolute.\n")
    raise SystemExit(2)
site = os.path.realpath(site)
project = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(site)

import importlib.machinery
import importlib.util


class _OriginGuard:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourcelessFileLoader):
            raise ImportError("sourceless Python imports are not permitted")
        if (spec is not None and fullname.startswith("looking_glass.")
                and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)):
            raise ImportError("native extensions cannot replace project source modules")
        return spec


sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), _OriginGuard)
package = os.path.join(project, "looking_glass")
spec = importlib.util.spec_from_file_location(
    "looking_glass", os.path.join(package, "__init__.py"), submodule_search_locations=[package])
if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
    raise ImportError("the project package must be loaded from its explicit source path")
module = importlib.util.module_from_spec(spec)
sys.modules["looking_glass"] = module
spec.loader.exec_module(module)
os.environ["LOOKING_GLASS_BENCHMARK_SITE_PACKAGES"] = site
sys.argv = [sys.argv[0], *sys.argv[3:]]

from looking_glass.gpu_tap_latency_owner import main

raise SystemExit(main())
