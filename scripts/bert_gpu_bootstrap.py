"""Load the calibration package from this checkout with isolated bytecode."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import stat
import sys

_MODULES = frozenset(("bert_gpu_calibration", "bert_gpu_evidence", "gpu_oneshot", "device_identity", "device_lease",
                      "supervisor", "memory_budget", "intervention_plan", "interventions",
                      "model_ownership"))


class _ProjectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("looking_glass."):
            return None
        name = fullname.removeprefix("looking_glass.")
        if name not in _MODULES:
            raise ImportError("project module is outside the calibration source allowlist")
        source = self.package / (name + ".py")
        if not source.is_file() or source.is_symlink():
            raise ImportError("project source is unavailable")
        return importlib.util.spec_from_file_location(fullname, source)


def main():
    prefix = sys.pycache_prefix
    if (not isinstance(prefix, str) or sys._xoptions.get("pycache_prefix") != prefix
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode):
        raise RuntimeError("start with -I -S -B and an absolute empty -X pycache_prefix")
    directory = Path(prefix)
    if not directory.is_absolute():
        raise RuntimeError("the bytecode prefix must be absolute")
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or directory.is_symlink()
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or next(directory.iterdir(), None) is not None):
        raise RuntimeError("the bytecode prefix must be a plain empty directory")
    if any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules):
        raise RuntimeError("project modules were imported before the bootstrap")
    if len(sys.argv) < 3 or sys.argv[1] != "--bootstrap-site":
        raise ValueError("an explicit dependency directory is required")
    site = Path(sys.argv[2])
    if not site.is_absolute() or not site.is_dir():
        raise ValueError("the dependency directory must exist and be absolute")
    del sys.argv[1:3]
    sys.path.append(str(site))
    package = Path(__file__).resolve().parent.parent / "looking_glass"
    source = package / "__init__.py"
    if not source.is_file() or source.is_symlink():
        raise ImportError("the project package is unavailable")
    spec = importlib.util.spec_from_file_location("looking_glass", source,
                                                  submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.meta_path.insert(0, _ProjectFinder(package))
    sys.modules["looking_glass"] = module
    spec.loader.exec_module(module)
    os.environ["LOOKING_GLASS_BERT_BOOTSTRAP_PID"] = str(os.getpid())
    os.environ["LOOKING_GLASS_BERT_BOOTSTRAP_SITE"] = str(site.resolve(strict=True))
    runpy.run_module("looking_glass.bert_gpu_calibration", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
