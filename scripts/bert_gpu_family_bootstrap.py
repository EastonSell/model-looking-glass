"""Load the family profile package from this checkout with isolated bytecode."""
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import runpy
import stat
import sys

_MODULES = frozenset(("bert_gpu_family_controller", "bert_gpu_family_owner",
    "bert_gpu_family_profile", "bert_gpu_study_adapter", "bert_gpu_calibration",
    "bert_gpu_evidence", "bert_agreement_runner", "bert_agreement_audit",
    "bert_agreement_study", "bert_agreement_cases", "pretrained", "gpu_oneshot",
    "device_identity", "device_lease", "supervisor", "memory_budget",
    "intervention_plan", "interventions", "model_ownership"))


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
        if not fullname.startswith("looking_glass."):
            return None
        name = fullname.removeprefix("looking_glass.")
        if name not in _MODULES:
            raise ImportError("project module is outside the family profile source allowlist")
        source = _checked_path(self.package / (name + ".py"))
        if source.parent != self.package or not source.is_file():
            raise ImportError("project source is unavailable")
        return importlib.util.spec_from_file_location(fullname, source)


def main():
    prefix = sys.pycache_prefix
    if (not isinstance(prefix, str) or sys._xoptions.get("pycache_prefix") != prefix
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode):
        raise RuntimeError("start with -I -S -B and an absolute empty -X pycache_prefix")
    if not Path(prefix).is_absolute():
        raise RuntimeError("the bytecode prefix must be absolute")
    directory = _checked_path(prefix)
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or directory.is_symlink()
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            or next(directory.iterdir(), None) is not None):
        raise RuntimeError("the bytecode prefix must be a plain empty directory")
    if any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules):
        raise RuntimeError("project modules were imported before the bootstrap")
    if len(sys.argv) < 3 or sys.argv[1] != "--bootstrap-site":
        raise ValueError("an explicit dependency directory is required")
    site_argument = Path(sys.argv[2])
    if not site_argument.is_absolute():
        raise ValueError("the dependency directory must exist and be absolute")
    site = _checked_path(site_argument)
    checkout = _checked_path(__file__).parent.parent
    if (not site.is_dir() or site == checkout or site.is_relative_to(checkout)
            or checkout.is_relative_to(site)):
        raise ValueError("the dependency directory must exist and be absolute")
    for entry in sys.path:
        location = Path(entry or os.getcwd()).resolve()
        if location == checkout or location.is_relative_to(checkout):
            raise RuntimeError('checkout paths must not appear on the import search path')
    del sys.argv[1:3]
    sys.path.append(str(site))
    sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), _RejectSourceless())
    package = _checked_path(checkout / "looking_glass")
    source = _checked_path(package / "__init__.py")
    if source.parent != package or not source.is_file():
        raise ImportError("the project package is unavailable")
    spec = importlib.util.spec_from_file_location("looking_glass", source,
                                                  submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.meta_path.insert(0, _ProjectFinder(package))
    sys.modules["looking_glass"] = module
    spec.loader.exec_module(module)
    os.environ["LOOKING_GLASS_BERT_BOOTSTRAP_PID"] = str(os.getpid())
    os.environ["LOOKING_GLASS_BERT_BOOTSTRAP_SITE"] = str(site.resolve(strict=True))
    runpy.run_module("looking_glass.bert_gpu_family_controller", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
