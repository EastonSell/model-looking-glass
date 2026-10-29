"""Development-only isolated source-loader probe; this cannot run the study.

The future external launcher must independently bind this script and its source
manifest to committed bytes and hold source/cache protections for the process
lifetime. This diagnostic deliberately does not claim those protections.
"""
import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


class BootstrapError(RuntimeError):
    """An isolated source-loading check failed."""


_START_TOKEN = b"model-looking-glass-development-start-v1\n"


def _await_guarded_start():
    """Check inherited handles and wait for the launcher before project imports.

    This is a cooperative start gate, not authentication of the launcher or
    proof of file identities. The owner must assign the actual interpreter to
    its job and validate its source protections before sending this token.
    """
    data = sys.stdin.buffer.read(262145)
    if len(data) > 262144 or not data.startswith(_START_TOKEN):
        raise BootstrapError("launcher start channel was closed or invalid")
    frame = json.loads(data[len(_START_TOKEN):], object_pairs_hook=_unique_object)
    if type(frame) is not dict or set(frame) != {"guard_handles"}:
        raise BootstrapError("invalid launcher start frame")
    handles = frame["guard_handles"]
    if (os.name != "nt" or type(handles) is not list or not 0 < len(handles) <= 8193 or
            any(type(handle) is not int or handle <= 0 for handle in handles) or
            len(set(handles)) != len(handles)):
        raise BootstrapError("invalid inherited protection handles")
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetHandleInformation.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetHandleInformation.restype = wintypes.BOOL
    for handle in handles:
        flags = wintypes.DWORD()
        if not kernel.GetHandleInformation(handle, ctypes.byref(flags)):
            raise BootstrapError("an inherited protection handle is unavailable")
        # Keep the protection in this process, but do not leak it to children.
        os.set_handle_inheritable(handle, False)
        if os.get_handle_inheritable(handle):
            raise BootstrapError("an inherited protection handle remains inheritable")


def _within(path, root):
    return path == root or path.is_relative_to(root)


def _checked_path(path):
    """Reject symlinks and Windows reparse points before path resolution."""
    lexical = Path(os.path.abspath(path))
    for entry in (lexical, *lexical.parents):
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise BootstrapError("reparse points and symlinks are unsupported")
    return lexical.resolve(strict=True)


def _check_startup(checkout, prefix):
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise BootstrapError("isolated, no-site and no-bytecode flags are required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise BootstrapError("the CPU-only diagnostic requires CUDA device masking")
    if any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules):
        raise BootstrapError("project modules were loaded before the bootstrap")
    if prefix.is_symlink() or not prefix.is_dir() or any(prefix.iterdir()):
        raise BootstrapError("a fresh empty cache prefix is required")
    if sys.pycache_prefix is None or _checked_path(sys.pycache_prefix) != prefix:
        raise BootstrapError("cache prefix does not match the launched process")
    _check_search_path(checkout)


def _check_search_path(checkout):
    for entry in sys.path:
        if _within(Path(entry or os.getcwd()).resolve(), checkout):
            raise BootstrapError("checkout paths must not appear on sys.path")


def _source_map(checkout, manifest):
    if type(manifest) is not dict or not 0 < len(manifest) <= 4096:
        raise BootstrapError("source manifest must be a bounded nonempty object")
    by_name = {}
    total_bytes = 0
    for relative, digest in manifest.items():
        if type(relative) is not str or type(digest) is not str or len(digest) != 64:
            raise BootstrapError("invalid source manifest entry")
        if any(character not in "0123456789abcdef" for character in digest):
            raise BootstrapError("invalid source digest")
        path = PurePosixPath(relative)
        if (path.as_posix() != relative or path.is_absolute() or len(path.parts) < 2 or path.parts[0] != "looking_glass" or
                any(part in ("", ".", "..") for part in path.parts) or path.suffix != ".py"):
            raise BootstrapError("source path is not a project Python source")
        parts = list(path.parts)
        parts[-1] = path.stem
        if parts[-1] == "__init__":
            parts.pop()
            is_package = True
        else:
            is_package = False
        if not parts or any(not part.isidentifier() for part in parts):
            raise BootstrapError("source path is not a valid module name")
        name = ".".join(parts)
        if name in by_name:
            raise BootstrapError("ambiguous project module")
        resolved = _checked_path(checkout.joinpath(*path.parts))
        if not _within(resolved, checkout) or not resolved.is_file():
            raise BootstrapError("project source resolves outside checkout")
        with resolved.open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        total_bytes += len(data)
        if len(data) > 8 * 1024 * 1024 or total_bytes > 64 * 1024 * 1024:
            raise BootstrapError("project source manifest exceeds bounds")
        if hashlib.sha256(data).hexdigest() != digest:
            raise BootstrapError("project source differs from its supplied digest")
        by_name[name] = {"path": resolved, "relative": relative, "sha256": digest,
                         "is_package": is_package}
    if "looking_glass" not in by_name or not by_name["looking_glass"]["is_package"]:
        raise BootstrapError("project package initializer is required")
    return by_name


class _VerifiedLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, source, inventory):
        super().__init__(name, str(source["path"]))
        self._source = source
        self._inventory = inventory

    def get_code(self, fullname):
        # Compile exactly the checked source bytes, without consulting .pyc.
        if fullname != self.name:
            raise BootstrapError("source loader was invoked for another module")
        with self._source["path"].open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise BootstrapError("source exceeds the size limit")
        digest = hashlib.sha256(data).hexdigest()
        if digest != self._source["sha256"]:
            raise BootstrapError("source changed before import")
        code = self.source_to_code(data, str(self._source["path"]))
        self._inventory[self._source["relative"]] = digest
        return code


class _ProjectFinder:
    def __init__(self, sources):
        self.sources = sources
        self.inventory = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "looking_glass" and not fullname.startswith("looking_glass."):
            return None
        source = self.sources.get(fullname)
        if source is None:
            raise ModuleNotFoundError("project module is absent from the source allowlist")
        loader = _VerifiedLoader(fullname, source, self.inventory)
        return importlib.util.spec_from_file_location(
            fullname, source["path"], loader=loader,
            submodule_search_locations=[str(source["path"].parent)] if source["is_package"] else None,
        )

    def verify_inventory(self):
        if not sys.meta_path or sys.meta_path[0] is not self:
            raise BootstrapError("project source finder is no longer first")
        for source in self.sources.values():
            if _checked_path(source["path"]) != source["path"]:
                raise BootstrapError("project source path changed")
            with source["path"].open("rb") as stream:
                data = stream.read(8 * 1024 * 1024 + 1)
            if (len(data) > 8 * 1024 * 1024 or
                    hashlib.sha256(data).hexdigest() != source["sha256"]):
                raise BootstrapError("project source changed after import")
        for name, module in tuple(sys.modules.items()):
            if name != "looking_glass" and not name.startswith("looking_glass."):
                continue
            source = self.sources.get(name)
            spec = getattr(module, "__spec__", None)
            loader = getattr(spec, "loader", None)
            path = getattr(module, "__file__", None)
            if (source is None or not isinstance(loader, _VerifiedLoader) or
                    loader.name != name or not path or Path(path).resolve() != source["path"] or
                    self.inventory.get(source["relative"]) != source["sha256"]):
                raise BootstrapError("project module provenance is inconsistent")
        return dict(sorted(self.inventory.items()))


def _add_dependencies(checkout, venv, dependencies):
    if not dependencies:
        raise BootstrapError("explicit dependency paths are required")
    seen = set()
    for item in dependencies:
        path = _checked_path(item)
        if (not path.is_dir() or not _within(path, venv) or _within(path, checkout) or path in seen):
            raise BootstrapError("dependency path is not a unique directory within the venv")
        seen.add(path)
        sys.path.append(str(path))  # Keep the isolated interpreter's stdlib first.


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapError("duplicate source manifest key")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-probe", action="store_true", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--venv-root", required=True)
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--require-start", action="store_true")
    args = parser.parse_args()
    checkout = _checked_path(args.checkout)
    prefix_argument = Path(args.prefix)
    if not prefix_argument.is_absolute() or prefix_argument.is_symlink():
        raise BootstrapError("cache prefix must be an absolute non-symlink directory")
    prefix = _checked_path(prefix_argument)
    venv = _checked_path(args.venv_root)
    _check_startup(checkout, prefix)
    if args.require_start:
        _await_guarded_start()
    manifest_path = _checked_path(args.sources)
    with manifest_path.open("rb") as stream:
        manifest_bytes = stream.read(2 * 1024 * 1024 + 1)
    if len(manifest_bytes) > 2 * 1024 * 1024:
        raise BootstrapError("source manifest exceeds the size limit")
    manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_object)
    finder = _ProjectFinder(_source_map(checkout, manifest))
    _add_dependencies(checkout, venv, args.dependency)
    # Install before the package initializer can execute any relative import.
    sys.meta_path.insert(0, finder)
    if args.module != "looking_glass" and not args.module.startswith("looking_glass."):
        raise BootstrapError("only a project diagnostic import is permitted")
    importlib.import_module(args.module)
    inventory = finder.verify_inventory()
    _check_search_path(checkout)
    if any(prefix.iterdir()):
        raise BootstrapError("cache prefix changed during the diagnostic")
    print(json.dumps({"status": "development_only", "study_ready": False,
                      "project_sources": inventory}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not print private local paths or arbitrary source exception text.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
