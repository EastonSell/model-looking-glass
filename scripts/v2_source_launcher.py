"""Development-only launcher for the isolated source import diagnostic.

There is no experiment or study entry point. The checks are operational
controls against ordinary concurrent changes, not same-account attestation.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import types


class LaunchError(RuntimeError):
    def __init__(self, message, child_reaped=None, *, status="failed", quarantine=None):
        super().__init__(message)
        self.child_reaped = child_reaped
        self.status = status
        self.quarantine = quarantine


_SCRIPTS = ("scripts/v2_source_launcher.py", "scripts/v2_source_bootstrap.py",
            "scripts/v2_source_guards.py")
_TOKEN = b"model-looking-glass-development-start-v1\n"


def _check_single_threaded():
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise LaunchError("launcher requires a dedicated Python process")


def _checked_path(value):
    path = Path(os.path.abspath(value))
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise LaunchError("reparse points and symlinks are unsupported")
    return path.resolve(strict=True)


def _check_startup(checkout, venv, dependencies, work):
    _check_single_threaded()
    if os.name != "nt" or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise LaunchError("Windows isolated, no-site, no-bytecode startup is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise LaunchError("CPU device masking is required")
    if any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules):
        raise LaunchError("project modules were already imported")
    if _checked_path(__file__) != checkout / _SCRIPTS[0]:
        raise LaunchError("the launcher is not in the target checkout")
    for entry in sys.path:
        path = Path(entry or os.getcwd()).resolve()
        if path == checkout or path.is_relative_to(checkout):
            raise LaunchError("checkout paths must not be on the import path")
    if work == checkout or work.is_relative_to(checkout) or checkout.is_relative_to(work):
        raise LaunchError("the temporary work directory must be separate from checkout")
    if not work.is_dir() or not venv.is_dir():
        raise LaunchError("work and virtual environment must be directories")
    # The installed interpreter's version alias can be a Windows junction.
    # Resolve it to the real executable, then check the canonical path. Source
    # and cache paths remain subject to the stricter no-reparse policy.
    base = _checked_path(Path(sys.base_prefix).resolve(strict=True))
    executable = _checked_path((base / "python.exe").resolve(strict=True))
    if not os.path.samefile(_checked_path(Path(sys.executable).resolve(strict=True)), executable):
        raise LaunchError("launch directly with the base interpreter, not a venv shim")
    config = _checked_path(venv / "pyvenv.cfg")
    data = config.read_bytes()
    if len(data) > 16384:
        raise LaunchError("virtual environment configuration is too large")
    entries = {}
    for line in data.decode("utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in entries:
            raise LaunchError("duplicate virtual environment configuration key")
        entries[key] = value
    home = entries.get("home")
    if (not home or not Path(home).is_absolute() or
            entries.get("implementation") != "CPython" or
            entries.get("version_info") != ".".join(map(str, sys.version_info[:3])) or
            entries.get("include-system-site-packages", "").lower() != "false" or
            _checked_path(Path(home).resolve(strict=True)) != base):
        raise LaunchError("virtual environment does not identify this base interpreter")
    if not dependencies or len(set(dependencies)) != len(dependencies):
        raise LaunchError("explicit unique dependency directories are required")
    if any(not path.is_dir() or not path.is_relative_to(venv) or path.is_relative_to(checkout)
           for path in dependencies):
        raise LaunchError("dependency directories must lie within the virtual environment")
    return executable


def _git(checkout, *args):
    executable = shutil.which("git")
    if not executable:
        raise LaunchError("git is unavailable")
    result = subprocess.run([executable, "-C", str(checkout), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
    if result.returncode or len(result.stdout) > 8 * 1024 * 1024:
        raise LaunchError("cannot inspect the committed checkout")
    return result.stdout


def _revision(checkout):
    value = _git(checkout, "rev-parse", "--verify", "HEAD").strip().decode("ascii")
    if not re.fullmatch("[0-9a-f]{40}", value) or _git(checkout, "cat-file", "-t", value).strip() != b"commit":
        raise LaunchError("unsupported source revision")
    return value


def _committed_sources(checkout, revision):
    raw = _git(checkout, "ls-tree", "-rz", "--full-tree", revision)
    found = {}
    casefolded = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, encoded_path = record.split(b"\t", 1)
        mode, kind, sha = header.decode("ascii").split(" ")
        relative = encoded_path.decode("utf-8")
        selected = relative in _SCRIPTS or (relative.startswith("looking_glass/") and relative.endswith(".py"))
        if not selected:
            continue
        path = PurePosixPath(relative)
        if (path.is_absolute() or path.as_posix() != relative or
                any(part in ("", ".", "..") for part in path.parts) or
                mode not in ("100644", "100755") or kind != "blob" or
                not re.fullmatch("[0-9a-f]{40}", sha) or relative.casefold() in casefolded):
            raise LaunchError("unsupported committed source entry")
        casefolded.add(relative.casefold())
        found[relative] = sha
        if len(found) > 4096:
            raise LaunchError("too many committed source files")
    if any(path not in found for path in _SCRIPTS) or "looking_glass/__init__.py" not in found:
        raise LaunchError("required committed source is missing")
    return dict(sorted(found.items()))


def _verify_sources(checkout, sources):
    hashes = {}
    total = 0
    for relative, committed_blob in sources.items():
        path = _checked_path(checkout.joinpath(*PurePosixPath(relative).parts))
        if not path.is_relative_to(checkout) or not path.is_file():
            raise LaunchError("source resolves outside checkout")
        with path.open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        total += len(data)
        if len(data) > 8 * 1024 * 1024 or total > 64 * 1024 * 1024:
            raise LaunchError("committed source inventory exceeds bounds")
        git_hash = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        if git_hash != committed_blob:
            raise LaunchError("source bytes do not match the committed blob")
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return hashes


def _load_guards(checkout, sources):
    # The module is loaded only from bytes checked against this revision.
    path = checkout / _SCRIPTS[2]
    with path.open("rb") as stream:
        data = stream.read(8 * 1024 * 1024 + 1)
    git_hash = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    if len(data) > 8 * 1024 * 1024 or git_hash != sources[_SCRIPTS[2]]:
        raise LaunchError("source guard does not match the committed blob")
    module = types.ModuleType("_v2_development_guards")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _stop_and_reap(child, job):
    if child is not None and child.stdin is not None:
        try:
            child.stdin.close()
        except OSError:
            pass
        child.stdin = None
    # The child may have created descendants; the entire assigned job must
    # drain before source/cache protections are released.
    job_error = None
    if job is not None:
        try:
            job.terminate_and_wait()
        except BaseException as error:
            job_error = error
    if child is not None and child.poll() is None:
        try:
            child.kill()  # Also covers a child for which assignment failed.
        except OSError:
            pass
    if child is not None:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise LaunchError("child could not be reaped; protections retained", False) from error
    if job_error is not None:
        if isinstance(job_error, KeyboardInterrupt):
            job_error.child_reaped = child is not None
            raise job_error
        raise LaunchError("process job could not be drained", child is not None) from job_error
    if job is not None:
        job.close()
    return True if child is not None else None


class LaunchQuarantine:
    """Retain unfinished protection resources so an embedding caller can retry."""
    def __init__(self, temporary, manifest):
        self.temporary, self.manifest = temporary, manifest
        self.stack = ExitStack()
        self.source_lease = self.manifest_lease = self.cache = None
        self.job = self.child = None
        self.child_reaped = None
        self.complete = False

    def retry_cleanup(self):
        if self.complete:
            return
        if self.job is not None or self.child is not None:
            try:
                self.child_reaped = _stop_and_reap(self.child, self.job)
            except BaseException as error:
                if getattr(error, "child_reaped", None) is not None:
                    self.child_reaped = error.child_reaped
                raise
            self.job = self.child = None
        if self.cache is not None and self.cache._used:
            self.cache.close()
        if self.manifest_lease is not None and self.manifest_lease._kernel is not None:
            self.manifest_lease.close()
        if self.source_lease is not None and self.source_lease._kernel is not None:
            self.source_lease.close()
        self.stack.close()
        self.manifest.unlink(missing_ok=True)
        if self.temporary.exists():
            self.temporary.rmdir()
        self.complete = True


def run(checkout, venv, dependencies, work, module, timeout):
    executable = _check_startup(checkout, venv, dependencies, work)
    if module != "looking_glass.trace_schema" or not 1 <= timeout <= 30:
        raise LaunchError("invalid development diagnostic target or timeout")
    revision = _revision(checkout)
    sources = _committed_sources(checkout, revision)
    before = _verify_sources(checkout, sources)
    guards = _load_guards(checkout, sources)
    temporary = Path(tempfile.mkdtemp(prefix="v2-launch-", dir=work))
    manifest = temporary / "sources.json"
    manifest_bytes = json.dumps({key: before[key] for key in before if key.startswith("looking_glass/")},
                                sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    quarantine = LaunchQuarantine(temporary, manifest)
    try:
        manifest.write_bytes(manifest_bytes)
        paths = [checkout.joinpath(*PurePosixPath(name).parts) for name in sources]
        quarantine.source_lease = guards.SourceFileLease(checkout, paths)
        quarantine.stack.enter_context(quarantine.source_lease)
        quarantine.manifest_lease = guards.SourceFileLease(temporary, [manifest])
        quarantine.stack.enter_context(quarantine.manifest_lease)
        quarantine.cache = guards.PrivateEmptyCache(temporary)
        prefix = quarantine.stack.enter_context(quarantine.cache)
        if (_revision(checkout) != revision or _verify_sources(checkout, sources) != before or
                manifest.read_bytes() != manifest_bytes or any(prefix.iterdir())):
            raise LaunchError("guarded source or cache validation failed")
        quarantine.job = guards.WindowsJob()
        handles = (quarantine.source_lease.handles() + quarantine.manifest_lease.handles() +
                   quarantine.cache.handles())
        _check_single_threaded()
        arguments = [str(executable), "-I", "-S", "-B", "-X", f"pycache_prefix={prefix}",
                     str(checkout / _SCRIPTS[1]), "--development-probe", "--checkout", str(checkout),
                     "--sources", str(manifest), "--venv-root", str(venv), "--prefix", str(prefix),
                     "--module", module, "--require-start"]
        for dependency in dependencies:
            arguments.extend(("--dependency", str(dependency)))
        if len(subprocess.list2cmdline(arguments)) > 30000:
            raise LaunchError("guarded child command exceeds the Windows bound")
        quarantine.child = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, close_fds=True,
                                            creationflags=subprocess.DETACHED_PROCESS, cwd=work)
        quarantine.job.assign(quarantine.child)
        remote_handles = guards.duplicate_to_process(handles, quarantine.child)
        frame = _TOKEN + json.dumps({"guard_handles": remote_handles}, separators=(",", ":")).encode("ascii")
        if len(frame) > 262144:
            raise LaunchError("guarded start frame exceeds its bound")
        quarantine.child.stdin.write(frame)
        quarantine.child.stdin.close()  # EOF completes the child's bounded start read.
        quarantine.child.stdin = None
        code = quarantine.child.wait(timeout=timeout)
        if code:
            raise LaunchError("isolated child diagnostic failed")
        if _revision(checkout) != revision or _verify_sources(checkout, sources) != before:
            raise LaunchError("source verification failed after the diagnostic")
        quarantine.retry_cleanup()
        return {"status": "development_only", "study_ready": False,
                "source_revision": revision, "source_sha256": before, "child_reaped": True}
    except BaseException as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        try:
            quarantine.retry_cleanup()
        except BaseException as cleanup_error:
            if isinstance(cleanup_error, KeyboardInterrupt):
                status = "cancelled"
            raise LaunchError("development cleanup requires retry", quarantine.child_reaped,
                              status=status, quarantine=quarantine) from cleanup_error
        raise LaunchError("development launch failed", quarantine.child_reaped, status=status) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-probe", action="store_true", required=True)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--venv-root", required=True)
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--module", default="looking_glass.trace_schema")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    result = run(_checked_path(args.checkout), _checked_path(args.venv_root),
                 tuple(_checked_path(item) for item in args.dependency),
                 _checked_path(args.work_dir), args.module, args.timeout)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else getattr(error, "status", "failed")
        quarantine = getattr(error, "quarantine", None)
        protection_state = "released"
        if quarantine is not None:
            try:
                quarantine.retry_cleanup()
            except BaseException as retry_error:
                if isinstance(retry_error, KeyboardInterrupt):
                    status = "cancelled"
                protection_state = "unknown_on_process_exit"
        print(json.dumps({"status": status, "study_ready": False, "protection_state": protection_state,
                          "error_type": type(error).__name__,
                          "child_reaped": (quarantine.child_reaped if quarantine is not None
                                            else getattr(error, "child_reaped", None))}), file=sys.stderr)
        raise SystemExit(130 if status == "cancelled" else 1)
