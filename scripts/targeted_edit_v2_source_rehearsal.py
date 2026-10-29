"""Source-bound, supervised no-checkpoint rehearsal for targeted-edit v2.

This Windows-only diagnostic does not authorize model loading or a study
forward. Parent-held file protections do not survive parent death.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import threading
import types
import uuid


class RehearsalError(RuntimeError):
    pass


_PARENT = "scripts/targeted_edit_v2_source_rehearsal.py"
_CHILD = "scripts/targeted_edit_v2_source_child.py"
_REFERENCE = "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"
_REFERENCE_SHA = "cb666aa911db020b8904dab6f08f989e506d85e875d7cac65e72d3b94cef2fa5"
_EXTRA = (_PARENT, _CHILD, _REFERENCE)
_ENVIRONMENT_KEYS = (
    "python_version", "platform", "machine", "processor", "torch_version", "numpy_version",
    "transformers_version", "torch_build_sha256", "torch_cpu_capability", "gpt2_source_sha256",
    "intraop_threads", "interop_threads", "deterministic_algorithms", "mkldnn_enabled",
    "omp_num_threads", "mkl_num_threads", "cuda_visibility", "cuda_initialized",
    "bytecode_attested", "denormal_probe",
)
_LOCK = threading.Lock()
_PENDING = {}
_MAX_PENDING = 8
_PARENT_USED = False


def _checked_module(path, expected):
    with path.open("rb") as stream:
        data = stream.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected:
        raise RehearsalError("checked helper differs from its committed source")
    module = types.ModuleType("_targeted_edit_source_helper")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _launcher():
    # The later commit/hash check does not attest this already-executing parent.
    path = Path(__file__).resolve().with_name("v2_source_launcher.py")
    with path.open("rb") as stream:
        data = stream.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise RehearsalError("source launcher exceeds its bound")
    module = types.ModuleType("_targeted_edit_source_launcher")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RehearsalError("duplicate JSON key")
        result[key] = value
    return result


def _bounded_json(path, limit):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or getattr(info, "st_nlink", 1) != 1):
        raise RehearsalError("a single regular JSON file is required")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
            raise RehearsalError("the regular JSON file changed during opening")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise RehearsalError("bounded JSON input is too large")
    return json.loads(raw, object_pairs_hook=_unique_object)


def _json_equal(left, right):
    try:
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return canonical(left) == canonical(right)
    except (TypeError, ValueError):
        return False


def _attempt_directory_exists(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RehearsalError("attempt directory could not be inspected") from error
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise RehearsalError("attempt directory is not an ordinary directory")
    return True


def _check_child_report(child, result, revision, hashes, reference):
    fields = {"status", "source_revision", "owner_pid", "owner_run_id", "claim_mask", "model_import_mask",
              "runtime", "denormal_probe_before", "denormal_probe_after", "project_sources",
              "checkpoint_loaded", "model_forwarded"}
    required_sources = {"looking_glass/" + name + ".py" for name in
                        ("__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
                         "targeted_edit_cpu_v1", "targeted_edit_v2_edits")}
    expected = {name: reference["environment"][name] for name in _ENVIRONMENT_KEYS}
    inventory = child.get("project_sources") if type(child) is dict else None
    if (type(child) is not dict or set(child) != fields or type(result.pid) is not int or result.pid <= 0
            or type(result.run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", result.run_id) is None
            or child["status"] != "matching_v1_runtime" or child["source_revision"] != revision
            or type(child["owner_pid"]) is not int or child["owner_pid"] != result.pid
            or child["owner_run_id"] != result.run_id or child["claim_mask"] != ""
            or child["model_import_mask"] != "-1" or not _json_equal(child["runtime"], expected)
            or not _json_equal(child["denormal_probe_before"], expected["denormal_probe"])
            or not _json_equal(child["denormal_probe_after"], expected["denormal_probe"])
            or child["checkpoint_loaded"] is not False or child["model_forwarded"] is not False
            or type(inventory) is not dict or not required_sources <= set(inventory)
            or any(type(name) is not str or type(digest) is not str or hashes.get(name) != digest
                   for name, digest in inventory.items())):
        raise RehearsalError("child report and supervisor identity differ")


class RehearsalQuarantine:
    """Retain exact protections while the supervisor outcome is unresolved."""

    def __init__(self, attempt_id):
        self.attempt_id = attempt_id
        # Published in the registry already owned by the admitting thread.
        self.running = True
        self.disposing = False
        self.complete = False
        self.temp = self.manifest = self.report = None
        self.temp_owned = False
        self.source_lease = self.manifest_lease = self.cache = self.job = None

    def release(self):
        """Release after a verified normal result, or before any launch attempt."""
        with _LOCK:
            if _PENDING.get(self.attempt_id) is not self or self.running or self.disposing:
                raise RehearsalError("quarantine is stale or still running")
            if not self.complete and getattr(self, "launch_started", False):
                raise RehearsalError("the supervisor outcome is uncertain; retain protections")
            if self.temp is not None and not self.temp_owned:
                raise RehearsalError("attempt directory ownership is uncertain; retain it")
            # An interrupted cleanup remains rooted and cannot be raced or
            # retried blindly; the hosting process must resolve the outcome.
            self.disposing = True
        if self.temp is not None:
            _attempt_directory_exists(self.temp)
        if self.job is not None:
            if self.complete:
                if getattr(self.job, "_handle", None) is not None or not getattr(self.job, "_confirmed", False):
                    raise RehearsalError("whole-job drain is not confirmed")
            else:
                if self.job.process_counts() != (0, 0):
                    raise RehearsalError("unused job is not empty")
                self.job.terminate_and_wait(timeout=3)
                self.job.close()
            self.job = None
        if self.cache is not None and getattr(self.cache, "_used", False):
            self.cache.close()
        self.cache = None
        for name in ("manifest_lease", "source_lease"):
            lease = getattr(self, name)
            if lease is not None and getattr(lease, "_kernel", None) is not None:
                lease.close()
            setattr(self, name, None)
        if self.report is not None:
            self.report.unlink(missing_ok=True)
        if self.manifest is not None:
            self.manifest.unlink(missing_ok=True)
        if self.temp is not None and _attempt_directory_exists(self.temp):
            self.temp.rmdir()
        with _LOCK:
            if _PENDING.get(self.attempt_id) is not self:
                raise RehearsalError("quarantine identity changed")
            del _PENDING[self.attempt_id]


def _validate_attempt(attempt_id):
    if type(attempt_id) is not str or re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise RehearsalError("a caller-known attempt identity is required")


def register_quarantine(attempt_id, *, _for_run=False):
    global _PARENT_USED
    _validate_attempt(attempt_id)
    with _LOCK:
        if _for_run and _PARENT_USED:
            raise RehearsalError("the isolated parent is single-use")
        if attempt_id in _PENDING or len(_PENDING) >= _MAX_PENDING:
            raise RehearsalError("quarantine identity or capacity is unavailable")
        quarantine = RehearsalQuarantine(attempt_id)
        if _for_run:
            _PARENT_USED = True
        _PENDING[attempt_id] = quarantine
    return quarantine


def get_quarantine(attempt_id):
    _validate_attempt(attempt_id)
    with _LOCK:
        quarantine = _PENDING.get(attempt_id)
    return quarantine


def _create_attempt_dir(quarantine, work):
    quarantine.temp = work / ("targeted-edit-source-" + quarantine.attempt_id)
    quarantine.temp.mkdir()
    quarantine.temp_owned = True


def _execute(q, checkout, venv, dependency, work, study_root, timeout):
    launcher = _launcher()
    checkout, venv = launcher._checked_path(checkout), launcher._checked_path(venv)
    dependency, work = launcher._checked_path(dependency), launcher._checked_path(work)
    study_root = launcher._checked_path(study_root)
    executable = launcher._check_startup(checkout, venv, (dependency,), work)
    if (launcher._checked_path(__file__) != checkout / _PARENT or not study_root.is_dir()
            or study_root == checkout or study_root.is_relative_to(checkout)
            or type(timeout) is not int or not 1 <= timeout <= 300):
        raise RehearsalError("invalid source rehearsal inputs")
    platform = launcher._parent_platform_observation()
    revision = launcher._revision(checkout)
    sources = launcher._committed_sources(checkout, revision, extra=_EXTRA)
    hashes = launcher._verify_sources(checkout, sources)
    if hashes.get(_REFERENCE) != _REFERENCE_SHA:
        raise RehearsalError("the historical precommit differs")
    guards = launcher._load_guards(checkout, sources)
    _create_attempt_dir(q, work)
    q.manifest = q.temp / "sources.json"
    q.report = q.temp / "child-report.json"
    manifest_bytes = json.dumps(hashes, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode("utf-8")
    with q.manifest.open("xb") as stream:
        if stream.write(manifest_bytes) != len(manifest_bytes):
            raise RehearsalError("short source inventory write")
        stream.flush()
        os.fsync(stream.fileno())
    q.source_lease = object.__new__(guards.SourceFileLease)
    q.source_lease.__init__(checkout, [checkout.joinpath(*PurePosixPath(name).parts) for name in sources])
    q.source_lease.__enter__()
    q.manifest_lease = object.__new__(guards.SourceFileLease)
    q.manifest_lease.__init__(q.temp, [q.manifest])
    q.manifest_lease.__enter__()
    q.cache = object.__new__(guards.PrivateEmptyCache)
    q.cache.__init__(q.temp)
    prefix = q.cache.__enter__()
    if (launcher._revision(checkout) != revision or launcher._verify_sources(checkout, sources) != hashes
            or q.manifest.read_bytes() != manifest_bytes or any(prefix.iterdir())):
        raise RehearsalError("protected source or cache check failed")
    bootstrap = _checked_module(checkout / "scripts/v2_source_bootstrap.py",
                                hashes["scripts/v2_source_bootstrap.py"])
    project = {name: digest for name, digest in hashes.items()
               if name.startswith("looking_glass/") and name.endswith(".py")}
    finder = bootstrap._ProjectFinder(bootstrap._source_map(checkout, project))
    sys.meta_path.insert(0, finder)
    from looking_glass.cpu_study_supervision import run_cpu_study
    finder.verify_inventory()
    q.job = object.__new__(guards.WindowsJob)
    q.job.__init__()
    command = [str(executable), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(checkout / _CHILD), "--checkout", str(checkout), "--sources", str(q.manifest),
               "--venv-root", str(venv), "--dependency", str(dependency), "--prefix", str(prefix),
               "--study-root", str(study_root), "--report", str(q.report), "--source-revision", revision,
               "--parent-platform", json.dumps(platform, separators=(",", ":"))]
    env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
           "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "PROCESSOR_IDENTIFIER") if name in os.environ}
    env.update({"PATH": str(Path(env.get("SYSTEMROOT", "C:/Windows")) / "System32"),
                "TEMP": str(q.temp), "TMP": str(q.temp), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    q.launch_started = True
    result = run_cpu_study(command, study_root=study_root, timeout=timeout, lease_timeout=5,
                           terminate_grace=5, cwd=q.temp, env=env, process_job=q.job)
    if (not result.clean or result.exit_code != 0 or result.timed_out
            or q.job._handle is not None or not q.job._confirmed):
        raise RehearsalError("supervisor did not confirm clean whole-job teardown")
    child = _bounded_json(q.report, 1024 * 1024)
    reference = _bounded_json(checkout / _REFERENCE, 65536)
    _check_child_report(child, result, revision, hashes, reference)
    if (launcher._revision(checkout) != revision or launcher._verify_sources(checkout, sources) != hashes
            or q.manifest.read_bytes() != manifest_bytes or any(prefix.iterdir())):
        raise RehearsalError("source or cache verification changed after child exit")
    finder.verify_inventory()
    return {**child, "reaped_child_pid": result.pid, "supervisor_run_id": result.run_id,
            "clean": True, "job_drained": True, "study_ready": False,
            "parent_death_source_protection": False}


def run(checkout, venv, dependency, work, study_root, *, attempt_id, timeout=120):
    quarantine = register_quarantine(attempt_id, _for_run=True)
    try:
        report = _execute(quarantine, checkout, venv, dependency, work, study_root, timeout)
        quarantine.complete = True
    finally:
        with _LOCK:
            quarantine.running = False
    quarantine.release()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "venv-root", "dependency", "work-dir", "study-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    attempt_id = uuid.uuid4().hex
    try:
        report = run(Path(args.checkout), Path(args.venv_root), Path(args.dependency), Path(args.work_dir),
                     Path(args.study_root), attempt_id=attempt_id, timeout=args.timeout)
    except BaseException as error:
        quarantine = get_quarantine(attempt_id)
        print(json.dumps({"status": "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                          "error_type": type(error).__name__, "attempt_id": attempt_id,
                          "quarantine_rooted_at_report": quarantine is not None,
                          "parent_exit_releases_handles": True,
                          "quarantine_location": str(quarantine.temp) if quarantine and quarantine.temp else None}),
              file=sys.stderr)
        if isinstance(error, SystemExit):
            raise
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
