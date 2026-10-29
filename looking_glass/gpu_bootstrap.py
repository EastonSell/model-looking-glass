"""Fail-closed evidence handling for the supervised two-token GPU bootstrap.

The GPU owner is deliberately not enabled by this module until its source,
memory-ledger and execution contracts have been integrated and reviewed.
"""

from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile
from types import MappingProxyType

from .resource_profile import _canonical_bytes
from .supervisor import OwnerSession, SupervisionResult
from .device_identity import DeviceIdentity, normalize_gpu_uuid
from .device_lease import DeviceLease


class BootstrapError(RuntimeError):
    """A bootstrap run cannot safely continue or publish its evidence."""


_GIB = 1024 ** 3
_MIB = 1024 ** 2
_REQUESTED = {
    "load_commitment_bytes": 2 * _GIB,
    "job_commitment_bytes": 512 * _MIB,
    "reserve_min_bytes": _GIB,
    "reserve_percent": 15,
    "allocator_fraction": "0.20",
    "input_limit_bytes": 64 * 1024,
    "output_tensor_limit_bytes": 512 * 1024,
    "response_wire_limit_bytes": 4 * _MIB,
    "max_gpu_trace_bytes": 0,
    "max_host_trace_bytes": 0,
}
_DOCUMENTS = frozenset({
    "owner.json", "source.json", "runtime.json", "assets.json", "load-case.json", "forward-case.json",
    "warmup.json", "capture.json", "checks.json", "load.profile.json", "load.evidence.json",
    "cold.profile.json", "cold.evidence.json", "warm.profile.json", "warm.evidence.json",
})
_JOURNALS = frozenset({"observations.jsonl", "admissions.jsonl"})
_STAGES = ("load", "cold", "warm", "warm", "warm")


@dataclass(frozen=True)
class OwnerLaunch:
    """Direct-child command and pre-exec controls; the cache directory is private."""

    command: tuple
    environment: object
    pycache_prefix: Path
    _prefix_identity: tuple
    source_index: Path
    _index_sha256: str

    def verify_ready(self):
        """Refuse a replaced or populated bytecode cache before child exec."""
        try:
            info = os.lstat(self.pycache_prefix)
            if (not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                    or (info.st_dev, info.st_ino) != self._prefix_identity
                    or any(self.pycache_prefix.iterdir())):
                raise BootstrapError("private bytecode prefix was replaced or is not empty")
            if self.source_index.stat().st_size > 32 * _MIB:
                raise BootstrapError("private source index is too large")
            with self.source_index.open("rb") as stream:
                index = stream.read(32 * _MIB + 1)
            if len(index) > 32 * _MIB or hashlib.sha256(index).hexdigest() != self._index_sha256:
                raise BootstrapError("private source index was changed")
        except OSError as exc:
            raise BootstrapError("private bytecode prefix is unavailable") from exc


_DIRECT_BOOTSTRAP = r'''\
# Only frozen/builtin modules are imported before the trusted path is selected.
import os, sys
try:
    startup_arguments = sys.orig_argv[1:sys.orig_argv.index("-c")]
except ValueError:
    raise RuntimeError("direct interpreter startup is required")
if not (sys.flags.no_site and sys.flags.safe_path and sys.dont_write_bytecode and "-B" in startup_arguments):
    raise RuntimeError("unsafe interpreter startup flags")
if (os.environ.get("PYTHONHASHSEED") != "17" or os.environ.get("PYTHONNOUSERSITE") != "1"
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or "PYTHONPATH" in os.environ or "PYTHONHOME" in os.environ):
    raise RuntimeError("unsafe interpreter startup environment")
def normalized(p):
    return os.path.normcase(os.path.realpath(p))
def inside(path, root):
    try:
        return os.path.commonpath((normalized(path), normalized(root))) == normalized(root)
    except (ValueError, TypeError):
        return False
base = normalized(sys.base_prefix)
sys.path[:] = [p for p in sys.path if p and inside(p, base)]
if not sys.path:
    raise RuntimeError("trusted standard library is unavailable")
import hashlib, importlib.machinery, importlib.metadata, json, runpy, stat
spec = json.loads(sys.argv.pop(1))
prefix = spec["prefix"]
info = os.lstat(prefix)
if (not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
        or (info.st_dev, info.st_ino) != tuple(spec["prefix_identity"])
        or normalized(sys.pycache_prefix) != normalized(prefix)
        or normalized(os.environ.get("PYTHONPYCACHEPREFIX", "")) != normalized(prefix)
        or os.listdir(prefix) or os.environ.get("PATH") != spec["path"]):
    raise RuntimeError("unsafe private cache or executable path")
project, site = spec["project"], spec["site"]
if inside(project, base) or inside(site, base):
    raise RuntimeError("source roots overlap the trusted standard library")
with open(spec["index"], "rb") as stream:
    raw = stream.read(32 * 1024 * 1024 + 1)
if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != spec["index_sha256"]:
    raise RuntimeError("source index was changed")
index = json.loads(raw)
allowed = {entry["path"]: (entry["size"], entry["sha256"]) for entry in index["files"]}
if len(allowed) != len(index["files"]):
    raise RuntimeError("duplicate executable source")
def checked_bytes(path, expected, materialize=True):
    if normalized(path) != os.path.normcase(os.path.abspath(path)):
        raise ImportError("source path has an alias or reparse point")
    info = os.lstat(path)
    if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or info.st_size != expected[0]):
        raise ImportError("source is not an ordinary file")
    digest = hashlib.sha256()
    chunks = []
    count = 0
    with open(path, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise ImportError("source changed while opening")
        while True:
            block = stream.read(min(1024 * 1024, expected[0] - count + 1))
            if not block:
                break
            count += len(block)
            if count > expected[0]:
                raise ImportError("source exceeds its approved byte size")
            digest.update(block)
            if materialize:
                chunks.append(block)
    if (count, digest.hexdigest()) != expected:
        raise ImportError("source differs from verified manifest")
    return b"".join(chunks) if materialize else None
class VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        return self.source_to_code(checked_bytes(path, allowed[normalized(path)]), path)
class VerifiedDistribution(importlib.metadata.Distribution):
    def __init__(self, distribution):
        self.distribution = distribution
    def read_text(self, filename):
        if (not isinstance(filename, str) or os.path.basename(filename) != filename
                or filename in ("", ".", "..")):
            raise ImportError("invalid distribution metadata path")
        target = os.path.join(os.fspath(self.distribution._path), filename)
        if not os.path.exists(target):
            return None
        key = normalized(target)
        if not inside(target, site) or key not in allowed:
            raise ImportError("distribution metadata is absent from verified source manifest")
        return checked_bytes(target, allowed[key]).decode("utf-8")
    def locate_file(self, path):
        return self.distribution.locate_file(path)
class Guard:
    @classmethod
    def assert_bound(cls, source_sha256):
        if (sys.meta_path != [cls]
                or getattr(sys, "_looking_glass_import_guard", None) is not cls
                or source_sha256 != spec["source_sha256"]
                or os.environ.get("MLG_SOURCE_MANIFEST_SHA256") != source_sha256
                or tuple(sys.path) != cls.approved_paths or os.listdir(prefix)
                or normalized(sys.pycache_prefix) != normalized(prefix)
                or not (sys.flags.no_site and sys.flags.safe_path and sys.dont_write_bytecode)):
            raise RuntimeError("verified import guard or source binding changed")
    @classmethod
    def assert_torch_dll_inventory(cls, source_sha256):
        cls.assert_bound(source_sha256)
        directory = os.path.join(site, "torch", "lib")
        if not os.path.isdir(directory) or normalized(directory) != os.path.normcase(os.path.abspath(directory)):
            raise ImportError("Torch native library directory is unavailable or redirected")
        expected = {path: record for path, record in allowed.items()
                    if os.path.dirname(path) == normalized(directory) and path.lower().endswith(".dll")}
        actual = {}
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.lower().endswith(".dll"):
                    key = normalized(entry.path)
                    if key in actual:
                        raise ImportError("duplicate Torch native library")
                    actual[key] = entry.path
        if not expected or set(actual) != set(expected):
            raise ImportError("Torch native library inventory differs from verified preflight")
        for key, path in actual.items():
            checked_bytes(path, expected[key], materialize=False)
        cls.assert_bound(source_sha256)
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if sys.meta_path != [cls]:
            raise ImportError("the approved import finder list changed")
        for builtin in (importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter):
            found = builtin.find_spec(fullname, path, target)
            if found is not None:
                return found
        found = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if found is None:
            return None
        if found.loader is None:
            if found.submodule_search_locations is None or any(
                    not (inside(p, base) or inside(p, project) or inside(p, site))
                    for p in found.submodule_search_locations):
                raise ImportError("unknown namespace path")
            return found
        loader = type(found.loader)
        if loader not in (importlib.machinery.SourceFileLoader, importlib.machinery.ExtensionFileLoader):
            raise ImportError("unsupported or sourceless module loader")
        origin = found.origin
        if not origin:
            raise ImportError("module has no source origin")
        if inside(origin, project) or inside(origin, site):
            key = normalized(origin)
            if key not in allowed:
                raise ImportError("module is absent from verified source manifest")
            checked_bytes(origin, allowed[key], materialize=False)
            if loader is importlib.machinery.SourceFileLoader:
                found.loader = VerifiedSourceLoader(fullname, origin)
        elif not inside(origin, base):
            raise ImportError("module origin is outside trusted roots")
        return found
    @classmethod
    def find_distributions(cls, context):
        if sys.meta_path != [cls]:
            raise ImportError("the approved import finder list changed")
        for distribution in importlib.machinery.PathFinder.find_distributions(context):
            if (type(distribution) is not importlib.metadata.PathDistribution
                    or not inside(os.fspath(distribution._path), site)):
                raise ImportError("distribution origin is outside the approved environment")
            yield VerifiedDistribution(distribution)
sys.meta_path[:] = [Guard]
sys.path.extend((project, site))
Guard.approved_paths = tuple(sys.path)
sys._looking_glass_import_guard = Guard
Guard.assert_bound(spec["source_sha256"])
runpy.run_module(spec["module"], run_name="__main__", alter_sys=True)
'''


def _source_index(source_manifest, project, environment_root, executable):
    """Build a bounded origin index from the trusted CPU preflight's manifest.

    This validates the file identities supplied by preflight; callers must also
    establish the clean Git snapshot and distribution inventory. Individual
    executable files are rehashed in the child immediately before import.
    """
    if type(source_manifest) is not bytes or len(source_manifest) > 32 * 1024 * 1024:
        raise ValueError("source manifest must be bounded canonical bytes")
    try:
        manifest = json.loads(source_manifest)
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError("invalid source manifest") from exc
    if (canonical != source_manifest or type(manifest) is not dict
            or set(manifest) != {"schema", "git_commit", "git_tree", "repository_files",
                                 "python_executable", "distributions"}
            or manifest["schema"] != "resource-bootstrap-source-v1"):
        raise ValueError("invalid canonical source manifest")
    for field in ("git_commit", "git_tree"):
        if type(manifest[field]) is not str or re.fullmatch(r"[0-9a-f]{40}", manifest[field]) is None:
            raise ValueError("invalid source revision")
    def validate_record(record, root=None):
        expected = {"size", "sha256"} if root is None else {"path", "size", "sha256"}
        if (type(record) is not dict or set(record) != expected
                or type(record["size"]) is not int or record["size"] < 0
                or type(record["sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
            raise ValueError("invalid source file record")
        if root is None:
            return
        name = record["path"]
        if (type(name) is not str or not name or "\\" in name or ":" in name
                or "\0" in name or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("invalid source-relative path")
        path = root.joinpath(*name.split("/"))
        try:
            resolved = path.resolve(strict=True)
            info = os.lstat(path)
        except OSError as exc:
            raise ValueError("manifest source is unavailable") from exc
        if (not resolved.is_relative_to(root) or os.path.normcase(str(resolved)) != os.path.normcase(str(path))
                or not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("manifest source is not an ordinary file beneath its root")
        return {"path": os.path.normcase(str(resolved)), "size": record["size"], "sha256": record["sha256"]}
    validate_record(manifest["python_executable"])
    executable_bytes = executable.read_bytes()
    if manifest["python_executable"] != {"size": len(executable_bytes),
                                          "sha256": hashlib.sha256(executable_bytes).hexdigest()}:
        raise ValueError("Python executable differs from verified preflight")
    if type(manifest["repository_files"]) is not list or type(manifest["distributions"]) is not list:
        raise ValueError("source file inventory is unavailable")
    records = [validate_record(record, project) for record in manifest["repository_files"]]
    for distribution in manifest["distributions"]:
        if (type(distribution) is not dict or set(distribution) != {"name", "version", "files"}
                or type(distribution["name"]) is not str or type(distribution["version"]) is not str
                or type(distribution["files"]) is not list):
            raise ValueError("invalid distribution file inventory")
        records.extend(validate_record(record, environment_root) for record in distribution["files"])
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("duplicate source origin")
    index = json.dumps({"files": records}, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
    if len(index) > 32 * 1024 * 1024:
        raise ValueError("source index exceeds its byte cap")
    return index


def prepare_owner_launch(*, project_root, site_packages, environment_root, private_parent, gpu_uuid,
                         module, arguments, source_manifest, trusted_path, inherited_environment=None):
    """Prepare a direct base-Python child with pre-exec reproducibility controls.

    The caller must preserve and inspect the private cache prefix before and
    after the child. No site hooks, inherited Python paths, credentials or
    unreviewed accelerator settings enter the child environment.
    """
    gpu_uuid = normalize_gpu_uuid(gpu_uuid)
    if type(module) is not str or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) is None:
        raise ValueError("module must be a dotted Python identifier")
    if type(arguments) not in (list, tuple) or any(type(item) is not str or "\0" in item for item in arguments):
        raise ValueError("arguments must be a sequence of strings")
    roots = []
    for item in (project_root, site_packages, environment_root, private_parent):
        path = Path(item)
        if not path.is_absolute():
            raise ValueError("launcher directories must be absolute")
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError("launcher paths must be directories")
        roots.append(path)
    project, site, environment_root, private = roots
    if not site.is_relative_to(environment_root):
        raise ValueError("site-packages must be within the selected environment")
    base = Path(sys._base_executable).resolve(strict=True)
    if not base.is_file():
        raise ValueError("base Python executable is unavailable")
    if type(trusted_path) is not str or not trusted_path or "\0" in trusted_path:
        raise ValueError("an explicit controlled executable path is required")
    path_entries = trusted_path.split(os.pathsep)
    if any(not Path(entry).is_absolute() or not Path(entry).is_dir() for entry in path_entries):
        raise ValueError("controlled executable path entries must be absolute directories")
    checked_path = os.pathsep.join(str(Path(entry).resolve(strict=True)) for entry in path_entries)
    index = _source_index(source_manifest, project, environment_root, base)
    source_sha256 = hashlib.sha256(source_manifest).hexdigest()
    source = os.environ if inherited_environment is None else inherited_environment
    if type(source) is not dict and source is not os.environ:
        raise TypeError("inherited environment must be a dictionary")
    environment = {}
    keep = {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP"}
    for key, value in source.items():
        if type(key) is str and key.upper() in keep and type(value) is str:
            environment[key] = value
    prefix = Path(tempfile.mkdtemp(prefix="looking-glass-pyc-", dir=private))
    descriptor, index_name = tempfile.mkstemp(prefix="looking-glass-source-", suffix=".json", dir=private)
    index_path = Path(index_name)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(index)
        stream.flush()
        os.fsync(stream.fileno())
    index_sha256 = hashlib.sha256(index).hexdigest()
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONHASHSEED": "17",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(prefix),
        "PATH": checked_path,
        "MLG_SOURCE_MANIFEST_SHA256": source_sha256,
    })
    info = os.lstat(prefix)
    spec = json.dumps({"project": str(project), "site": str(site), "module": module,
                       "prefix": str(prefix), "prefix_identity": (info.st_dev, info.st_ino),
                       "index": str(index_path), "index_sha256": index_sha256,
                       "source_sha256": source_sha256, "path": checked_path},
                      separators=(",", ":"))
    command = (str(base), "-B", "-S", "-P", "-X", "pycache_prefix=" + str(prefix),
               "-c", _DIRECT_BOOTSTRAP, spec, *arguments)
    return OwnerLaunch(command=command, environment=MappingProxyType(environment), pycache_prefix=prefix,
                       _prefix_identity=(info.st_dev, info.st_ino), source_index=index_path,
                       _index_sha256=index_sha256)


def assert_import_guard(source_sha256):
    """Require the launcher's still-active guard for this exact source manifest.

    This is an integrity check for cooperating trusted code, not a security
    boundary against code that intentionally mutates interpreter internals.
    """
    _digest(source_sha256, 64, "source_sha256")
    guard = getattr(sys, "_looking_glass_import_guard", None)
    if not isinstance(guard, type) or not sys.meta_path or sys.meta_path[0] is not guard:
        raise BootstrapError("the verified import guard is absent")
    guard.assert_bound(source_sha256)


# A complete reviewed profiling, disposal and admission path is not yet present.
GPU_BOOTSTRAP_READY = False
_PROCESS_LIFETIME_DEVICE_LEASES = []


def _torch_runtime_uuid(torch, value):
    """Decode the exact PyTorch 2.8 CUDA UUID object without accepting ordinals."""
    import uuid
    uuid_type = getattr(getattr(torch, "_C", None), "_CUuuid", None)
    if not isinstance(uuid_type, type) or type(value) is not uuid_type:
        raise BootstrapError("unsupported CUDA runtime UUID type")
    rendered = str(value)
    try:
        parsed = uuid.UUID(rendered)
    except (ValueError, AttributeError) as exc:
        raise BootstrapError("unsupported CUDA runtime UUID format") from exc
    if rendered != str(parsed):
        raise BootstrapError("CUDA runtime UUID must use canonical lowercase format")
    return normalize_gpu_uuid("GPU-" + rendered)


def claim_owner_and_load_dependencies(directory, *, parent_run_id, started_sha256,
                                      source_sha256, lease_timeout=30):
    """Claim the supervised device before importing CUDA-capable dependencies.

    This is an owner-process startup primitive, not an enabled GPU runner.
    The lease is intentionally retained until process exit, even on exceptions.
    """
    if GPU_BOOTSTRAP_READY is not True:
        raise BootstrapError("GPU bootstrap is disabled until its complete path is reviewed")
    _digest(parent_run_id, 32, "parent_run_id")
    _digest(started_sha256, 64, "started_sha256")
    assert_import_guard(source_sha256)
    forbidden = {"torch", "transformers", "accelerate", "triton", "torchvision", "torchaudio", "cupy"}
    if any(name.partition(".")[0] in forbidden for name in sys.modules):
        raise BootstrapError("a CUDA-capable dependency was imported before owner startup")
    session = OwnerSession.from_environment()
    if type(session) is not OwnerSession:
        raise BootstrapError("an exact supervised owner session is required")
    lease = DeviceLease.acquire(session.device_lease_path, timeout=lease_timeout)
    _PROCESS_LIFETIME_DEVICE_LEASES.append(lease)
    OwnerSession.check(session, lease)
    if type(session.identity) is not DeviceIdentity:
        raise BootstrapError("a supervised physical GPU identity is required")
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError("bootstrap evidence directory must be absolute")
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise BootstrapError("bootstrap evidence directory is not an ordinary directory")
    start_path = path / "started.json"
    info = os.lstat(start_path)
    if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
            or info.st_nlink != 1 or info.st_size > _MIB):
        raise BootstrapError("bootstrap start evidence is invalid")
    raw = start_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != started_sha256:
        raise BootstrapError("bootstrap start evidence changed")
    started = json.loads(raw)
    if (_canonical_bytes(started) != raw or type(started) is not dict
            or started.get("schema") != "resource-bootstrap-start-v1"
            or started.get("run_id") != parent_run_id):
        raise BootstrapError("bootstrap start evidence does not match this owner")
    owner = {"schema": "resource-bootstrap-owner-v1", "parent_run_id": parent_run_id,
             "supervisor_run_id": session.run_id, "pid": os.getpid()}
    _write_new(path / "owner.json", _canonical_bytes(owner))
    OwnerSession.check(session, lease)
    assert_import_guard(source_sha256)
    sys._looking_glass_import_guard.assert_torch_dll_inventory(source_sha256)
    OwnerSession.check(session, lease)
    assert_import_guard(source_sha256)
    import importlib
    torch = importlib.import_module("torch")
    OwnerSession.check(session, lease)
    assert_import_guard(source_sha256)
    if torch.cuda.is_initialized() is not False:
        raise BootstrapError("CUDA was initialized during dependency import")
    OwnerSession.check(session, lease)
    count = torch.cuda.device_count()
    if type(count) is not int or count != 1:
        raise BootstrapError("the runtime must expose exactly one GPU")
    OwnerSession.check(session, lease)
    runtime_uuid = _torch_runtime_uuid(torch, torch.cuda.get_device_properties(0).uuid)
    OwnerSession.verify_runtime_binding(session, lease, runtime_uuid=runtime_uuid, device_count=count)
    assert_import_guard(source_sha256)
    OwnerSession.check(session, lease)
    transformers = importlib.import_module("transformers")
    OwnerSession.check(session, lease)
    assert_import_guard(source_sha256)
    return session, lease, torch, transformers, runtime_uuid


def _utc(value):
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if type(value) is not str:
        raise ValueError("timestamp must be a UTC ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be a UTC ISO-8601 string") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp must be UTC")
    return value


def _digest(value, length, name):
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _write_new(path, data):
    """Publish new bytes without overwriting an existing result or scratch file."""
    if os.path.lexists(path):
        raise FileExistsError(path)
    scratch = path.with_name(path.name + ".partial")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(scratch, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # A hard link is an atomic no-replace publication on the same filesystem.
    # If publication fails, the complete or partial scratch remains for audit.
    os.link(scratch, path)
    scratch.unlink()


def _write_terminal_new(path, data):
    """Atomically publish terminal bytes without leaving scratch in its inventory."""
    if os.path.lexists(path):
        raise FileExistsError(path)
    if os.name == "nt":
        # Windows rename is atomic on this filesystem and refuses an existing
        # destination. No post-publication scratch unlink is needed.
        scratch = path.with_name(path.name + ".partial")
        descriptor = os.open(scratch, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(scratch, path)
        return
    # On POSIX, rename may replace a competing destination. Stage beside the
    # run so link publication is no-replace and a failed cleanup cannot add an
    # unlisted file to an already visible terminal inventory.
    staging = Path(tempfile.mkdtemp(prefix=".looking-glass-terminal-", dir=path.parent.parent))
    scratch = staging / "terminal.json.partial"
    published = False
    try:
        descriptor = os.open(scratch, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(scratch, path)
        published = True
    finally:
        try:
            scratch.unlink()
            staging.rmdir()
        except OSError:
            if not published:
                raise


def _inventory(directory):
    records = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == "terminal.json":
            raise FileExistsError(path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise BootstrapError("run inventory contains a link or non-file entry")
        before = path.stat()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise BootstrapError("run evidence changed during inventory")
        if size != after.st_size:
            raise BootstrapError("run evidence size changed during inventory")
        records.append({"path": path.name, "size": size, "sha256": digest.hexdigest()})
    return records


class BootstrapJournal:
    """Single-run evidence writer owned by the supervising parent process."""

    def __init__(self, directory, started):
        self.directory = directory
        self.started = started
        self._parent_pid = os.getpid()
        info = os.lstat(directory)
        self._directory_identity = (info.st_dev, info.st_ino)
        self._started_bytes = _canonical_bytes(started)

    def _check_run(self):
        try:
            info = os.lstat(self.directory)
            if (not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                    or (info.st_dev, info.st_ino) != self._directory_identity):
                raise BootstrapError("bootstrap run directory was replaced or redirected")
            start = self.directory / "started.json"
            info = os.lstat(start)
            if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                    or info.st_nlink != 1 or start.read_bytes() != self._started_bytes):
                raise BootstrapError("bootstrap start evidence was replaced or changed")
        except OSError as exc:
            raise BootstrapError("bootstrap run directory or start evidence is unavailable") from exc

    @classmethod
    def create(cls, directory, *, protocol_sha256, git_commit, git_tree,
               run_id=None, started_utc=None):
        path = Path(directory)
        if not path.is_absolute():
            raise ValueError("run directory must be absolute")
        _digest(protocol_sha256, 64, "protocol_sha256")
        _digest(git_commit, 40, "git_commit")
        _digest(git_tree, 40, "git_tree")
        if run_id is None:
            run_id = secrets.token_hex(16)
        _digest(run_id, 32, "run_id")
        started = {"schema": "resource-bootstrap-start-v1", "run_id": run_id,
                   "started_utc": _utc(started_utc), "protocol_sha256": protocol_sha256,
                   "git_commit": git_commit, "git_tree": git_tree, "requested": dict(_REQUESTED)}
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        _write_new(path / "started.json", _canonical_bytes(started))
        return cls(path, started)

    def write_document(self, name, value):
        if name not in _DOCUMENTS:
            raise ValueError("unsupported bootstrap evidence document")
        self._check_run()
        _write_new(self.directory / name, _canonical_bytes(value))
        self._check_run()

    def append_record(self, name, value):
        if name not in _JOURNALS or type(value) is not dict:
            raise ValueError("unsupported bootstrap evidence journal")
        self._check_run()
        path = self.directory / name
        if path.is_symlink():
            raise BootstrapError("journal cannot be a symbolic link")
        if path.exists():
            previous = path.read_bytes()
            if len(previous) > 2 * _MIB or (previous and not previous.endswith(b"\n")):
                raise BootstrapError("journal is too large or has a truncated record")
            import json
            try:
                records = [json.loads(line) for line in previous.splitlines()]
            except (ValueError, UnicodeError) as exc:
                raise BootstrapError("journal contains invalid JSON") from exc
            if any(_canonical_bytes(record) != line for record, line in zip(records, previous.splitlines())):
                raise BootstrapError("journal contains noncanonical records")
        else:
            records = []
        sequence = len(records) + 1
        if sequence > len(_STAGES) or value.get("sequence") != sequence or value.get("stage") != _STAGES[sequence - 1]:
            raise ValueError("journal stage or sequence is not the next prescribed record")
        for index, record in enumerate(records):
            if type(record) is not dict or record.get("sequence") != index + 1 or record.get("stage") != _STAGES[index]:
                raise BootstrapError("existing journal stages are not in order")
        data = _canonical_bytes(value) + b"\n"
        if len(data) > 2 * _MIB:
            raise ValueError("journal record is too large")
        flags = os.O_APPEND | os.O_WRONLY | os.O_CREAT | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self._check_run()

    def finish(self, supervision, *, failure_stage=None, reason=None, cancelled=False,
               owner_may_have_started=True, ended_utc=None):
        """Publish a terminal record only after the supervising parent regains control."""
        if os.getpid() != self._parent_pid:
            raise BootstrapError("only the creating parent may publish the terminal")
        if supervision is not None and type(supervision) is not SupervisionResult:
            raise TypeError("supervision must be the actual supervisor result")
        if supervision is not None:
            if (type(supervision.clean) is not bool or type(supervision.timed_out) is not bool
                    or type(supervision.pid) is not int or supervision.pid <= 0
                    or (supervision.exit_code is not None and type(supervision.exit_code) is not int)
                    or type(supervision.reason) is not str):
                raise TypeError("supervisor result fields have invalid types")
            _digest(supervision.run_id, 32, "supervisor run_id")
            if supervision.clean and (supervision.timed_out or supervision.exit_code is None):
                raise ValueError("supervisor reports an impossible clean state")
        if type(cancelled) is not bool or type(owner_may_have_started) is not bool:
            raise TypeError("cancelled and owner_may_have_started must be boolean")
        self._check_run()
        if os.path.lexists(self.directory / "terminal.json"):
            raise FileExistsError(self.directory / "terminal.json")
        clean = bool(supervision is not None and supervision.clean and not supervision.timed_out)
        exit_code = None if supervision is None else supervision.exit_code
        if ((supervision is not None and not clean)
                or (supervision is None and (owner_may_have_started or os.path.lexists(self.directory / "owner.json")))):
            status = "quarantined"
        elif cancelled:
            status = "cancelled"
        elif supervision is None or exit_code != 0:
            status = "failed"
        else:
            status = "success"
        if clean:
            try:
                path = self.directory / "owner.json"
                info = os.lstat(path)
                if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400
                        or info.st_nlink != 1):
                    raise BootstrapError("owner record is not a private regular file")
                raw = path.read_bytes()
                owner = json.loads(raw)
                expected = {"schema": "resource-bootstrap-owner-v1",
                            "parent_run_id": self.started["run_id"],
                            "supervisor_run_id": supervision.run_id, "pid": supervision.pid}
                if owner != expected or _canonical_bytes(owner) != raw:
                    raise BootstrapError("owner record does not match the reaped supervisor child")
            except (OSError, ValueError, TypeError) as exc:
                raise BootstrapError("missing or invalid supervisor-bound owner evidence") from exc
        if status == "success":
            if failure_stage is not None or reason is not None:
                raise ValueError("successful terminal cannot have a failure reason")
        elif (type(failure_stage) is not str or not failure_stage.strip()
              or type(reason) is not str or not reason.strip()):
            raise ValueError("non-success terminal requires a failure stage and reason")
        ended = _utc(ended_utc)
        if datetime.fromisoformat(ended.replace("Z", "+00:00")) < datetime.fromisoformat(
                self.started["started_utc"].replace("Z", "+00:00")):
            raise ValueError("terminal timestamp predates run start")
        terminal = {"schema": "resource-bootstrap-terminal-v1", "run_id": self.started["run_id"],
                    "ended_utc": ended, "status": status, "child_exit_code": exit_code,
                    "clean_teardown": clean, "failure_stage": failure_stage,
                    "reason": reason, "supervisor": (None if supervision is None else {
                        "run_id": supervision.run_id, "pid": supervision.pid,
                        "exit_code": supervision.exit_code, "clean": supervision.clean,
                        "timed_out": supervision.timed_out,
                    }), "inventory": _inventory(self.directory)}
        self._check_run()
        if status == "success":
            try:
                from .resource_bootstrap_audit import audit_candidate
                audit = audit_candidate(self.directory, terminal)
            except (ImportError, ValueError, RuntimeError) as exc:
                raise BootstrapError("independent candidate audit did not approve publication") from exc
            if type(audit) is not dict or audit.get("status") != "success":
                raise BootstrapError("independent candidate audit did not approve publication")
            if _inventory(self.directory) != terminal["inventory"]:
                raise BootstrapError("run evidence changed after candidate audit")
        _write_terminal_new(self.directory / "terminal.json", _canonical_bytes(terminal))
        return terminal
