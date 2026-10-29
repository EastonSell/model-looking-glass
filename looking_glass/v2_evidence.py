"""Bounded per-cell evidence publication for a planned synthetic study.

This is a development-only storage helper, not an experiment runner, a schema
validator for condition records, or proof of causal precommitment. See
docs/V2_EVIDENCE.md for the cooperating-writer and disk-accounting assumptions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import threading
from contextlib import contextmanager
from types import MappingProxyType

from . import registry


class EvidenceError(RuntimeError):
    """Evidence cannot be safely stored or treated as complete."""


_ONE_GIB = 1024 ** 3
_FAILURE = "failure.json"
_CONFIG = "v2_evidence"
_SCHEMA = "causal-v2-chunk-v1"
_FAILURE_SCHEMA = "causal-v2-failure-v1"
_WRITE_TOKEN = object()


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _plan(protocol, metadata, chunk_limits, extra_limits, required_extra,
          failure_limit, external_reserve_bytes, disk_limit_bytes):
    if (type(protocol) is not dict or type(metadata) is not dict
            or type(chunk_limits) is not dict or not chunk_limits
            or type(extra_limits) is not dict or type(required_extra) is not list):
        raise ValueError("identities, artifact declarations and required names are required")
    if _CONFIG in protocol or _CONFIG in metadata:
        raise ValueError("v2_evidence is reserved")
    _nonnegative(failure_limit, "failure_limit")
    _nonnegative(external_reserve_bytes, "external_reserve_bytes")
    _nonnegative(disk_limit_bytes, "disk_limit_bytes")
    if failure_limit == 0 or not 0 < disk_limit_bytes <= _ONE_GIB:
        raise ValueError("invalid failure or disk limit")
    if (_FAILURE in chunk_limits or _FAILURE in extra_limits
            or set(chunk_limits) & set(extra_limits)):
        raise ValueError("artifact declarations overlap")
    if (any(type(name) is not str for name in required_extra)
            or len(set(required_extra)) != len(required_extra)
            or not set(required_extra) <= set(extra_limits)):
        raise ValueError("required extras must be distinct declared names")
    limits = dict(extra_limits)
    limits.update(chunk_limits)
    limits[_FAILURE] = failure_limit
    for name, limit in limits.items():
        registry._validate_name(name)
        _nonnegative(limit, "artifact limit")
    # Registry enforces the 128-name, case-alias and 512 MiB payload limits.
    if len(limits) > registry.MAX_ARTIFACTS or sum(limits.values()) > registry.MAX_TOTAL_BYTES:
        raise EvidenceError("registry artifact count or payload limit exceeded")
    if len({name.lower() for name in limits}) != len(limits):
        raise ValueError("artifact names collide on case-insensitive filesystems")
    # Count all declared artifact payloads, the largest simultaneously retained
    # scratch path, manifest and outcome, and any external public/private files.
    pending_max = max(registry.MAX_METADATA_BYTES, max(limits.values()))
    worst_case = sum(limits.values()) + pending_max + 2 * registry.MAX_METADATA_BYTES
    if worst_case + external_reserve_bytes > disk_limit_bytes:
        raise EvidenceError("declared evidence exceeds the host disk envelope")
    config = {"schema": "causal-v2-evidence-layout-v1",
              "chunks": sorted(chunk_limits), "external_reserve_bytes": external_reserve_bytes,
              "disk_limit_bytes": disk_limit_bytes}
    identity = dict(protocol)
    identity[_CONFIG] = config
    return identity, limits, sorted(chunk_limits) + list(required_extra)


def _read_bounded(path, maximum):
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise EvidenceError("artifact exceeds its declared size")
    return data


def _chunk(data, name, maximum):
    if type(data) is not bytes or len(data) > maximum:
        raise EvidenceError("chunk exceeds its declared size or is not bytes")
    try:
        value = json.loads(data)
        if (type(value) is not dict
                or set(value) != {"schema", "name", "state", "records"}
                or value["schema"] != _SCHEMA or value["name"] != name
                or type(value["state"]) is not str
                or value["state"] not in ("complete", "incomplete")
                or type(value["records"]) is not list
                or registry._json_bytes(value) != data):
            raise ValueError("invalid chunk")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise EvidenceError("chunk is not a canonical valid envelope") from exc
    return value


def _failure(data, maximum):
    if type(data) is not bytes or len(data) > maximum:
        raise EvidenceError("failure evidence exceeds its declared size or is not bytes")
    try:
        value = json.loads(data)
        required = {"schema", "outcome", "stage", "error_type", "message", "submissions", "root_forwards"}
        if (type(value) is not dict or not required <= set(value)
                or not set(value) <= required | {"cell_id", "phase"}
                or value["schema"] != _FAILURE_SCHEMA
                or type(value["outcome"]) is not str or value["outcome"] not in ("failed", "cancelled")
                or type(value["stage"]) is not str or not value["stage"]
                or len(value["stage"]) > 256
                or type(value["error_type"]) is not str or not value["error_type"]
                or len(value["error_type"]) > 256
                or type(value["message"]) is not str
                or type(value["submissions"]) is not int
                or type(value["root_forwards"]) is not int
                or not 0 <= value["root_forwards"] <= value["submissions"] <= 60000
                or any(key in value and value[key] is not None
                       and (type(value[key]) is not str or len(value[key]) > 256)
                       for key in ("cell_id", "phase"))
                or registry._json_bytes(value) != data):
            raise ValueError("invalid failure envelope")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise EvidenceError("failure is not a complete canonical envelope") from exc
    return value


def _ordinary(info, *, directory=False):
    valid = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    return valid and not (getattr(info, "st_file_attributes", 0) & 0x400)


def _run_disk_bytes(path):
    """Count named payloads conservatively, including a pending hard-link alias."""
    names = {"manifest.json", "outcome.json", ".lock", ".pending", "artifacts"}
    total = 0
    for entry in os.scandir(path):
        if entry.name not in names:
            raise EvidenceError("unexpected path in evidence run")
        info = entry.stat(follow_symlinks=False)
        if entry.name == "artifacts":
            if not _ordinary(info, directory=True):
                raise EvidenceError("artifact directory is not ordinary")
            count = 0
            for item in os.scandir(entry.path):
                count += 1
                if count > registry.MAX_ARTIFACTS:
                    raise EvidenceError("too many artifact paths")
                child = item.stat(follow_symlinks=False)
                if not _ordinary(child):
                    raise EvidenceError("artifact is not an ordinary file")
                total += child.st_size
        else:
            if not _ordinary(info):
                raise EvidenceError("run entry is not an ordinary file")
            total += info.st_size
    return total


def _sync_directory(path):
    if os.name != "posix":
        return  # Windows has no portable directory fsync through os.open.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _raise_cleanup(primary, cleanup):
    if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
        raise cleanup from primary
    raise primary from cleanup


@contextmanager
def _locked(path, timeout):
    context = registry._lock(path, timeout)
    context.__enter__()
    try:
        yield
    except BaseException as primary:
        try:
            context.__exit__(type(primary), primary, primary.__traceback__)
        except BaseException as cleanup:
            _raise_cleanup(primary, cleanup)
        raise
    else:
        context.__exit__(None, None, None)


@contextmanager
def _pending_stream(path):
    stream = path.open("xb")
    try:
        yield stream
    except BaseException as primary:
        try:
            stream.close()
        except BaseException as cleanup:
            _raise_cleanup(primary, cleanup)
        raise
    else:
        stream.close()


def _publish_preserving_pending(path, destination, data):
    """Keep any partial scratch and make successful publication no-replace."""
    pending = path / ".pending"
    with _pending_stream(pending) as stream:
        remaining = memoryview(data)
        while remaining:
            count = stream.write(remaining)
            if type(count) is not int or count <= 0 or count > len(remaining):
                raise OSError("evidence write did not make valid progress")
            remaining = remaining[count:]
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path)
    os.link(pending, destination)
    _sync_directory(destination.parent)
    pending.unlink()
    _sync_directory(path)


def create_evidence(path, *, protocol, inputs, model, code, metadata, chunk_limits,
                    extra_limits, required_extra, failure_limit,
                    external_reserve_bytes=0, disk_limit_bytes=_ONE_GIB,
                    free_margin_bytes=0):
    """Create a new run after a conservative bound and live disk-free check."""
    _nonnegative(free_margin_bytes, "free_margin_bytes")
    identity, limits, required = _plan(protocol, metadata, chunk_limits, extra_limits,
                                       required_extra, failure_limit,
                                       external_reserve_bytes, disk_limit_bytes)
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("run path must be absolute")
    if shutil.disk_usage(path.parent).free < max(2 * _ONE_GIB, disk_limit_bytes + free_margin_bytes):
        raise EvidenceError("insufficient host disk headroom")
    run = registry.create_run(path, protocol=identity, inputs=inputs, model=model, code=code,
                              metadata=metadata, artifact_limits=limits,
                              required_artifacts=required, max_total_bytes=sum(limits.values()))
    return Evidence(run, _token=_WRITE_TOKEN)


def open_evidence(path):
    """Open evidence for inspection; an uncertain pending file still blocks writes."""
    return Evidence(registry.open_run(path))


class Evidence:
    def __init__(self, run, *, _token=None):
        self._run = run
        self.path = run.path
        self._writer = _token is _WRITE_TOKEN
        self._owner_pid = os.getpid()
        self._owner_thread = threading.current_thread()
        self._poisoned = False
        self._failure_only = False
        with _locked(self.path, run.lock_timeout):
            manifest, manifest_hash = registry._read_manifest(self.path)
            try:
                config = manifest["identities"]["protocol"]["value"][_CONFIG]
                chunks = config["chunks"]
                if (set(config) != {"schema", "chunks", "external_reserve_bytes", "disk_limit_bytes"}
                        or config["schema"] != "causal-v2-evidence-layout-v1"
                        or type(chunks) is not list or not chunks
                        or chunks != sorted(set(chunks))
                        or any(type(name) is not str for name in chunks)):
                    raise ValueError("invalid layout")
                _nonnegative(config["external_reserve_bytes"], "external reserve")
                _nonnegative(config["disk_limit_bytes"], "disk limit")
                if not 0 < config["disk_limit_bytes"] <= _ONE_GIB:
                    raise ValueError("invalid disk limit")
                limits = manifest["artifact_limits"]
                if (_FAILURE not in limits or any(name not in limits for name in chunks)
                        or any(name not in manifest["required_artifacts"] for name in chunks)):
                    raise ValueError("layout does not match manifest")
                pending_max = max(registry.MAX_METADATA_BYTES, max(limits.values()))
                if (sum(limits.values()) + pending_max + 2 * registry.MAX_METADATA_BYTES
                        + config["external_reserve_bytes"] > config["disk_limit_bytes"]):
                    raise ValueError("layout exceeds disk envelope")
            except (KeyError, ValueError, TypeError) as exc:
                raise EvidenceError("invalid frozen evidence layout") from exc
            self._manifest_hash = manifest_hash
            self._config = MappingProxyType({**config, "chunks": tuple(chunks)})
            self._limits = MappingProxyType(dict(limits))
            self._check_disk()

    @property
    def config(self):
        return self._config

    def __copy__(self):
        raise TypeError("evidence handles cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("evidence handles cannot be copied")

    def __reduce__(self):
        raise TypeError("evidence handles cannot be serialized")

    def _require_writer(self, *, allow_failure_only=False):
        if (not self._writer or self._poisoned or os.getpid() != self._owner_pid
                or threading.current_thread() is not self._owner_thread
                or (self._failure_only and not allow_failure_only)):
            raise EvidenceError("writer is unavailable; preserve the incomplete attempt")

    def _check_disk(self):
        if _run_disk_bytes(self.path) + self.config["external_reserve_bytes"] > self.config["disk_limit_bytes"]:
            raise EvidenceError("host evidence disk envelope exceeded")

    def _checked_manifest(self):
        manifest, digest = registry._read_manifest(self.path)
        if digest != self._manifest_hash:
            raise EvidenceError("frozen run manifest changed")
        return manifest, digest

    def _writable(self):
        if (self.path / ".pending").exists():
            raise EvidenceError("uncertain pending publication; preserve it and inspect")
        if (self.path / "outcome.json").exists():
            raise EvidenceError("run is already terminal")
        self._check_disk()

    def _write(self, name, data, *, kind):
        self._require_writer(allow_failure_only=(kind == "failure"))
        # These checks perform no filesystem mutation. A rejected chunk can
        # still produce one bounded evidence-limit failure before termination.
        try:
            if type(data) is not bytes:
                raise TypeError("artifact must be bytes")
            if name not in self._limits:
                raise KeyError(name)
            if kind == "chunk":
                if name not in self.config["chunks"]:
                    raise EvidenceError("undeclared chunk")
                _chunk(data, name, self._limits[name])
            elif kind == "failure":
                record = _failure(data, self._limits[_FAILURE])
                if self._failure_only and record["outcome"] != "failed":
                    raise EvidenceError("prepublication validation failure cannot be labelled cancelled")
            elif name in self.config["chunks"] or name == _FAILURE:
                raise EvidenceError("use the specific chunk or failure method")
            if len(data) > self._limits[name]:
                raise EvidenceError("artifact exceeds its declared size")
        except (EvidenceError, ValueError, TypeError, KeyError):
            if kind == "failure":
                self._poisoned = True
            else:
                self._failure_only = True
            raise
        except BaseException:
            self._poisoned = True
            raise
        try:
            with _locked(self.path, self._run.lock_timeout):
                self._writable()
                manifest, _ = self._checked_manifest()
                limits = manifest["artifact_limits"]
                destination = self.path / "artifacts" / name
                if destination.exists():
                    raise FileExistsError(destination)
                # The temporary and final paths can coexist after link publication.
                if (_run_disk_bytes(self.path) + 2 * len(data) + self.config["external_reserve_bytes"]
                        + registry.MAX_METADATA_BYTES > self.config["disk_limit_bytes"]):
                    raise EvidenceError("write would exhaust the terminal reserve")
                _publish_preserving_pending(self.path, destination, data)
        except BaseException:
            self._poisoned = True
            raise

    def publish_chunk(self, name, canonical_json_bytes):
        self._write(name, canonical_json_bytes, kind="chunk")

    def publish_artifact(self, name, data):
        self._write(name, data, kind="extra")

    def publish_failure(self, data):
        self._write(_FAILURE, data, kind="failure")

    def status(self):
        with _locked(self.path, self._run.lock_timeout):
            self._check_disk()
            base = registry._inspect(self.path)
            manifest, _ = self._checked_manifest()
            inventory = registry._inventory(self.path, manifest)
            pending = self.path / ".pending"
            evidence = None
            if pending.exists():
                size = pending.stat().st_size
                digest = registry.sha256_file(pending)
                linked = [name for name in inventory
                          if os.path.samefile(pending, self.path / "artifacts" / name)]
                if (self.path / "outcome.json").exists() and os.path.samefile(pending, self.path / "outcome.json"):
                    linked.append("outcome.json")
                evidence = {"bytes": size, "sha256": digest, "linked": linked}
            return {"state": base["state"], "artifacts": inventory,
                    "pending_write": evidence is not None, "pending": evidence,
                    "run_disk_bytes": _run_disk_bytes(self.path)}

    def finalize(self, state, details):
        self._require_writer(allow_failure_only=(state in ("failed", "cancelled")))
        try:
            if state not in ("succeeded", "failed", "cancelled") or type(details) is not dict:
                raise ValueError("invalid terminal state or details")
            with _locked(self.path, self._run.lock_timeout):
                self._writable()
                manifest, manifest_hash = self._checked_manifest()
                inventory = registry._inventory(self.path, manifest)
                if state == "succeeded":
                    if _FAILURE in inventory:
                        raise EvidenceError("a run with failure evidence cannot succeed")
                    for name in self.config["chunks"]:
                        if name not in inventory:
                            raise EvidenceError("successful run is missing a chunk")
                        data = _read_bounded(self.path / "artifacts" / name, manifest["artifact_limits"][name])
                        if _chunk(data, name, manifest["artifact_limits"][name])["state"] != "complete":
                            raise EvidenceError("successful run has an incomplete chunk")
                    if not set(manifest["required_artifacts"]) <= set(inventory):
                        raise EvidenceError("successful run is missing required evidence")
                else:
                    if _FAILURE not in inventory:
                        raise EvidenceError("failure or cancellation needs a complete failure artifact")
                    failure = _read_bounded(self.path / "artifacts" / _FAILURE,
                                            manifest["artifact_limits"][_FAILURE])
                    if _failure(failure, manifest["artifact_limits"][_FAILURE])["outcome"] != state:
                        raise EvidenceError("failure evidence and terminal state differ")
                outcome = {"schema_version": registry.SCHEMA_VERSION, "state": state,
                           "details": details, "manifest_sha256": manifest_hash, "artifacts": inventory}
                raw = registry._json_bytes(outcome)
                if len(raw) > registry.MAX_METADATA_BYTES:
                    raise EvidenceError("terminal outcome exceeds its bound")
                if (_run_disk_bytes(self.path) + 2 * len(raw) + self.config["external_reserve_bytes"]
                        > self.config["disk_limit_bytes"]):
                    raise EvidenceError("terminal outcome exceeds the disk envelope")
                _publish_preserving_pending(self.path, self.path / "outcome.json", raw)
        except BaseException:
            self._poisoned = True
            raise
