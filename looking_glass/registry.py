"""Bounded, append-only experiment evidence for cooperating local processes.

See docs/REGISTRY.md for the identity and filesystem trust assumptions.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


SCHEMA_VERSION = 1
MAX_METADATA_BYTES = 1024 * 1024
MAX_ARTIFACTS = 128
MAX_TOTAL_BYTES = 512 * 1024 * 1024
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
_IDENTITY_NAMES = ("protocol", "inputs", "model", "code")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
}


class RegistryIntegrityError(RuntimeError):
    """A published record is malformed or its contents no longer match."""


def _check_json(value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError("JSON numbers must be finite")
    if type(value) is list:
        for item in value:
            _check_json(item)
        return
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _check_json(item)
        return
    raise ValueError("values must consist of JSON types")


def _json_bytes(value):
    try:
        _check_json(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except RecursionError as exc:
        raise ValueError("JSON value is too deeply nested") from exc


def _validate_name(name):
    if (type(name) is not str or not _SAFE_NAME.fullmatch(name)
            or name.endswith(".") or name.split(".", 1)[0].upper() in _WINDOWS_DEVICES):
        raise ValueError(f"unsafe artifact name: {name!r}")


def _manifest(protocol, inputs, model, code, metadata, artifact_limits,
              required_artifacts, max_total_bytes):
    if any(type(value) is not dict for value in (protocol, inputs, model, code, metadata)):
        raise ValueError("identities and metadata must be JSON objects")
    if type(max_total_bytes) is not int or not 0 <= max_total_bytes <= MAX_TOTAL_BYTES:
        raise ValueError("invalid total artifact byte budget")
    if type(artifact_limits) is not dict or len(artifact_limits) > MAX_ARTIFACTS:
        raise ValueError("invalid artifact limits")
    folded_names = set()
    for name, limit in artifact_limits.items():
        _validate_name(name)
        if name.lower() in folded_names:
            raise ValueError("artifact names must be unique on case-insensitive filesystems")
        folded_names.add(name.lower())
        if type(limit) is not int or limit < 0:
            raise ValueError("artifact limits must be nonnegative integer bytes")
    if sum(artifact_limits.values()) > max_total_bytes:
        raise ValueError("declared artifact limits exceed the total budget")
    if (type(required_artifacts) is not list or
            any(type(name) is not str for name in required_artifacts) or
            len(set(required_artifacts)) != len(required_artifacts) or
            not set(required_artifacts) <= set(artifact_limits)):
        raise ValueError("required artifacts must be unique, declared names")
    values = (protocol, inputs, model, code)
    identities = {name: {"value": value, "sha256": hashlib.sha256(_json_bytes(value)).hexdigest()}
                  for name, value in zip(_IDENTITY_NAMES, values)}
    manifest = {"schema_version": SCHEMA_VERSION, "identities": identities,
                "metadata": metadata, "artifact_limits": artifact_limits,
                "required_artifacts": required_artifacts, "max_total_bytes": max_total_bytes}
    raw = _json_bytes(manifest)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("manifest exceeds metadata byte limit")
    return raw


def _read_json(path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("record exceeds metadata byte limit")
        value = json.loads(raw)
        if _json_bytes(value) != raw:
            raise ValueError("record is not canonical JSON")
        return value, raw
    except (OSError, ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise RegistryIntegrityError(f"invalid registry record: {path}") from exc


def _read_manifest(path):
    manifest, raw = _read_json(path / "manifest.json")
    try:
        identities = manifest["identities"]
        values = [identities[name]["value"] for name in _IDENTITY_NAMES]
        rebuilt = _manifest(*values, manifest["metadata"], manifest["artifact_limits"],
                            manifest["required_artifacts"], manifest["max_total_bytes"])
        if raw != rebuilt:
            raise ValueError("manifest fields or identity hashes do not match")
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistryIntegrityError("invalid manifest or identity hash") from exc
    return manifest, hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    """Hash file contents incrementally; callers must define what files identify."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _lock(path, timeout):
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError("lock timeout must be a finite nonnegative number")
    deadline = time.monotonic() + timeout
    with (path / ".lock").open("r+b") as stream:
        if os.name == "nt":
            import msvcrt

            def acquire():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out acquiring experiment registry lock") from exc
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            release()


def _clean_pending(path):
    # The lock ensures a previous writer is no longer using the single scratch file.
    (path / ".pending").unlink(missing_ok=True)


def _publish(path, destination, data):
    pending = path / ".pending"
    try:
        with pending.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link is atomically visible and fails if the target exists.
        os.link(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def _inventory(path, manifest):
    artifacts = path / "artifacts"
    try:
        entries = list(artifacts.iterdir())
        if any(not entry.is_file() or entry.name not in manifest["artifact_limits"]
               for entry in entries):
            raise RegistryIntegrityError("unexpected artifact or directory")
        inventory = {}
        for entry in entries:
            size = entry.stat().st_size
            if size > manifest["artifact_limits"][entry.name]:
                raise RegistryIntegrityError("artifact exceeds its declared byte limit")
            inventory[entry.name] = {"bytes": size, "sha256": sha256_file(entry)}
        return inventory
    except OSError as exc:
        raise RegistryIntegrityError("cannot read artifact inventory") from exc


def _inspect(path):
    manifest, manifest_hash = _read_manifest(path)
    inventory = _inventory(path, manifest)
    outcome_path = path / "outcome.json"
    if not outcome_path.exists():
        return {"state": "incomplete", "manifest": manifest, "artifacts": inventory,
                "pending_write": (path / ".pending").exists()}
    outcome, _ = _read_json(outcome_path)
    try:
        if (set(outcome) != {"schema_version", "state", "details", "manifest_sha256", "artifacts"}
                or type(outcome["schema_version"]) is not int
                or outcome["schema_version"] != SCHEMA_VERSION
                or outcome["state"] not in _TERMINAL_STATES
                or type(outcome["details"]) is not dict
                or outcome["manifest_sha256"] != manifest_hash
                or type(outcome["artifacts"]) is not dict
                or any(type(record) is not dict or type(record.get("bytes")) is not int
                       for record in outcome["artifacts"].values())
                or outcome["artifacts"] != inventory
                or (outcome["state"] == "succeeded" and
                    not set(manifest["required_artifacts"]) <= set(inventory))):
            raise ValueError("outcome does not match manifest or artifacts")
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistryIntegrityError("invalid or inconsistent terminal outcome") from exc
    return {"state": outcome["state"], "manifest": manifest, "outcome": outcome,
            "pending_write": (path / ".pending").exists()}


class ExperimentRun:
    def __init__(self, path, *, lock_timeout=10.0):
        self.path = Path(path).resolve()
        self.lock_timeout = lock_timeout

    def status(self):
        return verify_run(self.path, lock_timeout=self.lock_timeout)

    def write_artifact(self, name, data):
        if type(data) is not bytes:
            raise TypeError("artifact data must be bytes")
        with _lock(self.path, self.lock_timeout):
            _clean_pending(self.path)
            manifest, _ = _read_manifest(self.path)
            if (self.path / "outcome.json").exists():
                raise RuntimeError("run already has a terminal outcome")
            if name not in manifest["artifact_limits"]:
                raise KeyError(f"undeclared artifact: {name!r}")
            if len(data) > manifest["artifact_limits"][name]:
                raise ValueError("artifact exceeds its declared byte limit")
            destination = self.path / "artifacts" / name
            if destination.exists():
                raise FileExistsError(destination)
            _publish(self.path, destination, data)

    def finalize(self, state, details):
        if state not in _TERMINAL_STATES or type(details) is not dict:
            raise ValueError("terminal state and JSON-object details are required")
        with _lock(self.path, self.lock_timeout):
            _clean_pending(self.path)
            manifest, manifest_hash = _read_manifest(self.path)
            if (self.path / "outcome.json").exists():
                raise RuntimeError("run already has a terminal outcome")
            inventory = _inventory(self.path, manifest)
            if state == "succeeded" and not set(manifest["required_artifacts"]) <= set(inventory):
                raise ValueError("successful run is missing required artifacts")
            outcome = {"schema_version": SCHEMA_VERSION, "state": state, "details": details,
                       "manifest_sha256": manifest_hash, "artifacts": inventory}
            raw = _json_bytes(outcome)
            if len(raw) > MAX_METADATA_BYTES:
                raise ValueError("outcome exceeds metadata byte limit")
            _publish(self.path, self.path / "outcome.json", raw)


def create_run(path, *, protocol, inputs, model, code, metadata,
               artifact_limits, required_artifacts, max_total_bytes, lock_timeout=10.0):
    """Create a new exclusive run, preserving incomplete directories on failure."""
    raw = _manifest(protocol, inputs, model, code, metadata, artifact_limits,
                    required_artifacts, max_total_bytes)
    path = Path(path).resolve()
    path.mkdir()
    (path / "artifacts").mkdir()
    # Locking byte zero works with an empty file on Windows. Do not write after
    # creation: another process may already hold the byte-range lock.
    with (path / ".lock").open("xb"):
        pass
    with _lock(path, lock_timeout):
        _publish(path, path / "manifest.json", raw)
    return ExperimentRun(path, lock_timeout=lock_timeout)


def open_run(path, *, lock_timeout=10.0):
    """Open a previously initialized run for another cooperating process."""
    run = ExperimentRun(path, lock_timeout=lock_timeout)
    if run.status()["state"] == "uninitialized":
        raise ValueError("run was never initialized")
    return run


def verify_run(path, *, lock_timeout=10.0):
    """Check identities and, for terminal runs, every recorded artifact hash."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    lock = path / ".lock"
    if not lock.exists():
        # During the earliest creation window there is no lock yet. If it appears
        # while observing the directory, acquire it before deciding its state.
        artifacts = path / "artifacts"
        published = ((path / "manifest.json").exists() or (path / "outcome.json").exists() or
                     (artifacts.is_dir() and any(artifacts.iterdir())))
        if not lock.exists():
            if published:
                raise RegistryIntegrityError("published evidence is missing its registry lock")
            return {"state": "uninitialized", "pending_write": (path / ".pending").exists()}
    with _lock(path, lock_timeout):
        if not (path / "manifest.json").exists():
            artifacts = path / "artifacts"
            if (path / "outcome.json").exists() or (artifacts.is_dir() and any(artifacts.iterdir())):
                raise RegistryIntegrityError("published evidence is missing its manifest")
            return {"state": "uninitialized", "pending_write": (path / ".pending").exists()}
        return _inspect(path)
