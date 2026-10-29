"""Cooperative reservations for the matched-magnitude BERT CPU owner.

The controller must derive each binding from its verified committed source,
manifest and checkpoint before calling this module. A lease and a reservation
are operational coordination, not proof of execution or a security boundary.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading

from .bert_evaluation_ledger import read_record, write_record
from .device_lease import DeviceLease, LeaseTimeoutError
from .supervisor import OwnerSession, SupervisionResult, _supervisor_birth


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ATTEMPT = re.compile(r"[0-9a-f]{32}\Z")
_ROW = re.compile(r"evaluation-matched-[A-Za-z0-9_-]{1,80}\Z")
_PROTOCOL = "docs/BERT_MATCHED_MAGNITUDE_V1.md"
_MANIFEST = "experiments/bert_matched_public_cases_v1.json"
_ASSETS = frozenset(("config.json", "tokenizer_config.json", "vocab.txt",
                     "tokenizer.json", "model.safetensors"))
_LOCK = "bert-matched-ledger.lock"
_UNSAFE = "cpu-study.unsafe.json"
_SCOPE = "cooperative-cpu-study-custody-only"
_WINDOWS_DEVICES = frozenset(("CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                              *(f"LPT{i}" for i in range(1, 10))))


def _directory(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("study roots must be absolute")
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("study roots cannot contain links or reparse points")
    if not path.is_dir():
        raise ValueError("study root must be an existing directory")
    return path.resolve(strict=True)


def _identity(path):
    return hashlib.sha256(os.path.normcase(str(path)).encode("utf-8")).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _observation(value):
    if (type(value) is not dict or set(value) != {"scope", "run_id", "pid", "exit_code", "clean", "timed_out", "reason"}
            or type(value["scope"]) is not str or value["scope"] != _SCOPE
            or type(value["run_id"]) is not str or not _ATTEMPT.fullmatch(value["run_id"])
            or type(value["pid"]) is not int or value["pid"] <= 0
            or type(value["exit_code"]) is not int
            or type(value["clean"]) is not bool or type(value["timed_out"]) is not bool
            or type(value["reason"]) is not str or not 1 <= len(value["reason"]) <= 256
            or not value["reason"].isascii() or not value["reason"].isprintable()):
        raise ValueError("the direct-owner exit is not a bounded reaped observation")
    return dict(value)


def _snapshot(value):
    """Copy bounded built-in values before evaluating any caller-owned value."""
    remaining = [100000, 1024 * 1024]

    def copy(item, depth):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 8:
            raise ValueError("binding is too large or nested")
        if type(item) is str:
            remaining[1] -= len(item)
            if len(item) > 512 or remaining[1] < 0:
                raise ValueError("binding strings exceed their bound")
            return item
        if type(item) is int:
            if not -(2 ** 63) <= item < 2 ** 63:
                raise ValueError("binding integer is outside its bound")
            return item
        if type(item) is bool or item is None:
            return item
        if type(item) is list:
            if len(item) > 4096:
                raise ValueError("binding list exceeds its bound")
            return [copy(member, depth + 1) for member in item.copy()]
        if type(item) is dict:
            if len(item) > 4096:
                raise ValueError("binding object exceeds its bound")
            pairs = item.copy()
            if any(type(key) is not str for key in pairs):
                raise ValueError("binding keys must be plain strings")
            return {copy(key, depth + 1): copy(member, depth + 1) for key, member in pairs.items()}
        raise ValueError("binding contains a non-built-in value")

    return copy(value, 0)


def _binding(value):
    """Validate bounded input and take an owned snapshot before reservation."""
    value = _snapshot(value)
    if (type(value) is not dict or set(value) != {"source_commit", "source_sha256",
            "protocol_sha256", "manifest_sha256", "prior_exposure_sha256", "model", "rows"}
            or type(value["source_commit"]) is not str or not _COMMIT.fullmatch(value["source_commit"])
            or any(type(value[name]) is not str or not _HASH.fullmatch(value[name]) for name in
                   ("protocol_sha256", "manifest_sha256", "prior_exposure_sha256"))):
        raise ValueError("invalid matched-study identity")
    sources = value["source_sha256"]
    if (type(sources) is not dict or not 1 <= len(sources) <= 4096
            or sources.get(_PROTOCOL) != value["protocol_sha256"]
            or sources.get(_MANIFEST) != value["manifest_sha256"]):
        raise ValueError("protocol and manifest must be present in the source inventory")
    owned_sources, folded = {}, set()
    for name, digest in sources.items():
        if type(name) is not str or len(name) > 512 or "\\" in name:
            raise ValueError("invalid source path")
        path = PurePosixPath(name)
        if (path.is_absolute() or path.as_posix() != name or any(p in ("", ".", "..") for p in path.parts)
                or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p.endswith(".")
                       or p.split(".", 1)[0].upper() in _WINDOWS_DEVICES for p in path.parts)
                or name.casefold() in folded or type(digest) is not str or not _HASH.fullmatch(digest)):
            raise ValueError("invalid source path or digest")
        folded.add(name.casefold())
        owned_sources[name] = digest
    model = value["model"]
    if (type(model) is not dict or set(model) != {"model", "revision", "assets"}
            or model["model"] != "distilbert/distilbert-base-uncased"
            or model["revision"] != "12040accade4e8a0f71eabdb258fecc2e7e948be"
            or type(model["assets"]) is not dict or set(model["assets"]) != _ASSETS):
        raise ValueError("invalid matched-study model identity")
    assets = {}
    for name, item in model["assets"].items():
        if (type(item) is not dict or set(item) != {"size", "sha256"}
                or type(item["size"]) is not int or item["size"] <= 0
                or type(item["sha256"]) is not str or not _HASH.fullmatch(item["sha256"])):
            raise ValueError("invalid checkpoint asset identity")
        assets[name] = dict(item)
    rows = value["rows"]
    if type(rows) is not list or len(rows) != 64:
        raise ValueError("the full 64-row exposure inventory is required")
    ids, inputs, owned_rows = set(), set(), []
    for row in rows:
        if (type(row) is not dict or set(row) != {"row_id", "input_ids", "attention_mask"}
                or type(row["row_id"]) is not str or not _ROW.fullmatch(row["row_id"])
                or type(row["input_ids"]) is not list or len(row["input_ids"]) != 10
                or any(type(x) is not int or not 0 <= x < 30522 for x in row["input_ids"])
                or type(row["attention_mask"]) is not list or row["attention_mask"] != [1] * 10
                or any(type(x) is not int for x in row["attention_mask"])
                or row["input_ids"][0] != 101 or row["input_ids"][6] != 103 or row["input_ids"][-1] != 102
                or row["row_id"] in ids or tuple(row["input_ids"]) in inputs):
            raise ValueError("invalid or duplicate matched-study input row")
        ids.add(row["row_id"])
        inputs.add(tuple(row["input_ids"]))
        owned_rows.append({"row_id": row["row_id"], "input_ids": list(row["input_ids"]),
                           "attention_mask": list(row["attention_mask"])})
    return {"source_commit": value["source_commit"], "source_sha256": owned_sources,
            "protocol_sha256": value["protocol_sha256"], "manifest_sha256": value["manifest_sha256"],
            "prior_exposure_sha256": value["prior_exposure_sha256"],
            "model": {"model": model["model"], "revision": model["revision"], "assets": assets},
            "rows": owned_rows}


def verify_owner_claim(root, custody, attempt_id, reservation_sha256, binding, *, session, lease):
    """Check a direct child's reservation and cooperative controller lease.

    The caller must independently rederive ``binding`` from committed sources,
    the canonical manifest and pinned assets. Lock contention alone cannot
    authenticate the controller or make this an adversarial security boundary.
    """
    root, custody, owned = _directory(root), _directory(custody), _binding(binding)
    if (type(attempt_id) is not str or not _ATTEMPT.fullmatch(attempt_id)
            or type(reservation_sha256) is not str or not _HASH.fullmatch(reservation_sha256)
            or type(session) is not OwnerSession or type(lease) is not DeviceLease):
        raise ValueError("invalid matched-study owner claim")
    OwnerSession.check(session, lease)
    if (session.identity is not None or session.supervisor_pid != os.getppid()
            or os.path.normcase(str(session.device_lease_path)) != os.path.normcase(str(custody / "cpu-study.lock"))
            or os.path.normcase(str(session.marker_path)) != os.path.normcase(str(custody / _UNSAFE))):
        raise ValueError("owner session is not bound to the expected CPU custody")
    path = _directory(root / attempt_id)
    if path.parent != root:
        raise ValueError("reservation is outside the ledger")
    record_path = path / "reservation.json"
    info = record_path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("reservation must be a plain file")
    reservation, digest = read_record(record_path)
    if (digest != reservation_sha256 or reservation.get("schema") != "bert-matched-reservation-v1"
            or reservation.get("attempt_id") != attempt_id
            or type(reservation.get("controller_pid")) is not int
            or reservation["controller_pid"] != session.supervisor_pid
            or reservation.get("controller_birth") != session.supervisor_birth
            or reservation.get("ledger_root_sha256") != _identity(root)
            or reservation.get("custody_root_sha256") != _identity(custody)
            or _canonical(_binding(reservation.get("binding"))) != _canonical(owned)):
        raise ValueError("owner reservation differs from the frozen binding")
    try:
        unexpected = DeviceLease.acquire(root / _LOCK, timeout=0)
    except LeaseTimeoutError:
        pass
    else:
        unexpected.release()
        raise ValueError("the controller ledger lease is not held")
    OwnerSession.check(session, lease)
    if read_record(record_path)[1] != digest:
        raise ValueError("reservation changed during the owner claim")
    return reservation


class MatchedLedger:
    """Retain the controller's ledger lease across reservation and owner exit."""

    def __init__(self, root, custody, binding, lease):
        self.root, self.custody, self.lease = root, custody, lease
        self._binding = _canonical(binding)
        self._thread, self._pid, self._closed, self._reserved = threading.current_thread(), os.getpid(), False, False
        self._attempt = None

    @property
    def binding(self):
        return json.loads(self._binding)

    @classmethod
    def acquire(cls, root, custody, binding, *, timeout=5):
        root, custody, binding = _directory(root), _directory(custody), _binding(binding)
        if root == custody or root.is_relative_to(custody) or custody.is_relative_to(root):
            raise ValueError("ledger and CPU custody roots must be separate")
        lease = DeviceLease.acquire(root / _LOCK, timeout=timeout)
        try:
            return cls(root, custody, binding, lease)
        except BaseException:
            lease.release()
            raise

    def _check(self):
        if self._closed or os.getpid() != self._pid or threading.current_thread() is not self._thread:
            raise RuntimeError("ledger is closed or belongs to another owner")
        self.lease.check()

    def _prior_attempts(self):
        attempts = []
        for path in self.root.iterdir():
            if path.name == _LOCK:
                continue
            if (len(attempts) >= 32 or not _ATTEMPT.fullmatch(path.name)
                    or _directory(path).parent != self.root):
                raise RuntimeError("ledger contains unknown or excessive entries")
            try:
                reservation, reservation_sha = read_record(path / "reservation.json")
                result, result_sha = read_record(path / "result.json")
            except (OSError, ValueError) as error:
                raise RuntimeError("prior matched-study attempt is unresolved") from error
            if (set(reservation) != {"schema", "attempt_id", "controller_pid", "controller_birth",
                    "ledger_root_sha256", "custody_root_sha256", "binding", "prior_attempts", "prior_may_have_exposed"}
                    or reservation["schema"] != "bert-matched-reservation-v1"
                    or reservation["attempt_id"] != path.name
                    or type(reservation["controller_pid"]) is not int or reservation["controller_pid"] <= 0
                    or type(reservation["controller_birth"]) is not str or not reservation["controller_birth"]
                    or reservation["ledger_root_sha256"] != _identity(self.root)
                    or reservation["custody_root_sha256"] != _identity(self.custody)
                    or _canonical(_binding(reservation["binding"])) != self._binding
                    or type(reservation["prior_attempts"]) is not list
                    or type(reservation["prior_may_have_exposed"]) is not bool
                    or reservation["prior_may_have_exposed"] != bool(reservation["prior_attempts"])
                    or set(result) != {"schema", "attempt_id", "status", "reservation_sha256", "supervision"}
                    or result["schema"] != "bert-matched-attempt-result-v1"
                    or result["attempt_id"] != path.name or result["status"] != "incomplete"
                    or result["reservation_sha256"] != reservation_sha):
                raise RuntimeError("prior matched-study attempt is unresolved or mismatched")
            _observation(result["supervision"])
            attempts.append((reservation, {"attempt_id": path.name, "reservation_sha256": reservation_sha,
                                           "result_sha256": result_sha, "status": "incomplete"}))
        attempts.sort(key=lambda item: len(item[0]["prior_attempts"]))
        previous = []
        for reservation, summary in attempts:
            if _canonical(reservation["prior_attempts"]) != _canonical(previous):
                raise RuntimeError("matched-study attempt history is not a single chain")
            previous.append(summary)
        return previous

    def reserve(self, attempt_id):
        self._check()
        if self._reserved or type(attempt_id) is not str or not _ATTEMPT.fullmatch(attempt_id):
            raise ValueError("invalid or duplicate attempt reservation")
        marker = self.custody / _UNSAFE
        if os.path.lexists(marker) or os.path.lexists(str(marker) + ".attestation"):
            raise RuntimeError("CPU custody is quarantined")
        prior = self._prior_attempts()
        if len(prior) >= 32 or any(item["attempt_id"] == attempt_id for item in prior):
            raise ValueError("matched-study history is full or the attempt exists")
        self._reserved = True
        path = self.root / attempt_id
        path.mkdir()
        write_record(path / "reservation.json", {
            "schema": "bert-matched-reservation-v1", "attempt_id": attempt_id,
            "controller_pid": os.getpid(), "controller_birth": _supervisor_birth(),
            "ledger_root_sha256": _identity(self.root), "custody_root_sha256": _identity(self.custody),
            "binding": self.binding, "prior_attempts": prior, "prior_may_have_exposed": bool(prior),
        })
        self._attempt = path
        return path

    def record_incomplete(self, observed):
        """Record a reaped direct owner while conservatively retaining exposure."""
        self._check()
        if self._attempt is None or type(observed) is not SupervisionResult:
            raise ValueError("an active attempt and a supervisor result are required")
        observation = _observation({"scope": _SCOPE, "run_id": observed.run_id, "pid": observed.pid,
                                    "exit_code": observed.exit_code, "clean": observed.clean,
                                    "timed_out": observed.timed_out, "reason": observed.reason})
        if not observation["clean"] and not os.path.lexists(self.custody / _UNSAFE):
            raise RuntimeError("unclean ownership has no quarantine marker")
        reservation, digest = read_record(self._attempt / "reservation.json")
        if (reservation.get("attempt_id") != self._attempt.name
                or _canonical(_binding(reservation.get("binding"))) != self._binding):
            raise RuntimeError("the active reservation changed")
        return write_record(self._attempt / "result.json", {
            "schema": "bert-matched-attempt-result-v1", "attempt_id": self._attempt.name,
            "status": "incomplete", "reservation_sha256": digest, "supervision": observation,
        })

    def release(self):
        self._check()
        self._closed = True
        self.lease.release()
