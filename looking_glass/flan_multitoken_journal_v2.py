"""Bounded per-subject saved evidence for the multi-subtoken protocol.

This journal preserves bytes and partial failures. It does not authenticate
the process, reserve disk space, select cases, or replace the owner's combined
4.5-GiB preflight and independent source/custody checks. Its parent is trusted.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import threading

from .flan_t5_sentiment_journal import (
    EvidenceJournal, JournalError, MAX_COMPONENT_BYTES, METADATA_BYTES,
    FAILURE_BYTES, _encoded, _write_new,
)


_DISCOVERY_CAP = 5 * 1024**3 // 4
_HOLDOUT_CAP = 3 * 1024**3 // 4
_PROFILES = {
    "story": (_DISCOVERY_CAP, 3516, 580, 36, 8),
    "concert": (_DISCOVERY_CAP, 3516, 580, 36, 8),
    "painting": (_HOLDOUT_CAP, 1758, 290, 18, 4),
}
_MAX_VECTOR_BYTES = 512 * 8


@dataclass(frozen=True, slots=True)
class LimitsV2:
    subject: str
    total_bytes: int
    metadata_bytes: int
    failure_bytes: int
    forward_bytes: int
    max_forwards: int
    free_bytes: int

    def __post_init__(self):
        if type(self.subject) is not str or self.subject not in _PROFILES:
            raise ValueError("invalid pass subject")
        cap, calls, _, _, _ = _PROFILES[self.subject]
        values = ((self.total_bytes, cap), (self.metadata_bytes, METADATA_BYTES),
                  (self.failure_bytes, FAILURE_BYTES), (self.forward_bytes, MAX_COMPONENT_BYTES),
                  (self.max_forwards, calls))
        if (any(type(value) is not int or not 0 < value <= maximum for value, maximum in values)
                or type(self.free_bytes) is not int or not 0 <= self.free_bytes <= cap
                or self.metadata_bytes + self.failure_bytes >= self.total_bytes):
            raise ValueError("invalid evidence limits")

    @classmethod
    def for_subject(cls, subject):
        if type(subject) is not str or subject not in _PROFILES:
            raise ValueError("invalid pass subject")
        cap, calls, _, _, _ = _PROFILES[subject]
        return cls(subject, cap, METADATA_BYTES, FAILURE_BYTES, MAX_COMPONENT_BYTES, calls, cap)


def require_protocol_allocation(limits):
    """Validate the complete pass allowance, including saved binary64 vectors."""
    if type(limits) is not LimitsV2 or limits != LimitsV2.for_subject(limits.subject):
        raise JournalError("insufficient protocol allocation")
    cap, _, evaluations, encoders, groups = _PROFILES[limits.subject]
    retained = (evaluations * (14 * 32128 * 4 + 2 * 64 * 512 * 4)
                + encoders * (64 * 512 * 4) + groups * (40 * 512 * 4)
                + METADATA_BYTES + FAILURE_BYTES
                + groups * 4 * 2 * _MAX_VECTOR_BYTES)
    if retained > cap:
        raise JournalError("insufficient protocol allocation")
    return retained


class EvidenceJournalV2(EvidenceJournal):
    """The inherited component journal plus exact binary64 vector storage."""

    def __init__(self, directory, metadata, *, limits):
        directory = Path(directory)
        if (type(limits) is not LimitsV2 or type(metadata) is not dict
                or not directory.is_absolute() or not directory.parent.is_dir()):
            raise ValueError("expected an absolute new attempt in an existing trusted parent")
        if shutil.disk_usage(directory.parent)[2] < limits.free_bytes:
            raise JournalError("insufficient free space before model loading")
        raw = _encoded(metadata)
        if len(raw) > limits.metadata_bytes or len(raw) + limits.failure_bytes > limits.total_bytes:
            raise JournalError("initial metadata exceeds the retained budget")
        directory.mkdir()
        self.directory = directory
        self.limits = limits
        self._thread = threading.current_thread()
        self._owner_local = threading.local()
        self._owner_token = object()
        self._owner_local.token = self._owner_token
        self._used = self._metadata = self._tensor_index = self.component_count = 0
        self._vector_index = 0
        self._active = None
        self._component_bytes = self._component_limit = 0
        self._faulted = self._finished = False
        self._faulted = True
        (directory / "tensors").mkdir()
        (directory / "vectors").mkdir()
        _write_new(directory / "started.json", raw)
        _write_new(directory / "events.jsonl", b"")
        self._used = self._metadata = len(raw)
        self._faulted = False

    def vector64(self, raw):
        """Store exact bits on a completed write; uncertain writes remain faulted."""
        self._check()
        if (self._active is not None or type(raw) is not bytes or not raw
                or len(raw) > _MAX_VECTOR_BYTES or len(raw) % 8):
            raise JournalError("invalid bounded binary64 vector")
        self._space(payload=len(raw))
        path = f"vectors/{self._vector_index:06d}.f64le"
        ref = {"path": path, "shape": [len(raw) // 8], "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest(), "dtype": "float64-le"}
        self._faulted = True
        _write_new(self.directory / path, raw)
        self._used += len(raw)
        self._vector_index += 1
        self._faulted = False
        return ref

    def read_vector64(self, ref):
        self._check()
        if (type(ref) is not dict or set(ref) != {"path", "shape", "bytes", "sha256", "dtype"}
                or type(ref["path"]) is not str or not ref["path"].startswith("vectors/")
                or ref["path"].count("/") != 1 or not ref["path"].endswith(".f64le")
                or not ref["path"][8:-6].isdigit() or type(ref["shape"]) is not list
                or len(ref["shape"]) != 1 or type(ref["shape"][0]) is not int
                or not 1 <= ref["shape"][0] <= 512 or type(ref["bytes"]) is not int
                or ref["bytes"] != 8 * ref["shape"][0] or ref["dtype"] != "float64-le"
                or type(ref["sha256"]) is not str):
            raise JournalError("invalid saved binary64 reference")
        path = self.directory / ref["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != ref["bytes"]:
            raise JournalError("saved vector is absent, aliased or changed")
        with path.open("rb") as stream:
            raw = stream.read(ref["bytes"] + 1)
        if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
            raise JournalError("saved vector bytes changed")
        return raw
