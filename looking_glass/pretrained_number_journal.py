"""Private, fail-stop discovery-pair journal for a trusted CPU custodian.

All files, receipts, evidence, and exception objects are private until the
protocol permits reveal. This component neither runs a model nor establishes
publication order, exclusive execution, stable-media durability, or a study
result. The caller must independently observe and retain every receipt.
"""
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import struct
import threading

from .pretrained_number_archive import _MAX_RECORD_BYTES, _ROLES, _read
from .pretrained_number_commitments import verify_site_reveal
from .pretrained_number_requests import family_request_schedule, load_family_request_plan, verify_request
from .pretrained_number_selection import CHECKPOINT_REVISION, PROTOCOL_SHA256
from .pretrained_number_witness import _MAX_PRIVATE_RECORD_BYTES, audit_private_forward, decode_private_forward


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FRAME = struct.Struct(">8sHII32s32s")
_ACK = struct.Struct(">8sH32s")
_PARTIAL = struct.Struct(">8sHI32s32s")
_FRAME_MAGIC = b"MLGNJ001"
_ACK_MAGIC = b"MLGNA001"
_PARTIAL_MAGIC = b"MLGNP001"
_FOOTER = b"MLGNEND1"
_FRAME_DOMAIN = b"MLG-number-private-pair-v1\0"
_PARTIAL_DOMAIN = b"MLG-number-private-partial-v1\0"
_BINDING_DOMAIN = b"MLG-number-journal-binding-v1\0"
_MAX_FORWARD_BYTES = 1024 * 1024
_TERMINAL_RESERVE = 4096
_COUNT = 8 * len(_ROLES)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _hex64(value):
    if type(value) is not str or not _HEX64.fullmatch(value):
        raise ValueError("a full lowercase SHA-256 is required")


def _digest_parts(*parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.digest()


def _bind(plan_bytes, expected_source_commit, expected_plan_sha256, runtime_sha256,
          site_nonce, site_record_bytes, site_commitment):
    plan = load_family_request_plan(plan_bytes, expected_source_commit=expected_source_commit,
                                    expected_sha256=expected_plan_sha256)
    _hex64(runtime_sha256)
    layer = verify_site_reveal(site_nonce, site_record_bytes, site_commitment,
                               expected_source_commit=expected_source_commit)
    if hmac.compare_digest(site_nonce, bytes.fromhex(json.loads(plan.record_bytes)["public_entropy"])):
        raise ValueError("the private site nonce must differ from public request entropy")
    binding = _canonical({"schema": "mlg-number-private-journal-binding-v1",
                          "source_commit": expected_source_commit,
                          "plan_sha256": expected_plan_sha256, "runtime_sha256": runtime_sha256,
                          "protocol_sha256": PROTOCOL_SHA256, "checkpoint_revision": CHECKPOINT_REVISION,
                          "site_commitment": site_commitment})
    return plan, layer, hashlib.sha256(_BINDING_DOMAIN + binding).digest()


def _frame(ordinal, public, private, binding, previous):
    prefix = _FRAME.pack(_FRAME_MAGIC, ordinal, len(public), len(private), binding, previous)
    digest = _digest_parts(_FRAME_DOMAIN, prefix, public, private)
    return (prefix, public, private, digest, _FOOTER), digest


def _write_parts(path, parts):
    """Exclusive create, full writes, fsync and close, preserving failures."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    try:
        for part in parts:
            remaining = memoryview(part)
            while remaining:
                count = os.write(fd, remaining)
                if type(count) is not int or count <= 0 or count > len(remaining):
                    raise OSError("short or invalid journal write")
                remaining = remaining[count:]
        os.fsync(fd)
    except BaseException as primary:
        try:
            os.close(fd)
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
    else:
        os.close(fd)


def _parse_frame(raw, ordinal, binding, previous):
    if len(raw) < _FRAME.size + 32 + len(_FOOTER):
        raise ValueError("private frame is incomplete")
    magic, number, public_size, private_size, found_binding, found_previous = _FRAME.unpack_from(raw)
    if (magic != _FRAME_MAGIC or number != ordinal or found_binding != binding or found_previous != previous or
            not 0 < public_size <= _MAX_RECORD_BYTES or not 0 < private_size <= _MAX_PRIVATE_RECORD_BYTES or
            public_size + private_size > _MAX_FORWARD_BYTES or
            len(raw) != _FRAME.size + public_size + private_size + 32 + len(_FOOTER) or
            raw[-len(_FOOTER):] != _FOOTER):
        raise ValueError("private frame identity, bounds, or length is invalid")
    end = _FRAME.size + public_size
    public, private = raw[_FRAME.size:end], raw[end:end + private_size]
    expected = _digest_parts(_FRAME_DOMAIN, raw[:_FRAME.size], public, private)
    if not hmac.compare_digest(expected, raw[end + private_size:end + private_size + 32]):
        raise ValueError("private frame checksum is invalid")
    return public, private, expected


def _audit_pair(public, private, ordinal, plan, source_commit, plan_sha256, site, baselines):
    index, role, observation = _read(public, "cpu")
    expected_index, expected_role = family_request_schedule()[ordinal]
    if index != expected_index or role != expected_role:
        raise ValueError("public record is out of planned order")
    verify_request(plan, index, role, observation.request_id, observation.input_sha256,
                   expected_source_commit=source_commit, expected_sha256=plan_sha256)
    record = decode_private_forward(private)
    if ((record.case_index, record.role, record.request_id, record.input_sha256, record.condition) !=
            (index, role, observation.request_id, observation.input_sha256, observation.condition)):
        raise ValueError("private and public record identities differ")
    state = dict(baselines)
    if role == "clean" and index in state:
        raise ValueError("a family baseline was already established")
    clean, injected = state.get(index, (None, None))
    audit_private_forward(index, role, observation, record.evidence, site=site, clean=clean, injected=injected)
    if role == "clean":
        state[index] = (observation, None)
    elif role == "injected":
        state[index] = (clean, observation)
    elif role == "verify_clean":
        del state[index]
    return state


@dataclass(frozen=True, repr=False)
class PairReceipt:
    """Private acknowledgment; the supervisor must retain it independently."""
    ordinal: int
    frame_sha256: str
    public_sha256: str


@dataclass(frozen=True, repr=False)
class PrivatePair:
    public: bytes = field(repr=False)
    private: bytes = field(repr=False)


@dataclass(frozen=True, repr=False)
class JournalInspection:
    status: str
    tail: str
    verified_pairs: tuple
    tail_pair: PrivatePair | None = field(repr=False)
    partial_evidence: bytes | None = field(repr=False)
    terminal_present: bool


@dataclass(frozen=True)
class JournalReservation:
    """Conservative, no-refund reservations for this journal only."""
    disk_bytes: int
    payload_bytes: int
    acknowledged_pairs: int
    failed: bool


class DiscoveryJournal:
    """Single-owner, non-resumable journal; no model or publication authority."""
    def __init__(self, directory, *, plan_bytes, expected_source_commit, expected_plan_sha256,
                 runtime_sha256, site_nonce, site_record_bytes, site_commitment,
                 max_disk_bytes, max_retained_payload_bytes):
        plan, site, binding = _bind(plan_bytes, expected_source_commit, expected_plan_sha256, runtime_sha256,
                                    site_nonce, site_record_bytes, site_commitment)
        if (type(max_disk_bytes) is not int or max_disk_bytes < _TERMINAL_RESERVE or
                type(max_retained_payload_bytes) is not int or max_retained_payload_bytes <= 0):
            raise ValueError("positive disk and retained-payload limits are required")
        self._directory = Path(directory)
        if not self._directory.is_absolute():
            raise ValueError("an absolute trusted attempt directory is required")
        self._directory.mkdir(mode=0o700)  # One new trusted directory per attempt.
        self._owner = threading.current_thread()
        self._plan, self._site, self._binding = plan, site, binding
        self._source_commit, self._plan_sha256 = expected_source_commit, expected_plan_sha256
        self._max_disk, self._max_payload = max_disk_bytes, max_retained_payload_bytes
        self._reserved_disk, self._reserved_payload = _TERMINAL_RESERVE, 0
        self._ordinal, self._previous = 0, bytes(32)
        self._baselines = {}
        self._failed, self._busy = False, False

    def _check_owner(self):
        if threading.current_thread() is not self._owner:
            raise RuntimeError("journal belongs to another thread")
        if self._failed or self._busy:
            raise RuntimeError("journal is failed or busy")

    def _reserve(self, disk_bytes, payload_bytes):
        if (self._reserved_disk + disk_bytes > self._max_disk or
                self._reserved_payload + payload_bytes > self._max_payload):
            raise ValueError("journal reservation exceeds its configured budget")
        self._reserved_disk += disk_bytes
        self._reserved_payload += payload_bytes

    def _terminal(self, phase, cancelled, staged):
        record = _canonical({"schema": "mlg-number-private-terminal-v1", "ordinal": self._ordinal,
                             "status": "cancelled" if cancelled else "failed", "phase": phase,
                             "evidence": "staged-unverified" if staged else "partial-or-unavailable",
                             "exception_details": "omitted-or-possibly-truncated"})
        if len(record) > _TERMINAL_RESERVE:
            raise RuntimeError("terminal record exceeds reserved bytes")
        _write_parts(self._directory / "terminal.json", (record,))

    def reservation(self):
        if threading.current_thread() is not self._owner or self._busy:
            raise RuntimeError("journal is busy or belongs to another thread")
        return JournalReservation(self._reserved_disk, self._reserved_payload, self._ordinal, self._failed)

    def append(self, public, private):
        """Stage, audit and acknowledge the next pair, or fail permanently."""
        self._check_owner()
        self._busy = True
        phase, staged = "preflight", False
        try:
            if self._ordinal >= _COUNT:
                raise ValueError("all planned observations are already present")
            if (type(public) is not bytes or type(private) is not bytes or
                    not 0 < len(public) <= _MAX_RECORD_BYTES or not 0 < len(private) <= _MAX_PRIVATE_RECORD_BYTES or
                    len(public) + len(private) > _MAX_FORWARD_BYTES):
                raise ValueError("bounded immutable pair bytes are required")
            parts, digest = _frame(self._ordinal, public, private, self._binding, self._previous)
            ack_raw = _ACK.pack(_ACK_MAGIC, self._ordinal, digest)
            # Encoded lengths conservatively exceed the represented raw tensor
            # bytes; reserve attempted frames without deduplication or refunds.
            self._reserve(sum(map(len, parts)) + len(ack_raw), len(public) + len(private))
            phase = "stage"
            _write_parts(self._directory / f"{self._ordinal:03d}.frame", parts)
            staged = True
            phase = "audit"
            baselines = _audit_pair(public, private, self._ordinal, self._plan, self._source_commit,
                                    self._plan_sha256, self._site, self._baselines)
            phase = "acknowledge"
            _write_parts(self._directory / f"{self._ordinal:03d}.ok", (ack_raw,))
            receipt = PairReceipt(self._ordinal, digest.hex(), hashlib.sha256(public).hexdigest())
            self._previous, self._baselines = digest, baselines
            self._ordinal += 1
            return receipt
        except BaseException as primary:
            self._failed = True
            try:
                self._terminal(phase, isinstance(primary, (KeyboardInterrupt, SystemExit)), staged)
            except BaseException as secondary:
                if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                    raise secondary from primary
                raise primary from secondary
            raise
        finally:
            self._busy = False

    def abort(self, *, phase, cancelled=False, partial_evidence=None):
        """Stop after an external failure, preserving bounded opaque evidence.

        Partial evidence is private caller-supplied bytes, not a complete
        forward and not a claim that failed plan captures were available.
        """
        self._check_owner()
        self._busy = True
        self._failed = True
        staged = False
        terminal_attempted = False
        try:
            if type(phase) is not str or phase not in ("forward", "cleanup", "external", "resource") or type(cancelled) is not bool:
                raise ValueError("a sanitized abort phase and status are required")
            if partial_evidence is not None:
                if type(partial_evidence) is not bytes or not 0 < len(partial_evidence) <= _MAX_FORWARD_BYTES:
                    raise ValueError("partial evidence must be bounded immutable bytes")
                header = _PARTIAL.pack(_PARTIAL_MAGIC, self._ordinal, len(partial_evidence), self._binding, self._previous)
                digest = _digest_parts(_PARTIAL_DOMAIN, header, partial_evidence)
                parts = (header, partial_evidence, digest, _FOOTER)
                self._reserve(sum(map(len, parts)), len(partial_evidence))
                _write_parts(self._directory / f"{self._ordinal:03d}.partial", parts)
                staged = True
            terminal_attempted = True
            self._terminal(phase, cancelled, staged)
        except BaseException as primary:
            # A partial-file write failure can still leave recoverable bytes.
            # A failed terminal write itself is not retried.
            if not terminal_attempted:
                try:
                    safe_phase = phase if type(phase) is str and phase in ("forward", "cleanup", "external", "resource") else "external"
                    self._terminal(safe_phase, cancelled is True or isinstance(primary, (KeyboardInterrupt, SystemExit)), staged)
                except BaseException as secondary:
                    if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                        raise secondary from primary
                    raise primary from secondary
            raise
        finally:
            self._busy = False


def _read_bounded(path, limit):
    with open(path, "rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("journal file exceeds its read limit")
    return raw


def _parse_partial(raw, ordinal, binding, previous):
    if len(raw) < _PARTIAL.size + 32 + len(_FOOTER):
        raise ValueError("partial evidence file is incomplete")
    magic, number, size, found_binding, found_previous = _PARTIAL.unpack_from(raw)
    if (magic != _PARTIAL_MAGIC or number != ordinal or found_binding != binding or found_previous != previous or
            not 0 < size <= _MAX_FORWARD_BYTES or len(raw) != _PARTIAL.size + size + 32 + len(_FOOTER) or
            raw[-len(_FOOTER):] != _FOOTER):
        raise ValueError("partial evidence identity or size is invalid")
    evidence = raw[_PARTIAL.size:_PARTIAL.size + size]
    digest = _digest_parts(_PARTIAL_DOMAIN, raw[:_PARTIAL.size], evidence)
    if not hmac.compare_digest(digest, raw[_PARTIAL.size + size:_PARTIAL.size + size + 32]):
        raise ValueError("partial evidence checksum is invalid")
    return evidence


def _valid_inventory(path):
    allowed = {"terminal.json"} | {f"{i:03d}.{suffix}" for i in range(_COUNT)
                                   for suffix in ("frame", "ok", "partial")}
    count = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                count += 1
                if count > 2 * _COUNT + 2 or entry.name not in allowed or not entry.is_file(follow_symlinks=False):
                    return False
    except OSError:
        return False
    return True


def inspect_discovery_journal(directory, *, plan_bytes, expected_source_commit, expected_plan_sha256,
                              runtime_sha256, site_nonce, site_record_bytes, site_commitment,
                              acknowledgments, max_read_bytes, externally_quiescent=False):
    """Privately inspect a bounded prefix after independently verified quiescence."""
    if externally_quiescent is not True:
        raise ValueError("external quiescence is required")
    if (type(acknowledgments) is not tuple or len(acknowledgments) > _COUNT or
            type(max_read_bytes) is not int or max_read_bytes <= 0):
        raise ValueError("bounded immutable acknowledgments and read limit are required")
    for ordinal, item in enumerate(acknowledgments):
        if (type(item) is not PairReceipt or type(item.ordinal) is not int or item.ordinal != ordinal):
            raise ValueError("acknowledgments must form the exact ordered prefix")
        _hex64(item.frame_sha256)
        _hex64(item.public_sha256)
    plan, site, binding = _bind(plan_bytes, expected_source_commit, expected_plan_sha256, runtime_sha256,
                                site_nonce, site_record_bytes, site_commitment)
    path = Path(directory)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("journal directory is unavailable")
    if not _valid_inventory(path):
        return JournalInspection("incomplete", "invalid", (), None, None, (path / "terminal.json").exists())
    previous, baselines = bytes(32), {}
    pairs, consumed, tail, tail_pair, partial_evidence = [], 0, "empty", None, None
    for ordinal in range(_COUNT):
        frame_path = path / f"{ordinal:03d}.frame"
        ack_path = path / f"{ordinal:03d}.ok"
        partial_path = path / f"{ordinal:03d}.partial"
        if not frame_path.exists():
            if ack_path.exists() or ordinal < len(acknowledgments):
                tail = "invalid"
            elif partial_path.exists():
                maximum_partial = _PARTIAL.size + _MAX_FORWARD_BYTES + 32 + len(_FOOTER)
                limit = min(maximum_partial, max_read_bytes - consumed)
                if limit <= 0:
                    tail = "capacity"
                else:
                    try:
                        partial_raw = _read_bounded(partial_path, limit)
                    except (ValueError, OSError):
                        tail = "capacity" if limit < maximum_partial else "invalid"
                    else:
                        try:
                            partial_evidence = _parse_partial(partial_raw, ordinal, binding, previous)
                            tail = "partial"
                        except ValueError:
                            tail = "invalid"
            break
        if partial_path.exists():
            tail = "invalid"
            break
        maximum_frame = _FRAME.size + _MAX_FORWARD_BYTES + 32 + len(_FOOTER)
        limit = min(maximum_frame, max_read_bytes - consumed)
        if limit <= 0:
            tail = "capacity"
            break
        try:
            raw = _read_bounded(frame_path, limit)
            consumed += len(raw)
        except (ValueError, OSError):
            tail = "capacity" if limit < maximum_frame else "invalid"
            break
        try:
            public, private, digest = _parse_frame(raw, ordinal, binding, previous)
        except ValueError:
            tail = "invalid"
            break
        if ordinal >= len(acknowledgments):
            tail, tail_pair = "unacknowledged", PrivatePair(public, private)
            break
        if max_read_bytes - consumed < _ACK.size:
            tail = "capacity"
            break
        try:
            ack_raw = _read_bounded(ack_path, _ACK.size)
            consumed += len(ack_raw)
            receipt = acknowledgments[ordinal]
            if (ack_raw != _ACK.pack(_ACK_MAGIC, ordinal, digest) or
                    not hmac.compare_digest(receipt.frame_sha256, digest.hex()) or
                    not hmac.compare_digest(receipt.public_sha256, hashlib.sha256(public).hexdigest())):
                raise ValueError("acknowledgment does not match the frame")
            baselines = _audit_pair(public, private, ordinal, plan, expected_source_commit,
                                    expected_plan_sha256, site, baselines)
        except (ValueError, TypeError, OSError):
            tail, tail_pair = "invalid", PrivatePair(public, private)
            break
        pairs.append(PrivatePair(public, private))
        previous = digest
    terminal = (path / "terminal.json").exists()
    if tail == "empty":
        # A later slot after a gap is an invalid non-sequential attempt.
        for ordinal in range(len(pairs) + 1, _COUNT):
            if ((path / f"{ordinal:03d}.frame").exists() or (path / f"{ordinal:03d}.ok").exists() or
                    (path / f"{ordinal:03d}.partial").exists()):
                tail = "invalid"
                break
    if not _valid_inventory(path):
        tail = "invalid"
    status = "complete-prefix" if len(pairs) == _COUNT and tail == "empty" and not terminal and not baselines else "incomplete"
    return JournalInspection(status, tail, tuple(pairs), tail_pair, partial_evidence, terminal)
