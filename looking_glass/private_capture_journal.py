"""Private, bounded CPU capture frames for failed-forward diagnostics.

This trusted prototype is not an IPC service, broker, GPU path, confidentiality
boundary or attestation that a model forward completed. See its protocol note.
"""

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from threading import RLock, current_thread

import torch

from .intervention_plan import ActivationPlan


_MAGIC = b"MLGPCJ1\0"
_PREFIX = struct.Struct("<IQ")
_FOOTER = b"END!"
_FORMAT = "private-cpu-capture-journal-v1"
_FLOAT_BYTES = {str(torch.float16): 2, str(torch.bfloat16): 2,
                str(torch.float32): 4, str(torch.float64): 8}


class CaptureJournalFull(RuntimeError):
    """A frame would exceed the private journal's reserved limits."""


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _encode_bounded(value, limit):
    """Stop serializing once the metadata byte budget would be exceeded."""
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False)
    result = bytearray()
    for part in encoder.iterencode(value):
        if len(part) > limit - len(result):
            raise CaptureJournalFull("capture metadata exceeds its bound")
        result.extend(part.encode("ascii"))
    return bytes(result)


def _cpu_bytes(value):
    """Copy a validated CPU tensor without requiring an optional NumPy install."""
    count = value.numel() * value.element_size()
    if not count:
        return b""
    packed = value.detach().contiguous()
    pointer = packed.data_ptr()
    if type(pointer) is not int or pointer <= 0:
        raise ValueError("nonempty capture has no CPU data pointer")
    # Keep `packed` strongly referenced through the synchronous native copy.
    return ctypes.string_at(pointer, count)


def _validate_identity(identity):
    keys = {"case_id", "input_sha256", "model_sha256", "plan_sha256", "byte_order"}
    if type(identity) is not dict or set(identity) != keys:
        raise ValueError("identity must contain the exact case, digest and byte-order fields")
    case_id = identity["case_id"]
    if type(case_id) is not str or not case_id.strip() or len(case_id) > 128:
        raise ValueError("case_id must be a bounded nonempty string")
    for key in ("input_sha256", "model_sha256", "plan_sha256"):
        value = identity[key]
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("identity digests must be lowercase SHA-256 hex")
    if type(identity["byte_order"]) is not str or identity["byte_order"] not in ("little", "big"):
        raise ValueError("invalid identity byte order")


def _valid_event(event, element_size, payload_size, previous_order):
    keys = {"name", "operation", "order", "module_path", "module_aliases", "call_site_known",
            "forward_index", "invocation", "batch_index", "token_index", "output_selector",
            "output_shape", "selected_shape", "dtype", "device", "source_capture",
            "preceding_same_site_actions"}
    if type(event) is not dict or set(event) != keys:
        return False
    name, path = event["name"], event["module_path"]
    aliases, preceding = event["module_aliases"], event["preceding_same_site_actions"]
    order, invocation = event["order"], event["invocation"]
    batch, token = event["batch_index"], event["token_index"]
    output_shape, selected_shape = event["output_shape"], event["selected_shape"]
    selector = event["output_selector"]
    if (type(name) is not str or not name or len(name) > 128
            or type(path) is not str or len(path) > 4096
            or type(aliases) is not list or any(type(item) is not str or len(item) > 4096 for item in aliases)
            or path not in aliases
            or type(preceding) is not list or any(type(item) is not str or not item or len(item) > 128 for item in preceding)
            or type(order) is not int or order <= previous_order
            or type(invocation) is not int or invocation < 0
            or type(event["forward_index"]) is not int or event["forward_index"] != 0
            or event["call_site_known"] is not False or event["source_capture"] is not None
            or event["operation"] != "capture" or event["device"] != "cpu"
            or type(event["dtype"]) is not str or event["dtype"] not in _FLOAT_BYTES
            or type(element_size) is not int or element_size != _FLOAT_BYTES[event["dtype"]]
            or type(output_shape) is not list or type(selected_shape) is not list
            or any(type(n) is not int or n < 0 for n in output_shape + selected_shape)
            or type(selector) is not list or (selector != [] and selector != [0])
            or any(type(index) is not int for index in selector)):
        return False
    if batch is None and token is None:
        if selected_shape != output_shape:
            return False
    elif (type(batch) is not int or type(token) is not int or batch < 0 or token < 0
          or len(output_shape) != 3 or batch >= output_shape[0] or token >= output_shape[1]
          or selected_shape != output_shape[2:]):
        return False
    return math.prod(selected_shape) * element_size == payload_size


class PrivateCaptureJournal:
    """Own one exclusively created file for one trusted CPU activation plan."""

    def __init__(self, path, *, case_id, input_sha256, model_sha256, plan_sha256,
                 max_journal_bytes, max_records, max_metadata_bytes, max_workspace_bytes):
        for name, value in (("max_journal_bytes", max_journal_bytes),
                            ("max_records", max_records),
                            ("max_metadata_bytes", max_metadata_bytes),
                            ("max_workspace_bytes", max_workspace_bytes)):
            _positive(value, name)
        self.max_journal_bytes = max_journal_bytes
        self.max_records = max_records
        self.max_metadata_bytes = max_metadata_bytes
        self.max_workspace_bytes = max_workspace_bytes
        self._lock = RLock()
        self._plan = None
        self._entry_token = None
        self._closed = False
        self._failed = False
        self._claimed_bytes = len(_MAGIC)
        self._claimed_records = 0
        self._last_order = -1
        identity = {"case_id": case_id, "input_sha256": input_sha256,
                    "model_sha256": model_sha256, "plan_sha256": plan_sha256,
                    "byte_order": sys.byteorder}
        _validate_identity(identity)
        metadata = _encode_bounded({"kind": "header", "format": _FORMAT, "identity": identity},
                                   self.max_metadata_bytes)
        self._reserve(len(metadata), 0, capture=False)
        self._stream = open(Path(path), "xb", buffering=0)
        try:
            self._write_all(_MAGIC)
            self._write_frame(metadata, b"")
        except BaseException as error:
            self._failed = True
            try:
                self._stream.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _reserve(self, metadata_bytes, payload_bytes, *, capture):
        if metadata_bytes > self.max_metadata_bytes:
            raise CaptureJournalFull("capture metadata exceeds its bound")
        size = _PREFIX.size + metadata_bytes + payload_bytes + 32 + len(_FOOTER)
        if (self._claimed_bytes + size > self.max_journal_bytes
                or (capture and self._claimed_records >= self.max_records)):
            raise CaptureJournalFull("capture journal capacity exceeded")
        self._claimed_bytes += size
        if capture:
            self._claimed_records += 1

    def _write_all(self, data):
        remaining = memoryview(data)
        while remaining:
            written = self._stream.write(remaining)
            if type(written) is not int or written <= 0 or written > len(remaining):
                raise OSError("journal write returned an invalid byte count")
            remaining = remaining[written:]

    def _write_frame(self, metadata, payload):
        prefix = _PREFIX.pack(len(metadata), len(payload))
        digest = hashlib.sha256()
        for part in (prefix, metadata, payload):
            digest.update(part)
            self._write_all(part)
        self._write_all(digest.digest())
        self._write_all(_FOOTER)
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def _bind(self, plan):
        with self._lock:
            if self._closed or self._failed or self._plan is not None:
                raise RuntimeError("journal is unavailable or already bound")
            self._plan = plan

    def _ensure_open(self, plan):
        with self._lock:
            if self._closed or self._failed or self._plan is not plan:
                raise RuntimeError("journal is unavailable for this plan")

    def _begin_entry(self, plan, token):
        with self._lock:
            if self._plan is not plan or self._closed or self._failed or self._entry_token is not None:
                raise RuntimeError("journal is unavailable or plan entry is in progress")
            self._entry_token = token

    def _end_entry(self, token):
        with self._lock:
            if self._entry_token is token:
                self._entry_token = None

    def _append(self, plan, value, event):
        payload = None
        with self._lock:
            try:
                self._ensure_open(plan)
                if current_thread() is not plan._owner_thread or not plan._in_forward:
                    raise RuntimeError("journal capture requires the active owner thread")
                if (type(value) is not torch.Tensor or value.device.type != "cpu"
                        or value.layout != torch.strided or value.is_nested or value.is_quantized
                        or str(value.dtype) not in _FLOAT_BYTES):
                    raise ValueError("journal captures require plain strided CPU tensors")
                nbytes = value.numel() * value.element_size()
                if 2 * nbytes > self.max_workspace_bytes:
                    raise CaptureJournalFull("capture copy workspace exceeds its bound")
                if not _valid_event(event, value.element_size(), nbytes, self._last_order):
                    raise ValueError("capture event metadata is invalid")
                sequence = self._claimed_records
                metadata = _encode_bounded({"kind": "capture", "sequence": sequence,
                                            "element_size": value.element_size(), "event": event},
                                           self.max_metadata_bytes)
                self._reserve(len(metadata), nbytes, capture=True)
                payload = _cpu_bytes(value)
                if len(payload) != nbytes:
                    raise RuntimeError("capture copy has an unexpected byte count")
                self._write_frame(metadata, payload)
                self._last_order = event["order"]
            except BaseException:
                self._failed = True
                raise
            finally:
                value = payload = None

    def close(self):
        """Close only after the plan and all its known hooks are quiescent."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("journal work is still active")
        try:
            if self._closed:
                return
            plan = self._plan
            if self._entry_token is not None or (plan is not None and (plan._in_forward or plan._entering or plan._entered or plan._closing
                                     or plan._registration_uncertain or any(plan._handles)
                                     or plan._lease._held)):
                raise RuntimeError("plan or hook cleanup may still be active")
            try:
                self._stream.close()
            except BaseException:
                self._failed = True
                raise
            self._closed = True
            self._plan = None
        finally:
            self._lock.release()


class JournaledActivationPlan(ActivationPlan):
    """Opt-in private journal; ordinary plan captures still publish atomically."""

    def __init__(self, model, steps, *, journal, **kwargs):
        if type(journal) is not PrivateCaptureJournal:
            raise TypeError("journal must be a PrivateCaptureJournal")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        for tensor in (*model.parameters(), *model.buffers()):
            if tensor.device.type != "cpu":
                raise ValueError("journaled plans require a CPU model")
        if isinstance(steps, (tuple, list)):
            for step in steps:
                if isinstance(step, dict) and step.get("value") is not None:
                    source = step["value"]
                    if type(source) is not torch.Tensor or source.device.type != "cpu":
                        raise ValueError("journaled plans require plain CPU tensor sources")
        super().__init__(model, steps, **kwargs)
        self._journal = journal
        journal._bind(self)

    def __enter__(self):
        token = object()
        first_entry = False
        try:
            self._journal._begin_entry(self, token)
            with self._lifecycle_lock:
                first_entry = not self._used
            self._journal._ensure_open(self)
            entered = super().__enter__()
            self._journal._end_entry(token)
            return entered
        except BaseException as error:
            cleanup_error = None
            if first_entry:
                try:
                    with self._lifecycle_lock:
                        should_abort = self._entered and not self._in_forward
                    if should_abort:
                        self.abort()
                except BaseException as failure:
                    cleanup_error = failure
            try:
                self._journal._end_entry(token)
            except BaseException as failure:
                if cleanup_error is None:
                    cleanup_error = failure
                else:
                    cleanup_error.__cause__ = failure
            if cleanup_error is not None:
                raise error from cleanup_error
            raise

    def _apply_step(self, step, output, preceding):
        result, changed = super()._apply_step(step, output, preceding)
        if step["operation"] == "capture":
            self._journal._append(self, self._captures[step["name"]], self._events[-1])
        return result, changed

    def recover_interrupted_entry(self, *, externally_quiescent=False):
        """Recover only once the entire entry/body stack cannot resume.

        The caller must independently establish both that condition and that no
        forward or hook activity can run. This method cannot infer either fact.
        """
        if externally_quiescent is not True:
            raise RuntimeError("trusted caller must establish quiescence before recovery")
        with self._lifecycle_lock:
            if self._owner_thread is not None and current_thread() is not self._owner_thread:
                raise RuntimeError("only the plan owner can recover an interrupted entry")
            if self._in_forward or self._entering or self._closing:
                raise RuntimeError("plan execution or cleanup may still be active")
            entered = self._entered
        if entered:
            self.abort()
        with self._lifecycle_lock:
            if (self._in_forward or self._entering or self._entered or self._closing
                    or self._registration_uncertain or any(self._handles) or self._lease._held):
                raise RuntimeError("plan hooks or cleanup remain uncertain")
        with self._journal._lock:
            if self._journal._plan is not self or self._journal._closed:
                raise RuntimeError("journal is not bound to this plan")
            had_marker = self._journal._entry_token is not None
            self._journal._entry_token = None
            if entered or had_marker:
                self._journal._failed = True
        return entered or had_marker


def _read_exact(stream, count):
    data = stream.read(count)
    return data if len(data) == count else None


def read_capture_journal(path, *, expected_identity, max_journal_bytes, max_records,
                         max_metadata_bytes, max_read_payload_bytes, externally_quiescent=False):
    """Read a validated prefix only after the trusted caller attests quiescence."""
    for name, value in (("max_journal_bytes", max_journal_bytes),
                        ("max_records", max_records), ("max_metadata_bytes", max_metadata_bytes),
                        ("max_read_payload_bytes", max_read_payload_bytes)):
        _positive(value, name)
    _validate_identity(expected_identity)
    if externally_quiescent is not True:
        raise RuntimeError("trusted caller must establish external quiescence before reading")
    captures, identity, tail = [], None, "clean"
    seen_names = set()
    retained_bytes, previous_order = 0, -1
    with open(Path(path), "rb") as stream:
        if os.fstat(stream.fileno()).st_size > max_journal_bytes:
            raise ValueError("journal exceeds the reader's byte bound")
        if _read_exact(stream, len(_MAGIC)) != _MAGIC:
            raise ValueError("invalid or incomplete journal header")
        while True:
            prefix = stream.read(_PREFIX.size)
            if not prefix:
                break
            if len(prefix) != _PREFIX.size:
                tail = "incomplete"
                break
            metadata_size, payload_size = _PREFIX.unpack(prefix)
            if (metadata_size > max_metadata_bytes or payload_size > max_journal_bytes
                    or _PREFIX.size + metadata_size + payload_size + 32 + len(_FOOTER) > max_journal_bytes):
                tail = "invalid"
                break
            if identity is not None and len(captures) >= max_records:
                tail = "invalid"
                break
            if retained_bytes + payload_size > max_read_payload_bytes:
                tail = "capacity"
                break
            metadata = _read_exact(stream, metadata_size)
            payload = _read_exact(stream, payload_size)
            checksum = _read_exact(stream, 32)
            footer = _read_exact(stream, len(_FOOTER))
            if any(part is None for part in (metadata, payload, checksum, footer)):
                tail = "incomplete"
                break
            hasher = hashlib.sha256()
            for part in (prefix, metadata, payload):
                hasher.update(part)
            digest = hasher.digest()
            if digest != checksum or footer != _FOOTER:
                tail = "invalid"
                break
            try:
                value = json.loads(metadata)
                if _encode_bounded(value, max_metadata_bytes) != metadata:
                    raise ValueError("metadata is not canonical")
            except (ValueError, TypeError, UnicodeDecodeError, RecursionError, CaptureJournalFull):
                tail = "invalid"
                break
            if identity is None:
                if (type(value) is not dict or set(value) != {"kind", "format", "identity"}
                        or value.get("kind") != "header"
                        or value.get("format") != _FORMAT or type(value.get("identity")) is not dict or payload):
                    raise ValueError("journal has no valid identity header")
                identity = value["identity"]
                _validate_identity(identity)
                if _encode(identity) != _encode(expected_identity):
                    raise ValueError("journal identity differs from the trusted expectation")
            else:
                event = value.get("event") if type(value) is dict else None
                size = value.get("element_size") if type(value) is dict else None
                if (type(value) is not dict or set(value) != {"kind", "sequence", "element_size", "event"}
                        or value.get("kind") != "capture"
                        or type(value.get("sequence")) is not int or value.get("sequence") != len(captures)
                        or not _valid_event(event, size, len(payload), previous_order)
                        or event["name"] in seen_names):
                    tail = "invalid"
                    break
                captures.append({"metadata": event, "payload": payload,
                                 "payload_sha256": hashlib.sha256(payload).hexdigest()})
                seen_names.add(event["name"])
                previous_order = event["order"]
                retained_bytes += len(payload)
    if identity is None:
        raise ValueError("journal has no complete identity header")
    return {"identity": identity, "captures": captures, "tail": tail,
            "durability": "not-established-by-file"}
