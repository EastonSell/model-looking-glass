"""Private per-cell evidence records for the proposed synthetic v2 study.

This adapter does not run a model, generate cases, validate causal controls,
enforce phase order, or provide access control. The custodian must do those
jobs and keep these target-identifying records private until reveal.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading

from . import registry


class CellEvidenceError(ValueError):
    """A cell record is incomplete, malformed, or unavailable for writing."""


MAX_RECORD_BYTES = 64 * 1024
MAX_CONDITIONS = 32
# Thirty-two complete conditions and one optional partial record, plus framing.
CELL_CHUNK_LIMIT = (MAX_CONDITIONS + 1) * MAX_RECORD_BYTES + 4096
_RECORD_SCHEMA = "causal-v2-condition-record-v1"
_PARTIAL_SCHEMA = "causal-v2-partial-record-v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = tuple((kind, site) for kind in ("forward", "reverse", "null", "self")
               for site in (1, 2, 3)) + (("upstream", 0),)
_FIELDS = {"schema", "cell_id", "phase", "case_id", "case_index", "condition_id",
           "model_seed", "model_state_sha256", "input_sha256", "strength", "prediction_ref",
           "traces", "logits", "plans", "private_integrity", "comparison"}


def _text(value):
    return type(value) is str and 0 < len(value) <= 256


def _digest(value):
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _parse(raw):
    if type(raw) is not bytes or len(raw) > MAX_RECORD_BYTES:
        raise CellEvidenceError("record is not bounded immutable bytes")
    try:
        value = json.loads(raw)
        if type(value) is not dict or registry._json_bytes(value) != raw:
            raise ValueError("record is not canonical JSON")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise CellEvidenceError("invalid canonical JSON record") from error
    return value


def _tensor(value, width):
    if (type(value) is not dict or set(value) != {"dtype", "device", "shape", "bits"}
            or value["dtype"] != "torch.float32" or value["device"] != "cpu"
            or registry._json_bytes(value["shape"]) != registry._json_bytes([1, width])
            or type(value["bits"]) is not list or len(value["bits"]) != width
            or any(type(bits) is not int or not 0 <= bits <= 0xffffffff
                   or ((bits >> 23) & 0xff) == 0xff for bits in value["bits"])):
        raise CellEvidenceError("tensor must contain finite CPU float32 bit patterns")


def _trace(events):
    if type(events) is not list or len(events) != 4:
        raise CellEvidenceError("sparse trace must have four events")
    run_id = None
    for index, event in enumerate(events):
        if type(event) is not dict or set(event) != {"metadata", "value"}:
            raise CellEvidenceError("invalid sparse event")
        metadata = event["metadata"]
        if type(metadata) is not dict or not _text(metadata.get("run_id")):
            raise CellEvidenceError("invalid sparse event identity")
        if run_id is None:
            run_id = metadata["run_id"]
        expected = {"run_id": run_id, "node_id": f"block.{index}.output",
                    "sequence": index, "occurrence": 0, "shape": [1, 8],
                    "dtype": "torch.float32", "device": "cpu", "tensor_bytes": 32}
        # Canonical bytes distinguish False from 0 and 1.0 from 1.
        if registry._json_bytes(metadata) != registry._json_bytes(expected):
            raise CellEvidenceError("sparse event metadata differs")
        _tensor(event["value"], 8)


def _logits(value):
    keys = {"clean", "repeat_clean", "perturbed", "forward", "reverse", "null", "self",
            "upstream", "post_hook_clean"}
    if type(value) is not dict or set(value) != keys:
        raise CellEvidenceError("full logit inventory is required")
    for name in ("clean", "repeat_clean", "perturbed", "upstream", "post_hook_clean"):
        _tensor(value[name], 2)
    for name in ("forward", "reverse", "null", "self"):
        group = value[name]
        if type(group) is not list or len(group) != 3:
            raise CellEvidenceError("three candidate-site logits are required")
        for item in group:
            _tensor(item, 2)


def _plans(value):
    if type(value) is not list or len(value) != len(_KINDS):
        raise CellEvidenceError("thirteen ordered plan records are required")
    for plan, (kind, site) in zip(value, _KINDS):
        if (type(plan) is not dict
                or set(plan) != {"kind", "site", "metadata", "captures", "source_id", "base_id"}
                or plan["kind"] != kind or type(plan["site"]) is not int or plan["site"] != site
                or type(plan["metadata"]) is not dict or not _text(plan["base_id"])):
            raise CellEvidenceError("invalid ordered plan record")
        source = plan["source_id"]
        if ((kind == "null" and source is not None)
                or (kind == "self" and source != "same_forward_self_source")
                or (kind not in ("null", "self") and not _text(source))):
            raise CellEvidenceError("invalid plan source identity")
        captures = plan["captures"]
        names = {"self_source", "post_action"} if kind == "self" else {"post_action"}
        if type(captures) is not dict or set(captures) != names:
            raise CellEvidenceError("missing plan capture")
        for item in captures.values():
            _tensor(item, 8)


def _private(value, strength):
    keys = {"target", "delta", "clean_target", "perturbed_target", "clean_capture_metadata",
            "perturbed_capture_metadata"}
    if (type(value) is not dict or set(value) != keys or not _text(value["target"])
            or type(value["delta"]) is not float or not math.isfinite(value["delta"])
            or abs(value["delta"]) != strength
            or type(value["clean_capture_metadata"]) is not dict
            or type(value["perturbed_capture_metadata"]) is not dict):
        raise CellEvidenceError("private target integrity evidence is incomplete")
    _tensor(value["clean_target"], 8)
    _tensor(value["perturbed_target"], 8)


class CellEvidence:
    """Buffer bounded complete condition records and publish one private chunk.

    Use from one creating process and thread. A failure poisons further append;
    already accepted records can still be published as incomplete. A publication
    attempt is one-shot even if its acknowledgement fails.
    """
    def __init__(self, writer, *, name, cell_id, phase, model_seed, model_state_sha256):
        try:
            registry._validate_name(name)
        except (ValueError, TypeError) as error:
            raise CellEvidenceError("invalid chunk name") from error
        if (not callable(getattr(writer, "publish_chunk", None)) or not _text(cell_id)
                or phase not in ("discovery", "heldout") or type(phase) is not str
                or type(model_seed) is not int or model_seed < 0 or not _digest(model_state_sha256)):
            raise CellEvidenceError("invalid cell identity")
        self._writer, self._name, self._cell_id = writer, name, cell_id
        self._phase, self._model_seed, self._model_state_sha256 = phase, model_seed, model_state_sha256
        self._owner_pid, self._owner_thread = os.getpid(), threading.current_thread()
        # A single immutable tuple assignment commits both bytes and identity
        # indexes, avoiding a separately updated count or repeated JSON parsing.
        self._records = ()
        self._failed = False
        self._closed = False
        self._partial_unavailable = False

    @property
    def name(self):
        return self._name

    @property
    def cell_id(self):
        return self._cell_id

    @property
    def phase(self):
        return self._phase

    @property
    def model_seed(self):
        return self._model_seed

    @property
    def model_state_sha256(self):
        return self._model_state_sha256

    def __copy__(self):
        raise TypeError("cell evidence buffers cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("cell evidence buffers cannot be copied")

    def __reduce__(self):
        raise TypeError("cell evidence buffers cannot be serialized")

    def _require_owner(self, *, allow_failed=False):
        if (self._closed or (self._failed and not allow_failed) or os.getpid() != self._owner_pid
                or threading.current_thread() is not self._owner_thread):
            raise CellEvidenceError("cell buffer is unavailable; preserve the attempt")

    def append(self, canonical_record_bytes):
        self._require_owner()
        try:
            value = _parse(canonical_record_bytes)
            count = len(self._records)
            index, strength = divmod(count, 2)
            strength = (0.25, 0.75)[strength]
            if (count >= MAX_CONDITIONS or set(value) != _FIELDS or value["schema"] != _RECORD_SCHEMA
                    or value["cell_id"] != self.cell_id or value["phase"] != self.phase
                    or type(value["case_index"]) is not int or value["case_index"] != index
                    or type(value["model_seed"]) is not int or value["model_seed"] != self.model_seed
                    or value["model_state_sha256"] != self.model_state_sha256
                    or not _text(value["case_id"]) or not _text(value["condition_id"])
                    or not _digest(value["input_sha256"])
                    or type(value["strength"]) is not float or value["strength"] != strength
                    or (self.phase == "discovery" and value["prediction_ref"] is not None)
                    or (self.phase == "heldout" and not _text(value["prediction_ref"]))):
                raise CellEvidenceError("condition identity or order differs")
            previous = self._records
            if any(row[3] == value["condition_id"] for row in previous):
                raise CellEvidenceError("condition identity is repeated")
            if count % 2:
                last = previous[-1]
                if last[1] != value["case_id"] or last[2] != value["input_sha256"]:
                    raise CellEvidenceError("strengths do not share the same case")
            elif any(row[1] == value["case_id"] or row[2] == value["input_sha256"]
                     for row in previous):
                raise CellEvidenceError("case identity or input digest is repeated")
            traces = value["traces"]
            if type(traces) is not dict or set(traces) != {"clean", "repeat_clean", "perturbed"}:
                raise CellEvidenceError("paired trace inventory is incomplete")
            for events in traces.values():
                _trace(events)
            _logits(value["logits"])
            _plans(value["plans"])
            _private(value["private_integrity"], strength)
            if type(value["comparison"]) is not dict:
                raise CellEvidenceError("comparison record is required")
            updated = self._records + ((canonical_record_bytes, value["case_id"],
                                        value["input_sha256"], value["condition_id"]),)
            self._records = updated
        except BaseException:
            self._failed = True
            raise

    def _publish(self, state, records):
        # Set before the attempt: a thrown acknowledgement never authorizes retry.
        self._closed = True
        raw = (b'{"name":' + registry._json_bytes(self.name) + b',"records":[' + b",".join(records)
               + b'],"schema":"causal-v2-chunk-v1","state":' + registry._json_bytes(state) + b"}")
        if len(raw) > CELL_CHUNK_LIMIT:
            raise CellEvidenceError("cell chunk exceeds its declared maximum")
        self._writer.publish_chunk(self.name, raw)

    def publish_complete(self):
        self._require_owner()
        if len(self._records) != MAX_CONDITIONS:
            self._failed = True
            raise CellEvidenceError("all 32 ordered conditions are required")
        try:
            self._publish("complete", [row[0] for row in self._records])
        except BaseException:
            self._failed = True
            raise

    def publish_incomplete(self, canonical_partial_bytes=None):
        self._require_owner(allow_failed=True)
        records = [row[0] for row in self._records]
        if self._partial_unavailable:
            if canonical_partial_bytes is not None:
                raise CellEvidenceError("only completed-prefix salvage is available")
            # Explicitly report that the attempted partial record was not saved;
            # the caller must also retain the original error in failure evidence.
            marker = {"schema": "causal-v2-partial-unavailable-v1", "cell_id": self.cell_id,
                      "phase": self.phase, "model_seed": self.model_seed,
                      "reason": "invalid_or_oversize_partial"}
            records.append(registry._json_bytes(marker))
        if canonical_partial_bytes is not None:
            try:
                partial = _parse(canonical_partial_bytes)
                if (set(partial) != {"schema", "cell_id", "phase", "model_seed", "stage", "evidence"}
                        or partial["schema"] != _PARTIAL_SCHEMA or partial["cell_id"] != self.cell_id
                        or partial["phase"] != self.phase or type(partial["model_seed"]) is not int
                        or partial["model_seed"] != self.model_seed or not _text(partial["stage"])
                        or type(partial["evidence"]) is not dict):
                    raise CellEvidenceError("invalid partial evidence identity")
                records = [*records, canonical_partial_bytes]
            except CellEvidenceError:
                # No writer call has begun. Permit one incomplete salvage while
                # making the missing partial explicit in the saved record.
                self._failed = True
                self._partial_unavailable = True
                raise
            except BaseException:
                self._closed = True
                raise
        self._publish("incomplete", records)
