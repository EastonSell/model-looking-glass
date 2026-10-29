"""Validate a complete trusted CPU GPT-2 trace job before batch publication.

This module does not run a model, own tensors, reserve claims, seal results,
or authenticate callers. A trusted owner must keep every source stable and
charged during validation and subsequent staging.
"""
from dataclasses import dataclass

import torch

from .capture_dispatcher import CapturedSlice
from .trace_schema import validate_trace
from .traced_broker_worker import BoundTracePlan, SiteTrace

_MAX_INT = 2**63 - 1
_MAX_SLOTS = 65536
_FINITE_CHUNK_ELEMENTS = 4096
_ERROR_CLEANUP_NODES = 128
_ERROR_CLEANUP_DEPTH = 16
_FACTORY = object()
try:
    _BUILTIN_GROUP_TYPES = (BaseExceptionGroup, ExceptionGroup)
except NameError:  # Python 3.10 has no built-in exception groups.
    _BUILTIN_GROUP_TYPES = ()


def _positive(value, name):
    if type(value) is not int or not 1 <= value <= _MAX_INT:
        raise ValueError(f"{name} must be a positive bounded plain integer")


def _plain_text(value):
    return type(value) is str and len(value) <= 128 and bool(value.strip())


def _exact_match(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is tuple:
        return (len(actual) == len(expected)
                and all(type(left) is type(right) and left == right
                        for left, right in zip(actual, expected)))
    return actual == expected


def _finite_chunks(value):
    flat = value.view(-1)
    for start in range(0, flat.numel(), _FINITE_CHUNK_ELEMENTS):
        chunk = flat.narrow(0, start, min(_FINITE_CHUNK_ELEMENTS, flat.numel() - start))
        if not bool(torch.isfinite(chunk).all().item()):
            return False
    return True


def _strip_failure_frames(error):
    """Best-effort bounded cleanup, not proof that source aliases are gone."""
    pending = [(error, 0)]
    seen = set()
    while pending and len(seen) < _ERROR_CLEANUP_NODES:
        current, depth = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if type(current) in _BUILTIN_GROUP_TYPES and depth < _ERROR_CLEANUP_DEPTH:
            for child in current.exceptions:
                if len(pending) + len(seen) >= _ERROR_CLEANUP_NODES:
                    break
                pending.append((child, depth + 1))
        try:
            current.__traceback__ = None
            current.__cause__ = None
            current.__context__ = None
        except Exception:
            # Uncooperative exception objects remain an owner's custody risk.
            pass


@dataclass(frozen=True, slots=True)
class TraceJobSlot:
    """Expected one-invocation output for one sample and one GPT-2 site."""

    slot_index: int
    batch_position: int
    sample_id: str
    token_positions: tuple[int, ...]
    layer_index: int
    module_kind: str
    module_path: str
    output_path: tuple[int, ...]
    payload_bytes: int


@dataclass(frozen=True, slots=True, init=False)
class TraceJobLayout:
    """Immutable private metadata for one bounded, trusted trace selection."""

    _identity: tuple
    _forward_index: int
    _generation_step: int | None
    _sequence_start: int
    _batch_size: int
    _sequence_length: int
    _hidden_width: int
    _slots: tuple
    _total_bytes: int

    def __init__(self, identity, forward_index, generation_step, sequence_start,
                 batch_size, sequence_length, hidden_width, slots, total_bytes,
                 *, _factory=None):
        if _factory is not _FACTORY:
            raise TypeError("use make_trace_job_layout")
        for name, value in (("_identity", identity), ("_forward_index", forward_index),
                            ("_generation_step", generation_step),
                            ("_sequence_start", sequence_start),
                            ("_batch_size", batch_size),
                            ("_sequence_length", sequence_length),
                            ("_hidden_width", hidden_width), ("_slots", slots),
                            ("_total_bytes", total_bytes)):
            object.__setattr__(self, name, value)

    @property
    def slots(self):
        return tuple(TraceJobSlot(*slot) for slot in self._slots)

    @property
    def total_payload_bytes(self):
        return self._total_bytes

    def validate(self, events):
        """Return tensor-free, sequence-ordered metadata for a complete batch.

        Only exact CapturedSlice objects with finite, contiguous, detached
        float32 CPU payloads are accepted. This does not prove source ownership,
        check external aliases, or preserve tensor values against mutation.
        """
        result = failure = None
        try:
            result = self._validate(events)
        except BaseException as error:
            # Nested schema/PyTorch frames can retain the complete source
            # bundle. Strip their traceback before reraising outside the
            # handler, preserving the failure type (including interruptions).
            failure = error
            try:
                _strip_failure_frames(failure)
            except Exception:
                # A hostile exception subclass is outside this trusted API;
                # the owner must quarantine if it cannot prove disposal.
                pass
        finally:
            events = None
        if failure is not None:
            try:
                raise failure from None
            except BaseException:
                # Python can attach an unrelated caller's active exception
                # as implicit context even with ``from None``.
                try:
                    failure.__context__ = None
                    failure.__cause__ = None
                except Exception:
                    pass
                raise
        return result

    def _validate(self, events):
        if type(events) is not tuple or len(events) != len(self._slots):
            raise ValueError("the entire expected batch must be present")
        expected = {(slot[1], slot[4], slot[5]): slot for slot in self._slots}
        remaining = set(expected)
        metadata = []
        spans = []
        sequences = set()
        run_id, job_id, request_id = self._identity
        for event in events:
            if type(event) is not CapturedSlice:
                raise ValueError("events must be exact captured slices")
            if (type(event.batch_position) is not int or type(event.layer_index) is not int
                    or type(event.module_kind) is not str):
                raise ValueError("event coordinates must be plain values")
            key = (event.batch_position, event.layer_index, event.module_kind)
            if key not in remaining:
                raise ValueError("event slot is missing, duplicated, or unauthorized")
            remaining.remove(key)
            slot = expected[key]
            slot_index, row, sample_id, positions, layer, kind, path, output_path, nbytes = slot
            expected_fields = (
                (event.run_id, run_id), (event.job_id, job_id), (event.request_id, request_id),
                (event.sample_id, sample_id), (event.batch_position, row),
                (event.token_positions, positions), (event.layer_index, layer),
                (event.module_kind, kind), (event.module_path, path),
                (event.output_path, output_path), (event.invocation, 0),
                (event.forward_index, self._forward_index),
                (event.generation_step, self._generation_step),
                (event.source_shape, (self._batch_size, self._sequence_length, self._hidden_width)),
                (event.source_dtype, "torch.float32"), (event.source_device, "cpu"),
                (event.saved_shape, (len(positions), self._hidden_width)),
                (event.saved_dtype, "torch.float32"), (event.saved_device, "cpu"),
            )
            if any(not _exact_match(actual, wanted) for actual, wanted in expected_fields):
                raise ValueError("event metadata does not match its authorized slot")
            if (type(event.sequence) is not int
                    or event.sequence != self._sequence_start + slot_index
                    or event.sequence in sequences):
                raise ValueError("event sequence does not match its expected invocation order")
            sequences.add(event.sequence)
            value = event.value
            if (type(value) is not torch.Tensor or value.device.type != "cpu"
                    or value.layout != torch.strided or value.is_nested or value.is_quantized
                    or value.dtype != torch.float32 or value.requires_grad or value.is_inference()
                    or not value.is_contiguous() or value.storage_offset() != 0
                    or tuple(value.shape) != (len(positions), self._hidden_width)
                    or value.numel() * value.element_size() != nbytes):
                raise ValueError("event payload does not match its authorized slot")
            storage = value.untyped_storage()
            if storage.nbytes() != nbytes:
                raise ValueError("event payload does not span its reported storage")
            start = storage.data_ptr()
            if start == 0:
                raise ValueError("event payload is not materialized")
            spans.append((start, start + nbytes))
            if not _finite_chunks(value):
                raise ValueError("GPT-2 trace values must be finite")
            metadata.append(event.to_trace_event())
        spans.sort()
        if any(previous[1] > current[0] for previous, current in zip(spans, spans[1:])):
            raise ValueError("event tensor storage spans overlap")
        if remaining or len(sequences) != len(self._slots):
            raise ValueError("the entire expected batch must be present")
        validate_trace(metadata)
        return tuple(sorted(metadata, key=lambda event: event.sequence))


def make_trace_job_layout(plan, *, batch_size, sequence_length, hidden_width,
                          layer_count, max_slots, max_event_bytes):
    """Create an exact one-invocation-per-site layout from trusted owner facts."""
    if type(plan) is not BoundTracePlan:
        raise ValueError("expected an exact bound trace plan")
    for name, value in (("batch_size", batch_size), ("sequence_length", sequence_length),
                        ("hidden_width", hidden_width), ("layer_count", layer_count),
                        ("max_slots", max_slots), ("max_event_bytes", max_event_bytes)):
        _positive(value, name)
    if max_slots > _MAX_SLOTS:
        raise ValueError("max_slots exceeds the supported bound")
    # Bound the plan's existing tuple lengths before its accessors allocate
    # reconstructed sample and site objects. The cooperative owner must not
    # mutate private plan fields while the snapshot is being made.
    if (type(plan._samples) is not tuple or type(plan._sites) is not tuple
            or len(plan._samples) != batch_size or not plan._sites
            or len(plan._samples) * len(plan._sites) > max_slots):
        raise ValueError("the plan exceeds the declared batch or slot bound")
    if (type(plan._identity) is not tuple or len(plan._identity) != 4
            or any(type(site) is not tuple or len(site) != 2 for site in plan._sites)
            or any(type(sample) is not tuple or len(sample) != 2 for sample in plan._samples)):
        raise ValueError("the plan snapshot is invalid")
    broker_id, job_number, team_id, external_request_id = plan._identity
    if (type(broker_id) is not str or len(broker_id) != 32
            or any(char not in "0123456789abcdef" for char in broker_id)
            or type(job_number) is not int or not 1 <= job_number <= _MAX_INT
            or not _plain_text(team_id) or not _plain_text(external_request_id)
            or not _plain_text(plan._run_id)):
        raise ValueError("the plan identity is invalid")
    request = plan.capture_request()
    sites = plan.sites
    if len(request.samples) != batch_size or not sites:
        raise ValueError("the plan does not match the declared batch")
    count = len(request.samples) * len(sites)
    starts = plan.sequence_starts()
    if (type(request.max_events) is not int or not 1 <= request.max_events <= _MAX_INT
            or type(request.max_trace_bytes) is not int or not 1 <= request.max_trace_bytes <= _MAX_INT
            or type(request.forward_index) is not int or not 0 <= request.forward_index <= _MAX_INT
            or (request.generation_step is not None
                and (type(request.generation_step) is not int
                     or not 0 <= request.generation_step <= _MAX_INT))
            or set(starts) != {request.run_id} or type(starts[request.run_id]) is not int
            or not 0 <= starts[request.run_id] <= _MAX_INT
            or starts[request.run_id] + count - 1 > _MAX_INT):
        raise ValueError("the plan has an invalid sequence start")
    if count > max_slots or count > request.max_events:
        raise ValueError("the authorized slot count exceeds its bound")
    slots = []
    total_bytes = 0
    seen_samples = set()
    seen_sites = set()
    for site in sites:
        if (type(site) is not SiteTrace or type(site.layer_index) is not int
                or not 0 <= site.layer_index < layer_count
                or type(site.module_kind) is not str
                or site.module_kind not in ("block", "attn", "mlp")
                or (site.layer_index, site.module_kind) in seen_sites):
            raise ValueError("invalid or duplicate GPT-2 trace site")
        seen_sites.add((site.layer_index, site.module_kind))
    for row, sample in enumerate(request.samples):
        if (type(sample.batch_position) is not int or sample.batch_position != row
                or not _plain_text(sample.sample_id)
                or sample.sample_id in seen_samples or type(sample.token_positions) is not tuple
                or not sample.token_positions or len(sample.token_positions) > 128):
            raise ValueError("invalid sample selection")
        seen_samples.add(sample.sample_id)
        if (any(type(index) is not int or not 0 <= index < sequence_length
                for index in sample.token_positions)
                or len(set(sample.token_positions)) != len(sample.token_positions)):
            raise ValueError("invalid token positions")
    order = {"attn": 0, "mlp": 1, "block": 2}
    for site in sorted(sites, key=lambda item: (item.layer_index, order[item.module_kind])):
        for row, sample in enumerate(request.samples):
            size = len(sample.token_positions) * hidden_width * 4
            if size > max_event_bytes:
                raise ValueError("an event exceeds its byte bound")
            path = f"transformer.h.{site.layer_index}"
            if site.module_kind != "block":
                path += f".{site.module_kind}"
            output_path = () if site.module_kind == "mlp" else (0,)
            slots.append((len(slots), row, sample.sample_id, sample.token_positions,
                          site.layer_index, site.module_kind, path, output_path, size))
            total_bytes += size
            if total_bytes > request.max_trace_bytes:
                raise ValueError("the expected payload exceeds the trace byte bound")
    return TraceJobLayout((request.run_id, request.job_id, request.request_id),
                          request.forward_index, request.generation_step, starts[request.run_id],
                          batch_size, sequence_length, hidden_width, tuple(slots), total_bytes,
                          _factory=_FACTORY)
