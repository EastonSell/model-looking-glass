"""Bounded request-local slices from one trusted synchronous forward.

This primitive neither owns model hooks nor schedules GPU work. A trusted
serial model owner must establish and tear down hooks around exactly one
forward; see docs/CAPTURE_DISPATCHER.md for its limits and ownership contract.
"""
import asyncio
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
import inspect
import math
import threading

import torch

from .trace_schema import GenerationContext, RunIdentity, TraceEvent


class CaptureLimitError(RuntimeError):
    """The next complete batch event would exceed a declared limit."""


class CaptureAbortedError(RuntimeError):
    """A caught capture failure made the whole forward unusable."""


@dataclass(frozen=True, slots=True)
class SampleSelection:
    batch_position: int
    sample_id: str
    token_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    run_id: str
    job_id: str
    request_id: str
    samples: tuple[SampleSelection, ...]
    forward_index: int
    generation_step: int | None
    max_events: int
    max_trace_bytes: int


@dataclass(frozen=True, slots=True)
class CapturedSlice:
    """Immutable metadata and an owned, mutable detached tensor payload."""

    run_id: str
    job_id: str
    request_id: str
    sample_id: str
    batch_position: int
    token_positions: tuple[int, ...]
    forward_index: int
    generation_step: int | None
    sequence: int
    module_path: str
    layer_index: int | None
    module_kind: str
    output_path: tuple[int | str, ...] | None
    invocation: int
    source_shape: tuple[int, ...]
    source_dtype: str
    source_device: str
    saved_shape: tuple[int, ...]
    saved_dtype: str
    saved_device: str
    value: torch.Tensor

    @property
    def tensor_bytes(self):
        return self.value.numel() * self.value.element_size()

    def to_trace_event(self):
        """Build validated metadata; tensor payload remains separately owned."""
        if (tuple(self.value.shape) != self.saved_shape
                or str(self.value.dtype) != self.saved_dtype
                or str(self.value.device) != self.saved_device):
            raise ValueError("the owned payload no longer matches its capture metadata")
        return TraceEvent(
            identity=RunIdentity(run_id=self.run_id, job_id=self.job_id,
                                 request_id=self.request_id, sample_id=self.sample_id),
            sequence=self.sequence, forward_index=self.forward_index,
            module_path=self.module_path, invocation=self.invocation,
            shape=self.source_shape, dtype=self.source_dtype, device=self.source_device,
            capture_mode="direct", module_kind=self.module_kind, layer_index=self.layer_index,
            output_path=self.output_path, batch_positions=(self.batch_position,),
            token_positions=self.token_positions, batch_axis=0, token_axis=1,
            generation=GenerationContext(generation_step=self.generation_step),
            saved_shape=self.saved_shape, saved_dtype=self.saved_dtype,
            saved_device=self.saved_device,
        )


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _text(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


# Prevent overlapping dispatcher executions within this process. This is not
# a cross-process GPU/model lock; the model owner still needs one of its own.
_EXECUTION_LOCK = threading.Lock()


class CaptureDispatcher:
    """Split one normalized hook output into bounded, request-owned slices.

    Every batch row must be assigned once. The owner alone installs hooks and
    calls record inside run; this object cannot prove hook/model ownership.
    Payload caps exclude metadata, temporary copies, allocator overhead, model
    allocations, and any results retained by prior callers.
    """

    def __init__(self, requests, *, batch_size, sequence_length, max_events,
                 max_trace_bytes, capture_device="cpu", sequence_starts=None):
        _positive("batch_size", batch_size)
        _positive("sequence_length", sequence_length)
        _positive("max_events", max_events)
        _positive("max_trace_bytes", max_trace_bytes)
        if capture_device not in ("cpu", "source"):
            raise ValueError("capture_device must be cpu or source")
        if not isinstance(requests, (list, tuple)) or not requests:
            raise ValueError("requests must be a nonempty list or tuple")
        requests = tuple(requests)
        rows = set()
        request_ids = set()
        run_ids = set()
        run_forwards = {}
        for request in requests:
            if not isinstance(request, CaptureRequest):
                raise ValueError("each request must be a CaptureRequest")
            for name in ("run_id", "job_id", "request_id"):
                _text(name, getattr(request, name))
            if request.request_id in request_ids:
                raise ValueError("request IDs must be distinct within a batch")
            request_ids.add(request.request_id)
            run_ids.add(request.run_id)
            _nonnegative("forward_index", request.forward_index)
            if request.run_id in run_forwards and run_forwards[request.run_id] != request.forward_index:
                raise ValueError("one run ID must identify the same forward within a batch")
            run_forwards[request.run_id] = request.forward_index
            if request.generation_step is not None:
                _nonnegative("generation_step", request.generation_step)
            _positive("request max_events", request.max_events)
            _positive("request max_trace_bytes", request.max_trace_bytes)
            if type(request.samples) is not tuple or not request.samples:
                raise ValueError("request samples must be a nonempty tuple")
            sample_ids = set()
            for sample in request.samples:
                if not isinstance(sample, SampleSelection):
                    raise ValueError("each sample must be a SampleSelection")
                _nonnegative("batch_position", sample.batch_position)
                if sample.batch_position >= batch_size or sample.batch_position in rows:
                    raise ValueError("batch rows must have exactly one valid owner")
                rows.add(sample.batch_position)
                _text("sample_id", sample.sample_id)
                if sample.sample_id in sample_ids:
                    raise ValueError("sample IDs must be distinct within a request")
                sample_ids.add(sample.sample_id)
                if type(sample.token_positions) is not tuple or not sample.token_positions:
                    raise ValueError("token positions must be a nonempty tuple")
                positions = set()
                for position in sample.token_positions:
                    _nonnegative("token position", position)
                    if position >= sequence_length or position in positions:
                        raise ValueError("token positions must be distinct and in range")
                    positions.add(position)
        if len(rows) != batch_size:
            raise ValueError("every batch row must have exactly one owner")
        if not isinstance(sequence_starts, Mapping) or set(sequence_starts) != run_ids:
            raise ValueError("explicit sequence_starts must map every run ID exactly once")
        for value in sequence_starts.values():
            _nonnegative("sequence start", value)

        self._requests = requests
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._max_events = max_events
        self._max_trace_bytes = max_trace_bytes
        self._capture_device = capture_device
        self._sequences = dict(sequence_starts)
        self._module_counts = defaultdict(int)
        self._module_metadata = {}
        self._request_events = defaultdict(int)
        self._request_bytes = defaultdict(int)
        self._events = []
        self._retained_bytes = 0
        self._state = "new"
        self._failure_kind = None
        self._owner_thread = None
        self._recording = False
        self._state_lock = threading.RLock()

    @property
    def retained_bytes(self):
        """Payload bytes still held internally, excluding results already returned."""
        with self._state_lock:
            return self._retained_bytes

    def _abort(self, failure_kind):
        # Do not keep exception objects or their traceback frames and tensors.
        with self._state_lock:
            if self._failure_kind is None:
                self._failure_kind = failure_kind
            self._events.clear()
            self._retained_bytes = 0

    def run(self, forward):
        """Call trusted synchronous forward once; release events only on success."""
        if not callable(forward):
            raise ValueError("forward must be callable")
        if inspect.iscoroutinefunction(forward) or inspect.isasyncgenfunction(forward) \
                or inspect.isgeneratorfunction(forward):
            raise RuntimeError("forward must be synchronous and eager")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("dispatch inside an asyncio event loop is unsupported")
        if not _EXECUTION_LOCK.acquire(blocking=False):
            raise RuntimeError("overlapping or reentrant dispatch is unsupported")
        try:
            with self._state_lock:
                if self._state != "new":
                    raise RuntimeError("CaptureDispatcher is single-use")
                self._state = "active"
                self._owner_thread = threading.get_ident()
            output = events = None
            try:
                output = forward()
                if isinstance(output, Future) or inspect.isawaitable(output) or inspect.isgenerator(output) \
                        or inspect.isasyncgen(output):
                    if inspect.iscoroutine(output) or inspect.isgenerator(output):
                        output.close()
                    raise RuntimeError("forward returned deferred work")
                with self._state_lock:
                    if self._failure_kind is not None:
                        raise CaptureAbortedError("capture failed during forward")
                    events = tuple(self._events)
                    self._events.clear()
                    self._retained_bytes = 0
                    self._state = "finished"
                return output, events
            except BaseException as exc:
                # Retained tracebacks include this frame. Drop references to
                # model outputs and closures as well as owned trace payloads.
                output = events = None
                forward = None
                self._abort(type(exc).__name__)
                with self._state_lock:
                    self._state = "failed"
                raise
        finally:
            _EXECUTION_LOCK.release()

    def record(self, module_path, *, layer_index, module_kind, output, output_path=None):
        """Record one module invocation; the surrounding hook must return None."""
        with self._state_lock:
            if self._state != "active":
                output = None
                raise RuntimeError("record is only valid inside an active forward")
            if threading.get_ident() != self._owner_thread:
                self._abort("wrong thread")
                output = None
                raise RuntimeError("cross-thread record is unsupported")
            if self._recording:
                self._abort("reentrant record")
                output = None
                raise RuntimeError("reentrant record is unsupported")
            if self._failure_kind is not None:
                output = None
                raise CaptureAbortedError("capture was already aborted")
            self._recording = True

        pending = []
        detached = indices = selected = None
        try:
            _text("module_path", module_path)
            _text("module_kind", module_kind)
            if layer_index is not None:
                _nonnegative("layer_index", layer_index)
            if output_path is not None:
                if type(output_path) is not tuple:
                    raise ValueError("output_path must be a tuple or None")
                for part in output_path:
                    if type(part) is int:
                        _nonnegative("output path index", part)
                    elif type(part) is str:
                        _text("output path key", part)
                    else:
                        raise ValueError("output path parts must be indices or keys")
            metadata = (layer_index, module_kind, output_path)
            if module_path in self._module_metadata and self._module_metadata[module_path] != metadata:
                raise ValueError("module metadata changed within the forward")
            if type(output) is not torch.Tensor or output.device.type == "meta" \
                    or output.layout != torch.strided or not output.is_floating_point():
                raise TypeError("output must be a materialized dense strided floating tensor, not a subclass")
            shape = tuple(output.shape)
            if len(shape) < 3 or shape[:2] != (self._batch_size, self._sequence_length) \
                    or any(size <= 0 for size in shape[2:]):
                raise ValueError("output must have shape [batch, token, ...] with positive trailing dimensions")

            element_bytes = output.element_size() * math.prod(shape[2:])
            increments = {}
            for request in self._requests:
                n_events = len(request.samples)
                n_bytes = sum(len(sample.token_positions) * element_bytes
                              for sample in request.samples)
                if self._request_events[request.request_id] + n_events > request.max_events \
                        or self._request_bytes[request.request_id] + n_bytes > request.max_trace_bytes:
                    raise CaptureLimitError("request capture limit exceeded")
                increments[request.request_id] = (n_events, n_bytes)
            new_events = sum(item[0] for item in increments.values())
            new_bytes = sum(item[1] for item in increments.values())
            if len(self._events) + new_events > self._max_events \
                    or self._retained_bytes + new_bytes > self._max_trace_bytes:
                raise CaptureLimitError("batch capture limit exceeded")

            # Capture into a private pending list. Nothing is published until
            # the full event and the containing forward both succeed.
            sequence_numbers = dict(self._sequences)
            detached = output.detach()
            invocation = self._module_counts[module_path]
            for request in self._requests:
                for sample in request.samples:
                    # Worker forwards may run in inference_mode.  Build the
                    # owned snapshot outside it so a caller can use or mutate
                    # that separate tensor after the worker exits its context.
                    with torch.inference_mode(False):
                        indices = torch.tensor(sample.token_positions, dtype=torch.long,
                                               device=output.device)
                        selected = detached[sample.batch_position].index_select(0, indices)
                        if self._capture_device == "cpu" and selected.device.type != "cpu":
                            selected = selected.to(device="cpu", non_blocking=False)
                    sequence = sequence_numbers[request.run_id]
                    sequence_numbers[request.run_id] += 1
                    pending.append(CapturedSlice(
                        request.run_id, request.job_id, request.request_id,
                        sample.sample_id, sample.batch_position, sample.token_positions,
                        request.forward_index, request.generation_step, sequence,
                        module_path, layer_index, module_kind, output_path, invocation,
                        shape, str(output.dtype), str(output.device), tuple(selected.shape),
                        str(selected.dtype), str(selected.device), selected,
                    ))
            with self._state_lock:
                if self._failure_kind is not None:
                    raise CaptureAbortedError("capture failed during slicing")
                self._events.extend(pending)
                self._retained_bytes += new_bytes
                self._sequences = sequence_numbers
                self._module_counts[module_path] += 1
                self._module_metadata[module_path] = metadata
                for request_id, (n_events, n_bytes) in increments.items():
                    self._request_events[request_id] += n_events
                    self._request_bytes[request_id] += n_bytes
        except BaseException as exc:
            # A consumer may retain this exception and its traceback. Clear
            # this frame's private payloads before propagating the failure.
            pending.clear()
            detached = indices = selected = output = None
            self._abort(type(exc).__name__)
            raise
        finally:
            with self._state_lock:
                self._recording = False
