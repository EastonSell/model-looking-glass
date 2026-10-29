"""CPU-only trusted case prototype for ordered intervention plans.

This is not a broker, GPU worker, IPC endpoint, or untrusted-code boundary.
The trusted owner must exclude arbitrary direct model access and control model
code. Cooperating hook APIs are excluded by one continuous ModelLease.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from threading import Event, Lock, RLock, current_thread
from types import MappingProxyType

import torch

from .intervention_plan import ActivationPlan
from .model_ownership import ModelLease
from .resident_worker import _rng_lock, registered_model_state
from .trace_store import HostTraceStore, TraceStoreFull


class PlanWorkerError(RuntimeError):
    """A private case failed and cannot publish its results."""


class PlanWorkerBusy(PlanWorkerError):
    """Another case owns this model, or retained case capacity is full."""


class PlanWorkerOutOfMemory(PlanWorkerError):
    """An allocation failure left the worker quarantined."""


class PlanCaseCancelled(PlanWorkerError):
    """A cooperative cancellation was observed at a forward boundary."""


class PlanControlFailed(PlanWorkerError):
    """A trusted private control did not match its expected raw bits."""


class PlanTensorRef:
    """Opaque, case-local identity for a retained CPU tensor."""

    __slots__ = ()


class PlanExport:
    """A separately charged, mutable consumer copy owned by a trusted client."""

    def __init__(self, worker, size):
        self._worker = worker
        self._size = size
        self._state = "pending"
        self._tensor = None
        self._lock = RLock()

    @property
    def tensor(self):
        with self._lock:
            if self._state != "ready":
                raise PlanWorkerError("export is not available")
            return self._tensor

    @property
    def state(self):
        with self._lock:
            return self._state

    def close(self):
        """Acknowledge that all client aliases and in-flight uses are gone."""
        with self._lock, self._worker._lock:
            if self._state == "closed":
                return
            if self._state != "ready":
                raise PlanWorkerError("uncertain export requires worker disposal")
            self._tensor = None
            self._worker._exports.discard(self)
            self._state = "closed"


@dataclass(frozen=True)
class PlanRun:
    """Private references and input provenance for one completed forward."""

    captures: Mapping
    sources: Mapping
    logits: PlanTensorRef
    input_sha256: str
    seed: int
    index: int


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _cpu_tensor(value, name):
    if (type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.layout != torch.strided or value.is_nested or value.is_quantized):
        raise ValueError(f"{name} must be a plain, strided CPU tensor")
    return value.numel() * value.element_size()


def _bits_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    a = left.reshape(-1).contiguous().view(torch.uint8)
    b = right.reshape(-1).contiguous().view(torch.uint8)
    return bool(torch.equal(a, b))


def _sanitized_exception(error):
    """Detach execution tracebacks and preserve only bounded plain messages."""
    message = "plan case failed"
    if len(error.args) == 1 and type(error.args[0]) is str and len(error.args[0]) <= 256:
        message = error.args[0]
    if isinstance(error, (MemoryError, torch.OutOfMemoryError)):
        clean = PlanWorkerOutOfMemory("case allocation failed; worker quarantined")
    elif type(error) in (ValueError, TypeError, PlanWorkerError, PlanWorkerOutOfMemory,
                       PlanWorkerBusy, PlanCaseCancelled, PlanControlFailed, TraceStoreFull):
        clean = type(error)(message)
    elif isinstance(error, KeyboardInterrupt):
        clean = KeyboardInterrupt()
    elif isinstance(error, SystemExit):
        clean = SystemExit(1)
    else:
        clean = PlanWorkerError("plan execution failed")
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None
    return clean


class CPUPlanWorker:
    """Own one trusted CPU model and a bounded store across private cases."""

    def __init__(self, model, *, max_sequence_length=32, max_input_bytes=1024,
                 max_output_bytes=2 * 1024 * 1024, max_tensor_bytes=2 * 1024 * 1024,
                 max_capture_bytes=1024 * 1024, max_source_bytes=1024 * 1024,
                 max_check_bytes=4 * 1024 * 1024, max_steps=64,
                 max_runs_per_case=32, max_stored_bytes=16 * 1024 * 1024,
                 max_stored_items=512, max_stored_cases=4,
                 max_export_bytes=16 * 1024 * 1024, max_export_items=512):
        for name, value in (("max_sequence_length", max_sequence_length),
                            ("max_input_bytes", max_input_bytes),
                            ("max_output_bytes", max_output_bytes),
                            ("max_tensor_bytes", max_tensor_bytes),
                            ("max_capture_bytes", max_capture_bytes),
                            ("max_source_bytes", max_source_bytes),
                            ("max_check_bytes", max_check_bytes), ("max_steps", max_steps),
                            ("max_runs_per_case", max_runs_per_case),
                            ("max_stored_bytes", max_stored_bytes),
                            ("max_stored_items", max_stored_items),
                            ("max_stored_cases", max_stored_cases),
                            ("max_export_bytes", max_export_bytes),
                            ("max_export_items", max_export_items)):
            _positive(value, name)
        self._model = model
        self._baseline = registered_model_state(model, "cpu")
        self.max_sequence_length = max_sequence_length
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_tensor_bytes = max_tensor_bytes
        self.max_capture_bytes = max_capture_bytes
        self.max_source_bytes = max_source_bytes
        self.max_check_bytes = max_check_bytes
        self.max_steps = max_steps
        self.max_runs_per_case = max_runs_per_case
        self.max_stored_cases = max_stored_cases
        self.max_export_bytes = max_export_bytes
        self.max_export_items = max_export_items
        self._store = HostTraceStore(max_tensor_bytes=max_stored_bytes, max_items=max_stored_items)
        self._lock = RLock()
        self._active = None
        self._cases = set()
        self._exports = set()
        self._poisoned = False
        self._closed = False

    @property
    def poisoned(self):
        with self._lock:
            return self._poisoned

    def usage(self):
        """Logical retained/pending tensor quota; not total process memory."""
        return self._store.usage()

    def export_usage(self):
        """Return separately retained consumer-copy commitments."""
        with self._lock:
            return {"items": len(self._exports), "bytes": sum(item._size for item in self._exports),
                    "max_items": self.max_export_items, "max_bytes": self.max_export_bytes}

    def active_exports(self):
        """Recover retained export handles after an interrupted return."""
        with self._lock:
            return tuple(self._exports)

    def _export(self, entry, size):
        holder = PlanExport(self, size)
        value = None
        try:
            with self._lock:
                if self._poisoned or self._closed:
                    raise PlanWorkerError("worker is unavailable")
                if (self._active is not None
                        or any(case._closing_token is not None for case in self._cases)):
                    raise PlanWorkerBusy("cannot copy an export while a case is active")
                if (len(self._exports) >= self.max_export_items
                        or sum(item._size for item in self._exports) + size > self.max_export_bytes):
                    raise PlanWorkerBusy("consumer export quota exceeded")
                self._exports.add(holder)
            # A failed/ambiguous copy keeps its reservation quarantined.
            value = entry.read()
            with self._lock:
                holder._tensor = value
                holder._state = "ready"
            value = None
            return holder
        except BaseException as error:
            with self._lock:
                if holder in self._exports:
                    holder._state = "quarantined"
                    self._poisoned = True
                elif isinstance(error, (MemoryError, torch.OutOfMemoryError)):
                    self._poisoned = True
            sanitized = _sanitized_exception(error)
            entry = None
            value = None
        raise sanitized from None

    def case(self, case_id):
        if type(case_id) is not str or not case_id.strip() or len(case_id) > 256:
            raise ValueError("case_id must be a nonempty bounded string")
        return _PlanCase(self, case_id)

    def close(self):
        with self._lock:
            if self._active is not None:
                raise PlanWorkerBusy("cannot close a worker with an active case")
            if self._cases:
                raise PlanWorkerBusy("discard retained cases before closing the worker")
            if self._exports:
                raise PlanWorkerBusy("close all consumer exports before closing the worker")
            if self._poisoned:
                raise PlanWorkerError("quarantined worker requires process/model disposal")
            self._closed = True
            self._model = None


class _PlanCase:
    """A trusted owner-thread scope holding one model lease across forwards."""

    def __init__(self, worker, case_id):
        self._worker = worker
        self._case_id = case_id
        self._lease = ModelLease(worker._model)
        self._request = worker._store.request(case_id)
        self._lock = RLock()
        self._state = "new"
        self._entry_token = None
        self._closing_token = None
        self._owner = None
        self._in_forward = False
        self._failed = False
        self._entries = {}
        self._next_item = 0
        self._records = []
        self._failure = None
        self._attempt = None

    def __enter__(self):
        worker = self._worker
        token = object()
        try:
            if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
                raise PlanWorkerError("ambient autocast is incompatible with case execution")
            with self._lock, worker._lock:
                if self._state != "new":
                    raise PlanWorkerError("case scopes are single-use")
                if worker._closed or worker._poisoned:
                    raise PlanWorkerError("worker is unavailable")
                if (worker._active is not None or len(worker._cases) >= worker.max_stored_cases
                        or any(case._closing_token is not None for case in worker._cases)
                        or any(export._state != "ready" for export in worker._exports)
                        or any(case._case_id == self._case_id for case in worker._cases)):
                    raise PlanWorkerBusy("worker case capacity is unavailable")
                self._entry_token = token
                self._state = "entering"
                self._owner = current_thread()
                worker._active = self
                worker._cases.add(self)
            self._lease.acquire()
            try:
                self._check_state()
            except BaseException as error:
                self._quarantine(type(error).__name__)
                raise
            with self._lock:
                self._state = "active"
            return self
        except BaseException:
            with self._lock:
                owns_entry = self._entry_token is token
            if not owns_entry:
                raise
            with worker._lock:
                registered = worker._active is self or self in worker._cases
            if registered:
                try:
                    if worker.poisoned:
                        with self._lock:
                            self._state = "quarantined"
                    else:
                        self._lease.release()
                except BaseException:
                    with worker._lock:
                        worker._poisoned = True
                    with self._lock:
                        self._state = "quarantined"
                    raise
                finally:
                    with worker._lock:
                        worker._active = None
                        if self._state != "quarantined":
                            worker._cases.discard(self)
                    if self._state != "quarantined":
                        self._state = "failed"
            else:
                with self._lock:
                    self._state = "failed"
            raise

    def _check_owner(self):
        if current_thread() is not self._owner:
            self._failed = True
            raise PlanWorkerError("case operations require the owner thread")
        if self._state != "active" or self._failed:
            raise PlanWorkerError("case is not active")
        if self._worker.poisoned:
            self._failed = True
            self._failure = "worker quarantined"
            raise PlanWorkerError("worker is unavailable")

    def _check_state(self):
        self._lease.verify_tree()
        if registered_model_state(self._worker._model, "cpu") != self._worker._baseline:
            raise PlanWorkerError("model state changed or could not be verified")

    def _quarantine(self, reason):
        self._failed = True
        self._failure = reason
        with self._worker._lock:
            self._worker._poisoned = True
        try:
            self._lease._quarantine()
        except BaseException:
            # A lost/uncertain lease remains a process-level disposal condition.
            pass

    def _put(self, tensor, kind, *, input_sha256=None):
        size = _cpu_tensor(tensor, "stored value")
        ref = PlanTensorRef()
        entry = self._request.put(self._next_item, tensor)
        self._next_item += 1
        self._entries[ref] = (entry, kind, size, input_sha256)
        return ref

    def _entry(self, ref, *, kind=None, input_sha256=None):
        if type(ref) is not PlanTensorRef or ref not in self._entries:
            raise ValueError("tensor reference does not belong to this case")
        entry, actual_kind, size, origin_input = self._entries[ref]
        if kind is not None and actual_kind != kind:
            raise ValueError("tensor reference is not a capture")
        if input_sha256 is not None and origin_input != input_sha256:
            raise ValueError("source capture was recorded for different inputs")
        return entry, size

    def _input_snapshot(self, inputs):
        worker = self._worker
        if type(inputs) is not dict or "input_ids" not in inputs or not set(inputs) <= {"input_ids", "attention_mask"}:
            raise ValueError("inputs must contain only input_ids and optional attention_mask")
        total = 0
        shape = None
        for name, value in inputs.items():
            total += _cpu_tensor(value, name)
            allowed = (torch.int32, torch.int64, torch.bool) if name == "attention_mask" else (torch.int32, torch.int64)
            if value.dtype not in allowed or value.ndim != 2:
                raise ValueError("inputs must be rank-two integer token tensors")
            if shape is None:
                shape = tuple(value.shape)
            if tuple(value.shape) != shape:
                raise ValueError("input tensor shapes must match")
        if shape[0] != 1 or not 0 < shape[1] <= worker.max_sequence_length or total > worker.max_input_bytes:
            raise ValueError("input shape or byte bound exceeded")
        snapshot = {name: value.detach().clone(memory_format=torch.contiguous_format)
                    for name, value in inputs.items()}
        digest = hashlib.sha256()
        for name in sorted(snapshot):
            value = snapshot[name]
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(repr(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(bytes(value.untyped_storage()))
        return snapshot, digest.hexdigest()

    def run(self, inputs, steps, *, seed, cancel_event=None):
        """Run one plan without exposing tensor-bearing exception frames."""
        try:
            return self._run(inputs, steps, seed=seed, cancel_event=cancel_event)
        except BaseException as error:
            if not self._failed:
                self._quarantine(type(error).__name__)
            sanitized = _sanitized_exception(error)
            inputs = steps = cancel_event = None
        raise sanitized from None

    def _run(self, inputs, steps, *, seed, cancel_event=None):
        """Run one private plan; no tensor values are published during a case."""
        with self._lock:
            self._check_owner()
            if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
                self._failed = True
                self._failure = "ambient autocast"
                raise PlanWorkerError("ambient autocast is incompatible with case execution")
            if self._in_forward:
                self._failed = True
                raise PlanWorkerError("reentrant case execution is unsupported")
            if len(self._records) >= self._worker.max_runs_per_case:
                self._failed = True
                raise PlanWorkerError("case forward limit exceeded")
            if type(seed) is not int or not 0 <= seed < 2**63:
                self._failed = True
                raise ValueError("seed must be an integer in [0, 2**63)")
            if cancel_event is not None and type(cancel_event) is not Event:
                self._failed = True
                raise ValueError("cancel_event must be a threading.Event")
            if cancel_event is not None and cancel_event.is_set():
                self._failed = True
                self._failure = "cancelled before forward"
                raise PlanCaseCancelled("case cancelled before forward")
        worker = self._worker
        started = False
        completed_and_checked = False
        try:
            self._check_state()
            snapshot, input_digest = self._input_snapshot(inputs)
            if type(steps) not in (tuple, list) or not steps or len(steps) > worker.max_steps:
                raise ValueError("steps must be a bounded nonempty list")
            staged = []
            source_bytes = 0
            for raw in steps:
                if type(raw) is not dict:
                    raise ValueError("steps must be plain dictionaries")
                step = dict(raw)
                source_ref = step.pop("value_ref", None)
                if source_ref is not None:
                    if "value" in step or "source_capture" in step:
                        raise ValueError("value_ref cannot be combined with another source")
                    entry, size = self._entry(source_ref, kind="capture", input_sha256=input_digest)
                    source_bytes += size
                    staged.append((step, source_ref, entry))
                elif "value" in step:
                    size = _cpu_tensor(step["value"], "source value")
                    source_bytes += size
                    staged.append((step, None, None))
                else:
                    staged.append((step, None, None))
            if source_bytes > worker.max_source_bytes:
                raise ValueError("aggregate source values exceed their bound")
            sources = {}
            for step, ref, entry in staged:
                if ref is None and "value" in step:
                    ref = self._put(step.pop("value"), "source")
                    entry, _ = self._entry(ref)
                if ref is not None:
                    sources[step.get("name")] = ref
                    step["value"] = entry.read()
            plan = ActivationPlan(worker._model, [item[0] for item in staged],
                                  max_tensor_bytes=worker.max_tensor_bytes,
                                  max_capture_bytes=worker.max_capture_bytes,
                                  max_source_bytes=worker.max_source_bytes,
                                  max_steps=worker.max_steps, max_invocations=1,
                                  _parent_lease=self._lease)
            self._attempt = {"index": len(self._records), "input_sha256": input_digest,
                             "seed": seed,
                             "steps": [{key: value for key, value in step.items() if key != "value"}
                                       for step, _, _ in staged],
                             "sources": dict(sources), "captures": {}, "logits": None,
                             "plan_metadata": None}
            with self._lock:
                self._check_owner()
                self._in_forward = True
            started = True
            try:
                with _rng_lock, torch.random.fork_rng(devices=[]), torch.inference_mode():
                    torch.random.default_generator.manual_seed(seed)
                    with plan:
                        output = worker._model(**snapshot)
            finally:
                with self._lock:
                    self._in_forward = False
            self._check_state()
            with self._lock:
                self._check_owner()
            completed_and_checked = True
            self._attempt["plan_metadata"] = plan.metadata
            logits = output if type(output) is torch.Tensor else output.get("logits") if isinstance(output, Mapping) else None
            if isinstance(output, Mapping) and output.get("past_key_values") is not None:
                self._quarantine("model returned an unsupported key/value cache")
                raise PlanWorkerError("model returned an unsupported key/value cache")
            if (type(logits) is not torch.Tensor or logits.device.type != "cpu"
                    or not logits.is_floating_point() or logits.ndim != 3
                    or tuple(logits.shape[:2]) != tuple(snapshot["input_ids"].shape)):
                raise ValueError("model must return CPU rank-three logits matching the input")
            if _cpu_tensor(logits, "logits") > worker.max_output_bytes or not bool(torch.isfinite(logits).all()):
                raise ValueError("logits exceed their bound or contain nonfinite values")
            if cancel_event is not None and cancel_event.is_set():
                self._failed = True
                self._failure = "cancelled after forward"
                raise PlanCaseCancelled("case cancelled after forward")
            captures = self._attempt["captures"]
            for name, value in plan.captures.items():
                captures[name] = self._put(value, "capture", input_sha256=input_digest)
            logit_ref = self._put(logits[0, -1], "logits", input_sha256=input_digest)
            self._attempt["logits"] = logit_ref
            run = PlanRun(MappingProxyType(captures), MappingProxyType(sources),
                          logit_ref, input_digest, seed, len(self._records))
            self._records.append((run, plan.metadata))
            self._attempt = None
            return run
        except PlanCaseCancelled:
            if started and not completed_and_checked:
                self._quarantine("cancellation during forward")
            else:
                self._failed = True
                if self._failure is None:
                    self._failure = "cancelled at forward boundary"
            raise
        except (ValueError, TypeError, TraceStoreFull) as error:
            self._failed = True
            self._failure = type(error).__name__
            if started and not completed_and_checked:
                self._quarantine(type(error).__name__)
            raise
        except BaseException as error:
            self._quarantine(type(error).__name__)
            raise

    def private_metadata(self, run):
        """Return private plan evidence to the trusted custodian only."""
        with self._lock:
            if (self._closing_token is not None
                    or self._state not in ("active", "finished", "failed", "quarantined")):
                raise PlanWorkerError("case metadata is unavailable")
            if type(run) is not PlanRun or run.index >= len(self._records) or self._records[run.index][0] is not run:
                raise ValueError("run does not belong to this case")
            return deepcopy(self._records[run.index][1])

    def _check_pair(self, left, right):
        left_entry, left_size = self._entry(left)
        right_entry, right_size = self._entry(right)
        if left_size + right_size > self._worker.max_check_bytes:
            raise ValueError("private control exceeds check workspace bound")
        return left_entry.read(), right_entry.read()

    def require_equal(self, left, right):
        """Require exact raw bits without exposing a retained backing tensor."""
        try:
            return self._require_equal(left, right)
        except BaseException as error:
            self._failed = True
            self._failure = "private equality control failed"
            if isinstance(error, (MemoryError, torch.OutOfMemoryError)) or not isinstance(error, Exception):
                self._quarantine(type(error).__name__)
            sanitized = _sanitized_exception(error)
            left = right = None
        raise sanitized from None

    def _require_equal(self, left, right):
        with self._lock:
            self._check_owner()
            if self._in_forward:
                raise PlanWorkerError("controls must run between forwards")
            try:
                first, second = self._check_pair(left, right)
                if not _bits_equal(first, second):
                    raise PlanControlFailed("private equality control failed")
            except BaseException:
                self._failed = True
                self._failure = "private equality control failed"
                raise

    def require_added(self, before, delta, after):
        """Require raw bits of before + retained delta to equal after."""
        try:
            return self._require_added(before, delta, after)
        except BaseException as error:
            self._failed = True
            self._failure = "private addition control failed"
            if isinstance(error, (MemoryError, torch.OutOfMemoryError)) or not isinstance(error, Exception):
                self._quarantine(type(error).__name__)
            sanitized = _sanitized_exception(error)
            before = delta = after = None
        raise sanitized from None

    def _require_added(self, before, delta, after):
        with self._lock:
            self._check_owner()
            if self._in_forward:
                raise PlanWorkerError("controls must run between forwards")
            try:
                before_entry, before_size = self._entry(before)
                delta_entry, delta_size = self._entry(delta)
                after_entry, after_size = self._entry(after)
                if 2 * before_size + delta_size + after_size > self._worker.max_check_bytes:
                    raise ValueError("private control exceeds check workspace bound")
                first, difference, result = before_entry.read(), delta_entry.read(), after_entry.read()
                if first.shape != difference.shape or first.dtype != difference.dtype:
                    raise PlanControlFailed("private addition control failed")
                with torch.no_grad():
                    expected = first + difference
                if not _bits_equal(expected, result):
                    raise PlanControlFailed("private addition control failed")
            except BaseException:
                self._failed = True
                self._failure = "private addition control failed"
                raise

    def private_failure(self):
        """Describe retained partial evidence to the trusted custodian."""
        with self._lock:
            if self._closing_token is not None or self._state not in ("failed", "quarantined"):
                raise PlanWorkerError("case has no completed failure state")
            attempt = None
            if self._attempt is not None:
                attempt = dict(self._attempt)
                attempt["steps"] = deepcopy(attempt["steps"])
                attempt["plan_metadata"] = deepcopy(attempt["plan_metadata"])
                attempt["sources"] = dict(attempt["sources"])
                attempt["captures"] = dict(attempt["captures"])
            return {"reason": self._failure, "completed_runs": len(self._records),
                    "metadata": deepcopy([record[1] for record in self._records]),
                    "attempt": attempt}

    def export_private_failure(self, ref):
        """Copy a completed prior tensor to a charged private export."""
        with self._lock:
            if self._closing_token is not None or self._state not in ("failed", "quarantined"):
                raise PlanWorkerError("case has no completed failure state")
            entry, size = self._entry(ref)
            return self._worker._export(entry, size)

    def __exit__(self, exc_type, exc, tb):
        worker = self._worker
        token = object()
        cleanup_error = None
        try:
            with self._lock:
                if current_thread() is not self._owner or self._state != "active":
                    self._failed = True
                    raise PlanWorkerError("case exit requires its active owner thread")
                if self._in_forward:
                    self._failed = True
                    raise PlanWorkerError("cannot exit while a forward may be active")
                self._closing_token = token
                self._state = "closing"
            with worker._lock:
                if not worker._poisoned:
                    self._check_state()
                    self._lease.release()
                else:
                    self._lease._quarantine()
        except BaseException as error:
            with self._lock:
                owns_cleanup = self._closing_token is token
            if not owns_cleanup:
                raise
            cleanup_error = error
            self._quarantine(type(error).__name__)
        finally:
            with self._lock:
                owns_cleanup = self._closing_token is token
            if owns_cleanup:
                with self._lock, worker._lock:
                    worker._active = None
                    self._state = "quarantined" if worker._poisoned else (
                        "failed" if exc is not None or self._failed else "finished")
                    self._closing_token = None
        if cleanup_error is not None:
            if exc is not None:
                raise exc.with_traceback(tb) from cleanup_error
            raise cleanup_error
        if exc is None and self._failed:
            raise PlanWorkerError("case failed; partial results were not published")
        return False

    def abort(self):
        """Abort after the owner establishes that no forward is active."""
        with self._lock:
            self._check_owner()
            if self._in_forward:
                self._failed = True
                raise PlanWorkerError("cannot abort while a forward may be active")
            self._failed = True
            self._failure = "case aborted"
        self.__exit__(PlanWorkerError, PlanWorkerError("case aborted"), None)

    @property
    def status(self):
        """Expose a possibly interrupted exit without publishing tensor data."""
        with self._lock:
            return "recovery_required" if self._closing_token is not None else self._state

    def quarantine_interrupted_exit(self):
        """After exit unwinds, fail closed if its finalization was interrupted."""
        worker = self._worker
        with self._lock, worker._lock:
            if current_thread() is not self._owner:
                raise PlanWorkerError("only the case owner can quarantine interrupted exit")
            if self._closing_token is None or self._in_forward:
                raise PlanWorkerError("no quiescent interrupted exit to quarantine")
            self._failed = True
            self._failure = "interrupted case exit"
            worker._poisoned = True
            try:
                self._lease._quarantine()
            except BaseException:
                # Release may already have happened; never attest a clean case.
                pass
            self._state = "quarantined"
            self._closing_token = None
            if worker._active is self:
                worker._active = None

    def export(self, ref):
        """Create a separately charged consumer copy after full-case success."""
        with self._lock:
            if self._closing_token is not None or self._state != "finished" or self._failed:
                raise PlanWorkerError("case results are not published")
            entry, size = self._entry(ref)
            return self._worker._export(entry, size)

    def discard(self):
        """Drop retained case payloads after a safe success or failure."""
        with self._lock:
            if self._closing_token is not None or self._state not in ("finished", "failed"):
                raise PlanWorkerError("cannot discard an active or quarantined case")
            self._request.cancel()
            self._entries.clear()
            self._records.clear()
            self._attempt = None
            self._state = "discarded"
            with self._worker._lock:
                self._worker._cases.discard(self)
