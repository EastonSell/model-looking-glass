"""Fail-closed admission for a trusted serial resident-model worker.

The service must route all device work through one shared DeviceAdmission,
bind profiles and telemetry to the actual model and device, and replace its
process when that admission object is quarantined. See BUDGETED_WORKER.md.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import re
import threading

import torch

from .memory_budget import MemoryBudget
from .resident_worker import (
    ForwardResult, InvalidRequest, RequestCancelled, ResidentWorker,
    WorkerExecutionError, WorkerStateError, WorkerSynchronizationError, WorkerUnavailable,
)


class ProfileUnavailable(ValueError):
    """No trusted profile covers the exact request signature."""


class DeviceQuarantined(WorkerUnavailable):
    """Device completion or accounting is uncertain; replace the process."""


class WorkerContractError(WorkerExecutionError):
    """The configured worker or its result violated the admitted contract."""


@dataclass(frozen=True)
class WorkloadProfile:
    """Service-owned, exact-shape estimate; never derived from a request."""

    profile_id: str
    model_id: str
    device: str
    shape: tuple[int, int]
    input_dtypes: tuple[tuple[str, str], ...]
    output_limit_bytes: int
    estimated_incremental_bytes: int


def _device(device):
    try:
        result = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError('invalid admission device') from exc
    if result.type not in ('cpu', 'cuda') or (result.type == 'cuda' and result.index is None):
        raise ValueError('device must be cpu or an explicit CUDA index')
    if result.type == 'cpu' and result.index is not None:
        raise ValueError('CPU admission device must not have an index')
    return result


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _describe_error(exc):
    try:
        return f'{type(exc).__name__}: {str(exc)[:1024]}'
    except BaseException:
        return f'{type(exc).__name__}: <message unavailable>'


class DeviceAdmission:
    """One process-local, serialized admission and quarantine gate per device."""

    def __init__(self, budget, *, device, max_estimated_incremental_bytes):
        if type(budget) is not MemoryBudget:
            raise TypeError('budget must be a MemoryBudget')
        _positive_int(max_estimated_incremental_bytes, 'max_estimated_incremental_bytes')
        self._budget = budget
        self.device = _device(device)
        self.max_estimated_incremental_bytes = max_estimated_incremental_bytes
        self._lock = threading.Lock()
        self._quarantined = threading.Event()
        # On uncertainty, keep leases and objects alive until process exit.
        self._holds = []

    @property
    def quarantined(self):
        return self._quarantined.is_set()

    def _quarantine(self, lease, worker, detail=None):
        # Set first: retaining additional references may itself fail during OOM.
        # MemoryBudget already keeps the lease alive even if this append fails.
        self._quarantined.set()
        try:
            self._holds.append((lease, worker, detail))
        except BaseException:
            pass  # No further work is admitted; process replacement is required.


def _worker_binding(worker):
    return (worker.model_id, worker.device, worker.max_batch_size,
            worker.max_sequence_length, worker.max_input_bytes, worker.max_output_bytes)


def _model_cache_binding(worker):
    model = worker._model  # ResidentWorker is exclusively owned by this wrapper.
    if model is None:
        raise ValueError('resident model is unavailable')
    config = getattr(model, 'config', None)
    if config is not None and hasattr(config, 'use_cache'):
        if config.use_cache is not False:
            raise ValueError('configured model cache must be explicitly disabled')
        return id(model), id(config), False
    # Arbitrary trusted model code can still cache internally; see the docs.
    return id(model), id(config), None


def _require_no_autocast():
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda'):
        raise RuntimeError('ambient autocast is incompatible with profiled execution')


def _validate_profile(profile, worker, admission):
    if type(profile) is not WorkloadProfile:
        raise TypeError('profiles must be WorkloadProfile records')
    if type(profile.profile_id) is not str or not profile.profile_id.strip() or len(profile.profile_id) > 256:
        raise ValueError('profile_id must be a nonempty bounded string')
    if (type(profile.model_id) is not str or not re.fullmatch(r'[0-9a-f]{64}', profile.model_id)
            or profile.model_id != worker.model_id):
        raise ValueError('profile must match the service-verified model fingerprint')
    if type(profile.device) is not str or _device(profile.device) != admission.device:
        raise ValueError('profile device must match the admission device')
    if type(profile.shape) is not tuple or len(profile.shape) != 2:
        raise ValueError('profile shape must be a pair of positive integers')
    for value in profile.shape:
        _positive_int(value, 'profile dimension')
    batch, sequence = profile.shape
    if batch > worker.max_batch_size or sequence > worker.max_sequence_length:
        raise ValueError('profile shape exceeds worker bounds')
    if type(profile.input_dtypes) is not tuple or not profile.input_dtypes:
        raise ValueError('input_dtypes must be an immutable sorted tuple')
    dtype_bytes = {'torch.int32': 4, 'torch.int64': 8, 'torch.bool': 1}
    allowed_names = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids'}
    names = []
    element_bytes = 0
    for item in profile.input_dtypes:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError('each input signature must be a name and dtype pair')
        name, dtype = item
        if (type(name) is not str or name not in allowed_names or type(dtype) is not str
                or dtype not in dtype_bytes or (dtype == 'torch.bool' and name != 'attention_mask')):
            raise ValueError('unsupported input name or dtype')
        names.append(name)
        element_bytes += dtype_bytes[dtype]
    if names != sorted(set(names)) or 'input_ids' not in names:
        raise ValueError('input names must be unique, sorted, and include input_ids')
    input_bytes = batch * sequence * element_bytes
    if input_bytes > worker.max_input_bytes:
        raise ValueError('profile input bytes exceed worker bounds')
    _positive_int(profile.output_limit_bytes, 'profile output limit')
    _positive_int(profile.estimated_incremental_bytes, 'profile estimate')
    if profile.output_limit_bytes != worker.max_output_bytes:
        raise ValueError('profile output limit must match the worker')
    if profile.estimated_incremental_bytes < input_bytes + profile.output_limit_bytes:
        raise ValueError('profile estimate must at least cover input and output limits')
    if profile.estimated_incremental_bytes > admission.max_estimated_incremental_bytes:
        raise ValueError('profile estimate exceeds service hard cap')
    return profile.shape, profile.input_dtypes


class BudgetedWorker:
    """Broker worker interface around one trusted worker and admission gate."""

    cancellation_exception = RequestCancelled

    def __init__(self, worker, admission, profiles):
        # Exception-to-completion guarantees are specific to this worker class.
        if type(worker) is not ResidentWorker or type(admission) is not DeviceAdmission:
            raise TypeError('expected a ResidentWorker and DeviceAdmission')
        if worker.device != admission.device:
            raise ValueError('worker device and admission device must match')
        profiles = tuple(profiles)
        if not profiles:
            raise ValueError('at least one trusted profile is required')
        self._worker, self._admission = worker, admission
        self._binding = _worker_binding(worker)
        self._cache_binding = _model_cache_binding(worker)
        self._admission_binding = (admission._budget, admission.device,
                                   admission.max_estimated_incremental_bytes)
        self._local_poisoned = threading.Event()
        self._closed = False
        self._profiles = {}
        ids = set()
        for profile in profiles:
            signature = _validate_profile(profile, worker, admission)
            if signature in self._profiles or profile.profile_id in ids:
                raise ValueError('duplicate workload signature or profile ID')
            ids.add(profile.profile_id)
            self._profiles[signature] = profile

    @property
    def quarantined(self):
        """Nonblocking signal that the owner must replace the execution process."""
        if self._process_is_quarantined():
            self._admission._quarantined.set()
        return self._admission.quarantined

    @property
    def failed(self):
        """Nonblocking failure signal; normal closure alone does not set it."""
        return self._local_poisoned.is_set() or self.quarantined or self._worker.poisoned

    @property
    def poisoned(self):
        # Broker queries this while holding its condition; never acquire the gate.
        return self._closed or self.failed

    @property
    def max_input_bytes(self):
        """Per-snapshot input-tensor limit; not an aggregate host-memory cap."""
        self._check_available()
        return self._binding[4]

    def _process_is_quarantined(self):
        try:
            return bool(self._worker.quarantined)
        except BaseException:
            # If the supervisor cannot read the latch, completion is unknown.
            return True

    def _binding_matches(self):
        admission = self._admission
        return (_worker_binding(self._worker) == self._binding
                and (admission._budget, admission.device,
                     admission.max_estimated_incremental_bytes) == self._admission_binding)

    def _cache_matches(self):
        try:
            return _model_cache_binding(self._worker) == self._cache_binding
        except BaseException:
            return False

    def _check_available(self):
        if self._process_is_quarantined():
            self._admission._quarantined.set()
        if self._admission.quarantined:
            raise DeviceQuarantined('device completion is unverified; replace the process')
        if not self._binding_matches() or not self._cache_matches():
            self._local_poisoned.set()
            raise WorkerUnavailable('worker identity, cache policy or limits changed after profiling')
        if self.poisoned:
            raise WorkerUnavailable('budgeted worker is unavailable')

    def snapshot_request(self, request):
        """Take the resident worker's bounded CPU snapshot without device admission."""
        self._check_available()
        return self._worker.snapshot_request(request)

    def _release(self, lease, original=None):
        try:
            lease.release()
        except BaseException as exc:
            self._admission._quarantine(lease, self._worker, exc)
            message = f'lease cleanup failed; device quarantined: {_describe_error(exc)}'
            if original is not None:
                message += f'; original execution failure: {_describe_error(original)}'
            raise WorkerContractError(message) from exc

    @contextmanager
    def _gate(self, cancel_event):
        # A waiter has launched no device work and can be cancelled without
        # waiting for an unrelated, potentially stuck forward to finish.
        while not self._admission._lock.acquire(timeout=0.05):
            self._check_available()
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled('request cancelled while awaiting admission')
        try:
            yield
        finally:
            self._admission._lock.release()

    def execute(self, request, cancel_event=None):
        """Admit and run one request, holding its lease through verified cleanup."""
        if cancel_event is not None and type(cancel_event) is not threading.Event:
            raise InvalidRequest('cancel_event must be a threading.Event')
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelled('request cancelled before admission')
        _require_no_autocast()
        self._check_available()
        # Own a snapshot for selection, so a caller cannot alter the selected
        # shape or dtype after admission. ResidentWorker snapshots again.
        snapshot = self._worker.snapshot_request(request)
        signature = (tuple(snapshot.inputs['input_ids'].shape),
                     tuple(sorted((name, str(value.dtype)) for name, value in snapshot.inputs.items())))
        profile = self._profiles.get(signature)
        if profile is None:
            raise ProfileUnavailable('no trusted profile matches this request')
        with self._gate(cancel_event):
            self._check_available()
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled('request cancelled while awaiting admission')
            _require_no_autocast()
            lease = self._admission._budget.acquire(profile.estimated_incremental_bytes,
                                                     max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            try:
                self._check_available()
                _require_no_autocast()
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelled('request cancelled before execution')
            except BaseException as exc:
                if self.quarantined:
                    self._admission._quarantine(lease, self._worker, exc)
                else:
                    self._release(lease, exc)
                raise
            try:
                result = self._worker.execute(snapshot, cancel_event)
            except WorkerSynchronizationError as exc:
                self._admission._quarantine(lease, self._worker, exc)
                raise
            except WorkerStateError as exc:
                self._admission._quarantine(lease, self._worker, exc)
                raise WorkerContractError(
                    f'worker model state or cache policy changed during execution: {_describe_error(exc)}') from exc
            except (InvalidRequest, RequestCancelled, WorkerUnavailable, WorkerExecutionError) as exc:
                if self._process_is_quarantined():
                    self._admission._quarantine(lease, self._worker, exc)
                    raise WorkerSynchronizationError(
                        f'worker process quarantined; original failure: {_describe_error(exc)}') from exc
                if (not self._binding_matches()
                        or (not self._worker.poisoned and not self._cache_matches())):
                    self._admission._quarantine(lease, self._worker, exc)
                    raise WorkerContractError('worker binding changed during execution') from exc
                self._release(lease, exc)
                raise
            except BaseException as exc:
                self._admission._quarantine(lease, self._worker, exc)
                raise
            if self._process_is_quarantined():
                self._admission._quarantine(lease, self._worker, result)
                raise WorkerSynchronizationError('worker process quarantined after execution')
            if not self._binding_matches() or not self._cache_matches():
                self._admission._quarantine(lease, self._worker, result)
                raise WorkerContractError('worker binding changed during execution')
            if (type(result) is not ForwardResult or result.request_id != snapshot.request_id
                    or type(result.logits) is not torch.Tensor
                    or result.logits.device.type != 'cpu'
                    or result.logits.requires_grad or not result.logits.is_floating_point()
                    or result.logits.ndim != 3 or tuple(result.logits.shape[:2]) != profile.shape
                    or result.logits.numel() * result.logits.element_size() > profile.output_limit_bytes):
                self._admission._quarantine(lease, self._worker, result)
                raise WorkerContractError('worker returned an unsupported or oversized result')
            self._release(lease)
            return result

    def close(self):
        """Close safely; a quarantined worker and its lease remain retained."""
        with self._admission._lock:
            if self._closed:
                return
            self._closed = True
            if self.quarantined:
                self._admission._quarantine(None, self._worker)
                return
            try:
                self._worker.close()
            except BaseException as exc:
                self._admission._quarantine(None, self._worker, exc)
                raise
