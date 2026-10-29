"""Bounded assembly and owner-isolated splitting of compatible eval forwards."""

from dataclasses import dataclass
import math
from numbers import Real
import re
import threading
from types import MappingProxyType
from typing import Mapping
import uuid

import torch

from .resident_worker import WorkerOutOfMemory

_DTYPE_BYTES = {torch.bool: 1, torch.int32: 4, torch.int64: 8,
                torch.float16: 2, torch.bfloat16: 2, torch.float32: 4, torch.float64: 8}


@dataclass(frozen=True)
class ForwardContract:
    model_id: str
    checkpoint: str
    adapter: str
    device: str
    model_dtype: torch.dtype
    runtime_id: str
    seed: int
    deterministic: bool
    batch_separable: bool
    mode: str
    rng_policy: str
    cache_policy: str
    mutable_state_policy: str
    tracing: bool
    intervention: bool


@dataclass(frozen=True)
class BatchProfile:
    profile_id: str
    certification_id: str
    contract: ForwardContract
    input_schema: tuple[tuple[str, torch.dtype], ...]
    input_shape: tuple[int, int]
    output_dtype: torch.dtype
    output_width: int
    serial_atol: float
    serial_rtol: float
    approved_rows: tuple[int, ...]
    max_requests: int
    max_per_owner: int
    max_batch_rows: int
    max_token_positions: int
    max_logical_input_bytes: int
    max_logical_output_bytes: int
    max_logical_total_bytes: int


@dataclass(frozen=True)
class MicrobatchRequest:
    job_id: str
    owner: str
    request_id: str
    contract: ForwardContract
    inputs: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class BatchLaunch:
    profile_id: str
    launch_id: str
    job_ids: tuple[str, ...]
    inputs: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class LaunchIdentity:
    profile: BatchProfile
    profile_id: str
    certification_id: str
    launch_id: str
    job_ids: tuple[str, ...]


@dataclass(frozen=True)
class RequestResult:
    job_id: str
    owner: str
    request_id: str
    status: str
    logits: torch.Tensor | None = None
    reason: str | None = None


@dataclass(frozen=True)
class BatchOutcome:
    results: tuple[RequestResult, ...]
    quarantine_required: bool = False


class Microbatch:
    def __init__(self, requests, profile):
        try:
            self._initialize(requests, profile)
        except BaseException as exc:
            # A retained exception must not retain the constructor's input
            # iterator, partial snapshots, or failed clone frames.
            self = requests = profile = None
            _detach_exception(exc)
            raise exc from None

    def _initialize(self, requests, profile):
        _validate_profile(profile)
        try:
            iterator = iter(requests)
        except TypeError:
            raise ValueError('requests must be an iterable') from None
        bounded = []
        for job in iterator:
            bounded.append(job)
            if len(bounded) > profile.max_requests:
                raise ValueError('request limit exceeded')
        requests = tuple(bounded)
        if not requests:
            raise ValueError('at least one request is required')
        seen_jobs, seen_requests, owners = set(), set(), {}
        expected_names = {name for name, _ in profile.input_schema}
        for job in requests:
            if not isinstance(job, MicrobatchRequest):
                raise ValueError('requests must contain MicrobatchRequest records')
            for value in (job.job_id, job.owner, job.request_id):
                if type(value) is not str or not value.strip():
                    raise ValueError('job ID, owner, and request ID must be nonempty strings')
            if job.job_id in seen_jobs or (job.owner, job.request_id) in seen_requests:
                raise ValueError('duplicate job or owner/request identity')
            seen_jobs.add(job.job_id)
            seen_requests.add((job.owner, job.request_id))
            owners[job.owner] = owners.get(job.owner, 0) + 1
            if job.contract != profile.contract:
                raise ValueError('request execution contract differs from profile')
            if type(job.inputs) not in (dict, MappingProxyType) or set(job.inputs) != expected_names:
                raise ValueError('request input names differ from profile')
            for name, dtype in profile.input_schema:
                value = job.inputs[name]
                if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                        or value.layout != torch.strided or value.dtype != dtype
                        or tuple(value.shape) != profile.input_shape or value.requires_grad):
                    raise ValueError(f'invalid input tensor: {name}')
        if any(count > profile.max_per_owner for count in owners.values()):
            raise ValueError('per-owner request limit exceeded')
        _check_limits(profile, len(requests))
        self._profile = profile
        # Retain only immutable identities and owned snapshots, not client dictionaries.
        self._identities = tuple((job.job_id, job.owner, job.request_id) for job in requests)
        self._job_ids = frozenset(seen_jobs)
        with torch.inference_mode(False), torch.no_grad():
            self._snapshots = {job.job_id: {name: job.inputs[name].detach().clone()
                                           for name, _ in profile.input_schema} for job in requests}
        self._cancelled = set()
        self._state = 'planned'
        self._result = None
        self._active = ()
        self._launch_id = None
        self._launch = None
        self._launch_versions = ()
        self._launch_claimed = False
        self._publication_quarantine = False
        self._lock = threading.Lock()

    @property
    def publication_failed(self):
        with self._lock:
            return self._state == 'unpublishable'

    @property
    def publication_quarantine_required(self):
        with self._lock:
            return self._publication_quarantine

    def cancel(self, job_id):
        with self._lock:
            if type(job_id) is not str or job_id not in self._job_ids:
                raise KeyError(job_id)
            accepted = self._state in ('planned', 'launched') and job_id not in self._cancelled
            if accepted:
                try:
                    self._cancelled.add(job_id)
                    if self._state == 'planned':
                        self._snapshots.pop(job_id, None)
                except BaseException:
                    self._mark_unpublishable(False)
                    raise
        return accepted

    def cancellation_state(self, job_id):
        """Return a recorded cancellation or None when publication is uncertain."""
        with self._lock:
            if type(job_id) is not str or job_id not in self._job_ids:
                raise KeyError(job_id)
            cancelled = None if self._state == 'unpublishable' else job_id in self._cancelled
        return cancelled

    def launch(self):
        started = False
        inputs = launch = None
        try:
            with self._lock:
                if self._state != 'planned':
                    raise RuntimeError('batch has already launched or finished')
                started = True
                self._active = tuple(identity[0] for identity in self._identities
                                     if identity[0] not in self._cancelled)
                if not self._active:
                    self._complete({}, None, False)
                else:
                    rows = len(self._active) * self._profile.input_shape[0]
                    if rows not in self._profile.approved_rows:
                        self._complete({}, 'unprofiled_batch', False)
                        raise ValueError('combined batch size lacks an approved workload profile')
                    with torch.inference_mode(False), torch.no_grad():
                        inputs = {name: torch.cat([self._snapshots[job_id][name] for job_id in self._active], dim=0)
                                  for name, _ in self._profile.input_schema}
                    launch_id = uuid.uuid4().hex
                    launch = BatchLaunch(self._profile.profile_id, launch_id, self._active, MappingProxyType(inputs))
                    versions = tuple((name, id(inputs[name]), inputs[name]._version)
                                     for name, _ in self._profile.input_schema)
                    self._launch_id = launch_id
                    self._launch = launch
                    self._launch_versions = versions
                    self._state = 'launched'
            return launch
        except BaseException as exc:
            try:
                with self._lock:
                    if started and self._state in ('planned', 'launched'):
                        if self._state == 'launched' and self._launch_claimed:
                            # An independently claimed handoff may have begun work.
                            self._mark_unpublishable(True)
                        else:
                            # No launch was claimed and no worker forward was issued.
                            self._state = 'planned'
                            self._launch_id = None
                            self._launch = None
                            self._launch_versions = ()
                            self._complete_failure(exc, False)
            except BaseException as reporting_error:
                inputs = launch = None
                _detach_exception(exc)
                _detach_exception(reporting_error)
                raise reporting_error from exc
            inputs = launch = None
            _detach_exception(exc)
            raise exc from None

    def pending_launch(self):
        """Recover the same launch object after an ambiguous trusted-owner handoff."""
        with self._lock:
            launch = self._launch if self._state == 'launched' else None
        return launch

    @property
    def launch_claimed(self):
        with self._lock:
            claimed = self._launch_claimed
        return claimed

    def _inspect_launch_locked(self, launch):
        if self._state != 'launched':
            raise RuntimeError('batch is not awaiting a launched forward')
        if type(launch) is not BatchLaunch or launch is not self._launch:
            raise ValueError('launch is not the exact plan-owned capability')
        try:
            versions = tuple((name, id(launch.inputs[name]), launch.inputs[name]._version)
                             for name, _ in self._profile.input_schema)
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError('launch inputs cannot be validated') from exc
        if versions != self._launch_versions:
            raise ValueError('launch inputs changed after assembly')
        return LaunchIdentity(self._profile, self._profile.profile_id, self._profile.certification_id,
                              self._launch_id, self._active)

    def inspect_launch(self, launch):
        """Validate the live launch and return immutable identity metadata."""
        with self._lock:
            identity = self._inspect_launch_locked(launch)
        return identity

    def claim_launch(self, launch):
        """Validate and consume a launch capability once before worker dispatch."""
        with self._lock:
            identity = self._inspect_launch_locked(launch)
            if self._launch_claimed:
                raise RuntimeError('launch capability has already been claimed')
            self._launch_claimed = True
        return identity

    def abort_unexecuted(self, launch, error):
        """Finish before device work; the trusted owner must establish no forward was issued."""
        with self._lock:
            if self._state != 'launched':
                raise RuntimeError('batch is not awaiting a launched forward')
            if type(launch) is not BatchLaunch or launch is not self._launch:
                raise ValueError('launch is not the exact plan-owned capability')
            if not isinstance(error, BaseException):
                raise ValueError('error must be an exception')
            try:
                self._state = 'planned'
                self._launch = None
                self._launch_id = None
                self._launch_versions = ()
                result = self._complete_failure(error, False)
            except BaseException as exc:
                if self._state != 'unpublishable':
                    self._mark_unpublishable(False)
                raise exc from error
        return result

    def finish(self, logits, *, launch_id):
        outputs = None
        with self._lock:
            if self._state != 'launched':
                raise RuntimeError('batch is not awaiting a launched forward')
            try:
                if launch_id != self._launch_id:
                    return self._complete({}, 'invalid_result', True)
                rows_per_job, sequence = self._profile.input_shape
                shape = (len(self._active) * rows_per_job, sequence, self._profile.output_width)
                if (type(logits) is not torch.Tensor or logits.device.type != 'cpu'
                        or logits.layout != torch.strided or tuple(logits.shape) != shape
                        or logits.dtype != self._profile.output_dtype or logits.requires_grad
                        or logits.grad_fn is not None):
                    return self._complete({}, 'invalid_result', True)
                finite = bool(torch.isfinite(logits).all())
                outputs = {}
                if finite:
                    with torch.inference_mode(False), torch.no_grad():
                        outputs = {job_id: logits[index * rows_per_job:(index + 1) * rows_per_job].detach().clone()
                                   for index, job_id in enumerate(self._active) if job_id not in self._cancelled}
                if not finite:
                    return self._complete({}, 'invalid_result', True)
                return self._complete(outputs, None, False)
            except BaseException as exc:
                # Failed split/publication must not pin logits through retained
                # exception frames. The caller still owns any other references.
                logits = None
                if outputs is not None:
                    outputs.clear()
                _detach_exception(exc)
                if self._state != 'launched':
                    raise exc from None
                result = self._complete_failure(exc, True)
                if not isinstance(exc, Exception):
                    raise exc from None
                return result

    def fail(self, error, *, launch_id):
        with self._lock:
            if self._state != 'launched':
                raise RuntimeError('batch is not awaiting a launched forward')
            try:
                if launch_id != self._launch_id:
                    return self._complete({}, 'invalid_result', True)
                if not isinstance(error, BaseException):
                    raise ValueError('error must be an exception')
                return self._complete_failure(error, True)
            except BaseException as exc:
                if self._state != 'unpublishable':
                    self._mark_unpublishable(True)
                if isinstance(error, BaseException):
                    raise exc from error
                raise

    def outcome(self):
        with self._lock:
            if self._state == 'unpublishable':
                raise RuntimeError('batch outcome could not be published; broker recovery is required')
            if self._state != 'finished':
                raise RuntimeError('batch has not finished')
            return self._result

    def _complete(self, outputs, reason, quarantine):
        was_launched = self._state == 'launched'
        records = []
        result = None
        try:
            for job_id, owner, request_id in self._identities:
                if job_id in self._cancelled:
                    records.append(RequestResult(job_id, owner, request_id, 'cancelled'))
                elif reason is not None:
                    records.append(RequestResult(job_id, owner, request_id, 'failed', reason=reason))
                else:
                    records.append(RequestResult(job_id, owner, request_id, 'ok', logits=outputs[job_id]))
            result = BatchOutcome(tuple(records), quarantine)
            self._snapshots.clear()
            self._launch = None
            self._launch_versions = ()
            self._result = result
            self._state = 'finished'
            return result
        except BaseException as exc:
            self._mark_unpublishable(quarantine or was_launched)
            outputs.clear()
            records.clear()
            result = None
            _detach_exception(exc)
            raise exc from None

    def _complete_failure(self, error, quarantine):
        try:
            return self._complete({}, _failure_reason(error), quarantine)
        except BaseException:
            if self._state != 'unpublishable':
                self._mark_unpublishable(quarantine)
            raise

    def _mark_unpublishable(self, quarantine):
        self._publication_quarantine = quarantine or self._state == 'launched'
        self._result = None
        self._state = 'unpublishable'
        self._snapshots.clear()
        self._launch = None
        self._launch_versions = ()


def _detach_exception(error):
    """Drop common chained tracebacks; arbitrary exception-owned aliases may remain."""
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None


def _failure_reason(error):
    return 'out_of_memory' if isinstance(error, (MemoryError, torch.OutOfMemoryError, WorkerOutOfMemory)) else 'execution_failed'


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _validate_profile(profile):
    if not isinstance(profile, BatchProfile):
        raise ValueError('a BatchProfile is required')
    for value in (profile.profile_id, profile.certification_id):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('profile and certification IDs must be nonempty')
    contract = profile.contract
    if not isinstance(contract, ForwardContract):
        raise ValueError('a ForwardContract is required')
    if any(not isinstance(value, str) or not value.strip()
           for value in (contract.model_id, contract.checkpoint, contract.adapter, contract.runtime_id)):
        raise ValueError('model, checkpoint, adapter, and runtime identities are required')
    if not isinstance(contract.device, str) or not (contract.device == 'cpu' or re.fullmatch(r'cuda:[0-9]+', contract.device)):
        raise ValueError('device must be cpu or an indexed CUDA device')
    float_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if contract.model_dtype not in float_dtypes or profile.output_dtype not in float_dtypes:
        raise ValueError('model and output dtypes must be supported floating types')
    if type(contract.seed) is not int or not 0 <= contract.seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if (contract.deterministic is not True or contract.batch_separable is not True
            or contract.mode != 'eval' or contract.rng_policy != 'none'
            or contract.cache_policy != 'disabled' or contract.mutable_state_policy != 'none'
            or contract.tracing is not False or contract.intervention is not False):
        raise ValueError('only certified separable deterministic eval forwards without RNG, tracing or interventions are supported')
    if (not isinstance(profile.input_shape, tuple) or len(profile.input_shape) != 2
            or any(type(value) is not int or value <= 0 for value in profile.input_shape)):
        raise ValueError('input shape must be two positive integers')
    if not isinstance(profile.input_schema, tuple) or not profile.input_schema:
        raise ValueError('input schema must be a nonempty tuple')
    allowed = {'input_ids': (torch.int32, torch.int64),
               'attention_mask': (torch.bool, torch.int32, torch.int64),
               'position_ids': (torch.int32, torch.int64),
               'token_type_ids': (torch.int32, torch.int64)}
    names = []
    for entry in profile.input_schema:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError('invalid input schema entry')
        name, dtype = entry
        if not isinstance(name, str) or name not in allowed or dtype not in allowed[name] or name in names:
            raise ValueError('unsupported or duplicate input schema entry')
        names.append(name)
    if 'input_ids' not in names:
        raise ValueError('input_ids are required')
    for name in ('output_width', 'max_requests', 'max_per_owner', 'max_batch_rows',
                 'max_token_positions', 'max_logical_input_bytes', 'max_logical_output_bytes',
                 'max_logical_total_bytes'):
        _positive_integer(getattr(profile, name), name)
    for name in ('serial_atol', 'serial_rtol'):
        value = getattr(profile, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f'{name} must be finite and nonnegative')
        try:
            finite = math.isfinite(value)
        except (OverflowError, ValueError):
            raise ValueError(f'{name} must be finite and nonnegative') from None
        if not finite or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if (not isinstance(profile.approved_rows, tuple) or not profile.approved_rows
            or any(type(value) is not int or value <= 0 or value > profile.max_batch_rows
                   or value % profile.input_shape[0] for value in profile.approved_rows)
            or len(set(profile.approved_rows)) != len(profile.approved_rows)):
        raise ValueError('approved rows must be unique positive profiled batch sizes within the cap')


def _check_limits(profile, request_count):
    rows = request_count * profile.input_shape[0]
    tokens = rows * profile.input_shape[1]
    input_bytes = tokens * sum(_DTYPE_BYTES[dtype] for _, dtype in profile.input_schema)
    output_bytes = tokens * profile.output_width * _DTYPE_BYTES[profile.output_dtype]
    if (request_count > profile.max_requests or rows > profile.max_batch_rows
            or tokens > profile.max_token_positions or input_bytes > profile.max_logical_input_bytes
            or output_bytes > profile.max_logical_output_bytes
            or input_bytes + output_bytes > profile.max_logical_total_bytes):
        raise ValueError('combined request exceeds the profile limits')
