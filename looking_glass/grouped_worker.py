"""Opt-in owner-coordinated grouped execution with explicit host custody.

This module is a cooperative in-process interface, not a sandbox or an OS
memory limit. A trusted service supplies conservative profiles and retains all
input/output/exception aliases until it explicitly disposes the group.
"""
from hashlib import sha256
import ctypes
import threading
from types import MappingProxyType
import uuid

import torch

from .budgeted_worker import BudgetedWorker
from .host_budget import HostBudget, HostBusy, HostClaim, HostRequest, HostRequestCancelled
from .microbatch import BatchProfile, Microbatch, MicrobatchRequest, _validate_profile
from .resident_worker import ForwardRequest, RequestCancelled


def _identity(value, name):
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError(f'{name} must be a nonempty bounded string')
    return value


def _digests(inputs, profile):
    """Hash CPU tensor bytes without making another payload copy.

    Exclusive trusted ownership must exclude concurrent tensor mutation while
    these hashes and the later worker snapshot are taken.
    """
    result = []
    for name, _ in profile.input_schema:
        tensor = inputs[name]
        if type(tensor) is not torch.Tensor or tensor.device.type != 'cpu' or not tensor.is_contiguous():
            raise ValueError('group inputs must be contiguous CPU tensors')
        nbytes = tensor.numel() * tensor.element_size()
        if not nbytes or not tensor.data_ptr():
            raise ValueError('group input has no readable CPU storage')
        # A ctypes view hashes existing contiguous CPU storage without NumPy or
        # another payload copy. The tensor reference stays live during hashing.
        raw = (ctypes.c_ubyte * nbytes).from_address(tensor.data_ptr())
        result.append((name, sha256(raw).digest()))
    return tuple(result)


class _GroupRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.groups = {}


class GroupedWorker:
    """A trusted owner for exact, profiled grouped forwards on one worker."""

    def __init__(self, worker, profiles, *, host_budget, finite_scratch_bytes, finite_scratch_items):
        if type(worker) is not BudgetedWorker or type(host_budget) is not HostBudget:
            raise TypeError('expected exact BudgetedWorker and HostBudget objects')
        if type(profiles) not in (tuple, list) or not profiles:
            raise ValueError('profiles must be a nonempty bounded sequence')
        if type(finite_scratch_bytes) is not dict or type(finite_scratch_items) is not dict:
            raise ValueError('service must provide profiled finite-check scratch bounds')
        self._worker = worker
        self._host = host_budget
        self._profiles = {}
        self._scratch = {}
        self._scratch_items = {}
        # All grouped owners using this exact ledger share one bounded
        # recovery registry, including records interrupted before claim binding.
        with host_budget._lock:
            registry = getattr(host_budget, '_group_registry', None)
            if registry is None:
                registry = _GroupRegistry()
                host_budget._group_registry = registry
            if type(registry) is not _GroupRegistry:
                raise ValueError('host ledger group registry is incompatible')
        self._groups = registry.groups
        self._lock = registry.lock
        for profile in profiles:
            if type(profile) is not BatchProfile:
                raise ValueError('profiles must contain exact BatchProfile records')
            _validate_profile(profile)
            if profile.profile_id in self._profiles:
                raise ValueError('duplicate batch profile ID')
            if (profile.contract.model_id != worker._binding[0]
                    or profile.contract.device != str(worker._binding[1])):
                raise ValueError('profile does not match the resident worker binding')
            scratch = finite_scratch_bytes.get(profile.profile_id)
            scratch_items = finite_scratch_items.get(profile.profile_id)
            if type(scratch) is not int or scratch <= 0 or type(scratch_items) is not int or scratch_items <= 0:
                raise ValueError('each profile requires a positive trusted scratch bound')
            self._profiles[profile.profile_id] = profile
            self._scratch[profile.profile_id] = scratch
            self._scratch_items[profile.profile_id] = scratch_items
        if set(finite_scratch_bytes) != set(self._profiles) or set(finite_scratch_items) != set(self._profiles):
            raise ValueError('scratch bounds must match configured profile IDs')

    def required_host_capacity(self, profile_id):
        profile = self._profiles[profile_id]
        # Conservatively overlap broker-selected inputs, plan snapshots,
        # concatenation, wrapper and resident snapshots, full logits, and row
        # clones. The separate service-configured bound covers finite-check
        # scratch. Model-owned storage and arbitrary runtime allocations need
        # additional independent admission.
        nbytes = (5 * profile.max_logical_input_bytes
                  + 2 * self._worker._binding[5] + self._scratch[profile_id])
        items = ((2 * profile.max_requests + 3) * len(profile.input_schema)
                 + profile.max_requests + 1 + self._scratch_items[profile_id])
        return nbytes, items

    def snapshot_request(self, request):
        return self._worker.snapshot_request(request)

    def active_groups(self):
        """Return exact group capabilities for trusted interrupted-call recovery."""
        with self._lock:
            return tuple(group for group in self._groups.values() if group._owner is self)

    def prepare(self, profile_id, jobs, host_request, host_claim, group_id):
        profile = self._profiles[profile_id]
        _identity(group_id, 'group_id')
        if type(jobs) not in (list, tuple) or not 0 < len(jobs) <= profile.max_requests:
            raise ValueError('jobs must be a bounded nonempty sequence')
        for job in jobs:
            if type(job) is not MicrobatchRequest:
                raise ValueError('jobs must contain exact MicrobatchRequest records')
            for value, name in ((job.job_id, 'job_id'), (job.owner, 'owner'),
                                (job.request_id, 'request_id')):
                _identity(value, name)
            if type(job.inputs) not in (dict, MappingProxyType):
                raise ValueError('job inputs must be an ordinary mapping')
            for name, _ in profile.input_schema:
                try:
                    tensor = job.inputs[name]
                except (KeyError, TypeError):
                    raise ValueError('job inputs differ from the profile') from None
                if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu'
                        or tensor.layout != torch.strided or not tensor.is_contiguous()
                        or tensor.storage_offset() != 0
                        or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
                    raise ValueError('job inputs must have exact contiguous CPU storage')
        if (type(host_request) is not HostRequest or type(host_claim) is not HostClaim
                or host_request._budget is not self._host or host_claim._budget is not self._host):
            raise ValueError('group custody must use the configured host ledger')
        required_bytes, required_items = self.required_host_capacity(profile_id)
        with self._host._lock:
            request_record = self._host._request(host_request)
            claim_record = self._host._claim(host_claim)
            if claim_record.request is not host_request or request_record.cancelled:
                raise ValueError('claim must belong to the active group request')
            if claim_record.nbytes < required_bytes or claim_record.items < required_items:
                raise ValueError('group claim does not cover the trusted host envelope')
            if claim_record.binding is not None or self._host._operation_count(host_claim._claim_id):
                raise HostBusy('group claim is already bound or in use')
        token = uuid.uuid4().hex
        group = GroupPlan(self, profile, host_request, host_claim, group_id, token)
        with self._lock:
            if group_id in self._groups:
                raise ValueError('group ID is already active')
            if len(self._groups) >= self._host.max_items:
                raise HostBusy('group recovery registry is full')
            # The owner retains the group before binding or pinning, so an
            # interrupted return is recoverable within a bounded registry.
            self._groups[group_id] = group
        try:
            with self._host._lock:
                request_record = self._host._request(host_request)
                claim_record = self._host._claim(host_claim)
                if claim_record.request is not host_request or request_record.cancelled:
                    raise ValueError('claim must belong to the active group request')
                if claim_record.nbytes < required_bytes or claim_record.items < required_items:
                    raise ValueError('group claim does not cover the trusted host envelope')
                if claim_record.binding is not None or self._host._operation_count(host_claim._claim_id):
                    raise HostBusy('group claim is already bound or in use')
                if not host_claim.bind_once(group._binding):
                    raise HostBusy('claim was already bound')
                group._operation = host_request.begin_operation(group._operation_label, [host_claim])
            # Potentially expensive tensor copies must not hold the host ledger
            # or owner registry locks; the bound operation pins their envelope.
            group._plan = Microbatch(jobs, profile)
            group._prepared = True
        except BaseException:
            group._preparation_failed = True
            raise
        finally:
            with group._lock:
                group._preparing = False
        return group

    def execute(self, group, launch, *, cancel_event=None):
        if type(group) is not GroupPlan or group._owner is not self:
            raise ValueError('group belongs to another owner')
        token = object()
        try:
            group._begin_execution(launch, token)
            request = ForwardRequest(group.group_id, launch.inputs, seed=group._profile.contract.seed)
            return self._worker._execute_grouped(request, group, launch, cancel_event)
        except BaseException as exc:
            if group._execution is token and not group._started:
                try:
                    group.abort_unexecuted(exc)
                except BaseException:
                    # Keep the group, claims and any operation for recovery.
                    group._uncertain = True
            elif group._execution is token:
                group._uncertain = True
            raise
        finally:
            group._end_execution(token)


class GroupPlan:
    """Single-use group capability; disposal is an explicit owner attestation."""

    def __init__(self, owner, profile, request, claim, group_id, token):
        self._owner, self._profile = owner, profile
        self._request, self._claim = request, claim
        self.group_id = group_id
        self._binding = 'group:' + token
        self._operation_label = 'group-operation:' + token
        self._operation = None
        self._plan = None
        # Set before this capability is published in the recovery registry.
        self._preparing = True
        self._prepared = self._preparation_failed = False
        self._execution = None
        self._started = self._uncertain = self._disposed = False
        self._aborted = False
        self._disposal_ready = False
        self._launch = None
        self._digest = None
        self._lock = threading.Lock()

    def cancel(self, job_id):
        if not self._prepared or self._disposed:
            raise RuntimeError('group is unavailable')
        return self._plan.cancel(job_id)

    def _request_cancelled(self):
        with self._owner._host._lock:
            return self._owner._host._request(self._request).cancelled

    def launch(self):
        with self._lock:
            if not self._prepared or self._disposed or self._aborted:
                raise RuntimeError('group is unavailable')
            if self._request_cancelled():
                raise HostRequestCancelled('group request was cancelled before launch')
            launch = self._plan.launch()
            if launch is not None:
                self._digest = _digests(launch.inputs, self._profile)
                self._launch = launch
            return launch

    def pending_launch(self):
        return None if self._plan is None else self._plan.pending_launch()

    def outcome(self):
        if self._plan is None:
            raise RuntimeError('group has no prepared plan')
        return self._plan.outcome()

    def _begin_execution(self, launch, token):
        with self._lock:
            if (not self._prepared or self._disposed or self._aborted or self._uncertain
                    or self._execution is not None or self._started or launch is not self._launch):
                raise RuntimeError('group launch is not available for execution')
            self._plan.inspect_launch(launch)
            self._execution = token

    def _end_execution(self, token):
        with self._lock:
            if self._execution is token:
                self._execution = None

    def _abort_if_all_cancelled(self, launch):
        with self._lock:
            if self._execution is None or self._started or launch is not self._launch:
                raise RuntimeError('group launch is not owned by this execution')
            if self._request_cancelled():
                raise HostRequestCancelled('group request was cancelled before dispatch')
            if all(self._plan.cancellation_state(job_id) is True for job_id in launch.job_ids):
                result = self._plan.abort_unexecuted(launch, RequestCancelled('all members cancelled'))
                self._aborted = True
                return result
        return None

    def _start_under_gate(self, launch, workload, snapshot):
        # The aggregate host operation has remained pinned since prepare.
        with self._lock:
            if self._execution is None or self._started or launch is not self._launch:
                raise RuntimeError('group launch is not owned by this execution')
            if self._request_cancelled():
                raise HostRequestCancelled('group request was cancelled before dispatch')
            identity = self._plan.inspect_launch(launch)
            if identity.profile != self._profile or identity.job_ids != launch.job_ids:
                raise ValueError('launch identity differs from its service profile')
            if (workload.model_id != self._profile.contract.model_id
                    or workload.device != self._profile.contract.device
                    or workload.shape != tuple(snapshot.inputs['input_ids'].shape)
                    or workload.input_dtypes != tuple(sorted((k, str(v.dtype)) for k, v in snapshot.inputs.items()))
                    or _digests(snapshot.inputs, self._profile) != self._digest):
                raise ValueError('worker snapshot differs from the certified launch')
            # Set before claim/forward: interruption cannot be interpreted as
            # evidence that device work never started.
            self._started = True
            self._plan.claim_launch(launch)

    def _finish_under_gate(self, launch, result):
        if self._request_cancelled():
            for job_id in launch.job_ids:
                self._plan.cancel(job_id)
        return self._plan.finish(result.logits, launch_id=launch.launch_id)

    def _fail_under_gate(self, launch, error):
        return self._plan.fail(error, launch_id=launch.launch_id)

    def abort_unexecuted(self, error):
        if not isinstance(error, BaseException):
            raise ValueError('error must be an exception')
        with self._lock:
            if self._preparing or self._started or self._disposed:
                raise RuntimeError('cannot abort a started group')
            if self._plan is None:
                self._aborted = True
                return None
            if self._launch is not None:
                result = self._plan.abort_unexecuted(self._launch, error)
            else:
                result = None
            self._aborted = True
            return result

    def recover_preparation(self):
        """Attest that an interrupted prepare has stopped and all aliases are gone.

        This is a trusted recovery operation, not proof that another thread
        stopped or that an exception retained no tensors.
        """
        with self._lock:
            if not self._preparing or self._started or self._disposed:
                raise RuntimeError('group has no interrupted preparation to recover')
            self._preparation_failed = True
            self._aborted = True
            self._preparing = False

    def dispose(self):
        """Attest all covered aliases are gone, then finish the pin and release.

        The caller must have discarded every plan/launch/outcome/traceback and
        consumer tensor alias. This is a cooperative declaration, not proof.
        """
        with self._owner._lock:
            with self._lock:
                if self._disposed:
                    if self._owner._groups.get(self.group_id) is self:
                        del self._owner._groups[self.group_id]
                    return False
                if self._preparing or self._execution is not None or self._uncertain:
                    raise HostBusy('group preparation, execution or cleanup is uncertain')
                if not self._disposal_ready:
                    if self._prepared and not self._aborted:
                        self._plan.outcome()  # Must be terminal before disposal.
                    self._disposal_ready = True
                self._plan = self._launch = self._digest = None
                # Recover a pin if interruption happened after its registration
                # but before assignment to this group.
                if self._operation is None:
                    with self._owner._host._lock:
                        matches = [record.handle for record in self._owner._host._operations.values()
                                   if record.request is self._request and record.label == self._operation_label]
                    if len(matches) > 1:
                        raise HostBusy('ambiguous group operation')
                    if matches:
                        self._operation = matches[0]
                if self._operation is not None:
                    with self._owner._host._lock:
                        record = self._owner._host._operations.get(self._operation._operation_id)
                        if record is not None:
                            if record.handle is not self._operation:
                                raise HostBusy('group operation identity changed')
                            self._operation.finish()
                        self._operation = None
                with self._owner._host._lock:
                    record = self._owner._host._claims.get(self._claim._claim_id)
                    if record is not None:
                        if record.handle is not self._claim:
                            raise HostBusy('group claim identity changed')
                        if record.binding == self._binding:
                            self._claim.release()
                self._disposed = True
                if self._owner._groups.get(self.group_id) is self:
                    del self._owner._groups[self.group_id]
        return True
