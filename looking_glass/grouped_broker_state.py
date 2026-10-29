"""One-commit cohort publication and cancellation for a trusted broker.

This module does not execute models or manage memory claims. The coordinator
must retain custody of all data and failures until verified disposal.
"""

from dataclasses import dataclass
import math
from threading import Condition, Lock
from time import monotonic

import torch

from .microbatch import (BatchOutcome, BatchProfile, ForwardContract, RequestResult,
                         _check_limits, _validate_profile)
from .grouped_observation import GroupedObservation


class CohortUnavailable(RuntimeError):
    """Publication failed and its status cannot safely be inferred."""


class CohortRetired(RuntimeError):
    """The trusted broker retired this publication after disposal."""


@dataclass(frozen=True, slots=True)
class CohortMember:
    job_id: str
    owner: str
    request_id: str


@dataclass(frozen=True, slots=True)
class MemberCompletion:
    member: CohortMember
    status: str
    cancel_requested: bool
    logits: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class CohortPublication:
    cohort_id: str
    launch_committed: bool
    error_code: str | None
    quarantine_required: bool
    members: tuple[MemberCompletion, ...]


@dataclass(frozen=True, slots=True)
class _Open:
    launch_committed: bool
    cancel_mask: int


_UNKNOWN_BEFORE = object()
_UNKNOWN_AFTER = object()
_RETIRED = object()


def _name(value, limit=128):
    return type(value) is str and bool(value) and len(value) <= limit


def _valid_owned_logits(value):
    if (type(value) is not torch.Tensor or value.device.type != 'cpu'
            or value.layout != torch.strided or value.ndim != 3
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            or value.numel() <= 0 or not value.is_contiguous()
            or value.requires_grad or value.grad_fn is not None or value.is_inference()
            or value.storage_offset() != 0):
        return False
    return value.untyped_storage().nbytes() == value.numel() * value.element_size()


class CohortState:
    """Coordinate one cohort; an individual broker handle must expose only its row.

    Exactly one dispatcher owns launch, publication and retirement. Concurrent
    handle cancellations are serialized with those decisions. Inputs, logits,
    exceptions and host/device reservations remain the coordinator's custody.
    """

    def __init__(self, cohort_id, members, profile, *, observation=None):
        if observation is not None and type(observation) is not GroupedObservation:
            raise ValueError('an exact trusted observation is required')
        if not _name(cohort_id) or type(members) is not tuple or not 2 <= len(members) <= 256:
            raise ValueError('a bounded, identified cohort of at least two members is required')
        if type(profile) is not BatchProfile or type(profile.contract) is not ForwardContract:
            raise ValueError('an exact batch profile is required')
        _validate_profile(profile)
        if len(members) * profile.input_shape[0] not in profile.approved_rows:
            raise ValueError('cohort row count is not approved by the profile')
        _check_limits(profile, len(members))
        if any(type(member) is not CohortMember or not _name(member.job_id)
               or not _name(member.owner) or not _name(member.request_id, 256)
               for member in members):
            raise ValueError('members require exact nonempty bounded identities')
        if len({member.job_id for member in members}) != len(members):
            raise ValueError('job IDs must be unique')
        if len({member.request_id for member in members}) != len(members):
            raise ValueError('worker request IDs must be unique')
        if any(sum(other.owner == member.owner for other in members) > profile.max_per_owner
               for member in members):
            raise ValueError('cohort exceeds the per-owner profile limit')
        self._cohort_id = cohort_id
        self._members = members
        self._output_shape = (*profile.input_shape, profile.output_width)
        self._output_dtype = profile.output_dtype
        self._indices = {member.job_id: i for i, member in enumerate(members)}
        self._condition = Condition(Lock())
        self._state = _Open(False, 0)
        self._observation = observation
        self._published_ns = None

    def _index(self, job_id):
        if type(job_id) is not str or job_id not in self._indices:
            raise ValueError('job is not a member of this cohort')
        return self._indices[job_id]

    @staticmethod
    def _check_state(state):
        if state is _UNKNOWN_BEFORE or state is _UNKNOWN_AFTER:
            raise CohortUnavailable('cohort publication is unknown')
        if state is _RETIRED:
            raise CohortRetired('cohort publication has been retired')

    @property
    def phase(self):
        with self._condition:
            state = self._state
            if state is _UNKNOWN_BEFORE or state is _UNKNOWN_AFTER:
                phase = 'unknown'
            elif state is _RETIRED:
                phase = 'retired'
            elif type(state) is CohortPublication:
                phase = 'terminal'
            else:
                phase = 'launch_committed' if state.launch_committed else 'staged'
        return phase

    @property
    def quarantine_required(self):
        """Whether an in-flight device or published batch needs quarantine."""
        with self._condition:
            state = self._state
            required = (state is _UNKNOWN_AFTER or
                        (type(state) is CohortPublication and state.quarantine_required))
        return required

    def cancellation_requested(self, job_id):
        index = self._index(job_id)
        with self._condition:
            state = self._state
            if state is _UNKNOWN_BEFORE or state is _UNKNOWN_AFTER:
                value = None
            elif state is _RETIRED:
                raise CohortRetired('cohort publication has been retired')
            elif type(state) is CohortPublication:
                value = state.members[index].cancel_requested
            else:
                value = bool(state.cancel_mask & (1 << index))
        return value

    def cancel(self, job_id):
        """Accept a cancellation request once, until terminal publication."""
        index = self._index(job_id)
        bit = 1 << index
        with self._condition:
            state = self._state
            self._check_state(state)
            if type(state) is CohortPublication or state.cancel_mask & bit:
                accepted = False
            else:
                self._state = _Open(state.launch_committed, state.cancel_mask | bit)
                accepted = True
        return accepted

    def try_commit_launch(self):
        """Commit once, unless a prelaunch cancellation already won."""
        with self._condition:
            state = self._state
            self._check_state(state)
            if type(state) is CohortPublication or state.launch_committed:
                raise RuntimeError('launch decision was already made')
            if state.cancel_mask:
                committed = False
            else:
                self._state = _Open(True, 0)
                committed = True
        return committed

    def mark_unknown(self):
        """Fail closed for an external uncertain handoff; never erase a commit."""
        with self._condition:
            state = self._state
            if type(state) is _Open:
                self._state = _UNKNOWN_AFTER if state.launch_committed else _UNKNOWN_BEFORE
                changed = True
            else:
                changed = False
        return changed

    def finish_cancelled_before_launch(self):
        return self._publish('prelaunch_cancel', None, False)

    def publish_success(self, outcome):
        """Publish a trusted, plan-issued complete split outcome in one write."""
        return self._publish('success', outcome, False)

    def publish_failure(self, error_code, *, quarantine_required):
        if not _name(error_code) or type(quarantine_required) is not bool:
            raise ValueError('a bounded failure code and boolean quarantine flag are required')
        return self._publish('failure', error_code, quarantine_required)

    def _publish(self, kind, data, quarantine):
        try:
            with self._condition:
                old = self._state
                self._check_state(old)
                if type(old) is not _Open:
                    raise RuntimeError('cohort was already published')
                if kind == 'success' and not old.launch_committed:
                    raise RuntimeError('cannot publish successful results before launch')
                if kind == 'prelaunch_cancel' and (old.launch_committed or not old.cancel_mask):
                    raise RuntimeError('a prelaunch cancellation is required')
                if kind == 'failure' and old.launch_committed and not quarantine:
                    raise ValueError('a failure after launch requires quarantine')
                if kind == 'success':
                    self._validate_outcome(data)
                completions = []
                for index, member in enumerate(self._members):
                    cancelled = bool(old.cancel_mask & (1 << index))
                    if kind == 'success':
                        status = 'cancelled' if cancelled else 'ok'
                        logits = None if cancelled else data.results[index].logits
                    elif kind == 'prelaunch_cancel':
                        status, logits = ('cancelled' if cancelled else 'failed'), None
                    else:
                        status, logits = 'failed', None
                    completions.append(MemberCompletion(member, status, cancelled, logits))
                error = (None if kind == 'success' else
                         'cohort_cancelled_before_launch' if kind == 'prelaunch_cancel' else data)
                publication = CohortPublication(self._cohort_id, old.launch_committed, error,
                                                quarantine, tuple(completions))
                if self._observation is not None:
                    self._published_ns = self._observation._stamp()
                self._state = publication
                self._condition.notify_all()
        except BaseException:
            # The outer exception region begins before acquiring the lock. A
            # single state pointer is the publication commit: mark_unknown
            # preserves a publication that was already committed.
            self.mark_unknown()
            raise
        return publication

    def _validate_outcome(self, outcome):
        if (type(outcome) is not BatchOutcome or type(outcome.results) is not tuple
                or type(outcome.quarantine_required) is not bool or outcome.quarantine_required
                or len(outcome.results) != len(self._members)):
            raise ValueError('invalid successful cohort outcome')
        regions = []
        for member, result in zip(self._members, outcome.results):
            if (type(result) is not RequestResult or result.job_id != member.job_id
                    or result.owner != member.owner or result.request_id != member.request_id
                    or result.status != 'ok' or result.reason is not None
                    or not _valid_owned_logits(result.logits)
                    or tuple(result.logits.shape) != self._output_shape
                    or result.logits.dtype != self._output_dtype
                    or not bool(torch.isfinite(result.logits).all())):
                raise ValueError('split result identity, status or ownership mismatch')
            storage = result.logits.untyped_storage()
            start = storage.data_ptr()
            stop = start + storage.nbytes()
            if any(start < end and prior < stop for prior, end in regions):
                raise ValueError('split result storage overlaps another member')
            regions.append((start, stop))

    def publication(self):
        with self._condition:
            state = self._state
            self._check_state(state)
            if type(state) is _Open:
                raise RuntimeError('cohort has not been published')
        return state

    def wait(self, timeout=None):
        if timeout is not None:
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
                raise ValueError('timeout must be a finite nonnegative number or None')
            deadline = monotonic() + timeout
        else:
            deadline = None
        with self._condition:
            while type(self._state) is _Open:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError('cohort has not been published')
                # A notification can fail after the pointer commits. Periodic
                # rechecks avoid depending on notification for visibility.
                self._condition.wait(0.1 if remaining is None else min(0.1, remaining))
            state = self._state
            self._check_state(state)
        return state

    def retire(self):
        """Drop this state object's result references after trusted disposal."""
        with self._condition:
            state = self._state
            self._check_state(state)
            if type(state) is not CohortPublication:
                raise RuntimeError('only a terminal cohort can be retired')
            if state.quarantine_required:
                raise CohortUnavailable('a quarantined cohort requires process teardown')
            self._state = _RETIRED
        return None
