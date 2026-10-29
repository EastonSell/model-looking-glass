"""Serial, cooperative in-process clients for one bounded CPU trace owner.

This is not IPC, authentication, a physical memory limit, or a GPU service.
The owner and every client must obey the trace and host-custody contracts.
"""
from dataclasses import dataclass
import threading
import uuid

import torch

from .cpu_trace_owner import CpuTraceOwner
from .host_budget import HostBudgetExceeded
from .producer_slot import ProducerSlotError
from .resident_worker import ForwardRequest
from .trace_batch_handoff import SealedBatchRegistry
from .trace_job_validation import _strip_failure_frames
from .trace_producer import ClaimLimit, ProducerProfile
from .traced_broker_worker import BrokerJobIdentity, TraceLimits, TraceSpec, bind_trace_plan


class SharedTraceError(RuntimeError):
    """The shared trace service is closed, busy, or requires supervision."""


class PostForwardCapacityError(HostBudgetExceeded):
    """The CPU forward returned, but trace delivery admission was denied."""


def _text(value):
    return type(value) is str and 0 < len(value) <= 128 and bool(value.strip())


@dataclass(frozen=True, slots=True)
class TraceReceipt:
    broker_id: str
    process_generation: str
    job_id: int
    generation: int
    team_id: str
    request_id: str
    run_id: str
    event_count: int
    trace_bytes: int


class _Attempt:
    def __init__(self, team_id, request_id, inputs, input_owner, input_claim, on_event):
        self.team_id, self.request_id = team_id, request_id
        self.inputs, self.input_owner, self.input_claim = inputs, input_owner, input_claim
        self.on_event = on_event
        self.job_id = self.generation = None
        self.registration_id = self.transfer_id = self.run_id = None
        self.job = self.bridge = self.recipient = None
        self.exposure_request_guard = self.exposure_request = None
        self.exposure_claim_guard = self.exposure_claim = self.exposed_payload = None
        self.callback_interruption = None
        self.stage = 'waiting'
        self.receipt = None


class CpuSharedTraceService:
    """One exact CPU owner, one active job, and two trusted team capabilities."""

    def __init__(self, *, owner, profile, limits, team_ids):
        if (type(owner) is not CpuTraceOwner or type(profile) is not ProducerProfile
                or type(limits) is not TraceLimits or type(team_ids) is not tuple
                or len(team_ids) != 2 or any(not _text(team) for team in team_ids)
                or len(set(team_ids)) != 2):
            raise ValueError('expected an owner, policy and two distinct plain team IDs')
        limits = TraceLimits(limits.max_batch_size, limits.max_sequence_length, limits.max_layers,
                             limits.max_sites, limits.max_events, limits.max_trace_bytes,
                             limits.max_token_positions)
        stages = (profile.input_copies, profile.output, profile.trace_bundle, profile.workspace)
        if any(type(stage) is not ClaimLimit for stage in stages):
            raise ValueError('invalid producer policy')
        profile = ProducerProfile(*(ClaimLimit(stage.nbytes, stage.items) for stage in stages),
                                  device=profile.device)
        if (type(limits.max_events) is not int or not 1 <= limits.max_events <= min(owner._max_slots, 256)
                or type(limits.max_trace_bytes) is not int or limits.max_trace_bytes < 1):
            raise ValueError('invalid service trace limits')
        self._owner, self._profile, self._limits = owner, profile, limits
        self._condition = threading.Condition()
        self._clients = {team: _Client(self, team) for team in team_ids}
        self._pending = set()
        self._receipts = {}
        self._acknowledged = {}
        self._active = None
        self._thread_id = None
        self._generation = 0
        self._closed = self._quarantined = False
        self._handoffs = SealedBatchRegistry(
            budget=owner._budget, broker_id=owner._broker_id,
            process_generation=owner._process_generation, max_jobs=1, max_events=limits.max_events)
        # The CPU owner has its own monotonic generation. Bind a fresh owner
        # once for its lifetime, including after this service closes.
        with owner._lock:
            if (owner._job is not None or owner._generation != 0
                    or getattr(owner, '_shared_trace_service', None) is not None):
                raise ValueError('CPU trace owner is already used or bound')
            owner._shared_trace_service = self

    def client(self, team_id):
        if not _text(team_id):
            raise SharedTraceError('unknown client')
        with self._condition:
            if self._closed or team_id not in self._clients:
                raise SharedTraceError('unknown client or service closed')
            return self._clients[team_id]

    @property
    def quarantined(self):
        with self._condition:
            return self._quarantined

    def _admit(self, attempt):
        team = attempt.team_id
        with self._condition:
            if (self._closed or self._quarantined or team in self._pending or team in self._receipts
                    or (self._active is not None and self._active.team_id == team)):
                raise SharedTraceError('client or service is unavailable')
            if self._thread_id == threading.get_ident():
                raise SharedTraceError('a trace callback cannot reenter the service')
            try:
                self._pending.add(team)
                while self._active is not None and not self._closed and not self._quarantined:
                    self._condition.wait()
                if self._closed or self._quarantined:
                    raise SharedTraceError('service is unavailable')
                self._generation += 1
                if self._generation >= 2**63:
                    raise SharedTraceError('job generations exhausted')
                attempt.job_id = attempt.generation = self._generation
                attempt.registration_id, attempt.transfer_id = uuid.uuid4().hex, uuid.uuid4().hex
                attempt.run_id = f'run-{self._owner._broker_id}-{attempt.job_id}'
                attempt.stage = 'admitted'
                self._active = attempt
                self._thread_id = threading.get_ident()
            finally:
                self._pending.discard(team)

    def _release(self, attempt, *, receipt=None, quarantine=False):
        with self._condition:
            if self._active is not attempt:
                self._condition.notify_all()
                return
            if quarantine:
                self._quarantined = True
            else:
                if receipt is not None:
                    self._receipts[attempt.team_id] = receipt
                self._thread_id = None
                self._active = None
            self._condition.notify_all()

    def _submit(self, team_id, *, request_id, inputs, input_owner, input_claim,
                samples, sites, on_event):
        if not _text(request_id) or type(inputs) is not dict or not callable(on_event):
            raise ValueError('invalid request identity, inputs or event sink')
        ids = inputs.get('input_ids')
        if type(ids) is not torch.Tensor or ids.ndim != 2:
            raise ValueError('input_ids must be a two-dimensional tensor')
        attempt = _Attempt(team_id, request_id, inputs, input_owner, input_claim, on_event)
        try:
            self._admit(attempt)
            receipt = self._execute(attempt, samples, sites)
            attempt.receipt = receipt
            attempt.stage = 'completed'
            self._release(attempt, receipt=receipt)
            return receipt
        except BaseException as error:
            # A known pre-forward failure is disposed explicitly by _execute.
            # Every other exposed or ambiguous transition remains rooted.
            clean = (attempt.stage == 'clean_rejection'
                     or (attempt.stage == 'clean_postforward_failure'
                         and type(error) is PostForwardCapacityError))
            self._release(attempt, quarantine=not clean)
            raise

    def _execute(self, attempt, samples, sites):
        owner = self._owner
        ids = attempt.inputs['input_ids']
        batch_size, sequence_length = ids.shape
        identity = BrokerJobIdentity(owner._broker_id, attempt.job_id, attempt.team_id, attempt.request_id)
        spec = TraceSpec(attempt.run_id, samples, sites, 0, 0, None,
                         self._limits.max_events, self._limits.max_trace_bytes)
        try:
            plan = bind_trace_plan(identity, spec, self._limits, batch_size=batch_size,
                                   sequence_length=sequence_length,
                                   layer_count=len(owner._model.transformer.h))
            owner.register(plan, profile=self._profile, registration_id=attempt.registration_id,
                           generation=attempt.generation, batch_size=batch_size,
                           sequence_length=sequence_length)
            attempt.job = owner.lookup(attempt.job_id, registration_id=attempt.registration_id,
                                       generation=attempt.generation)
            attempt.stage = 'registered'
            try:
                attempt.job.prepare()
            except (HostBudgetExceeded, ProducerSlotError) as error:
                if attempt.job.state == 'admission_failed':
                    try:
                        drained = attempt.job.abort_before_forward(timeout=5)
                        if drained:
                            owner.retire(attempt.job)
                            attempt.job = None
                            attempt.stage = 'clean_rejection'
                    except BaseException as cleanup:
                        if not isinstance(cleanup, Exception):
                            raise cleanup from error
                        raise error from cleanup
                raise
        except (ValueError, TypeError):
            # owner.register validates before allocating producer/delivery claims.
            if attempt.stage == 'admitted' and owner._job is None:
                attempt.stage = 'clean_rejection'
            raise

        attempt.stage = 'forward'
        request = ForwardRequest(plan.worker_request_id, attempt.inputs)
        try:
            attempt.job.execute(request, input_owner=attempt.input_owner, input_claim=attempt.input_claim)
        except Exception as error:
            # The owner's prepared state proves its producer never started.
            if attempt.job.state == 'prepared':
                try:
                    drained = attempt.job.abort_before_forward(timeout=5)
                    if drained:
                        owner.retire(attempt.job)
                        attempt.job = None
                        attempt.stage = 'clean_rejection'
                except BaseException as cleanup:
                    if not isinstance(cleanup, Exception):
                        raise cleanup from error
                    raise error from cleanup
            raise
        request = None
        attempt.job.discard_result()
        attempt.job.confirm_producer_disposed()
        attempt.job.seal()
        attempt.stage = 'sealed'
        self._handoffs.register(owner, attempt.job, transfer_id=attempt.transfer_id,
                                recipient_id=attempt.team_id)
        attempt.bridge = self._handoffs.lookup(attempt.job_id, registration_id=attempt.registration_id,
                                               generation=attempt.generation, transfer_id=attempt.transfer_id)
        try:
            attempt.bridge.prepare()
        except HostBudgetExceeded as error:
            # Only admission_failed establishes that no manifest was copied
            # and no recipient could have observed any trace metadata.
            if attempt.bridge.state == 'admission_failed':
                try:
                    drained = attempt.bridge.abort_before_transfer(timeout=5)
                    if drained:
                        self._handoffs.retire(attempt.bridge)
                        attempt.bridge = attempt.job = None
                        attempt.stage = 'clean_postforward_failure'
                except BaseException as cleanup:
                    if not isinstance(cleanup, Exception):
                        raise cleanup from error
                    raise error from cleanup
                if attempt.stage == 'clean_postforward_failure':
                    raise PostForwardCapacityError('forward returned; trace delivery capacity was denied') from error
            raise
        attempt.bridge.offer()
        attempt.recipient = attempt.bridge.accept(recipient_id=attempt.team_id,
                                                  transfer_id=attempt.transfer_id)
        attempt.stage = 'delivery'
        recipient = attempt.recipient
        manifest = recipient.manifest()
        keys = recipient.events()
        recipient.ack_manifest()
        attempt.exposure_request_guard = owner._budget.request_guard(f'callback-{attempt.registration_id}')
        attempt.exposure_request_guard.open()
        attempt.exposure_request = attempt.exposure_request_guard.adopt()
        event_count = len(keys)
        total_bytes = 0
        event = key = sink = marker = None
        for event, key in zip(manifest.events, keys):
            if (event.identity.team_id != attempt.team_id or event.identity.request_id != attempt.request_id
                    or event.identity.job_id != attempt.job_id or event.identity.run_id != attempt.run_id):
                raise SharedTraceError('delivery identity does not match the submitting client')
            attempt.exposure_claim_guard = attempt.exposure_request.claim_guard(
                'client-callback', 'callback_exposure', event.payload_bytes, items=1)
            attempt.exposure_claim_guard.reserve()
            attempt.exposure_claim = attempt.exposure_claim_guard.adopt()
            def sink(payload, current=event):
                attempt.exposed_payload = payload
                try:
                    return attempt.on_event(current, payload)
                except BaseException as error:
                    if not isinstance(error, Exception):
                        attempt.callback_interruption = error
                    raise
            try:
                recipient.send(key, sink)
            except BaseException as error:
                if attempt.callback_interruption is not None:
                    raise attempt.callback_interruption from error
                raise
            sink = None
            attempt.exposed_payload = None
            attempt.exposure_claim.release()
            attempt.exposure_claim = attempt.exposure_claim_guard = None
            recipient.ack_received(key)
            total_bytes += event.payload_bytes
        marker = recipient.complete()
        if marker.event_count != event_count or total_bytes > self._limits.max_trace_bytes:
            raise SharedTraceError('complete trace exceeds the declared request limit')
        for key in keys:
            recipient.ack_release(key)
        attempt.exposure_request.close()
        attempt.exposure_request = attempt.exposure_request_guard = None
        # The cooperative callback must not retain metadata or payload aliases.
        # Drop this coordinator's own views before attesting their disposal.
        event = key = sink = manifest = keys = marker = None
        recipient.confirm_disposed()
        if not attempt.bridge.dispose(timeout=5):
            raise SharedTraceError('trace delivery did not drain')
        self._handoffs.retire(attempt.bridge)
        attempt.bridge = attempt.job = attempt.recipient = None
        return TraceReceipt(owner._broker_id, owner._process_generation, attempt.job_id,
                            attempt.generation, attempt.team_id, attempt.request_id,
                            attempt.run_id, event_count, total_bytes)

    def close(self):
        with self._condition:
            if self._active is not None or self._pending or self._quarantined or self._receipts:
                raise SharedTraceError('active or quarantined work requires supervision')
            self._closed = True
            self._acknowledged.clear()
            self._condition.notify_all()


class _Client:
    def __init__(self, service, team_id):
        self._service, self._team_id = service, team_id

    def submit(self, *, request_id, inputs, input_owner, input_claim, samples, sites, on_event):
        failure = None
        try:
            return self._service._submit(self._team_id, request_id=request_id, inputs=inputs,
                                         input_owner=input_owner, input_claim=input_claim,
                                         samples=samples, sites=sites, on_event=on_event)
        except BaseException as error:
            failure = error
            try:
                _strip_failure_frames(failure)
            except BaseException:
                pass
        finally:
            inputs = input_owner = input_claim = samples = sites = on_event = None
        raise failure from None

    def recover(self, request_id):
        if not _text(request_id):
            raise SharedTraceError('unknown receipt')
        with self._service._condition:
            receipt = self._service._receipts.get(self._team_id)
            if receipt is None or receipt.request_id != request_id:
                raise SharedTraceError('unknown receipt')
            return receipt

    def acknowledge(self, receipt):
        if type(receipt) is not TraceReceipt:
            raise SharedTraceError('invalid receipt')
        with self._service._condition:
            if self._service._receipts.get(self._team_id) is receipt:
                already = self._service._acknowledged.get(self._team_id) is receipt
                self._service._acknowledged[self._team_id] = receipt
                del self._service._receipts[self._team_id]
                return not already
            if self._service._acknowledged.get(self._team_id) is receipt:
                return False
            raise SharedTraceError('stale or foreign receipt')
