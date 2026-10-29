"""CPU-only, broker-owned scheduling for the cooperative trace service.

The generic Broker still sees exact forward-only request/result types. Its
private worker adapter runs the traced job and returns a zero-byte scheduling
token; trace payloads and receipts never travel in that result.
"""
from threading import Lock, RLock, current_thread
from uuid import uuid4
import math
import time
import weakref

import torch

from .broker import Broker, RequestState
from .cpu_shared_trace_service import CpuSharedTraceService, SharedTraceError, TraceReceipt
from .cpu_trace_owner import CpuTraceOwner
from .host_budget import HostBudget, HostClaim, HostRequest
from .resident_worker import ForwardRequest, ForwardResult, RequestCancelled
from .trace_producer import ProducerProfile
from .trace_store import HostTraceStore
from .traced_broker_worker import TraceLimits
from .traced_worker import TracedGPT2Worker


class BrokerTraceIncomplete(RuntimeError):
    """A trace or its broker handoff requires owner reconciliation."""


_owners = set()
_owners_lock = Lock()


class _Attempt:
    def __init__(self, team, request_id, token, samples, sites, on_event):
        self.team, self.request_id, self.token = team, request_id, token
        self.samples, self.sites, self.on_event = samples, sites, on_event
        self.guard = self.request = self.claim_guard = self.claim = None
        self.snapshot_ref = self.handle = self.receipt = self.failure = None
        self.completed = self.acknowledged = False


class _DispatchWorker:
    """The broker's only worker; its result is not the trace or model logits."""

    cancellation_exception = RequestCancelled

    def __init__(self, owner, worker):
        self._owner, self._worker = owner, worker
        self._token = torch.empty((0,), dtype=torch.float32, device='cpu')

    @property
    def poisoned(self):
        owner = self._owner
        if owner._inner is None or owner._broker is None:
            return bool(owner._faulted or self._worker.poisoned)
        return owner.quarantined

    @property
    def max_input_bytes(self):
        return self._worker.max_input_bytes

    @property
    def max_output_bytes(self):
        return self._worker.max_output_bytes

    def snapshot_request(self, request):
        attempt = self._owner._ticket(request.seed)
        try:
            snapshot = self._worker.snapshot_request(request)
            attempt.snapshot_ref = weakref.ref(snapshot.inputs['input_ids'])
            return snapshot
        except BaseException as error:
            self._owner._fail(attempt, error)
            raise

    def execute(self, request, cancel_event=None):
        owner = self._owner
        attempt = owner._ticket(request.seed)
        try:
            prefix = 'lg-' + owner._broker.broker_id + '-'
            if (type(request.request_id) is not str or not request.request_id.startswith(prefix)
                    or not request.request_id[len(prefix):].isdigit()):
                raise BrokerTraceIncomplete('broker request identity changed')
            job_id = int(request.request_id[len(prefix):])
            if not 1 <= job_id < 2**63 or request.request_id != prefix + str(job_id):
                raise BrokerTraceIncomplete('broker job identity is invalid')
            with owner._broker._condition:
                running = owner._broker._running
                if (running is None or running.team_id != attempt.team or running.job_id != job_id):
                    raise BrokerTraceIncomplete('broker client identity changed')
            with owner._condition:
                if owner._inner._generation + 1 != job_id:
                    raise BrokerTraceIncomplete('broker and trace job order differ')
            if self.poisoned:
                raise BrokerTraceIncomplete('trace broker is unavailable before execution')
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled('trace cancelled before execution')
            def deliver(manifest, payload):
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelled('trace cancelled before delivery')
                result = attempt.on_event(manifest, payload)
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelled('trace cancelled during delivery')
                return result
            inputs = dict(request.inputs)
            receipt = owner._inner.client(attempt.team).submit(
                request_id=attempt.request_id, inputs=inputs,
                input_owner=attempt.request, input_claim=attempt.claim,
                samples=attempt.samples, sites=attempt.sites, on_event=deliver)
            attempt.receipt = receipt
            if (type(receipt) is not TraceReceipt or receipt.broker_id != owner._broker.broker_id
                    or receipt.job_id != job_id or receipt.generation != job_id
                    or receipt.team_id != attempt.team or receipt.request_id != attempt.request_id):
                raise BrokerTraceIncomplete('trace receipt does not match the broker job')
            worker_id = request.request_id
            inputs = request = deliver = receipt = None
            return ForwardResult(worker_id, self._token)
        except BaseException as error:
            owner._fail(attempt, error)
            raise

    def close(self):
        self._owner._close_worker()


class BrokeredCpuTraceService:
    """One broker-owned CPU model and at most one trace per configured team.

    The operator must provide a fresh worker exclusively owned by this service.
    Any incomplete attempt retains its graph and claims for process recovery.
    """

    def __init__(self, *, worker, budget, store, profile, limits, team_ids,
                 max_slots, max_event_bytes):
        if (type(worker) is not TracedGPT2Worker or type(budget) is not HostBudget
                or type(store) is not HostTraceStore or type(profile) is not ProducerProfile
                or type(limits) is not TraceLimits or type(team_ids) is not tuple
                or len(team_ids) != 2 or any(type(team) is not str or not team for team in team_ids)
                or len(set(team_ids)) != 2):
            raise ValueError('a fresh CPU worker, trace policy and two teams are required')
        if worker.poisoned:
            raise ValueError('the CPU trace worker is already unavailable')
        self._worker, self._budget, self._bound_model = worker, budget, worker._model
        self._lock, self._submit_gate = RLock(), Lock()
        self._inner = self._owner = self._broker = self._server = None
        self._adapter = None
        self._attempts, self._tickets = {}, {}
        self._registering = None
        self._faulted = self._closing = self._closed = False
        self._failure = None
        with _owners_lock:
            if any(other._worker is worker or other._bound_model is self._bound_model for other in _owners):
                raise ValueError('CPU worker or model already has a brokered trace owner')
            _owners.add(self)
        try:
            self._adapter = _DispatchWorker(self, worker)
            self._broker = Broker(self._adapter, max_pending=2, max_pending_per_team=1)
            self._owner = CpuTraceOwner(worker=worker, budget=budget, store=store,
                broker_id=self._broker.broker_id, process_generation=uuid4().hex,
                max_slots=max_slots, max_event_bytes=max_event_bytes)
            self._inner = CpuSharedTraceService(owner=self._owner, profile=profile,
                                               limits=limits, team_ids=team_ids)
            self._condition, self._limits = self._inner._condition, self._inner._limits
            self._clients = {team: _Client(self, team) for team in team_ids}
        except BaseException as error:
            self._failure = error
            self._faulted = True
            raise

    @property
    def _active(self):
        return self._inner._active

    @property
    def quarantined(self):
        server = self._server
        model_changed = (not self._closing and self._owner is not None
                         and (self._worker._model is not self._bound_model
                              or self._owner._model is not self._bound_model))
        return bool(self._faulted or self._worker.poisoned or model_changed
                    or self._inner._quarantined or self._broker.quarantined
                    or (server is not None and server._incomplete))

    def _bind_server(self, server):
        with self._lock:
            if self._server is not None or self._closing or self._faulted:
                raise BrokerTraceIncomplete('trace server is already bound or unavailable')
            self._server = server

    def client(self, team_id):
        with self._lock:
            if self._closing or self.quarantined or team_id not in self._clients:
                raise SharedTraceError('unknown client or brokered trace service unavailable')
            return self._clients[team_id]

    def _ticket(self, token):
        with self._lock:
            attempt = self._tickets.get(token)
            if attempt is None or self.quarantined:
                raise BrokerTraceIncomplete('unknown or unavailable broker trace ticket')
            return attempt

    def _fail(self, attempt, error):
        with self._lock:
            if attempt.failure is None:
                attempt.failure = error
            self._faulted = True

    def _submit(self, team, *, request_id, inputs, input_owner, input_claim,
                samples, sites, on_event):
        if current_thread() is self._broker._thread:
            raise SharedTraceError('a trace callback cannot reenter its broker')
        if (type(request_id) is not str or not request_id or len(request_id) > 64
                or type(inputs) is not dict or set(inputs) != {'input_ids'}
                or type(inputs['input_ids']) is not torch.Tensor or not callable(on_event)
                or type(input_owner) is not HostRequest or type(input_claim) is not HostClaim
                or input_owner._budget is not self._budget or input_claim._budget is not self._budget):
            raise ValueError('invalid brokered trace input custody')
        source = inputs['input_ids']
        info = input_claim.info()
        size = source.numel() * source.element_size()
        if info.request_id != input_owner._request_id or info.nbytes < size or info.items < 1:
            raise ValueError('input claim does not cover the source tensor')
        attempt = None
        registered = False
        try:
            with self._lock:
                if (self._closing or self.quarantined or self._registering is not None
                        or team in self._attempts):
                    raise SharedTraceError('client or brokered trace service unavailable')
                token = len(self._attempts) + 1
                attempt = _Attempt(team, request_id, token, samples, sites, on_event)
                self._registering = attempt
                self._attempts[team] = attempt
                self._tickets[token] = attempt
                registered = True
                self._registering = None
            attempt.guard = self._budget.request_guard('broker-trace-' + uuid4().hex)
            attempt.guard.open()
            attempt.request = attempt.guard.adopt()
            attempt.claim_guard = attempt.request.claim_guard(
                'broker-trace-input', 'broker_trace_snapshot', size, items=1)
            attempt.claim_guard.reserve()
            attempt.claim = attempt.claim_guard.adopt()
            # Serialize preparation so broker job IDs and trace generations
            # have the same order even if one snapshot takes longer.
            with self._submit_gate:
                if self.quarantined or self._closing:
                    raise BrokerTraceIncomplete('trace broker is unavailable')
                request = ForwardRequest(request_id, inputs, token)
                attempt.handle = self._broker.submit(team, request)
                request = None
            outcome = attempt.handle.wait()
            receipt = attempt.receipt
            if (outcome.state is not RequestState.SUCCEEDED or type(receipt) is not TraceReceipt
                    or outcome.broker_id != self._broker.broker_id or outcome.job_id != receipt.job_id
                    or outcome.team_id != team or outcome.request_id != request_id
                    or type(outcome.result) is not ForwardResult
                    or outcome.result.request_id != request_id
                    or outcome.result.logits is not self._adapter._token
                    or receipt.team_id != team or receipt.request_id != request_id):
                raise BrokerTraceIncomplete('broker trace completion is uncertain')
            with self._lock:
                if self.quarantined or self._closing:
                    raise BrokerTraceIncomplete('broker trace completion requires reconciliation')
                attempt.completed = True
                attempt.samples = attempt.sites = attempt.on_event = None
                return receipt
        except BaseException as error:
            if attempt is not None:
                self._fail(attempt, error)
            elif isinstance(error, SharedTraceError):
                raise
            if not isinstance(error, Exception):
                raise
            raise BrokerTraceIncomplete('broker trace request is incomplete') from None
        finally:
            if attempt is not None and not registered:
                self._faulted = True
            source = inputs = input_owner = input_claim = on_event = None

    def _close_worker(self):
        if self.quarantined:
            raise BrokerTraceIncomplete('quarantined broker trace owner requires process recovery')
        if (self._worker._model is not self._bound_model
                or self._owner._model is not self._bound_model):
            raise BrokerTraceIncomplete('broker trace model ownership changed')
        for attempt in self._attempts.values():
            if (not attempt.completed or not attempt.acknowledged or attempt.snapshot_ref is None
                    or attempt.snapshot_ref() is not None):
                raise BrokerTraceIncomplete('broker trace snapshots or receipts remain live')
        self._inner.close()
        for attempt in self._attempts.values():
            attempt.claim.release()
            attempt.claim = attempt.claim_guard = None
            attempt.request.close()
            attempt.request = attempt.guard = None
        model = self._owner._model
        self._worker.close()
        if self._worker._model is not None or self._worker.poisoned:
            raise BrokerTraceIncomplete('worker model release is uncertain')
        self._owner._model = None
        model = None

    def close(self, timeout=5):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('invalid brokered trace close timeout')
        deadline = time.monotonic() + timeout
        remaining = lambda: max(0.0, deadline - time.monotonic())
        if not self._lock.acquire(timeout=remaining()):
            return False
        try:
            already_closed = self._closed
            if self.quarantined:
                return False
            server = self._server
            if server is not None:
                if not server._lock.acquire(timeout=remaining()):
                    return False
                try:
                    if (not server._closing or server._incomplete or server._sessions
                            or (server._accept_thread is not None and server._accept_thread.is_alive())):
                        return False
                finally:
                    server._lock.release()
            self._closing = True
        finally:
            self._lock.release()
        if not already_closed:
            try:
                if not self._broker.close(wait=True, timeout=remaining()):
                    return False
            except Exception as error:
                self._failure = error
                self._faulted = True
                return False
            if not self._lock.acquire(timeout=remaining()):
                return False
            try:
                self._closed = True
            finally:
                self._lock.release()
        model = None
        if not _owners_lock.acquire(timeout=remaining()):
            return False
        try:
            model = self._bound_model
            _owners.discard(self)
            self._bound_model = None
        finally:
            _owners_lock.release()
        model = None
        return True


class _Client:
    def __init__(self, owner, team):
        self._owner, self._team = owner, team

    def submit(self, *, request_id, inputs, input_owner, input_claim, samples, sites, on_event):
        return self._owner._submit(self._team, request_id=request_id, inputs=inputs,
            input_owner=input_owner, input_claim=input_claim, samples=samples, sites=sites, on_event=on_event)

    def recover(self, request_id):
        with self._owner._lock:
            if self._owner.quarantined or self._owner._closing:
                raise SharedTraceError('brokered trace receipt requires reconciliation')
            attempt = self._owner._attempts.get(self._team)
            if attempt is None or not attempt.completed or attempt.request_id != request_id:
                raise SharedTraceError('unknown brokered trace receipt')
            return attempt.receipt

    def acknowledge(self, receipt):
        owner = self._owner
        with owner._lock:
            if owner.quarantined or owner._closing:
                raise SharedTraceError('brokered trace receipt requires reconciliation')
            attempt = owner._attempts.get(self._team)
            if attempt is None or not attempt.completed or attempt.receipt is not receipt:
                raise SharedTraceError('foreign brokered trace receipt')
            try:
                result = owner._inner.client(self._team).acknowledge(receipt)
                if owner.quarantined:
                    raise BrokerTraceIncomplete('brokered trace receipt became uncertain')
                attempt.acknowledged = True
                return result
            except BaseException as error:
                owner._fail(attempt, error)
                raise
