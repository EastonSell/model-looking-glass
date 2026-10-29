"""Experimental owner-private IPC-to-fair-broker bridge, tested on CPU.

The budgets below are declared custody commitments, not physical memory
limits. The owner must use the paired server, codec, broker and supervisor.
"""
from dataclasses import dataclass, replace
import json
from queue import Full
import re
import threading
import time
from types import MappingProxyType
from uuid import uuid4

import torch

from .broker import Broker, BrokerUnavailable, Outcome, RequestHandle, RequestState
from .broker_fair_handoff import BrokerFairHandoff
from .fair_adapter import ExperimentalFairAdapter, HandoffFailure
from .host_budget import HostBudget
from .ipc import IPCServer
from .ipc_custody import IPCCustodyCapacity, IPCCustodyKey
from .resident_worker import ForwardRequest


_IDENTIFIER = re.compile(r'[A-Za-z0-9_.-]{1,64}\Z')
_INPUTS = frozenset(('input_ids', 'attention_mask', 'token_type_ids', 'position_ids'))


class FatalBridgeFailure(BaseException):
    """An unclassified fatal bridge failure whose private cause stays rooted."""


class ProxyBusy(BrokerUnavailable):
    """The trusted handler must retry after its current proxy operation ends."""


def _failure_kind(error):
    if isinstance(error, KeyboardInterrupt):
        return 1
    if isinstance(error, SystemExit):
        return 2
    return 0 if isinstance(error, Exception) else 3


def _raise_public_failure(kind):
    try:
        if kind == 1:
            raise KeyboardInterrupt('bridge custody is indeterminate') from None
        if kind == 2:
            raise SystemExit('bridge custody is indeterminate') from None
        if kind == 3:
            raise FatalBridgeFailure('bridge custody is indeterminate') from None
        raise BrokerUnavailable('bridge custody is indeterminate') from None
    except BaseException as public_error:
        public_error.__context__ = None
        raise


def _identifier(value):
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate field')
        result[key] = value
    return result


class ForwardCodec:
    """Bounded JSON codec for the IPC server's CPU int64 forward inputs."""

    def __init__(self, *, max_batch, max_tokens, max_encoded_bytes, max_decoded_bytes):
        for value, ceiling in ((max_batch, 256), (max_tokens, 8192), (max_encoded_bytes, 1048576),
                               (max_decoded_bytes, 64 * 1024 * 1024)):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError('invalid codec limit')
        self.max_batch = max_batch
        self.max_tokens = max_tokens
        self.max_encoded_bytes = max_encoded_bytes
        self.max_decoded_bytes = max_decoded_bytes

    def _validate(self, request_id, seed, inputs):
        if not _identifier(request_id) or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('invalid forward identity')
        if type(inputs) is not dict or 'input_ids' not in inputs or not set(inputs) <= _INPUTS:
            raise ValueError('invalid forward inputs')
        expected = None
        for name, rows in inputs.items():
            if type(rows) is not list or not 1 <= len(rows) <= self.max_batch:
                raise ValueError('invalid forward batch')
            width = None
            for row in rows:
                if type(row) is not list or not 1 <= len(row) <= self.max_tokens:
                    raise ValueError('invalid forward sequence')
                if width is None:
                    width = len(row)
                if len(row) != width:
                    raise ValueError('ragged forward input')
                for value in row:
                    if type(value) is not int or not 0 <= value < 2**31:
                        raise ValueError('invalid input value')
                    if name == 'attention_mask' and value not in (0, 1):
                        raise ValueError('invalid attention mask')
            shape = len(rows), width
            if expected is not None and shape != expected:
                raise ValueError('forward shapes differ')
            expected = shape

    def _check_tensor_request(self, request):
        if type(request) is not ForwardRequest or type(request.inputs) not in (dict, MappingProxyType):
            raise ValueError('invalid forward request')
        if not _identifier(request.request_id) or type(request.seed) is not int or not 0 <= request.seed < 2**63:
            raise ValueError('invalid forward identity')
        if 'input_ids' not in request.inputs or not set(request.inputs) <= _INPUTS:
            raise ValueError('invalid forward inputs')
        total = 0
        shape = None
        for name, tensor in request.inputs.items():
            if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu' or
                    tensor.dtype is not torch.int64 or tensor.ndim != 2 or
                    tensor.layout is not torch.strided or not tensor.is_contiguous() or
                    tensor.requires_grad or tensor.storage_offset() != 0 or
                    tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size() or
                    not 1 <= tensor.shape[0] <= self.max_batch or
                    not 1 <= tensor.shape[1] <= self.max_tokens):
                raise ValueError('unsupported forward tensor')
            if shape is not None and tuple(tensor.shape) != shape:
                raise ValueError('forward shapes differ')
            shape = tuple(tensor.shape)
            total += tensor.numel()
            if total > self.max_encoded_bytes or total * 8 > self.max_decoded_bytes:
                raise ValueError('forward expansion exceeds encoded limit')

    def snapshot(self, request):
        """Copy caller-owned inputs before trusted model-domain validation."""
        self._check_tensor_request(request)
        inputs = {}
        with torch.inference_mode(False):
            for name, tensor in request.inputs.items():
                inputs[name] = tensor.clone()
        return ForwardRequest(request.request_id, inputs, request.seed)

    def encode(self, request):
        self._check_tensor_request(request)
        inputs = {}
        for name, tensor in request.inputs.items():
            inputs[name] = tensor.tolist()
        self._validate(request.request_id, request.seed, inputs)
        data = json.dumps({'request_id': request.request_id, 'seed': request.seed, 'inputs': inputs},
                          sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False).encode('ascii')
        if len(data) > self.max_encoded_bytes:
            raise ValueError('encoded request exceeds its limit')
        return data

    def decode(self, payload):
        if type(payload) is not bytes or len(payload) > self.max_encoded_bytes:
            raise ValueError('invalid encoded request')
        try:
            value = json.loads(payload.decode('ascii'), object_pairs_hook=_unique_object)
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ValueError('invalid encoded request') from exc
        if type(value) is not dict or set(value) != {'request_id', 'seed', 'inputs'}:
            raise ValueError('invalid encoded request')
        self._validate(value['request_id'], value['seed'], value['inputs'])
        total = sum(len(rows) * len(rows[0]) for rows in value['inputs'].values())
        if total * 8 > self.max_decoded_bytes:
            raise ValueError('decoded tensor storage exceeds limit')
        tensors = {}
        for name, rows in value['inputs'].items():
            tensors[name] = torch.tensor(rows, dtype=torch.long, device='cpu')
        return ForwardRequest(value['request_id'], tensors, value['seed'])


@dataclass
class _Record:
    key: IPCCustodyKey
    request_guard: object = None
    claim_guard: object = None
    claim: object = None
    team: str = None
    request_id: str = None
    proxy: object = None
    request: object = None
    owned_request: object = None
    payload: bytes = None
    guard: object = None
    ticket: object = None
    target_handle: object = None
    submitted_ns: int = 0
    synthetic: object = None
    cancel_requested: bool = False
    cancel_forwarded: bool = False
    handler_closed: bool = False
    handler_thread: object = None
    notice_thread: object = None
    retired: bool = False
    failure: object = None
    transition_failure: object = None
    wait_failure: object = None
    gate_failure: object = None
    retiring: bool = False
    disposal: object = None


class _Proxy:
    """One server-handler-owned handle, never a client capability."""

    def __init__(self, owner, record, job_id):
        self._owner, self._record, self.job_id = owner, record, job_id
        self._operation_lock = threading.Lock()

    def _invoke(self, method, *args):
        gate_complete = False
        try:
            if not self._operation_lock.acquire(blocking=False):
                gate_complete = True
                raise ProxyBusy('proxy operation already in progress')
            try:
                return method(self._record, *args)
            finally:
                self._operation_lock.release()
                gate_complete = True
        except BaseException as error:
            if gate_complete:
                raise
            # Acquisition or release may have taken effect without returning.
            # Root the private failure and require owner-process recovery.
            self._owner._retain_gate_failure(error, self._record)
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def cancel(self):
        return self._invoke(self._owner._cancel)

    def wait(self, timeout=None):
        return self._invoke(self._owner._wait, timeout)

    @property
    def cancel_requested(self):
        return self._invoke(self._owner._cancellation_requested)


class FairIPCBridge:
    """Single-owner fair dispatch with separately budgeted IPC custody."""

    def __init__(self, adapter, *, codec, team_credentials, custody_epoch,
                 max_connections, max_request_bytes, max_response_bytes,
                 connection_commitment_bytes, request_validator=None):
        if type(adapter) is not ExperimentalFairAdapter or type(codec) is not ForwardCodec:
            raise ValueError('exact trusted adapter and codec are required')
        target = adapter._target
        if (type(target) is not BrokerFairHandoff or type(target._broker) is not Broker or
                target._broker._fair_handoff is not target):
            raise ValueError('the adapter must use a bound fair broker target')
        if type(team_credentials) is not dict or not team_credentials:
            raise ValueError('team credentials are required')
        if request_validator is not None and not callable(request_validator):
            raise ValueError('request validator must be trusted callable code')
        if any(not _identifier(team) or type(credential) is not bytes or not credential
               for team, credential in team_credentials.items()):
            raise ValueError('invalid team credential')
        if len(set(team_credentials.values())) != len(team_credentials):
            raise ValueError('credentials must be distinct per team')
        IPCCustodyKey(custody_epoch, 1)
        for value in (max_connections, max_request_bytes, max_response_bytes, connection_commitment_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('connection limits must be positive integers')
        if (max_connections > 128 or not 256 <= max_request_bytes <= 1048576 or
                not 512 <= max_response_bytes <= 4194304):
            raise ValueError('invalid IPC limits')
        minimum = 256 * (max_request_bytes + max_response_bytes + codec.max_encoded_bytes)
        if connection_commitment_bytes < minimum:
            raise ValueError('connection commitment is below declared envelope')
        decoder = getattr(adapter._target, '_decoder', None)
        if getattr(decoder, '__self__', None) is not codec or getattr(decoder, '__func__', None) is not ForwardCodec.decode:
            raise ValueError('target must use this exact codec decoder')
        if (codec.max_encoded_bytes > adapter._max_item_bytes or
                codec.max_encoded_bytes > adapter._target.max_item_bytes or
                codec.max_decoded_bytes > adapter._target.max_decoded_bytes):
            raise ValueError('target envelope is smaller than codec limits')
        if set(team_credentials) != set(adapter._target.teams):
            raise ValueError('bridge teams must match the target')
        for team, credential in team_credentials.items():
            adapter.registration_guard(credential, b'', team_id=team)
        self._adapter, self._codec = adapter, codec
        self._target, self._target_broker = target, target._broker
        self._request_validator = request_validator
        self._credentials = dict(team_credentials)
        self._epoch = custody_epoch
        self._max_connections = max_connections
        self._max_request_bytes, self._max_response_bytes = max_request_bytes, max_response_bytes
        self._commitment = connection_commitment_bytes
        self._budget = HostBudget(max_bytes=max_connections * connection_commitment_bytes,
                                  max_items=max_connections, max_requests=max_connections,
                                  max_operations=1)
        self._condition = threading.Condition()
        self._records = {}
        self._transition = None
        self._quarantined = False
        self._owner_failure = None
        self._owner_transition_failure = None
        self._dispatch_failure = None
        self._disposal_failure = None
        self._accepting = True
        self._stop = False
        self._thread = None
        self._next_job = 1
        self._server_created = False
        self._server = None
        self.broker_id = uuid4().hex
        self.custody_guardian = self

    @property
    def quarantined(self):
        return (self._quarantined or self._adapter.indeterminate or
                self._target_broker.quarantined or
                (self._server is not None and self._server.custody_quarantined))

    @property
    def committed_bytes(self):
        return self._budget.snapshot().used_bytes

    def _check_locked(self):
        if self._transition is not None:
            self._quarantined = True
        if self.quarantined:
            raise BrokerUnavailable('bridge custody is indeterminate')

    def _retain_failure(self, error, record=None):
        if record is not None:
            if record.failure is None:
                record.failure = error
        else:
            if self._owner_failure is None:
                self._owner_failure = error
        self._quarantined = True

    def _retain_wait_failure(self, error, record):
        # A proxy has at most one active operation. Keep the wait's original
        # error separately from a dispatcher/observer failure for this record.
        if record.wait_failure is None:
            record.wait_failure = error
        self._quarantined = True

    def _retain_gate_failure(self, error, record):
        if record.gate_failure is None:
            record.gate_failure = error
        self._quarantined = True

    def _latch(self, marker, error, record=None):
        if self._transition is marker:
            # Only the transition's own caller has this marker; an observer
            # reporting quarantine cannot displace its original exception.
            if record is None:
                if self._owner_transition_failure is None:
                    self._owner_transition_failure = error
            elif record.transition_failure is None:
                record.transition_failure = error
            self._quarantined = True
            return True
        if self.quarantined:
            self._retain_failure(error, record)
            return True
        return False

    def _key(self, key):
        if type(key) is not IPCCustodyKey or key.server_epoch != self._epoch:
            raise ValueError('foreign custody key')

    def reserve(self, key):
        self._key(key)
        marker = object()
        try:
            return self._reserve(key, marker)
        except BaseException as error:
            if not self._latch(marker, error, self._records.get(key)):
                raise
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _reserve(self, key, marker):
        with self._condition:
            self._check_locked()
            if not self._accepting:
                raise BrokerUnavailable('connection custody is closed')
            if key in self._records:
                raise ValueError('custody key already registered')
            if len(self._records) >= self._max_connections:
                # Exact, trusted no-commit denial: no new record, claim,
                # payload, or asynchronous registration was created.
                raise IPCCustodyCapacity('connection custody is full')
            record = _Record(key, handler_thread=threading.current_thread())
            self._transition = marker
            self._records[key] = record
            record.request_guard = self._budget.request_guard(f'{self._epoch}:{key.connection_generation}')
            record.request_guard.open()
            host_request = record.request_guard.adopt()
            record.claim_guard = host_request.claim_guard('ipc-bridge', 'connection', self._commitment)
            record.claim_guard.reserve()
            record.claim = record.claim_guard.adopt()
            self._transition = None

    def authenticate(self, key, team_id, request_id):
        self._key(key)
        if team_id not in self._credentials or not _identifier(request_id):
            raise PermissionError('unconfigured team or request')
        marker = object()
        try:
            return self._authenticate(key, team_id, request_id, marker)
        except BaseException as error:
            if not self._latch(marker, error, self._records.get(key)):
                raise
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _authenticate(self, key, team_id, request_id, marker):
        with self._condition:
            self._check_locked()
            record = self._records.get(key)
            if record is None or record.team is not None:
                raise ValueError('unknown or already authenticated connection')
            self._transition = marker
            record.team, record.request_id = team_id, request_id
            self._transition = None

    def submit_registered(self, key, team_id, request):
        self._key(key)
        marker = object()
        try:
            return self._submit(key, team_id, request, marker)
        except BaseException as error:
            if not self._latch(marker, error, self._records.get(key)):
                raise
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _submit(self, key, team_id, request, marker):
        with self._condition:
            self._check_locked()
            record = self._records.get(key)
            if (record is None or record.team != team_id or type(request) is not ForwardRequest or
                    request.request_id != record.request_id or record.proxy is not None):
                raise PermissionError('request does not match authenticated custody')
            if self._next_job >= 2**63:
                raise BrokerUnavailable('bridge job IDs exhausted')
            proxy = _Proxy(self, record, self._next_job)
            self._transition = marker
            self._next_job += 1
            record.proxy = proxy
            record.submitted_ns = time.perf_counter_ns()
            record.request = request
            try:
                record.owned_request = self._codec.snapshot(request)
                if self._request_validator is not None:
                    if self._request_validator(record.owned_request) is not None:
                        raise ValueError('request validator must return None')
                record.payload = self._codec.encode(record.owned_request)
            except ValueError:
                record.synthetic = self._synthetic(record, RequestState.FAILED, 'InvalidRequest')
            else:
                record.guard = self._adapter.registration_guard(
                    self._credentials[team_id], record.payload, team_id=team_id)
                try:
                    record.guard.submit()
                except Full:
                    # This adapter's capacity failures occur before queue
                    # publication. Keep the separately funded IPC custody.
                    if self._adapter.indeterminate:
                        raise
                    record.synthetic = self._synthetic(record, RequestState.FAILED, 'QueueFull')
                else:
                    record.ticket = record.guard.adopt()
            self._condition.notify_all()
            self._transition = None
        return proxy

    def recover_proxy(self, key, team_id, request_id):
        """Trusted owner recovers the exact proxy after a lost submit return."""
        self._key(key)
        with self._condition:
            self._check_locked()
            record = self._records.get(key)
            if record is None:
                return None
            if record.team != team_id or record.request_id != request_id:
                raise PermissionError('connection identity does not match')
            if record.proxy is not None and record.ticket is None and record.synthetic is None:
                self._quarantined = True
                raise BrokerUnavailable('submission state is indeterminate')
            proxy = record.proxy
        return proxy

    def handler_closed(self, key):
        self._key(key)
        marker = object()
        try:
            return self._handler_closed(key, marker)
        except BaseException as error:
            if not self._latch(marker, error, self._records.get(key)):
                raise
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _handler_closed(self, key, marker):
        with self._condition:
            self._check_locked()
            record = self._records.get(key)
            if record is None:
                raise ValueError('unknown custody key')
            if self._server is not None and record.handler_thread is not threading.current_thread():
                raise PermissionError('notice must come from the registered handler')
            self._transition = marker
            record.notice_thread = threading.current_thread()
            record.handler_closed = True
            self._condition.notify_all()
            self._transition = None

    def _synthetic(self, record, state, error=None):
        now = time.perf_counter_ns()
        return Outcome(self.broker_id, record.proxy.job_id, record.team, record.request_id,
                       state, record.submitted_ns, record.submitted_ns, None, now,
                       error_type=error)

    def _target_locked(self, record):
        if record.ticket is None:
            return None
        try:
            # Hold the broker's reentrant condition across both lookups. A
            # publication between two independent lookups would otherwise
            # make a valid None-to-handle transition look like a mismatch.
            with self._target_broker._condition:
                if (self._adapter._target is not self._target or
                        self._target._broker is not self._target_broker or
                        self._target_broker._fair_handoff is not self._target or
                        not self._adapter._owns_ticket_namespace(record.ticket)):
                    raise BrokerUnavailable('target or ticket identity mismatch')
                if record.target_handle is None:
                    record.target_handle = self._adapter.recover_receipt(record.ticket)
                handle = record.target_handle
                authoritative = self._target.recover_receipt(record.ticket.sequence)
                if (authoritative is not handle or self._target_broker.quarantined or
                        (handle is not None and
                         (type(handle) is not RequestHandle or handle._broker is not self._target_broker or
                          handle.team_id != record.team or handle.request_id != record.request_id))):
                    raise BrokerUnavailable('target receipt identity or custody mismatch')
            return handle
        except BaseException as error:
            self._retain_failure(error, record)
            raise

    def _forward_cancel_locked(self, record, handle):
        if handle is not None and record.cancel_requested and not record.cancel_forwarded:
            handle.cancel()
            record.cancel_forwarded = True

    def _cancel(self, record):
        marker = object()
        try:
            return self._cancel_inner(record, marker)
        except BaseException as error:
            if not self._latch(marker, error, record):
                raise
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _cancel_inner(self, record, marker):
        with self._condition:
            self._check_locked()
            if record.retired or record.handler_closed or record.synthetic is not None or record.cancel_requested:
                return False
            handle = self._target_locked(record)
            if handle is not None:
                try:
                    handle.wait(timeout=0)
                except TimeoutError:
                    pass
                else:
                    return False
            self._transition = marker
            record.cancel_requested = True
            accepted = True
            if handle is not None:
                accepted = handle.cancel()
                record.cancel_forwarded = accepted
            elif self._adapter.cancel(self._credentials[record.team], record.ticket):
                record.synthetic = self._synthetic(record, RequestState.CANCELLED)
            else:
                handle = self._target_locked(record)
                if handle is not None:
                    accepted = handle.cancel()
                    record.cancel_forwarded = accepted
            if not accepted:
                record.cancel_requested = False
            self._condition.notify_all()
            self._transition = None
        return accepted

    def _cancellation_requested(self, record):
        with self._condition:
            self._check_locked()
            if record.retired:
                raise BrokerUnavailable('handler custody has ended')
            requested = record.cancel_requested
        return requested

    def _wait(self, record, timeout):
        if timeout is not None and (type(timeout) not in (int, float) or timeout < 0):
            raise ValueError('timeout must be nonnegative')
        try:
            return self._wait_inner(record, timeout)
        except TimeoutError:
            raise
        except BrokerUnavailable as error:
            if not self.quarantined:
                raise
            self._retain_wait_failure(error, record)
            kind = _failure_kind(error)
        except BaseException as error:
            self._retain_wait_failure(error, record)
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _wait_inner(self, record, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._condition:
                self._check_locked()
                if record.retired or record.handler_closed:
                    raise BrokerUnavailable('handler custody has ended')
                if record.synthetic is not None:
                    return record.synthetic
                try:
                    handle = self._target_locked(record)
                    self._forward_cancel_locked(record, handle)
                    outcome = handle.wait(timeout=0) if handle is not None else None
                except TimeoutError:
                    outcome = None
                if outcome is not None:
                    self._check_locked()
                    if self._target_locked(record) is not handle:
                        raise BrokerUnavailable('target receipt changed before delivery')
                    if (outcome.team_id != record.team or outcome.request_id != record.request_id or
                            outcome.job_id != handle.job_id or outcome.broker_id != handle._broker.broker_id):
                        self._quarantined = True
                        raise BrokerUnavailable('target outcome identity mismatch')
                    return replace(outcome, broker_id=self.broker_id, job_id=record.proxy.job_id,
                                   submitted_ns=record.submitted_ns)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError('request is still in flight')
                self._condition.wait(.02 if remaining is None else min(.02, remaining))

    def start(self):
        marker = object()
        try:
            return self._start(marker)
        except BaseException as error:
            if not self._latch(marker, error):
                raise
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _start(self, marker):
        with self._condition:
            self._check_locked()
            if self._thread is not None or not self._accepting:
                raise BrokerUnavailable('bridge already started or closed')
            thread = threading.Thread(target=self._run, daemon=True, name='fair-ipc-dispatch')
            self._transition = marker
            self._thread = thread
            self._thread.start()
            self._transition = None
        return self

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._check_locked()
                    if self._stop:
                        return
                try:
                    self._adapter.transfer_next()
                except HandoffFailure as error:
                    self._handle_handoff_failure(error.ticket)
                self._collect_closed()
                with self._condition:
                    self._condition.notify_all()
                    self._condition.wait(.01)
        except BaseException as error:
            self._dispatch_failure = error
            self._quarantined = True

    def _handle_handoff_failure(self, ticket):
        with self._condition:
            self._check_locked()
            matches = [record for record in self._records.values() if record.ticket == ticket]
            if len(matches) != 1:
                raise BrokerUnavailable('claimed ticket has no unique custody owner')
            record = matches[0]
            handle = self._target_locked(record)
            if handle is None and self._target_broker.available:
                record.synthetic = self._synthetic(record, RequestState.FAILED, 'HandoffFailure')
            elif handle is None:
                raise BrokerUnavailable('failed handoff may retain unknown work')
            self._condition.notify_all()

    def _retire_target(self, record):
        handle = self._target_locked(record)
        if handle is None:
            return False
        self._forward_cancel_locked(record, handle)
        try:
            handle.wait_for_retirement(timeout=0)
        except TimeoutError:
            return False
        if not self._adapter.acknowledge_receipt(record.ticket, handle):
            raise BrokerUnavailable('receipt acknowledgement failed')
        # Keep the acknowledged handle in the record until it is staged with
        # the other aliases and disposed outside the bridge condition.
        return True

    def _collect_closed(self):
        try:
            return self._collect_closed_inner()
        except BaseException as error:
            if self._disposal_failure is None:
                self._disposal_failure = error
            self._quarantined = True
            kind = _failure_kind(error)
        _raise_public_failure(kind)

    def _collect_closed_inner(self):
        # One dispatcher owns this method. Detach each connection's aliases
        # into a rooted holder under the condition, then dispose them without
        # holding the condition so synchronous finalizers can reenter safely.
        while True:
            candidate = None
            with self._condition:
                self._check_locked()
                for record in tuple(self._records.values()):
                    if not record.handler_closed or record.retiring:
                        continue
                    if record.notice_thread is None or record.notice_thread.is_alive():
                        continue
                    # A handler can latch failure just before its exit.
                    self._check_locked()
                    if record.proxy is not None and record.synthetic is None:
                        if not record.cancel_requested:
                            record.cancel_requested = True
                            if self._adapter.cancel(self._credentials[record.team], record.ticket):
                                record.synthetic = self._synthetic(record, RequestState.CANCELLED)
                        if record.synthetic is None and not self._retire_target(record):
                            continue
                    self._check_locked()
                    record.disposal = (record.request, record.owned_request, record.payload,
                                       record.guard, record.synthetic, record.target_handle, record.proxy)
                    record.retiring = True
                    record.request = record.owned_request = record.payload = record.guard = record.synthetic = None
                    record.target_handle = None
                    record.proxy = None
                    candidate = record
                    break
                if candidate is None:
                    self._condition.notify_all()
                    return
            candidate.disposal = None
            with self._condition:
                self._check_locked()
                candidate.retired = True
                candidate.claim.release()
                candidate.claim = candidate.claim_guard = None
                candidate.request_guard.cancel_and_close_empty()
                candidate.request_guard = None
                del self._records[candidate.key]
                self._condition.notify_all()

    def make_server(self, *, io_timeout=3.0, job_timeout=60.0, request_factory=None):
        with self._condition:
            self._check_locked()
            if self._server_created:
                raise ValueError('only one server is supported per bridge epoch')
            self._server_created = True
        server = IPCServer(self, team_ids=tuple(self._credentials), custody_guardian=self,
                           request_factory=request_factory,
                           custody_epoch=self._epoch, max_connections=self._max_connections,
                           max_request_bytes=self._max_request_bytes, max_response_bytes=self._max_response_bytes,
                           max_batch=self._codec.max_batch, max_tokens=self._codec.max_tokens,
                           io_timeout=io_timeout, job_timeout=job_timeout)
        self._server = server
        return server

    def close(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            self._accepting = False
            while self._records and not self.quarantined:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(.05, remaining))
            if self.quarantined:
                return False
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive():
                return False
        self._adapter.close()
        return True
