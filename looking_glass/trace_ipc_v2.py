"""Cooperative, bounded, CPU-only trace delivery over loopback IPC.

This is a separate protocol from the forward-only v1 broker transport.  A
receiver must dispose borrowed event storage before sending its release
attestation. A disconnect does not prove that a peer disposed its copies.
"""
from dataclasses import dataclass
import math
import secrets
import select
import socket
import struct
import threading
import time
from types import MappingProxyType
import uuid

from .host_budget import HostBudget
from .ipc import _identifier, _read_exact, _receive, _remaining, _send
from .trace_schema import TraceEvent
from .traced_broker_worker import SampleTrace, SiteTrace


class TraceIPCIncomplete(RuntimeError):
    """A delivery or its completion is uncertain and requires reconciliation."""


class TraceIPCRejected(RuntimeError):
    """The server rejected the request before starting the traced job."""


@dataclass(frozen=True, slots=True)
class TraceIPCEvent:
    broker_id: str
    process_generation: str
    job_id: int
    generation: int
    transfer_id: str
    event_id: str
    team_id: str
    request_id: str
    event: TraceEvent
    payload_bytes: int
    byte_order: str


@dataclass(frozen=True, slots=True)
class TraceIPCCompletion:
    broker_id: str
    process_generation: str
    job_id: int
    generation: int
    transfer_id: str
    team_id: str
    request_id: str
    run_id: str
    event_count: int
    trace_bytes: int


def _hex(value):
    return (type(value) is str and len(value) == 32
            and all(char in '0123456789abcdef' for char in value))


def _limits(control, event, trace, timeout):
    # The fixed request and event metadata bounds require an 8 KiB frame.
    # Smaller frames can admit a request but fail its event after execution.
    if (type(control) is not int or not 8192 <= control <= 65536
            or type(event) is not int or not 1 <= event <= 2**24
            or type(trace) is not int or not event <= trace <= 2**26
            or type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 300):
        raise ValueError('invalid bounded trace IPC configuration')


def _send_bytes(sock, payload, deadline):
    view = memoryview(payload)
    try:
        offset = 0
        while offset < len(view):
            wait = _remaining(deadline)
            if wait <= 0 or not select.select([], [sock], [], wait)[1]:
                raise TimeoutError('trace IPC write timed out')
            try:
                count = sock.send(view[offset:])
            except BlockingIOError:
                continue
            if not count:
                raise EOFError('trace IPC peer disconnected')
            offset += count
    finally:
        view.release()
        view = payload = None


def _recv_into(sock, payload, deadline):
    view = memoryview(payload)
    try:
        offset = 0
        while offset < len(view):
            wait = _remaining(deadline)
            if wait <= 0 or not select.select([sock], [], [], wait)[0]:
                raise TimeoutError('trace IPC read timed out')
            try:
                count = sock.recv_into(view[offset:], len(view) - offset)
            except BlockingIOError:
                continue
            if not count:
                raise EOFError('trace IPC peer disconnected')
            offset += count
    finally:
        view.release()
        view = payload = None


def _control_request(budget, session, name, control_bytes):
    # Both guards are rooted before the corresponding ledger mutation.
    session.request_guard = budget.request_guard(name)
    session.request_guard.open()
    session.request = session.request_guard.adopt()
    # Covers overlapping bounded control byte representations. Python object
    # overhead and kernel socket buffers are not covered by this ledger.
    session.control_guard = session.request.claim_guard(
        'trace-ipc', 'trace_ipc_control', 8 * max(control_bytes, 65536), items=1)
    session.control_guard.reserve()
    session.control_claim = session.control_guard.adopt()


def _clean_control(session):
    session.message = session.event = session.result = None
    if session.input_claim is not None:
        session.inputs = None
        session.input_claim.release()
        session.input_claim = session.input_guard = None
    if session.control_claim is not None:
        session.control_claim.release()
        session.control_claim = session.control_guard = None
    if session.request is not None:
        session.request.close()
        session.request = session.request_guard = None


class _Session:
    def __init__(self, sock=None):
        self.sock = sock
        self.request_guard = self.request = None
        self.control_guard = self.control_claim = None
        self.input_guard = self.input_claim = self.inputs = None
        self.payload_guard = self.payload_claim = self.payload = None
        self.message = self.event = self.result = self.failure = None
        self.identity = None
        self.event_ids = set()
        self.event_count = self.trace_bytes = 0
        self.clean = self.finished = False
        self.thread = None


_EVENT_FIELDS = {'version', 'op', 'broker_id', 'process_generation', 'job_id',
                 'generation', 'transfer_id', 'event_id', 'team_id', 'request_id',
                 'event', 'payload_bytes', 'byte_order'}
_ACK_FIELDS = ('broker_id', 'process_generation', 'job_id', 'generation',
               'transfer_id', 'event_id', 'team_id', 'request_id')


def _ack(header, op):
    return {'version': 2, 'op': op, **{name: header[name] for name in _ACK_FIELDS},
            'sequence': header['event']['sequence']}


def _same_wire(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected)
                and all(_same_wire(actual[key], value) for key, value in expected.items()))
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same_wire(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _exact_message(actual, expected):
    if not _same_wire(actual, expected):
        raise TraceIPCIncomplete('trace IPC acknowledgement does not match')


class TraceIPCServer:
    """Serve two authenticated trusted clients through one CPU trace owner."""

    def __init__(self, service, *, max_control_bytes=8192, max_event_bytes,
                 max_trace_bytes, timeout=30):
        from .cpu_shared_trace_service import CpuSharedTraceService
        if type(service) is not CpuSharedTraceService:
            raise ValueError('an exact CPU shared trace service is required')
        _limits(max_control_bytes, max_event_bytes, max_trace_bytes, timeout)
        # The wire schema and pre-forward byte calculation are float32 CPU.
        import torch
        model = service._owner._model
        if (model is None or any(p.device.type != 'cpu' or p.dtype != torch.float32
                                 for p in model.parameters())):
            raise ValueError('trace IPC requires a CPU float32 model')
        teams = tuple(service._clients)
        if len(teams) != 2 or any(not _identifier(team) for team in teams):
            raise ValueError('two bounded IPC team identifiers are required')
        self._service = service
        self._budget = service._owner._budget
        self._control = max_control_bytes
        self._event = max_event_bytes
        self._trace = max_trace_bytes
        self._timeout = float(timeout)
        self._tokens = {team: secrets.token_hex(32) for team in teams}
        self._invalid_token = secrets.token_hex(32)
        self._lock = threading.RLock()
        self._sessions = {}
        self._listener = self._accept_thread = None
        self._closing = self._incomplete = False
        self.address = None
        self.broker_id = service._owner._broker_id
        self.process_generation = service._owner._process_generation

    @property
    def team_tokens(self):
        return MappingProxyType(self._tokens)

    @property
    def incomplete(self):
        with self._lock:
            return self._incomplete

    @property
    def active_count(self):
        with self._lock:
            return len(self._sessions)

    def start(self):
        with self._lock:
            if self._listener is not None or self._closing:
                raise RuntimeError('trace IPC server has already started')
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(('127.0.0.1', 0))
            listener.listen(2)
            listener.settimeout(.1)
            self._listener = listener
            self.address = listener.getsockname()
            thread = threading.Thread(target=self._accept, name='trace-ipc-accept', daemon=True)
            self._accept_thread = thread
            try:
                thread.start()
            except BaseException:
                self._incomplete = True
                raise
        return self

    def _mark_incomplete(self):
        with self._lock:
            self._incomplete = True

    def _accept(self):
        try:
            while True:
                with self._lock:
                    if self._closing or self._incomplete:
                        return
                    listener = self._listener
                try:
                    sock, _address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    with self._lock:
                        if self._closing:
                            return
                    raise
                sock.setblocking(False)
                session = _Session(sock)
                key = uuid.uuid4().hex
                with self._lock:
                    if self._closing or self._incomplete or len(self._sessions) >= 2:
                        sock.close()
                        continue
                    self._sessions[key] = session
                thread = threading.Thread(target=self._serve, args=(key, session),
                                          name='trace-ipc-client', daemon=True)
                session.thread = thread
                try:
                    thread.start()
                except BaseException as error:
                    session.failure = error
                    self._mark_incomplete()
                    sock.close()
                    return
        except BaseException as error:
            with self._lock:
                self._accept_failure = error
                self._incomplete = True

    def _publish(self, session, message, deadline):
        # Order the decision against a prior known fault without holding the
        # lock during socket I/O. A decision that wins may send afterward.
        with self._lock:
            if self._incomplete:
                raise TraceIPCIncomplete('another trace IPC session is incomplete')
        _send(session.sock, message, self._control, deadline)

    def _validate_request(self, message):
        fields = {'version', 'op', 'token', 'team_id', 'request_id', 'input_ids',
                  'sample_id', 'positions', 'sites'}
        if (type(message) is not dict or set(message) != fields
                or type(message['version']) is not int or message['version'] != 2
                or message['op'] != 'trace'):
            raise TraceIPCRejected('invalid_request')
        team, token = message['team_id'], message['token']
        expected = self._tokens.get(team, self._invalid_token) if type(team) is str else self._invalid_token
        if (not _identifier(team) or type(token) is not str or not token.isascii()
                or not secrets.compare_digest(token, expected) or team not in self._tokens):
            raise TraceIPCRejected('unauthorized')
        row, positions, sites = message['input_ids'], message['positions'], message['sites']
        limits, owner = self._service._limits, self._service._owner
        if (not _identifier(message['request_id']) or not _identifier(message['sample_id'])
                or type(row) is not list or not 1 <= len(row) <= min(limits.max_sequence_length, 128)
                or any(type(v) is not int or not 0 <= v < owner._model.config.vocab_size for v in row)
                or type(positions) is not list or not 1 <= len(positions) <= limits.max_token_positions
                or any(type(v) is not int or not 0 <= v < len(row) for v in positions)
                or len(set(positions)) != len(positions)
                or type(sites) is not list or not 1 <= len(sites) <= min(limits.max_sites, limits.max_events)):
            raise TraceIPCRejected('invalid_request')
        seen = set()
        for site in sites:
            if (type(site) is not list or len(site) != 2 or type(site[0]) is not int
                    or not 0 <= site[0] < min(limits.max_layers, len(owner._model.transformer.h))
                    or type(site[1]) is not str or site[1] not in ('attn', 'mlp', 'block')
                    or (site[0], site[1]) in seen):
                raise TraceIPCRejected('invalid_request')
            seen.add((site[0], site[1]))
        size = len(positions) * owner._model.config.n_embd * 4
        if (size > min(self._event, owner._max_event_bytes)
                or size * len(sites) > min(self._trace, limits.max_trace_bytes)):
            raise TraceIPCRejected('trace_limit')

    def _serve(self, key, session):
        try:
            self._run(session)
        except BaseException as error:
            session.failure = error
            self._mark_incomplete()
        finally:
            if not session.clean:
                self._mark_incomplete()
            try:
                session.sock.close()
            except OSError:
                pass
            session.finished = True
            if session.clean:
                with self._lock:
                    self._sessions.pop(key, None)

    def _run(self, session):
        import torch
        deadline = time.monotonic() + self._timeout
        _control_request(self._budget, session, 'trace-ipc-server-' + uuid.uuid4().hex, self._control)
        session.message = _receive(session.sock, self._control, deadline)
        rejection_code = None
        try:
            self._validate_request(session.message)
        except TraceIPCRejected as rejection:
            rejection_code = str(rejection)
        if rejection_code is not None:
            try:
                _send(session.sock, {'version': 2, 'op': 'rejected', 'code': rejection_code},
                      self._control, deadline)
            except (OSError, EOFError, TimeoutError):
                # No job or tensor was created; a failed rejection write is
                # rooted as incomplete rather than claiming disposal.
                raise
            rejection_code = None
            _clean_control(session)
            session.clean = True
            return
        message = session.message
        team, request_id = message['team_id'], message['request_id']
        transfer = uuid.uuid4().hex
        session.input_guard = session.request.claim_guard(
            'trace-ipc-input', 'trace_ipc_input', len(message['input_ids']) * 8, items=1)
        session.input_guard.reserve()
        session.input_claim = session.input_guard.adopt()
        session.inputs = {'input_ids': torch.tensor([message['input_ids']], dtype=torch.long, device='cpu')}
        accepted = {'version': 2, 'op': 'accepted', 'broker_id': self.broker_id,
                    'process_generation': self.process_generation, 'transfer_id': transfer,
                    'team_id': team, 'request_id': request_id}
        self._publish(session, accepted, deadline)
        # A different handler may have failed while admission was sent.
        # The check is an ordered start decision; later faults cannot revoke it.
        with self._lock:
            if self._incomplete:
                raise TraceIPCIncomplete('another trace IPC session is incomplete')
        def send_event(manifest, payload):
            identity = manifest.identity
            with self._service._condition:
                attempt = self._service._active
                if (attempt is None or attempt.team_id != team or attempt.request_id != request_id
                        or attempt.job_id != identity.job_id or identity.team_id != team
                        or identity.request_id != request_id or identity.broker_id != self.broker_id):
                    raise TraceIPCIncomplete('trace IPC source identity changed')
                generation = attempt.generation
            current = (identity.job_id, generation, identity.run_id)
            if session.identity is None:
                session.identity = current
            if session.identity != current or manifest.event.sequence != session.event_count:
                raise TraceIPCIncomplete('trace IPC event ordering changed')
            if (type(payload) is not bytes or len(payload) != manifest.payload_bytes
                    or not 1 <= len(payload) <= self._event
                    or session.trace_bytes + len(payload) > self._trace):
                raise TraceIPCIncomplete('trace IPC event exceeds its bound')
            header = {'version': 2, 'op': 'event', 'broker_id': self.broker_id,
                'process_generation': self.process_generation, 'job_id': identity.job_id,
                'generation': generation, 'transfer_id': transfer, 'event_id': identity.delivery_id,
                'team_id': team, 'request_id': request_id, 'event': manifest.event.to_dict(),
                'payload_bytes': len(payload), 'byte_order': manifest.byte_order}
            self._publish(session, header, deadline)
            _exact_message(_receive(session.sock, self._control, deadline), _ack(header, 'ready'))
            _send_bytes(session.sock, struct.pack('!I', len(payload)), deadline)
            _send_bytes(session.sock, payload, deadline)
            _exact_message(_receive(session.sock, self._control, deadline), _ack(header, 'received'))
            _exact_message(_receive(session.sock, self._control, deadline), _ack(header, 'released'))
            session.event_count += 1
            session.trace_bytes += len(payload)
        samples = (SampleTrace(message['sample_id'], tuple(message['positions'])),)
        sites = tuple(SiteTrace(site[0], site[1]) for site in message['sites'])
        capability = self._service.client(team)
        session.result = capability.submit(request_id=request_id, inputs=session.inputs,
            input_owner=session.request, input_claim=session.input_claim,
            samples=samples, sites=sites, on_event=send_event)
        result = session.result
        if (session.identity != (result.job_id, result.generation, result.run_id)
                or result.team_id != team or result.request_id != request_id
                or result.broker_id != self.broker_id or result.process_generation != self.process_generation
                or result.event_count != session.event_count or result.trace_bytes != session.trace_bytes):
            raise TraceIPCIncomplete('complete trace identity changed')
        complete = {'version': 2, 'op': 'complete', 'broker_id': self.broker_id,
            'process_generation': self.process_generation, 'job_id': result.job_id,
            'generation': result.generation, 'transfer_id': transfer, 'team_id': team,
            'request_id': request_id, 'run_id': result.run_id, 'event_count': result.event_count,
            'trace_bytes': result.trace_bytes}
        self._publish(session, complete, deadline)
        confirmation = dict(complete, op='complete_received')
        _exact_message(_receive(session.sock, self._control, deadline), confirmation)
        capability.acknowledge(result)
        self._publish(session, dict(complete, op='closed'), deadline)
        _exact_message(_receive(session.sock, self._control, deadline), dict(complete, op='closed_received'))
        message = samples = sites = result = capability = send_event = None
        _clean_control(session)
        session.clean = True

    def close(self, timeout=5):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError('invalid close timeout')
        deadline = time.monotonic() + timeout
        if not self._lock.acquire(timeout=_remaining(deadline)):
            return False
        try:
            self._closing = True
            listener, accept_thread = self._listener, self._accept_thread
        finally:
            self._lock.release()
        if listener is not None:
            listener.close()
        if accept_thread is not None and accept_thread is not threading.current_thread():
            accept_thread.join(_remaining(deadline))
        if not self._lock.acquire(timeout=_remaining(deadline)):
            return False
        try:
            threads = tuple(s.thread for s in self._sessions.values() if s.thread is not None)
        finally:
            self._lock.release()
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(_remaining(deadline))
        if not self._lock.acquire(timeout=_remaining(deadline)):
            return False
        try:
            return (not self._incomplete and not self._sessions
                    and (accept_thread is None or not accept_thread.is_alive()))
        finally:
            self._lock.release()


class TraceIPCClient:
    """One cooperative receiver with its own independent host budget."""

    def __init__(self, address, *, token, team_id, broker_id, process_generation,
                 budget, max_control_bytes=8192, max_event_bytes, max_trace_bytes, timeout=30):
        _limits(max_control_bytes, max_event_bytes, max_trace_bytes, timeout)
        if (type(address) is not tuple or len(address) != 2 or address[0] != '127.0.0.1'
                or type(address[1]) is not int or not 1 <= address[1] <= 65535
                or type(token) is not str or not token.isascii() or not 1 <= len(token) <= 128
                or not _identifier(team_id) or not _hex(broker_id) or not _hex(process_generation)
                or type(budget) is not HostBudget):
            raise ValueError('invalid trusted trace IPC client configuration')
        self.address, self._token, self.team_id = address, token, team_id
        self.broker_id, self.process_generation = broker_id, process_generation
        self._budget, self._control = budget, max_control_bytes
        self._event, self._trace, self._timeout = max_event_bytes, max_trace_bytes, float(timeout)
        self._lock = threading.Lock()
        self._active = self._completion = None
        self._incomplete = False

    @property
    def incomplete(self):
        with self._lock:
            return self._incomplete

    def recover(self):
        with self._lock:
            if self._completion is None:
                raise TraceIPCIncomplete('no completed receipt is available')
            return self._completion

    def trace(self, *, request_id, input_ids, sample_id, positions, sites, on_event):
        # Snapshot bounded slices before validating or serializing. The caller
        # may mutate its lists concurrently; never copy an unbounded list whole.
        if type(input_ids) is not list or type(positions) is not list or type(sites) is not list:
            raise ValueError('invalid bounded trace IPC request')
        input_ids = input_ids[:129]
        positions = positions[:129]
        site_refs = sites[:257]
        if any(type(site) is not list for site in site_refs):
            raise ValueError('invalid bounded trace IPC sites')
        sites = [site[:3] for site in site_refs]
        site_refs = None
        if (not _identifier(request_id) or not _identifier(sample_id)
                or not 1 <= len(input_ids) <= 128
                or any(type(v) is not int or not 0 <= v < 2**31 for v in input_ids)
                or not 1 <= len(positions) <= 128
                or any(type(v) is not int or not 0 <= v < len(input_ids) for v in positions)
                or len(set(positions)) != len(positions)
                or not 1 <= len(sites) <= 256 or not callable(on_event)):
            raise ValueError('invalid bounded trace IPC request')
        if (any(type(site) is not list or len(site) != 2 or type(site[0]) is not int
                or not 0 <= site[0] < 2**31 or type(site[1]) is not str
                or site[1] not in ('attn', 'mlp', 'block') for site in sites)
                or len({tuple(site) for site in sites}) != len(sites)):
            raise ValueError('invalid bounded trace IPC sites')
        session = _Session()
        deadline = time.monotonic() + self._timeout
        try:
            with self._lock:
                if self._active is not None or self._incomplete or self._completion is not None:
                    raise TraceIPCIncomplete('trace IPC client is single-use or incomplete')
                self._active = session
            _control_request(self._budget, session, 'trace-ipc-client-' + uuid.uuid4().hex, self._control)
            session.sock = socket.create_connection(self.address, timeout=_remaining(deadline))
            session.sock.setblocking(False)
            session.message = {'version': 2, 'op': 'trace', 'token': self._token,
                'team_id': self.team_id, 'request_id': request_id, 'input_ids': input_ids,
                'sample_id': sample_id, 'positions': positions, 'sites': sites}
            _send(session.sock, session.message, self._control, deadline)
            session.message = _receive(session.sock, self._control, deadline)
            if session.message.get('op') == 'rejected' and set(session.message) == {'version', 'op', 'code'} \
                    and type(session.message['version']) is int and session.message['version'] == 2 \
                    and session.message['code'] in ('unauthorized', 'invalid_request', 'trace_limit'):
                code = session.message['code']
                _clean_control(session)
                session.clean = True
                with self._lock:
                    self._active = None
                raise TraceIPCRejected(code)
            accepted = session.message
            expected = {'version': 2, 'op': 'accepted', 'broker_id': self.broker_id,
                        'process_generation': self.process_generation, 'transfer_id': accepted.get('transfer_id'),
                        'team_id': self.team_id, 'request_id': request_id}
            if not _same_wire(accepted, expected) or not _hex(accepted['transfer_id']):
                raise TraceIPCIncomplete('invalid trace IPC admission')
            transfer = accepted['transfer_id']
            expected_sites = sorted((tuple(site) for site in sites),
                                    key=lambda item: (item[0], {'attn': 0, 'mlp': 1, 'block': 2}[item[1]]))
            while True:
                session.message = _receive(session.sock, self._control, deadline)
                if session.message.get('op') == 'complete':
                    break
                header = session.message
                if (set(header) != _EVENT_FIELDS or type(header['version']) is not int
                        or header['version'] != 2 or header['op'] != 'event'
                        or header['broker_id'] != self.broker_id
                        or header['process_generation'] != self.process_generation
                        or header['transfer_id'] != transfer or header['team_id'] != self.team_id
                        or header['request_id'] != request_id or not _hex(header['event_id'])
                        or header['event_id'] in session.event_ids
                        or type(header['job_id']) is not int or header['job_id'] < 1
                        or type(header['generation']) is not int or header['generation'] < 1
                        or type(header['payload_bytes']) is not int
                        or not 1 <= header['payload_bytes'] <= self._event
                        or session.trace_bytes + header['payload_bytes'] > self._trace
                        or header['byte_order'] not in ('little', 'big')):
                    raise TraceIPCIncomplete('invalid bounded trace event')
                event = TraceEvent.from_dict(header['event'])
                current = (header['job_id'], header['generation'], event.identity.run_id)
                if session.identity is None:
                    session.identity = current
                if (session.identity != current or event.sequence != session.event_count
                        or event.identity.request_id != request_id or event.identity.sample_id != sample_id
                        or event.identity.job_id != f'lg-{self.broker_id}-{header["job_id"]}'
                        or event.identity.run_id != f'run-{self.broker_id}-{header["job_id"]}'
                        or header['generation'] != header['job_id']
                        or event.token_positions != tuple(positions) or event.batch_positions != (0,)
                        or session.event_count >= len(expected_sites)
                        or (event.layer_index, event.module_kind) != expected_sites[session.event_count]
                        or event.module_path != (f'transformer.h.{event.layer_index}'
                            + ('' if event.module_kind == 'block' else f'.{event.module_kind}'))
                        or event.output_path != (() if event.module_kind == 'mlp' else (0,))
                        or event.forward_index != 0 or event.invocation != 0
                        or event.batch_axis != 0 or event.token_axis != 1
                        or event.dtype != 'torch.float32' or event.device != 'cpu'
                        or event.capture_mode != 'direct'
                        or event.saved_dtype != 'torch.float32' or event.saved_device != 'cpu'
                        or event.saved_shape is None or len(event.saved_shape) != 2
                        or event.saved_shape[0] != len(positions) or event.saved_shape[1] < 1
                        or event.shape != (1, len(input_ids), event.saved_shape[1])
                        or math.prod(event.saved_shape) * 4 != header['payload_bytes']):
                    raise TraceIPCIncomplete('trace event identity or layout changed')
                session.event_ids.add(header['event_id'])
                session.event = TraceIPCEvent(self.broker_id, self.process_generation, header['job_id'],
                    header['generation'], transfer, header['event_id'], self.team_id, request_id,
                    event, header['payload_bytes'], header['byte_order'])
                session.payload_guard = session.request.claim_guard(
                    'trace-ipc-receiver', 'trace_ipc_receive', header['payload_bytes'], items=1)
                session.payload_guard.reserve()
                session.payload_claim = session.payload_guard.adopt()
                _send(session.sock, _ack(header, 'ready'), self._control, deadline)
                length = struct.unpack('!I', _read_exact(session.sock, 4, deadline))[0]
                if length != header['payload_bytes']:
                    raise TraceIPCIncomplete('trace payload length changed')
                session.payload = bytearray(length)
                _recv_into(session.sock, session.payload, deadline)
                _send(session.sock, _ack(header, 'received'), self._control, deadline)
                session.result = on_event(session.event, session.payload)
                if session.result is not None:
                    raise TraceIPCIncomplete('trace receiver must finish synchronously and return None')
                session.payload = session.event = session.result = None
                session.payload_claim.release()
                session.payload_claim = session.payload_guard = None
                _send(session.sock, _ack(header, 'released'), self._control, deadline)
                session.event_count += 1
                session.trace_bytes += header['payload_bytes']
                event = header = None
            complete = session.message
            required = {'version', 'op', 'broker_id', 'process_generation', 'job_id', 'generation',
                        'transfer_id', 'team_id', 'request_id', 'run_id', 'event_count', 'trace_bytes'}
            if (set(complete) != required or type(complete['version']) is not int
                    or complete['version'] != 2
                    or complete['broker_id'] != self.broker_id
                    or complete['process_generation'] != self.process_generation
                    or complete['transfer_id'] != transfer or complete['team_id'] != self.team_id
                    or complete['request_id'] != request_id
                    or type(complete['job_id']) is not int or type(complete['generation']) is not int
                    or type(complete['event_count']) is not int or type(complete['trace_bytes']) is not int
                    or session.identity != (complete['job_id'], complete['generation'], complete['run_id'])
                    or complete['event_count'] != session.event_count or session.event_count != len(expected_sites)
                    or complete['trace_bytes'] != session.trace_bytes):
                raise TraceIPCIncomplete('incomplete or mismatched trace completion')
            completion = TraceIPCCompletion(self.broker_id, self.process_generation, complete['job_id'],
                complete['generation'], transfer, self.team_id, request_id, complete['run_id'],
                complete['event_count'], complete['trace_bytes'])
            _send(session.sock, dict(complete, op='complete_received'), self._control, deadline)
            _exact_message(_receive(session.sock, self._control, deadline), dict(complete, op='closed'))
            _send(session.sock, dict(complete, op='closed_received'), self._control, deadline)
            complete = accepted = expected = expected_sites = on_event = None
            _clean_control(session)
            session.clean = True
            with self._lock:
                self._completion = completion
                self._active = None
            return completion
        except BaseException as error:
            with self._lock:
                owned = self._active is session
            if not owned:
                raise
            session.failure = error
            with self._lock:
                self._incomplete = True
            if not isinstance(error, Exception):
                raise
            raise TraceIPCIncomplete('trace IPC delivery is incomplete') from None
        finally:
            with self._lock:
                if self._active is session and not session.clean:
                    self._incomplete = True
            if session.sock is not None:
                try:
                    session.sock.close()
                except OSError:
                    pass
