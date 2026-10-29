"""Lifecycle of the trusted GPU-owner service, independent of transport setup.

The enclosing process owns the cross-process lease and durable recovery
marker. Only a separate supervisor may clear a marker after this process exits.
"""
from dataclasses import dataclass
from enum import Enum
from threading import Event, Thread
from time import monotonic


class OwnerStatus(str, Enum):
    CLEAN = 'clean'
    QUARANTINED = 'quarantined'
    WORKER_FAILED = 'worker_failed'
    SHUTDOWN_INCOMPLETE = 'shutdown_incomplete'


@dataclass(frozen=True)
class OwnerResult:
    status: OwnerStatus
    error_type: str | None = None
    error_message: str | None = None
    teardown_verified: bool = False


def make_gpt2_request_factory(*, vocab_size, max_positions, max_batch, max_tokens):
    """Validate wire token matrices against a trusted pinned GPT-2 config."""
    for name, value, maximum in (
        ('vocab_size', vocab_size, 2**31), ('max_positions', max_positions, 2**31),
        ('max_batch', max_batch, 256), ('max_tokens', max_tokens, 8192),
    ):
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(f'invalid {name}')
    if max_tokens > max_positions:
        raise ValueError('max_tokens exceeds model positions')
    limits = {'input_ids': vocab_size, 'token_type_ids': vocab_size,
              'position_ids': max_positions, 'attention_mask': 2}

    def factory(request_id, inputs, seed):
        if type(inputs) is not dict or 'input_ids' not in inputs or not set(inputs) <= set(limits):
            raise ValueError('unsupported input names')
        shape = None
        for name, matrix in inputs.items():
            if type(matrix) is not list or not 1 <= len(matrix) <= max_batch:
                raise ValueError('invalid input batch')
            width = None
            for row in matrix:
                if type(row) is not list or not 1 <= len(row) <= max_tokens:
                    raise ValueError('invalid token sequence')
                if width is None:
                    width = len(row)
                if len(row) != width:
                    raise ValueError('ragged token sequence')
                if any(type(value) is not int or not 0 <= value < limits[name] for value in row):
                    raise ValueError('input value outside model bounds')
                if name == 'attention_mask' and not any(row):
                    raise ValueError('attention mask must expose at least one token')
            actual_shape = (len(matrix), width)
            if shape is None:
                shape = actual_shape
            elif shape != actual_shape:
                raise ValueError('input shapes differ')
        import torch
        from .resident_worker import ForwardRequest
        return ForwardRequest(request_id, {name: torch.tensor(matrix, dtype=torch.long)
                                           for name, matrix in inputs.items()}, seed)

    return factory


def _error(exc):
    try:
        message = str(exc)
    except BaseException:
        message = '<exception message unavailable>'
    return type(exc).__name__, message


class _CpuServiceWorker:
    """Adapt the resident worker's close state to the owner health contract."""

    def __init__(self, worker):
        self._worker = worker
        self._closed = False
        self.cancellation_exception = worker.cancellation_exception

    @property
    def poisoned(self):
        return self._closed or self._worker.poisoned

    @property
    def failed(self):
        return self._worker.poisoned

    @property
    def quarantined(self):
        return self._worker.quarantined

    def snapshot_request(self, request):
        return self._worker.snapshot_request(request)

    def execute(self, request, cancel_event=None):
        return self._worker.execute(request, cancel_event)

    def close(self):
        self._worker.close()
        self._closed = True


def serve_cpu_model(model, *, model_id, team_ids, stop_event, announce,
                    max_tokens=2, max_pending=8, max_pending_per_team=2,
                    max_connections=8, shutdown_timeout=10):
    """Run a trusted CPU model behind per-team local IPC; no GPU is used."""
    from .broker import Broker
    from .ipc import IPCServer
    from .resident_worker import ResidentWorker

    config = getattr(model, 'config', None)
    factory = make_gpt2_request_factory(vocab_size=getattr(config, 'vocab_size', None),
                                       max_positions=getattr(config, 'n_positions', None),
                                       max_batch=1, max_tokens=max_tokens)
    output_limit = 2 * 1024 * 1024
    response_limit = 4 * 1024 * 1024
    max_by_tensor = output_limit // (8 * config.vocab_size)
    max_by_wire = (response_limit - 512) // (32 * config.vocab_size)
    if max_tokens > min(max_by_tensor, max_by_wire):
        raise ValueError('max_tokens exceeds bounded full-logit response capacity')
    resident = ResidentWorker(model_id, model, device='cpu', max_batch_size=1,
                              max_sequence_length=max_tokens, max_input_bytes=64 * 1024,
                              max_output_bytes=output_limit)
    worker = _CpuServiceWorker(resident)
    broker = Broker(worker, max_pending=max_pending, max_pending_per_team=max_pending_per_team)
    try:
        server = IPCServer(broker, team_ids=team_ids, request_factory=factory,
                           max_connections=max_connections, max_batch=1, max_tokens=max_tokens,
                           max_response_bytes=response_limit)
    except BaseException:
        broker.close(timeout=shutdown_timeout)
        raise
    return serve_owner(server, broker, worker, stop_event, announce=announce,
                       verify_drain=lambda: True, shutdown_timeout=shutdown_timeout)


def serve_owner(server, broker, worker, stop_event, *, announce, verify_drain,
                poll_interval=0.1, shutdown_timeout=10):
    """Run an already-configured owner and report if teardown was verified.

    `announce(server)` must deliver credentials only to trusted recipients.
    `verify_drain()` must verify all device work and memory leases are complete
    after server/broker shutdown and return exactly True. Only a result with
    teardown_verified=True permits a clean-shutdown attestation; workload failure
    may coexist with verified teardown and must still be reported as failure.
    If QUARANTINED, the owner process must terminate promptly while retaining
    its lease until process exit.
    """
    if not callable(announce) or not callable(verify_drain):
        raise TypeError('announce and verify_drain must be callable')
    if type(poll_interval) not in (int, float) or not 0 < poll_interval <= 1:
        raise ValueError('poll_interval must be in (0, 1]')
    if type(shutdown_timeout) not in (int, float) or not 0 <= shutdown_timeout <= 300:
        raise ValueError('shutdown_timeout must be in [0, 300]')
    status = None
    failure = None
    startup_done = Event()
    startup_errors = []

    def startup():
        try:
            server.start()
            announce(server)
        except BaseException as exc:
            startup_errors.append(_error(exc))
        finally:
            startup_done.set()

    Thread(target=startup, name='looking-glass-owner-startup', daemon=True).start()
    startup_deadline = monotonic() + shutdown_timeout
    while not startup_done.wait(poll_interval):
        try:
            if worker.quarantined:
                status = OwnerStatus.QUARANTINED
                break
            if worker.poisoned or not broker.available:
                status = OwnerStatus.WORKER_FAILED
                break
        except BaseException as exc:
            status = OwnerStatus.QUARANTINED
            failure = _error(exc)
            break
        if stop_event.is_set() or monotonic() >= startup_deadline:
            status = OwnerStatus.SHUTDOWN_INCOMPLETE
            break

    if status is None and startup_errors:
        status = OwnerStatus.SHUTDOWN_INCOMPLETE
        failure = startup_errors[0]
    if status is None:
        while True:
            try:
                if worker.quarantined:
                    status = OwnerStatus.QUARANTINED
                    break
                if worker.poisoned or not broker.available:
                    status = OwnerStatus.WORKER_FAILED
                    break
            except BaseException as exc:
                status = OwnerStatus.QUARANTINED
                failure = _error(exc)
                break
            if stop_event.wait(poll_interval):
                break

    if status is OwnerStatus.QUARANTINED:
        _request_quarantine_cleanup(server, broker)
        return OwnerResult(status, *(failure or (None, None)))

    # A startup thread that has not finished can still create or announce a
    # server after cleanup. Request shutdown, but never attest that teardown.
    if not startup_done.is_set():
        _request_quarantine_cleanup(server, broker)
        return OwnerResult(status or OwnerStatus.SHUTDOWN_INCOMPLETE,
                           *(failure or (None, None)))

    deadline = monotonic() + shutdown_timeout
    done = Event()
    results = []

    def finish():
        try:
            if status is OwnerStatus.WORKER_FAILED:
                # Let already accepted jobs receive their terminal failures.
                # Admission is closed first; a bounded grace cannot delay a
                # concurrent quarantine, which the owner thread still polls.
                server.stop_accepting(timeout=max(0, min(1.0, deadline - monotonic())))
                server.drain(timeout=max(0, min(1.0, deadline - monotonic())))
            server_stopped = server.close(timeout=max(0, deadline - monotonic()))
            broker_stopped = broker.close(timeout=max(0, deadline - monotonic()))
            if worker.quarantined:
                result = OwnerResult(OwnerStatus.QUARANTINED)
            elif not server_stopped or not broker_stopped:
                result = OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE)
            elif verify_drain() is not True or worker.quarantined:
                result = OwnerResult(OwnerStatus.QUARANTINED)
            else:
                final_status = status or (OwnerStatus.WORKER_FAILED if worker.failed or broker.failed else OwnerStatus.CLEAN)
                result = OwnerResult(final_status, *(failure or (None, None)), teardown_verified=True)
        except BaseException as exc:
            result = OwnerResult(OwnerStatus.QUARANTINED, *_error(exc))
        results.append(result)
        done.set()

    Thread(target=finish, name='looking-glass-owner-shutdown', daemon=True).start()
    while True:
        try:
            if worker.quarantined:
                _request_quarantine_cleanup(server, broker)
                return OwnerResult(OwnerStatus.QUARANTINED)
        except BaseException as exc:
            _request_quarantine_cleanup(server, broker)
            return OwnerResult(OwnerStatus.QUARANTINED, *_error(exc))
        remaining = deadline - monotonic()
        if done.wait(max(0, min(poll_interval, remaining))):
            try:
                if worker.quarantined:
                    return OwnerResult(OwnerStatus.QUARANTINED)
                if results[0].status is OwnerStatus.CLEAN and (worker.failed or broker.failed):
                    return OwnerResult(OwnerStatus.WORKER_FAILED, teardown_verified=True)
            except BaseException as exc:
                return OwnerResult(OwnerStatus.QUARANTINED, *_error(exc))
            return results[0]
        if remaining <= 0:
            return OwnerResult(OwnerStatus.SHUTDOWN_INCOMPLETE)


def _request_quarantine_cleanup(server, broker):
    """Best-effort signals; a failed device drain must not block process exit."""
    def close_server():
        try:
            server.close(timeout=0)
        except BaseException:
            pass

    def close_broker():
        try:
            broker.close(wait=False)
        except BaseException:
            pass

    Thread(target=close_server, name='looking-glass-close-ipc', daemon=True).start()
    Thread(target=close_broker, name='looking-glass-close-broker', daemon=True).start()
