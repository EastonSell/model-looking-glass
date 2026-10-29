"""Two bounded concurrent clients for the separately gated direct owner.

The caller starts and retains the prepared owner. This client does not certify
owner teardown, GPU admission, host RSS, sustained fairness, or device overlap.
"""
from dataclasses import dataclass
import math
import sys
import threading
import time

from . import gpu_service as _owner
from .ipc import IPCClient, IPCRejected, CancelAckObservation, _IDENTIFIER, _supported_handler_runtime
from .two_team_service_smoke import (_CLAIMED, _CLAIM_LOCK, _REQUEST_ID,
                                     _REQUIRED_DIRECT_CONTRACT, _result_identity)
from .two_token_service import _PreparedService, _REQUIRED_COMPLETION_CONTRACT


class ConcurrentClientError(RuntimeError):
    """The bounded client plan or an owner compatibility gate failed."""


@dataclass(frozen=True)
class RequestObservation:
    team_id: str
    seed: int
    status: str
    broker_id: str | None = None
    job_id: int | None = None
    submitted_ns: int | None = None
    ready_ns: int | None = None
    started_ns: int | None = None
    finished_ns: int | None = None
    cancel_ack: bool | None = None
    float32_sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ConcurrentClientReport:
    """Client observations only; supervisor and marker checks are always required.

    A false ``requires_owner_reconciliation`` means these clients detected no
    listed fault. It does not prove that client thread machinery or the owner
    terminated cleanly, and is not an owner-teardown attestation.
    """
    requests: tuple[RequestObservation, RequestObservation]
    serial_execution_observed: bool | None
    queueing_observed: bool | None
    requires_owner_reconciliation: bool
    fairness_under_backlog: str = 'not_measured'
    activation_trace: str = 'disabled'


class _Slot:
    def __init__(self, team, seed, cancel):
        self.team, self.seed, self.cancel = team, seed, cancel
        self.admitted = threading.Event()
        self.client = self.handle = self.thread = None
        self.observation = self.failure = self.close_failure = self.hook_failure = None
        self.cancel_failure = None
        self.cancel_inflight = False
        self.completed = False


def _times(response):
    submitted, ready, started, finished = (response.get(name) for name in
        ('submitted_ns', 'ready_ns', 'started_ns', 'finished_ns'))
    if any(type(value) is not int or not 0 <= value < 2**63
           for value in (submitted, ready, finished)):
        raise ConcurrentClientError('invalid broker timestamps')
    if not submitted <= ready <= finished:
        raise ConcurrentClientError('invalid broker timestamp order')
    if started is not None and (type(started) is not int or not ready <= started <= finished):
        raise ConcurrentClientError('invalid broker timestamp order')
    return submitted, ready, started, finished


def _terminal(slot, response):
    handle = slot.handle
    if type(response) is not dict:
        raise ConcurrentClientError('invalid terminal response')
    ack = handle.cancel_ack_observation
    if handle.cancel_ack_observation_failed:
        raise ConcurrentClientError('cancellation observation is incomplete')
    if ack is not None and (type(ack) is not CancelAckObservation or type(ack.accepted) is not bool
                            or type(ack.received_ns) is not int or not 0 <= ack.received_ns < 2**63
                            or not slot.cancel):
        raise ConcurrentClientError('invalid cancellation acknowledgement')
    accepted = None if ack is None else ack.accepted
    status = response.get('status')
    if status == 'timeout':
        return RequestObservation(slot.team, slot.seed, 'unresolved', handle.broker_id,
                                  handle.job_id, cancel_ack=accepted, error='timeout')
    if status not in ('succeeded', 'cancelled', 'failed'):
        raise ConcurrentClientError('invalid terminal status')
    submitted, ready, started, finished = _times(response)
    digest = None
    error = None
    if status == 'succeeded':
        if accepted is True or started is None:
            raise ConcurrentClientError('inconsistent successful outcome')
        broker_id, job_id, digest = _result_identity(response, slot.team)
        if broker_id != handle.broker_id or job_id != handle.job_id:
            raise ConcurrentClientError('inconsistent outcome identity')
    elif status == 'failed':
        error = ('out_of_memory' if response.get('error') in
                 ('WorkerOutOfMemory', 'OutOfMemoryError', 'MemoryError', 'serialization_out_of_memory')
                 else 'worker_or_transport_failure')
    return RequestObservation(slot.team, slot.seed, status, handle.broker_id, handle.job_id,
                              submitted, ready, started, finished, accepted, digest, error)


class ConcurrentClients:
    """A one-shot pair of fixed requests through an already-started service.

    Retain this object and the prepared service until the caller has joined the
    clients and reconciled the service supervisor and unsafe marker. A None
    wait result means clients remain in flight; wait again or perform supervised
    shutdown. A terminal client report alone never proves owner teardown.
    """
    def __init__(self, service, *, seeds=(29, 29), cancel=(False, False)):
        if type(service) is not _PreparedService:
            raise TypeError('a prepared fixed service is required')
        direct = getattr(_owner, 'DIRECT_SERVICE_OWNER_CONTRACT', None)
        completion = getattr(_owner, 'OWNER_COMPLETION_CUSTODY_CONTRACT', None)
        if (type(direct) is not str or direct != _REQUIRED_DIRECT_CONTRACT
                or type(completion) is not str or completion != _REQUIRED_COMPLETION_CONTRACT):
            raise ConcurrentClientError('reviewed direct service and completion contracts are unavailable')
        teams = service._teams
        if (type(teams) is not tuple or len(teams) != 2 or teams[0] == teams[1]
                or any(type(team) is not str or _IDENTIFIER.fullmatch(team) is None for team in teams)):
            raise ValueError('exactly two configured teams are required')
        if (type(seeds) is not tuple or len(seeds) != 2
                or any(type(seed) is not int or seed not in (29, 43, 71) for seed in seeds)
                or type(cancel) is not tuple or len(cancel) != 2
                or any(type(value) is not bool for value in cancel)):
            raise ValueError('invalid fixed request plan')
        if not _supported_handler_runtime():
            raise ConcurrentClientError('client thread fault observation is unavailable')
        self._service = service
        self._slots = tuple(_Slot(team, seed, flag) for team, seed, flag in zip(teams, seeds, cancel))
        self._go = threading.Event()
        self._aborted = False
        self._started = False
        self._starting = False
        self._start_failure = None
        self._control_lock = threading.Lock()
        self._final_report = None

    def _run(self, slot):
        try:
            self._go.wait()
            if self._aborted:
                return
            slot.handle = slot.client.submit(team_id=slot.team, request_id=_REQUEST_ID,
                inputs={'input_ids': [[15496, 995]]}, seed=slot.seed)
            slot.admitted.set()
            if slot.cancel:
                slot.handle.cancel()
            response = slot.handle.wait(timeout=60)
            slot.observation = _terminal(slot, response)
            response = None
        except BaseException as exc:
            slot.failure = exc
            if type(exc) is IPCRejected and type(exc.code) is str and exc.code in ('busy', 'queue_full', 'invalid_request'):
                slot.observation = RequestObservation(slot.team, slot.seed, 'rejected', error=exc.code)
            else:
                slot.observation = RequestObservation(slot.team, slot.seed, 'unresolved', error='client_failure')
        finally:
            # IPCHandle.wait closes the socket even on a timeout. A submission
            # exception may have followed server acceptance, so never retry it.
            if slot.handle is not None:
                try:
                    slot.handle.close()
                except BaseException as exc:
                    slot.close_failure = exc
                    slot.observation = RequestObservation(slot.team, slot.seed, 'unresolved', error='close_failure')
            slot.completed = True

    @staticmethod
    def _observe_thread(slot, thread):
        previous = thread._invoke_excepthook
        def observe(target):
            try:
                slot.hook_failure = sys.exc_info()[1]
                slot.completed = False
            finally:
                previous(target)
        thread._invoke_excepthook = observe

    def start(self):
        """Start both clients once. On any interruption retain and reconcile."""
        with self._control_lock:
            if self._started:
                raise ConcurrentClientError('client plan already started')
            self._started = True
            self._starting = True
        try:
            with _CLAIM_LOCK:
                if self._service in _CLAIMED:
                    raise ConcurrentClientError('prepared service already has a client operation')
                _CLAIMED.add(self._service)
            for slot in self._slots:
                original = self._service.client(slot.team)
                if type(original) is not IPCClient:
                    raise ConcurrentClientError('service returned an invalid client')
                slot.client = IPCClient(original.address, original.token,
                    io_timeout=original.io_timeout, max_response_bytes=original.max_response_bytes,
                    observe_cancel_ack=True)
                slot.thread = threading.Thread(target=self._run, args=(slot,), daemon=True,
                                               name='looking-glass-direct-client')
                self._observe_thread(slot, slot.thread)
            for slot in self._slots:
                slot.thread.start()
        except BaseException as exc:
            self._aborted = True
            self._start_failure = exc
            raise
        finally:
            try:
                self._go.set()
            except BaseException as exc:
                self._aborted = True
                self._start_failure = exc
                raise
            else:
                with self._control_lock:
                    self._starting = False

    def cancel(self, team):
        """Send a cooperative cancellation after admission, without claiming acceptance.

        Read the terminal report for the optional acknowledgement. An error
        sending the request leaves its outcome unknown and requires supervisor
        reconciliation; the caller retains this object even if this raises.
        """
        if not self._started:
            raise ConcurrentClientError('client plan has not started')
        with self._control_lock:
            if self._final_report is not None:
                raise ConcurrentClientError('client report is already final')
            slot = next((item for item in self._slots if type(team) is str and item.team == team), None)
            if slot is None or not slot.admitted.is_set():
                raise ConcurrentClientError('client request has not been admitted')
            if slot.cancel_inflight:
                raise ConcurrentClientError('cancellation is already in progress')
            slot.cancel = True
            slot.cancel_inflight = True
        try:
            slot.handle.cancel()
        except BaseException as exc:
            with self._control_lock:
                slot.cancel_failure = exc
                slot.cancel_inflight = False
            raise
        else:
            with self._control_lock:
                slot.cancel_inflight = False

    def wait_admitted(self, team, *, timeout=60):
        """Wait for a validated admission; False does not establish rejection."""
        if not self._started:
            raise ConcurrentClientError('client plan has not started')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('invalid admission timeout')
        slot = next((item for item in self._slots if type(team) is str and item.team == team), None)
        if slot is None:
            raise ValueError('unknown team')
        return slot.admitted.wait(timeout)

    def wait(self, *, timeout=60):
        """Return None while a client runs, otherwise bounded terminal metadata."""
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('invalid wait timeout')
        with self._control_lock:
            if not self._started:
                raise ConcurrentClientError('client plan has not started')
            if self._starting:
                return None
        deadline = time.monotonic() + timeout
        for slot in self._slots:
            thread = slot.thread
            if thread is not None and thread.ident is not None:
                thread.join(max(0, deadline - time.monotonic()))
                if thread.is_alive():
                    return None
        with self._control_lock:
            if self._final_report is not None:
                return self._final_report
            if any(slot.cancel_inflight for slot in self._slots):
                return None
            rows = tuple(slot.observation if slot.completed and slot.hook_failure is None and slot.observation is not None
                         else RequestObservation(slot.team, slot.seed, 'unresolved', error='client_thread_failure')
                         for slot in self._slots)
            needs_reconciliation = (self._start_failure is not None or
                                    any(slot.cancel_failure is not None for slot in self._slots) or
                                    any(row.status in ('failed', 'unresolved', 'rejected') for row in rows))
            admitted = [row for row in rows if row.broker_id is not None]
            if len(admitted) == 2 and (admitted[0].broker_id != admitted[1].broker_id
                                      or admitted[0].job_id == admitted[1].job_id):
                needs_reconciliation = True
            started = [row for row in rows if row.started_ns is not None and row.finished_ns is not None]
            serial = queueing = None
            if len(started) == 2:
                first, second = sorted(started, key=lambda row: row.started_ns)
                if first.broker_id != second.broker_id or first.job_id == second.job_id:
                    needs_reconciliation = True
                else:
                    serial = first.finished_ns <= second.started_ns
                    queueing = second.ready_ns < first.finished_ns <= second.started_ns
                    if not serial:
                        needs_reconciliation = True
            self._final_report = ConcurrentClientReport(rows, serial, queueing, needs_reconciliation)
            return self._final_report
