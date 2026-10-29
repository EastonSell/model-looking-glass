"""Owner-private serial observations for the fixed-workspace service path.

This metadata adapter does not provide device admission, session verification,
resource custody, IPC ownership or a measured performance result.
"""

import threading

from .broker import Broker, RequestHandle, RequestState
from .ipc import IPCServer
from .owner_completion import LifetimeWorkspaceWorker, OwnerCompletionAuthority
from .serial_observation import SerialObservation


class _ObservationCallback:
    """Pre-root one metadata callback and its private failure owners."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.handle = None
        self.outcome = None
        self.failure = None
        self.interruption = None


class ServiceSerialObservation(SerialObservation):
    """Observe one exact bound broker and lifetime-workspace owner epoch."""

    def __init__(self, recorder):
        super().__init__(recorder)
        self._worker = None
        self._authority = None
        self._resident = None
        self._guard = None
        self._server = None
        self._terminal_capture = _ObservationCallback()
        self._registration_callback = _ObservationCallback()

    def bind_service(self, worker, broker, server):
        """Bind the exact authenticated service graph before listener startup."""
        if (type(server) is not IPCServer or type(broker) is not Broker
                or broker._worker is not worker or server.broker is not broker):
            raise ValueError('the service must own the exact worker, broker and server')
        try:
            with server._lifecycle:
                if (server._listener is not None or server._accept_thread is not None
                        or server._accepting_stopped.is_set() or not server.team_tokens):
                    raise ValueError('bind before an authenticated listener starts')
                self.bind_broker(broker)
                self._server = server
        except BaseException:
            # If broker attachment committed but the service binding did not,
            # the owner must not expose a partly bound service.
            if self._broker is not None and self._server is not server:
                self._unsafe_owner()
            raise

    def bind_broker(self, broker):
        """Bind after warmup and before the broker is exposed or accepts jobs."""
        if type(broker) is not Broker:
            raise ValueError('an exact service broker is required')
        worker = broker._worker
        authority = broker._completion_authority
        if type(worker) is not LifetimeWorkspaceWorker or type(authority) is not OwnerCompletionAuthority:
            raise ValueError('an exact lifetime worker and completion authority are required')
        if worker._worker.device.type != 'cpu':
            raise ValueError('CUDA service observation is disabled pending a measured instrumented profile')
        if (getattr(worker, '_cpu_observation', None) is not self
                or getattr(worker, '_cpu_observation_binding', None) is not self):
            raise ValueError('worker was not constructed with this CPU observation capability')
        mutating = False
        try:
            with broker._condition:
                if (self._broker is not None or self._worker is not None
                        or not hasattr(broker, '_serial_observation')
                        or broker._serial_observation is not None
                        or worker._observation_bound
                        or worker._authority is not authority or authority._worker is not worker
                        or authority._broker is not broker or authority.unsafe or worker.quarantined
                        or worker._worker._owner_completion is not authority
                        or worker._worker._model is not worker._model
                        or authority._device_guard is not worker._device_guard
                        or worker._cold_pending or broker._next_job_id != 0 or broker._pending_slots
                        or broker._running is not None or broker._closed or broker.quarantined):
                    raise ValueError('bind the exact warm owner before any broker submission')
                mutating = True
                self._attach(broker)
                self._worker, self._authority = worker, authority
                self._resident, self._guard = worker._worker, authority._device_guard
                broker._serial_observation = self
                worker._observation_bound = True
        except BaseException:
            # A partial attachment cannot be used as evidence or as a service
            # startup. The owning process retains its normal custody roots.
            self._fault()
            if mutating:
                authority.mark_unsafe()
            raise

    def _submit(self, team_id, request):
        # The exact Broker source hook registers its returned handle. Avoid a
        # second registration; IPC continues to use the ordinary Broker API.
        return self._broker.submit(team_id, request)

    def wait(self, handle, timeout=None):
        """Wait on the real handle; an unresolved poll is not event loss."""
        if type(handle) is not RequestHandle or handle._broker is not self._broker:
            raise ValueError('handle belongs to another serial broker')
        try:
            with self._registration_lock:
                registration = self._registered.get(handle.job_id)
                if registration is None:
                    self._fault()
                elif registration[0]() is not handle:
                    self._fault()
                    raise ValueError('handle does not match its registered submission')
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception) or isinstance(exc, ValueError):
                raise
        try:
            # RequestHandle.wait already captures the real terminal outside
            # the broker/authority locks. Do not emit a duplicate capture.
            return handle.wait(timeout)
        except TimeoutError:
            raise
        except BaseException:
            self._fault()
            raise

    def _registered_handle(self, handle):
        try:
            self._register(handle, handle.team_id, handle.request_id)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise

    def _captured_handle(self, handle, outcome):
        try:
            self._capture(handle, outcome)
        except BaseException as exc:
            self._fault()
            if not isinstance(exc, Exception):
                raise

    def _capture_bounded(self, handle, outcome):
        """Run at most one terminal callback; never block competing waits."""
        return self._callback_bounded(self._terminal_capture, handle, outcome, capture=True)

    def _register_bounded(self, handle):
        """Run at most one registration callback without delaying others."""
        return self._callback_bounded(self._registration_callback, handle, None, capture=False)

    def _callback_bounded(self, state, handle, outcome, *, capture):
        try:
            return self._callback_bounded_inner(state, handle, outcome, capture=capture)
        except BaseException as exc:
            # The pre-rooted state keeps both the callback failure and a
            # distinct interruption in its cleanup/entry. The service owner
            # retains this observation for the lifetime of the unsafe owner.
            if state.interruption is None:
                state.interruption = exc
            self._unsafe_owner()
            raise

    def _callback_bounded_inner(self, state, handle, outcome, *, capture):
        if self._broker.quarantined or state.active or not state.lock.acquire(False):
            self._fault()
            return
        try:
            # Recheck after taking the gate. An interrupted preceding callback
            # leaves active set until process replacement.
            if self._broker.quarantined or state.active:
                self._fault()
                return
            state.active = True
            state.handle, state.outcome = handle, outcome
            try:
                if capture:
                    self._captured_handle(handle, outcome)
                else:
                    self._registered_handle(handle)
            except BaseException as exc:
                state.failure = exc
                raise
            else:
                state.handle = state.outcome = None
                state.active = False
        finally:
            state.lock.release()

    def _unsafe_owner(self):
        self._fault()
        authority = self._authority
        if authority is not None:
            try:
                authority.mark_unsafe()
            finally:
                # Retain the shared nonblocking latch if the signaling call
                # is itself interrupted once.
                authority._unsafe = True

    def _identity(self, worker, handle, request):
        broker = self._broker
        if broker is None or threading.current_thread() is not broker._thread:
            raise RuntimeError('only the bound broker owner may execute this worker')
        try:
            with broker._condition:
                if (broker._serial_observation is not self or broker._worker is not worker
                        or worker is not self._worker or worker._authority is not self._authority
                        or worker._cpu_observation is not self
                        or worker._cpu_observation_binding is not self
                        or broker._completion_authority is not self._authority
                        or self._authority._broker is not broker or self._authority._worker is not worker
                        or worker._worker is not self._resident
                        or self._resident._owner_completion is not self._authority
                        or self._resident._model is not worker._model
                        or self._authority._device_guard is not self._guard
                        or worker._device_guard is not self._guard
                        or (self._server is not None and self._server.broker is not broker)
                        or type(handle) is not RequestHandle or broker._running is not handle
                        or handle._state is not RequestState.RUNNING
                        or handle._worker_request_id != request.request_id):
                    raise RuntimeError('running owner, worker or broker identity changed')
                try:
                    # Only this bounded metadata allocation is optional after
                    # the exact ownership checks have succeeded.
                    return (handle.job_id, handle.team_id, handle.request_id, handle._worker_request_id)
                except Exception:
                    self._fault()
                    return None
        except BaseException:
            self._unsafe_owner()
            raise

    def _execute(self, worker, handle, request, cancel_event):
        """Execute the real worker on its broker thread with an optional scope."""
        if self._broker is None or threading.current_thread() is not self._broker._thread:
            self._unsafe_owner()
            raise RuntimeError('only the bound broker owner may execute this worker')
        scope = None
        try:
            identity = self._identity(worker, handle, request)
            if identity is None:
                # Optional metadata construction failed after ownership was
                # verified; the underlying request can still run.
                self._abandon_forward(None)
                return worker.execute(request, cancel_event)
            self._note_execution(identity[0])
            scope = self._start_serial(identity)
            result = worker.execute(request, cancel_event)
        except BaseException:
            self._settle_serial(scope, failed=True)
            raise
        self._settle_serial(scope, failed=False)
        return result
