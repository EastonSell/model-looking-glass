"""Private lifecycle composition for a fair IPC service.

The components and their health observations are trusted owner-side objects.
This module does not construct a service, authorize a model, provide a process
supervisor, or measure GPU or host memory. The caller must retain this graph
through owner-process exit and replace the process after uncertain teardown.
"""

from threading import Lock, current_thread
from time import perf_counter


class _FairOwnerLifecycle:
    """Adapt a server/bridge/broker graph to the existing owner watchdog.

    The real broker's ``health_snapshot`` and all quarantine properties must
    be nonblocking. The bridge's public dispatcher observation is only a
    liveness signal; it cannot establish that dispatch is making progress.
    No work may be submitted except through the paired server and bridge.
    """

    def __init__(self, worker, broker, bridge, server, *, owner_quarantined):
        if not callable(owner_quarantined):
            raise TypeError('a nonblocking owner-quarantine observation is required')
        self.worker, self.broker, self.bridge, self.server = worker, broker, bridge, server
        self._owner_quarantined = owner_quarantined
        self._unsafe = False
        # Distinct in-flight categories have independent, preallocated slots;
        # each pair retains a primary error and an interrupted retention error.
        self._errors = [None] * 16
        self._phase = 'initial'
        self._start_pending = self._server_pending = False
        self._bridge_pending = self._broker_pending = False
        self._start_thread = self._server_thread = None
        self._bridge_thread = self._broker_thread = None
        self._server_stopped = self._bridge_stopped = self._broker_stopped = False
        self._server_lock, self._broker_lock = Lock(), Lock()
        self._health_locks = tuple(Lock() for _ in range(4))
        self._broker_failed = False
        self.server_view, self.broker_view = _ServerView(self), _BrokerView(self)

    def _fail(self, error, slot):
        try:
            self._unsafe = True
            if self._errors[slot] is None:
                self._errors[slot] = error
            elif self._errors[slot] is not error and self._errors[slot + 1] is None:
                self._errors[slot + 1] = error
        except BaseException as secondary:
            # Retry while the original traceback is still in this frame. A
            # bounded pair of slots cannot guarantee retention against repeated
            # arbitrary asynchronous interruptions; process replacement is
            # still mandatory when this path is entered.
            self._unsafe = True
            self._recover(error, secondary, slot)
            raise

    def _recover(self, error, secondary, slot):
        self._unsafe = True
        if self._errors[slot] is None:
            self._errors[slot] = error
        if self._errors[slot + 1] is None and secondary is not self._errors[slot]:
            self._errors[slot + 1] = secondary

    @staticmethod
    def _dead_pending(pending, thread):
        return pending and (thread is None or not thread.is_alive())

    def _health(self):
        health = self.broker.health_snapshot
        values = tuple(getattr(health, name) for name in
                       ('failed', 'quarantined', 'closed', 'owner_alive'))
        if any(type(value) is not bool for value in values):
            raise RuntimeError('broker health fields must be exact booleans')
        return health

    def _observe(self):
        if self._unsafe:
            return True, None
        # Separate bounded slots allow the owner and shutdown threads to read
        # health simultaneously, without waiting on a possibly stuck reader.
        for index, lock in enumerate(self._health_locks):
            acquired = False
            try:
                acquired = lock.acquire(False)
                if not acquired:
                    continue
                if self._unsafe:
                    return True, None
                if any(self._dead_pending(pending, thread) for pending, thread in (
                    (self._start_pending, self._start_thread),
                    (self._server_pending, self._server_thread),
                    (self._bridge_pending, self._bridge_thread),
                    (self._broker_pending, self._broker_thread),
                )):
                    self._unsafe = True
                    return True, None
                health = self._health()
                if health.failed:
                    self._broker_failed = True
                flags = (self.worker.quarantined, self.broker.quarantined,
                         self.bridge.quarantined, self.server.custody_quarantined,
                         health.quarantined, self._owner_quarantined())
                if any(type(flag) is not bool for flag in flags):
                    raise RuntimeError('quarantine signals must be exact booleans')
                if any(flags):
                    self._unsafe = True
                    return True, None
                if self._phase in ('serving', 'stopping'):
                    dispatcher_alive = self.bridge.dispatcher_alive
                    if type(dispatcher_alive) is not bool:
                        raise RuntimeError('dispatcher liveness must be an exact boolean')
                    unexpected_dispatch_exit = (not self._bridge_pending and not self._bridge_stopped
                                                and not dispatcher_alive)
                    unexpected_broker_exit = (not self._broker_pending and not self._broker_stopped
                                               and (health.closed or
                                                    (not health.owner_alive and not health.failed)))
                    if unexpected_dispatch_exit or unexpected_broker_exit:
                        self._unsafe = True
                        return True, None
                return self._unsafe, health
            except BaseException as error:
                slot = 8 + index * 2
                try:
                    self._fail(error, slot)
                except BaseException as secondary:
                    self._recover(error, secondary, slot)
                    raise
                return True, None
            finally:
                if acquired:
                    lock.release()
        self._unsafe = True
        return True, None

    @property
    def quarantined(self):
        return self._observe()[0]

    def start(self):
        if self._phase != 'initial' or self.quarantined:
            raise RuntimeError('fair owner cannot start')
        self._start_thread = current_thread()
        try:
            self._start_pending = True
            self._phase = 'starting'
            self.bridge.start()
            if self.quarantined:
                raise RuntimeError('fair owner became unsafe during dispatcher start')
            dispatcher_alive = self.bridge.dispatcher_alive
            if type(dispatcher_alive) is not bool or not dispatcher_alive:
                raise RuntimeError('fair dispatcher did not remain alive during startup')
            if self.quarantined:
                raise RuntimeError('fair owner became unsafe before listener start')
            self.server.start()
            if self.quarantined:
                raise RuntimeError('fair owner became unsafe during listener start')
            self._phase = 'serving'
            if self.quarantined:
                raise RuntimeError('fair owner is unsafe after listener start')
            self._start_pending = False
            return self.server_view
        except BaseException as error:
            try:
                self._fail(error, 2)
            except BaseException as secondary:
                self._recover(error, secondary, 2)
                raise
            raise
        finally:
            if self._start_pending:
                self._unsafe = True

    def control_server(self, operation, timeout):
        if operation not in ('stop_accepting', 'drain'):
            raise ValueError('unsupported server control operation')
        if self._start_pending or self.quarantined:
            return False
        deadline = perf_counter() + max(0, timeout)
        acquired = False
        try:
            acquired = self._server_lock.acquire(timeout=max(0, timeout))
            if not acquired:
                return False
            if self.quarantined:
                return False
            self._server_thread = current_thread()
            self._server_pending = True
            try:
                result = getattr(self.server, operation)(timeout=max(0.0, deadline - perf_counter()))
                if type(result) is not bool:
                    raise RuntimeError('server control must return an exact boolean')
                self._server_pending = False
                return result
            except BaseException as error:
                try:
                    self._fail(error, 0)
                except BaseException as secondary:
                    self._recover(error, secondary, 0)
                    raise
                raise
            finally:
                if self._server_pending:
                    self._unsafe = True
        except BaseException:
            self._unsafe = True
            raise
        finally:
            if self._server_pending:
                self._unsafe = True
            if acquired:
                self._server_lock.release()

    def stop_server(self, timeout):
        deadline = perf_counter() + max(0, timeout)
        if self._start_pending or self.quarantined:
            return False
        acquired = False
        try:
            acquired = self._server_lock.acquire(timeout=max(0, timeout))
            if not acquired:
                return False
            if self.quarantined:
                return False
            if self._server_stopped:
                return True
            self._phase = 'stopping'
            self._server_thread = current_thread()
            self._server_pending = True
            try:
                stopped = self.server.close(timeout=max(0.0, deadline - perf_counter()))
                if type(stopped) is not bool:
                    raise RuntimeError('server close must return an exact boolean')
                if stopped and perf_counter() > deadline:
                    self._unsafe = True
                    return False
                self._server_stopped = stopped
                self._server_pending = False
                return stopped
            except BaseException as error:
                try:
                    self._fail(error, 0)
                except BaseException as secondary:
                    self._recover(error, secondary, 0)
                    raise
                raise
            finally:
                if self._server_pending:
                    self._unsafe = True
        except BaseException:
            self._unsafe = True
            raise
        finally:
            if self._server_pending:
                self._unsafe = True
            if acquired:
                self._server_lock.release()

    def stop_broker(self, *, wait=True, timeout=None):
        # Quarantine cleanup may request nonblocking signals while server
        # cleanup is still in flight. Never close a downstream owner then.
        if not wait or not self._server_stopped or self.quarantined:
            return False
        timeout = 0.0 if timeout is None else max(0.0, timeout)
        deadline = perf_counter() + timeout
        acquired = False
        try:
            acquired = self._broker_lock.acquire(timeout=timeout)
            if not acquired:
                return False
            if self.quarantined:
                return False
            if self._broker_stopped:
                return True
            if not self._bridge_stopped:
                self._bridge_thread = current_thread()
                self._bridge_pending = True
                try:
                    stopped = self.bridge.close(timeout=max(0.0, deadline - perf_counter()))
                    if type(stopped) is not bool:
                        raise RuntimeError('bridge close must return an exact boolean')
                    if stopped and perf_counter() > deadline:
                        self._unsafe = True
                        return False
                    self._bridge_stopped = stopped
                    self._bridge_pending = False
                except BaseException as error:
                    try:
                        self._fail(error, 4)
                    except BaseException as secondary:
                        self._recover(error, secondary, 4)
                        raise
                    raise
                finally:
                    if self._bridge_pending:
                        self._unsafe = True
                if not stopped or self.quarantined:
                    return False
            self._broker_thread = current_thread()
            self._broker_pending = True
            try:
                stopped = self.broker.close(timeout=max(0.0, deadline - perf_counter()))
                if type(stopped) is not bool:
                    raise RuntimeError('broker close must return an exact boolean')
                if stopped and perf_counter() > deadline:
                    self._unsafe = True
                    return False
                self._broker_stopped = stopped
                self._broker_pending = False
            except BaseException as error:
                try:
                    self._fail(error, 6)
                except BaseException as secondary:
                    self._recover(error, secondary, 6)
                    raise
                raise
            finally:
                if self._broker_pending:
                    self._unsafe = True
            if stopped:
                self._phase = 'stopped'
            return stopped and not self.quarantined
        except BaseException:
            self._unsafe = True
            raise
        finally:
            if self._bridge_pending or self._broker_pending:
                self._unsafe = True
            if acquired:
                self._broker_lock.release()


class _ServerView:
    def __init__(self, owner):
        self._owner = owner

    @property
    def address(self):
        return self._owner.server.address

    @property
    def broker_id(self):
        return self._owner.server.broker_id

    @property
    def team_tokens(self):
        return self._owner.server.team_tokens

    def start(self):
        return self._owner.start()

    def stop_accepting(self, timeout):
        return self._owner.control_server('stop_accepting', timeout)

    def drain(self, timeout):
        return self._owner.control_server('drain', timeout)

    def close(self, timeout):
        return self._owner.stop_server(timeout)


class _BrokerView:
    def __init__(self, owner):
        self._owner = owner

    @property
    def quarantined(self):
        return self._owner.quarantined

    @property
    def available(self):
        unsafe, health = self._owner._observe()
        if unsafe:
            return False
        return not self._owner._broker_failed and not health.closed and health.owner_alive

    @property
    def failed(self):
        unsafe, health = self._owner._observe()
        if unsafe:
            return True
        return self._owner._broker_failed

    def close(self, *, wait=True, timeout=None):
        return self._owner.stop_broker(wait=wait, timeout=timeout)
