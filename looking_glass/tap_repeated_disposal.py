"""CPU-testable teardown for a prospective repeated-tap owner.

This private seam does not enable a GPU owner. Its callbacks and the process
are trusted. The supervised owner must terminate after any escaped failure or
ambiguous return; this class is not an asynchronous-interruption guarantee.
"""

import gc
import weakref

from . import tap_repeated_owner as owner


_PERSISTENT = ('model', 'converted', 'input_ids', 'default_stream', 'start_event', 'end_event')
_EMPTY = ('pending_return', 'failure', 'cleanup_failure')


class _Disposal:
    """Single-use, fail-stop disposal of the exact owner's named roots.

    The caller must remove its own aliases before ``finish``. The callbacks
    are supplied by the eventual fixed owner; success only checks their
    reported results and cannot prove physical device quiescence.
    """

    def __init__(self, operations, *, synchronize, cleanup, final_counters):
        if (type(operations) is not owner._OwnerOperations
                or not callable(synchronize) or type(cleanup) is not tuple
                or len(cleanup) > 3 or not all(callable(x) for x in cleanup)
                or type(final_counters) is not tuple or len(final_counters) != 2
                or not all(callable(x) for x in final_counters)):
            raise ValueError('invalid repeated-tap disposal configuration')
        self.operations = operations
        self.session = operations.session
        self.device_lease = operations.device_lease
        self.admitted = operations.admitted
        self.roots = operations.roots
        self.synchronize = synchronize
        self.cleanup = cleanup
        self.final_counters = final_counters
        self.state = 'ready'
        self.failure = None
        self.pending_return = None
        self.counters = None

    def _binding(self):
        ops = self.operations
        if gc.isenabled():
            raise RuntimeError('automatic garbage collection must remain disabled')
        if (ops.session is not self.session or ops.device_lease is not self.device_lease
                or ops.admitted is not self.admitted or ops.roots is not self.roots
                or ops.unsafe is not False):
            raise RuntimeError('owner disposal binding changed')
        owner._OwnerOperations._check(ops)

    def _empty_errors(self):
        if any(not hasattr(self.roots, name) or getattr(self.roots, name) is not None
               for name in _EMPTY):
            raise RuntimeError('owner has an unresolved operation or cleanup failure')

    def _check_roots(self, *, before):
        self._empty_errors()
        for name in owner._SLOTS:
            value = getattr(self.roots, name)
            if before and name in _PERSISTENT:
                if value is None:
                    raise RuntimeError('persistent owner root is missing')
            elif value is not None:
                raise RuntimeError('unexpected owner root remains occupied')

    def _none_call(self, callback):
        def run():
            result = None
            try:
                result = callback()
                self.pending_return = result
                if result is not None:
                    raise RuntimeError('disposal callback returned an unexpected object')
                return None
            except BaseException:
                # Preserve an assigned callback return even if the first root
                # store was interrupted. A second interruption here is still
                # an uncertain transition requiring process exit.
                if result is not None:
                    self.pending_return = result
                raise

        self._binding()
        owner._OwnerOperations.call(self.operations, run)
        self._binding()

    def _check_remaining(self, refs, cleared):
        self._empty_errors()
        for index, name in enumerate(_PERSISTENT):
            expected = None if index < cleared else refs[index]()
            if (index >= cleared and expected is None) or getattr(self.roots, name) is not expected:
                raise RuntimeError('owner resource root changed during disposal')
        if any(getattr(self.roots, name) is not None
               for name in owner._SLOTS if name not in _PERSISTENT):
            raise RuntimeError('unexpected owner root remains occupied')

    def _capture_counter(self, name, callback):
        def read():
            result = None
            try:
                result = callback()
                self.pending_return = result
                if type(result) is not int or result != 0:
                    raise RuntimeError('final allocator counters are not exactly zero')
                self.counters[name] = result
                self.pending_return = None
            except BaseException:
                if result is not None:
                    self.pending_return = result
                raise

        self._none_call(read)
        self._check_roots(before=False)

    def _post_release(self):
        ops = self.operations
        if gc.isenabled():
            raise RuntimeError('automatic garbage collection must remain disabled')
        if (ops.session is not self.session or ops.device_lease is not self.device_lease
                or ops.admitted is not self.admitted or ops.roots is not self.roots
                or ops.unsafe is not False):
            raise RuntimeError('released owner binding changed')
        owner.OwnerSession.check(self.session, self.device_lease)
        owner._OwnerOperations._check_identity(ops)
        self._check_roots(before=False)
        if (self.admitted.state != 'released' or self.admitted.budget.active_leases()
                or self.admitted.lease.trace_bytes('gpu') != 0
                or self.admitted.lease.trace_bytes('host') != 0):
            raise RuntimeError('admission claims remain after disposal')
        owner.OwnerSession.check(self.session, self.device_lease)
        owner._OwnerOperations._check_identity(ops)

    def finish(self):
        if self.state != 'ready':
            raise RuntimeError('disposal is single-use')
        self.state = 'disposing'
        try:
            self._binding()
            self._check_roots(before=True)
            try:
                refs = tuple(weakref.ref(getattr(self.roots, name)) for name in _PERSISTENT)
            except TypeError as exc:
                raise RuntimeError('owner resource cannot be weak-referenced') from exc

            # Keep every named resource rooted through the first drain.
            self._none_call(self.synchronize)
            self._check_remaining(refs, 0)
            for index, name in enumerate(_PERSISTENT):
                self._binding()
                owner._OwnerOperations.clear_after_synchronization(self.operations, name)
                self._binding()
                self._check_remaining(refs, index + 1)
            self._check_roots(before=False)
            if any(ref() is not None for ref in refs):
                raise RuntimeError('an owner resource is still reachable')
            self._none_call(self.synchronize)
            self._check_roots(before=False)

            for callback in self.cleanup:
                self._none_call(callback)
                self._check_roots(before=False)
            self._none_call(self.synchronize)
            self._check_roots(before=False)
            self.counters = {}
            self._capture_counter('allocated_bytes', self.final_counters[0])
            self._capture_counter('reserved_bytes', self.final_counters[1])
            self.admitted.release_after_disposal()
            self._post_release()
            self.state = 'released'
        except BaseException as exc:
            self.state = 'failed'
            self.failure = exc
            self.operations.unsafe = True
            raise
