"""Bounded, opt-in memory observations inside a supervised service owner.

Device free/total are point observations shared by the device. Allocator
values and peaks cover only this process's PyTorch allocator. A completed
observation is not an admission decision or a teardown attestation.
"""

from time import perf_counter_ns

from .memory_budget import CudaMemorySnapshot


class ServingMemoryObservation:
    """Observe one service lifetime; the caller must join producers first.

    The caller supplies the owner's lease/authority guard and a failure latch.
    Start runs before the service accepts clients; finish runs after its
    handlers and worker have joined. Only fixed-size scalar data is retained.
    """

    def __init__(self, *, cuda, guard, mark_unsafe, expected_total_bytes, clock=perf_counter_ns,
                 max_samples=128):
        if (not callable(guard) or not callable(mark_unsafe) or not callable(clock)
                or type(expected_total_bytes) is not int or expected_total_bytes <= 0
                or type(max_samples) is not int or not 1 <= max_samples <= 1024):
            raise ValueError('invalid serving-memory observation configuration')
        self._cuda = cuda
        self._guard = guard
        self._mark_unsafe = mark_unsafe
        self._clock = clock
        self._limit = max_samples
        self._expected_total_bytes = expected_total_bytes
        self._state = 'new'
        self._failure = None
        self._baseline = None
        self._samples = []
        self._dropped = 0
        self._last_time = None
        self._max_allocated = 0
        self._max_reserved = 0

    @property
    def active(self):
        return self._state == 'active'

    def _fail(self, error):
        try:
            self._mark_unsafe()
        finally:
            self._failure = error
            self._state = 'failed'

    def _call(self, method):
        self._guard()
        value = method(0)
        self._guard()
        return value

    def _snapshot(self):
        free, total = self._call(self._cuda.mem_get_info)
        if (type(free) is not int or type(total) is not int or total != self._expected_total_bytes
                or not 0 <= free <= total):
            raise ValueError('invalid device memory observation')
        allocated = self._call(self._cuda.memory_allocated)
        if type(allocated) is not int or not 0 <= allocated <= total:
            raise ValueError('invalid allocated memory observation')
        reserved = self._call(self._cuda.memory_reserved)
        return CudaMemorySnapshot(free, total, allocated, reserved)

    def _point(self, snapshot):
        if type(snapshot) is not CudaMemorySnapshot:
            raise TypeError('invalid serving-memory snapshot')
        if self._baseline is not None and snapshot.device_total_bytes != self._baseline['device_total_bytes']:
            raise ValueError('device total changed during observation')
        when = self._clock()
        if type(when) is not int or when < 0 or (self._last_time is not None and when < self._last_time):
            raise ValueError('invalid serving-memory timestamp')
        self._last_time = when
        self._max_allocated = max(self._max_allocated, snapshot.process_allocated_bytes)
        self._max_reserved = max(self._max_reserved, snapshot.process_reserved_bytes)
        return {'time_ns': when,
                'device_free_bytes': snapshot.device_free_bytes,
                'device_total_bytes': snapshot.device_total_bytes,
                'process_allocated_bytes': snapshot.process_allocated_bytes,
                'process_reserved_bytes': snapshot.process_reserved_bytes}

    def start(self):
        if self._state != 'new':
            raise RuntimeError('serving-memory observation is single-use')
        self._state = 'starting'
        try:
            self._call(self._cuda.synchronize)
            self._call(self._cuda.reset_peak_memory_stats)
            self._baseline = self._point(self._snapshot())
            self._state = 'active'
        except BaseException as error:
            self._fail(error)
            raise

    def record(self, snapshot):
        if self._state != 'active':
            raise RuntimeError('serving-memory observation is not active')
        try:
            point = self._point(snapshot)
            if len(self._samples) < self._limit:
                self._samples.append(point)
            else:
                self._dropped += 1
        except BaseException as error:
            self._fail(error)
            raise

    def finish(self):
        if self._state != 'active':
            raise RuntimeError('serving-memory observation is not active')
        self._state = 'finishing'
        try:
            self._call(self._cuda.synchronize)
            endpoint = self._point(self._snapshot())
            peak_allocated = self._call(self._cuda.max_memory_allocated)
            if (type(peak_allocated) is not int or peak_allocated < self._max_allocated
                    or peak_allocated > endpoint['device_total_bytes']):
                raise ValueError('invalid serving-memory allocated peak')
            peak_reserved = self._call(self._cuda.max_memory_reserved)
            if (type(peak_reserved) is not int or peak_reserved < self._max_reserved
                    or peak_allocated > peak_reserved or peak_reserved > endpoint['device_total_bytes']):
                raise ValueError('invalid serving-memory allocator peaks')
            result = {'schema': 'serial-serving-memory-v1', 'baseline': self._baseline,
                      'samples': list(self._samples), 'dropped_samples': self._dropped,
                      'endpoint': endpoint, 'process_peak_allocated_bytes': peak_allocated,
                      'process_peak_reserved_bytes': peak_reserved}
            self._state = 'complete'
            return result
        except BaseException as error:
            self._fail(error)
            raise
