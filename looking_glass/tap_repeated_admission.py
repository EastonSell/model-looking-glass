"""Private, CPU-testable admission custody for a prospective tap owner.

This module does not access CUDA or launch an owner. The calling process must
already hold and check the real supervised physical lease, provide guarded live
telemetry, and stop after any escaping failure. Its commitments are logical
and cannot reserve physical GPU memory or bound Python/native intermediates.
"""

from .memory_budget import CudaMemorySnapshot, MemoryBudget
from .tap_repeated_protocol import GIB, HEADROOM, OUTSIDE_CAP, check_memory


GPU_TRACE_CLAIM = 128 * 1024


class _Admission:
    """Single-use custody; the caller must root this object before acquire.

    No cleanup or retry is attempted on an uncertain transition. The owner is
    responsible for process termination and for keeping all known resources
    rooted. This class is not a replacement for an external watchdog.
    """

    def __init__(self, *, check_owner, snapshot, total_bytes, cap_bytes):
        if (not callable(check_owner) or not callable(snapshot)
                or type(total_bytes) is not int or total_bytes <= 0
                or type(cap_bytes) is not int or cap_bytes != total_bytes // 4
                or cap_bytes <= HEADROOM):
            raise ValueError('invalid owner admission configuration')
        self.check_owner, self.snapshot = check_owner, snapshot
        self.total_bytes, self.cap_bytes = total_bytes, cap_bytes
        self.commitment_bytes = cap_bytes + HEADROOM + OUTSIDE_CAP
        self.state = 'new'
        self.budget = None
        self.lease = None
        self.failure = None

    def _snapshot(self):
        self.check_owner()
        row = self.snapshot()
        self.check_owner()
        check_memory(row, expected_total=self.total_bytes, cap_bytes=self.cap_bytes)
        return row

    def _telemetry(self):
        row = self._snapshot()
        return CudaMemorySnapshot(row['free_bytes'], row['total_bytes'],
                                  row['allocated_bytes'], row['reserved_bytes'])

    def acquire(self):
        if self.state != 'new':
            raise RuntimeError('admission is single-use')
        self.state = 'admitting'
        try:
            first = self._snapshot()
            if first['free_bytes'] < 3 * GIB:
                raise ValueError('at least 3 GiB must be free before model transfer')
            # The budget is rooted before acquire. If its return is lost after
            # publication, its active set still owns the issued memory lease.
            self.budget = MemoryBudget(self._telemetry, max_gpu_trace_bytes=GPU_TRACE_CLAIM,
                                       max_host_trace_bytes=0)
            self.lease = self.budget.acquire(self.commitment_bytes,
                                              max_gpu_trace_bytes=GPU_TRACE_CLAIM,
                                              max_host_trace_bytes=0)
            self.lease.reserve_trace('gpu', GPU_TRACE_CLAIM)
            self.check_owner()
            self.state = 'active'
            return first
        except BaseException as exc:
            self.state = 'failed'
            self.failure = exc
            raise

    def verify(self):
        if self.state != 'active':
            raise RuntimeError('admission is not active')
        self.state = 'verifying'
        try:
            self.check_owner()
            if (type(self.budget) is not MemoryBudget or self.budget.active_leases() != (self.lease,)
                    or self.lease.estimated_incremental_bytes != self.commitment_bytes
                    or self.lease.trace_bytes('gpu') != GPU_TRACE_CLAIM
                    or self.lease.trace_bytes('host') != 0):
                raise RuntimeError('whole-run or trace custody differs')
            self.check_owner()
            self.state = 'active'
        except BaseException as exc:
            self.state = 'failed'
            self.failure = exc
            raise

    def check_snapshot(self):
        """Guard and validate an actual live sample while custody is active."""
        self.verify()
        self.state = 'sampling'
        try:
            row = self._snapshot()
            self.state = 'active'
            self.verify()
            return row
        except BaseException as exc:
            self.state = 'failed'
            if self.failure is None:
                self.failure = exc
            raise

    def release_after_disposal(self):
        """Release only when the trusted caller has already synchronized disposal.

        This private method does not itself observe disposal. The future owner
        must hold every result, trace, model and input through a successful
        completion synchronization, dispose them, and synchronize again before
        invoking it. Any error or interruption makes the owner unpublishable.
        """
        self.verify()
        self.state = 'releasing'
        try:
            self.check_owner()
            self.lease.release_trace('gpu', GPU_TRACE_CLAIM)
            self.check_owner()
            if self.lease.trace_bytes('gpu') != 0 or self.lease.trace_bytes('host') != 0:
                raise RuntimeError('trace claims remain after release')
            self.lease.release()
            self.check_owner()
            if self.budget.active_leases():
                raise RuntimeError('whole-run commitment remains after release')
            self.state = 'released'
        except BaseException as exc:
            self.state = 'failed'
            self.failure = exc
            raise
