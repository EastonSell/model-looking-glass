"""Conservative, process-local admission for shared GPU work.

Estimates are commitments, not CUDA allocations or guarantees of success.
"""
from dataclasses import dataclass
from threading import RLock

GIB = 1024 ** 3


def _nonnegative_bytes(value, name):
    if type(value) is not int:
        raise TypeError(f'{name} must be an integer number of bytes')
    if value < 0:
        raise ValueError(f'{name} must be nonnegative')
    return value


class TelemetryError(RuntimeError):
    """A CUDA telemetry read failed or CUDA is unavailable."""


class AdmissionError(RuntimeError):
    """A request cannot be admitted within the current memory budget."""


class TraceLimitError(RuntimeError):
    """A trace claim exceeds a per-lease or aggregate trace cap."""


class TraceClaimsOutstanding(RuntimeError):
    """A lease cannot be released while it still owns trace claims."""


@dataclass(frozen=True)
class CudaMemorySnapshot:
    device_free_bytes: int
    device_total_bytes: int
    process_allocated_bytes: int
    process_reserved_bytes: int

    def __post_init__(self):
        for name in ('device_free_bytes', 'device_total_bytes',
                     'process_allocated_bytes', 'process_reserved_bytes'):
            _nonnegative_bytes(getattr(self, name), name)
        if not self.device_total_bytes:
            raise ValueError('device_total_bytes must be positive')
        if self.device_free_bytes > self.device_total_bytes:
            raise ValueError('device free exceeds total')
        if self.process_allocated_bytes > self.process_reserved_bytes:
            raise ValueError('process allocated exceeds reserved')
        if self.process_reserved_bytes > self.device_total_bytes:
            raise ValueError('process reserved exceeds device total')


@dataclass(frozen=True)
class BudgetSnapshot:
    telemetry: CudaMemorySnapshot
    reserve_bytes: int
    committed_incremental_bytes: int
    active_leases: int
    gpu_trace_bytes: int
    host_trace_bytes: int
    max_gpu_trace_bytes: int
    max_host_trace_bytes: int
    admissible_estimate_bytes: int


def cuda_memory_snapshot(device=0):
    """Sample device-wide free/total and this process's PyTorch allocator."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        return CudaMemorySnapshot(free, total, allocated, reserved)
    except Exception as exc:
        raise TelemetryError(f'cannot read CUDA memory telemetry: {exc}') from exc


class MemoryLease:
    """A commitment obtained from ``MemoryBudget.acquire`` and explicitly released."""

    def __init__(self, budget, estimated_incremental_bytes, max_gpu_trace_bytes,
                 max_host_trace_bytes):
        self._budget = budget
        self._estimated_incremental_bytes = estimated_incremental_bytes
        self._trace_limits = {'gpu': max_gpu_trace_bytes, 'host': max_host_trace_bytes}
        self._trace_used = {'gpu': 0, 'host': 0}
        self._issued = False
        self._entered = False

    @property
    def estimated_incremental_bytes(self):
        return self._estimated_incremental_bytes

    @property
    def released(self):
        with self._budget._lock:
            released = self._issued and self not in self._budget._leases
        return released

    def trace_bytes(self, location):
        if location not in ('gpu', 'host'):
            raise ValueError('location must be gpu or host')
        with self._budget._lock:
            used = self._trace_used[location]
        return used

    def _check_active(self):
        if not self._issued or self not in self._budget._leases:
            raise RuntimeError('lease is not active in its issuing budget')

    def reserve_trace(self, location, byte_count):
        """Claim bytes before allocating a retained trace; allocation can still fail."""
        _nonnegative_bytes(byte_count, 'byte_count')
        if location not in ('gpu', 'host'):
            raise ValueError('location must be gpu or host')
        with self._budget._lock:
            self._check_active()
            new_used = self._trace_used[location] + byte_count
            new_total = self._budget._trace_total(location) + byte_count
            if (new_used > self._trace_limits[location]
                    or new_total > self._budget._trace_limits[location]):
                raise TraceLimitError(f'{location} trace limit exceeded')
            self._trace_used[location] = new_used
        return None

    def release_trace(self, location, byte_count):
        """Release bytes after storage disposal; reconcile an interrupted call before retry.

        This count-based API cannot identify a retried release. If the result is
        uncertain, inspect the lease's outstanding bytes and reconcile actual
        storage ownership, or retain the lease. Do not blindly repeat the call.
        """
        _nonnegative_bytes(byte_count, 'byte_count')
        if location not in ('gpu', 'host'):
            raise ValueError('location must be gpu or host')
        with self._budget._lock:
            self._check_active()
            if byte_count > self._trace_used[location]:
                raise ValueError('cannot release more trace bytes than claimed')
            self._trace_used[location] -= byte_count
        return None

    def release(self):
        """Release after all device work, offload and retained storage are finished."""
        with self._budget._lock:
            if not (self._issued and self not in self._budget._leases):
                self._check_active()
                if any(self._trace_used.values()):
                    raise TraceClaimsOutstanding('release trace claims after disposing retained storage')
                self._budget._leases = self._budget._leases - {self}
        return None

    def __enter__(self):
        with self._budget._lock:
            self._check_active()
            if self._entered:
                raise RuntimeError('lease context is single-use')
            self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            if hasattr(exc, 'add_note'):
                exc.add_note('Memory lease remains active until device completion and '
                             'storage disposal are verified, then claims and lease are released.')
            return False
        self.release()
        return False


class MemoryBudget:
    """Thread-safe commitments for one broker-owned device and process."""

    def __init__(self, telemetry, *, max_gpu_trace_bytes, max_host_trace_bytes,
                 minimum_reserve_bytes=0):
        if not callable(telemetry):
            raise TypeError('telemetry must be callable')
        _nonnegative_bytes(max_gpu_trace_bytes, 'max_gpu_trace_bytes')
        _nonnegative_bytes(max_host_trace_bytes, 'max_host_trace_bytes')
        self._minimum_reserve_bytes = _nonnegative_bytes(minimum_reserve_bytes, 'minimum_reserve_bytes')
        self._telemetry = telemetry
        self._lock = RLock()
        # This immutable set is the sole authority for committed leases. Each
        # update replaces it in one assignment, so interruption cannot split a
        # membership change from a separate cached total.
        self._leases = frozenset()
        self._trace_limits = {'gpu': max_gpu_trace_bytes, 'host': max_host_trace_bytes}

    def _committed_bytes(self):
        return sum(lease._estimated_incremental_bytes for lease in self._leases)

    def _trace_total(self, location):
        return sum(lease._trace_used[location] for lease in self._leases)

    def _read_snapshot(self):
        snapshot = self._telemetry()
        if not isinstance(snapshot, CudaMemorySnapshot):
            raise TypeError('telemetry must return a CudaMemorySnapshot')
        return snapshot

    def _reserve(self, telemetry):
        return max(GIB, (telemetry.device_total_bytes * 15 + 99) // 100,
                   self._minimum_reserve_bytes)

    def active_leases(self):
        """Return retained lease objects; reclaim only after verified cleanup."""
        with self._lock:
            leases = tuple(self._leases)
        return leases

    def snapshot(self):
        """Return policy state and current telemetry, sampled under the policy lock."""
        with self._lock:
            telemetry = self._read_snapshot()
            reserve = self._reserve(telemetry)
            committed = self._committed_bytes()
            available = max(0, min(
                telemetry.device_total_bytes - telemetry.process_reserved_bytes
                - reserve - committed,
                telemetry.device_free_bytes - reserve - committed))
            result = BudgetSnapshot(telemetry, reserve, committed,
                                    len(self._leases), self._trace_total('gpu'),
                                    self._trace_total('host'), self._trace_limits['gpu'],
                                    self._trace_limits['host'], available)
        return result

    def acquire(self, estimated_incremental_bytes, *, max_gpu_trace_bytes,
                max_host_trace_bytes):
        """Admit an estimate, or raise without changing the current commitments."""
        _nonnegative_bytes(estimated_incremental_bytes, 'estimated_incremental_bytes')
        _nonnegative_bytes(max_gpu_trace_bytes, 'max_gpu_trace_bytes')
        _nonnegative_bytes(max_host_trace_bytes, 'max_host_trace_bytes')
        if max_gpu_trace_bytes > estimated_incremental_bytes:
            raise ValueError('GPU trace cap must fit the incremental estimate')
        if (max_gpu_trace_bytes > self._trace_limits['gpu']
                or max_host_trace_bytes > self._trace_limits['host']):
            raise ValueError('lease trace cap exceeds the aggregate cap')
        with self._lock:
            snapshot = self._read_snapshot()
            reserve = self._reserve(snapshot)
            committed = self._committed_bytes() + estimated_incremental_bytes
            if (snapshot.process_reserved_bytes + committed + reserve > snapshot.device_total_bytes
                    or committed + reserve > snapshot.device_free_bytes):
                raise AdmissionError('insufficient memory budget')
            lease = MemoryLease(self, estimated_incremental_bytes, max_gpu_trace_bytes,
                                max_host_trace_bytes)
            lease._issued = True
            self._leases = self._leases | {lease}
        return lease
