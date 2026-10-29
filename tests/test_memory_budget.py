import gc
import importlib
import weakref
import threading
import types
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor


GIB = 1024 ** 3


class MemoryBudgetTests(unittest.TestCase):
    def test_active_estimates_limit_admission_until_released(self):
        try:
            memory = importlib.import_module('looking_glass.memory_budget')
        except ModuleNotFoundError as exc:
            self.fail(f'memory budget implementation is unavailable: {exc}')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        lease = budget.acquire(3 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(4 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease.release()
        budget.acquire(4 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0).release()


    def test_device_free_includes_other_applications(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(2 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)

    def test_reserve_floor_and_fraction_at_exact_boundary(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        small = memory.CudaMemorySnapshot(4 * GIB, 4 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: small, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        budget.acquire(3 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0).release()
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(3 * GIB + 1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)

        large = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: large, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        # ceil(8 GiB * .15) == 1288490189 bytes
        budget.acquire(8 * GIB - 1288490189, max_gpu_trace_bytes=0,
                       max_host_trace_bytes=0).release()
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(8 * GIB - 1288490189 + 1, max_gpu_trace_bytes=0,
                           max_host_trace_bytes=0)


    def test_invalid_estimates_and_telemetry_fail_closed(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        valid = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: valid, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        for value in (-1, True, 1.5, float('nan'), float('inf'), '1'):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                budget.acquire(value, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        for fields in ((-1, 8 * GIB, 0, 0), (9 * GIB, 8 * GIB, 0, 0),
                       (8 * GIB, 0, 0, 0), (8 * GIB, 8 * GIB, 2, 1),
                       (8 * GIB, 8 * GIB, True, 0)):
            with self.subTest(fields=fields), self.assertRaises((TypeError, ValueError)):
                memory.CudaMemorySnapshot(*fields)

    def test_reserve_can_be_raised_for_desktop_pressure(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0, minimum_reserve_bytes=2 * GIB)
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(6 * GIB + 1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)


    def test_context_manager_retains_commitment_on_failure_until_cleanup(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        lease = budget.acquire(4 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        with self.assertRaisesRegex(RuntimeError, 'worker failed'):
            with lease:
                raise RuntimeError('worker failed')
        self.assertFalse(lease.released)
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(6 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        lease.release()  # only after synchronization and cleanup are verified
        lease.release()
        budget.acquire(6 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0).release()
        with self.assertRaises(RuntimeError):
            with lease:
                pass

    def test_concurrent_admission_never_oversubscribes_commitments(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        barrier = threading.Barrier(8)

        def submit():
            try:
                lease = budget.acquire(2 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
            except memory.AdmissionError:
                lease = None
            barrier.wait(timeout=5)
            if lease is not None:
                lease.release()
                return True
            return False

        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(lambda _: submit(), range(8)))
        self.assertEqual(sum(accepted), 3)
        budget.acquire(6 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0).release()

    def test_trace_claims_enforce_per_lease_and_shared_caps_atomically(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=100,
                                     max_host_trace_bytes=200)
        a = budget.acquire(60, max_gpu_trace_bytes=60, max_host_trace_bytes=150)
        b = budget.acquire(60, max_gpu_trace_bytes=60, max_host_trace_bytes=150)
        a.reserve_trace('gpu', 60)
        with self.assertRaises(memory.TraceLimitError):
            a.reserve_trace('gpu', 1)
        with self.assertRaises(memory.TraceLimitError):
            b.reserve_trace('gpu', 50)
        a.release_trace('gpu', 10)
        b.reserve_trace('gpu', 50)
        a.reserve_trace('host', 150)
        with self.assertRaises(memory.TraceLimitError):
            b.reserve_trace('host', 51)
        b.reserve_trace('host', 50)
        with self.assertRaises(ValueError):
            a.release_trace('gpu', 51)
        a.release_trace('gpu', 50)
        a.release_trace('host', 150)
        b.release_trace('gpu', 50)
        b.release_trace('host', 50)
        a.release()
        b.release()
        with self.assertRaises(RuntimeError):
            b.reserve_trace('gpu', 1)
        c = budget.acquire(100, max_gpu_trace_bytes=100, max_host_trace_bytes=200)
        c.reserve_trace('gpu', 100)
        c.reserve_trace('host', 200)
        c.release_trace('gpu', 100)
        c.release_trace('host', 200)
        c.release()

    def test_gpu_trace_cap_must_fit_job_estimate_and_global_cap(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=100,
                                     max_host_trace_bytes=200)
        with self.assertRaises(ValueError):
            budget.acquire(5, max_gpu_trace_bytes=6, max_host_trace_bytes=0)
        with self.assertRaises(ValueError):
            budget.acquire(101, max_gpu_trace_bytes=101, max_host_trace_bytes=0)
        with self.assertRaises(ValueError):
            budget.acquire(0, max_gpu_trace_bytes=0, max_host_trace_bytes=201)


    def test_snapshot_reports_separate_device_and_process_values_and_commitments(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(5 * GIB, 8 * GIB, GIB, 2 * GIB)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=100,
                                     max_host_trace_bytes=200)
        lease = budget.acquire(100, max_gpu_trace_bytes=100, max_host_trace_bytes=200)
        lease.reserve_trace('gpu', 30)
        lease.reserve_trace('host', 20)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot.telemetry, telemetry)
        self.assertEqual(snapshot.reserve_bytes, 1288490189)
        self.assertEqual(snapshot.committed_incremental_bytes, 100)
        self.assertEqual(snapshot.active_leases, 1)
        self.assertEqual(snapshot.gpu_trace_bytes, 30)
        self.assertEqual(snapshot.host_trace_bytes, 20)
        lease.release_trace('gpu', 30)
        lease.release_trace('host', 20)
        lease.release()
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_live_telemetry_failure_or_pressure_cannot_admit_or_mutate_state(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        state = [memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)]
        budget = memory.MemoryBudget(lambda: state[0], max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        first = budget.acquire(2 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        state[0] = memory.CudaMemorySnapshot(2 * GIB, 8 * GIB, 0, 0)
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        state[0] = None
        with self.assertRaises(TypeError):
            budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        state[0] = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        self.assertEqual(budget.snapshot().committed_incremental_bytes, 2 * GIB)
        budget.acquire(4 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0).release()
        first.release()

    def test_concurrent_trace_claims_cannot_exceed_shared_cap(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=100,
                                     max_host_trace_bytes=0)
        leases = [budget.acquire(20, max_gpu_trace_bytes=20, max_host_trace_bytes=0)
                  for _ in range(8)]
        barrier = threading.Barrier(8)
        def claim(lease):
            try:
                lease.reserve_trace('gpu', 20)
                accepted = True
            except memory.TraceLimitError:
                accepted = False
            barrier.wait(timeout=5)
            if accepted:
                lease.release_trace('gpu', 20)
            lease.release()
            return accepted
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(claim, leases))
        self.assertEqual(sum(accepted), 5)
        self.assertEqual(budget.snapshot().gpu_trace_bytes, 0)

    def test_cuda_telemetry_reads_device_and_process_metrics_separately(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        cuda = types.SimpleNamespace(is_available=lambda: True,
            mem_get_info=lambda device: (5000 + device, 8000),
            memory_allocated=lambda device: 100 + device,
            memory_reserved=lambda device: 400 + device)
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace(cuda=cuda)}):
            snapshot = memory.cuda_memory_snapshot(2)
        self.assertEqual(snapshot, memory.CudaMemorySnapshot(5002, 8000, 102, 402))

    def test_unavailable_cuda_telemetry_fails_closed(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        cuda = types.SimpleNamespace(is_available=lambda: False)
        with patch.dict('sys.modules', {'torch': types.SimpleNamespace(cuda=cuda)}):
            with self.assertRaises(memory.TelemetryError):
                memory.cuda_memory_snapshot(0)


    def test_outstanding_trace_claims_prevent_lease_release(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=100)
        a = budget.acquire(0, max_gpu_trace_bytes=0, max_host_trace_bytes=100)
        b = budget.acquire(0, max_gpu_trace_bytes=0, max_host_trace_bytes=100)
        a.reserve_trace('host', 100)
        with self.assertRaises(memory.TraceClaimsOutstanding):
            a.release()
        with self.assertRaises(memory.TraceLimitError):
            b.reserve_trace('host', 1)
        a.release_trace('host', 100)  # after the storage is disposed
        a.release()
        b.reserve_trace('host', 100)
        b.release_trace('host', 100)
        b.release()


    def test_cleanup_failure_preserves_worker_oom_and_retains_commitment(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=4)
        lease = budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=4)
        class FakeOOM(RuntimeError):
            pass
        retained = None
        with self.assertRaises(FakeOOM) as result:
            with lease:
                lease.reserve_trace('host', 4)
                retained = bytearray(4)
                raise FakeOOM('worker failed')
        self.assertEqual(str(result.exception), 'worker failed')
        self.assertFalse(lease.released)
        self.assertEqual(lease.trace_bytes('host'), 4)
        self.assertEqual(budget.snapshot().active_leases, 1)
        self.assertEqual(budget.snapshot().host_trace_bytes, 4)
        retained = None
        lease.release_trace('host', 4)
        lease.release()
        self.assertEqual(budget.snapshot().active_leases, 0)

    def test_lease_estimate_cannot_be_reassigned_via_public_property(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        lease = budget.acquire(100, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        with self.assertRaises(AttributeError):
            lease.estimated_incremental_bytes = 1
        lease.release()
        self.assertEqual(budget.snapshot().committed_incremental_bytes, 0)

    def test_process_reserved_limit_is_checked_even_for_inconsistent_sample(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        # Samples are separate API calls, so changing external pressure can make
        # the separately observed free and reserved values temporarily inconsistent.
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, GIB, 6 * GIB)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)


    def test_budget_retains_unreleased_lease_for_recovery(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=4)
        lease = budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=4)
        lease.reserve_trace('host', 4)
        reference = weakref.ref(lease)
        del lease
        gc.collect()
        recovered, = budget.active_leases()
        self.assertIs(recovered, reference())
        self.assertEqual(recovered.trace_bytes('host'), 4)
        recovered.release_trace('host', 4)  # after storage disposal is verified
        recovered.release()
        self.assertEqual(budget.active_leases(), ())


    def test_normal_context_exit_releases_after_caller_confirms_completion(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=0,
                                     max_host_trace_bytes=0)
        lease = budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        with lease:
            pass  # Worker has already confirmed work is complete.
        self.assertTrue(lease.released)
        self.assertEqual(budget.snapshot().committed_incremental_bytes, 0)


    def test_unregistered_lease_cannot_corrupt_budget_or_claim_trace(self):
        memory = importlib.import_module('looking_glass.memory_budget')
        telemetry = memory.CudaMemorySnapshot(8 * GIB, 8 * GIB, 0, 0)
        budget = memory.MemoryBudget(lambda: telemetry, max_gpu_trace_bytes=100,
                                     max_host_trace_bytes=100)
        unregistered = memory.MemoryLease(budget, 100 * GIB, 100, 100)
        with self.assertRaises(RuntimeError):
            unregistered.release()
        with self.assertRaises(RuntimeError):
            unregistered.reserve_trace('host', 10)
        self.assertEqual(budget.snapshot().committed_incremental_bytes, 0)
        self.assertEqual(budget.snapshot().host_trace_bytes, 0)
        with self.assertRaises(memory.AdmissionError):
            budget.acquire(50 * GIB, max_gpu_trace_bytes=0, max_host_trace_bytes=0)


if __name__ == '__main__':
    unittest.main()
