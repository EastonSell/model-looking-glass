"""CPU-only tests for an unambiguous capacity-rejection result."""
import sys
import unittest

from looking_glass.memory_budget import AdmissionError, CudaMemorySnapshot, GIB, MemoryBudget


def make_budget(free=3 * GIB, telemetry=None):
    if telemetry is None:
        telemetry = lambda: CudaMemorySnapshot(free, 4 * GIB, 0, 0)
    return MemoryBudget(telemetry, max_gpu_trace_bytes=0, max_host_trace_bytes=0)


class TryAcquireTests(unittest.TestCase):
    def test_returns_none_only_for_capacity_and_legacy_api_still_raises(self):
        budget = make_budget(free=GIB)
        self.assertIsNone(budget.try_acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0))
        self.assertEqual(budget.active_leases(), ())
        with self.assertRaises(AdmissionError):
            budget.acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        self.assertEqual(budget.active_leases(), ())

    def test_successful_try_acquire_publishes_a_normal_lease(self):
        budget = make_budget()
        lease = budget.try_acquire(100, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        self.assertEqual(budget.active_leases(), (lease,))
        self.assertEqual(budget.snapshot().committed_incremental_bytes, 100)
        lease.release()
        self.assertEqual(budget.active_leases(), ())

    def test_telemetry_admission_error_is_not_converted_to_capacity_result(self):
        error = AdmissionError("telemetry failed")
        def telemetry():
            raise error
        budget = make_budget(telemetry=telemetry)
        with self.assertRaises(AdmissionError) as caught:
            budget.try_acquire(1, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        self.assertIs(caught.exception, error)
        self.assertEqual(budget.active_leases(), ())

    def test_typed_error_at_return_after_publication_is_not_swallowed(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        budget = make_budget()
        error = AdmissionError("injected after admission")
        code = MemoryBudget.try_acquire.__code__
        def interrupt(frame, event, arg):
            if frame.f_code is code and event == "return":
                raise error
            return interrupt
        try:
            sys.settrace(interrupt)
            with self.assertRaises(AdmissionError) as caught:
                budget.try_acquire(100, max_gpu_trace_bytes=0, max_host_trace_bytes=0)
        finally:
            sys.settrace(None)
        self.assertIs(caught.exception, error)
        lease, = budget.active_leases()
        self.assertEqual(budget.snapshot().committed_incremental_bytes, 100)
        lease.release()


if __name__ == "__main__":
    unittest.main()
