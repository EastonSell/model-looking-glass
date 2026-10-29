"""CPU-only interruption tests for the process-local memory ledger."""
import dis
import sys
import threading
import unittest

from looking_glass.memory_budget import (
    CudaMemorySnapshot, GIB, MemoryBudget, MemoryLease, TraceLimitError,
)


def make_budget():
    sample = CudaMemorySnapshot(3 * GIB, 4 * GIB, 0, 0)
    return MemoryBudget(lambda: sample, max_gpu_trace_bytes=0, max_host_trace_bytes=100)


def acquire(budget, estimate=100):
    return budget.acquire(estimate, max_gpu_trace_bytes=0, max_host_trace_bytes=100)


def opcode_count(code, call):
    count = 0
    old_trace = sys.gettrace()

    def tracer(frame, event, arg):
        nonlocal count
        if frame.f_code is code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode":
                count += 1
        return tracer

    try:
        sys.settrace(tracer)
        call()
    finally:
        sys.settrace(old_trace)
    return count


def interrupted(code, call, target):
    seen = 0
    fired = False
    old_trace = sys.gettrace()

    def tracer(frame, event, arg):
        nonlocal seen, fired
        if frame.f_code is code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode":
                if seen == target:
                    fired = True
                    raise KeyboardInterrupt("injected opcode interruption")
                seen += 1
        return tracer

    try:
        sys.settrace(tracer)
        try:
            call()
        except KeyboardInterrupt:
            pass
    finally:
        sys.settrace(old_trace)
    if not fired:
        raise AssertionError("target opcode was not reached")


def interrupt_return(code, call):
    fired = False
    old_trace = sys.gettrace()

    def tracer(frame, event, arg):
        nonlocal fired
        if frame.f_code is code and event == "return":
            fired = True
            raise KeyboardInterrupt("injected return-delivery interruption")
        return tracer

    try:
        sys.settrace(tracer)
        try:
            call()
        except KeyboardInterrupt:
            pass
    finally:
        sys.settrace(old_trace)
    if not fired:
        raise AssertionError("return event was not reached")


def interrupt_return_line(code, call):
    return_offset = next(item.offset for item in dis.get_instructions(code)
                         if item.opname == "RETURN_VALUE")
    return_line = max((offset, line) for offset, line in dis.findlinestarts(code)
                      if offset <= return_offset)[1]
    fired = False
    old_trace = sys.gettrace()

    def tracer(frame, event, arg):
        nonlocal fired
        if frame.f_code is code and event == "line" and frame.f_lineno == return_line:
            fired = True
            raise KeyboardInterrupt("injected source-line interruption")
        return tracer

    try:
        sys.settrace(tracer)
        try:
            call()
        except KeyboardInterrupt:
            pass
    finally:
        sys.settrace(old_trace)
    if not fired:
        raise AssertionError("return line was not reached")


def lock_responds_from_other_thread(budget):
    results = []

    def probe():
        acquired = budget._lock.acquire(timeout=0.01)
        results.append(acquired)
        if acquired:
            budget._lock.release()

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(1)
    if worker.is_alive() or len(results) != 1:
        raise AssertionError("bounded lock probe did not finish")
    return results[0]


class InterruptedLedgerTests(unittest.TestCase):
    def assert_consistent(self, budget):
        leases = budget.active_leases()
        snapshot = budget.snapshot()
        self.assertEqual(snapshot.active_leases, len(leases))
        self.assertEqual(snapshot.committed_incremental_bytes,
                         sum(lease.estimated_incremental_bytes for lease in leases))
        self.assertEqual(snapshot.host_trace_bytes,
                         sum(lease.trace_bytes("host") for lease in leases))
        self.assertEqual(snapshot.gpu_trace_bytes,
                         sum(lease.trace_bytes("gpu") for lease in leases))
        self.assertGreaterEqual(snapshot.committed_incremental_bytes, 0)
        self.assertGreaterEqual(snapshot.host_trace_bytes, 0)

    def test_acquire_interruption_at_each_opcode_never_undercounts(self):
        initial = make_budget()
        code = MemoryBudget.acquire.__code__
        count = opcode_count(code, lambda: acquire(initial))
        for target in range(count):
            with self.subTest(opcode=target):
                budget = make_budget()
                interrupted(code, lambda: acquire(budget), target)
                self.assert_consistent(budget)
                for lease in budget.active_leases():
                    lease.release()
                self.assert_consistent(budget)

    def test_release_interruption_and_retry_cannot_undercharge_other_lease(self):
        initial = make_budget()
        first = acquire(initial)
        code = MemoryLease.release.__code__
        count = opcode_count(code, first.release)
        for target in range(count):
            with self.subTest(opcode=target):
                budget = make_budget()
                first, second = acquire(budget), acquire(budget, 200)
                interrupted(code, first.release, target)
                self.assert_consistent(budget)
                first.release()
                self.assert_consistent(budget)
                self.assertEqual(budget.active_leases(), (second,))
                self.assertEqual(budget.snapshot().committed_incremental_bytes, 200)
                second.release()

    def test_reserve_interruption_cannot_hide_claim_from_global_cap(self):
        initial = make_budget()
        first = acquire(initial)
        code = MemoryLease.reserve_trace.__code__
        count = opcode_count(code, lambda: first.reserve_trace("host", 60))
        for target in range(count):
            with self.subTest(opcode=target):
                budget = make_budget()
                first, second = acquire(budget), acquire(budget)
                interrupted(code, lambda: first.reserve_trace("host", 60), target)
                self.assert_consistent(budget)
                used = first.trace_bytes("host")
                self.assertIn(used, (0, 60))
                if used:
                    with self.assertRaises(TraceLimitError):
                        second.reserve_trace("host", 41)
                else:
                    second.reserve_trace("host", 100)
                    second.release_trace("host", 100)
                if used:
                    first.release_trace("host", used)
                first.release()
                second.release()
                self.assert_consistent(budget)

    def test_trace_release_interruption_never_desynchronizes_totals(self):
        initial = make_budget()
        first = acquire(initial)
        first.reserve_trace("host", 60)
        code = MemoryLease.release_trace.__code__
        count = opcode_count(code, lambda: first.release_trace("host", 30))
        for target in range(count):
            with self.subTest(opcode=target):
                budget = make_budget()
                first, second = acquire(budget), acquire(budget)
                first.reserve_trace("host", 60)
                second.reserve_trace("host", 40)
                interrupted(code, lambda: first.release_trace("host", 30), target)
                self.assert_consistent(budget)
                self.assertIn(first.trace_bytes("host"), (30, 60))
                self.assertEqual(second.trace_bytes("host"), 40)
                # Reconcile the interrupted call before releasing remaining bytes.
                first.release_trace("host", first.trace_bytes("host"))
                second.release_trace("host", 40)
                first.release()
                second.release()
                self.assert_consistent(budget)

    def test_interrupted_return_delivery_is_reconcilable(self):
        budget = make_budget()
        interrupt_return(MemoryBudget.acquire.__code__, lambda: acquire(budget))
        first, = budget.active_leases()
        self.assert_consistent(budget)
        second = acquire(budget)
        interrupt_return(MemoryLease.reserve_trace.__code__,
                         lambda: first.reserve_trace("host", 60))
        second.reserve_trace("host", 40)
        self.assertEqual(first.trace_bytes("host"), 60)
        self.assert_consistent(budget)
        interrupt_return(MemoryLease.release_trace.__code__,
                         lambda: first.release_trace("host", 30))
        # The caller must reconcile, not repeat 30: that would release another
        # claim. The two leases remain exactly accounted.
        self.assertEqual(first.trace_bytes("host"), 30)
        self.assertEqual(second.trace_bytes("host"), 40)
        self.assert_consistent(budget)
        first.release_trace("host", 30)
        second.release_trace("host", 40)
        interrupt_return(MemoryLease.release.__code__, first.release)
        self.assertTrue(first.released)
        first.release()  # Unlike count releases, the lease release is idempotent.
        self.assert_consistent(budget)
        second.release()

    def test_cross_thread_lock_probe_observes_stalls(self):
        budget = make_budget()
        with budget._lock:
            self.assertFalse(lock_responds_from_other_thread(budget))
        self.assertTrue(lock_responds_from_other_thread(budget))

    def test_source_return_line_interruption_leaves_lock_available(self):
        budget = make_budget()
        interrupt_return_line(MemoryBudget.acquire.__code__, lambda: acquire(budget))
        self.assertTrue(lock_responds_from_other_thread(budget))
        first, = budget.active_leases()
        self.assert_consistent(budget)
        interrupt_return_line(MemoryLease.release.__code__, first.release)
        self.assertTrue(lock_responds_from_other_thread(budget))
        self.assertTrue(first.released)
        self.assert_consistent(budget)

    def test_opcode_interruptions_check_lock_liveness_from_another_thread(self):
        def case(kind):
            budget = make_budget()
            if kind == "acquire":
                return budget, MemoryBudget.acquire.__code__, lambda: acquire(budget)
            lease = acquire(budget)
            if kind == "release":
                return budget, MemoryLease.release.__code__, lease.release
            if kind == "reserve":
                return budget, MemoryLease.reserve_trace.__code__, lambda: lease.reserve_trace("host", 60)
            lease.reserve_trace("host", 60)
            return budget, MemoryLease.release_trace.__code__, lambda: lease.release_trace("host", 30)

        for kind in ("acquire", "release", "reserve", "release_trace"):
            _, code, baseline = case(kind)
            count = opcode_count(code, baseline)
            for target in range(count):
                with self.subTest(operation=kind, opcode=target):
                    budget, code, call = case(kind)
                    interrupted(code, call, target)
                    if lock_responds_from_other_thread(budget):
                        self.assert_consistent(budget)
                    # If the lock is stranded, do not attempt same-process
                    # recovery: an external supervisor must replace the worker.


if __name__ == "__main__":
    unittest.main()
