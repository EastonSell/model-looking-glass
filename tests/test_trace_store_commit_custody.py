import concurrent.futures
import dis
import gc
import inspect
import sys
import threading
import unittest
import weakref

import torch

if __package__:
    from .opcode_support import offsets_at_line, run_opcode_trace
else:
    from opcode_support import offsets_at_line, run_opcode_trace
from looking_glass.trace_store import HostTraceStore, TraceReservation


def commit_return_offsets():
    instructions = list(dis.get_instructions(TraceReservation.commit))
    returns = {op.offset for op in instructions if op.opname == "RETURN_VALUE"}
    observed = []

    def trace(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
        if (frame.f_code is TraceReservation.commit.__code__ and event == "opcode"
                and frame.f_lasti in returns):
            observed.append(frame.f_lasti)
        return trace

    calibration = HostTraceStore(max_tensor_bytes=4, max_items=1)
    handle = calibration.request("calibration").reserve(0, 4)
    run_opcode_trace(trace, lambda: handle.commit(torch.ones(1)))
    if len(observed) != 1:
        raise AssertionError(f"expected one executed commit return, got {observed}")
    index = next(i for i, op in enumerate(instructions) if op.offset == observed[0])
    previous = instructions[index - 1]
    if previous.opname != "LOAD_FAST" or previous.argval != "self":
        raise AssertionError(f"unexpected instruction before commit return: {previous}")
    return previous.offset, observed[0]


class CommitCustodyTests(unittest.TestCase):
    def test_interruption_after_pre_copy_check_releases_claim_and_lock(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        handle = store.request("request").reserve(0, 4)
        source = torch.ones(1)
        targets = [op.offset for op in dis.get_instructions(TraceReservation.commit)
                   if op.opname in ("LOAD_FAST", "LOAD_FAST_CHECK")
                   and op.argval == "cancelled_before_copy"]
        self.assertEqual(len(targets), 1)

        def trace(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is TraceReservation.commit.__code__ and event == "opcode"
                    and frame.f_lasti == targets[0]):
                sys.settrace(None)
                raise KeyboardInterrupt("injected after pre-copy check")
            return trace

        with self.assertRaises(KeyboardInterrupt):
            run_opcode_trace(trace, lambda: handle.commit(source))
        observed = []
        thread = threading.Thread(target=lambda: observed.append(store.usage()), daemon=True)
        try:
            thread.start()
            thread.join(1)
            self.assertFalse(thread.is_alive(), "pre-copy interruption stranded store lock")
            self.assertEqual(observed[0].claimed_bytes, 0)
            self.assertEqual(observed[0].claimed_items, 0)
        finally:
            # The synthetic fault may deliberately expose a stranded RLock.
            if store._lock._is_owned():
                store._lock.release()
            thread.join(1)

    def test_cancel_waits_until_commit_has_cleared_its_snapshot_local(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("request")
        handle = request.reserve(0, 4)
        source = torch.ones(1)
        entered, release, cancel_started = threading.Event(), threading.Event(), threading.Event()
        references = []
        lines, first = inspect.getsourcelines(TraceReservation.commit)
        assignment = next(i for i, line in enumerate(lines)
                          if line.strip() == "self._value = snapshot")
        clear_line = next(first + i for i in range(assignment + 1, len(lines))
                          if lines[i].strip() == "snapshot = None")
        offsets = offsets_at_line(TraceReservation.commit, clear_line, "STORE_FAST")
        self.assertEqual(len(offsets), 1)
        target = offsets[0]

        def trace(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is TraceReservation.commit.__code__ and event == "opcode"
                    and frame.f_lasti == target):
                references.append(weakref.ref(handle._value))
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release publication")
            return trace

        def cancel():
            cancel_started.set()
            request.cancel()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            producer = pool.submit(run_opcode_trace, trace, lambda: handle.commit(source))
            try:
                self.assertTrue(entered.wait(5), "publication gate was not reached")
                self.assertEqual(handle._state, "copying")
                self.assertEqual(store._claimed_bytes, 4)
                cancellation = pool.submit(cancel)
                self.assertTrue(cancel_started.wait(5))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    cancellation.result(timeout=0.1)
            finally:
                release.set()
            producer.result(timeout=5)
            cancellation.result(timeout=5)
        self.assertEqual(store.usage().claimed_bytes, 0)
        gc.collect()
        self.assertIsNone(references[0]())

    def test_dropping_ready_entry_during_commit_does_not_retain_snapshot(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("request")
        handle = request.reserve(0, 4)
        source = torch.ones(1)
        entered = threading.Event()
        release = threading.Event()
        references = []
        target, _ = commit_return_offsets()

        def trace(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is TraceReservation.commit.__code__ and event == "opcode"
                    and frame.f_lasti == target):
                references.append(weakref.ref(handle._value))
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release commit")
            return trace

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_opcode_trace, trace, lambda: handle.commit(source))
            try:
                self.assertTrue(entered.wait(5), "commit return was not reached")
                request.cancel()
                self.assertEqual(store.usage().claimed_bytes, 0)
                gc.collect()
                self.assertIsNone(references[0](), "running commit retained a dropped snapshot")
            finally:
                release.set()
            future.result(timeout=5)

    def test_interrupted_return_does_not_retain_dropped_snapshot(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("request")
        handle = request.reserve(0, 4)
        source = torch.ones(1)
        references = []
        _, target = commit_return_offsets()

        def trace(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is TraceReservation.commit.__code__ and event == "opcode"
                    and frame.f_lasti == target):
                references.append(weakref.ref(handle._value))
                sys.settrace(None)
                raise KeyboardInterrupt("injected commit return interruption")
            return trace

        error = None
        try:
            run_opcode_trace(trace, lambda: handle.commit(source))
        except KeyboardInterrupt as caught:
            error = caught
        self.assertIsNotNone(error, "return interruption did not fire")
        handle.drop()
        self.assertEqual(store.usage().claimed_bytes, 0)
        gc.collect()
        self.assertIsNone(references[0](), "commit traceback retained a dropped snapshot")
