import dis
import gc
import inspect
import sys
import unittest
import weakref

import torch

from looking_glass.trace_store import HostTraceStore


def start_opcode_trace(tracer):
    # CPython 3.12 may only emit opcode events after one trace session has
    # requested them. Preserve any pre-existing tracer across the warm-up.
    old_trace = sys.gettrace()

    def warmup():
        return None

    try:
        sys.settrace(tracer)
        warmup()
    finally:
        sys.settrace(old_trace)
    sys.settrace(tracer)
    return old_trace


class TakeInterruptionTests(unittest.TestCase):
    def test_interrupted_internal_return_does_not_retain_uncharged_tensor(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("request")
        handle = request.put(0, torch.ones(1))
        snapshot = weakref.ref(handle._value)
        method = HostTraceStore._remove_locked
        return_offsets = {i.offset for i in dis.get_instructions(method)
                          if i.opname == "RETURN_VALUE"}
        observed = []

        def observe_return(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is method.__code__ and event == "opcode"
                    and frame.f_lasti in return_offsets):
                observed.append(frame.f_lasti)
            return observe_return

        # Conditional-return bytecode differs across CPython versions. Find the
        # transfer branch that actually executes, then inject into a fresh take.
        calibration = HostTraceStore(max_tensor_bytes=4, max_items=1)
        calibration_handle = calibration.request("calibration").put(0, torch.ones(1))
        old_trace = start_opcode_trace(observe_return)
        try:
            calibration_handle.take()
        finally:
            sys.settrace(old_trace)
        self.assertEqual(len(observed), 1)
        offset = observed[0]

        def tracer(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is method.__code__ and event == "opcode"
                    and frame.f_lasti == offset):
                sys.settrace(None)
                raise KeyboardInterrupt("injected interruption")
            return tracer

        error = None
        old_trace = start_opcode_trace(tracer)
        try:
            handle.take()
        except KeyboardInterrupt as caught:
            error = caught
        finally:
            sys.settrace(old_trace)
        self.assertIsNotNone(error)
        self.assertEqual(store.usage().claimed_bytes, 0)
        gc.collect()
        self.assertIsNone(snapshot(), "internal traceback retained the removed tensor")

    def test_interrupted_setup_does_not_keep_box_after_entry_drop(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("request")
        handle = request.put(0, torch.ones(1))
        snapshot = weakref.ref(handle._value)
        method = HostTraceStore._remove_locked
        source, start = inspect.getsourcelines(method)
        lines = [start + i for i, line in enumerate(source)
                 if 'entry._state = "releasing"' in line]
        self.assertGreaterEqual(len(lines), 1)
        offset = next(i.offset for i in dis.get_instructions(method)
                      if i.starts_line == lines[0])

        def tracer(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is method.__code__ and event == "opcode"
                    and frame.f_lasti == offset):
                sys.settrace(None)
                raise KeyboardInterrupt("injected interruption")
            return tracer

        error = None
        old_trace = start_opcode_trace(tracer)
        try:
            handle.take()
        except KeyboardInterrupt as caught:
            error = caught
        finally:
            sys.settrace(old_trace)
        self.assertIsNotNone(error)
        handle.drop()
        self.assertEqual(store.usage().claimed_bytes, 0)
        gc.collect()
        self.assertIsNone(snapshot(), "interrupted setup retained the transfer box")

    def test_successful_take_preserves_owned_value_and_store_accounting(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        handle = store.request("request").put(0, torch.ones(1))
        taken = handle.take()
        self.assertEqual(taken.tolist(), [1.0])
        self.assertEqual(store.usage().claimed_bytes, 0)


if __name__ == "__main__":
    unittest.main()
