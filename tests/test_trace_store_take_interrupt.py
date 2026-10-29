import dis
import gc
import inspect
import sys
import unittest
import weakref

import torch

from looking_glass.trace_store import HostTraceStore


class TakeInterruptionTests(unittest.TestCase):
    def test_interrupted_internal_return_does_not_retain_uncharged_tensor(self):
        store = HostTraceStore(max_tensor_bytes=4, max_items=1)
        request = store.request("request")
        handle = request.put(0, torch.ones(1))
        snapshot = weakref.ref(handle._value)
        method = HostTraceStore._remove_locked
        offset = next(i.offset for i in dis.get_instructions(method)
                      if i.opname == "RETURN_VALUE")

        def tracer(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            if (frame.f_code is method.__code__ and event == "opcode"
                    and frame.f_lasti == offset):
                sys.settrace(None)
                raise KeyboardInterrupt("injected interruption")
            return tracer

        error = None
        sys.settrace(tracer)
        try:
            handle.take()
        except KeyboardInterrupt as caught:
            error = caught
        finally:
            sys.settrace(None)
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
        sys.settrace(tracer)
        try:
            handle.take()
        except KeyboardInterrupt as caught:
            error = caught
        finally:
            sys.settrace(None)
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
