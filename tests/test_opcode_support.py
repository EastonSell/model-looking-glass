import sys
import unittest

from opcode_support import run_opcode_trace


class OpcodeTraceSupportTests(unittest.TestCase):
    def test_prior_tracer_is_restored_when_traced_call_raises(self):
        previous = sys.gettrace()
        calls = []

        def original(frame, event, arg):
            return original

        def injected(frame, event, arg):
            if event == "call":
                frame.f_trace_opcodes = True
            return injected

        def fail():
            calls.append("called")
            raise RuntimeError("synthetic failure")

        try:
            sys.settrace(original)
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                run_opcode_trace(injected, fail)
            self.assertIs(sys.gettrace(), original)
            self.assertEqual(calls, ["called"])
        finally:
            sys.settrace(previous)
