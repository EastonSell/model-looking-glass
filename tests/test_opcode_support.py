import subprocess
import sys
import unittest
from pathlib import Path

if __package__:
    from .opcode_support import run_opcode_trace
else:
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

    def test_first_use_primes_a_target_selective_tracer(self):
        # A subprocess makes this the interpreter's first tracing session even
        # when this test runs after other opcode tests in the parent process.
        script = """
import sys
sys.path.insert(0, sys.argv[1])
from opcode_support import run_opcode_trace
observed = []
def target():
    return 7
def trace(frame, event, arg):
    if frame.f_code is target.__code__:
        if event == 'call':
            frame.f_trace_opcodes = True
        elif event == 'opcode':
            observed.append(frame.f_lasti)
    return trace
run_opcode_trace(trace, target)
if not observed:
    raise AssertionError('no opcode events were delivered to the target')
"""
        process = subprocess.run(
            [sys.executable, "-I", "-c", script, str(Path(__file__).resolve().parent)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
