"""Development-only numeric-policy tests; no v2 cases or schedule are made."""
import os
import subprocess
import sys
import tempfile
import unittest


def _run(code):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    with tempfile.TemporaryDirectory() as prefix:
        return subprocess.run(
            [sys.executable, "-B", "-X", f"pycache_prefix={prefix}", "-c", code],
            capture_output=True, text=True, env=env,
        )


class V2NumericPolicyTests(unittest.TestCase):
    def test_setup_and_rechecking_are_bound_to_execution_thread(self):
        process = _run(r'''
import threading, torch
from looking_glass.causal_v2_runtime import establish_numeric_policy, NumericPolicy, NumericPolicyError

try: NumericPolicy({}, {})
except NumericPolicyError: pass
else: raise AssertionError("direct policy construction accepted")

policy = establish_numeric_policy()
record = policy.record()
assert record["flush_denormal_requested"] is False
assert record["preservation_request_succeeded"] is True
assert record["initial_probes"] == {"input_preserved": True, "output_preserved": True}
assert record["single_thread_probes"] == {"input_preserved": True, "output_preserved": True}
assert record["intraop_threads"] == 1
policy.verify()
errors = []
def foreign():
    try: policy.verify()
    except NumericPolicyError: return
    errors.append("foreign thread accepted")
t = threading.Thread(target=foreign); t.start(); t.join()
assert not errors, errors
assert torch.set_flush_denormal(True)
try: policy.verify()
except NumericPolicyError: pass
else: raise AssertionError("changed denormal policy accepted")
''')
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_policy_fails_closed_when_mode_request_is_unsupported(self):
        process = _run(r'''
import torch
from looking_glass.causal_v2_runtime import establish_numeric_policy, NumericPolicyError
torch.set_flush_denormal = lambda mode: False
try: establish_numeric_policy()
except NumericPolicyError: pass
else: raise AssertionError("unsupported preservation accepted")
''')
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_exact_float32_checks_are_finite_shape_strict_and_mode_independent(self):
        process = _run(r'''
import torch
from looking_glass.causal_v2_runtime import float32_exact_equal, NumericPolicyError
def raw(values): return torch.tensor(values, dtype=torch.int32, device="cpu").view(torch.float32)
for flush in (False, True):
    assert torch.set_flush_denormal(flush)
    assert float32_exact_equal(raw([0, 1]), raw([-2147483648, 1]))
    assert not float32_exact_equal(raw([0]), raw([1]))
    assert not float32_exact_equal(raw([1, 2, 3, 4])[::2], raw([1, 2]))
    for left, right in ((raw([0x7f800000]), raw([0])),
                        (raw([0x7fc00001]), raw([0])),
                        (raw([0, 0]), raw([0])),
                        (raw([0]), torch.tensor([0.], dtype=torch.float64)),
                        (raw([]), raw([]))):
        try: float32_exact_equal(left, right)
        except NumericPolicyError: pass
        else: raise AssertionError(("invalid input accepted", left, right))
''')
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
