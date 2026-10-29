"""Fabricated discovery observations only; no v2 inputs or sites are drawn."""
import os
import subprocess
import sys
import tempfile
import unittest


class V2DiscoverySelectionTests(unittest.TestCase):
    def test_fixed_selection_controls_and_prediction_means(self):
        code = r'''
from dataclasses import replace
import json, torch
from looking_glass.probes import TraceEvent
from looking_glass.causal_v2_runtime import establish_numeric_policy
from looking_glass.causal_v2_selection import (
    DiscoveryCase, DiscoveryCondition, DiscoveryInvalid, freeze_discovery)

def tensor(value): return torch.tensor([[0., value]], dtype=torch.float32, device="cpu")
def trace(run, first=None, magnitude=0.):
    result = []
    for i in range(4):
        x = torch.zeros((1, 8), dtype=torch.float32, device="cpu")
        if first is not None and i >= first: x[0, 0] = magnitude
        result.append(TraceEvent(run, f"block.{i}.output", i, 0, (1, 8),
                                 "torch.float32", "cpu", x))
    return tuple(result)

def case(i, first=2, last_boost=1.):
    polarity = 1 if i % 2 == 0 else -1
    clean, repeat, logits = trace(f"c{i}"), trace(f"r{i}"), tensor(1.)
    conditions = []
    for k, strength in enumerate((0.25, 0.75)):
        changed = trace(f"p{i}-{k}", first=first, magnitude=strength)
        if first is not None: changed[3].value[0, 0] = strength * last_boost
        perturbed = tensor(1. if first is None else 1. + strength * polarity)
        patched = tuple(perturbed if first is not None and site < first else logits
                        for site in (1, 2, 3))
        reverse = tuple(logits if first is not None and site < first else perturbed
                        for site in (1, 2, 3))
        conditions.append(DiscoveryCondition(strength, changed, perturbed, patched,
            reverse, (perturbed,)*3, (perturbed,)*3,
            perturbed, logits))
    return DiscoveryCase(i, polarity, clean, repeat, logits, logits, tuple(conditions))

policy = establish_numeric_policy()
cases = tuple(case(i) for i in range(16))
freeze = freeze_discovery(cases, numeric_policy=policy,
                          pairing_verified=True, private_controls_verified=True)
assert freeze["first_changed"] == 2, freeze
assert freeze["magnitude"] == 2, freeze  # Blocks 2 and 3 tie.
assert freeze["first_votes"] == {"1": 0, "2": 32, "3": 0}
assert freeze["observation_misses"] == 0
assert freeze["no_injection"] == {"cases": 16, "false_detections": 0,
                                  "first_changed": None, "magnitude": None}
assert freeze["random"]["each_probability"] == "1/3"
assert len(freeze["measurements"]) == 16
means = freeze["predictions"]["2"]["0.25"]
assert means["positive"] == -0.25 and means["negative"] == 0.25
assert means["constant"] == 0.0 and means["zero"] == 0.0
json.dumps(freeze, allow_nan=False)

miss_cases = tuple(case(i, first=None) for i in range(16))
miss = freeze_discovery(miss_cases, numeric_policy=policy,
                        pairing_verified=True, private_controls_verified=True)
assert miss["first_changed"] is None and miss["magnitude"] is None
assert miss["observation_misses"] == 32

mixed_cases = tuple(case(i, first=1 if i < 8 else 2 if i < 12 else None,
                         last_boost=4.) for i in range(16))
mixed = freeze_discovery(mixed_cases, numeric_policy=policy,
                         pairing_verified=True, private_controls_verified=True)
assert mixed["first_changed"] == 1 and mixed["magnitude"] == 3
assert mixed["first_votes"] == {"1": 16, "2": 8, "3": 0}
assert mixed["observation_misses"] == 8
tie_cases = tuple(case(i, first=1 if i < 8 else 2, last_boost=4.) for i in range(16))
tie = freeze_discovery(tie_cases, numeric_policy=policy,
                       pairing_verified=True, private_controls_verified=True)
assert tie["first_votes"] == {"1": 16, "2": 16, "3": 0}
assert tie["first_changed"] == 1 and tie["magnitude"] == 3

def invalid(items, **kwargs):
    try:
        freeze_discovery(items, numeric_policy=policy, pairing_verified=True,
                         private_controls_verified=True, **kwargs)
    except DiscoveryInvalid: return
    raise AssertionError("invalid discovery observations accepted")

invalid(cases[:-1])
invalid((replace(cases[0], case_index=1),) + cases[1:])
bad = replace(cases[0].conditions[0], upstream_logits=tensor(9.))
invalid((replace(cases[0], conditions=(bad, cases[0].conditions[1])),) + cases[1:])
for field in ("null_logits", "identity_logits", "clean_to_perturbed_logits"):
    values = list(getattr(cases[0].conditions[0], field))
    values[-1] = tensor(9.)
    bad = replace(cases[0].conditions[0], **{field: tuple(values)})
    invalid((replace(cases[0], conditions=(bad, cases[0].conditions[1])),) + cases[1:])
bad = replace(cases[0].conditions[0], clean_verify_logits=tensor(9.))
invalid((replace(cases[0], conditions=(bad, cases[0].conditions[1])),) + cases[1:])
bad = replace(cases[0].conditions[0], perturbed_trace=trace("upstream", first=0, magnitude=1.))
invalid((replace(cases[0], conditions=(bad, cases[0].conditions[1])),) + cases[1:])
bad = replace(cases[0], repeated_clean_trace=trace("repeat", first=1, magnitude=1.))
try:
    freeze_discovery((bad,) + cases[1:], numeric_policy=policy,
                     pairing_verified=True, private_controls_verified=True)
except DiscoveryInvalid as error:
    assert error.evidence["no_injection_failure"]["false_detection"] is True
    assert error.evidence["no_injection_failure"]["comparison"]["first_unequal"] == "block.1.output"
    json.dumps(error.evidence, allow_nan=False)
else:
    raise AssertionError("no-injection false detection accepted")
try:
    freeze_discovery(cases, numeric_policy=policy, pairing_verified=False,
                     private_controls_verified=True)
except DiscoveryInvalid: pass
else: raise AssertionError("unattested pairing accepted")
'''
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        with tempfile.TemporaryDirectory() as prefix:
            process = subprocess.run(
                [sys.executable, "-B", "-X", f"pycache_prefix={prefix}", "-c", code],
                capture_output=True, text=True, env=env,
            )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
