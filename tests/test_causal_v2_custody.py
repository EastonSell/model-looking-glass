import hashlib
import json
import math
import subprocess
import sys
import unittest

from looking_glass.causal_v2_custody import (
    build_exclusion_union,
    canonical_json,
    commitment,
    validate_nonce_set,
    verify_commitment,
    verify_split_disjointness,
)


def case(case_id, first=1.0, other=0.0):
    return {"case_id": case_id, "values": [first, other, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


class CustodyEvidenceTests(unittest.TestCase):
    def test_union_is_canonical_sorted_and_preserves_float32_bits(self):
        records = [{"inputs": [case("b", -1.0)]}, {"inputs": [case("a", 0.5, -0.0)]}]
        raw = build_exclusion_union(records)
        self.assertEqual(raw, (b'{"inputs":[{"case_id":"a","values":[0.5,-0.0,0.0,0.0,0.0,0.0,0.0,0.0]},'
                               b'{"case_id":"b","values":[-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}]}'))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), hashlib.sha256(canonical_json(json.loads(raw))).hexdigest())

    def test_union_rejects_duplicate_identifiers_and_signed_zero_equivalent_tuples(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_exclusion_union([{"inputs": [case("x"), case("x", -1.0)]}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_exclusion_union([{"inputs": [case("x", 1.0, 0.0), case("y", 1.0, -0.0)]}])
        with self.assertRaisesRegex(ValueError, "empty"):
            build_exclusion_union([{"inputs": []}])

    def test_union_rejects_nonfinite_or_not_exact_float32_values(self):
        for bad in (True, math.nan, math.inf, 0.1, 1e40, 2**53 + 1, "0.5"):
            with self.subTest(value=bad), self.assertRaises((TypeError, ValueError)):
                build_exclusion_union([{"inputs": [case("x", bad)]}])

    def test_split_disjointness_includes_ids_and_all_exclusion_records(self):
        excluded = build_exclusion_union([{"inputs": [case("dev", 0.5)]}])
        historical = [case("old", 0.25)]
        verify_split_disjointness([case("discovery", 1.0)], [case("heldout", -1.0)], excluded,
                                  historical_cases=historical)
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_split_disjointness([case("discovery", 0.5)], [case("heldout", -1.0)], excluded,
                                      historical_cases=historical)
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_split_disjointness([case("discovery", 1.0)], [case("heldout", 1.0, -0.0)], excluded,
                                      historical_cases=historical)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify_split_disjointness([case("same", 1.0)], [case("same", -1.0)], excluded,
                                      historical_cases=historical)
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_split_disjointness([case("discovery", 1.0)], [case("heldout", 0.25)], excluded,
                                      historical_cases=historical)
        with self.assertRaisesRegex(ValueError, "historical"):
            verify_split_disjointness([case("discovery", 1.0)], [case("heldout", -1.0)], excluded,
                                      historical_cases=[])
        with self.assertRaisesRegex(ValueError, "invalid exclusion"):
            verify_split_disjointness([], [], b"[" * 2000 + b"]" * 2000, historical_cases=historical)

    def test_commitment_matches_protocol_domain_and_canonical_bytes(self):
        nonce = bytes(range(32))
        record = {"context_sha256": "a" * 64, "cases": [{"case_id": "fake", "values": [0.5]}]}
        expected = hashlib.sha256(b"MLG-v2-holdout\0" + nonce + canonical_json(record)).hexdigest()
        self.assertEqual(commitment("heldout", nonce, record), expected)
        self.assertTrue(verify_commitment("heldout", nonce, record, expected))
        self.assertFalse(verify_commitment("heldout", nonce, record, "0" * 64))
        self.assertFalse(verify_commitment("heldout", nonce, {**record, "cases": []}, expected))

    def test_commitments_require_exact_record_shape_nonce_and_digest(self):
        nonce = bytes(32)
        with self.assertRaises(ValueError):
            commitment("heldout", nonce, {"context_sha256": "a" * 64, "cases": [], "extra": 0})
        with self.assertRaises(ValueError):
            commitment("predictions", nonce, {"pre_manifest_sha256": "a" * 64,
                                               "freeze_sha256": "b" * 64, "predictions": [math.nan]})
        with self.assertRaises(ValueError):
            commitment("heldout", b"short", {"context_sha256": "a" * 64, "cases": []})
        with self.assertRaises(ValueError):
            verify_commitment("schedule", nonce, {"context_sha256": "a" * 64, "assignments": []}, "A" * 64)

    def test_nonce_set_rejects_reuse_invalid_types_and_public_root(self):
        validate_nonce_set(bytes([1]) * 32, bytes([2]) * 32, bytes([3]) * 32)
        with self.assertRaises(ValueError):
            validate_nonce_set(bytes([1]) * 32, bytes([1]) * 32, bytes([3]) * 32)
        with self.assertRaises(ValueError):
            validate_nonce_set(bytearray(32), bytes([2]) * 32, bytes([3]) * 32)
        with self.assertRaises(ValueError):
            validate_nonce_set((2026102803).to_bytes(32, "big"), bytes([2]) * 32, bytes([3]) * 32)

    def test_canonical_json_rejects_coerced_keys_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            canonical_json({1: "not a JSON string key"})
        with self.assertRaises(ValueError):
            canonical_json({"bad": float("nan")})
        self.assertEqual(canonical_json({"unicode": "\u00e9"}), b'{"unicode":"\\u00e9"}')

    def test_exclusions_are_independent_of_flush_mode(self):
        for flush in (True, False):
            with self.subTest(flush=flush):
                command = f'''
import torch
from looking_glass.causal_v2_custody import build_exclusion_union
assert torch.set_flush_denormal({flush!r})
small = float.fromhex("0x1p-149")
def case(name, value, zero):
    return {{"case_id": name, "values": [value, zero, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}
raw = build_exclusion_union([{{"inputs": [case("tiny", small, 0.0), case("zero", 0.0, 0.0)]}}])
assert b"1.401298464324817e-45" in raw
try:
    build_exclusion_union([{{"inputs": [case("one", small, 0.0), case("two", small, -0.0)]}}])
except ValueError:
    pass
else:
    raise AssertionError("signed-zero duplicate accepted")
'''
                result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
