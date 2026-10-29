"""Deterministic public request plans use no model or private site."""
import hashlib
import json
import unittest

from looking_glass.pretrained_number_requests import create_request_plan, load_request_plan, verify_request
from looking_glass.pretrained_exposures_v2 import PUBLIC_DISCOVERY


class NumberInjectionRequestPlanTests(unittest.TestCase):
    def test_fixed_role_order_and_independent_request_derivation(self):
        seed = bytes(range(32))
        commit = "a" * 40
        raw = create_request_plan(seed, commit)
        plan = load_request_plan(raw, expected_source_commit=commit, expected_sha256=hashlib.sha256(raw).hexdigest())
        record = json.loads(raw)
        self.assertEqual(record["public_entropy"], seed.hex())
        self.assertEqual(record["source_commit"], commit)
        self.assertEqual(len(record["requests"]), 128)
        self.assertEqual(len({row["request_id"] for row in record["requests"]}), 128)
        first = record["requests"][0]
        expected = hashlib.sha256(b"MLG-number-injection-request-v1\0" + seed + (0).to_bytes(4, "big")).digest()[:16].hex()
        self.assertEqual(first["request_id"], expected)
        self.assertEqual((first["case_index"], first["role"]), (0, "clean"))
        self.assertEqual(record["requests"][15]["role"], "verify_clean")
        self.assertEqual((record["requests"][16]["case_index"], record["requests"][16]["role"]), (1, "clean"))
        self.assertTrue(verify_request(plan, first["case_index"], first["role"], first["request_id"],
                                       first["input_sha256"], expected_source_commit=commit,
                                       expected_sha256=hashlib.sha256(raw).hexdigest()))
        self.assertEqual(raw, create_request_plan(seed, commit))
        self.assertNotIn(b"site_nonce", raw)
        self.assertNotIn(b"hidden_site", raw)

    def test_all_ordinals_roles_and_input_digests_match_independent_oracle(self):
        seed, commit = b"\x05" * 32, "c" * 40
        raw = create_request_plan(seed, commit)
        plan = load_request_plan(raw, expected_source_commit=commit, expected_sha256=hashlib.sha256(raw).hexdigest())
        expected_roles = ("clean", "repeat", "injected", "forward_1", "forward_2", "forward_3", "forward_4",
                          "reverse_1", "reverse_2", "reverse_3", "reverse_4", "null", "self", "upstream",
                          "downstream", "verify_clean")
        for ordinal, row in enumerate(json.loads(raw)["requests"]):
            index, within_case = divmod(ordinal, 16)
            ids = PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]
            expected_input = hashlib.sha256(b"MLG-number-input-v1\0" + bytes([len(ids)]) +
                                           b"".join(x.to_bytes(4, "big") for x in ids)).hexdigest()
            expected_id = hashlib.sha256(b"MLG-number-injection-request-v1\0" + seed +
                                         ordinal.to_bytes(4, "big")).digest()[:16].hex()
            self.assertEqual(row, {"case_index": index, "role": expected_roles[within_case],
                                   "input_sha256": expected_input, "request_id": expected_id})
            self.assertTrue(verify_request(plan, index, row["role"], expected_id, expected_input,
                                           expected_source_commit=commit, expected_sha256=hashlib.sha256(raw).hexdigest()))

    def test_wrong_request_input_role_or_tampered_plan_is_rejected(self):
        raw = create_request_plan(bytes(range(32)), "b" * 40)
        digest = hashlib.sha256(raw).hexdigest()
        plan = load_request_plan(raw, expected_source_commit="b" * 40, expected_sha256=digest)
        row = json.loads(plan.record_bytes)["requests"][0]
        for index, role, request, digest in ((0, "clean", "0" * 32, row["input_sha256"]),
                                             (0, "repeat", row["request_id"], row["input_sha256"]),
                                             (0, "clean", row["request_id"], "0" * 64),
                                             (True, "clean", row["request_id"], row["input_sha256"])):
            with self.subTest(index=index, role=role), self.assertRaises(ValueError):
                verify_request(plan, index, role, request, digest, expected_source_commit="b" * 40,
                               expected_sha256=plan.sha256)
        tampered = json.loads(plan.record_bytes)
        tampered["requests"][0]["request_id"] = "f" * 32
        raw = json.dumps(tampered, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        with self.assertRaises(ValueError):
            load_request_plan(raw, expected_source_commit="b" * 40, expected_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            load_request_plan(plan.record_bytes + b"\n", expected_source_commit="b" * 40,
                              expected_sha256=hashlib.sha256(plan.record_bytes + b"\n").hexdigest())
        with self.assertRaises(ValueError):
            load_request_plan(plan.record_bytes, expected_source_commit="c" * 40, expected_sha256=plan.sha256)
        with self.assertRaises(ValueError):
            load_request_plan(plan.record_bytes, expected_source_commit="b" * 40, expected_sha256="0" * 64)
        other_raw = create_request_plan(bytes(reversed(range(32))), "d" * 40)
        other = load_request_plan(other_raw, expected_source_commit="d" * 40,
                                  expected_sha256=hashlib.sha256(other_raw).hexdigest())
        other_row = json.loads(other_raw)["requests"][0]
        with self.assertRaises(ValueError):
            verify_request(other, 0, "clean", other_row["request_id"], other_row["input_sha256"],
                           expected_source_commit="b" * 40, expected_sha256=plan.sha256)

    def test_requires_exact_nonce_and_commit_types(self):
        for nonce, commit in ((b"x" * 31, "a" * 40), (bytearray(32), "a" * 40),
                              (bytes(32), "A" * 40), (bytes(32), "a" * 39)):
            with self.subTest(nonce=nonce, commit=commit), self.assertRaises(ValueError):
                create_request_plan(nonce, commit)


if __name__ == "__main__":
    unittest.main()
