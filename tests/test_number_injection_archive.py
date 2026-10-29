"""Fabricated archive controls; no model or held-out data."""
import base64
import json
import struct
import unittest
from dataclasses import asdict, replace

from looking_glass.pretrained_number_archive import (
    _read, audit_discovery_archive, serialize_discovery_observation,
)
from looking_glass.pretrained_number_adapter import to_selector_observation
from looking_glass.pretrained_number_injection import InjectionObservation, PublicTraceBoundary
from looking_glass.pretrained_number_selection import freeze_discovery
from tests.test_pretrained_number_selection import fabricated_cases


def roles(case):
    return (("clean", case.clean), ("repeat", case.repeat), ("injected", case.injected),
            *((f"forward_{i}", value) for i, value in enumerate(case.forward, 1)),
            *((f"reverse_{i}", value) for i, value in enumerate(case.reverse, 1)),
            ("null", case.null), ("self", case.self_patch), ("upstream", case.upstream),
            ("downstream", case.downstream), ("verify_clean", case.verify_clean))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


class NumberInjectionArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = fabricated_cases()
        cls.prediction = freeze_discovery(cls.cases, private_controls_verified=True, execution_device="cpu")
        cls.records = tuple(serialize_discovery_observation(index, role, observation, execution_device="cpu")
                            for index, case in enumerate(cls.cases) for role, observation in roles(case))

    def test_all_observations_recompute_the_exact_freeze(self):
        result = audit_discovery_archive(self.records, self.prediction, execution_device="cpu")
        self.assertEqual(result["record_count"], 128)
        self.assertEqual(len(result["archive_sha256"]), 64)
        self.assertEqual(result, audit_discovery_archive(tuple(reversed(self.records)), self.prediction,
                                                         execution_device="cpu"))

    def test_missing_duplicate_or_swapped_role_is_rejected(self):
        for records in (self.records[:-1], (self.records[1],) + self.records[1:]):
            with self.subTest(count=len(records)), self.assertRaises(ValueError):
                audit_discovery_archive(records, self.prediction, execution_device="cpu")
        parsed = json.loads(self.records[0])
        parsed["role"] = "repeat"
        with self.assertRaises(ValueError):
            audit_discovery_archive((canonical(parsed),) + self.records[1:], self.prediction,
                                    execution_device="cpu")

    def test_changed_bits_nonfinite_and_noncanonical_are_rejected(self):
        parsed = json.loads(self.records[0])
        logits = bytearray(base64.b64decode(parsed["payload"]["logits"], validate=True))
        logits[3] ^= 0x80  # preserve a finite value but change +0 to -0
        parsed["payload"]["logits"] = base64.b64encode(logits).decode("ascii")
        with self.assertRaises(ValueError):
            audit_discovery_archive((canonical(parsed),) + self.records[1:], self.prediction,
                                    execution_device="cpu")
        struct.pack_into("<I", logits, 0, 0x7f800000)
        parsed["payload"]["logits"] = base64.b64encode(logits).decode("ascii")
        with self.assertRaises(ValueError):
            audit_discovery_archive((canonical(parsed),) + self.records[1:], self.prediction,
                                    execution_device="cpu")
        with self.assertRaises(ValueError):
            audit_discovery_archive((self.records[0] + b"\n",) + self.records[1:], self.prediction,
                                    execution_device="cpu")

    def test_public_projection_is_strict_and_excludes_private_sentinels(self):
        for record in self.records:
            self.assertNotIn(b"SITE_NONCE_PRIVATE_SENTINEL", record)
            self.assertNotIn(b"OWNER_TOKEN_PRIVATE_SENTINEL", record)
            self.assertNotIn(b"inject_before", record)
            self.assertNotIn(b"preceding_same_site_actions", record)
        observation = self.cases[0].forward[0]
        poisoned = replace(observation, action=replace(observation.action,
                            module_path="SITE_NONCE_PRIVATE_SENTINEL"))
        with self.assertRaises(ValueError):
            serialize_discovery_observation(0, "forward_1", poisoned, execution_device="cpu")
        with self.assertRaises(TypeError):
            serialize_discovery_observation(0, "injected", object(), execution_device="cpu")
        public = self.cases[0].clean
        source = InjectionObservation(public.request_id, public.input_sha256, public.condition, public.attention,
                    public.mlp, public.block, public.logits,
                    tuple(PublicTraceBoundary(**asdict(item)) for item in public.trace), None,
                    "OWNER_TOKEN_PRIVATE_SENTINEL")
        raw = serialize_discovery_observation(0, "clean", to_selector_observation(source), execution_device="cpu")
        self.assertNotIn(b"OWNER_TOKEN_PRIVATE_SENTINEL", raw)
        parsed = json.loads(raw)
        parsed["site_nonce"] = "SITE_NONCE_PRIVATE_SENTINEL"
        with self.assertRaises(ValueError):
            _read(canonical(parsed), "cpu")

    def test_signed_zero_and_subnormal_bits_round_trip_unchanged(self):
        source = self.cases[0].clean
        logits = bytearray(source.logits)
        struct.pack_into("<IIII", logits, 0, 0x80000000, 0x00000001, 0x80000001, 0x00000000)
        original = replace(source, logits=bytes(logits))
        record = serialize_discovery_observation(0, "clean", original, execution_device="cpu")
        _, _, restored = _read(record, "cpu")
        self.assertEqual(restored.logits, original.logits)


if __name__ == "__main__":
    unittest.main()
