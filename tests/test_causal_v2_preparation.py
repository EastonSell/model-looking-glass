import hashlib
import unittest

from looking_glass.causal_v2_custody import build_exclusion_union, canonical_json, commitment
from looking_glass.causal_v2_preparation import (DISCOVERY_ROOT, MODEL_SEEDS, NODES,
                                                  prepare_development, verify_prepared)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.seed = 41977  # Development only; no study seed or model forward.
        self.context = {"protocol_id": "synthetic-causal-v2", "protocol_sha256": "a" * 64,
                        "source_revision": "b" * 40,
                        "model_state_hashes": [{"seed": self.seed, "sha256": "c" * 64}]}
        self.pilot = build_exclusion_union([{"inputs": [
            {"case_id": "pilot", "values": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]}])
        self.union = self.pilot
        self.historical = [{"case_id": "historical", "values": [-2.0, 0.0, 0.0, 0.0,
                                                                   0.0, 0.0, 0.0, 0.0]}]
        self.roots = {"heldout_root": b"h" * 32, "site_nonce": b"s" * 32,
                      "prediction_nonce": b"p" * 32}

    def prepare(self, **overrides):
        args = dict(context=self.context, seeds=(self.seed,), exclusion_bytes=self.union,
                    pilot_bytes=self.pilot, historical_cases=self.historical,
                    shuffle=lambda values: values.reverse(), count=2, **self.roots)
        args.update(overrides)
        return prepare_development(**args)

    def test_fixed_formula_order_and_disjointness(self):
        prepared = self.prepare()
        discovery = prepared.public["discovery_cases"]
        heldout = prepared.heldout["cases"]
        self.assertEqual([item["case_id"] for item in discovery],
                         ["dev-v2-m41977-d00", "dev-v2-m41977-d01"])
        self.assertEqual([item["case_id"] for item in heldout],
                         ["dev-v2-m41977-h00", "dev-v2-m41977-h01"])
        digest = hashlib.sha256(b"MLG-v2-cases\0" + DISCOVERY_ROOT
                                + self.seed.to_bytes(8, "big") + b"D" + b"\0\0" + b"\0\0").digest()
        expected = [((digest[0] & 15) + 1) / 16]
        expected += [((digest[j] & 31) - 16) / 16 for j in range(1, 8)]
        self.assertEqual(discovery[0]["values"], expected)
        self.assertGreater(discovery[0]["values"][0], 0)
        self.assertLess(discovery[1]["values"][0], 0)
        self.assertEqual(len({tuple(row["values"]) for row in discovery + heldout}), 4)
        self.assertEqual([row["injection_node"] for row in prepared.schedule["assignments"]],
                         list(reversed(NODES)))

    def test_commitments_bind_private_records_and_context(self):
        prepared = self.prepare()
        public = prepared.public
        context_sha = hashlib.sha256(canonical_json(self.context)).hexdigest()
        self.assertEqual(public["context_sha256"], context_sha)
        self.assertEqual(public["heldout_commitment"],
                         commitment("heldout", self.roots["heldout_root"], prepared.heldout))
        self.assertEqual(public["schedule_commitment"],
                         commitment("schedule", self.roots["site_nonce"], prepared.schedule))
        self.assertNotIn("assignments", public)
        self.assertNotIn("heldout_root", public)
        self.assertEqual(public["mode"], "development")
        self.assertEqual(public["historical_exclusions_sha256"], hashlib.sha256(
            build_exclusion_union([{"inputs": self.historical}])).hexdigest())
        self.assertEqual(public["historical_exclusions_count"], 1)
        displayed = repr(prepared)
        self.assertNotIn("dev-v2-m41977-h00", displayed)
        self.assertNotIn("injection_node", displayed)
        self.assertNotIn("hhhhhhhh", displayed)
        self.assertNotIn("ssssssss", displayed)
        self.assertNotIn("pppppppp", displayed)

    def test_collision_increments_attempt_before_accepting(self):
        first = self.prepare().public["discovery_cases"][0]["values"]
        union = build_exclusion_union([{"inputs": [
            {"case_id": "pilot", "values": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"case_id": "collision", "values": first}]}])
        prepared = self.prepare(exclusion_bytes=union)
        next_digest = hashlib.sha256(b"MLG-v2-cases\0" + DISCOVERY_ROOT
                                     + self.seed.to_bytes(8, "big") + b"D" + b"\0\0" + b"\0\1").digest()
        expected = [((next_digest[0] & 15) + 1) / 16]
        expected += [((next_digest[j] & 31) - 16) / 16 for j in range(1, 8)]
        self.assertEqual(prepared.public["discovery_cases"][0]["values"], expected)

    def test_rejects_unbound_pilot_reused_roots_or_bad_schedule(self):
        other = build_exclusion_union([{"inputs": [
            {"case_id": "other", "values": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]}])
        with self.assertRaisesRegex(ValueError, "pilot"):
            self.prepare(exclusion_bytes=other)
        with self.assertRaises(ValueError):
            self.prepare(site_nonce=self.roots["heldout_root"])
        with self.assertRaisesRegex(ValueError, "permutation"):
            self.prepare(shuffle=lambda values: values.pop())
        with self.assertRaises(ValueError):
            self.prepare(exclusion_bytes=self.union + b" ")
        signed_zero = build_exclusion_union([{"inputs": [
            {"case_id": "pilot", "values": [2.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]}])
        with self.assertRaisesRegex(ValueError, "pilot"):
            self.prepare(exclusion_bytes=signed_zero)

    def test_rejects_wrong_context_or_reserved_development_seeds(self):
        context = dict(self.context, source_revision="not-a-commit")
        with self.assertRaises(ValueError):
            self.prepare(context=context)
        context = dict(self.context, model_state_hashes=[{"seed": True, "sha256": "c" * 64}])
        with self.assertRaises(ValueError):
            self.prepare(context=context)
        with self.assertRaisesRegex(ValueError, "different model seeds"):
            self.prepare(seeds=(MODEL_SEEDS[0],))

    def test_snapshot_revalidates_all_bindings_and_detects_mutation(self):
        prepared = self.prepare()
        def checked(value, **overrides):
            args = dict(expected_context=self.context, exclusion_bytes=self.union,
                        pilot_bytes=self.pilot, historical_cases=self.historical,
                        development=True, seeds=(self.seed,), count=2)
            args.update(overrides)
            return verify_prepared(value, **args)
        snapshot = checked(prepared)
        self.assertEqual(snapshot.public, canonical_json(prepared.public))
        self.assertNotIn("hhhhhhhh", repr(snapshot))
        prepared.heldout["cases"][0]["values"][0] = 9.0
        with self.assertRaisesRegex(ValueError, "bindings"):
            checked(prepared)
        self.assertNotEqual(snapshot.heldout, canonical_json(prepared.heldout))
        other = self.prepare()
        other.schedule["assignments"][0]["model_seed"] = True
        with self.assertRaisesRegex(ValueError, "bindings"):
            checked(other)
        with self.assertRaises(ValueError):
            checked(self.prepare(), historical_cases=[{"case_id": "other-history",
                    "values": [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}])


if __name__ == "__main__":
    unittest.main()
