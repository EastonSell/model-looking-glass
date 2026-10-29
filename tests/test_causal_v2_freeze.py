import copy
from dataclasses import replace
import json
import math
import unittest

from looking_glass.causal_v2_custody import build_exclusion_union, commitment
from looking_glass.causal_v2_freeze import _REQUIRED_SOURCE_FILES, build_freeze, public_pre_manifest
from looking_glass.causal_v2_preparation import prepare_development, verify_prepared


def trace(value):
    return [{"node_id": f"block.{index}.output", "sequence": index, "occurrence": 0,
             "shape": [1, 8], "dtype": "torch.float32", "device": "cpu",
             "values": [float(value if index else 0.0)] + [0.0] * 7}
            for index in range(4)]


def selection():
    rows = []
    for index in range(2):
        polarity = 1 if index == 0 else -1
        conditions = []
        for strength in (0.25, 0.75):
            base = [0.0, strength * polarity]
            clean = [0.0, 0.0]
            conditions.append({"strength": strength, "perturbed_trace": trace(strength * polarity),
                "first_unequal": "block.1.output", "last_equal": "block.0.output",
                "l2_by_block": [0.0, strength, strength, strength],
                "perturbed_logits": base, "clean_to_perturbed_logits": [clean] * 3,
                "perturbed_to_clean_logits": [base] * 3, "null_logits": [base] * 3,
                "identity_logits": [base] * 3, "upstream_logits": base,
                "clean_verify_logits": clean, "effects": [-strength * polarity] * 3})
        rows.append({"case_index": index, "polarity": polarity, "clean_trace": trace(0.0),
                     "repeated_clean_trace": trace(0.0), "clean_logits": [0.0, 0.0],
                     "repeated_clean_logits": [0.0, 0.0], "conditions": conditions})
    predictions = {str(site): {str(strength): {"positive": -strength, "negative": strength,
                    "constant": 0.0, "zero": 0.0} for strength in (0.25, 0.75)} for site in (1, 2, 3)}
    return {"first_changed": 1, "magnitude": 1, "first_votes": {"1": 4, "2": 0, "3": 0},
            "mean_l2": {"1": 0.5, "2": 0.5, "3": 0.5}, "observation_misses": 0,
            "no_injection": {"cases": 2, "false_detections": 0, "first_changed": None, "magnitude": None},
            "random": {"sites": [1, 2, 3], "each_probability": "1/3"},
            "predictions": predictions, "measurements": rows}


class FreezeTests(unittest.TestCase):
    def setUp(self):
        seed = 41977
        context = {"protocol_id": "synthetic-causal-v2", "protocol_sha256": "a" * 64,
                   "source_revision": "b" * 40,
                   "model_state_hashes": [{"seed": seed, "sha256": "c" * 64}]}
        pilot = build_exclusion_union([{"inputs": [{"case_id": "pilot", "values":
                              [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]}])
        historic = [{"case_id": "historic", "values": [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}]
        prepared = prepare_development(context, (seed,), pilot, pilot, historic, heldout_root=b"h" * 32,
            site_nonce=b"s" * 32, prediction_nonce=b"p" * 32, shuffle=lambda values: values.reverse(), count=2)
        self.snapshot = verify_prepared(prepared, context, pilot, pilot, historic,
                                        development=True, seeds=(seed,), count=2)
        self.clarifications = {name + "_clarification_sha256": str(index) * 64 for index, name in
                               enumerate(("numeric", "source", "self_patch", "resource"), start=1)}
        source = {path: "f" * 64 for path in _REQUIRED_SOURCE_FILES}
        source["docs/CAUSAL_CONFIRMATION_V2.md"] = "a" * 64
        for name, label in (("numeric", "NUMERIC"), ("source", "SOURCE"),
                            ("self_patch", "SELF_PATCH"), ("resource", "RESOURCE")):
            source[f"docs/CAUSAL_CONFIRMATION_V2_{label}_CLARIFICATION.md"] = self.clarifications[name + "_clarification_sha256"]
        self.source = source
        self.runtime = {"python_version": "3.11.15", "torch_version": "2.8.0+cu128", "system": "Windows",
                        "machine": "AMD64", "byteorder": "little", "intraop_threads": 1,
                        "denormal_preserved": True, "torch_build_sha256": "e" * 64}
        self.manifest = public_pre_manifest(self.snapshot, source_sha256=source,
            clarification_sha256=self.clarifications, runtime=self.runtime, created_utc="2026-10-29T02:00:00+00:00")
        self.cells = [{"cell_id": row["cell_id"], "model_seed": seed, "selection": selection()}
                      for row in prepared.schedule["assignments"]]

    def test_global_freeze_and_private_prediction_order(self):
        freeze, private, public_commit = build_freeze(self.snapshot, self.manifest, self.cells)
        frozen = json.loads(freeze)
        predictions = json.loads(private)
        sealed = json.loads(public_commit)
        self.assertEqual(len(frozen["cells"]), 9)
        self.assertEqual(len(predictions["predictions"]), 9 * 2 * 2 * 3)
        self.assertEqual([row["site"] for row in predictions["predictions"][:3]], [1, 2, 3])
        self.assertEqual(predictions["predictions"][0]["polarity_prediction"], -0.25)
        self.assertEqual(predictions["predictions"][6]["polarity_prediction"], 0.25)
        self.assertEqual(sealed["commitment"], commitment("predictions", b"p" * 32, predictions))
        self.assertNotIn(b"dev-v2-m41977-h00", freeze)
        self.assertNotIn(b"injection_node", freeze)

    def test_missing_or_reordered_cells_fail(self):
        with self.assertRaises(ValueError):
            build_freeze(self.snapshot, self.manifest, self.cells[:-1])
        with self.assertRaises(ValueError):
            build_freeze(self.snapshot, self.manifest, list(reversed(self.cells)))

    def test_changed_trace_control_or_prediction_fails(self):
        for mutation in ("trace", "control", "prediction", "effect_and_mean", "final_state"):
            cells = copy.deepcopy(self.cells)
            selected = cells[0]["selection"]
            if mutation == "trace":
                selected["measurements"][0]["conditions"][0]["perturbed_trace"][1]["values"][0] = 0.5
            elif mutation == "control":
                selected["measurements"][0]["conditions"][0]["null_logits"][0] = [0.0, 2.0]
            elif mutation == "prediction":
                selected["predictions"]["1"]["0.25"]["positive"] = 0.0
            elif mutation == "effect_and_mean":
                selected["measurements"][0]["conditions"][0]["effects"][0] = 123.0
                selected["predictions"]["1"]["0.25"]["positive"] = 123.0
                selected["predictions"]["1"]["0.25"]["constant"] = 61.625
            else:
                condition = selected["measurements"][0]["conditions"][0]
                condition["perturbed_trace"] = trace(0.0)
                condition["first_unequal"], condition["last_equal"] = None, "block.3.output"
                condition["l2_by_block"] = [0.0] * 4
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_freeze(self.snapshot, self.manifest, cells)

    def test_manifest_rejects_wrong_protocol_or_clarification(self):
        source = dict(self.source)
        source["docs/CAUSAL_CONFIRMATION_V2.md"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            public_pre_manifest(self.snapshot, source_sha256=source,
                clarification_sha256=self.clarifications, runtime=self.runtime, created_utc="2026-10-29T02:00:00+00:00")
        with self.assertRaises(ValueError):
            build_freeze(self.snapshot, self.manifest + b" ", self.cells)
        source = dict(self.source, **{"C:\\private\\file.py": "f" * 64})
        with self.assertRaises(ValueError):
            public_pre_manifest(self.snapshot, source_sha256=source,
                clarification_sha256=self.clarifications, runtime=self.runtime, created_utc="2026-10-29T02:00:00+00:00")
        from looking_glass.causal_v2_custody import canonical_json
        malformed = json.loads(self.manifest)
        del malformed["settings"]
        with self.assertRaises(ValueError):
            build_freeze(self.snapshot, canonical_json(malformed), self.cells)

    def test_forged_snapshot_cannot_publish_private_fields_or_bad_commitment(self):
        from looking_glass.causal_v2_custody import canonical_json
        public = json.loads(self.snapshot.public)
        public["injection_node"] = "private-target"
        forged = replace(self.snapshot, public=canonical_json(public))
        with self.assertRaises(ValueError):
            public_pre_manifest(forged, source_sha256=self.source,
                clarification_sha256=self.clarifications, runtime=self.runtime,
                created_utc="2026-10-29T02:00:00+00:00")
        heldout = json.loads(self.snapshot.heldout)
        heldout["cases"][0]["values"][0] = 0.1
        forged = replace(self.snapshot, heldout=canonical_json(heldout))
        with self.assertRaises(ValueError):
            public_pre_manifest(forged, source_sha256=self.source,
                clarification_sha256=self.clarifications, runtime=self.runtime,
                created_utc="2026-10-29T02:00:00+00:00")

    def test_non_float32_trace_value_is_rejected(self):
        cells = copy.deepcopy(self.cells)
        cells[0]["selection"]["measurements"][0]["clean_trace"][1]["values"][0] = 0.1
        with self.assertRaises(ValueError):
            build_freeze(self.snapshot, self.manifest, cells)

    def test_same_model_case_must_reuse_identical_clean_baseline(self):
        cells = copy.deepcopy(self.cells)
        row = cells[1]["selection"]["measurements"][0]
        for name in ("clean_logits", "repeated_clean_logits"):
            row[name] = [value + 1.0 for value in row[name]]
        for condition in row["conditions"]:
            for name in ("perturbed_logits", "upstream_logits", "clean_verify_logits"):
                condition[name] = [value + 1.0 for value in condition[name]]
            for name in ("clean_to_perturbed_logits", "perturbed_to_clean_logits", "null_logits", "identity_logits"):
                condition[name] = [[value + 1.0 for value in output] for output in condition[name]]
        with self.assertRaisesRegex(ValueError, "clean baseline differs"):
            build_freeze(self.snapshot, self.manifest, cells)

    def test_same_model_case_must_reuse_identical_clean_trace(self):
        cells = copy.deepcopy(self.cells)
        row = cells[1]["selection"]["measurements"][0]
        for event in row["clean_trace"] + row["repeated_clean_trace"]:
            event["values"][1] = 0.5
        for condition in row["conditions"]:
            for event in condition["perturbed_trace"]:
                event["values"][1] = 0.5
        with self.assertRaisesRegex(ValueError, "clean baseline differs"):
            build_freeze(self.snapshot, self.manifest, cells)

    def test_float32_metric_preserves_subnormals_and_rejects_overflow(self):
        from looking_glass.causal_v2_freeze import _metric
        minimum = math.ldexp(1.0, -149)
        self.assertEqual(_metric([0.0, minimum]), minimum)
        self.assertEqual(_metric([minimum, 0.0]), -minimum)
        self.assertEqual(math.copysign(1.0, _metric([0.0, -0.0])), -1.0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _metric([-float.fromhex("0x1.fffffep+127"), float.fromhex("0x1.fffffep+127")])


if __name__ == "__main__":
    unittest.main()
