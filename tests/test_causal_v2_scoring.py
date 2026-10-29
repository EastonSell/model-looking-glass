"""Fabricated evidence only; no study cases, schedules, or model execution."""
import copy
import hashlib
import json
import os
import subprocess
import struct
import sys
import unittest

from looking_glass.causal_v2_custody import canonical_json, commitment
from looking_glass import registry


SEEDS = tuple(range(41977, 41983))
NODES = tuple(f"block.{block}.{operation}" for block in (1, 2, 3)
              for operation in ("branch.0", "activation", "merge"))
NONCES = (b"h" * 32, b"s" * 32, b"p" * 32)
ATTESTATIONS = {name: True for name in ("phase_order_verified", "input_model_pairing_verified",
                "private_plan_metadata_verified", "durable_evidence_verified",
                "case_generation_verified", "discovery_freeze_verified")}
ATTESTATIONS["exposure_classification"] = "unexposed"


def bits(value):
    return struct.unpack(">I", struct.pack(">f", value))[0]


def tensor(values):
    return {"dtype": "torch.float32", "device": "cpu", "shape": [1, len(values)],
            "bits": [bits(x) for x in values]}


def trace(run, values):
    return [{"metadata": {"run_id": run, "node_id": f"block.{i}.output", "sequence": i,
                           "occurrence": 0, "shape": [1, 8], "dtype": "torch.float32",
                           "device": "cpu", "tensor_bytes": 32}, "value": tensor(values[i])}
            for i in range(4)]


def input_hash(values):
    return hashlib.sha256(b"MLG-v2-input-f32be-v1\0" + struct.pack(">II", 1, 8) +
                          b"".join(struct.pack(">I", bits(x)) for x in values)).hexdigest()


def capture_metadata(path):
    return {"module_path": path, "module_aliases": [path], "call_site_known": False,
            "forward_index": 0, "invocation": 0, "batch_index": None, "token_index": None,
            "output_shape": [1, 8], "selected_shape": [1, 8], "dtype": "torch.float32", "device": "cpu"}


def plan_metadata(kind, site):
    path = f"blocks.{site}.output_tap"
    if kind in ("forward", "reverse", "upstream"):
        names, operations, capture_bytes, source_bytes = ("patch", "post_action"), ("patch", "capture"), 32, 32
    elif kind == "null":
        names, operations, capture_bytes, source_bytes = ("null", "post_action"), ("null", "capture"), 32, 0
    else:
        names, operations, capture_bytes, source_bytes = (("self_source", "self_patch", "post_action"),
            ("capture", "patch", "capture"), 64, 0)
    steps = []
    for i, (name, operation) in enumerate(zip(names, operations)):
        steps.append({"name": name, "operation": operation, "order": i, "module_path": path,
                      "module_aliases": [path], "call_site_known": False, "forward_index": 0,
                      "invocation": 0, "batch_index": None, "token_index": None, "output_selector": [],
                      "output_shape": [1, 8], "selected_shape": [1, 8], "dtype": "torch.float32", "device": "cpu",
                      "source_capture": "self_source" if kind == "self" and operation == "patch" else None,
                      "preceding_same_site_actions": list(names[:i])})
    return {"steps": steps, "capture_bytes": capture_bytes, "source_bytes": source_bytes}


def fixture():
    """Use out-of-protocol vectors and development IDs, not generated study data."""
    context = {"protocol_id": "synthetic-causal-v2", "protocol_sha256": "a" * 64,
               "source_revision": "b" * 40,
               "model_state_hashes": [{"seed": s, "sha256": f"{s:064x}"} for s in SEEDS]}
    context_hash = hashlib.sha256(canonical_json(context)).hexdigest()
    cases, discovery, assignments, cells = [], [], [], []
    for mi, seed in enumerate(SEEDS):
        for i in range(16):
            values = [2.0 if i % 2 == 0 else -2.0, float(100 + 16 * mi + i)] + [0.0] * 6
            cases.append({"case_id": f"dev-v2-m{seed}-h{i:02d}", "model_seed": seed,
                          "split": "heldout", "index": i, "values": values})
            discovery.append({"case_id": f"dev-v2-m{seed}-d{i:02d}", "model_seed": seed,
                              "split": "discovery", "index": i,
                              "values": [values[0], -values[1]] + [0.0] * 6})
        for ci, node in enumerate(NODES):
            cid = f"dev-v2-m{seed}-c{ci:02d}"
            assignments.append({"model_seed": seed, "cell_id": cid, "injection_node": node})
            means = {str(site): {str(q): {"positive": -q, "negative": q, "constant": 0.0,
                                         "zero": 0.0} for q in (0.25, 0.75)} for site in (1, 2, 3)}
            cells.append({"model_seed": seed, "cell_id": cid,
                          "selection": {"first_changed": None if mi == 0 or ci == 8 else 1,
                                        "magnitude": 3, "predictions": means,
                                        "no_injection": {"cases": 16, "false_detections": 0,
                                                         "first_changed": None, "magnitude": None}}})
    holdout = {"context_sha256": context_hash, "cases": cases}
    schedule = {"context_sha256": context_hash, "assignments": assignments}
    manifest = {"schema": "synthetic-v2-pre-manifest-v1", "mode": "development",
                "context": context, "context_sha256": context_hash,
                "discovery_cases": discovery,
                "heldout_commitment": commitment("heldout", NONCES[0], holdout),
                "schedule_commitment": commitment("schedule", NONCES[1], schedule)}
    manifest_raw = canonical_json(manifest)
    mh = hashlib.sha256(manifest_raw).hexdigest()
    freeze = {"schema": "synthetic-v2-prediction-freeze-v1", "pre_manifest_sha256": mh,
              "cells": cells}
    freeze_raw = canonical_json(freeze)
    fh = hashlib.sha256(freeze_raw).hexdigest()
    predictions = []
    for ci, cell in enumerate(cells):
        seed = cell["model_seed"]
        for case in cases[(ci // 9) * 16:(ci // 9 + 1) * 16]:
            polarity = 1 if case["index"] % 2 == 0 else -1
            for q in (0.25, 0.75):
                for site in (1, 2, 3):
                    predictions.append({"model_seed": seed, "cell_id": cell["cell_id"],
                        "case_id": case["case_id"], "strength": q, "site": site,
                        "polarity": polarity, "polarity_prediction": -q * polarity,
                        "constant_prediction": 0.0, "zero_prediction": 0.0})
    pred = {"pre_manifest_sha256": mh, "freeze_sha256": fh, "predictions": predictions}
    pred_raw = canonical_json(pred)
    pred_hash = hashlib.sha256(pred_raw).hexdigest()
    public_commit = {"schema": "synthetic-v2-prediction-commitment-v1", "pre_manifest_sha256": mh,
                     "freeze_sha256": fh, "commitment": commitment("predictions", NONCES[2], pred)}
    chunks = []
    for ci, cell in enumerate(cells):
        rows = []
        seed, cid = cell["model_seed"], cell["cell_id"]
        block = int(assignments[ci]["injection_node"].split(".")[1])
        for case in cases[(ci // 9) * 16:(ci // 9 + 1) * 16]:
            i = case["index"]
            polarity = 1 if i % 2 == 0 else -1
            for qi, q in enumerate((0.25, 0.75)):
                delta = q * polarity
                zero8, delta8 = [0.0] * 8, [delta] + [0.0] * 7
                clean_values = [zero8] * 4
                changed_values = [delta8 if j >= block else zero8 for j in range(4)]
                clean_run, repeat_run, changed_run = (f"clean-{seed}-{i}", f"repeat-{seed}-{i}",
                                                       f"changed-{ci}-{i}-{qi}")
                clean, changed = tensor([0.0, 0.0]), tensor([0.0, delta])
                plans = []
                for kind, sites in (("forward", (1, 2, 3)), ("reverse", (1, 2, 3)),
                                    ("null", (1, 2, 3)), ("self", (1, 2, 3)), ("upstream", (0,))):
                    for site in sites:
                        source = changed_values[site] if kind in ("reverse", "null", "self") else clean_values[site]
                        captures = {"post_action": tensor(source)}
                        if kind == "self":
                            captures["self_source"] = tensor(source)
                        source_role = {"forward": "clean_sparse_trace", "reverse": "perturbed_sparse_trace",
                                       "null": None, "self": "same_forward_self_source",
                                       "upstream": "clean_sparse_trace"}[kind]
                        comparison_role = ("perturbed_sparse_trace" if kind in ("reverse", "null", "self")
                                           else "clean_sparse_trace")
                        base_role = "clean" if kind == "reverse" else "perturbed"
                        comparison_id = f"{clean_run if comparison_role == 'clean_sparse_trace' else changed_run}/block.{site}.output"
                        plans.append({"kind": kind, "site": site, "metadata": plan_metadata(kind, site),
                                      "captures": captures, "source_role": source_role,
                                      "source_id": None if kind == "null" else
                                      "same_forward_self_source" if kind == "self" else comparison_id,
                                      "comparison_role": comparison_role, "comparison_id": comparison_id,
                                      "base_role": base_role, "base_id": f"{clean_run if base_role == 'clean' else changed_run}/logits"})
                first = ci * 96 + i * 6 + qi * 3
                operation = assignments[ci]["injection_node"].split(".", 2)[2]
                target_path = f"blocks.{block}.branch_taps.0" if operation == "branch.0" else f"blocks.{block}.{operation}_tap"
                rows.append({"schema": "causal-v2-condition-record-v2", "cell_id": cid,
                    "phase": "heldout", "case_id": case["case_id"], "case_index": i,
                    "condition_id": f"{cid}-{i}-{qi}", "model_seed": seed,
                    "model_state_sha256": f"{seed:064x}", "input_sha256": input_hash(case["values"]),
                    "strength": q, "prediction_ref": f"sha256:{pred_hash}:{first}:3",
                    "traces": {"clean": trace(clean_run, clean_values),
                               "repeat_clean": trace(repeat_run, clean_values),
                               "perturbed": trace(changed_run, changed_values)},
                    "logits": {"clean": clean, "repeat_clean": clean, "perturbed": changed,
                               "forward": [clean if j >= block else changed for j in (1, 2, 3)],
                               "reverse": [changed if j >= block else clean for j in (1, 2, 3)],
                               "null": [changed] * 3, "self": [changed] * 3,
                               "upstream": changed, "post_hook_clean": clean},
                    "plans": plans,
                    "private_integrity": {"target": assignments[ci]["injection_node"], "delta": delta,
                        "clean_target": tensor(zero8), "perturbed_target": tensor(delta8),
                        "clean_capture_metadata": capture_metadata(target_path),
                        "perturbed_capture_metadata": capture_metadata(target_path)},
                    "comparison": {"status": "difference_observed", "first_unequal": f"block.{block}.output",
                        "last_equal": f"block.{block-1}.output", "l2_by_block":
                        [q if j >= block else 0.0 for j in range(4)]}})
        chunks.append(registry._json_bytes({"name": f"dev-{ci}.json", "records": rows,
                                            "schema": "causal-v2-chunk-v1", "state": "complete"}))
    return [manifest_raw, freeze_raw, pred_raw, canonical_json(public_commit),
            canonical_json(holdout), canonical_json(schedule), *NONCES, chunks]


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = fixture()

    def score(self, args=None, **attestations):
        from looking_glass.causal_v2_scoring import score_heldout
        kwargs = dict(ATTESTATIONS)
        kwargs.update(attestations)
        return json.loads(score_heldout(*(args or self.args), **kwargs))

    def test_prespecified_model_and_random_summaries(self):
        result = self.score()
        self.assertEqual(result["status"], "complete")
        models = result["models"]
        self.assertEqual(len(models), 6)
        self.assertEqual(models[0]["selection"]["first_changed"]["selected_cells"], 0)
        self.assertIsNone(models[0]["selection"]["first_changed"]["polarity_mae"])
        self.assertEqual(models[1]["selection"]["first_changed"]["selected_cells"], 8)
        self.assertAlmostEqual(models[1]["selection"]["first_changed"]["polarity_mae"], 0.3125)
        self.assertAlmostEqual(models[1]["selection"]["first_changed"]["constant_mae"], 0.1875)
        self.assertAlmostEqual(models[1]["selection"]["first_changed"]["constant_improvement"], -0.125)
        self.assertAlmostEqual(models[1]["selection"]["magnitude"]["polarity_mae"], 0.0)
        self.assertEqual(models[1]["selection"]["magnitude"]["sign"]["eligible"], 288)
        self.assertEqual(models[1]["selection"]["magnitude"]["sign"]["polarity_accuracy"], 1.0)
        self.assertEqual(models[1]["selection"]["magnitude"]["sign"]["constant_accuracy"], 0.0)
        self.assertEqual(models[1]["selection"]["first_changed"]["sign"]["eligible"], 96)
        self.assertEqual(models[1]["selection"]["first_changed"]["sign"]["excluded"], 160)
        self.assertAlmostEqual(models[1]["random"]["polarity_mae"], 1 / 6)
        self.assertAlmostEqual(models[1]["random"]["zero_mae"], 1 / 3)
        self.assertIsNone(result["equal_model_means"]["selection"]["first_changed"]["polarity_mae"])
        self.assertEqual(result["conditions_count"], 1728)
        for split in ("discovery", "heldout"):
            self.assertEqual(models[1]["no_injection"][split]["cases"], 16)
            self.assertEqual(models[1]["no_injection"][split]["false_detections"], 0)
        self.assertEqual(result["mode"], "development")
        self.assertEqual(result["interpretation"], "development_only")
        self.assertEqual(len(result["repair_groups"]), 6 * 9 * 2 * 2 * 3)
        self.assertEqual(result["cells"][0]["chunk_sha256"], hashlib.sha256(self.args[9][0]).hexdigest())
        self.assertEqual(result["conditions"][0]["sites"][0]["effect"], -0.25)
        self.assertEqual(result["conditions"][0]["sites"][0]["reverse_effect"], 0.25)
        self.assertEqual(result["repair_groups"][0]["prediction_mae"]["polarity_mae"], 0.0)

    def test_zero_signs_do_not_change_reused_baseline_or_controls(self):
        chunks = list(self.args[9])
        chunk = json.loads(chunks[1])
        row = chunk["records"][0]
        for name in ("clean", "repeat_clean", "perturbed"):
            row["traces"][name][0]["value"]["bits"][1] = 0x80000000
        for name in ("clean", "repeat_clean", "post_hook_clean"):
            row["logits"][name]["bits"][0] = 0x80000000
        chunks[1] = registry._json_bytes(chunk)
        args = list(self.args); args[9] = chunks
        result = self.score(args)
        self.assertEqual(result["status"], "complete")

    def test_encoder_v2_output_is_accepted_from_saved_chunk(self):
        from types import SimpleNamespace
        import torch
        from looking_glass.causal_v2_evidence_encoder import _record
        from looking_glass.causal_v2_selection import DiscoveryCondition
        from looking_glass.probes import TraceEvent
        chunk = json.loads(self.args[9][0])
        row = chunk["records"][0]

        def torch_value(value):
            signed = [bit if bit < 0x80000000 else bit - 0x100000000 for bit in value["bits"]]
            return torch.tensor([signed], dtype=torch.int32).view(torch.float32)

        def events(items):
            return tuple(TraceEvent(event["metadata"]["run_id"], event["metadata"]["node_id"],
                                    event["metadata"]["sequence"], 0, (1, 8), "torch.float32", "cpu",
                                    torch_value(event["value"])) for event in items)

        logits, traces = row["logits"], row["traces"]
        condition = DiscoveryCondition(row["strength"], events(traces["perturbed"]),
            torch_value(logits["perturbed"]),
            *(tuple(torch_value(item) for item in logits[name]) for name in ("forward", "reverse", "null", "self")),
            torch_value(logits["upstream"]), torch_value(logits["post_hook_clean"]))
        plans = [{"kind": p["kind"], "site": p["site"], "metadata": p["metadata"],
                  "captures": {key: value["bits"] for key, value in p["captures"].items()},
                  "source_identity": p["source_role"], "comparison_identity": p["comparison_role"],
                  "base_identity": p["base_role"]} for p in row["plans"]]
        private = row["private_integrity"]
        source = {"clean_trace": events(traces["clean"]), "repeat_trace": events(traces["repeat_clean"]),
                  "clean_logits": torch_value(logits["clean"]), "repeat_logits": torch_value(logits["repeat_clean"]),
                  "condition": condition, "plans": plans,
                  "private_integrity": {"target": private["target"], "delta": private["delta"],
                    "clean_target_bits": private["clean_target"]["bits"],
                    "perturbed_target_bits": private["perturbed_target"]["bits"],
                    "clean_capture_metadata": private["clean_capture_metadata"],
                    "perturbed_capture_metadata": private["perturbed_capture_metadata"]},
                  "comparison": row["comparison"], "input_sha256": row["input_sha256"],
                  "model_state_sha256": row["model_state_sha256"]}
        clean_id = traces["clean"][0]["metadata"]["run_id"]
        changed_id = traces["perturbed"][0]["metadata"]["run_id"]
        refs = {"clean_sparse_trace": clean_id, "perturbed_sparse_trace": changed_id,
                "clean": clean_id + "/logits", "perturbed": changed_id + "/logits"}
        buffer = SimpleNamespace(cell_id=row["cell_id"], phase="heldout", model_seed=row["model_seed"])
        encoded = _record(buffer, source, case_id=row["case_id"], case_index=row["case_index"],
                          condition_id=row["condition_id"], prediction_ref=row["prediction_ref"], references=refs)
        self.assertEqual(json.loads(encoded), row)
        chunks = list(self.args[9]); chunk["records"][0] = json.loads(encoded)
        chunks[0] = registry._json_bytes(chunk)
        args = list(self.args); args[9] = chunks
        self.assertEqual(self.score(args)["status"], "complete")
        self.assertFalse(torch.cuda.is_initialized())

    def test_sign_rule_is_strict_and_abstention_is_undefined(self):
        from looking_glass.causal_v2_scoring import _sign_summary
        rows = [{"effect": observed, "polarity_prediction": prediction,
                 "constant_prediction": baseline} for observed, prediction, baseline in
                ((1e-7, 1.0, 1.0), (1.01e-7, 1e-7, 1.0), (-0.5, -0.25, 0.0))]
        result = _sign_summary(rows)
        self.assertEqual((result["eligible"], result["excluded"]), (2, 1))
        self.assertEqual(result["polarity_accuracy"], 0.5)
        self.assertEqual(result["constant_accuracy"], 0.5)
        self.assertTrue(result["sign_variation_tested"])
        self.assertIsNone(_sign_summary([])["polarity_accuracy"])

    def test_metric_rounds_in_float32_and_preserves_subnormals_under_ftz(self):
        from looking_glass.causal_v2_scoring import _metric
        self.assertEqual(_metric({"bits": [bits(1.0), bits(2.0 ** -25)]}), -1.0)
        with self.assertRaises(ValueError):
            _metric({"bits": [0xff7fffff, 0x7f7fffff]})
        code = r'''
import torch
from looking_glass.causal_v2_scoring import _metric
for flush in (False, True):
    if not torch.set_flush_denormal(flush):
        raise SystemExit(77)
    assert _metric({"bits": [0, 1]}) == 2.0 ** -149
    assert _metric({"bits": [0, 0x80000001]}) == -(2.0 ** -149)
assert not torch.cuda.is_initialized()
'''
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=30)
        if child.returncode == 77:
            self.skipTest("thread-local flush mode is unavailable")
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_masked_injection_is_a_retained_observation_miss(self):
        chunks = list(self.args[9])
        chunk = json.loads(chunks[0])
        row = chunk["records"][0]
        zeros = tensor([0.0] * 8)
        zero_logits = tensor([0.0, 0.0])
        for event in row["traces"]["perturbed"]:
            event["value"] = zeros
        for name in ("clean", "repeat_clean", "perturbed", "upstream", "post_hook_clean"):
            row["logits"][name] = zero_logits
        for name in ("forward", "reverse", "null", "self"):
            row["logits"][name] = [zero_logits] * 3
        for plan in row["plans"]:
            plan["captures"]["post_action"] = zeros
            if plan["kind"] == "self":
                plan["captures"]["self_source"] = zeros
        row["comparison"] = {"status": "no_difference_observed", "first_unequal": None,
                             "last_equal": "block.3.output", "l2_by_block": [0.0] * 4}
        chunks[0] = registry._json_bytes(chunk)
        args = list(self.args); args[9] = chunks
        result = self.score(args)
        self.assertEqual(result["conditions_count"], 1728)
        self.assertEqual(result["cells"][0]["observation_misses"], 1)
        self.assertEqual(result["conditions"][0]["comparison"]["status"], "no_difference_observed")
        self.assertEqual(result["models"][0]["no_injection"]["heldout"]["cases"], 16)

    def test_requires_explicit_external_attestations(self):
        for name in ATTESTATIONS:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.score(**{name: False})

    def test_unblinded_or_unknown_exposure_is_explicit(self):
        for status, expected in (("unblinded", "exploratory_unblinded"), ("unknown", "development_only")):
            result = self.score(exposure_classification=status)
            self.assertEqual(result["exposure_classification"], status)
            self.assertEqual(result["interpretation"], expected)

    def test_shared_control_validator_accepts_discovery_without_prediction_reference(self):
        from looking_glass.causal_v2_scoring import validate_saved_controls
        row = json.loads(self.args[9][0])["records"][0]
        row["phase"], row["prediction_ref"] = "discovery", None
        outcome = validate_saved_controls(row, target="block.1.branch.0", strength=0.25, polarity=1)
        self.assertEqual(outcome["forward_effects"], (-0.25, -0.25, -0.25))
        self.assertEqual(outcome["reverse_effects"], (0.25, 0.25, 0.25))
        row["plans"][0]["metadata"]["steps"][0]["invocation"] = True
        with self.assertRaises(ValueError):
            validate_saved_controls(row, target="block.1.branch.0", strength=0.25, polarity=1)

    def test_rejects_wrong_salt_incomplete_chunk_and_extra_chunk(self):
        args = list(self.args)
        args[6] = b"x" * 32
        with self.assertRaises(ValueError):
            self.score(args)
        args = list(self.args)
        args[9] = self.args[9][:-1]
        with self.assertRaises(ValueError):
            self.score(args)
        args[9] = self.args[9] + [self.args[9][0]]
        with self.assertRaises(ValueError):
            self.score(args)

    def test_rejects_unsafe_and_casefold_colliding_chunk_names(self):
        for name in ("../escape.json", "DEV-0.JSON"):
            chunks = list(self.args[9])
            chunk = json.loads(chunks[1]); chunk["name"] = name
            chunks[1] = registry._json_bytes(chunk)
            args = list(self.args); args[9] = chunks
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.score(args)

    def test_rejects_trace_run_id_reuse_across_distinct_forwards_or_cases(self):
        for role in ("perturbed", "clean", "repeat_clean"):
            chunks = list(self.args[9]); chunk = json.loads(chunks[0])
            original = chunk["records"][0]
            other = chunk["records"][1 if role == "perturbed" else 2]
            aliased = original["traces"][role][0]["metadata"]["run_id"]
            for event in other["traces"][role]:
                event["metadata"]["run_id"] = aliased
            comparison_role = {"clean": "clean_sparse_trace", "perturbed": "perturbed_sparse_trace"}.get(role)
            for plan in other["plans"]:
                if plan["comparison_role"] == comparison_role:
                    plan["comparison_id"] = f"{aliased}/block.{plan['site']}.output"
                    if plan["source_role"] == comparison_role:
                        plan["source_id"] = plan["comparison_id"]
                if plan["base_role"] == role:
                    plan["base_id"] = aliased + "/logits"
            chunks[0] = registry._json_bytes(chunk)
            args = list(self.args); args[9] = chunks
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.score(args)

    def test_rejects_prediction_alias_and_control_corruption(self):
        for name in ("prediction", "wrong_site", "null", "capture", "target", "comparison", "input",
                     "source_ref", "colluding_source_refs", "plan_metadata", "capture_metadata"):
            chunks = list(self.args[9])
            chunk = json.loads(chunks[0])
            row = chunk["records"][0]
            if name == "prediction": row["prediction_ref"] = row["prediction_ref"].replace(":0:3", ":3:3")
            if name == "wrong_site": row["logits"]["reverse"][0] = row["logits"]["clean"]
            if name == "null": row["logits"]["null"][0] = row["logits"]["clean"]
            if name == "capture": row["plans"][0]["captures"]["post_action"]["bits"][0] = bits(1.0)
            if name == "target": row["private_integrity"]["perturbed_target"]["bits"][0] = bits(0.0)
            if name == "comparison": row["comparison"]["l2_by_block"][1] = 999.0
            if name == "input": row["input_sha256"] = "1" * 64
            if name == "source_ref":
                row["plans"][0]["comparison_id"] = row["plans"][0]["source_id"] = "other/block.1.output"
            if name == "colluding_source_refs":
                for plan in row["plans"]:
                    if plan["comparison_role"] == "clean_sparse_trace":
                        plan["comparison_id"] = f"other/block.{plan['site']}.output"
                        plan["source_id"] = plan["comparison_id"]
                    if plan["base_role"] == "clean": plan["base_id"] = "other/logits"
            if name == "plan_metadata": row["plans"][0]["metadata"]["steps"][0]["invocation"] = 1
            if name == "capture_metadata": row["private_integrity"]["clean_capture_metadata"]["invocation"] = 1
            chunks[0] = registry._json_bytes(chunk)
            args = list(self.args); args[9] = chunks
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.score(args)


if __name__ == "__main__":
    unittest.main()
