"""Fabricated-only evidence binding; no model or study data are generated."""
import copy
import hashlib
import json
import unittest

from looking_glass import registry
from looking_glass.causal_v2_custody import _float32_bits
from looking_glass.causal_v2_freeze import verify_discovery_evidence

import test_causal_v2_evidence_adapter as adapter_fixture
import test_causal_v2_freeze as freeze_fixture


def tensor(values):
    return {"dtype": "torch.float32", "device": "cpu", "shape": [1, len(values)],
            "bits": [_float32_bits(value) for value in values]}


def trace(events, run_id):
    return [{"metadata": {"run_id": run_id, "node_id": event["node_id"],
                          "sequence": event["sequence"], "occurrence": event["occurrence"],
                          "shape": event["shape"], "dtype": event["dtype"],
                          "device": event["device"], "tensor_bytes": 32},
             "value": tensor(event["values"])} for event in events]


def input_digest(values):
    prefix = b"MLG-v2-input-f32be-v1\0" + (1).to_bytes(4, "big") + (8).to_bytes(4, "big")
    return hashlib.sha256(prefix + b"".join(_float32_bits(value).to_bytes(4, "big")
                                            for value in values)).hexdigest()


def selection(block):
    result = freeze_fixture.selection()
    result["first_changed"] = result["magnitude"] = block
    result["first_votes"] = {str(site): 4 if site == block else 0 for site in (1, 2, 3)}
    result["mean_l2"] = {str(site): 0.5 if site >= block else 0.0 for site in (1, 2, 3)}
    for row in result["measurements"]:
        for condition in row["conditions"]:
            value = condition["strength"] * row["polarity"]
            for index, event in enumerate(condition["perturbed_trace"]):
                event["values"][0] = value if index >= block else 0.0
            condition["first_unequal"] = f"block.{block}.output"
            condition["last_equal"] = f"block.{block - 1}.output"
            condition["l2_by_block"] = [abs(value) if index >= block else 0.0 for index in range(4)]
            condition["clean_to_perturbed_logits"] = [[0.0, 0.0] if site >= block else [0.0, value]
                                                        for site in (1, 2, 3)]
            condition["perturbed_to_clean_logits"] = [[0.0, value] if site >= block else [0.0, 0.0]
                                                       for site in (1, 2, 3)]
            condition["effects"] = [-value if site >= block else 0.0 for site in (1, 2, 3)]
    for site in (1, 2, 3):
        for strength in (0.25, 0.75):
            value = strength if site >= block else 0.0
            result["predictions"][str(site)][str(strength)] = {
                "positive": -value if value else 0.0, "negative": value, "constant": 0.0, "zero": 0.0}
    return result


def capture_metadata(path):
    return {"module_path": path, "module_aliases": [path], "call_site_known": False,
            "forward_index": 0, "invocation": 0, "batch_index": None, "token_index": None,
            "output_shape": [1, 8], "selected_shape": [1, 8],
            "dtype": "torch.float32", "device": "cpu"}


def plan_metadata(kind, site):
    path = f"blocks.{site}.output_tap"
    if kind == "self":
        names, operations, capture_bytes, source_bytes = (
            ("self_source", "self_patch", "post_action"), ("capture", "patch", "capture"), 64, 0)
    elif kind == "null":
        names, operations, capture_bytes, source_bytes = (("null", "post_action"), ("null", "capture"), 32, 0)
    else:
        names, operations, capture_bytes, source_bytes = (("patch", "post_action"), ("patch", "capture"), 32, 32)
    steps = []
    for index, (name, operation) in enumerate(zip(names, operations)):
        event = capture_metadata(path)
        event.update(name=name, operation=operation, order=index, output_selector=[],
                     source_capture="self_source" if kind == "self" and operation == "patch" else None,
                     preceding_same_site_actions=list(names[:index]))
        steps.append(event)
    return {"steps": steps, "capture_bytes": capture_bytes, "source_bytes": source_bytes}


class DiscoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        prepared = freeze_fixture.FreezeTests(methodName="runTest")
        prepared.setUp()
        self.snapshot, self.cells = prepared.snapshot, prepared.cells
        public = json.loads(self.snapshot.public)
        schedule = json.loads(self.snapshot.schedule)["assignments"]
        model_cases = public["discovery_cases"]
        for cell, assignment in zip(self.cells, schedule):
            cell["selection"] = selection(int(assignment["injection_node"].split(".")[1]))
        self.chunks = []
        for cell_index, (cell, assignment) in enumerate(zip(self.cells, schedule)):
            records = []
            for case, row in zip(model_cases, cell["selection"]["measurements"]):
                for condition in row["conditions"]:
                    strength, index = condition["strength"], row["case_index"]
                    record = adapter_fixture.record_v2(index, strength)
                    record.update(cell_id=cell["cell_id"], model_seed=cell["model_seed"],
                                  model_state_sha256="c" * 64, case_id=case["case_id"],
                                  input_sha256=input_digest(case["values"]),
                                  condition_id=f"condition-{cell_index}-{index}-{strength}")
                    record["private_integrity"].update(target=assignment["injection_node"],
                                                       delta=strength * row["polarity"])
                    record["traces"] = {"clean": trace(row["clean_trace"], f"clean-{index}"),
                        "repeat_clean": trace(row["repeated_clean_trace"], f"repeat-{index}"),
                        "perturbed": trace(condition["perturbed_trace"], f"perturbed-{cell_index}-{index}-{strength}")}
                    record["logits"] = {"clean": tensor(row["clean_logits"]),
                        "repeat_clean": tensor(row["repeated_clean_logits"]),
                        "perturbed": tensor(condition["perturbed_logits"]),
                        "forward": [tensor(x) for x in condition["clean_to_perturbed_logits"]],
                        "reverse": [tensor(x) for x in condition["perturbed_to_clean_logits"]],
                        "null": [tensor(x) for x in condition["null_logits"]],
                        "self": [tensor(x) for x in condition["identity_logits"]],
                        "upstream": tensor(condition["upstream_logits"]),
                        "post_hook_clean": tensor(condition["clean_verify_logits"])}
                    record["comparison"] = {"status": "difference_observed",
                        "first_unequal": condition["first_unequal"],
                        "last_equal": condition["last_equal"], "l2_by_block": condition["l2_by_block"]}
                    parts = assignment["injection_node"].split(".")
                    block, op = parts[1], parts[2]
                    path = (f"blocks.{block}.branch_taps.0" if op == "branch" else f"blocks.{block}.{op}_tap")
                    private = record["private_integrity"]
                    private["clean_target"] = tensor([0.0] * 8)
                    private["perturbed_target"] = tensor([strength * row["polarity"]] + [0.0] * 7)
                    private["clean_capture_metadata"] = capture_metadata(path)
                    private["perturbed_capture_metadata"] = capture_metadata(path)
                    for plan in record["plans"]:
                        kind, site = plan["kind"], plan["site"]
                        plan["metadata"] = plan_metadata(kind, site)
                        refs = {"clean_sparse_trace": f"clean-{index}",
                                "perturbed_sparse_trace": f"perturbed-{cell_index}-{index}-{strength}"}
                        plan["comparison_id"] = f"{refs[plan['comparison_role']]}/block.{site}.output"
                        if kind not in ("null", "self"):
                            plan["source_id"] = f"{refs[plan['source_role']]}/block.{site}.output"
                        base = refs["clean_sparse_trace" if plan["base_role"] == "clean"
                                    else "perturbed_sparse_trace"]
                        plan["base_id"] = f"{base}/logits"
                        source = (record["traces"]["perturbed"] if kind in ("reverse", "null", "self")
                                  else record["traces"]["clean"])[site]["value"]
                        plan["captures"]["post_action"] = copy.deepcopy(source)
                        if kind == "self":
                            plan["captures"]["self_source"] = copy.deepcopy(source)
                    records.append(record)
            self.chunks.append({"name": f"cell-{cell_index:02d}-d.json", "records": records,
                                "schema": "causal-v2-chunk-v1", "state": "complete"})

    def encoded(self, chunks=None):
        return [registry._json_bytes(chunk) for chunk in (self.chunks if chunks is None else chunks)]

    def test_matches_all_ordered_saved_records(self):
        self.assertIsNone(verify_discovery_evidence(self.snapshot, self.cells, iter(self.encoded())))

    def test_mismatched_saved_traces_logits_controls_and_comparison_fail(self):
        mutations = (
            lambda record: record["traces"]["clean"][1]["value"]["bits"].__setitem__(0, 1),
            lambda record: record["traces"]["repeat_clean"][1]["value"]["bits"].__setitem__(0, 1),
            lambda record: record["traces"]["perturbed"][1]["value"]["bits"].__setitem__(0, 1),
            *[lambda record, key=key: record["logits"][key]["bits"].__setitem__(0, 1)
              for key in ("clean", "repeat_clean", "perturbed", "upstream", "post_hook_clean")],
            *[lambda record, key=key: record["logits"][key][1]["bits"].__setitem__(0, 1)
              for key in ("forward", "reverse", "null", "self")],
            lambda record: record["comparison"].update(first_unequal=None),
        )
        for index, change in enumerate(mutations):
            chunks = copy.deepcopy(self.chunks)
            change(chunks[0]["records"][0])
            with self.subTest(index=index), self.assertRaises(ValueError):
                verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))

    def test_signed_zero_equivalence_and_subnormal_difference(self):
        chunks = copy.deepcopy(self.chunks)
        record = chunks[0]["records"][0]
        record["traces"]["clean"][1]["value"]["bits"][0] = 0x80000000
        record["logits"]["clean"]["bits"][0] = 0x80000000
        verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))
        record["logits"]["clean"]["bits"][0] = 1
        with self.assertRaises(ValueError):
            verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))

    def test_wrong_identity_order_pairing_or_private_target_fails_without_leak(self):
        changes = (
            lambda record: record.update(case_id="wrong"),
            lambda record: record.update(input_sha256="e" * 64),
            lambda record: record.update(model_state_sha256="f" * 64),
            lambda record: record["private_integrity"].update(target="fake-private-target"),
            lambda record: record["private_integrity"].update(delta=-0.25),
            lambda record: [event["metadata"].update(run_id="clean-0")
                            for event in record["traces"]["perturbed"]],
        )
        for index, change in enumerate(changes):
            chunks = copy.deepcopy(self.chunks)
            change(chunks[0]["records"][0])
            with self.subTest(index=index):
                with self.assertRaises(ValueError) as raised:
                    verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))
                self.assertNotIn("fake-private-target", str(raised.exception))

    def test_invalid_private_injection_plan_and_cross_strength_capture_fail(self):
        def unchanged_target(record):
            record["private_integrity"]["perturbed_target"] = copy.deepcopy(
                record["private_integrity"]["clean_target"])

        def wrong_other_target_coordinate(record):
            record["private_integrity"]["perturbed_target"]["bits"][1] = 0x3f800000

        def bad_capture(record):
            record["plans"][0]["captures"]["post_action"]["bits"][0] = 0x3f800000

        def bad_plan(record):
            record["plans"][0]["metadata"]["steps"][0]["module_path"] = "private-fake-path"

        def bad_target_metadata(record):
            record["private_integrity"]["clean_capture_metadata"]["invocation"] = 1

        def coordinated_wrong_trace_references(record):
            for plan in record["plans"]:
                for key in ("source_id", "comparison_id", "base_id"):
                    if isinstance(plan[key], str) and plan[key].startswith("clean-0/"):
                        plan[key] = plan[key].replace("clean-0/", "unrelated-forward/", 1)

        for index, change in enumerate((unchanged_target, wrong_other_target_coordinate, bad_capture,
                                        bad_plan, bad_target_metadata, coordinated_wrong_trace_references)):
            chunks = copy.deepcopy(self.chunks)
            change(chunks[0]["records"][0])
            with self.subTest(index=index):
                with self.assertRaises(ValueError) as raised:
                    verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))
                self.assertNotIn("private-fake-path", str(raised.exception))

        chunks = copy.deepcopy(self.chunks)
        second = chunks[0]["records"][1]["private_integrity"]
        second["clean_target"]["bits"][0] = _float32_bits(1.0)
        second["perturbed_target"]["bits"][0] = _float32_bits(1.75)
        with self.assertRaisesRegex(ValueError, "reused private target"):
            verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))

    def test_hidden_target_rejects_consistent_but_upstream_changed_evidence(self):
        schedule = json.loads(self.snapshot.schedule)["assignments"]
        early = next(i for i, item in enumerate(schedule) if ".1." in item["injection_node"])
        later = next(i for i, item in enumerate(schedule) if ".3." in item["injection_node"])
        cells, chunks = copy.deepcopy(self.cells), copy.deepcopy(self.chunks)
        cells[later]["selection"] = copy.deepcopy(cells[early]["selection"])
        for saved, donor in zip(chunks[later]["records"], chunks[early]["records"]):
            old_id = donor["traces"]["perturbed"][0]["metadata"]["run_id"]
            new_id = saved["traces"]["perturbed"][0]["metadata"]["run_id"]
            for key in ("traces", "logits", "comparison", "plans"):
                saved[key] = copy.deepcopy(donor[key])
            for event in saved["traces"]["perturbed"]:
                event["metadata"]["run_id"] = new_id
            for plan in saved["plans"]:
                for field in ("source_id", "comparison_id", "base_id"):
                    if isinstance(plan[field], str) and plan[field].startswith(old_id + "/"):
                        plan[field] = new_id + plan[field][len(old_id):]
        with self.assertRaisesRegex(ValueError, "upstream"):
            verify_discovery_evidence(self.snapshot, cells, self.encoded(chunks))

    def test_condition_ids_are_unique_across_cells(self):
        chunks = copy.deepcopy(self.chunks)
        chunks[1]["records"][0]["condition_id"] = chunks[0]["records"][0]["condition_id"]
        with self.assertRaises(ValueError):
            verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))

    def test_trace_ids_cannot_alias_a_different_input_or_perturbed_forward(self):
        for role in ("clean", "perturbed"):
            chunks = copy.deepcopy(self.chunks)
            first = chunks[0]["records"][0]
            second = chunks[0]["records"][2]
            old_id = second["traces"][role][0]["metadata"]["run_id"]
            new_id = first["traces"][role][0]["metadata"]["run_id"]
            for event in second["traces"][role]:
                event["metadata"]["run_id"] = new_id
            for plan in second["plans"]:
                for field in ("source_id", "comparison_id", "base_id"):
                    if isinstance(plan[field], str) and plan[field].startswith(old_id + "/"):
                        plan[field] = new_id + plan[field][len(old_id):]
            with self.subTest(role=role), self.assertRaises(ValueError):
                verify_discovery_evidence(self.snapshot, self.cells, self.encoded(chunks))

    def test_missing_extra_incomplete_wrong_order_or_noncanonical_chunks_fail(self):
        raw = self.encoded()
        variants = [raw[:-1], raw + raw[:1], list(reversed(raw)), [raw[0] + b" "] + raw[1:]]
        incomplete = copy.deepcopy(self.chunks)
        incomplete[0]["state"] = "incomplete"
        variants.append(self.encoded(incomplete))
        for index, chunks in enumerate(variants):
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                verify_discovery_evidence(self.snapshot, self.cells, chunks)


if __name__ == "__main__":
    unittest.main()
