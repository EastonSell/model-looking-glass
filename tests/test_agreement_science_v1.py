"""Model-free tests for the frozen agreement-direction scoring rules."""
import math
import struct
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import hashlib
import tempfile
from pathlib import Path
from looking_glass import agreement_science_saved_v1 as saved
from tools import agreement_direction_science_replay_v1 as cli
from collections import Counter

from looking_glass import agreement_science_v1 as science
from looking_glass.agreement_science_v1 import scalar_response


class DirectionResponseTests(unittest.TestCase):
    def test_real_runtime_contract_rejects_contradictory_recorded_settings(self):
        environment = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None}
        numeric = [['deterministic_algorithms', True], ['deterministic_warn_only', False],
                   ['cpu_threads', 1], ['cpu_interop_threads', 1],
                   ['float32_matmul_precision', 'highest'], ['calling_thread_denormal_product_bits', 4194304]]
        modules = [['', [], [['use_cache', ['bool', False]], ['_attn_implementation', ['str', 'eager']],
                             ['n_layer', ['int', 6]], ['n_embd', ['int', 768]], ['vocab_size', ['int', 50257]]]]]
        contract = {'kind': 'run_contract', 'expected_forwards': 784, 'seed': 20261030,
                    'environment': environment, 'execution_configuration': [numeric, modules],
                    'model_state_sha256': 'a' * 64, 'external_binding': {}}
        self.assertTrue(saved.validate_recorded_runtime(contract))
        for bad in (
            {**contract, 'environment': {**environment, 'CUDA_VISIBLE_DEVICES': '0'}},
            {**contract, 'environment': {key: value for key, value in environment.items() if key != 'MKL_CBWR'}},
            {**contract, 'execution_configuration': [[["cpu_threads", 1]], modules]},
            {**contract, 'execution_configuration': [[*numeric[:2], ['cpu_threads', True], *numeric[3:]], modules]},
            {**contract, 'execution_configuration': [numeric, [['', [], [['use_cache', ['bool', True]]]]]]},
            {**contract, 'execution_configuration': [numeric, [['', [], [field if field[0] != 'n_layer'
                                                                          else ['n_layer', ['int', 5]]
                                                                          for field in modules[0][2]]]]]},
            {**contract, 'execution_configuration': [numeric, [['', [], [field for field in modules[0][2]
                                                                          if field[0] != 'n_layer']]]]},
            {**contract, 'unexpected': True},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                saved.validate_recorded_runtime(bad)
        # The public-width audit must apply the same check, before reading any tensor.
        one_hot = lambda at: tuple(float(index == at) for index in range(768))
        directions = {'u': one_hot(0), 'v1': one_hot(1), 'v2': one_hot(2)}
        with patch.object(science, 'validate_journal_layout'):
            with self.assertRaises(ValueError):
                saved.audit_records([{'event': {**contract, 'environment': {}}, 'tensors': {}}],
                                    [], '.', directions, expected_binding={})

    def test_saved_directory_rejects_reparse_points_at_root_and_tensor_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'attempt'
            tensors = root / 'tensors'
            tensors.mkdir(parents=True)
            for name in ('started.json', 'events.jsonl', 'terminal.json'):
                (root / name).write_bytes(b'{}')
            real_lstat = Path.lstat
            for reparse in (root, tensors):
                def lstat(path, *args, **kwargs):
                    info = real_lstat(path, *args, **kwargs)
                    return (SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
                            if path == reparse else info)
                with self.subTest(reparse=reparse), patch.object(Path, 'lstat', lstat):
                    with self.assertRaisesRegex(ValueError, 'directory'):
                        saved.audit_attempt(root, [], {}, expected_binding=None, expected_source_revision=None,
                                            width=3, vocabulary=548, allow_synthetic=True)

    def test_report_publication_is_exclusive_and_rejects_nonfinite_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            cli.write_report(path, {"complete_candidate": True, "value": 1.0})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                cli.write_report(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), original)
            invalid = Path(temporary) / "invalid.json"
            with self.assertRaises(ValueError):
                cli.write_report(invalid, {"value": float("nan")})
            self.assertFalse(invalid.exists())

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        self.assertEqual(saved.strict_json(b'{"a": 1, "b": -0.0}')["a"], 1)
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                saved.strict_json(raw)

    def test_terminal_accounting_rejects_candidate_or_storage_mismatch(self):
        state = "a" * 64
        terminal = {"status": "complete_candidate", "events": 809, "event_log_bytes": 123,
                    "retained_tensor_bytes": 8, "storage_accounting_complete": True,
                    "detail": {"status": "complete_candidate", "forwards": 784,
                               "model_state_sha256_before_after": state, "retained_tensor_bytes": 8}}
        self.assertTrue(saved.validate_terminal(terminal, state, event_bytes=123, tensor_bytes=8))
        for altered in ({**terminal, "status": "failed"}, {**terminal, "events": 808},
                        {**terminal, "storage_accounting_complete": False},
                        {**terminal, "retained_tensor_bytes": 4},
                        {**terminal, "detail": {**terminal["detail"], "model_state_sha256_before_after": "b" * 64}}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                saved.validate_terminal(altered, state, event_bytes=123, tensor_bytes=8)

    def test_independent_source_pin_rejects_changed_or_missing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.bin"
            path.write_bytes(b"frozen")
            digest = hashlib.sha256(b"frozen").hexdigest()
            self.assertEqual(saved.pinned_file(path, digest, limit=100), b"frozen")
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                saved.pinned_file(path, digest, limit=100)
            with self.assertRaises(ValueError):
                saved.pinned_file(path, "not-a-digest", limit=100)

    def test_odd_and_even_responses_do_not_conflate_signed_and_common_changes(self):
        self.assertIsNotNone(scalar_response)
        # The + arm moves the margin by 2 and the - arm leaves it unchanged.
        result = scalar_response(clean=3.0, plus=5.0, minus=3.0, q=2.0)
        self.assertEqual(result["plus_change"], 2.0)
        self.assertEqual(result["minus_change"], 0.0)
        self.assertEqual(result["odd"], 1.0)
        self.assertEqual(result["even"], 1.0)
        self.assertEqual(result["odd_per_unit"], 0.5)

    def test_zero_dose_keeps_the_observation_and_marks_slope_undefined(self):
        result = scalar_response(clean=3.0, plus=3.0, minus=3.0, q=0.0)
        self.assertEqual(result["odd"], 0.0)
        self.assertEqual(result["even"], 0.0)
        self.assertIsNone(result["odd_per_unit"])

    def test_invalid_scalar_or_negative_dose_cannot_be_scored(self):
        for values in (
            {"clean": float("nan"), "plus": 1.0, "minus": 0.0, "q": 1.0},
            {"clean": 0.0, "plus": float("inf"), "minus": 0.0, "q": 1.0},
            {"clean": 0.0, "plus": 1.0, "minus": 0.0, "q": -1.0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                scalar_response(**values)

    def test_overlarge_integer_is_rejected_as_an_invalid_number(self):
        with self.assertRaises(ValueError):
            scalar_response(clean=10**1000, plus=0.0, minus=0.0, q=1.0)

    def test_zero_dose_cannot_hide_a_changed_margin(self):
        with self.assertRaises(ValueError):
            scalar_response(clean=1.0, plus=2.0, minus=1.0, q=0.0)

    def test_projection_loss_keeps_negative_values(self):
        self.assertTrue(callable(getattr(science, "projection_loss", None)))
        self.assertEqual(science.projection_loss(4.0, -3.0, 2.0, -1.0),
                         {"clean_gap": 7.0, "arm_gap": 3.0, "loss": 4.0})
        self.assertEqual(science.projection_loss(4.0, -3.0, 5.0, -4.0)["loss"], -2.0)

    def test_donor_weakening_has_the_same_orientation_in_both_number_classes(self):
        self.assertTrue(callable(getattr(science, "donor_weakening", None)))
        self.assertEqual(science.donor_weakening(3.0, 1.0, "singular"), 2.0)
        self.assertEqual(science.donor_weakening(-3.0, -1.0, "plural"), 2.0)
        self.assertEqual(science.donor_weakening(-3.0, -5.0, "plural"), -2.0)

    def test_four_fixed_margins_use_all_finite_vocabulary_logits(self):
        logits = [0.0] * 50257
        logits[318], logits[389] = 2.0, -1.0
        logits[373], logits[547] = -3.0, 1.0
        logits[262], logits[257] = 2.0, 0.5
        logits[287], logits[319] = -1.0, 2.0
        self.assertEqual(science.fixed_margins(logits),
                         {"present": 3.0, "past": -4.0,
                          "the_vs_a": 1.5, "in_vs_on": -3.0})
        with self.assertRaises(ValueError):
            science.fixed_margins(logits[:-1])
        logits[10000] = float("nan")
        with self.assertRaises(ValueError):
            science.fixed_margins(logits)

    def test_donor_contrasts_keep_number_orientation_and_nonwinning_cases(self):
        for sign, number in ((1, "singular"), (-1, "plural")):
            result = science.donor_comparisons(
                clean=sign * 2.0, same_diff=sign * 1.5,
                opposite_same=sign * -0.5, opposite_diff=sign * 0.5, number=number)
            self.assertEqual(result["weakening"],
                             {"same_diff": 0.5, "opposite_same": 2.5, "opposite_diff": 1.5})
            self.assertEqual(result["absolute_change"],
                             {"same_diff": 0.5, "opposite_same": 2.5, "opposite_diff": 1.5})
            self.assertEqual(result["contrasts"], {"opposite_same_minus_same_diff": 2.0,
                                                   "opposite_diff_minus_same_diff": 1.0})
            self.assertTrue(result["clean_correct"])
            self.assertTrue(result["margin_reversal"]["opposite_same"])
            self.assertFalse(result["margin_reversal"]["opposite_diff"])
        incorrect = science.donor_comparisons(clean=-1.0, same_diff=-2.0,
                      opposite_same=0.0, opposite_diff=1.0, number="singular")
        self.assertFalse(incorrect["clean_correct"])
        self.assertTrue(incorrect["margin_reversal"]["opposite_diff"])
        self.assertFalse(incorrect["margin_reversal"]["opposite_same"])

    def test_direction_rule_uses_absolute_controls_and_all_families(self):
        self.assertTrue(callable(getattr(science, "evaluate_signed", None)))
        rows = [{"family": f, "variant": v, "u": 0.125, "v1": 0.03125,
                 "v2": -0.03125} for f in range(8) for v in range(4)]
        result = science.evaluate_signed(rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["families_passed"], 8)
        self.assertEqual(result["classes"]["singular"]["n"], 16)
        # Opposite-signed controls have a zero mean but large mean absolute value.
        for row in rows:
            row["v1"] = 0.25 if row["variant"] < 2 else -0.25
        result = science.evaluate_signed(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["classes"]["singular"]["mean_abs_v1"], 0.25)
        # Two zero-dose families remain in the denominator and meet 6/8 exactly.
        for row in rows:
            row["v1"] = 0.03125
            if row["family"] < 2:
                row["u"] = row["v1"] = row["v2"] = 0.0
        result = science.evaluate_signed(rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["families_passed"], 6)
        self.assertEqual(result["classes"]["plural"]["n"], 16)

    def test_projection_rule_counts_all_sixteen_pairs(self):
        self.assertTrue(callable(getattr(science, "evaluate_projection", None)))
        pairs = [{"family": f, "morphology": kind, "u": 0.25,
                  "v1": 0.03125, "v2": -0.03125}
                 for f in range(8) for kind in ("regular", "irregular")]
        for pair in pairs[:4]:
            pair["u"] = pair["v1"] = pair["v2"] = 0.0
        result = science.evaluate_projection(pairs)
        self.assertTrue(result["passed"])
        self.assertEqual(result["pairs_passed"], 12)
        self.assertEqual(result["n"], 16)
        pairs[4]["u"] = pairs[4]["v1"] = pairs[4]["v2"] = 0.0
        self.assertFalse(science.evaluate_projection(pairs)["passed"])

    def test_floor_includes_equality_but_control_ties_do_not_pass(self):
        signed = [{"family": f, "variant": v, "u": 0.05, "v1": 0.0, "v2": 0.0}
                  for f in range(8) for v in range(4)]
        self.assertTrue(science.evaluate_signed(signed)["passed"])
        for row in signed:
            row["v1"] = 0.05
        self.assertFalse(science.evaluate_signed(signed)["passed"])
        pairs = [{"family": f, "morphology": k, "u": 0.10, "v1": 0.0, "v2": 0.0}
                 for f in range(8) for k in ("regular", "irregular")]
        self.assertTrue(science.evaluate_projection(pairs)["passed"])
        for row in pairs:
            row["v2"] = -0.10
        self.assertFalse(science.evaluate_projection(pairs)["passed"])

    def test_generic_linear_output_bias_can_satisfy_both_predictions(self):
        # m(x) = x + 4 is a generic output bias on a coordinate, not a
        # demonstration that the coordinate is a grammatical-number feature.
        margin = lambda x: x + 4.0
        responses = []
        projections = []
        for family in range(8):
            for variant in range(4):
                x = 1.0 if variant in (0, 2) else -1.0
                response = scalar_response(clean=margin(x), plus=margin(x + 0.25),
                                           minus=margin(x - 0.25), q=0.25)
                responses.append({"family": family, "variant": variant,
                                  "u": response["odd"], "v1": 0.0, "v2": 0.0})
            loss = science.projection_loss(margin(1.0), margin(-1.0), margin(0.0), margin(0.0))["loss"]
            for kind in ("regular", "irregular"):
                projections.append({"family": family, "morphology": kind,
                                    "u": loss, "v1": 0.0, "v2": 0.0})
        self.assertTrue(science.evaluate_signed(responses)["passed"])
        self.assertTrue(science.evaluate_projection(projections)["passed"])

    def test_float32_bit_decoder_retains_signed_zero_and_subnormals(self):
        self.assertTrue(callable(getattr(science, "decode_float32", None)))
        raw = struct.pack("<6I", 0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000)
        values = science.decode_float32(raw, expected_count=6)
        self.assertEqual(math.copysign(1, values[0]), 1)
        self.assertEqual(math.copysign(1, values[1]), -1)
        self.assertEqual(values[2:], (2.0**-149, -(2.0**-149), 1.0, -1.0))
        for raw in (b"\x00", struct.pack("<I", 0x7f800000), struct.pack("<I", 0x7fc00000)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                science.decode_float32(raw, expected_count=1)

    def test_integer_rounding_to_float32_uses_even_midpoints_and_subnormals(self):
        self.assertEqual(science.float32_bits(1.0 + 2.0**-24), 0x3f800000)
        self.assertEqual(science.float32_bits(1.0 + 3 * 2.0**-24), 0x3f800002)
        self.assertEqual(science.float32_bits(2.0**-150), 0)
        self.assertEqual(science.float32_bits(3 * 2.0**-150), 2)
        self.assertEqual(science.float32_bits(-0.0), 0x80000000)
        self.assertEqual(science.float32_bits(-2.0**-150), 0x80000000)
        with self.assertRaises(ValueError):
            science.float32_bits(2.0**128 - 2.0**103)
        with self.assertRaises(ValueError):
            science.float32_bits(float("nan"))

    def test_geometry_uses_both_members_six_donors_and_signed_centering(self):
        rows = [struct.pack("<2f", *xy) for xy in ((0, 0), (4, 0), (0, 6), (4, 6))]
        geometry = science.family_geometry(rows, (1.0, 0.0))
        self.assertEqual(geometry["distances"][0], [6.0, 4.0, math.sqrt(52)])
        self.assertEqual(geometry["doses"], [2.0] * 4)
        self.assertEqual(geometry["centering_gaps"], [-4.0, -4.0])
        self.assertEqual(geometry["centering"], [-2.0, 2.0, -2.0, 2.0])
        tiny = [struct.pack("<2f", x, 0) for x in (0.0, 2**-20, 2**-19, 3 * 2**-20)]
        geometry = science.family_geometry(tiny, (1.0, 0.0))
        self.assertEqual(geometry["doses"], [0.0] * 4)
        self.assertEqual(geometry["centering"], [0.0] * 4)

    def test_delivery_recomputes_donor_direction_and_zero_bit_custody(self):
        base = struct.pack("<2f", 1.0, 2.0)
        donor = struct.pack("<2f", 5.0, 5.0)
        delivered, meta = science.donor_delivery(base, donor, 2.5)
        self.assertEqual(struct.unpack("<2f", delivered), (3.0, 3.5))
        self.assertEqual(meta["donor_distance"], 5.0)
        self.assertEqual(meta["fraction"], 0.5)
        self.assertEqual(meta["realized_norm"], 2.5)
        self.assertTrue(meta["dose_valid"])
        moved, info = science.direction_delivery(base, (1.0, 0.0), -2.0)
        self.assertEqual(struct.unpack("<2f", moved), (-1.0, 2.0))
        self.assertEqual(info["signed_amount"], -2.0)
        self.assertTrue(info["dose_valid"])
        signed_zero = struct.pack("<2I", 0x80000000, 0)
        self.assertEqual(science.direction_delivery(signed_zero, (1.0, 0.0), 0.0)[0], signed_zero)
        self.assertEqual(science.donor_delivery(signed_zero, signed_zero, 0.0)[0], signed_zero)
        with self.assertRaises(ValueError):
            science.donor_delivery(base, donor, 5.5)

    def test_hook_witness_binds_the_selected_slice_and_unchanged_other_positions(self):
        before = struct.pack("<2f", 20.0, 21.0)
        after = struct.pack("<2f", 25.0, 26.0)
        whole_before = struct.pack("<6f", 10.0, 11.0, 20.0, 21.0, 30.0, 31.0)
        whole_after = struct.pack("<6f", 10.0, 11.0, 25.0, 26.0, 30.0, 31.0)
        self.assertTrue(science.verify_hook_witness(before, after, whole_before, whole_after,
                                                    width=2, sequence_length=3, token_index=1))
        tampered = struct.pack("<6f", 9.0, 11.0, 25.0, 26.0, 30.0, 31.0)
        with self.assertRaises(ValueError):
            science.verify_hook_witness(before, after, whole_before, tampered,
                                        width=2, sequence_length=3, token_index=1)
        with self.assertRaises(ValueError):
            science.verify_hook_witness(after, after, whole_before, whole_after,
                                        width=2, sequence_length=3, token_index=1)

    def test_hook_plan_names_selected_position_and_exact_capture_budget(self):
        site, plan = science.expected_hook_plan("subject_u_plus", sequence_length=5, width=8)
        self.assertEqual(site["module_path"], "transformer.h.3.attn")
        self.assertEqual(site["token_role"], "subject")
        self.assertEqual(plan["capture_bytes"], 384)
        self.assertEqual(plan["source_bytes"], 32)
        self.assertEqual([row["name"] for row in plan["steps"]],
                         ["whole_before", "before", "action", "after", "whole_after"])
        self.assertEqual(plan["steps"][2]["operation"], "patch")
        self.assertEqual(plan["steps"][2]["token_index"], 1)
        site, noop = science.expected_hook_plan("noop_l1_final", sequence_length=5, width=8)
        self.assertEqual(site["module_path"], "transformer.h.1.attn")
        self.assertEqual(noop["source_bytes"], 0)
        self.assertEqual(noop["steps"][2]["operation"], "null")
        self.assertEqual(noop["steps"][2]["token_index"], 4)

    def test_saved_tensor_reference_is_bounded_shape_checked_and_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = struct.pack("<2I", 0x3f800000, 0x80000000)
            digest = hashlib.sha256(raw).hexdigest()
            path = directory / (digest + ".f32le")
            path.write_bytes(raw)
            reference = {"sha256": digest, "shape": [2], "dtype": "float32-le", "bytes": 8}
            self.assertEqual(science.read_saved_tensor(directory, reference, (2,)), raw)
            for altered in ({**reference, "shape": [1, 2]}, {**reference, "bytes": True},
                            {**reference, "sha256": "../not-a-digest"},
                            {**reference, "dtype": "float64-le"}):
                with self.subTest(altered=altered), self.assertRaises(ValueError):
                    science.read_saved_tensor(directory, altered, (2,))
            path.write_bytes(raw + b"x")
            with self.assertRaises(ValueError):
                science.read_saved_tensor(directory, reference, (2,))
            path.write_bytes(struct.pack("<2I", 0x40000000, 0x80000000))
            with self.assertRaises(ValueError):
                science.read_saved_tensor(directory, reference, (2,))

    def test_exact_metadata_comparison_does_not_conflate_boolean_or_zero_sign(self):
        self.assertTrue(science.same_recorded_value({"x": [1, 1.0, -0.0]}, {"x": [1, 1.0, -0.0]}))
        self.assertFalse(science.same_recorded_value({"x": 1}, {"x": True}))
        self.assertFalse(science.same_recorded_value({"x": 1}, {"x": 1.0}))
        self.assertFalse(science.same_recorded_value({"x": -0.0}, {"x": 0.0}))
        self.assertFalse(science.same_recorded_value({"x": float("nan")}, {"x": float("nan")}))

    def test_fixed_schedule_covers_every_arm_slot_and_both_number_classes(self):
        cases = [{"case_id": f"t{f}-{v}", "role": "target", "family_index": f,
                  "variant": v, "subject_number": "singular" if v % 2 == 0 else "plural"}
                 for f in range(8) for v in range(4)]
        cases += [{"case_id": f"c{f}", "role": "context", "family_index": f}
                  for f in range(8)]
        schedule = science.expected_schedule(cases)
        self.assertEqual(len(schedule), 784)
        self.assertEqual([row["index"] for row in schedule], list(range(784)))
        arms = [row for row in schedule if row["role"] == "target" and row["slot"] is not None]
        self.assertEqual(len(arms), 32 * 16)
        for number in ("singular", "plural"):
            counts = Counter((row["action"], row["slot"]) for row in arms if row["subject_number"] == number)
            self.assertEqual(len(counts), 16 * 16)
            self.assertEqual(set(counts.values()), {1})
        self.assertEqual(schedule[20]["action"], "same_diff")
        damaged = list(cases)
        damaged[1] = damaged[0]
        with self.assertRaises(ValueError):
            science.expected_schedule(damaged)

    def test_forward_inventory_rejects_missing_reordered_or_forged_inputs(self):
        cases = [{"case_id": f"t{f}-{v}", "role": "target", "family_index": f,
                  "variant": v, "subject_number": "singular" if v % 2 == 0 else "plural",
                  "token_ids": [464, 10 + v, 262, 30, 40]}
                 for f in range(8) for v in range(4)]
        cases += [{"case_id": f"c{f}", "role": "context", "family_index": f,
                   "token_ids": [464, 262, 30]} for f in range(8)]
        by_id = {case["case_id"]: case for case in cases}
        events = [{**action, "kind": "forward", "input_ids": by_id[action["case_id"]]["token_ids"][:],
                   "attention_mask": [1] * len(by_id[action["case_id"]]["token_ids"]),
                   "all_sequence_logits_finite": True}
                  for action in science.expected_schedule(cases)]
        self.assertEqual(science.validate_forward_inventory(events, cases), 784)
        with self.assertRaises(ValueError):
            science.validate_forward_inventory(events[:-1], cases)
        altered = [dict(event) for event in events]
        altered[20], altered[21] = altered[21], altered[20]
        with self.assertRaises(ValueError):
            science.validate_forward_inventory(altered, cases)
        altered = [dict(event) for event in events]
        altered[1]["index"] = True  # bool must not alias the integer 1.
        with self.assertRaises(ValueError):
            science.validate_forward_inventory(altered, cases)
        altered = [dict(event) for event in events]
        altered[0]["input_ids"] = [464, True, 262, 30, 40]
        with self.assertRaises(ValueError):
            science.validate_forward_inventory(altered, cases)
        altered = [dict(event) for event in events]
        altered[-1]["all_sequence_logits_finite"] = False
        with self.assertRaises(ValueError):
            science.validate_forward_inventory(altered, cases)

    def test_journal_layout_requires_all_interleaved_geometry_and_payloads(self):
        cases = [{"case_id": f"t{f}-{v}", "role": "target", "family_index": f,
                  "variant": v, "subject_number": "singular" if v % 2 == 0 else "plural",
                  "token_ids": [464, 10 + v, 262, 30, 40]} for f in range(8) for v in range(4)]
        cases += [{"case_id": f"c{f}", "role": "context", "family_index": f,
                   "token_ids": [464, 262, 30]} for f in range(8)]
        by_id = {case["case_id"]: case for case in cases}
        schedule = science.expected_schedule(cases)
        records = []
        def add(event, tensors=None):
            records.append({"index": len(records), "event": event, "tensors": tensors or {}})
        def add_forward(item):
            tokens = by_id[item["case_id"]]["token_ids"]
            event = {**item, "kind": "forward", "input_ids": tokens, "attention_mask": [1]*len(tokens),
                     "all_sequence_logits_finite": True}
            hooked = item["action"] not in ("clean", "clean_repeat", "clean_after")
            tensors = {name: {} for name in (("last_logits", "whole_before", "before", "after", "whole_after")
                                              if hooked else ("last_logits",))}
            add(event, tensors)
        add({"kind": "run_contract", "expected_forwards": 784, "seed": 20261030})
        for family in range(8):
            group = schedule[family*98:(family+1)*98]
            for item in group[:20]: add_forward(item)
            add({"kind": "family_geometry", "family_index": family})
            for item in group[20:88]: add_forward(item)
            for variant in (0, 2):
                add({"kind": "centering_result", "family_index": family, "singular_variant": variant})
            for item in group[88:]: add_forward(item)
        self.assertEqual(science.validate_journal_layout(records, cases), 784)
        with self.assertRaises(ValueError):
            science.validate_journal_layout(records[:-1], cases)
        bad = [dict(row) for row in records]
        bad[21] = {**bad[21], "event": {**bad[21]["event"], "family_index": 3}}
        with self.assertRaises(ValueError):
            science.validate_journal_layout(bad, cases)
        bad = [dict(row) for row in records]
        bad[3] = {**bad[3], "tensors": {"last_logits": {}}}
        with self.assertRaises(ValueError):
            science.validate_journal_layout(bad, cases)

    def test_full_study_reductions_keep_context_and_all_fixed_cases(self):
        cases = [{"case_id": f"t{f}-{v}", "role": "target", "family_index": f,
                  "variant": v, "subject_number": "singular" if v % 2 == 0 else "plural",
                  "morphology": "regular" if v < 2 else "irregular",
                  "attractor_number": "singular" if f % 2 == 0 else "plural"}
                 for f in range(8) for v in range(4)]
        cases += [{"case_id": f"c{f}", "role": "context", "family_index": f}
                  for f in range(8)]
        lookup = {case["case_id"]: case for case in cases}
        scores = {}
        for entry in science.expected_schedule(cases):
            case = lookup[entry["case_id"]]
            action = entry["action"]
            x = 0 if case["role"] == "context" else (1 if case["variant"] % 2 == 0 else -1)
            value = float(x + 4)
            if action == "u_plus":
                value += 0.25
            elif action == "u_minus":
                value -= 0.25
            elif action == "center_u":
                value = 4.0
            scores[(case["case_id"], action)] = {"present": value, "past": 0.0,
                                                   "the_vs_a": 0.0, "in_vs_on": 0.0}
        doses = {case["case_id"]: 0.25 for case in cases}
        report = science.evaluate_study(cases, scores, doses)
        self.assertEqual(len(report["targets"]), 32)
        self.assertEqual(len(report["contexts"]), 8)
        self.assertEqual(report["signed_prediction"]["families_passed"], 8)
        self.assertTrue(report["signed_prediction"]["passed"])
        self.assertEqual(report["projection_prediction"]["pairs_passed"], 16)
        self.assertTrue(report["projection_prediction"]["passed"])
        self.assertEqual(report["contexts"][0]["responses"]["u"]["present"]["odd"], 0.25)
        self.assertEqual(report["groups"]["subject_number"]["singular"]["donors"]["n"], 16)
        self.assertEqual(report["groups"]["subject_number"]["plural"]["donors"]["n"], 16)
        self.assertIn("opp_same", report["targets"][0]["arm_margins"])
        bad = dict(scores)
        del bad[("t0-0", "v2_minus")]
        with self.assertRaises(ValueError):
            science.evaluate_study(cases, bad, doses)
        bad_doses = dict(doses)
        bad_doses["t0-1"] = 0.5
        with self.assertRaises(ValueError):
            science.evaluate_study(cases, scores, bad_doses)
        # Only target-pair q is floored. Its average with a zero pair may be
        # a positive context dose at or below the original target floor.
        small_doses = dict(doses)
        small_scores = dict(scores)
        for name in ("t0-0", "t0-1"):
            small_doses[name] = 0.0
            for direction in ("u", "v1", "v2", "l1_u", "subject_u"):
                for sign in ("plus", "minus"):
                    small_scores[name, direction + "_" + sign] = scores[name, "clean"]
        small_doses["t0-2"] = small_doses["t0-3"] = 2 ** -18
        small_doses["c0"] = 2 ** -19
        self.assertEqual(science.evaluate_study(cases, small_scores, small_doses)["contexts"][0]["dose"], 2 ** -19)
        changed = dict(small_scores)
        changed["t0-0", "opp_diff"] = {**scores["t0-0", "clean"], "present": 12.0}
        with self.assertRaises(ValueError):
            science.evaluate_study(cases, changed, small_doses)

    def test_ranks_ties_and_full_vocabulary_kl_follow_protocol(self):
        self.assertTrue(callable(getattr(science, "vocabulary_diagnostics", None)))
        tied = science.vocabulary_diagnostics([1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0, 1, 2], top_n=2)
        self.assertEqual(tied["top_clean"], [0, 1])
        self.assertEqual(tied["top_clean_logits"], [1.0, 1.0])
        self.assertEqual(tied["target_logits_arm"], [1.0, 1.0, 0.0])
        self.assertEqual(tied["ranks_clean"], [1, 2, 3])
        self.assertEqual(tied["kl_clean_to_arm"], 0.0)
        # p=(1/2,1/2), q=(2/3,1/3) has KL = log(9/8)/2.
        changed = science.vocabulary_diagnostics([0.0, 0.0], [math.log(2), 0.0], [1], top_n=2)
        self.assertAlmostEqual(changed["kl_clean_to_arm"], math.log(9/8) / 2, places=15)


if __name__ == "__main__":
    unittest.main()
