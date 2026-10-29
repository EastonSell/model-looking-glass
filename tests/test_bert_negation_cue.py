import unittest
from unittest import mock

import torch
from torch import nn

from looking_glass.bert_negation_cue import (
    SITES, baseline_summary, choose_radius, evaluate_prediction, make_patch, observe, run_case, run_study,
    transfer_score,
)
from looking_glass import bert_negation_cue


class Block(nn.Module):
    def forward(self, hidden):
        return (hidden + hidden.mean(dim=1, keepdim=True) * 0.1,)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)
        self.distilbert = nn.Module()
        self.distilbert.transformer = nn.Module()
        self.distilbert.transformer.layer = nn.ModuleList([Block() for _ in range(6)])
        self.head = nn.Linear(8, 7)

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.distilbert.transformer.layer:
            hidden = layer(hidden)[0]
        return type("Output", (), {"logits": self.head(hidden)})()


class NegationCueTests(unittest.TestCase):
    def test_common_radius_uses_every_site_and_pair(self):
        states = {cue: {site: torch.tensor([value, 0.0]) for site, _, _ in SITES}
                  for cue, value in (("not", 4.0), ("never", 8.0), ("just", 0.0), ("also", 2.0))}
        radius, distances = choose_radius(states)
        self.assertEqual(radius, 2.0)
        self.assertEqual(len(distances), 20)
        states["also"]["object2"] = states["just"]["object2"] + torch.tensor([0.0005, 0.0])
        with self.assertRaisesRegex(ValueError, "radius"):
            choose_radius(states)

    def test_patch_has_requested_nonzero_dose_without_amplification(self):
        target = torch.tensor([1.0, 2.0], dtype=torch.float32)
        donor = torch.tensor([5.0, 2.0], dtype=torch.float32)
        patched, norm, fraction = make_patch(target, donor, 1.0)
        self.assertTrue(torch.equal(patched, torch.tensor([2.0, 2.0])))
        self.assertEqual(norm, 1.0)
        self.assertEqual(fraction, 0.25)
        with self.assertRaises(ValueError):
            make_patch(target, donor, 5.0)

    def test_bidirectional_sign_and_strict_heldout_boundary(self):
        actions = {("not", "just"): 0.08, ("just", "not"): -0.04,
                   ("never", "just"): 0.04, ("just", "never"): -0.02}
        self.assertAlmostEqual(transfer_score(actions), 0.045)
        self.assertTrue(evaluate_prediction([0.03, 0.04, 0.05, 0.06, -0.01, -0.01])["passed"])
        self.assertFalse(evaluate_prediction([0.02] * 6)["passed"])
        self.assertFalse(evaluate_prediction([0.1, 0.1, 0.1, 0.0, -0.01, -0.01])["passed"])
        with self.assertRaises(ValueError):
            evaluate_prediction([0.1] * 5)

    def test_full_vocabulary_baseline_uses_stable_probabilities_and_ties(self):
        value = torch.tensor([2.0, 2.0, 1.0, -1.0, -2.0, -3.0], dtype=torch.float32)
        summary = baseline_summary(value, any_id=0, some_id=2)
        self.assertEqual([entry["token_id"] for entry in summary["top_five"]], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(summary["any_minus_some"], 1.0)
        self.assertAlmostEqual(summary["any_probability"] / summary["some_probability"], 2.718281828459045)

    def test_self_and_final_cue_patch_leave_mask_logits_unchanged(self):
        torch.manual_seed(17)
        model = ToyModel().eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.int64)
        clean = observe(model, ids)
        self_patch = observe(model, ids, site="cue2", self_patch=True)
        self.assertTrue(torch.equal(clean["mask_logits"], self_patch["mask_logits"]))
        self.assertTrue(self_patch["other_tokens_equal"])
        replacement = clean["captures"]["cue5"] + 1
        final = observe(model, ids, site="cue5", source=replacement)
        self.assertTrue(torch.equal(clean["mask_logits"], final["mask_logits"]))
        self.assertTrue(torch.equal(final["after"], replacement))
        self.assertTrue(final["other_tokens_equal"])

    def test_complete_toy_case_runs_frozen_schedule_and_repeats(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        rows = {}
        for cue, token in (("not", 5), ("never", 6), ("just", 7), ("also", 8)):
            ids = base.copy()
            ids[4] = token
            rows[cue] = torch.tensor([ids], dtype=torch.int64)
        records = []
        result = run_case(model, "toy", rows, records.append, any_id=2, some_id=3)
        self.assertEqual(result["forwards"], 56)
        self.assertEqual(sum(record["kind"] in ("clean", "self", "transfer", "repeat")
                             for record in records), 56)
        self.assertEqual([record["kind"] for record in records[-4:]], ["repeat"] * 4)
        self.assertEqual(len([record for record in records if record["kind"] == "transfer"]), 32)
        self.assertTrue(result["radius"] >= 0.001)

    def test_public_split_requires_all_eight_complete_cases(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        manifest = {"cues": {"not": 5, "never": 6, "just": 7, "also": 8},
                    "scored_tokens": {"any": 2, "some": 3},
                    "cases": [{"id": f"case-{i}", "split": "development" if i < 2 else "evaluation",
                               "just_ids": [1, 2, 3, 4, 7, 6, 7, 8, 9, 10, 11, 12]}
                              for i in range(8)]}
        result = run_study(model, manifest, lambda _record: None)
        self.assertEqual(result["forwards"], 448)
        self.assertEqual(len(result["evaluation_cases"]), 6)
        del manifest["cases"][-1]
        with self.assertRaises(ValueError):
            run_study(model, manifest, lambda _record: None)

    def test_internal_landing_failure_keeps_completed_witness(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        rows = {}
        for cue, token in (("not", 5), ("never", 6), ("just", 7), ("also", 8)):
            ids = base.copy()
            ids[4] = token
            rows[cue] = torch.tensor([ids], dtype=torch.int64)
        records = []
        with mock.patch.object(bert_negation_cue, "_landing_equal", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "land"):
                run_case(model, "toy", rows, records.append, any_id=2, some_id=3)
        failure = records[-1]
        self.assertEqual(failure["kind"], "forward_failed")
        self.assertEqual(failure["phase"], "landing")
        self.assertEqual(failure["site"], "cue2")
        self.assertIn("mask_logits_f32le", failure)
        self.assertIn("full_before_f32le", failure)
        self.assertIn("full_after_f32le", failure)

    def test_completed_clean_witness_precedes_summary_failure(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        rows = {cue: torch.tensor([[1, 2, 3, 4, token, 6, 7, 8, 9, 10, 11, 12]])
                for cue, token in (("not", 5), ("never", 6), ("just", 7), ("also", 8))}
        records = []
        with mock.patch.object(bert_negation_cue, "baseline_summary", side_effect=RuntimeError("synthetic")):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                run_case(model, "toy", rows, records.append, any_id=2, some_id=3)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["kind"], "clean")
        self.assertIn("mask_logits_f32le", records[0])

    def test_completed_transfer_witness_survives_dose_measurement_failure(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        rows = {cue: torch.tensor([[1, 2, 3, 4, token, 6, 7, 8, 9, 10, 11, 12]])
                for cue, token in (("not", 5), ("never", 6), ("just", 7), ("also", 8))}
        records, state = [], {"after_transfer": False}
        real_observe = bert_negation_cue._observed_or_record_failure
        real_distance = bert_negation_cue.distance

        def observe_then_arm(*args, **kwargs):
            value = real_observe(*args, **kwargs)
            if "source" in kwargs:
                state["after_transfer"] = True
            return value

        def failing_distance(*args):
            if state["after_transfer"]:
                raise RuntimeError("synthetic dose failure")
            return real_distance(*args)

        with mock.patch.object(bert_negation_cue, "_observed_or_record_failure", side_effect=observe_then_arm), \
                mock.patch.object(bert_negation_cue, "distance", side_effect=failing_distance):
            with self.assertRaisesRegex(RuntimeError, "synthetic dose failure"):
                run_case(model, "toy", rows, records.append, any_id=2, some_id=3)
        failure = records[-1]
        self.assertEqual((failure["kind"], failure["phase"]), ("forward_failed", "dose_measurement"))
        self.assertEqual((failure["site"], failure["donor"], failure["target"]), ("cue2", "not", "just"))
        for name in ("mask_logits_f32le", "before_f32le", "after_f32le", "submitted_f32le",
                     "full_before_f32le", "full_after_f32le"):
            self.assertIn(name, failure)


if __name__ == "__main__":
    unittest.main()
