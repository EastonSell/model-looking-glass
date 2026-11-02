import copy
import unittest
from unittest import mock

import torch
from torch import nn

from looking_glass.bert_cue_class import (
    baseline_screen, context_specificity, evaluate, run_case, run_study,
)
from looking_glass import bert_cue_class


CUES = ("never", "rarely", "always", "often")
IDS = {"never": 5, "rarely": 6, "always": 7, "often": 8}


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
        for block in self.distilbert.transformer.layer:
            hidden = block(hidden)[0]
        return type("Output", (), {"logits": self.head(hidden)})()


def fixture_case():
    rows = {}
    for cue in CUES:
        ids = [1, 2, 3, IDS[cue], 4, 9, 10, 11, 12, 13, 14]
        rows[cue] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    return {"id": "toy", "split": "evaluation", "frame": "main",
            "indices": {"cue": 3, "mask": 5, "object": 8}, "rows": rows}


def strong_effects():
    values = {}
    for target in CUES:
        negative = target in ("never", "rarely")
        for donor in CUES:
            if donor == target:
                continue
            opposite = (donor in ("never", "rarely")) != negative
            sign = -1 if negative else 1
            values[donor, target] = sign * (0.2 if opposite else 0.05)
    return values


class CueClassTests(unittest.TestCase):
    def test_baseline_requires_all_four_strict_gaps(self):
        result = baseline_screen({"never": 1.0, "rarely": 0.75, "always": 0.0, "often": 0.1})
        self.assertTrue(result["eligible"])
        self.assertEqual(len(result["gaps"]), 4)
        self.assertFalse(baseline_screen({"never": 1.0, "rarely": 0.6, "always": 0.0, "often": 0.1})["eligible"])

    def test_both_opposite_donors_must_beat_same_class_control(self):
        effects = strong_effects()
        result = context_specificity(effects)
        self.assertTrue(result["passed"])
        effects["never", "always"] = 0.099
        result = context_specificity(effects)
        self.assertFalse(result["passed"])
        self.assertFalse(result["targets"]["always"]["passed"])
        self.assertTrue(result["targets"]["often"]["passed"])
        effects = strong_effects()
        effects["never", "always"] = 0.1
        self.assertFalse(context_specificity(effects)["targets"]["always"]["passed"])

    def test_evaluation_distinguishes_unevaluable_from_failed(self):
        rows = [{"id": f"{frame}-{i}", "frame": frame, "split": "evaluation",
                 "baseline_eligible": True, "dose_eligible": True, "specificity_passed": True}
                for frame in ("main", "embedded", "relative") for i in range(3)]
        self.assertEqual(evaluate(rows)["status"], "passed")
        changed = copy.deepcopy(rows)
        changed[0]["specificity_passed"] = False
        changed[1]["specificity_passed"] = False
        self.assertEqual(evaluate(changed)["status"], "failed")
        changed[1]["baseline_eligible"] = False
        changed[2]["dose_eligible"] = False
        result = evaluate(changed)
        self.assertEqual(result["status"], "unevaluable")
        self.assertEqual(result["frames"]["main"]["eligible"], 1)

    def test_baseline_ineligible_case_still_runs_all_patches(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        events = []
        with mock.patch("looking_glass.bert_cue_class.baseline_screen", return_value={"eligible": False, "gaps": []}):
            result = run_case(model, fixture_case(), events.append, any_id=2, some_id=3)
        self.assertFalse(result["baseline_eligible"])
        self.assertTrue(result["dose_eligible"])
        self.assertEqual(result["forwards"], 72)
        self.assertEqual(sum(row["kind"] == "transfer" for row in events), 48)
        self.assertEqual([row["kind"] for row in events[-4:]], ["repeat"] * 4)

    def test_small_radius_preserves_baselines_and_repeats(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        events = []
        with mock.patch("looking_glass.bert_cue_class.choose_radius", return_value=(0.0005, [])):
            result = run_case(model, fixture_case(), events.append, any_id=2, some_id=3)
        self.assertFalse(result["dose_eligible"])
        self.assertEqual(result["forwards"], 8)
        self.assertEqual(sum(row["kind"] == "transfer" for row in events), 0)
        self.assertIn("dose_ineligible", [row["kind"] for row in events])
        self.assertEqual([row["kind"] for row in events[-4:]], ["repeat"] * 4)

    def test_completed_transfer_is_retained_on_dose_calculation_failure(self):
        torch.manual_seed(22)
        model, events, state = ToyModel().eval(), [], {"armed": False}
        actual_observe, actual_distance = bert_cue_class._observed, bert_cue_class.distance
        def observe_then_arm(*args, **kwargs):
            value = actual_observe(*args, **kwargs)
            if "source" in kwargs:
                state["armed"] = True
            return value
        def bad_distance(*args):
            if state["armed"]:
                raise RuntimeError("synthetic dose failure")
            return actual_distance(*args)
        with mock.patch.object(bert_cue_class, "_observed", side_effect=observe_then_arm), \
                mock.patch.object(bert_cue_class, "distance", side_effect=bad_distance):
            with self.assertRaisesRegex(RuntimeError, "synthetic dose failure"):
                run_case(model, fixture_case(), events.append, any_id=2, some_id=3)
        self.assertEqual((events[-1]["kind"], events[-1]["phase"]), ("forward_failed", "dose_measurement"))
        for key in ("mask_logits_f32le", "submitted_f32le", "before_f32le", "after_f32le",
                    "full_before_f32le", "full_after_f32le"):
            self.assertIn(key, events[-1])

    def test_landing_failure_retains_submitted_vector(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        vector = torch.tensor([1.0, 2.0])
        payload = {"mask_logits": vector, "before": vector, "after": vector,
                   "full_before": vector, "full_after": vector}
        error = bert_cue_class.CompletedObservationFailure("landing", payload)
        events = []
        with mock.patch.object(bert_cue_class, "observe", side_effect=error):
            with self.assertRaises(bert_cue_class.CompletedObservationFailure):
                bert_cue_class._observed(None, ids, {}, events.append, {"case_id": "toy"},
                                        site="cue2", source=vector)
        self.assertIn("submitted_f32le", events[-1])

    def test_nonfinite_or_validation_exception_retains_completed_witness(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        case = fixture_case()
        ids = torch.tensor([case["rows"]["never"]["input_ids"]])
        indices = case["indices"]
        clean = bert_cue_class.observe(model, ids, indices)
        actual_forward = model.forward
        def nonfinite(*args, **kwargs):
            result = actual_forward(*args, **kwargs)
            result.logits[0, indices["mask"], 0] = float("nan")
            return result
        for site in (None, "cue2"):
            kwargs = {} if site is None else {"site": site, "source": clean["captures"][site]}
            for failure in ("nonfinite", "validation_exception"):
                with self.subTest(site=site, failure=failure):
                    events = []
                    target = mock.patch.object(model, "forward", side_effect=nonfinite) if failure == "nonfinite" else \
                             mock.patch.object(bert_cue_class, "_logits_finite", side_effect=ValueError("synthetic"))
                    with target, self.assertRaises(bert_cue_class.CompletedObservationFailure):
                        bert_cue_class._observed(model, ids, indices, events.append, {"case_id": "toy"}, **kwargs)
                    record = events[-1]
                    self.assertEqual(record["kind"], "forward_failed")
                    self.assertIn("mask_logits_f32le", record)
                    self.assertIn("captures_f32le" if site is None else "full_before_f32le", record)
                    if site is not None:
                        self.assertIn("submitted_f32le", record)

    def test_unavailable_or_oversized_logits_are_explicitly_omitted(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        case = fixture_case()
        ids = torch.tensor([case["rows"]["never"]["input_ids"]])
        indices = case["indices"]
        clean = bert_cue_class.observe(model, ids, indices)
        actual_forward = model.forward
        for site in (None, "cue2"):
            kwargs = {} if site is None else {"site": site, "source": clean["captures"][site]}
            for failure in ("missing", "oversized"):
                def invalid(*args, **kw):
                    result = actual_forward(*args, **kw)
                    if failure == "missing":
                        return object()
                    result.logits = torch.zeros((1, ids.shape[1], 100001), dtype=torch.float32)
                    return result
                events = []
                with self.subTest(site=site, failure=failure), mock.patch.object(model, "forward", side_effect=invalid):
                    with self.assertRaises(bert_cue_class.CompletedObservationFailure):
                        bert_cue_class._observed(model, ids, indices, events.append, {"case_id": "toy"}, **kwargs)
                self.assertEqual(events[-1]["omitted_components"], ["mask_logits_unavailable_or_unbounded"])
                self.assertIn("captures_f32le" if site is None else "full_before_f32le", events[-1])

    def test_wrong_bounded_vocabulary_width_is_retained_and_rejected(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        case = fixture_case()
        ids = torch.tensor([case["rows"]["never"]["input_ids"]])
        indices = case["indices"]
        clean = bert_cue_class.observe(model, ids, indices)
        actual_forward = model.forward
        def short(*args, **kw):
            result = actual_forward(*args, **kw)
            result.logits = result.logits[:, :, :6]
            return result
        for site in (None, "cue2"):
            kwargs = {} if site is None else {"site": site, "source": clean["captures"][site]}
            events = []
            with self.subTest(site=site), mock.patch.object(model, "forward", side_effect=short):
                with self.assertRaises(bert_cue_class.CompletedObservationFailure):
                    bert_cue_class._observed(model, ids, indices, events.append, {"case_id": "toy"},
                                             expected_vocab=7, **kwargs)
            self.assertIn("mask_logits_f32le", events[-1])
            self.assertEqual(events[-1]["omitted_components"], [])

    def test_wrong_bounded_capture_width_is_retained_and_rejected(self):
        torch.manual_seed(22)
        model = ToyModel().eval()
        case = fixture_case()
        ids = torch.tensor([case["rows"]["never"]["input_ids"]])
        indices = case["indices"]
        clean = bert_cue_class.observe(model, ids, indices)
        for site in (None, "cue2"):
            kwargs = {} if site is None else {"site": site, "source": clean["captures"][site]}
            events = []
            with self.subTest(site=site), self.assertRaises(bert_cue_class.CompletedObservationFailure):
                bert_cue_class._observed(model, ids, indices, events.append, {"case_id": "toy"},
                                         expected_width=9, **kwargs)
            self.assertEqual(events[-1]["phase"], "capture_shape")
            self.assertIn("mask_logits_f32le", events[-1])
            self.assertIn("captures_f32le" if site is None else "full_before_f32le", events[-1])

    def test_full_toy_schedule_preserves_all_cases(self):
        torch.manual_seed(22)
        cases = []
        for split, counts in (("development", 1), ("evaluation", 3)):
            for frame in ("main", "embedded", "relative"):
                for index in range(counts):
                    case = fixture_case()
                    case.update(id=f"{split}-{frame}-{index}", split=split, frame=frame)
                    cases.append(case)
        manifest = {"cases": cases, "scored_tokens": {"any": 2, "some": 3}}
        events = []
        result = run_study(ToyModel().eval(), manifest, events.append)
        self.assertEqual(result["forwards"], 864)
        self.assertEqual(len(result["cases"]), 12)
        self.assertEqual(len(events), 888)
        self.assertEqual(sum(row["kind"] == "transfer" for row in events), 576)


if __name__ == "__main__":
    unittest.main()
