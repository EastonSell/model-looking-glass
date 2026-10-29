"""Small fabricated checks for the independent matched-dose review."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


def load_review():
    path = Path(__file__).with_name("review_bert_taste_matched_closed.py")
    if not path.exists():
        path = Path(__file__).with_name("independent-custody-auditor.py")
    spec = importlib.util.spec_from_file_location("_matched_custody_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClosedReviewPrimitives(unittest.TestCase):
    def test_published_package_paths_use_the_public_source_commit(self):
        review = load_review()
        package = Path("repo/results/taste-package")
        paths = review.evidence_paths(package, published_package=True)
        self.assertEqual(paths["run"], package / "run")
        self.assertEqual(paths["observer"], package / "receipts/observer.json")
        self.assertEqual(paths["exit"], package / "receipts/operator-exit-projection.json")
        self.assertEqual(paths["repo"], Path("repo"))
        self.assertEqual(paths["source_ref"], review.SOURCE_COMMIT)

    def test_public_exit_projection_is_bound_without_a_process_identifier(self):
        review = load_review()
        record = {"status": "observed-zero-exit", "exit_code": 0, "parent_prefix_empty": True,
                  "source_commit": review.SOURCE_COMMIT, "observer_sha256": review.OBSERVER_SHA,
                  "result_sha256": review.RESULT_SHA, "original_sha256": review.EXIT_SHA}
        review.verify_public_exit_projection(record)
        for changed in ({**record, "process_id": 1}, {**record, "exit_code": False},
                        {**record, "original_sha256": "0" * 64},
                        {**record, "parent_prefix_empty": False}):
            with self.assertRaises(ValueError):
                review.verify_public_exit_projection(changed)

    def test_private_exit_requires_a_positive_exact_integer_pid(self):
        review = load_review()
        record = {"status": "observed-zero-exit", "process_id": 1, "exit_code": 0,
                  "parent_prefix_empty": True, "source_commit": review.SOURCE_COMMIT,
                  "observer_sha256": review.OBSERVER_SHA, "result_sha256": review.RESULT_SHA}
        review.verify_private_exit(record)
        for invalid_pid in (True, 0, -1, "1"):
            with self.assertRaises(ValueError):
                review.verify_private_exit({**record, "process_id": invalid_pid})

    def test_binary32_extremes_and_signed_zero(self):
        review = load_review()
        self.assertEqual(review.float32(1), 2.0 ** -149)
        self.assertEqual(review.float32(0x80000001), -(2.0 ** -149))
        self.assertEqual(math.copysign(1.0, review.float32(0x80000000)), -1.0)
        for bits in (0x7F800000, 0x7FC00000, -1, True):
            with self.assertRaises(ValueError):
                review.float32(bits)

    def test_selected_margin_requires_the_exact_saved_bits(self):
        review = load_review()
        summary = {"sweet_bits": 0x3F800000, "sour_bits": 0xBF000000,
                   "sweet_logit": 1.0, "sour_logit": -0.5, "sweet_minus_sour": 1.5}
        self.assertEqual(review.selected_margin(summary), 1.5)
        for bad in ({**summary, "sweet_minus_sour": 1.0}, {**summary, "sweet_bits": True},
                    {**summary, "sour_logit": -0.4}):
            with self.assertRaises(ValueError):
                review.selected_margin(bad)

    def test_direction_sign_and_eligible_mean(self):
        review = load_review()
        differences = {"a": (1.0, 0.25), "b": (0.5, 0.75), "c": (3.0, 0.0)}
        result = review.direction_summary(differences, {"a": True, "b": True, "c": False}, sign=-1)
        self.assertEqual(result, {"a": -0.75, "b": 0.25, "c": -3.0, "n": 2, "mean": -0.25})

    def test_fabricated_full_schedule_and_rehashed_effect_fault(self):
        review = load_review()
        checkout = Path(__file__).parent / "teams/bert_taste_matched_dose"
        if not checkout.exists():
            checkout = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("_matched_fake", checkout / "tests/bert_taste_matched_fake_fixture.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        cases = json.loads((checkout / "experiments/bert_taste_matched_dose_public_cases_v1.json").read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fixture.make_fixture(root, cases)
            journals = {frame + ".jsonl": (root / (frame + ".jsonl")).read_bytes() for frame in review.FRAMES}
            report = review.audit_numeric(cases, result, journals)
            self.assertEqual((report["completed_forwards"], report["interventions"]), (516, 360))
            result["effects"][0]["effect"] = 0.25
            with self.assertRaises(ValueError):
                review.audit_numeric(cases, result, journals)

    def test_environment_rejects_boolean_integer_aliases_and_cuda(self):
        review = load_review()
        result = {"model": {"model": "distilbert/distilbert-base-uncased",
                            "revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
                            "device": "cpu", "dtype": "float32", "model_class": "DistilBertForMaskedLM",
                            "tokenizer_class": "DistilBertTokenizerFast", "attention_implementation": "eager",
                            "parameter_count": 66985530, "pytorch_cuda_initialized": False,
                            "assets": review.EXPECTED_ASSETS, "torch_version": "2.8.0+cu128",
                            "transformers_version": "4.57.3"},
                  "runtime": {"python": "3.11.15", "torch": "2.8.0+cu128", "threads": 1, "interop_threads": 1,
                              "deterministic_algorithms": True, "cuda_initialized": False,
                              "default_dtype": "torch.float32", "mkldnn_enabled": True,
                              "omp_num_threads": "1", "mkl_num_threads": "1", "mkl_cbwr": "AUTO",
                              "aten_cpu_capability": None, "denormal_probe": {"smallest_times_one_bits": 1,
                                                                               "normal_half_bits": 4194304},
                              "rounding_probe": "nearest-even-halfway-cases-pass",
                              "torch_build_sha256": "9835a6eed581f616d1e294cc42162b701425ea24bf4723678b8233e78f30e8c6"},
                  "model_state_sha256_before_after": "a" * 64,
                  "exposure": {"prior_row_counts": [160, 64, 8], "exact_or_complete_prefix_collisions": 0,
                               "complete_exposure_history_certified": False}}
        review.verify_environment(result)
        for changed in ({**result, "runtime": {**result["runtime"], "threads": True}},
                        {**result, "runtime": {**result["runtime"], "cuda_initialized": True}},
                        {**result, "model": {**result["model"], "device": "cuda"}}):
            with self.assertRaises(ValueError):
                review.verify_environment(changed)


if __name__ == "__main__":
    unittest.main()
