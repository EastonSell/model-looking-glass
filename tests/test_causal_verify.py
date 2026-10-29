import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from looking_glass.causal_verify import EvidenceError, audit_payload, audit_run
from looking_glass.causal_metrics import EffectCase, evaluate_effects


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "synthetic-causal-v1-20261028"


def _load_payloads():
    return (json.loads((RUN / "manifest.json").read_text(encoding="utf-8")),
            json.loads((RUN / "artifacts" / "precommit.json").read_text(encoding="utf-8")),
            json.loads((RUN / "artifacts" / "results.json").read_text(encoding="utf-8")))


def _rebind_precommit(frozen, result):
    raw = json.dumps(frozen, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    result["precommit_sha256"] = hashlib.sha256(raw).hexdigest()


class CausalEvidenceAuditTests(unittest.TestCase):
    def test_published_evidence_passes_without_claiming_an_explanation(self):
        report = audit_run(RUN)
        self.assertEqual(report["status"], "consistent")
        self.assertEqual(report["heldout_cases"], 16)
        self.assertAlmostEqual(report["summaries"]["first_observed"]["intervention_prediction_mae"],
                               0.002000067150220275)
        self.assertIn("discovery trace selection", report["not_verified"])
        self.assertIn("full control logits", report["not_verified"])
        self.assertIn("prior blindness and ordering", report["not_verified"])

    def test_case_effect_must_match_saved_base_and_patched_metrics(self):
        manifest, frozen, result = _load_payloads()
        result["cases"][0]["interventions"]["first_observed"]["effect"] += 0.1
        with self.assertRaises(EvidenceError):
            audit_payload(manifest, frozen, result)

    def test_prediction_must_match_frozen_discovery_effect_mean(self):
        manifest, frozen, result = _load_payloads()
        frozen["predictions"]["first_observed"]["predicted_effect"] += 0.1
        _rebind_precommit(frozen, result)
        with self.assertRaises(EvidenceError):
            audit_payload(manifest, frozen, result)

    def test_frozen_comparison_plan_is_checked(self):
        manifest, frozen, result = _load_payloads()
        frozen["comparisons"] = ["only compare against an easier baseline"]
        _rebind_precommit(frozen, result)
        with self.assertRaises(EvidenceError):
            audit_payload(manifest, frozen, result)

    def test_boolean_is_not_accepted_as_numeric_injection_strength(self):
        manifest, frozen, result = _load_payloads()
        frozen["injection"]["delta"] = True
        _rebind_precommit(frozen, result)
        with self.assertRaises(EvidenceError):
            audit_payload(manifest, frozen, result)

    def test_boolean_is_not_accepted_as_zero_baseline_or_invocation(self):
        for field in ("zero_prediction", "invocation"):
            with self.subTest(field=field):
                manifest, frozen, result = _load_payloads()
                if field == "zero_prediction":
                    frozen["zero_prediction"] = False
                else:
                    frozen["predictions"]["first_observed"]["invocation"] = False
                _rebind_precommit(frozen, result)
                with self.assertRaises(EvidenceError):
                    audit_payload(manifest, frozen, result)

    def test_missing_or_duplicate_heldout_cases_are_rejected(self):
        for mutation in ("missing", "duplicate"):
            with self.subTest(mutation=mutation):
                manifest, frozen, result = _load_payloads()
                if mutation == "missing":
                    result["cases"].pop()
                else:
                    result["cases"][-1]["case_id"] = result["cases"][0]["case_id"]
                with self.assertRaises(EvidenceError):
                    audit_payload(manifest, frozen, result)

    def test_summary_is_recomputed_instead_of_trusted(self):
        manifest, frozen, result = _load_payloads()
        result["summaries"]["first_observed"]["intervention_prediction"]["point"]["mae"] = 0.0
        with self.assertRaises(EvidenceError):
            audit_payload(manifest, frozen, result)

    def test_nonfinite_case_value_is_rejected(self):
        manifest, frozen, result = _load_payloads()
        result["cases"][0]["target_correction"] = float("nan")
        with self.assertRaises(EvidenceError):
            audit_payload(manifest, frozen, result)

    def test_no_bootstrap_metadata_must_be_self_consistent(self):
        for field in ("samples", "defined", "intervals"):
            with self.subTest(field=field):
                manifest, frozen, result = _load_payloads()
                bootstrap = result["summaries"]["first_observed"]["intervention_prediction"]["bootstrap"]
                if field == "samples":
                    bootstrap["samples"] = False
                elif field == "defined":
                    bootstrap["defined_samples"]["mae"] = 42
                else:
                    bootstrap["intervals"] = {}
                with self.assertRaises(EvidenceError):
                    audit_payload(manifest, frozen, result)

    def test_integer_model_seed_does_not_accept_float(self):
        manifest, frozen, result = _load_payloads()
        manifest["identities"]["model"]["value"]["seed"] = 2301.0
        with self.assertRaises(EvidenceError):
            audit_payload(manifest, frozen, result)

    def test_reverse_discrepancy_is_reported_not_assumed_away(self):
        manifest, frozen, result = _load_payloads()
        result["cases"][0]["controls"]["reverse_patch_effect"] = 123.0
        report = audit_payload(manifest, frozen, result)
        self.assertGreater(report["reverse_opposition_max_error"], 100.0)

    def test_duplicate_site_effect_discrepancy_is_reported(self):
        manifest, frozen, result = _load_payloads()
        row = result["cases"][0]
        row["interventions"]["magnitude"]["patched_metric"] += 0.1
        row["interventions"]["magnitude"]["effect"] = (
            row["interventions"]["magnitude"]["patched_metric"] - row["corrupt_metric"])
        kind = "magnitude"
        for name in ("intervention_prediction", "repair_residual"):
            pairs = []
            for case in result["cases"]:
                observed = case["interventions"][kind]["effect"] if name == "intervention_prediction" else case["target_correction"]
                predicted = case["interventions"][kind]["predicted_effect"] if name == "intervention_prediction" else case["interventions"][kind]["effect"]
                pairs.append(EffectCase(case["case_id"], predicted, observed))
            result["summaries"][kind][name] = evaluate_effects(pairs, zero_tolerance=1e-7, bootstrap_samples=0)
        report = audit_payload(manifest, frozen, result)
        self.assertGreater(report["coincident_site_effect_max_delta"], 0.09)

    def test_failed_terminal_run_is_not_a_successful_study(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "failed"
            shutil.copytree(RUN, destination)
            outcome_path = destination / "outcome.json"
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcome["state"] = "failed"
            outcome_path.write_text(json.dumps(outcome, sort_keys=True, separators=(",", ":")),
                                    encoding="utf-8")
            with self.assertRaises(EvidenceError):
                audit_run(destination)

    def test_file_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "tampered"
            shutil.copytree(RUN, destination)
            with (destination / "artifacts" / "precommit.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaises(EvidenceError):
                audit_run(destination)

    def test_outcome_heldout_count_must_match_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "wrong-count"
            shutil.copytree(RUN, destination)
            outcome_path = destination / "outcome.json"
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcome["details"]["heldout_cases"] = 999
            outcome_path.write_text(json.dumps(outcome, sort_keys=True, separators=(",", ":")),
                                    encoding="utf-8")
            with self.assertRaises(EvidenceError):
                audit_run(destination)

    def test_auditor_import_does_not_load_pytorch(self):
        command = "import sys; import looking_glass.causal_verify; assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", command], cwd=ROOT,
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
