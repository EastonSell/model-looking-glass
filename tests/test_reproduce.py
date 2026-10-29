"""Tests of the saved-results audit; no model execution or GPU required."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from looking_glass.reproduce import AuditError, audit_run


ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "results" / "verified-cpu-v1"


class ReproducibilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        shutil.copytree(RECORDED, self.run)

    def test_saved_run_recomputes_claims_and_paired_seed_ratios(self):
        report = audit_run(self.run, ROOT / "looking_glass")
        self.assertEqual(report["timing_cells"], 56)
        self.assertEqual(report["raw_samples"], 5600)
        self.assertEqual(report["transparency_passed"], 48)
        self.assertEqual(report["reconstruction_elements"], 51200)
        self.assertEqual(report["localization"]["dense"], {"trials": 96, "detected": 96,
                         "exact_first_point": 96, "first_point_after_injection": 0})
        self.assertEqual(report["localization"]["sparse"]["first_point_after_injection"], 96)
        cell = report["timing_aggregates"]["16"]["sparse"]
        self.assertAlmostEqual(cell["direct_median_us"], 170.91275)
        self.assertAlmostEqual(cell["gradient_over_direct_paired_median"], 2.7918576159863573)

    def test_changed_raw_csv_sample_is_rejected(self):
        path = self.run / "raw_timing.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["nanoseconds"] = str(int(rows[0]["nanoseconds"]) + 1)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(AuditError, "raw CSV"):
            audit_run(self.run)

    def test_summary_is_recomputed_even_if_json_and_csv_agree(self):
        path = self.run / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["timing"][0]["median_us"] += 100
        path.write_text(json.dumps(data), encoding="utf-8")
        path = self.run / "timing.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["median_us"] = str(data["timing"][0]["median_us"])
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(AuditError, "median"):
            audit_run(self.run)

    def test_duplicate_transparency_case_cannot_replace_missing_case(self):
        path = self.run / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["transparency"][0] = data["transparency"][1]
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "transparency"):
            audit_run(self.run)

    def test_seed_must_be_an_integer_not_a_boolean(self):
        path = self.run / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["config"]["seeds"][1] = True
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "protocol"):
            audit_run(self.run)

    def test_malformed_environment_and_source_hashes_fail_as_audit_errors(self):
        path = self.run / "results.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for mutation in ("boolean_threads", "source_list"):
            with self.subTest(mutation=mutation):
                data = json.loads(json.dumps(original))
                if mutation == "boolean_threads":
                    data["environment"]["threads"] = True
                else:
                    data["source_sha256"] = list(data["source_sha256"])
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(AuditError):
                    audit_run(self.run)

    def test_first_difference_cannot_precede_the_injection(self):
        path = self.run / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        row = next(r for r in data["localization"]
                   if r["placement"] == "dense" and r["injected_at"] == "block.0.activation")
        row["observed_interval"] = {"last_equal": None, "first_unequal": "block.0.branch.0"}
        row["exact_first_point"] = False
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(AuditError, "precedes"):
            audit_run(self.run)

    def test_source_differences_are_reported_separately_from_saved_data(self):
        sources = Path(self.temp.name) / "sources"
        sources.mkdir()
        (sources / "experiment.py").write_text("different source\n", encoding="utf-8")
        report = audit_run(self.run, sources)
        self.assertEqual(report["source_hashes"]["experiment.py"]["status"], "mismatch")
        self.assertEqual(report["source_hashes"]["model.py"]["status"], "missing")
        self.assertEqual(report["raw_samples"], 5600)

    def test_additional_package_source_is_compared_for_later_v1_runs(self):
        path = self.run / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        source = ROOT / "looking_glass" / "reproduce.py"
        data["source_sha256"][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        path.write_text(json.dumps(data), encoding="utf-8")
        report = audit_run(self.run, ROOT / "looking_glass")
        self.assertEqual(report["source_hashes"][source.name]["status"], "match")


if __name__ == "__main__":
    unittest.main()
