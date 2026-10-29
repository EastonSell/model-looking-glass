"""Tests use only frozen public IDs; no checkpoint forward is performed."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from looking_glass import bert_agreement_cases as cases


MANIFEST = Path(__file__).resolve().parents[1] / "experiments" / "bert_agreement_public_cases_v1.json"
EXPECTED = "20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5"
DEVELOPMENT = (("farmer", "nurse"), ("pilot", "soldier"),
               ("driver", "singer"), ("doctor", "dancer"))
EVALUATION = (("painter", "artist"), ("actor", "guard"),
              ("teacher", "worker"), ("chef", "clerk"),
              ("student", "player"), ("lawyer", "coach"),
              ("writer", "sailor"), ("leader", "cook"))


class BertAgreementCaseTests(unittest.TestCase):
    def setUp(self):
        self.data = MANIFEST.read_bytes()
        self.manifest = json.loads(self.data)
        self.rows = {r["row_id"]: r for r in self.manifest["rows"]}

    def test_saved_manifest_and_known_exposure_scope(self):
        self.assertEqual(hashlib.sha256(self.data).hexdigest(), EXPECTED)
        self.assertEqual(cases.manifest_digest(), EXPECTED)
        report = cases.verify_manifest_bytes(self.data)
        self.assertEqual((report.development_target_count, report.evaluation_target_count), (16, 32))
        self.assertEqual((report.development_row_count, report.evaluation_row_count), (32, 64))
        self.assertEqual((report.observed_pilot_count, report.exact_token_overlap_count), (4, 0))
        self.assertFalse(report.historical_exposure_complete)
        self.assertFalse(report.execution_authorized)
        self.assertEqual(Counter(r["exposure_status"] for r in self.rows.values()),
                         {"observed-pilot-and-replay": 4, "reserved-development": 28,
                          "public-nonblind-evaluation": 64})
        observed = {tuple(r["input_ids"]) for r in self.rows.values()
                    if r["exposure_status"] == "observed-pilot-and-replay"}
        self.assertEqual(observed, cases.PILOT_ROWS)
        self.assertEqual(self.manifest["pilot_evidence"]["original_result_sha256"],
                         "2602232c7b0d5382f578b455b83496721ce6a68ff4a9fef398cdb033862387e0")
        self.assertEqual(self.manifest["pilot_evidence"]["independent_replay_sha256"],
                         "d0c4acfec26b80fd8609b3a005e3970c73eff76f6686ddc30823d1df1f89076a")

    def test_exact_balanced_target_donor_pairings_and_split_disjointness(self):
        rows = self.rows
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({tuple(r["input_ids"]) for r in rows.values()}), 96)
        self.assertEqual(len(self.manifest["targets"]), 48)
        self.assertEqual(len({r["case_id"] for r in self.manifest["targets"]}), 48)
        for split, families in (("development", DEVELOPMENT), ("evaluation", EVALUATION)):
            subset = [target for target in self.manifest["targets"] if target["split"] == split]
            self.assertEqual(len(subset), 4 * len(families))
            for index, (subject, distractor) in enumerate(families):
                family_targets = [t for t in subset if t["family"] == index + 1]
                target_pairs = [(rows[t["target_row_id"]]["subject_number"],
                                 rows[t["target_row_id"]]["distractor_number"]) for t in family_targets]
                self.assertEqual(target_pairs, [("singular", "singular"), ("singular", "plural"),
                                                ("plural", "singular"), ("plural", "plural")])
                next_subject = families[(index + 1) % len(families)][0]
                for target in family_targets:
                    row = rows[target["target_row_id"]]
                    self.assertEqual((row["subject_lemma"], row["distractor_lemma"]), (subject, distractor))
                    self.assertEqual(row["subject_source"], "A")
                    self.assertEqual(set(target["donor_row_ids"]), {"A_same", "A_opposite", "B_same", "B_opposite"})
                    self.assertEqual(target["donor_row_ids"]["A_same"], target["target_row_id"])
                    for key, donor_id in target["donor_row_ids"].items():
                        donor = rows[donor_id]
                        self.assertEqual(donor["split"], split)
                        self.assertEqual(donor["family"], index + 1)
                        self.assertEqual((donor["distractor_lemma"], donor["distractor_number"]),
                                         (row["distractor_lemma"], row["distractor_number"]))
                        self.assertEqual(donor["subject_lemma"], subject if key.startswith("A") else next_subject)
                        self.assertEqual(donor["subject_source"], key[0])
                        self.assertEqual(donor["subject_number"] == row["subject_number"], key.endswith("same"))
        dev_lemmas = {lemma for pair in DEVELOPMENT for lemma in pair}
        eval_lemmas = {lemma for pair in EVALUATION for lemma in pair}
        self.assertTrue(dev_lemmas.isdisjoint(eval_lemmas))
        dev_ids = {tuple(r["input_ids"]) for r in rows.values() if r["split"] == "development"}
        eval_ids = {tuple(r["input_ids"]) for r in rows.values() if r["split"] == "evaluation"}
        self.assertTrue(dev_ids.isdisjoint(eval_ids))

    def test_pinned_coordinates_masks_and_identity(self):
        identity = self.manifest["identity"]
        self.assertEqual(identity["model"], "distilbert/distilbert-base-uncased")
        self.assertEqual(identity["revision"], "12040accade4e8a0f71eabdb258fecc2e7e948be")
        self.assertEqual(identity["asset_sha256"]["model.safetensors"],
                         "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063")
        self.assertEqual(identity["asset_sha256"]["tokenizer.json"],
                         "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98")
        self.assertEqual(self.manifest["coordinates"],
                         {"subject": 2, "distractor": 5, "mask": 6, "sequence_length": 10,
                          "output_token_ids": {"is": 2003, "are": 2024}})
        for row in self.rows.values():
            with self.subTest(row=row["row_id"]):
                ids = row["input_ids"]
                self.assertEqual(len(ids), 10)
                self.assertTrue(all(type(i) is int for i in ids))
                self.assertEqual((ids[0], ids[1], ids[4], ids[6:]), (101, 1996, 1996, [103, 3201, 1012, 102]))
                self.assertEqual(ids[3], 2379 if row["split"] == "development" else 3875)
                self.assertEqual(row["attention_mask"], [1] * 10)

    def test_tampered_or_noncanonical_payload_cannot_be_reported_as_frozen(self):
        for change in ("token", "boolean", "donor", "mask"):
            payload = json.loads(self.data)
            if change == "token":
                payload["rows"][0]["input_ids"][2] += 1
            elif change == "boolean":
                payload["coordinates"]["subject"] = True
            elif change == "donor":
                payload["targets"][0]["donor_row_ids"]["B_same"] = payload["targets"][0]["target_row_id"]
            else:
                payload["rows"][0]["attention_mask"][0] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                cases.verify_manifest_bytes(json.dumps(payload).encode("utf-8"))
        with self.assertRaises(ValueError):
            cases.verify_manifest_bytes(self.data + b" ")
        with self.assertRaises(ValueError):
            cases.verify_manifest_bytes(bytearray(self.data))
        with self.assertRaises(ValueError):
            cases.verify_manifest_bytes(b"x" * 1_000_001)

    def test_saved_tokenizer_check_binds_manifest_and_source(self):
        root = MANIFEST.parents[1]
        data = (root / "results" / "bert-agreement-tokenizer-check-v1-20261028.json").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         "2c7ad4548af28032d7b77b25f388c6855085bbbf1baca0698e6d4fe54c7dc958")
        report = json.loads(data)
        self.assertEqual(report["manifest_sha256"], EXPECTED)
        self.assertEqual(report["verified_rows"], 96)
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["forward_executed"])
        source = (root / "scripts" / "verify_bert_agreement_tokenizer.py").read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), report["verifier_source_sha256"])


if __name__ == "__main__":
    unittest.main()
