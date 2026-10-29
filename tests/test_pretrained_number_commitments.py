"""Fabricated pools/nonces only: no real study allocation or model forward."""
import copy
import hashlib
import json
from pathlib import Path
from unittest import mock
import unittest

from looking_glass import pretrained_exposures_v3 as exposures
from looking_glass import pretrained_exposures_v4 as later_exposures
from looking_glass import pretrained_number_commitments as study


NOUNS = ("alpha/alphas", "beta/betas", "gamma/gammas", "delta/deltas",
         "epsilon/epsilons", "zeta/zetas", "eta/etas", "theta/thetas")
PLURAL = ("near the boxes", "behind the cars", "above the lamps", "below the desks")
SINGULAR = ("near the box", "behind the car", "above the lamp", "below the desk")
FAKE_NONCE = bytes(range(32))
FAKE_SITE_NONCE = bytes(reversed(range(32)))
TRACKS = {
    "natural": {"source_commit": "a" * 40, "runtime_sha256": "b" * 64, "prediction_sha256": "c" * 64},
    "injection": {"source_commit": "d" * 40, "runtime_sha256": "e" * 64, "prediction_sha256": "f" * 64},
}


class NumberCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.pool_patch = mock.patch.multiple(study, NOUNS=NOUNS, PLURAL_CONTEXTS=PLURAL,
                                              SINGULAR_CONTEXTS=SINGULAR)
        self.pool_patch.start()
        self.addCleanup(self.pool_patch.stop)
        self.freeze = study.make_prediction_freeze(copy.deepcopy(TRACKS))
        self.site = study.seal_site(FAKE_SITE_NONCE, "d" * 40)
        self.site_kwargs = {"site_nonce": FAKE_SITE_NONCE, "site_record_bytes": self.site.record_bytes,
                            "site_commitment": self.site.commitment_sha256}
        self.inventory = study.load_inventory_snapshot(exposures.canonical_inventory_bytes(),
                                                        exposures.inventory_digest())
        self.inventory_kwargs = {"inventory_snapshot": self.inventory,
                                 "current_inventory_sha256": self.inventory.sha256}
        self.allocation = study.sample_families(self.freeze, FAKE_NONCE,
                                                **self.inventory_kwargs, **self.site_kwargs)

    def fake_cases(self):
        result = []
        for family in self.allocation.families:
            rows = []
            for index, text in enumerate(family.texts):
                ids = [10, 20000 + 4 * family.family_index + index, 3000 + family.family_index]
                rows.append({"text": text, "input_ids": ids,
                             "attention_mask": [1] * len(ids), "subject_index": 1})
            result.append(rows)
        return result

    def test_protocol_canonical_encoding_and_aggregate_gate(self):
        protocol = Path(__file__).resolve().parents[1] / "docs" / "PRETRAINED_NUMBER_PATCH_V1.md"
        self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(), study.PROTOCOL_SHA256)
        self.assertEqual(study.canonical_json({"z": [2, "é"], "a": 1}),
                         b'{"a":1,"z":[2,"\\u00e9"]}')
        for bad in (True, None, 1.0, {"bad": False}, {1: "bad"}, (1, 2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                study.canonical_json(bad)
        record = json.loads(self.freeze.record_bytes)
        self.assertEqual(set(record), {"schema", "protocol_sha256", "checkpoint_revision", "tracks"})
        self.assertEqual(record["tracks"], TRACKS)
        self.assertEqual(hashlib.sha256(self.freeze.record_bytes).hexdigest(), self.freeze.sha256)
        self.assertEqual(study.load_prediction_freeze(self.freeze.record_bytes, self.freeze.sha256), self.freeze)
        for bad in (None, study.PredictionFreeze(b"{}", "0" * 64),
                    study.PredictionFreeze(self.freeze.record_bytes, "0" * 64)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                study.sample_families(bad, FAKE_NONCE, **self.inventory_kwargs, **self.site_kwargs)
        with self.assertRaises(ValueError):
            study.load_prediction_freeze(self.freeze.record_bytes + b"\n", self.freeze.sha256)
        for tracks in ({}, {"other": TRACKS["natural"]}, {"natural": {"source_commit": "a" * 40}},
                       {"natural": {**TRACKS["natural"], "runtime_sha256": "A" * 64}}):
            with self.subTest(tracks=tracks), self.assertRaises(ValueError):
                study.make_prediction_freeze(tracks)

    def test_later_inventory_is_canonical_identity_bound_and_superset(self):
        def encoded(payload):
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                              allow_nan=False).encode("utf-8")

        def load(payload):
            raw = encoded(payload)
            return study.load_inventory_snapshot(raw, hashlib.sha256(raw).hexdigest())

        original = exposures.inventory_payload()
        with self.assertRaises(ValueError):
            study.load_inventory_snapshot(self.inventory.record_bytes + b"\n", self.inventory.sha256)
        changed = copy.deepcopy(original)
        changed["identity"]["weights_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            load(changed)
        missing = copy.deepcopy(original)
        missing["records"] = [r for r in missing["records"] if r["token_ids"][:4] != [50256] * 4]
        with self.assertRaises(ValueError):
            load(missing)
        duplicate = copy.deepcopy(original)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        with self.assertRaises(ValueError):
            load(duplicate)
        malformed = copy.deepcopy(original)
        malformed["records"][0]["token_ids"][0] = True
        with self.assertRaises(ValueError):
            load(malformed)
        with mock.patch.object(study, "_sort_pool", side_effect=AssertionError("allocation started")):
            with self.assertRaises(ValueError):
                study.sample_families(self.freeze, FAKE_NONCE, inventory_snapshot=self.inventory,
                                      current_inventory_sha256="0" * 64, **self.site_kwargs)

        # The extra row is entirely fabricated and is equal to the first fake
        # held-out row; a later snapshot must exclude it at sealing.
        later = copy.deepcopy(original)
        later["version"] = "distilgpt2-development-exposures-v4"
        extra = copy.deepcopy(later["records"][0])
        extra.update({"record_id": "synthetic-extra", "category": "planned",
                      "token_ids": self.fake_cases()[0][0]["input_ids"][:]})
        later["records"].append(extra)
        snapshot = load(later)
        allocation = study.sample_families(self.freeze, FAKE_NONCE, inventory_snapshot=snapshot,
                                            current_inventory_sha256=snapshot.sha256, **self.site_kwargs)
        with self.assertRaises(ValueError):
            study.seal_heldout(self.freeze, FAKE_NONCE, self.fake_cases(), allocation=allocation,
                               inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256,
                               **self.site_kwargs)
        with self.assertRaises(ValueError):
            study.seal_heldout(self.freeze, FAKE_NONCE, self.fake_cases(), allocation=self.allocation,
                               inventory_snapshot=snapshot, current_inventory_sha256=snapshot.sha256,
                               **self.site_kwargs)

    def test_fabricated_allocation_has_exact_order_and_tie_break(self):
        allocation = study.sample_families(self.freeze, FAKE_NONCE, **self.inventory_kwargs, **self.site_kwargs)
        rows = allocation.families
        self.assertEqual([r.context for r in rows],
                         ["near the boxes", "behind the car", "behind the cars", "near the box"])
        self.assertEqual([r.subjects for r in rows],
                         [("theta", "thetas", "epsilon", "epsilons"),
                          ("beta", "betas", "eta", "etas"),
                          ("zeta", "zetas", "delta", "deltas"),
                          ("gamma", "gammas", "alpha", "alphas")])
        self.assertEqual([r.family_index for r in rows], [0, 1, 2, 3])
        self.assertNotIn("theta", repr(rows[0]))
        self.assertEqual(study.sample_families(self.freeze, FAKE_NONCE, **self.inventory_kwargs,
                                               **self.site_kwargs), allocation)
        class SameDigest:
            def digest(self):
                return b"\0" * 32
        with mock.patch.object(study.hashlib, "sha256", return_value=SameDigest()):
            self.assertEqual(study._sort_pool(b"fake", FAKE_NONCE, ("é", "b", "a")), ["a", "b", "é"])
        for nonce in (b"", b"x" * 31, b"x" * 33, bytearray(32), "0" * 64):
            with self.subTest(nonce_type=type(nonce)), self.assertRaises(ValueError):
                study.sample_families(self.freeze, nonce, **self.inventory_kwargs, **self.site_kwargs)

    def test_reviewed_v4_snapshot_binds_fabricated_seal_and_reveal(self):
        snapshot = study.load_inventory_snapshot(later_exposures.canonical_inventory_bytes(),
                                                 later_exposures.INVENTORY_SHA256)
        options = {"inventory_snapshot": snapshot, "current_inventory_sha256": snapshot.sha256}
        allocation = study.sample_families(self.freeze, FAKE_NONCE, **options, **self.site_kwargs)
        self.assertEqual(allocation.families, self.allocation.families)
        sealed = study.seal_heldout(self.freeze, FAKE_NONCE, self.fake_cases(), allocation=allocation,
                                    **options, **self.site_kwargs)
        self.assertEqual(json.loads(sealed.record_bytes)["exposure_inventory_sha256"], snapshot.sha256)
        self.assertTrue(study.verify_heldout_reveal(self.freeze, FAKE_NONCE, sealed.record_bytes,
                                                    sealed.commitment_sha256, inventory_snapshot=snapshot,
                                                    **self.site_kwargs))

    def test_site_commitment_and_reveal_follow_exact_domains(self):
        sealed = study.seal_site(FAKE_NONCE, "d" * 40)
        site = json.loads(sealed.record_bytes)
        self.assertEqual(site["layer"], 1)
        self.assertEqual(site["source_commit"], "d" * 40)
        self.assertEqual(sealed.commitment_sha256, hashlib.sha256(
            b"MLG-number-v1-site-commit\0" + FAKE_NONCE + b"\0" + sealed.record_bytes).hexdigest())
        self.assertEqual(study.verify_site_reveal(FAKE_NONCE, sealed.record_bytes,
                                                 sealed.commitment_sha256, expected_source_commit="d" * 40), 1)
        self.assertNotIn("layer", repr(sealed))
        with self.assertRaises(ValueError):
            study.verify_site_reveal(FAKE_NONCE, sealed.record_bytes,
                                     sealed.commitment_sha256, expected_source_commit="a" * 40)
        with self.assertRaises(ValueError):
            study.verify_site_reveal(FAKE_NONCE, sealed.record_bytes, "0" * 64, expected_source_commit="d" * 40)
        altered = dict(site, layer=True)
        with self.assertRaises(ValueError):
            study.verify_site_reveal(FAKE_NONCE, json.dumps(altered).encode(), sealed.commitment_sha256,
                                     expected_source_commit="d" * 40)

    def test_fabricated_heldout_commitment_binds_all_tracks_cases_and_inventory(self):
        tokenized = self.fake_cases()
        sealed = study.seal_heldout(self.freeze, FAKE_NONCE, tokenized,
                                    allocation=self.allocation, **self.inventory_kwargs, **self.site_kwargs)
        record = json.loads(sealed.record_bytes)
        self.assertEqual(record["source_commits"], {"natural": "a" * 40, "injection": "d" * 40})
        self.assertEqual(record["prediction_freeze_sha256"], self.freeze.sha256)
        self.assertEqual(len(record["families"]), 4)
        self.assertEqual(sealed.commitment_sha256, hashlib.sha256(
            b"MLG-number-v1-heldout-commit\0" + FAKE_NONCE + b"\0" + sealed.record_bytes).hexdigest())
        self.assertTrue(study.verify_heldout_reveal(self.freeze, FAKE_NONCE, sealed.record_bytes,
                                                    sealed.commitment_sha256, inventory_snapshot=self.inventory,
                                                    **self.site_kwargs))
        self.assertNotIn("theta", repr(sealed))
        for mode in ("bad_mask", "bad_token", "duplicate", "wrong_text", "unaligned", "bad_index",
                     "first_subject", "last_subject"):
            altered = copy.deepcopy(tokenized)
            if mode == "bad_mask":
                altered[0][0]["attention_mask"][0] = True
            elif mode == "bad_token":
                altered[0][0]["input_ids"][0] = 50257
            elif mode == "duplicate":
                altered[0][1]["input_ids"] = altered[0][0]["input_ids"][:]
            elif mode == "wrong_text":
                altered[0][0]["text"] += " "
            elif mode == "unaligned":
                altered[0][1]["input_ids"][2] += 1
            elif mode == "bad_index":
                altered[0][0]["subject_index"] = True
            elif mode == "first_subject":
                altered[0][0]["subject_index"] = 0
            else:
                altered[0][0]["subject_index"] = len(altered[0][0]["input_ids"]) - 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                study.seal_heldout(self.freeze, FAKE_NONCE, altered,
                                   allocation=self.allocation, **self.inventory_kwargs, **self.site_kwargs)
        with self.assertRaises(ValueError):
            study.seal_heldout(self.freeze, FAKE_NONCE, tokenized,
                               allocation=self.allocation, inventory_snapshot=self.inventory,
                               current_inventory_sha256="0" * 64, **self.site_kwargs)
        with self.assertRaises(ValueError):
            study.verify_heldout_reveal(self.freeze, FAKE_NONCE, sealed.record_bytes, "0" * 64,
                                        inventory_snapshot=self.inventory, **self.site_kwargs)
        other = study.make_prediction_freeze({"natural": TRACKS["natural"]})
        with self.assertRaises(ValueError):
            study.verify_heldout_reveal(other, FAKE_NONCE, sealed.record_bytes, sealed.commitment_sha256,
                                        inventory_snapshot=self.inventory)

    def test_site_custody_is_tied_to_injection_track_and_distinct_nonce(self):
        tokenized = self.fake_cases()
        options = {"allocation": self.allocation, **self.inventory_kwargs}
        with mock.patch.object(study, "_sort_pool", side_effect=AssertionError("allocation started")):
            with self.assertRaises(ValueError):
                study.sample_families(self.freeze, FAKE_NONCE, **self.inventory_kwargs)
            repeated = study.seal_site(FAKE_NONCE, "d" * 40)
            with self.assertRaises(ValueError):
                study.sample_families(self.freeze, FAKE_NONCE, **self.inventory_kwargs, site_nonce=FAKE_NONCE,
                                      site_record_bytes=repeated.record_bytes,
                                      site_commitment=repeated.commitment_sha256)
            wrong_source = study.seal_site(FAKE_SITE_NONCE, "a" * 40)
            with self.assertRaises(ValueError):
                study.sample_families(self.freeze, FAKE_NONCE, **self.inventory_kwargs, site_nonce=FAKE_SITE_NONCE,
                                      site_record_bytes=wrong_source.record_bytes,
                                      site_commitment=wrong_source.commitment_sha256)
        with self.assertRaises(ValueError):
            study.seal_heldout(self.freeze, FAKE_NONCE, tokenized, **options)
        repeated = study.seal_site(FAKE_NONCE, "d" * 40)
        with self.assertRaises(ValueError):
            study.seal_heldout(self.freeze, FAKE_NONCE, tokenized, **options,
                               site_nonce=FAKE_NONCE, site_record_bytes=repeated.record_bytes,
                               site_commitment=repeated.commitment_sha256)
        wrong_source = study.seal_site(FAKE_SITE_NONCE, "a" * 40)
        with self.assertRaises(ValueError):
            study.seal_heldout(self.freeze, FAKE_NONCE, tokenized, **options,
                               site_nonce=FAKE_SITE_NONCE, site_record_bytes=wrong_source.record_bytes,
                               site_commitment=wrong_source.commitment_sha256)
        colliding = copy.deepcopy(tokenized)
        for item, (_, ids) in zip(colliding[0], exposures.PUBLIC_DISCOVERY[:4]):
            item["input_ids"] = list(ids)
            item["attention_mask"] = [1] * len(ids)
            item["subject_index"] = 1
        with self.assertRaises(ValueError):
            study.seal_heldout(self.freeze, FAKE_NONCE, colliding, **options, **self.site_kwargs)
        natural_only = study.make_prediction_freeze({"natural": TRACKS["natural"]})
        natural_allocation = study.sample_families(natural_only, FAKE_NONCE, **self.inventory_kwargs)
        natural_options = {"allocation": natural_allocation, **self.inventory_kwargs}
        self.assertTrue(study.seal_heldout(natural_only, FAKE_NONCE, tokenized, **natural_options).commitment_sha256)
        with self.assertRaises(ValueError):
            study.seal_heldout(natural_only, FAKE_NONCE, tokenized, **natural_options, **self.site_kwargs)


if __name__ == "__main__":
    unittest.main()
