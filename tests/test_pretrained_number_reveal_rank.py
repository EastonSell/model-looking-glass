"""Post-reveal ranking uses fabricated records and no model forwards."""
import hashlib
import json
import unittest

from looking_glass import pretrained_number_selection as selection
from test_pretrained_number_selection import fabricated_cases


class RevealRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prediction = selection.freeze_discovery(
            fabricated_cases(), private_controls_verified=True, execution_device="cpu"
        )
        cls.digest = hashlib.sha256(cls.prediction).hexdigest()

    def test_uses_the_frozen_ranking_for_a_one_based_rank(self):
        # The fabricated first difference is layer 2; the vote tie among the
        # others is resolved in ascending layer order: [2, 1, 3, 4].
        lookup = getattr(selection, "revealed_site_rank", None)
        self.assertTrue(callable(lookup), "post-reveal rank lookup is missing")
        for site, expected in ((2, 1), (1, 2), (3, 3), (4, 4)):
            with self.subTest(site=site):
                self.assertEqual(
                    lookup(self.prediction, prediction_sha256=self.digest,
                           execution_device="cpu", revealed_layer=site),
                    expected,
                )

    def test_rejects_a_non_layer_or_malformed_ranking(self):
        for site in (None, False, True, 0, 5, 1.0, "1"):
            with self.subTest(site=site), self.assertRaises(selection.EvaluationInvalid):
                selection.revealed_site_rank(
                    self.prediction, prediction_sha256=self.digest,
                    execution_device="cpu", revealed_layer=site,
                )
        for ranking in ([1, 1, 2, 3], [1, 2, 3], [1, 2, 3, 5], [True, 2, 3, 4]):
            with self.subTest(ranking=ranking), self.assertRaises(selection.EvaluationInvalid):
                record = json.loads(self.prediction)
                record["ranking"] = ranking
                raw = json.dumps(record, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False).encode("utf-8")
                selection.revealed_site_rank(
                    raw, prediction_sha256=hashlib.sha256(raw).hexdigest(),
                    execution_device="cpu", revealed_layer=1,
                )

    def test_requires_the_bound_immutable_prediction_and_device(self):
        for raw, digest, device in (
                (self.prediction, "0" * 64, "cpu"),
                (bytearray(self.prediction), self.digest, "cpu"),
                (self.prediction + b" ", hashlib.sha256(self.prediction + b" ").hexdigest(), "cpu"),
                (self.prediction, self.digest, "cuda:0"),
                (self.prediction, self.digest, "unrecognized")):
            with self.subTest(device=device, digest=digest), self.assertRaises(selection.EvaluationInvalid):
                selection.revealed_site_rank(
                    raw, prediction_sha256=digest, execution_device=device, revealed_layer=1
                )

    def test_zero_vote_ranking_has_a_position_but_still_abstains(self):
        raw = selection.freeze_discovery(
            fabricated_cases(changed=False), private_controls_verified=True, execution_device="cpu"
        )
        self.assertIsNone(json.loads(raw)["first_changed"])
        self.assertEqual(
            selection.revealed_site_rank(raw, prediction_sha256=hashlib.sha256(raw).hexdigest(),
                                         execution_device="cpu", revealed_layer=3), 3
        )


if __name__ == "__main__":
    unittest.main()
