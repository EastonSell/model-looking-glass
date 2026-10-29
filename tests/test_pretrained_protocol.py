"""CPU-only tests of the preregistration's public and private records."""
import copy
import json
import unittest

from looking_glass import pretrained_protocol as protocol


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.discovery_ids = [[100 + i] for i in range(8)]
        self.holdout_ids = [[200 + i] for i in range(8)]
        self.options = {
            "site": 2,
            "site_nonce": bytes(range(32)),
            "split_nonce": bytes(range(32, 64)),
            "source_commit": "1" * 40,
            "protocol_sha256": "2" * 64,
            "created_utc": "2026-10-28T20:00:00Z",
            "runtime": {"python": "3.11.15", "torch": "2.8.0+cu128",
                        "transformers": "4.57.3", "device": "cpu",
                        "driver": "none", "deterministic_settings": "test"},
        }

    def seal(self):
        return protocol.seal_manifest(self.discovery_ids, self.holdout_ids, **self.options)

    def test_manifest_is_reproducible_and_reveal_verifies(self):
        public, reveal = self.seal()
        self.assertEqual((public, reveal), self.seal())
        self.assertTrue(protocol.verify_reveal(public, reveal))
        self.assertEqual(len(protocol.canonical_sha256(public)), 64)
        self.assertEqual(public["model"]["revision"], "2290a62682d06624634c1f46a6ad5be0f47f38aa")
        self.assertEqual(len(set(reveal["heldout_prompts"])), 8)

    def test_public_record_does_not_reveal_nonces_or_selected_prompts(self):
        public, reveal = self.seal()
        text = json.dumps(public)
        self.assertNotIn(reveal["site_nonce"], text)
        self.assertNotIn(reveal["split_nonce"], text)
        for prompt in reveal["heldout_prompts"]:
            self.assertNotIn(prompt, text)
        self.assertNotIn("site", public["injection"])

    def test_mutating_site_data_or_public_configuration_breaks_reveal(self):
        public, reveal = self.seal()
        for kind in ("site", "site_nonce", "split_nonce", "heldout_ids", "runtime"):
            altered_public, altered_reveal = copy.deepcopy(public), copy.deepcopy(reveal)
            if kind == "site":
                altered_reveal["site"] = 3
            elif kind == "site_nonce":
                altered_reveal["site_nonce"] = "00" * 32
            elif kind == "split_nonce":
                altered_reveal["split_nonce"] = "ff" * 32
            elif kind == "heldout_ids":
                altered_reveal["heldout_token_ids"][0][0] += 1
            else:
                altered_public["runtime"]["torch"] = "different"
            self.assertFalse(protocol.verify_reveal(altered_public, altered_reveal), kind)

    def test_invalid_sites_nonces_and_token_ids_are_rejected(self):
        for key, value in (("site", 0), ("site", True), ("site_nonce", b"short"),
                           ("split_nonce", b"short"), ("protocol_sha256", "x" * 64)):
            options = dict(self.options)
            options[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                protocol.seal_manifest(self.discovery_ids, self.holdout_ids, **options)
        for bad_ids in ([[]] * 8, [[True]] * 8, [[50257]] * 8, [[1] * 33] * 8, [[1]] * 7):
            with self.subTest(ids=bad_ids), self.assertRaises(ValueError):
                protocol.seal_manifest(self.discovery_ids, bad_ids, **self.options)


if __name__ == "__main__":
    unittest.main()
