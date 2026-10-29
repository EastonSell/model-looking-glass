"""Family-final request ordering uses the existing site-independent IDs."""
import hashlib
import json
import unittest

from looking_glass.pretrained_number_archive import _ROLES
from looking_glass.pretrained_number_requests import (create_family_request_plan, create_request_plan,
                                                       load_family_request_plan, verify_request)


class FamilyRequestPlanTests(unittest.TestCase):
    def test_full_schedule_places_both_cleanup_checks_after_family_actions(self):
        entropy, commit = bytes(range(32)), "a" * 40
        old = json.loads(create_request_plan(entropy, commit))["requests"]
        raw = create_family_request_plan(entropy, commit)
        plan = load_family_request_plan(raw, expected_source_commit=commit,
                                        expected_sha256=hashlib.sha256(raw).hexdigest())
        rows = json.loads(raw)["requests"]
        old_by_key = {(row["case_index"], row["role"]): row for row in old}
        expected = []
        for family in range(4):
            first, second = 2 * family, 2 * family + 1
            expected.extend((case, role) for case in (first, second) for role in _ROLES[:-1])
            expected.extend(((first, "verify_clean"), (second, "verify_clean")))
        self.assertEqual([(row["case_index"], row["role"]) for row in rows], expected)
        self.assertEqual(len(rows), 128)
        for row in rows:
            key = (row["case_index"], row["role"])
            self.assertEqual(row, old_by_key[key])
            self.assertTrue(verify_request(plan, *key, row["request_id"], row["input_sha256"],
                                           expected_source_commit=commit, expected_sha256=plan.sha256))
        self.assertNotIn(b"site_nonce", raw)
        self.assertNotIn(b"hidden_site", raw)

    def test_case_major_v1_is_not_accepted_as_family_final(self):
        raw = create_request_plan(bytes(range(32)), "a" * 40)
        with self.assertRaises(ValueError):
            load_family_request_plan(raw, expected_source_commit="a" * 40,
                                     expected_sha256=hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main()
