"""Tests of the fixed research inputs and their provenance checks."""
from dataclasses import replace
import math
import unittest

from looking_glass.research_cases import (
    DISCOVERY_CASES,
    HELDOUT_CASES,
    PRETRAINED_SMOKE_CASES,
    SYNTHETIC_CASES,
    PromptCase,
    SyntheticCase,
    float32_values,
    pretrained_content_sha256,
    pretrained_smoke_sha256,
    synthetic_content_sha256,
    synthetic_suite_sha256,
    validate_case_sets,
    validate_pretrained_smoke_cases,
    validate_synthetic_cases,
)


class ResearchCaseTests(unittest.TestCase):
    def test_fixed_suites_validate_and_have_requested_independent_cases(self):
        self.assertEqual(len(DISCOVERY_CASES), 8)
        self.assertEqual(len(HELDOUT_CASES), 16)
        self.assertEqual(SYNTHETIC_CASES, DISCOVERY_CASES + HELDOUT_CASES)
        self.assertGreaterEqual(len(PRETRAINED_SMOKE_CASES), 4)
        validate_case_sets(SYNTHETIC_CASES, PRETRAINED_SMOKE_CASES)

    def test_published_v1_suites_match_their_frozen_digests(self):
        self.assertEqual(synthetic_suite_sha256(),
                         '883b28112ccee7605125959a4c8209c9e20bb438b73938b323e3fc80994f1e8e')
        self.assertEqual(pretrained_smoke_sha256(),
                         '9ce08d4a321c9e3e1cd6c165a35ec290986a9548bb960804573733f46e89dd4f')

    def test_numeric_content_hash_has_a_reproducible_canonical_form(self):
        values = (0., 1., -1., 0.5, 0.25, 0.125, -0.25, 0.75)
        # SHA-256 of UTF-8 bytes of {"kind":"numeric-vector-v1","values":
        # [0.0,1.0,-1.0,0.5,0.25,0.125,-0.25,0.75]} without whitespace.
        self.assertEqual(
            synthetic_content_sha256(values),
            'b441e8891816dd502d88e00bb4db786ca465c05d4cd695795bbd1ed7580259cc',
        )
        # Signed zero is observable by models; identities must preserve its sign.
        self.assertNotEqual(synthetic_content_sha256(values),
                            synthetic_content_sha256((-0.,) + values[1:]))
        self.assertNotEqual(synthetic_content_sha256(values),
                            synthetic_content_sha256((0.5,) + values[1:]))

    def test_public_float32_conversion_matches_hashing_and_preserves_signed_zero(self):
        values = (-0., 1. + 1e-9, 2., 3., 4., 5., 6., 7.)
        actual = float32_values(values)
        self.assertEqual(actual[1], 1.0)
        self.assertEqual(math.copysign(1., actual[0]), -1.)
        self.assertEqual(synthetic_content_sha256(values), synthetic_content_sha256(actual))

    def test_suite_hash_covers_case_identity_and_model_inputs(self):
        renamed = (replace(SYNTHETIC_CASES[0], case_id='renamed'),) + SYNTHETIC_CASES[1:]
        self.assertNotEqual(synthetic_suite_sha256(SYNTHETIC_CASES),
                            synthetic_suite_sha256(renamed))
        changed_prompt = (replace(PRETRAINED_SMOKE_CASES[0], text='Different text'),) + PRETRAINED_SMOKE_CASES[1:]
        self.assertNotEqual(pretrained_smoke_sha256(PRETRAINED_SMOKE_CASES),
                            pretrained_smoke_sha256(changed_prompt))

    def test_smoke_content_hash_uses_exact_text(self):
        self.assertEqual(pretrained_content_sha256('hello'),
                         'cb078d719ba729eb5a23c587cb662c4f165c5d237625d8fe914ccc31f91d5b19')
        self.assertNotEqual(pretrained_content_sha256('hello'), pretrained_content_sha256('hello '))
        with self.assertRaises(ValueError):
            pretrained_content_sha256('  ')

    def test_synthetic_rejects_duplicate_inputs_under_different_ids_across_splits(self):
        leaked = (DISCOVERY_CASES[0], replace(HELDOUT_CASES[0], values=DISCOVERY_CASES[0].values))
        with self.assertRaisesRegex(ValueError, 'duplicate.*(input|vector|content)'):
            validate_synthetic_cases(leaked)

    def test_synthetic_detects_overlap_after_float32_rounding_and_signed_zero(self):
        discovery = SyntheticCase('disc', 'discovery', (0., 1., 2., 3., 4., 5., 6., 7.))
        heldout = SyntheticCase('held', 'heldout', (-0., 1. + 1e-9, 2., 3., 4., 5., 6., 7.))
        with self.assertRaisesRegex(ValueError, 'duplicate.*(input|vector|content)'):
            validate_synthetic_cases((discovery, heldout))
        self.assertNotEqual(synthetic_content_sha256(discovery.values),
                            synthetic_content_sha256(heldout.values))
        rounded_only = (0., 1. + 1e-9, 2., 3., 4., 5., 6., 7.)
        self.assertEqual(synthetic_content_sha256(discovery.values),
                         synthetic_content_sha256(rounded_only))

    def test_synthetic_rejects_duplicate_within_discovery_split(self):
        repeated = (DISCOVERY_CASES[0], replace(DISCOVERY_CASES[1], values=DISCOVERY_CASES[0].values),
                    HELDOUT_CASES[0])
        with self.assertRaisesRegex(ValueError, 'duplicate.*(input|vector|content)'):
            validate_synthetic_cases(repeated)

    def test_rejects_duplicate_case_ids_even_with_different_content(self):
        leaked = (DISCOVERY_CASES[0], replace(HELDOUT_CASES[0], case_id=DISCOVERY_CASES[0].case_id))
        with self.assertRaisesRegex(ValueError, 'duplicate.*ID'):
            validate_synthetic_cases(leaked)
        same_id_prompt = (replace(PRETRAINED_SMOKE_CASES[0], case_id=DISCOVERY_CASES[0].case_id),)
        with self.assertRaisesRegex(ValueError, 'duplicate.*ID'):
            validate_case_sets(SYNTHETIC_CASES, same_id_prompt)

    def test_synthetic_rejects_incomplete_or_invalid_split(self):
        with self.assertRaises(ValueError):
            validate_synthetic_cases(DISCOVERY_CASES)
        wrong_split = replace(HELDOUT_CASES[0], split='validation')
        with self.assertRaises(ValueError):
            validate_synthetic_cases((DISCOVERY_CASES[0], wrong_split))

    def test_synthetic_rejects_nonfinite_nonnumeric_and_mutable_vectors(self):
        original = HELDOUT_CASES[0]
        invalid = [
            (0.,) * 7,
            (0.,) * 7 + (float('nan'),),
            (0.,) * 7 + (float('inf'),),
            (0.,) * 7 + (1e100,),
            (0.,) * 7 + (True,),
            [0.] * 8,
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_synthetic_cases((DISCOVERY_CASES[0], replace(original, values=values)))

    def test_smoke_inputs_reject_empty_or_duplicate_prompts(self):
        for cases in ((), (PromptCase('prompt-a', ' '),),
                      (PromptCase('prompt-a', 'hello'), PromptCase('prompt-b', 'hello'))):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                validate_pretrained_smoke_cases(cases)


if __name__ == '__main__':
    unittest.main()
