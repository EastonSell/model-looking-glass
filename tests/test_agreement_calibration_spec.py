"""CPU checks of declared agreement-plan payloads, not GPU memory bounds."""
import unittest

from looking_glass.agreement_calibration_spec import (
    CALIBRATION_SEED, DONOR_INPUT_IDS, LAST_TOKEN_INDEX, MIN_MARGIN_BPS, MIN_MARGIN_BYTES,
    MIN_TRACE_ALLOWANCE_BYTES, SAMPLE_KINDS, SUBJECT_INDEX, TARGET_INPUT_IDS,
    logical_payloads, validate_plan_limits, validate_profile_requirements)


class AgreementCalibrationSpecTests(unittest.TestCase):
    def test_fixed_payload_arithmetic_for_clean_and_actions(self):
        clean = logical_payloads('clean_capture')
        self.assertEqual((clean.input_and_mask_bytes, clean.selected_vector_bytes,
                          clean.whole_activation_bytes, clean.full_logits_bytes),
                         (80, 3072, 15360, 1005140))
        self.assertEqual((clean.plan_capture_bytes, clean.plan_source_clone_bytes,
                          clean.patch_output_clone_bytes, clean.plan_steps),
                         (30720, 0, 0, 10))
        baseline = logical_payloads('baseline')
        self.assertEqual((baseline.plan_capture_bytes, baseline.plan_source_clone_bytes,
                          baseline.patch_output_clone_bytes, baseline.plan_steps), (0, 0, 0, 0))
        for action in ('self_patch', 'cross_patch', 'null'):
            with self.subTest(action=action):
                payload = logical_payloads(action)
                self.assertEqual((payload.plan_capture_bytes, payload.plan_steps), (36864, 5))
                self.assertEqual(payload.patch_output_clone_bytes, 15360)
                self.assertEqual(payload.plan_source_clone_bytes, 3072 if action == 'cross_patch' else 0)

    def test_required_plan_caps_reject_one_byte_or_one_step_too_small(self):
        for action in ('baseline', 'clean_capture', 'self_patch', 'cross_patch', 'null'):
            payload = logical_payloads(action)
            caps = dict(max_tensor_bytes=max(1, payload.plan_max_tensor_bytes),
                        max_capture_bytes=max(1, payload.plan_capture_bytes),
                        max_source_bytes=max(1, payload.plan_source_clone_bytes),
                        max_steps=max(1, payload.plan_steps), max_invocations=1,
                        max_output_bytes=payload.full_logits_bytes)
            validate_plan_limits(action, **caps)
            for key in caps:
                with self.subTest(action=action, cap=key):
                    smaller = dict(caps)
                    smaller[key] -= 1
                    with self.assertRaises(ValueError):
                        validate_plan_limits(action, **smaller)
            for key in caps:
                with self.subTest(action=action, noninteger=key):
                    invalid = dict(caps)
                    invalid[key] = True
                    with self.assertRaises(ValueError):
                        validate_plan_limits(action, **invalid)

    def test_plan_description_agrees_with_actual_cpu_executor(self):
        from looking_glass.agreement_patch_runner import (
            AgreementCase, _RNG_SEED, _SITES, _steps_for_action, _steps_for_capture)
        case = AgreementCase('case', 'quartet', 'lemma', 'singular', (1, 2, 3, 4, 5), 1)
        clean = _steps_for_capture(case)
        self.assertEqual(_RNG_SEED, CALIBRATION_SEED)
        self.assertEqual(len(_SITES), 10)
        self.assertEqual([(site['layer'], site['component'], site['token_role'], site['output_selector'])
                          for site in _SITES],
                         [(layer, component, 'last', [0] if component == 'attn' else [])
                          for layer in (1, 2, 3, 4) for component in ('attn', 'mlp')]
                         + [(5, 'block', 'last', [0]), (5, 'block', 'subject', [0])])
        self.assertEqual(len(clean), logical_payloads('clean_capture').plan_steps)
        self.assertEqual(sum(1 for step in clean if step['operation'] == 'capture'), 10)
        for site in _SITES:
            for mode, operation, donor in (('self_patch', 'patch', None),
                                           ('cross_patch', 'patch', object()),
                                           ('null', 'null', None)):
                with self.subTest(site=site['id'], mode=mode):
                    steps = _steps_for_action(case, site, operation, donor)
                    self.assertEqual(len(steps), logical_payloads(mode).plan_steps)
                    captures = [step for step in steps if step['operation'] == 'capture']
                    vectors = sum(1 if 'token_index' in step else 5 for step in captures)
                    self.assertEqual(vectors * 768 * 4, logical_payloads(mode).plan_capture_bytes)
                    self.assertEqual('value' in steps[2], mode == 'cross_patch')
                    self.assertEqual(steps[2].get('source_capture'),
                                     'before' if mode == 'self_patch' else None)

    def test_unknown_mode_is_rejected(self):
        for mode in ('', 'capture', 'cross', None, True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                logical_payloads(mode)

    def test_profile_policy_requires_13_ordered_samples_and_minimums(self):
        self.assertEqual(len(SAMPLE_KINDS), 13)
        self.assertEqual(TARGET_INPUT_IDS, (464, 3797, 1474, 262, 9168))
        self.assertEqual(DONOR_INPUT_IDS, (464, 11875, 1474, 262, 9168))
        self.assertEqual((SUBJECT_INDEX, LAST_TOKEN_INDEX), (1, 4))
        settings = dict(sample_kinds=SAMPLE_KINDS, seed=CALIBRATION_SEED,
                        gpu_trace_bytes=MIN_TRACE_ALLOWANCE_BYTES,
                        host_trace_bytes=MIN_TRACE_ALLOWANCE_BYTES,
                        margin_bytes=MIN_MARGIN_BYTES, margin_bps=MIN_MARGIN_BPS)
        validate_profile_requirements(**settings)
        for key in ('gpu_trace_bytes', 'host_trace_bytes', 'margin_bytes', 'margin_bps'):
            with self.subTest(key=key):
                bad = dict(settings)
                bad[key] -= 1
                with self.assertRaises(ValueError): validate_profile_requirements(**bad)
                bad[key] = True
                with self.assertRaises(ValueError): validate_profile_requirements(**bad)
        for samples in (SAMPLE_KINDS[:-1], SAMPLE_KINDS[::-1], list(SAMPLE_KINDS),
                        SAMPLE_KINDS[:-1] + (SAMPLE_KINDS[0],)):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                validate_profile_requirements(**{**settings, 'sample_kinds': samples})
        with self.assertRaises(ValueError):
            validate_profile_requirements(**{**settings, 'seed': CALIBRATION_SEED + 1})
        for mode in ('baseline', 'clean_capture', 'self_patch', 'cross_patch', 'null'):
            validate_plan_limits(mode, max_tensor_bytes=1024 ** 2, max_capture_bytes=512 * 1024,
                                 max_source_bytes=128 * 1024, max_steps=16,
                                 max_invocations=1, max_output_bytes=1005140)


if __name__ == '__main__':
    unittest.main()
