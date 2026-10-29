"""Fabricated CPU GPT-2 checks; no checkpoint or public input is loaded."""
import copy
import os
import subprocess
import sys
import unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
try:
    import distractor_core as core
except ModuleNotFoundError:
    core = None

class CausalDistractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        cfg = GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=5,
                         n_head=2, use_cache=False, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
        cfg._attn_implementation = 'eager'
        cls.model = GPT2LMHeadModel(cfg).eval()
        cls.tokens = torch.tensor([[1, 7, 11, 21, 31]], dtype=torch.long)

    def test_future_distractor_and_four_candidate_controls(self):
        self.assertIsNotNone(core, 'the two-position intervention core is not implemented')
        with torch.inference_mode():
            clean = core.observe_clean(self.model, self.tokens, 'fabricated-clean')
            decoy = core.intervene(self.model, self.tokens, clean['vectors'], 1, 'decoy')
            core.check_landing(decoy, self.tokens, clean['vectors'], 1, 'decoy')
            self.assertTrue(core.bitwise_equal(decoy['logits'][0], clean['logits'][0]))
            for layer in (1, 2, 3):
                target = core.intervene(self.model, self.tokens, clean['vectors'], layer, 'target')
                combined = core.intervene(self.model, self.tokens, clean['vectors'], layer, 'combined')
                core.check_landing(target, self.tokens, clean['vectors'], layer, 'target')
                core.check_landing(combined, self.tokens, clean['vectors'], layer, 'combined')
                self.assertTrue(core.bitwise_equal(target['logits'][0], combined['logits'][0]))
                mapping = core.candidate_map(layer)
                for q, item in mapping.items():
                    patched = core.intervene(self.model, self.tokens, clean['vectors'], layer, 'patch', q=q)
                    core.check_landing(patched, self.tokens, clean['vectors'], layer, 'patch', q=q)
                    if item['role'] == 'source':
                        self.assertTrue(core.bitwise_equal(patched['logits'][0], clean['logits'][0]))
                    if item['role'] in ('upstream', 'distractor'):
                        self.assertTrue(core.bitwise_equal(patched['logits'][0], combined['logits'][0]))
                for module in self.model.modules():
                    self.assertFalse(module._forward_hooks)
                    self.assertFalse(module._forward_pre_hooks)

    def test_shared_module_order_and_rotation_landing(self):
        self.assertIsNotNone(core, 'the two-position intervention core is not implemented')
        with torch.inference_mode():
            clean = core.observe_clean(self.model, self.tokens, 'fabricated-clean')
            mapping = core.candidate_map(1)
            upstream = next(q for q, x in mapping.items() if x['role'] == 'upstream')
            a = core.intervene(self.model, self.tokens, clean['vectors'], 1, 'patch', q=upstream)
            events = [e for e in a['events'] if e['module_path'] == 'transformer.h.0.mlp']
            names = [e['name'] for e in events]
            self.assertLess(names.index('decoy_add'), names.index('patch'))
            self.assertEqual(next(e['token_index'] for e in events if e['name'] == 'decoy_add'), 4)
            self.assertEqual(next(e['token_index'] for e in events if e['name'] == 'patch'), 3)
            core.check_landing(a, self.tokens, clean['vectors'], 1, 'patch', q=upstream)
            combined = core.intervene(self.model, self.tokens, clean['vectors'], 1, 'combined')
            base = combined['captures']['tap_2_t']
            rotated = core.intervene(self.model, self.tokens, clean['vectors'], 1, 'rotation', base=base)
            core.check_landing(rotated, self.tokens, clean['vectors'], 1, 'rotation', base=base)
            self.assertTrue(core.bitwise_equal(rotated['captures']['patch_before'][0, 3], base))
            planned = rotated['captures']['rotation_planned']
            self.assertTrue(core.bitwise_equal(planned, rotated['captures']['patch_after'][0, 3]))
            wrong_plan = copy.copy(rotated)
            wrong_plan['captures'] = dict(rotated['captures'])
            wrong_plan['captures']['rotation_planned'] = planned + 1
            with self.assertRaises(ValueError):
                core.check_landing(wrong_plan, self.tokens, clean['vectors'], 1, 'rotation', base=base)
            bad = copy.copy(rotated)
            bad['captures'] = dict(rotated['captures'])
            bad['captures']['patch_after'] = rotated['captures']['patch_after'].clone()
            bad['captures']['patch_after'][0, 0, 1] += 1
            with self.assertRaises(ValueError):
                core.check_landing(bad, self.tokens, clean['vectors'], 1, 'rotation', base=base)

    def test_null_self_preserves_both_scored_positions(self):
        with torch.inference_mode():
            clean = core.observe_clean(self.model, self.tokens, 'null-test')
            result = core.null_self(self.model, self.tokens)
            self.assertTrue(core.bitwise_equal(result['logits'], clean['logits']))
            self.assertTrue(core.bitwise_equal(result['captures']['before'], result['captures']['after']))
            self.assertTrue(core.bitwise_equal(result['captures']['self_source'],
                                               result['captures']['before'][0, -2]))

    def test_zero_rotation_is_undefined_not_a_false_match(self):
        self.assertIsNotNone(core, 'the two-position intervention core is not implemented')
        z, report = core.rotate_displacement(torch.zeros(8), torch.zeros(8))
        self.assertTrue(core.bitwise_equal(z, torch.zeros(8)))
        self.assertIsNone(report['cosine'])
        self.assertFalse(report['matched'])


class RoundingModeTests(unittest.TestCase):
    def test_f32_nearest_even_midpoints(self):
        self.assertEqual(core.f64_to_f32_bits(float.fromhex("0x1.000001p+0")), 0x3f800000)
        self.assertEqual(core.f64_to_f32_bits(float.fromhex("0x1.000003p+0")), 0x3f800002)
        self.assertEqual(core.f64_to_f32_bits(float.fromhex("-0x1.000001p+0")), 0xbf800000)

    @unittest.skipUnless(os.name == 'nt', 'Windows runtime test')
    def test_all_non_nearest_modes_fail_closed_in_subprocess(self):
        code = r"""
import ctypes
import torch
import distractor_core as core
control = ctypes.CDLL('msvcrt')._controlfp
control.argtypes = [ctypes.c_uint, ctypes.c_uint]
control.restype = ctypes.c_uint
base = torch.tensor([1.0, 0.0], dtype=torch.float32)
donor = torch.tensor([1.0, 2**-24], dtype=torch.float32)
before = torch.zeros((1, 2, 2), dtype=torch.float32)
after = before.clone()
after[0, 0, 0] = 4.0
captures = {'x_before': before, 'x_after': after}
old = control(0, 0)
try:
    for mode in (0x100, 0x200, 0x300):
        control(mode, 0x300)
        assert control(0, 0) & 0x300 == mode
        for label, action in (('rotate', lambda: core.rotate_displacement(base, donor)),
                              ('add witness', lambda: core._check_add(captures, 'x', 0))):
            try:
                action()
            except RuntimeError as error:
                assert 'round' in str(error).lower(), (label, str(error))
            else:
                raise AssertionError(f'{label} accepted non-nearest mode {mode:#x}')
finally:
    control(old, 0x300)
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == '__main__':
    unittest.main()
