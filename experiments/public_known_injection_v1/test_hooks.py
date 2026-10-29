"""Fabricated, freshly initialized GPT-2 controls; no pretrained model is loaded."""
import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from hook_core import (UndeliveredInjection, observe_clean, perform_action,
                       perform_null_self, check_landing, bitwise_equal)


class OrderedInjectionTests(unittest.TestCase):
    def test_rounded_away_addition_is_not_counted_as_injection(self):
        before = torch.zeros((1, 2, 8), dtype=torch.float32)
        before[0, -1, 0] = 2.0 ** 28
        same = before.clone()
        with self.assertRaises(UndeliveredInjection):
            check_landing({'captures': {'inject_before': before, 'inject_after': same}}, 'injected')

    def test_returned_logits_recovered_when_context_exit_fails(self):
        class ExitFailure:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                raise MemoryError('fabricated exit boundary')

        class FakeOutput:
            logits = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

        class FakeModel:
            def __call__(self, **kwargs):
                return FakeOutput()

        recovered = []
        with patch('hook_core.GPT2TraceSession', ExitFailure):
            with self.assertRaisesRegex(MemoryError, 'fabricated exit boundary'):
                observe_clean(FakeModel(), torch.tensor([[1, 2, 3]]), 'fake',
                              lambda logits, error: recovered.append(logits.clone()))
        self.assertEqual(len(recovered), 1)
        self.assertTrue(bitwise_equal(recovered[0], FakeOutput.logits[0, -1]))

    def test_injected_tap_is_after_add_and_controls_land(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            config = GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=6,
                                n_head=2, use_cache=False, attn_pdrop=0,
                                resid_pdrop=0, embd_pdrop=0)
            config._attn_implementation = 'eager'
            torch.manual_seed(71)
            model = GPT2LMHeadModel(config).eval()
            tokens = torch.tensor([[1, 7, 11, 21, 31]], dtype=torch.long)
            with torch.inference_mode():
                clean = observe_clean(model, tokens, 'fake-clean')
                null = perform_null_self(model, tokens)
                self.assertTrue(bitwise_equal(null['logits'], clean['logits']))
                self.assertTrue(bitwise_equal(null['captures']['before'], null['captures']['after']))
                for site in (1, 4):
                    injected = perform_action(model, tokens, clean['vectors'], site, 'injected')
                    check_landing(injected, 'injected')
                    first = next(layer for layer in (1, 2, 3, 4)
                                 if not bitwise_equal(injected['captures'][f'tap_{layer}'],
                                                      clean['vectors'][layer]))
                    self.assertEqual(first, site)
                    self.assertTrue(bitwise_equal(injected['captures'][f'tap_{site}'],
                                                  injected['captures']['inject_after'][0, -1]))
                    for candidate in (1, 2, 3, 4):
                        patched = perform_action(model, tokens, clean['vectors'], site,
                                                 'patch', candidate)
                        check_landing(patched, 'patch', clean['vectors'], candidate)
                        if candidate == site:
                            self.assertTrue(bitwise_equal(patched['logits'], clean['logits']))
                        if candidate < site:
                            self.assertTrue(bitwise_equal(patched['logits'], injected['logits']))
                    self_action = perform_action(model, tokens, clean['vectors'], site, 'self')
                    check_landing(self_action, 'self')
                    self.assertTrue(bitwise_equal(self_action['logits'], injected['logits']))
                    bad = dict(injected, captures=dict(injected['captures']))
                    bad['captures']['inject_after'] = injected['captures']['inject_after'].clone()
                    bad['captures']['inject_after'][0, 0, 1] += 1.0
                    with self.assertRaises(ValueError):
                        check_landing(bad, 'injected')
                for module in model.modules():
                    self.assertFalse(module._forward_hooks)
                    self.assertFalse(module._forward_pre_hooks)
        finally:
            torch.set_num_threads(old)


if __name__ == '__main__':
    unittest.main()
