"""CPU-only mechanism checks; these never load the pretrained checkpoint."""

import unittest
from unittest.mock import patch

import torch

from looking_glass.copy_head_intervention import (
    HeadProjectionWitness, ProjectionWitnessError, assert_no_forward_hooks,
)


class HeadProjectionWitnessTests(unittest.TestCase):
    def test_tiny_gpt2_head_layout_and_unhooked_controls(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from transformers.models.gpt2 import modeling_gpt2

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(71)
            config = GPT2Config(vocab_size=29, n_positions=8, n_embd=8, n_layer=1,
                                n_head=2, resid_pdrop=0, embd_pdrop=0,
                                attn_pdrop=0, use_cache=False)
            config._attn_implementation = "eager"
            model = GPT2LMHeadModel(config).eval()
        projection = model.transformer.h[0].attn.c_proj
        ids = torch.tensor([[1, 4, 7, 2]], dtype=torch.int64)
        reference = []
        original = modeling_gpt2.eager_attention_forward

        def capture_head_shaped(*args, **kwargs):
            output, weights = original(*args, **kwargs)
            reference.append(output.detach().clone())
            return output, weights

        assert_no_forward_hooks(model)
        with torch.inference_mode():
            clean = model(input_ids=ids, use_cache=False).logits.clone()
            with patch.object(modeling_gpt2, "eager_attention_forward", capture_head_shaped):
                with HeadProjectionWitness(projection, sequence_length=4, width=8,
                                           heads=2, mode="noop") as noop:
                    noop_logits = model(input_ids=ids, use_cache=False).logits.clone()
            with HeadProjectionWitness(projection, sequence_length=4, width=8,
                                       heads=2, mode="zero", head=1) as zero:
                model(input_ids=ids, use_cache=False)
            assert_no_forward_hooks(model)
            after = model(input_ids=ids, use_cache=False).logits.clone()

        self.assertEqual(len(reference), 1)
        self.assertTrue(torch.equal(noop.before.view(1, 4, 2, 4), reference[0]))
        self.assertTrue(torch.equal(noop.before, noop.after))
        self.assertTrue(torch.equal(noop.before, zero.before))
        self.assertTrue(torch.equal(zero.after[:, :-1], zero.before[:, :-1]))
        self.assertTrue(torch.equal(zero.after[:, -1, :4], zero.before[:, -1, :4]))
        self.assertTrue(torch.equal(zero.after[:, -1, 4:], torch.zeros(1, 4)))
        self.assertTrue(torch.equal(clean.view(torch.int32), noop_logits.view(torch.int32)))
        self.assertTrue(torch.equal(clean.view(torch.int32), after.view(torch.int32)))

    def test_failure_and_missing_call_remove_hooks(self):
        projection = torch.nn.Linear(8, 8).eval()
        with self.assertRaisesRegex(ValueError, "forward failed"):
            with HeadProjectionWitness(projection, sequence_length=4, width=8,
                                       heads=2, mode="noop"):
                raise ValueError("forward failed")
        self.assertFalse(projection._forward_hooks)
        self.assertFalse(projection._forward_pre_hooks)
        with self.assertRaises(ProjectionWitnessError):
            with HeadProjectionWitness(projection, sequence_length=4, width=8,
                                       heads=2, mode="noop"):
                pass
        self.assertFalse(projection._forward_hooks)
        self.assertFalse(projection._forward_pre_hooks)

    def test_foreign_hook_and_wrong_shape_are_rejected(self):
        projection = torch.nn.Linear(8, 8).eval()
        handle = projection.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaises(ProjectionWitnessError):
                with HeadProjectionWitness(projection, sequence_length=4, width=8,
                                           heads=2, mode="noop"):
                    pass
        finally:
            handle.remove()
        with self.assertRaises(ProjectionWitnessError):
            with HeadProjectionWitness(projection, sequence_length=4, width=8,
                                       heads=2, mode="zero", head=1):
                projection(torch.zeros(1, 3, 8))
        self.assertFalse(projection._forward_hooks)
        self.assertFalse(projection._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
