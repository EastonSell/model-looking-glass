"""CPU-only position-specific head witness checks on a synthetic GPT-2."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from looking_glass.copy_head_followup import ARMS, CONTROL_HASHES, action_order, arm_witness, selected_controls, summarize
from looking_glass.copy_head_intervention import HeadProjectionWitness, assert_no_forward_hooks


class PositionWitnessTests(unittest.TestCase):
    def test_earlier_position_matches_head_shaped_reference_and_preserves_bits(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from transformers.models.gpt2 import modeling_gpt2

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(137)
            config = GPT2Config(vocab_size=31, n_positions=8, n_embd=8, n_layer=2,
                                n_head=2, resid_pdrop=0, embd_pdrop=0,
                                attn_pdrop=0, use_cache=False)
            config._attn_implementation = "eager"
            model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        projection = model.transformer.h[0].attn.c_proj
        output_by_head = []
        eager = modeling_gpt2.eager_attention_forward

        def capture(*args, **kwargs):
            output, weights = eager(*args, **kwargs)
            output_by_head.append(output.detach().clone())
            return output, weights

        with torch.inference_mode():
            clean = model(input_ids=ids, use_cache=False).logits.clone()
            with patch.object(modeling_gpt2, "eager_attention_forward", capture):
                with HeadProjectionWitness(projection, sequence_length=4, width=8, heads=2,
                                           mode="zero", head=1, position=1) as witness:
                    ablated = model(input_ids=ids, use_cache=False).logits.clone()
            assert_no_forward_hooks(model)
            clean_after = model(input_ids=ids, use_cache=False).logits.clone()

        self.assertEqual(len(output_by_head), 2)
        self.assertTrue(torch.equal(witness.before.view(1, 4, 2, 4), output_by_head[0]))
        expected = witness.before.clone()
        expected[0, 1, 4:8] = 0.0
        self.assertTrue(torch.equal(witness.after.view(torch.int32), expected.view(torch.int32)))
        self.assertTrue(torch.equal(clean[:, 0, :].view(torch.int32), ablated[:, 0, :].view(torch.int32)))
        self.assertFalse(torch.equal(clean[:, -1, :].view(torch.int32), ablated[:, -1, :].view(torch.int32)))
        self.assertTrue(torch.equal(clean.view(torch.int32), clean_after.view(torch.int32)))

    def test_invalid_position_fails_before_hook_installation(self):
        projection = torch.nn.Linear(8, 8)
        for position in (-1, 4, True, 0.5):
            with self.subTest(position=position), self.assertRaises(ValueError):
                HeadProjectionWitness(projection, sequence_length=4, width=8, heads=2,
                                      mode="zero", head=1, position=position)
        with self.assertRaises(ValueError):
            HeadProjectionWitness(projection, sequence_length=4, width=8, heads=2,
                                  mode="noop", position=1)
        self.assertFalse(projection._forward_pre_hooks)
        self.assertFalse(projection._forward_hooks)

    def test_rotation_covers_all_positions_for_each_variant(self):
        self.assertEqual(selected_controls(), ((CONTROL_HASHES[0], 3), (CONTROL_HASHES[1], 1)))
        names = tuple(name for name, _, _ in ARMS)
        for variant in range(4):
            first_arms = set()
            for family in range(8):
                order = action_order(family, variant)
                self.assertEqual(order[:2], ("clean", "noop"))
                self.assertEqual(order[-1], "clean_after")
                self.assertEqual(set(order[2:-1]), set(names))
                first_arms.add(order[2])
            self.assertEqual(first_arms, set(names))
        for invalid in (True, -1, 8):
            with self.assertRaises(ValueError):
                action_order(invalid, 0)

    def test_position_witness_and_negative_paired_contrasts(self):
        before = np.arange(1, 1 + 11 * 768, dtype="<f4").reshape(1, 11, 768)
        after = before.copy()
        after[0, 5, 640:704] = 0
        witness = arm_witness("h10_target", {"cproj_before": before.tobytes(),
                                             "cproj_after": after.tobytes()})
        self.assertEqual((witness["head"], witness["position"], witness["channels"]), (10, 5, [640, 704]))
        self.assertEqual(witness["changed_float32_bit_patterns"], 64)
        self.assertGreater(witness["pre_head_l2"], 0)
        records = []
        for family in range(8):
            for variant in range(4):
                for action in action_order(family, variant):
                    delta = (-2 if variant == 0 else -1) if action == "h10_final" else 0
                    records.append({"action": action, "target_id": 1,
                                    "score": {"target_log_probability": delta, "target_rank": 1,
                                              "top_token_id": 1}})
        summary = summarize(records)
        self.assertEqual(summary["mean_contrast_by_arm"]["h10_final"], -1)
        self.assertEqual(summary["mean_candidate_minus_arm"]["h10_target"], -1)
        self.assertEqual(summary["top1_by_arm"]["h10_target"], {"repeated": 8, "shuffled": 24})
        self.assertEqual(len(summary["families"]), 8)


if __name__ == "__main__":
    unittest.main()
