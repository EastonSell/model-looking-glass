"""CPU-only software controls on a random model, not causal-study evidence."""

import importlib.util
import unittest

import torch


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'transformers is optional')
class RandomGPT2PlanReferenceTests(unittest.TestCase):
    """Compare a six-block ordered plan with separately written serial hooks."""

    def test_injection_and_patch_controls_match_serial_reference(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from looking_glass.intervention_plan import ActivationPlan

        config = GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=6, n_head=2,
                            use_cache=False, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
        config._attn_implementation = 'eager'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(310)
            model = GPT2LMHeadModel(config).eval()
        tokens = torch.tensor([[1, 7, 11, 21, 31, 10],
                               [2, 9, 13, 17, 23, 35]], dtype=torch.long)
        token = tokens.shape[1] - 1
        delta = torch.linspace(-0.8, 0.6, config.n_embd)

        def same_float32_bits(left, right):
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.dtype, torch.float32)
            self.assertEqual(right.dtype, torch.float32)
            return torch.equal(left.contiguous().view(torch.int32),
                               right.contiguous().view(torch.int32))

        def clean_reference():
            observations, handles = {}, []
            try:
                for layer, block in enumerate(model.transformer.h):
                    def record_mlp(module, inputs, output, layer=layer):
                        observations[f'mlp_{layer}'] = output[0, token].detach().clone()
                    handles.append(block.mlp.register_forward_hook(record_mlp))
                def record_block(module, inputs, output):
                    self.assertIs(type(output), tuple)
                    self.assertGreater(len(output), 1)
                    observations['last_block'] = output[0][0, token].detach().clone()
                handles.append(model.transformer.h[5].register_forward_hook(record_block))
                logits = model(tokens, use_cache=False, output_attentions=True).logits.detach().clone()
                return observations, logits
            finally:
                for handle in reversed(handles):
                    handle.remove()

        def serial_reference(mode, clean):
            """Independent hook path: never reads ActivationPlan output."""
            observations, handles = {}, []
            upstream = clean['mlp_1'] + torch.linspace(0.02, 0.16, config.n_embd)
            try:
                for layer, block in enumerate(model.transformer.h):
                    def apply_mlp(module, inputs, output, layer=layer):
                        current = output
                        changed = False
                        if mode in ('upstream_clean', 'upstream_perturb', 'upstream_null') and layer == 1:
                            current = current.clone()
                            if mode == 'upstream_clean':
                                current[0, token] = clean['mlp_1']
                            elif mode == 'upstream_perturb':
                                current[0, token] = upstream
                            else:
                                current[0, token] = current[0, token]
                            changed = True
                        if layer == 2:
                            current = current.clone()
                            current[0, token] = current[0, token] + delta
                            changed = True
                        observations[f'mlp_{layer}'] = current[0, token].detach().clone()
                        if layer == 2 and mode in ('same_site', 'self_patch'):
                            current = current.clone()
                            donor = clean['mlp_2'] if mode == 'same_site' else observations['mlp_2']
                            current[0, token] = donor
                            observations['after_local_patch'] = current[0, token].detach().clone()
                            changed = True
                        return current if changed else None
                    handles.append(block.mlp.register_forward_hook(apply_mlp))

                def apply_final_block(module, inputs, output):
                    self.assertIs(type(output), tuple)
                    self.assertGreater(len(output), 1)
                    current = output[0]
                    observations['last_before'] = current[0, token].detach().clone()
                    if mode == 'final_block':
                        current = current.clone()
                        current[0, token] = clean['last_block']
                        observations['last_after'] = current[0, token].detach().clone()
                        return (current,) + output[1:]
                    return None
                handles.append(model.transformer.h[5].register_forward_hook(apply_final_block))
                result = model(tokens, use_cache=False, output_attentions=True)
                logits = result.logits.detach().clone()
                attention = tuple(value.detach().clone() for value in result.attentions)
                return observations, logits, attention
            finally:
                for handle in reversed(handles):
                    handle.remove()

        def plan_steps(mode, clean):
            steps = []
            for layer in range(6):
                site = {'module_path': f'transformer.h.{layer}.mlp',
                        'batch_index': 0, 'token_index': token}
                if mode == 'upstream_clean' and layer == 1:
                    steps.append(dict(site, name='upstream', operation='patch', value=clean['mlp_1']))
                if mode == 'upstream_perturb' and layer == 1:
                    steps.append(dict(site, name='upstream', operation='patch',
                                      value=clean['mlp_1'] + torch.linspace(0.02, 0.16, config.n_embd)))
                if mode == 'upstream_null' and layer == 1:
                    steps.append(dict(site, name='upstream', operation='null'))
                if layer == 2:
                    steps.append(dict(site, name='injection', operation='add', value=delta))
                steps.append(dict(site, name=f'mlp_{layer}', operation='capture'))
                if layer == 2 and mode == 'same_site':
                    steps.append(dict(site, name='local_patch', operation='patch', value=clean['mlp_2']))
                    steps.append(dict(site, name='after_local_patch', operation='capture'))
                if layer == 2 and mode == 'self_patch':
                    steps.append(dict(site, name='local_patch', operation='patch', source_capture='mlp_2'))
                    steps.append(dict(site, name='after_local_patch', operation='capture'))
            last = {'module_path': 'transformer.h.5', 'batch_index': 0, 'token_index': token}
            steps.append(dict(last, name='last_before', operation='capture'))
            if mode == 'final_block':
                steps.append(dict(last, name='final_patch', operation='patch', value=clean['last_block']))
                steps.append(dict(last, name='last_after', operation='capture'))
            return steps

        previous_threads = torch.get_num_threads()
        try:
            # These tiny CPU matrices are not a throughput benchmark.
            torch.set_num_threads(1)
            with torch.no_grad():
                clean, clean_logits = clean_reference()
                seen = {}
                for mode in ('injection', 'same_site', 'final_block', 'upstream_clean',
                             'upstream_perturb', 'upstream_null', 'self_patch'):
                    with self.subTest(mode=mode):
                        expected_captures, expected_logits, expected_attention = serial_reference(mode, clean)
                        steps = plan_steps(mode, clean)
                        plan = ActivationPlan(model, steps, max_invocations=1)
                        with plan:
                            result = model(tokens, use_cache=False, output_attentions=True)
                            logits = result.logits.detach().clone()
                            attention = tuple(value.detach().clone() for value in result.attentions)
                        self.assertTrue(same_float32_bits(logits, expected_logits))
                        self.assertEqual(len(attention), len(expected_attention))
                        self.assertEqual(len(attention), 6)
                        for actual, expected in zip(attention, expected_attention):
                            self.assertTrue(same_float32_bits(actual, expected))
                        self.assertEqual(set(plan.captures), set(expected_captures))
                        for name, expected in expected_captures.items():
                            self.assertTrue(same_float32_bits(plan.captures[name], expected), name)
                        self.assertEqual([event['name'] for event in plan.metadata['steps']],
                                         [step['name'] for step in steps])
                        self.assertEqual(plan.metadata['steps'][-1]['output_selector'], [0])
                        self.assertTrue(same_float32_bits(logits[:, :token], clean_logits[:, :token]))
                        self.assertTrue(same_float32_bits(logits[1], clean_logits[1]))
                        for module in model.modules():
                            self.assertEqual(len(module._forward_hooks), 0)
                            self.assertEqual(len(module._forward_pre_hooks), 0)
                        seen[mode] = logits
                self.assertFalse(torch.equal(seen['injection'], clean_logits))
                self.assertTrue(same_float32_bits(seen['same_site'], clean_logits))
                self.assertTrue(same_float32_bits(seen['final_block'], clean_logits))
                self.assertTrue(same_float32_bits(seen['upstream_clean'], seen['injection']))
                self.assertTrue(same_float32_bits(seen['upstream_null'], seen['injection']))
                self.assertFalse(same_float32_bits(seen['upstream_perturb'], seen['injection']))
                self.assertTrue(same_float32_bits(seen['self_patch'], seen['injection']))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
