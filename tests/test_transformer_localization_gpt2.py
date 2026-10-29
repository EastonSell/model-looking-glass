"""Offline integration checks, using random GPT-2 weights and no downloads."""
from dataclasses import replace
import importlib.util
import unittest

import torch

TRANSFORMERS_INSTALLED = importlib.util.find_spec("transformers") is not None
if TRANSFORMERS_INSTALLED:
    with torch.device("cpu"):
        from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass.trace_schema import GenerationContext
from looking_glass.transformer_localization import TensorObservation, compare_transformer_traces
from looking_glass.transformer_trace import GPT2TraceSession


@unittest.skipUnless(TRANSFORMERS_INSTALLED, "transformers optional dependency is not installed")
class GPT2LocalizationIntegrationTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(527)
            self.model = GPT2LMHeadModel(GPT2Config(
                n_layer=2, n_embd=8, n_head=2, n_positions=16, vocab_size=64,
                use_cache=False,
            )).to(device="cpu", dtype=torch.float32).eval()
        self.ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cpu")

    def capture(self, run_id, *, injection=False, verified_context=True):
        handle = None
        if injection:
            def inject(_module, _inputs, output):
                changed = output.clone()
                changed[0, 1, 0] += .125
                return changed
            handle = self.model.transformer.h[0].mlp.register_forward_hook(inject)
        try:
            with torch.no_grad(), torch.device("cpu"), GPT2TraceSession(
                self.model, run_id=run_id, samples=(0,), tokens=(1,),
                max_trace_bytes=4096,
            ) as session:
                self.model(self.ids, past_key_values=None, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
        self.assertTrue(session.successful)
        self.assertEqual(len(session.events), 6)
        observations = []
        for event in session.events:
            metadata = event.to_schema()
            if verified_context:
                # This test explicitly supplies no input cache and disables output caching.
                metadata = replace(metadata, generation=GenerationContext(
                    stage="forward", use_cache=False))
            observations.append(TensorObservation(metadata, event.value))
        return observations

    def compare(self, left, right):
        return compare_transformer_traces(left, right, reference_complete=True,
            candidate_complete=True, pairing_verified=True, execution_order_verified=True)

    def test_repeated_no_injection_runs_match_selected_snapshots(self):
        result = self.compare(self.capture("reference"), self.capture("candidate"))
        self.assertEqual(result["status"], "no_difference_observed")
        self.assertEqual(result["coverage"]["compared"], 6)

    def test_known_injection_is_observed_at_the_selected_mlp_boundary(self):
        result = self.compare(self.capture("reference"), self.capture("injected", injection=True))
        self.assertEqual(result["status"], "difference_observed")
        self.assertEqual(result["first_observed_interval"]["first_unequal"]["module_path"],
                         "transformer.h.0.mlp")
        self.assertEqual(result["causal_origin"], "unknown")

    def test_unverified_adapter_generation_context_is_inconclusive(self):
        result = self.compare(self.capture("reference", verified_context=False),
                              self.capture("candidate", verified_context=False))
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(result["schema_coverage"]["reference_metadata_issues"])


if __name__ == "__main__":
    unittest.main()
