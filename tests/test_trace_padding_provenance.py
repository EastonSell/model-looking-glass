"""CPU-only checks that selected coordinates remain local to each forward.

These fixed development inputs exercise the instrumentation contract; matching
snapshots do not explain what the model computes or establish causality.
"""

import importlib.util
import unittest

import torch

if importlib.util.find_spec("transformers") is not None:
    from transformers import GPT2Config, GPT2LMHeadModel
else:
    GPT2Config = GPT2LMHeadModel = None


@unittest.skipUnless(GPT2Config is not None, "requires transformers")
class TracePaddingProvenanceTests(unittest.TestCase):
    def make_model(self):
        config = GPT2Config(vocab_size=32, n_positions=32, n_ctx=32, n_embd=8,
                            n_layer=1, n_head=2, use_cache=False,
                            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026)
            return GPT2LMHeadModel(config).eval()

    def assert_same_bits(self, actual, expected):
        self.assertEqual((actual.dtype, tuple(actual.shape)),
                         (expected.dtype, tuple(expected.shape)))
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))

    def baseline(self, model, inputs, *, cached=False):
        """Record the whole attention output before attaching either producer."""
        outputs = []

        def save_attention(_module, _args, output):
            outputs.append(output[0].detach().clone())

        hook = model.transformer.h[0].attn.register_forward_hook(save_attention)
        try:
            with torch.inference_mode():
                if cached:
                    first = model(**inputs[0], use_cache=True)
                    second = model(**inputs[1], past_key_values=first.past_key_values,
                                   use_cache=True)
                    logits = (first.logits.clone(), second.logits.clone())
                else:
                    logits = tuple(model(**item).logits.clone() for item in inputs)
        finally:
            hook.remove()
        return logits, tuple(outputs)

    def test_masked_repeated_tokens_and_reordered_rows_keep_local_coordinates(self):
        from looking_glass.capture_dispatcher import CaptureRequest, SampleSelection
        from looking_glass.resident_worker import ForwardRequest
        from looking_glass.trace_schema import GenerationContext, RunIdentity, TraceEvent, validate_trace
        from looking_glass.traced_worker import TraceSite, TracedGPT2Worker
        from looking_glass.transformer_trace import GPT2TraceSession

        model = self.make_model()
        inputs = (
            {"input_ids": torch.tensor([[0, 0, 7, 7, 4], [0, 7, 7, 0, 5]]),
             "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]),
             "position_ids": torch.tensor([[7, 8, 9, 10, 11], [2, 3, 4, 5, 6]])},
            {"input_ids": torch.tensor([[0, 7, 7, 0, 6], [0, 0, 7, 7, 9]]),
             "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
             "position_ids": torch.tensor([[3, 4, 5, 6, 7], [9, 10, 11, 12, 13]])},
        )
        baseline_logits, baseline_outputs = self.baseline(model, inputs)
        # Every row includes a masked position and a pair of repeated token IDs.
        self.assertEqual(inputs[0]["input_ids"][0, 2].item(), inputs[0]["input_ids"][0, 3].item())
        self.assertEqual(inputs[0]["input_ids"][1, 1].item(), inputs[0]["input_ids"][1, 2].item())
        self.assertEqual(inputs[1]["input_ids"][0, 1].item(), inputs[1]["input_ids"][0, 2].item())
        self.assertEqual(inputs[1]["input_ids"][1, 2].item(), inputs[1]["input_ids"][1, 3].item())

        tokens = (4, 3, 2, 1, 0)
        with GPT2TraceSession(model, run_id="padded-adapter", layers=(0,), targets=("attn",),
                              samples=(1, 0), tokens=tokens, max_events=2,
                              max_trace_bytes=4096) as trace:
            with torch.inference_mode():
                logits = tuple(model(**item).logits.clone() for item in inputs)
        self.assertTrue(trace.successful)
        self.assertEqual(len(trace.events), 2)
        for index, event in enumerate(trace.events):
            metadata = event.to_schema()
            self.assert_same_bits(logits[index], baseline_logits[index])
            self.assertEqual(metadata.identity, RunIdentity("padded-adapter"))
            self.assertEqual((metadata.sequence, metadata.forward_index, metadata.invocation),
                             (index, index, 0))
            self.assertEqual((metadata.batch_positions, metadata.token_positions), ((1, 0), tokens))
            self.assertEqual((metadata.batch_axis, metadata.token_axis), (0, 1))
            self.assertEqual(metadata.generation, GenerationContext())
            self.assertEqual(TraceEvent.from_json(metadata.to_json()), metadata)
            expected = baseline_outputs[index][[1, 0]][:, list(tokens)]
            self.assert_same_bits(event.value, expected)

        # The caller supplies the row-to-sample association anew per forward.
        rows = (("sample-A", "sample-B"), ("sample-B", "sample-A"))
        selections = (((3, 2, 0), (2, 1, 0)), ((2, 1, 0), (3, 2, 0)))
        worker = TracedGPT2Worker("padded-tiny", model, device="cpu",
                                  max_batch_size=2, max_sequence_length=5)
        all_metadata = []
        for forward_index, item in enumerate(inputs):
            captures = tuple(CaptureRequest(
                "padded-worker", "job", f"request-{rows[forward_index][row]}",
                (SampleSelection(row, rows[forward_index][row], selections[forward_index][row]),),
                forward_index, None, 1, 4096) for row in (1, 0))
            result = worker.execute_traced(
                ForwardRequest(f"batch-{forward_index}", item, seed=7),
                captures=captures, sites=(TraceSite(0, "attn"),),
                sequence_starts={"padded-worker": forward_index * 2},
                max_events=2, max_trace_bytes=4096)
            self.assert_same_bits(result.forward_result.logits, baseline_logits[forward_index])
            self.assertEqual(len(result.events), 2)
            seen_rows = set()
            for event in result.events:
                row = event.batch_position
                seen_rows.add(row)
                metadata = event.to_trace_event()
                all_metadata.append(metadata)
                expected_id = rows[forward_index][row]
                selected = selections[forward_index][row]
                self.assertEqual(metadata.identity, RunIdentity(
                    "padded-worker", "job", f"request-{expected_id}", expected_id))
                self.assertEqual((metadata.forward_index, metadata.invocation), (forward_index, 0))
                self.assertEqual((metadata.batch_positions, metadata.token_positions), ((row,), selected))
                self.assertEqual((metadata.batch_axis, metadata.token_axis), (0, 1))
                self.assertEqual(metadata.generation, GenerationContext())
                self.assertEqual(TraceEvent.from_json(metadata.to_json()), metadata)
                self.assertEqual(item["attention_mask"][row, 0].item(), 0)
                self.assertIn(0, selected)
                self.assert_same_bits(event.value, baseline_outputs[forward_index][row, list(selected)])
            self.assertEqual(seen_rows, {0, 1})
        validate_trace(all_metadata)
        self.assertEqual([event.sequence for event in all_metadata], [0, 1, 2, 3])
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))

    def test_cached_decode_keeps_zero_local_and_full_generation_context_unknown(self):
        from looking_glass.trace_schema import GenerationContext, RunIdentity, TraceEvent, validate_trace
        from looking_glass.transformer_trace import GPT2TraceSession

        model = self.make_model()
        inputs = (
            {"input_ids": torch.tensor([[0, 7, 7], [0, 0, 7]]),
             "attention_mask": torch.tensor([[0, 1, 1], [0, 0, 1]]),
             "position_ids": torch.tensor([[3, 4, 5], [7, 8, 9]])},
            {"input_ids": torch.tensor([[7], [7]]),
             "attention_mask": torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]]),
             "position_ids": torch.tensor([[6], [10]])},
        )
        baseline_logits, baseline_outputs = self.baseline(model, inputs, cached=True)
        with GPT2TraceSession(model, run_id="cached-local", layers=(0,), targets=("attn",),
                              samples=(1, 0), tokens=(0,), max_events=2,
                              max_trace_bytes=4096) as trace:
            with torch.inference_mode():
                first = model(**inputs[0], use_cache=True)
                self.assertIsNotNone(first.past_key_values)
                # The cache can be mutated in place by the next forward.
                prefill_length = first.past_key_values.get_seq_length()
                second = model(**inputs[1], past_key_values=first.past_key_values,
                               use_cache=True)
                self.assertIsNotNone(second.past_key_values)
                decode_length = second.past_key_values.get_seq_length()
        self.assertTrue(trace.successful)
        self.assertEqual((prefill_length, decode_length), (3, 4))
        self.assert_same_bits(first.logits, baseline_logits[0])
        self.assert_same_bits(second.logits, baseline_logits[1])
        self.assertEqual(len(trace.events), 2)
        metadata = [event.to_schema() for event in trace.events]
        validate_trace(metadata)
        for index, event in enumerate(trace.events):
            item = metadata[index]
            self.assertEqual(item.identity, RunIdentity("cached-local"))
            self.assertEqual((item.sequence, item.forward_index, item.invocation), (index, index, 0))
            self.assertEqual((item.batch_positions, item.token_positions), ((1, 0), (0,)))
            self.assertEqual(item.shape, (2, 3 if index == 0 else 1, 8))
            self.assertEqual(item.generation, GenerationContext())
            self.assertEqual(TraceEvent.from_json(item.to_json()), item)
            self.assert_same_bits(event.value, baseline_outputs[index][[1, 0]][:, [0]])
        self.assertEqual(inputs[1]["position_ids"].tolist(), [[6], [10]])
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
