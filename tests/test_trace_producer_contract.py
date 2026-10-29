"""Compare two CPU trace producers on identical fixed GPT-2 forwards.

Matching observations establish a producer contract, not causal provenance.
"""
import importlib.util
import unittest

import torch

if importlib.util.find_spec("transformers") is not None:
    from transformers import GPT2Config, GPT2LMHeadModel
else:
    GPT2Config = GPT2LMHeadModel = None


@unittest.skipUnless(GPT2Config is not None, "requires transformers")
class TraceProducerContractTests(unittest.TestCase):
    def assert_same_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))

    def test_adapter_and_worker_agree_on_two_forwards_and_each_sample_slice(self):
        from looking_glass.capture_dispatcher import CaptureRequest, SampleSelection
        from looking_glass.resident_worker import ForwardRequest
        from looking_glass.trace_schema import GenerationContext, RunIdentity, validate_trace
        from looking_glass.traced_worker import TraceSite, TracedGPT2Worker
        from looking_glass.transformer_trace import GPT2TraceSession

        config = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16, n_embd=8,
                            n_layer=2, n_head=2, use_cache=False,
                            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026)
            model = GPT2LMHeadModel(config).eval()
        inputs = (torch.tensor([[1, 3, 5], [2, 4, 6]]),
                  torch.tensor([[3, 1, 7], [6, 2, 8]]))
        positions = (2, 0)

        # The adapter keeps both selected batch rows in each event.
        with GPT2TraceSession(model, run_id="adapter-run", targets=("mlp", "block", "attn"),
                              layers=(1, 0), samples=(1, 0), tokens=positions,
                              max_events=12, max_trace_bytes=8192) as session:
            with torch.inference_mode():
                adapter_logits = tuple(model(input_ids=item).logits.clone() for item in inputs)
        self.assertTrue(session.successful)
        adapter_events = session.events
        self.assertEqual(len(adapter_events), 12)
        adapter_metadata = tuple(event.to_schema() for event in adapter_events)
        validate_trace(adapter_metadata)
        self.assertEqual([event.sequence for event in adapter_metadata], list(range(12)))
        self.assertTrue(all(event.identity == RunIdentity("adapter-run") for event in adapter_metadata))
        self.assertTrue(all(event.generation == GenerationContext() for event in adapter_metadata))

        # The worker emits one separately identified slice for each row/site.
        worker = TracedGPT2Worker("fixed-tiny-gpt2", model, device="cpu",
                                  max_batch_size=2, max_sequence_length=3)
        sites = (TraceSite(1, "block"), TraceSite(0, "mlp"), TraceSite(1, "attn"),
                 TraceSite(0, "block"), TraceSite(1, "mlp"), TraceSite(0, "attn"))
        worker_events = []
        for forward_index, item in enumerate(inputs):
            requests = tuple(CaptureRequest(
                "worker-run", "job", f"request-{row}",
                (SampleSelection(row, f"sample-{row}", positions),),
                forward_index, None, 6, 4096) for row in (1, 0))
            result = worker.execute_traced(
                ForwardRequest(f"batch-{forward_index}", {"input_ids": item}, seed=7),
                captures=requests, sites=sites,
                sequence_starts={"worker-run": forward_index * 12},
                max_events=12, max_trace_bytes=8192)
            self.assert_same_bits(result.forward_result.logits, adapter_logits[forward_index])
            worker_events.extend(result.events)

        worker_metadata = tuple(event.to_trace_event() for event in worker_events)
        validate_trace(worker_metadata)
        self.assertEqual([event.sequence for event in worker_metadata], list(range(24)))
        self.assertTrue(all(event.generation == GenerationContext() for event in worker_metadata))

        def key(event):
            return (event.forward_index, event.module_path, event.layer_index,
                    event.module_kind, event.invocation, event.output_path)

        adapter_by_site = {key(event): event for event in adapter_events}
        self.assertEqual(len(adapter_by_site), len(adapter_events))
        seen = set()
        for captured, metadata in zip(worker_events, worker_metadata):
            with self.subTest(forward=metadata.forward_index, path=metadata.module_path,
                              row=captured.batch_position):
                site = key(metadata)
                self.assertIn(site, adapter_by_site)
                adapter = adapter_by_site[site]
                adapter_schema = adapter.to_schema()
                row = captured.batch_position
                self.assertNotIn((site, row), seen)
                seen.add((site, row))
                self.assertEqual(metadata.identity.run_id, "worker-run")
                self.assertEqual(metadata.identity.job_id, "job")
                self.assertEqual(metadata.identity.request_id, f"request-{row}")
                self.assertEqual(metadata.identity.sample_id, f"sample-{row}")
                self.assertIn(metadata.layer_index, (0, 1))
                self.assertIn(metadata.module_kind, ("block", "attn", "mlp"))
                suffix = "" if metadata.module_kind == "block" else f".{metadata.module_kind}"
                self.assertEqual(metadata.module_path, f"transformer.h.{metadata.layer_index}{suffix}")
                self.assertEqual(metadata.output_path, () if metadata.module_kind == "mlp" else (0,))
                self.assertEqual(metadata.batch_positions, (row,))
                self.assertEqual(adapter.batch_positions, (1, 0))
                self.assertEqual(metadata.token_positions, positions)
                self.assertEqual(adapter.token_positions, positions)
                self.assertEqual(metadata.shape, adapter.shape)
                self.assertEqual((metadata.dtype, metadata.device), (adapter.dtype, adapter.device))
                self.assertEqual((metadata.batch_axis, metadata.token_axis), (0, 1))
                self.assertEqual((adapter_schema.batch_axis, adapter_schema.token_axis), (0, 1))
                self.assertEqual((metadata.capture_mode, adapter.capture_mode), ("direct", "direct"))
                self.assertEqual(metadata.saved_shape, (len(positions), 8))
                self.assertEqual(adapter.saved_shape, (2, len(positions), 8))
                self.assertEqual((metadata.saved_dtype, metadata.saved_device), ("torch.float32", "cpu"))
                self.assertEqual((adapter_schema.saved_dtype, adapter.saved_device),
                                 ("torch.float32", "cpu"))
                self.assert_same_bits(captured.value, adapter.value[adapter.batch_positions.index(row)])
        self.assertEqual(len(seen), 24)
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
