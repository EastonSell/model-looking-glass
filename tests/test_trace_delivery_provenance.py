"""CPU-only producer-to-delivery metadata and byte consistency.

The sink deliberately examines bounded pieces of the synchronous payload. It
does not model a network protocol or an atomic multi-event publication. Model,
input, baseline, and test fixture memory is outside these logical test claims.
"""

import gc
import importlib.util
import unittest
import weakref

import torch

if importlib.util.find_spec("transformers") is not None:
    from transformers import GPT2Config, GPT2LMHeadModel
else:
    GPT2Config = GPT2LMHeadModel = None


@unittest.skipUnless(GPT2Config is not None, "requires transformers")
class TraceDeliveryProvenanceTests(unittest.TestCase):
    def test_padded_repeated_tokens_survive_individual_delivery(self):
        from looking_glass.host_budget import HostBudget
        from looking_glass.resident_worker import ForwardRequest
        from looking_glass.trace_job_validation import make_trace_job_layout
        from looking_glass.trace_producer import ClaimLimit, ProducerEnvelope, ProducerProfile
        from looking_glass.trace_schema import GenerationContext, TraceEvent, validate_trace
        from looking_glass.trace_store import HostTraceStore
        from looking_glass.trace_transport import TraceDelivery
        from looking_glass.traced_broker_worker import (
            BrokerJobIdentity, SampleTrace, SiteTrace, TraceLimits, TraceSpec, bind_trace_plan,
        )
        from looking_glass.traced_worker import TraceSite, TracedGPT2Worker

        config = GPT2Config(vocab_size=32, n_positions=32, n_ctx=32, n_embd=8,
                            n_layer=2, n_head=2, use_cache=False,
                            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            model = GPT2LMHeadModel(config).eval()
        inputs = (
            {"input_ids": torch.tensor([[0, 0, 7, 7, 4], [0, 7, 7, 0, 5]]),
             "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]),
             "position_ids": torch.tensor([[7, 8, 9, 10, 11], [2, 3, 4, 5, 6]])},
            {"input_ids": torch.tensor([[0, 7, 7, 0, 6], [0, 0, 7, 7, 9]]),
             "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
             "position_ids": torch.tensor([[3, 4, 5, 6, 7], [9, 10, 11, 12, 13]])},
        )
        # These are fixed development inputs, not held-out experiment cases.
        sites = (SiteTrace(1, "block"), SiteTrace(0, "mlp"),
                 SiteTrace(1, "mlp"), SiteTrace(0, "attn"))
        row_samples = (("sample-A", "sample-B"), ("sample-B", "sample-A"))
        row_tokens = (((3, 2, 0), (2, 1, 0)), ((2, 1, 0), (3, 2, 0)))
        observed = {}
        active_forward = [0]
        hooks = []
        for site in sites:
            block = model.transformer.h[site.layer_index]
            target = block if site.module_kind == "block" else getattr(block, site.module_kind)

            def save(_module, _args, output, site=site):
                value = output[0] if type(output) is tuple else output
                observed[(active_forward[0], site.layer_index, site.module_kind)] = value.detach().clone()

            hooks.append(target.register_forward_hook(save))
        try:
            with torch.inference_mode():
                baseline_logits = []
                for index, item in enumerate(inputs):
                    active_forward[0] = index
                    baseline_logits.append(model(**item).logits.clone())
        finally:
            for hook in hooks:
                hook.remove()

        budget = HostBudget(max_bytes=2 * 1024 * 1024, max_items=128,
                            max_requests=32, max_operations=32)
        store = HostTraceStore(max_tensor_bytes=8192, max_items=32)
        worker = TracedGPT2Worker("delivery-tiny", model, device="cpu",
                                  max_batch_size=2, max_sequence_length=5)
        all_delivered = []
        for case, item in enumerate(inputs):
            identity = BrokerJobIdentity("a" * 32, case + 7, "team", f"request-{case}")
            samples = tuple(SampleTrace(row_samples[case][row], row_tokens[case][row])
                            for row in (0, 1))
            spec = TraceSpec("delivery-run", samples, sites, case + 3, 19 + case * 8,
                             None, 8, 768)
            plan = bind_trace_plan(identity, spec, TraceLimits(2, 5, 2, 4, 8, 768, 3),
                                   batch_size=2, sequence_length=5, layer_count=2)
            layout = make_trace_job_layout(plan, batch_size=2, sequence_length=5,
                                           hidden_width=8, layer_count=2,
                                           max_slots=8, max_event_bytes=96)
            # Illustrative cooperative logical commitments, not physical RAM bounds.
            profile = ProducerProfile(ClaimLimit(1024, 2), ClaimLimit(4096, 1),
                                      ClaimLimit(1024, 8), ClaimLimit(262144, 1))
            envelope = ProducerEnvelope.acquire(budget, f"producer-{case}", profile)
            deliveries = {}
            for slot in layout.slots:
                key = (slot.batch_position, slot.layer_index, slot.module_kind)
                delivery = TraceDelivery(
                    budget, store, broker_id=identity.broker_id, job_id=identity.job_id,
                    team_id=identity.team_id, run_id=spec.run_id,
                    request_id=identity.request_id, sample_id=slot.sample_id,
                    batch_position=slot.batch_position, max_payload_bytes=96)
                # Each slot's source claim is held before the forward.
                delivery.reserve_source(slot.payload_bytes)
                deliveries[key] = delivery

            request = ForwardRequest(plan.worker_request_id, item, seed=7)

            def forward():
                return worker.execute_traced(
                    request, captures=(plan.capture_request(),),
                    sites=tuple(TraceSite(site.layer_index, site.module_kind) for site in plan.sites),
                    sequence_starts=plan.sequence_starts(), max_events=8, max_trace_bytes=768)

            result = envelope.run(forward)
            self.assertTrue(torch.equal(result.forward_result.logits.contiguous().view(torch.uint8),
                                        baseline_logits[case].contiguous().view(torch.uint8)))
            expected_metadata = layout.validate(result.events)
            by_sequence = {event.sequence: event for event in expected_metadata}
            self.assertEqual(list(by_sequence), list(range(spec.sequence_start, spec.sequence_start + 8)))
            source_refs = [weakref.ref(event.value) for event in result.events]
            logits_ref = weakref.ref(result.forward_result.logits)
            pending = {}
            for event in result.events:
                key = (event.batch_position, event.layer_index, event.module_kind)
                self.assertIn(key, deliveries)
                self.assertNotIn(key, pending)
                metadata = event.to_trace_event()
                self.assertEqual(metadata, by_sequence[metadata.sequence])
                self.assertEqual(metadata.generation, GenerationContext())
                self.assertEqual(metadata.identity.sample_id, row_samples[case][event.batch_position])
                self.assertEqual(metadata.token_positions, row_tokens[case][event.batch_position])
                self.assertIn(0, metadata.token_positions)
                self.assertEqual(item["attention_mask"][event.batch_position, 0].item(), 0)
                self.assertNotEqual(item["position_ids"][event.batch_position, 0].item(), 0)
                self.assertEqual(sum(item["input_ids"][event.batch_position, token].item() == 7
                                     for token in metadata.token_positions), 2)
                deliveries[key].stage(event)
                pending[key] = metadata
            self.assertEqual(set(pending), set(deliveries))
            event = result = None
            gc.collect()
            self.assertTrue(all(reference() is None for reference in source_refs))
            self.assertIsNone(logits_ref())
            for delivery in deliveries.values():
                delivery.confirm_source_disposed()
            for stage in ("input_copies", "output", "trace_bundle", "workspace"):
                envelope.confirm_disposed(stage)
            envelope.close()

            # Deliver individual events out of producer order. The test sink
            # reads seven-byte pieces synchronously and retains no payload;
            # this is not a chunked network or batch-sealing protocol.
            case_sequences = []
            for key in reversed(list(deliveries)):
                delivery = deliveries[key]
                manifest = delivery.publish()
                metadata = pending[key]
                row, layer, kind = key
                self.assertEqual(manifest.event, metadata)
                self.assertEqual(TraceEvent.from_json(manifest.event.to_json()), metadata)
                self.assertEqual((manifest.identity.broker_id, manifest.identity.job_id,
                                  manifest.identity.team_id, manifest.identity.request_id,
                                  manifest.identity.sample_id, manifest.identity.batch_position),
                                 (identity.broker_id, identity.job_id, identity.team_id,
                                  identity.request_id, row_samples[case][row], row))
                self.assertEqual((metadata.forward_index, metadata.layer_index,
                                  metadata.module_kind, metadata.invocation), (case + 3, layer, kind, 0))
                suffix = "" if kind == "block" else f".{kind}"
                self.assertEqual(metadata.module_path, f"transformer.h.{layer}{suffix}")
                self.assertEqual(metadata.output_path, () if kind == "mlp" else (0,))
                self.assertEqual((metadata.batch_positions, metadata.token_positions),
                                 ((row,), row_tokens[case][row]))
                self.assertEqual((metadata.batch_axis, metadata.token_axis), (0, 1))
                self.assertEqual((metadata.shape, metadata.saved_shape), ((2, 5, 8), (3, 8)))
                self.assertEqual((metadata.dtype, metadata.device,
                                  metadata.saved_dtype, metadata.saved_device),
                                 ("torch.float32", "cpu", "torch.float32", "cpu"))
                self.assertEqual((manifest.payload_bytes, manifest.byte_order), (96, "little"))
                expected = observed[(case, layer, kind)][row, list(row_tokens[case][row])]
                reference = expected.contiguous().numpy().tobytes(order="C")
                pieces = []

                def sink(data):
                    view = memoryview(data)
                    for start in range(0, len(view), 7):
                        end = min(start + 7, len(view))
                        self.assertEqual(view[start:end], memoryview(reference)[start:end])
                        pieces.append(end - start)

                delivery.send(sink)
                self.assertGreater(len(pieces), 1)
                self.assertEqual(sum(pieces), manifest.payload_bytes)
                delivery.ack_received(manifest.identity)
                delivery.ack_release(manifest.identity)
                all_delivered.append(metadata)
                case_sequences.append(metadata.sequence)
            self.assertEqual(case_sequences, list(reversed(range(spec.sequence_start,
                                                                  spec.sequence_start + 8))))
            for delivery in deliveries.values():
                self.assertTrue(delivery.drain(3))
            snapshot = budget.snapshot()
            self.assertEqual((snapshot.used_bytes, snapshot.used_items), (0, 0))
            self.assertEqual((snapshot.requests, snapshot.claims, snapshot.operations), ((), (), ()))
            self.assertEqual(store.usage().claimed_items, 0)
        validate_trace(all_delivered)
        self.assertEqual({event.sequence for event in all_delivered}, set(range(19, 35)))
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))
        worker.close()


if __name__ == "__main__":
    unittest.main()
