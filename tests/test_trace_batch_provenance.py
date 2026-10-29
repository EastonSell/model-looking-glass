"""CPU integration of trace provenance through private batch sealing.

The inputs are fixed development fixtures. This test covers metadata sealing,
not payload delivery, an authenticated service, GPU admission, or causality.
"""

from dataclasses import replace
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
class TraceBatchProvenanceTests(unittest.TestCase):
    def test_real_padded_traces_seal_only_with_matching_coordinates(self):
        from looking_glass.host_budget import HostBudget
        from looking_glass.resident_worker import ForwardRequest
        from looking_glass.trace_batch_owner import BatchStateError, TraceBatchRegistry
        from looking_glass.trace_producer import ClaimLimit, ProducerEnvelope, ProducerProfile
        from looking_glass.trace_schema import GenerationContext, RunIdentity, TraceEvent, validate_trace
        from looking_glass.trace_store import HostTraceStore
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
        sites = (SiteTrace(1, "block"), SiteTrace(0, "mlp"),
                 SiteTrace(1, "mlp"), SiteTrace(0, "attn"))
        row_samples = (("sample-A", "sample-B"), ("sample-B", "sample-A"))
        row_tokens = (((3, 2, 0), (2, 1, 0)), ((2, 1, 0), (3, 2, 0)))

        # Independent full outputs are observed before any worker hooks exist.
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
        broker_id, process_id = "a" * 32, "b" * 32
        registry = TraceBatchRegistry(budget=budget, store=store, broker_id=broker_id,
                                      process_generation=process_id, max_jobs=1,
                                      max_slots_per_job=8)
        worker = TracedGPT2Worker("batch-tiny", model, device="cpu",
                                  max_batch_size=2, max_sequence_length=5)
        published = []
        for case, input_index in enumerate((0, 1, 0)):
            item = inputs[input_index]
            identity = BrokerJobIdentity(broker_id, case + 7, "team", f"request-{case}")
            samples = tuple(SampleTrace(row_samples[input_index][row], row_tokens[input_index][row])
                            for row in (0, 1))
            spec = TraceSpec("batch-run", samples, sites, case + 3, 19 + case * 8,
                             None, 8, 768)
            plan = bind_trace_plan(identity, spec, TraceLimits(2, 5, 2, 4, 8, 768, 3),
                                   batch_size=2, sequence_length=5, layer_count=2)
            registration = f"{case + 1:032x}"
            registry.register(plan, registration_id=registration, generation=case + 1,
                              batch_size=2, sequence_length=5, hidden_width=8,
                              layer_count=2, max_event_bytes=96)
            job = registry.lookup(identity.job_id, registration_id=registration, generation=case + 1)
            with self.assertRaises(BatchStateError):
                job.manifest()
            # Logical fixture commitments only; model and test reference
            # allocations are outside this cooperative ledger.
            profile = ProducerProfile(ClaimLimit(1024, 2), ClaimLimit(4096, 1),
                                      ClaimLimit(1024, 8), ClaimLimit(262144, 1))
            envelope = ProducerEnvelope.acquire(budget, f"producer-{case}", profile)
            job.prepare()  # Reserve every delivery source before capture.
            request = ForwardRequest(plan.worker_request_id, item, seed=7)

            def forward():
                return worker.execute_traced(
                    request, captures=(plan.capture_request(),),
                    sites=tuple(TraceSite(site.layer_index, site.module_kind) for site in plan.sites),
                    sequence_starts=plan.sequence_starts(), max_events=8, max_trace_bytes=768)

            result = envelope.run(forward)
            self.assertTrue(torch.equal(result.forward_result.logits.contiguous().view(torch.uint8),
                                        baseline_logits[input_index].contiguous().view(torch.uint8)))
            references = [weakref.ref(event.value) for event in result.events]
            logits_ref = weakref.ref(result.forward_result.logits)
            expected = {}
            for event in result.events:
                row = event.batch_position
                metadata = event.to_trace_event()
                expected[metadata.sequence] = metadata
                self.assertEqual(metadata.generation, GenerationContext())
                self.assertEqual(metadata.identity, RunIdentity(
                    "batch-run", f"lg-{broker_id}-{identity.job_id}",
                    identity.request_id, row_samples[input_index][row]))
                self.assertEqual(metadata.token_positions, row_tokens[input_index][row])
                self.assertEqual(item["attention_mask"][row, 0].item(), 0)
                self.assertNotEqual(item["position_ids"][row, 0].item(), 0)
                self.assertEqual(sum(item["input_ids"][row, token].item() == 7
                                     for token in metadata.token_positions), 2)
                reference = observed[(input_index, event.layer_index, event.module_kind)][
                    row, list(metadata.token_positions)]
                self.assertTrue(torch.equal(event.value.contiguous().view(torch.uint8),
                                            reference.contiguous().view(torch.uint8)))
            self.assertEqual(set(expected), set(range(spec.sequence_start, spec.sequence_start + 8)))

            if case < 2:
                # Arrival order does not identify a slot: validate each event.
                job.stage(tuple(reversed(result.events)))
            else:
                # Preserve the actual payload but mislabel one token order.
                first = result.events[0]
                bad = (replace(first, token_positions=tuple(reversed(first.token_positions))),
                       *result.events[1:])
                with self.assertRaises(ValueError):
                    job.stage(bad)
                bad = first = None
            with self.assertRaises(BatchStateError):
                job.manifest()
            event = result = reference = None
            gc.collect()
            self.assertTrue(all(ref() is None for ref in references))
            self.assertIsNone(logits_ref())
            job.confirm_sources_disposed()
            for stage in ("input_copies", "output", "trace_bundle", "workspace"):
                envelope.confirm_disposed(stage)
            envelope.close()

            if case < 2:
                with self.assertRaises(BatchStateError):
                    job.manifest()
                job.seal()
                manifest = job.manifest()
                self.assertEqual((manifest.broker_id, manifest.process_generation, manifest.job_id,
                                  manifest.registration_id, manifest.generation),
                                 (broker_id, process_id, identity.job_id, registration, case + 1))
                self.assertEqual([part.event.sequence for part in manifest.events],
                                 list(range(spec.sequence_start, spec.sequence_start + 8)))
                expected_order = [(layer, kind, row)
                                  for layer, kind in ((0, "attn"), (0, "mlp"),
                                                      (1, "mlp"), (1, "block"))
                                  for row in (0, 1)]
                self.assertEqual([(part.event.layer_index, part.event.module_kind,
                                   part.event.batch_positions[0]) for part in manifest.events],
                                 expected_order)
                self.assertEqual(len({part.identity.delivery_id for part in manifest.events}), 8)
                for part in manifest.events:
                    meta = part.event
                    self.assertEqual(meta, expected[meta.sequence])
                    self.assertEqual(TraceEvent.from_json(meta.to_json()), meta)
                    self.assertEqual((meta.forward_index, meta.invocation), (case + 3, 0))
                    row = meta.batch_positions[0]
                    self.assertEqual(meta.token_positions, row_tokens[input_index][row])
                    self.assertEqual((part.identity.broker_id, part.identity.job_id,
                                      part.identity.team_id, part.identity.run_id,
                                      part.identity.request_id, part.identity.sample_id,
                                      part.identity.batch_position),
                                     (broker_id, identity.job_id, "team", "batch-run",
                                      identity.request_id, row_samples[input_index][row], row))
                    self.assertEqual(meta.identity, RunIdentity(
                        "batch-run", f"lg-{broker_id}-{identity.job_id}",
                        identity.request_id, row_samples[input_index][row]))
                    suffix = "" if meta.module_kind == "block" else f".{meta.module_kind}"
                    self.assertEqual(meta.module_path, f"transformer.h.{meta.layer_index}{suffix}")
                    self.assertEqual(meta.output_path, () if meta.module_kind == "mlp" else (0,))
                    self.assertEqual((part.payload_bytes, part.byte_order), (96, "little"))
                    published.append(meta)
            else:
                with self.assertRaises(BatchStateError):
                    job.seal()
                with self.assertRaises(BatchStateError):
                    job.manifest()
            self.assertTrue(job.dispose(timeout=5))
            registry.retire(job)
            snapshot = budget.snapshot()
            self.assertEqual((snapshot.used_bytes, snapshot.used_items), (0, 0))
            self.assertEqual((snapshot.requests, snapshot.claims, snapshot.operations), ((), (), ()))
            self.assertEqual(registry.active_jobs(), ())
            self.assertEqual(store.usage().claimed_items, 0)
        validate_trace(published)
        self.assertEqual({event.sequence for event in published}, set(range(19, 35)))
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in model.modules()))
        worker.close()


if __name__ == "__main__":
    unittest.main()
