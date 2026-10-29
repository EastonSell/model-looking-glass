"""CPU-only producer/worker/batch custody integration."""
import gc
import inspect
import importlib
import importlib.util
import sys
import threading
import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch

import torch

from looking_glass.host_budget import HostBudget, HostBusy, HostRequest
from looking_glass.resident_worker import ForwardRequest
from looking_glass.trace_producer import ClaimLimit, ProducerProfile
from looking_glass.trace_store import HostTraceStore
from looking_glass.traced_broker_worker import (
    BrokerJobIdentity, SampleTrace, SiteTrace, TraceLimits, TraceSpec, bind_trace_plan,
)

if __package__:
    from .opcode_support import offsets_at_line, run_opcode_trace
else:
    from opcode_support import offsets_at_line, run_opcode_trace


def plan():
    identity = BrokerJobIdentity("a" * 32, 7, "team", "client-request")
    samples = (SampleTrace("row-a", (0,)), SampleTrace("row-b", (2,)))
    sites = (SiteTrace(0, "mlp"), SiteTrace(0, "attn"))
    spec = TraceSpec("cpu-run", samples, sites, 0, 10, None, 4, 128)
    return bind_trace_plan(identity, spec, TraceLimits(2, 3, 1, 2, 4, 128, 3),
                           batch_size=2, sequence_length=3, layer_count=1)


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None,
                     "requires the optional transformers dependency")
class CpuTraceOwnerTests(unittest.TestCase):
    def make_owner(self, *, budget_bytes=2_000_000, max_output_bytes=768):
        from transformers import GPT2Config, GPT2LMHeadModel
        from looking_glass.cpu_trace_owner import CpuTraceOwner
        from looking_glass.traced_worker import TracedGPT2Worker
        config = GPT2Config(vocab_size=32, n_positions=8, n_ctx=8, n_embd=8, n_layer=1,
                            n_head=2, use_cache=False, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(71)
            model = GPT2LMHeadModel(config).eval()
        worker = TracedGPT2Worker("tiny-gpt2", model, device="cpu", max_batch_size=2,
                                  max_sequence_length=3, max_input_bytes=48,
                                  max_output_bytes=max_output_bytes)
        self.addCleanup(worker.close)
        budget = HostBudget(max_bytes=budget_bytes, max_items=64, max_requests=16,
                            max_operations=16)
        owner = CpuTraceOwner(worker=worker, budget=budget,
                              store=HostTraceStore(max_tensor_bytes=4096, max_items=8),
                              broker_id="a" * 32, process_generation="b" * 32,
                              max_slots=4, max_event_bytes=64)
        profile = ProducerProfile(input_copies=ClaimLimit(96, 8),
                                  output=ClaimLimit(2 * max_output_bytes, 2),
                                  trace_bundle=ClaimLimit(128, 4),
                                  workspace=ClaimLimit(100_000, 1))
        return owner, worker, budget, profile

    def register_job(self, owner, profile):
        selected = plan()
        owner.register(selected, profile=profile, registration_id="c" * 32, generation=1,
                       batch_size=2, sequence_length=3)
        return selected, owner.lookup(7, registration_id="c" * 32, generation=1)

    def test_raw_logits_are_rejected_before_any_forward_or_reservation(self):
        owner, worker, budget, profile = self.make_owner(max_output_bytes=4)
        with patch.object(worker._model.lm_head, "forward", side_effect=AssertionError("forward ran")):
            with self.assertRaisesRegex(ValueError, "raw model logits"):
                self.register_job(owner, profile)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_profile_minima_are_checked_before_admission(self):
        owner, worker, budget, profile = self.make_owner()
        shortages = (
            ("input_copies", ClaimLimit(95, 8)), ("input_copies", ClaimLimit(96, 7)),
            ("output", ClaimLimit(1535, 2)), ("output", ClaimLimit(1536, 1)),
            ("trace_bundle", ClaimLimit(127, 4)), ("trace_bundle", ClaimLimit(128, 3)),
            ("workspace", ClaimLimit(0, 1)),
        )
        for generation, (stage, limit) in enumerate(shortages, 1):
            with self.subTest(stage=stage, limit=limit):
                with self.assertRaisesRegex(ValueError, "overlapping stages"):
                    owner.register(plan(), profile=replace(profile, **{stage: limit}),
                                   registration_id="c" * 32, generation=generation,
                                   batch_size=2, sequence_length=3)
                self.assertEqual(budget.snapshot().used_bytes, 0)
        owner.register(plan(), profile=profile, registration_id="d" * 32,
                       generation=len(shortages) + 1, batch_size=2, sequence_length=3)
        job = owner.lookup(7, registration_id="d" * 32, generation=len(shortages) + 1)
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)

    def test_model_auxiliary_outputs_are_rejected_before_reservation(self):
        owner, worker, budget, profile = self.make_owner()
        worker._model.config.output_hidden_states = True
        with self.assertRaisesRegex(ValueError, "auxiliary"):
            self.register_job(owner, profile)
        self.assertEqual(budget.snapshot().used_bytes, 0)
        worker._model.config.output_hidden_states = False

    def test_tampered_raw_plan_entries_are_rejected_before_iteration(self):
        class UnexpectedIterable:
            def __iter__(self):
                raise AssertionError("untrusted plan entry was iterated")

        owner, worker, budget, profile = self.make_owner()
        selected = plan()
        object.__setattr__(selected, "_samples", (UnexpectedIterable(), selected._samples[1]))
        with self.assertRaisesRegex(ValueError, "bounded sample"):
            owner.register(selected, profile=profile, registration_id="c" * 32, generation=1,
                           batch_size=2, sequence_length=3)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_lookup_rejects_comparison_spoofing_before_owner_lock(self):
        from looking_glass.cpu_trace_owner import CpuTraceStateError

        class SpoofedKey(str):
            def __eq__(self, other):
                raise AssertionError("custom equality must not run")

            def __ne__(self, other):
                raise AssertionError("custom inequality must not run")

        owner, worker, budget, profile = self.make_owner()
        _, job = self.register_job(owner, profile)
        with self.assertRaises(CpuTraceStateError):
            owner.lookup(7, registration_id=SpoofedKey("c" * 32), generation=1)
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)

    def test_cpu_fixture_does_not_initialize_or_seed_cuda(self):
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")), \
                patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("CUDA seeded")), \
                patch.object(torch.cuda, "device_count", side_effect=AssertionError("CUDA enumerated")):
            owner, worker, budget, profile = self.make_owner()
            selected, job = self.register_job(owner, profile)
            job.prepare()
            caller = budget.open_request("caller")
            claim = caller.reserve("caller", "input", 48, items=1)
            ids = torch.zeros((2, 3), dtype=torch.int64)
            request = ForwardRequest(selected.worker_request_id, {"input_ids": ids})
            job.execute(request, input_owner=caller, input_claim=claim)
            job.discard_result()
            job.confirm_producer_disposed()
            job.seal()
            self.assertEqual(len(job.manifest().events), 4)
            self.assertTrue(job.dispose(timeout=5))
            owner.retire(job)
            request = ids = None
            claim.release()
            caller.close()
            self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_known_worker_out_of_memory_cannot_publish_and_can_drain(self):
        from looking_glass.cpu_trace_owner import CpuTraceStateError
        from looking_glass.resident_worker import WorkerOutOfMemory
        owner, worker, budget, profile = self.make_owner()
        selected, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
        request = ForwardRequest(selected.worker_request_id, {"input_ids": ids})
        with patch.object(worker._model, "forward", side_effect=MemoryError("injected")):
            with self.assertRaises(WorkerOutOfMemory):
                job.execute(request, input_owner=caller, input_claim=claim)
        self.assertTrue(worker.poisoned)
        self.assertEqual(job.state, "failed")
        self.assertGreater(budget.snapshot().used_bytes, 48)
        with self.assertRaises(CpuTraceStateError):
            job.manifest()
        job.discard_result()
        # The handled exception and its model/input references must be gone
        # before the trusted owner attests disposal.
        gc.collect()
        job.confirm_producer_disposed()
        self.assertTrue(job.dispose(timeout=5))
        owner.retire(job)
        request = ids = None
        claim.release()
        caller.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_known_unstarted_cancellation_releases_only_job_claims(self):
        for prepare in (False, True):
            with self.subTest(prepare=prepare):
                owner, worker, budget, profile = self.make_owner()
                _, job = self.register_job(owner, profile)
                if prepare:
                    job.prepare()
                    self.assertGreater(budget.snapshot().used_bytes, 0)
                job.cancel()
                self.assertTrue(job.abort_before_forward(timeout=5))
                owner.retire(job)
                self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_source_admission_denial_can_be_cleanly_aborted(self):
        # Producer claims and the first child metadata fit; its source does not.
        owner, worker, budget, profile = self.make_owner(budget_bytes=167_296)
        _, job = self.register_job(owner, profile)
        with self.assertRaises(Exception):
            job.prepare()
        self.assertEqual(job.state, "admission_failed")
        self.assertGreater(budget.snapshot().used_bytes, 0)
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_unstarted_abort_can_retry_after_a_drain_timeout(self):
        owner, worker, budget, profile = self.make_owner()
        _, job = self.register_job(owner, profile)
        job.prepare()
        with patch.object(job._batch, "dispose", return_value=False):
            self.assertFalse(job.abort_before_forward(timeout=0))
        self.assertEqual(job.state, "abort_pending")
        self.assertGreater(budget.snapshot().used_bytes, 0)
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_bad_input_claim_and_request_identity_fail_before_model_execution(self):
        from looking_glass.cpu_trace_owner import CpuTraceStateError
        owner, worker, budget, profile = self.make_owner()
        selected, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 47, items=1)
        ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
        with patch.object(worker, "execute_traced", side_effect=AssertionError("forward ran")):
            bad = ForwardRequest("wrong", {"input_ids": ids})
            with self.assertRaisesRegex(ValueError, "match the job"):
                job.execute(bad, input_owner=caller, input_claim=claim)
            with self.assertRaisesRegex(ValueError, "input claim"):
                job.execute(ForwardRequest(selected.worker_request_id, {"input_ids": ids}),
                            input_owner=caller, input_claim=claim)
        self.assertEqual(job.state, "prepared")
        with self.assertRaises(CpuTraceStateError):
            job.manifest()
        job.cancel()
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)
        ids = bad = None
        claim.release()
        caller.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_finalizer_reenters_owner_while_result_is_dropped(self):
        owner, worker, budget, profile = self.make_owner()
        selected, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
        request = ForwardRequest(selected.worker_request_id, {"input_ids": ids})
        job.execute(request, input_owner=caller, input_claim=claim)
        observed = []
        reference = weakref.ref(job._result.events[0].value,
                                lambda _: observed.append(job.state))
        job.discard_result()
        self.assertIsNone(reference())
        self.assertEqual(observed, ["discarding"])
        job.confirm_producer_disposed()
        self.assertTrue(job.dispose(timeout=5))
        owner.retire(job)
        request = ids = None
        claim.release()
        caller.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_rejected_request_error_does_not_keep_ordinary_input_alive(self):
        owner, worker, budget, profile = self.make_owner()
        _, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        ids = torch.zeros((2, 3), dtype=torch.int64)
        reference = weakref.ref(ids)
        request = ForwardRequest("wrong", {"input_ids": ids})
        retained = None
        try:
            job.execute(request, input_owner=caller, input_claim=claim)
        except ValueError as error:
            retained = error
        self.assertIsNotNone(retained)
        request = ids = None
        gc.collect()
        self.assertIsNone(reference())
        job.cancel()
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)
        claim.release()
        caller.close()

    def test_larger_backing_storage_is_rejected_before_capture(self):
        owner, worker, budget, profile = self.make_owner()
        selected, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        backing = torch.zeros((1024,), dtype=torch.int64)
        ids = backing[:6].view(2, 3)
        with self.assertRaisesRegex(ValueError, "full contiguous reported storage"):
            job.execute(ForwardRequest(selected.worker_request_id, {"input_ids": ids}),
                        input_owner=caller, input_claim=claim)
        self.assertEqual(job.state, "prepared")
        job.cancel()
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)
        backing = ids = None
        claim.release()
        caller.close()

    def test_ambient_autocast_is_rejected_before_input_pin(self):
        owner, worker, budget, profile = self.make_owner()
        selected, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        ids = torch.zeros((2, 3), dtype=torch.int64)
        request = ForwardRequest(selected.worker_request_id, {"input_ids": ids})
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                job.execute(request, input_owner=caller, input_claim=claim)
        self.assertEqual(job.state, "prepared")
        self.assertFalse(budget.snapshot().operations)
        job.cancel()
        self.assertTrue(job.abort_before_forward(timeout=5))
        owner.retire(job)
        request = ids = None
        claim.release()
        caller.close()

    def test_cancel_during_actual_stage_copy_keeps_claims_until_disposal(self):
        from looking_glass import trace_store
        from looking_glass.cpu_trace_owner import CpuTraceStateError
        owner, worker, budget, profile = self.make_owner()
        selected, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
        request = ForwardRequest(selected.worker_request_id, {"input_ids": ids})
        entered = threading.Event()
        resume = threading.Event()
        failures = []
        original = trace_store._clone_tensor

        def blocked_clone(value):
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("test clone timed out")
            return original(value)

        def run():
            try:
                job.execute(request, input_owner=caller, input_claim=claim)
            except BaseException as error:
                failures.append(error)

        with patch.object(trace_store, "_clone_tensor", blocked_clone):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                references = [weakref.ref(event.value) for event in job._result.events]
                job.cancel()
                snapshot = budget.snapshot()
                self.assertTrue(snapshot.operations)
                self.assertGreater(snapshot.used_bytes, 48)
                with self.assertRaises(CpuTraceStateError):
                    job.manifest()
            finally:
                resume.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertEqual(job.state, "failed")
        job.discard_result()
        # The caught error and all its traceback/attribute aliases must be gone
        # before the trusted owner attests source disposal.
        failures.clear()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        job.confirm_producer_disposed()
        self.assertTrue(job.dispose(timeout=5))
        owner.retire(job)
        request = ids = None
        claim.release()
        caller.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_cancel_after_input_pin_before_forward_can_be_drained(self):
        owner, worker, budget, profile = self.make_owner()
        selected, job = self.register_job(owner, profile)
        job.prepare()
        caller = budget.open_request("caller")
        claim = caller.reserve("caller", "input", 48, items=1)
        ids = torch.zeros((2, 3), dtype=torch.int64)
        request = ForwardRequest(selected.worker_request_id, {"input_ids": ids})
        entered, resume = threading.Event(), threading.Event()
        failures = []
        original = HostRequest.begin_operation

        def blocked_begin(instance, label, claims):
            result = original(instance, label, claims)
            if instance is caller:
                entered.set()
                if not resume.wait(10):
                    raise RuntimeError("test input pin timed out")
            return result

        def run():
            try:
                job.execute(request, input_owner=caller, input_claim=claim)
            except BaseException as error:
                failures.append(error)

        with patch.object(HostRequest, "begin_operation", blocked_begin), \
                patch.object(worker, "execute_traced", side_effect=AssertionError("forward ran")):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                job.cancel()
                self.assertEqual(claim.info().active_operations, 1)
            finally:
                resume.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)
        self.assertEqual(job.state, "failed")
        job.discard_result()
        failures.clear()
        gc.collect()
        job.confirm_producer_disposed()
        self.assertTrue(job.dispose(timeout=5))
        owner.retire(job)
        request = ids = None
        claim.release()
        caller.close()
        self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_interrupted_input_root_is_never_retained_without_pin(self):
        if sys.gettrace() is not None:
            self.skipTest("preserve an existing tracing session")
        for after_pin in (False, True):
            with self.subTest(after_pin=after_pin):
                owner, worker, budget, profile = self.make_owner()
                selected, job = self.register_job(owner, profile)
                job.prepare()
                caller = budget.open_request("caller")
                claim = caller.reserve("caller", "input", 48, items=1)
                ids = torch.zeros((2, 3), dtype=torch.int64)
                reference = weakref.ref(ids)
                request = ForwardRequest(selected.worker_request_id, {"input_ids": ids})
                method = type(job).execute
                source, first = inspect.getsourcelines(method)
                if after_pin:
                    index = next(i for i, line in enumerate(source)
                                 if "self._input_request = request" in line)
                    index = next(i for i in range(index + 1, len(source))
                                 if "with self._lock:" in source[i])
                else:
                    index = next(i for i, line in enumerate(source)
                                 if "self._input_operation = input_owner.begin_operation" in line)
                offsets = (offsets_at_line(method, first + index, "LOAD_FAST")
                           or offsets_at_line(method, first + index, "LOAD_DEREF"))
                self.assertTrue(offsets)
                target = offsets[0]

                def interrupt(frame, event, arg):
                    if frame.f_code is method.__code__:
                        if event == "call":
                            frame.f_trace_opcodes = True
                        elif event == "opcode" and frame.f_lasti == target:
                            sys.settrace(None)
                            raise KeyboardInterrupt("input pin boundary")
                    return interrupt

                with self.assertRaises(KeyboardInterrupt):
                    run_opcode_trace(interrupt, lambda: job.execute(
                        request, input_owner=caller, input_claim=claim))
                self.assertEqual(job.state, "quarantined")
                request = ids = None
                gc.collect()
                if after_pin:
                    self.assertIsNotNone(reference())
                    self.assertEqual(claim.info().active_operations, 1)
                    self.assertEqual([operation.label for operation in budget.snapshot().operations],
                                     [job._input_label])
                    with self.assertRaises(HostBusy):
                        claim.release()
                    # The test supervisor knows the worker never started and
                    # explicitly disposes its exact retained input reference.
                    job._input_request = None
                    gc.collect()
                    self.assertIsNone(reference())
                    job._input_operation.finish()
                    job._input_operation = None
                    job._input_state = "finished"
                else:
                    self.assertIsNone(job._input_request)
                    self.assertIsNone(reference())
                    self.assertEqual(claim.info().active_operations, 0)
                claim.release()
                caller.close()
                # Reconciliation here is test-only: no public API guesses how
                # to recover a quarantined production job.
                with job._lock:
                    job._phase = "prepared"
                self.assertTrue(job.abort_before_forward(timeout=5))
                owner.retire(job)
                self.assertEqual(budget.snapshot().used_bytes, 0)

    def test_real_worker_stages_a_complete_private_batch_under_claims(self):
        try:
            module = importlib.import_module("looking_glass.cpu_trace_owner")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "CPU trace owner is not implemented")
        from transformers import GPT2Config, GPT2LMHeadModel
        from looking_glass.traced_worker import TracedGPT2Worker
        config = GPT2Config(vocab_size=32, n_positions=8, n_ctx=8, n_embd=8, n_layer=1,
                            n_head=2, use_cache=False, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(71)
            model = GPT2LMHeadModel(config).eval()
        worker = TracedGPT2Worker("tiny-gpt2", model, device="cpu", max_batch_size=2,
                                  max_sequence_length=3, max_input_bytes=48,
                                  max_output_bytes=768)
        budget = HostBudget(max_bytes=2_000_000, max_items=64, max_requests=16,
                            max_operations=16)
        store = HostTraceStore(max_tensor_bytes=4096, max_items=8)
        owner = module.CpuTraceOwner(worker=worker, budget=budget, store=store,
                                     broker_id="a" * 32, process_generation="b" * 32,
                                     max_slots=4, max_event_bytes=64)
        profile = ProducerProfile(input_copies=ClaimLimit(96, 8),
                                  output=ClaimLimit(1536, 2),
                                  trace_bundle=ClaimLimit(128, 4),
                                  workspace=ClaimLimit(100_000, 1))
        selected = plan()
        owner.register(selected, profile=profile, registration_id="c" * 32, generation=1,
                       batch_size=2, sequence_length=3)
        job = owner.lookup(7, registration_id="c" * 32, generation=1)
        job.prepare()
        input_owner = budget.open_request("caller-input")
        input_claim = input_owner.reserve("caller", "input", 48, items=1)
        ids = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.int64)
        input_ref = weakref.ref(ids)
        request = ForwardRequest(selected.worker_request_id, {"input_ids": ids}, seed=71)
        # The owner uses a private, revalidated plan rather than caller-owned
        # frozen instances, which ordinary Python can still forcibly mutate.
        object.__setattr__(selected, "_run_id", "caller-mutated")
        object.__setattr__(selected, "_forward_index", 99)
        try:
            job.execute(request, input_owner=input_owner, input_claim=input_claim)
            with self.assertRaises(module.CpuTraceStateError):
                job.manifest()
            self.assertGreater(budget.snapshot().used_bytes, 48)
            job.discard_result()
            job.confirm_producer_disposed()
            job.seal()
            manifest = job.manifest()
            self.assertTrue(all(item.event.identity.run_id == "cpu-run"
                                and item.event.forward_index == 0 for item in manifest.events))
            self.assertEqual([(item.event.sequence, item.event.batch_positions,
                               item.event.layer_index, item.event.module_kind)
                              for item in manifest.events],
                             [(10, (0,), 0, "attn"), (11, (1,), 0, "attn"),
                              (12, (0,), 0, "mlp"), (13, (1,), 0, "mlp")])
            self.assertTrue(job.dispose(timeout=5))
            owner.retire(job)
            self.assertEqual(budget.snapshot().used_bytes, 48)
            request = ids = None
            gc.collect()
            self.assertIsNone(input_ref())
            input_claim.release()
            input_owner.close()
            self.assertEqual(budget.snapshot().used_bytes, 0)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
