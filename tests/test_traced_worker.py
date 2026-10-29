"""Offline CPU integration checks for trusted GPT-2 worker trace ownership."""
import gc
import importlib.util
import threading
import unittest
from unittest.mock import patch
import weakref

import torch
if importlib.util.find_spec("transformers") is not None:
    from transformers import GPT2Config, GPT2LMHeadModel
else:
    GPT2Config = GPT2LMHeadModel = None


@unittest.skipUnless(GPT2Config is not None, "requires transformers")
class TracedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("looking_glass.traced_worker"),
                             "trusted traced worker is missing")
        from looking_glass.capture_dispatcher import CaptureRequest, SampleSelection
        from looking_glass.resident_worker import ForwardRequest
        from looking_glass.traced_worker import TraceSite, TracedGPT2Worker
        self.CaptureRequest, self.SampleSelection = CaptureRequest, SampleSelection
        self.ForwardRequest, self.TraceSite, self.Worker = ForwardRequest, TraceSite, TracedGPT2Worker
        config = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16, n_embd=8,
                            n_layer=1, n_head=2, use_cache=False,
                            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            self.model = GPT2LMHeadModel(config).eval()
        self.inputs = torch.tensor([[1, 3, 5], [2, 4, 6]])

    def make_worker(self, **kwargs):
        return self.Worker("tiny-gpt2", self.model, device="cpu", max_batch_size=2,
                           max_sequence_length=3, **kwargs)

    def captures(self, forward_index=0):
        return tuple(self.CaptureRequest(
            run_id="run-1", job_id=f"job-{i}", request_id=f"request-{i}",
            samples=(self.SampleSelection(i, f"sample-{i}", (2, 0)),),
            forward_index=forward_index, generation_step=None,
            max_events=8, max_trace_bytes=4096,
        ) for i in range(2))

    def forward_request(self):
        return self.ForwardRequest("batch-1", {"input_ids": self.inputs}, seed=7)

    def traced(self, worker, *, captures=None, sites=None, sequence_starts=None,
               max_events=8, max_trace_bytes=4096, cancel_event=None):
        return worker.execute_traced(
            self.forward_request(), captures=self.captures() if captures is None else captures,
            sites=(self.TraceSite(0, "attn"), self.TraceSite(0, "mlp")) if sites is None else sites,
            sequence_starts={"run-1": 0} if sequence_starts is None else sequence_starts,
            max_events=max_events, max_trace_bytes=max_trace_bytes, cancel_event=cancel_event)

    def assert_no_hooks(self):
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in self.model.modules()))

    def test_traced_batch_keeps_request_identity_coordinates_and_logits(self):
        from looking_glass.trace_schema import TraceEvent, validate_trace
        reference_outputs = {}
        def capture_reference(kind):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                reference_outputs[kind] = tensor.detach().clone()
            return hook
        handles = [self.model.transformer.h[0].attn.register_forward_hook(capture_reference("attn")),
                   self.model.transformer.h[0].mlp.register_forward_hook(capture_reference("mlp"))]
        try:
            with torch.inference_mode():
                self.model(self.inputs)
        finally:
            for handle in handles:
                handle.remove()
        worker = self.make_worker()
        reference = worker.execute(self.forward_request())
        traced = self.traced(worker)
        self.assertTrue(torch.equal(reference.logits, traced.forward_result.logits))
        self.assertEqual(traced.forward_result.request_id, "batch-1")
        self.assertEqual(len(traced.events), 4)
        metadata = [event.to_trace_event() for event in traced.events]
        validate_trace(metadata)
        self.assertEqual([TraceEvent.from_json(event.to_json()) for event in metadata], metadata)
        self.assertEqual([e.identity.request_id for e in metadata],
                         ["request-0", "request-1", "request-0", "request-1"])
        self.assertEqual([e.identity.sample_id for e in metadata],
                         ["sample-0", "sample-1", "sample-0", "sample-1"])
        self.assertEqual([e.sequence for e in metadata], [0, 1, 2, 3])
        self.assertEqual([e.forward_index for e in metadata], [0, 0, 0, 0])
        self.assertEqual([e.invocation for e in metadata], [0, 0, 0, 0])
        self.assertEqual([e.output_path for e in metadata], [(0,), (0,), (), ()])
        self.assertTrue(all(e.shape == (2, 3, 8) and e.saved_shape == (2, 8)
                            for e in metadata))
        self.assertTrue(all(e.token_positions == (2, 0) and e.token_axis == 1
                            and e.generation.stage == "unknown"
                            and e.generation.cache_position is None for e in metadata))
        self.assertEqual([e.batch_positions for e in metadata], [(0,), (1,), (0,), (1,)])
        for event in traced.events:
            expected = reference_outputs[event.module_kind][event.batch_position,
                                                           list(event.token_positions), :]
            self.assertTrue(torch.equal(event.value, expected))
        self.assert_no_hooks()
        self.assertTrue(torch.equal(worker.execute(self.forward_request()).logits, reference.logits))

    def test_distinct_runs_in_one_batch_keep_independent_sequences_and_selections(self):
        from looking_glass.trace_schema import validate_trace
        worker = self.make_worker()
        captures = (
            self.CaptureRequest("run-a", "job-a", "request-a",
                                (self.SampleSelection(0, "sample-a", (2, 0)),),
                                4, None, 8, 4096),
            self.CaptureRequest("run-b", "job-b", "request-b",
                                (self.SampleSelection(1, "sample-b", (1,)),),
                                9, 7, 8, 4096),
        )
        traced = self.traced(worker, captures=captures,
                             sequence_starts={"run-a": 12, "run-b": 35})
        events = [event.to_trace_event() for event in traced.events]
        validate_trace(events)
        self.assertEqual([(e.identity.run_id, e.sequence, e.forward_index) for e in events],
                         [("run-a", 12, 4), ("run-b", 35, 9),
                          ("run-a", 13, 4), ("run-b", 36, 9)])
        self.assertEqual([e.token_positions for e in events], [(2, 0), (1,), (2, 0), (1,)])
        self.assertEqual([e.generation.generation_step for e in events], [None, 7, None, 7])
        self.assertTrue(all(e.generation.stage == "unknown" and e.generation.cache_position is None
                            for e in events))
        self.assert_no_hooks()

    def test_request_metadata_mutated_after_admission_cannot_reassign_a_sample(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        worker, captures, changes = self.make_worker(), self.captures(), []
        record = CaptureDispatcher.record
        def mutate_then_record(dispatcher, *args, **kwargs):
            if not changes:
                object.__setattr__(captures[0].samples[0], "batch_position", 1)
                object.__setattr__(captures[0], "request_id", "forged")
                changes.append(True)
            return record(dispatcher, *args, **kwargs)
        with patch.object(CaptureDispatcher, "record", mutate_then_record):
            traced = self.traced(worker, captures=captures)
        self.assertTrue(changes)
        self.assertEqual([event.batch_position for event in traced.events], [0, 1, 0, 1])
        self.assertEqual([event.request_id for event in traced.events],
                         ["request-0", "request-1", "request-0", "request-1"])
        self.assert_no_hooks()

    def test_returned_payload_is_ordinary_owned_mutable_cpu_tensor(self):
        worker = self.make_worker()
        captured = self.traced(worker).events[0].value
        original_inputs = self.inputs.clone()
        self.assertFalse(captured.is_inference())
        self.assertFalse(captured.requires_grad)
        self.assertEqual(captured.device.type, "cpu")
        captured.add_(1)
        self.assertTrue(torch.equal(self.inputs, original_inputs))

    def test_reused_site_counts_every_successful_invocation(self):
        worker = self.make_worker()
        model_forward, mlp = self.model.forward, self.model.transformer.h[0].mlp

        def two_calls(*args, **kwargs):
            mlp(torch.zeros(2, 3, 8))
            return model_forward(*args, **kwargs)

        with patch.object(self.model, "forward", two_calls):
            result = self.traced(worker, sites=(self.TraceSite(0, "mlp"),))
        self.assertEqual([event.invocation for event in result.events], [0, 0, 1, 1])
        self.assert_no_hooks()

    def test_caught_failed_site_invocation_rejects_later_capture(self):
        for exception_type in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception_type=exception_type):
                self.setUp()
                worker, calls = self.make_worker(), []
                model_forward, mlp = self.model.forward, self.model.transformer.h[0].mlp
                mlp_forward = mlp.forward

                def flaky(*args, **kwargs):
                    calls.append(True)
                    if len(calls) == 1:
                        raise exception_type("TEST first site invocation failed")
                    return mlp_forward(*args, **kwargs)

                def recover(*args, **kwargs):
                    try:
                        mlp(torch.zeros(2, 3, 8))
                    except exception_type:
                        pass
                    return model_forward(*args, **kwargs)

                with patch.object(mlp, "forward", flaky), \
                        patch.object(self.model, "forward", recover):
                    with self.assertRaisesRegex(RuntimeError, "invocation"):
                        self.traced(worker, sites=(self.TraceSite(0, "mlp"),))
                self.assertTrue(calls)
                self.assertTrue(worker.poisoned)
                self.assert_no_hooks()

    def test_caught_failed_site_without_retry_rejects_trace(self):
        for exception_type in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception_type=exception_type):
                self.setUp()
                worker, mlp = self.make_worker(), self.model.transformer.h[0].mlp

                def recover_without_retry(*args, **kwargs):
                    try:
                        mlp(torch.zeros(2, 3, 8))
                    except exception_type:
                        pass
                    return {"logits": torch.zeros(2, 3, 32)}

                with patch.object(mlp, "forward", side_effect=exception_type("site failure")), \
                        patch.object(self.model, "forward", recover_without_retry):
                    with self.assertRaisesRegex(RuntimeError, "invocation"):
                        self.traced(worker, sites=(self.TraceSite(0, "mlp"),))
                self.assertTrue(worker.poisoned)
                self.assert_no_hooks()

    def test_reentrant_site_invocation_is_rejected(self):
        worker, calls = self.make_worker(), []
        mlp = self.model.transformer.h[0].mlp
        original = mlp.forward

        def recurse(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                return mlp(*args, **kwargs)
            return original(*args, **kwargs)

        with patch.object(mlp, "forward", recurse):
            with self.assertRaisesRegex(RuntimeError, "invocation"):
                self.traced(worker, sites=(self.TraceSite(0, "mlp"),))
        self.assertTrue(worker.poisoned)
        self.assert_no_hooks()

    def test_forward_and_sequence_allocation_span_two_calls_without_collisions(self):
        from looking_glass.trace_schema import validate_trace
        worker = self.make_worker()
        first = self.traced(worker)
        second = self.traced(worker, captures=self.captures(forward_index=1),
                             sequence_starts={"run-1": 4})
        metadata = [e.to_trace_event() for e in first.events + second.events]
        validate_trace(metadata)
        self.assertEqual([e.sequence for e in metadata], list(range(8)))
        self.assertEqual([e.forward_index for e in metadata], [0] * 4 + [1] * 4)
        self.assert_no_hooks()

    def test_cancel_after_capture_discards_private_events_and_removes_hooks(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.resident_worker import RequestCancelled
        worker, cancelled = self.make_worker(), threading.Event()
        record = CaptureDispatcher.record
        references = []
        def capture_then_cancel(dispatcher, *args, **kwargs):
            record(dispatcher, *args, **kwargs)
            references.extend(weakref.ref(item.value) for item in dispatcher._events)
            cancelled.set()
        with patch.object(CaptureDispatcher, "record", capture_then_cancel):
            with self.assertRaises(RequestCancelled):
                self.traced(worker, cancel_event=cancelled)
        gc.collect()
        self.assertTrue(references)
        self.assertTrue(all(ref() is None for ref in references))
        self.assertFalse(worker.poisoned)
        self.assert_no_hooks()
        worker.execute(self.forward_request())

    def test_already_cancelled_request_never_snapshots_inputs(self):
        from looking_glass.resident_worker import RequestCancelled
        worker, cancelled = self.make_worker(), threading.Event()
        cancelled.set()
        with patch.object(worker, "snapshot_request", side_effect=AssertionError("should not snapshot")):
            with self.assertRaises(RequestCancelled):
                self.traced(worker, cancel_event=cancelled)
        self.assertFalse(worker.poisoned)
        self.assert_no_hooks()

    def test_output_limit_failure_discards_already_captured_events(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.resident_worker import OutputLimitExceeded
        worker = self.make_worker(max_output_bytes=1)
        record = CaptureDispatcher.record
        references = []
        def capture_and_track(dispatcher, *args, **kwargs):
            record(dispatcher, *args, **kwargs)
            references.extend(weakref.ref(item.value) for item in dispatcher._events)
        with patch.object(CaptureDispatcher, "record", capture_and_track):
            with self.assertRaises(OutputLimitExceeded):
                self.traced(worker)
        gc.collect()
        self.assertTrue(references)
        self.assertTrue(all(ref() is None for ref in references))
        self.assertFalse(worker.poisoned)
        self.assert_no_hooks()

    def test_capacity_failure_releases_partial_trace_with_exception_retained(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.resident_worker import WorkerExecutionError
        worker, references, errors = self.make_worker(), [], []
        record = CaptureDispatcher.record
        def track(dispatcher, *args, **kwargs):
            try:
                return record(dispatcher, *args, **kwargs)
            finally:
                references.extend(weakref.ref(item.value) for item in dispatcher._events)
        with patch.object(CaptureDispatcher, "record", track):
            try:
                self.traced(worker, max_trace_bytes=128)
            except WorkerExecutionError as exc:
                errors.append(exc)
        self.assertEqual(len(errors), 1)
        self.assertTrue(references)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertTrue(worker.poisoned)
        self.assert_no_hooks()

    def test_oom_during_slicing_releases_prior_events_and_poisons_worker(self):
        from looking_glass.resident_worker import WorkerOutOfMemory
        worker, references, errors, calls = self.make_worker(), [], [], []
        index_select = torch.Tensor.index_select
        def fail_after_first_event(tensor, *args, **kwargs):
            calls.append(True)
            if len(calls) == 3:
                raise torch.OutOfMemoryError("simulated CPU allocation failure")
            value = index_select(tensor, *args, **kwargs)
            references.append(weakref.ref(value))
            return value
        with patch.object(torch.Tensor, "index_select", fail_after_first_event):
            try:
                self.traced(worker)
            except WorkerOutOfMemory as exc:
                errors.append(exc)
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(calls), 3)
        gc.collect()
        self.assertTrue(references)
        self.assertTrue(all(ref() is None for ref in references))
        self.assertTrue(worker.poisoned)
        self.assert_no_hooks()

    def test_failed_worker_postcheck_releases_completed_trace(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.model_ownership import ModelLease
        from looking_glass.resident_worker import WorkerExecutionError
        worker, references, errors = self.make_worker(), [], []
        record, forward = CaptureDispatcher.record, self.model.forward
        def track(dispatcher, *args, **kwargs):
            record(dispatcher, *args, **kwargs)
            references.extend(weakref.ref(item.value) for item in dispatcher._events)
        def mutate_after_forward(*args, **kwargs):
            output = forward(*args, **kwargs)
            self.model.transformer.wte.weight.add_(1)
            return output
        with patch.object(CaptureDispatcher, "record", track), \
                patch.object(self.model, "forward", side_effect=mutate_after_forward):
            try:
                self.traced(worker)
            except WorkerExecutionError as exc:
                errors.append(exc)
        self.assertEqual(len(errors), 1)
        gc.collect()
        self.assertTrue(references)
        self.assertTrue(all(ref() is None for ref in references))
        self.assertTrue(worker.poisoned)
        self.assert_no_hooks()
        # A changed model instance must not be leased as though still verified.
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_interrupted_worker_postcheck_quarantines_mutated_model(self):
        from looking_glass.model_ownership import ModelLease
        worker, checks = self.make_worker(), []
        model_state, forward = worker._model_state, self.model.forward
        def interrupted_postcheck():
            checks.append(True)
            if len(checks) == 2:
                raise KeyboardInterrupt("TEST interrupted postcheck")
            return model_state()
        def mutate_after_forward(*args, **kwargs):
            output = forward(*args, **kwargs)
            self.model.transformer.wte.weight.add_(1)
            return output
        with patch.object(worker, "_model_state", interrupted_postcheck), \
                patch.object(self.model, "forward", side_effect=mutate_after_forward):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.traced(worker)
        self.assertIn("TEST interrupted postcheck", str(caught.exception))
        self.assertTrue(worker.poisoned)
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_partial_hook_registration_failure_removes_earlier_hooks(self):
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        mlp = self.model.transformer.h[0].mlp
        with patch.object(mlp, "register_forward_hook", side_effect=RuntimeError("registration failed")):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                self.traced(worker)
        self.assert_no_hooks()
        self.assertTrue(worker.poisoned)
        # Even a registration error before a handle is returned is indeterminate.
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_failed_hook_removal_keeps_model_lease_quarantined(self):
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        mlp = self.model.transformer.h[0].mlp
        register = mlp.register_forward_hook
        class BrokenHandle:
            def __init__(self, real):
                self.real = real
            def remove(self):
                raise RuntimeError("simulated removal failure")
        with patch.object(mlp, "register_forward_hook",
                          side_effect=lambda hook, **kw: BrokenHandle(register(hook, **kw))):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                self.traced(worker)
        self.assertTrue(worker.poisoned)
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_silent_hook_removal_failure_also_quarantines_model(self):
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        mlp, register = self.model.transformer.h[0].mlp, self.model.transformer.h[0].mlp.register_forward_hook
        class NoopHandle:
            def __init__(self, real):
                self.real = real
            def remove(self):
                return None
        with patch.object(mlp, "register_forward_hook",
                          side_effect=lambda hook, **kw: NoopHandle(register(hook, **kw))):
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                self.traced(worker)
        self.assertTrue(worker.poisoned)
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_oom_and_hook_cleanup_failure_both_remain_diagnosable(self):
        worker = self.make_worker()
        mlp, register = self.model.transformer.h[0].mlp, self.model.transformer.h[0].mlp.register_forward_hook
        class BrokenHandle:
            def __init__(self, real):
                self.real = real
            def remove(self):
                raise RuntimeError("TEST cleanup removal failed")
        with patch.object(mlp, "register_forward_hook",
                          side_effect=lambda hook, **kw: BrokenHandle(register(hook, **kw))), \
                patch.object(torch.Tensor, "index_select", side_effect=torch.OutOfMemoryError("TEST OOM")):
            with self.assertRaises(RuntimeError) as caught:
                self.traced(worker)
        message = str(caught.exception)
        self.assertIn("quarantined", message)
        self.assertIn("OutOfMemory", message)
        self.assertIn("TEST OOM", message)
        self.assertIn("TEST cleanup removal failed", message)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(worker.poisoned)

    def test_interrupt_during_hook_cleanup_propagates_and_quarantines(self):
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        mlp, register = self.model.transformer.h[0].mlp, self.model.transformer.h[0].mlp.register_forward_hook
        class InterruptedHandle:
            def __init__(self, real):
                self.real = real
            def remove(self):
                raise KeyboardInterrupt("TEST interrupted cleanup")
        with patch.object(mlp, "register_forward_hook",
                          side_effect=lambda hook, **kw: InterruptedHandle(register(hook, **kw))):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.traced(worker)
        self.assertIn("TEST interrupted cleanup", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(worker.poisoned)
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_interrupt_during_final_verification_propagates_and_quarantines(self):
        from looking_glass.model_ownership import ModelLease
        worker, acquired = self.make_worker(), []
        acquire, verify = ModelLease.acquire, ModelLease.verify_tree
        def remember_acquire(lease):
            result = acquire(lease)
            acquired.append(lease)
            return result
        def interrupt_final(lease):
            if lease in acquired:
                raise KeyboardInterrupt("TEST interrupted verification")
            return verify(lease)
        with patch.object(ModelLease, "acquire", remember_acquire), \
                patch.object(ModelLease, "verify_tree", interrupt_final):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.traced(worker)
        self.assertIn("TEST interrupted verification", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(worker.poisoned)
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_interrupted_lease_release_recovers_before_and_after_publication(self):
        from looking_glass.model_ownership import ModelLease
        for after_publication in (False, True):
            with self.subTest(after_publication=after_publication):
                self.setUp()
                worker, calls, other = self.make_worker(), [], []
                release = ModelLease.release

                def interrupt_once(lease):
                    calls.append(True)
                    if len(calls) == 1:
                        if after_publication:
                            release(lease)
                            other.append(ModelLease(self.model).acquire())
                        raise KeyboardInterrupt("TEST interrupted release")
                    return release(lease)

                with patch.object(ModelLease, "release", interrupt_once):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        self.traced(worker)
                self.assertIn("TEST interrupted release", str(caught.exception))
                self.assertEqual(len(calls), 2)
                self.assertFalse(worker.poisoned)
                self.assert_no_hooks()
                if after_publication:
                    # Retrying the stale lease must not revoke a new owner.
                    with self.assertRaises(RuntimeError):
                        ModelLease(self.model).acquire()
                    release(other[0])
                else:
                    ModelLease(self.model).acquire().release()
                self.assertEqual(len(self.traced(worker).events), 4)

    def test_failed_lease_release_retry_quarantines_and_keeps_both_diagnostics(self):
        from looking_glass.model_ownership import ModelLease
        for interrupted_first in (False, True):
            with self.subTest(interrupted_first=interrupted_first):
                self.setUp()
                worker, calls = self.make_worker(), []

                def fail_release(_lease):
                    calls.append(True)
                    if (len(calls) == 1) == interrupted_first:
                        raise KeyboardInterrupt("TEST interrupted release")
                    raise RuntimeError("TEST release failure")

                with patch.object(ModelLease, "release", fail_release):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        self.traced(worker)
                self.assertEqual(len(calls), 2)
                self.assertIn("TEST interrupted release", str(caught.exception))
                self.assertIn("TEST release failure", str(caught.exception))
                self.assertIsNone(caught.exception.__context__)
                self.assertIsNone(caught.exception.__cause__)
                self.assertTrue(worker.poisoned)
                self.assertIsNotNone(worker._quarantined_lease)
                with self.assertRaises(RuntimeError):
                    ModelLease(self.model).acquire()

    def test_failed_stale_release_retry_never_revokes_later_owner(self):
        from looking_glass.model_ownership import ModelLease
        worker, calls, other = self.make_worker(), [], []
        release = ModelLease.release

        def fail_after_publication(lease):
            calls.append(True)
            if len(calls) == 1:
                release(lease)
                other.append(ModelLease(self.model).acquire())
                raise KeyboardInterrupt("TEST interrupted after publication")
            raise RuntimeError("TEST retry failure")

        with patch.object(ModelLease, "release", fail_after_publication):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.traced(worker)
        self.assertIn("TEST interrupted after publication", str(caught.exception))
        self.assertIn("TEST retry failure", str(caught.exception))
        self.assertTrue(worker.poisoned)
        self.assertIsNotNone(worker._quarantined_lease)
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()
        release(other[0])
        # The retained stale handle is not proof of continued ownership.
        ModelLease(self.model).acquire().release()

    def test_interrupt_while_poisoning_snapshot_failure_is_preserved(self):
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        with patch.object(torch.Tensor, "clone", side_effect=torch.OutOfMemoryError("TEST OOM")), \
                patch.object(worker, "_model_state", side_effect=KeyboardInterrupt("TEST poison interrupted")):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.traced(worker)
        self.assertIn("TEST poison interrupted", str(caught.exception))
        self.assertTrue(worker.poisoned)
        with self.assertRaises(RuntimeError):
            ModelLease(self.model).acquire()

    def test_interrupt_during_forward_clears_partial_trace_and_hooks(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.model_ownership import ModelLease
        for interrupt_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(interrupt_type=interrupt_type):
                self.setUp()
                worker, references = self.make_worker(), []
                record = CaptureDispatcher.record

                def track(dispatcher, *args, **kwargs):
                    record(dispatcher, *args, **kwargs)
                    references.extend(weakref.ref(item.value) for item in dispatcher._events)

                with patch.object(CaptureDispatcher, "record", track), \
                        patch.object(self.model.transformer.h[0].mlp, "forward",
                                     side_effect=interrupt_type("TEST forward interrupted")):
                    with self.assertRaises(interrupt_type) as caught:
                        self.traced(worker)
                self.assertIn("TEST forward interrupted", str(caught.exception))
                self.assertTrue(references)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in references))
                self.assertTrue(worker.poisoned)
                self.assert_no_hooks()
                # The synchronous forward has unwound; no external accesses
                # are in flight under this trusted-owner contract.
                ModelLease(self.model).acquire().release()

    def test_forward_failure_survives_interrupted_poison_for_both_apis(self):
        from looking_glass.model_ownership import ModelLease
        for traced in (False, True):
            with self.subTest(traced=traced):
                self.setUp()
                worker = self.make_worker()
                original_poison, calls = worker._poison, []

                def interrupted_poison():
                    calls.append(True)
                    with patch.object(worker, "_model_state",
                                      side_effect=KeyboardInterrupt("TEST interrupted poison")):
                        return original_poison()

                def failed_forward(*args, **kwargs):
                    raise RuntimeError("TEST initial forward failure")

                with patch.object(worker, "_poison", interrupted_poison), \
                        patch.object(self.model, "forward", failed_forward):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        if traced:
                            self.traced(worker)
                        else:
                            worker.execute(self.forward_request())
                self.assertIn("TEST interrupted poison", str(caught.exception))
                self.assertIn("TEST initial forward failure", str(caught.exception))
                self.assertTrue(calls)
                self.assertIsNone(caught.exception.__context__)
                self.assertIsNone(caught.exception.__cause__)
                self.assertTrue(worker.poisoned)
                with self.assertRaises(RuntimeError):
                    ModelLease(self.model).acquire()

    def test_forward_failure_diagnostic_does_not_leak_into_later_call(self):
        from looking_glass.resident_worker import OutputLimitExceeded
        for first_is_traced in (False, True):
            with self.subTest(first_is_traced=first_is_traced):
                self.setUp()
                worker = self.make_worker(max_output_bytes=1)
                with self.assertRaises(OutputLimitExceeded):
                    if first_is_traced:
                        self.traced(worker)
                    else:
                        worker.execute(self.forward_request())
                self.assertFalse(worker.poisoned)
                state, checks = worker._model_state, []

                def interrupt_next_check():
                    checks.append(True)
                    if len(checks) == 1:
                        raise KeyboardInterrupt("TEST unrelated interrupt")
                    return state()

                with patch.object(worker, "_model_state", interrupt_next_check):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        if first_is_traced:
                            worker.execute(self.forward_request())
                        else:
                            self.traced(worker)
                self.assertIn("TEST unrelated interrupt", str(caught.exception))
                self.assertNotIn("OutputLimitExceeded", str(caught.exception))
                self.assertNotIn("logits exceed", str(caught.exception))
                self.assert_no_hooks()
    def test_final_lease_failure_does_not_retain_results_via_exception_traceback(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.model_ownership import ModelLease
        worker, references, errors, acquired = self.make_worker(), [], [], []
        record, acquire, verify = CaptureDispatcher.record, ModelLease.acquire, ModelLease.verify_tree
        def remember_acquire(lease):
            result = acquire(lease)
            acquired.append(lease)
            return result
        def track(dispatcher, *args, **kwargs):
            record(dispatcher, *args, **kwargs)
            references.extend(weakref.ref(item.value) for item in dispatcher._events)
        def fail_final_verification(lease):
            if lease in acquired:
                raise RuntimeError("verification failed")
            return verify(lease)
        with patch.object(CaptureDispatcher, "record", track), \
                patch.object(ModelLease, "acquire", remember_acquire), \
                patch.object(ModelLease, "verify_tree", fail_final_verification):
            try:
                self.traced(worker)
            except RuntimeError as exc:
                errors.append(exc)
        self.assertEqual(len(errors), 1)
        self.assertTrue(references)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertTrue(worker.poisoned)

    def test_failed_preflight_does_not_retain_inputs_or_snapshot_in_traceback(self):
        worker, references, errors = self.make_worker(), [], []
        snapshot_request = worker.snapshot_request
        def track_snapshot(request):
            snapshot = snapshot_request(request)
            references.append(weakref.ref(snapshot.inputs["input_ids"]))
            return snapshot
        ephemeral = torch.tensor([[1, 3, 5], [2, 4, 6]])
        references.append(weakref.ref(ephemeral))
        request = self.ForwardRequest("temporary", {"input_ids": ephemeral})
        with patch.object(worker, "snapshot_request", track_snapshot):
            try:
                worker.execute_traced(request, captures=self.captures(),
                                      sites=(self.TraceSite(8, "attn"),),
                                      sequence_starts={"run-1": 0},
                                      max_events=8, max_trace_bytes=4096)
            except ValueError as exc:
                errors.append(exc)
        request = ephemeral = None
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(references), 2)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        self.assertFalse(worker.poisoned)
        self.assert_no_hooks()

    def test_invalid_input_does_not_survive_only_via_exception_traceback(self):
        from looking_glass.resident_worker import InvalidRequest
        worker, errors = self.make_worker(), []
        ephemeral = torch.tensor([[1.0, 2.0, 3.0]])
        reference = weakref.ref(ephemeral)
        request = self.ForwardRequest("temporary", {"input_ids": ephemeral})
        try:
            worker.execute_traced(request, captures=self.captures(),
                                  sites=(self.TraceSite(0, "attn"),),
                                  sequence_starts={"run-1": 0},
                                  max_events=8, max_trace_bytes=4096)
        except InvalidRequest as exc:
            errors.append(exc)
        request = ephemeral = None
        self.assertEqual(len(errors), 1)
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(worker.poisoned)

    def test_untraced_invalid_input_does_not_survive_only_via_traceback(self):
        from looking_glass.resident_worker import InvalidRequest
        worker, errors = self.make_worker(), []
        ephemeral = torch.tensor([[1.0, 2.0, 3.0]])
        reference = weakref.ref(ephemeral)
        request = self.ForwardRequest("temporary", {"input_ids": ephemeral})
        try:
            worker.execute(request)
        except InvalidRequest as exc:
            errors.append(exc)
        request = ephemeral = None
        self.assertEqual(len(errors), 1)
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(worker.poisoned)

    def test_other_model_owner_blocks_trace_without_poisoning_worker(self):
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        lease = ModelLease(self.model).acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.traced(worker)
            self.assertFalse(worker.poisoned)
            self.assert_no_hooks()
        finally:
            lease.release()
        self.assertEqual(len(self.traced(worker).events), 4)

    def test_busy_model_owner_is_rejected_before_input_snapshot(self):
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        lease = ModelLease(self.model).acquire()
        try:
            with patch.object(worker, "snapshot_request", side_effect=AssertionError("snapshot while busy")):
                with self.assertRaises(RuntimeError):
                    self.traced(worker)
            self.assertFalse(worker.poisoned)
        finally:
            lease.release()

    def test_running_trace_blocks_another_owner_until_hook_cleanup(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.model_ownership import ModelLease
        worker = self.make_worker()
        entered, release = threading.Event(), threading.Event()
        record, outcomes = CaptureDispatcher.record, []
        def pause_in_hook(dispatcher, *args, **kwargs):
            record(dispatcher, *args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test hook timed out")
        def call():
            try:
                outcomes.append(self.traced(worker))
            except BaseException as exc:
                outcomes.append(exc)
        with patch.object(CaptureDispatcher, "record", pause_in_hook):
            thread = threading.Thread(target=call)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RuntimeError):
                    ModelLease(self.model).acquire()
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], self.import_result_type())
        self.assert_no_hooks()
        ModelLease(self.model).acquire().release()

    def test_legacy_trace_session_and_worker_exclude_each_other_both_ways(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.probes import TraceSession
        worker = self.make_worker()
        with TraceSession(self.model, run_id="legacy"):
            with self.assertRaises(RuntimeError):
                self.traced(worker)
        self.assertFalse(worker.poisoned)
        entered, release, results = threading.Event(), threading.Event(), []
        record = CaptureDispatcher.record
        def hold(dispatcher, *args, **kwargs):
            record(dispatcher, *args, **kwargs)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test hook timed out")
        def call():
            try:
                results.append(self.traced(worker))
            except BaseException as exc:
                results.append(exc)
        with patch.object(CaptureDispatcher, "record", hold):
            thread = threading.Thread(target=call, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RuntimeError):
                    with TraceSession(self.model, run_id="legacy-overlap"):
                        pass
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], self.import_result_type())
        self.assert_no_hooks()

    def test_competing_callers_keep_batch_logits_and_trace_values_separate(self):
        worker, start, outcomes, errors = self.make_worker(), threading.Event(), {}, []
        inputs = {"a": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                  "b": torch.tensor([[9, 8, 7], [6, 5, 4]])}
        def run(name):
            captures = tuple(self.CaptureRequest(
                f"run-{name}", f"job-{name}{i}", f"request-{name}{i}",
                (self.SampleSelection(i, f"sample-{name}{i}", (0, 2)),), 0, None, 8, 4096)
                for i in range(2))
            request = self.ForwardRequest(f"batch-{name}", {"input_ids": inputs[name]}, seed=7)
            return worker.execute_traced(
                request, captures=captures, sites=(self.TraceSite(0, "attn"), self.TraceSite(0, "mlp")),
                sequence_starts={f"run-{name}": 0}, max_events=8, max_trace_bytes=4096)
        expected = {name: run(name) for name in ("a", "b")}
        def call(name):
            try:
                if not start.wait(10):
                    raise RuntimeError("test start timed out")
                outcomes[name] = run(name)
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=call, args=(name,), daemon=True) for name in ("a", "b")]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(20)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertFalse(errors)
        for name in ("a", "b"):
            actual, reference = outcomes[name], expected[name]
            self.assertEqual(actual.forward_result.request_id, f"batch-{name}")
            self.assertTrue(torch.equal(actual.forward_result.logits, reference.forward_result.logits))
            self.assertEqual(len(actual.events), 4)
            for event, baseline in zip(actual.events, reference.events):
                self.assertEqual(event.request_id, baseline.request_id)
                self.assertEqual(event.run_id, f"run-{name}")
                self.assertTrue(torch.equal(event.value, baseline.value))
        self.assert_no_hooks()

    @staticmethod
    def import_result_type():
        from looking_glass.traced_worker import TracedForwardResult
        return TracedForwardResult

    def test_repeated_root_forward_within_one_dispatch_fails_closed(self):
        from looking_glass.resident_worker import ResidentWorker
        worker = self.make_worker()
        perform = ResidentWorker._perform
        def perform_twice(instance, request):
            perform(instance, request)
            return perform(instance, request)
        with patch.object(ResidentWorker, "_perform", perform_twice):
            with self.assertRaises(RuntimeError):
                self.traced(worker)
        self.assertTrue(worker.poisoned)
        self.assert_no_hooks()

    def test_tracing_gpu_is_rejected_without_admission(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.Worker("tiny-gpt2", self.model, device="cuda:0")
        self.assert_no_hooks()

    def test_cache_inputs_are_rejected_before_a_forward(self):
        from looking_glass.resident_worker import InvalidRequest
        worker = self.make_worker()
        request = self.ForwardRequest("batch", {"input_ids": self.inputs,
                                                 "past_key_values": torch.ones(1, 1)})
        with self.assertRaises(InvalidRequest):
            worker.execute_traced(request, captures=self.captures(),
                                  sites=(self.TraceSite(0, "attn"),),
                                  sequence_starts={"run-1": 0},
                                  max_events=8, max_trace_bytes=4096)
        self.assertFalse(worker.poisoned)
        self.assert_no_hooks()

    def test_invalid_site_and_aliased_module_fail_closed(self):
        worker = self.make_worker()
        with self.assertRaises((TypeError, ValueError)):
            self.traced(worker, sites=(self.TraceSite(0, "arbitrary"),))
        self.assertFalse(worker.poisoned)
        self.assert_no_hooks()
        worker.close()
        self.model.other_attention_reference = self.model.transformer.h[0].attn
        with self.assertRaises((TypeError, ValueError)):
            self.make_worker()
        self.assert_no_hooks()

    def test_string_subclass_cannot_spoof_the_trace_site_whitelist(self):
        class ImpersonatedRole(str):
            def __eq__(self, other):
                return other == "attn"
            __hash__ = str.__hash__
        with self.assertRaises((TypeError, ValueError)):
            self.TraceSite(0, ImpersonatedRole("ln_1"))

    def test_mutated_site_object_cannot_bypass_the_site_whitelist(self):
        worker = self.make_worker()
        site = self.TraceSite(0, "attn")
        object.__setattr__(site, "module_kind", "ln_1")
        with self.assertRaises((TypeError, ValueError)):
            self.traced(worker, sites=(site,))
        self.assertFalse(worker.poisoned)
        self.assert_no_hooks()

    def test_replaced_attention_child_is_not_labeled_as_gpt2_attention(self):
        self.model.transformer.h[0].attn = torch.nn.Identity()
        with self.assertRaises((TypeError, ValueError)):
            self.make_worker()
        self.assert_no_hooks()


if __name__ == "__main__":
    unittest.main()
