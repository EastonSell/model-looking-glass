import gc
import inspect
import sys
from contextlib import nullcontext
import threading
import unittest
from unittest.mock import Mock, patch
import weakref
from types import SimpleNamespace

import torch
from torch import nn

from looking_glass import resident_worker as worker_module
from looking_glass.resident_worker import (
    ForwardRequest, InvalidRequest, OutputLimitExceeded, RequestCancelled,
    ResidentWorker, WorkerExecutionError, WorkerOutOfMemory, WorkerSynchronizationError,
    WorkerUnavailable,
)


class SmallModel(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        value = input_ids.float()
        if attention_mask is not None:
            value = value * attention_mask
        return {"logits": torch.stack((value, value + 1), dim=-1)}


class RandomModel(nn.Module):
    def forward(self, input_ids):
        return {"logits": torch.rand((*input_ids.shape, 2))}


class ResidentWorkerTests(unittest.TestCase):
    def setUp(self):
        # Keep the deliberately permanent quarantine local to each test.
        patcher = patch.object(worker_module, "_process_quarantined", threading.Event(), create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        references = patch.object(worker_module, "_quarantine_references", [], create=True)
        references.start()
        self.addCleanup(references.stop)

    def test_snapshot_and_result_are_independently_owned(self):
        worker = ResidentWorker("small", SmallModel().eval(), device="cpu")
        original = torch.tensor([[2, 3]])
        snapshot = worker.snapshot_request(ForwardRequest("r1", {"input_ids": original}))
        original.fill_(9)
        result = worker.execute(snapshot)
        self.assertEqual(result.request_id, "r1")
        self.assertTrue(torch.equal(result.logits, torch.tensor([[[2., 3.], [3., 4.]]])))
        self.assertFalse(result.logits.requires_grad)
        self.assertEqual(result.logits.device.type, "cpu")
        result.logits.fill_(7)
        again = worker.execute(snapshot)
        self.assertTrue(torch.equal(again.logits, torch.tensor([[[2., 3.], [3., 4.]]])))

    def test_seeded_requests_restore_rng_and_ignore_execution_order(self):
        worker = ResidentWorker("random", RandomModel().eval(), device="cpu")
        before = torch.random.get_rng_state().clone()
        first = worker.execute(ForwardRequest("first", {"input_ids": torch.tensor([[1, 2]])}, seed=123))
        worker.execute(ForwardRequest("middle", {"input_ids": torch.tensor([[1, 2]])}, seed=999))
        last = worker.execute(ForwardRequest("last", {"input_ids": torch.tensor([[1, 2]])}, seed=123))
        self.assertTrue(torch.equal(first.logits, last.logits))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_unsupported_input_is_rejected_without_poisoning_worker(self):
        worker = ResidentWorker("small", SmallModel().eval(), device="cpu")
        for inputs in ({"input_ids": torch.tensor([1, 2])},
                       {"input_ids": torch.tensor([[1]]), "past_key_values": torch.tensor([[1]])},
                       {"input_ids": torch.tensor([[1.0]])}):
            with self.subTest(inputs=inputs):
                with self.assertRaises(InvalidRequest):
                    worker.execute(ForwardRequest("bad", inputs))
        self.assertFalse(worker.poisoned)
        self.assertTrue(torch.equal(worker.execute(ForwardRequest("ok", {"input_ids": torch.tensor([[5]])})).logits,
                                    torch.tensor([[[5., 6.]]])))

    def test_running_cancellation_discards_result_then_worker_remains_usable(self):
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()

        class BlockingModel(SmallModel):
            def forward(self, input_ids, attention_mask=None):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test forward timed out")
                return super().forward(input_ids, attention_mask)

        worker = ResidentWorker("blocking", BlockingModel().eval())
        errors, results = [], []

        def run():
            try:
                results.append(worker.execute(ForwardRequest("one", {"input_ids": torch.tensor([[1]])}), cancelled))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(entered.wait(5))
        cancelled.set()
        release.set()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RequestCancelled)
        self.assertFalse(worker.poisoned)
        result = worker.execute(ForwardRequest("next", {"input_ids": torch.tensor([[2]])}))
        self.assertTrue(torch.equal(result.logits, torch.tensor([[[2., 3.]]])))

    def test_cancelled_request_waiting_on_worker_never_enters_model(self):
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        seen = []

        class BlockingModel(SmallModel):
            def forward(self, input_ids, attention_mask=None):
                seen.append(int(input_ids[0, 0]))
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test forward timed out")
                return super().forward(input_ids, attention_mask)

        worker = ResidentWorker("serial", BlockingModel().eval())
        errors = []

        def call(token, event=None):
            try:
                worker.execute(ForwardRequest(str(token), {"input_ids": torch.tensor([[token]])}), event)
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(target=call, args=(1,))
        second = threading.Thread(target=call, args=(2, cancelled))
        first.start()
        self.assertTrue(entered.wait(5))
        second.start()
        cancelled.set()
        release.set()
        first.join(5)
        second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(seen, [1])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RequestCancelled)

    def test_oom_poison_drops_model_and_temporary_traceback_tensors(self):
        temporary_refs = []

        class OomModel(nn.Module):
            def forward(self, input_ids):
                temporary = torch.ones(64)
                temporary_refs.append(weakref.ref(temporary))
                raise torch.OutOfMemoryError("simulated failure")

        model = OomModel().eval()
        model_ref = weakref.ref(model)
        worker = ResidentWorker("oom", model)
        del model
        with self.assertRaises(WorkerOutOfMemory) as caught:
            worker.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}))
        self.assertTrue(worker.poisoned)
        self.assertIsNone(caught.exception.__context__)
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertIsNone(temporary_refs[0]())
        with self.assertRaises(WorkerUnavailable):
            worker.execute(ForwardRequest("again", {"input_ids": torch.tensor([[1]])}))

    def test_failed_forward_restores_rng_and_poison_rejects_reuse(self):
        class FailingModel(nn.Module):
            def forward(self, input_ids):
                torch.rand(5)
                raise RuntimeError("failure after allocation")

        worker = ResidentWorker("failure", FailingModel().eval())
        before = torch.random.get_rng_state().clone()
        with self.assertRaises(WorkerExecutionError):
            worker.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}, seed=10))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(worker.poisoned)
        with self.assertRaises(WorkerUnavailable):
            worker.execute(ForwardRequest("again", {"input_ids": torch.tensor([[1]])}))

    def test_output_limit_fails_without_silent_truncation(self):
        worker = ResidentWorker("small", SmallModel().eval(), max_output_bytes=8)
        with self.assertRaises(OutputLimitExceeded):
            worker.execute(ForwardRequest("large", {"input_ids": torch.tensor([[1, 2]])}))
        self.assertFalse(worker.poisoned)
        result = worker.execute(ForwardRequest("small", {"input_ids": torch.tensor([[3]])}))
        self.assertTrue(torch.equal(result.logits, torch.tensor([[[3., 4.]]])))

    def test_output_is_copied_into_exact_contiguous_cpu_storage(self):
        class NoncontiguousModel(nn.Module):
            def forward(self, input_ids):
                values = input_ids.float().transpose(0, 1)
                return {'logits': torch.stack((values, values + 1), dim=-1).transpose(0, 1)}

        worker = ResidentWorker('layout', NoncontiguousModel().eval(), max_batch_size=2,
                                max_output_bytes=48)
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = worker.execute(ForwardRequest('r', {'input_ids': inputs}))
        self.assertTrue(result.logits.is_contiguous())
        self.assertEqual(result.logits.storage_offset(), 0)
        self.assertEqual(result.logits.untyped_storage().nbytes(),
                         result.logits.numel() * result.logits.element_size())
        self.assertTrue(torch.equal(result.logits[..., 0], inputs.float()))

    def test_mutating_registered_buffer_poisoned_before_result_is_returned(self):
        class StatefulModel(SmallModel):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros(1))

            def forward(self, input_ids, attention_mask=None):
                self.counter.add_(1)
                return super().forward(input_ids, attention_mask)

        worker = ResidentWorker("stateful", StatefulModel().eval())
        with self.assertRaises(WorkerExecutionError):
            worker.execute(ForwardRequest("stateful", {"input_ids": torch.tensor([[1]])}))
        self.assertTrue(worker.poisoned)

    def test_hooks_and_training_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            ResidentWorker("training", SmallModel())
        model = SmallModel().eval()
        handle = model.register_forward_hook(lambda module, inputs, output: None)
        try:
            with self.assertRaises(ValueError):
                ResidentWorker("hooks", model)
        finally:
            handle.remove()

    def test_global_forward_hooks_cannot_intercept_worker_requests(self):
        worker = ResidentWorker("small", SmallModel().eval())
        handle = nn.modules.module.register_module_forward_hook(lambda module, inputs, output: None)
        try:
            with self.assertRaises(WorkerUnavailable):
                worker.execute(ForwardRequest("blocked", {"input_ids": torch.tensor([[1]])}))
            self.assertTrue(worker.poisoned)
        finally:
            handle.remove()

    def test_close_is_idempotent_and_rejects_subsequent_work(self):
        worker = ResidentWorker("small", SmallModel().eval())
        worker.close()
        worker.close()
        with self.assertRaises(WorkerUnavailable):
            worker.execute(ForwardRequest("late", {"input_ids": torch.tensor([[1]])}))

    def test_pre_cancelled_request_does_not_allocate_snapshot(self):
        worker = ResidentWorker("small", SmallModel().eval())
        cancelled = threading.Event()
        cancelled.set()
        with patch.object(worker, "snapshot_request", side_effect=AssertionError("copied after cancellation")):
            with self.assertRaises(RequestCancelled):
                worker.execute(ForwardRequest("cancelled", {"input_ids": torch.tensor([[1]])}), cancelled)

    def test_failed_cuda_rng_setup_still_attempts_device_drain(self):
        worker = ResidentWorker("small", SmallModel().eval())
        worker._device = torch.device("cuda:0")  # Exercise cleanup without initializing a GPU.
        with patch("torch.random.fork_rng", side_effect=RuntimeError("RNG setup failed")):
            with patch("torch.cuda.synchronize", side_effect=RuntimeError("drain failed")) as drain:
                with self.assertRaises(WorkerSynchronizationError):
                    worker.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}))
        drain.assert_called_once_with(torch.device("cuda:0"))
        self.assertTrue(worker.poisoned)

    def test_failed_cuda_drain_takes_precedence_over_interrupt(self):
        class InterruptModel(nn.Module):
            def forward(self, input_ids):
                raise KeyboardInterrupt()

        for drain_error in (RuntimeError("drain failed"), KeyboardInterrupt()):
            with self.subTest(drain_error=type(drain_error).__name__):
                with patch.object(worker_module, "_process_quarantined", threading.Event()):
                    worker = ResidentWorker("interrupt", InterruptModel().eval())
                    worker._device = torch.device("cuda:0")
                    with patch("torch.random.fork_rng", return_value=nullcontext()):
                        with patch("torch.cuda.default_generators", [Mock()]):
                            with patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor):
                                with patch("torch.cuda.synchronize", side_effect=drain_error):
                                    with self.assertRaises(WorkerSynchronizationError):
                                        worker.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}))
                    self.assertTrue(worker.poisoned)

    def test_failed_drain_takes_precedence_over_post_forward_state_check(self):
        worker = ResidentWorker("small", SmallModel().eval())
        with patch.object(worker, "_model_state", side_effect=[worker._baseline, RuntimeError("unreadable")]) as state:
            with patch.object(worker, "_perform", return_value=(None, (WorkerSynchronizationError, "drain failed"))):
                with self.assertRaisesRegex(WorkerSynchronizationError, "drain failed"):
                    worker.execute(ForwardRequest("r", {"input_ids": torch.tensor([[1]])}))
        self.assertEqual(state.call_count, 1)
        self.assertTrue(worker.poisoned)

    def test_snapshot_oom_blocks_other_queued_forwards_immediately(self):
        entered, release, failed_copy = threading.Event(), threading.Event(), threading.Event()
        seen = []

        class BlockingModel(SmallModel):
            def forward(self, input_ids, attention_mask=None):
                seen.append(int(input_ids[0, 0]))
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test forward timed out")
                return super().forward(input_ids, attention_mask)

        worker = ResidentWorker("serial", BlockingModel().eval())
        first_request = ForwardRequest("first", {"input_ids": torch.tensor([[1]])})
        bad_tensor = torch.tensor([[2]])
        bad_request = ForwardRequest("bad-copy", {"input_ids": bad_tensor})
        third_request = ForwardRequest("queued", {"input_ids": torch.tensor([[3]])})
        failures = []
        original_clone = torch.Tensor.clone

        def clone_with_failure(tensor, *args, **kwargs):
            if tensor.data_ptr() == bad_tensor.data_ptr():
                failed_copy.set()
                raise torch.OutOfMemoryError("snapshot OOM")
            return original_clone(tensor, *args, **kwargs)

        def call(request, snapshot_only=False):
            try:
                if snapshot_only:
                    worker.snapshot_request(request)
                else:
                    worker.execute(request)
            except Exception as exc:
                failures.append(exc)

        first = threading.Thread(target=call, args=(first_request,))
        first.start()
        self.assertTrue(entered.wait(5))
        with patch.object(torch.Tensor, "clone", clone_with_failure):
            oom_thread = threading.Thread(target=call, args=(bad_request, True))
            oom_thread.start()
            self.assertTrue(failed_copy.wait(5))
            for _ in range(100):
                if worker.poisoned:
                    break
                threading.Event().wait(0.01)
            self.assertTrue(worker.poisoned)
            queued = threading.Thread(target=call, args=(third_request,))
            queued.start()
            release.set()
            for thread in (first, oom_thread, queued):
                thread.join(5)
                self.assertFalse(thread.is_alive())
        self.assertEqual(seen, [1])
        self.assertTrue(any(isinstance(exc, WorkerOutOfMemory) for exc in failures))

    def test_failed_drain_quarantines_other_workers_and_retains_model(self):
        model = SmallModel().eval()
        model_ref = weakref.ref(model)
        failed = ResidentWorker("cuda", model)
        other_model = SmallModel().eval()
        other_ref = weakref.ref(other_model)
        other = ResidentWorker("other", other_model)
        del model
        del other_model
        failed._device = torch.device("cuda:0")
        with patch("torch.random.fork_rng", return_value=nullcontext()):
            with patch("torch.cuda.default_generators", [Mock()]):
                with patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor):
                    with patch("torch.cuda.synchronize", side_effect=RuntimeError("drain failed")):
                        with self.assertRaises(WorkerSynchronizationError):
                            failed.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}))
        failed.close()
        del failed
        gc.collect()
        self.assertIsNotNone(model_ref())
        self.assertTrue(other.poisoned)
        with self.assertRaises(WorkerSynchronizationError):
            other.execute(ForwardRequest("later", {"input_ids": torch.tensor([[1]])}))
        other.close()
        del other
        gc.collect()
        self.assertIsNotNone(other_ref())

    def test_failed_drain_quarantines_even_if_diagnostics_or_retention_fail(self):
        class UndisplayableError(RuntimeError):
            def __str__(self):
                raise RuntimeError("formatting failed")

        class NoRetentionSpace:
            def append(self, item):
                raise MemoryError("no space to retain references")

        worker = ResidentWorker("cuda", SmallModel().eval())
        worker._device = torch.device("cuda:0")
        with patch.object(worker_module, "_quarantine_references", NoRetentionSpace()):
            with patch("torch.random.fork_rng", side_effect=RuntimeError("setup failed")):
                with patch("torch.cuda.synchronize", side_effect=UndisplayableError()):
                    with self.assertRaises(WorkerSynchronizationError):
                        worker.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}))
        self.assertTrue(worker.quarantined)

    def test_failed_drain_keeps_known_tensors_if_global_retention_cannot_grow(self):
        references = {}

        class NoRetentionSpace:
            def append(self, item):
                raise MemoryError('no space to retain references')

        class ObservedModel(nn.Module):
            def forward(self, input_ids):
                logits = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
                references['input'] = weakref.ref(input_ids)
                references['logits'] = weakref.ref(logits)
                return {'logits': logits}

        model = ObservedModel().eval()
        model_ref = weakref.ref(model)
        worker = ResidentWorker('cuda', model)
        del model
        worker._device = torch.device('cuda:0')

        def fake_to(tensor, *args, **kwargs):
            if kwargs.get('copy'):
                copied = tensor.clone()
                references['offload'] = weakref.ref(copied)
                return copied
            return tensor

        other = ResidentWorker('other', SmallModel().eval())
        with patch.object(worker_module, '_quarantine_references', NoRetentionSpace()):
            with patch.object(worker_module._process_quarantined, 'set', side_effect=MemoryError('event notification')):
                with patch('torch.random.fork_rng', return_value=nullcontext()):
                    with patch('torch.cuda.default_generators', [Mock()]):
                        with patch.object(torch.Tensor, 'to', fake_to):
                            with patch('torch.cuda.synchronize', side_effect=RuntimeError('drain failed')):
                                with self.assertRaises(WorkerSynchronizationError):
                                    worker.execute(ForwardRequest('fail', {'input_ids': torch.tensor([[1]])}))
            worker.close()
            gc.collect()
            self.assertTrue(worker.quarantined)
            self.assertTrue(other.quarantined)
            with self.assertRaises(WorkerSynchronizationError):
                other.execute(ForwardRequest('later', {'input_ids': torch.tensor([[2]])}))
            self.assertIsNotNone(model_ref())
            self.assertIsNotNone(references['input']())
            self.assertIsNotNone(references['logits']())
            self.assertIsNotNone(references['offload']())

    def test_failed_drain_recovers_output_interrupted_before_slot_assignment(self):
        references = {}

        class NoRetentionSpace:
            def append(self, item):
                raise MemoryError('no space to retain references')

        class ObservedModel(nn.Module):
            def forward(self, input_ids):
                logits = torch.stack((input_ids.float(), input_ids.float() + 1), dim=-1)
                references['logits'] = weakref.ref(logits)
                return {'logits': logits}

        worker = ResidentWorker('cuda', ObservedModel().eval())
        worker._device = torch.device('cuda:0')
        source, first = inspect.getsourcelines(ResidentWorker._perform_with_custody)
        target = first + max(i for i, line in enumerate(source)
                             if line.strip() == 'self._live_buffers[2] = output')
        fired = []

        def trace(frame, event, arg):
            if (frame.f_code is ResidentWorker._perform_with_custody.__code__
                    and event == 'line' and frame.f_lineno == target):
                sys.settrace(None)
                fired.append(True)
                raise RuntimeError('output-slot assignment interrupted')
            return trace

        with patch.object(worker_module, '_quarantine_references', NoRetentionSpace()):
            with patch('torch.random.fork_rng', return_value=nullcontext()):
                with patch('torch.cuda.default_generators', [Mock()]):
                    with patch.object(torch.Tensor, 'to', lambda tensor, *args, **kwargs: tensor):
                        with patch('torch.cuda.synchronize', side_effect=RuntimeError('drain failed')):
                            try:
                                sys.settrace(trace)
                                with self.assertRaises(WorkerSynchronizationError):
                                    worker.execute(ForwardRequest('fail', {'input_ids': torch.tensor([[1]])}))
                            finally:
                                sys.settrace(None)
            worker.close()
            gc.collect()
            self.assertEqual(fired, [True])
            self.assertTrue(worker.quarantined)
            self.assertIsNotNone(references['logits']())

    def test_successful_forward_clears_fixed_custody_slots(self):
        worker = ResidentWorker('cpu', SmallModel().eval())
        worker.execute(ForwardRequest('r', {'input_ids': torch.tensor([[1]])}))
        self.assertEqual(worker._live_buffers, [None] * 5)

    def test_worker_already_waiting_for_execution_lock_observes_quarantine(self):
        entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
        seen = []

        class ObservedModel(SmallModel):
            def forward(self, input_ids, attention_mask=None):
                seen.append(int(input_ids[0, 0]))
                return super().forward(input_ids, attention_mask)

        failed = ResidentWorker("cuda", SmallModel().eval())
        other = ResidentWorker("other", ObservedModel().eval())
        failed._device = torch.device("cuda:0")
        original_snapshot = other.snapshot_request
        failures = []

        def snapshot(request):
            result = original_snapshot(request)
            submitted.set()
            return result

        def drain(device):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test drain timed out")
            raise RuntimeError("drain failed")

        def call(worker, request):
            try:
                worker.execute(request)
            except Exception as exc:
                failures.append(exc)

        with patch("torch.random.fork_rng", return_value=nullcontext()):
            with patch("torch.cuda.default_generators", [Mock()]):
                with patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor):
                    with patch("torch.cuda.synchronize", side_effect=drain):
                        with patch.object(other, "snapshot_request", side_effect=snapshot):
                            first = threading.Thread(target=call, args=(failed, ForwardRequest("one", {"input_ids": torch.tensor([[1]])})))
                            second = threading.Thread(target=call, args=(other, ForwardRequest("two", {"input_ids": torch.tensor([[2]])})))
                            first.start()
                            self.assertTrue(entered.wait(5))
                            second.start()
                            self.assertTrue(submitted.wait(5))
                            release.set()
                            first.join(5)
                            second.join(5)
                            self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(seen, [])
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(isinstance(exc, WorkerSynchronizationError) for exc in failures))

    def test_invalid_request_exceptions_do_not_retain_input_tensors(self):
        for boundary in ("validate_request", "snapshot_request", "execute", "invalid_cancel"):
            with self.subTest(boundary=boundary):
                worker = ResidentWorker("small", SmallModel().eval())
                tensor = torch.tensor([1.0])
                tensor_ref = weakref.ref(tensor)
                request = ForwardRequest("invalid", {"input_ids": tensor})
                caught = None
                try:
                    if boundary == "invalid_cancel":
                        worker.execute(request, object())
                    else:
                        getattr(worker, boundary)(request)
                except InvalidRequest as exc:
                    caught = exc
                self.assertIsNotNone(caught)
                self.assertIsNone(caught.__context__)
                request = None
                tensor = None
                gc.collect()
                self.assertIsNone(tensor_ref())

    def test_autocast_is_rejected_in_the_execution_thread(self):
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(1, 2)

            def forward(self, input_ids):
                return {"logits": self.projection(input_ids.float().unsqueeze(-1))}

        worker = ResidentWorker("linear", LinearModel().eval())
        request = ForwardRequest("r", {"input_ids": torch.tensor([[1]])})
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(WorkerExecutionError, "autocast"):
                worker.execute(request)
        self.assertFalse(worker.poisoned)
        self.assertEqual(worker.execute(request).logits.dtype, torch.float32)

    def test_enabled_or_changed_model_cache_is_rejected(self):
        for setting in (True, None):
            enabled = SmallModel().eval()
            enabled.config = SimpleNamespace(use_cache=setting)
            with self.subTest(setting=setting), self.assertRaisesRegex(ValueError, "cache"):
                ResidentWorker("enabled", enabled)
        model = SmallModel().eval()
        model.config = SimpleNamespace(use_cache=False)
        worker = ResidentWorker("changing", model)
        model.config.use_cache = True
        with self.assertRaises(WorkerUnavailable):
            worker.execute(ForwardRequest("r", {"input_ids": torch.tensor([[1]])}))
        self.assertTrue(worker.poisoned)

    def test_returned_cache_fails_and_poisons_even_when_config_is_disabled(self):
        class CacheModel(SmallModel):
            def forward(self, input_ids, attention_mask=None):
                result = super().forward(input_ids, attention_mask)
                result["past_key_values"] = (torch.ones(1),)
                return result

        model = CacheModel().eval()
        model.config = SimpleNamespace(use_cache=False)
        worker = ResidentWorker("cached", model)
        with self.assertRaises(WorkerExecutionError):
            worker.execute(ForwardRequest("r", {"input_ids": torch.tensor([[1]])}))
        self.assertTrue(worker.poisoned)

    def test_interrupt_during_state_validation_poison_worker(self):
        for failure_point in ("before", "after"):
            with self.subTest(failure_point=failure_point):
                worker = ResidentWorker("small", SmallModel().eval())
                outcomes = [KeyboardInterrupt("state interrupted")]
                if failure_point == "after":
                    outcomes.insert(0, worker._baseline)
                with patch.object(worker, "_model_state", side_effect=outcomes):
                    with self.assertRaises(KeyboardInterrupt):
                        worker.execute(ForwardRequest("r", {"input_ids": torch.tensor([[1]])}))
                self.assertTrue(worker.poisoned)

    def test_binding_properties_are_read_only(self):
        worker = ResidentWorker("small", SmallModel().eval())
        for name in ("model_id", "device", "max_batch_size", "max_sequence_length",
                     "max_input_bytes", "max_output_bytes"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(worker, name, 999)

    def test_snapshot_oom_with_unprintable_error_still_drops_model(self):
        class UnprintableOom(MemoryError):
            def __str__(self):
                raise RuntimeError("cannot format OOM")

        model = SmallModel().eval()
        model_ref = weakref.ref(model)
        worker = ResidentWorker("small", model)
        del model
        with patch.object(torch.Tensor, "clone", side_effect=UnprintableOom()):
            with self.assertRaises(WorkerOutOfMemory):
                worker.snapshot_request(ForwardRequest("r", {"input_ids": torch.tensor([[1]])}))
        gc.collect()
        self.assertTrue(worker.poisoned)
        self.assertIsNone(model_ref())

    def test_interrupted_snapshot_oom_cleanup_does_not_retain_inputs(self):
        def fail_clone(tensor, *args, **kwargs):
            raise MemoryError("injected")

        for boundary in ("snapshot_request", "execute"):
            with self.subTest(boundary=boundary):
                worker = ResidentWorker("small", SmallModel().eval())
                tensor = torch.tensor([[1]])
                tensor_ref = weakref.ref(tensor)
                request = ForwardRequest("r", {"input_ids": tensor})
                caught = None
                with patch.object(torch.Tensor, "clone", fail_clone):
                    with patch.object(worker, "_poison", side_effect=KeyboardInterrupt("cleanup")):
                        try:
                            getattr(worker, boundary)(request)
                        except KeyboardInterrupt as exc:
                            caught = exc
                self.assertIsNotNone(caught)
                self.assertTrue(worker.poisoned)
                tensor = None
                request = None
                gc.collect()
                self.assertIsNone(tensor_ref())

    def test_forward_oom_with_unprintable_error_keeps_oom_classification(self):
        class UnprintableOom(MemoryError):
            def __str__(self):
                raise RuntimeError("cannot format OOM")

        class FailingModel(nn.Module):
            def forward(self, input_ids):
                raise UnprintableOom()

        worker = ResidentWorker("failing", FailingModel().eval())
        with self.assertRaises(WorkerOutOfMemory):
            worker.execute(ForwardRequest("r", {"input_ids": torch.tensor([[1]])}))
        self.assertTrue(worker.poisoned)

    def test_shared_registered_state_helper_matches_worker_baseline(self):
        from looking_glass.resident_worker import registered_model_state

        model = SmallModel().eval()
        worker = ResidentWorker("small", model)
        self.assertEqual(registered_model_state(model, worker.device), worker._baseline)


if __name__ == "__main__":
    unittest.main()
