import gc
from contextlib import nullcontext
import threading
import unittest
from unittest.mock import Mock, patch
import weakref

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
        worker.device = torch.device("cuda:0")  # Exercise cleanup without initializing a GPU.
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
                    worker.device = torch.device("cuda:0")
                    with patch("torch.random.fork_rng", return_value=nullcontext()):
                        with patch("torch.cuda.default_generators", [Mock()]):
                            with patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor):
                                with patch("torch.cuda.synchronize", side_effect=drain_error):
                                    with self.assertRaises(WorkerSynchronizationError):
                                        worker.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}))
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
        failed.device = torch.device("cuda:0")
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
        worker.device = torch.device("cuda:0")
        with patch.object(worker_module, "_quarantine_references", NoRetentionSpace()):
            with patch("torch.random.fork_rng", side_effect=RuntimeError("setup failed")):
                with patch("torch.cuda.synchronize", side_effect=UndisplayableError()):
                    with self.assertRaises(WorkerSynchronizationError):
                        worker.execute(ForwardRequest("fail", {"input_ids": torch.tensor([[1]])}))
        self.assertTrue(worker.quarantined)

    def test_worker_already_waiting_for_execution_lock_observes_quarantine(self):
        entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
        seen = []

        class ObservedModel(SmallModel):
            def forward(self, input_ids, attention_mask=None):
                seen.append(int(input_ids[0, 0]))
                return super().forward(input_ids, attention_mask)

        failed = ResidentWorker("cuda", SmallModel().eval())
        other = ResidentWorker("other", ObservedModel().eval())
        failed.device = torch.device("cuda:0")
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


if __name__ == "__main__":
    unittest.main()
