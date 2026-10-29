"""CPU checks for ownership across resident tracing and interventions."""
import importlib.util
import threading
import unittest
from unittest.mock import patch

import torch

if importlib.util.find_spec("transformers") is not None:
    from transformers import GPT2Config, GPT2LMHeadModel
else:
    GPT2Config = GPT2LMHeadModel = None


@unittest.skipUnless(GPT2Config is not None, "requires transformers")
class TraceInterventionOwnershipTests(unittest.TestCase):
    def setUp(self):
        from looking_glass.capture_dispatcher import CaptureRequest, SampleSelection
        from looking_glass.resident_worker import ForwardRequest
        from looking_glass.traced_worker import TraceSite, TracedGPT2Worker

        config = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16, n_embd=8,
                            n_layer=1, n_head=2, use_cache=False,
                            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
        with torch.random.fork_rng():
            torch.manual_seed(2026)
            self.model = GPT2LMHeadModel(config).eval()
        self.inputs = torch.tensor([[1, 3, 5]])
        self.request = ForwardRequest("batch", {"input_ids": self.inputs}, seed=7)
        self.captures = (CaptureRequest(
            "run", "job", "request", (SampleSelection(0, "sample", (2, 0)),),
            0, None, 8, 4096),)
        self.site = TraceSite(0, "mlp")
        self.worker = TracedGPT2Worker("tiny-gpt2", self.model, device="cpu",
                                       max_batch_size=1, max_sequence_length=3)
        self.module_path = "transformer.h.0.mlp"

    def traced(self):
        return self.worker.execute_traced(
            self.request, captures=self.captures, sites=(self.site,),
            sequence_starts={"run": 0}, max_events=8, max_trace_bytes=4096)

    def session(self, kind):
        from looking_glass.interventions import ActivationCapture, ActivationIntervention
        from looking_glass.intervention_plan import ActivationPlan

        coordinates = {"batch_index": 0, "token_index": 1}
        if kind == "capture":
            return ActivationCapture(self.model, self.module_path, **coordinates)
        if kind == "intervention":
            return ActivationIntervention(self.model, self.module_path,
                                          operation="zero", **coordinates)
        return ActivationPlan(self.model, [
            {"name": "zero", "operation": "zero", "module_path": self.module_path,
             **coordinates}])

    def assert_no_hooks(self):
        self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                            for module in self.model.modules()))

    def test_intervention_owner_blocks_worker_then_leaves_model_reusable(self):
        for kind in ("capture", "intervention", "plan"):
            with self.subTest(kind=kind):
                self.setUp()
                baseline = self.worker.execute(self.request).logits
                with self.session(kind) as session:
                    with self.assertRaisesRegex(RuntimeError, "already leased"):
                        self.traced()
                    with torch.inference_mode():
                        output = self.model(self.inputs).logits
                self.assertFalse(self.worker.poisoned)
                self.assert_no_hooks()
                if kind == "capture":
                    self.assertEqual(tuple(session.value.shape), (8,))
                    self.assertTrue(torch.equal(output, baseline))
                else:
                    self.assertFalse(torch.equal(output, baseline))
                self.assertTrue(torch.equal(self.worker.execute(self.request).logits, baseline))
                self.assertEqual(len(self.traced().events), 1)
                self.assert_no_hooks()

    def test_active_worker_blocks_interventions_then_all_can_reacquire(self):
        from looking_glass.capture_dispatcher import CaptureDispatcher
        from looking_glass.traced_worker import TracedForwardResult

        baseline = self.worker.execute(self.request).logits
        entered, release, results = threading.Event(), threading.Event(), []
        record = CaptureDispatcher.record

        def hold(dispatcher, *args, **kwargs):
            record(dispatcher, *args, **kwargs)
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test hook timed out")

        def call():
            try:
                results.append(self.traced())
            except BaseException as exc:
                results.append(exc)

        with patch.object(CaptureDispatcher, "record", hold):
            thread = threading.Thread(target=call, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(15))
                for kind in ("capture", "intervention", "plan"):
                    with self.subTest(kind=kind):
                        with self.assertRaisesRegex(RuntimeError, "already leased"):
                            with self.session(kind):
                                pass
            finally:
                release.set()
                thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], TracedForwardResult)
        self.assertTrue(torch.equal(results[0].forward_result.logits, baseline))
        self.assertEqual(len(results[0].events), 1)
        self.assertEqual(results[0].events[0].request_id, "request")
        self.assert_no_hooks()

        # Each API can run only after the worker has released its hooks.
        for kind in ("capture", "intervention", "plan"):
            with self.subTest(reacquire=kind):
                with self.session(kind) as session:
                    with torch.inference_mode():
                        output = self.model(self.inputs).logits
                if kind == "capture":
                    self.assertEqual(tuple(session.value.shape), (8,))
                    self.assertTrue(torch.equal(output, baseline))
                else:
                    self.assertFalse(torch.equal(output, baseline))
                self.assert_no_hooks()
                self.assertTrue(torch.equal(self.worker.execute(self.request).logits, baseline))


if __name__ == "__main__":
    unittest.main()
