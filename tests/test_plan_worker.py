"""CPU-only case-worker tests with artificial weights and inputs."""

import threading
import unittest
import gc
import inspect
import importlib.util
import sys
import weakref

import torch
from torch import nn

from looking_glass.model_ownership import ModelLease
from looking_glass.plan_worker import (CPUPlanWorker, PlanCaseCancelled, PlanControlFailed,
                                       PlanTensorRef, PlanWorkerBusy, PlanWorkerError,
                                       PlanWorkerOutOfMemory)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.tap = nn.Identity()
        self.output = nn.Linear(4, 7, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(44).reshape(11, 4) / 13)
            self.output.weight.copy_(torch.arange(28).reshape(7, 4) / 17)
        self.eval()

    def forward(self, input_ids, attention_mask=None):
        return {"logits": self.output(self.tap(self.embedding(input_ids)))}


def selected(name, operation, **extra):
    return {"name": name, "operation": operation, "module_path": "tap",
            "invocation": 0, "batch_index": 0, "token_index": 2, **extra}


class PlanWorkerTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.worker = CPUPlanWorker(self.model)
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

    def test_multi_forward_case_holds_model_and_checks_controls(self):
        case = self.worker.case("synthetic-development")
        with case:
            clean = case.run(self.inputs, [selected("clean", "capture")], seed=19)
            with self.assertRaisesRegex(PlanWorkerError, "not published"):
                case.export(clean.logits)
            with self.assertRaisesRegex(RuntimeError, "already leased"):
                ModelLease(self.model).acquire()
            delta = torch.tensor([1., 2., -1., .5])
            patched = case.run(self.inputs, [
                selected("before", "capture"),
                selected("inject", "add", value=delta),
                selected("after", "capture"),
                selected("repair", "patch", value_ref=clean.captures["clean"]),
                selected("repaired", "capture"),
            ], seed=19)
            case.require_added(patched.captures["before"], patched.sources["inject"],
                               patched.captures["after"])
            case.require_equal(clean.captures["clean"], patched.captures["repaired"])
            case.require_equal(clean.logits, patched.logits)
            meta = case.private_metadata(patched)
            self.assertEqual([e["name"] for e in meta["steps"]],
                             ["before", "inject", "after", "repair", "repaired"])
            self.assertEqual(clean.input_sha256, patched.input_sha256)
        clean_export = case.export(clean.logits)
        patched_export = case.export(patched.logits)
        self.assertTrue(torch.equal(clean_export.tensor, patched_export.tensor))
        self.assertEqual(self.worker.usage().ready_items, 7)
        capture_export = case.export(clean.captures["clean"])
        self.assertFalse(capture_export.tensor.requires_grad)
        case.discard()
        self.assertEqual(self.worker.usage().claimed_bytes, 0)
        self.assertEqual(self.worker.export_usage()["items"], 3)
        clean_export.close()
        patched_export.close()
        capture_export.close()
        self.assertEqual(self.worker.export_usage()["bytes"], 0)
        ModelLease(self.model).acquire().release()
        self.worker.close()

    def test_same_plan_self_patch_and_private_bits(self):
        case = self.worker.case("self-patch")
        with case:
            run = case.run(self.inputs, [selected("before", "capture"),
                selected("self", "patch", source_capture="before"),
                selected("after", "capture")], seed=0)
            case.require_equal(run.captures["before"], run.captures["after"])
        case.discard()

    def test_failed_control_prevents_publication_and_keeps_private_evidence(self):
        case = self.worker.case("failed-control")
        with self.assertRaises(PlanControlFailed):
            with case:
                run = case.run(self.inputs, [selected("before", "capture"),
                    selected("add", "add", value=torch.ones(4)),
                    selected("after", "capture")], seed=0)
                case.require_equal(run.captures["before"], run.captures["after"])
        self.assertFalse(self.worker.poisoned)
        self.assertEqual(case.private_failure()["completed_runs"], 1)
        after = case.export_private_failure(run.captures["after"])
        before = case.export_private_failure(run.captures["before"])
        self.assertTrue(torch.equal(after.tensor, before.tensor + 1))
        after.close()
        before.close()
        with self.assertRaisesRegex(PlanWorkerError, "not published"):
            case.export(run.logits)
        case.discard()

    def test_caught_control_error_still_invalidates_case(self):
        case = self.worker.case("caught-control")
        with self.assertRaisesRegex(PlanWorkerError, "partial results"):
            with case:
                run = case.run(self.inputs, [selected("before", "capture"),
                    selected("add", "add", value=torch.ones(4)),
                    selected("after", "capture")], seed=0)
                with self.assertRaises(PlanControlFailed):
                    case.require_equal(run.captures["before"], run.captures["after"])
        case.discard()

    def test_input_source_limits_and_same_case_reference(self):
        case = self.worker.case("invalid-ref")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            with case:
                case.run(self.inputs, [selected("patch", "patch", value_ref=PlanTensorRef())], seed=0)
        self.assertFalse(self.worker.poisoned)
        case.discard()
        case2 = self.worker.case("invalid-input")
        with self.assertRaisesRegex(ValueError, "rank-two"):
            with case2:
                case2.run({"input_ids": torch.ones(1, 3)}, [selected("x", "capture")], seed=0)
        case2.discard()

    def test_prior_capture_requires_identical_input_snapshot(self):
        case = self.worker.case("different-input")
        with self.assertRaisesRegex(ValueError, "different inputs"):
            with case:
                first = case.run(self.inputs, [selected("clean", "capture")], seed=0)
                other = {"input_ids": torch.tensor([[1, 2, 4]])}
                case.run(other, [selected("patch", "patch", value_ref=first.captures["clean"])], seed=0)
        self.assertFalse(self.worker.poisoned)
        self.assertEqual(case.private_failure()["completed_runs"], 1)
        case.discard()

    def test_retained_case_and_tensor_caps_are_aggregate(self):
        worker = CPUPlanWorker(self.model, max_stored_cases=1)
        first = worker.case("one")
        with first:
            first.run(self.inputs, [selected("capture", "capture")], seed=0)
        with self.assertRaises(PlanWorkerBusy):
            worker.case("two").__enter__()
        self.assertGreater(worker.usage().claimed_bytes, 0)
        first.discard()
        second = worker.case("two")
        with second:
            second.run(self.inputs, [selected("capture", "capture")], seed=0)
        second.discard()
        self.assertEqual(worker.usage().claimed_items, 0)
        worker.close()

    def test_two_case_owners_cannot_overlap(self):
        first = self.worker.case("one")
        with first:
            with self.assertRaises(PlanWorkerBusy):
                self.worker.case("two").__enter__()
            first.run(self.inputs, [selected("capture", "capture")], seed=0)
        first.discard()

    def test_reentering_case_cannot_release_the_active_owner(self):
        case = self.worker.case("reentry")
        errors = []
        with case:
            with self.assertRaisesRegex(PlanWorkerError, "single-use"):
                case.__enter__()

            def foreign_entry():
                try:
                    case.__enter__()
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=foreign_entry)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIn("single-use", str(errors[0]))
            with self.assertRaisesRegex(RuntimeError, "already leased"):
                ModelLease(self.model).acquire()
            case.run(self.inputs, [selected("capture", "capture")], seed=0)
        case.discard()

    def test_reentry_during_forward_does_not_release_parent_or_child(self):
        shared = {}

        class Reentering(ToyModel):
            def forward(self, input_ids, attention_mask=None):
                try:
                    shared["case"].__enter__()
                except PlanWorkerError:
                    pass
                else:
                    raise AssertionError("case reentry must fail")
                try:
                    ModelLease(self).acquire()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("competing model lease must fail")
                return super().forward(input_ids, attention_mask)

        model = Reentering()
        worker = CPUPlanWorker(model)
        case = worker.case("reenter-model")
        shared["case"] = case
        with case:
            case.run(self.inputs, [selected("capture", "capture")], seed=0)
        case.discard()

    def test_rejects_ambient_autocast_at_entry_and_before_each_forward(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(PlanWorkerError, "ambient autocast"):
                self.worker.case("entry").__enter__()
        case = self.worker.case("run")
        with self.assertRaisesRegex(PlanWorkerError, "ambient autocast"):
            with case:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    case.run(self.inputs, [selected("capture", "capture")], seed=0)
        self.assertFalse(self.worker.poisoned)
        case.discard()

    def test_duplicate_case_id_is_rejected_until_prior_case_is_discarded(self):
        case = self.worker.case("same-id")
        with case:
            case.run(self.inputs, [selected("capture", "capture")], seed=0)
        with self.assertRaises(PlanWorkerBusy):
            self.worker.case("same-id").__enter__()
        case.discard()
        replacement = self.worker.case("same-id")
        with replacement:
            replacement.run(self.inputs, [selected("capture", "capture")], seed=0)
        replacement.discard()

    def test_cooperative_cancellation_before_forward_preserves_model(self):
        event = threading.Event()
        event.set()
        case = self.worker.case("cancelled")
        with self.assertRaises(PlanCaseCancelled):
            with case:
                case.run(self.inputs, [selected("capture", "capture")], seed=0, cancel_event=event)
        self.assertFalse(self.worker.poisoned)
        self.assertEqual(case.private_failure()["completed_runs"], 0)
        case.discard()
        ModelLease(self.model).acquire().release()

    def test_cancellation_during_forward_is_seen_only_after_return(self):
        event = threading.Event()

        class Cancelling(ToyModel):
            def forward(self, input_ids, attention_mask=None):
                output = super().forward(input_ids, attention_mask)
                event.set()
                return output

        model = Cancelling()
        worker = CPUPlanWorker(model)
        case = worker.case("cancel-after")
        with self.assertRaises(PlanCaseCancelled):
            with case:
                case.run(self.inputs, [selected("capture", "capture")], seed=0, cancel_event=event)
        self.assertFalse(worker.poisoned)
        failure = case.private_failure()
        self.assertEqual(failure["reason"], "cancelled after forward")
        self.assertEqual(failure["attempt"]["plan_metadata"]["steps"][0]["name"], "capture")
        self.assertEqual(len(model.tap._forward_hooks), 0)
        case.discard()
        ModelLease(model).acquire().release()

    def test_store_limit_after_forward_preserves_prior_private_evidence(self):
        # One four-float capture plus seven float32 logits uses 44 bytes.
        worker = CPUPlanWorker(self.model, max_stored_bytes=60)
        case = worker.case("bounded")
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            with case:
                first = case.run(self.inputs, [selected("a", "capture")], seed=0)
                case.run(self.inputs, [selected("b", "capture")], seed=0)
        self.assertFalse(worker.poisoned)
        failure = case.private_failure()
        self.assertEqual(failure["completed_runs"], 1)
        self.assertEqual(failure["attempt"]["steps"][0]["name"], "b")
        export = case.export_private_failure(first.logits)
        self.assertTrue(torch.isfinite(export.tensor).all())
        export.close()
        self.assertLessEqual(worker.usage().claimed_bytes, 60)
        case.discard()
        ModelLease(self.model).acquire().release()

    def test_source_snapshot_is_independent_of_caller_mutation(self):
        case = self.worker.case("snapshot")
        source = torch.tensor([1., 2., 3., 4.])
        with case:
            run = case.run(self.inputs, [selected("before", "capture"),
                selected("add", "add", value=source), selected("after", "capture")], seed=0)
            source.zero_()
            case.require_added(run.captures["before"], run.sources["add"], run.captures["after"])
        export = case.export(run.sources["add"])
        self.assertTrue(torch.equal(export.tensor, torch.tensor([1., 2., 3., 4.])))
        export.close()
        case.discard()

    def test_model_exception_poisons_and_keeps_the_reservation(self):
        class Failing(ToyModel):
            def forward(self, input_ids, attention_mask=None):
                self.tap(self.embedding(input_ids))
                raise MemoryError("simulated failure")

        model = Failing()
        worker = CPUPlanWorker(model)
        case = worker.case("failing")
        with self.assertRaises(PlanWorkerOutOfMemory):
            with case:
                case.run(self.inputs, [selected("capture", "capture")], seed=0)
        self.assertTrue(worker.poisoned)
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model).acquire()
        with self.assertRaisesRegex(PlanWorkerError, "quarantined"):
            case.discard()
        with self.assertRaisesRegex(PlanWorkerError, "unavailable"):
            worker.case("next").__enter__()

    def test_model_raised_cancellation_is_not_a_boundary_cancellation(self):
        class CancelFromModel(ToyModel):
            def __init__(self):
                super().__init__()
                self.fail = False

            def forward(self, input_ids, attention_mask=None):
                if self.fail:
                    raise PlanCaseCancelled("model interruption")
                return super().forward(input_ids, attention_mask)

        model = CancelFromModel()
        worker = CPUPlanWorker(model)
        case = worker.case("model-cancellation")
        with self.assertRaisesRegex(PlanWorkerError, "partial results"):
            with case:
                first = case.run(self.inputs, [selected("capture", "capture")], seed=0)
                model.fail = True
                with self.assertRaises(PlanCaseCancelled):
                    case.run(self.inputs, [selected("capture", "capture")], seed=0)
        self.assertTrue(worker.poisoned)
        self.assertEqual(case.private_failure()["completed_runs"], 1)
        with self.assertRaisesRegex(PlanWorkerError, "not published"):
            case.export(first.logits)
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model).acquire()

    def test_torch_out_of_memory_has_specific_sanitized_classification(self):
        class Failing(ToyModel):
            def forward(self, input_ids, attention_mask=None):
                raise torch.OutOfMemoryError("synthetic CPU fault")

        model = Failing()
        worker = CPUPlanWorker(model)
        case = worker.case("torch-oom")
        with self.assertRaises(PlanWorkerOutOfMemory):
            with case:
                case.run(self.inputs, [selected("capture", "capture")], seed=0)
        self.assertTrue(worker.poisoned)
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model).acquire()

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers is optional")
    def test_random_gpt2_same_site_repair_matches_direct_clean_logits(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2701)
            config = GPT2Config(vocab_size=23, n_positions=16, n_layer=2,
                                n_head=2, n_embd=8, use_cache=False)
            config._attn_implementation = "eager"
            model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[2, 5, 8, 4]])
        with torch.inference_mode():
            direct = model(input_ids=ids).logits[0, -1].clone()
        worker = CPUPlanWorker(model)
        case = worker.case("random-gpt2-software-control")
        site = {"module_path": "transformer.h.0.mlp", "batch_index": 0, "token_index": 3}
        with case:
            clean = case.run({"input_ids": ids}, [
                {"name": "clean", "operation": "capture", **site}], seed=17)
            changed = case.run({"input_ids": ids}, [
                {"name": "before", "operation": "capture", **site},
                {"name": "inject", "operation": "add", "value": torch.ones(8), **site},
                {"name": "after", "operation": "capture", **site},
                {"name": "repair", "operation": "patch", "value_ref": clean.captures["clean"], **site},
                {"name": "repaired", "operation": "capture", **site}], seed=17)
            case.require_added(changed.captures["before"], changed.sources["inject"],
                               changed.captures["after"])
            case.require_equal(clean.captures["clean"], changed.captures["repaired"])
            case.require_equal(clean.logits, changed.logits)
        export = case.export(changed.logits)
        self.assertTrue(torch.equal(export.tensor.view(torch.int32), direct.view(torch.int32)))
        export.close()
        case.discard()
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks
                             for module in model.modules()))
        worker.close()

    def test_returned_model_cache_quarantines_even_after_forward_returns(self):
        class Caching(ToyModel):
            def forward(self, input_ids, attention_mask=None):
                result = super().forward(input_ids, attention_mask)
                result["past_key_values"] = (torch.ones(1),)
                return result

        model = Caching()
        worker = CPUPlanWorker(model)
        case = worker.case("cache")
        with self.assertRaisesRegex(PlanWorkerError, "cache"):
            with case:
                case.run(self.inputs, [selected("capture", "capture")], seed=0)
        self.assertTrue(worker.poisoned)
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(model).acquire()

    def test_retained_clean_failure_traceback_does_not_hold_forward_logits(self):
        references = []

        class Recording(ToyModel):
            def forward(self, input_ids, attention_mask=None):
                output = super().forward(input_ids, attention_mask)
                references.append(weakref.ref(output["logits"]))
                return output

        worker = CPUPlanWorker(Recording(), max_stored_bytes=60)
        case = worker.case("retained-error")
        caught = None
        try:
            with case:
                case.run(self.inputs, [selected("a", "capture")], seed=0)
                case.run(self.inputs, [selected("b", "capture")], seed=0)
        except RuntimeError as error:
            caught = error
        self.assertIsNotNone(caught)
        case.discard()
        gc.collect()
        self.assertEqual(worker.usage().claimed_bytes, 0)
        self.assertIsNone(references[-1]())
        self.assertIsNotNone(caught.__traceback__)

    def test_interrupted_export_remains_charged_and_recoverable(self):
        case = self.worker.case("export-interrupt")
        with case:
            run = case.run(self.inputs, [selected("capture", "capture")], seed=0)
        caught = None

        def tracer(frame, event, arg):
            if frame.f_code.co_name == "export" and event == "return":
                raise KeyboardInterrupt()
            return tracer

        try:
            sys.settrace(tracer)
            case.export(run.captures["capture"])
        except KeyboardInterrupt as error:
            caught = error
        finally:
            sys.settrace(None)
        self.assertIsNotNone(caught)
        case.discard()
        self.assertEqual(self.worker.usage().claimed_bytes, 0)
        self.assertEqual(self.worker.export_usage()["bytes"], 16)
        exports = self.worker.active_exports()
        self.assertEqual(len(exports), 1)
        caught = None
        exports[0].close()
        self.assertEqual(self.worker.export_usage()["bytes"], 0)

    def test_interrupted_exit_transition_quarantines_instead_of_releasing(self):
        case = self.worker.case("exit-interrupt")
        case.__enter__()
        case.run(self.inputs, [selected("capture", "capture")], seed=0)
        lines, first_line = inspect.getsourcelines(type(case).__exit__)
        line = first_line + next(i for i, text in enumerate(lines) if "if not worker._poisoned:" in text)

        def tracer(frame, event, arg):
            if frame.f_code is type(case).__exit__.__code__ and event == "line" and frame.f_lineno == line:
                raise KeyboardInterrupt()
            return tracer

        try:
            sys.settrace(tracer)
            with self.assertRaises(KeyboardInterrupt):
                case.__exit__(None, None, None)
        finally:
            sys.settrace(None)
        self.assertTrue(self.worker.poisoned)
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(self.model).acquire()
        with self.assertRaisesRegex(PlanWorkerError, "quarantined"):
            case.discard()

    def test_interrupted_exit_finalization_requires_owner_quarantine(self):
        for needle in ("with self._lock:", "worker._active = None",
                       "self._state = \"quarantined\"", "self._closing_token = None"):
            with self.subTest(boundary=needle):
                model = ToyModel()
                worker = CPUPlanWorker(model)
                case = worker.case("exit-finalization")
                case.__enter__()
                run = case.run(self.inputs, [selected("capture", "capture")], seed=0)
                lines, first_line = inspect.getsourcelines(type(case).__exit__)
                start = next(i for i, text in enumerate(lines) if text.strip() == "finally:")
                line = first_line + next(i for i in range(start + 1, len(lines)) if needle in lines[i])

                def tracer(frame, event, arg):
                    if frame.f_code is type(case).__exit__.__code__ and event == "line" and frame.f_lineno == line:
                        raise KeyboardInterrupt()
                    return tracer

                try:
                    sys.settrace(tracer)
                    with self.assertRaises(KeyboardInterrupt):
                        case.__exit__(None, None, None)
                finally:
                    sys.settrace(None)
                self.assertEqual(case.status, "recovery_required")
                with self.assertRaisesRegex(PlanWorkerError, "not published"):
                    case.export(run.logits)
                with self.assertRaises(PlanWorkerError):
                    case.private_metadata(run)
                with self.assertRaises(PlanWorkerError):
                    case.discard()
                with self.assertRaises(PlanWorkerBusy):
                    worker.case("competitor").__enter__()
                case.quarantine_interrupted_exit()
                self.assertEqual(case.status, "quarantined")
                self.assertTrue(worker.poisoned)
                with self.assertRaisesRegex(PlanWorkerError, "unavailable"):
                    worker.case("later").__enter__()
                self.assertGreater(worker.usage().claimed_bytes, 0)

    def test_export_quota_is_separate_from_retained_store(self):
        worker = CPUPlanWorker(self.model, max_export_bytes=16)
        case = worker.case("export-cap")
        with case:
            run = case.run(self.inputs, [selected("capture", "capture")], seed=0)
        export = case.export(run.captures["capture"])
        with self.assertRaises(PlanWorkerBusy):
            case.export(run.logits)
        self.assertEqual(worker.export_usage()["bytes"], 16)
        self.assertGreater(worker.usage().claimed_bytes, 0)
        case.discard()
        export.close()
        worker.close()

    def test_export_copy_is_serialized_against_active_case_and_oom_quarantines(self):
        old = self.worker.case("prior")
        with old:
            prior = old.run(self.inputs, [selected("capture", "capture")], seed=0)
        entry, _ = old._entry(prior.captures["capture"])

        def fail_copy():
            raise MemoryError("simulated export OOM")

        entry.read = fail_copy
        case = self.worker.case("concurrent")
        errors = []
        with case:
            def try_export():
                try:
                    old.export(prior.captures["capture"])
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=try_export)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], PlanWorkerBusy)
            case.run(self.inputs, [selected("capture", "capture")], seed=0)
        case.discard()
        with self.assertRaises(PlanWorkerOutOfMemory):
            old.export(prior.captures["capture"])
        self.assertTrue(self.worker.poisoned)
        self.assertEqual(self.worker.export_usage()["items"], 1)
        self.assertEqual(self.worker.active_exports()[0].state, "quarantined")
        with self.assertRaisesRegex(PlanWorkerError, "unavailable"):
            self.worker.case("later").__enter__()

    def test_pending_export_blocks_admission_until_copy_completes(self):
        old = self.worker.case("prior")
        with old:
            prior = old.run(self.inputs, [selected("capture", "capture")], seed=0)
        entry, _ = old._entry(prior.captures["capture"])
        original = entry.read
        entered, resume = threading.Event(), threading.Event()
        exports, errors = [], []

        def gated_read():
            entered.set()
            if not resume.wait(5):
                raise RuntimeError("test gate timed out")
            return original()

        entry.read = gated_read

        def do_export():
            try:
                exports.append(old.export(prior.captures["capture"]))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=do_export)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaises(PlanWorkerBusy):
                self.worker.case("blocked").__enter__()
        finally:
            resume.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(exports), 1)
        later = self.worker.case("later")
        with later:
            later.run(self.inputs, [selected("capture", "capture")], seed=0)
        later.discard()
        old.discard()
        exports[0].close()

    def test_seeded_forward_restores_external_cpu_rng_state(self):
        class Stochastic(ToyModel):
            def forward(self, input_ids, attention_mask=None):
                output = super().forward(input_ids, attention_mask)
                output["logits"] = output["logits"] + torch.rand_like(output["logits"])
                return output

        model = Stochastic()
        worker = CPUPlanWorker(model)
        original_rng = torch.random.get_rng_state().clone()
        case = worker.case("rng-isolation")
        with case:
            first = case.run(self.inputs, [selected("capture", "capture")], seed=33)
            second = case.run(self.inputs, [selected("capture", "capture")], seed=33)
            case.require_equal(first.logits, second.logits)
        self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
        case.discard()

    def test_model_mutation_between_worker_construction_and_entry_quarantines(self):
        with torch.no_grad():
            self.model.embedding.weight.add_(1)
        case = self.worker.case("mutated")
        with self.assertRaisesRegex(PlanWorkerError, "state changed"):
            case.__enter__()
        self.assertTrue(self.worker.poisoned)
        with self.assertRaisesRegex(RuntimeError, "already leased"):
            ModelLease(self.model).acquire()
