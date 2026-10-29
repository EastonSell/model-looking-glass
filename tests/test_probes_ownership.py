import unittest
import threading
import dis
import sys
from unittest.mock import patch

import torch
from torch import nn

from looking_glass.interventions import ActivationIntervention
from looking_glass.probes import TapPoint, TraceSession


class FailOnceHandle:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.failed = False

    def remove(self):
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected removal failure")
        self.wrapped.remove()


class FailOnceTap(TapPoint):
    def __init__(self, node_id):
        super().__init__(node_id)
        self.fail_next = True

    def register_forward_hook(self, *args, **kwargs):
        handle = super().register_forward_hook(*args, **kwargs)
        if self.fail_next:
            self.fail_next = False
            return FailOnceHandle(handle)
        return handle


class SkipInterruptedFinish(TraceSession):
    """Model a PyTorch version that omits an interrupted completion callback."""
    def _finish_forward(self, module, inputs, output):
        if output is None:
            return None
        return super()._finish_forward(module, inputs, output)


class TraceOwnershipTests(unittest.TestCase):
    def test_intervention_inside_legacy_trace_is_rejected_before_hooking(self):
        tap = TapPoint("tap")
        model = nn.Sequential(tap)
        source = torch.tensor([1.0])
        with TraceSession(model) as trace:
            with self.assertRaises(RuntimeError):
                with ActivationIntervention(model, "0", operation="zero"):
                    model(source)
            self.assertTrue(torch.equal(model(source), source))
        self.assertEqual(len(trace.events), 1)
        self.assertEqual(len(tap._forward_hooks), 0)

    def test_legacy_trace_inside_intervention_is_rejected_before_hooking(self):
        tap = TapPoint("tap")
        model = nn.Sequential(tap)
        source = torch.tensor([1.0])
        with ActivationIntervention(model, "0", operation="zero"):
            with self.assertRaises(RuntimeError):
                with TraceSession(model):
                    pass
            self.assertTrue(torch.equal(model(source), torch.zeros_like(source)))
        self.assertEqual(len(tap._forward_hooks), 0)
        with TraceSession(model) as trace:
            model(source)
        self.assertEqual(len(trace.events), 1)

    def test_overlapping_legacy_sessions_reject_and_failed_close_preserves_owner(self):
        tap = TapPoint("tap")
        source = torch.tensor([1.0])
        with TraceSession(tap) as outer:
            rejected = TraceSession(tap)
            with self.assertRaises(RuntimeError):
                rejected.__enter__()
            rejected.close()
            with self.assertRaises(RuntimeError):
                with TraceSession(tap):
                    pass
            tap(source)
        self.assertEqual(len(outer.events), 1)
        self.assertEqual(len(tap._forward_hooks), 0)
        with TraceSession(tap):
            tap(source)

    def test_shared_submodule_rejects_overlapping_sessions_on_different_roots(self):
        tap = TapPoint("shared")
        first = nn.Sequential(tap)
        second = nn.Sequential(tap)
        with TraceSession(first) as trace:
            with self.assertRaises(RuntimeError):
                with TraceSession(second):
                    pass
            first(torch.ones(1))
        self.assertEqual(len(trace.events), 1)
        self.assertEqual(len(tap._forward_hooks), 0)

    def test_independent_models_can_be_traced_at_the_same_time(self):
        first = TapPoint("first")
        second = TapPoint("second")
        with TraceSession(first) as a, TraceSession(second) as b:
            first(torch.ones(1))
            second(torch.zeros(1))
        self.assertEqual([event.node_id for event in a.events], ["first"])
        self.assertEqual([event.node_id for event in b.events], ["second"])

    def test_duplicate_labels_failure_releases_model_for_intervention(self):
        model = nn.Sequential(TapPoint("same"), TapPoint("same"))
        with self.assertRaises(ValueError):
            with TraceSession(model):
                pass
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_registration_failure_without_handle_keeps_model_leased(self):
        class RegistrationFailure(TapPoint):
            def register_forward_hook(self, *args, **kwargs):
                raise RuntimeError("registration failed")

        first = TapPoint("first")
        model = nn.Sequential(first, RegistrationFailure("second"))
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "registration failed") as caught:
            trace.__enter__()
        self.assertIn("discard", str(caught.exception.__cause__))
        self.assertEqual(len(first._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "discard"):
            trace.close()
        fresh = nn.Sequential(TapPoint("fresh"))
        with ActivationIntervention(fresh, "0", operation="zero"):
            self.assertTrue(torch.equal(fresh(torch.ones(1)), torch.zeros(1)))

    def test_capture_overflow_releases_model_and_keeps_byte_cap(self):
        model = nn.Sequential(TapPoint("tap"))
        with self.assertRaisesRegex(RuntimeError, "tensor capacity"):
            with TraceSession(model, max_tensor_bytes=3):
                model(torch.ones(1))
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_manual_close_releases_model_and_is_idempotent(self):
        model = nn.Sequential(TapPoint("tap"))
        session = TraceSession(model)
        with session:
            model(torch.ones(1))
            session.close()
            session.close()
            with ActivationIntervention(model, "0", operation="zero"):
                self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))
        self.assertEqual(len(session.events), 1)
        with self.assertRaises(RuntimeError):
            session.__enter__()

    def test_changed_module_tree_before_forward_aborts_and_releases(self):
        model = nn.Sequential(TapPoint("tap"))
        with self.assertRaisesRegex(RuntimeError, "tree changed"):
            with TraceSession(model):
                model.add_module("new", nn.Identity())
                model(torch.ones(1))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_changed_module_tree_after_forward_invalidates_session(self):
        model = nn.Sequential(TapPoint("tap"))
        with self.assertRaisesRegex(RuntimeError, "tree changed"):
            with TraceSession(model):
                model(torch.ones(1))
                model.add_module("new", nn.Identity())
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_failed_hook_removal_retains_lease_until_cleanup_retried(self):
        first, second = FailOnceTap("first"), TapPoint("second")
        model = nn.Sequential(first, second)
        trace = TraceSession(model)
        trace.__enter__()
        with self.assertRaisesRegex(RuntimeError, "injected removal failure"):
            trace.close()
        self.assertEqual(len(first._forward_hooks), 1)
        self.assertEqual(len(second._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        trace.close()
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_registration_that_installs_then_raises_poisoned_model(self):
        class InstallsThenFails(TapPoint):
            def register_forward_hook(self, *args, **kwargs):
                super().register_forward_hook(*args, **kwargs)
                raise MemoryError("registration failed after installation")

        tap = InstallsThenFails("tap")
        model = nn.Sequential(tap)
        trace = TraceSession(model)
        with self.assertRaisesRegex(MemoryError, "registration failed after installation") as caught:
            trace.__enter__()
        self.assertIn("discard", str(caught.exception.__cause__))
        self.assertEqual(len(tap._forward_hooks), 1)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "discard"):
            trace.close()

    def test_pytorch_auxiliary_hook_registration_oom_poisoned_model(self):
        class FailAuxiliaryDict(dict):
            def __setitem__(self, key, value):
                raise MemoryError("always-call bookkeeping failed")

        model = nn.Sequential(TapPoint("tap"))
        model._forward_hooks_always_called = FailAuxiliaryDict()
        trace = TraceSession(model)
        with self.assertRaisesRegex(MemoryError, "always-call bookkeeping failed") as caught:
            trace.__enter__()
        self.assertIn("discard", str(caught.exception.__cause__))
        self.assertEqual(len(model._forward_hooks), 1)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "discard"):
            trace.close()

    def test_registration_with_malformed_returned_handle_quarantines_model(self):
        for malformed in (None, object()):
            with self.subTest(malformed=malformed):
                class BadReturn(TapPoint):
                    def register_forward_hook(self, *args, **kwargs):
                        super().register_forward_hook(*args, **kwargs)
                        return malformed

                model = nn.Sequential(BadReturn("tap"))
                trace = TraceSession(model)
                with self.assertRaisesRegex(RuntimeError, "handle") as caught:
                    trace.__enter__()
                self.assertIn("discard", str(caught.exception.__cause__))
                self.assertEqual(len(model[0]._forward_hooks), 1)
                with self.assertRaises(RuntimeError):
                    with ActivationIntervention(model, "0", operation="zero"):
                        model(torch.ones(1))
                with self.assertRaisesRegex(RuntimeError, "discard"):
                    trace.close()

    def test_registration_error_remains_primary_if_hook_cleanup_also_fails(self):
        class RegistrationFailure(TapPoint):
            def register_forward_hook(self, *args, **kwargs):
                raise RuntimeError("registration failed")

        first, second = FailOnceTap("first"), TapPoint("second")
        model = nn.Sequential(first, second, RegistrationFailure("third"))
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "registration failed") as caught:
            trace.__enter__()
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("discard", str(caught.exception.__cause__))
        self.assertIn("injected removal failure", str(caught.exception.__cause__.__cause__))
        self.assertEqual(len(first._forward_hooks), 1)
        self.assertEqual(len(second._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "discard"):
            trace.close()
        self.assertEqual(len(first._forward_hooks), 0)

    def test_body_error_remains_primary_if_hook_cleanup_also_fails(self):
        model = nn.Sequential(FailOnceTap("tap"))
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "body failed") as caught:
            with trace:
                raise RuntimeError("body failed")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("injected removal failure", str(caught.exception.__cause__))
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        trace.close()
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_handle_recording_oom_removes_the_just_registered_hook(self):
        class FailSecondAppend(list):
            def append(self, handle):
                if len(self) == 1:
                    raise MemoryError("recording handle failed")
                super().append(handle)

        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        trace._handles = FailSecondAppend()
        with self.assertRaisesRegex(MemoryError, "recording handle failed"):
            trace.__enter__()
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_recording_oom_with_failed_removal_retains_lease_for_retry(self):
        class FailAppend(list):
            def append(self, handle):
                raise MemoryError("recording handle failed")

        class FailOnceRoot(nn.Sequential):
            def __init__(self):
                super().__init__(TapPoint("tap"))
                self.fail_next = True

            def register_forward_pre_hook(self, *args, **kwargs):
                handle = super().register_forward_pre_hook(*args, **kwargs)
                if self.fail_next:
                    self.fail_next = False
                    return FailOnceHandle(handle)
                return handle

        model = FailOnceRoot()
        trace = TraceSession(model)
        trace._handles = FailAppend()
        with self.assertRaisesRegex(MemoryError, "recording handle failed") as caught:
            trace.__enter__()
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("injected removal failure", str(caught.exception.__cause__))
        self.assertEqual(len(model._forward_pre_hooks), 1)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        trace.close()
        self.assertEqual(len(model._forward_pre_hooks), 0)
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_acquire_exception_after_reservation_releases_model(self):
        from looking_glass.model_ownership import ModelLease

        model = nn.Sequential(TapPoint("tap"))
        acquire = ModelLease.acquire

        def fail_after_reservation(lease):
            acquire(lease)
            raise MemoryError("after reservation")

        with patch.object(ModelLease, "acquire", fail_after_reservation):
            with self.assertRaisesRegex(MemoryError, "after reservation"):
                TraceSession(model).__enter__()
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_interruption_after_handle_append_does_not_lose_or_double_remove(self):
        class AppendThenInterrupt(list):
            def append(self, handle):
                super().append(handle)
                raise KeyboardInterrupt("after append")

        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        trace._handles = AppendThenInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            trace.__enter__()
        self.assertFalse(model._forward_pre_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            self.assertTrue(torch.equal(model(torch.ones(1)), torch.zeros(1)))

    def test_changed_module_tree_during_forward_aborts_and_cleans_up(self):
        class MutableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TapPoint("tap")

            def forward(self, x):
                result = self.tap(x)
                self.add_module("new", nn.Identity())
                return result

        model = MutableModel()
        with self.assertRaisesRegex(RuntimeError, "tree changed"):
            with TraceSession(model):
                model(torch.ones(1))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_caught_mutation_remains_failed_even_after_tree_is_restored(self):
        class MutableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TapPoint("tap")

            def forward(self, x):
                output = self.tap(x)
                self.add_module("new", nn.Identity())
                return output

        model = MutableModel()
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace:
                with self.assertRaisesRegex(RuntimeError, "tree changed"):
                    model(torch.ones(1))
                del model.new
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    model(torch.ones(1) * 2)
        self.assertTrue(trace.failed)
        self.assertEqual(len(trace.events), 1)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_caught_overflow_cannot_exit_as_successful_trace(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model, max_events=1)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace:
                model(torch.ones(1))
                with self.assertRaisesRegex(RuntimeError, "capacity"):
                    model(torch.ones(1))
        self.assertTrue(trace.failed)
        self.assertEqual(len(trace.events), 1)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_caught_model_exception_cannot_exit_as_successful_trace(self):
        class FailsOnce(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TapPoint("tap")
                self.should_fail = True

            def forward(self, x):
                x = self.tap(x)
                if self.should_fail:
                    self.should_fail = False
                    raise ValueError("forward failed")
                return x

        model = FailsOnce()
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace:
                with self.assertRaisesRegex(ValueError, "forward failed"):
                    model(torch.ones(1))
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    model(torch.ones(1))
        self.assertTrue(trace.failed)
        self.assertEqual(len(trace.events), 1)
        self.assertFalse(model.tap._forward_hooks)

    def test_caught_cleanup_error_cannot_exit_as_successful_trace(self):
        model = nn.Sequential(FailOnceTap("tap"))
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace:
                model(torch.ones(1))
                with self.assertRaisesRegex(RuntimeError, "injected removal failure"):
                    trace.close()
        self.assertTrue(trace.failed)
        self.assertFalse(model[0]._forward_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_failed_lease_release_marks_trace_failed_and_can_be_retried(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        trace.__enter__()
        with patch.object(trace._lease, "release", side_effect=RuntimeError("release failed")):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                trace.close()
        self.assertTrue(trace.failed)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                model(torch.ones(1))
        trace.close()
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_interrupted_cleanup_transition_remains_retryable(self):
        for operation in ("close", "exit"):
            with self.subTest(operation=operation):
                model = nn.Sequential(TapPoint("tap"))
                trace = TraceSession(model)
                trace.__enter__()
                model(torch.ones(1))

                class InterruptOnce:
                    def __init__(self):
                        self.lock = threading.RLock()
                        self.interrupted = False

                    def __enter__(self):
                        self.lock.acquire()
                        return self

                    def __exit__(self, *args):
                        self.lock.release()
                        if trace._closing and not self.interrupted:
                            self.interrupted = True
                            raise KeyboardInterrupt("injected cleanup interruption")

                trace._lifecycle_lock = InterruptOnce()
                with self.assertRaisesRegex(KeyboardInterrupt, "cleanup interruption"):
                    if operation == "exit":
                        trace.__exit__(None, None, None)
                    else:
                        trace.close()
                with self.assertRaises(RuntimeError):
                    with ActivationIntervention(model, "0", operation="zero"):
                        pass

                if operation == "exit":
                    errors = []

                    def retry_as_supervisor():
                        try:
                            trace.close()
                        except BaseException as error:
                            errors.append(error)

                    supervisor = threading.Thread(target=retry_as_supervisor)
                    supervisor.start()
                    supervisor.join(5)
                    self.assertFalse(supervisor.is_alive())
                    self.assertEqual(errors, [])
                else:
                    trace.close()
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        trace.__exit__(None, None, None)
                self.assertTrue(trace.failed)
                self.assertFalse(model[0]._forward_hooks)
                with ActivationIntervention(model, "0", operation="zero"):
                    model(torch.ones(1))

    def test_interrupted_entry_transition_does_not_strand_session(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)

        class InterruptOnce:
            def __init__(self):
                self.lock = threading.RLock()
                self.interrupted = False

            def __enter__(self):
                self.lock.acquire()
                return self

            def __exit__(self, *args):
                self.lock.release()
                if trace._entering and not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt("injected entry interruption")

        trace._lifecycle_lock = InterruptOnce()
        with self.assertRaisesRegex(KeyboardInterrupt, "entry interruption"):
            trace.__enter__()
        errors = []

        def retry_as_supervisor():
            try:
                trace.close()
            except BaseException as error:
                errors.append(error)

        supervisor = threading.Thread(target=retry_as_supervisor)
        supervisor.start()
        supervisor.join(5)
        self.assertFalse(supervisor.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(trace.failed)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_repeated_enter_cannot_clean_up_active_session(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        with trace:
            with self.assertRaisesRegex(RuntimeError, "single-use"):
                trace.__enter__()
            with self.assertRaises(RuntimeError):
                with ActivationIntervention(model, "0", operation="zero"):
                    pass
            model(torch.ones(1))
        self.assertFalse(trace.failed)
        self.assertEqual(len(trace.events), 1)

    def test_interrupted_entry_return_cleans_installed_hooks(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        code = TraceSession.__enter__.__code__
        return_offset = next(i.offset for i in dis.get_instructions(code)
                             if i.opname == "RETURN_VALUE")
        return_line = max((offset, line) for offset, line in dis.findlinestarts(code)
                          if offset <= return_offset)[1]

        def interrupt(frame, event, arg):
            if (frame.f_code is TraceSession.__enter__.__code__ and event == "line"
                    and frame.f_lineno == return_line):
                raise KeyboardInterrupt("injected entry return interruption")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaisesRegex(KeyboardInterrupt, "entry return interruption"):
                trace.__enter__()
        finally:
            sys.settrace(None)
        self.assertTrue(trace.failed)
        self.assertFalse(model[0]._forward_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_interrupted_return_delivery_can_be_closed_by_owner(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)

        def interrupt(frame, event, arg):
            if frame.f_code is TraceSession.__enter__.__code__ and event == "return":
                raise KeyboardInterrupt("injected return delivery interruption")
            return interrupt

        try:
            sys.settrace(interrupt)
            with self.assertRaisesRegex(KeyboardInterrupt, "return delivery interruption"):
                trace.__enter__()
        finally:
            sys.settrace(None)
        # The method completed before its result was delivered to the caller.
        # Retaining the session lets the entering thread explicitly close it.
        self.assertTrue(model[0]._forward_hooks)
        trace.close()
        self.assertFalse(model[0]._forward_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_base_exception_in_forward_quarantines_and_preserves_original(self):
        for kind in (KeyboardInterrupt, SystemExit):
            for caught_inside in (False, True):
                with self.subTest(kind=kind.__name__, caught_inside=caught_inside):
                    original = kind("injected forward interruption")

                    class Bomb(nn.Module):
                        def forward(self, x):
                            raise original

                    model = nn.Sequential(TapPoint("tap"), Bomb())
                    trace = SkipInterruptedFinish(model)
                    if caught_inside:
                        with self.assertRaisesRegex(RuntimeError, "active forward"):
                            with trace:
                                with self.assertRaises(kind):
                                    model(torch.ones(1))
                    else:
                        with self.assertRaises(kind) as caught:
                            with trace:
                                model(torch.ones(1))
                        self.assertIs(caught.exception, original)
                        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
                        self.assertIn("active forward", str(caught.exception.__cause__))
                    self.assertTrue(trace.failed)
                    self.assertTrue(model[0]._forward_hooks)
                    with self.assertRaisesRegex(RuntimeError, "active forward"):
                        trace.close()
                    with self.assertRaises(RuntimeError):
                        with ActivationIntervention(model, "0", operation="zero"):
                            pass

    def test_caught_base_exception_detection_can_be_delayed_until_exit(self):
        class InterruptOnce(nn.Module):
            def __init__(self):
                super().__init__()
                self.interrupted = False

            def forward(self, x):
                if not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt("injected interruption")
                return x

        model = nn.Sequential(TapPoint("tap"), InterruptOnce())
        trace = SkipInterruptedFinish(model)
        with self.assertRaisesRegex(RuntimeError, "active forward"):
            with trace:
                with self.assertRaises(KeyboardInterrupt):
                    model(torch.ones(1))
                self.assertFalse(trace.failed)
                # This illustrates why a caller must stop immediately: from
                # public hooks alone the stale frame resembles recursion.
                model(torch.ones(1))
                self.assertEqual(len(trace.events), 2)
        self.assertTrue(trace.failed)
        with self.assertRaises(RuntimeError):
            with ActivationIntervention(model, "0", operation="zero"):
                pass

    def test_native_interrupted_forward_keeps_ownership_until_safe(self):
        original = KeyboardInterrupt("injected interruption")

        class Bomb(nn.Module):
            def forward(self, x):
                raise original

        model = nn.Sequential(TapPoint("tap"), Bomb())
        trace = TraceSession(model)
        with self.assertRaises(KeyboardInterrupt) as caught:
            with trace:
                model(torch.ones(1))
        self.assertIs(caught.exception, original)
        self.assertTrue(trace.failed)
        if model[0]._forward_hooks:
            # Some PyTorch versions skip always-call hooks for BaseException.
            with self.assertRaisesRegex(RuntimeError, "active forward"):
                trace.close()
            with self.assertRaises(RuntimeError):
                with ActivationIntervention(model, "0", operation="zero"):
                    pass
        else:
            # A version that runs the completion hook can cleanly release it.
            trace.close()
            with ActivationIntervention(model, "0", operation="zero"):
                pass

    def test_recursive_root_forwards_keep_completion_state_per_invocation(self):
        class Recursive(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TapPoint("tap")

            def forward(self, x, depth):
                y = self.tap(x)
                return self(y, depth - 1) if depth else y

        model = Recursive()
        with TraceSession(model) as trace:
            model(torch.ones(1), 2)
        self.assertFalse(trace.failed)
        self.assertEqual([event.occurrence for event in trace.events], [0, 1, 2])

    def test_foreign_thread_root_forward_is_rejected_without_events(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        errors = []

        def foreign_forward():
            try:
                model(torch.ones(1))
            except BaseException as error:
                errors.append(error)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace:
                thread = threading.Thread(target=foreign_forward)
                thread.start()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertRegex(str(errors[0]), "thread")
        self.assertTrue(trace.failed)
        self.assertEqual(trace.events, [])
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_foreign_thread_direct_tap_is_rejected_without_events(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        errors = []

        def foreign_forward():
            try:
                model[0](torch.ones(1))
            except BaseException as error:
                errors.append(error)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace:
                thread = threading.Thread(target=foreign_forward)
                thread.start()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertRegex(str(errors[0]), "thread")
        self.assertEqual(trace.events, [])
        self.assertTrue(trace.failed)

    def test_foreign_thread_close_does_not_release_active_model(self):
        model = nn.Sequential(TapPoint("tap"))
        trace = TraceSession(model)
        errors = []

        def foreign_close():
            try:
                trace.close()
            except BaseException as error:
                errors.append(error)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace:
                thread = threading.Thread(target=foreign_close)
                thread.start()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertRegex(str(errors[0]), "thread")
                with self.assertRaises(RuntimeError):
                    with ActivationIntervention(model, "0", operation="zero"):
                        model(torch.ones(1))
        self.assertTrue(trace.failed)
        self.assertFalse(model[0]._forward_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_supervisor_can_retry_cleanup_after_owner_context_exits(self):
        model = nn.Sequential(FailOnceTap("tap"))
        trace = TraceSession(model)
        errors = []

        def run_owner():
            try:
                with trace:
                    model(torch.ones(1))
            except BaseException as error:
                errors.append(error)

        owner = threading.Thread(target=run_owner)
        owner.start()
        owner.join(5)
        self.assertFalse(owner.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "injected removal failure")
        self.assertTrue(trace.failed)
        trace.close()  # The owner context has exited; this supervisor retries cleanup.
        self.assertFalse(model[0]._forward_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_foreign_forward_does_not_pop_an_active_owner_forward(self):
        class PausedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TapPoint("tap")
                self.entered = threading.Event()
                self.resume = threading.Event()

            def forward(self, x):
                self.entered.set()
                if not self.resume.wait(5):
                    raise TimeoutError("owner was not resumed")
                return self.tap(x)

        model = PausedModel()
        trace = TraceSession(model)
        errors = []

        def run_owner():
            try:
                with trace:
                    model(torch.ones(1))
            except BaseException as error:
                errors.append(error)

        owner = threading.Thread(target=run_owner)
        owner.start()
        self.assertTrue(model.entered.wait(5))
        try:
            with self.assertRaisesRegex(RuntimeError, "thread"):
                model(torch.ones(1) * 101)
        finally:
            model.resume.set()
            owner.join(5)
        self.assertFalse(owner.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "failed")
        self.assertTrue(trace.failed)
        self.assertEqual(trace.events, [])
        self.assertEqual(trace._forward_stack, [])
        self.assertFalse(model.tap._forward_hooks)

    def test_close_from_inside_active_forward_is_rejected(self):
        class CloseDuringForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TapPoint("tap")

            def forward(self, x):
                x = self.tap(x)
                trace.close()
                return x

        model = CloseDuringForward()
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "active forward"):
            with trace:
                model(torch.ones(1))
        self.assertTrue(trace.failed)
        self.assertFalse(model.tap._forward_hooks)

    def test_foreign_close_during_entry_cannot_release_model_lease(self):
        entered, resume = threading.Event(), threading.Event()

        class PausedTrace(TraceSession):
            def _install_hook(self, register, callback):
                super()._install_hook(register, callback)
                if not entered.is_set():
                    entered.set()
                    if not resume.wait(5):
                        raise TimeoutError("entry was not resumed")

        model = nn.Sequential(TapPoint("tap"))
        trace = PausedTrace(model)
        errors = []

        def run_owner():
            try:
                with trace:
                    model(torch.ones(1))
            except BaseException as error:
                errors.append(error)

        owner = threading.Thread(target=run_owner)
        owner.start()
        self.assertTrue(entered.wait(5))
        try:
            with self.assertRaisesRegex(RuntimeError, "thread|entry"):
                trace.close()
            with self.assertRaises(RuntimeError):
                with ActivationIntervention(model, "0", operation="zero"):
                    model(torch.ones(1))
        finally:
            resume.set()
            owner.join(5)
        self.assertFalse(owner.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "failed")
        self.assertFalse(model[0]._forward_hooks)
        with ActivationIntervention(model, "0", operation="zero"):
            model(torch.ones(1))

    def test_competing_supervisor_retries_cannot_release_during_cleanup(self):
        removing, resume = threading.Event(), threading.Event()

        class PausedHandle:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.calls = 0

            def remove(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first removal failed")
                removing.set()
                if not resume.wait(5):
                    raise TimeoutError("cleanup was not resumed")
                self.wrapped.remove()

        class PausedTap(TapPoint):
            def register_forward_hook(self, *args, **kwargs):
                return PausedHandle(super().register_forward_hook(*args, **kwargs))

        model = nn.Sequential(PausedTap("tap"))
        trace = TraceSession(model)
        with self.assertRaisesRegex(RuntimeError, "first removal failed"):
            with trace:
                model(torch.ones(1))
        errors = []

        def retry():
            try:
                trace.close()
            except BaseException as error:
                errors.append(error)

        supervisor = threading.Thread(target=retry)
        supervisor.start()
        self.assertTrue(removing.wait(5))
        try:
            with self.assertRaisesRegex(RuntimeError, "cleanup|closing"):
                trace.close()
            with self.assertRaises(RuntimeError):
                with ActivationIntervention(model, "0", operation="zero"):
                    model(torch.ones(1))
        finally:
            resume.set()
            supervisor.join(5)
        self.assertFalse(supervisor.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(model[0]._forward_hooks)

    def test_foreign_tap_during_cleanup_invalidates_the_owner_session(self):
        removing, resume = threading.Event(), threading.Event()

        class PausedHandle:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def remove(self):
                removing.set()
                if not resume.wait(5):
                    raise TimeoutError("cleanup was not resumed")
                self.wrapped.remove()

        class PausedTap(TapPoint):
            def register_forward_hook(self, *args, **kwargs):
                return PausedHandle(super().register_forward_hook(*args, **kwargs))

        model = nn.Sequential(PausedTap("tap"))
        trace = TraceSession(model)
        errors = []

        def run_owner():
            try:
                with trace:
                    model(torch.ones(1))
            except BaseException as error:
                errors.append(error)

        owner = threading.Thread(target=run_owner)
        owner.start()
        self.assertTrue(removing.wait(5))
        try:
            with self.assertRaisesRegex(RuntimeError, "thread"):
                model[0](torch.tensor([101.0]))
        finally:
            resume.set()
            owner.join(5)
        self.assertFalse(owner.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertRegex(str(errors[0]), "failed")
        self.assertTrue(trace.failed)
        self.assertEqual([event.value.item() for event in trace.events], [1.0])
        self.assertFalse(model[0]._forward_hooks)


if __name__ == "__main__":
    unittest.main()
