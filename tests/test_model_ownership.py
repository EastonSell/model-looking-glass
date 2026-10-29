import importlib.util
import inspect
import sys
import threading
import unittest

from torch import nn


class ModelOwnershipTests(unittest.TestCase):
    def test_overlapping_model_trees_are_exclusive_and_release_is_reusable(self):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.model_ownership'))
        from looking_glass.model_ownership import ModelLease

        shared = nn.Identity()
        first = nn.Sequential(shared)
        second = nn.Sequential(shared)
        lease = ModelLease(first)
        lease.acquire()
        with self.assertRaises(RuntimeError):
            ModelLease(second).acquire()
        lease.release()
        with ModelLease(second):
            pass

    def test_failed_or_released_lease_cannot_release_another_owner(self):
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        first = ModelLease(model)
        first.acquire()
        failed = ModelLease(model)
        with self.assertRaises(RuntimeError):
            failed.acquire()
        failed.release()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        first.release()
        current = ModelLease(model)
        current.acquire()
        first.release()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        current.release()
        with ModelLease(model):
            pass

    def test_context_failure_releases_lease_and_original_exception(self):
        from looking_glass.model_ownership import ModelLease
        model = nn.Identity()
        with self.assertRaisesRegex(ValueError, 'original failure'):
            with ModelLease(model):
                raise ValueError('original failure')
        with ModelLease(model):
            pass

    def test_context_preserves_body_error_if_release_fails(self):
        from looking_glass.model_ownership import ModelLease
        class FailingLease(ModelLease):
            fail = True
            def release(self):
                if self.fail:
                    raise RuntimeError('cleanup failure')
                return super().release()
        lease = FailingLease(nn.Sequential(nn.Identity()))
        try:
            with self.assertRaisesRegex(ValueError, 'body failure') as caught:
                with lease:
                    raise ValueError('body failure')
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        finally:
            lease.fail = False
            lease.release()

    def test_lease_is_single_use(self):
        from looking_glass.model_ownership import ModelLease
        lease = ModelLease(nn.Identity())
        lease.acquire()
        with self.assertRaises(RuntimeError):
            lease.acquire()
        lease.release()
        with self.assertRaises(RuntimeError):
            lease.acquire()

    def test_concurrent_acquisition_admits_only_one_owner(self):
        from looking_glass.model_ownership import ModelLease
        model = nn.Identity()
        start = threading.Barrier(3)
        attempted = threading.Barrier(3)
        outcomes = []
        def runner():
            lease = ModelLease(model)
            start.wait()
            try:
                lease.acquire()
                outcomes.append('acquired')
            except RuntimeError:
                outcomes.append('rejected')
            attempted.wait()
            lease.release()
        threads = [threading.Thread(target=runner) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        attempted.wait()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertCountEqual(outcomes, ['acquired', 'rejected'])
        with ModelLease(model):
            pass

    def test_non_module_is_rejected(self):
        from looking_glass.model_ownership import ModelLease
        with self.assertRaises(TypeError):
            ModelLease(object())

    def test_verify_tree_rejects_added_alias_and_registered_child(self):
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        lease = ModelLease(model).acquire()
        model.add_module('alias', model[0])
        with self.assertRaises(RuntimeError):
            lease.verify_tree()
        lease.release()
        with ModelLease(model):
            pass

    def test_tree_change_during_acquisition_releases_all_modules(self):
        from looking_glass.model_ownership import ModelLease
        class ChangingTree(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Identity()
                self.changed = False
            def named_modules(self, *args, **kwargs):
                yield from super().named_modules(*args, **kwargs)
                if not self.changed:
                    self.add_module('late', nn.Identity())
                    self.changed = True
        model = ChangingTree()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with ModelLease(model.first):
            pass
        with ModelLease(model.late):
            pass

    def test_intervention_and_shared_lease_reject_both_nesting_orders(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        with ModelLease(model):
            with self.assertRaises(RuntimeError):
                with ActivationCapture(model, '0'):
                    pass
        with ActivationCapture(model, '0'):
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
            model(torch.ones(1, 2))
        with ModelLease(model):
            pass

    def test_failed_hook_removal_retains_ownership_until_retried(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class FailOnceHandle:
            def __init__(self, handle):
                self.handle = handle
                self.failed = False
            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('remove failure')
                self.handle.remove()
        class FlakyTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                return FailOnceHandle(super().register_forward_hook(*args, **kwargs))
        model = nn.Sequential(FlakyTap())
        capture = ActivationCapture(model, '0')
        with self.assertRaisesRegex(RuntimeError, 'remove failure'):
            with capture:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            _ = capture.value
        capture.close()
        with ModelLease(model):
            pass

    def test_hook_removal_failure_does_not_mask_forward_error(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class FailOnceHandle:
            def __init__(self, handle):
                self.handle = handle
                self.failed = False
            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('remove failure')
                self.handle.remove()
        class FlakyTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                return FailOnceHandle(super().register_forward_hook(*args, **kwargs))
        class Failing(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = FlakyTap()
            def forward(self, x):
                self.tap(x)
                raise ValueError('original failure')
        model = Failing()
        capture = ActivationCapture(model, 'tap')
        with self.assertRaisesRegex(ValueError, 'original failure') as caught:
            with capture:
                model(torch.ones(1, 2))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn('remove failure', str(caught.exception.__cause__))
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        capture.close()
        with ModelLease(model):
            pass

    def test_registration_and_cleanup_failures_preserve_both_errors_and_lease(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class FailOnceHandle:
            def __init__(self, handle):
                self.handle = handle
                self.failed = False
            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('cleanup failure')
                self.handle.remove()
        class FailingTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                raise ValueError('registration failure')
        class FailingRoot(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = FailingTap()
            def register_forward_pre_hook(self, *args, **kwargs):
                return FailOnceHandle(super().register_forward_pre_hook(*args, **kwargs))
        model = FailingRoot()
        capture = ActivationCapture(model, 'tap')
        with self.assertRaisesRegex(ValueError, 'registration failure') as caught:
            with capture:
                pass
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn('cleanup failure', str(caught.exception.__cause__))
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            capture.close()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()

    def test_hook_registration_never_appends_after_installing_a_hook(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class AppendWouldFail(list):
            def append(self, value):
                raise MemoryError('list append failed')
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        capture._handles = AppendWouldFail([None] * 6)
        with capture:
            model(torch.ones(1, 2))
        self.assertTrue(torch.equal(capture.value, torch.ones(1, 2)))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model[0]._forward_hooks), 0)
        with ModelLease(model):
            pass

    def test_error_immediately_after_acquisition_releases_lease(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        acquire = capture._lease.acquire
        def acquire_then_fail():
            acquire()
            raise RuntimeError('after acquire')
        capture._lease.acquire = acquire_then_fail
        with self.assertRaisesRegex(RuntimeError, 'after acquire'):
            with capture:
                pass
        with ModelLease(model):
            pass

    def test_acquisition_rolls_back_if_recording_held_state_fails(self):
        from looking_glass.model_ownership import ModelLease
        class FaultyLease(ModelLease):
            def __setattr__(self, name, value):
                if name == '_held' and value and getattr(self, 'inject_failure', False):
                    self.inject_failure = False
                    raise MemoryError('held-state failure')
                super().__setattr__(name, value)
        model = nn.Sequential(nn.Identity())
        lease = FaultyLease(model)
        lease.inject_failure = True
        with self.assertRaisesRegex(MemoryError, 'held-state failure'):
            lease.acquire()
        with ModelLease(model):
            pass

    def test_tree_change_after_forward_invalidates_capture_on_exit(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        with self.assertRaises(RuntimeError):
            with capture:
                model(torch.ones(1, 2))
                model[0] = nn.Identity()
        with self.assertRaises(RuntimeError):
            _ = capture.value

    def test_partial_pytorch_registration_quarantines_the_model(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class AllocationFailure(dict):
            def __setitem__(self, key, value):
                raise MemoryError('registration allocation failed')
        model = nn.Sequential(nn.Identity())
        model._forward_hooks_always_called = AllocationFailure()
        capture = ActivationCapture(model, '0')
        with self.assertRaisesRegex(MemoryError, 'registration allocation failed'):
            with capture:
                pass
        # PyTorch installed a hook before raising and returned no handle.
        self.assertGreater(len(model._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            capture.close()

    def test_custom_registration_that_installs_then_raises_quarantines_model(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class InstalledThenRaises(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                super().register_forward_hook(*args, **kwargs)
                raise MemoryError('registration failed after install')
        model = nn.Sequential(InstalledThenRaises())
        capture = ActivationCapture(model, '0')
        with self.assertRaisesRegex(MemoryError, 'registration failed after install'):
            with capture:
                pass
        self.assertGreater(len(model[0]._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            capture.close()
        with ModelLease(nn.Sequential(nn.Identity())):
            pass

    def test_registration_returning_invalid_handle_quarantines_model(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class InvalidHandle(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                super().register_forward_hook(*args, **kwargs)
                return None
        model = nn.Sequential(InvalidHandle())
        capture = ActivationCapture(model, '0')
        with self.assertRaises(TypeError):
            with capture:
                pass
        self.assertGreater(len(model[0]._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            capture.close()

    def test_foreign_thread_target_cannot_contaminate_owner_capture(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        started, release = threading.Event(), threading.Event()
        errors = []
        class PausedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                started.set()
                if not release.wait(5):
                    raise RuntimeError('test thread timed out')
                return self.tap(x)
        model = PausedModel()
        capture = ActivationCapture(model, 'tap')
        def foreign():
            if started.wait(5):
                try:
                    model.tap(torch.tensor([[101.]]))
                except RuntimeError as error:
                    errors.append(error)
            release.set()
        worker = threading.Thread(target=foreign)
        worker.start()
        with self.assertRaises(RuntimeError):
            with capture:
                model(torch.tensor([[1.]]))
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = capture.value

    def test_foreign_thread_exit_does_not_release_active_session(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        capture.__enter__()
        errors = []
        def foreign_exit():
            try:
                capture.__exit__(None, None, None)
            except RuntimeError as error:
                errors.append(error)
        worker = threading.Thread(target=foreign_exit)
        worker.start()
        worker.join(timeout=5)
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            capture.__exit__(None, None, None)
        with ModelLease(model):
            pass

    def test_supervisor_thread_can_retry_removing_known_handles(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class FailOnceHandle:
            def __init__(self, handle):
                self.handle = handle
                self.failed = False
            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('remove failure')
                self.handle.remove()
        class FlakyTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                return FailOnceHandle(super().register_forward_hook(*args, **kwargs))
        model = nn.Sequential(FlakyTap())
        capture = ActivationCapture(model, '0')
        with self.assertRaises(RuntimeError):
            with capture:
                model(torch.ones(1, 2))
        errors = []
        def retry():
            try:
                capture.close()
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=retry)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        with ModelLease(model):
            pass

    def test_close_during_hook_registration_cannot_release_the_lease(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        installed, resume = threading.Event(), threading.Event()
        errors = []
        register = capture._register
        def pause_after_first(slot, *args, **kwargs):
            register(slot, *args, **kwargs)
            if slot == 0:
                installed.set()
                if not resume.wait(5):
                    raise RuntimeError('test thread timed out')
        capture._register = pause_after_first
        def owner():
            try:
                with capture:
                    model(torch.ones(1, 2))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=owner)
        worker.start()
        self.assertTrue(installed.wait(5))
        try:
            with self.assertRaises(RuntimeError):
                capture.close()
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(capture.value, torch.ones(1, 2)))

    def test_simultaneous_cleanup_retries_cannot_release_early(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        started, resume = threading.Event(), threading.Event()
        class PausingHandle:
            def __init__(self, handle):
                self.handle = handle
                self.calls = 0
            def remove(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('remove failure')
                started.set()
                if not resume.wait(5):
                    raise RuntimeError('cleanup timed out')
                self.handle.remove()
        class FlakyTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                return PausingHandle(super().register_forward_hook(*args, **kwargs))
        model = nn.Sequential(FlakyTap())
        capture = ActivationCapture(model, '0')
        with self.assertRaises(RuntimeError):
            with capture:
                model(torch.ones(1, 2))
        errors = []
        def retry():
            try:
                capture.close()
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=retry)
        worker.start()
        self.assertTrue(started.wait(5))
        try:
            with self.assertRaises(RuntimeError):
                capture.close()
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        with ModelLease(model):
            pass

    def test_close_before_entry_is_not_a_cancellation_api(self):
        from looking_glass.interventions import ActivationCapture
        capture = ActivationCapture(nn.Sequential(nn.Identity()), '0')
        with self.assertRaises(RuntimeError):
            capture.close()

    def test_foreign_callback_during_cleanup_prevents_publication(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        started, resume = threading.Event(), threading.Event()
        errors = []
        class PausingHandle:
            def __init__(self, handle):
                self.handle = handle
            def remove(self):
                started.set()
                if not resume.wait(5):
                    raise RuntimeError('test thread timed out')
                self.handle.remove()
        class PausingTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                return PausingHandle(super().register_forward_hook(*args, **kwargs))
        model = nn.Sequential(PausingTap())
        capture = ActivationCapture(model, '0')
        def owner():
            try:
                with capture:
                    model(torch.tensor([[1.]]))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=owner)
        worker.start()
        self.assertTrue(started.wait(5))
        try:
            with self.assertRaises(RuntimeError):
                model[0](torch.tensor([[101.]]))
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = capture.value

    def test_interrupted_release_never_exposes_a_partial_module_tree(self):
        import looking_glass.model_ownership as ownership
        from looking_glass.model_ownership import ModelLease
        class InterruptedSnapshot(dict):
            def __delitem__(self, key):
                super().__delitem__(key)
                self.deletes = getattr(self, 'deletes', 0) + 1
                if self.deletes == 2:
                    raise KeyboardInterrupt('interrupted release')
        class InterruptingRegistry(dict):
            def copy(self):
                return InterruptedSnapshot(self)
            def __delitem__(self, key):
                super().__delitem__(key)
                self.deletes = getattr(self, 'deletes', 0) + 1
                if self.deletes == 2:
                    raise KeyboardInterrupt('interrupted release')
        model = nn.Sequential(nn.Identity(), nn.Identity())
        lease = ModelLease(model).acquire()
        with ownership._lock:
            ownership._owners = InterruptingRegistry(
                {key: entry for key, entry in ownership._owners.items() if entry[0]() is not None})
        try:
            with self.assertRaises(KeyboardInterrupt):
                lease.release()
            # Remove only the fault injector, preserving the live registry state.
            with ownership._lock:
                ownership._owners = dict(ownership._owners)
            for part in (model, model[0], model[1]):
                with self.assertRaises(RuntimeError):
                    ModelLease(part).acquire()
        finally:
            with ownership._lock:
                ownership._owners = dict(ownership._owners)
            lease.release()
        with ModelLease(model):
            pass

    def test_retry_after_release_linearization_cannot_revoke_a_new_owner(self):
        from looking_glass.model_ownership import ModelLease
        class InterruptAfterSwap(ModelLease):
            def __setattr__(self, name, value):
                if name == '_modules' and value == () and getattr(self, 'inject_failure', False):
                    self.inject_failure = False
                    raise KeyboardInterrupt('after registry swap')
                super().__setattr__(name, value)
        model = nn.Sequential(nn.Identity())
        first = InterruptAfterSwap(model).acquire()
        first.inject_failure = True
        with self.assertRaises(KeyboardInterrupt):
            first.release()
        with self.assertRaises(RuntimeError):
            _ = first.registered_paths
        with self.assertRaises(RuntimeError):
            first.verify_tree()
        second = ModelLease(model).acquire()
        first.release()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        second.release()
        with ModelLease(model):
            pass

    def test_foreign_callback_during_registration_cannot_complete_entry(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        installed, resume = threading.Event(), threading.Event()
        errors = []
        register = capture._register
        def pause_after_target(slot, *args, **kwargs):
            register(slot, *args, **kwargs)
            if slot == 1:
                installed.set()
                if not resume.wait(5):
                    raise RuntimeError('test thread timed out')
        capture._register = pause_after_target
        def owner():
            try:
                with capture:
                    model(torch.ones(1, 2))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=owner)
        worker.start()
        self.assertTrue(installed.wait(5))
        try:
            with self.assertRaises(RuntimeError):
                model[0](torch.full((1, 2), 101.))
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = capture.value
        with ModelLease(model):
            pass

    def test_interruption_after_exit_closing_transition_cleans_hooks(self):
        import torch
        from looking_glass.interventions import ActivationCapture, _ActivationSession
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        source, start = inspect.getsourcelines(_ActivationSession.__exit__)
        claim = next(i for i, line in enumerate(source) if line.strip().startswith('self._closing = '))
        inject_line = start + claim + 1
        def trace(frame, event, arg):
            if (frame.f_code is _ActivationSession.__exit__.__code__ and event == 'line'
                    and frame.f_lineno == inject_line):
                sys.settrace(None)
                raise KeyboardInterrupt('exit cancellation')
            return trace
        with self.assertRaisesRegex(KeyboardInterrupt, 'exit cancellation'):
            sys.settrace(trace)
            try:
                with capture:
                    model(torch.ones(1, 2))
            finally:
                sys.settrace(None)
        with self.assertRaises(RuntimeError):
            _ = capture.value
        self.assertEqual(len(model[0]._forward_hooks), 0)
        with ModelLease(model):
            pass

    def test_interruption_at_entry_return_cleans_hooks(self):
        from looking_glass.interventions import ActivationCapture, _ActivationSession
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        capture = ActivationCapture(model, '0')
        source, start = inspect.getsourcelines(_ActivationSession.__enter__)
        return_line = start + next(i for i, line in enumerate(source) if line.strip() == 'return self')
        def trace(frame, event, arg):
            if (frame.f_code is _ActivationSession.__enter__.__code__ and event == 'line'
                    and frame.f_lineno == return_line):
                sys.settrace(None)
                raise KeyboardInterrupt('entry cancellation')
            return trace
        with self.assertRaisesRegex(KeyboardInterrupt, 'entry cancellation'):
            sys.settrace(trace)
            try:
                with capture:
                    self.fail('context should not have entered')
            finally:
                sys.settrace(None)
        self.assertEqual(len(model[0]._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            _ = capture.value
        with ModelLease(model):
            pass

    def test_abort_recovers_after_entry_return_event_is_interrupted(self):
        import torch
        from looking_glass.interventions import ActivationCapture, ActivationIntervention, _ActivationSession
        from looking_glass.model_ownership import ModelLease
        for intervention in (False, True):
            with self.subTest(intervention=intervention):
                model = nn.Sequential(nn.Identity())
                session = (ActivationIntervention(model, '0', operation='patch', value=torch.ones(1, 2))
                           if intervention else ActivationCapture(model, '0'))
                entry = ActivationIntervention.__enter__ if intervention else _ActivationSession.__enter__
                def trace(frame, event, arg):
                    if frame.f_code is entry.__code__ and event == 'return' and arg is session:
                        sys.settrace(None)
                        raise KeyboardInterrupt('entry return event')
                    return trace
                with self.assertRaisesRegex(KeyboardInterrupt, 'entry return event'):
                    sys.settrace(trace)
                    try:
                        with session:
                            self.fail('context should not have entered')
                    finally:
                        sys.settrace(None)
                session.abort()
                self.assertEqual(len(model[0]._forward_hooks), 0)
                with self.assertRaises(RuntimeError):
                    _ = session.metadata
                if intervention:
                    self.assertIsNone(session._source)
                with ModelLease(model):
                    pass

    def test_abort_during_forward_rejects_and_retains_the_lease(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        errors = []
        test = self
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                try:
                    session.abort()
                except RuntimeError as error:
                    errors.append(error)
                with test.assertRaises(RuntimeError):
                    ModelLease(self).acquire()
                test.assertGreater(len(self.tap._forward_hooks), 0)
                return self.tap(x)
        model = Root()
        session = ActivationCapture(model, 'tap')
        with self.assertRaisesRegex(RuntimeError, 'already invalid'):
            with session:
                model(torch.ones(1, 2))
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = session.value
        with ModelLease(model):
            pass

    def test_foreign_abort_rejects_without_releasing_active_owner(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        model = nn.Sequential(nn.Identity())
        session = ActivationCapture(model, '0')
        errors = []
        def foreign():
            try:
                session.abort()
            except BaseException as error:
                errors.append(error)
        with self.assertRaisesRegex(RuntimeError, 'already invalid'):
            with session:
                worker = threading.Thread(target=foreign)
                worker.start()
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
                model(torch.ones(1, 2))
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = session.value
        with ModelLease(model):
            pass

    def test_abort_cleanup_failure_can_be_retried_by_close(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class FailOnceHandle:
            def __init__(self, handle):
                self.handle = handle
                self.failed = False
            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('abort removal failure')
                self.handle.remove()
        class FlakyTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                return FailOnceHandle(super().register_forward_hook(*args, **kwargs))
        model = nn.Sequential(FlakyTap())
        session = ActivationCapture(model, '0')
        session.__enter__()
        with self.assertRaisesRegex(RuntimeError, 'abort removal failure'):
            session.abort()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            _ = session.value
        session.close()
        with ModelLease(model):
            pass

    def test_keyboard_interrupt_from_forward_has_a_safe_outcome(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class Interrupted(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                raise KeyboardInterrupt('forward cancellation')
        model = Interrupted()
        session = ActivationCapture(model, 'tap')
        with self.assertRaisesRegex(KeyboardInterrupt, 'forward cancellation'):
            with session:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = session.value
        if session._in_forward:
            self.assertGreater(len(model.tap._forward_hooks), 0)
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
            with self.assertRaises(RuntimeError):
                session.close()
        else:
            self.assertEqual(len(model.tap._forward_hooks), 0)
            with ModelLease(model):
                pass

    def test_caught_keyboard_interrupt_remains_unpublishable_and_abort_refuses(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class Interrupted(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                raise KeyboardInterrupt('forward cancellation')
        model = Interrupted()
        session = ActivationCapture(model, 'tap')
        with self.assertRaises(RuntimeError):
            with session:
                with self.assertRaises(KeyboardInterrupt):
                    model(torch.ones(1, 2))
                if session._in_forward:
                    with self.assertRaisesRegex(RuntimeError, 'forward may be active'):
                        session.abort()
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            _ = session.value
        if session._in_forward:
            self.assertGreater(len(model.tap._forward_hooks), 0)
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
            with self.assertRaises(RuntimeError):
                session.close()
        else:
            self.assertEqual(len(model.tap._forward_hooks), 0)
            with ModelLease(model):
                pass

    def test_skipped_forward_completion_retains_lease_deterministically(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        class SkipFinishCapture(ActivationCapture):
            def _finish_forward(self, module, inputs, output):
                return None
        class Interrupted(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                raise ValueError('forward cancellation')
        model = Interrupted()
        session = SkipFinishCapture(model, 'tap')
        with self.assertRaisesRegex(ValueError, 'forward cancellation'):
            with session:
                model(torch.ones(1, 2))
        self.assertTrue(session._in_forward)
        with self.assertRaises(RuntimeError):
            _ = session.value
        with self.assertRaises(RuntimeError):
            session.close()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()

    def test_reentrant_exit_cannot_release_lease_before_root_returns(self):
        import torch
        from looking_glass.interventions import ActivationCapture
        from looking_glass.model_ownership import ModelLease
        errors = []
        test = self
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                try:
                    session.__exit__(None, None, None)
                except RuntimeError as error:
                    errors.append(error)
                with test.assertRaises(RuntimeError):
                    ModelLease(self).acquire()
                test.assertGreater(len(self.tap._forward_hooks), 0)
                return self.tap(x)
        model = Root()
        session = ActivationCapture(model, 'tap')
        with self.assertRaisesRegex(RuntimeError, 'already invalid'):
            with session:
                model(torch.ones(1, 2))
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = session.value
        with ModelLease(model):
            pass

    def test_interrupted_close_claim_can_be_retried(self):
        import torch
        from looking_glass.interventions import ActivationCapture, _ActivationSession
        from looking_glass.model_ownership import ModelLease
        class FailOnceHandle:
            def __init__(self, handle):
                self.handle = handle
                self.failed = False
            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('one removal failure')
                self.handle.remove()
        class FlakyTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                return FailOnceHandle(super().register_forward_hook(*args, **kwargs))
        model = nn.Sequential(FlakyTap())
        capture = ActivationCapture(model, '0')
        with self.assertRaisesRegex(RuntimeError, 'one removal failure'):
            with capture:
                model(torch.ones(1, 2))
        source, start = inspect.getsourcelines(_ActivationSession.close)
        claim = next(i for i, line in enumerate(source) if line.strip().startswith('self._closing = '))
        inject_line = start + claim + 1
        def trace(frame, event, arg):
            if (frame.f_code is _ActivationSession.close.__code__ and event == 'line'
                    and frame.f_lineno == inject_line):
                sys.settrace(None)
                raise KeyboardInterrupt('close cancellation')
            return trace
        with self.assertRaisesRegex(KeyboardInterrupt, 'close cancellation'):
            sys.settrace(trace)
            try:
                capture.close()
            finally:
                sys.settrace(None)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        capture.close()
        with ModelLease(model):
            pass


if __name__ == '__main__':
    unittest.main()
