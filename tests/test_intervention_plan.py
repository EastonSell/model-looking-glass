import copy
import importlib.util
import inspect
import sys
import threading
import unittest

import torch
from torch import nn


class TwoSites(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Identity()
        self.second = nn.Identity()

    def forward(self, x):
        return self.second(self.first(x) * 2) * 3


class InterventionPlanTests(unittest.TestCase):
    def test_ordered_same_site_add_capture_and_rescue(self):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.intervention_plan'))
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        x = torch.tensor([[[1., 2.], [3., 4.]]])
        steps = [
            {'name': 'inject', 'operation': 'add', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1, 'value': torch.tensor([10., 20.])},
            {'name': 'post_add', 'operation': 'capture', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1},
            {'name': 'rescue', 'operation': 'patch', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1, 'value': torch.tensor([3., 4.])},
            {'name': 'post_patch', 'operation': 'capture', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1},
            {'name': 'downstream', 'operation': 'capture', 'module_path': 'second',
             'batch_index': 0, 'token_index': 1},
        ]
        with ActivationPlan(model, steps) as plan:
            output = model(x)
        self.assertTrue(torch.equal(plan.captures['post_add'], torch.tensor([13., 24.])))
        self.assertTrue(torch.equal(plan.captures['post_patch'], torch.tensor([3., 4.])))
        self.assertTrue(torch.equal(plan.captures['downstream'], torch.tensor([6., 8.])))
        self.assertTrue(torch.equal(output, torch.tensor([[[6., 12.], [18., 24.]]])))
        self.assertEqual([step['name'] for step in plan.metadata['steps']],
                         ['inject', 'post_add', 'rescue', 'post_patch', 'downstream'])
        self.assertEqual(plan.metadata['steps'][1]['preceding_same_site_actions'], ['inject'])
        self.assertEqual(plan.metadata['steps'][3]['preceding_same_site_actions'],
                         ['inject', 'post_add', 'rescue'])

    def test_same_site_self_patch_uses_an_earlier_captured_value(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        plan = ActivationPlan(model, [
            {'name': 'inject', 'operation': 'add', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1, 'value': torch.tensor([10., 20.])},
            {'name': 'post_injection', 'operation': 'capture', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1},
            {'name': 'self_patch', 'operation': 'patch', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1, 'source_capture': 'post_injection'},
            {'name': 'after_patch', 'operation': 'capture', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1},
        ], max_capture_bytes=16, max_source_bytes=8, max_invocations=1)
        with plan:
            result = model(torch.tensor([[[1., 2.], [3., 4.]]]))
        self.assertTrue(torch.equal(plan.captures['post_injection'], torch.tensor([13., 24.])))
        self.assertTrue(torch.equal(plan.captures['after_patch'], torch.tensor([13., 24.])))
        self.assertTrue(torch.equal(result, torch.tensor([[[6., 12.], [78., 144.]]])))
        self.assertEqual(plan.metadata['capture_bytes'], 16)
        self.assertEqual(plan.metadata['source_bytes'], 8)
        self.assertEqual(plan.metadata['steps'][2]['source_capture'], 'post_injection')

    def test_self_patch_reference_must_be_prior_and_match_the_exact_site(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        capture = {'name': 'seen', 'operation': 'capture', 'module_path': 'first',
                   'batch_index': 0, 'token_index': 0}
        patch = {'name': 'patch', 'operation': 'patch', 'module_path': 'first',
                 'batch_index': 0, 'token_index': 0, 'source_capture': 'seen'}
        cases = [[patch, capture],
                 [capture, dict(patch, module_path='second')],
                 [capture, dict(patch, token_index=1)],
                 [capture, dict(patch, value=torch.ones(2))],
                 [capture, dict(patch, operation='add')]]
        for steps in cases:
            with self.subTest(steps=steps):
                with self.assertRaises((TypeError, ValueError)):
                    ActivationPlan(model, steps)

    def test_self_patch_cuts_selected_gradient_while_null_preserves_it(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        x_null = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
        x_patch = x_null.detach().clone().requires_grad_(True)
        site = {'module_path': 'first', 'batch_index': 0, 'token_index': 1}
        null = ActivationPlan(model, [dict(site, name='control', operation='null')])
        with null:
            output_null = model(x_null)
        output_null.sum().backward()
        patch = ActivationPlan(model, [
            dict(site, name='snapshot', operation='capture'),
            dict(site, name='self_patch', operation='patch', source_capture='snapshot'),
        ])
        with patch:
            output_patch = model(x_patch)
        output_patch.sum().backward()
        self.assertTrue(torch.equal(output_null, output_patch))
        self.assertTrue(torch.equal(x_null.grad, torch.full_like(x_null, 6)))
        self.assertTrue(torch.equal(x_patch.grad, torch.tensor([[[6., 6.], [0., 0.]]])))

    def test_out_of_order_and_missing_sites_fail_without_publishing_capture(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        steps = [
            {'name': 'late', 'operation': 'capture', 'module_path': 'second'},
            {'name': 'early', 'operation': 'capture', 'module_path': 'first'},
        ]
        plan = ActivationPlan(model, steps)
        with self.assertRaises(RuntimeError):
            with plan:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with self.assertRaises(RuntimeError):
            _ = plan.metadata
        missing = ActivationPlan(model, [
            {'name': 'first', 'operation': 'capture', 'module_path': 'first'},
            {'name': 'missing', 'operation': 'capture', 'module_path': 'second', 'invocation': 1},
        ])
        with self.assertRaises(RuntimeError):
            with missing:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = missing.captures
        self.assertEqual(len(model.first._forward_hooks), 0)
        self.assertEqual(len(model.second._forward_hooks), 0)

    def test_aggregate_capture_and_source_limits(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        sources = [
            {'name': 'a', 'operation': 'add', 'module_path': 'first', 'value': torch.ones(1, 2)},
            {'name': 'b', 'operation': 'patch', 'module_path': 'second', 'value': torch.ones(1, 2)},
        ]
        with self.assertRaises(ValueError):
            ActivationPlan(model, sources, max_source_bytes=15)
        plan = ActivationPlan(model, [
            {'name': 'a', 'operation': 'capture', 'module_path': 'first'},
            {'name': 'b', 'operation': 'capture', 'module_path': 'second'},
        ], max_capture_bytes=15)
        with self.assertRaises(ValueError):
            with plan:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = plan.captures

    def test_invalid_and_ambiguous_steps_are_rejected(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        good = {'name': 'a', 'operation': 'capture', 'module_path': 'first'}
        cases = [[], [good, good],
                 [dict(good, operation='unknown')],
                 [dict(good, batch_index=0)],
                 [dict(good, extra_field=1)],
                 [dict(good, operation='patch', value='prior capture')],
                 [good, {'name': 'b', 'operation': 'capture', 'module_path': 'second'},
                  {'name': 'c', 'operation': 'capture', 'module_path': 'first'}]]
        for steps in cases:
            with self.subTest(steps=steps):
                with self.assertRaises((TypeError, ValueError)):
                    ActivationPlan(model, steps)

    def test_fixed_sources_are_detached_and_untouched_input_gradients_survive(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        value = torch.tensor([8., 9.], requires_grad=True)
        plan = ActivationPlan(model, [
            {'name': 'patch', 'operation': 'patch', 'module_path': 'first',
             'batch_index': 0, 'token_index': 1, 'value': value},
            {'name': 'observe', 'operation': 'capture', 'module_path': 'second',
             'batch_index': 0, 'token_index': 1},
        ])
        with torch.no_grad():
            value.fill_(100)
        x = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
        with plan:
            result = model(x)
        result.sum().backward()
        self.assertTrue(torch.equal(plan.captures['observe'], torch.tensor([16., 18.])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([[[6., 6.], [0., 0.]]])))
        self.assertIsNone(value.grad)

    def test_root_completion_order_and_tuple_output(self):
        from looking_glass.intervention_plan import ActivationPlan
        class TupleTap(nn.Module):
            def forward(self, x):
                return x, 'sentinel'
        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TupleTap()
            def forward(self, x):
                return self.tap(x)
        model = TupleModel()
        plan = ActivationPlan(model, [
            {'name': 'tap_zero', 'operation': 'zero', 'module_path': 'tap'},
            {'name': 'root_capture', 'operation': 'capture', 'module_path': ''},
        ])
        with plan:
            output = model(torch.tensor([[1., 2.]]))
        self.assertEqual(output[1], 'sentinel')
        self.assertTrue(torch.equal(output[0], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(plan.captures['root_capture'], torch.zeros(1, 2)))
        self.assertEqual(plan.metadata['steps'][0]['output_selector'], [0])
        self.assertEqual(plan.metadata['steps'][1]['output_selector'], [0])

    def test_other_owned_sessions_cannot_overlap(self):
        from looking_glass.interventions import ActivationCapture
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        steps = [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}]
        with ActivationCapture(model, 'first'):
            with self.assertRaises(RuntimeError):
                with ActivationPlan(model, steps):
                    pass
            model(torch.ones(1, 2))
        with ActivationPlan(model, steps):
            with self.assertRaises(RuntimeError):
                with ActivationCapture(model, 'first'):
                    pass
            model(torch.ones(1, 2))

    def test_capture_and_null_preserve_output_and_all_gradients(self):
        from looking_glass.intervention_plan import ActivationPlan
        torch.manual_seed(5)
        original = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
        baseline = copy.deepcopy(original)
        instrumented = copy.deepcopy(original)
        x0 = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        x1 = x0.detach().clone().requires_grad_(True)
        expected = baseline(x0)
        expected.sum().backward()
        plan = ActivationPlan(instrumented, [
            {'name': 'observe', 'operation': 'capture', 'module_path': '0'},
            {'name': 'control', 'operation': 'null', 'module_path': '1'},
        ])
        with plan:
            actual = instrumented(x1)
        actual.sum().backward()
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(x0.grad, x1.grad))
        for first, second in zip(baseline.parameters(), instrumented.parameters()):
            self.assertTrue(torch.equal(first.grad, second.grad))

    def test_null_can_change_layout_sensitive_downstream_behavior(self):
        from looking_glass.intervention_plan import ActivationPlan
        class LayoutSensitive(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                return self.tap(x[:, ::2, :]).stride(1)
        model = LayoutSensitive()
        x = torch.zeros(1, 4, 3)
        sliced = x[:, ::2, :]
        if sliced.clone().stride() == sliced.stride():
            self.skipTest('this torch build preserves the example layout on clone')
        untouched = model(x)
        plan = ActivationPlan(model, [{'name': 'null', 'operation': 'null', 'module_path': 'tap'}])
        with plan:
            control = model(x)
        self.assertNotEqual(untouched, control)

    def test_reentrant_target_fails_without_capture(self):
        from looking_glass.intervention_plan import ActivationPlan
        class Recursive(nn.Module):
            def forward(self, x, depth):
                return self(x, depth - 1) if depth else x
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.rec = Recursive()
            def forward(self, x):
                return self.rec(x, 1)
        model = Root()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'rec'}])
        with self.assertRaises(RuntimeError):
            with plan:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        self.assertEqual(len(model.rec._forward_hooks), 0)

    def test_tree_change_after_forward_discards_captures(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        with self.assertRaises(RuntimeError):
            with plan:
                model(torch.ones(1, 2))
                model.first = nn.Identity()
        with self.assertRaises(RuntimeError):
            _ = plan.captures

    def test_partial_registration_quarantines_model(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        class AllocationFailure(dict):
            def __setitem__(self, key, value):
                raise MemoryError('registration allocation failed')
        model = TwoSites()
        model.first._forward_hooks_always_called = AllocationFailure()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        with self.assertRaisesRegex(MemoryError, 'registration allocation failed'):
            with plan:
                pass
        self.assertGreater(len(model.first._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            plan.close()

    def test_foreign_thread_target_cannot_supply_a_planned_capture(self):
        from looking_glass.intervention_plan import ActivationPlan
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
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'tap'}])
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
            with plan:
                model(torch.tensor([[1.]]))
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = plan.captures

    def test_aliases_share_invocation_count_and_metadata_does_not_claim_route(self):
        from looking_glass.intervention_plan import ActivationPlan
        class Aliased(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Identity()
                self.alias = self.first
            def forward(self, x):
                self.alias(x)
                return self.first(x + 1)
        model = Aliased()
        plan = ActivationPlan(model, [
            {'name': 'second_call', 'operation': 'capture', 'module_path': 'alias', 'invocation': 1},
        ])
        with plan:
            model(torch.ones(1, 2))
        self.assertTrue(torch.equal(plan.captures['second_call'], torch.full((1, 2), 2.)))
        self.assertEqual(plan.metadata['steps'][0]['module_aliases'], ['first', 'alias'])
        self.assertFalse(plan.metadata['steps'][0]['call_site_known'])
        self.assertEqual(plan.metadata['steps'][0]['invocation'], 1)

    def test_max_invocations_rejects_a_repeated_target_even_after_capture(self):
        from looking_glass.intervention_plan import ActivationPlan
        class Repeated(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                return self.tap(self.tap(x))
        model = Repeated()
        plan = ActivationPlan(model, [
            {'name': 'once', 'operation': 'capture', 'module_path': 'tap'},
        ], max_invocations=1)
        with self.assertRaisesRegex(RuntimeError, 'max_invocations'):
            with plan:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        self.assertEqual(len(model.tap._forward_hooks), 0)

    def test_caught_later_keyboard_interrupt_cannot_publish_earlier_capture(self):
        from looking_glass.intervention_plan import ActivationPlan
        class FailsSecond(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt('later target failed')
                return x
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = FailsSecond()
            def forward(self, x):
                first = self.tap(x)
                try:
                    self.tap(x)
                except KeyboardInterrupt:
                    pass
                return first
        model = Root()
        plan = ActivationPlan(model, [{'name': 'first', 'operation': 'capture', 'module_path': 'tap'}])
        with self.assertRaises(RuntimeError):
            with plan:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = plan.captures

    def test_caught_target_exception_never_applies_a_later_planned_patch(self):
        from looking_glass.intervention_plan import ActivationPlan
        class FailsFirst(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError('first target failed')
                return x
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = FailsFirst()
            def forward(self, x):
                try:
                    self.tap(x)
                except ValueError:
                    pass
                return self.tap(x)
        model = Root()
        plan = ActivationPlan(model, [
            {'name': 'later', 'operation': 'patch', 'module_path': 'tap', 'invocation': 1,
             'value': torch.ones(1, 2)},
        ], max_invocations=2)
        with self.assertRaises(RuntimeError):
            with plan:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = plan.metadata

    def test_known_handle_cleanup_failure_is_retryable_and_hides_captures(self):
        from looking_glass.intervention_plan import ActivationPlan
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
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': '0'}])
        with self.assertRaisesRegex(RuntimeError, 'remove failure'):
            with plan:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        plan.close()
        with ModelLease(model):
            pass

    def test_add_overflow_invalidates_the_whole_plan(self):
        from looking_glass.intervention_plan import ActivationPlan
        model = TwoSites()
        largest = torch.finfo(torch.float32).max
        plan = ActivationPlan(model, [
            {'name': 'before', 'operation': 'capture', 'module_path': 'first'},
            {'name': 'overflow', 'operation': 'add', 'module_path': 'first',
             'value': torch.tensor([[largest]])},
        ])
        with self.assertRaises(ValueError):
            with plan:
                model(torch.tensor([[largest]]))
        with self.assertRaises(RuntimeError):
            _ = plan.captures

    def test_close_during_registration_cannot_release_the_lease(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        installed, resume = threading.Event(), threading.Event()
        errors = []
        register = plan._register
        def pause(slot, *args, **kwargs):
            register(slot, *args, **kwargs)
            if slot == 0:
                installed.set()
                if not resume.wait(5):
                    raise RuntimeError('test thread timed out')
        plan._register = pause
        def owner():
            try:
                with plan:
                    model(torch.ones(1, 2))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=owner)
        worker.start()
        self.assertTrue(installed.wait(5))
        try:
            with self.assertRaises(RuntimeError):
                plan.close()
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(plan.captures['a'], torch.ones(1, 2)))

    def test_close_before_entry_is_not_a_cancellation_api(self):
        from looking_glass.intervention_plan import ActivationPlan
        plan = ActivationPlan(TwoSites(), [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        with self.assertRaises(RuntimeError):
            plan.close()

    def test_foreign_callback_during_registration_cannot_complete_entry(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        installed, resume = threading.Event(), threading.Event()
        errors = []
        register = plan._register
        def pause(slot, *args, **kwargs):
            register(slot, *args, **kwargs)
            if slot == 2:
                installed.set()
                if not resume.wait(5):
                    raise RuntimeError('test thread timed out')
        plan._register = pause
        def owner():
            try:
                with plan:
                    model(torch.ones(1, 2))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=owner)
        worker.start()
        self.assertTrue(installed.wait(5))
        try:
            with self.assertRaises(RuntimeError):
                model.first(torch.ones(1, 2))
        finally:
            resume.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    def test_foreign_callback_during_cleanup_prevents_publication(self):
        from looking_glass.intervention_plan import ActivationPlan
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
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': '0'}])
        def owner():
            try:
                with plan:
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
            _ = plan.captures

    def test_interruption_at_entry_return_cleans_hooks(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        source, start = inspect.getsourcelines(ActivationPlan.__enter__)
        return_line = start + next(i for i, line in enumerate(source) if line.strip() == 'return self')
        def trace(frame, event, arg):
            if (frame.f_code is ActivationPlan.__enter__.__code__ and event == 'line'
                    and frame.f_lineno == return_line):
                sys.settrace(None)
                raise KeyboardInterrupt('entry cancellation')
            return trace
        with self.assertRaisesRegex(KeyboardInterrupt, 'entry cancellation'):
            sys.settrace(trace)
            try:
                with plan:
                    self.fail('context should not have entered')
            finally:
                sys.settrace(None)
        self.assertEqual(len(model.first._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    def test_interruption_after_exit_closing_transition_cleans_hooks(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        source, start = inspect.getsourcelines(ActivationPlan.__exit__)
        claim = next(i for i, line in enumerate(source) if line.strip().startswith('self._closing = '))
        inject_line = start + claim + 1
        def trace(frame, event, arg):
            if (frame.f_code is ActivationPlan.__exit__.__code__ and event == 'line'
                    and frame.f_lineno == inject_line):
                sys.settrace(None)
                raise KeyboardInterrupt('exit cancellation')
            return trace
        with self.assertRaisesRegex(KeyboardInterrupt, 'exit cancellation'):
            sys.settrace(trace)
            try:
                with plan:
                    model(torch.ones(1, 2))
            finally:
                sys.settrace(None)
        self.assertEqual(len(model.first._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    def test_failed_source_clear_does_not_strand_hooks_or_publish(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        class InterruptClear(dict):
            def clear(self):
                raise KeyboardInterrupt('source clearing cancellation')
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        plan._sources = InterruptClear(plan._sources)
        with self.assertRaisesRegex(KeyboardInterrupt, 'source clearing cancellation'):
            with plan:
                model(torch.ones(1, 2))
        self.assertEqual(len(model.first._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    def test_failed_source_clear_during_entry_preserves_error_and_cleans(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        class InterruptClear(dict):
            def clear(self):
                raise KeyboardInterrupt('clear failed')
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        plan._sources = InterruptClear(plan._sources)
        register = plan._register
        def interrupt_after_known_handle(slot, *args, **kwargs):
            register(slot, *args, **kwargs)
            if slot == 2:
                raise ValueError('entry failed')
        plan._register = interrupt_after_known_handle
        with self.assertRaisesRegex(ValueError, 'entry failed') as caught:
            with plan:
                self.fail('context should not have entered')
        self.assertIsInstance(caught.exception.__cause__, KeyboardInterrupt)
        self.assertEqual(len(model.first._forward_hooks), 0)
        with ModelLease(model):
            pass

    def test_interrupted_close_claim_can_be_retried(self):
        from looking_glass.intervention_plan import ActivationPlan
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
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': '0'}])
        with self.assertRaisesRegex(RuntimeError, 'one removal failure'):
            with plan:
                model(torch.ones(1, 2))
        source, start = inspect.getsourcelines(ActivationPlan.close)
        claim = next(i for i, line in enumerate(source) if line.strip().startswith('self._closing = '))
        inject_line = start + claim + 1
        def trace(frame, event, arg):
            if (frame.f_code is ActivationPlan.close.__code__ and event == 'line'
                    and frame.f_lineno == inject_line):
                sys.settrace(None)
                raise KeyboardInterrupt('close cancellation')
            return trace
        with self.assertRaisesRegex(KeyboardInterrupt, 'close cancellation'):
            sys.settrace(trace)
            try:
                plan.close()
            finally:
                sys.settrace(None)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        plan.close()
        with ModelLease(model):
            pass

    def test_abort_recovers_after_entry_return_event_is_interrupted(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        def trace(frame, event, arg):
            if frame.f_code is ActivationPlan.__enter__.__code__ and event == 'return' and arg is plan:
                sys.settrace(None)
                raise KeyboardInterrupt('entry return event')
            return trace
        with self.assertRaisesRegex(KeyboardInterrupt, 'entry return event'):
            sys.settrace(trace)
            try:
                with plan:
                    self.fail('context should not have entered')
            finally:
                sys.settrace(None)
        plan.abort()
        self.assertEqual(len(model.first._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    def test_abort_during_forward_rejects_and_retains_the_lease(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        errors = []
        test = self
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                try:
                    plan.abort()
                except RuntimeError as error:
                    errors.append(error)
                with test.assertRaises(RuntimeError):
                    ModelLease(self).acquire()
                test.assertGreater(len(self.tap._forward_hooks), 0)
                return self.tap(x)
        model = Root()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'tap'}])
        with self.assertRaisesRegex(RuntimeError, 'already invalid'):
            with plan:
                model(torch.ones(1, 2))
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    def test_foreign_abort_rejects_without_releasing_active_owner(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        model = TwoSites()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'first'}])
        errors = []
        def foreign():
            try:
                plan.abort()
            except BaseException as error:
                errors.append(error)
        with self.assertRaisesRegex(RuntimeError, 'already invalid'):
            with plan:
                worker = threading.Thread(target=foreign)
                worker.start()
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
                model(torch.ones(1, 2))
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    def test_abort_cleanup_failure_can_be_retried_by_close(self):
        from looking_glass.intervention_plan import ActivationPlan
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
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': '0'}])
        plan.__enter__()
        with self.assertRaisesRegex(RuntimeError, 'abort removal failure'):
            plan.abort()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        plan.close()
        with ModelLease(model):
            pass

    def test_keyboard_interrupt_forward_is_never_published(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        class Interrupted(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                raise KeyboardInterrupt('forward cancellation')
        for caught in (False, True):
            with self.subTest(caught=caught):
                model = Interrupted()
                plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'tap'}])
                if caught:
                    with self.assertRaises(RuntimeError):
                        with plan:
                            with self.assertRaises(KeyboardInterrupt):
                                model(torch.ones(1, 2))
                            if plan._in_forward:
                                with self.assertRaisesRegex(RuntimeError, 'forward may be active'):
                                    plan.abort()
                            with self.assertRaises(RuntimeError):
                                ModelLease(model).acquire()
                else:
                    with self.assertRaisesRegex(KeyboardInterrupt, 'forward cancellation'):
                        with plan:
                            model(torch.ones(1, 2))
                with self.assertRaises(RuntimeError):
                    _ = plan.captures
                if plan._in_forward:
                    self.assertGreater(len(model.tap._forward_hooks), 0)
                    with self.assertRaises(RuntimeError):
                        ModelLease(model).acquire()
                    with self.assertRaises(RuntimeError):
                        plan.close()
                else:
                    self.assertEqual(len(model.tap._forward_hooks), 0)
                    with ModelLease(model):
                        pass

    def test_skipped_forward_completion_retains_lease_deterministically(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        class SkipFinishPlan(ActivationPlan):
            def _finish_forward(self, module, inputs, output):
                return None
        class Interrupted(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                raise ValueError('forward cancellation')
        model = Interrupted()
        plan = SkipFinishPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'tap'}])
        with self.assertRaisesRegex(ValueError, 'forward cancellation'):
            with plan:
                model(torch.ones(1, 2))
        self.assertTrue(plan._in_forward)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with self.assertRaises(RuntimeError):
            plan.close()
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()

    def test_reentrant_exit_cannot_release_lease_before_root_returns(self):
        from looking_glass.intervention_plan import ActivationPlan
        from looking_glass.model_ownership import ModelLease
        errors = []
        test = self
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                try:
                    plan.__exit__(None, None, None)
                except RuntimeError as error:
                    errors.append(error)
                with test.assertRaises(RuntimeError):
                    ModelLease(self).acquire()
                test.assertGreater(len(self.tap._forward_hooks), 0)
                return self.tap(x)
        model = Root()
        plan = ActivationPlan(model, [{'name': 'a', 'operation': 'capture', 'module_path': 'tap'}])
        with self.assertRaisesRegex(RuntimeError, 'already invalid'):
            with plan:
                model(torch.ones(1, 2))
        self.assertEqual(len(errors), 1)
        with self.assertRaises(RuntimeError):
            _ = plan.captures
        with ModelLease(model):
            pass

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'transformers is optional')
    def test_random_gpt2_tuple_attention_rescue_preserves_logits(self):
        from looking_glass.intervention_plan import ActivationPlan
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=2, n_head=2,
                            use_cache=False, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            model = GPT2LMHeadModel(config).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        site = {'module_path': 'transformer.h.0.attn', 'batch_index': 0, 'token_index': 3}
        plan = ActivationPlan(model, [
            dict(site, name='clean', operation='capture'),
            dict(site, name='inject', operation='add', value=torch.full((8,), 0.5)),
            dict(site, name='corrupt', operation='capture'),
            dict(site, name='restore', operation='patch', source_capture='clean'),
            dict(site, name='restored', operation='capture'),
            {'name': 'later', 'operation': 'capture', 'module_path': 'transformer.h.1.mlp',
             'batch_index': 0, 'token_index': 3},
        ], max_invocations=1)
        with torch.no_grad():
            expected = model(ids).logits
            with plan:
                actual = model(ids).logits
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(plan.captures['clean'], plan.captures['restored']))
        self.assertTrue(torch.allclose(plan.captures['corrupt'], plan.captures['clean'] + 0.5))
        self.assertEqual(plan.metadata['steps'][0]['output_selector'], [0])
        self.assertEqual(tuple(plan.captures['later'].shape), (8,))


if __name__ == '__main__':
    unittest.main()
