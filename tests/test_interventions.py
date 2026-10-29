import importlib.util
import threading
import unittest

import torch
from torch import nn


class Reused(nn.Module):
    def __init__(self):
        super().__init__()
        self.tap = nn.Identity()

    def forward(self, x):
        self.tap(x)
        return self.tap(x + 10)


class Doubler(nn.Module):
    def __init__(self):
        super().__init__()
        self.tap = nn.Identity()

    def forward(self, x):
        return self.tap(x) * 2


class InterventionTests(unittest.TestCase):
    def test_caught_target_failure_never_mislabels_or_patches_a_later_call(self):
        from looking_glass.interventions import ActivationCapture, ActivationIntervention
        class FailsFirst(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError('first target call failed')
                return x * 2
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
        for operation in ('capture', 'patch'):
            with self.subTest(operation=operation):
                model = Root()
                session = (ActivationCapture(model, 'tap', invocation=0, max_invocations=1)
                           if operation == 'capture' else
                           ActivationIntervention(model, 'tap', invocation=0, max_invocations=1,
                                                  operation='patch', value=torch.ones(1, 2)))
                with self.assertRaises(RuntimeError):
                    with session:
                        model(torch.ones(1, 2))
                with self.assertRaises(RuntimeError):
                    _ = session.metadata
                if operation == 'capture':
                    with self.assertRaises(RuntimeError):
                        _ = session.value
                self.assertEqual(len(model.tap._forward_hooks), 0)

    def test_reentrant_target_is_not_mislabeled_by_output_completion_order(self):
        from looking_glass.interventions import ActivationCapture
        class Recursive(nn.Module):
            def forward(self, x, depth):
                return self(x, depth - 1) if depth else x
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = Recursive()
            def forward(self, x):
                return self.tap(x, 1)
        model = Root()
        session = ActivationCapture(model, 'tap')
        with self.assertRaisesRegex(RuntimeError, 'reentrant'):
            with session:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = session.value

    def test_caught_target_keyboard_interrupt_cannot_publish(self):
        from looking_glass.interventions import ActivationCapture
        class Interrupted(nn.Module):
            def forward(self, x):
                raise KeyboardInterrupt('target cancellation')
        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = Interrupted()
            def forward(self, x):
                try:
                    self.tap(x)
                except KeyboardInterrupt:
                    return x
        model = Root()
        session = ActivationCapture(model, 'tap')
        with self.assertRaises(RuntimeError):
            with session:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = session.value

    def test_root_model_can_be_targeted_by_its_empty_path(self):
        from looking_glass.interventions import ActivationCapture
        model = nn.Sequential(nn.Identity())
        with ActivationCapture(model, '') as capture:
            result = model(torch.tensor([[1., 2.]]))
        self.assertTrue(torch.equal(capture.value, result))
        self.assertEqual(capture.metadata['invocation'], 0)

    def test_null_can_change_layout_sensitive_downstream_behavior(self):
        from looking_glass.interventions import ActivationIntervention
        class LayoutSensitive(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                return self.tap(x[:, ::2, :]).stride(1)
        model = LayoutSensitive()
        x = torch.zeros(1, 4, 3)
        untouched = model(x)
        with ActivationIntervention(model, 'tap', operation='null'):
            control = model(x)
        self.assertNotEqual(untouched, control)

    def test_repeated_entry_does_not_clear_an_active_patch_or_delta(self):
        from looking_glass.interventions import ActivationIntervention
        for operation in ('patch', 'add'):
            with self.subTest(operation=operation):
                model = nn.Sequential(nn.Identity())
                session = ActivationIntervention(model, '0', operation=operation,
                                                 value=torch.tensor([[8., 9.]]))
                with session:
                    with self.assertRaisesRegex(RuntimeError, 'single-use'):
                        session.__enter__()
                    result = model(torch.tensor([[1., 2.]]))
                expected = torch.tensor([[8., 9.]]) if operation == 'patch' else torch.tensor([[9., 11.]])
                self.assertTrue(torch.equal(result, expected))

    def test_rejected_foreign_exit_preserves_source_until_owner_cleanup(self):
        from looking_glass.interventions import ActivationIntervention
        model = nn.Sequential(nn.Identity())
        session = ActivationIntervention(model, '0', operation='patch',
                                         value=torch.tensor([[8., 9.]]))
        errors = []
        def foreign_exit():
            try:
                session.__exit__(None, None, None)
            except BaseException as error:
                errors.append(error)
        with self.assertRaisesRegex(RuntimeError, 'already invalid'):
            with session:
                worker = threading.Thread(target=foreign_exit)
                worker.start()
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(errors), 1)
                self.assertIsNotNone(session._source)
                model(torch.ones(1, 2))
        self.assertIsNone(session._source)

    def test_capture_selected_token_from_second_module_invocation(self):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.interventions'),
                             'the intervention API must exist')
        from looking_glass.interventions import ActivationCapture

        model = Reused()
        x = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], requires_grad=True)
        with ActivationCapture(model, 'tap', invocation=1,
                               batch_index=1, token_index=0) as capture:
            result = model(x)
        self.assertTrue(torch.equal(capture.value, torch.tensor([15., 16.])))
        self.assertTrue(torch.equal(result, x + 10))
        self.assertEqual(capture.metadata['invocation'], 1)
        self.assertEqual(capture.metadata['module_path'], 'tap')
        with torch.no_grad():
            x.fill_(0)
        self.assertTrue(torch.equal(capture.value, torch.tensor([15., 16.])))
        self.assertFalse(capture.value.requires_grad)

    def test_clean_and_altered_whole_output_patches_and_rescue(self):
        from looking_glass.interventions import ActivationCapture, ActivationIntervention
        model = Doubler()
        clean = torch.tensor([[1., 2.]])
        altered = torch.tensor([[4., 5.]])
        with ActivationCapture(model, 'tap') as clean_capture:
            clean_result = model(clean)
        with ActivationCapture(model, 'tap') as altered_capture:
            altered_result = model(altered)
        with ActivationIntervention(model, 'tap', operation='patch', value=clean_capture.value):
            rescued_result = model(altered)
        with ActivationIntervention(model, 'tap', operation='patch', value=altered_capture.value):
            patched_result = model(clean)
        self.assertTrue(torch.equal(clean_result, torch.tensor([[2., 4.]])))
        self.assertTrue(torch.equal(altered_result, torch.tensor([[8., 10.]])))
        self.assertTrue(torch.equal(rescued_result, clean_result))
        self.assertTrue(torch.equal(patched_result, altered_result))

    def test_targeted_operations_leave_other_tokens_and_original_input_unchanged(self):
        from looking_glass.interventions import ActivationIntervention
        model = nn.Sequential(nn.Identity())
        x = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        original = x.clone()
        cases = [
            ('patch', torch.tensor([10., 20.]), [10., 20.]),
            ('zero', None, [0., 0.]),
            ('add', torch.tensor([10., 20.]), [15., 26.]),
            ('null', None, [5., 6.]),
        ]
        for operation, value, replacement in cases:
            with self.subTest(operation=operation):
                with ActivationIntervention(model, '0', operation=operation,
                                            value=value, batch_index=1, token_index=0):
                    result = model(x)
                expected = original.clone()
                expected[1, 0] = torch.tensor(replacement)
                self.assertTrue(torch.equal(result, expected))
                self.assertTrue(torch.equal(x, original))

    def test_patch_stops_gradient_only_at_replaced_token(self):
        from looking_glass.interventions import ActivationIntervention
        model = nn.Sequential(nn.Identity())
        x = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
        with ActivationIntervention(model, '0', operation='patch',
                                    value=torch.tensor([8., 9.]),
                                    batch_index=0, token_index=1):
            result = model(x)
        result.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([[[1., 1.], [0., 0.]]])))

    def test_tuple_output_preserves_non_activation_items(self):
        from looking_glass.interventions import ActivationIntervention
        class TupleTap(nn.Module):
            def forward(self, x):
                return (x, 'marker', None)
        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = TupleTap()
            def forward(self, x):
                return self.tap(x)
        model = TupleModel()
        with ActivationIntervention(model, 'tap', operation='zero'):
            result = model(torch.tensor([[1., 2.]]))
        self.assertEqual(result[1:], ('marker', None))
        self.assertTrue(torch.equal(result[0], torch.tensor([[0., 0.]])))

    def test_missing_invocation_fails_and_hooks_are_removed(self):
        from looking_glass.interventions import ActivationCapture
        model = Doubler()
        with self.assertRaises(RuntimeError):
            with ActivationCapture(model, 'tap', invocation=1):
                model(torch.ones(1, 2))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model.tap._forward_hooks), 0)
        self.assertTrue(torch.equal(model(torch.ones(1, 2)), torch.full((1, 2), 2.)))

    def test_forward_exception_is_not_masked_and_hooks_are_removed(self):
        from looking_glass.interventions import ActivationIntervention
        model = Doubler()
        with self.assertRaisesRegex(ValueError, 'original failure'):
            with ActivationIntervention(model, 'tap', operation='zero', invocation=1):
                raise ValueError('original failure')
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model.tap._forward_hooks), 0)

    def test_multiple_model_forwards_and_direct_module_calls_fail_closed(self):
        from looking_glass.interventions import ActivationCapture
        model = Doubler()
        with self.assertRaises(RuntimeError):
            with ActivationCapture(model, 'tap'):
                model(torch.ones(1, 2))
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            with ActivationCapture(model, 'tap'):
                model.tap(torch.ones(1, 2))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model.tap._forward_hooks), 0)

    def test_shape_dtype_nonfinite_and_bounds_are_rejected_without_hooks_leaking(self):
        from looking_glass.interventions import ActivationIntervention
        model = Doubler()
        x = torch.ones(1, 2)
        for bad in (torch.tensor([1., 2.]), torch.tensor([[1., 2.]], dtype=torch.float64),
                    torch.tensor([[float('nan'), 1.]])):
            with self.subTest(value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    with ActivationIntervention(model, 'tap', operation='patch', value=bad):
                        model(x)
                self.assertEqual(len(model._forward_pre_hooks), 0)
                self.assertEqual(len(model.tap._forward_hooks), 0)
        with self.assertRaises(ValueError):
            with ActivationIntervention(model, 'tap', operation='zero', max_tensor_bytes=7):
                model(x)
        with self.assertRaises((IndexError, ValueError)):
            with ActivationIntervention(model, 'tap', operation='zero',
                                        batch_index=0, token_index=1):
                model(x)
        with self.assertRaises(ValueError):
            with ActivationIntervention(model, 'tap', operation='null'):
                model(torch.tensor([[float('inf'), 1.]]))

    def test_invalid_configuration_fails_before_execution(self):
        from looking_glass.interventions import ActivationCapture, ActivationIntervention
        model = Doubler()
        configs = [
            (ActivationCapture, {'invocation': -1}),
            (ActivationCapture, {'invocation': True}),
            (ActivationCapture, {'batch_index': 0}),
            (ActivationCapture, {'max_tensor_bytes': 0}),
            (ActivationIntervention, {'operation': 'missing'}),
            (ActivationIntervention, {'operation': 'patch'}),
            (ActivationIntervention, {'operation': 'zero', 'value': torch.tensor([1.])}),
        ]
        for cls, kwargs in configs:
            with self.subTest(cls=cls, kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    cls(model, 'tap', **kwargs)
        with self.assertRaises((AttributeError, ValueError)):
            ActivationCapture(model, 'does.not.exist')


    def test_add_rejects_overflow_even_if_operands_are_finite(self):
        from looking_glass.interventions import ActivationIntervention
        model = Doubler()
        largest = torch.finfo(torch.float32).max
        with self.assertRaises(ValueError):
            with ActivationIntervention(model, 'tap', operation='add',
                                        value=torch.tensor([[largest]])):
                model(torch.tensor([[largest]]))
        self.assertEqual(len(model.tap._forward_hooks), 0)

    def test_intervention_sources_are_fixed_detached_snapshots(self):
        from looking_glass.interventions import ActivationIntervention
        model = Doubler()
        source = torch.tensor([[3., 4.]], requires_grad=True)
        session = ActivationIntervention(model, 'tap', operation='patch', value=source)
        with torch.no_grad():
            source.fill_(8.)
        x = torch.tensor([[1., 2.]], requires_grad=True)
        with session:
            y = model(x)
        y.sum().backward()
        self.assertTrue(torch.equal(y, torch.tensor([[6., 8.]])))
        self.assertIsNone(source.grad)
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_null_and_add_keep_identity_gradient(self):
        from looking_glass.interventions import ActivationIntervention
        model = Doubler()
        for operation, value in (('null', None), ('add', torch.tensor([[2., 3.]]))):
            x = torch.tensor([[1., 2.]], requires_grad=True)
            with ActivationIntervention(model, 'tap', operation=operation, value=value):
                y = model(x)
            y.sum().backward()
            self.assertTrue(torch.equal(x.grad, torch.tensor([[2., 2.]])))

    def test_direct_target_calls_after_root_forward_fail_and_hooks_are_removed(self):
        from looking_glass.interventions import ActivationCapture
        model = Doubler()
        with self.assertRaises(RuntimeError):
            with ActivationCapture(model, 'tap'):
                model(torch.ones(1, 2))
                model.tap(torch.ones(1, 2))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model.tap._forward_hooks), 0)

    def test_partial_hook_registration_failure_cleans_up(self):
        from looking_glass.interventions import ActivationCapture
        class BadTap(nn.Identity):
            def register_forward_hook(self, *args, **kwargs):
                raise RuntimeError('registration failure')
        model = Doubler()
        model.tap = BadTap()
        with self.assertRaisesRegex(RuntimeError, 'registration failure'):
            with ActivationCapture(model, 'tap'):
                pass
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)

    def test_out_of_range_token_and_concurrent_contexts_fail(self):
        from looking_glass.interventions import ActivationCapture, ActivationIntervention
        model = Doubler()
        with self.assertRaises((ValueError, IndexError)):
            with ActivationCapture(model, 'tap', batch_index=0, token_index=2):
                model(torch.ones(1, 2, 3))
        with ActivationCapture(model, 'tap'):
            with self.assertRaises(RuntimeError):
                with ActivationIntervention(model, 'tap', operation='zero'):
                    pass
            model(torch.ones(1, 2))

    def test_root_module_can_be_targeted_and_a_session_is_single_use(self):
        from looking_glass.interventions import ActivationCapture
        model = nn.Identity()
        session = ActivationCapture(model, '')
        with session:
            model(torch.tensor([1., 2.]))
        self.assertTrue(torch.equal(session.value, torch.tensor([1., 2.])))
        with self.assertRaises(RuntimeError):
            with session:
                pass

    def test_caught_model_failure_does_not_commit_capture(self):
        from looking_glass.interventions import ActivationCapture
        class Failing(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                self.tap(x)
                raise ValueError('model failed')
        model = Failing()
        capture = ActivationCapture(model, 'tap')
        with self.assertRaises(RuntimeError):
            with capture:
                try:
                    model(torch.ones(1, 2))
                except ValueError:
                    pass
        self.assertEqual(len(model.tap._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            _ = capture.value
        with self.assertRaises(RuntimeError):
            _ = capture.metadata

    def test_existing_unrelated_hooks_survive_cleanup(self):
        from looking_glass.interventions import ActivationCapture
        model = Doubler()
        called = []
        handle = model.tap.register_forward_hook(lambda module, args, output: called.append(True))
        try:
            with ActivationCapture(model, 'tap'):
                model(torch.ones(1, 2))
            self.assertEqual(len(model.tap._forward_hooks), 1)
            model(torch.ones(1, 2))
            self.assertEqual(called, [True, True])
        finally:
            handle.remove()

    def test_caught_second_forward_failure_invalidates_prior_snapshot(self):
        from looking_glass.interventions import ActivationCapture
        model = Doubler()
        capture = ActivationCapture(model, 'tap')
        with self.assertRaises(RuntimeError):
            with capture:
                model(torch.ones(1, 2))
                try:
                    model(torch.ones(1, 2))
                except RuntimeError:
                    pass
        with self.assertRaises(RuntimeError):
            _ = capture.value
        with self.assertRaises(RuntimeError):
            _ = capture.metadata

    def test_overlapping_module_trees_cannot_attach_concurrent_sessions(self):
        from looking_glass.interventions import ActivationCapture
        shared = nn.Identity()
        first = nn.Sequential(shared)
        second = nn.Sequential(shared)
        with ActivationCapture(first, '0'):
            with self.assertRaises(RuntimeError):
                with ActivationCapture(second, '0'):
                    pass
            first(torch.ones(1, 2))
        with ActivationCapture(second, '0') as capture:
            second(torch.ones(1, 2))
        self.assertTrue(torch.equal(capture.value, torch.ones(1, 2)))

    def test_invocation_limit_invalidates_capture(self):
        from looking_glass.interventions import ActivationCapture
        model = Reused()
        capture = ActivationCapture(model, 'tap', max_invocations=1)
        with self.assertRaises(RuntimeError):
            with capture:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = capture.value
        with self.assertRaises(ValueError):
            ActivationCapture(model, 'tap', invocation=1, max_invocations=1)

    def test_replacement_device_mismatch_fails(self):
        from looking_glass.interventions import ActivationIntervention
        model = Doubler()
        with self.assertRaises((TypeError, ValueError)):
            with ActivationIntervention(model, 'tap', operation='patch',
                                        value=torch.tensor([[1., 2.]])):
                model(torch.empty((1, 2), device='meta'))
        self.assertEqual(len(model.tap._forward_hooks), 0)

    def test_changed_module_path_is_rejected_before_hooks_are_registered(self):
        from looking_glass.interventions import ActivationCapture
        class ReplacedTap(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
                object.__setattr__(self, 'orphan', self.tap)
            def forward(self, x):
                self.orphan(x + 100)
                return self.tap(x)
        model = ReplacedTap()
        session = ActivationCapture(model, 'tap')
        model.tap = nn.Identity()
        with self.assertRaises(RuntimeError):
            with session:
                model(torch.ones(1, 2))
        self.assertEqual(len(model.orphan._forward_hooks), 0)
        self.assertEqual(len(model.tap._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            _ = session.value

    def test_module_tree_change_during_forward_invalidates_capture(self):
        from looking_glass.interventions import ActivationCapture
        class ChangingTree(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                result = self.tap(x)
                self.tap = nn.Identity()
                return result
        model = ChangingTree()
        session = ActivationCapture(model, 'tap')
        with self.assertRaises(RuntimeError):
            with session:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = session.value

    def test_shared_module_aliases_are_metadata_not_call_site_claims(self):
        from looking_glass.interventions import ActivationCapture
        class Aliased(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Identity()
                self.second = self.first
            def forward(self, x):
                self.second(x)
                return self.first(x + 1)
        model = Aliased()
        with ActivationCapture(model, 'first', invocation=1) as capture:
            model(torch.ones(1, 2))
        self.assertTrue(torch.equal(capture.value, torch.full((1, 2), 2.)))
        self.assertEqual(capture.metadata['module_path'], 'first')
        self.assertEqual(capture.metadata['module_aliases'], ['first', 'second'])
        self.assertFalse(capture.metadata['call_site_known'])

    def test_new_module_alias_during_forward_invalidates_capture(self):
        from looking_glass.interventions import ActivationCapture
        class AddingAlias(nn.Module):
            def __init__(self):
                super().__init__()
                self.tap = nn.Identity()
            def forward(self, x):
                result = self.tap(x)
                self.new_alias = self.tap
                return result
        model = AddingAlias()
        session = ActivationCapture(model, 'tap')
        with self.assertRaises(RuntimeError):
            with session:
                model(torch.ones(1, 2))
        with self.assertRaises(RuntimeError):
            _ = session.value

if __name__ == '__main__':
    unittest.main()
