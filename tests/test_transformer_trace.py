"""CPU contract tests for the native GPT-2 trace adapter; no downloads."""
import copy
import importlib.util
import inspect
import sys
import threading
import unittest
from collections import OrderedDict
from unittest.mock import patch

import torch

if importlib.util.find_spec('transformers') is not None:
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
else:
    GPT2Config = GPT2LMHeadModel = GPT2Model = None


class AdapterValidationTests(unittest.TestCase):
    def test_cleanup_diagnostic_preserves_primary_on_python_310_style_exception(self):
        from looking_glass.transformer_trace import _note_cleanup
        class NoAddNote(RuntimeError):
            add_note = None
        primary = NoAddNote('primary failure')
        original_cause = ValueError('original cause')
        primary.__cause__ = original_cause
        _note_cleanup(primary, 'cleanup also failed')
        self.assertEqual(str(primary), 'primary failure')
        self.assertIn('cleanup also failed', str(primary.__cause__))
        self.assertIs(primary.__cause__.__cause__, original_cause)

    def test_event_metadata_keeps_source_and_snapshot_coordinates_distinct(self):
        from looking_glass import transformer_trace
        self.assertTrue(hasattr(transformer_trace, 'GPT2TraceEvent'),
                        'trace event metadata is not implemented')
        value = torch.tensor([[[2., 3.]]])
        event = transformer_trace.GPT2TraceEvent(
            run_id='unit', sequence=0, forward_index=3, module_path='transformer.h.1.attn',
            module_kind='attn', layer_index=1, invocation=0, output_path=(0,),
            batch_positions=(1,), token_positions=(2,), shape=(2, 4, 2),
            dtype='torch.float32', device='cuda:0', capture_mode='direct', value=value)
        data = event.metadata()
        self.assertEqual(data['identity']['run_id'], 'unit')
        self.assertEqual(data['batch_positions'], [1])
        self.assertEqual(data['token_positions'], [2])
        self.assertEqual(data['shape'], [2, 4, 2])
        self.assertEqual(data['saved_shape'], [1, 1, 2])
        self.assertEqual(data['device'], 'cuda:0')
        self.assertEqual(data['saved_device'], 'cpu')
        self.assertEqual(data['output_path'], [0])
        self.assertEqual(data['generation']['stage'], 'unknown')
        self.assertIsNone(data['generation']['cache_position'])
        self.assertEqual(event.tensor_bytes, 8)
        from looking_glass.trace_schema import TraceEvent
        self.assertTrue(hasattr(event, 'to_schema'), 'schema conversion is not implemented')
        schema_event = event.to_schema()
        self.assertIsInstance(schema_event, TraceEvent)
        self.assertEqual(schema_event.to_dict(), data)
        self.assertEqual(TraceEvent.from_json(schema_event.to_json()), schema_event)

    def test_invalid_mode_is_rejected_before_any_hooks_are_installed(self):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.transformer_trace'),
                             'native GPT-2 trace adapter missing')
        from looking_glass.transformer_trace import GPT2TraceSession
        model = torch.nn.Identity()
        with self.assertRaises(ValueError):
            GPT2TraceSession(model, mode='not-a-capture-mode')
        self.assertEqual(len(model._forward_hooks), 0)

    def test_non_gpt2_models_are_rejected_without_installing_hooks(self):
        from looking_glass.transformer_trace import GPT2TraceSession
        model = torch.nn.Identity()
        try:
            GPT2TraceSession(model, run_id='unit')
        except (TypeError, ValueError):
            pass
        except NotImplementedError:
            self.fail('non-GPT-2 model validation is not implemented')
        else:
            self.fail('non-GPT-2 model was accepted')
        self.assertEqual(len(model._forward_hooks), 0)


@unittest.skipIf(GPT2Config is None, "optional transformers package not installed")
class GPT2TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small CPU-only fixture; avoid changing seeds or thread counts at import.
        torch.set_num_threads(1)

    def make_model(self, base=False):
        torch.manual_seed(71)
        config = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16,
                            n_embd=8, n_layer=2, n_head=2,
                            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                            use_cache=False)
        config._attn_implementation = 'eager'
        model = GPT2Model(config) if base else GPT2LMHeadModel(config)
        return model.eval()

    def session(self, model, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('looking_glass.transformer_trace'),
                             'native GPT-2 trace adapter missing')
        from looking_glass.transformer_trace import GPT2TraceSession
        return GPT2TraceSession(model, run_id='unit', **kwargs)

    def test_boundary_values_and_coordinates_match_independent_hooks(self):
        model = self.make_model()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        observed = {}
        handles = []
        for layer in range(2):
            for kind, module in (("attn", model.transformer.h[layer].attn),
                                 ("mlp", model.transformer.h[layer].mlp),
                                 ("block", model.transformer.h[layer])):
                def capture(_module, _args, output, key=(layer, kind)):
                    selected = output[0] if isinstance(output, tuple) else output
                    observed[key] = (selected.detach().clone(), (0,) if isinstance(output, tuple) else ())
                handles.append(module.register_forward_hook(capture))
        try:
            baseline = model(ids, use_cache=False).logits.detach()
            observed.clear()
            with self.session(model, samples=(1, 0), tokens=(2, 0)) as trace:
                actual = model(ids, use_cache=False).logits.detach()
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(torch.equal(baseline, actual))
        self.assertEqual([(e.layer_index, e.module_kind) for e in trace.events],
                         [(0, 'attn'), (0, 'mlp'), (0, 'block'),
                          (1, 'attn'), (1, 'mlp'), (1, 'block')])
        self.assertEqual(trace.tensor_bytes, 6 * 2 * 2 * 8 * 4)
        for index, event in enumerate(trace.events):
            self.assertEqual(event.sequence, index)
            self.assertEqual(event.forward_index, 0)
            self.assertEqual(event.invocation, 0)
            self.assertEqual(event.batch_positions, (1, 0))
            self.assertEqual(event.token_positions, (2, 0))
            self.assertEqual(event.shape, (2, 3, 8))
            self.assertEqual(event.saved_shape, (2, 2, 8))
            self.assertEqual(event.module_path,
                             f"transformer.h.{event.layer_index}" +
                             ('' if event.module_kind == 'block' else '.' + event.module_kind))
            original, expected_path = observed[(event.layer_index, event.module_kind)]
            self.assertEqual(event.output_path, expected_path)
            expected = original[[1, 0]][:, [2, 0]]
            self.assertTrue(torch.equal(event.value, expected))
            self.assertFalse(event.value.requires_grad)
            meta = event.metadata()
            self.assertEqual(meta['batch_axis'], 0)
            self.assertEqual(meta['token_axis'], 1)

    def test_base_model_paths_and_repeated_calls_have_forward_indices(self):
        model = self.make_model(base=True)
        with self.session(model, layers=(1,), targets=('block',), mode='id') as trace:
            model(torch.tensor([[1, 2]]))
            model(torch.tensor([[3]]))
        self.assertEqual([e.module_path for e in trace.events], ['h.1', 'h.1'])
        self.assertEqual([e.forward_index for e in trace.events], [0, 1])
        self.assertEqual([e.invocation for e in trace.events], [0, 0])
        self.assertEqual([e.token_positions for e in trace.events], [(0, 1), (0,)])
        self.assertTrue(all(e.value is None for e in trace.events))
        self.assertEqual(trace.tensor_bytes, 0)

    def test_invocations_count_module_entries_within_one_forward(self):
        model = self.make_model()
        block, mlp = model.transformer.h[0], model.transformer.h[0].mlp
        original_block = block.forward
        def call_twice(*args, **kwargs):
            mlp(args[0])
            return original_block(*args, **kwargs)
        with patch.object(block, 'forward', side_effect=call_twice):
            with self.session(model, layers=(0,), targets=('mlp',), mode='id') as trace:
                model(torch.tensor([[1, 2]]))
        self.assertTrue(trace.successful)
        self.assertEqual([event.invocation for event in trace.events], [0, 1])

    def test_caught_target_failure_cannot_mislabel_a_later_invocation(self):
        model = self.make_model()
        block, mlp = model.transformer.h[0], model.transformer.h[0].mlp
        original_block, original_mlp = block.forward, mlp.forward
        calls = []
        def fail_first(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise ValueError('injected target failure')
            return original_mlp(*args, **kwargs)
        def recover_then_continue(*args, **kwargs):
            try:
                mlp(args[0])
            except ValueError:
                pass
            return original_block(*args, **kwargs)
        trace = self.session(model, layers=(0,), targets=('mlp',), mode='id')
        with patch.object(block, 'forward', side_effect=recover_then_continue), \
             patch.object(mlp, 'forward', side_effect=fail_first):
            with trace:
                with self.assertRaises(RuntimeError):
                    model(torch.tensor([[1, 2]]))
        self.assertFalse(trace.successful)
        self.assertEqual(trace.events, [])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in model.modules()))

    def test_caught_earlier_target_hook_failure_cannot_publish_a_trace(self):
        model = self.make_model()
        block, mlp = model.transformer.h[0], model.transformer.h[0].mlp
        original_block = block.forward
        calls = []
        def fail_first_hook(_module, _args, _output):
            calls.append(True)
            if len(calls) == 1:
                raise ValueError('injected target hook failure')
        def recover_then_continue(*args, **kwargs):
            try:
                mlp(args[0])
            except ValueError:
                pass
            return original_block(*args, **kwargs)
        handle = mlp.register_forward_hook(fail_first_hook)
        trace = self.session(model, layers=(0,), targets=('mlp',), mode='id')
        try:
            with patch.object(block, 'forward', side_effect=recover_then_continue):
                with trace:
                    with self.assertRaises(RuntimeError):
                        model(torch.tensor([[1, 2]]))
            self.assertFalse(trace.successful)
            self.assertEqual(trace.events, [])
        finally:
            handle.remove()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in model.modules()))

    def test_separate_sessions_require_explicit_run_identity(self):
        from looking_glass.transformer_trace import GPT2TraceSession
        from looking_glass.trace_schema import validate_trace
        model = self.make_model()
        with self.assertRaises((TypeError, ValueError)):
            GPT2TraceSession(model)
        events = []
        for run_id in ('first-run', 'second-run'):
            with GPT2TraceSession(model, run_id=run_id, targets=('block',), layers=(0,)) as trace:
                model(torch.tensor([[1, 2]]))
            events.extend(event.to_schema() for event in trace.events)
        validate_trace(events)

    def test_direct_and_gradient_preserve_training_and_match_snapshots(self):
        original = self.make_model()
        base_embeddings = torch.randn(1, 3, 8)
        results = []
        for mode in (None, 'direct', 'gradient', 'id'):
            model = copy.deepcopy(original)
            inputs = base_embeddings.clone().requires_grad_()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad(set_to_none=True)
            if mode is None:
                output = model(inputs_embeds=inputs, use_cache=False).logits
                events = None
            else:
                with self.session(model, mode=mode, targets=('block',)) as trace:
                    output = model(inputs_embeds=inputs, use_cache=False).logits
                events = trace.events
            loss = output.square().mean()
            loss.backward()
            result = (output.detach().clone(), loss.detach().clone(), inputs.grad.clone(),
                      [None if p.grad is None else p.grad.clone() for p in model.parameters()])
            optimizer.step()
            results.append((result, copy.deepcopy(model.state_dict()), events))
        expected, expected_state, _ = results[0]
        for actual, state, _ in results[1:]:
            for first, second in zip(expected[:3], actual[:3]):
                self.assertTrue(torch.equal(first, second))
            for first, second in zip(expected[3], actual[3]):
                self.assertTrue(first is second if first is None else torch.equal(first, second))
            self.assertTrue(all(torch.equal(expected_state[k], state[k]) for k in state))
        for direct, gradient in zip(results[1][2], results[2][2]):
            self.assertTrue(torch.equal(direct.value, gradient.value))

    def test_native_lm_loss_and_updates_are_unchanged_with_dropout(self):
        original = self.make_model().train()
        for module in original.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.2
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        results = []
        for mode in (None, 'direct', 'gradient', 'id'):
            model = copy.deepcopy(original)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            torch.manual_seed(125)
            if mode is None:
                output = model(ids, labels=ids, use_cache=False)
            else:
                with self.session(model, mode=mode) as trace:
                    output = model(ids, labels=ids, use_cache=False)
                self.assertTrue(trace.successful)
            output.loss.backward()
            gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
            logits, loss = output.logits.detach().clone(), output.loss.detach().clone()
            optimizer.step()
            results.append((logits, loss, gradients, copy.deepcopy(model.state_dict())))
        expected_logits, expected_loss, expected_grads, expected_state = results[0]
        for logits, loss, grads, state in results[1:]:
            self.assertTrue(torch.equal(expected_logits, logits))
            self.assertTrue(torch.equal(expected_loss, loss))
            for first, second in zip(expected_grads, grads):
                self.assertTrue(first is second if first is None else torch.equal(first, second))
            self.assertTrue(all(torch.equal(expected_state[k], state[k]) for k in state))

    def test_limits_reject_before_extra_snapshot_and_remove_hooks(self):
        model = self.make_model()
        for kwargs in ({'max_events': 1}, {'max_trace_bytes': 31}):
            trace = self.session(model, targets=('block',), **kwargs)
            with self.assertRaises(RuntimeError):
                with trace:
                    model(torch.tensor([[1]]))
            self.assertEqual(trace.tensor_bytes, 0)
            self.assertEqual(trace.events, [])
            self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        # The model remains usable after capture failure.
        self.assertEqual(tuple(model(torch.tensor([[1]])).logits.shape), (1, 1, 32))

    def test_bad_positions_raise_and_cleanup(self):
        model = self.make_model()
        for kwargs in ({'tokens': (-1,)}, {'samples': (0, 0)}, {'layers': (2,)},
                       {'tokens': (True,)}, {'targets': ('unknown',)}, {'max_trace_bytes': -1}):
            with self.assertRaises((TypeError, ValueError)):
                self.session(model, **kwargs)
        for kwargs in ({'tokens': (2,)}, {'samples': (1,)}):
            with self.assertRaises((IndexError, ValueError)):
                with self.session(model, **kwargs):
                    model(torch.tensor([[1, 2]]))
            self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_prefill_and_cached_decode_do_not_claim_absolute_positions(self):
        model = self.make_model()
        baseline_prefill = model(torch.tensor([[1, 2, 3]]), use_cache=True,
                                 output_attentions=True)
        baseline_prefill_logits = baseline_prefill.logits.detach().clone()
        baseline_decode = model(torch.tensor([[4]]), past_key_values=baseline_prefill.past_key_values,
                                use_cache=True, output_attentions=True)
        with self.session(model, layers=(0,), targets=('attn', 'block')) as trace:
            prefill = model(torch.tensor([[1, 2, 3]]), use_cache=True,
                            output_attentions=True)
            decode = model(torch.tensor([[4]]), past_key_values=prefill.past_key_values,
                           use_cache=True, output_attentions=True)
        self.assertTrue(torch.equal(baseline_prefill_logits, prefill.logits))
        self.assertTrue(torch.equal(baseline_decode.logits, decode.logits))
        for expected_layer, actual_layer in zip(baseline_decode.past_key_values,
                                                decode.past_key_values):
            for expected, actual in zip(expected_layer, actual_layer):
                self.assertTrue(torch.equal(expected, actual))
        self.assertEqual([e.forward_index for e in trace.events], [0, 0, 1, 1])
        self.assertEqual([e.token_positions for e in trace.events],
                         [(0, 1, 2), (0, 1, 2), (0,), (0,)])
        self.assertEqual([e.invocation for e in trace.events], [0, 0, 0, 0])
        self.assertTrue(all(e.metadata()['generation']['stage'] == 'unknown' for e in trace.events))

    def test_nonfinite_and_model_exception_release_hooks_and_snapshots(self):
        model = self.make_model()
        trace = self.session(model, targets=('block',))
        invalid_embeddings = torch.full((1, 2, 8), float('nan'))
        with self.assertRaises(ValueError):
            with trace:
                model(inputs_embeds=invalid_embeddings)
        self.assertEqual(trace.events, [])
        self.assertEqual(trace.tensor_bytes, 0)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        trace = self.session(model, targets=('block',))
        with self.assertRaises(RuntimeError):
            with trace:
                model(torch.tensor([[1]]))
                raise RuntimeError('test failure after a forward')
        self.assertEqual(trace.events, [])
        self.assertEqual(trace.tensor_bytes, 0)

    def test_caught_capture_failure_invalidates_the_entire_session(self):
        model = self.make_model()
        trace = self.session(model, targets=('block',), max_events=1)
        with trace:
            with self.assertRaises(RuntimeError):
                model(torch.tensor([[1]]))
            self.assertEqual(trace.events, [])
            self.assertEqual(trace.tensor_bytes, 0)
            with self.assertRaises(RuntimeError):
                model(torch.tensor([[2]]))
        self.assertFalse(trace.successful)
        self.assertEqual(trace.events, [])
        self.assertEqual(trace.tensor_bytes, 0)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_caught_top_level_forward_hook_failure_invalidates_session(self):
        model = self.make_model()
        def fail_after_model(_module, _args, _output):
            raise ValueError('foreign top-level hook failed')
        handle = model.register_forward_hook(fail_after_model)
        trace = self.session(model, layers=(0,), targets=('block',))
        try:
            with trace:
                with self.assertRaises(ValueError):
                    model(torch.tensor([[1]]))
                with self.assertRaises(RuntimeError):
                    model(torch.tensor([[2]]))
            self.assertFalse(trace.successful)
            self.assertEqual(trace.events, [])
        finally:
            handle.remove()

    def test_failed_cached_forward_can_partially_mutate_caller_cache(self):
        model = self.make_model()
        prefix = torch.tensor([[1, 2, 3]])
        cache = model(prefix, use_cache=True).past_key_values
        before = [key.shape[-2] for key, _value in cache]
        with self.assertRaises(RuntimeError):
            with self.session(model, targets=('attn', 'block'), max_events=1):
                model(torch.tensor([[4]]), past_key_values=cache, use_cache=True)
        after = [key.shape[-2] for key, _value in cache]
        self.assertEqual(before, [3, 3])
        self.assertEqual(after, [4, 3])
        # Re-prefill, rather than retrying with the partially updated cache.
        clean_cache = model(prefix, use_cache=True).past_key_values
        clean = model(torch.tensor([[4]]), past_key_values=clean_cache, use_cache=True)
        self.assertEqual([key.shape[-2] for key, _value in clean.past_key_values], [4, 4])

    def test_overlapping_sessions_are_rejected(self):
        model = self.make_model()
        with self.session(model):
            with self.assertRaises(RuntimeError):
                with self.session(model):
                    pass
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_intervention_and_trace_cannot_overlap_in_either_order(self):
        from looking_glass.interventions import ActivationIntervention
        model = self.make_model()
        intervention = lambda: ActivationIntervention(
            model, 'transformer.h.0.attn', operation='zero', batch_index=0, token_index=0)
        with self.session(model):
            with self.assertRaises(RuntimeError):
                with intervention():
                    pass
        with intervention():
            with self.assertRaises(RuntimeError):
                with self.session(model):
                    pass
            model(torch.tensor([[1, 2]]))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_target_replacement_between_construction_and_entry_is_rejected(self):
        model = self.make_model()
        trace = self.session(model, targets=('mlp',), layers=(0,))
        model.transformer.h[0].mlp = torch.nn.Identity()
        with self.assertRaises(RuntimeError):
            with trace:
                pass
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_shared_layer_objects_are_rejected_before_hooks_can_mislabel_them(self):
        model = self.make_model()
        model.transformer.h[1] = model.transformer.h[0]
        with self.assertRaises(ValueError):
            self.session(model)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_replaced_child_module_is_not_mislabeled_as_standard_gpt2(self):
        model = self.make_model()
        model.transformer.h[0].mlp = torch.nn.Identity()
        with self.assertRaises((TypeError, ValueError)):
            self.session(model)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_snapshot_does_not_alias_the_original_module_output(self):
        model = self.make_model()
        observed = []
        def keep_source(_module, _args, output):
            observed.append(output[0] if isinstance(output, tuple) else output)
        handle = model.transformer.h[0].register_forward_hook(keep_source)
        try:
            with self.session(model, layers=(0,), targets=('block',), mode='direct') as trace:
                model(torch.tensor([[1, 2]]))
            expected = trace.events[0].value.clone()
            observed[0].detach().fill_(42)
            self.assertTrue(torch.equal(expected, trace.events[0].value))
        finally:
            handle.remove()

    def test_registration_exception_quarantines_model_even_without_an_observed_hook(self):
        model = self.make_model()
        failing = model.transformer.h[1]
        with patch.object(failing, 'register_forward_hook', side_effect=RuntimeError('cannot register')):
            with self.assertRaisesRegex(RuntimeError, 'cannot register'):
                with self.session(model):
                    pass
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        with self.assertRaises(RuntimeError):
            with self.session(model):
                pass
        fresh_model = self.make_model()
        with self.session(fresh_model) as trace:
            fresh_model(torch.tensor([[1]]))
        self.assertTrue(trace.successful)

    def test_bookkeeping_failure_removes_just_registered_hook(self):
        class FailAppend(list):
            def __init__(self, fail_on):
                super().__init__()
                self.fail_on = fail_on
                self.calls = 0
            def append(self, item):
                self.calls += 1
                if self.calls == self.fail_on:
                    raise MemoryError('injected handle bookkeeping failure')
                super().append(item)
        for fail_on in (1, 4):
            model = self.make_model()
            trace = self.session(model)
            trace._handles = FailAppend(fail_on)
            with self.assertRaises(MemoryError):
                with trace:
                    pass
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                                for m in model.modules()))
            with self.session(model) as fresh:
                model(torch.tensor([[1]]))
            self.assertTrue(fresh.successful)

    def test_failure_inside_pytorch_registration_quarantines_the_model(self):
        class FailAuxiliaryMap(OrderedDict):
            def __setitem__(self, key, value):
                raise MemoryError('injected torch registration failure')
        model = self.make_model()
        original = model._forward_pre_hooks_with_kwargs
        model._forward_pre_hooks_with_kwargs = FailAuxiliaryMap()
        trace = self.session(model)
        try:
            with self.assertRaises(MemoryError):
                with trace:
                    pass
            self.assertTrue(model._forward_pre_hooks)
            with self.assertRaises(RuntimeError):
                with self.session(model):
                    pass
        finally:
            model._forward_pre_hooks_with_kwargs = original
        with self.assertRaises(RuntimeError):
            trace.close()
        fresh_model = self.make_model()
        with self.session(fresh_model) as fresh:
            fresh_model(torch.tensor([[1]]))
        self.assertTrue(fresh.successful)

    def test_malformed_registration_handle_quarantines_model(self):
        model = self.make_model()
        original = model.register_forward_pre_hook
        def install_but_return_none(*args, **kwargs):
            original(*args, **kwargs)
            return None
        trace = self.session(model)
        with patch.object(model, 'register_forward_pre_hook', side_effect=install_but_return_none):
            with self.assertRaises(RuntimeError):
                with trace:
                    pass
        with self.assertRaises(RuntimeError):
            with self.session(model):
                pass

    def test_failed_handle_removal_keeps_ownership_until_retry(self):
        class FailOnce:
            def __init__(self, handle):
                self.handle = handle
                self.failed = False
            def remove(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError('injected cleanup failure')
                self.handle.remove()
        model = self.make_model()
        trace = self.session(model)
        with self.assertRaisesRegex(RuntimeError, 'cleanup failure'):
            with trace:
                model(torch.tensor([[1]]))
                trace._handles[1] = FailOnce(trace._handles[1])
        self.assertFalse(trace.successful)
        with self.assertRaises(RuntimeError):
            with self.session(model):
                pass
        trace.close()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                            for m in model.modules()))
        with self.session(model) as fresh:
            model(torch.tensor([[1]]))
        self.assertTrue(fresh.successful)

    def test_interrupted_lease_release_is_retried_without_claiming_success(self):
        from looking_glass.model_ownership import ModelLease
        for interrupt_after_release in (False, True):
            model = self.make_model()
            trace = self.session(model)
            original = trace._lease.release
            calls = []
            def interrupted_release():
                calls.append(True)
                if len(calls) == 1:
                    if interrupt_after_release:
                        original()
                    raise KeyboardInterrupt('injected release interruption')
                original()
            with patch.object(trace._lease, 'release', side_effect=interrupted_release):
                with self.assertRaises(KeyboardInterrupt):
                    with trace:
                        model(torch.tensor([[1]]))
            self.assertFalse(trace.successful)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
            with ModelLease(model):
                pass

    def test_reentrant_model_forward_fails_without_misattributed_events(self):
        model = self.make_model()
        def reenter(_module, _args, _output):
            model(torch.tensor([[2]]))
        handle = model.transformer.h[0].mlp.register_forward_hook(reenter)
        trace = self.session(model, targets=('block',))
        try:
            with self.assertRaises(RuntimeError):
                with trace:
                    model(torch.tensor([[1]]))
            self.assertFalse(trace.successful)
            self.assertEqual(trace.events, [])
        finally:
            handle.remove()
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_foreign_thread_child_call_poisoning_cannot_publish_a_trace(self):
        model = self.make_model()
        owner_thread = threading.get_ident()
        foreign_errors = []
        def inject_foreign_call(_module, _args, _output):
            if threading.get_ident() != owner_thread:
                return
            def foreign():
                try:
                    model.transformer.h[0].mlp(torch.zeros(1, 1, 8))
                except BaseException as error:
                    foreign_errors.append(error)
            thread = threading.Thread(target=foreign)
            thread.start()
            thread.join()
        handle = model.transformer.h[0].mlp.register_forward_hook(inject_foreign_call)
        trace = self.session(model, targets=('mlp',), layers=(0,))
        try:
            with self.assertRaises(RuntimeError):
                with trace:
                    model(torch.tensor([[1]]))
            self.assertEqual(len(foreign_errors), 1)
            self.assertFalse(trace.successful)
            self.assertEqual(trace.events, [])
        finally:
            handle.remove()

    def test_foreign_close_cannot_release_lease_mid_registration(self):
        from looking_glass.model_ownership import ModelLease
        model = self.make_model()
        trace = self.session(model)
        installed, resume = threading.Event(), threading.Event()
        errors = []
        original = model.register_forward_pre_hook
        def slow_register(*args, **kwargs):
            handle = original(*args, **kwargs)
            installed.set()
            if not resume.wait(5):
                raise TimeoutError('registration test timed out')
            return handle
        def owner():
            try:
                with trace:
                    model(torch.tensor([[1]]))
            except BaseException as error:
                errors.append(error)
        with patch.object(model, 'register_forward_pre_hook', side_effect=slow_register):
            thread = threading.Thread(target=owner)
            thread.start()
            try:
                self.assertTrue(installed.wait(5))
                with self.assertRaises(RuntimeError):
                    trace.close()
                with self.assertRaises(RuntimeError):
                    ModelLease(model).acquire()
            finally:
                resume.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(trace.successful)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_same_thread_close_during_active_forward_cannot_release_lease(self):
        from looking_glass.model_ownership import ModelLease
        model = self.make_model()
        trace = self.session(model, targets=('block',))
        close_errors = []
        def try_close(_module, _args, _output):
            try:
                trace.close()
            except RuntimeError as error:
                close_errors.append(error)
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
        handle = model.transformer.h[0].mlp.register_forward_hook(try_close)
        try:
            with trace:
                model(torch.tensor([[1]]))
            self.assertEqual(len(close_errors), 1)
            self.assertTrue(trace.successful)
        finally:
            handle.remove()

    def test_reentrant_close_during_registration_cannot_release_lease(self):
        from looking_glass.model_ownership import ModelLease
        model = self.make_model()
        trace = self.session(model)
        original = model.register_forward_pre_hook
        def try_close(*args, **kwargs):
            with self.assertRaises(RuntimeError):
                trace.close()
            with self.assertRaises(RuntimeError):
                ModelLease(model).acquire()
            return original(*args, **kwargs)
        with patch.object(model, 'register_forward_pre_hook', side_effect=try_close):
            with trace:
                model(torch.tensor([[1]]))
        self.assertTrue(trace.successful)

    def test_interruption_after_lease_acquisition_cleans_up(self):
        from looking_glass.transformer_trace import GPT2TraceSession
        model = self.make_model()
        trace = self.session(model)
        source, start = inspect.getsourcelines(GPT2TraceSession.__enter__)
        target = start + next(i for i, line in enumerate(source) if 'self._lease.verify_tree()' in line)
        def cancel(frame, event, _arg):
            if frame.f_code is GPT2TraceSession.__enter__.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('injected entry interruption')
            return cancel
        try:
            sys.settrace(cancel)
            with self.assertRaises(KeyboardInterrupt):
                trace.__enter__()
        finally:
            sys.settrace(None)
        with self.session(model) as fresh:
            model(torch.tensor([[1]]))
        self.assertTrue(fresh.successful)

    def test_interruption_before_returned_handle_is_recorded_quarantines(self):
        from looking_glass.model_ownership import ModelLease
        from looking_glass.transformer_trace import GPT2TraceSession
        model = self.make_model()
        trace = self.session(model)
        source, start = inspect.getsourcelines(GPT2TraceSession._register)
        target = start + next(i for i, line in enumerate(source) if 'self._untracked_handle = handle' in line)
        def cancel(frame, event, _arg):
            if frame.f_code is GPT2TraceSession._register.__code__ and event == 'line' and frame.f_lineno == target:
                raise KeyboardInterrupt('injected registration interruption')
            return cancel
        try:
            sys.settrace(cancel)
            with self.assertRaises(KeyboardInterrupt):
                trace.__enter__()
        finally:
            sys.settrace(None)
        with self.assertRaises(RuntimeError):
            ModelLease(model).acquire()

    def test_interrupted_root_prehook_cannot_publish_prior_events(self):
        from looking_glass.transformer_trace import GPT2TraceSession
        source, start = inspect.getsourcelines(GPT2TraceSession._begin_forward)
        targets = [start + next(i for i, line in enumerate(source) if 'frames.append(False)' in line),
                   start + next(i for i, line in enumerate(source) if 'frames[-1] = True' in line)]
        for target in targets:
            model = self.make_model()
            trace = self.session(model, targets=('block',))
            with trace:
                model(torch.tensor([[1]]))
                def cancel(frame, event, _arg):
                    if frame.f_code is GPT2TraceSession._begin_forward.__code__ and event == 'line' and frame.f_lineno == target:
                        raise KeyboardInterrupt('injected prehook interruption')
                    return cancel
                try:
                    sys.settrace(cancel)
                    with self.assertRaises(KeyboardInterrupt):
                        model(torch.tensor([[2]]))
                finally:
                    sys.settrace(None)
                self.assertEqual(trace.events, [])
                with self.assertRaises(RuntimeError):
                    model(torch.tensor([[3]]))
            self.assertFalse(trace.successful)

    def test_interruption_during_end_hook_cannot_publish_prior_events(self):
        from looking_glass.model_ownership import ModelLease
        from looking_glass.transformer_trace import GPT2TraceSession
        source, start = inspect.getsourcelines(GPT2TraceSession._end_forward)
        target = start + next(i for i, line in enumerate(source)
                              if 'if output is None or not self._normal_completed' in line)
        for attempt_another_forward in (False, True):
            model = self.make_model()
            trace = self.session(model, targets=('block',), layers=(0,))
            with trace:
                model(torch.tensor([[1]]))
                def cancel(frame, event, _arg):
                    if frame.f_code is GPT2TraceSession._end_forward.__code__ and event == 'line' and frame.f_lineno == target:
                        raise KeyboardInterrupt('injected end-hook interruption')
                    return cancel
                try:
                    sys.settrace(cancel)
                    with self.assertRaises(KeyboardInterrupt):
                        model(torch.tensor([[2]]))
                finally:
                    sys.settrace(None)
                if attempt_another_forward:
                    with self.assertRaises(RuntimeError):
                        model(torch.tensor([[3]]))
            self.assertFalse(trace.successful)
            self.assertEqual(trace.events, [])
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                                for m in model.modules()))
            with ModelLease(model):
                pass

    def test_unwound_keyboard_interrupt_invalidates_and_releases_trace(self):
        from looking_glass.model_ownership import ModelLease
        for location in ('model', 'capture', 'completion'):
            for caught_inside in (False, True):
                with self.subTest(location=location, caught_inside=caught_inside):
                    model = self.make_model()
                    trace = self.session(model, targets=('block',), layers=(0,))
                    patches = []
                    if location == 'model':
                        patches.append(patch.object(model.transformer.h[0].mlp, 'forward',
                                                    side_effect=KeyboardInterrupt('model interrupted')))
                    elif location == 'capture':
                        original_factory = trace._make_hook
                        def interrupted_factory(*args):
                            original = original_factory(*args)
                            def interrupted_capture(*hook_args):
                                original(*hook_args)
                                raise KeyboardInterrupt('capture interrupted')
                            return interrupted_capture
                        patches.append(patch.object(trace, '_make_hook', side_effect=interrupted_factory))
                    else:
                        original_marker = trace._mark_forward_complete
                        def interrupted_marker(*args):
                            original_marker(*args)
                            raise KeyboardInterrupt('completion interrupted')
                        patches.append(patch.object(trace, '_mark_forward_complete', side_effect=interrupted_marker))
                    with patches[0]:
                        if caught_inside:
                            with trace:
                                with self.assertRaises(KeyboardInterrupt):
                                    model(torch.tensor([[1]]))
                        else:
                            with self.assertRaises(KeyboardInterrupt):
                                with trace:
                                    model(torch.tensor([[1]]))
                    self.assertFalse(trace.successful)
                    self.assertEqual(trace.events, [])
                    self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                                        for m in model.modules()))
                    with ModelLease(model):
                        pass


if __name__ == '__main__':
    unittest.main()
