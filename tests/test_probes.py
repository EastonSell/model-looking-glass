import unittest

import torch
from torch import nn

from looking_glass.probes import TapPoint, TraceSession, gradient_readout


class ProbeTests(unittest.TestCase):
    def test_gradient_readout_exact_and_detached(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            a = torch.tensor([[1.25, -2.0], [0.0, 0.5]], dtype=dtype, requires_grad=True)
            b = gradient_readout(a)
            self.assertTrue(torch.equal(a, b))
            self.assertFalse(b.requires_grad)
            self.assertIsNone(a.grad)
            self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_gradient_readout_under_disabled_grad(self):
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                a = torch.tensor([1.0, 2.0])
                b = gradient_readout(a)
                self.assertTrue(torch.equal(a, b))
            self.assertFalse(torch.is_grad_enabled() is False)

    def test_nonfinite_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError):
                gradient_readout(torch.tensor([value]))

    def test_hooks_preserve_output_and_input_gradient(self):
        for mode in ('direct', 'gradient', 'id'):
            model = nn.Sequential(TapPoint('entry', 'sparse'), nn.Linear(3, 3), TapPoint('exit', 'sparse'))
            x = torch.tensor([[0.25, 0.5, -1.0]], requires_grad=True)
            y = model(x)
            expected = torch.autograd.grad(y.sum(), x)[0]
            with TraceSession(model, mode=mode, placement='dense', run_id='test') as session:
                result = model(x)
                actual = torch.autograd.grad(result.sum(), x)[0]
            self.assertTrue(torch.equal(y, result))
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual([e.node_id for e in session.events], ['entry', 'exit'])
            self.assertEqual([e.sequence for e in session.events], [0, 1])
            self.assertTrue(all(e.run_id == 'test' for e in session.events))
            self.assertTrue(all(e.value is None for e in session.events) if mode == 'id' else all(not e.value.requires_grad for e in session.events))
            self.assertEqual(session.tensor_bytes, 0 if mode == 'id' else 24)

    def test_snapshot_does_not_alias(self):
        model = TapPoint('point', 'sparse')
        x = torch.tensor([1.0])
        for mode in ('direct', 'gradient'):
            x.fill_(1)
            with TraceSession(model, mode=mode) as s:
                model(x)
            x.fill_(8)
            self.assertEqual(s.events[0].value.item(), 1)

    def test_reused_point_has_occurrence_ids(self):
        p = TapPoint('shared', 'sparse')
        with TraceSession(p) as s:
            p(torch.ones(2)); p(torch.zeros(2))
        self.assertEqual([e.occurrence for e in s.events], [0, 1])

    def test_sparse_filters_group(self):
        model = nn.Sequential(TapPoint('a', 'dense'), TapPoint('b', 'sparse'))
        with TraceSession(model, placement='sparse') as s:
            model(torch.ones(2))
        self.assertEqual([e.node_id for e in s.events], ['b'])

    def test_invalid_configuration_rejected(self):
        p = TapPoint('point', 'sparse')
        for kw in ({'mode':'oops'}, {'placement':'oops'}, {'max_events':0}, {'max_events':True}):
            with self.assertRaises(ValueError):
                TraceSession(p, **kw)

    def test_duplicate_node_ids_rejected(self):
        model = nn.Sequential(TapPoint('a','sparse'), TapPoint('a','sparse'))
        with self.assertRaises(ValueError):
            with TraceSession(model):
                pass
        self.assertTrue(all(len(m._forward_hooks) == 0 for m in model.modules()))

    def test_overflow_and_exception_cleanup(self):
        p = TapPoint('point', 'sparse')
        with self.assertRaises(RuntimeError):
            with TraceSession(p, max_events=1):
                p(torch.ones(1)); p(torch.ones(1))
        self.assertEqual(len(p._forward_hooks), 0)
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with TraceSession(p):
                raise RuntimeError('deliberate')
        self.assertEqual(len(p._forward_hooks), 0)

    def test_partial_registration_failure_removes_earlier_hooks(self):
        class RegistrationFailure(TapPoint):
            def register_forward_hook(self, *args, **kwargs):
                raise RuntimeError("registration failure")
        first = TapPoint('first', 'sparse')
        model = nn.Sequential(first, RegistrationFailure('second', 'sparse'))
        with self.assertRaisesRegex(RuntimeError, 'registration failure'):
            with TraceSession(model):
                pass
        self.assertEqual(len(first._forward_hooks), 0)

    def test_session_cannot_be_entered_twice(self):
        p = TapPoint('point', 'sparse')
        session = TraceSession(p)
        with session:
            with self.assertRaises(RuntimeError):
                session.__enter__()
        with self.assertRaises(RuntimeError):
            session.__enter__()


if __name__ == '__main__':
    unittest.main()
