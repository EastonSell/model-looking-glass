import copy
import unittest

import torch

from looking_glass.model import TinyResidual
from looking_glass.probes import TraceSession


def training_step(model, x):
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    y = model(x, route=1)
    loss = y.square().mean()
    loss.backward()
    gradients = {k: p.grad.detach().clone() if p.grad is not None else None for k, p in model.named_parameters()}
    xgrad = x.grad.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.step()
    return y.detach(), loss.detach(), gradients, xgrad, copy.deepcopy(model.state_dict())


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)

    def test_all_channels_preserve_training_step(self):
        original = TinyResidual(width=4, depth=2)
        x = torch.randn(3, 4)
        baseline = training_step(copy.deepcopy(original), x)
        for channel in ('direct', 'gradient', 'id'):
            for placement in ('sparse', 'dense'):
                model = copy.deepcopy(original)
                with TraceSession(model, mode=channel, placement=placement):
                    actual = training_step(model, x)
                for i in (0, 1, 3):
                    self.assertTrue(torch.equal(baseline[i], actual[i]), (channel, placement, i))
                for i in (2, 4):
                    for k, v in baseline[i].items():
                        if v is None:
                            self.assertIsNone(actual[i][k])
                        else:
                            self.assertTrue(torch.equal(v, actual[i][k]), (channel, placement, k))

    def test_identical_branches_need_metadata(self):
        model = TinyResidual(width=4, depth=2, identical_branches=True)
        x = torch.randn(3, 4)
        with TraceSession(model, mode='direct', placement='dense') as a:
            ya = model(x, route=0)
        with TraceSession(model, mode='direct', placement='dense') as b:
            yb = model(x, route=1)
        self.assertTrue(torch.equal(ya, yb))
        self.assertNotEqual([e.node_id for e in a.events], [e.node_id for e in b.events])
        self.assertEqual(len(a.events), len(b.events))
        self.assertTrue(all(torch.equal(e.value, f.value) for e, f in zip(a.events, b.events)))

    def test_full_model_capture_in_inference_mode(self):
        model = TinyResidual(width=4,depth=2)
        with torch.inference_mode():
            x = torch.randn(3,4)
            expected = model(x)
            with TraceSession(model,mode='gradient') as trace:
                actual = model(x)
            self.assertTrue(torch.equal(expected,actual))
            self.assertEqual(len(trace.events),8)

    def test_capture_channels_agree(self):
        model = TinyResidual(width=4, depth=2)
        x = torch.randn(3, 4)
        with TraceSession(model, mode='direct') as a:
            model(x)
        with TraceSession(model, mode='gradient') as b:
            model(x)
        self.assertEqual([e.node_id for e in a.events], [e.node_id for e in b.events])
        self.assertTrue(all(torch.equal(e.value, f.value) for e, f in zip(a.events, b.events)))

    def test_invalid_model_inputs(self):
        for width, depth in ((0, 1), (4, 0), (True, 1)):
            with self.assertRaises(ValueError):
                TinyResidual(width=width, depth=depth)
        model = TinyResidual(width=4)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4), route=2)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4), perturb=('unknown', .125))

    def test_injection_is_before_named_point_and_does_not_change_input(self):
        model = TinyResidual(width=4, depth=2)
        x = torch.randn(2, 4)
        saved = x.clone()
        label = 'block.0.branch.0'
        with TraceSession(model) as a:
            model(x, route=0)
        with TraceSession(model) as b:
            model(x, route=0, perturb=(label, .125))
        first = next(e.node_id for e, f in zip(a.events, b.events) if not torch.equal(e.value, f.value))
        self.assertEqual(first, label)
        self.assertTrue(torch.equal(x, saved))


if __name__ == '__main__':
    unittest.main()
