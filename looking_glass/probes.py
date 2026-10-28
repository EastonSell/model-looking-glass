"""Identity observation points and two equivalent value-reporting channels.

The hook always returns None. Recording never replaces the main-path tensor.
Sessions are local, single-use and single-threaded. They are not a general
profiler and do not support concurrent use on the same model.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def _validate_tensor(a):
    if not isinstance(a, torch.Tensor) or not a.is_floating_point():
        raise TypeError("capture requires one floating-point tensor")
    if not bool(torch.isfinite(a).all()):
        raise ValueError("capture requires finite activations")


def gradient_readout(a: torch.Tensor) -> torch.Tensor:
    """Return a detached copy via grad_q sum(stopgrad(a)*q), where q=0.

    The separate q is ephemeral, not a model parameter, and never optimized.
    Local contexts also permit this isolated computation during inference mode.
    """
    _validate_tensor(a)
    with torch.inference_mode(False), torch.enable_grad():
        saved = a.detach().clone()
        q = torch.zeros_like(saved, requires_grad=True)
        report = (saved * q).sum()
        (value,) = torch.autograd.grad(report, q, create_graph=False)
        return value.detach().clone()


class TapPoint(nn.Identity):
    """A genuine identity operation with an explicit instrumentation label."""
    def __init__(self, node_id: str, group: str = "dense"):
        super().__init__()
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node_id must be a nonempty string")
        if group not in ("sparse", "dense"):
            raise ValueError("group must be sparse or dense")
        self.node_id = node_id
        self.group = group


@dataclass
class TraceEvent:
    run_id: str
    node_id: str
    sequence: int
    occurrence: int
    shape: tuple
    dtype: str
    device: str
    value: Optional[torch.Tensor]

    @property
    def tensor_bytes(self) -> int:
        return 0 if self.value is None else self.value.numel() * self.value.element_size()

    def metadata(self) -> dict:
        return {"run_id": self.run_id, "node_id": self.node_id,
                "sequence": self.sequence, "occurrence": self.occurrence,
                "shape": list(self.shape), "dtype": self.dtype,
                "device": self.device, "tensor_bytes": self.tensor_bytes}


class TraceSession:
    """Record labeled identity outputs, without replacing values or gradients.

    max_events is an explicit bound; exceeding it raises rather than silently
    dropping events. Tensor bytes count persistent value snapshots only.
    """
    def __init__(self, model: nn.Module, mode="direct", placement="dense",
                 run_id="run", max_events=10000):
        if mode not in ("direct", "gradient", "id"):
            raise ValueError("mode must be direct, gradient, or id")
        if placement not in ("sparse", "dense"):
            raise ValueError("placement must be sparse or dense")
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        self.model, self.mode, self.placement = model, mode, placement
        self.run_id, self.max_events = run_id, max_events
        self.events = []
        self._handles = []
        self._counts = defaultdict(int)
        self._used = False

    @property
    def tensor_bytes(self):
        return sum(e.tensor_bytes for e in self.events)

    def __enter__(self):
        if self._used:
            raise RuntimeError("TraceSession is single-use")
        self._used = True
        points = [m for m in self.model.modules() if isinstance(m, TapPoint)
                  and (self.placement == "dense" or m.group == "sparse")]
        labels = [m.node_id for m in points]
        if len(set(labels)) != len(labels):
            raise ValueError("distinct tap modules must have distinct node IDs")
        try:
            for module in points:
                self._handles.append(module.register_forward_hook(self._capture))
        except BaseException:
            self.close()
            raise
        return self

    def _capture(self, module, inputs, output):
        if len(self.events) >= self.max_events:
            raise RuntimeError("trace capacity exceeded; increase max_events explicitly")
        if not isinstance(output, torch.Tensor):
            raise TypeError("tap output must be one tensor")
        if self.mode == "id":
            value = None
        elif self.mode == "gradient":
            value = gradient_readout(output)
        else:
            _validate_tensor(output)
            value = output.detach().clone()
        count = self._counts[module.node_id]
        self._counts[module.node_id] += 1
        self.events.append(TraceEvent(self.run_id, module.node_id,
            len(self.events), count, tuple(output.shape), str(output.dtype),
            str(output.device), value))
        return None

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
