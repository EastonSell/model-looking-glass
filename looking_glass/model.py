"""Small deterministic residual network with an explicit controlled route."""
import math

import torch
from torch import nn

from .probes import TapPoint


class ResidualBlock(nn.Module):
    def __init__(self, width, index, identical_branches=False):
        super().__init__()
        self.branches = nn.ModuleList([nn.Linear(width, width, device="cpu", dtype=torch.float32), nn.Linear(width, width, device="cpu", dtype=torch.float32)])
        if identical_branches:
            self.branches[1].load_state_dict(self.branches[0].state_dict())
        self.branch_taps = nn.ModuleList([
            TapPoint(f"block.{index}.branch.{route}") for route in (0, 1)])
        self.activation_tap = TapPoint(f"block.{index}.activation")
        self.merge_tap = TapPoint(f"block.{index}.merge")
        self.output_tap = TapPoint(f"block.{index}.output", "sparse")
        self.projection = nn.Linear(width, width, device="cpu", dtype=torch.float32)

    @staticmethod
    def observe(point, value, perturb):
        if perturb is not None and perturb[0] == point.node_id:
            value = value.clone()
            value.reshape(-1)[0] = value.reshape(-1)[0] + perturb[1]
        return point(value)

    def forward(self, x, route, perturb):
        branch = self.observe(self.branch_taps[route], self.branches[route](x), perturb)
        activated = self.observe(self.activation_tap, torch.tanh(branch), perturb)
        merged = self.observe(self.merge_tap, x + self.projection(activated), perturb)
        return self.observe(self.output_tap, merged, perturb)


class TinyResidual(nn.Module):
    """Route is external ground truth, not inferred from captured activations."""
    def __init__(self, width, depth=4, identical_branches=False):
        super().__init__()
        if type(width) is not int or width <= 0 or type(depth) is not int or depth <= 0:
            raise ValueError("width and depth must be positive integers")
        self.width, self.depth = width, depth
        self.blocks = nn.ModuleList([
            ResidualBlock(width, i, identical_branches) for i in range(depth)])
        self.head = nn.Linear(width, 2, device="cpu", dtype=torch.float32)

    def active_nodes(self, route=0):
        return [label for i in range(self.depth) for label in (
            f"block.{i}.branch.{route}", f"block.{i}.activation",
            f"block.{i}.merge", f"block.{i}.output")]

    def forward(self, x, route=0, perturb=None):
        if type(route) is not int or route not in (0, 1):
            raise ValueError("route must be integer 0 or 1")
        if x.ndim != 2 or x.shape[1] != self.width or x.shape[0] == 0:
            raise ValueError("input must have shape (positive batch, width)")
        if perturb is not None:
            if (not isinstance(perturb, tuple) or len(perturb) != 2
                    or perturb[0] not in self.active_nodes(route)
                    or not isinstance(perturb[1], (float, int))
                    or not math.isfinite(perturb[1])):
                raise ValueError("perturb must identify an active node and finite delta")
        for block in self.blocks:
            x = block(x, route, perturb)
        return self.head(x)
