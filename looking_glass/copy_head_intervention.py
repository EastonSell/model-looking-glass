"""Trusted CPU witness for a token-position GPT-2 head-output ablation.

The caller owns the model exclusively.  This helper does not authenticate the
model, reserve workspace, or establish the provenance of the chosen head.
"""

from __future__ import annotations

import torch


class ProjectionWitnessError(RuntimeError):
    """The projection call or its recorded intervention was not as expected."""


def _equal_float_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.int32),
                            right.contiguous().view(torch.int32)))


def assert_no_forward_hooks(model) -> None:
    """Reject competing Python forward hooks under the trusted-owner contract."""
    registry = torch.nn.modules.module
    if registry._global_forward_pre_hooks or registry._global_forward_hooks:
        raise ProjectionWitnessError("a global forward hook is installed")
    for module in model.modules():
        if module._forward_pre_hooks or module._forward_hooks:
            raise ProjectionWitnessError("model already has a forward hook")


class HeadProjectionWitness:
    """Observe or alter one head slice entering one projection call.

    ``observe`` leaves the original argument untouched. ``noop`` passes an
    identical clone; ``zero`` passes a clone with one head at a specified token
    position zeroed (the default is the final position). All three capture the
    input actually passed to the projection and the projection output. Only
    CPU float32, batch size one and the declared
    fixed shape are supported.
    """

    def __init__(self, projection, *, sequence_length: int, width: int,
                 heads: int, mode: str, head: int | None = None,
                 position: int | None = None):
        if (type(sequence_length) is not int or sequence_length < 1
                or type(width) is not int or width < 1
                or type(heads) is not int or heads < 1 or width % heads
                or mode not in ("observe", "noop", "zero")):
            raise ValueError("invalid witness dimensions or mode")
        if mode == "zero":
            if type(head) is not int or not 0 <= head < heads:
                raise ValueError("zero mode needs an in-range head")
            if position is not None and (type(position) is not int or not 0 <= position < sequence_length):
                raise ValueError("zero mode position must be an in-range integer")
        elif head is not None or position is not None:
            raise ValueError("only zero mode takes a head or position")
        self.projection = projection
        self.sequence_length = sequence_length
        self.width = width
        self.heads = heads
        self.mode = mode
        self.head = head
        self.position = sequence_length - 1 if position is None else position
        self.before = None
        self.after = None
        self.output = None
        self.pre_calls = 0
        self.post_calls = 0
        self._handles = []

    def _tensor(self, value):
        if (type(value) is not torch.Tensor or value.device.type != "cpu"
                or value.dtype != torch.float32
                or tuple(value.shape) != (1, self.sequence_length, self.width)
                or not value.is_contiguous() or not bool(torch.isfinite(value).all())):
            raise ProjectionWitnessError("unexpected projection tensor")
        return value

    def _pre(self, module, arguments):
        if (module is not self.projection or self.pre_calls != 0
                or type(arguments) is not tuple or len(arguments) != 1):
            raise ProjectionWitnessError("unexpected projection pre-hook call")
        value = self._tensor(arguments[0])
        self.pre_calls = 1
        self.before = value.detach().clone().contiguous()
        if self.mode == "observe":
            delivered = value
        else:
            delivered = value.clone().contiguous()
            if self.mode == "zero":
                size = self.width // self.heads
                delivered[0, self.position, self.head * size:(self.head + 1) * size] = 0.0
        self.after = delivered.detach().clone().contiguous()
        expected = self.before.clone()
        if self.mode == "zero":
            size = self.width // self.heads
            expected[0, self.position, self.head * size:(self.head + 1) * size] = 0.0
        if not _equal_float_bits(self.after, expected):
            raise ProjectionWitnessError("projection intervention changed other bits")
        if self.mode != "observe":
            return (delivered,)
        return None

    def _post(self, module, arguments, output):
        if (module is not self.projection or self.pre_calls != 1 or self.post_calls != 0
                or type(arguments) is not tuple or len(arguments) != 1):
            raise ProjectionWitnessError("unexpected projection post-hook call")
        actual_input = self._tensor(arguments[0])
        if self.after is None or not _equal_float_bits(actual_input, self.after):
            raise ProjectionWitnessError("projection received different input bits")
        self.output = self._tensor(output).detach().clone().contiguous()
        self.post_calls = 1
        return None

    def __enter__(self):
        if self._handles or self.projection._forward_pre_hooks or self.projection._forward_hooks:
            raise ProjectionWitnessError("projection hook ownership is not exclusive")
        try:
            self._handles.append(self.projection.register_forward_pre_hook(self._pre))
            self._handles.append(self.projection.register_forward_hook(self._post))
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            raise
        return self

    def __exit__(self, error_type, error, traceback):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if self.projection._forward_pre_hooks or self.projection._forward_hooks:
            raise ProjectionWitnessError("projection hook cleanup is uncertain")
        if error_type is None and (self.pre_calls != 1 or self.post_calls != 1):
            raise ProjectionWitnessError("projection was not called exactly once")
        return False
