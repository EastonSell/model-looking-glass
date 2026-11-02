"""Bit-defined matched-dose block-output interventions for the fixed study."""
from __future__ import annotations

import math
import struct

import torch

from .copy_head_factorial import f32_bits_to_f64, f64_to_f32_bits
from .year_transfer import CONDITIONS, ROLES

WIDTH = 768
ROW_BYTES = WIDTH * 4
MIN_DOSE = 2.0**-20
ACTIONS = ('self', 'year_final', 'year_at_zero', 'rotated_final')


def _bits(raw):
    if type(raw) is not bytes or len(raw) % 4:
        raise ValueError('a complete little-endian float32 byte string is required')
    return tuple(item[0] for item in struct.iter_unpack('<I', raw))


def _values(raw):
    return tuple(f32_bits_to_f64(value) for value in _bits(raw))


def _norm(values):
    value = math.sqrt(math.fsum(x*x for x in values))
    if not math.isfinite(value):
        raise ValueError('nonfinite displacement norm')
    return value


def _delivery(source, displacement, dose):
    before = _values(source)
    if len(before) != WIDTH or len(displacement) != WIDTH:
        raise ValueError('unexpected vector width')
    after_bits = tuple(f64_to_f32_bits(value + delta) for value, delta in zip(before, displacement))
    after = struct.pack(f'<{WIDTH}I', *after_bits)
    realized = _norm(a-b for a, b in zip(_values(after), before))
    if abs(realized - dose) > 1e-5 * dose:
        raise ValueError('realized float32 dose is outside the fixed tolerance')
    return {'before': source, 'after': after, 'requested': dose, 'achieved': realized,
            'changed_bits': sum(a != b for a, b in zip(_bits(source), after_bits))}


def plan_family(noops):
    """Plan all four actions from six saved low/high no-op block outputs."""
    if type(noops) is not dict or set(noops) != set(CONDITIONS):
        raise ValueError('expected three conditions')
    vectors = {}
    norms = {}
    for condition in CONDITIONS:
        pair = noops[condition]
        if type(pair) is not dict or set(pair) != set(ROLES):
            raise ValueError('expected both year roles')
        low, high = pair['low'], pair['high']
        if (type(low) is not bytes or type(high) is not bytes or len(low) != len(high)
                or len(low) % ROW_BYTES or not 2 <= len(low)//ROW_BYTES <= 32):
            raise ValueError('unexpected bounded block-output shape')
        _values(low)
        _values(high)
        if low[:ROW_BYTES] != high[:ROW_BYTES]:
            raise ValueError('position-zero no-op vectors differ')
        vectors[condition] = {role: _values(pair[role][-ROW_BYTES:]) for role in ROLES}
        delta = [b-a for a, b in zip(vectors[condition]['low'], vectors[condition]['high'])]
        norms[condition] = _norm(delta)
    dose = 0.5 * min(norms.values())
    if not math.isfinite(dose) or dose <= MIN_DOSE:
        raise ValueError('the common input dose is nonfinite or too small')
    plans = {}
    for condition in CONDITIONS:
        plans[condition] = {}
        for role in ROLES:
            donor = 'high' if role == 'low' else 'low'
            raw = noops[condition][role]
            final = len(raw)//ROW_BYTES - 1
            delta = [b-a for a, b in zip(vectors[condition][role], vectors[condition][donor])]
            scale = dose / norms[condition]
            displacement = tuple(scale*x for x in delta)
            rotated = tuple(displacement[(i-1) % WIDTH] for i in range(WIDTH))
            self_raw = raw[-ROW_BYTES:]
            self_action = {'position': final, 'before': self_raw, 'after': self_raw,
                           'requested': 0.0, 'achieved': 0.0, 'changed_bits': 0}
            final_action = {'position': final, **_delivery(self_raw, displacement, dose)}
            zero_action = {'position': 0, **_delivery(raw[:ROW_BYTES], displacement, dose)}
            rotated_action = {'position': final, **_delivery(self_raw, rotated, dose)}
            plans[condition][role] = {'self': self_action, 'year_final': final_action,
                                      'year_at_zero': zero_action, 'rotated_final': rotated_action}
    return {'dose': dose, 'norms': norms, 'plans': plans}


class BlockOutputWitness:
    """Capture one block output and optionally deliver a precomputed vector."""
    def __init__(self, block, *, sequence_length, expected_before=None, plan=None):
        if (type(sequence_length) is not int or not 2 <= sequence_length <= 32
                or (expected_before is not None and
                    (type(expected_before) is not bytes or len(expected_before) != sequence_length*ROW_BYTES))
                or (plan is not None and expected_before is None)):
            raise ValueError('invalid block witness configuration')
        self.block = block
        self.sequence_length = sequence_length
        self.expected_before = expected_before
        self.plan = plan
        self.before = self.after = None
        self.observed = None
        self.calls = 0
        self.handle = None

    def __enter__(self):
        if self.handle is not None or self.block._forward_hooks:
            raise ValueError('block already has a forward hook')
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, arguments, output):
        if module is not self.block or self.calls != 0 or type(output) is not tuple or len(output) != 1:
            raise ValueError('unexpected block call or output')
        value = output[0]
        if (type(value) is not torch.Tensor or value.device.type != 'cpu' or value.dtype != torch.float32
                or tuple(value.shape) != (1, self.sequence_length, WIDTH) or not value.is_contiguous()):
            raise ValueError('unexpected block-output tensor')
        self.calls += 1
        self.before = value.detach().numpy().tobytes()
        _values(self.before)
        if self.expected_before is not None and self.before != self.expected_before:
            raise ValueError('pre-action block output differs from the saved no-op')
        if self.plan is None:
            self.after = self.before
            return None
        if (type(self.plan) is not dict or type(self.plan.get('position')) is not int
                or not 0 <= self.plan['position'] < self.sequence_length
                or type(self.plan.get('before')) is not bytes or type(self.plan.get('after')) is not bytes
                or len(self.plan['before']) != ROW_BYTES or len(self.plan['after']) != ROW_BYTES):
            raise ValueError('invalid planned replacement')
        start = self.plan['position'] * ROW_BYTES
        if self.before[start:start+ROW_BYTES] != self.plan['before']:
            raise ValueError('pre-action site differs from the planned source')
        expected = bytearray(self.before)
        expected[start:start+ROW_BYTES] = self.plan['after']
        replacement = torch.frombuffer(bytearray(self.plan['after']), dtype=torch.float32)
        patched = value.clone().contiguous()
        patched[0, self.plan['position']].copy_(replacement)
        self.after = patched.detach().numpy().tobytes()
        if self.after != bytes(expected):
            raise ValueError('actual block replacement differs from the planned bits')
        return (patched,)

    def verify_complete(self):
        if self.calls != 1 or self.before is None or self.after is None:
            raise ValueError('block output was not observed exactly once')
        if self.plan is not None:
            start = self.plan['position'] * ROW_BYTES
            before = self.before[start:start+ROW_BYTES]
            after = self.after[start:start+ROW_BYTES]
            achieved = _norm(a-b for a, b in zip(_values(after), _values(before)))
            changed = sum(a != b for a, b in zip(_bits(after), _bits(before)))
            requested = self.plan['requested']
            if (abs(achieved-requested) > 1e-5*requested
                    or achieved != self.plan['achieved'] or changed != self.plan['changed_bits']):
                raise ValueError('observed same-invocation dose differs from the plan')
            self.observed = {'achieved': achieved, 'changed_bits': changed}
