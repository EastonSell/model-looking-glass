"""Ordered two-position GPT-2 interventions for a public artificial control."""
import ctypes
import math
import os
import struct

import torch
from looking_glass.intervention_plan import ActivationPlan
from looking_glass.trace_schema import GenerationContext
from looking_glass.transformer_trace import GPT2TraceSession

def ensure_round_to_nearest():
    """Fail closed unless the pinned CPU thread uses round-to-nearest-even.

    The Windows control-word check covers all three alternate modes; the
    arithmetic probes also check the binary64 operations used by this module.
    This assumes the interpreter and native libraries do not change the
    thread's mode concurrently between checks.
    """
    if os.name == 'nt':
        control = ctypes.CDLL('msvcrt')._controlfp
        control.argtypes = [ctypes.c_uint, ctypes.c_uint]
        control.restype = ctypes.c_uint
        if control(0, 0) & 0x300:
            raise RuntimeError('floating-point rounding mode is not nearest-even')
    else:
        try:
            getround = ctypes.CDLL(None).fegetround
            getround.argtypes = []
            getround.restype = ctypes.c_int
            mode = getround()
        except (AttributeError, OSError) as error:
            raise RuntimeError('cannot inspect floating-point rounding mode') from error
        if mode != 0:
            raise RuntimeError('floating-point rounding mode is not nearest-even')
    one = 1.0
    half = float.fromhex('0x1p-53')
    next_one = float.fromhex('0x1.0000000000001p+0')
    if (one + half != one
            or next_one + half != float.fromhex('0x1.0000000000002p+0')
            or one - float.fromhex('0x1p-54') != one
            or next_one * 1.5 != float.fromhex('0x1.8000000000002p+0')
            or one / 10.0 != float.fromhex('0x1.999999999999ap-4')
            or math.sqrt(2.0) != float.fromhex('0x1.6a09e667f3bcdp+0')):
        raise RuntimeError('binary64 rounding probe failed')


def _round_shift_even(value, shift):
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1))


def f64_to_f32_bits(value):
    """Convert a finite binary64 to IEEE binary32 RNE by integer arithmetic."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('expected finite binary64')
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    sign = (bits >> 32) & 0x80000000
    exponent, fraction = (bits >> 52) & 0x7ff, bits & ((1 << 52) - 1)
    if exponent == 0:
        return sign | _round_shift_even(fraction, 925)
    power, significand = exponent - 1023, (1 << 52) | fraction
    if power >= -126:
        rounded = _round_shift_even(significand, 29)
        if rounded == 1 << 24:
            rounded >>= 1
            power += 1
        if power > 127:
            return sign | 0x7f800000
        return sign | ((power + 127) << 23) | (rounded & 0x7fffff)
    return sign | _round_shift_even(significand, -(power + 97))


def f32_bits_to_f64(bits):
    sign, exponent, fraction = bits >> 31, (bits >> 23) & 0xff, bits & 0x7fffff
    if exponent == 0xff:
        raise ValueError('nonfinite float32')
    significand = fraction if exponent == 0 else (1 << 23) | fraction
    value = math.ldexp(float(significand), -149 if exponent == 0 else exponent - 150)
    return -value if sign else value


def _signed32(bits):
    return bits if bits < 0x80000000 else bits - 0x100000000


def _vector_bits(vector):
    return [x & 0xffffffff for x in vector.detach().contiguous().view(torch.int32).tolist()]


LAYERS = (0, 1, 2, 3, 4)
MAPPINGS = {
    1: ((2, 't', 'downstream'), (0, 'f', 'distractor'), (1, 't', 'source'), (0, 't', 'upstream')),
    2: ((2, 't', 'source'), (1, 't', 'upstream'), (0, 'f', 'distractor'), (3, 't', 'downstream')),
    3: ((2, 't', 'upstream'), (4, 't', 'downstream'), (3, 't', 'source'), (0, 'f', 'distractor')),
}


def bitwise_equal(a, b):
    return (a.dtype == b.dtype == torch.float32 and a.shape == b.shape
            and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def candidate_map(layer):
    if layer not in MAPPINGS:
        raise ValueError('target layer is not fixed')
    return {f'Q{i}': {'layer': block, 'position': pos, 'role': role}
            for i, (block, pos, role) in enumerate(MAPPINGS[layer], 1)}


def _positioned(layer, token):
    return {'module_path': f'transformer.h.{layer}.mlp', 'batch_index': 0,
            'token_index': token}


def _full(layer):
    return {'module_path': f'transformer.h.{layer}.mlp'}


def _saved_logits(output, t, f):
    return torch.stack((output.logits[0, t], output.logits[0, f])).detach().clone()


def _partial(output, tokens, callback, error):
    if output is not None and callback is not None:
        try:
            callback(_saved_logits(output, tokens.shape[1] - 2, tokens.shape[1] - 1), error)
        except BaseException as nested:
            if hasattr(error, 'add_note'):
                error.add_note('partial returned-logit save failed: ' + type(nested).__name__)


def observe_clean(model, tokens, run_id, on_partial=None):
    ensure_round_to_nearest()
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 3:
        raise ValueError('expected one input with at least three tokens')
    t, f = tokens.shape[1] - 2, tokens.shape[1] - 1
    session = GPT2TraceSession(model, run_id=run_id, targets=('mlp',), layers=LAYERS,
        samples=(0,), tokens=(t, f), mode='direct', max_events=5, max_trace_bytes=2_000_000,
        capture_device='cpu', forward_context=GenerationContext(stage='forward', use_cache=False))
    output = None
    try:
        with session:
            output = model(input_ids=tokens, use_cache=False)
        ensure_round_to_nearest()
        if not session.successful or len(session.events) != len(LAYERS):
            raise RuntimeError('incomplete clean two-position trace')
        vectors = {}
        for event in session.events:
            vectors[(event.layer_index, 't')] = event.value[0, 0].detach().clone()
            vectors[(event.layer_index, 'f')] = event.value[0, 1].detach().clone()
        if set(vectors) != {(layer, pos) for layer in LAYERS for pos in ('t', 'f')}:
            raise RuntimeError('clean trace coordinates differ')
        return {'logits': _saved_logits(output, t, f), 'vectors': vectors,
                'events': [event.metadata() for event in session.events]}
    except BaseException as error:
        _partial(output, tokens, on_partial, error)
        raise


def rotate_displacement(base, donor):
    ensure_round_to_nearest()
    if (base.dtype != torch.float32 or donor.dtype != torch.float32
            or base.device.type != 'cpu' or donor.device.type != 'cpu'
            or base.ndim != 1 or donor.shape != base.shape or not base.numel()):
        raise ValueError('rotation requires finite equal CPU float32 vectors')
    b = [f32_bits_to_f64(x) for x in _vector_bits(base)]
    c = [f32_bits_to_f64(x) for x in _vector_bits(donor)]
    d = [x - y for x, y in zip(c, b)]
    intended = [d[(i - 1) % len(d)] for i in range(len(d))]
    if not all(math.isfinite(x) for x in d):
        raise ValueError('nonfinite rotation')
    zbits = [f64_to_f32_bits(b[i] + intended[i]) for i in range(len(b))]
    if any((x & 0x7f800000) == 0x7f800000 for x in zbits):
        raise ValueError('rotation overflowed float32')
    z = [f32_bits_to_f64(x) for x in zbits]
    actual = [z[i] - b[i] for i in range(len(b))]
    q = math.sqrt(math.fsum(x*x for x in d))
    qr = math.sqrt(math.fsum(x*x for x in actual))
    cosine = math.fsum(x*y for x,y in zip(d, actual))/(q*qr) if q > 0 and qr > 0 else None
    report = {'clean_norm': q, 'rotated_norm': qr, 'cosine': cosine,
              'matched': bool(q > 0 and qr > 0 and abs(qr-q)/q <= 0.001)}
    ensure_round_to_nearest()
    return torch.tensor([_signed32(x) for x in zbits], dtype=torch.int32,
                        device=base.device).view(torch.float32), report


def intervene(model, tokens, clean, layer, kind, q=None, base=None, on_partial=None):
    ensure_round_to_nearest()
    if layer not in MAPPINGS or kind not in ('decoy', 'target', 'combined', 'patch', 'rotation', 'self'):
        raise ValueError('unsupported fixed intervention')
    if (kind == 'patch') != (q in candidate_map(layer)):
        raise ValueError('a candidate patch requires one fixed Q identity')
    if (kind == 'rotation') != (base is not None):
        raise ValueError('rotation requires the previously captured combined vector')
    t, f = tokens.shape[1] - 2, tokens.shape[1] - 1
    if t < 1 or tokens.shape[0] != 1:
        raise ValueError('invalid token positions')
    hidden = clean[(0, 't')].numel()
    delta = torch.zeros(hidden, dtype=torch.float32, device=tokens.device)
    delta[0] = 4.0
    steps = []
    rotation_info = None
    candidate = candidate_map(layer).get(q) if kind == 'patch' else None
    z = None
    if kind == 'rotation':
        z, rotation_info = rotate_displacement(base, clean[(layer + 1, 't')])
    for block in LAYERS:
        full = _full(block)
        if block == 0 and kind != 'target':
            steps.extend([dict(full, name='decoy_before', operation='capture'),
                          dict(_positioned(block, f), name='decoy_add', operation='add', value=delta),
                          dict(full, name='decoy_after', operation='capture')])
        if block == layer and kind != 'decoy':
            steps.extend([dict(full, name='target_before', operation='capture'),
                          dict(_positioned(block, t), name='target_add', operation='add', value=delta),
                          dict(full, name='target_after', operation='capture')])
        if candidate is not None and block == candidate['layer']:
            pos = candidate['position']
            token = t if pos == 't' else f
            steps.extend([dict(full, name='patch_before', operation='capture'),
                          dict(_positioned(block, token), name='patch', operation='patch',
                               value=clean[(block, pos)]),
                          dict(full, name='patch_after', operation='capture')])
        if kind == 'rotation' and block == layer + 1:
            steps.extend([dict(full, name='patch_before', operation='capture'),
                          dict(_positioned(block, t), name='patch', operation='patch', value=z),
                          dict(full, name='patch_after', operation='capture')])
        if kind == 'self' and block == layer + 1:
            steps.extend([dict(_positioned(block, t), name='self_source', operation='capture'),
                          dict(full, name='self_before', operation='capture'),
                          dict(_positioned(block, t), name='self_patch', operation='patch',
                               source_capture='self_source'),
                          dict(full, name='self_after', operation='capture')])
        for name, token in (('t', t), ('f', f)):
            steps.append(dict(_positioned(block, token), name=f'tap_{block}_{name}', operation='capture'))
    plan = ActivationPlan(model, steps, max_tensor_bytes=1_000_000,
                          max_capture_bytes=6_000_000, max_source_bytes=1_000_000,
                          max_invocations=1)
    output = None
    try:
        with plan:
            output = model(input_ids=tokens, use_cache=False)
        ensure_round_to_nearest()
        captures = dict(plan.captures)
        if z is not None:
            captures['rotation_planned'] = z.detach().clone()
        return {'logits': _saved_logits(output, t, f), 'captures': captures,
                'events': plan.metadata['steps'], 'rotation_info': rotation_info}
    except BaseException as error:
        _partial(output, tokens, on_partial, error)
        raise


def null_self(model, tokens, on_partial=None):
    ensure_round_to_nearest()
    t, f = tokens.shape[1] - 2, tokens.shape[1] - 1
    if tokens.ndim != 2 or tokens.shape[0] != 1 or t < 1:
        raise ValueError('invalid token positions')
    full, position = _full(0), _positioned(0, t)
    steps = [dict(full, name='before', operation='capture'),
             dict(position, name='null', operation='null'),
             dict(position, name='self_source', operation='capture'),
             dict(position, name='self_patch', operation='patch', source_capture='self_source'),
             dict(full, name='after', operation='capture')]
    plan = ActivationPlan(model, steps, max_tensor_bytes=1_000_000,
                          max_capture_bytes=2_000_000, max_source_bytes=1_000_000,
                          max_invocations=1)
    output = None
    try:
        with plan:
            output = model(input_ids=tokens, use_cache=False)
        ensure_round_to_nearest()
        return {'logits': _saved_logits(output, t, f), 'captures': plan.captures,
                'events': plan.metadata['steps']}
    except BaseException as error:
        _partial(output, tokens, on_partial, error)
        raise


def _check_add(captures, name, pos):
    ensure_round_to_nearest()
    before, after = captures[name + '_before'], captures[name + '_after']
    if (before.dtype != torch.float32 or after.dtype != torch.float32
            or after.shape != before.shape or before.ndim != 3
            or before.device.type != 'cpu' or after.device.type != 'cpu'
            or not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all())):
        raise ValueError('invalid injection witness')
    expected = before.contiguous().clone()
    old_bits = int(before.contiguous().view(torch.int32)[0, pos, 0].item()) & 0xffffffff
    wanted_bits = f64_to_f32_bits(f32_bits_to_f64(old_bits) + 4.0)
    if (wanted_bits & 0x7f800000) == 0x7f800000 or wanted_bits == old_bits:
        raise ValueError('injection was nonfinite or undelivered')
    expected.view(torch.int32)[0, pos, 0] = _signed32(wanted_bits)
    if not bitwise_equal(expected, after):
        raise ValueError('injection landed differently or changed another coordinate')
    ensure_round_to_nearest()


def check_landing(result, tokens, clean, layer, kind, q=None, base=None):
    ensure_round_to_nearest()
    captures = result['captures']
    t, f = tokens.shape[1] - 2, tokens.shape[1] - 1
    if kind != 'target':
        _check_add(captures, 'decoy', f)
    if kind != 'decoy':
        _check_add(captures, 'target', t)
    if kind in ('patch', 'rotation'):
        before, after = captures['patch_before'], captures['patch_after']
        expected = before.clone()
        if kind == 'patch':
            candidate = candidate_map(layer)[q]
            block, pos = candidate['layer'], candidate['position']
            token = t if pos == 't' else f
            expected[0, token] = clean[(block, pos)]
        else:
            if not bitwise_equal(before[0, t], base):
                raise ValueError('rotation pre-patch vector differs from saved combined state')
            planned = rotate_displacement(base, clean[(layer + 1, 't')])[0]
            if not bitwise_equal(planned, captures['rotation_planned']):
                raise ValueError('recorded rotation plan differs')
            expected[0, t] = planned
        if not bitwise_equal(expected, after):
            raise ValueError('patch landed differently or changed an untouched coordinate')
    if kind == 'self' and not bitwise_equal(captures['self_before'], captures['self_after']):
        raise ValueError('downstream self patch changed a bit')
    ensure_round_to_nearest()
