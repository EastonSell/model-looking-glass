"""Prospective FLAN sentiment action order and bit-defined CPU dose geometry.

These helpers neither select cases nor load a checkpoint or authorize a
forward. Their inputs must be supplied by a separately verified, exclusive owner.
"""
import math
import struct
import sys

import torch

from .bert_matched_numeric import binary64_rounding_probe, f32_bits_to_f64, f64_to_f32_bits

RECEIVERS = ('P1', 'N1', 'P2', 'N2')
ACTIONS = ('opposite_adjective', 'same_adjective', 'opposite_instruction', 'opposite_subject')
SELF_SITES = ('adjective', 'instruction', 'subject')
OPPOSITE = {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
SAME = {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}
_ACTION_INFO = {
    'opposite_adjective': ('adjective', OPPOSITE),
    'same_adjective': ('adjective', SAME),
    'opposite_instruction': ('instruction', OPPOSITE),
    'opposite_subject': ('subject', OPPOSITE),
}


def _rotate(items, amount):
    return items[amount:] + items[:amount]


def fixed_schedule():
    """Return all 484 potential evaluations, before eligibility-based skips."""
    groups = []
    for unit in range(4):
        for instruction in ((0, 1) if unit % 2 == 0 else (1, 0)):
            groups.append((unit, instruction, _rotate(RECEIVERS, (unit + instruction) % 4)))
    schedule = []
    for unit, instruction, receivers in groups:
        for receiver in receivers:
            schedule.append(dict(kind='clean', unit=unit, instruction=instruction, receiver=receiver))
    for unit, instruction, receivers in groups:
        for receiver in receivers:
            base = dict(unit=unit, instruction=instruction, receiver=receiver)
            schedule.append(dict(kind='passthrough', **base))
            schedule.extend(dict(kind='self', site=site, **base) for site in SELF_SITES)
            schedule.append(dict(kind='full_swap', donor=OPPOSITE[receiver], **base))
    for unit in range(4):
        for u, instruction, receivers in groups:
            if u != unit:
                continue
            for receiver in receivers:
                r = RECEIVERS.index(receiver)
                actions = _rotate(tuple(enumerate(ACTIONS)), (unit + instruction + r) % 4)
                for a, action in actions:
                    levels = ('low', 'high') if (unit + instruction + r + a) % 2 == 0 else ('high', 'low')
                    for level in levels:
                        schedule.append(dict(kind='matched', unit=unit, instruction=instruction,
                                             receiver=receiver, action=action, dose=level))
                schedule.append(dict(kind='natural', unit=unit, instruction=instruction,
                                     receiver=receiver, donor=OPPOSITE[receiver]))
        schedule.append(dict(kind='repeat', unit=unit, instruction=0 if unit % 2 == 0 else 1,
                             receiver='P1'))
    return tuple(schedule)


def runtime_numeric_probe():
    """Probe representative binary32/binary64 ties and denormals on this thread."""
    binary64_rounding_probe()
    values = torch.tensor([0x3f800000, 0x3f800001], dtype=torch.int32).view(torch.float32)
    rounded = (values + math.ldexp(1.0, -24)).view(torch.int32).tolist()
    if rounded != [0x3f800000, 0x3f800002]:
        raise RuntimeError('binary32 rounding probe failed')
    results = []
    for integer, floating, minimum, sign in ((torch.int32, torch.float32, 0x00800000, 1 << 31),
                                             (torch.int64, torch.float64, 0x0010000000000000, 1 << 63)):
        bits = torch.tensor([1, minimum, 1 - sign, minimum - sign], dtype=integer).view(floating)
        input_result = int((bits[0] * 2.0).view(integer).item())
        output_result = int((bits[1] * 0.5).view(integer).item())
        negative_input = int((bits[2] * 2.0).view(integer).item())
        negative_output = int((bits[3] * 0.5).view(integer).item())
        if (input_result != 2 or output_result != minimum // 2
                or negative_input != 2 - sign or negative_output != minimum // 2 - sign):
            raise RuntimeError('CPU denormal preservation probe failed')
        results.append((input_result, output_result, negative_input, negative_output))
    return {'binary32_ties': rounded, 'float32_denormals': results[0],
            'float64_denormals': results[1]}


def _norm(values):
    value = math.sqrt(math.fsum(x ** 2 for x in values))
    if not math.isfinite(value):
        raise ValueError('nonfinite vector norm')
    return value


def _state_vectors(states, positions):
    if (type(states) is not dict or set(states) != set(RECEIVERS)
            or type(positions) is not dict or set(positions) != set(SELF_SITES)
            or any(type(x) is not int or x < 0 for x in positions.values())
            or len(set(positions.values())) != 3 or sys.byteorder != 'little'):
        raise ValueError('invalid group state or positions')
    shape = None
    result = {}
    for receiver in RECEIVERS:
        tensor = states[receiver]
        if (type(tensor) is not torch.Tensor or tensor.dtype != torch.float32
                or tensor.device.type != 'cpu' or tensor.layout != torch.strided
                or tensor.ndim != 3 or tensor.shape[0] != 1
                or not 1 <= tensor.shape[1] <= 64 or not 1 <= tensor.shape[2] <= 512
                or max(positions.values()) >= tensor.shape[1]):
            raise ValueError('expected bounded CPU float32 encoder states')
        if shape is not None and tuple(tensor.shape) != shape:
            raise ValueError('group encoder state shapes differ')
        shape = tuple(tensor.shape)
        raw = tensor.detach().contiguous().view(torch.int32).numpy().tobytes()
        words = tuple(word for (word,) in struct.iter_unpack('<I', raw))
        if any((word & 0x7f800000) == 0x7f800000 for word in words):
            raise ValueError('nonfinite encoder state')
        width = shape[2]
        result[receiver] = {site: tuple(f32_bits_to_f64(w) for w in words[pos * width:(pos + 1) * width])
                            for site, pos in positions.items()}
    return result


def _cosine(left, right, left_norm, right_norm):
    if left_norm == 0 or right_norm == 0:
        return None
    value = math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    if not math.isfinite(value):
        raise ValueError('nonfinite cosine')
    return value


def plan_group(states, positions):
    """Compute all 16 distances and, if eligible, all 32 proposed landings.

    Values are little-endian binary32 bytes. The calling owner must persist
    them all and verify actual state bytes before invoking an intervention.
    """
    runtime_numeric_probe()
    vectors = _state_vectors(states, positions)
    differences, distances = {}, {}
    for receiver in RECEIVERS:
        for action in ACTIONS:
            site, mapping = _ACTION_INFO[action]
            difference = tuple(b - a for a, b in zip(vectors[receiver][site], vectors[mapping[receiver]][site]))
            key = f'{receiver}:{action}'
            differences[key] = difference
            distances[key] = _norm(difference)
    q = min(distances.values())
    donor_cosines, q_over_word = {}, {}
    for receiver in RECEIVERS:
        opposite, same = f'{receiver}:opposite_adjective', f'{receiver}:same_adjective'
        cosine = _cosine(differences[opposite], differences[same], distances[opposite], distances[same])
        donor_cosines[receiver] = {'value': cosine, 'reason': 'zero donor norm' if cosine is None else None}
        q_over_word[receiver] = None if distances[opposite] == 0 else q / distances[opposite]
    result = {'status': 'dose-ineligible' if q / 2 <= 1e-6 else 'matchable',
              'q': q, 'distances': distances, 'q_over_word_distance': q_over_word,
              'donor_cosines': donor_cosines, 'landings': {}}
    if result['status'] == 'dose-ineligible':
        return result
    failures = False
    for receiver in RECEIVERS:
        for action in ACTIONS:
            site, mapping = _ACTION_INFO[action]
            base = vectors[receiver][site]
            key = f'{receiver}:{action}'
            distance = distances[key]
            for level, dose in (('low', q / 2), ('high', q)):
                fraction = dose / distance
                if not 0 < fraction <= 1:
                    raise ValueError('invalid interpolation fraction')
                # Keep subtraction, division, multiply and addition distinct;
                # integer bit conversion supplies one final ties-to-even cast.
                words = tuple(f64_to_f32_bits(float(a + fraction * delta))
                              for a, delta in zip(base, differences[key]))
                raw = struct.pack('<' + str(len(words)) + 'I', *words)
                finite = all((word & 0x7f800000) != 0x7f800000 for word in words)
                actual = tuple(f32_bits_to_f64(w) for w in words) if finite else None
                displacement = tuple(a - b for a, b in zip(actual, base)) if finite else None
                achieved = _norm(displacement) if finite else None
                cosine = _cosine(displacement, differences[key], achieved, distance) if finite else None
                tolerance = min(max(1e-5, 1e-5 * dose), .01 * dose)
                base_norm = _norm(base)
                reason = None
                if not finite:
                    reason = 'nonfinite rounded landing'
                elif achieved == 0:
                    reason = 'zero achieved displacement'
                elif abs(achieved - dose) > tolerance:
                    reason = 'landing norm outside tolerance'
                elif cosine is None or cosine < .99:
                    reason = 'landing cosine below threshold'
                failures |= reason is not None
                result['landings'][f'{key}:{level}'] = {
                    'receiver': receiver, 'action': action, 'site': site,
                    'donor': mapping[receiver], 'level': level, 'value': raw,
                    'target_dose': dose, 'fraction': fraction, 'achieved_norm': achieved,
                    'cosine': cosine, 'tolerance': tolerance, 'receiver_norm': base_norm,
                    'achieved_over_receiver_norm': None if not finite or base_norm == 0 else achieved / base_norm,
                    'ratio_reason': 'zero receiver norm' if base_norm == 0 else None,
                    'failure': reason,
                }
    for receiver in RECEIVERS:
        for level in ('low', 'high'):
            reference = result['landings'][f'{receiver}:opposite_adjective:{level}']['achieved_norm']
            for action in ACTIONS[1:]:
                landing = result['landings'][f'{receiver}:{action}:{level}']
                actual = landing['achieved_norm']
                if reference is None or actual is None or abs(actual - reference) > .01 * reference:
                    failures = True
                    landing['failure'] = landing['failure'] or 'control norm differs from opposite adjective'
    if failures:
        result['status'] = 'dose-unmatchable'
    return result


def group_decision(clean_margins, action_margins):
    """Compute the fixed larger-dose rule for one actually measured group."""
    if (type(clean_margins) is not dict or set(clean_margins) != set(RECEIVERS)
            or type(action_margins) is not dict or set(action_margins) != set(ACTIONS)
            or any(type(row) is not dict or set(row) != set(RECEIVERS)
                   for row in action_margins.values())):
        raise ValueError('expected all four receivers and four action kinds')
    values = list(clean_margins.values()) + [x for row in action_margins.values() for x in row.values()]
    if any(type(x) is not float or not math.isfinite(x) for x in values):
        raise ValueError('margins must be finite binary64 values')
    movements = {}
    for action in ACTIONS:
        movements[action] = {
            receiver: (clean_margins[receiver] - action_margins[action][receiver]
                       if receiver.startswith('P') else action_margins[action][receiver] - clean_margins[receiver])
            for receiver in RECEIVERS
        }
    a = math.fsum(movements[ACTIONS[0]][receiver] for receiver in RECEIVERS) / 4.0
    controls = {action: math.fsum(abs(movements[action][receiver]) for receiver in RECEIVERS) / 4.0
                for action in ACTIONS[1:]}
    b = max(controls.values())
    count = sum(movements[ACTIONS[0]][receiver] > 0 for receiver in RECEIVERS)
    return {'status': 'measured', 'movements': movements, 'A': a, 'B': b,
            'control_means': controls, 'positive_count': count,
            'passes': bool(a >= .10 and a - b >= .05 and count >= 3)}


def primary_decision(units, *, attempt_complete):
    """Keep the four fixed units in the denominator; incomplete has no result."""
    if (type(units) not in (list, tuple) or len(units) != 4 or type(attempt_complete) is not bool
            or any(type(pair) not in (list, tuple) or len(pair) != 2 for pair in units)):
        raise ValueError('expected four units and two instruction groups per unit')
    if not attempt_complete:
        return {'status': 'incomplete', 'passed_units': None, 'total_units': 4}
    allowed = {'measured', 'baseline-unsupported', 'dose-ineligible', 'dose-unmatchable'}
    for pair in units:
        for group in pair:
            if (type(group) is not dict or group.get('status') not in allowed
                    or (group['status'] == 'measured' and type(group.get('passes')) is not bool)):
                raise ValueError('group outcome is missing or invalid')
    passed = sum(all(group['status'] == 'measured' and group['passes'] for group in pair) for pair in units)
    return {'status': 'supported' if passed >= 3 else 'threshold-not-met',
            'passed_units': passed, 'total_units': 4}
