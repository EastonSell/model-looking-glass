"""Exact-rational fidelity checks for the exposed-family continuation."""
from fractions import Fraction
from decimal import Decimal, localcontext
import math
import struct

from . import year_label_control as study
from .copy_head_factorial import f64_to_f32_bits
from .year_transfer_intervention import (
    BlockOutputWitness as _Witness, MIN_DOSE, ROW_BYTES, WIDTH, _bits, _norm, _values,
)

ACTIONS = study.ACTIONS


def _fraction(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError('a finite binary64 value is required')
    return Fraction.from_float(value)


def _ratio(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def _preserve_binary64_subnormals():
    tiny = struct.unpack('<d', struct.pack('<Q', 1))[0]
    if (struct.unpack('<Q', struct.pack('<d', tiny + tiny))[0] != 2
            or struct.unpack('<Q', struct.pack('<d', tiny * 1.0))[0] != 1):
        raise RuntimeError('binary64 subnormal arithmetic is not preserved')


def _decimal_sqrt(value):
    with localcontext() as context:
        context.prec = 25
        return str((Decimal(value.numerator) / Decimal(value.denominator)).sqrt())


def delivery_certificate(before, after, displacement, dose):
    """Compare exact squared ideal, realized and error norms to fixed rational bounds."""
    if (type(before) is not bytes or type(after) is not bytes
            or len(before) != ROW_BYTES or len(after) != ROW_BYTES
            or type(displacement) is not tuple or len(displacement) != WIDTH
            or type(dose) is not float or not math.isfinite(dose) or dose <= MIN_DOSE):
        raise ValueError('invalid finite vector or dose')
    _preserve_binary64_subnormals()
    q = _fraction(dose)
    source, target = _values(before), _values(after)
    ideal = tuple(_fraction(x) for x in displacement)
    realized = tuple(_fraction(z) - _fraction(b) for b, z in zip(source, target))
    error = tuple(r-d for r, d in zip(realized, ideal))
    D2 = sum((x*x for x in ideal), Fraction())
    R2 = sum((x*x for x in realized), Fraction())
    E2 = sum((x*x for x in error), Fraction())
    one = Fraction(1)
    ideal_low, ideal_high = (q*(one-Fraction(1, 10**12)))**2, (q*(one+Fraction(1, 10**12)))**2
    actual_low, actual_high = (q*(one-Fraction(1, 5000)))**2, (q*(one+Fraction(1, 5000)))**2
    error_high = (q/Fraction(5000))**2
    if not (ideal_low <= D2 <= ideal_high and actual_low <= R2 <= actual_high
            and 0 < R2 and E2 <= error_high):
        raise ValueError('delivered vector is outside the fixed fidelity bounds')
    rounded_error2 = float(E2)
    return {'D2': _ratio(D2), 'R2': _ratio(R2), 'E2': _ratio(E2),
            'limits': {'D2_min': _ratio(ideal_low), 'D2_max': _ratio(ideal_high),
                       'R2_min': _ratio(actual_low), 'R2_max': _ratio(actual_high),
                       'E2_max': _ratio(error_high)},
            'achieved': math.sqrt(float(R2)),
            'vector_error': None if E2 and rounded_error2 == 0.0 else math.sqrt(rounded_error2),
            'vector_error_decimal': _decimal_sqrt(E2)}


def _delivery(source, displacement, dose):
    if type(displacement) is not tuple or len(displacement) != WIDTH:
        raise ValueError('unexpected displacement width')
    before = _values(source)
    if len(before) != WIDTH:
        raise ValueError('unexpected vector width')
    after_bits = tuple(f64_to_f32_bits(value + delta) for value, delta in zip(before, displacement))
    after = struct.pack(f'<{WIDTH}I', *after_bits)
    certificate = delivery_certificate(source, after, displacement, dose)
    return {'before': source, 'after': after, 'requested': dose,
            'achieved': certificate['achieved'], 'displacement': displacement,
            'certificate': certificate,
            'changed_bits': sum(a != b for a, b in zip(_bits(source), after_bits))}


def plan_family(noops, labels):
    """Use all 12 crossed edges to set a single family-wide dose."""
    if (type(labels) is not tuple or labels not in study.LABEL_PAIRS
            or type(noops) is not dict or set(noops) != set(study.CONDITIONS)):
        raise ValueError('expected the fixed three wordings and label pair')
    vectors, norms, lengths = {}, {}, set()
    for condition in study.CONDITIONS:
        group = noops[condition]
        if type(group) is not dict or set(group) != set(labels):
            raise ValueError('expected both labels')
        vectors[condition], first_zero = {}, None
        for label in labels:
            if type(group[label]) is not dict or set(group[label]) != set(study.ROLES):
                raise ValueError('expected both anchor roles')
            vectors[condition][label] = {}
            for role in study.ROLES:
                raw = group[label][role]
                if (type(raw) is not bytes or len(raw) % ROW_BYTES
                        or not 2 <= len(raw)//ROW_BYTES <= 32):
                    raise ValueError('unexpected bounded block-output shape')
                lengths.add(len(raw))
                _values(raw)
                zero = raw[:ROW_BYTES]
                if first_zero is None:
                    first_zero = zero
                elif first_zero != zero:
                    raise ValueError('position-zero no-op vectors differ')
                vectors[condition][label][role] = _values(raw[-ROW_BYTES:])
        for label in labels:
            low, high = (vectors[condition][label][role] for role in study.ROLES)
            norms[f'{condition}:year:{label}'] = _norm(b-a for a, b in zip(low, high))
        for role in study.ROLES:
            left, right = (vectors[condition][label][role] for label in labels)
            norms[f'{condition}:label:{role}'] = _norm(b-a for a, b in zip(left, right))
    if len(lengths) != 1:
        raise ValueError('family block-output lengths differ')
    dose = 0.5 * min(norms.values())
    if not math.isfinite(dose) or dose <= MIN_DOSE:
        raise ValueError('the common input dose is nonfinite or too small')

    plans = {}
    for condition in study.CONDITIONS:
        plans[condition] = {}
        for label_index, label in enumerate(labels):
            plans[condition][label] = {}
            for role_index, role in enumerate(study.ROLES):
                raw = noops[condition][label][role]
                final = len(raw)//ROW_BYTES - 1
                source = vectors[condition][label][role]
                year_donor = vectors[condition][label][study.ROLES[1-role_index]]
                label_donor = vectors[condition][labels[1-label_index]][role]
                year_scale = dose / norms[f'{condition}:year:{label}']
                label_scale = dose / norms[f'{condition}:label:{role}']
                year = tuple(year_scale * (b-a) for a, b in zip(source, year_donor))
                lexical = tuple(label_scale * (b-a) for a, b in zip(source, label_donor))
                rotated = tuple(year[(i-1) % WIDTH] for i in range(WIDTH))
                before = raw[-ROW_BYTES:]
                plans[condition][label][role] = {
                    'self': {'position': final, 'before': before, 'after': before,
                             'requested': 0.0, 'achieved': 0.0, 'changed_bits': 0,
                             'displacement': None, 'certificate': None},
                    'year_final': {'position': final, **_delivery(before, year, dose)},
                    'label_final': {'position': final, **_delivery(before, lexical, dose)},
                    'year_at_zero': {'position': 0, **_delivery(raw[:ROW_BYTES], year, dose)},
                    'rotated_year_final': {'position': final, **_delivery(before, rotated, dose)},
                }
    return {'dose': dose, 'norms': norms, 'plans': plans}


class BlockOutputWitness(_Witness):
    """Use v1's exact landing hook and recheck v2 fidelity from actual bits."""
    def verify_complete(self):
        if self.calls != 1 or self.before is None or self.after is None:
            raise ValueError('block output was not observed exactly once')
        if self.plan is None:
            return
        start = self.plan['position'] * ROW_BYTES
        before = self.before[start:start+ROW_BYTES]
        after = self.after[start:start+ROW_BYTES]
        changed = sum(a != b for a, b in zip(_bits(after), _bits(before)))
        if changed != self.plan.get('changed_bits'):
            raise ValueError('same-forward changed-bit count differs from the plan')
        if self.plan['requested'] == 0.0:
            if before != after or self.plan['displacement'] is not None or self.plan['certificate'] is not None:
                raise ValueError('self replacement is not bit-identical')
            self.observed = {'achieved': 0.0, 'changed_bits': changed, 'certificate': None}
            return
        certificate = delivery_certificate(before, after, self.plan['displacement'], self.plan['requested'])
        if certificate != self.plan['certificate']:
            raise ValueError('same-forward fidelity certificate differs from the plan')
        self.observed = {'achieved': certificate['achieved'], 'changed_bits': changed,
                         'certificate': certificate}
