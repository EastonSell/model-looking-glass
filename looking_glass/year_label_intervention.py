"""Bit-defined crossed year and label donor interventions."""
import math

from . import year_label_control as study
from .year_transfer_intervention import (
    BlockOutputWitness, MIN_DOSE, ROW_BYTES, WIDTH, _delivery, _norm, _values,
)

ACTIONS = study.ACTIONS


def plan_family(noops, labels):
    """Use all twelve unique donor edges to set one family-wide dose."""
    if (type(labels) is not tuple or labels not in study.LABEL_PAIRS
            or type(noops) is not dict or set(noops) != set(study.CONDITIONS)):
        raise ValueError('expected the fixed three wordings and label pair')
    vectors, norms = {}, {}
    lengths = set()
    for condition in study.CONDITIONS:
        group = noops[condition]
        if type(group) is not dict or set(group) != set(labels):
            raise ValueError('expected both labels')
        vectors[condition] = {}
        first_zero = None
        for label in labels:
            if type(group[label]) is not dict or set(group[label]) != set(study.ROLES):
                raise ValueError('expected both anchor roles')
            vectors[condition][label] = {}
            for role in study.ROLES:
                raw = group[label][role]
                if (type(raw) is not bytes or len(raw) % ROW_BYTES
                        or not 2 <= len(raw) // ROW_BYTES <= 32):
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
                final = len(raw) // ROW_BYTES - 1
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
                             'requested': 0.0, 'achieved': 0.0, 'changed_bits': 0},
                    'year_final': {'position': final, **_delivery(before, year, dose)},
                    'label_final': {'position': final, **_delivery(before, lexical, dose)},
                    'year_at_zero': {'position': 0, **_delivery(raw[:ROW_BYTES], year, dose)},
                    'rotated_year_final': {'position': final, **_delivery(before, rotated, dose)},
                }
    return {'dose': dose, 'norms': norms, 'plans': plans}
