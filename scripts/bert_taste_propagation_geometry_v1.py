"""Fixed common-radius geometry for the public propagation study."""
import hashlib
import math
import struct

from scripts import bert_taste_template_geometry_v1 as base

SITES = ('early_ingredient', 'late_ingredient', 'late_the', 'late_tasted')
TEMPLATES = ('after', 'with')
INGREDIENTS = ('sugar', 'lemon')


def signs(stem, template, site, index, width):
    if (type(stem) is not str or template not in TEMPLATES or site not in SITES
            or type(index) is not int or index not in (1, 2) or type(width) is not int or width <= 0):
        raise ValueError('invalid orthogonal identity')
    seed = hashlib.sha256(b'bert-taste-context-propagation-orth-v1\0' + stem.encode('utf-8')
                          + b'\0' + template.encode('ascii') + b'\0' + site.encode('ascii')
                          + b'\0' + str(index).encode('ascii')).digest()
    values, counter = [], 0
    while len(values) < width:
        block = hashlib.sha256(seed + counter.to_bytes(4, 'big')).digest()
        for byte in block:
            for shift in range(7, -1, -1):
                values.append(1. if byte & (1 << shift) else -1.)
                if len(values) == width:
                    return tuple(values)
        counter += 1
    return tuple(values)


def dot(left, right):
    if len(left) != len(right):
        raise ValueError('vector widths differ')
    return math.fsum(a * b for a, b in zip(left, right))


def orthogonals(stem, template, site, lexical):
    lexical = base.values(lexical)
    denominator = math.fsum(value ** 2 for value in lexical)
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError('invalid lexical direction')
    units = []
    for index in (1, 2):
        vector = signs(stem, template, site, index, len(lexical))
        coefficient = dot(vector, lexical) / denominator
        projected = tuple(math.fsum((v, -coefficient * d)) for v, d in zip(vector, lexical))
        if units:
            coefficient = dot(projected, units[0]) / math.fsum(value ** 2 for value in units[0])
            projected = tuple(math.fsum((v, -coefficient * u)) for v, u in zip(projected, units[0]))
        length = base.norm(projected)
        if not math.isfinite(length) or length < 1e-8:
            raise ValueError('orthogonal direction is degenerate')
        units.append(tuple(value / length for value in projected))
    return tuple(units)


def _actual(action, recipient):
    rounded = tuple(base.f32_value(bits[0]) for bits in struct.iter_unpack('<I', action['raw']))
    return base.displacement(rounded, recipient)


def cosine(left, right):
    denominator = base.norm(left) * base.norm(right)
    if not math.isfinite(denominator) or denominator == 0:
        raise ValueError('zero or invalid cosine denominator')
    value = dot(left, right) / denominator
    if not math.isfinite(value):
        raise ValueError('nonfinite cosine')
    return value


def plan(stem, captures, *, width=768):
    if type(captures) is not dict or set(captures) != set(TEMPLATES):
        raise ValueError('both templates are required')
    clean = {}
    for template in TEMPLATES:
        if type(captures[template]) is not dict or set(captures[template]) != set(INGREDIENTS):
            raise ValueError('both ingredients are required')
        clean[template] = {}
        for ingredient in INGREDIENTS:
            raw = captures[template][ingredient]
            if type(raw) is not dict or set(raw) != {*SITES, 'late_mask'}:
                raise ValueError('all five clean captures are required')
            clean[template][ingredient] = {site: base.values(raw[site]) for site in raw}
            if any(len(value) != width for value in clean[template][ingredient].values()):
                raise ValueError('capture width differs')

    distances = {}
    for site in SITES:
        distances[site] = {
            'lex_after': base.norm(base.displacement(clean['after']['sugar'][site], clean['after']['lemon'][site])),
            'lex_with': base.norm(base.displacement(clean['with']['sugar'][site], clean['with']['lemon'][site])),
            'context_sugar': base.norm(base.displacement(clean['after']['sugar'][site], clean['with']['sugar'][site])),
            'context_lemon': base.norm(base.displacement(clean['after']['lemon'][site], clean['with']['lemon'][site])),
        }
    radius = min(value for group in distances.values() for value in group.values())
    if not math.isfinite(radius) or radius < 1e-5:
        raise ValueError('common radius or distance is invalid')

    actions, mutual, fractions = {}, {}, {}
    for template in TEMPLATES:
        other_template = 'with' if template == 'after' else 'after'
        actions[template], mutual[template], fractions[template] = {}, {}, {}
        for ingredient in INGREDIENTS:
            other_ingredient = 'lemon' if ingredient == 'sugar' else 'sugar'
            actions[template][ingredient], mutual[template][ingredient] = {}, {}
            fractions[template][ingredient] = {}
            for site in SITES:
                recipient = clean[template][ingredient][site]
                lexical = base.displacement(clean[template][other_ingredient][site], recipient)
                context = base.displacement(clean[other_template][ingredient][site], recipient)
                lex_norm, context_norm = base.norm(lexical), base.norm(context)
                if min(lex_norm, context_norm) < 1e-5:
                    raise ValueError('donor distance is invalid')
                lexical_axis = base.displacement(clean[template]['sugar'][site], clean[template]['lemon'][site])
                u1, u2 = orthogonals(stem, template, site, lexical_axis)
                units = {'lex': tuple(x / lex_norm for x in lexical),
                         'context': tuple(x / context_norm for x in context), 'orth1': u1, 'orth2': u2}
                generated = {}
                for axis, unit in units.items():
                    for label, sign in (('plus', 1), ('minus', -1)):
                        reference = context if axis == 'context' else lexical
                        action = base.source(recipient, unit, sign, radius, reference, orth=axis.startswith('orth'))
                        actual = _actual(action, recipient)
                        action['lexical_cosine'] = cosine(actual, lexical)
                        action['context_cosine'] = cosine(actual, context)
                        generated[axis + '_' + label] = action
                orth_cosines = {}
                for sign1 in ('plus', 'minus'):
                    for sign2 in ('plus', 'minus'):
                        name = 'orth1_' + sign1 + '__orth2_' + sign2
                        value = cosine(_actual(generated['orth1_' + sign1], recipient),
                                       _actual(generated['orth2_' + sign2], recipient))
                        if abs(value) > .01:
                            raise ValueError('post-rounding orthogonal directions are not mutually orthogonal')
                        orth_cosines[name] = value
                actions[template][ingredient][site] = generated
                mutual[template][ingredient][site] = orth_cosines
                fractions[template][ingredient][site] = {'lexical': radius / lex_norm,
                                                          'context': radius / context_norm,
                                                          'lexical_context_cosine': cosine(lexical, context)}
    return {'radius': radius, 'distances': distances, 'fractions': fractions,
            'actions': actions, 'mutual_cosines': mutual}
