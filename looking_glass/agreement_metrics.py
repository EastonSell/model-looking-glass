"""Bounded factorial arithmetic for a prespecified agreement-patching study.

This does not execute a model, establish provenance, or certify that patches
landed. The caller must validate those controls before presenting observations
to this pure calculation and preserve incomplete attempts separately.
"""

import math


class StudyInvalid(ValueError):
    """The supplied observations do not form the declared complete factorial."""


_FLOAT32_MAX = 3.4028234663852886e38
_CELL_KEYS = (
    'same_lemma_same_number', 'same_lemma_opposite_number',
    'different_lemma_same_number', 'different_lemma_opposite_number',
)


def _identifier(value):
    if type(value) is not str or not value or len(value) > 128:
        raise StudyInvalid('identifiers must be nonempty bounded strings')
    return value


def _score(value):
    # Values must originate from finite float32 logits decoded to binary64 by
    # the execution layer. Restrict range as well as finiteness before sums.
    if type(value) not in (int, float) or abs(value) > _FLOAT32_MAX or not math.isfinite(value):
        raise StudyInvalid('a logit is nonfinite, out of float32 range or invalid')
    result = float(value)
    numerator, denominator = result.as_integer_ratio()
    if numerator:
        significand = abs(numerator)
        trailing = (significand & -significand).bit_length() - 1
        significand >>= trailing
        exponent = trailing - (denominator.bit_length() - 1)
        if significand.bit_length() > 24 or exponent < -149 or exponent + significand.bit_length() - 1 > 127:
            raise StudyInvalid('a logit is not an exact decoded float32 value')
    return result


def score_factorial(cases, patches, sites, *, expected_quartets=4):
    """Return all pair effects, per-quartet cells and equally weighted site means.

    ``cases`` contains id, quartet, lemma, number and two decoded clean logits.
    ``patches`` contains site, target/donor IDs and two decoded patched logits.
    Every quartet must contain the full 2-lemma by 2-number factorial; for each
    requested site every target must have each of the four same-quartet donors.
    Missing, duplicate, cross-quartet or invalid observations raise instead of
    shrinking a denominator. The default is the protocol's four quartets.
    """
    if (type(expected_quartets) is not int or not 1 <= expected_quartets <= 16
            or type(cases) not in (list, tuple) or type(patches) not in (list, tuple)
            or type(sites) not in (list, tuple) or not 1 <= len(sites) <= 32):
        raise StudyInvalid('expected bounded cases, patches, sites and quartet count')
    if len(cases) != 4 * expected_quartets or len(patches) != 16 * expected_quartets * len(sites):
        raise StudyInvalid('the declared factorial is incomplete or oversized')
    selected_sites = tuple(_identifier(site) for site in sites)
    if len(set(selected_sites)) != len(selected_sites):
        raise StudyInvalid('duplicate site')
    known = {}
    groups = {}
    baseline = {'n': 0, 'correct': 0, 'incorrect': 0, 'ties': 0}
    for row in cases:
        if type(row) is not dict or set(row) != {'id', 'quartet', 'lemma', 'number', 'clean_is', 'clean_are'}:
            raise StudyInvalid('invalid case record fields')
        item = {key: _identifier(row[key]) for key in ('id', 'quartet', 'lemma', 'number')}
        if item['number'] not in ('singular', 'plural') or item['id'] in known:
            raise StudyInvalid('invalid number or duplicate case')
        item['clean_is'], item['clean_are'] = _score(row['clean_is']), _score(row['clean_are'])
        item['clean_margin'] = math.fsum((item['clean_is'], -item['clean_are']))
        item['weight'] = 1 if item['number'] == 'singular' else -1
        signed_margin = item['weight'] * item['clean_margin']
        baseline['n'] += 1
        baseline['correct'] += int(signed_margin > 0)
        baseline['incorrect'] += int(signed_margin <= 0)
        baseline['ties'] += int(signed_margin == 0)
        known[item['id']] = item
        groups.setdefault(item['quartet'], []).append(item)
    if len(groups) != expected_quartets:
        raise StudyInvalid('wrong number of quartets')
    for rows in groups.values():
        if len(rows) != 4 or len({item['lemma'] for item in rows}) != 2:
            raise StudyInvalid('a quartet must contain two lemmas')
        if len({(item['lemma'], item['number']) for item in rows}) != 4:
            raise StudyInvalid('a quartet must contain both numbers for each lemma')

    indexed = {}
    for row in patches:
        if type(row) is not dict or set(row) != {'site', 'target', 'donor', 'patched_is', 'patched_are'}:
            raise StudyInvalid('invalid patch record fields')
        site, target, donor = (_identifier(row[name]) for name in ('site', 'target', 'donor'))
        if (site not in selected_sites or target not in known or donor not in known
                or known[target]['quartet'] != known[donor]['quartet']):
            raise StudyInvalid('unknown site, case or cross-quartet donor')
        key = (site, target, donor)
        if key in indexed:
            raise StudyInvalid('duplicate patch observation')
        indexed[key] = (_score(row['patched_is']), _score(row['patched_are']))

    pairs, quartets, site_summaries = [], [], []
    for site in selected_sites:
        site_rows = []
        for quartet in sorted(groups):
            family = sorted(groups[quartet], key=lambda item: item['id'])
            cells = {key: [] for key in _CELL_KEYS}
            for target in family:
                for donor in family:
                    key = (site, target['id'], donor['id'])
                    if key not in indexed:
                        raise StudyInvalid('a target-donor cell is missing')
                    patched_is, patched_are = indexed[key]
                    if target['id'] == donor['id']:
                        for clean, patched in ((target['clean_is'], patched_is),
                                               (target['clean_are'], patched_are)):
                            if abs(clean - patched) > 1e-5 + 1e-5 * max(abs(clean), abs(patched)):
                                raise StudyInvalid('self-patch selected logits differ from the clean reference')
                    patched_margin = math.fsum((patched_is, -patched_are))
                    u = target['weight'] * math.fsum((target['clean_margin'], -patched_margin))
                    lemma_factor = 'same_lemma' if target['lemma'] == donor['lemma'] else 'different_lemma'
                    number_factor = 'same_number' if target['number'] == donor['number'] else 'opposite_number'
                    cell = lemma_factor + '_' + number_factor
                    cells[cell].append(u)
                    pairs.append({'site': site, 'quartet': quartet, 'target': target['id'],
                                  'donor': donor['id'], 'cell': cell, 'weight': target['weight'],
                                  'clean_is': target['clean_is'], 'clean_are': target['clean_are'],
                                  'clean_margin': target['clean_margin'], 'patched_is': patched_is,
                                  'patched_are': patched_are, 'patched_margin': patched_margin,
                                  'signed_target_shift': u})
            if any(len(values) != 4 for values in cells.values()):
                raise StudyInvalid('a factorial cell does not have four targets')
            means = {key: math.fsum(values) / 4 for key, values in cells.items()}
            so, ss = means['same_lemma_opposite_number'], means['same_lemma_same_number']
            do, ds = means['different_lemma_opposite_number'], means['different_lemma_same_number']
            row = {'site': site, 'quartet': quartet, 'cell_means': means,
                   'cell_counts': {key: len(values) for key, values in cells.items()},
                   'number_main_contrast': math.fsum((so, -ss, do, -ds)) / 2,
                   'lemma_number_interaction': math.fsum((so, -ss, -do, ds))}
            quartets.append(row)
            site_rows.append(row)
        site_summaries.append({'site': site, 'n_quartets': len(site_rows),
                               'mean_number_main_contrast': math.fsum(
                                   row['number_main_contrast'] for row in site_rows) / len(site_rows),
                               'mean_lemma_number_interaction': math.fsum(
                                   row['lemma_number_interaction'] for row in site_rows) / len(site_rows)})
    return {'baseline': baseline, 'pairs': pairs, 'quartets': quartets,
            'site_summaries': site_summaries}
