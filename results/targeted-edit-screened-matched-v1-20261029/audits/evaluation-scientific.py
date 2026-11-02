"""Independently recompute screened edit outcomes from saved logit matrices.

This reader never loads a model. It checks numerical and row consistency, not
the provenance or actual execution of the records it reads.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VOCAB = 50257
LONDON, PARIS = 3576, 6342
BASE = ('original', 'projected', 'unprojected', 'projected_random', 'opposite', 'nuisance')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check(actual, expected, label):
    if isinstance(expected, bool) or expected is None or isinstance(expected, (str, int)):
        require(type(actual) is type(expected) and actual == expected, label)
    else:
        require(isinstance(actual, (float, int)) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), label)


def dist(row):
    z = np.asarray(row, dtype=np.float64)
    require(z.shape == (VOCAB,) and bool(np.isfinite(z).all()), 'nonfinite or malformed row')
    shifted = z - float(np.max(z))
    log_normalizer = math.log(math.fsum(map(float, np.exp(shifted))))
    return z, shifted - log_normalizer, int(np.argmax(z))


def audit(case_path, screen_dir, candidate_path, data_dir):
    case_bytes = case_path.read_bytes()
    cases = json.loads(case_bytes)['cases']
    record_bytes = (screen_dir / 'screen-selection.json').read_bytes()
    record = json.loads(record_bytes)
    candidate_bytes = candidate_path.read_bytes()
    candidate = json.loads(candidate_bytes)
    require(sha(case_bytes) == record['case_sha256'] and len(cases) == 44, 'case identity')
    require(sha(candidate_bytes) == record['identity']['candidate_sha256'], 'candidate identity')
    selected = record['selection']['selected_offsets']
    require(type(selected) is list and len(selected) <= 8 and len(set(selected)) == len(selected)
            and all(type(i) is int and 0 <= i < 32 for i in selected), 'selected offsets')
    fixed = [cases[i] for i in selected] + cases[32:]
    require(all(c['role'] == 'target' for c in fixed[:len(selected)]) and
            len(fixed[len(selected):]) == 12 and
            all(c['role'] == 'locality' for c in fixed[len(selected):]), 'case roles')
    match_available = candidate['discovery_match']['available']
    require(type(match_available) is bool, 'discovery match status')
    shared = candidate['discovery_match']['shared_equal_norm_arm']
    require(type(shared) is bool and (not shared or match_available), 'shared-arm status')
    if match_available:
        require(shared == (candidate['discovery_match']['selected_dose'] == .001), 'shared-arm dose')
    names = BASE + (('matched',) if match_available else ())
    raw, arrays = {}, {}
    for name in names:
        raw[name] = (data_dir / (name + '.f32')).read_bytes()
        require(len(raw[name]) == len(fixed) * VOCAB * 4, f'{name} size')
        arrays[name] = np.frombuffer(raw[name], dtype='<f4').reshape(len(fixed), VOCAB)
        require(bool(np.isfinite(arrays[name]).all()), f'{name} finiteness')
    if shared:
        require(raw['matched'] == raw['unprojected'], 'shared arm bytes differ')
    screen = (screen_dir / 'screen.f32').read_bytes()
    info = record['artifacts']['screen.f32']
    require(len(screen) == 32 * VOCAB * 4 and sha(screen) == info['sha256'], 'screen bytes')
    for position, offset in enumerate(selected):
        expected = screen[offset * VOCAB * 4:(offset + 1) * VOCAB * 4]
        actual = raw['original'][position * VOCAB * 4:(position + 1) * VOCAB * 4]
        require(actual == expected, f'selected baseline row {offset} was not reused')
    observations = [json.loads(line) for line in (data_dir / 'observations.jsonl').read_bytes().splitlines()]
    require(len(observations) == len(names) * len(fixed), 'observation count')
    for condition_index, name in enumerate(names):
        for index, case in enumerate(fixed):
            entry = observations[condition_index * len(fixed) + index]
            row = raw[name][index * VOCAB * 4:(index + 1) * VOCAB * 4]
            require(entry['condition'] == name and entry['case_id'] == case['case_id'] and
                    entry['index'] == index and entry['offset'] == index * VOCAB * 4 and
                    entry['bytes'] == VOCAB * 4 and entry['sha256'] == sha(row), 'observation binding')
            expected_source = ('screen' if name == 'original' and index < len(selected) else
                               'unprojected' if name == 'matched' and shared else 'forward')
            expected_offset = selected[index] if expected_source == 'screen' else None
            require(entry.get('source') == expected_source and entry.get('screen_offset') == expected_offset,
                    'observation source/reuse')

    report = json.loads((data_dir / 'provisional-report.json').read_bytes())
    require(report['case_sha256'] == sha(case_bytes) and report['selected_count'] == len(selected)
            and report['discovery_match_available'] is match_available, 'reported identity')
    require(set(report['conditions']) == set(names), 'reported condition set')
    original = [dist(row) for row in arrays['original']]
    conditions = {}
    for name in names:
        out = []
        for i, case in enumerate(fixed):
            base_z, base_lp, base_top = original[i]
            edit_z, edit_lp, edit_top = dist(arrays[name][i])
            kl = math.fsum(map(float, np.exp(base_lp) * (base_lp - edit_lp)))
            common = {'case_id': case['case_id'], 'stratum': case['stratum'],
                      'original_top': base_top, 'condition_top': edit_top, 'kl': kl,
                      'original_margin': float(base_z[LONDON] - base_z[PARIS]),
                      'condition_margin': float(edit_z[LONDON] - edit_z[PARIS]),
                      'original_london_nll': -float(base_lp[LONDON]),
                      'condition_london_nll': -float(edit_lp[LONDON]),
                      'original_paris_nll': -float(base_lp[PARIS]),
                      'condition_paris_nll': -float(edit_lp[PARIS])}
            if i < len(selected):
                m0 = float(base_z[LONDON] - base_z[PARIS])
                m1 = float(edit_z[LONDON] - edit_z[PARIS])
                require(m0 < 0, 'selected target baseline sign')
                out.append({**common, 'change': m1 - m0, 'reversed': m1 > 0})
            else:
                label = case['continuations'][0]
                require(len(label['ids']) == 1, 'locality label')
                token = label['ids'][0]
                n0, n1 = -float(base_lp[token]), -float(edit_lp[token])
                out.append({**common, 'expected_text': label['text'], 'expected_id': token,
                            'original_nll': n0, 'condition_nll': n1,
                            'nll_deterioration': n1 - n0})
        targets, locality = out[:len(selected)], out[len(selected):]
        mean = lambda xs: math.fsum(xs) / len(xs) if xs else None
        groups = {}
        for st in sorted({r['stratum'] for r in locality}):
            group = [r for r in locality if r['stratum'] == st]
            groups[st] = {'count': len(group), 'mean_kl': mean([r['kl'] for r in group]),
                          'mean_nll_deterioration': mean([r['nll_deterioration'] for r in group])}
        summary = {'targets': targets, 'locality': locality, 'locality_strata': groups,
                   'mean_target_change': mean([r['change'] for r in targets]),
                   'reversed_targets': sum(r['reversed'] for r in targets) if targets else None,
                   'all_top_preserved': all(r['original_top'] == r['condition_top'] for r in locality),
                   'mean_kl': mean([r['kl'] for r in locality]), 'max_kl': max(r['kl'] for r in locality),
                   'mean_nll_deterioration': mean([r['nll_deterioration'] for r in locality]),
                   'max_nll_deterioration': max(r['nll_deterioration'] for r in locality)}
        recorded = report['conditions'][name]
        for key in ('targets', 'locality'):
            require(len(summary[key]) == len(recorded[key]), f'{name} {key} count')
            for index, row in enumerate(summary[key]):
                for field, expected in row.items():
                    check(recorded[key][index][field], expected, f'{name} {key} {index} {field}')
        for st, group in groups.items():
            for field, expected in group.items():
                check(recorded['locality_strata'][st][field], expected, f'{name} {st} {field}')
        for field in ('mean_target_change', 'reversed_targets', 'all_top_preserved', 'mean_kl',
                      'max_kl', 'mean_nll_deterioration', 'max_nll_deterioration'):
            check(recorded[field], summary[field], f'{name} {field}')
        conditions[name] = summary

    p, u, rnd = (conditions[n] for n in ('projected', 'unprojected', 'projected_random'))
    has = bool(selected)
    common = {'at_least_two_reversals': p['reversed_targets'] >= 2 if has else None,
              'mean_target_gain': p['mean_target_change'] > .25 if has else None,
              'random_advantage': p['mean_target_change'] - rnd['mean_target_change'] >= .10 if has else None,
              'all_locality_top_preserved': p['all_top_preserved'], 'mean_locality_kl': p['mean_kl'] <= .02,
              'max_locality_kl': p['max_kl'] <= .10,
              'mean_locality_nll_deterioration': p['mean_nll_deterioration'] <= .05,
              'max_locality_nll_deterioration': p['max_nll_deterioration'] <= .20}
    def comparison(control):
        return {**common, 'projected_kl_ratio': p['mean_kl'] <= .9 * control['mean_kl'],
                'projected_kl_difference': control['mean_kl'] - p['mean_kl'] >= 1e-6}
    equal = comparison(u)
    match = None
    if match_available:
        m = conditions['matched']
        gap = abs(p['mean_target_change'] - m['mean_target_change']) if has else None
        tolerance = max(.05, .25 * max(abs(p['mean_target_change']), abs(m['mean_target_change']))) if has else None
        match = {**comparison(m), 'unprojected_target_gain': m['mean_target_change'] > .25 if has else None,
                 'evaluated_target_gain_match': gap <= tolerance if has else None,
                 'evaluated_target_gap': gap, 'evaluated_target_tolerance': tolerance}
    def verify_criteria(field, expected):
        if expected is None:
            require(report[field] is None, field)
        else:
            require(set(report[field]) == set(expected), field + ' keys')
            for k, value in expected.items():
                check(report[field][k], value, field + ' ' + k)
    verify_criteria('equal_norm_criteria', equal)
    verify_criteria('matched_discovery_criteria', match)
    eligible = len(selected) == 8
    equal_decision = 'ineligible' if not eligible else ('success' if all(equal.values()) else 'negative')
    match_decision = ('ineligible' if not eligible or not match_available else
        'success' if all(v for k, v in match.items() if k not in ('evaluated_target_gap', 'evaluated_target_tolerance'))
        else 'negative')
    check(report['provisional_equal_norm_decision'], equal_decision, 'equal decision')
    check(report['provisional_matched_discovery_decision'], match_decision, 'matched decision')
    return {'schema': 'screened-evaluation-independent-scientific-audit-v1', 'status': 'passed',
            'scope': ('Saved-byte scientific/score comparison only; not proof of execution, model identity, '
                      'cleanup or independently authenticated chronology.'),
            'auditor_sha256': sha(Path(__file__).read_bytes()), 'case_sha256': sha(case_bytes),
            'screen_selection_sha256': sha(record_bytes), 'candidate_sha256': sha(candidate_bytes),
            'provisional_report_sha256': sha((data_dir / 'provisional-report.json').read_bytes()),
            'selected_offsets': selected, 'discovery_match': candidate['discovery_match'],
            'equal_norm_criteria': equal, 'matched_discovery_criteria': match,
            'equal_norm_decision': equal_decision, 'matched_discovery_decision': match_decision,
            'conditions': conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'screen-dir', 'candidate', 'data-dir', 'report'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.cases, args.screen_dir, args.candidate, args.data_dir)
    with args.report.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)
        file.write('\n')
    print('passed:', result['equal_norm_decision'], result['matched_discovery_decision'])


if __name__ == '__main__':
    main()
