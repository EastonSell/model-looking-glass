"""Summarize already-audited saved vision records without a model forward.

This is a presentation of the pinned replayed pair bytes, not a second replay or
an authentication of the source, images, execution, chronology or secrecy.
Baseline-ineligible effects are descriptive and have no primary denominator.
"""
import collections
import hashlib
import json
import math
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


CONFIGS = (
    {
        'split': 'development', 'ids': [f'pair-{i:02d}' for i in range(4)],
        'data': Path('work/vision-coco-development-evidence-v2/data'),
        'audit': Path('work/vision-coco-development-v2-independent-audit.json'),
        'audit_sha256': '4b26c66c53c709c02a79b3a3a1bdfb2068b0f4dbd71247833eb4d3f594bd815a',
        'terminal_sha256': 'a2ae5f0280168609e5f5b6f26d3e3bbef3183718906099568187e2fff6626b71',
        'forwards': 114,
    },
    {
        'split': 'heldout', 'ids': [f'pair-{i:02d}' for i in range(4, 12)],
        'data': Path('work/vision-coco-heldout-evidence-v1/data'),
        'audit': Path('work/vision-coco-heldout-v1-independent-audit.json'),
        'audit_sha256': '60567da0f6db38555ccaa905e4414c079f0d817e4a87f87394c234527fbadab9',
        'terminal_sha256': '2b61dfc59a8ab6c1072293e97d8e8ad42ceea8c4c1a0956bc9e7c992c2f3a9cc',
        'forwards': 228,
    },
)


def summarize(config):
    raw = config['audit'].read_bytes()
    require(sha(raw) == config['audit_sha256'], 'audit bytes changed')
    audit = json.loads(raw)
    require(audit['split'] == config['split'] and not audit['synthetic'] and
            audit['forwards'] == config['forwards'] and
            audit['terminal_sha256'] == config['terminal_sha256'], 'audit binding mismatch')
    require(sha((config['data'] / 'terminal.json').read_bytes()) == config['terminal_sha256'],
            'terminal bytes changed')
    require(set(audit['pair_hashes']) == set(config['ids']), 'pair inventory mismatch')
    by_id = {p['pair_id']: p for p in audit['pairs']}
    require(set(by_id) == set(config['ids']), 'audit pair identities mismatch')
    counters = collections.Counter()
    q_values, relative_errors, token_doses = [], [], []
    relative_receiver = collections.defaultdict(list)
    pairs = []
    for pair_id in config['ids']:
        pair_raw = (config['data'] / (pair_id + '.json')).read_bytes()
        require(sha(pair_raw) == audit['pair_hashes'][pair_id], 'pair bytes changed')
        pair = json.loads(pair_raw)
        checked = by_id[pair_id]
        require(pair['pair_id'] == pair_id and pair['split'] == config['split'] and
                pair['status'] == checked['status'] and
                pair['baseline_eligibility'] == checked['baseline_eligibility'] and
                pair['dose_matchability'] == checked['dose_statuses'], 'pair audit mismatch')
        gap, eligible = pair['baseline_eligibility']['gap'], pair['baseline_eligibility']['eligible']
        counters['pair_' + pair['status']] += 1
        sides = {}
        for side in ('cat', 'dog'):
            item = pair['sides'][side]
            dose, conditions = item['dose'], item['conditions']
            counters['dose_' + dose['status']] += 1
            counters['same_species_' + dose['same_species_status']] += 1
            for name in conditions:
                counters['condition_' + name] += 1
            if dose['status'] == 'matchable':
                q = dose['q']
                require(q > 0, 'nonpositive matched norm')
                q_values.append(q)
                for name in ('matched_inside', 'matched_outside', 'matched_random', 'matched_same_species'):
                    if name not in conditions:
                        continue
                    condition = conditions[name]
                    relative_errors.append(abs(condition['actual_dose'] - q) / q)
                    token_doses.extend(condition['token_doses'])
                    if condition['q_over_receiver_norm'] is not None:
                        relative_receiver[name].append(condition['q_over_receiver_norm'])
            decision = None
            if checked['movements'] is not None:
                inside = checked['movements'][side]['inside']
                outside = checked['movements'][side]['outside']
                decision = {'inside': inside, 'outside': outside,
                            'minimum_inside': .02 * gap,
                            'minimum_advantage': .01 * gap,
                            'actual_inside_minus_abs_outside': inside - abs(outside),
                            'passes_inside': inside >= .02 * gap,
                            'passes_advantage': inside - abs(outside) >= .01 * gap}
            # Every condition and norm is retained, including negative changes.
            sides[side] = {'baseline': item['baseline'], 'dose': dose,
                           'conditions': conditions, 'primary_direction': decision}
        if checked['movements'] is not None:
            passes = all(sides[s]['primary_direction']['passes_inside'] and
                         sides[s]['primary_direction']['passes_advantage'] for s in ('cat', 'dog'))
            require((pair['status'] == 'positive') == passes, 'decision mismatch')
        pairs.append({'pair_id': pair_id, 'pair_sha256': sha(pair_raw), 'status': pair['status'],
                      'baseline_eligibility': pair['baseline_eligibility'], 'sides': sides})
    require(bool(q_values) and bool(relative_errors), 'no matched doses')
    return {'split': config['split'], 'audit_sha256': config['audit_sha256'],
            'terminal_sha256': config['terminal_sha256'], 'forwards': config['forwards'],
            'aggregate': audit['aggregate'], 'counts': dict(sorted(counters.items())),
            'matched_q_range': [min(q_values), max(q_values)],
            'max_absolute_relative_dose_error': max(relative_errors),
            'token_dose_range': [min(token_doses), max(token_doses)],
            'q_over_receiver_norm_ranges_by_condition': {
                name: {'sides': len(values), 'range': [min(values), max(values)]}
                for name, values in sorted(relative_receiver.items())},
            'pairs': pairs}


def main():
    result = {'schema': 'vision-coco-audited-descriptive-summary-v1',
              'scope': 'Presentation of audited saved records; baseline-ineligible results are descriptive only. '
                       'No execution, input authenticity, chronology, pretraining exposure or population claim.',
              'splits': [summarize(config) for config in CONFIGS]}
    raw = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    path = Path('work/vision-coco-both-splits-descriptive-summary-v2.json')
    with path.open('xb') as output:
        output.write(raw)
    print('summary', path, sha(raw))
    for split in result['splits']:
        print('\n', split['split'], 'aggregate', split['aggregate'], 'counts', split['counts'])
        print('q', split['matched_q_range'], 'max relative q error', split['max_absolute_relative_dose_error'],
              'token doses', split['token_dose_range'], 'q/receiver by condition',
              split['q_over_receiver_norm_ranges_by_condition'])
        for pair in split['pairs']:
            values = []
            for side in ('cat', 'dog'):
                s = pair['sides'][side]
                c = s['conditions']
                values.append(f"{side}:mass={s['baseline'][side+'_mass']:.6f} "
                              f"inside={c['matched_inside']['annotation_oriented_change']:+.6f} "
                              f"outside={c['matched_outside']['annotation_oriented_change']:+.6f} "
                              f"random={c['matched_random']['annotation_oriented_change']:+.6f} "
                              f"same={c['matched_same_species']['annotation_oriented_change']:+.6f}"
                              if 'matched_same_species' in c else
                              f"{side}:mass={s['baseline'][side+'_mass']:.6f} "
                              f"inside={c['matched_inside']['annotation_oriented_change']:+.6f} "
                              f"outside={c['matched_outside']['annotation_oriented_change']:+.6f} "
                              f"random={c['matched_random']['annotation_oriented_change']:+.6f} "
                              f"same=unavailable({s['dose']['same_species_status']})")
            print(pair['pair_id'], pair['status'], pair['baseline_eligibility']['reasons'], 'gap',
                  round(pair['baseline_eligibility']['gap'], 6), ' | '.join(values))


if __name__ == '__main__':
    main()
