"""Fixed public explicit-direction cases and model-free scoring."""
import math

from . import greater_than_screen as prior
from .year_transfer import _same_typed, band_score

PAIRS = ((7, 16), (29, 38), (51, 60), (73, 82))
EVENTS = ('war', 'reign')
CONDITIONS = ('after', 'before', 'near')
ROLES = ('low', 'high')
WORDS = {'after': 706, 'before': 878, 'near': 1474}
MODEL_ID = 'distilbert/distilgpt2'
MODEL_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
TOKENIZER_SHA256 = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
PROTOCOL_REVISION = 'e5f71eb66e233a54f1171d1e6f489c0fca587156'
PROTOCOL_SHA256 = '68f028e401fd4c43d1cac89772da10c92cf5e5de7af30c4043c18c21678b902e'
EXPOSURE_VERSION = 'distilgpt2-development-exposures-v17'
EXPOSURE_SHA256 = 'e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a'
TEMPLATE = 'The {event} took place {condition} the year 17{year:02d} in the year 17'
TOKEN_TEMPLATE = (464, None, 1718, 1295, None, 262, 614, 1596, None, 287, 262, 614, 1596)
ACTIONS = ('self', 'year_final', 'year_at_zero', 'rotated_final')


def _case_id(pair_index, event, condition, role):
    return f'year-direction-{pair_index + 1:02d}-{event}-{condition}-{role}'


def make_inventory():
    cases = []
    for pair_index, (low, high) in enumerate(PAIRS):
        middle, above = list(range(low + 3, low + 6)), list(range(high + 3, high + 6))
        for event_index, event in enumerate(EVENTS):
            pair_noun_index = 2 * pair_index + event_index
            for condition_index, condition in enumerate(CONDITIONS):
                for year_index, (role, year) in enumerate(zip(ROLES, (low, high))):
                    tokens = list(TOKEN_TEMPLATE)
                    tokens[1], tokens[4], tokens[8] = prior.EVENT_IDS[event], WORDS[condition], prior.digit(year)
                    below, local_above = list(range(year-3, year)), list(range(year+1, year+4))
                    cases.append({
                        'case_id': _case_id(pair_index, event, condition, role),
                        'donor_case_id': _case_id(pair_index, event, condition, ROLES[1-year_index]),
                        'pair_index': pair_index, 'pair_noun_index': pair_noun_index,
                        'event': event, 'condition': condition, 'condition_index': condition_index,
                        'year_role': role, 'year_index': year_index, 'year_suffix': year,
                        'text': TEMPLATE.format(event=event, condition=condition, year=year),
                        'input_ids': tokens, 'year_position': 8, 'condition_position': 4,
                        'final_position': len(tokens)-1,
                        'middle_values': middle.copy(), 'above_values': above.copy(),
                        'middle_ids': [prior.digit(x) for x in middle],
                        'above_ids': [prior.digit(x) for x in above],
                        'local_below_values': below, 'local_above_values': local_above,
                        'local_below_ids': [prior.digit(x) for x in below],
                        'local_above_ids': [prior.digit(x) for x in local_above],
                        'rotation': (pair_noun_index + 2*condition_index + year_index) % 4,
                    })
    return {'schema': 'distilgpt2-explicit-year-direction-cases-v1', 'model_id': MODEL_ID,
            'model_revision': MODEL_REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
            'protocol_revision': PROTOCOL_REVISION, 'protocol_sha256': PROTOCOL_SHA256,
            'known_exposure_version': EXPOSURE_VERSION, 'known_exposure_sha256': EXPOSURE_SHA256,
            'pairs': [list(p) for p in PAIRS], 'events': list(EVENTS), 'conditions': list(CONDITIONS),
            'year_roles': list(ROLES), 'numeric_ids': list(prior.DIGIT_IDS), 'planned_forwards': 336,
            'max_logit_bytes': 67545408, 'max_block_snapshot_bytes': 47185920, 'cases': cases}


def validate_inventory(data):
    if not _same_typed(data, make_inventory()):
        raise ValueError('case inventory differs from the fixed explicit-direction protocol')
    rows = data['cases']
    if len({tuple(r['input_ids']) for r in rows}) != 48:
        raise ValueError('case inputs are not distinct')
    for start in range(0, 48, 6):
        group = rows[start:start+6]
        for index in range(0, 6, 2):
            a, b = group[index:index+2]
            if ([i for i, (x, y) in enumerate(zip(a['input_ids'], b['input_ids'])) if x != y] != [8]
                    or len(a['input_ids']) != len(b['input_ids'])):
                raise ValueError('matched year inputs differ outside their suffix')
        for role in ROLES:
            variants = [row for row in group if row['year_role'] == role]
            if any([i for i, (x, y) in enumerate(zip(variants[0]['input_ids'], row['input_ids'])) if x != y] != [4]
                   or len(variants[0]['input_ids']) != len(row['input_ids']) for row in variants[1:]):
                raise ValueError('wording inputs differ outside the temporal word')
    return rows


def validate_tokenizer(tokenizer, rows):
    if not _same_typed(rows, make_inventory()['cases']):
        raise ValueError('unexpected cases')
    def encode(text):
        result = tokenizer.encode(text, add_special_tokens=False)
        ids = result if type(result) is list else getattr(result, 'ids', None)
        if type(ids) is not list or any(type(value) is not int for value in ids):
            raise ValueError('unexpected tokenizer encoding')
        return ids
    if len(set(prior.DIGIT_IDS)) != 100:
        raise ValueError('numeric suffix IDs are not unique')
    for row in rows:
        if encode(row['text']) != row['input_ids'] or row['input_ids'][-1] != 1596 or len(row['input_ids']) > 32:
            raise ValueError('prompt does not match the fixed tokenizer')
        for value in range(1, 100):
            token = prior.digit(value)
            if (encode(f'{value:02d}') != [token]
                    or encode(row['text'] + f'{value:02d}') != row['input_ids'] + [token]):
                raise ValueError('numeric suffix is not a matching contextual token')
        if (encode('00') != [prior.digit(0)]
                or encode(row['text']+'00') == row['input_ids']+[prior.digit(0)]):
            raise ValueError('the separately reported 00 boundary changed')
        if (len(set(row['middle_ids'] + row['above_ids'])) != 6
                or len(set(row['local_below_ids'] + row['local_above_ids'])) != 6):
            raise ValueError('candidate IDs are not distinct within a comparison')


def score_logits(logits, row):
    values = tuple(float(x) for x in logits)
    if len(values) != 50257 or any(not math.isfinite(x) for x in values):
        raise ValueError('expected finite full-vocabulary logits')
    middle = band_score(values, row['middle_ids'], row['above_ids'])
    local = band_score(values, row['local_above_ids'], row['local_below_ids'])
    maximum = max(values)
    weights = tuple(math.exp(x-maximum) for x in values)
    total = math.fsum(weights)
    numeric = frozenset(prior.DIGIT_IDS)
    aligned = math.fsum(weights[i] for i in prior.DIGIT_IDS[1:]) / total
    zero = weights[prior.digit(0)] / total
    other = math.fsum(weight for i, weight in enumerate(weights) if i not in numeric) / total
    return {'band_score': middle, 'local_score': local, 'aligned_01_99_full_mass': aligned,
            'zero_00_next_token_mass': zero, 'other_vocabulary_mass': other,
            'top_token_id': values.index(maximum)}


def _valid_metrics(rows, metrics):
    if (not _same_typed(rows, make_inventory()['cases']) or type(metrics) is not dict
            or set(metrics) != {row['case_id'] for row in rows}
            or any(type(value) is not dict or any(type(value.get(k)) not in (int, float)
                       or not math.isfinite(value[k]) for k in ('band_score', 'local_score'))
                   for value in metrics.values())):
        raise ValueError('one finite pair of fixed scores is required per case')


def summarize_clean(rows, metrics):
    _valid_metrics(rows, metrics)
    groups = []
    for pair_index, _ in enumerate(PAIRS):
        for event in EVENTS:
            local = {c: {r: metrics[_case_id(pair_index, event, c, r)]['local_score'] for r in ROLES}
                     for c in CONDITIONS}
            fixed = {c: metrics[_case_id(pair_index, event, c, 'low')]['band_score']
                     - metrics[_case_id(pair_index, event, c, 'high')]['band_score'] for c in CONDITIONS}
            checks = {'after_low': local['after']['low'] > 0.5, 'after_high': local['after']['high'] > 0.5,
                      'before_low': local['before']['low'] < -0.5, 'before_high': local['before']['high'] < -0.5,
                      'D_after': fixed['after'] > 0.5, 'D_before': fixed['before'] < -0.5}
            local_differences = {f'{left}_minus_{right}': {role: local[left][role]-local[right][role]
                                                           for role in ROLES}
                                 for left, right in (('after', 'before'), ('after', 'near'), ('before', 'near'))}
            fixed_differences = {f'{left}_minus_{right}': fixed[left]-fixed[right]
                                 for left, right in (('after', 'before'), ('after', 'near'), ('before', 'near'))}
            groups.append({'pair_index': pair_index, 'event': event, 'local_scores': local,
                           'local_wording_differences': local_differences, 'fixed_band_D': fixed,
                           'fixed_band_wording_differences': fixed_differences,
                           'checks': checks, 'comparator_pass': all(checks.values())})
    passed = [g for g in groups if g['comparator_pass']]
    check_counts = {key: sum(group['checks'][key] for group in groups) for key in groups[0]['checks']}
    return {'pairs': groups, 'check_counts': check_counts, 'comparator_gate_count': len(passed),
            'comparator_gate_passed': len(passed) >= 6 and len({g['pair_index'] for g in passed}) >= 3}


def summarize_interventions(rows, metrics, action_scores):
    clean = summarize_clean(rows, metrics)
    if (type(action_scores) is not dict or set(action_scores) != set(metrics)
            or any(type(actions) is not dict or set(actions) != set(ACTIONS) for actions in action_scores.values())
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for actions in action_scores.values() for x in actions.values())
            or any(actions['self'] != metrics[key]['band_score'] for key, actions in action_scores.items())):
        raise ValueError('one finite fixed-band score for each action is required')
    groups = []
    for pair_index, _ in enumerate(PAIRS):
        for event in EVENTS:
            conditions = {}
            for condition in CONDITIONS:
                low, high = (_case_id(pair_index, event, condition, role) for role in ROLES)
                actions = {}
                for action in ACTIONS:
                    lo = action_scores[low][action] - metrics[low]['band_score']
                    hi = action_scores[high][action] - metrics[high]['band_score']
                    actions[action] = {'low_E': lo, 'high_E': hi, 'low_donorward': -lo,
                                       'high_donorward': hi, 'donorward_mean': (-lo+hi)/2}
                conditions[condition] = actions
            contrasts = {a: conditions['after'][a]['donorward_mean'] - conditions['before'][a]['donorward_mean']
                         for a in ACTIONS}
            generic = all(conditions[c]['year_final']['low_donorward'] > 0
                          and conditions[c]['year_final']['high_donorward'] > 0
                          and conditions[c]['year_final']['donorward_mean']
                              > conditions[c]['year_at_zero']['donorward_mean'] + 0.1
                          and conditions[c]['year_final']['donorward_mean']
                              > conditions[c]['rotated_final']['donorward_mean'] + 0.1 for c in CONDITIONS)
            after, before = conditions['after']['year_final'], conditions['before']['year_final']
            directional = (after['low_donorward'] > 0 and after['high_donorward'] > 0
                           and before['low_donorward'] < 0 and before['high_donorward'] < 0)
            controlled = (directional and contrasts['year_final'] > contrasts['year_at_zero'] + 0.1
                          and contrasts['year_final'] > contrasts['rotated_final'] + 0.1)
            groups.append({'pair_index': pair_index, 'event': event, 'conditions': conditions,
                           'after_minus_before': contrasts, 'generic_pattern': generic,
                           'comparator_direction': directional, 'comparator_control': controlled})
    counts = {key: sum(g[key] for g in groups) for key in
              ('generic_pattern', 'comparator_direction', 'comparator_control')}
    means = {c: {a: math.fsum(g['conditions'][c][a]['donorward_mean'] for g in groups)/8 for a in ACTIONS}
             for c in CONDITIONS}
    contrast_means = {a: math.fsum(g['after_minus_before'][a] for g in groups)/8 for a in ACTIONS}
    eligible = clean['comparator_gate_passed']
    return {'pairs': groups, 'mean_donorward_by_condition_and_action': means,
            'mean_after_minus_before_by_action': contrast_means,
            'generic_pattern_count': counts['generic_pattern'], 'generic_pattern_supported': counts['generic_pattern'] >= 6,
            'comparator_direction_count': counts['comparator_direction'],
            'comparator_direction_supported': counts['comparator_direction'] >= 6 if eligible else None,
            'comparator_control_count': counts['comparator_control'],
            'comparator_control_supported': counts['comparator_control'] >= 6 if eligible else None}
