"""Fixed public year/label donor cases and model-free scoring."""
import math
from . import greater_than_screen as prior
from .explicit_year_direction import score_logits
from .year_transfer import _same_typed

PAIRS = ((9, 18), (54, 63))
LABEL_PAIRS = (('alpha', 'beta'), ('red', 'blue'))
LABEL_IDS = {'alpha': 17130, 'beta': 12159, 'red': 2266, 'blue': 4171}
CONDITIONS = ('after', 'before', 'near')
CONDITION_IDS = {'after': 706, 'before': 878, 'near': 1474}
ROLES = ('low', 'high')
ACTIONS = ('self', 'year_final', 'label_final', 'year_at_zero', 'rotated_year_final')
TEMPLATE = 'The event tagged {label} took place {condition} the year 17{year:02d} in the year 17'
TOKEN_TEMPLATE = (464, 1785, 30509, None, 1718, 1295, None, 262, 614, 1596, None, 287, 262, 614, 1596)
PROTOCOL_REVISION = 'd35d685950e36e91fc5cf576c03fe65e8d02aac3'
PROTOCOL_SHA256 = '5da05c4d35708b420e5843b1eb77fda8699c74755233b510a01683ead2081f4a'
BASELINE_VERSION = 'distilgpt2-development-exposures-v24'
BASELINE_SHA256 = '877fa0376925be34ac3de14ff066e5465a67953660bce81011e9c8fd87810bd3'


def _case_id(pair_index, labels_index, condition, label, role):
    return f'year-label-{pair_index + 1:02d}-{labels_index + 1:02d}-{condition}-{label}-{role}'


def make_inventory():
    cases = []
    for pair_index, (low, high) in enumerate(PAIRS):
        middle, above = list(range(low + 3, low + 6)), list(range(high + 3, high + 6))
        for labels_index, labels in enumerate(LABEL_PAIRS):
            family_index = 2 * pair_index + labels_index
            for condition_index, condition in enumerate(CONDITIONS):
                for label_index, label in enumerate(labels):
                    for anchor_index, (role, year) in enumerate(zip(ROLES, (low, high))):
                        tokens = list(TOKEN_TEMPLATE)
                        tokens[3], tokens[6], tokens[10] = LABEL_IDS[label], CONDITION_IDS[condition], prior.digit(year)
                        below, local_above = list(range(year - 3, year)), list(range(year + 1, year + 4))
                        cases.append({
                            'case_id': _case_id(pair_index, labels_index, condition, label, role),
                            'year_donor_case_id': _case_id(pair_index, labels_index, condition, label, ROLES[1-anchor_index]),
                            'label_donor_case_id': _case_id(pair_index, labels_index, condition, labels[1-label_index], role),
                            'pair_index': pair_index, 'label_pair_index': labels_index, 'family_index': family_index,
                            'condition': condition, 'condition_index': condition_index,
                            'label': label, 'label_index': label_index,
                            'year_role': role, 'anchor_index': anchor_index, 'year_suffix': year,
                            'text': TEMPLATE.format(label=label, condition=condition, year=year),
                            'input_ids': tokens, 'label_position': 3, 'condition_position': 6,
                            'year_position': 10, 'final_position': 14,
                            'middle_values': middle.copy(), 'above_values': above.copy(),
                            'middle_ids': [prior.digit(x) for x in middle],
                            'above_ids': [prior.digit(x) for x in above],
                            'local_below_values': below, 'local_above_values': local_above,
                            'local_below_ids': [prior.digit(x) for x in below],
                            'local_above_ids': [prior.digit(x) for x in local_above],
                            'rotation': (family_index + 2*condition_index + label_index + 3*anchor_index) % 5,
                        })
    return {'schema': 'distilgpt2-year-label-control-cases-v1',
            'model_id': 'distilbert/distilgpt2',
            'model_revision': '2290a62682d06624634c1f46a6ad5be0f47f38aa',
            'tokenizer_sha256': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
            'protocol_revision': PROTOCOL_REVISION, 'protocol_sha256': PROTOCOL_SHA256,
            'known_exposure_version': BASELINE_VERSION, 'known_exposure_sha256': BASELINE_SHA256,
            'pairs': [list(pair) for pair in PAIRS], 'label_pairs': [list(pair) for pair in LABEL_PAIRS],
            'conditions': list(CONDITIONS), 'actions': list(ACTIONS), 'numeric_ids': list(prior.DIGIT_IDS),
            'planned_forwards': 384, 'max_logit_bytes': 77194752, 'max_block_snapshot_bytes': 56623104,
            'cases': cases}


def validate_inventory(data):
    if not _same_typed(data, make_inventory()):
        raise ValueError('case inventory differs from the fixed year-label protocol')
    rows = data['cases']
    by_id = {row['case_id']: row for row in rows}
    if len(by_id) != 48 or len({tuple(row['input_ids']) for row in rows}) != 48:
        raise ValueError('case inputs or identities are not distinct')
    for row in rows:
        for name, position in (('year_donor_case_id', 10), ('label_donor_case_id', 3)):
            donor = by_id[row[name]]
            if (len(row['input_ids']) != len(donor['input_ids'])
                    or [i for i, (x, y) in enumerate(zip(row['input_ids'], donor['input_ids'])) if x != y] != [position]
                    or donor[name] != row['case_id']):
                raise ValueError('donor does not make the fixed one-token change')
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
    if (encode('00') != [prior.digit(0)]
            or tokenizer.decode([prior.digit(0)], skip_special_tokens=False) != '00'):
        raise ValueError('the separate 00 next-token identity changed')
    for row in rows:
        if (encode(row['text']) != row['input_ids'] or row['input_ids'][-1] != 1596
                or len(row['input_ids']) > 32):
            raise ValueError('prompt does not match the fixed tokenizer')
        for value in range(1, 100):
            token = prior.digit(value)
            if (encode(f'{value:02d}') != [token]
                    or encode(row['text'] + f'{value:02d}') != row['input_ids'] + [token]):
                raise ValueError('numeric suffix is not a matching contextual token')
        if encode(row['text'] + '00') == row['input_ids'] + [prior.digit(0)]:
            raise ValueError('the separately reported 00 boundary changed')
        if (len(set(row['middle_ids'] + row['above_ids'])) != 6
                or len(set(row['local_below_ids'] + row['local_above_ids'])) != 6):
            raise ValueError('candidate IDs are not distinct within a comparison')


def _valid_clean(rows, clean):
    expected_ids = {row['case_id'] for row in rows}
    if (not _same_typed(rows, make_inventory()['cases']) or type(clean) is not dict
            or set(clean) != expected_ids
            or any(type(value) is not dict or any(type(value.get(k)) not in (int, float)
                       or not math.isfinite(value[k]) for k in ('band_score', 'local_score'))
                   for value in clean.values())):
        raise ValueError('one finite clean score pair is required per case')


def summarize_clean(rows, clean):
    _valid_clean(rows, clean)
    families = []
    differences = []
    for index in range(4):
        family_rows = rows[index * 12:(index + 1) * 12]
        conditions = {}
        for condition in CONDITIONS:
            labels = {}
            for label in LABEL_PAIRS[index % 2]:
                pair = [row for row in family_rows if row['condition'] == condition and row['label'] == label]
                low, high = (clean[row['case_id']] for row in pair)
                difference = low['band_score'] - high['band_score']
                differences.append(difference)
                labels[label] = {'low': dict(low), 'high': dict(high), 'low_minus_high': difference}
            conditions[condition] = labels
        families.append({'family_index': index, 'conditions': conditions})
    return {'families': families,
            'mean_band_score': math.fsum(clean[row['case_id']]['band_score'] for row in rows) / len(rows),
            'mean_local_score': math.fsum(clean[row['case_id']]['local_score'] for row in rows) / len(rows),
            'mean_low_minus_high': math.fsum(differences) / len(differences)}


def summarize_interventions(rows, clean, actions):
    _valid_clean(rows, clean)
    expected_ids = {row['case_id'] for row in rows}
    if (type(actions) is not dict or set(actions) != expected_ids
            or any(type(value) is not dict or set(value) != set(ACTIONS)
                   or any(type(x) not in (int, float) or not math.isfinite(x) for x in value.values())
                   for value in actions.values())
            or any(actions[key]['self'] != clean[key]['band_score'] for key in expected_ids)):
        raise ValueError('one finite clean score and five action scores are required per case')
    families = []
    for index in range(4):
        targets = []
        for row in rows[index * 12:(index + 1) * 12]:
            key = row['case_id']
            effects = {name: actions[key][name] - clean[key]['band_score'] for name in ACTIONS}
            year = -effects['year_final'] if row['year_role'] == 'low' else effects['year_final']
            controls = ('label_final', 'year_at_zero', 'rotated_year_final')
            margins = {name: year - abs(effects[name]) for name in controls}
            checks = {name: year > abs(effects[name]) + 0.1 for name in controls}
            targets.append({'case_id': key, 'condition': row['condition'], 'label': row['label'],
                            'year_role': row['year_role'], 'effects': effects, 'year_donorward': year,
                            'margins': margins, 'checks': checks, 'pass': all(checks.values())})
        families.append({'family_index': index, 'targets': targets,
                         'pass': all(target['pass'] for target in targets)})
    count = sum(family['pass'] for family in families)
    targets = [target for family in families for target in family['targets']]
    means = {condition: {name: math.fsum(target['effects'][name] for target in targets
                                          if target['condition'] == condition) / 16
                         for name in ACTIONS} for condition in CONDITIONS}
    donorward = {condition: math.fsum(target['year_donorward'] for target in targets
                                      if target['condition'] == condition) / 16
                 for condition in CONDITIONS}
    return {'families': families, 'mean_effect_by_condition': means,
            'mean_year_donorward_by_condition': donorward,
            'pattern_count': count, 'pattern_supported': count >= 3}
