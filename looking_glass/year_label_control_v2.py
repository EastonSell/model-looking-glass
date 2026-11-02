"""Fixed exposed-family selection and descriptive v2 scoring."""
import math

from . import year_label_control as v1

ACTIONS = v1.ACTIONS
CONDITIONS = v1.CONDITIONS
LABEL_PAIRS = v1.LABEL_PAIRS
ROLES = v1.ROLES
PROTOCOL_REVISION = '0e302e93baad1d15605ace6c2794a171a22bfb7a'
PROTOCOL_SHA256 = '3edb6f8736f98a21a467a0028e7781a531ed5a43558451147ff6e5c50836a49f'
BASELINE_VERSION = v1.BASELINE_VERSION
BASELINE_SHA256 = v1.BASELINE_SHA256
score_logits = v1.score_logits


def select_cases(full_rows):
    if not v1._same_typed(full_rows, v1.make_inventory()['cases']):
        raise ValueError('the complete v1 case inventory differs')
    return full_rows[12:]


def validate_inventory(data):
    return select_cases(v1.validate_inventory(data))


def validate_tokenizer(tokenizer, rows):
    if not v1._same_typed(rows, select_cases(v1.make_inventory()['cases'])):
        raise ValueError('unexpected selected cases')
    v1.validate_tokenizer(tokenizer, v1.make_inventory()['cases'])


def _valid_clean(rows, clean):
    expected = select_cases(v1.make_inventory()['cases'])
    if (not v1._same_typed(rows, expected) or type(clean) is not dict
            or set(clean) != {row['case_id'] for row in rows}
            or any(type(value) is not dict or any(type(value.get(k)) not in (int, float)
                       or not math.isfinite(value[k]) for k in ('band_score', 'local_score'))
                   for value in clean.values())):
        raise ValueError('one finite clean score pair is required per selected case')


def summarize_clean(rows, clean):
    _valid_clean(rows, clean)
    families, differences = [], []
    for offset, index in enumerate((1, 2, 3)):
        family_rows = rows[offset*12:(offset+1)*12]
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
            'mean_band_score': math.fsum(clean[row['case_id']]['band_score'] for row in rows)/len(rows),
            'mean_local_score': math.fsum(clean[row['case_id']]['local_score'] for row in rows)/len(rows),
            'mean_low_minus_high': math.fsum(differences)/len(differences)}


def summarize_interventions(rows, clean, actions):
    _valid_clean(rows, clean)
    expected = {row['case_id'] for row in rows}
    if (type(actions) is not dict or set(actions) != expected
            or any(type(value) is not dict or set(value) != set(ACTIONS)
                   or any(type(x) not in (int, float) or not math.isfinite(x) for x in value.values())
                   for value in actions.values())
            or any(actions[key]['self'] != clean[key]['band_score'] for key in expected)):
        raise ValueError('one finite clean score and five action scores are required per case')
    families = []
    for offset, index in enumerate((1, 2, 3)):
        targets = []
        for row in rows[offset*12:(offset+1)*12]:
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
                                          if target['condition'] == condition)/12
                         for name in ACTIONS} for condition in CONDITIONS}
    donorward = {condition: math.fsum(target['year_donorward'] for target in targets
                                      if target['condition'] == condition)/12
                 for condition in CONDITIONS}
    return {'families': families, 'mean_effect_by_condition': means,
            'mean_year_donorward_by_condition': donorward,
            'pattern_count': count, 'pattern_supported': count == 3}
