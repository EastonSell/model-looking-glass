"""Fixed public year-transfer inventory and clean behavioral gates.

This module does not load or execute a model. Inputs are public exploratory
examples selected after the earlier year screen and before this evaluation.
"""
import math

from . import greater_than_screen as prior

PAIRS = ((5, 14), (27, 36), (49, 58), (71, 80))
EVENTS = ('war', 'reign')
CONDITIONS = ('forward', 'reverse', 'neutral')
ROLES = ('low', 'high')
MODEL_ID = 'distilbert/distilgpt2'
MODEL_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
TOKENIZER_SHA256 = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
PROTOCOL_REVISION = '38ffcab52256e3952315b7bd9b58d6f10d7921c2'
PROTOCOL_SHA256 = '6f31697b09b2b68bf701494b4ceedde185b9f8501aea61caf50b647edac1fee7'
EXPOSURE_VERSION = 'distilgpt2-development-exposures-v15'
EXPOSURE_SHA256 = '6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34'


def _case_id(pair_index, event, condition, role):
    return f'year-transfer-{pair_index + 1:02d}-{event}-{condition}-{role}'


def make_inventory():
    cases = []
    for pair_index, (low, high) in enumerate(PAIRS):
        middle = list(range(low + 3, low + 6))
        above = list(range(high + 3, high + 6))
        for event_index, event in enumerate(EVENTS):
            pair_noun_index = 2 * pair_index + event_index
            for condition_index, condition in enumerate(CONDITIONS):
                for year_index, (role, year) in enumerate(zip(ROLES, (low, high))):
                    positions = list(prior.TOKEN_TEMPLATES[condition])
                    year_position = 8 if condition == 'neutral' else 7
                    positions[1] = prior.EVENT_IDS[event]
                    positions[year_position] = prior.digit(year)
                    cases.append({
                        'case_id': _case_id(pair_index, event, condition, role),
                        'donor_case_id': _case_id(pair_index, event, condition, ROLES[1-year_index]),
                        'pair_index': pair_index, 'pair_noun_index': pair_noun_index,
                        'event': event, 'condition': condition, 'condition_index': condition_index,
                        'year_role': role, 'year_index': year_index, 'year_suffix': year,
                        'input_ids': positions, 'year_position': year_position,
                        'final_position': len(positions)-1,
                        'text': prior.TEMPLATES[condition].format(event=event, year=year),
                        'middle_values': middle.copy(), 'above_values': above.copy(),
                        'middle_ids': [prior.digit(x) for x in middle],
                        'above_ids': [prior.digit(x) for x in above],
                        'rotation': (pair_noun_index + 2*condition_index + year_index) % 4,
                    })
    return {
        'schema': 'distilgpt2-year-transfer-cases-v1', 'model_id': MODEL_ID,
        'model_revision': MODEL_REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
        'protocol_revision': PROTOCOL_REVISION, 'protocol_sha256': PROTOCOL_SHA256,
        'known_exposure_version': EXPOSURE_VERSION, 'known_exposure_sha256': EXPOSURE_SHA256,
        'pairs': [list(pair) for pair in PAIRS], 'events': list(EVENTS),
        'conditions': list(CONDITIONS), 'year_roles': list(ROLES),
        'gate_only_forwards': 96, 'gate_pass_forwards': 336,
        'max_logit_bytes': 67545408, 'max_block_snapshot_bytes': 47185920,
        'cases': cases,
    }


def _same_typed(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(type(k) is str and _same_typed(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(_same_typed(x, y) for x, y in zip(a, b))
    return a == b


def validate_inventory(data):
    expected = make_inventory()
    if not _same_typed(data, expected):
        raise ValueError('case inventory differs from the fixed year-transfer protocol')
    rows = data['cases']
    if len({tuple(row['input_ids']) for row in rows}) != 48:
        raise ValueError('case inputs are not distinct')
    for index in range(0, len(rows), 2):
        low, high = rows[index:index+2]
        changes = [i for i, (a, b) in enumerate(zip(low['input_ids'], high['input_ids'])) if a != b]
        if (len(low['input_ids']) != len(high['input_ids']) or changes != [low['year_position']]
                or low['input_ids'][-1] != 1596 or len(low['input_ids']) > 32):
            raise ValueError('matched prompts do not have the specified token relationship')
    return rows


def validate_tokenizer(tokenizer, rows):
    """Validate a Rust tokenizer or Hugging Face tokenizer without a model."""
    if not _same_typed(rows, make_inventory()['cases']):
        raise ValueError('unexpected cases')
    def encode(text):
        result = tokenizer.encode(text, add_special_tokens=False)
        ids = result if type(result) is list else getattr(result, 'ids', None)
        if type(ids) is not list or any(type(item) is not int for item in ids):
            raise ValueError('unexpected tokenizer encoding')
        return ids
    for row in rows:
        if encode(row['text']) != row['input_ids']:
            raise ValueError('prompt does not match the fixed tokenizer')
        for value, token_id in zip(row['middle_values'] + row['above_values'],
                                   row['middle_ids'] + row['above_ids']):
            if (encode(f'{value:02d}') != [token_id]
                    or encode(row['text'] + f'{value:02d}') != row['input_ids'] + [token_id]):
                raise ValueError('candidate is not a matching single-token continuation')


def band_score(logits, middle, above):
    values = [float(x) for x in logits]
    indices = list(middle) + list(above)
    if (not values or any(not math.isfinite(x) for x in values)
            or len(middle) != 3 or len(above) != 3 or len(set(indices)) != 6
            or any(type(i) is not int or not 0 <= i < len(values) for i in indices)):
        raise ValueError('invalid logits or fixed candidate bands')
    lo = [values[i] for i in middle]
    hi = [values[i] for i in above]
    u, v = max(lo), max(hi)
    a = math.fsum(math.exp(x-u) for x in lo)
    b = math.fsum(math.exp(x-v) for x in hi)
    return ((u-v)+math.log(a))-math.log(b)


def summarize_clean(rows, scores):
    expected = make_inventory()['cases']
    if (not _same_typed(rows, expected) or type(scores) is not dict
            or set(scores) != {row['case_id'] for row in expected}
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores.values())):
        raise ValueError('one finite fixed-band score is required per case')
    groups = []
    for pair_index, _ in enumerate(PAIRS):
        for event in EVENTS:
            delta = {condition: scores[_case_id(pair_index, event, condition, 'low')]
                     - scores[_case_id(pair_index, event, condition, 'high')]
                     for condition in CONDITIONS}
            f, r, n = (delta[x] for x in CONDITIONS)
            groups.append({'pair_index': pair_index, 'event': event, **delta,
                           'forward_minus_neutral': f-n,
                           'comparator_pass': f > 0.5 and r < -0.5,
                           'preference_pass': f > 0.5 and n > 0.5 and f-n > 0.1})
    def gate(key):
        passed = [group for group in groups if group[key]]
        return len(passed) >= 6 and len({group['pair_index'] for group in passed}) >= 3
    return {'pairs': groups, 'comparator_gate_passed': gate('comparator_pass'),
            'preference_gate_passed': gate('preference_pass'),
            'comparator_gate_count': sum(group['comparator_pass'] for group in groups),
            'preference_gate_count': sum(group['preference_pass'] for group in groups)}
