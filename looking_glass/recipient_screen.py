"""Fixed public DistilGPT-2 recipient screen; no model access is needed here."""

import math

PAIRS = (('John', 'Mary'), ('Michael', 'Sarah'), ('Daniel', 'Emma'), ('Peter', 'Lisa'),
         ('Alice', 'Claire'), ('Anna', 'Susan'), ('Paul', 'Mark'), ('David', 'Robert'))
NAME_IDS = (1757, 5335, 3899, 10490, 7806, 18966, 5613, 15378,
            14862, 20758, 11735, 14919, 3362, 2940, 3271, 5199)
TEMPLATES = ('When {first} and {second} went to the store, {giver} gave a book to',
             'After {first} and {second} went to the park, {giver} handed a gift to')
TOKEN_TEMPLATES = ((2215, None, 290, None, 1816, 284, 262, 3650, 11, None, 2921, 257, 1492, 284),
                   (3260, None, 290, None, 1816, 284, 262, 3952, 11, None, 10158, 257, 6979, 284))


def validate_cases(data):
    if (type(data) is not dict or data.get('schema') != 'distilgpt2-recipient-cases-v1'
            or data.get('pairs') != [list(p) for p in PAIRS]
            or data.get('templates') != list(TEMPLATES)
            or type(data.get('cases')) is not list or len(data['cases']) != 64):
        raise ValueError('case inventory does not match the fixed public design')
    expected = []
    for pair, names in enumerate(PAIRS):
        tokens = NAME_IDS[2 * pair:2 * pair + 2]
        for template, pattern in enumerate(TEMPLATES):
            for order in (0, 1):
                order_indices = (0, 1) if order == 0 else (1, 0)
                for giver in (0, 1):
                    first, second = (names[i] for i in order_indices)
                    ids = list(TOKEN_TEMPLATES[template])
                    ids[1], ids[3], ids[9] = tokens[order_indices[0]], tokens[order_indices[1]], tokens[giver]
                    expected.append({'case_id': f'recipient-{pair+1:02d}-{template}-{order}-{giver}',
                                     'pair': pair, 'template': template, 'order': order,
                                     'giver_index': giver,
                                     'text': pattern.format(first=first, second=second, giver=names[giver]),
                                     'input_ids': ids, 'giver_id': tokens[giver],
                                     'recipient_id': tokens[1-giver]})
    if data['cases'] != expected:
        raise ValueError('expanded cases or token IDs differ from the fixed design')
    return expected


def validate_tokenizer(tokenizer, cases):
    for case in cases:
        names = PAIRS[case['pair']]
        if tokenizer.encode(case['text'], add_special_tokens=False) != case['input_ids']:
            raise ValueError('prompt tokenization does not match the fixed input')
        for index, name in enumerate(names):
            token = NAME_IDS[2 * case['pair'] + index]
            if (tokenizer.encode(' ' + name, add_special_tokens=False) != [token]
                    or tokenizer.encode(case['text'] + ' ' + name, add_special_tokens=False)
                    != case['input_ids'] + [token]):
                raise ValueError('answer continuation is not the expected single token')


def score_logits(logits, *, recipient, giver):
    values = [float(x) for x in logits]
    if (not values or any(not math.isfinite(x) for x in values)
            or type(recipient) is not int or type(giver) is not int
            or recipient == giver or not (0 <= recipient < len(values) and 0 <= giver < len(values))):
        raise ValueError('invalid logits or answer IDs')
    maximum = max(values)
    log_sum = math.log(math.fsum(math.exp(v - maximum) for v in values))
    def rank(index):
        return 1 + sum(v > values[index] or (v == values[index] and i < index)
                       for i, v in enumerate(values))
    return {'recipient_logit': values[recipient], 'giver_logit': values[giver],
            'margin': values[recipient] - values[giver],
            'recipient_log_probability': (values[recipient] - maximum) - log_sum,
            'giver_log_probability': (values[giver] - maximum) - log_sum,
            'recipient_rank': rank(recipient), 'giver_rank': rank(giver),
            'top_token_id': values.index(maximum)}


def summarize(cases, margins):
    if (type(margins) is not dict or len(margins) != 64
            or set(margins) != {c['case_id'] for c in cases}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in margins.values())):
        raise ValueError('one finite margin is required for every fixed case')
    groups = {}
    for c in cases:
        groups.setdefault((c['pair'], c['template'], c['order']), []).append(margins[c['case_id']])
    if len(groups) != 32 or any(len(values) != 2 for values in groups.values()):
        raise ValueError('incomplete matched pairs')
    paired = {key: all(value > 0 for value in values) for key, values in groups.items()}
    by_template = [sum(paired[(p, t, o)] for p in range(8) for o in (0, 1)) for t in (0, 1)]
    by_name_pair = [sum(paired[(p, t, o)] for t in (0, 1) for o in (0, 1)) for p in range(8)]
    both_positive = sum(paired.values())
    sufficient_names = sum(x >= 3 for x in by_name_pair)
    return {'positive_cases': sum(value > 0 for value in margins.values()),
            'both_positive_pairs': both_positive, 'template_both_positive': by_template,
            'name_pair_both_positive': by_name_pair,
            'name_pairs_meeting_requirement': sufficient_names,
            'mean_margin': math.fsum(margins[c['case_id']] for c in cases) / 64,
            'gate_passed': both_positive >= 24 and min(by_template) >= 12 and sufficient_names >= 6}
