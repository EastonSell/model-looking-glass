"""Fixed public year-comparison cases and saved-logit scoring."""

import math

YEARS = (12, 21, 32, 43, 54, 65, 76, 87)
EVENTS = ('war', 'reign')
TEMPLATES = {'forward': 'The {event} lasted from the year 17{year:02d} to the year 17',
             'reverse': 'The {event} lasted until the year 17{year:02d}, after starting in the year 17',
             'neutral': 'The {event} was recorded in the year 17{year:02d} beside the year 17'}
EVENT_IDS = {'war': 1175, 'reign': 13580}
DIGIT_IDS = (405, 486, 2999, 3070, 3023, 2713, 3312, 2998, 2919, 2931, 940, 1157, 1065, 1485, 1415, 1314, 1433, 1558,
             1507, 1129, 1238, 2481, 1828, 1954, 1731, 1495, 2075, 1983, 2078, 1959, 1270, 3132,
             2624, 2091, 2682, 2327, 2623, 2718, 2548, 2670, 1821, 3901, 3682, 3559, 2598, 2231,
             3510, 2857, 2780, 2920, 1120, 4349, 4309, 4310, 4051, 2816, 3980, 3553, 3365, 3270,
             1899, 5333, 5237, 5066, 2414, 2996, 2791, 3134, 3104, 3388, 2154, 4869, 4761, 4790,
             4524, 2425, 4304, 3324, 3695, 3720, 1795, 6659, 6469, 5999, 5705, 5332, 4521, 5774,
             3459, 4531, 3829, 6420, 5892, 6052, 5824, 3865, 4846, 5607, 4089, 2079)
TOKEN_TEMPLATES = {
    'forward': (464, None, 15436, 422, 262, 614, 1596, None, 284, 262, 614, 1596),
    'reverse': (464, None, 15436, 1566, 262, 614, 1596, None, 11, 706, 3599, 287, 262, 614, 1596),
    'neutral': (464, None, 373, 6264, 287, 262, 614, 1596, None, 13970, 262, 614, 1596),
}


def digit(value):
    return DIGIT_IDS[value]


def validate_cases(data):
    if (type(data) is not dict or data.get('schema') != 'distilgpt2-greater-than-cases-v1'
            or data.get('years') != list(YEARS) or data.get('events') != list(EVENTS)
            or data.get('templates') != TEMPLATES or data.get('numeric_ids') != list(DIGIT_IDS)
            or type(data.get('cases')) is not list or len(data['cases']) != 48):
        raise ValueError('case inventory differs from the fixed year-comparison design')
    expected = []
    for year in YEARS:
        for event in EVENTS:
            for condition, pattern in TEMPLATES.items():
                ids = list(TOKEN_TEMPLATES[condition])
                position = 8 if condition == 'neutral' else 7
                ids[1], ids[position] = EVENT_IDS[event], digit(year)
                expected.append({'case_id': f'greater-{year:02d}-{event}-{condition}',
                                 'year_suffix': year, 'event': event, 'condition': condition,
                                 'text': pattern.format(event=event, year=year), 'input_ids': ids,
                                 'year_position': position,
                                 'lower_ids': [digit(v) for v in range(year-8, year)],
                                 'upper_ids': [digit(v) for v in range(year+1, year+9)],
                                 'equal_id': digit(year)})
    if data['cases'] != expected:
        raise ValueError('expanded cases or token IDs differ from the fixed design')
    return expected


def validate_tokenizer(tokenizer, cases):
    for case in cases:
        if tokenizer.encode(case['text'], add_special_tokens=False) != case['input_ids']:
            raise ValueError('prompt tokenization differs')
        for value in range(1, 100):
            token = digit(value)
            if (tokenizer.encode(f'{value:02d}', add_special_tokens=False) != [token]
                    or tokenizer.encode(case['text'] + f'{value:02d}', add_special_tokens=False)
                    != case['input_ids'] + [token]):
                raise ValueError('candidate is not the expected single-token continuation')
        if (tokenizer.encode('00', add_special_tokens=False) != [digit(0)]
                or tokenizer.encode(case['text'] + '00', add_special_tokens=False)
                == case['input_ids'] + [digit(0)]):
            raise ValueError('expected 00 boundary mismatch changed')


def score_logits(logits, *, lower, upper, equal, numeric=None):
    values = [float(x) for x in logits]
    ids = list(lower) + list(upper) + [equal]
    if (not values or any(not math.isfinite(v) for v in values) or len(lower) != 8 or len(upper) != 8
            or len(set(ids)) != 17 or any(type(i) is not int or not 0 <= i < len(values) for i in ids)):
        raise ValueError('invalid logits or candidate IDs')
    lo, hi = [values[i] for i in lower], [values[i] for i in upper]
    ml, mh = max(lo), max(hi)
    log_lower = math.log(math.fsum(math.exp(v - ml) for v in lo))
    log_upper = math.log(math.fsum(math.exp(v - mh) for v in hi))
    ratio = (mh - ml) + log_upper - log_lower
    maximum = max(values)
    full_logsum = math.log(math.fsum(math.exp(v - maximum) for v in values))
    probability = lambda value: math.exp((value - maximum) - full_logsum)
    exp_ratio = math.exp(-abs(ratio))
    fraction = 1 / (1 + exp_ratio) if ratio >= 0 else exp_ratio / (1 + exp_ratio)
    result = {'log_mass_ratio': ratio, 'upper_conditional_fraction': fraction,
            'lower_full_mass': math.fsum(probability(v) for v in lo),
            'upper_full_mass': math.fsum(probability(v) for v in hi),
            'equal_full_mass': probability(values[equal]), 'lower_logits': lo,
            'upper_logits': hi, 'equal_logit': values[equal], 'top_token_id': values.index(maximum)}
    if numeric is not None:
        if (len(numeric) < 2 or len(set(numeric)) != len(numeric)
                or any(type(i) is not int or not 0 <= i < len(values) for i in numeric)):
            raise ValueError('invalid numeric-token inventory')
        aligned = math.fsum(probability(values[i]) for i in numeric[1:])
        zero = probability(values[numeric[0]])
        result.update({'aligned_01_99_full_mass': aligned, 'zero_00_next_token_mass': zero,
                       'other_vocabulary_mass': max(0.0, 1.0 - aligned - zero)})
    return result


def summarize(cases, metrics):
    if (type(metrics) is not dict or len(metrics) != 48
            or set(metrics) != {case['case_id'] for case in cases}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values())):
        raise ValueError('one finite log-mass ratio is required for each fixed case')
    groups = {}
    for case in cases:
        key = (case['year_suffix'], case['event'])
        groups.setdefault(key, {})[case['condition']] = metrics[case['case_id']]
    if len(groups) != 16 or any(set(x) != set(TEMPLATES) for x in groups.values()):
        raise ValueError('matched triples are incomplete')
    triples = []
    for year in YEARS:
        for event in EVENTS:
            values = groups[(year, event)]
            f, r, n = values['forward'], values['reverse'], values['neutral']
            triples.append({'year_suffix': year, 'event': event, 'forward': f, 'reverse': r, 'neutral': n,
                            'forward_minus_reverse': f-r, 'forward_minus_neutral': f-n,
                            'reverse_minus_neutral': r-n})
    fields = ('forward_minus_reverse', 'forward_minus_neutral', 'reverse_minus_neutral')
    return {'triples': triples,
            **{'mean_' + key: math.fsum(x[key] for x in triples) / 16 for key in fields},
            'positive_forward_minus_reverse': sum(x['forward_minus_reverse'] > 0 for x in triples)}
