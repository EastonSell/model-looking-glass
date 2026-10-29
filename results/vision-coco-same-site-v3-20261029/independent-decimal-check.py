"""Sixty-digit recomputation of closed v3 logits; never opens images or a model."""
import argparse
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True, help='directory of extracted data/ records')
data = parser.parse_args().data
terminal = json.loads((data / 'terminal.json').read_bytes())
records = {}
for i in range(1, terminal['saved_forwards'] + 1):
    row = json.loads((data / f'forward-{i:03d}.json').read_bytes())
    assert row['label'] not in records
    records[row['label']] = row


def logits(label):
    row = records[label]
    raw = (data / row['file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
    item = row['layout']['logits']
    result = raw[item['offset']:item['offset'] + item['bytes']]
    assert len(result) == 4000 and item['shape'] == [1, 1000]
    return [x[0] for x in struct.iter_unpack('<f', result)]


def score(label):
    raw = logits(label)
    assert len(raw) == 1000 and all(math.isfinite(x) for x in raw)
    with localcontext() as ctx:
        ctx.prec = 60
        values = [Decimal.from_float(x) for x in raw]
        maximum = max(values)
        weights = [(x - maximum).exp() for x in values]
        dog, cat, total = sum(weights[151:269]), sum(weights[281:286]), sum(weights)
        return {'margin': (dog / cat).ln(), 'dog_mass': dog / total, 'cat_mass': cat / total}


statuses = Counter()
max_score_error, max_change_error = 0, 0
for pair in terminal['pairs']:
    pair_id = pair['pair_id']
    clean = {side: score(f'{pair_id}:{side}:initial') for side in ('cat', 'dog')}
    gap = clean['dog']['margin'] - clean['cat']['margin']
    eligible = (clean['cat']['margin'] < 0 and clean['dog']['margin'] > 0 and gap >= 1 and
                clean['cat']['cat_mass'] >= Decimal('.2') and clean['dog']['dog_mass'] >= Decimal('.2'))
    assert eligible is pair['baseline_eligibility']['eligible']
    movements = {}
    for side in ('cat', 'dog'):
        saved = pair['sides'][side]
        for key in ('margin', 'cat_mass', 'dog_mass'):
            max_score_error = max(max_score_error, abs(float(clean[side][key]) - saved['baseline'][key]))
            assert math.isclose(float(clean[side][key]), saved['baseline'][key], rel_tol=1e-12, abs_tol=1e-12)
        movements[side] = {}
        for kind in ('opposite', 'same_species', 'random'):
            current = score(f'{pair_id}:{side}:matched_{kind}')
            oriented = (current['margin'] - clean[side]['margin']) * (1 if side == 'cat' else -1)
            expected = saved['conditions']['matched_' + kind]['annotation_oriented_change']
            max_change_error = max(max_change_error, abs(float(oriented) - expected))
            assert math.isclose(float(oriented), expected, rel_tol=1e-12, abs_tol=1e-12)
            movements[side][kind] = oriented
    passed = all(movements[s]['opposite'] >= Decimal('.05') and
                 movements[s]['opposite'] >= max(abs(movements[s]['same_species']),
                                                 abs(movements[s]['random'])) + Decimal('.025')
                 for s in ('cat', 'dog'))
    status = ('positive' if passed else 'negative') if eligible else 'baseline-ineligible'
    assert status == pair['status']
    statuses[status] += 1
qualified, positive = statuses['positive'] + statuses['negative'], statuses['positive']
result = 'inconclusive' if qualified < 4 else 'positive' if 4 * positive >= 3 * qualified else 'negative'
assert terminal['aggregate'] == {'status': result, 'selected': 8, 'evaluable': qualified, 'positive': positive}
print(json.dumps({'result': result, 'statuses': statuses, 'initial_logits': 16, 'matched_logits': 48,
                  'max_abs_score_error': max_score_error, 'max_abs_movement_error': max_change_error,
                  'scope': '60-digit decimal check of saved logits; no model or image access'}))
