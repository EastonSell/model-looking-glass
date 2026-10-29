"""Decimal oracle for closed v4 intervention scores; no model inputs or calls."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import struct
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--audit', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = args.data
audit_path = args.audit
audit_raw = audit_path.read_bytes()
if not hashlib.sha256(audit_raw).hexdigest() == 'eacf6827403103716c43ff9e78f2921e2cd3bcd405bd0a45cf4aa1ecce91061c':
    raise ValueError('saved-evidence check failed')
audit = json.loads(audit_raw)
evidence = audit['intervention']
terminal_raw = (data / 'terminal.json').read_bytes()
if not hashlib.sha256(terminal_raw).hexdigest() == evidence['terminal_sha256']:
    raise ValueError('saved-evidence check failed')
terminal = json.loads(terminal_raw)
if not (len(terminal['pairs']) == 4 and evidence['forwards'] == 102):
    raise ValueError('saved-evidence check failed')
maximum_error = {k: Decimal(0) for k in ('margin', 'dog_mass', 'cat_mass', 'off_mass', 'off_conditional_tv', 'movement')}
scores = {}
off_references = {}
observations = []
with localcontext() as context:
    context.prec = 60
    for n in range(1, 103):
        meta_raw = (data / f'forward-{n:03d}.json').read_bytes()
        meta = json.loads(meta_raw)
        payload = (data / meta['file']).read_bytes()
        expected = evidence['forward_hashes'][n - 1]
        if not (expected['submission'] == n and hashlib.sha256(meta_raw).hexdigest() == expected['json_sha256']):
            raise ValueError('saved-evidence check failed')
        if not hashlib.sha256(payload).hexdigest() == expected['binary_sha256'] == meta['sha256']:
            raise ValueError('saved-evidence check failed')
        entry = meta['layout']['logits']
        raw = payload[entry['offset']:entry['offset'] + entry['bytes']]
        if not (len(raw) == 4000 and entry['shape'] == [1, 1000]):
            raise ValueError('saved-evidence check failed')
        logits = [Decimal.from_float(v[0]) for v in struct.iter_unpack('<f', raw)]
        top = max(range(1000), key=logits.__getitem__)
        high = logits[top]
        weights = [(z - high).exp() for z in logits]
        dog = sum(weights[151:269], Decimal(0))
        cat = sum(weights[281:286], Decimal(0))
        off_weights = weights[:151] + weights[269:281] + weights[286:]
        off = sum(off_weights, Decimal(0))
        total = dog + cat + off
        calculated = {'margin': dog.ln() - cat.ln(), 'dog_mass': dog / total, 'cat_mass': cat / total, 'off_mass': off / total}
        reference = [w / off for w in off_weights]
        saved = meta['score']
        if not saved['top1_id'] == top:
            raise ValueError('saved-evidence check failed')
        for key, value in calculated.items():
            error = abs(Decimal.from_float(saved[key]) - value)
            maximum_error[key] = max(maximum_error[key], error)
            if not error < Decimal('1e-12'):
                raise ValueError((n, key, error))
        label = meta['label']
        parts = label.split(':')
        if not len(parts) == 3:
            raise ValueError('saved-evidence check failed')
        pair, side, kind = parts
        if kind == 'initial' and side in ('cat', 'dog'):
            off_references[pair, side] = reference
        if 'off_conditional_tv' in saved:
            ref = off_references[pair, side]
            tv = sum((abs(x - y) for x, y in zip(reference, ref)), Decimal(0)) / 2
            error = abs(Decimal.from_float(saved['off_conditional_tv']) - tv)
            maximum_error['off_conditional_tv'] = max(maximum_error['off_conditional_tv'], error)
            if not error < Decimal('1e-12'):
                raise ValueError((n, 'off_conditional_tv', error))
        scores[label] = calculated
    for pair in terminal['pairs']:
        pair_id = pair['pair_id']
        raw = (data / f'{pair_id}.json').read_bytes()
        if not hashlib.sha256(raw).hexdigest() == evidence['pair_hashes'][pair_id]:
            raise ValueError('saved-evidence check failed')
        for side in ('cat', 'dog'):
            baseline = scores[f'{pair_id}:{side}:initial']['margin']
            moves = {}
            for name in ('opposite', 'same_species', 'random'):
                new = scores[f'{pair_id}:{side}:matched_{name}']['margin']
                oriented = (new - baseline) * (1 if side == 'cat' else -1)
                recorded = pair['sides'][side]['conditions']['matched_' + name]['annotation_oriented_change']
                error = abs(Decimal.from_float(recorded) - oriented)
                maximum_error['movement'] = max(maximum_error['movement'], error)
                if not error < Decimal('1e-12'):
                    raise ValueError('saved-evidence check failed')
                moves[name] = oriented
            side_pass = moves['opposite'] >= Decimal('.05') and moves['opposite'] >= max(abs(moves['same_species']), abs(moves['random'])) + Decimal('.025')
            observations.append({'pair_id': pair_id, 'side': side, 'side_pass': side_pass, 'movements': {name: str(value) for name, value in moves.items()}})
    if not sum((row['side_pass'] for row in observations)) == 1:
        raise ValueError('saved-evidence check failed')
    if not [row['pair_id'] for row in observations if row['side_pass']] == ['pair-02']:
        raise ValueError('saved-evidence check failed')
    if not [row['side'] for row in observations if row['side_pass']] == ['dog']:
        raise ValueError('saved-evidence check failed')
out = {'schema': 'independent-v4-intervention-decimal-audit-v1', 'scope': '60-digit score and off-group-distribution check of saved bytes; not source, images, model, execution or chronology attestation.', 'independent_replay_sha256': hashlib.sha256(audit_raw).hexdigest(), 'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(), 'forwards': 102, 'max_absolute_error': {k: str(v) for k, v in maximum_error.items()}, 'sides': observations}
payload = (json.dumps(out, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
path = args.output
with path.open('xb') as f:
    if not f.write(payload) == len(payload):
        raise ValueError('saved-evidence check failed')
print(json.dumps({'forwards': 102, 'sides_passing': 1, 'max_absolute_error': out['max_absolute_error'], 'sha256': hashlib.sha256(payload).hexdigest()}))
