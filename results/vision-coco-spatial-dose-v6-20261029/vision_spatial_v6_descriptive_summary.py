"""Recompute all v6 three-dose side effects from saved float32 logits and landed states.

Decimal arithmetic is independent of the producing score implementation. The
separate intervention replay must still validate donor vectors, selected sites,
complete controls, status aggregation, provenance, and execution chronology.
"""
from decimal import Decimal, localcontext
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = None
OUT = None
TERMINAL = None
SIDES = ('cat', 'dog')
LEVELS = ('low', 'mid', 'high')
ARMS = ('opposite', 'same_species', 'random')


def require(value, msg):
    if not value:
        raise ValueError(msg)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def float32_values(raw):
    require(len(raw) % 4 == 0, 'invalid float32 byte length')
    result = []
    for (word,) in struct.iter_unpack('<I', raw):
        exp, frac = (word >> 23) & 255, word & 0x7fffff
        require(exp != 255, 'nonfinite float32')
        mantissa = frac if exp == 0 else frac + (1 << 23)
        value = math.ldexp(float(mantissa), -149 if exp == 0 else exp - 150)
        result.append(-value if word >> 31 else value)
    return result


def read_records():
    terminal_raw = (ROOT/'terminal.json').read_bytes()
    require(sha(terminal_raw) == TERMINAL, 'terminal hash differs')
    terminal = json.loads(terminal_raw)
    count = terminal['submissions']
    require(type(count) is int and 0 < count <= 256 and count == terminal['saved_forwards'], 'forward count differs')
    records = {}
    for i in range(1, count+1):
        row = json.loads((ROOT/f'forward-{i:03d}.json').read_bytes())
        require(row['submission'] == i and row['file'] == f'forward-{i:03d}.bin' and
                row['label'] not in records, 'forward identity differs')
        raw = (ROOT/row['file']).read_bytes()
        require(sha(raw) == row['sha256'], 'forward payload hash differs')
        spans = {}
        offset = 0
        for name, item in sorted(row['layout'].items(), key=lambda v: v[1]['offset']):
            require(item['offset'] == offset and item['bytes'] == 4*math.prod(item['shape']), 'tensor layout differs')
            spans[name] = raw[offset:offset + item['bytes']]
            offset += item['bytes']
        require(offset == len(raw) and row['layout']['logits']['shape'] == [1, 1000], 'payload length differs')
        records[row['label']] = spans
    return terminal, records


def distribution(raw):
    z = [Decimal.from_float(v) for v in float32_values(raw)]
    require(len(z) == 1000, 'logit width differs')
    maximum = max(z)
    weights = [(v-maximum).exp() for v in z]
    denominator = sum(weights)
    probs = [v/denominator for v in weights]
    dog, cat = list(range(151,269)), list(range(281,286))
    off = list(range(151)) + list(range(269,281)) + list(range(286,1000))
    dog_sum, cat_sum = sum(weights[i] for i in dog), sum(weights[i] for i in cat)
    off_sum = sum(weights[i] for i in off)
    return {'z': z, 'probs': probs, 'logz': maximum + denominator.ln(),
            'margin': dog_sum.ln()-cat_sum.ln(),
            'cat_mass': cat_sum/denominator, 'dog_mass': dog_sum/denominator,
            'off_mass': off_sum/denominator, 'off_conditional': [weights[i]/off_sum for i in off],
            'top1_id': max(range(1000), key=z.__getitem__)}


def comparison(clean, patched):
    tv = sum(abs(a-b) for a,b in zip(clean['probs'],patched['probs']))/2
    off_tv = sum(abs(a-b) for a,b in zip(clean['off_conditional'],patched['off_conditional']))/2
    kl = sum(p*(a-b+patched['logz']-clean['logz'])
             for p,a,b in zip(clean['probs'],clean['z'],patched['z']))
    return {'margin': float(patched['margin']), 'cat_mass': float(patched['cat_mass']),
            'dog_mass': float(patched['dog_mass']), 'off_mass': float(patched['off_mass']),
            'tv': float(tv), 'off_conditional_tv': float(off_tv), 'kl_clean_to_patched': float(kl),
            'top1_id': patched['top1_id'], 'top1_changed': clean['top1_id'] != patched['top1_id']}


def near(a,b,why):
    require(type(a) in (int,float) and type(b) in (int,float) and
            math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-12), why)


def main():
    global ROOT, OUT, TERMINAL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terminal-sha256', required=True)
    args = parser.parse_args()
    ROOT, OUT, TERMINAL = args.data, args.output, args.terminal_sha256
    terminal, records = read_records()
    require(type(terminal.get('pairs')) is list and 1 <= len(terminal['pairs']) <= 4, 'pair inventory differs')
    result, max_residual = [], 0.0
    with localcontext() as ctx:
        ctx.prec = 60
        for ordinal, terminal_pair in enumerate(terminal['pairs']):
            pair_id = f'pair-{ordinal:02d}'
            pair = json.loads((ROOT / f'{pair_id}.json').read_bytes())
            require(pair == terminal_pair and pair['pair_id'] == pair_id, 'pair or terminal differs')
            for side in SIDES:
                saved = pair['sides'][side]
                clean = distribution(records[f'{pair_id}:{side}:initial']['logits'])
                base = saved['baseline']
                for key in ('margin', 'cat_mass', 'dog_mass', 'off_mass'):
                    near(float(clean[key]), base[key], 'baseline score differs')
                require(clean['top1_id'] == base['top1_id'], 'baseline top1 differs')
                dose = saved['dose']
                matchable = dose['status'] in ('matchable', 'limited-dose')
                names = ['repeat', 'null', 'self', 'full_self', 'full_swap']
                if matchable:
                    names.extend(f'{level}_{arm}' for level in LEVELS for arm in ARMS)
                names.extend(['final_spatial', 'post_clean'])
                require(set(names) == set(saved['conditions']), 'condition inventory differs')
                movements = {level: {} for level in LEVELS} if matchable else None
                conditions = {}
                for name in names:
                    payload = records[f'{pair_id}:{side}:{name}']
                    score = comparison(clean, distribution(payload['logits']))
                    recorded = saved['conditions'][name]
                    for key in ('margin', 'cat_mass', 'dog_mass', 'off_mass', 'tv', 'off_conditional_tv', 'kl_clean_to_patched'):
                        near(score[key], recorded[key], f'{name} {key} differs')
                        max_residual = max(max_residual, abs(score[key] - recorded[key]))
                    require(score['top1_id'] == recorded['top1_id'] and
                            score['top1_changed'] is recorded['top1_changed'], 'top1 differs')
                    direction = (score['margin'] - float(clean['margin'])) * (1 if side == 'cat' else -1)
                    if 'annotation_oriented_change' in recorded:
                        near(direction, recorded['annotation_oriented_change'], f'{name} movement differs')
                    condition = {**score, 'oriented_movement': direction}
                    if any(name.startswith(level + '_') for level in LEVELS):
                        level, arm = name.split('_', 1)
                        q = dose['levels'][level]
                        deltas, per_token = [], []
                        for i in range(4):
                            before, after = float32_values(payload[f'before_{i}']), float32_values(payload[f'after_{i}'])
                            require(len(before) == len(after) == 192, 'token witness width differs')
                            diff = [b-a for a,b in zip(before,after)]
                            deltas.extend(diff)
                            per_token.append(math.sqrt(math.fsum(v*v for v in diff)))
                        actual = math.sqrt(math.fsum(v*v for v in deltas))
                        receiver = dose['receiver_norm']
                        near(actual, recorded['actual_dose'], 'actual dose differs')
                        near(actual / receiver, recorded['realized_over_receiver_norm'], 'dose fraction differs')
                        for a, b in zip(per_token, recorded['token_doses']):
                            near(a, b, 'per-token dose differs')
                        movements[level][arm] = direction
                        condition.update(actual_dose=actual, dose_over_q=actual/q,
                                         dose_over_receiver_norm=actual/receiver, token_doses=per_token)
                    conditions[name] = condition
                comparison_values = None
                if matchable:
                    adjusted = {level: vals['opposite'] - max(abs(vals['same_species']),abs(vals['random']))
                                for level,vals in movements.items()}
                    low_pass = movements['low']['opposite'] >= .05 and adjusted['low'] >= .025
                    side_pass = (not low_pass and movements['high']['opposite'] >= .10 and
                                 adjusted['high'] >= .05 and adjusted['high']-adjusted['low'] >= .025)
                    comparison_values = {'adjusted': adjusted,
                        'raw_high_minus_low': movements['high']['opposite']-movements['low']['opposite'],
                        'adjusted_high_minus_low': adjusted['high']-adjusted['low'],
                        'low_directional_pass': low_pass, 'side_pass': side_pass}
                    for level in LEVELS:
                        near(adjusted[level], saved['comparison']['adjusted'][level], 'adjusted effect differs')
                        for arm in ARMS:
                            near(movements[level][arm], saved['movements'][level][arm], 'movement differs')
                    for key in ('raw_high_minus_low','adjusted_high_minus_low'):
                        near(comparison_values[key], saved['comparison'][key], f'{key} differs')
                    require(all(comparison_values[key] is saved['comparison'][key] for key in
                                ('low_directional_pass','side_pass')), 'decision boundary differs')
                result.append({'pair_id':pair_id, 'pair_status':pair['status'], 'side':side,
                               'image_id':saved['image_id'], 'same_species_donor_id':saved['same_species_donor_id'],
                               'baseline':{k:float(clean[k]) for k in ('margin','cat_mass','dog_mass','off_mass')},
                               'dose':dose, 'movements':movements, 'comparison':comparison_values,
                               'conditions':conditions})
    report = {'schema':'vision-spatial-v6-descriptive-from-raw-v1', 'terminal_sha256':TERMINAL,
              'aggregate_recorded':terminal['aggregate'], 'forwards_checked':len(records), 'sides':result,
              'max_decimal_vs_recorded_residual':max_residual,
              'scope':'Decimal score/distribution and landed token norms; not a donor-direction, full patch/control, '
                      'classification, source, execution or chronology attestation.'}
    raw = (json.dumps(report,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()
    with OUT.open('xb') as out:
        require(out.write(raw) == len(raw), 'short report write')
    print(json.dumps({'report_sha256':sha(raw), 'sides':len(result), 'forwards':len(records),
                      'recorded_aggregate':terminal['aggregate'], 'max_residual':max_residual}))


if __name__ == '__main__':
    main()
