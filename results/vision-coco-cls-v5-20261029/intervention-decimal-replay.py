"""Recompute saved v5 class margins and full-distribution diagnostics with Decimal.

This independently checks recorded logits and their summaries, not the model,
execution, image provenance, patch landings, or dose reconstruction.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_float32_bits(raw):
    require(len(raw) == 4000, 'logit payload is not 1,000 binary32 values')
    values = []
    for (word,) in struct.iter_unpack('<I', raw):
        exponent, fraction = (word >> 23) & 255, word & 0x7fffff
        require(exponent != 255, 'nonfinite binary32')
        mantissa = fraction if exponent == 0 else (1 << 23) | fraction
        number = Decimal(mantissa) * (Decimal(2) ** (-149 if exponent == 0 else exponent - 150))
        values.append(-number if word >> 31 else number)
    return values


def distribution(raw):
    z = read_float32_bits(raw)
    peak = max(z)
    e = [(x - peak).exp() for x in z]
    total = sum(e)
    p = [x / total for x in e]
    dog = sum(e[151:269])
    cat = sum(e[281:286])
    off = sum(e[:151]) + sum(e[269:281]) + sum(e[286:])
    return {'z': z, 'p': p, 'margin': dog.ln() - cat.ln(),
            'dog_mass': dog / total, 'cat_mass': cat / total, 'off_mass': off / total,
            'top1_id': z.index(peak)}


def effects(clean, patched):
    p, q = clean['p'], patched['p']
    tv = sum(abs(a - b) for a, b in zip(p, q)) / 2
    kl = sum(a * (a.ln() - b.ln()) for a, b in zip(p, q))
    indices = list(range(151)) + list(range(269, 281)) + list(range(286, 1000))
    off_tv = sum(abs(p[i] / clean['off_mass'] - q[i] / patched['off_mass']) for i in indices) / 2
    return {'tv': tv, 'kl_clean_to_patched': kl, 'off_conditional_tv': off_tv}


def near(recorded, computed, label, errors):
    require(type(recorded) in (int, float) and math.isfinite(recorded), f'{label}: invalid recorded value')
    error = abs(Decimal.from_float(float(recorded)) - computed)
    errors[label] = max(errors.get(label, Decimal(0)), error)
    require(error <= Decimal('1e-10'), f'{label}: mismatch {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    terminal = json.loads((args.data / 'terminal.json').read_bytes())
    count = terminal['submissions']
    require(type(count) is int and count == terminal['saved_forwards'] and 0 < count <= 256,
            'invalid forward count')
    records = {}
    errors = {}
    with localcontext() as ctx:
        ctx.prec = 60
        for index in range(1, count + 1):
            name = f'forward-{index:03d}'
            row = json.loads((args.data / (name + '.json')).read_bytes())
            raw = (args.data / (name + '.bin')).read_bytes()
            require(row['submission'] == index and row['file'] == name + '.bin' and
                    hashlib.sha256(raw).hexdigest() == row['sha256'], 'record identity/hash differs')
            layout = row['layout']['logits']
            require(layout == {'offset': 0, 'bytes': 4000, 'shape': [1, 1000]}, 'logit layout differs')
            label = row['label']
            require(label not in records, 'duplicate forward label')
            dist = distribution(raw[:4000])
            for key in ('margin', 'dog_mass', 'cat_mass', 'off_mass'):
                near(row['score'][key], dist[key], key, errors)
            require(row['score']['top1_id'] == dist['top1_id'], 'top-one differs')
            records[label] = {'score': dist, 'raw': raw[:4000], 'row': row}
        summaries = []
        positives = 0
        evaluable = 0
        for pair in terminal['pairs']:
            pairid = pair['pair_id']
            sides = []
            for side in ('cat', 'dog'):
                info = pair['sides'][side]
                base = records[f'{pairid}:{side}:initial']['score']
                for name, saved in info['conditions'].items():
                    record = records[f'{pairid}:{side}:{name}']
                    for key in ('margin', 'dog_mass', 'cat_mass', 'off_mass'):
                        near(saved[key], record['score'][key], 'condition_' + key, errors)
                    require(saved['top1_id'] == record['score']['top1_id'], 'condition top-one differs')
                    measured = effects(base, record['score'])
                    for key in ('tv', 'kl_clean_to_patched', 'off_conditional_tv'):
                        near(saved[key], measured[key], key, errors)
                if info['dose']['status'] != 'matchable':
                    sides.append({'side': side, 'dose_status': info['dose']['status']})
                    continue
                moves = {}
                for site in ('cls', 'spatial'):
                    moves[site] = {}
                    for kind in ('opposite', 'same_species', 'random'):
                        name = f'{site}_{kind}'
                        measured = (records[f'{pairid}:{side}:{name}']['score']['margin'] - base['margin'])
                        measured *= Decimal(1 if side == 'cat' else -1)
                        near(info['movements'][site][kind], measured, 'movement', errors)
                        moves[site][kind] = measured
                adj = {site: v['opposite'] - max(abs(v['same_species']), abs(v['random']))
                       for site, v in moves.items()}
                delta = adj['cls'] - adj['spatial']
                passed = (moves['cls']['opposite'] >= Decimal('.05') and
                          adj['cls'] >= Decimal('.025') and delta >= Decimal('.025'))
                require(passed is info['comparison']['side_pass'], 'side decision differs')
                for key, measured in [('adjusted_cls', adj['cls']), ('adjusted_spatial', adj['spatial']),
                                      ('adjusted_difference', delta),
                                      ('raw_opposite_difference', moves['cls']['opposite'] - moves['spatial']['opposite'])]:
                    near(info['comparison'][key], measured, key, errors)
                sides.append({'side': side, 'dose_status': 'matchable', 'side_pass': passed,
                              'movements': {site: {k: float(v) for k, v in values.items()}
                                            for site, values in moves.items()},
                              'adjusted_cls': float(adj['cls']), 'adjusted_spatial': float(adj['spatial']),
                              'adjusted_difference': float(delta)})
            if all(s['dose_status'] == 'matchable' for s in sides):
                evaluable += 1
                positive = all(s['side_pass'] for s in sides)
                positives += positive
                require(pair['status'] == ('positive' if positive else 'negative'), 'pair decision differs')
            summaries.append({'pair_id': pairid, 'status': pair['status'], 'sides': sides})
        status = ('inconclusive' if evaluable < 2 else
                  'positive' if positives * 4 >= evaluable * 3 else 'negative')
        require(terminal['aggregate'] == {'status': status, 'selected': len(summaries),
                                          'evaluable': evaluable, 'positive': positives},
                'aggregate differs')
        result = {'schema': 'vision-cls-v5-decimal-replay-v1', 'precision_digits': ctx.prec,
                  'scope': 'saved logits and numerical decision consistency only; '
                           'no model, source, image, execution, landing or dose attestation',
                  'terminal_sha256': hashlib.sha256((args.data / 'terminal.json').read_bytes()).hexdigest(),
                  'forward_count': count, 'aggregate': terminal['aggregate'], 'pairs': summaries,
                  'max_absolute_errors': {k: float(v) for k, v in sorted(errors.items())}}
    out = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.report.open('xb') as stream:
        if stream.write(out) != len(out):
            raise OSError('short report write')
    print(json.dumps({'report_sha256': hashlib.sha256(out).hexdigest(),
                      'forwards': count, 'aggregate': result['aggregate'],
                      'max_absolute_errors': result['max_absolute_errors']}))


if __name__ == '__main__':
    main()
