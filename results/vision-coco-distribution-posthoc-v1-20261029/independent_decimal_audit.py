"""Recompute a published saved-logit report with independent Decimal arithmetic.

This checks retained-byte consistency; it does not attest execution, model
identity, independent samples, or the origin of the images.
"""
import argparse
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import tarfile

def digest(data):
    return hashlib.sha256(data).hexdigest()

def check(condition, message):
    if not condition:
        raise ValueError(message)

def close(actual, expected, label):
    check(type(actual) in (int, float) and math.isfinite(actual) and
          math.isclose(actual, float(expected), abs_tol=1e-12, rel_tol=1e-12), label)
    return abs(actual - float(expected))

def decode(raw):
    check(len(raw) == 4000, 'expected 1000 binary32 logits')
    values = []
    for (word,) in struct.iter_unpack('<I', raw):
        exponent, fraction = (word >> 23) & 255, word & 0x7fffff
        check(exponent != 255, 'nonfinite saved binary32')
        mantissa = fraction if exponent == 0 else (1 << 23) | fraction
        value = math.ldexp(mantissa, -149 if exponent == 0 else exponent - 150)
        values.append(-value if word >> 31 else value)
    return values

def probabilities(logits):
    values = [Decimal.from_float(value) for value in logits]
    maximum = max(values)
    weights = [(value - maximum).exp() for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]

def cosine(a, b):
    aa, bb = sum(x*x for x in a), sum(x*x for x in b)
    return None if not aa or not bb else sum(x*y for x, y in zip(a, b)) / (aa*bb).sqrt()

def load_selected(archive_path, index, index_raw):
    expected = {item['path']: (item['bytes'], item['sha256']) for item in index['members']}
    check(len(expected) == len(index['members']) == 708, 'index member inventory')
    check('evidence-index.json' not in expected, 'index duplicates itself')
    expected['evidence-index.json'] = (len(index_raw), digest(index_raw))
    selected, pending, seen = {}, {}, set()
    regex = re.compile(r'(pair-(?:0[4-9]|1[01])):(cat|dog):(initial|matched_inside|matched_outside|matched_random)\Z')
    with tarfile.open(archive_path, mode='r|xz') as tar:
        for item in tar:
            check(item.isfile() and item.name in expected and item.name not in seen, 'unexpected archive member')
            size, expected_hash = expected[item.name]
            check(type(size) is int and 0 <= size <= 512 * 1024 and item.size == size, 'member length')
            stream = tar.extractfile(item)
            check(stream is not None, 'unreadable archive member')
            raw = stream.read(size + 1)
            check(len(raw) == size and digest(raw) == expected_hash, 'member hash')
            seen.add(item.name)
            found = re.fullmatch(r'heldout/data/forward-(\d{3})\.(bin|json)', item.name)
            if not found:
                continue
            number, extension = int(found.group(1)), found.group(2)
            check(1 <= number <= 228, 'forward index')
            if extension == 'bin':
                check(number not in pending and len(raw) >= 4000 and len(pending) < 228, 'forward binary')
                pending[number] = (raw[:4000], digest(raw))
            else:
                check(number in pending, 'missing forward binary')
                record = json.loads(raw)
                head, payload_hash = pending.pop(number)
                check(record.get('submission') == number and record.get('sha256') == payload_hash and
                      record.get('file') == f'forward-{number:03d}.bin' and
                      record.get('layout', {}).get('logits') == {'offset': 0, 'bytes': 4000, 'shape': [1, 1000]},
                      'forward binding')
                label = record.get('label')
                match = regex.fullmatch(label) if type(label) is str else None
                if match:
                    key = match.groups()
                    check(key not in selected, 'duplicate selected forward')
                    selected[key] = (head, number, payload_hash)
    check(seen == set(expected) and not pending and len(selected) == 64, 'incomplete saved inventory')
    return selected

def audit(archive, index_path, manifest_path, report_path):
    index_raw, manifest_raw = index_path.read_bytes(), manifest_path.read_bytes()
    with gzip.open(report_path, 'rb') as compressed:
        report_raw = compressed.read(16 * 1024 * 1024 + 1)
    check(len(report_raw) <= 16 * 1024 * 1024, 'report exceeds bound')
    report = json.loads(report_raw)
    pins = report['inputs']
    check(digest(index_raw) == pins['index_sha256'] and digest(manifest_raw) == pins['manifest_sha256'], 'input pin')
    h, size = hashlib.sha256(), 0
    with archive.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
            size += len(block)
    check(h.hexdigest() == pins['archive_sha256'] and size == pins['archive_bytes'], 'archive pin')
    index, manifest = json.loads(index_raw), json.loads(manifest_raw)
    cat = tuple(x['id'] for x in manifest['class_groups']['cat'])
    dog = tuple(x['id'] for x in manifest['class_groups']['dog'])
    check(len(cat) == 5 and len(dog) == 118 and len(set(cat) | set(dog)) == 123, 'class groups')
    animal = set(cat) | set(dog)
    selected = load_selected(archive, index, index_raw)
    rows = report['rows']
    keys = {(f'pair-{i:02d}', side) for i in range(4, 12) for side in ('cat', 'dog')}
    check(len(rows) == 16 and {(r['pair_id'], r['side']) for r in rows} == keys, 'report row inventory')
    maxima = dict(probability=0.0, tv=0.0, off_share=0.0, oriented_margin=0.0, cosine=0.0)
    gates = {}
    with localcontext() as context:
        context.prec = 60
        for row in rows:
            pair, side = row['pair_id'], row['side']
            clean_raw, number, binary_hash = selected[(pair, side, 'initial')]
            check(row['clean_submission'] == number and row['clean_binary_sha256'] == binary_hash, 'clean binding')
            clean = decode(clean_raw)
            p = probabilities(clean)
            check(len(row['probabilities']['clean']) == 1000, 'clean probability length')
            for value, expected in zip(row['probabilities']['clean'], p):
                maxima['probability'] = max(maxima['probability'], close(value, expected, 'clean probability'))
            cm, dm = sum(p[i] for i in cat), sum(p[i] for i in dog)
            margin = (dm / cm).ln()
            base = row['baseline']
            for name, expected in (('cat_mass', cm), ('dog_mass', dm), ('off_mass', 1-cm-dm), ('margin', margin)):
                close(base[name], expected, 'baseline ' + name)
            check(base['top1_id'] == max(range(1000), key=clean.__getitem__), 'baseline top-1')
            gates.setdefault(pair, {})[side] = (margin, cm, dm, row['eligible'])
            changes = {}
            for name in ('matched_inside', 'matched_outside', 'matched_random'):
                raw, number, binary_hash = selected[(pair, side, name)]
                condition = row['conditions'][name]
                check(condition['submission'] == number and condition['binary_sha256'] == binary_hash, 'patch binding')
                logits, q = decode(raw), probabilities(decode(raw))
                check(len(row['probabilities'][name]) == 1000, 'patch probability length')
                for value, expected in zip(row['probabilities'][name], q):
                    maxima['probability'] = max(maxima['probability'], close(value, expected, 'patch probability'))
                delta = [b-a for a, b in zip(p, q)]
                absolute = sum(abs(value) for value in delta)
                off_abs = sum(abs(value) for i, value in enumerate(delta) if i not in animal)
                expected_margin = ((sum(q[i] for i in dog) / sum(q[i] for i in cat)).ln() - margin)
                expected_margin *= 1 if side == 'cat' else -1
                maxima['tv'] = max(maxima['tv'], close(condition['total_variation'], absolute/2, 'TV'))
                maxima['oriented_margin'] = max(maxima['oriented_margin'],
                                                 close(condition['annotation_oriented_margin_change'], expected_margin, 'margin change'))
                if absolute:
                    maxima['off_share'] = max(maxima['off_share'],
                                               close(condition['offgroup_absolute_change_share'], off_abs/absolute, 'off-group share'))
                else:
                    check(condition['offgroup_absolute_change_share'] is None, 'undefined off-group share')
                close(condition['animal_mass_change'], sum(delta[i] for i in animal), 'animal mass change')
                close(condition['offgroup_mass_change'], sum(x for i, x in enumerate(delta) if i not in animal), 'off-group mass change')
                top = max(range(1000), key=logits.__getitem__)
                check(condition['patched_top1_id'] == top and condition['clean_top1_id'] == base['top1_id'] and
                      condition['top1_changed'] is (top != base['top1_id']), 'top-1 change')
                changes[name] = delta
            for name in ('outside', 'random'):
                expected = cosine(changes['matched_inside'], changes['matched_' + name])
                value = row[f'inside_vs_{name}_probability_displacement_cosine']
                if expected is None:
                    check(value is None, 'undefined cosine')
                else:
                    maxima['cosine'] = max(maxima['cosine'], close(value, expected, 'cosine'))
        for pair, sides in gates.items():
            c, d = sides['cat'], sides['dog']
            eligible = c[0] < 0 and d[0] > 0 and d[0] - c[0] >= 1 and c[1] >= Decimal('.2') and d[2] >= Decimal('.2')
            check(c[3] is eligible and d[3] is eligible, 'baseline eligibility')
    for group, eligible in (('evaluable', True), ('baseline_ineligible', False)):
        group_rows = [row for row in rows if row['eligible'] is eligible]
        saved = report['summary'][group]
        check(saved['sides'] == len(group_rows) and saved['pair_count'] == len({row['pair_id'] for row in group_rows}),
              'stratum counts')
        for name in ('matched_inside', 'matched_outside', 'matched_random'):
            entries = [row['conditions'][name] for row in group_rows]
            for field, median_name in (('total_variation', 'median_total_variation'),
                                       ('annotation_oriented_margin_change', 'median_annotation_oriented_margin_change'),
                                       ('offgroup_absolute_change_share', 'median_offgroup_absolute_change_share')):
                values = [item[field] for item in entries if item[field] is not None]
                if values:
                    close(saved[name][median_name], statistics.median(values), 'stratum median')
                else:
                    check(saved[name][median_name] is None, 'empty stratum median')
            check(saved[name]['undefined_offgroup_share_count'] ==
                  sum(item['offgroup_absolute_change_share'] is None for item in entries), 'undefined off-group count')
        for control in ('outside', 'random'):
            check(saved['inside_tv_exceeds_' + control] == sum(
                row['conditions']['matched_inside']['total_variation'] >
                row['conditions']['matched_' + control]['total_variation'] for row in group_rows), 'paired TV count')
            cosines = [row[f'inside_vs_{control}_probability_displacement_cosine'] for row in group_rows]
            values = [x for x in cosines if x is not None]
            recorded = saved[f'inside_vs_{control}_cosine']
            check(recorded['undefined_count'] == len(cosines) - len(values), 'undefined cosine count')
            if values:
                close(recorded['median'], statistics.median(values), 'cosine median')
            else:
                check(recorded['median'] is None, 'empty cosine median')
        check(saved['random_margin_at_least_inside'] == sum(
            row['conditions']['matched_random']['annotation_oriented_margin_change'] >=
            row['conditions']['matched_inside']['annotation_oriented_margin_change'] for row in group_rows),
            'paired margin count')
    return {'status': 'independent_decimal_saved_logits_pass', 'report_sha256': digest(report_path.read_bytes()),
            'uncompressed_report_sha256': digest(report_raw), 'vectors': len(selected), 'pair_sides': len(rows),
            'maximum_absolute_errors': maxima, 'scope': 'saved logits and report consistency; no model execution claim'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'index', 'manifest', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.index, args.manifest, args.report), sort_keys=True))

if __name__ == '__main__':
    main()
