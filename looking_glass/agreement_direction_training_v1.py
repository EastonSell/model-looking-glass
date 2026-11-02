"""Reconstruct fixed agreement directions from already-published captures.

This module does not import or run a model.  Its CLI requires the exact public
archive and writes reproducible binary64 vectors and a provenance manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile


ARCHIVE_SHA256 = '9ea7ebf83f0580335cbf35b65b2277d8990090adff5b3eb8d0b3b68409cd3a69'
PAIRS = (
    ('q1', 'cat', 'c00', 'c01'), ('q1', 'dog', 'c02', 'c03'),
    ('q2', 'doctor', 'c06', 'c07'), ('q2', 'teacher', 'c04', 'c05'),
    ('q3', 'artist', 'c08', 'c09'), ('q3', 'writer', 'c10', 'c11'),
    ('q4', 'farmer', 'c12', 'c13'), ('q4', 'sailor', 'c14', 'c15'),
)
SEED = b'mlg-agreement-direction-control-v1\x00'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode_f32_bits(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError('invalid float32 bit pattern')
    sign = -1.0 if bits >> 31 else 1.0
    exponent = (bits >> 23) & 0xff
    mantissa = bits & 0x7fffff
    if exponent == 0xff:
        raise ValueError('nonfinite float32')
    if exponent == 0:
        return math.copysign(math.ldexp(float(mantissa), -149), sign)
    return math.copysign(math.ldexp(float((1 << 23) + mantissa), exponent - 150), sign)


def dot(left, right):
    if len(left) != len(right):
        raise ValueError('incompatible vector widths')
    return math.fsum(a * b for a, b in zip(left, right))


def norm(values):
    return math.sqrt(math.fsum(value * value for value in values))


def unit(values):
    length = norm(values)
    if not math.isfinite(length) or length <= 2**-40:
        raise ValueError('nonfinite or degenerate direction')
    return tuple(value / length for value in values), length


def pair_differences(vectors, pairs):
    if len(pairs) != 8:
        raise ValueError('expected eight training pairs')
    ids = [key for pair in pairs for key in pair]
    if len(set(ids)) != 16 or set(ids) != set(vectors):
        raise ValueError('training IDs must be unique and exact')
    rows = []
    for singular, plural in pairs:
        a, b = vectors[singular], vectors[plural]
        if not a or len(a) != len(b) or not all(math.isfinite(x) for x in (*a, *b)):
            raise ValueError('invalid training capture')
        rows.append(tuple(x - y for x, y in zip(a, b)))
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('inconsistent training width')
    return tuple(rows)


def derive_directions(differences):
    if len(differences) != 8 or not differences[0]:
        raise ValueError('expected eight nonempty differences')
    width = len(differences[0])
    if width > 65536 or any(len(row) != width for row in differences):
        raise ValueError('invalid difference width')
    if any(not math.isfinite(x) for row in differences for x in row):
        raise ValueError('nonfinite difference')
    mean = tuple(math.fsum(row[j] for row in differences) / 8 for j in range(width))
    u, mean_norm = unit(mean)
    modulation = tuple(math.fsum((1 if k % 2 == 0 else -1) * differences[k][j]
                                 for k in range(8)) / 8 for j in range(width))
    modulation_dot = dot(modulation, u)
    v1, modulation_norm = unit(tuple(x - modulation_dot * y for x, y in zip(modulation, u)))
    random = tuple(int.from_bytes(hashlib.sha256(SEED + j.to_bytes(2, 'big')).digest()[:8], 'big')
                   / 2**63 - 1.0 for j in range(width))
    random_dot_u = dot(random, u)
    residual = tuple(x - random_dot_u * y for x, y in zip(random, u))
    random_dot_v1 = dot(residual, v1)
    v2, random_norm = unit(tuple(x - random_dot_v1 * y for x, y in zip(residual, v1)))
    vectors = (u, v1, v2)
    if any(abs(norm(v) - 1) > 1e-12 for v in vectors):
        raise ValueError('direction failed unit-norm check')
    if any(abs(dot(vectors[i], vectors[j])) > 1e-12 for i in range(3) for j in range(i)):
        raise ValueError('directions failed orthogonality check')
    metadata = {
        'mean_difference_norm': mean_norm,
        'interaction_raw_norm': norm(modulation),
        'interaction_residual_norm': modulation_norm,
        'random_raw_norm': norm(random),
        'random_residual_after_u_norm': norm(residual),
        'random_residual_norm': random_norm,
        'interaction_projection_on_u': modulation_dot,
        'random_projection_on_u': random_dot_u,
        'random_residual_projection_on_v1': random_dot_v1,
        'pairwise_dots': {'u_v1': dot(u, v1), 'u_v2': dot(u, v2), 'v1_v2': dot(v1, v2)},
        'norms': {'u': norm(u), 'v1': norm(v1), 'v2': norm(v2)},
        'training_pair_projections': [{'u': dot(row, u), 'v1': dot(row, v1),
                                       'v2': dot(row, v2), 'norm': norm(row)}
                                      for row in differences],
    }
    return u, v1, v2, metadata


def read_archive_snapshot(path):
    limit = 16 * 1024 * 1024
    with Path(path).open('rb') as stream:
        raw_archive = stream.read(limit + 1)
    if len(raw_archive) > limit:
        raise ValueError('public discovery archive exceeds the size limit')
    return raw_archive


def read_archive(path):
    raw_archive = read_archive_snapshot(path)
    if digest(raw_archive) != ARCHIVE_SHA256:
        raise ValueError('public discovery archive hash differs')
    expected = {item[2]: (item[0], item[1], 'singular') for item in PAIRS}
    expected.update({item[3]: (item[0], item[1], 'plural') for item in PAIRS})
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        entries = [json.loads(line) for line in archive.read('attempt/events.jsonl').splitlines()]
        cases = entries[0]['event']['cases']
        actual = {case['case_id']: (case['quartet'], case['lemma'], case['number']) for case in cases}
        if len(cases) != 16 or actual != expected:
            raise ValueError('public training identities differ')
        vectors, capture_hashes = {}, {}
        for entry in entries:
            event = entry['event']
            if event.get('kind') != 'clean_capture':
                continue
            case_id = event['case']
            if case_id not in expected or case_id in vectors:
                raise ValueError('unknown or repeated clean capture')
            selected = [step for step in event['plan_metadata']['steps'] if step.get('name') == 'capture_4']
            if (len(selected) != 1 or selected[0]['module_path'] != 'transformer.h.3.attn'
                    or selected[0]['output_selector'] != [0] or selected[0]['batch_index'] != 0
                    or selected[0]['invocation'] != 0 or selected[0]['token_index'] != 4
                    or selected[0]['output_shape'] != [1, 5, 768]
                    or selected[0]['selected_shape'] != [768]):
                raise ValueError('discovery capture site differs')
            info = entry['tensors']['capture_4']
            if info['shape'] != [768] or info['bytes'] != 3072 or info['dtype'] != 'float32-le':
                raise ValueError('discovery capture format differs')
            raw = archive.read('attempt/tensors/' + info['sha256'] + '.f32le')
            if len(raw) != 3072 or digest(raw) != info['sha256']:
                raise ValueError('discovery capture hash differs')
            vectors[case_id] = tuple(decode_f32_bits(bits) for (bits,) in struct.iter_unpack('<I', raw))
            capture_hashes[case_id] = info['sha256']
    if set(vectors) != set(expected):
        raise ValueError('missing public training capture')
    return vectors, capture_hashes


def write_vectors(target, vectors):
    if len(vectors) != 3:
        raise ValueError('expected three direction vectors')
    direction_hashes = {}
    for name, vector in zip(('u', 'v1', 'v2'), vectors):
        if len(vector) != 768 or not all(math.isfinite(x) for x in vector):
            raise ValueError('invalid direction vector')
        raw = struct.pack('<768d', *vector)
        path = target / (name + '.f64le')
        path.write_bytes(raw)
        if path.read_bytes() != raw:
            raise ValueError('saved direction readback differs')
        direction_hashes[name] = digest(raw)
    return direction_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output-dir', required=True, help='new, nonexistent output directory')
    args = parser.parse_args()
    vectors, hashes = read_archive(args.archive)
    differences = pair_differences(vectors, [(row[2], row[3]) for row in PAIRS])
    u, v1, v2, metadata = derive_directions(differences)
    target = Path(args.output_dir)
    target.mkdir(parents=False, exist_ok=False)
    direction_hashes = write_vectors(target, (u, v1, v2))
    manifest = {
        'schema': 'distilgpt2-agreement-direction-training-v1',
        'archive_sha256': ARCHIVE_SHA256,
        'builder_sha256': digest(Path(__file__).read_bytes()),
        'scope': 'Derived only from 16 exposed saved discovery activations; no model forward.',
        'site': 'transformer.h.3.attn output[0], batch 0, token 4',
        'pair_order': [{'family': q, 'lemma': lemma, 'singular': s, 'plural': p,
                        'singular_sha256': hashes[s], 'plural_sha256': hashes[p]}
                       for q, lemma, s, p in PAIRS],
        'seed_hex': SEED.hex(), 'vectors_sha256': direction_hashes, 'geometry': metadata,
    }
    (target / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)
                                          + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'manifest_sha256': digest((target / 'manifest.json').read_bytes()),
                      'vectors_sha256': direction_hashes}, sort_keys=True))


if __name__ == '__main__':
    main()
