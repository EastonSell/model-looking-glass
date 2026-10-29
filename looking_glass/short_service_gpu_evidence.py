"""Model-free readback of one fixed short-input GPU routing observation.

This validates saved bytes and the routing comparison. It cannot authenticate
execution, a process tree, native CUDA teardown, or a general capacity bound.
"""

import hashlib
from pathlib import Path
import re
import struct

from .bert_matched_numeric import f32_bits_to_f64


NAMES = ('reference-alpha', 'reference-beta', 'client-alpha', 'client-beta')
VOCAB = 50257
ATOL = 1e-6
RTOL = 1e-5
_HEX = re.compile(r'[0-9a-f]{64}\Z', re.ASCII)


def _bounded(path, size):
    with Path(path).open('rb') as stream:
        value = stream.read(size + 1)
    if len(value) != size:
        raise ValueError('saved full-logit payload has the wrong size')
    return value


def _compare(left, right):
    if type(left) is not bytes or type(right) is not bytes or len(left) != len(right) or len(left) % 4:
        raise ValueError('comparison requires equal little-endian float32 bytes')
    count = violations = 0
    maximum = 0.0
    for (a,), (b,) in zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right), strict=True):
        x, y = f32_bits_to_f64(a), f32_bits_to_f64(b)
        distance = abs(x - y)
        maximum = max(maximum, distance)
        violations += distance > ATOL + RTOL * abs(y)
        count += 1
    return {'count': count, 'violations': violations, 'max_abs': maximum,
            'atol': ATOL, 'rtol': RTOL}


def _compare_payloads(payloads, *, bucket, vocab, positions):
    if (type(payloads) is not dict or set(payloads) != set(NAMES)
            or type(bucket) is not int or not 1 <= bucket <= 32
            or type(vocab) is not int or not 1 <= vocab <= VOCAB
            or type(positions) is not tuple or len(positions) != 2
            or any(type(item) is not int or not 0 <= item < bucket for item in positions)):
        raise ValueError('invalid fixed output layout')
    size = bucket * vocab * 4
    if any(type(value) is not bytes or len(value) != size for value in payloads.values()):
        raise ValueError('invalid output byte count')
    row_bytes = vocab * 4
    def row(name, position):
        start = position * row_bytes
        return payloads[name][start:start + row_bytes]
    selected = (row('reference-alpha', positions[0]), row('reference-beta', positions[1]))
    separation = _compare(*selected)
    if separation['violations'] == 0:
        raise ValueError('serial references are indistinguishable at the fixed tolerance')
    comparisons = []
    for index, team in enumerate(('alpha', 'beta')):
        name, reference = 'client-' + team, 'reference-' + team
        full = _compare(payloads[name], payloads[reference])
        own = _compare(row(name, positions[index]), selected[index])
        other = _compare(row(name, positions[index]), selected[1 - index])
        if full['violations'] or own['violations'] or not other['violations']:
            raise ValueError('saved client output does not match its own distinct reference')
        comparisons.append({'team': team, 'active_position': positions[index],
                            'full': full, 'own_last': own, 'other_last': other})
    return {'reference_separation': separation, 'comparisons': comparisons}


def verify_payloads(run_dir, *, bucket, positions, expected_hashes):
    """Read the exact four bounded files and reconstruct the numerical checks."""
    if (type(bucket) is not int or bucket not in (16, 32)
            or type(expected_hashes) is not dict or set(expected_hashes) != set(NAMES)
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in expected_hashes.values())):
        raise ValueError('invalid saved-output identity')
    root = Path(run_dir)
    expected = {name + '.f32' for name in NAMES}
    if {path.name for path in root.glob('*.f32')} != expected:
        raise ValueError('full-logit inventory differs from the fixed four outputs')
    payloads = {name: _bounded(root / (name + '.f32'), bucket * VOCAB * 4) for name in NAMES}
    if any(hashlib.sha256(payloads[name]).hexdigest() != expected_hashes[name] for name in NAMES):
        raise ValueError('saved full-logit hash differs')
    return _compare_payloads(payloads, bucket=bucket, vocab=VOCAB, positions=positions)


def validate_supervision(record, *, run_id):
    """Require a separately observed, clean direct-child exit, with exact types."""
    if (type(record) is not dict or record.get('run_id') != run_id
            or type(record.get('exit_code')) is not int or record['exit_code'] != 0
            or record.get('timed_out') is not False or record.get('clean') is not True
            or type(record.get('pid')) is not int or record['pid'] <= 0
            or type(record.get('supervisor_run_id')) is not str or not record['supervisor_run_id']):
        raise ValueError('owner exit is not independently recorded as clean')
    return True
