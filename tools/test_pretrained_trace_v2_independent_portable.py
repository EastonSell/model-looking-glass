"""Fabricated-only fault checks for the independent retained-byte reader."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('trace_audit', HERE / 'pretrained_trace_v2_independent.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
SOURCE = 'a' * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    path.write_bytes(raw)
    return sha(raw)


def check(root, synthetic=True):
    return audit.audit(root, sha((root / 'candidate.json').read_bytes()), SOURCE, synthetic)


def expect_rejection(root, label, synthetic=True):
    try:
        check(root, synthetic)
    except (ValueError, KeyError, IndexError, TypeError):
        return
    raise AssertionError('accepted corrupted synthetic case: ' + label)


def update_capture(root, mutation):
    candidate_path = root / 'candidate.json'
    manifest_path = root / 'raw' / 'manifest.json'
    candidate = json.loads(candidate_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(x for x in manifest['captures'] if x['mode'] == 'gradient' and x['forward_index'] == 0)
    path = root / 'raw' / entry['file']
    raw = bytearray(path.read_bytes())
    if mutation == 'sign':
        raw[3] ^= 0x80
    elif mutation == 'reorder':
        n = entry['events'][0]['length']
        raw[:2*n] = raw[n:2*n] + raw[:n]
    else:
        raise ValueError('unknown mutation')
    path.write_bytes(raw)
    entry['sha256'] = sha(raw)
    for event in entry['events']:
        start, end = event['offset'], event['offset'] + event['length']
        event['sha256'] = sha(raw[start:end])
        candidate['measurement']['result']['modes']['gradient']['events'][event['sequence']]['value_sha256'] = event['sha256']
    candidate['measurement']['tensor_evidence']['manifest_sha256'] = write_json(manifest_path, manifest)
    write_json(candidate_path, candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True, help="fabricated tiny CPU fixture directory")
    parser.add_argument("--scratch", type=Path, required=True, help="existing scratch directory for temporary copies")
    args = parser.parse_args()
    fixture = args.fixture
    expected = check(fixture)
    if not (expected['synthetic'] is True and expected['forward_count'] == 10 and
            expected['capture_event_count'] == 54 and expected['capture_bit_comparisons'] == 36):
        raise AssertionError('valid synthetic fixture did not reconcile')
    expect_rejection(fixture, 'real geometry without synthetic flag', synthetic=False)
    cases = ('truncate', 'extra', 'sign', 'reorder', 'boolean_metadata', 'id_metadata', 'offset',
             'nonfinite', 'exponent_overflow')
    with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
        for name in cases:
            root = Path(tmp) / name
            shutil.copytree(fixture, root)
            candidate_path = root / 'candidate.json'
            if name == 'truncate':
                path = root / 'raw' / 'logits-003.f32'
                path.write_bytes(path.read_bytes()[:-1])
            elif name == 'extra':
                (root / 'raw' / 'unexpected.partial').write_bytes(b'x')
            elif name in ('sign', 'reorder'):
                update_capture(root, name)
            elif name in ('boolean_metadata', 'id_metadata'):
                candidate = json.loads(candidate_path.read_bytes())
                mode = 'id' if name == 'id_metadata' else 'direct'
                candidate['measurement']['result']['modes'][mode]['events'][1]['metadata']['sequence'] = True
                write_json(candidate_path, candidate)
            elif name == 'offset':
                manifest_path = root / 'raw' / 'manifest.json'
                manifest = json.loads(manifest_path.read_bytes())
                manifest['captures'][0]['events'][0]['offset'] = 1
                candidate = json.loads(candidate_path.read_bytes())
                candidate['measurement']['tensor_evidence']['manifest_sha256'] = write_json(manifest_path, manifest)
                write_json(candidate_path, candidate)
            elif name == 'nonfinite':
                path = root / 'raw' / 'logits-005.f32'
                data = b'\x00\x00\x80\x7f' + path.read_bytes()[4:]
                path.write_bytes(data)
                manifest_path = root / 'raw' / 'manifest.json'
                manifest = json.loads(manifest_path.read_bytes())
                manifest['logits'][4]['sha256'] = sha(data)
                forward_path = root / 'forward-005.json'
                forward = json.loads(forward_path.read_bytes())
                forward['logit_sha256'] = sha(data)
                write_json(forward_path, forward)
                candidate = json.loads(candidate_path.read_bytes())
                candidate['measurement']['result']['modes']['direct']['logit_sha256'][0] = sha(data)
                candidate['measurement']['tensor_evidence']['manifest_sha256'] = write_json(manifest_path, manifest)
                write_json(candidate_path, candidate)
            elif name == 'exponent_overflow':
                # Re-pin the mutated candidate; this value is in an otherwise
                # unused field, so checked tensor fields remain untouched.
                candidate = json.loads(candidate_path.read_bytes())
                candidate['unused'] = 1.0
                encoded = (json.dumps(candidate, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                needle = b'"unused":1.0'
                if encoded.count(needle) != 1:
                    raise AssertionError('synthetic exponent fixture is ambiguous')
                candidate_path.write_bytes(encoded.replace(needle, b'"unused":1e999'))
            expect_rejection(root, name)
    print('PASS: valid tiny fixture and ten fabricated geometry/byte/metadata fault cases')


if __name__ == '__main__':
    main()
