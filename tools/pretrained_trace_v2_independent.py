"""Independently reconcile a quiescent retained five-token trace attempt.

This stdlib-only reader checks saved-byte consistency, not execution, source
authenticity, output completeness before a crash, or publication chronology.
Synthetic mode is explicitly limited to the documented tiny CPU fixture.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct


PHASES = ('baseline_cold', 'baseline_warm', 'reference', 'reference', 'direct',
          'direct', 'gradient', 'gradient', 'id', 'id')
SITES = tuple((layer, kind) for layer in (0, 2, 5) for kind in ('attn', 'mlp', 'block'))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_limited(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing or linked file: ' + path.name)
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    require(0 < len(raw) <= limit, 'invalid byte count: ' + path.name)
    return raw


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject(_):
        raise ValueError('nonfinite JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def finite_f32(raw):
    require(len(raw) % 4 == 0, 'unaligned float32 bytes')
    return all((bits & 0x7f800000) != 0x7f800000 for (bits,) in struct.iter_unpack('<I', raw))


def exact(a, b):
    """Compare recursively without allowing Boolean/integer equivalence."""
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if type(a) in (tuple, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def expected_metadata(mode, sequence, hidden, device):
    forward, index = divmod(sequence, 9)
    layer, kind = SITES[index]
    path = f'transformer.h.{layer}' + ('' if kind == 'block' else '.' + kind)
    selector = [0] if kind in ('attn', 'block') else []
    saved = mode != 'id'
    return {'schema_version': '1.0', 'identity': {'run_id': f'pretrained-gpu-trace-pilot-v1-{mode}',
        'job_id': None, 'request_id': None, 'sample_id': None}, 'sequence': sequence,
        'forward_index': forward, 'invocation': 0, 'layer_index': layer, 'module_kind': kind,
        'module_path': path, 'output_path': selector, 'batch_axis': 0, 'token_axis': 1,
        'batch_positions': [0], 'token_positions': [0, 1], 'shape': [1, 5, hidden],
        'dtype': 'torch.float32', 'device': device, 'capture_mode': mode,
        'saved_shape': [1, 2, hidden] if saved else None,
        'saved_dtype': 'torch.float32' if saved else None, 'saved_device': 'cpu' if saved else None,
        'generation': {'stage': 'unknown', 'generation_step': None, 'cache_length_before': None,
                       'cache_length_after': None, 'cache_position': None, 'use_cache': None}}


def audit(root, candidate_sha256, source_commit, synthetic=False):
    root = Path(root)
    require(type(synthetic) is bool, 'synthetic flag must be Boolean')
    hidden, vocab, device = (8, 6767, 'cpu') if synthetic else (768, 50257, 'cuda:0')
    logit_bytes, event_bytes = 5 * vocab * 4, 2 * hidden * 4
    payload_bytes = 10 * logit_bytes + 54 * event_bytes
    require(type(source_commit) is str and len(source_commit) == 40 and
            all(c in '0123456789abcdef' for c in source_commit), 'invalid expected source commit')
    require(type(candidate_sha256) is str and len(candidate_sha256) == 64 and
            all(c in '0123456789abcdef' for c in candidate_sha256), 'invalid candidate digest')
    candidate_raw = read_limited(root / 'candidate.json', 1024 * 1024)
    require(digest(candidate_raw) == candidate_sha256, 'candidate digest differs')
    candidate = read_json(candidate_raw)
    require(candidate.get('schema') == 'pretrained-gpu-trace-candidate-v1' and
            candidate.get('source_commit') == source_commit, 'candidate identity differs')
    measurement = candidate['measurement']
    require(type(measurement['forward_count']) is int and measurement['forward_count'] == 10,
            'forward count differs')
    binding = measurement['tensor_evidence']
    require(type(binding) is dict and set(binding) == {'manifest_sha256', 'payload_bytes'} and
            type(binding['payload_bytes']) is int and binding['payload_bytes'] == payload_bytes,
            'candidate payload binding differs')
    rawdir = root / 'raw'
    manifest_raw = read_limited(rawdir / 'manifest.json', 65536)
    require(digest(manifest_raw) == binding['manifest_sha256'], 'manifest digest differs')
    manifest = read_json(manifest_raw)
    require(type(manifest) is dict and set(manifest) == {'schema', 'encoding', 'hidden', 'vocab',
            'payload_bytes', 'logits', 'captures'} and
            manifest['schema'] == 'pretrained-gpu-trace-tensor-evidence-v2' and
            manifest['encoding'] == 'float32-little-endian' and
            exact(manifest['hidden'], hidden) and exact(manifest['vocab'], vocab) and
            exact(manifest['payload_bytes'], payload_bytes), 'manifest geometry differs')
    require(type(manifest['logits']) is list and len(manifest['logits']) == 10 and
            type(manifest['captures']) is list and len(manifest['captures']) == 6,
            'manifest inventory differs')
    expected_files = {'manifest.json'} | {f'logits-{i:03d}.f32' for i in range(1, 11)}
    expected_files |= {f'{mode}-{i}.f32' for mode in ('reference', 'direct', 'gradient') for i in (1, 2)}
    require({path.name for path in rawdir.iterdir()} == expected_files, 'raw directory inventory differs')

    result = measurement['result']
    hashes = []
    baseline = None
    for index, (phase, item) in enumerate(zip(PHASES, manifest['logits']), 1):
        name = f'logits-{index:03d}.f32'
        data = read_limited(rawdir / name, logit_bytes)
        sha = digest(data)
        expected = {'index': index, 'phase': phase, 'file': name, 'bytes': logit_bytes, 'sha256': sha}
        require(len(data) == logit_bytes and finite_f32(data) and exact(item, expected),
                'logit payload or manifest differs: ' + name)
        record = read_json(read_limited(root / f'forward-{index:03d}.json', 4096))
        require(exact(record, {'schema': 'pretrained-gpu-trace-forward-v1', 'index': index,
                'phase': phase, 'dtype': 'float32', 'shape': [1, 5, vocab], 'logit_sha256': sha}),
                'forward record differs: ' + name)
        if baseline is None:
            baseline = data
        else:
            require(data == baseline, 'full-logit float32 bits differ: ' + name)
        hashes.append(sha)
    require(type(result['reference_value_sha256']) is list and len(result['reference_value_sha256']) == 18 and
            exact(result['baseline_logit_sha256'], hashes[:2]) and
            exact(result['reference_logit_sha256'], hashes[2:4]) and
            result['direct_gradient_equal'] is True, 'reported result differs')

    refpieces = []
    expected_pairs = [(mode, forward) for mode in ('reference', 'direct', 'gradient') for forward in (0, 1)]
    for item, (mode, forward) in zip(manifest['captures'], expected_pairs):
        name = f'{mode}-{forward + 1}.f32'
        raw = read_limited(rawdir / name, 9 * event_bytes)
        require(len(raw) == 9 * event_bytes and finite_f32(raw), 'capture payload differs: ' + name)
        require(type(item) is dict and set(item) == {'mode', 'forward_index', 'file', 'bytes', 'sha256', 'events'}
                and item['mode'] == mode and exact(item['forward_index'], forward) and item['file'] == name
                and exact(item['bytes'], len(raw)) and item['sha256'] == digest(raw)
                and type(item['events']) is list and len(item['events']) == 9, 'capture manifest differs: ' + name)
        for index, (entry, (layer, kind)) in enumerate(zip(item['events'], SITES)):
            sequence = forward * 9 + index
            start = index * event_bytes
            piece = raw[start:start + event_bytes]
            path = f'transformer.h.{layer}' + ('' if kind == 'block' else '.' + kind)
            selector = [0] if kind in ('attn', 'block') else []
            expected = {'sequence': sequence, 'forward_index': forward, 'layer_index': layer,
                        'module_kind': kind, 'module_path': path, 'output_path': selector,
                        'token_positions': [0, 1], 'offset': start, 'length': event_bytes,
                        'sha256': digest(piece)}
            require(exact(entry, expected), 'capture coordinates or segment digest differ: ' + name)
            if mode == 'reference':
                require(result['reference_value_sha256'][sequence] == digest(piece), 'reference digest differs')
                refpieces.append(piece)
            else:
                saved = result['modes'][mode]['events'][sequence]
                meta = saved['metadata']
                require(exact(meta, expected_metadata(mode, sequence, hidden, device)) and
                        saved['value_sha256'] == digest(piece) and
                        piece == refpieces[sequence], 'capture bits or metadata differ: ' + name)
    for mode, offset in (('direct', 4), ('gradient', 6), ('id', 8)):
        item = result['modes'][mode]
        require(exact(item['event_count'], 18) and exact(item['logit_sha256'], hashes[offset:offset+2]) and
                item['logits_equal_baseline'] is True and type(item['events']) is list and len(item['events']) == 18,
                'mode result differs: ' + mode)
        require(exact(item['tensor_bytes'], 0 if mode == 'id' else 18 * event_bytes) and
                item['values_equal_reference'] is (None if mode == 'id' else True), 'mode evidence differs')
        if mode == 'id':
            require(all(event['value_sha256'] is None and
                        exact(event['metadata'], expected_metadata('id', index, hidden, device))
                        for index, event in enumerate(item['events'])), 'ID mode metadata or value differs')
    return {'schema': 'independent-pretrained-trace-v2-byte-audit-v1',
            'status': 'saved-byte-consistent', 'synthetic': synthetic, 'source_commit': source_commit,
            'candidate_sha256': candidate_sha256, 'manifest_sha256': digest(manifest_raw),
            'payload_bytes': payload_bytes, 'forward_count': 10, 'capture_event_count': 54,
            'logit_bit_comparisons': 9, 'capture_bit_comparisons': 36,
            'scope': 'Quiescent saved-byte consistency only; no execution, source or timing attestation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    report = audit(args.data, args.candidate_sha256, args.source_commit, args.synthetic)
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with args.output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short audit report write')
        stream.flush()
    print(digest(raw))


if __name__ == '__main__':
    main()
