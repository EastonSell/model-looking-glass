"""Public saved-byte replay of the six-forward CPU reference.

This uses only the standard library and Git object reads. It does not load a
model, query a GPU, read private receipts, or attest model execution.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

LOCAL = '80f82a81c62403045c9c33295e6a61b4f410d446'
PUBLIC = '75e3ed56cad3f0586b8ea4f4010f0d116e1ca83b'
TREE = 'a15f3ebac5c129f03d9cb9e0d52eabac990958e2'
MANIFEST = '47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c'
INVENTORY = 'e586bae02145a62a7163eb030485fa85a33f9e94cd0949a9f88f3f74c7de194b'
OBSERVER = 'f9925fdda2fc3a754d4849dbfb344704105c44b1e3244851812b6e8711f1461d'
CANDIDATE = 'e56f9860b42eab1f7651d988b8120884d22e2e140e22e12bd87ceb462f794051'
STARTED = '323ddb81ef4cd1fa9408ffe0f2eae7bb16dd7095dcd735addaef5c56ee7f45c7'
EXIT = '4f2711b9064c38ec08936341179cc06151c1a848c97d97ba62b5f95903649e6b'
PRIVATE_AUDIT = '644c236c83ba2828148e9c6382cc3b772b80c9968e54231b10a085780cbfb0b9'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, limit, expected=None):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'bounded saved file exceeds its limit')
    actual = digest(raw)
    if expected is not None:
        require(actual == expected, 'saved file does not match its independent pin')
    return raw


def parse(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    def bad(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=bad)


def same_json_types_and_values(left, right):
    """Reject Boolean/integer and integral-float aliases in saved records."""
    return (json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False)
            == json.dumps(right, sort_keys=True, separators=(',', ':'), allow_nan=False))


def literal(source, name):
    tree = ast.parse(source)
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == name]
    require(len(values) == 1, 'source literal is missing or repeated')
    return ast.literal_eval(values[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'repo', 'git', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('run', 'repo', 'git', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    require(not args.output.exists() and args.output.parent.is_dir(), 'fresh report path is required')
    require(args.run.resolve(strict=True) not in args.output.resolve(strict=False).parents,
            'report must be outside the source run')
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                'GIT_NO_REPLACE_OBJECTS': '1', 'GIT_TERMINAL_PROMPT': '0'})
    def git(*values):
        return subprocess.run([str(args.git), '--no-replace-objects', '-C', str(args.repo), *values],
                              env=env, capture_output=True, check=True, timeout=15).stdout
    require(git('show', '-s', '--format=%T', PUBLIC).decode('ascii').strip() == TREE,
            'public source tree differs')
    source = git('show', PUBLIC + ':looking_glass/short_service_cpu_reference.py').decode('utf-8')
    paths = literal(source, 'SOURCE_PATHS')
    started_raw = read(args.run / 'started.json', 128 * 1024, STARTED)
    started = parse(started_raw)
    require(started['schema'] == 'short-service-cpu-reference-start-v1'
            and started['source_commit'] == LOCAL and started['manifest_sha256'] == MANIFEST
            and started['inventory_sha256'] == INVENTORY and set(started['source_sha256']) == set(paths),
            'started record or source tuple differs')
    for name in paths:
        require(digest(git('show', PUBLIC + ':' + name)) == started['source_sha256'][name],
                'a source record differs from the public immutable tree')
    manifest_raw = git('show', PUBLIC + ':experiments/short_cpu_service_reference_inputs_v1.json')
    require(digest(manifest_raw) == MANIFEST, 'input manifest differs')
    manifest = parse(manifest_raw)
    cases = {case['id']: case for case in manifest['comparisons']}
    require(list(cases) == ['synthetic-16', 'synthetic-32', 'gas', 'ligand'], 'case order differs')
    for case_id in cases:
        case = cases[case_id]
        for role in ('reference', 'service'):
            row = case[role]
            require(all(type(x) is int for x in row['input_ids'] + row['attention_mask']),
                    'an exact input or mask differs')
        active, padded = case['reference'], case['service']
        n = len(active['input_ids'])
        require(active['attention_mask'] == [1] * n
                and padded['input_ids'][:n] == active['input_ids']
                and padded['attention_mask'] == [1] * n + [0] * (len(padded['input_ids']) - n),
                'active prefix or right-padding mask differs')

    candidate = parse(read(args.run / 'candidate.json', 128 * 1024, CANDIDATE))
    require(candidate['schema'] == 'short-service-cpu-reference-v1' and candidate['status'] == 'passed'
            and candidate['source_commit'] == LOCAL and candidate['inventory_sha256'] == INVENTORY,
            'candidate contract differs')
    assets = literal(git('show', PUBLIC + ':looking_glass/pretrained.py').decode('utf-8'), 'ASSETS')
    expected_assets = {key: value[1] for key, value in assets.items()}
    meta = candidate['model_metadata']
    require(meta['asset_sha256'] == expected_assets and meta['weights_sha256'] == assets['model.safetensors'][1]
            and meta['model_id'] == 'distilbert/distilgpt2'
            and meta['model_revision'] == '2290a62682d06624634c1f46a6ad5be0f47f38aa'
            and meta['device'] == 'cpu' and meta['dtype'] == 'float32' and meta['use_cache'] is False,
            'recorded model identity differs from the source pins')
    order = [('synthetic-16', 'reference'), ('gas', 'reference'), ('gas', 'service'),
             ('synthetic-32', 'reference'), ('ligand', 'reference'), ('ligand', 'service')]
    require([(entry['case'], entry['role']) for entry in candidate['entries']] == order,
            'the six recorded forward entries differ')
    names = {'started.json', 'candidate.json'}
    rows, entries = {}, []
    for entry, (case_id, role) in zip(candidate['entries'], order, strict=True):
        stem = case_id + '-' + role
        names.update({stem + '-start.json', stem + '.json', stem + '.f32'})
        value = cases[case_id][role]
        start = parse(read(args.run / (stem + '-start.json'), 32 * 1024))
        require(same_json_types_and_values(start, {'case': case_id, 'role': role, **value}),
                'recorded forward input differs')
        require(same_json_types_and_values(parse(read(args.run / (stem + '.json'), 32 * 1024)), entry),
                'individual row metadata differs from candidate')
        shape = [1, len(value['input_ids']), 50257]
        require(entry['shape'] == shape and entry['filename'] == stem + '.f32'
                and type(entry['bytes']) is int and entry['bytes'] == shape[1] * shape[2] * 4,
                'recorded tensor shape or byte count differs')
        raw = read(args.run / entry['filename'], 32 * 50257 * 4, entry['sha256'])
        require(len(raw) == entry['bytes'] and all((int.from_bytes(raw[i:i+4], 'little') & 0x7f800000)
                                                    != 0x7f800000 for i in range(0, len(raw), 4)),
                'saved logits are incomplete or nonfinite')
        rows[(case_id, role)] = raw
        entries.append({'case': case_id, 'role': role, 'shape': shape,
                        'bytes': len(raw), 'sha256': digest(raw)})
    require({path.name for path in args.run.iterdir()} == names, 'saved run has missing or extra files')
    comparisons = []
    for case_id in ('gas', 'ligand'):
        reference, padded = rows[(case_id, 'reference')], rows[(case_id, 'service')]
        require(padded[:len(reference)] == reference, 'saved active logits differ at the bit level')
        value = {'case': case_id, 'comparison': {'atol': 1e-6, 'count': len(reference) // 4,
                 'max_abs': 0.0, 'rtol': 1e-5, 'violations': 0}}
        comparisons.append(value)
    require(candidate['padding_comparisons'] == comparisons, 'recorded padding comparisons differ')

    result = {'schema': 'public-short-cpu-reference-saved-replay-v1',
              'status': 'saved_byte_consistency', 'recorded_local_commit': LOCAL, 'public_commit': PUBLIC,
              'source_tree': TREE, 'source_file_count': len(paths),
              'pins': {'manifest': MANIFEST, 'candidate': CANDIDATE, 'started': STARTED},
              'unreplayed_private_commitments': {'inventory': INVENTORY, 'observer_source': OBSERVER,
                  'controller_exit': EXIT, 'independent_audit_report': PRIVATE_AUDIT},
              'entries': entries, 'padding_comparisons': comparisons,
              'scope': ['active logits of each saved padded/unpadded science pair are bit-identical',
                        'synthetic inputs each have one recorded reference forward',
                        '17 recorded source hashes match the pinned public tree; the local commit object is not required',
                        'private commitments are not replayed and do not establish controller or process identity',
                        'the checkpoint metadata and source hashes are records, not independent proof of execution',
                        'padded positions are not compared; no GPU or throughput conclusion follows']}
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    with args.output.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'short report write')
        stream.flush()
        os.fsync(stream.fileno())
    require(args.output.read_bytes() == raw, 'report readback differs')
    print(json.dumps({'status': result['status'], 'sha256': digest(raw), 'bytes': len(raw)}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'status': 'incomplete', 'reason': 'validation_failed'}))
        raise SystemExit(1)
