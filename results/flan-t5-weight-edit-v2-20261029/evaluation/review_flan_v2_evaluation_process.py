"""Independently reconcile closed FLAN v2 evaluation process/resource records.

This is a model-free check of recorded consistency, not execution attestation.
It requires phase and operator-exit digests obtained outside the attempt.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = 'b3c7e19a5349cd2fb76ef4908407342233c7e240'
MANIFEST = '09623ddf8716a1467cc35ef0469f7bb6d2119fdf574fd35aafe13f55e9fa669b'
CASE = 'a118dde671351d21c519b6a61957e9478c9501bac52d55aa34e8394772e933d5'
C_FLOOR, E_FLOOR = 1207959552, 2416967680
STUDY = Path(r'E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2')

def require(value, message):
    if not value:
        raise ValueError(message)

def read_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def invalid(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)

def file_digest(path, limit=None):
    p = Path(path)
    require(p.is_file() and (limit is None or p.stat().st_size <= limit), 'missing or oversized file')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def equivalent(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)

def process(args):
    root = args.attempt.resolve()
    require(root == (STUDY / 'fixed-attempt' / 'evaluation').resolve(), 'evaluation path differs')
    require(args.scope.resolve().is_relative_to(STUDY.resolve()) and args.claim.resolve() == (STUDY / 'claims/evaluation.json').resolve(), 'scope or claim path differs')
    require(file_digest(args.manifest) == MANIFEST, 'manifest digest differs')
    named = {name: root / name for name in ('operator-start.json', 'operator-spawn.json', 'operator-exit.json',
                                             'phase.json', 'stdout.log', 'stderr.log')}
    named.update({'scope': args.scope, 'claim': args.claim, 'manifest': args.manifest})
    hashes = {name: file_digest(path, 65536 if name.endswith('.log') else 1024 * 1024) for name, path in named.items()}
    require(hashes['phase.json'] == args.phase_sha256 and hashes['operator-exit.json'] == args.exit_sha256, 'external record pin differs')
    start, spawn, exit_record, phase, claim = [read_json(named[name].read_bytes()) for name in
        ('operator-start.json', 'operator-spawn.json', 'operator-exit.json', 'phase.json', 'claim')]
    require(start['phase'] == phase['phase'] == claim['phase'] == 'evaluation' and start['rehearsal'] is False, 'phase identity differs')
    require(start['source'] == phase['source_revision'] == SOURCE and phase['manifest_sha256'] == claim['manifest_sha256'] == MANIFEST, 'source or manifest differs')
    require(start['scope_sha256'] == phase['scope_sha256'] == hashes['scope'] and phase['case_sha256'] == CASE, 'scope or case differs')
    require(start['claim_sha256'] == hashes['claim'] and Path(claim['attempt']).resolve() == root, 'one-shot claim differs')
    disk = start['disk']
    require(type(disk) is dict and set(disk) == {'source_free', 'output_free', 'source_margin_scope'}
            and type(disk['source_free']) is int and C_FLOOR <= disk['source_free'] < 2**64
            and type(disk['output_free']) is int and E_FLOOR <= disk['output_free'] < 2**64
            and disk['source_margin_scope'] == 'heuristic; not an OS reservation', 'parent disk observation differs')
    parent_temp = start['parent_temp']
    require(type(parent_temp) is dict and set(parent_temp) == {'TEMP', 'TMP'}
            and all(type(x) is str and Path(x).is_absolute() and Path(x).resolve().is_relative_to(STUDY.resolve()) for x in parent_temp.values()), 'parent temp differs')
    temp = (root / 'temp').resolve()
    suffixes = {'HF_HOME': 'hf', 'HF_HUB_CACHE': 'hf-hub', 'HF_ASSETS_CACHE': 'hf-assets',
                'HF_DATASETS_CACHE': 'hf-datasets', 'HF_MODULES_CACHE': 'hf-modules',
                'HUGGINGFACE_HUB_CACHE': 'hf-hub-legacy', 'HUGGINGFACE_ASSETS_CACHE': 'hf-assets-legacy',
                'TRANSFORMERS_CACHE': 'transformers', 'TORCH_HOME': 'torch', 'XDG_CACHE_HOME': 'xdg', 'MPLCONFIGDIR': 'mpl'}
    expected_caches = {'TEMP': str(temp), 'TMP': str(temp), **{k: str(temp / 'library-cache' / v) for k, v in suffixes.items()}}
    require(equivalent(start['child_cache_paths'], expected_caches), 'child caches differ')
    require(all(type(v) is int and v > 0 for v in (spawn['pid'], spawn['parent_pid'], phase['pid'], phase['parent_pid'], exit_record['direct_child_pid'], exit_record['operator_pid']))
            and spawn['pid'] != spawn['parent_pid'] and spawn['pid'] == phase['pid'] == exit_record['direct_child_pid']
            and spawn['parent_pid'] == phase['parent_pid'] == exit_record['operator_pid'], 'process identity differs')
    require(exit_record['status'] == 'complete_candidate' and type(exit_record['exit_code']) is int and exit_record['exit_code'] == 0
            and all(exit_record[k] is True for k in ('reaped', 'drained', 'identity_matches', 'phase_consistent', 'within_evidence_cap'))
            and exit_record['timed_out'] is False and exit_record['error'] is None and exit_record['error_detail'] is None
            and exit_record['phase_sha256'] == hashes['phase.json'], 'direct-child completion differs')
    for stream in ('stdout', 'stderr'):
        record, path = exit_record['logs'][stream], named[stream + '.log']
        require(type(record['bytes']) is int and record['bytes'] == path.stat().st_size and record['sha256'] == hashes[stream + '.log']
                and record['finished'] is True and record['truncated'] is False and record['error'] is None, 'drained log differs')
    stdout = named['stdout.log'].read_bytes().splitlines()
    require(stdout, 'child identity is absent')
    identity = read_json(stdout[0])
    require(identity['pid'] == spawn['pid'] and identity['parent_pid'] == spawn['parent_pid']
            and equivalent(identity['cache_paths'], expected_caches), 'child identity differs')
    require(phase['execution_status'] == 'complete' and type(phase['calls']) is int and 0 <= phase['calls'] <= 1200, 'phase completion differs')
    resources = None
    if phase.get('model_loaded') is False:
        require(phase['calls'] == 0 and phase['scientific_status'] == 'ineligible'
                and phase.get('reason') == 'tokenizer-alignment', 'no-model phase differs')
    else:
        require('model_loaded' not in phase, 'model-load marker contradicts completed phase')
        require(type(args.previous_calls) is int and args.previous_calls >= 0
                and type(phase['cumulative_calls']) is int
                and phase['cumulative_calls'] == args.previous_calls + phase['calls'] <= 1200, 'cumulative call count differs')
        for key, path, limit in [('result_sha256', root / 'result.json', 256*1024**2),
                                 ('events_sha256', root / 'evidence/events.jsonl', 64*1024**2),
                                 ('logits_sha256', root / 'evidence/logits.f32', 256*1024**2),
                                 ('resources_sha256', root / 'resources.jsonl', 1024**2)]:
            require(file_digest(path, limit) == phase[key], 'phase artifact digest differs')
            named[key] = path
            hashes[key] = phase[key]
        rows = []
        with named['resources_sha256'].open('rb') as stream:
            for line in stream:
                require(len(line) <= 4096 and line.endswith(b'\n') and len(rows) < 2400, 'resource log bound differs')
                row = read_json(line)
                i = len(rows)
                require(type(row) is dict and set(row) == {'index', 'call', 'boundary', 'source_free', 'output_free', 'passed'}
                        and type(row['index']) is int and row['index'] == i and type(row['call']) is int and row['call'] == i // 2 + 1
                        and row['boundary'] == ('before' if i % 2 == 0 else 'after') and row['passed'] is True
                        and type(row['source_free']) is int and C_FLOOR <= row['source_free'] < 2**64
                        and type(row['output_free']) is int and E_FLOOR <= row['output_free'] < 2**64, 'resource record differs')
                rows.append(row)
        require(len(rows) == 2 * phase['calls'], 'resource count differs')
        resources = {'checks': len(rows), 'checked_calls': phase['calls'],
                     'min_source_free': min((r['source_free'] for r in rows), default=None),
                     'min_output_free': min((r['output_free'] for r in rows), default=None)}
        require(equivalent(resources, phase['resource_summary']), 'resource summary differs')
    required_bytes = sum(named[name].stat().st_size for name in
        ('operator-start.json', 'operator-spawn.json', 'phase.json', 'stdout.log', 'stderr.log'))
    if resources is not None:
        required_bytes += sum(named[name].stat().st_size for name in
            ('result_sha256', 'events_sha256', 'logits_sha256', 'resources_sha256'))
    observed_bytes = exit_record['cumulative_evidence_bytes_before_exit_record']
    require(type(observed_bytes) is int and required_bytes <= observed_bytes <= 255 * 1024**2,
            'recorded pre-exit evidence count differs')
    require(all(file_digest(path) == hashes[name] for name, path in named.items()), 'saved file changed during audit')
    return {'status': 'recorded_process_consistent', 'source_revision': SOURCE, 'manifest_sha256': MANIFEST,
            'phase_sha256': hashes['phase.json'], 'operator_exit_sha256': hashes['operator-exit.json'],
            'scope_sha256': hashes['scope'], 'claim_sha256': hashes['claim'], 'calls': phase['calls'],
            'recorded_scientific_status': phase['scientific_status'], 'parent_disk': disk,
            'resource_summary': resources, 'artifact_hashes': {k: v for k, v in hashes.items() if k.endswith('_sha256')},
            'scope': 'Saved cooperative process and resource observations; not independent execution, filesystem reservation, whole-tree teardown, or scientific replay.'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'scope', 'claim', 'manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('phase-sha256', 'exit-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--previous-calls', required=True, type=int)
    args = p.parse_args()
    print(json.dumps(process(args), sort_keys=True, allow_nan=False))

if __name__ == '__main__':
    main()
