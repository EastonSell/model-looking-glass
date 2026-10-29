"""Independently reconcile saved public copying-screen bytes; no model execution."""
import ast
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess


ATTEMPT = Path(__file__).resolve().parent
RUN = ATTEMPT / 'run'
REPO = ATTEMPT.parent.parent
REVISION = '55d11a53eb34a13eb6b14e19a5bd0c47172edda5'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def source(path):
    return subprocess.check_output(['git', 'show', f'{REVISION}:{path}'], cwd=REPO)


def constants(data, names):
    found = {}
    for node in ast.parse(data).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                found[name] = ast.literal_eval(node.value)
    assert set(found) == set(names)
    return found


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description='Reconcile retained public data without a model forward.')
    parser.add_argument('--cache-dir', type=Path, help='optional Hugging Face hub cache root')
    args = parser.parse_args()
    pre_bytes = (ATTEMPT / 'pre-run.json').read_bytes()
    supervisor = json.loads((ATTEMPT / 'supervisor.json').read_bytes())
    manifest_bytes = (RUN / 'manifest.json').read_bytes()
    result_bytes = (RUN / 'result.json').read_bytes()
    pre, manifest, result = map(json.loads, (pre_bytes, manifest_bytes, result_bytes))
    check(supervisor['pre_run_sha256'] == digest(pre_bytes), 'pre-run digest mismatch')
    check(supervisor['process_exit_code'] == 0 and supervisor['failure_file_present'] is False
          and not (RUN / 'failure.json').exists(), 'attempt exit/failure status')
    check(supervisor['source_revision'] == pre['source_revision'] == manifest['source_revision']
          == result['source_revision'] == REVISION, 'source revision mismatch')
    check(digest(supervisor['stdout'].encode('utf-8')) == supervisor['stdout_sha256'], 'stdout hash')
    check(result['manifest_sha256'] == digest(manifest_bytes), 'manifest digest mismatch')
    for name, expected in supervisor['run_sha256'].items():
        check(digest((RUN / name).read_bytes()) == expected, f'supervisor digest: {name}')
    for path, expected in manifest['source_sha256'].items():
        check(digest(source(path)) == expected, f'source digest: {path}')
    check(digest(source('docs/DISTILGPT2_COPYING_DEV_V1.md')) == pre['protocol_sha256'], 'protocol hash')
    check(pre['known_exposure_candidate_sha256'] ==
          '76803775e6929b4a20463a50bcae839a727b1a875a0d433f7c27c9d1d69f0681', 'exposure digest')
    script = constants(source('scripts/copying_dev_v1.py'), ('GROUPS', 'PERMUTATIONS'))
    pinned = constants(source('looking_glass/pretrained.py'), ('MODEL_ID', 'REVISION', 'ASSETS'))
    metadata = result['model_metadata']
    check(metadata['model_id'] == pinned['MODEL_ID'] and metadata['model_revision'] == pinned['REVISION'],
          'checkpoint identity')
    check(metadata['asset_sha256'] == {name: item[1] for name, item in pinned['ASSETS'].items()}, 'asset pins')
    check(metadata['weights_sha256'] == pinned['ASSETS']['model.safetensors'][1], 'weight pin')
    if args.cache_dir is not None:
        cache = args.cache_dir / 'models--distilbert--distilgpt2' / 'snapshots' / pinned['REVISION']
        for name, (size, expected) in pinned['ASSETS'].items():
            path = cache / name
            check(path.stat().st_size == size and file_digest(path) == expected, f'current cache bytes: {name}')
    check(result['model_state_sha256_before'] == result['model_state_sha256_after']
          and len(result['model_state_sha256_before']) == 64, 'saved model state comparison')
    runtime = result['runtime']
    check(runtime['cuda_visibility'] == '-1' and runtime['cuda_initialized'] is False
          and runtime['cpu_autocast'] is False and runtime['model_eval'] is True
          and runtime['attention_implementation'] == 'eager' and runtime['use_cache'] is False
          and runtime['intraop_threads'] == 1 and runtime['guard_blocked_calls'] == [], 'runtime')
    check(runtime['post_probe'] == manifest['pre_probe'] ==
          {'input_preserved': True, 'half_min_normal_bits': 0x00400000, 'numpy_cast_preserved': True},
          'denormal probe')

    cases = []
    for group, (words, ids) in enumerate(script['GROUPS'], 1):
        for variant, order in enumerate(script['PERMUTATIONS']):
            indices = order + (0, 1, 2, 3, 4)
            cases.append({'case_id': f'copy-dev-{group:02d}-{variant}', 'group': group, 'variant': variant,
                          'text': ''.join(' ' + words[i] for i in indices),
                          'token_ids': [ids[i] for i in indices], 'target_id': ids[5]})
    check(cases == manifest['cases'] and len(cases) == len(result['cases']) == 32, 'case schedule')
    check(len({x for _, ids in script['GROUPS'] for x in ids}) == 48, 'token IDs not disjoint')
    events = [json.loads(line) for line in (RUN / 'attempt.jsonl').read_bytes().splitlines()]
    check(len(events) == 66 and result['submitted_forwards'] == result['root_outputs_returned'] == 33,
          'forward/event count')
    check(events[0] == {'event': 'start', 'case_id': cases[0]['case_id'], 'mode': 'uncaptured'}
          and events[1]['event'] == 'complete' and events[1]['mode'] == 'uncaptured', 'uncaptured events')

    logits = (RUN / 'last_logits.f32').read_bytes()
    attentions = (RUN / 'attentions.f32').read_bytes()
    uncaptured = (RUN / 'uncaptured_logits.f32').read_bytes()
    log_size, att_size = 50257 * 4, 6 * 12 * 11 * 11 * 4
    check(len(logits) == 32 * log_size and len(attentions) == 32 * att_size and len(uncaptured) == log_size,
          'stream size')
    check(digest(logits) == result['logits_sha256'] and digest(attentions) == result['attentions_sha256'],
          'aggregate tensor hash')
    check(uncaptured == logits[:log_size] and result['captured_logits_bitwise_match'] is True
          and digest(uncaptured) == result['uncaptured_logits_sha256'] == events[1]['logits_sha256'],
          'capture/uncaptured bits')
    observations = []
    max_normalization_error = 0.0
    for i, (case, row) in enumerate(zip(cases, result['cases'])):
        lraw, araw = logits[i * log_size:(i + 1) * log_size], attentions[i * att_size:(i + 1) * att_size]
        expected_start = {'event': 'start', 'case_id': case['case_id']}
        expected_complete = {'event': 'complete', 'case_id': case['case_id'],
                             'logits_sha256': digest(lraw), 'attention_sha256': digest(araw)}
        check(events[2 + i * 2] == expected_start and events[3 + i * 2] == expected_complete, 'event pairing')
        check(all(row[k] == case[k] for k in ('case_id', 'group', 'variant', 'target_id'))
              and row['logits_sha256'] == digest(lraw) and row['attention_sha256'] == digest(araw)
              and row['logits_bytes'] == log_size and row['attention_bytes'] == att_size, 'per-case binding')
        values = struct.unpack('<50257f', lraw)
        check(all(map(math.isfinite, values)), 'nonfinite logits')
        maximum, target = max(values), values[case['target_id']]
        log_probability = target - maximum - math.log(math.fsum(math.exp(value - maximum) for value in values))
        rank = 1 + sum(value > target for value in values)
        top = next(j for j, value in enumerate(values) if value == maximum)
        check((row['target_logit'], row['target_log_probability'], row['target_rank'], row['top_token_id'])
              == (target, log_probability, rank, top), 'logit statistics')
        weights = struct.unpack('<8712f', araw)
        check(all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in weights), 'invalid attention')
        last_to_target = []
        for layer in range(6):
            layer_values = []
            for head in range(12):
                offset = (layer * 12 + head) * 121
                layer_values.append(weights[offset + 10 * 11 + 5])
                for query in range(11):
                    line = weights[offset + query * 11:offset + (query + 1) * 11]
                    error = abs(math.fsum(line) - 1.0)
                    max_normalization_error = max(max_normalization_error, error)
                    check(error <= 1e-5 and all(value == 0.0 for value in line[query + 1:]),
                          'attention mask or normalization')
            last_to_target.append(layer_values)
        check(last_to_target == row['attention_last_to_earlier_target']
              and row['attention_future_max'] == 0 and row['attention_normalization_max_error'] <= 1e-5,
              'attention summary')
        observations.append((log_probability, top == case['target_id'], last_to_target))

    pairs = []
    for group in range(8):
        differences = [observations[group * 4][0] - observations[group * 4 + k][0] for k in (1, 2, 3)]
        pairs.append({'group': group + 1, 'control_differences': differences,
                      'mean_log_probability_difference': math.fsum(differences) / 3})
    check(pairs == result['pairs'], 'per-group log-probability deltas')
    mean = math.fsum(x['mean_log_probability_difference'] for x in pairs) / 8
    positive = sum(x['mean_log_probability_difference'] > 0 for x in pairs)
    repeated_top = sum(observations[i * 4][1] for i in range(8))
    control_top = sum(observations[i * 4 + j][1] for i in range(8) for j in (1, 2, 3))
    ranking = sorted(({'layer': layer, 'head': head,
                       'mean_attention_difference': math.fsum(
                           observations[i * 4][2][layer][head] - observations[i * 4 + j][2][layer][head]
                           for i in range(8) for j in (1, 2, 3)) / 24}
                      for layer in range(6) for head in range(12)),
                     key=lambda row: (-row['mean_attention_difference'], row['layer'], row['head']))
    check(ranking == result['attention_ranking'], 'attention ranking')
    check((mean, positive, repeated_top, control_top) ==
          (result['mean_log_probability_difference'], result['positive_group_means'],
           result['repeated_top_one'], result['control_top_one']), 'aggregate statistics')
    printed = json.loads(supervisor['stdout'])
    check(printed == {'status': 'complete', 'forwards': 33, 'positive_group_means': positive,
                      'mean_log_probability_difference': mean, 'repeated_top_one': repeated_top,
                      'control_top_one': control_top}, 'stdout summary')
    print(json.dumps({'status': 'pass', 'source_revision': REVISION, 'forwards': 33,
                      'group_mean_log_probability_difference': mean, 'positive_groups': positive,
                      'repeated_top_one': repeated_top, 'control_top_one': control_top,
                      'top_attention_head': ranking[0],
                      'maximum_binary64_attention_row_error': max_normalization_error,
                      'cache_checked': args.cache_dir is not None,
                      'scope': 'retained-byte consistency; no model replay or proof of faithful execution'},
                     sort_keys=True))


if __name__ == '__main__':
    main()
