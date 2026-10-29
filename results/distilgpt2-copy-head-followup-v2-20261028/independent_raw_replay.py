"""Recompute public copying follow-up scores and witnesses from saved bytes.

No model or project source is imported.  Consistency of a saved journal is
not independent attestation that its producer faithfully executed the model.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = Path(__file__).resolve().parent
RUN = ATTEMPT / 'run'
TREE = ROOT
COMMIT = '40e98bc5e851a7be95035f8a60b266042422fff9'
ARMS = [('h10_final', 10, 10), ('h2_final', 2, 10), ('h7_final', 7, 10),
        ('h3_final', 3, 10), ('h1_final', 1, 10), ('h10_target', 10, 5)]
LOGITS = 50257 * 4
WITNESS = 11 * 768 * 4


def digest(data):
    return hashlib.sha256(data).hexdigest()


def describe_logits(data, target):
    numbers = struct.unpack('<50257f', data)
    assert all(math.isfinite(x) for x in numbers)
    largest = max(numbers)
    selected = numbers[target]
    logp = selected - largest - math.log(math.fsum(math.exp(x - largest) for x in numbers))
    return {'target_logit': selected, 'target_log_probability': logp,
            'target_rank': 1 + sum(x > selected for x in numbers), 'top_token_id': numbers.index(largest)}


def main():
    receipt = json.loads((ATTEMPT / 'supervisor.json').read_bytes())
    manifest_raw = (RUN / 'manifest.json').read_bytes()
    result_raw = (RUN / 'result.json').read_bytes()
    manifest, result = json.loads(manifest_raw), json.loads(result_raw)
    assert receipt['process_exit_code'] == 0 and receipt['interrupted'] is False
    assert receipt['failure_file_present'] is False and not (RUN / 'failure.json').exists()
    assert receipt['source_revision'] == COMMIT == manifest['source']['revision']
    assert result['status'] == 'complete' and result['submitted_forwards'] == result['returned_forwards'] == 288
    assert result['raw_payload_bytes'] == manifest['raw_payload_bytes'] == 80604288
    assert result['manifest_sha256'] == digest(manifest_raw)
    assert receipt['pre_run_sha256'] == digest((ATTEMPT / 'pre-run.json').read_bytes())
    for key in ('stdout', 'stderr'):
        raw = (ATTEMPT / (key + '.bin')).read_bytes()
        assert len(raw) == receipt[key + '_bytes'] and digest(raw) == receipt[key + '_sha256']
    assert receipt['stderr_bytes'] == 0
    for name, expected in receipt['run_sha256'].items():
        assert digest((RUN / name).read_bytes()) == expected
    assert len(manifest['source']['sha256']) == 30
    paths = list(manifest['source']['sha256'])
    batch = b''.join(f'{COMMIT}:{p}\n'.encode() for p in paths)
    response = subprocess.run(['git', 'cat-file', '--batch'], cwd=TREE, input=batch,
                              stdout=subprocess.PIPE, check=True).stdout
    i = 0
    for path in paths:
        end = response.index(b'\n', i)
        _, kind, n = response[i:end].split()
        n = int(n)
        assert kind == b'blob' and digest(response[end + 1:end + 1 + n]) == manifest['source']['sha256'][path]
        i = end + 1 + n
        assert response[i:i + 1] == b'\n'
        i += 1
    assert i == len(response)
    assert manifest['arms'] == [list(a) for a in ARMS]
    assert result['model_state_sha256_before'] == result['model_state_sha256_after']
    assert result['model_buffers_sha256_before'] == result['model_buffers_sha256_after']
    assert result['runtime']['device'] == 'cpu' and result['runtime']['cuda_initialized'] is False
    assert result['runtime']['guard_blocked_calls'] == []
    backend = {'MKL_CBWR': {'present': False, 'value': None},
               'ATEN_CPU_CAPABILITY': {'present': False, 'value': None}}
    assert result['runtime']['cpu_backend_environment'] == manifest['cpu_backend_environment_at_torch_import'] == backend
    events = [json.loads(x) for x in (RUN / 'attempt.jsonl').read_bytes().splitlines()]
    assert len(events) == 865 and events[0]['event'] == 'runtime_ready'
    assert events[0]['actual_runtime'] == result['runtime']
    rows = result['cases']
    assert len(manifest['cases']) == 32 and len(rows) == 288
    cursor = 1
    total = 0
    stored = {}
    changed = collections.defaultdict(list)
    clean_top1 = []
    for case_index, case in enumerate(manifest['cases']):
        group, variant = divmod(case_index, 4)
        assert case['group'] == group + 1 and case['variant'] == variant
        assert case['case_id'] == f'copy-followup-{group + 1:02d}-{variant}'
        raw_file = (RUN / (case['case_id'] + '.f32')).read_bytes()
        assert len(raw_file) == 2518884
        total += len(raw_file)
        order = [a[0] for a in ARMS]
        rotation = (group + 2 * variant) % 6
        order = ['clean', 'noop'] + order[rotation:] + order[:rotation] + ['clean_after']
        offset = 0
        clean = noop = None
        selected = {}
        for action_index, action in enumerate(order):
            row = rows[9 * case_index + action_index]
            assert (row['case_id'], row['group'], row['variant'], row['target_id'], row['action']) == (
                case['case_id'], group + 1, variant, case['target_id'], action)
            start, recorded, completed = events[cursor:cursor + 3]
            cursor += 3
            assert start == {'event': 'start', 'case_id': case['case_id'], 'action': action}
            assert recorded == {'event': 'raw_recorded', 'case_id': case['case_id'],
                                'action': action, 'raw': row['raw']}
            assert completed == {'event': 'complete', **row}
            payload = {}
            required = {'logits'} if action in ('clean', 'clean_after') else {
                'logits', 'cproj_before', 'cproj_after', 'cproj_output'}
            assert set(row['raw']) == required
            for key, meta in sorted(row['raw'].items(), key=lambda kv: kv[1]['offset']):
                n = LOGITS if key == 'logits' else WITNESS
                assert meta['offset'] == offset and meta['bytes'] == n
                raw = raw_file[offset:offset + n]
                assert len(raw) == n and digest(raw) == meta['sha256']
                offset += n
                payload[key] = raw
            score = describe_logits(payload['logits'], case['target_id'])
            assert score == row['score']
            selected[action] = score
            if action == 'clean':
                assert row['intervention'] is None
                clean = payload['logits']
                clean_top1.append(score['top_token_id'] == case['target_id'])
            elif action == 'clean_after':
                assert row['intervention'] is None and payload['logits'] == clean
            else:
                before, after, output = (payload[k] for k in ('cproj_before', 'cproj_after', 'cproj_output'))
                assert all(math.isfinite(x) for x in struct.unpack('<8448f', before))
                assert all(math.isfinite(x) for x in struct.unpack('<8448f', output))
                if action == 'noop':
                    assert row['intervention'] is None and before == after and payload['logits'] == clean
                    noop = before
                else:
                    assert before == noop
                    _, head, pos = next(a for a in ARMS if a[0] == action)
                    first, last = pos * 768 + head * 64, pos * 768 + (head + 1) * 64
                    a, b = struct.unpack('<8448I', before), struct.unpack('<8448I', after)
                    assert a[:first] == b[:first] and a[last:] == b[last:]
                    assert all(x == 0 for x in b[first:last])
                    values = struct.unpack('<8448f', before)[first:last]
                    changed_count = sum(x != y for x, y in zip(a[first:last], b[first:last]))
                    expected = {'layer': 3, 'head': head, 'position': pos, 'channels': [64 * head, 64 * (head + 1)],
                                'pre_head_l2': math.sqrt(math.fsum(float(x) * x for x in values)),
                                'changed_float32_bit_patterns': changed_count}
                    assert row['intervention'] == expected
                    changed[action].append(changed_count)
        assert offset == len(raw_file)
        stored[(group, variant)] = selected
    assert cursor == len(events) and total == 80604288

    families = []
    for group in range(8):
        cases = [stored[(group, variant)] for variant in range(4)]
        clean_ranks = [c['clean']['target_rank'] for c in cases]
        clean_top = [c['clean']['top_token_id'] == manifest['cases'][group * 4 + v]['target_id']
                     for v, c in enumerate(cases)]
        fam = {'group': group + 1, 'arms': {}, 'candidate_minus_arm': {},
               'clean_target_ranks': clean_ranks, 'clean_target_top1': clean_top}
        for name, _, _ in ARMS:
            delta = [c[name]['target_log_probability'] - c['clean']['target_log_probability'] for c in cases]
            shuffle = math.fsum(delta[1:]) / 3
            fam['arms'][name] = {'case_deltas': delta, 'repeat_delta': delta[0],
                'shuffle_mean_delta': shuffle, 'family_contrast': delta[0] - shuffle,
                'target_ranks': [c[name]['target_rank'] for c in cases],
                'target_top1': [c[name]['top_token_id'] == manifest['cases'][group * 4 + v]['target_id']
                                for v, c in enumerate(cases)]}
        candidate = fam['arms']['h10_final']['family_contrast']
        for name, _, _ in ARMS[1:]:
            fam['candidate_minus_arm'][name] = candidate - fam['arms'][name]['family_contrast']
        families.append(fam)
    summary = {'families': families,
        'top1_clean': {'repeated': sum(f['clean_target_top1'][0] for f in families),
                       'shuffled': sum(sum(f['clean_target_top1'][1:]) for f in families)},
        'mean_contrast_by_arm': {n: math.fsum(f['arms'][n]['family_contrast'] for f in families) / 8 for n, _, _ in ARMS},
        'mean_repeat_delta_by_arm': {n: math.fsum(f['arms'][n]['repeat_delta'] for f in families) / 8 for n, _, _ in ARMS},
        'mean_shuffle_delta_by_arm': {n: math.fsum(f['arms'][n]['shuffle_mean_delta'] for f in families) / 8 for n, _, _ in ARMS},
        'top1_by_arm': {n: {'repeated': sum(f['arms'][n]['target_top1'][0] for f in families),
                            'shuffled': sum(sum(f['arms'][n]['target_top1'][1:]) for f in families)} for n, _, _ in ARMS},
        'mean_candidate_minus_arm': {n: math.fsum(f['candidate_minus_arm'][n] for f in families) / 8 for n, _, _ in ARMS[1:]}}
    assert summary == result['summary']
    stdout = json.loads((ATTEMPT / 'stdout.bin').read_bytes())
    assert stdout == {'status': 'complete', 'forwards': 288, 'raw_payload_bytes': total, 'summary': summary}
    print('PASS: 288 logits, 192 ablation and 32 noop witnesses, 865 ordered events; 80,604,288 raw bytes.')
    print('means: arm, repeat-minus-shuffle contrast, repeat delta, shuffle delta, top1 repeated/shuffled')
    for name, _, _ in ARMS:
        top = summary['top1_by_arm'][name]
        print(f'{name:12} {summary["mean_contrast_by_arm"][name]:+12.9f} '
              f'{summary["mean_repeat_delta_by_arm"][name]:+12.9f} '
              f'{summary["mean_shuffle_delta_by_arm"][name]:+12.9f} '
              f'{top["repeated"]}/8 {top["shuffled"]}/24')
    print('clean top1:', summary['top1_clean'])
    print('family contrasts in arm order:')
    for fam in families:
        print(fam['group'], ' '.join(f'{fam["arms"][n]["family_contrast"]:+.9f}' for n, _, _ in ARMS))
    print('changed coordinates min/max:', {n: [min(v), max(v)] for n, v in changed.items()})
    print('source blobs and direct-process receipt verified; no model or descendant attestation.')


if __name__ == '__main__':
    main()
