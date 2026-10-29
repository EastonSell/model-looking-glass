"""Independently replay equal-length T5 response from saved bytes, never a model.

Usage: python flan_equal_length_independent_saved_audit.py RUN_DIR CASES_JSON SOURCE_CHECKOUT
This auditor is frozen before reading the outcome. It derives every case gate,
dose, intervention landing, cell and aggregate decision from retained tensors.
It checks the separately observed public source bytes and selected cases. It is
a consistency audit, not independent proof of model execution or chronology.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import torch

if not __debug__:
    raise RuntimeError('this evidence auditor must run without Python -O')
torch.set_num_threads(1)
assert torch.set_flush_denormal(False)
assert len(sys.argv) == 4
root, case_path, source_checkout = map(Path, sys.argv[1:])

def sha(data):
    return hashlib.sha256(data).hexdigest()

def same(a, b):
    return a.shape == b.shape and np.array_equal(a.view('<u4'), b.view('<u4'))

def distance(a, b):
    assert a.shape == b.shape
    delta = a.astype(np.float64).reshape(-1) - b.astype(np.float64).reshape(-1)
    return math.sqrt(math.fsum(float(x) * float(x) for x in delta))

def selected_logprobs(logits, ids):
    actual = torch.log_softmax(torch.from_numpy(logits[0].copy()), dim=-1)
    return [float(actual[i, token]) for i, token in enumerate(ids)]

def derive_cell(delta, curves):
    assert math.isfinite(delta) and delta > 0
    reasons = []
    for side in (0, 1):
        previous = 0.0
        for i, fraction in enumerate((1/3, 2/3, 1.0), 1):
            word = curves['word'][i][side]
            control = max(abs(curves['instruction'][i][side]), abs(curves['punctuation'][i][side]))
            assert math.isfinite(word) and math.isfinite(control)
            if word < -1e-4:
                reasons.append(f'reversed-side{side}-level{i}')
            if word < previous - 1e-4:
                reasons.append(f'nonmonotone-side{side}-level{i}')
            if word < control + .02 * fraction * delta:
                reasons.append(f'control_comparison-side{side}-level{i}')
            previous = word
        if previous < .05 * delta:
            reasons.append(f'weak_endpoint-side{side}')
    return ('negative' if reasons else 'positive'), reasons

def derive_aggregate(statuses):
    assert len(statuses) == 12
    pairs = []
    for i in range(6):
        x, y = statuses[2*i:2*i+2]
        pairs.append('unevaluable' if x not in ('positive', 'negative') or y not in ('positive', 'negative')
                     else 'positive' if x == y == 'positive' else 'negative')
    p, n = pairs.count('positive'), pairs.count('negative')
    return pairs, ('positive' if p >= 3 and p > n else 'negative' if n >= 3 and n > p else 'inconclusive')

def test_decision_helpers():
    # These fabricated checks run before any evidence is read. Ineligible is
    # never a negative, and a 3-to-3 evaluable tie is inconclusive.
    import itertools
    statuses = ('positive', 'negative', 'baseline-unsupported', 'dose-ineligible', 'dose-unmatchable')
    for x, y in itertools.product(statuses, repeat=2):
        p, d = derive_aggregate([x, y] + ['baseline-unsupported'] * 10)
        expected = ('unevaluable' if x not in statuses[:2] or y not in statuses[:2] else
                    'positive' if x == y == 'positive' else 'negative')
        assert p == [expected] + ['unevaluable'] * 5 and d == 'inconclusive'
    assert derive_aggregate(['positive'] * 6 + ['negative'] * 6)[1] == 'inconclusive'
    assert derive_aggregate(['positive'] * 6 + ['baseline-unsupported'] * 6)[1] == 'positive'
    assert derive_aggregate(['negative'] * 6 + ['baseline-unsupported'] * 6)[1] == 'negative'
    good = {'word': [[0, 0], [.25, .25], [.5, .5], [.75, .75]],
            'instruction': [[0, 0]] * 4, 'punctuation': [[0, 0]] * 4}
    assert derive_cell(10.0, good) == ('positive', [])
    bad = {'word': [[0, 0], [-.2, .25], [-.5, .5], [-.75, .75]],
           'instruction': [[0, 0]] * 4, 'punctuation': [[0, 0]] * 4}
    status, reasons = derive_cell(10.0, bad)
    assert status == 'negative' and any(r.startswith('reversed-side0') for r in reasons)
    assert any(r.startswith('nonmonotone-side0') for r in reasons)
    assert 'weak_endpoint-side0' in reasons and any(r.startswith('control_comparison-side0') for r in reasons)

test_decision_helpers()

cases_bytes = case_path.read_bytes()
manifest = json.loads(cases_bytes)
started = json.loads((root / 'started.json').read_bytes())
completed_bytes = (root / 'completed.json').read_bytes()
completed = json.loads(completed_bytes)
assert sha(completed_bytes) == '6e4caed8a4b5372c09d30e3140a3ce50dadf1b822dbcb69df40e8efd1d6713a8'
assert manifest['schema'] == 'mlg-flan-t5-equal-length-dose-response-cases-v1'
assert sha(cases_bytes) == '8d1ea3b124085ca8817dc80044dca6fd1e897bda7294510ad99e419c01e7708b'
assert manifest['revision'] == '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
assert manifest['model'] == 'google/flan-t5-small'
assert len(manifest['cases']) == len(completed['cases']) == 12
assert [c['candidate_index'] for c in manifest['cases']] == [0, 0, 1, 1, 2, 2, 3, 3, 5, 5, 7, 7]
assert [c['case_id'] for c in manifest['cases']] == [
    f'pair{pair}-{template}' for pair in range(6) for template in ('translate', 'question')]
assert all(c['alignment_eligible'] is True and c['eligibility_reasons'] == [] for c in manifest['cases'])
assert completed['schema'] == 'mlg-flant5-equal-length-response-v1' and completed['status'] == 'complete'
assert completed['source'] == started['source']
assert completed['case_manifest_sha256'] == started['case_manifest_sha256'] == sha(cases_bytes)
assert completed['asset_manifest_sha256'] == started['asset_manifest_sha256'] == manifest['asset_manifest_sha256']
assert manifest['asset_manifest_sha256'] == 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
assert completed['selection_plan_sha256'] == started['selection_plan_sha256'] == manifest['selection_plan_sha256'] == '1fa773268a2bbfd73979c2ce9f83f6a1d821f6607c37c12460b2368b2ca0668c'
assert completed['candidate_screen_sha256'] == started['candidate_screen_sha256'] == manifest['candidate_screen_sha256'] == 'adabba92e451fb0c27415b8ec67b67134b59bb5aac293b3c056936cfe872c0a2'
assert completed['hooks_restored'] is True
runtime = json.loads((root / 'environment.json').read_bytes())
assert runtime == completed['environment']
assert (runtime['device'], runtime['dtype'], runtime['threads'], runtime['interop_threads'],
        runtime['deterministic'], runtime['attention_implementation']) == ('cpu', 'float32', 1, 1, True, 'eager')
assert all(runtime['numeric_environment'][key] == value for key, value in
           (('MKL_CBWR', 'COMPATIBLE'), ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1')))
publication_bytes = (root / 'publication-observation.json').read_bytes()
assert sha(publication_bytes) == started['publication_observation_sha256']
assert type(completed['forwards']) is int and 0 <= completed['forwards'] <= 936
assert started['max_forwards'] == 936 and started['evidence_cap'] == 512 * 1024 * 1024
assert started['fixed_evidence_bound'] == completed['fixed_evidence_bound']
assert completed['fixed_evidence_bound']['total_bytes'] == 437178368
assert not (root / 'failure.json').exists()
assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= 512 * 1024 * 1024
if source_checkout is not None:
    local_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_checkout, text=True).strip()
    assert local_commit == '5db7543cb87cf8fc0c7737d74c8f217a9f186d0f'
    commit = '9d9aa69f208f8152db9c2832e0fc4ed63ef16df4'
    assert commit == completed['source']['commit']
    observation = json.loads(publication_bytes)
    assert observation['schema'] == 'mlg-flant5-public-observation-v1'
    assert observation['commit'] == commit and observation['source_sha256'] == completed['source']['source_sha256']
    assert sha(publication_bytes) == 'd7e78ae2ee1fb41fc41149ca45425819bcf359e41c00a28cb9f89892c0b7743e'
    assert observation['commit_url'] == f'https://github.com/EastonSell/model-looking-glass/commit/{commit}'
    assert observation['observer'] and observation['observed_utc']
    assert set(completed['source']['source_sha256']) == {
        'looking_glass/flan_t5_equal_length_response.py', 'looking_glass/flan_t5_dose_response.py', 'looking_glass/flan_t5_equal_dose.py', 'looking_glass/flan_t5_feasibility.py',
        'looking_glass/interventions.py', 'looking_glass/intervention_plan.py',
        'looking_glass/model_ownership.py', 'docs/FLAN_T5_EQUAL_LENGTH_RESPONSE_V1.md',
        'docs/FLAN_T5_EQUAL_LENGTH_SELECTION_V1.md', 'docs/FLAN_T5_EQUAL_LENGTH_SCREEN_V1.json',
        'docs/FLAN_T5_EQUAL_LENGTH_CASES_V1.json', 'tests/flan_t5_equal_length_tokenizer_screen.py'}
    for relative, expected in completed['source']['source_sha256'].items():
        working = (source_checkout / relative).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{local_commit}:{relative}'], cwd=source_checkout)
        assert working == committed and sha(working) == expected

by_name = {}
expected_order = []
max_f64_difference = 0.0
for index in range(completed['forwards']):
    row = json.loads((root / f'forward-{index:03d}.json').read_bytes())
    observed = json.loads((root / f'forward-{index:03d}-observed.json').read_bytes())
    assert row['index'] == index and row['file'] == f'forward-{index:03d}.bin'
    assert row['name'] not in by_name and row['dtype'] == 'little-endian-float32'
    assert observed['index'] == index and observed['forward'] == row['name']
    assert observed['phase'] == 'post-return-validation'
    assert observed['case_id'] == row['name'].split('-context-')[0].split('-base-')[0].split('-null_')[0].split('-self_')[0].split('-full_')[0].split('-zero_')[0].split('-matched_')[0].split('-repeat-')[0]
    assert observed['return_observed'] is True
    assert observed['saved_file'] == row['file'] and observed['saved_sha256'] == row['sha256']
    assert observed['tensor_layout'] == row['tensor_layout']
    assert observed['session_metadata'] == row['session_metadata']
    assert row['target_ids'][-1] == 1 and row['decoder_ids'] == [0] + row['target_ids'][:-1]
    assert row['decoder_mask'] == [1] * len(row['decoder_ids'])
    raw = (root / row['file']).read_bytes()
    assert sha(raw) == row['sha256']
    tensors, offset = {}, 0
    for name in ('logits', 'encoder', 'before', 'after', 'capture'):
        info = row['tensor_layout'][name]
        length = info['bytes']
        assert info['offset'] == offset
        if not info['available']:
            assert info['shape'] is None and length == 0
            tensors[name] = None
            continue
        assert type(info['shape']) is list and length == 4 * math.prod(info['shape'])
        tensors[name] = np.frombuffer(raw, dtype='<f4', count=length // 4, offset=offset).reshape(info['shape'])
        assert np.isfinite(tensors[name]).all()
        offset += length
    assert offset == len(raw)
    logits, state = tensors['logits'], tensors['encoder']
    assert logits.shape == (1, len(row['target_ids']), 32128)
    assert state.shape == (1, len(row['source_ids']), 512)
    selected = selected_logprobs(logits, row['target_ids'])
    assert selected == row['selected_logprobs'] and math.fsum(selected) == row['score']
    for token_index, token in enumerate(row['target_ids']):
        values = logits[0, token_index].astype(np.float64)
        peak = float(values.max())
        normalizer = peak + math.log(math.fsum(math.exp(float(x) - peak) for x in values))
        difference = abs((float(values[token]) - normalizer) - selected[token_index])
        max_f64_difference = max(max_f64_difference, difference)
        # Diagnostic bound fixed before any pretrained follow-up output. Exact
        # float32 selected log-probabilities were checked independently above.
        assert difference <= 1e-3
    by_name[row['name']] = {'row': row, **tensors}

def get(name, source, target):
    item = by_name[name]
    expected_order.append(name)
    row = item['row']
    assert row['source_ids'] == source['input_ids'] and row['source_mask'] == source['attention_mask']
    assert row['target_ids'] == target['input_ids']
    return item

def plan_check(item, action, position, sequence_length):
    steps = item['row']['session_metadata']['steps']
    assert len(steps) == 3
    for index, (step, name, operation) in enumerate(zip(steps, ('before', 'action', 'after'),
                                                        ('capture', action, 'capture'))):
        assert step['name'] == name and step['operation'] == operation and step['order'] == index
        assert step['module_path'] == 'encoder.final_layer_norm' and step['invocation'] == 0
        assert 'encoder.final_layer_norm' in step['module_aliases']
        assert step['forward_index'] == 0 and step['call_site_known'] is False
        assert step['batch_index'] == (None if position is None else 0) and step['token_index'] == position
        assert step['output_shape'] == [1, sequence_length, 512]
        assert step['selected_shape'] == ([1, sequence_length, 512] if position is None else [512])
        assert step['output_selector'] == [] and step['dtype'] == 'torch.float32' and step['device'] == 'cpu'
        assert step['source_capture'] is None and step['preceding_same_site_actions'] == ['before', 'action'][:index]

used = set()
outcomes = []
for ci, (case, saved) in enumerate(zip(manifest['cases'], completed['cases'])):
    assert saved['case_id'] == case['case_id'] and saved['pair_id'] == ci // 2
    assert saved['template_id'] == ('translate', 'question')[ci % 2]
    assert json.loads((root / f'case-{ci:02d}.json').read_bytes()) == saved
    assert case['alignment_eligible'] is True
    targets = case['targets']
    context = [get(f"{case['case_id']}-context-{t}", case['neutral'], targets[t]) for t in range(2)]
    bases = {(side, t): get(f"{case['case_id']}-base-{side}-{t}", case['source'][side], targets[t])
             for side in range(2) for t in range(2)}
    used.update(x['row']['name'] for x in context + list(bases.values()))
    assert all(x['row']['action'] == 'clean' and x['row']['token_index'] is None and x['row']['session_metadata'] is None
               and x['before'] is None and x['after'] is None and x['capture'] is None
               for x in context + list(bases.values()))
    assert all(same(bases[side, 0]['encoder'], bases[side, 1]['encoder']) for side in range(2))
    states = [bases[side, 0]['encoder'] for side in range(2)]
    margins = [bases[s, 0]['row']['score'] - bases[s, 1]['row']['score'] for s in range(2)]
    delta = margins[0] - margins[1]
    context_scores = [x['row']['score'] for x in context]
    assert saved['baseline_margins'] == margins and saved['delta'] == delta
    assert saved['context_scores'] == context_scores and saved['context_margin'] == context_scores[0] - context_scores[1]
    baseline_supported = margins[0] > 0 and margins[1] < 0 and delta >= 1.0
    if not baseline_supported:
        assert saved['status'] == 'baseline-unsupported'
        assert not (root / f'dose-{ci:02d}.json').exists() and not (root / f'dose-{ci:02d}.bin').exists()
        outcomes.append(saved)
        continue

    positions = {'word': case['word_position'], 'instruction': case['instruction_position'],
                 'punctuation': case['context_position']}
    assert positions == {'word': 6, 'instruction': 0, 'punctuation': (5 if ci % 2 == 0 else 7)}
    dose_record = json.loads((root / f'dose-{ci:02d}.json').read_bytes())
    dose = dose_record['dose']
    assert dose_record['case_id'] == case['case_id'] and dose == saved['dose']
    assert dose_record['file'] == f'dose-{ci:02d}.bin'
    dose_raw = (root / dose_record['file']).read_bytes()
    assert sha(dose_raw) == dose_record['sha256']
    vectors, cursor = {}, 0
    for item in dose_record['layout']:
        assert item['offset'] == cursor and item['bytes'] == 2048
        assert item['dtype'] == 'little-endian-float32' and item['shape'] == [512]
        key = (item['site'], item['side'], item['role'])
        assert key not in vectors
        vectors[key] = np.frombuffer(dose_raw[cursor:cursor + 2048], dtype='<f4')
        cursor += 2048
    assert cursor == len(dose_raw)
    distances = {site: distance(states[0][0, pos], states[1][0, pos]) for site, pos in positions.items()}
    assert distances == dose['distances']
    for site, pos in positions.items():
        for side in range(2):
            assert same(vectors[site, side, 'own'], states[side][0, pos])
            assert same(vectors[site, side, 'donor'], states[1-side][0, pos])
    q = min(distances.values())
    assert dose['q'] == q
    if q / 3 <= 1e-6:
        assert dose['status'] == saved['status'] == 'dose-ineligible' and len(vectors) == 12
        outcomes.append(saved)
        continue
    levels = ((1, 3), (2, 3), (1, 1))
    assert len(vectors) == 30 and len(dose['levels']) == len(levels)
    receiver_norms = {f'{site}-{side}': distance(states[side][0, pos], np.zeros_like(states[side][0, pos]))
                      for site, pos in positions.items() for side in range(2)}
    matchable = True
    for (numerator, denominator), level in zip(levels, dose['levels']):
        level_q = q * (numerator / denominator)
        tolerance = min(max(1e-5, 1e-5 * level_q), .01 * level_q)
        assert (level['numerator'], level['denominator'], level['q'], level['tolerance']) == (
            numerator, denominator, level_q, tolerance)
        achieved = {}
        for site, pos in positions.items():
            fraction = level_q / distances[site]
            for side in range(2):
                own, donor = states[side][0, pos], states[1-side][0, pos]
                delivered = vectors[site, side, f'{numerator}-{denominator}']
                expected = b''.join(struct.pack('<f', float(a) + fraction * (float(b) - float(a)))
                                    for a, b in zip(own, donor))
                assert delivered.tobytes() == expected
                achieved[f'{site}-{side}'] = distance(delivered, own)
        assert level['achieved'] == achieved and level['receiver_norms'] == receiver_norms
        assert level['q_over_receiver_norm'] == {
            key: level_q / norm if norm > 0 else None for key, norm in receiver_norms.items()}
        level_match = (all(x > 0 and abs(x-level_q) <= tolerance for x in achieved.values())
                       and all(abs(achieved[f'word-{side}'] - achieved[f'{site}-{side}'])
                               <= .01 * achieved[f'word-{side}']
                               for side in range(2) for site in ('instruction', 'punctuation')))
        assert level['status'] == ('matchable' if level_match else 'dose-unmatchable')
        matchable = matchable and level_match
    if not matchable:
        assert dose['status'] == saved['status'] == 'dose-unmatchable'
        outcomes.append(saved)
        continue
    assert dose['status'] == 'matchable'
    assert saved['positions'] == positions
    anchors = ('null_word', 'self_word', 'self_instruction', 'self_punctuation',
               'full_self', 'full_swap', 'full_word_swap', 'zero_word')
    conditions = (*anchors, *(f'matched_{site}_{a}of{b}' for a, b in levels for site in positions), 'repeat')
    controls = {}
    for condition in conditions:
        for side in range(2):
            own, donor = states[side], states[1-side]
            if condition == 'repeat':
                action, pos, expected = 'capture', None, None
            elif condition in ('full_self', 'full_swap'):
                action, pos, expected = 'patch', None, own if condition == 'full_self' else donor
            else:
                site = ('instruction' if condition.endswith('instruction') or condition.startswith('matched_instruction_') else
                        'punctuation' if condition.endswith('punctuation') or condition.startswith('matched_punctuation_') else 'word')
                pos = positions[site]
                action = 'null' if condition == 'null_word' else 'zero' if condition == 'zero_word' else 'patch'
                role = condition.rsplit('_', 1)[-1].replace('of', '-') if condition.startswith('matched_') else None
                expected = (np.zeros_like(own[0, pos]) if action == 'zero' else
                            own[0, pos] if action == 'null' else
                            vectors[site, side, role] if condition.startswith('matched_') else
                            donor[0, pos] if condition == 'full_word_swap' else own[0, pos])
            for t in range(2):
                item = get(f"{case['case_id']}-{condition}-{side}-{t}", case['source'][side], targets[t])
                used.add(item['row']['name'])
                controls[condition, side, t] = item
                assert item['row']['action'] == action and item['row']['token_index'] == pos
                if condition in ('repeat', 'null_word', 'self_word', 'self_instruction', 'self_punctuation', 'full_self'):
                    assert same(item['logits'], bases[side, t]['logits'])
                if condition == 'full_swap':
                    assert same(item['logits'], bases[1-side, t]['logits']) and same(item['encoder'], donor)
                if action == 'capture':
                    assert item['before'] is None and item['after'] is None and same(item['capture'], own)
                    assert same(item['encoder'], own)
                    metadata = item['row']['session_metadata']
                    assert metadata['module_path'] == 'encoder.final_layer_norm'
                    assert 'encoder.final_layer_norm' in metadata['module_aliases']
                    assert metadata['call_site_known'] is False and metadata['forward_index'] == 0
                    assert metadata['invocation'] == 0 and metadata['batch_index'] is None
                    assert metadata['token_index'] is None and metadata['output_shape'] == list(own.shape)
                    assert metadata['selected_shape'] == list(own.shape)
                    assert metadata['dtype'] == 'torch.float32' and metadata['device'] == 'cpu'
                else:
                    assert item['capture'] is None
                    plan_check(item, action, pos, len(case['source'][side]['input_ids']))
                    assert same(item['before'], own if pos is None else own[0, pos])
                    assert same(item['after'], expected)
                    if pos is None:
                        assert same(item['encoder'], expected)
                    else:
                        assert same(item['encoder'][0, pos], expected)
                        assert all(same(item['encoder'][0, p], own[0, p])
                                   for p in range(own.shape[1]) if p != pos)
    condition_margins = {c: [controls[c, s, 0]['row']['score'] - controls[c, s, 1]['row']['score']
                             for s in range(2)] for c in conditions}
    movements = {c: [margins[0] - v[0], v[1] - margins[1]] for c, v in condition_margins.items()}
    assert saved['condition_margins'] == condition_margins and saved['signed_movements'] == movements
    assert saved['absolute_movements'] == {c: [abs(x) for x in v] for c, v in movements.items()}
    curves = {site: [[0.0, 0.0], *[movements[f'matched_{site}_{a}of{b}'] for a, b in levels]]
              for site in positions}
    assert saved['response_curves'] == curves
    assert saved['absolute_response_curves'] == {
        site: [[abs(x) for x in row] for row in curve] for site, curve in curves.items()}
    decision, reasons = derive_cell(delta, curves)
    assert saved['reasons'] == reasons
    assert saved['status'] == decision
    outcomes.append(saved)

assert used == set(by_name)
assert expected_order == list(by_name), 'saved forward record order differs from frozen schedule'
assert len(list(root.glob('forward-*.bin'))) == completed['forwards']
assert len(list(root.glob('forward-*.json'))) == 2 * completed['forwards']
assert len(list(root.glob('case-*.json'))) == len(manifest['cases'])
assert len(list(root.glob('dose-*.json'))) == len(list(root.glob('dose-*.bin'))) == sum(
    saved['status'] in ('positive', 'negative', 'dose-ineligible', 'dose-unmatchable') for saved in outcomes)
pair_status, decision = derive_aggregate([case['status'] for case in outcomes])
assert completed['pair_status'] == pair_status
assert completed['decision'] == decision
assert (root / 'completed.json').read_bytes() == completed_bytes
assert (root / 'publication-observation.json').read_bytes() == publication_bytes
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_checkout, text=True).strip() == local_commit
for relative, expected in completed['source']['source_sha256'].items():
    assert sha((source_checkout / relative).read_bytes()) == expected
result = {'status': 'passed', 'terminal_sha256': sha(completed_bytes),
          'case_manifest_sha256': sha(cases_bytes), 'source': completed['source'],
          'source_checkout_verified': True,
          'forwards': len(by_name), 'max_abs_f64_vs_saved_f32_token_score': max_f64_difference,
          'f64_per_token_tolerance': 1e-3, 'decision': decision, 'pair_status': pair_status,
          'cases': outcomes,
          'scope': 'saved-evidence consistency only; no model execution or held-out data',
          'limits': ['Public related examples; no statistical or broad generalization claim.',
                     'Equal target length does not equalize lexical priors.',
                     'Scores include EOS and do not establish greedy generation.',
                     'Equal L2 does not imply equal functional perturbation.',
                     'The question context-only prompt also omits the question mark.']}
print(json.dumps(result, sort_keys=True, allow_nan=False))
