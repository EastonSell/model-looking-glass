"""Independently replay held-out FLAN-T5 scores, doses and decisions with stdlib.

This checks saved-byte consistency, not model execution or unique localization.
The f64 log-sum-exp check is diagnostic; exact f32 replay is a separate audit.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

if not __debug__ or sys.byteorder != 'little':
    raise RuntimeError('requires assertions and a little-endian CPU')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--custody-root', type=Path, required=True)
parser.add_argument('--terminal-sha256', required=True)
parser.add_argument('--auditor-sha256', required=True)
parser.add_argument('--source-checkout', type=Path)
parser.add_argument('--source-commit')
parser.add_argument('--synthetic', action='store_true')
args = parser.parse_args()
if args.synthetic == (args.source_checkout is not None or args.source_commit is not None):
    raise ValueError('choose synthetic mode or both source-checkout and source-commit')
if not args.synthetic and (args.source_checkout is None or args.source_commit is None):
    raise ValueError('production audit requires exact source checkout and commit')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


self_path = Path(__file__).resolve()
self_hash = sha(self_path.read_bytes())
if self_hash != args.auditor_sha256:
    raise ValueError('auditor bytes differ from the external pin')
def canonical_file(name, limit=4 * 1024 * 1024):
    raw = (args.custody_root / name).read_bytes()
    assert len(raw) <= limit
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            assert key not in obj
            obj[key] = value
        return obj
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
    assert type(value) is dict
    assert json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('utf-8') == raw
    return raw, value

pool_raw, pool = canonical_file('pool.json')
case_raw, payload = canonical_file('cases-payload.json')
_, screen = canonical_file('screen.json')
_, pool_card = canonical_file('pool-commitment.json')
_, case_card = canonical_file('case-commitment.json')
pool_salt = (args.custody_root / 'pool-salt.bin').read_bytes()
case_salt = (args.custody_root / 'case-salt.bin').read_bytes()
assert len(pool_salt) == len(case_salt) == 32 and pool_salt != case_salt
assert pool['schema'] == 'mlg-flant5-heldout-pool-v1'
assert payload['schema'] == 'mlg-flant5-heldout-cases-v1'
assert pool_card['commitment'] == sha(b'mlg-flant5-heldout-pool-v1\x00' + pool_salt + pool_raw)
assert case_card['commitment'] == sha(b'mlg-flant5-heldout-cases-v1\x00' + case_salt + case_raw)
assert payload['pool_commitment'] == screen['pool_commitment'] == pool_card['commitment']
assert case_card['pool_commitment'] == pool_card['commitment'] and payload['screen'] == screen
for key in ('source_commit', 'protocol_sha256', 'asset_manifest_sha256', 'tokenizer_revision'):
    assert pool[key] == payload[key] == pool_card[key] == case_card[key]
assert screen['status'] == 'complete' and len(screen['candidates']) == 24
eligible = [item['candidate_index'] for item in screen['candidates'] if item['eligible']]
assert screen['selected_candidate_indices'] == eligible[:6] and len(eligible) >= 6
manifest = {'cases': payload['cases']}
terminal_raw = (args.run_dir / 'completed.json').read_bytes()
assert sha(terminal_raw) == args.terminal_sha256
completed = json.loads(terminal_raw)
started = json.loads((args.run_dir / 'started.json').read_bytes())
assert completed['schema'] == 'mlg-flant5-heldout-response-v1' and completed['status'] == 'complete'
assert started['schema'] == 'mlg-flant5-heldout-response-attempt-v1' and started['status'] == 'started'
assert started['source'] == completed['source']
assert started['pool_commitment'] == completed['pool_commitment'] == pool_card['commitment']
assert started['case_commitment'] == completed['case_commitment'] == case_card['commitment']
assert started['asset_manifest_sha256'] == completed['asset_manifest_sha256'] == pool['asset_manifest_sha256']
assert completed['source']['commit'] == pool['source_commit']
assert completed['source']['source_sha256']['docs/FLAN_T5_HELDOUT_RESPONSE_V1.md'] == pool['protocol_sha256']
assert len(manifest['cases']) == len(completed['cases']) == 12
assert type(completed['forwards']) is int and 0 <= completed['forwards'] <= 936
assert completed['hooks_restored'] is True and not (args.run_dir / 'failure.json').exists()
runtime = json.loads((args.run_dir / 'environment.json').read_bytes())
assert runtime == completed['environment']
assert (runtime['device'], runtime['dtype'], runtime['threads'], runtime['interop_threads'],
        runtime['deterministic'], runtime['attention_implementation']) == ('cpu', 'float32', 1, 1, True, 'eager')
if not args.synthetic:
    assert completed['source']['commit'] == args.source_commit
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source_checkout, text=True).strip()
    assert head == args.source_commit
    assert 'looking_glass/flan_t5_heldout_response.py' in completed['source']['source_sha256']
    def verify_source_bytes():
        for name, digest in completed['source']['source_sha256'].items():
            current = (args.source_checkout / name).read_bytes()
            blob = subprocess.check_output(['git', 'show', f'{head}:{name}'], cwd=args.source_checkout)
            assert current == blob and sha(current) == digest
    verify_source_bytes()
    observed_raw = (args.custody_root / 'source-observation.json').read_bytes()
    assert sha(observed_raw) == started['source_observation_sha256']
    observation = json.loads(observed_raw)
    assert observation['schema'] == 'mlg-flant5-public-observation-v1'
    assert observation['commit'] == head and observation['source_sha256'] == completed['source']['source_sha256']
    assert observation['commit_url'] == f'https://github.com/EastonSell/model-looking-glass/commit/{head}'


def floats(raw):
    result = array.array('f')
    assert result.itemsize == 4 and len(raw) % 4 == 0
    result.frombytes(raw)
    assert all(math.isfinite(x) for x in result)
    return result


def distance(a, b):
    assert len(a) == len(b)
    differences = [float(x) - float(y) for x, y in zip(a, b)]
    return math.sqrt(math.fsum(value * value for value in differences))


records = {}
maximum_f64_difference = 0.0
for index in range(completed['forwards']):
    row = json.loads((args.run_dir / f'forward-{index:03d}.json').read_bytes())
    assert row['index'] == index and row['name'] not in records
    assert row['file'] == f'forward-{index:03d}.bin' and row['dtype'] == 'little-endian-float32'
    target = row['target_ids']
    assert len(target) == 2 and target[-1] == 1 and target.count(1) == 1
    assert row['decoder_ids'] == [0] + target[:-1] and row['decoder_mask'] == [1, 1]
    raw = (args.run_dir / row['file']).read_bytes()
    assert sha(raw) == row['sha256']
    info = row['tensor_layout']['logits']
    assert info['available'] is True and info['shape'] == [1, 2, 32128] and info['bytes'] == 2 * 32128 * 4
    values = floats(raw[info['offset']:info['offset'] + info['bytes']])
    assert len(values) == 2 * 32128
    selected = row['selected_logprobs']
    assert len(selected) == 2 and math.fsum(selected) == row['score']
    for position, token in enumerate(target):
        assert type(token) is int and 0 <= token < 32128 and math.isfinite(selected[position])
        values_at_position = values[position * 32128:(position + 1) * 32128]
        peak = max(values_at_position)
        normalizer = peak + math.log(math.fsum(math.exp(float(x) - peak) for x in values_at_position))
        recalculated = float(values_at_position[token]) - normalizer
        difference = abs(recalculated - selected[position])
        maximum_f64_difference = max(maximum_f64_difference, difference)
        assert difference <= 1e-3
    state_bytes = None
    if '-base-' in row['name']:
        state_info = row['tensor_layout']['encoder']
        assert state_info['available'] is True and state_info['shape'] == [1, len(row['source_ids']), 512]
        state_bytes = raw[state_info['offset']:state_info['offset'] + state_info['bytes']]
        assert len(state_bytes) == len(row['source_ids']) * 512 * 4
    records[row['name']] = {'row': row, 'state_bytes': state_bytes}

used = []


def get(name, source, target):
    item = records[name]
    row = item['row']
    assert row['source_ids'] == source['input_ids'] and row['source_mask'] == source['attention_mask']
    assert row['target_ids'] == target['input_ids']
    used.append(name)
    return item


def read_dose(index, case, states, saved):
    data = json.loads((args.run_dir / f'dose-{index:02d}.json').read_bytes())
    dose = data['dose']
    assert data['case_id'] == case['case_id'] and dose == saved['dose']
    assert data['file'] == f'dose-{index:02d}.bin'
    raw = (args.run_dir / data['file']).read_bytes()
    assert sha(raw) == data['sha256']
    vectors = {}
    offset = 0
    for part in data['layout']:
        assert part['offset'] == offset and part['bytes'] == 2048
        assert part['shape'] == [512] and part['dtype'] == 'little-endian-float32'
        key = (part['site'], part['side'], part['role'])
        assert key not in vectors
        vectors[key] = floats(raw[offset:offset + 2048])
        offset += 2048
    assert offset == len(raw)
    positions = {'word': case['word_position'], 'instruction': case['instruction_position'],
                 'punctuation': case['context_position']}
    selected = {(site, side): states[side][pos * 512:(pos + 1) * 512]
                for site, pos in positions.items() for side in range(2)}
    distances = {site: distance(selected[site, 0], selected[site, 1]) for site in positions}
    assert dose['distances'] == distances
    for site in positions:
        for side in range(2):
            assert vectors[site, side, 'own'].tobytes() == selected[site, side].tobytes()
            assert vectors[site, side, 'donor'].tobytes() == selected[site, 1-side].tobytes()
    q = min(distances.values())
    assert dose['q'] == q
    if q / 3 <= 1e-6:
        assert dose['status'] == 'dose-ineligible' and len(vectors) == 12
        return dose['status']
    assert len(vectors) == 30 and len(dose['levels']) == 3
    good = True
    for (a, b), level in zip(((1, 3), (2, 3), (1, 1)), dose['levels']):
        target_q = q * (a / b)
        tolerance = min(max(1e-5, 1e-5 * target_q), .01 * target_q)
        assert (level['numerator'], level['denominator'], level['q'], level['tolerance']) == (a, b, target_q, tolerance)
        achieved = {}
        for site in positions:
            for side in range(2):
                own, donor = selected[site, side], selected[site, 1-side]
                replacement = vectors[site, side, f'{a}-{b}']
                expected = b''.join(struct.pack('<f', float(x) + (target_q / distances[site]) * (float(y) - float(x)))
                                    for x, y in zip(own, donor))
                assert replacement.tobytes() == expected
                achieved[f'{site}-{side}'] = distance(replacement, own)
        assert level['achieved'] == achieved
        matches = (all(value > 0 and abs(value - target_q) <= tolerance for value in achieved.values())
                   and all(abs(achieved[f'word-{side}'] - achieved[f'{site}-{side}'])
                           <= .01 * achieved[f'word-{side}']
                           for side in range(2) for site in ('instruction', 'punctuation')))
        assert level['status'] == ('matchable' if matches else 'dose-unmatchable')
        good = good and matches
    status = 'matchable' if good else 'dose-unmatchable'
    assert dose['status'] == status
    return status


audited_cases = []
for index, (case, saved) in enumerate(zip(manifest['cases'], completed['cases'])):
    assert saved['case_id'] == case['case_id'] and saved['pair_id'] == index // 2
    assert saved['template_id'] == f'template{index % 2}'
    assert case['case_id'] == f'pair{index // 2}-template{index % 2}'
    candidate_index = screen['selected_candidate_indices'][index // 2]
    assert case['candidate_index'] == candidate_index
    screened = screen['candidates'][candidate_index]
    assert screened['candidate_index'] == candidate_index and screened['eligible'] is True
    cell = screened['cases'][index % 2]
    extras = {'case_id', 'pair_id', 'candidate_index', 'semantic_eligible', 'semantic_reasons'}
    assert {key: value for key, value in case.items() if key not in extras} == cell
    assert type(case['semantic_eligible']) is bool and type(case['semantic_reasons']) is list
    assert json.loads((args.run_dir / f'case-{index:02d}.json').read_bytes()) == saved
    if not case['semantic_eligible']:
        assert case['semantic_reasons'] and saved['reasons'] == case['semantic_reasons']
        assert saved['status'] == saved['raw_status'] == saved['descriptive_status'] == 'semantic-ineligible'
        audited_cases.append({'case_id': case['case_id'], 'status': saved['status'],
                              'raw_status': saved['raw_status'], 'descriptive_status': saved['descriptive_status']})
        continue
    assert not case['semantic_reasons']
    if saved['status'] == 'alignment-ineligible':
        assert case['alignment_eligible'] is False
        assert saved['raw_status'] == saved['descriptive_status'] == 'alignment-ineligible'
        audited_cases.append({'case_id': case['case_id'], 'status': saved['status'],
                              'raw_status': saved['raw_status'], 'descriptive_status': saved['descriptive_status']})
        continue
    assert case['alignment_eligible'] is True
    targets = case['targets']
    context = [get(f"{case['case_id']}-context-{t}", case['neutral'], targets[t]) for t in range(2)]
    bases = {(side, t): get(f"{case['case_id']}-base-{side}-{t}", case['source'][side], targets[t])
             for side in range(2) for t in range(2)}
    assert all(bases[side, 0]['state_bytes'] == bases[side, 1]['state_bytes'] for side in range(2))
    states = [floats(bases[side, 0]['state_bytes']) for side in range(2)]
    margins = [bases[side, 0]['row']['score'] - bases[side, 1]['row']['score'] for side in range(2)]
    delta = margins[0] - margins[1]
    context_scores = [item['row']['score'] for item in context]
    assert saved['baseline_margins'] == margins and saved['delta'] == delta
    assert saved['context_scores'] == context_scores and saved['context_margin'] == context_scores[0] - context_scores[1]
    context_margin = context_scores[0] - context_scores[1]
    raw_reasons = []
    if margins[0] <= 0: raw_reasons.append('first-margin-not-positive')
    if margins[1] >= 0: raw_reasons.append('second-margin-not-negative')
    if delta < 1.0: raw_reasons.append('gap-below-one')
    context_reasons = []
    if margins[0] - context_margin <= 0: context_reasons.append('first-not-above-context')
    if margins[1] - context_margin >= 0: context_reasons.append('second-not-below-context')
    raw_gate, strict_gate = not raw_reasons, not raw_reasons and not context_reasons
    expected_gate = {'delta': delta, 'context_relative_margins': [m - context_margin for m in margins],
                     'gap_supported': delta >= 1.0, 'raw_supported': raw_gate, 'strict_supported': strict_gate,
                     'raw_reasons': raw_reasons, 'context_reasons': context_reasons}
    assert saved['baseline_gate'] == expected_gate
    audit = {'case_id': case['case_id'], 'status': saved['status'], 'baseline_margins': margins,
             'raw_status': saved['raw_status'], 'descriptive_status': saved['descriptive_status'],
             'delta': delta, 'context_scores': context_scores, 'baseline_gate': expected_gate}
    if delta < 1.0:
        assert saved['status'] == saved['raw_status'] == 'baseline-unsupported'
        assert saved['descriptive_status'] == 'gap-ineligible'
        audited_cases.append(audit)
        continue
    dose_status = read_dose(index, case, states, saved)
    audit['q'] = saved['dose']['q']
    if dose_status != 'matchable':
        assert saved['status'] == (dose_status if strict_gate else 'baseline-unsupported')
        assert saved['raw_status'] == (dose_status if raw_gate else 'baseline-unsupported')
        assert saved['descriptive_status'] == dose_status
        audited_cases.append(audit)
        continue
    anchors = ('null_word', 'self_word', 'self_instruction', 'self_punctuation',
               'full_self', 'full_swap', 'full_word_swap', 'zero_word')
    levels = ((1, 3), (2, 3), (1, 1))
    conditions = (*anchors, *(f'matched_{site}_{a}of{b}' for a, b in levels
                              for site in ('word', 'instruction', 'punctuation')), 'repeat')
    scores = {}
    for condition in conditions:
        for side in range(2):
            for t in range(2):
                item = get(f"{case['case_id']}-{condition}-{side}-{t}", case['source'][side], targets[t])
                scores[condition, side, t] = item['row']['score']
    cm = {condition: [scores[condition, side, 0] - scores[condition, side, 1] for side in range(2)]
          for condition in conditions}
    moves = {condition: [margins[0] - value[0], value[1] - margins[1]] for condition, value in cm.items()}
    assert saved['condition_margins'] == cm and saved['signed_movements'] == moves
    curves = {site: [[0.0, 0.0], *[moves[f'matched_{site}_{a}of{b}'] for a, b in levels]]
              for site in ('word', 'instruction', 'punctuation')}
    assert saved['response_curves'] == curves
    reasons = []
    for side in range(2):
        previous = 0.0
        for j, (a, b) in enumerate(levels, 1):
            word = curves['word'][j][side]
            control = max(abs(curves['instruction'][j][side]), abs(curves['punctuation'][j][side]))
            if word < -1e-4:
                reasons.append(f'reversed-side{side}-level{j}')
            if word < previous - 1e-4:
                reasons.append(f'nonmonotone-side{side}-level{j}')
            if word < control + .02 * (a / b) * delta:
                reasons.append(f'control_comparison-side{side}-level{j}')
            previous = word
        if previous < .05 * delta:
            reasons.append(f'weak_endpoint-side{side}')
    decision = 'negative' if reasons else 'positive'
    assert saved['reasons'] == reasons and saved['descriptive_status'] == decision
    assert saved['status'] == (decision if strict_gate else 'baseline-unsupported')
    assert saved['raw_status'] == (decision if raw_gate else 'baseline-unsupported')
    audit.update(q=saved['dose']['q'], response_curves=curves, reasons=reasons)
    audited_cases.append(audit)

assert used == list(records)
decisions = {}
for track, key in (('primary', 'status'), ('raw_gate', 'raw_status'), ('gap_descriptive', 'descriptive_status')):
    pairs = []
    for pair in range(6):
        pair_cells = [audited_cases[2 * pair + j][key] for j in (0, 1)]
        if any(status not in ('positive', 'negative') for status in pair_cells):
            pairs.append('unevaluable')
        else:
            pairs.append('positive' if pair_cells == ['positive', 'positive'] else 'negative')
    p, n = pairs.count('positive'), pairs.count('negative')
    decision = ('positive' if p >= 3 and p > n else 'negative' if n >= 3 and n > p else 'inconclusive')
    decisions[track] = {'status': decision, 'pairs': pairs}
assert completed['decisions'] == decisions
assert sha(self_path.read_bytes()) == self_hash
if not args.synthetic:
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source_checkout, text=True).strip() == args.source_commit
    verify_source_bytes()
print(json.dumps({'status': 'passed', 'auditor_sha256': self_hash, 'terminal_sha256': sha(terminal_raw),
                  'case_commitment': case_card['commitment'], 'source_commit': completed['source']['commit'],
                  'source_checkout_verified': not args.synthetic, 'forwards': len(records),
                  'max_abs_f64_vs_saved_f32_token_score': maximum_f64_difference,
                  'f64_per_token_tolerance': 1e-3, 'decisions': decisions,
                  'cases': audited_cases,
                  'scope': 'saved-score, scalar dose and decision consistency; no model execution or full witness audit'},
                 sort_keys=True, allow_nan=False))
