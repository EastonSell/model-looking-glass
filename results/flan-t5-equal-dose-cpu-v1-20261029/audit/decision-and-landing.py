"""Independent metadata and bitwise-control audit of the public FLAN run.

No model is loaded and no hidden examples are read. Numeric reconstruction of
log-softmax from the raw logits is a separate independent audit.
"""
import hashlib
import json
import math
from pathlib import Path

root = Path('work/t5-results/flan-t5-equal-dose-cpu-v1-20261029')
manifest_path = Path('work/teams/flan_t5_equal_dose/docs/FLAN_T5_EQUAL_DOSE_CASES_V1.json')
manifest_bytes = manifest_path.read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == 'a0e320c9d24a8940fbca05cbc4ed6e3f2661b5b44b2e62e1140832c0e99e7cd6'
manifest = json.loads(manifest_bytes)
result_bytes = (root / 'completed.json').read_bytes()
assert hashlib.sha256(result_bytes).hexdigest() == '8d0d6e166fd8ced06b5066f7ccb6e27e0a12c97124bffb47e57ca304600a51d6'
result = json.loads(result_bytes)
started = json.loads((root / 'started.json').read_bytes())
observation = (root / 'publication-observation.json').read_bytes()
assert hashlib.sha256(observation).hexdigest() == started['publication_observation_sha256']
assert started['source']['commit'] == '6f74b7506b577154b0ed0b42eb3b7db8a5e3ed70'
assert started['source'] == result['source']
assert result['status'] == 'complete' and result['hooks_restored'] is True
assert result['case_manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()

cursor = 0
all_statuses = []
case_reports = []

def read_forward(name, source, target, action, pos):
    global cursor
    row = json.loads((root / f'forward-{cursor:03}.json').read_bytes())
    assert row['index'] == cursor and row['name'] == name
    assert row['action'] == action and row['token_index'] == pos
    assert row['source_ids'] == source['input_ids'] and row['source_mask'] == source['attention_mask']
    assert row['target_ids'] == target['input_ids']
    assert row['decoder_ids'] == [0] + target['input_ids'][:-1]
    assert row['decoder_mask'] == [1] * len(target['input_ids'])
    assert all(math.isfinite(x) for x in row['selected_logprobs'])
    assert len(row['selected_logprobs']) == len(target['input_ids'])
    assert math.fsum(row['selected_logprobs']) == row['score']
    assert row['file'] == f'forward-{cursor:03}.bin'
    raw = (root / row['file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['sha256']
    layout = row['tensor_layout']
    segments = {}
    offset = 0
    for key in ('logits', 'encoder', 'before', 'after', 'capture'):
        entry = layout[key]
        assert entry['offset'] == offset and entry['bytes'] >= 0
        segments[key] = raw[offset:offset + entry['bytes']]
        assert len(segments[key]) == entry['bytes']
        offset += entry['bytes']
    assert offset == len(raw)
    assert layout['logits']['shape'] == [1, len(target['input_ids']), 32128]
    assert layout['encoder']['shape'] == [1, len(source['input_ids']), 512]
    assert len(segments['logits']) == 4 * len(target['input_ids']) * 32128
    assert len(segments['encoder']) == 4 * len(source['input_ids']) * 512
    cursor += 1
    return row, segments

conditions = ('repeat', 'null_word', 'self_word', 'self_instruction', 'self_punctuation',
              'full_self', 'full_swap', 'full_word_swap', 'zero_word',
              'matched_word', 'matched_instruction', 'matched_punctuation')

for ci, case in enumerate(manifest['cases']):
    summary = json.loads((root / f'case-{ci:02}.json').read_bytes())
    assert summary == result['cases'][ci]
    assert summary['case_id'] == case['case_id'] and summary['pair_id'] == case['pair_id']
    assert summary['template_id'] == case['template_id']
    neutral = [read_forward(f"{case['case_id']}-context-{t}", case['neutral'], case['targets'][t], 'clean', None)[0]['score'] for t in range(2)]
    baselines = {}
    for side in range(2):
        for t in range(2):
            baselines[side, t] = read_forward(f"{case['case_id']}-base-{side}-{t}", case['source'][side], case['targets'][t], 'clean', None)
    base = [baselines[s, 0][0]['score'] - baselines[s, 1][0]['score'] for s in range(2)]
    for s in range(2):
        assert baselines[s, 0][1]['encoder'] == baselines[s, 1][1]['encoder']
    delta = base[0] - base[1]
    assert summary['baseline_margins'] == base and summary['delta'] == delta
    assert summary['context_scores'] == neutral and summary['context_margin'] == neutral[0] - neutral[1]
    baseline_eligible = base[0] > 0 and base[1] < 0 and delta >= 1
    if not baseline_eligible:
        assert summary['status'] == 'baseline-unsupported'
        all_statuses.append(summary['status'])
        case_reports.append({'case_id': case['case_id'], 'status': summary['status'], 'baseline_margins': base, 'delta': delta})
        continue

    dose_json = json.loads((root / f'dose-{ci:02}.json').read_bytes())
    assert dose_json['case_id'] == case['case_id'] and dose_json['dose'] == summary['dose']
    dose = summary['dose']
    dose_raw = (root / dose_json['file']).read_bytes()
    assert hashlib.sha256(dose_raw).hexdigest() == dose_json['sha256']
    delivered = {}
    for item in dose_json['layout']:
        segment = dose_raw[item['offset']:item['offset'] + item['bytes']]
        assert len(segment) == 2048 and item['shape'] == [512]
        if item['role'] == 'delivered':
            delivered[item['site'], item['side']] = segment
    assert dose['status'] == 'matchable'
    q = dose['q']
    tolerance = min(max(1e-5, 1e-5*q), .01*q)
    assert q == min(dose['distances'].values()) and dose['tolerance'] == tolerance
    for side in range(2):
        w = dose['achieved'][f'word-{side}']
        for site in ('word', 'instruction', 'punctuation'):
            actual = dose['achieved'][f'{site}-{side}']
            assert actual > 0 and abs(actual-q) <= tolerance
            assert abs(actual-w) <= .01*w

    pos_by_site = {'word':case['word_position'], 'instruction':case['instruction_position'], 'punctuation':case['context_position']}
    scores = {}
    for condition in conditions:
        for side in range(2):
            if condition == 'repeat':
                action, pos, site = 'capture', None, None
            elif condition in ('full_self', 'full_swap'):
                action, pos, site = 'patch', None, None
            else:
                site = ('instruction' if condition.endswith('instruction') else 'punctuation' if condition.endswith('punctuation') else 'word')
                pos = pos_by_site[site]
                action = 'null' if condition == 'null_word' else 'zero' if condition == 'zero_word' else 'patch'
            for t in range(2):
                row, seg = read_forward(f"{case['case_id']}-{condition}-{side}-{t}", case['source'][side], case['targets'][t], action, pos)
                scores[condition, side, t] = row['score']
                own = baselines[side, t][1]
                donor = baselines[1-side, t][1]
                if condition in ('repeat','null_word','self_word','self_instruction','self_punctuation','full_self'):
                    assert seg['logits'] == own['logits']
                if condition == 'full_swap':
                    assert seg['logits'] == donor['logits']
                if condition == 'repeat':
                    assert seg['encoder'] == own['encoder'] == seg['capture']
                elif pos is None:
                    expected = own['encoder'] if condition == 'full_self' else donor['encoder']
                    assert seg['before'] == own['encoder'] and seg['after'] == seg['encoder'] == expected
                else:
                    start, end = pos * 2048, (pos+1) * 2048
                    assert seg['encoder'][:start] == own['encoder'][:start] and seg['encoder'][end:] == own['encoder'][end:]
                    assert seg['before'] == own['encoder'][start:end] and seg['after'] == seg['encoder'][start:end]
                    if condition.startswith('matched_'):
                        assert seg['after'] == delivered[site, side]
                    elif condition == 'full_word_swap':
                        assert seg['after'] == donor['encoder'][start:end]
                    elif condition == 'zero_word':
                        assert seg['after'] == bytes(2048)
                    else:
                        assert seg['after'] == own['encoder'][start:end]

    cmargins = {c:[scores[c,s,0] - scores[c,s,1] for s in range(2)] for c in conditions}
    movements = {c:[base[0]-v[0], v[1]-base[1]] for c,v in cmargins.items()}
    assert summary['condition_margins'] == cmargins and summary['signed_movements'] == movements
    assert summary['absolute_movements'] == {k:[abs(v) for v in values] for k,values in movements.items()}
    word = movements['matched_word']
    controls = [max(abs(movements['matched_instruction'][s]),abs(movements['matched_punctuation'][s])) for s in range(2)]
    ok = all(word[s] >= .20*delta and word[s] >= controls[s] + .05*delta for s in range(2))
    status = 'positive' if ok else 'negative'
    assert summary['status'] == status
    all_statuses.append(status)
    case_reports.append({'case_id':case['case_id'], 'status':status, 'baseline_margins':base, 'delta':delta,
                         'q':q, 'matched_word_movement_over_delta':[x/delta for x in word],
                         'max_abs_control_movement_over_delta':[x/delta for x in controls],
                         'word_over_controls_by_5pct':[word[s] >= controls[s]+.05*delta for s in range(2)]})

assert cursor == result['forwards'] == 228
pair_results = []
for i in range(3):
    two = all_statuses[2*i:2*i+2]
    pair_results.append('positive' if two == ['positive','positive'] else
                        'negative' if all(x in ('positive','negative') for x in two) else 'unevaluable')
decision = 'positive' if pair_results.count('positive') >= 2 else 'negative' if pair_results.count('negative') >= 2 else 'inconclusive'
assert result['decision'] == decision
report = {'schema':'independent-flan-equal-dose-decision-check-v1','result_sha256':hashlib.sha256(result_bytes).hexdigest(),
          'forwards':cursor,'case_reports':case_reports,'pair_statuses':pair_results,'aggregate':decision,
          'scope':'Saved score/metadata arithmetic, file digests and bitwise controls; raw log-softmax reconstruction and execution provenance are separate.'}
out = Path('work/flan-equal-dose-independent-decision-audit-v1.json')
raw = (json.dumps(report,sort_keys=True,separators=(',',':'))+'\n').encode()
with out.open('xb') as stream:
    stream.write(raw)
print(json.dumps({'report':str(out.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),**report},sort_keys=True))
