"""Independent no-model saved-byte replay for the closed public dose-response attempt."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

work = Path(__file__).resolve().parent
attempt = work/'t5-results/flan-t5-dose-response-cpu-v1-20261029'
manifest_path = work/'teams/flan_t5_dose_response/docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json'
observation_path = work/'flan-t5-dose-response-public-observation-ecdbf562.json'
terminal_raw = (attempt/'completed.json').read_bytes()
assert hashlib.sha256(terminal_raw).hexdigest() == '253a70bd1e722f6ec642961c1385681386c2be28caff13416001a3dd1a54dbde'
terminal = json.loads(terminal_raw)
manifest_raw = manifest_path.read_bytes()
assert hashlib.sha256(manifest_raw).hexdigest() == terminal['case_manifest_sha256']
manifest = json.loads(manifest_raw)
observation_raw = observation_path.read_bytes()
assert (attempt/'publication-observation.json').read_bytes() == observation_raw
observation = json.loads(observation_raw)
assert terminal['source'] == {'commit': observation['commit'], 'source_sha256': observation['source_sha256']}
assert terminal['status'] == 'complete' and terminal['forwards'] == 36 and terminal['decision'] == 'inconclusive'
assert not (attempt/'failure.json').exists()
assert len(list(attempt.glob('forward-[0-9][0-9][0-9].json'))) == 36
assert len(list(attempt.glob('forward-[0-9][0-9][0-9]-observed.json'))) == 36
assert len(list(attempt.glob('forward-[0-9][0-9][0-9].bin'))) == 36
assert not list(attempt.glob('dose-*'))
maximum_difference = 0.0
summaries = []
for case_index, case in enumerate(manifest['cases']):
    assert case['alignment_eligible'] and not case['eligibility_reasons']
    sources = case['source']
    positions = (case['word_position'], case['instruction_position'], case['context_position'])
    assert len(set(positions)) == 3 and positions[0] == 6 and positions[1] == 0
    assert positions[2] == (5 if case['template_id'] == 'translate' else 7)
    assert len(sources[0]['input_ids']) == len(sources[1]['input_ids']) <= 32
    assert sources[0]['attention_mask'] == sources[1]['attention_mask']
    assert sources[0]['special_tokens_mask'] == sources[1]['special_tokens_mask']
    assert [i for i, pair in enumerate(zip(sources[0]['input_ids'], sources[1]['input_ids']))
            if pair[0] != pair[1]] == [positions[0]]
    assert all(sources[0]['input_ids'][p] == sources[1]['input_ids'][p] for p in positions[1:])
    for row in (case['neutral'], *sources, *case['targets']):
        assert row['input_ids'][-1] == 1 and row['input_ids'].count(1) == 1
        assert all(mask == 1 for mask in row['attention_mask'])
    expected = [('context-0', case['neutral'], case['targets'][0]),
                ('context-1', case['neutral'], case['targets'][1]),
                ('base-0-0', sources[0], case['targets'][0]),
                ('base-0-1', sources[0], case['targets'][1]),
                ('base-1-0', sources[1], case['targets'][0]),
                ('base-1-1', sources[1], case['targets'][1])]
    scores, encoders = {}, {}
    for index, (suffix, source, target) in enumerate(expected, start=case_index*6):
        record = json.loads((attempt/f'forward-{index:03}.json').read_bytes())
        observed = json.loads((attempt/f'forward-{index:03}-observed.json').read_bytes())
        data = (attempt/f'forward-{index:03}.bin').read_bytes()
        assert record['index'] == observed['index'] == index
        assert record['name'] == f"{case['case_id']}-{suffix}" == observed['forward']
        assert observed['case_id'] == case['case_id'] and observed['return_observed'] is True
        assert record['action'] == 'clean' and record['token_index'] is None
        assert record['session_metadata'] is None and observed['session_metadata'] is None
        assert record['source_ids'] == source['input_ids'] and record['source_mask'] == source['attention_mask']
        target_ids = target['input_ids']
        assert record['target_ids'] == target_ids and record['decoder_ids'] == [0] + target_ids[:-1]
        assert record['decoder_mask'] == [1]*len(target_ids)
        assert record['file'] == observed['saved_file'] == f'forward-{index:03}.bin'
        assert hashlib.sha256(data).hexdigest() == record['sha256'] == observed['saved_sha256']
        assert record['tensor_layout'] == observed['tensor_layout']
        layout = record['tensor_layout']
        logits, encoder = layout['logits'], layout['encoder']
        assert logits['available'] and logits['offset'] == 0 and logits['shape'] == [1, len(target_ids), 32128]
        assert logits['bytes'] == len(target_ids)*32128*4
        assert encoder['available'] and encoder['offset'] == logits['bytes']
        assert encoder['shape'] == [1, len(source['input_ids']), 512]
        assert encoder['bytes'] == len(source['input_ids'])*512*4
        assert len(data) == logits['bytes'] + encoder['bytes']
        for name in ('before', 'after', 'capture'):
            assert layout[name] == {'available': False, 'bytes': 0, 'offset': len(data), 'shape': None}
        token_values = []
        for pos, target_id in enumerate(target_ids):
            values = struct.unpack_from('<32128f', data, pos*32128*4)
            assert all(math.isfinite(x) for x in values)
            maximum = max(values)
            lse = math.log(math.fsum(math.exp(x-maximum) for x in values))
            token_values.append((values[target_id]-maximum)-lse)
        assert len(record['selected_logprobs']) == len(token_values)
        for position, (reproduced, saved) in enumerate(zip(token_values, record['selected_logprobs'])):
            difference = abs(reproduced - saved)
            maximum_difference = max(maximum_difference, difference)
            assert difference < 2e-4, (index, position, reproduced, saved, difference)
        assert record['score'] == math.fsum(record['selected_logprobs'])
        scores[suffix] = record['score']
        encoders[suffix] = data[encoder['offset']:]
    assert encoders['context-0'] == encoders['context-1']
    assert encoders['base-0-0'] == encoders['base-0-1']
    assert encoders['base-1-0'] == encoders['base-1-1']
    first = scores['base-0-0'] - scores['base-0-1']
    second = scores['base-1-0'] - scores['base-1-1']
    context = scores['context-0'] - scores['context-1']
    assert first > 0 and second > 0
    assert not (first > 0 and second < 0 and first-second >= 1.0)
    cell = terminal['cases'][case_index]
    assert cell == json.loads((attempt/f'case-{case_index:02}.json').read_bytes())
    assert cell['case_id'] == case['case_id'] and cell['status'] == 'baseline-unsupported'
    assert cell['baseline_margins'] == [first, second] and cell['delta'] == first-second
    assert cell['context_scores'] == [scores['context-0'], scores['context-1']] and cell['context_margin'] == context
    summaries.append({'case_id': case['case_id'], 'status': cell['status'],
                      'baseline_margins': [first, second], 'context_margin': context,
                      'target_lengths_including_eos': [len(item['input_ids']) for item in case['targets']]})
assert len(summaries) == 6
report = {'schema': 'independent-flant5-dose-response-baseline-review-v1',
          'scope': 'saved-byte consistency only; no execution or pretraining attestation',
          'terminal_sha256': hashlib.sha256(terminal_raw).hexdigest(),
          'case_manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
          'public_source_observation_sha256': hashlib.sha256(observation_raw).hexdigest(),
          'forwards': 36, 'clean_forwards': 36, 'patch_forwards': 0,
          'maximum_float64_vs_saved_float32_token_logprob_difference': maximum_difference,
          'aggregate': 'inconclusive', 'cases': summaries}
data = (json.dumps(report, sort_keys=True, separators=(',', ':'))+'\n').encode()
output = work/'flan-t5-dose-response-independent-baseline-audit.json'
with output.open('xb') as stream:
    stream.write(data)
print(json.dumps({'report': str(output), 'sha256': hashlib.sha256(data).hexdigest(), **report}, sort_keys=True))
