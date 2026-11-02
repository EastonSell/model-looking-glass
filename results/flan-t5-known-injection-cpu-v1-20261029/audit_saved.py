"""Model-free replay of a closed FLAN-T5 known-injection evidence journal.

Inputs must remain quiescent. This verifies saved-byte, coordinate, landing,
control and arithmetic consistency. It does not attest model execution,
source/asset identity, call-site return, process supervision, publication
chronology, pretraining independence, or a natural localization claim.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

CASE_SHA = '1b21870b8d2c83a7c144ecbf2911728fcdc14a828bf2e8078fca43780df64c11'
RAW_LIMIT = 32 * 1024 * 1024
SITES = tuple(f'{stack}.block.{layer}' for stack in ('encoder', 'decoder') for layer in range(6))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bounded(path, limit):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, f'file too large: {path.name}')
    return data


def read_json(path, limit):
    raw = bounded(path, limit)
    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    return raw, value


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def typed_equal(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        b, sort_keys=True, separators=(',', ':'), allow_nan=False)


def nearest():
    one, delta = float.fromhex('0x1p0'), float.fromhex('0x1.8p-52')
    expected = float.fromhex('0x1.0000000000002p0')
    require(one + delta == expected and -one - delta == -expected, 'host rounding is not nearest-even')


def floats(raw):
    require(len(raw) % 4 == 0, 'invalid float32 bytes')
    result = []
    for (word,) in struct.iter_unpack('<I', raw):
        exp, mantissa = (word >> 23) & 255, word & 0x7fffff
        require(exp != 255, 'nonfinite float32 evidence')
        value = math.ldexp(mantissa if exp == 0 else (1 << 23) + mantissa,
                           -149 if exp == 0 else exp - 150)
        result.append(-value if word >> 31 else value)
    return result


def distance(a, b):
    require(len(a) == len(b), 'distance shape differs')
    av, bv = floats(a), floats(b)
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(av, bv)))


def near(actual, expected, message):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and actual >= 0 and (actual == 0) == (expected == 0)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), message)


def random_guesses(case_id, stack):
    raw = hashlib.sha256(('FLAN-T5-known-injection-random-v1|' + case_id + '|' + stack).encode('ascii')).digest()
    return int.from_bytes(raw[:8], 'big') % 12, (0 if stack == 'encoder' else 6) + int.from_bytes(raw[8:16], 'big') % 6


def input_record(case, decoder):
    src = case['input_ids']
    return {name: {'dtype': 'torch.int64', 'shape': [1, len(values)], 'values': [values]}
            for name, values in (('input_ids', src), ('attention_mask', [1] * len(src)),
                                 ('decoder_input_ids', decoder), ('decoder_attention_mask', [1] * len(decoder)))}


def plan_steps(stack, arm):
    target = 2 if stack == 'encoder' else 8
    result = []
    for i in range(12):
        if i == target:
            result += [(i, 'inject_before', 'capture', False, None), (i, 'inject', 'patch', True, None),
                       (i, 'inject_after', 'capture', False, None)]
            if arm == 'self':
                result += [(i, 'self_source', 'capture', True, None), (i, 'self_before', 'capture', False, None),
                           (i, 'self_patch', 'patch', True, 'self_source'), (i, 'self_after', 'capture', False, None)]
            if arm == 'restore':
                result += [(i, 'restore_before', 'capture', False, None), (i, 'restore_patch', 'patch', True, None),
                           (i, 'restore_after', 'capture', False, None)]
        if i == target + 1 and arm == 'alternate':
            result += [(i, 'alternate_before', 'capture', False, None), (i, 'alternate_patch', 'patch', True, None),
                       (i, 'alternate_after', 'capture', False, None)]
        result.append((i, f'trace_{i}', 'capture', True, None))
    return result


class Reader:
    def __init__(self, root, events, hidden, vocab):
        self.root, self.events, self.hidden, self.vocab = root, events, hidden, vocab
        self.index, self.refs, self.forward_count = 0, {}, 0

    def event(self, kind, context=None):
        require(self.index < len(self.events), 'missing journal event')
        row = self.events[self.index]
        self.index += 1
        require(type(row) is dict and row.get('event') == kind, f'wrong event at {self.index}')
        if context is not None:
            require(all(typed_equal(row.get(k), v) for k, v in context.items()), 'event context differs')
            require((row.get('stack') if 'stack' in context else None) == context.get('stack'), 'extra stack context')
        return row

    def tensor(self, ref, shape):
        require(type(ref) is dict and set(ref) == {'sha256', 'shape', 'bytes', 'path'}, 'invalid tensor reference')
        require(typed_equal(ref['shape'], shape) and type(ref['bytes']) is int, 'tensor shape differs')
        size = math.prod(shape) * 4
        name = ref['sha256']
        require(type(name) is str and len(name) == 64 and all(c in '0123456789abcdef' for c in name), 'bad digest')
        require(ref['path'] == f'tensors/{name}.f32le' and ref['bytes'] == size and 0 < size <= 1048576,
                'invalid tensor path or length')
        raw = bounded(self.root / ref['path'], size)
        require(len(raw) == size and digest(raw) == name, 'tensor bytes differ')
        floats(raw)
        if name in self.refs:
            require(self.refs[name] == len(raw), 'inconsistent duplicate tensor')
        self.refs[name] = len(raw)
        return raw

    def trace_meta(self, events, case, label, source_len):
        require(type(events) is list and len(events) == 12, 'trace inventory differs')
        for i, row in enumerate(events):
            stack, layer = ('encoder', i) if i < 6 else ('decoder', i - 6)
            seq = source_len if i < 6 else 2
            expected = {'module_path': SITES[i], 'layer_index': layer, 'module_kind': f't5_{stack}_block_output',
                        'invocation': 0, 'sequence': i, 'output_path': [0], 'batch_positions': [0],
                        'token_positions': [1], 'shape': [1, seq, self.hidden], 'saved_shape': [self.hidden],
                        'dtype': 'torch.float32', 'saved_dtype': 'torch.float32', 'device': 'cpu', 'saved_device': 'cpu',
                        'forward_index': 0, 'batch_axis': 0, 'token_axis': 1, 'capture_mode': 'direct'}
            require(type(row) is dict and all(typed_equal(row.get(k), v) for k, v in expected.items()),
                    'clean trace coordinate differs')
            require(row.get('identity', {}).get('run_id') == case['case_id'] + '/' + label
                    and row.get('generation', {}).get('stage') == 'forward'
                    and row['generation'].get('use_cache') is False, 'trace identity/cache differs')

    def plan_meta(self, metadata, stack, arm, source_len):
        expected = plan_steps(stack, arm)
        require(type(metadata) is dict and type(metadata.get('steps')) is list
                and len(metadata['steps']) == len(expected), 'plan step count differs')
        preceding = {path: [] for path in SITES}
        for order, ((i, name, op, selected, source), row) in enumerate(zip(expected, metadata['steps'])):
            shape = [1, source_len if i < 6 else 2, self.hidden]
            wanted = {'name': name, 'operation': op, 'order': order, 'module_path': SITES[i],
                      'module_aliases': [SITES[i]], 'invocation': 0, 'forward_index': 0, 'output_selector': [0],
                      'batch_index': 0 if selected else None, 'token_index': 1 if selected else None,
                      'source_capture': source, 'output_shape': shape,
                      'selected_shape': [self.hidden] if selected else shape,
                      'dtype': 'torch.float32', 'device': 'cpu', 'call_site_known': False,
                      'preceding_same_site_actions': preceding[SITES[i]]}
            require(type(row) is dict and all(typed_equal(row.get(k), v) for k, v in wanted.items()),
                    f'plan metadata differs: {name}')
            preceding[SITES[i]] = preceding[SITES[i]] + [name]

    def forward(self, case, label, decoder, stack=None):
        context = {'case_id': case['case_id'], 'arm': label}
        if stack is not None:
            context['stack'] = stack
        start = self.event('forward_start', context)
        require(typed_equal(start.get('inputs'), input_record(case, decoder)), 'forward IDs/masks differ')
        returned = self.event('wrapper_return_observed', context)
        require(returned.get('return_scope') == ('trace_wrapper' if stack is None else 'plan_wrapper_and_context'),
                'wrapper return scope differs')
        observation = self.event('forward_observation', context)
        require(observation.get('unavailable') == [] and type(observation.get('refs')) is dict,
                'missing forward evidence')
        source_len = len(case['input_ids'])
        expected = {'logits': [1, 2, self.vocab], **{f'trace_{i}': [self.hidden] for i in range(12)}}
        if stack is None:
            self.trace_meta(observation.get('trace_events'), case, label, source_len)
        else:
            self.plan_meta(observation.get('plan_metadata'), stack, label, source_len)
            shape = [1, source_len if stack == 'encoder' else 2, self.hidden]
            expected.update({'inject_before': shape, 'inject_after': shape})
            if label == 'self':
                expected.update({'self_before': shape, 'self_after': shape, 'self_source': [self.hidden]})
            if label == 'restore':
                expected.update({'restore_before': shape, 'restore_after': shape})
            if label == 'alternate':
                expected.update({'alternate_before': shape, 'alternate_after': shape})
        require(set(observation['refs']) == set(expected), 'forward tensor inventory differs')
        if stack is not None:
            metadata = observation['plan_metadata']
            capture_bytes = sum(math.prod(shape) * 4 for name, shape in expected.items() if name != 'logits')
            source_bytes = self.hidden * 4 * (2 if label in ('restore', 'alternate') else 1)
            require(type(metadata.get('capture_bytes')) is int and metadata['capture_bytes'] == capture_bytes
                    and type(metadata.get('source_bytes')) is int and metadata['source_bytes'] == source_bytes,
                    'plan byte counts differ')
        payloads = {name: self.tensor(observation['refs'][name], shape) for name, shape in expected.items()}
        done = self.event('forward_complete', context)
        self.forward_count += 1
        return payloads, done


def token(raw, hidden):
    width = hidden * 4
    require(len(raw) >= width * 2, 'missing token 1')
    return raw[width:2 * width]


def landing(before, after, replacement, hidden):
    width = hidden * 4
    require(len(replacement) == width and after == before[:width] + replacement + before[2 * width:],
            'requested landing or untouched positions differ')


def run_trial(reader, case, decoder, stack, clean):
    target = 2 if stack == 'encoder' else 8
    clean_target = clean[f'trace_{target}']
    first = floats(clean_target)[0]
    injection = struct.pack('<f', first + 4.0) + clean_target[4:]
    require(distance(clean_target, injection) > 0, 'zero injection')
    arms = {}
    for arm in ('inject', 'self', 'restore', 'alternate'):
        payload, done = reader.forward(case, arm, decoder, stack)
        require(token(payload['inject_before'], reader.hidden) == clean_target, 'injection receiver differs')
        landing(payload['inject_before'], payload['inject_after'], injection, reader.hidden)
        effective_target = clean_target if arm == 'restore' else injection
        require(payload[f'trace_{target}'] == effective_target, 'effective target trace differs')
        if arm == 'self':
            require(payload['self_source'] == injection and payload['self_before'] == payload['inject_after'],
                    'self source/before differs')
            landing(payload['self_before'], payload['self_after'], injection, reader.hidden)
        if arm == 'restore':
            require(payload['restore_before'] == payload['inject_after'], 'restore before differs')
            landing(payload['restore_before'], payload['restore_after'], clean_target, reader.hidden)
            require(payload['restore_after'] == payload['inject_before'], 'restoration is not immediate undo')
        if arm == 'alternate':
            require(token(payload['alternate_before'], reader.hidden) == arms['inject'][f'trace_{target + 1}'],
                    'alternate receiver differs from injection trace')
            landing(payload['alternate_before'], payload['alternate_after'], clean[f'trace_{target + 1}'], reader.hidden)
            require(payload[f'trace_{target + 1}'] == clean[f'trace_{target + 1}'], 'alternate trace differs')
            alt_dose = distance(token(payload['alternate_before'], reader.hidden), clean[f'trace_{target + 1}'])
            near(done.get('alternate_dose'), alt_dose, 'alternate dose differs')
        else:
            require(done.get('alternate_dose') is None, 'unexpected alternate dose')
        checks = {'inject_before_matches_clean', 'injection_landing_and_untouched'}
        if arm == 'self':
            checks |= {'self_landing_and_untouched', 'self_source_is_injected'}
        elif arm == 'restore':
            checks |= {'restore_landing_and_untouched', 'restore_full_equals_preinjection'}
        elif arm == 'alternate':
            checks |= {'alternate_landing_and_untouched'}
        require(type(done.get('checks')) is dict and set(done['checks']) == checks
                and all(v is True for v in done['checks'].values()), 'recorded landing checks failed')
        arms[arm] = payload
    require(all(arms['inject'][k] == arms['self'][k] for k in ('logits', *(f'trace_{i}' for i in range(12)))),
            'self control differs')
    require(all(clean[k] == arms['restore'][k] for k in ('logits', *(f'trace_{i}' for i in range(12)))),
            'restore control differs')
    distances = [distance(clean[f'trace_{i}'], arms['inject'][f'trace_{i}']) for i in range(12)]
    changed = [i for i in range(12) if clean[f'trace_{i}'] != arms['inject'][f'trace_{i}']]
    require(changed, 'no trace change')
    first_index = changed[0]
    magnitude_index = max(range(12), key=lambda i: (distances[i], -i))
    across, within = random_guesses(case['case_id'], stack)
    trial = {'case_id': case['case_id'], 'split': case['split'], 'stack': stack, 'target_index': target,
             'first_change_index': first_index, 'magnitude_index': magnitude_index, 'random_across_index': across,
             'random_within_index': within, 'known_site_index': target, 'trace_l2': distances,
             'injection_l2': distance(clean_target, injection), 'alternate_l2': alt_dose,
             'alternate_active': alt_dose > 0, 'injected_logit_l2': distance(clean['logits'], arms['inject']['logits']),
             'alternate_logit_l2': distance(clean['logits'], arms['alternate']['logits']),
             'same_site_restore_matches_clean': True, 'self_matches_injected': True}
    saved = reader.event('trial_complete', {'case_id': case['case_id'], 'stack': stack})
    compare_trial(saved, trial)
    return trial


def compare_trial(saved, trial):
    require(type(saved) is dict and set(saved) in (set(trial), set(trial) | {'event'}), 'trial fields differ')
    for key, value in trial.items():
        if key == 'trace_l2':
            require(type(saved[key]) is list and len(saved[key]) == 12, 'trace distances differ')
            for actual, expected in zip(saved[key], value):
                near(actual, expected, 'trace distance differs')
        elif key in ('injection_l2', 'alternate_l2', 'injected_logit_l2', 'alternate_logit_l2'):
            near(saved[key], value, key + ' differs')
        else:
            require(typed_equal(saved[key], value), key + ' differs')
    reported_maximum = max(range(12), key=lambda i: (saved['trace_l2'][i], -i))
    require(reported_maximum == trial['magnitude_index'], 'reported trace distances change magnitude ranking')


def audit(root, manifest_path, synthetic):
    nearest()
    manifest_raw, manifest = read_json(manifest_path, 1000000)
    if synthetic:
        require(type(manifest) is list and 0 < len(manifest) <= 8, 'fake case list differs')
        cases, decoder = manifest, [0, 3]
        hidden, vocab = 16, 96
    else:
        require(digest(manifest_raw) == CASE_SHA and type(manifest) is dict, 'fixed public case manifest differs')
        cases, decoder = manifest['cases'], manifest['decoder_prefix']['decoder_input_ids']
        require(len(cases) == 8 and decoder == [0, 1368], 'fixed case count or prefix differs')
        hidden, vocab = 512, 32128
    require(len({c['case_id'] for c in cases}) == len(cases), 'duplicate cases')
    for case in cases:
        require(type(case['case_id']) is str and case['split'] in ('discovery', 'heldout')
                and type(case['input_ids']) is list and len(case['input_ids']) >= 3
                and all(type(i) is int and 0 <= i < vocab for i in case['input_ids'])
                and case['input_ids'][-1] == 1 and case['input_ids'][1] != 1, 'case coordinates differ')
    journal = bounded(root / 'attempt.jsonl', 16 * 1024 * 1024)
    events = [json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
              for line in journal.splitlines()]
    reader = Reader(root, events, hidden, vocab)
    begin = reader.event('begin')
    require(begin.get('case_sha256') == CASE_SHA and begin.get('case_count') == len(cases)
            and typed_equal(begin.get('decoder_input_ids'), decoder) and typed_equal(begin.get('sites'), list(SITES))
            and type(begin.get('raw_limit')) is int and 0 < begin['raw_limit'] <= RAW_LIMIT, 'begin differs')
    trials = []
    for case in cases:
        clean, _ = reader.forward(case, 'clean', decoder)
        repeat, _ = reader.forward(case, 'clean_repeat', decoder)
        require(all(clean[k] == repeat[k] for k in clean), 'clean repeat differs')
        for stack in ('encoder', 'decoder'):
            trials.append(run_trial(reader, case, decoder, stack, clean))
    groups = {}
    for split in ('discovery', 'heldout'):
        for stack in ('encoder', 'decoder'):
            chosen = [t for t in trials if t['split'] == split and t['stack'] == stack]
            if chosen:
                groups[split + '/' + stack] = {'trials': len(chosen), **{
                    key + '_hits': sum(t[key + '_index'] == t['target_index'] for t in chosen)
                    for key in ('first_change', 'magnitude', 'random_across', 'random_within', 'known_site')}}
    reader.event('postflight_start')
    post = reader.event('postflight_passed')
    require(post.get('production_callback_supplied') is True or synthetic, 'production postflight callback absent')
    complete = reader.event('complete')
    require(reader.index == len(events), 'trailing journal events')
    result_raw, result = read_json(root / 'result.json', 1000000)
    require(complete.get('result_sha256') == digest(result_raw), 'result digest differs')
    require(result.get('schema') == 'flan-t5-known-injection-result-v1' and result.get('case_sha256') == CASE_SHA
            and typed_equal(result.get('groups'), groups) and type(result.get('trials')) is list
            and len(result['trials']) == len(trials), 'result summary differs')
    require(typed_equal(result.get('model_metadata'), begin.get('model_metadata')),
            'begin and result model metadata differ')
    for saved, trial in zip(result['trials'], trials):
        compare_trial(saved, trial)
    count, total = len(cases) * 10, sum(reader.refs.values())
    require(reader.forward_count == count and result.get('forwards') == count and complete.get('forwards') == count,
            'forward count differs')
    require(total <= begin['raw_limit'] and result.get('raw_tensor_bytes') == total
            and complete.get('raw_tensor_bytes') == total, 'evidence budget differs')
    require({p.name for p in (root / 'tensors').iterdir()} == {k + '.f32le' for k in reader.refs},
            'tensor inventory differs')
    nearest()
    return {'manifest_sha256': digest(manifest_raw), 'journal_sha256': digest(journal),
            'result_sha256': digest(result_raw), 'forwards': count, 'raw_tensor_bytes': total,
            'unique_tensors': len(reader.refs), 'groups': groups, 'trials': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.manifest, args.synthetic)
    report = {'schema': 'flan-t5-known-injection-independent-audit-v1', 'synthetic': args.synthetic,
              'scope': 'Saved-byte consistency only; quiescent inputs required. No execution, source, asset, '
                       'call-site return, process, publication chronology or pretraining attestation.', 'result': result}
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short report write')
    print(json.dumps({'report_sha256': digest(raw), 'forwards': result['forwards'], 'groups': result['groups']}))


if __name__ == '__main__':
    main()
