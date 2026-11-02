"""Independent, model-free replay of saved year-transfer evidence.

Production mode requires a separately supplied receipt hash and source revision.
Synthetic mode deliberately does not assert execution provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

CASE_SHA = 'd6177125f8ac835676eebe59c5f65e379cb20d1d782eeeb17edf2e8700ac23d5'
PROTOCOL_SHA = '6f31697b09b2b68bf701494b4ceedde185b9f8501aea61caf50b647edac1fee7'
VOCAB, WIDTH = 50257, 768
ROW_BYTES, LOGITS_BYTES = WIDTH * 4, VOCAB * 4
CONDITIONS = ('forward', 'reverse', 'neutral')
ACTIONS = ('self', 'year_final', 'year_at_zero', 'rotated_final')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite_f32(bits):
    sign = -1.0 if bits >> 31 else 1.0
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    need(exponent != 255, 'nonfinite float32 value')
    if exponent == 0:
        return sign * math.ldexp(float(fraction), -149)
    return sign * math.ldexp(float((1 << 23) | fraction), exponent - 150)


def round_even_div(numerator, denominator):
    q, r = divmod(numerator, denominator)
    return q + int(2*r > denominator or (2*r == denominator and q % 2 == 1))


def round_f32(value):
    need(math.isfinite(value), 'nonfinite planned value')
    sign = 0x80000000 if math.copysign(1.0, value) < 0 else 0
    numerator, denominator = abs(value).as_integer_ratio()
    if not numerator:
        return sign
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        exponent -= int(numerator < denominator << exponent)
    else:
        exponent -= int(numerator << -exponent < denominator)
    if exponent < -126:
        fraction = round_even_div(numerator << 149, denominator)
        need(fraction <= 1 << 23, 'invalid subnormal rounding')
        return sign | fraction
    shift = 23 - exponent
    n = numerator << shift if shift >= 0 else numerator
    d = denominator if shift >= 0 else denominator << -shift
    significand = round_even_div(n, d)
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    need(exponent <= 127 and (1 << 23) <= significand < (1 << 24), 'float32 overflow')
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))


def values(raw):
    need(len(raw) % 4 == 0, 'incomplete float32 data')
    return [finite_f32(bits) for (bits,) in struct.iter_unpack('<I', raw)]


def norm(vector):
    return math.sqrt(math.fsum(x*x for x in vector))


def score(raw, row):
    need(len(raw) == LOGITS_BYTES, 'wrong logits length')
    for (bits,) in struct.iter_unpack('<I', raw):
        need((bits >> 23) & 255 != 255, 'nonfinite logits')
    bands = []
    for ids in (row['middle_ids'], row['above_ids']):
        bands.append([finite_f32(struct.unpack_from('<I', raw, 4*i)[0]) for i in ids])
    middle, above = bands
    u, v = max(middle), max(above)
    a = math.fsum(math.exp(x-u) for x in middle)
    b = math.fsum(math.exp(x-v) for x in above)
    return ((u-v)+math.log(a))-math.log(b)


def validate_provenance(args, manifest, result, run):
    if args.synthetic:
        return 'synthetic fixture only; no execution provenance asserted'
    need(args.receipt_sha256 and args.source_revision and args.source_root and args.reservation_sha256
         and args.current_exposure_sha256,
         'production requires independent receipt, source and exposure pins')
    root = run.parent
    receipt_raw = (root/'supervisor.json').read_bytes()
    need(digest(receipt_raw) == args.receipt_sha256, 'receipt hash mismatch')
    receipt = json.loads(receipt_raw)
    need(receipt['schema'] == 'distilgpt2-year-transfer-process-observation-v1', 'receipt schema mismatch')
    need(type(receipt['process_exit_code']) is int and receipt['process_exit_code'] == 0
         and receipt['interrupted'] is False and receipt['failure_file_present'] is False, 'process not clean')
    need(not (run/'failure.json').exists(), 'failure record exists')
    need(receipt['source_revision'] == args.source_revision == manifest['source']['revision'], 'source revision mismatch')
    need(receipt['source_tree'] == manifest['source']['tree'], 'source tree mismatch')
    need(digest((root/'pre-run.json').read_bytes()) == receipt['pre_run_sha256'], 'pre-run hash mismatch')
    pre = json.loads((root/'pre-run.json').read_bytes())
    need(pre['schema'] == 'distilgpt2-year-transfer-launch-v1', 'pre-run schema mismatch')
    need(pre['source_revision'] == args.source_revision and pre['source_tree'] == receipt['source_tree'], 'pre-run source mismatch')
    need(pre['case_sha256'] == CASE_SHA and pre['protocol_sha256'] == PROTOCOL_SHA, 'pre-run protocol mismatch')
    need(pre['script_sha256'] == manifest['source']['sha256']['scripts/year_transfer_v1.py'], 'pre-run script mismatch')
    intended = pre['intended_execution']
    for key, value in {'device':'cpu','dtype':'float32','attention':'eager','use_cache':False,
                       'cuda_visible_devices':'-1','hf_hub_offline':'1','transformers_offline':'1',
                       'threads':1,'cpu_rng_seed':20261029,'mkl_cbwr':None,'aten_cpu_capability':None,
                       'automatic_retry':False,'gate_only_forwards':96,'gate_pass_forwards':336,
                       'model_id':'distilbert/distilgpt2',
                       'model_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
                       'tokenizer_sha256':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
                       'weights_sha256':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'}.items():
        need(intended[key] == value and type(intended[key]) is type(value), 'pre-run execution mismatch: '+key)
    need(pre['reservation_version'] == 'distilgpt2-development-exposures-v16'
         and pre['reservation_sha256'] == args.reservation_sha256, 'pre-run reservation mismatch')
    need(pre['current_exposure_version'] == 'distilgpt2-development-exposures-v17'
         and pre['current_exposure_sha256'] == args.current_exposure_sha256
         and args.current_exposure_sha256 == 'e5734470ccd3b4e2655c91608caf8ffa158fa8650ff0bdd7f5aeaa1028046f3a',
         'pre-run current exposure mismatch')
    need(manifest['exposure_inventories'] == {
        'baseline_version':'distilgpt2-development-exposures-v15',
        'baseline_sha256':'6c6ed34755dd9daebdde9a2de07f65c0c5d20301656416d873f321fa2c146d34',
        'reservation_version':'distilgpt2-development-exposures-v16',
        'reservation_sha256':args.reservation_sha256}, 'exposure binding mismatch')
    extra = pre['additional_exposure_report']
    other = [tuple(row) for row in extra['token_rows']]
    need(len(other) == 8 and all(len(row) == 4 and all(type(x) is int for x in row) for row in other),
         'retrospective exposure rows malformed')
    inputs = [tuple(row['input_ids']) for row in manifest['cases']]
    continued = [tuple(row['input_ids']+[token]) for row in manifest['cases']
                 for token in row['middle_ids']+row['above_ids']]
    related = lambda a,b: a[:len(b)] == b or b[:len(a)] == a
    conflicts = sum(related(a,b) for a in inputs+continued for b in other)
    need(extra['input_bidirectional_prefix_comparisons'] == len(inputs)*len(other) == 384
         and extra['candidate_continuation_bidirectional_prefix_comparisons'] == len(continued)*len(other) == 2304
         and extra['conflicts'] == conflicts == 0, 'retrospective exposure comparison mismatch')
    need(set(receipt['run_sha256']) == {p.name for p in run.iterdir() if p.is_file()}, 'receipt file inventory mismatch')
    for name, expected in receipt['run_sha256'].items():
        need(Path(name).name == name and digest((run/name).read_bytes()) == expected, 'receipt run hash mismatch')
    need({'manifest.json', 'result.json', 'attempt.jsonl'} <= set(receipt['run_sha256']), 'receipt misses required files')
    for name, key in (('stdout.txt', 'stdout_sha256'), ('stderr.txt', 'stderr_sha256')):
        need(digest((root/name).read_bytes()) == receipt[key], 'process output hash mismatch')
    need(receipt['stderr_bytes'] == len((root/'stderr.txt').read_bytes()) == 0, 'nonempty stderr')
    source_root = Path(args.source_root)
    def git(*cmd):
        return subprocess.check_output(['git', *cmd], cwd=source_root, timeout=30).decode().strip()
    need(git('rev-parse', 'HEAD') == args.source_revision, 'checkout revision mismatch')
    need(git('show', '-s', '--format=%T', 'HEAD') == manifest['source']['tree'], 'checkout tree mismatch')
    need(not git('status', '--porcelain', '--untracked-files=normal'), 'checkout is not clean')
    paths = manifest['source']['sha256']
    expected = {'scripts/year_transfer_v1.py', 'looking_glass/year_transfer.py',
        'looking_glass/year_transfer_intervention.py', 'tests/test_year_transfer_cases.py',
        'tests/test_year_transfer_intervention.py', 'tests/test_year_transfer_runner.py',
        'experiments/distilgpt2_year_transfer_v1.json', 'docs/DISTILGPT2_YEAR_TRANSFER_V1.md',
        'docs/YEAR_TRANSFER_RUNNER_V1.md', 'looking_glass/greater_than_screen.py',
        'looking_glass/copy_head_factorial.py', 'looking_glass/copy_head_intervention.py',
        'scripts/copying_head_eval_v1.py', 'scripts/copying_dev_v1.py',
        'scripts/distilgpt2_intervention_dev_bootstrap.py', 'looking_glass/pretrained.py',
        'looking_glass/__init__.py', 'looking_glass/pretrained_exposures.py'}
    expected.update(f'looking_glass/pretrained_exposures_v{i}.py' for i in range(2,17))
    need(set(paths) == expected, 'source inventory mismatch')
    for path in expected:
        need(digest((source_root/path).read_bytes()) == paths[path], 'source file mismatch: '+path)
    runtime = result['runtime']
    for key, value in {'device':'cpu', 'dtype':'float32', 'model_eval':True, 'use_cache':False,
                       'attention_implementation':'eager', 'cuda_visibility':'-1',
                       'cuda_initialized':False, 'cpu_autocast':False,
                       'guard_blocked_calls':[], 'intraop_threads':1,
                       'cpu_rng_seed':20261029, 'deterministic_algorithms':False,
                       'float32_matmul_precision':'highest'}.items():
        need(runtime[key] == value and type(runtime[key]) is type(value), 'runtime mismatch: '+key)
    need(runtime['numeric_probe'] == {'half_min_normal_bits':4194304,'input_preserved':True,
                                      'numpy_cast_preserved':True}, 'numeric probe mismatch')
    env = {name:{'present':False,'value':None} for name in ('MKL_CBWR','ATEN_CPU_CAPABILITY')}
    need(manifest['cpu_backend_environment_at_torch_import'] == env
         and runtime['cpu_backend_environment'] == env, 'CPU backend environment mismatch')
    need(manifest['initial_numeric_probe'] == runtime['numeric_probe'], 'initial numeric probe mismatch')
    deps = manifest['dependencies']
    for key, value in {'python':'3.11.15','torch':'2.8.0+cu128','numpy':'2.4.6',
                       'transformers':'4.57.3','byteorder':'little',
                       'gpt2_source_sha256':'b591253d7acadd39635bbf6f9f6fd0a97d6be5d7cf901074e2d959ff3e478e80',
                       'conv1d_source_sha256':'e17301af0b11aa8982283208fb7b4af513853d643711b193283d1a429d6b2b86'}.items():
        need(deps[key] == value, 'dependency mismatch: '+key)
    checkpoint = result['checkpoint']
    for key, value in {'weights_sha256':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
                       'model_id':'distilbert/distilgpt2', 'tokenizer_id':'distilbert/distilgpt2',
                       'model_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
                       'tokenizer_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
                       'model_class':'GPT2LMHeadModel','tokenizer_class':'GPT2TokenizerFast',
                       'parameter_count':81912576,
                       'device':'cpu','dtype':'float32','attention_implementation':'eager',
                       'use_cache':False}.items():
        need(checkpoint[key] == value and type(checkpoint[key]) is type(value), 'checkpoint mismatch: '+key)
    need(checkpoint['asset_sha256'] == {
        'config.json':'4ec5947c1d59fee6212cdf3b0ec1a53eac02092554c5ff0a733488cbd2c64f3a',
        'generation_config.json':'b90eadacf585a743a30ea51e8b5c88b8d282a2a34dc0c7e556d0987cdbd68805',
        'merges.txt':'1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
        'model.safetensors':'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b',
        'tokenizer.json':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'tokenizer_config.json':'5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
        'vocab.json':'196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'},
        'checkpoint asset inventory mismatch')
    for key in ('model_state_sha256_before','model_state_sha256_after',
                'model_buffers_sha256_before','model_buffers_sha256_after'):
        value = result[key]
        need(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
             'invalid model hash: '+key)
    need(result['model_state_sha256_before'] == result['model_state_sha256_after'], 'model state changed')
    need(result['model_buffers_sha256_before'] == result['model_buffers_sha256_after'], 'model buffers changed')
    return 'direct-process observation and independently supplied pins checked; not execution attestation'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--source-revision')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--reservation-sha256')
    parser.add_argument('--current-exposure-sha256')
    args = parser.parse_args()
    run = args.run
    cases_raw = args.cases.read_bytes()
    need(digest(cases_raw) == CASE_SHA, 'fixed case file mismatch')
    cases = json.loads(cases_raw)['cases']
    need(len(cases) == 48 and len({r['case_id'] for r in cases}) == 48, 'case count mismatch')
    by_id = {r['case_id']: r for r in cases}
    manifest_raw = (run/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    result = json.loads((run/'result.json').read_bytes())
    need(manifest['schema'] == 'distilgpt2-year-transfer-run-v1'
         and result['schema'] == 'distilgpt2-year-transfer-result-v1', 'result schema mismatch')
    need(len(manifest_raw) <= 4*1024*1024 and (run/'result.json').stat().st_size <= 4*1024*1024
         and (run/'attempt.jsonl').stat().st_size <= 8*1024*1024, 'metadata bounds exceeded')
    need(manifest['cases'] == cases and manifest['cases_sha256'] == CASE_SHA, 'manifest case mismatch')
    need(manifest['protocol_sha256'] == PROTOCOL_SHA, 'manifest protocol mismatch')
    need(result['manifest_sha256'] == digest(manifest_raw) and result['status'] == 'complete', 'manifest/result mismatch')
    provenance = validate_provenance(args, manifest, result, run)
    events = [json.loads(line) for line in (run/'attempt.jsonl').read_bytes().splitlines()]
    need(events and events[0] == {'event':'runtime_ready', 'actual_runtime':result['runtime']}, 'runtime event mismatch')
    cursor, records, offsets, payload_bytes = 1, [], {r['case_id']:0 for r in cases}, 0
    cache = {r['case_id']:{} for r in cases}

    def event(expected):
        nonlocal cursor
        need(cursor < len(events), 'missing event')
        value = events[cursor]; cursor += 1
        need(value.get('event') == expected, 'event order mismatch: '+expected)
        return value

    def action(row, name, family=None):
        nonlocal payload_bytes
        identity = row['case_id']
        start = event('start')
        need(start.get('case_id') == identity and start.get('action') == name, 'action start mismatch')
        if not args.synthetic or 'intervention' in start:
            need(set(start) == {'event','case_id','action','intervention'}, 'start event shape mismatch')
            need((start['intervention'] is None) == (name in ('clean','noop','clean_after')), 'start plan presence mismatch')
        recorded = event('raw_recorded')
        need(recorded.get('case_id') == identity and recorded.get('action') == name, 'raw event identity mismatch')
        descriptors = recorded['raw']
        expected_names = {'logits'} if name in ('clean','clean_after') else {'logits','block_before','block_after'}
        need(set(descriptors) == expected_names, 'payload names mismatch')
        payload = {}
        with (run/(identity+'.f32')).open('rb') as stream:
            for part, desc in sorted(descriptors.items(), key=lambda item:item[1]['offset']):
                length = LOGITS_BYTES if part == 'logits' else len(row['input_ids'])*ROW_BYTES
                need(set(desc) == {'offset','bytes','sha256'} and desc['offset'] == offsets[identity] and desc['bytes'] == length, 'payload offset/size mismatch')
                stream.seek(desc['offset']); raw = stream.read(length)
                need(len(raw) == length and digest(raw) == desc['sha256'], 'payload digest mismatch')
                offsets[identity] += length; payload_bytes += length; payload[part] = raw
        completed = event('complete')
        need(completed.get('case_id') == identity and completed.get('action') == name and completed.get('raw') == descriptors, 'completion mismatch')
        item = dict(completed); del item['event']; records.append(item)
        if 'intervention' in start and start['intervention'] is not None:
            need({key:item['intervention'][key] for key in start['intervention']} == start['intervention'],
                 'pre-forward plan differs from completed record')
        observed_score = score(payload['logits'], row)
        need(item['band_score'] == observed_score, 'saved score mismatch')
        saved = cache[identity]
        if name == 'clean':
            saved['clean'], saved['score'] = payload['logits'], observed_score
            need(item['intervention'] is None, 'unexpected clean plan')
        elif name == 'noop':
            need(payload['logits'] == saved['clean'] and payload['block_before'] == payload['block_after'], 'no-op mismatch')
            values(payload['block_before'])
            saved['before'] = payload['block_before']
            need(item['intervention'] is None, 'unexpected no-op plan')
        elif name == 'clean_after':
            need(payload['logits'] == saved['clean'] and item['intervention'] is None, 'clean-after mismatch')
        else:
            need(payload['block_before'] == saved['before'], 'pre-action tensor mismatch')
            partner = by_id[row['donor_case_id']]
            target = values(saved['before'][-ROW_BYTES:])
            donor = values(cache[partner['case_id']]['before'][-ROW_BYTES:])
            delta = [b-a for a,b in zip(target,donor)]
            n = norm(delta); m = family['dose']
            displacement = [m/n*d for d in delta]
            position = 0 if name == 'year_at_zero' else row['final_position']
            start_byte = position*ROW_BYTES
            before = saved['before'][start_byte:start_byte+ROW_BYTES]
            if name == 'self':
                expected_after, requested = before, 0.0
            else:
                if name == 'rotated_final':
                    displacement = [displacement[(i-1)%WIDTH] for i in range(WIDTH)]
                out = [round_f32(a+b) for a,b in zip(values(before),displacement)]
                expected_after, requested = struct.pack('<768I', *out), m
            expected_full = saved['before'][:start_byte]+expected_after+saved['before'][start_byte+ROW_BYTES:]
            need(payload['block_after'] == expected_full, 'landing or untouched bits mismatch')
            info = item['intervention']
            need(info['position'] == position and info['expected_before_hex'] == before.hex() and info['expected_after_hex'] == expected_after.hex(), 'expected plan bits mismatch')
            achieved = norm(a-b for a,b in zip(values(expected_after), values(before)))
            changed = sum(a != b for a,b in zip(struct.iter_unpack('<I', before), struct.iter_unpack('<I', expected_after)))
            need(info['requested'] == requested and info['achieved'] == achieved and info['observed_achieved'] == achieved, 'dose mismatch')
            need(abs(achieved-requested) <= 1e-5*requested and info['changed_bits'] == changed == info['observed_changed_bits'], 'realized dose or bit count mismatch')
            need(name != 'self' or payload['logits'] == saved['clean'], 'self logits mismatch')
            saved.setdefault('scores',{})[name] = observed_score

    for row in cases:
        action(row,'clean'); action(row,'noop')
    zero = event('position_zero_check')
    for i in range(0,48,2):
        need(cache[cases[i]['case_id']]['before'][:ROW_BYTES] == cache[cases[i+1]['case_id']]['before'][:ROW_BYTES], 'position-zero mismatch')
    need(zero == {'event':'position_zero_check','pairs_checked':24}, 'position-zero event mismatch')
    groups = []
    for i in range(8):
        rows = cases[i*6:(i+1)*6]
        diffs = {c:cache[rows[j]['case_id']]['score']-cache[rows[j+1]['case_id']]['score'] for j,c in zip((0,2,4), CONDITIONS)}
        f,r,n = (diffs[c] for c in CONDITIONS)
        groups.append({'pair_index':rows[0]['pair_index'], 'event':rows[0]['event'], **diffs,
            'forward_minus_neutral':f-n, 'comparator_pass':f>0.5 and r<-.5,
            'preference_pass':f>.5 and n>.5 and f-n>.1})
    def gate(key):
        selected = [g for g in groups if g[key]]
        return len(selected)>=6 and len({g['pair_index'] for g in selected})>=3
    clean = {'pairs':groups, 'comparator_gate_passed':gate('comparator_pass'),
        'preference_gate_passed':gate('preference_pass'),
        'comparator_gate_count':sum(g['comparator_pass'] for g in groups),
        'preference_gate_count':sum(g['preference_pass'] for g in groups)}
    gates = event('clean_gates')
    need(gates['summary'] == clean == result['clean_summary'], 'clean gate mismatch')
    proceed = clean['comparator_gate_passed'] or clean['preference_gate_passed']
    need(result['interventions_performed'] is proceed, 'gate-dependent execution mismatch')
    families = []
    if proceed:
        for i in range(8):
            rows = cases[i*6:(i+1)*6]
            norms = {}
            for j,c in zip((0,2,4), CONDITIONS):
                low = values(cache[rows[j]['case_id']]['before'][-ROW_BYTES:])
                high = values(cache[rows[j+1]['case_id']]['before'][-ROW_BYTES:])
                norms[c] = norm(b-a for a,b in zip(low,high))
            m = 0.5*min(norms.values())
            need(math.isfinite(m) and m>2**-20, 'invalid family dose')
            family = {'pair_noun_index':i, 'dose':m, 'norms':norms,
                      'fractions':{k:m/v for k,v in norms.items()}}
            got = event('family_dose'); need(got == {'event':'family_dose', **family}, 'family dose mismatch')
            families.append(family)
            for row in rows:
                shift = row['rotation']
                for name in ACTIONS[shift:]+ACTIONS[:shift]: action(row,name,family)
                action(row,'clean_after')
    need(cursor == len(events), 'extra events')
    need(records == result['records'] and families == result['family_doses'], 'records or families mismatch')
    count = 336 if proceed else 96
    need(len(records) == count and all(result[k] == count for k in ('submitted_forwards','model_call_returns_observed','returned_forwards')), 'forward counts mismatch')
    need(payload_bytes == result['raw_payload_bytes'], 'aggregate byte count mismatch')
    for identity, end in offsets.items():
        need((run/(identity+'.f32')).stat().st_size == end, 'unaccounted raw bytes')
    need({p.name for p in run.glob('*.f32')} == {identity+'.f32' for identity in offsets}, 'unexpected raw file inventory')

    pairs = []
    if proceed:
        for i in range(8):
            rows = cases[i*6:(i+1)*6]; conditions = {}
            for j,c in zip((0,2,4), CONDITIONS):
                low,high = (cache[rows[k]['case_id']] for k in (j,j+1))
                actions = {}
                for name in ACTIONS:
                    le = low['scores'][name]-low['score']; he = high['scores'][name]-high['score']
                    actions[name] = {'low_E':le,'high_E':he,'low_donorward':-le,'high_donorward':he,'donorward_mean':(-le+he)/2}
                conditions[c] = actions
            f,n,r = (conditions[c]['year_final'] for c in ('forward','neutral','reverse'))
            pref = (f['low_donorward']>0 and f['high_donorward']>0 and n['low_donorward']>0 and n['high_donorward']>0
                    and f['donorward_mean']-n['donorward_mean']>.1
                    and f['donorward_mean']>conditions['forward']['year_at_zero']['donorward_mean']
                    and f['donorward_mean']>conditions['forward']['rotated_final']['donorward_mean'])
            comp = f['low_donorward']>0 and f['high_donorward']>0 and r['low_donorward']<0 and r['high_donorward']<0
            pairs.append({'pair_index':rows[0]['pair_index'],'event':rows[0]['event'],'conditions':conditions,
                'forward_minus_neutral':f['donorward_mean']-n['donorward_mean'],
                'preference_pattern':pref,'comparator_pattern':comp})
        pc,cc = sum(p['preference_pattern'] for p in pairs),sum(p['comparator_pattern'] for p in pairs)
        means = {c:{name:math.fsum(p['conditions'][c][name]['donorward_mean'] for p in pairs)/len(pairs) for name in ACTIONS} for c in CONDITIONS}
        summary = {'pairs':pairs,'mean_donorward_by_condition_and_action':means,'preference_pattern_count':pc,
            'comparator_pattern_count':cc,'preference_pattern_supported':pc>=6 if clean['preference_gate_passed'] else None,
            'comparator_pattern_supported':cc>=6 if clean['comparator_gate_passed'] else None}
        need(summary == result['intervention_summary'], 'intervention summary mismatch')
    else:
        need(result['intervention_summary'] is None, 'unexpected intervention summary')
    print(json.dumps({'status':'pass','provenance':provenance,'forwards':count,'bytes':payload_bytes,
                      'clean_gates':clean,'intervention_pairs':pairs}, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
