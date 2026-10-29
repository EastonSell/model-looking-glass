"""Independent audit of public CPU injection-discovery observations.

Uses only the standard library. Never reads private witnesses, the secret
site, a model, or held-out inputs. This checks saved public bytes and the
specified arithmetic, not source execution, chronology, or causal truth.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct


CASES = (
    ('The cat near the windows', (464, 3797, 1474, 262, 9168)),
    ('The cats near the windows', (464, 11875, 1474, 262, 9168)),
    ('The teacher in the room', (464, 4701, 287, 262, 2119)),
    ('The teachers in the room', (464, 7799, 287, 262, 2119)),
    ('The artist beside the rivers', (464, 6802, 13970, 262, 18180)),
    ('The artists beside the rivers', (464, 7912, 13970, 262, 18180)),
    ('The farmer behind the house', (464, 18739, 2157, 262, 2156)),
    ('The farmers behind the house', (464, 9818, 2157, 262, 2156)),
)
ROLES = ('clean', 'repeat', 'injected', *(f'forward_{i}' for i in range(1, 5)),
         *(f'reverse_{i}' for i in range(1, 5)), 'null', 'self', 'upstream', 'downstream', 'verify_clean')
CONDITIONS = {'clean': 'clean', 'repeat': 'clean', 'injected': 'injected', 'null': 'null',
              'self': 'self', 'upstream': 'upstream', 'downstream': 'downstream', 'verify_clean': 'clean',
              **{f'forward_{i}': 'clean_to_injected' for i in range(1, 5)},
              **{f'reverse_{i}': 'injected_to_clean' for i in range(1, 5)}}
PROTOCOL_SHA256 = '95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee'
REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def same(actual, expected):
    """Do not accept booleans as integers or other cross-type equality."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(same(actual[key], value) for key, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def read_json(path, limit):
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    require(0 < len(raw) <= limit, 'file is empty or exceeds its bound')
    value = json.loads(raw)
    require(type(value) is dict and canonical(value) == raw, 'JSON is not a canonical object')
    return raw, value


def float32(bits):
    """Decode bits to binary64 without FTZ-sensitive float32 conversion."""
    exponent = (bits >> 23) & 255
    require(exponent != 255, 'nonfinite float32 is not allowed')
    fraction = bits & 0x7fffff
    value = math.ldexp(float(fraction if exponent == 0 else (1 << 23) | fraction),
                       -149 if exponent == 0 else exponent - 150)
    return -value if bits >> 31 else value


def decode(value, count):
    require(type(value) is str and len(value) == ((count * 4 + 2) // 3) * 4,
            'encoded tensor length differs')
    raw = base64.b64decode(value, validate=True)
    require(len(raw) == count * 4 and base64.b64encode(raw).decode() == value, 'tensor encoding differs')
    for (bits,) in struct.iter_unpack('<I', raw):
        require(((bits >> 23) & 255) != 255, 'tensor contains a nonfinite value')
    return raw


def margin(raw, first=318, second=389):
    a = float32(struct.unpack_from('<I', raw, 4 * first)[0])
    b = float32(struct.unpack_from('<I', raw, 4 * second)[0])
    return math.fsum((a, -b))


def within(left, right):
    require(len(left) == len(right), 'control tensor size differs')
    for (a,), (b,) in zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right)):
        x, y = float32(a), float32(b)
        if abs(x - y) > 1e-5 + 1e-5 * max(abs(x), abs(y)):
            return False
    return True


def norm(left, right):
    require(len(left) == len(right) == 3072, 'candidate vector size differs')
    return math.sqrt(math.fsum((float32(a) - float32(b)) ** 2
        for (a,), (b,) in zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right))))


def input_hash(ids):
    return hashlib.sha256(b'MLG-number-input-v1\0' + bytes([len(ids)]) +
                          b''.join(value.to_bytes(4, 'big') for value in ids)).hexdigest()


def read_plan(path, source_commit, expected_sha256=None):
    require(type(source_commit) is str and re.fullmatch(r'[0-9a-f]{40}', source_commit), 'source commit is invalid')
    raw, plan = read_json(Path(path), 128 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    require(expected_sha256 is None or digest == expected_sha256, 'public plan digest differs')
    require(set(plan) == {'schema', 'protocol_sha256', 'checkpoint_revision', 'source_commit', 'public_entropy', 'requests'}
            and plan['schema'] == 'mlg-number-injection-family-request-plan-v2'
            and plan['protocol_sha256'] == PROTOCOL_SHA256 and plan['checkpoint_revision'] == REVISION
            and plan['source_commit'] == source_commit and type(plan['public_entropy']) is str
            and re.fullmatch(r'[0-9a-f]{64}', plan['public_entropy']) and type(plan['requests']) is list,
            'public plan identity differs')
    entropy = bytes.fromhex(plan['public_entropy'])
    schedule = []
    for family in range(4):
        first, second = 2 * family, 2 * family + 1
        schedule.extend((index, role) for index in (first, second) for role in ROLES[:-1])
        schedule.extend(((first, 'verify_clean'), (second, 'verify_clean')))
    expected = []
    for index, role in schedule:
        logical_ordinal = index * len(ROLES) + ROLES.index(role)
        request_id = hashlib.sha256(b'MLG-number-injection-request-v1\0' + entropy
                                    + logical_ordinal.to_bytes(4, 'big')).digest()[:16].hex()
        expected.append({'case_index': index, 'role': role, 'request_id': request_id,
                         'input_sha256': input_hash(CASES[index][1])})
    require(same(plan['requests'], expected) and len({row['request_id'] for row in expected}) == 128,
            'public plan requests, schedule or derived identities differ')
    return {(row['case_index'], row['role']): row for row in expected}, digest


def expected_boundary(path, order):
    return {'module_path': path, 'capture_order': order, 'invocation': 0, 'batch_index': 0,
            'token_index': 4, 'output_selector': [0] if path.endswith('.attn') or path == 'transformer.h.5' else [],
            'output_shape': [1, 5, 768], 'selected_shape': [768], 'dtype': 'torch.float32', 'device': 'cpu'}


def expected_action(role, record, clean_id, injected_id):
    if role in ('clean', 'repeat', 'injected', 'verify_clean'):
        return None
    if role in ('null', 'self'):
        path = 'transformer.h.1.mlp'
    elif role == 'upstream':
        path = 'transformer.h.0.mlp'
    elif role == 'downstream':
        path = 'transformer.h.5'
    else:
        path = f'transformer.h.{role[-1]}.mlp'
    selector = [0] if role == 'downstream' else []
    if role == 'null':
        donor = None
    else:
        donor_id = record['request_id'] if role == 'self' else injected_id if role.startswith('reverse_') else clean_id
        donor = {'request_id': donor_id, 'input_sha256': record['input_sha256'], 'module_path': path,
                 'invocation': 0, 'batch_index': 0, 'token_index': 4, 'output_selector': selector,
                 'selected_shape': [768], 'dtype': 'torch.float32', 'device': 'cpu'}
    return {'operation': 'null' if role == 'null' else 'patch', 'module_path': path,
            'invocation': 0, 'batch_index': 0, 'token_index': 4, 'output_selector': selector,
            'output_shape': [1, 5, 768], 'selected_shape': [768], 'dtype': 'torch.float32',
            'device': 'cpu', 'donor': donor}


def audit(directory, plan_path, source_commit, expected_prediction=None, expected_archive=None, expected_plan=None):
    directory = Path(directory)
    require(directory.is_dir(), 'public archive directory is unavailable')
    requests, plan_sha = read_plan(plan_path, source_commit, expected_plan)
    names = {f'observation-{i:02d}-{role}.json' for i in range(8) for role in ROLES} | {'prediction.json'}
    require({p.name for p in directory.iterdir()} == names, 'public archive inventory differs')
    prediction_raw, prediction = read_json(directory / 'prediction.json', 8 * 1024 ** 2)
    require(set(prediction) == {'schema', 'status', 'protocol_sha256', 'checkpoint_revision', 'execution_device',
                                'private_controls', 'first_changed', 'ranking', 'votes', 'mean_l2', 'earliest',
                                'random', 'magnitude', 'clean_accuracy', 'predictions', 'cases'}
            and prediction.get('schema') == 'mlg-number-injection-discovery-prediction-v1'
            and prediction.get('status') == 'complete' and prediction.get('execution_device') == 'cpu'
            and prediction.get('private_controls') == 'caller-attested'
            and prediction.get('protocol_sha256') == PROTOCOL_SHA256 and prediction.get('checkpoint_revision') == REVISION,
            'prediction identity differs')
    prediction_sha = hashlib.sha256(prediction_raw).hexdigest()
    require(expected_prediction is None or prediction_sha == expected_prediction, 'prediction digest differs')
    require(type(prediction.get('cases')) is list and len(prediction['cases']) == 8, 'prediction cases differ')
    seen, hashes, votes = set(), [], {site: 0 for site in range(1, 5)}
    all_l2, all_effects, correct, ties = {site: [] for site in range(1, 5)}, {site: [] for site in range(1, 5)}, 0, 0
    case_results = []
    paths = [f'transformer.h.{layer}.{kind}' for layer in range(6) for kind in ('attn', 'mlp')] + ['transformer.h.5']
    expected_trace = [expected_boundary(path, order) for order, path in enumerate(paths)]
    for index, (text, ids) in enumerate(CASES):
        records, payloads = {}, {}
        for role in ROLES:
            raw, record = read_json(directory / f'observation-{index:02d}-{role}.json', 512 * 1024)
            require(set(record) == {'schema', 'case_index', 'role', 'request_id', 'input_sha256', 'condition',
                                    'trace', 'action', 'payload'}
                    and record['schema'] == 'mlg-number-injection-raw-observation-v1'
                    and type(record['case_index']) is int and record['case_index'] == index and record['role'] == role
                    and record['condition'] == CONDITIONS[role] and record['input_sha256'] == input_hash(ids)
                    and record['request_id'] == requests[index, role]['request_id']
                    and type(record['request_id']) is str and re.fullmatch(r'[0-9a-f]{32}', record['request_id'])
                    and record['request_id'] not in seen and same(record['trace'], expected_trace),
                    'record identity, trace or request order differs')
            seen.add(record['request_id'])
            item = record['payload']
            require(type(item) is dict and set(item) == {'attention', 'mlp', 'block', 'logits'}
                    and type(item['attention']) is list and len(item['attention']) == 6
                    and type(item['mlp']) is list and len(item['mlp']) == 6, 'payload inventory differs')
            payloads[role] = {'attention': [decode(x, 768) for x in item['attention']],
                              'mlp': [decode(x, 768) for x in item['mlp']],
                              'block': decode(item['block'], 768), 'logits': decode(item['logits'], 50257)}
            records[role] = record
            hashes.append(hashlib.sha256(raw).digest())
        clean_id, injected_id = records['clean']['request_id'], records['injected']['request_id']
        clean, injected = payloads['clean'], payloads['injected']
        inventory = []
        for role in ROLES:
            record, data = records[role], payloads[role]
            require(same(record['action'], expected_action(role, record, clean_id, injected_id)), 'action or donor differs')
            inventory.append({'role': role, 'request_id': record['request_id'], 'input_sha256': record['input_sha256'],
                'condition': record['condition'], 'attention_sha256': [hashlib.sha256(x).hexdigest() for x in data['attention']],
                'mlp_sha256': [hashlib.sha256(x).hexdigest() for x in data['mlp']],
                'block_sha256': hashlib.sha256(data['block']).hexdigest(),
                'logits_sha256': hashlib.sha256(data['logits']).hexdigest(),
                'present_margin': margin(data['logits']), 'past_margin': margin(data['logits'], 373, 547),
                'trace': record['trace'], 'action': record['action']})
        for role in ('repeat', 'verify_clean'):
            current = payloads[role]
            require(all(within(a, b) for a, b in zip(clean['attention'] + clean['mlp'] + [clean['block'], clean['logits']],
                        current['attention'] + current['mlp'] + [current['block'], current['logits']])),
                    'clean repeat or family-final control differs')
        for role in ('null', 'self'):
            require(within(payloads[role]['mlp'][1], injected['mlp'][1])
                    and within(payloads[role]['logits'], injected['logits']), 'null or self control differs')
        require(within(payloads['upstream']['logits'], injected['logits'])
                and within(payloads['downstream']['logits'], clean['logits'])
                and payloads['upstream']['mlp'][0] == clean['mlp'][0]
                and payloads['downstream']['block'] == clean['block'], 'boundary control differs')
        require(all(within(a, b) for a, b in ((clean['attention'][0], injected['attention'][0]),
                (clean['mlp'][0], injected['mlp'][0]), (clean['attention'][1], injected['attention'][1]))),
                'pre-injection control differs')
        for site in range(1, 5):
            require(payloads[f'forward_{site}']['mlp'][site] == clean['mlp'][site]
                    and payloads[f'reverse_{site}']['mlp'][site] == injected['mlp'][site],
                    'selected candidate capture does not land on its donor')
        first = next((site for site in range(1, 5) if not within(clean['mlp'][site], injected['mlp'][site])), None)
        if first is not None:
            votes[first] += 1
        clean_m, injected_m = margin(clean['logits']), margin(injected['logits'])
        forward = [margin(payloads[f'forward_{site}']['logits']) for site in range(1, 5)]
        reverse = [margin(payloads[f'reverse_{site}']['logits']) for site in range(1, 5)]
        forward_effects = [math.fsum((x, -injected_m)) for x in forward]
        reverse_effects = [math.fsum((x, -clean_m)) for x in reverse]
        l2 = [norm(clean['mlp'][site], injected['mlp'][site]) for site in range(1, 5)]
        is_correct = clean_m > 0 if index % 2 == 0 else clean_m < 0
        correct += is_correct
        ties += clean_m == 0
        for site in range(1, 5):
            all_l2[site].append(l2[site - 1]); all_effects[site].append(forward_effects[site - 1])
        expected_case = {'index': index, 'family_index': index // 2,
            'number': 'singular' if index % 2 == 0 else 'plural', 'text': text, 'input_ids': list(ids),
            'first_changed': first, 'l2': l2, 'clean_margin': clean_m, 'injected_margin': injected_m,
            'clean_correct': is_correct, 'clean_past_margin': margin(clean['logits'], 373, 547),
            'injected_past_margin': margin(injected['logits'], 373, 547), 'forward_margins': forward,
            'reverse_margins': reverse, 'forward_effects': forward_effects, 'reverse_effects': reverse_effects,
            'observations': inventory}
        require(same(prediction['cases'][index], expected_case), 'prediction case does not match stored bytes')
        case_results.append({'case_index': index, 'first_changed': first, 'l2': l2,
                             'forward_effects': forward_effects, 'reverse_effects': reverse_effects})
    ranking = sorted(range(1, 5), key=lambda site: (-votes[site], site))
    means = {site: math.fsum(all_l2[site]) / 8 for site in range(1, 5)}
    predictions = {site: math.fsum(all_effects[site]) / 8 for site in range(1, 5)}
    random = 1 + hashlib.sha256(b'MLG-number-v1-injection-random\0' + (20261028).to_bytes(8, 'big')).digest()[0] % 4
    expected_aggregate = {'first_changed': ranking[0] if votes[ranking[0]] else None, 'ranking': ranking,
        'votes': {str(i): votes[i] for i in range(1, 5)}, 'mean_l2': {str(i): means[i] for i in range(1, 5)},
        'earliest': 1, 'random': random, 'magnitude': max(range(1, 5), key=lambda i: (means[i], -i)),
        'clean_accuracy': {'correct': correct, 'incorrect': 8 - correct, 'ties': ties},
        'predictions': {str(i): predictions[i] for i in range(1, 5)}}
    require(all(same(prediction.get(key), value) for key, value in expected_aggregate.items()),
            'prediction aggregate does not match stored bytes')
    archive_sha = hashlib.sha256(b'MLG-number-injection-archive-v1\0' + b''.join(hashes)).hexdigest()
    require(expected_archive is None or archive_sha == expected_archive, 'archive digest differs')
    return {'schema': 'mlg-independent-injection-public-audit-v1', 'status': 'passed',
            'source_commit': source_commit, 'plan_sha256': plan_sha, 'prediction_sha256': prediction_sha,
            'archive_sha256': archive_sha, 'records': 128,
            'cases': case_results, 'aggregate': expected_aggregate,
            'limitations': ['Public stored bytes only; no private injection witnesses or hidden site inspected.',
                            'Does not establish execution, publication chronology, held-out outcomes, or natural reasoning.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('public_directory')
    parser.add_argument('--request-plan', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--prediction-sha256', required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.public_directory, args.request_plan, args.source_commit,
                   args.prediction_sha256, args.archive_sha256, args.plan_sha256)
    result['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data = canonical(result) + b'\n'
    with open(args.output, 'xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'status': result['status'], 'records': result['records'],
                      'prediction_sha256': result['prediction_sha256'], 'archive_sha256': result['archive_sha256']}))


if __name__ == '__main__':
    main()
