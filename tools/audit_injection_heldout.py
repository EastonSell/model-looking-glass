"""Independent, private audit of saved injection held-out evidence.

This module does not load a model or establish publication, process execution,
custody, or quiescence. All inputs and results remain private until the study's
authorized reveal. Its arithmetic and action checks do not call the producer's
scorer or witness auditor.
"""
import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


WIDTH = 768
WIDTH_BYTES = 4 * WIDTH
PROTOCOL_SHA256 = '95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee'
CHECKPOINT_REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
ROLES = ('clean', 'repeat', 'injected', *(f'forward_{site}' for site in range(1, 5)),
         *(f'reverse_{site}' for site in range(1, 5)), 'null', 'self', 'upstream', 'downstream', 'verify_clean')
CONDITIONS = {'clean': 'clean', 'repeat': 'clean', 'injected': 'injected', 'null': 'null',
              'self': 'self', 'upstream': 'upstream', 'downstream': 'downstream', 'verify_clean': 'clean',
              **{f'forward_{site}': 'clean_to_injected' for site in range(1, 5)},
              **{f'reverse_{site}': 'injected_to_clean' for site in range(1, 5)}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(same(left[key], value) for key, value in right.items())
    if type(left) is list:
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def parse_object(raw, limit):
    require(type(raw) is bytes and 0 < len(raw) <= limit, 'JSON record size differs')
    try:
        value = json.loads(raw)
        require(type(value) is dict and canonical(value) == raw, 'JSON record is not a canonical object')
    except (TypeError, UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError('JSON record is malformed') from error
    return value


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate JSON object key')
        value[key] = item
    return value


def _reject_constant(_value):
    raise ValueError('nonfinite JSON number')


def tensor(value, length, *, finite=True):
    require(type(value) is str and len(value) == ((length + 2) // 3) * 4, 'tensor encoding size differs')
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('tensor encoding is invalid') from error
    require(len(raw) == length and base64.b64encode(raw).decode('ascii') == value, 'tensor encoding differs')
    if finite:
        require(length % 4 == 0, 'tensor width differs')
        for (bits,) in struct.iter_unpack('<I', raw):
            float32(bits)
    return raw


def within(left, right):
    require(type(left) is bytes and type(right) is bytes and len(left) == len(right) and len(left) % 4 == 0,
            'control tensor shapes differ')
    for (a,), (b,) in zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right)):
        x, y = float32(a), float32(b)
        if abs(x - y) > 1e-5 + 1e-5 * max(abs(x), abs(y)):
            return False
    return True


def input_hash(ids):
    require(type(ids) is tuple and 1 <= len(ids) <= 32
            and all(type(value) is int and 0 <= value < 50257 for value in ids), 'input IDs differ')
    return hashlib.sha256(b'MLG-number-input-v1\0' + bytes([len(ids)])
                          + b''.join(value.to_bytes(4, 'big') for value in ids)).hexdigest()


def request_id(commitment, index, role):
    require(type(commitment) is str and re.fullmatch('[0-9a-f]{64}', commitment) is not None
            and type(index) is int and 0 <= index < 8 and role in ROLES, 'request identity differs')
    ordinal = index * 16 + ROLES.index(role)
    return hashlib.sha256(b'MLG-injection-heldout-request-v1\0' + bytes.fromhex(commitment)
                          + ordinal.to_bytes(4, 'big')).digest()[:16].hex()


def parse_public(raw, *, index, role, ids, commitment):
    """Decode a single public observation without the producer's parser."""
    value = parse_object(raw, 512 * 1024)
    require(set(value) == {'schema', 'case_index', 'role', 'request_id', 'input_sha256', 'condition',
                           'trace', 'action', 'payload'}
            and value['schema'] == 'mlg-number-injection-raw-observation-v1'
            and type(value['case_index']) is int and value['case_index'] == index
            and value['role'] == role and value['condition'] == CONDITIONS[role]
            and value['request_id'] == request_id(commitment, index, role)
            and value['input_sha256'] == input_hash(ids), 'public observation identity differs')
    paths = [f'transformer.h.{layer}.{kind}' for layer in range(6) for kind in ('attn', 'mlp')] + ['transformer.h.5']
    expected = [{'module_path': path, 'capture_order': order, 'invocation': 0, 'batch_index': 0,
                 'token_index': len(ids) - 1, 'output_selector': [0] if path.endswith('.attn') or path == 'transformer.h.5' else [],
                 'output_shape': [1, len(ids), WIDTH], 'selected_shape': [WIDTH],
                 'dtype': 'torch.float32', 'device': 'cpu'} for order, path in enumerate(paths)]
    require(same(value['trace'], expected), 'public capture metadata differs')
    payload = value['payload']
    require(type(payload) is dict and set(payload) == {'attention', 'mlp', 'block', 'logits'}
            and type(payload['attention']) is list and len(payload['attention']) == 6
            and type(payload['mlp']) is list and len(payload['mlp']) == 6, 'public payload inventory differs')
    record = {key: item for key, item in value.items() if key != 'payload'}
    return {'record': record, 'attention': [tensor(item, WIDTH_BYTES) for item in payload['attention']],
            'mlp': [tensor(item, WIDTH_BYTES) for item in payload['mlp']],
            'block': tensor(payload['block'], WIDTH_BYTES), 'logits': tensor(payload['logits'], 50257 * 4)}


def expected_action(role, observation, clean, injected, ids):
    record = observation['record']
    if role in ('clean', 'repeat', 'injected', 'verify_clean'):
        return None
    path = ('transformer.h.5' if role == 'downstream' else
            'transformer.h.0.mlp' if role == 'upstream' else
            'transformer.h.1.mlp' if role in ('null', 'self') else f'transformer.h.{role[-1]}.mlp')
    selector = [0] if role == 'downstream' else []
    donor = None
    if role != 'null':
        source = observation if role == 'self' else injected if role.startswith('reverse_') else clean
        require(type(source) is dict, 'action source is missing')
        donor = {'request_id': source['record']['request_id'], 'input_sha256': record['input_sha256'],
                 'module_path': path, 'invocation': 0, 'batch_index': 0, 'token_index': len(ids) - 1,
                 'output_selector': selector, 'selected_shape': [WIDTH], 'dtype': 'torch.float32', 'device': 'cpu'}
    return {'operation': 'null' if role == 'null' else 'patch', 'module_path': path,
            'invocation': 0, 'batch_index': 0, 'token_index': len(ids) - 1,
            'output_selector': selector, 'output_shape': [1, len(ids), WIDTH], 'selected_shape': [WIDTH],
            'dtype': 'torch.float32', 'device': 'cpu', 'donor': donor}


def verify_witness(raw, *, index, role, ids, site, observation, clean, injected):
    """Recompute action landing and saved controls from private raw witnesses."""
    record = parse_object(raw, 1024 * 1024)
    public = observation['record']
    require(set(record) == {'schema', 'case_index', 'role', 'request_id', 'input_sha256', 'condition', 'actions'}
            and record['schema'] == 'mlg-number-injection-private-forward-v1'
            and type(record['case_index']) is int and record['case_index'] == index and record['role'] == role
            and record['request_id'] == public['request_id'] and record['input_sha256'] == public['input_sha256']
            and record['condition'] == public['condition'] and type(record['actions']) is list,
            'private record identity differs')
    expected_public = expected_action(role, observation, clean, injected, ids)
    require(same(public['action'], expected_public), 'public action or donor identity differs')
    injecting = role in ('injected', 'null', 'self', 'upstream', 'downstream') or role.startswith('forward_')
    action_needed = role in ('null', 'self', 'upstream', 'downstream') or role.startswith(('forward_', 'reverse_'))
    expected_ops = ({'add'} if injecting else set()) | ({'null' if role == 'null' else 'patch'} if action_needed else set())
    require((not injecting and not action_needed and role not in ('repeat', 'verify_clean')) or clean is not None,
            'clean baseline is missing')
    require(not action_needed or injected is not None, 'injected baseline is missing')
    events = expected_events(site=site, role=role, token_count=len(ids))
    actions, tensor_bytes = {}, 0
    fields = {'request_id', 'input_sha256', 'condition', 'operation', 'module_path', 'token_index',
              'before', 'after', 'donor', 'donor_identity', 'realized_delta_f64_le', 'metadata_json'}
    for item in record['actions']:
        require(type(item) is dict and set(item) == fields and item['operation'] in expected_ops
                and item['operation'] not in actions and item['request_id'] == public['request_id']
                and item['input_sha256'] == public['input_sha256'] and item['condition'] == public['condition']
                and type(item['token_index']) is int and item['token_index'] == len(ids) - 1,
                'private action identity differs')
        operation = item['operation']
        prefix = 'inject' if operation == 'add' else 'action'
        action_name = 'inject_add' if operation == 'add' else 'action_' + operation
        names = (prefix + '_before', action_name, prefix + '_after')
        require(item['module_path'] == events[action_name]['module_path']
                and type(item['metadata_json']) is str and len(item['metadata_json']) <= 65536,
                'private action coordinates differ')
        metadata_raw = item['metadata_json'].encode('utf-8')
        require(len(metadata_raw) <= 65536, 'private metadata exceeds its bound')
        try:
            metadata = json.loads(metadata_raw)
            require(canonical(metadata) == metadata_raw and same(metadata, [events[name] for name in names]),
                    'private action order differs')
        except (TypeError, UnicodeError, RecursionError, OverflowError) as error:
            raise ValueError('private action metadata is malformed') from error
        before = tensor(item['before'], len(ids) * WIDTH_BYTES)
        after = tensor(item['after'], len(ids) * WIDTH_BYTES)
        donor = None if item['donor'] is None else tensor(item['donor'], WIDTH_BYTES)
        delta = None if item['realized_delta_f64_le'] is None else tensor(item['realized_delta_f64_le'], WIDTH * 8, finite=False)
        tensor_bytes += len(before) + len(after) + (len(donor) if donor is not None else 0) + (len(delta) if delta is not None else 0)
        check_action_scope(before, after, token_count=len(ids), operation=operation, donor=donor, delta=delta)
        if operation == 'add':
            require(item['donor_identity'] is None, 'injection has a donor')
        elif operation == 'null':
            require(item['donor_identity'] is None, 'null action has a donor')
        else:
            require(same(item['donor_identity'], expected_public['donor']), 'private donor coordinates differ')
            expected_donor = (observation['record']['request_id'] if role == 'self' else
                              injected['record']['request_id'] if role.startswith('reverse_') else clean['record']['request_id'])
            require(item['donor_identity']['request_id'] == expected_donor, 'private donor request differs')
            donor_bits = (before[-WIDTH_BYTES:] if role == 'self' else clean['block'] if role == 'downstream' else
                          clean['mlp'][0] if role == 'upstream' else
                          injected['mlp'][int(role[-1])] if role.startswith('reverse_') else clean['mlp'][int(role[-1])])
            require(donor == donor_bits, 'private donor bytes differ')
        actions[operation] = (item, before, after)
    require(set(actions) == expected_ops, 'private actions are incomplete')
    action = actions.get('patch') or actions.get('null')
    if action is not None:
        target = (observation['block'] if role == 'downstream' else observation['mlp'][
            0 if role == 'upstream' else 1 if role in ('self', 'null') else int(role[-1])])
        require(action[2][-WIDTH_BYTES:] == target, 'public post-action capture differs')
    add = actions.get('add')
    if add is not None:
        require(within(add[1][-WIDTH_BYTES:], clean['mlp'][site]), 'pre-injection baseline differs')
        if action is not None and action[0]['module_path'] == add[0]['module_path']:
            require(add[2] == action[1], 'same-site action witnesses are not adjacent')
        else:
            require(add[2][-WIDTH_BYTES:] == observation['mlp'][site], 'post-injection capture differs')
        for layer in range(site):
            require(within(observation['attention'][layer], clean['attention'][layer])
                    and within(observation['mlp'][layer], clean['mlp'][layer]), 'upstream trace differs')
        require(within(observation['attention'][site], clean['attention'][site]), 'pre-injection attention differs')
    if role in ('repeat', 'verify_clean'):
        left = observation['attention'] + observation['mlp'] + [observation['block'], observation['logits']]
        right = clean['attention'] + clean['mlp'] + [clean['block'], clean['logits']]
        require(all(within(a, b) for a, b in zip(left, right)), 'clean control differs')
    if role in ('null', 'self'):
        require(within(observation['mlp'][1], injected['mlp'][1])
                and within(observation['logits'], injected['logits']), 'null or self control differs')
    elif role == 'upstream':
        require(within(observation['logits'], injected['logits']), 'upstream control differs')
    elif role == 'downstream':
        require(within(observation['logits'], clean['logits']), 'downstream control differs')
    return tensor_bytes


def float32(bits):
    """Decode finite IEEE binary32 to binary64 without host float32 casts."""
    require(type(bits) is int and 0 <= bits < (1 << 32), 'invalid float32 bits')
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    require(exponent != 255, 'nonfinite float32')
    value = math.ldexp(float(fraction if exponent == 0 else (1 << 23) | fraction),
                       -149 if exponent == 0 else exponent - 150)
    return -value if bits >> 31 else value


def direction_bits():
    """Reconstruct the public fixed seed and its exact signed components."""
    out = []
    for chunk in range(3):
        digest = hashlib.sha256(b'MLG-v1-direction\0' + (20261028).to_bytes(8, 'big')
                                + chunk.to_bytes(4, 'big')).digest()
        for byte in digest:
            out.extend(0x3c93cd3a if byte & (1 << bit) else 0xbc93cd3a
                       for bit in range(7, -1, -1))
    return tuple(out)


def rounded_add_bits(left, right):
    """Add two finite binary32 values and return nearest-even binary32 bits.

    The only supported right operand is one of the fixed direction values.
    Cancellation of those binary32 values cannot produce a nonzero subnormal,
    so the final host conversion cannot flush a nonzero result.
    """
    require(right in (0x3c93cd3a, 0xbc93cd3a), 'direction component differs')
    total = math.fsum((float32(left), float32(right)))
    try:
        result = struct.unpack('<I', struct.pack('<f', total))[0]
    except OverflowError as error:
        raise ValueError('injection result is not finite') from error
    float32(result)
    return result


def check_action_scope(before, after, *, token_count, operation, donor, delta=None):
    """Verify whole-output scope and selected-token patch, null or add bits."""
    require(type(token_count) is int and 1 <= token_count <= 32, 'invalid token count')
    length = token_count * WIDTH_BYTES
    require(type(before) is bytes and type(after) is bytes and len(before) == len(after) == length,
            'action snapshots differ in shape')
    for value in (before, after):
        for (bits,) in struct.iter_unpack('<I', value):
            float32(bits)
    start = length - WIDTH_BYTES
    require(before[:start] == after[:start], 'action changed an unselected token')
    if operation == 'null':
        require(donor is None and delta is None and before == after, 'null action changed the output')
    elif operation == 'patch':
        require(type(donor) is bytes and len(donor) == WIDTH_BYTES and delta is None,
                'patch donor is invalid')
        require(after[start:] == donor, 'patch did not land on the donor bits')
    elif operation == 'add':
        require(donor is None and type(delta) is bytes and len(delta) == 8 * WIDTH,
                'injection delta is invalid')
        before_bits = struct.unpack('<768I', before[start:])
        after_bits = struct.unpack('<768I', after[start:])
        expected = tuple(rounded_add_bits(x, d) for x, d in zip(before_bits, direction_bits()))
        require(after_bits == expected and before_bits != after_bits, 'injection did not land')
        realized = struct.pack('<768d', *(math.fsum((float32(a), -float32(b)))
                                                for a, b in zip(after_bits, before_bits)))
        require(delta == realized, 'recorded injection delta differs')
    else:
        raise ValueError('unknown action')


def read_pair(frame, ack, *, ordinal, binding, previous, receipt):
    """Independently check one frame, acknowledgment and retained receipt."""
    header_size = struct.calcsize('>8sHII32s32s')
    require(type(frame) is bytes and header_size + 40 < len(frame) <= 1024 * 1024 + 1024
            and type(ack) is bytes and len(ack) == struct.calcsize('>8sH32s')
            and type(ordinal) is int and 0 <= ordinal < 128
            and type(binding) is bytes and len(binding) == 32
            and type(previous) is bytes and len(previous) == 32, 'pair bounds differ')
    magic, number, public_size, private_size, bound, prior = struct.unpack_from('>8sHII32s32s', frame)
    require(magic == b'MLGNJ001' and number == ordinal and bound == binding and prior == previous
            and 0 < public_size <= 512 * 1024 and 0 < private_size <= 1024 * 1024
            and public_size + private_size <= 1024 * 1024
            and len(frame) == header_size + public_size + private_size + 40
            and frame[-8:] == b'MLGNEND1', 'pair framing differs')
    end_public = header_size + public_size
    end_private = end_public + private_size
    public, private = frame[header_size:end_public], frame[end_public:end_private]
    digest = hashlib.sha256(b'MLG-number-private-pair-v1\0' + frame[:end_private]).digest()
    require(frame[end_private:end_private + 32] == digest
            and struct.unpack('>8sH32s', ack) == (b'MLGNA001', ordinal, digest),
            'pair checksum or acknowledgment differs')
    require(type(receipt) is dict and set(receipt) == {'ordinal', 'frame_sha256', 'public_sha256'}
            and type(receipt['ordinal']) is int and receipt['ordinal'] == ordinal
            and receipt['frame_sha256'] == digest.hex()
            and receipt['public_sha256'] == hashlib.sha256(public).hexdigest(), 'pair receipt differs')
    return public, private, digest


def expected_events(*, site, role, token_count):
    """Build the declared serial operation order from protocol coordinates."""
    require(type(site) is int and site in range(1, 5) and type(token_count) is int
            and 1 <= token_count <= 32 and role in ROLES, 'invalid event coordinates')
    injecting = role in ('injected', 'null', 'self', 'upstream', 'downstream') or role.startswith('forward_')
    target = (int(role[-1]) if role.startswith(('forward_', 'reverse_')) else
              0 if role == 'upstream' else 1 if role in ('null', 'self') else None)
    steps = []
    for layer in range(6):
        attn, mlp = f'transformer.h.{layer}.attn', f'transformer.h.{layer}.mlp'
        steps.append((f'attention_{layer}', 'capture', attn, True, None))
        if injecting and layer == site:
            steps.extend((('inject_before', 'capture', mlp, False, None),
                          ('inject_add', 'add', mlp, True, None),
                          ('inject_after', 'capture', mlp, False, None)))
        if target == layer:
            if role == 'self':
                steps.append(('self_source', 'capture', mlp, True, None))
            kind = 'null' if role == 'null' else 'patch'
            steps.extend((('action_before', 'capture', mlp, False, None),
                          (f'action_{kind}', kind, mlp, True, 'self_source' if role == 'self' else None),
                          ('action_after', 'capture', mlp, False, None)))
        steps.append((f'mlp_{layer}', 'capture', mlp, True, None))
        if layer == 5:
            path = 'transformer.h.5'
            if role == 'downstream':
                steps.extend((('action_before', 'capture', path, False, None),
                              ('action_patch', 'patch', path, True, None),
                              ('action_after', 'capture', path, False, None)))
            steps.append(('block_5', 'capture', path, True, None))
    events, preceding, last_path = {}, [], None
    for ordinal, (name, operation, path, selected, source) in enumerate(steps):
        if path != last_path:
            preceding = []
        events[name] = {'name': name, 'operation': operation, 'order': ordinal, 'module_path': path,
                        'module_aliases': [path], 'call_site_known': False, 'forward_index': 0, 'invocation': 0,
                        'batch_index': 0 if selected else None, 'token_index': token_count - 1 if selected else None,
                        'output_selector': [0] if path.endswith('.attn') or path == 'transformer.h.5' else [],
                        'output_shape': [1, token_count, WIDTH], 'selected_shape': [WIDTH] if selected else [1, token_count, WIDTH],
                        'dtype': 'torch.float32', 'device': 'cpu', 'source_capture': source,
                        'preceding_same_site_actions': list(preceding)}
        preceding.append(name)
        last_path = path
    return events


def summarize_site(predicted, effects, reverse, *, numerator, denominator):
    """Recompute fixed prediction errors and whole-vocabulary restoration."""
    require(type(effects) is tuple and len(effects) == 8 and type(reverse) is tuple and len(reverse) == 8,
            'eight paired effects are required')
    values = (predicted, numerator, denominator, *effects, *reverse)
    require(all(type(value) in (int, float) and math.isfinite(value) for value in values)
            and numerator >= 0 and denominator >= 0, 'site summary contains invalid numbers')
    errors = [math.fsum((effect, -predicted)) for effect in effects]
    return {'predicted_effect': predicted, 'effects': list(effects), 'reverse_effects': list(reverse),
            'errors': errors, 'zero_errors': list(effects),
            'mae': math.fsum(abs(error) for error in errors) / 8,
            'rmse': math.sqrt(math.fsum(error * error for error in errors) / 8),
            'zero_mae': math.fsum(abs(effect) for effect in effects) / 8,
            'zero_rmse': math.sqrt(math.fsum(effect * effect for effect in effects) / 8),
            'families': [{'family_index': family, 'case_indices': [2 * family, 2 * family + 1],
                          'effects': list(effects[2 * family:2 * family + 2]),
                          'errors': errors[2 * family:2 * family + 2],
                          'zero_errors': list(effects[2 * family:2 * family + 2])}
                         for family in range(4)],
            'restoration': {'numerator': numerator, 'denominator': denominator,
                            'value': None if denominator <= 1e-8 else 1 - numerator / denominator}}


def _hex(value, length=64):
    return type(value) is str and re.fullmatch(f'[0-9a-f]{{{length}}}', value) is not None


def _read(path, limit, *, allow_empty=False):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400
            and (allow_empty or info.st_size > 0) and info.st_size <= limit,
            'saved evidence file is not bounded and ordinary')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) == info.st_size and len(raw) <= limit, 'saved evidence file changed')
    return raw


def verify_score_handoff(*, report, binding_bytes, evaluation_bytes, supervisor_bytes, stdout, stderr,
                         postscore_bytes, expected_supervisor_sha256, expected_postscore_sha256,
                         expected_command_sha256):
    """Check saved supervisor and score receipts against independent identities.

    These file checks do not themselves prove that the supervisor process
    returned or that its observations reflect the operating system.
    """
    require(type(report) is dict and all(_hex(report.get(key)) for key in
            ('binding_sha256', 'evaluation_sha256', 'receipts_sha256', 'last_frame_sha256'))
            and all(_hex(item) for item in (expected_supervisor_sha256, expected_postscore_sha256,
                                           expected_command_sha256))
            and type(stdout) is bytes and len(stdout) <= 1024 * 1024
            and type(stderr) is bytes and len(stderr) <= 1024 * 1024
            and hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding_bytes).hexdigest() == report['binding_sha256']
            and hashlib.sha256(evaluation_bytes).hexdigest() == report['evaluation_sha256']
            and hashlib.sha256(supervisor_bytes).hexdigest() == expected_supervisor_sha256
            and hashlib.sha256(postscore_bytes).hexdigest() == expected_postscore_sha256,
            'supervisor or score handoff identity differs')
    binding = parse_object(binding_bytes, 4096)
    postscore = parse_object(postscore_bytes, 65536)
    require(binding.get('disjoint_status') in ('pending', 'clear')
            and same(postscore, {'schema': 'mlg-injection-heldout-private-score-v1',
                'status': 'scored-private-awaiting-independent-audit',
                'binding_sha256': report['binding_sha256'], 'binding_file_sha256': hashlib.sha256(binding_bytes).hexdigest(),
                'supervisor_sha256': expected_supervisor_sha256, 'evaluation_sha256': report['evaluation_sha256'],
                'disjoint_status': binding['disjoint_status'],
                'source_qualification': 'prospective orchestration; same-full-source rule departed'}),
            'private post-score receipt differs')
    require(type(supervisor_bytes) is bytes and 0 < len(supervisor_bytes) <= 1024 * 1024,
            'supervisor receipt exceeds its bound')
    try:
        supervisor = json.loads(supervisor_bytes, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('supervisor receipt is malformed') from error
    require(type(supervisor) is dict and supervisor.get('status') == 'evidence_complete'
            and type(supervisor.get('exit_code')) is int and supervisor['exit_code'] == 0
            and type(supervisor.get('child_pid')) is int and 0 < supervisor['child_pid'] <= 0xffffffff
            and type(supervisor.get('postexit_active_processes')) is int and supervisor['postexit_active_processes'] == 0
            and all(supervisor.get(name) is True for name in ('child_reaped', 'job_drained', 'job_closed',
                                                               'logs_complete', 'cleanup_verified'))
            and all(supervisor.get(name) is False for name in ('retry_safe', 'cancellation_requested',
                                                                'quarantined', 'log_overflow'))
            and supervisor.get('supervisor_error') is None
            and not any(key in supervisor for key in ('reason', 'log_errors', 'audit_error_type'))
            and type(supervisor.get('command')) is list and supervisor['command']
            and all(type(part) is str and part for part in supervisor['command'])
            and hashlib.sha256(canonical(supervisor['command'])).hexdigest() == expected_command_sha256
            and supervisor.get('stdout_sha256') == hashlib.sha256(stdout).hexdigest()
            and supervisor.get('stderr_sha256') == hashlib.sha256(stderr).hexdigest()
            and same(supervisor.get('retained_tensor_files'), {'accounting_complete': True, 'total_bytes': 0, 'files': []}),
            'supervised owner cleanup or logs differ')
    structural = report.get('structural_audit')
    require(type(structural) is dict and set(structural) == {'status', 'forward_count', 'binding_sha256',
            'receipts_sha256', 'last_frame_sha256', 'inventory', 'scientific_validation'}
            and structural['status'] == 'evidence_complete' and type(structural['forward_count']) is int
            and structural['forward_count'] == 128 and structural['binding_sha256'] == report['binding_sha256']
            and structural['receipts_sha256'] == report['receipts_sha256']
            and structural['last_frame_sha256'] == report['last_frame_sha256']
            and structural['scientific_validation'] == 'pending independent audit', 'independent inventory identity differs')
    inventory = structural['inventory']
    require(type(inventory) is dict and set(inventory) == {'retained_tensor_bytes', 'tensor_encoding',
            'encoded_tensor_occurrence_bytes', 'journal_file_bytes', 'owner_side_file_bytes'}
            and type(inventory['retained_tensor_bytes']) is int and inventory['retained_tensor_bytes'] == 0
            and inventory['tensor_encoding'] == 'journal-frames'
            and type(inventory['encoded_tensor_occurrence_bytes']) is int
            and 0 < inventory['encoded_tensor_occurrence_bytes'] <= 72 * 1024 * 1024
            and type(inventory['journal_file_bytes']) is int and 0 < inventory['journal_file_bytes'] <= 192 * 1024 * 1024
            and type(inventory['owner_side_file_bytes']) is int and 0 < inventory['owner_side_file_bytes'] <= 128 * 512 + 12288
            and inventory['journal_file_bytes'] + inventory['owner_side_file_bytes'] <= 192 * 1024 * 1024,
            'independent inventory bounds differ')
    require(same(supervisor.get('audit'), structural),
            'supervisor audit or its independently measured inventory differs')


def _margin(raw, first=318, second=389):
    return math.fsum((float32(struct.unpack_from('<I', raw, first * 4)[0]),
                      -float32(struct.unpack_from('<I', raw, second * 4)[0])))


def _norm(left, right):
    require(len(left) == len(right) == WIDTH_BYTES, 'candidate vector size differs')
    return math.sqrt(math.fsum((float32(a) - float32(b)) ** 2
        for (a,), (b,) in zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right))))


def _inventory(role, observation):
    record = observation['record']
    return {'role': role, 'request_id': record['request_id'], 'input_sha256': record['input_sha256'],
            'condition': record['condition'],
            'attention_sha256': [hashlib.sha256(raw).hexdigest() for raw in observation['attention']],
            'mlp_sha256': [hashlib.sha256(raw).hexdigest() for raw in observation['mlp']],
            'block_sha256': hashlib.sha256(observation['block']).hexdigest(),
            'logits_sha256': hashlib.sha256(observation['logits']).hexdigest(),
            'present_margin': _margin(observation['logits']), 'past_margin': _margin(observation['logits'], 373, 547),
            'trace': record['trace'], 'action': record['action']}


def _case_score(index, text, ids, row):
    clean, injected = row['clean'], row['injected']
    first = next((site for site in range(1, 5) if not within(clean['mlp'][site], injected['mlp'][site])), None)
    clean_margin, injected_margin = _margin(clean['logits']), _margin(injected['logits'])
    forward = [_margin(row[f'forward_{site}']['logits']) for site in range(1, 5)]
    reverse = [_margin(row[f'reverse_{site}']['logits']) for site in range(1, 5)]
    return {'index': index, 'family_index': index // 2, 'number': 'singular' if index % 2 == 0 else 'plural',
            'text': text, 'input_ids': list(ids), 'first_changed': first,
            'l2': [_norm(clean['mlp'][site], injected['mlp'][site]) for site in range(1, 5)],
            'clean_margin': clean_margin, 'injected_margin': injected_margin,
            'clean_correct': bool(clean_margin > 0 if index % 2 == 0 else clean_margin < 0),
            'clean_past_margin': _margin(clean['logits'], 373, 547),
            'injected_past_margin': _margin(injected['logits'], 373, 547),
            'forward_margins': forward, 'reverse_margins': reverse,
            'forward_effects': [math.fsum((value, -injected_margin)) for value in forward],
            'reverse_effects': [math.fsum((value, -clean_margin)) for value in reverse],
            'observations': [_inventory(role, row[role]) for role in ROLES]}


def _squared(rows, role):
    for row in rows:
        for (a,), (b,) in zip(struct.iter_unpack('<I', row[role]['logits']),
                              struct.iter_unpack('<I', row['clean']['logits'])):
            yield (float32(a) - float32(b)) ** 2


def owner_side_size(started, terminal, receipts, binding):
    require(all(type(item) is bytes for item in (started, terminal, receipts, binding)), 'owner side files differ')
    return sum(len(item) for item in (started, terminal, receipts, binding))


def audit_saved(*, journal_dir, receipts_bytes, started_bytes, terminal_bytes, binding_bytes, split_bytes, split_nonce,
                site_bytes, site_nonce, freeze_bytes, prediction_bytes, evaluation_bytes,
                expected, externally_quiescent=False):
    """Recompute an externally quiescent, commitment-bound saved evaluation.

    Expected digests must come from the independent custodian's published and
    post-score bindings. This checks saved bytes, not publication, tokenization,
    actual execution, child exit, model integrity, or sampler disjointness.
    """
    keys = {'binding_sha256', 'heldout_commitment', 'site_commitment', 'prediction_sha256',
            'freeze_sha256', 'evaluation_sha256', 'model_state_sha256'}
    require(externally_quiescent is True and type(expected) is dict, 'independent bindings or quiescence are missing')
    expected = dict(expected)
    require(set(expected) == keys and all(_hex(value) for value in expected.values()),
            'independent bindings or quiescence are missing')
    require(type(site_nonce) is bytes and len(site_nonce) == 32 and type(split_nonce) is bytes
            and len(split_nonce) == 32 and site_nonce != split_nonce, 'private nonces differ')
    binding = parse_object(binding_bytes, 4096)
    require(set(binding) == {'schema', 'source_commit', 'runtime_sha256', 'prediction_sha256', 'freeze_sha256',
            'heldout_commitment', 'site_commitment', 'inventory_sha256', 'operation_sha256', 'gate_sha256',
            'disjoint_status', 'disjoint_receipt_sha256'}
            and binding['schema'] == 'mlg-injection-heldout-journal-binding-v1'
            and hashlib.sha256(b'MLG-injection-heldout-binding-v1\0' + binding_bytes).hexdigest() == expected['binding_sha256']
            and _hex(binding['source_commit'], 40)
            and all(_hex(binding[key]) for key in ('runtime_sha256', 'prediction_sha256', 'freeze_sha256',
                'heldout_commitment', 'site_commitment', 'inventory_sha256', 'operation_sha256', 'gate_sha256'))
            and all(binding[key] == expected[key] for key in ('heldout_commitment', 'site_commitment',
                                                              'prediction_sha256', 'freeze_sha256'))
            and ((binding['disjoint_status'] == 'pending' and binding['disjoint_receipt_sha256'] is None)
                 or (binding['disjoint_status'] == 'clear' and _hex(binding['disjoint_receipt_sha256']))),
            'held-out journal binding differs')
    site = parse_object(site_bytes, 4096)
    require(set(site) == {'schema', 'protocol_sha256', 'source_commit', 'checkpoint_revision', 'layer'}
            and site['schema'] == 'mlg-number-site-v1' and site['protocol_sha256'] == PROTOCOL_SHA256
            and site['source_commit'] == binding['source_commit'] and site['checkpoint_revision'] == CHECKPOINT_REVISION
            and type(site['layer']) is int
            and site['layer'] == 1 + (hashlib.sha256(b'MLG-number-v1-site\0' + site_nonce).digest()[0] >> 6)
            and hashlib.sha256(b'MLG-number-v1-site-commit\0' + site_nonce + b'\0' + site_bytes).hexdigest() == expected['site_commitment'],
            'private site commitment differs')
    split = parse_object(split_bytes, 1024 * 1024)
    require(set(split) == {'schema', 'protocol_sha256', 'source_commits', 'checkpoint_revision',
            'prediction_freeze_sha256', 'exposure_inventory_sha256', 'families'}
            and split['schema'] == 'mlg-number-heldout-v1' and split['protocol_sha256'] == PROTOCOL_SHA256
            and split['checkpoint_revision'] == CHECKPOINT_REVISION
            and same(split['source_commits'], {'injection': binding['source_commit']})
            and split['prediction_freeze_sha256'] == expected['freeze_sha256']
            and split['exposure_inventory_sha256'] == binding['inventory_sha256']
            and hashlib.sha256(b'MLG-number-v1-heldout-commit\0' + split_nonce + b'\0' + split_bytes).hexdigest() == expected['heldout_commitment']
            and type(split['families']) is list and len(split['families']) == 4, 'private split commitment differs')
    targets, all_ids = [], []
    for family_index, family in enumerate(split['families']):
        require(type(family) is dict and set(family) == {'family_index', 'context', 'subjects', 'cases'}
                and type(family['family_index']) is int and family['family_index'] == family_index
                and type(family['context']) is str and family['context']
                and type(family['subjects']) is list and len(family['subjects']) == 4
                and all(type(item) is str and item for item in family['subjects'])
                and type(family['cases']) is list and len(family['cases']) == 4, 'split family differs')
        alignment, subjects = None, set()
        for position, case in enumerate(family['cases']):
            require(type(case) is dict and set(case) == {'text', 'input_ids', 'attention_mask', 'subject_index'}
                    and type(case['text']) is str and 0 < len(case['text']) <= 4096
                    and type(case['input_ids']) is list and 3 <= len(case['input_ids']) <= 32
                    and all(type(token) is int and 0 <= token < 50257 for token in case['input_ids'])
                    and type(case['subject_index']) is int and 0 < case['subject_index'] < len(case['input_ids']) - 1
                    and same(case['attention_mask'], [1] * len(case['input_ids'])), 'split row differs')
            ids = tuple(case['input_ids'])
            subject_index = case['subject_index']
            coordinates = (len(ids), subject_index, ids[:subject_index] + ids[subject_index + 1:])
            if alignment is None:
                alignment = coordinates
            require(alignment == coordinates and ids[subject_index] not in subjects, 'split family alignment differs')
            subjects.add(ids[subject_index])
            require(not any(ids[:min(len(ids), len(other))] == other[:min(len(ids), len(other))]
                            for other in all_ids), 'split rows overlap')
            all_ids.append(ids)
            if position < 2:
                targets.append((case['text'], ids))
    freeze = parse_object(freeze_bytes, 2 * 1024 * 1024)
    prediction = parse_object(prediction_bytes, 2 * 1024 * 1024)
    evaluation = parse_object(evaluation_bytes, 4 * 1024 * 1024)
    require(hashlib.sha256(freeze_bytes).hexdigest() == expected['freeze_sha256']
            and hashlib.sha256(prediction_bytes).hexdigest() == expected['prediction_sha256']
            and hashlib.sha256(evaluation_bytes).hexdigest() == expected['evaluation_sha256']
            and same(freeze, {'schema': 'mlg-number-prediction-freezes-v1', 'protocol_sha256': PROTOCOL_SHA256,
                'checkpoint_revision': CHECKPOINT_REVISION, 'tracks': {'injection': {
                    'source_commit': binding['source_commit'], 'runtime_sha256': binding['runtime_sha256'],
                    'prediction_sha256': expected['prediction_sha256']}}}), 'frozen prediction binding differs')
    require(prediction.get('schema') == 'mlg-number-injection-discovery-prediction-v1'
            and prediction.get('status') == 'complete' and prediction.get('protocol_sha256') == PROTOCOL_SHA256
            and prediction.get('checkpoint_revision') == CHECKPOINT_REVISION
            and prediction.get('execution_device') == 'cpu', 'frozen prediction identity differs')
    require(type(receipts_bytes) is bytes and 0 < len(receipts_bytes) <= 128 * 512 and receipts_bytes.endswith(b'\n'),
            'receipt inventory is incomplete')
    lines = receipts_bytes.splitlines()
    require(len(lines) == 128, 'receipt inventory is incomplete')
    receipts = [parse_object(line, 512) for line in lines]
    started = parse_object(started_bytes, 4096)
    terminal = parse_object(terminal_bytes, 4096)
    require(same(started, {'schema': 'mlg-injection-heldout-owner-started-v1',
                'binding_sha256': expected['binding_sha256'], 'heldout_commitment': expected['heldout_commitment'],
                'forward_count_planned': 128})
            and set(terminal) == {'schema', 'status', 'forward_count', 'binding_sha256', 'receipts_sha256',
                'last_frame_sha256', 'model_state_sha256', 'retained_tensor_bytes', 'scientific_validation'}
            and terminal['schema'] == 'mlg-injection-heldout-owner-terminal-v1'
            and terminal['status'] == 'owner-produced' and type(terminal['forward_count']) is int
            and terminal['forward_count'] == 128 and terminal['binding_sha256'] == expected['binding_sha256']
            and terminal['receipts_sha256'] == hashlib.sha256(receipts_bytes).hexdigest()
            and terminal['model_state_sha256'] == expected['model_state_sha256']
            and type(terminal['retained_tensor_bytes']) is int
            and 0 <= terminal['retained_tensor_bytes'] <= 72 * 1024 * 1024
            and terminal['scientific_validation'] == 'pending independent audit', 'owner terminal differs')
    path = Path(journal_dir)
    require(path.is_absolute() and path.is_dir() and not path.is_symlink()
            and not getattr(path.lstat(), 'st_file_attributes', 0) & 0x400, 'journal directory differs')
    names = {f'{index:03d}.{suffix}' for index in range(128) for suffix in ('frame', 'ok')}
    found = set()
    for item in path.iterdir():
        require(len(found) < len(names) and item.name in names, 'journal file inventory differs')
        found.add(item.name)
    require(found == names, 'journal file inventory differs')
    schedule = []
    for family in range(4):
        first, second = 2 * family, 2 * family + 1
        schedule.extend((index, role) for index in (first, second) for role in ROLES[:-1])
        schedule.extend(((first, 'verify_clean'), (second, 'verify_clean')))
    previous, observed, baselines, disk_bytes, tensor_bytes = bytes(32), {}, {}, 0, 0
    for ordinal, (index, role) in enumerate(schedule):
        frame = _read(path / f'{ordinal:03d}.frame', 1024 * 1024 + 1024)
        ack = _read(path / f'{ordinal:03d}.ok', 128)
        disk_bytes += len(frame) + len(ack)
        require(disk_bytes <= 192 * 1024 * 1024, 'journal exceeds its disk allowance')
        public, private, previous = read_pair(frame, ack, ordinal=ordinal,
            binding=bytes.fromhex(expected['binding_sha256']), previous=previous, receipt=receipts[ordinal])
        ids = targets[index][1]
        observation = parse_public(public, index=index, role=role, ids=ids, commitment=expected['heldout_commitment'])
        clean, injected = baselines.get(index, (None, None))
        tensor_bytes += sum(map(len, observation['attention'] + observation['mlp']
                               + [observation['block'], observation['logits']]))
        tensor_bytes += verify_witness(private, index=index, role=role, ids=ids, site=site['layer'],
                                       observation=observation, clean=clean, injected=injected)
        require(tensor_bytes <= 72 * 1024 * 1024, 'saved tensor occurrence allowance is exceeded')
        if role == 'clean':
            baselines[index] = (observation, None)
        elif role == 'injected':
            baselines[index] = (clean, observation)
        elif role == 'verify_clean':
            del baselines[index]
        observed[index, role] = observation
    owner_side_bytes = owner_side_size(started_bytes, terminal_bytes, receipts_bytes, binding_bytes)
    require(not baselines and len(observed) == 128 and terminal['last_frame_sha256'] == previous.hex()
            and terminal['retained_tensor_bytes'] <= tensor_bytes and disk_bytes + owner_side_bytes <= 192 * 1024 * 1024,
            'family-final controls or owner inventory differ')
    structural = {'status': 'evidence_complete', 'forward_count': 128, 'binding_sha256': expected['binding_sha256'],
        'receipts_sha256': terminal['receipts_sha256'], 'last_frame_sha256': previous.hex(),
        'inventory': {'retained_tensor_bytes': 0, 'tensor_encoding': 'journal-frames',
            'encoded_tensor_occurrence_bytes': tensor_bytes, 'journal_file_bytes': disk_bytes,
            'owner_side_file_bytes': owner_side_bytes}, 'scientific_validation': 'pending independent audit'}
    rows = [{role: observed[index, role] for role in ROLES} for index in range(8)]
    case_scores = [_case_score(index, text, ids, rows[index]) for index, (text, ids) in enumerate(targets)]
    selection_keys = ('first_changed', 'ranking', 'votes', 'earliest', 'random', 'magnitude')
    require(all(key in prediction for key in selection_keys) and type(prediction.get('predictions')) is dict
            and set(prediction['predictions']) == {'1', '2', '3', '4'}, 'frozen selections differ')
    selection = {key: prediction[key] for key in selection_keys}
    accuracy = {'correct': sum(case['clean_correct'] for case in case_scores),
                'incorrect': sum(not case['clean_correct'] for case in case_scores),
                'ties': sum(case['clean_margin'] == 0 for case in case_scores)}
    denominator = math.fsum(_squared(rows, 'injected'))
    sites = {}
    for site_number in range(1, 5):
        effects = tuple(case['forward_effects'][site_number - 1] for case in case_scores)
        reverse = tuple(case['reverse_effects'][site_number - 1] for case in case_scores)
        numerator = math.fsum(_squared(rows, f'forward_{site_number}'))
        sites[str(site_number)] = summarize_site(prediction['predictions'][str(site_number)], effects, reverse,
                                                  numerator=numerator, denominator=denominator)
    expected_evaluation = {'schema': 'mlg-number-injection-evaluation-v1', 'status': 'complete',
        'protocol_sha256': PROTOCOL_SHA256, 'checkpoint_revision': CHECKPOINT_REVISION, 'execution_device': 'cpu',
        'private_controls': 'caller-attested', 'custody': 'caller-attested',
        'prediction_sha256': expected['prediction_sha256'], 'prediction_freeze_sha256': expected['freeze_sha256'],
        'heldout_commitment_sha256': expected['heldout_commitment'], 'selection': selection,
        'clean_accuracy': accuracy, 'sites': sites, 'cases': case_scores}
    require(same(evaluation, expected_evaluation), 'saved evaluation differs from independent recomputation')
    return {'schema': 'mlg-independent-injection-heldout-audit-v1',
            'status': 'independent_saved_evidence_pass', 'records': 128,
            'binding_sha256': expected['binding_sha256'], 'heldout_commitment': expected['heldout_commitment'],
            'site_commitment': expected['site_commitment'], 'freeze_sha256': expected['freeze_sha256'],
            'prediction_sha256': expected['prediction_sha256'], 'evaluation_sha256': expected['evaluation_sha256'],
            'receipts_sha256': hashlib.sha256(receipts_bytes).hexdigest(), 'last_frame_sha256': previous.hex(),
            'structural_audit': structural, 'source_deviation': True, 'selection': selection,
            'clean_accuracy': accuracy, 'sites': sites,
            'limitations': ['Saved bytes only; execution, publication chronology, custody and tokenization are external.',
                            'The frozen discovery prediction is digest-bound; its discovery archive is audited separately.']}


def write_private_receipt(path, result):
    """Create one bounded receipt; a failed or partial write must be retained."""
    path = Path(path)
    require(path.is_absolute() and path.name == 'independent-audit.json' and path.parent.is_dir()
            and not path.parent.is_symlink() and not getattr(path.parent.lstat(), 'st_file_attributes', 0) & 0x400,
            'private receipt path differs')
    require(type(result) is dict, 'private receipt must be an object')
    raw = canonical(result)
    require(0 < len(raw) <= 1024 * 1024, 'private receipt exceeds its allowance')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(path, flags, 0o600)
    primary = None
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(fd, remaining)
            if type(written) is not int or not 0 < written <= len(remaining):
                raise OSError('private receipt write was incomplete')
            remaining = remaining[written:]
        os.fsync(fd)
    except BaseException as error:
        primary = error
    try:
        os.close(fd)
    except BaseException as cleanup:
        if primary is not None:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(primary, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
    if primary is not None:
        raise primary
    return hashlib.sha256(raw).hexdigest()
