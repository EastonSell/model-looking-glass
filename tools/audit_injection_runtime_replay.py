"""Independent saved-evidence check for the fixed public runtime replay.

This module neither runs a model nor establishes publication, process exit,
runtime provenance, quiescence, custody, or permission to use a result. The
site, witnesses, per-role comparisons and detailed errors remain private.
"""
import hashlib
import os
from pathlib import Path
import stat

from tools import audit_injection_heldout as independent


SOURCE = 'e84f66ae63b6f20e6df81789f2e4741b13598fb1'
PLAN_SHA256 = 'e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8'
DISCOVERY_RUNTIME_SHA256 = 'b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e'
SITE_COMMITMENT = 'c3739cb173d3c1bc3e3adda70025ff7f26a73c4690e50ecf5baf06ff8b359b9d'
IDS = (464, 3797, 1474, 262, 9168)
REFERENCES = (
    '4d0815c9368397ffebb4257065e0fac1f5c7110b21bc4bbf53511199fb545fc2',
    '68f03dc86e14e653f9850aa241f4410fff121ec80d8c303987ed2f2966d795dc',
    '0a18dda10126cc532dbbe46712084a4013fabb00414b81141905fd1ae003dccc',
    '6350d70b8847ebf842169895c3a53ce6f469158ff7f4b44f405ce3ff3f83a8fc',
    '8e26558abae3fb1eac9a773b823bab94d9413cc9cec687941dbc16f23f9ae44d',
    'e67cb76ed14a693b8fe88bfcb39336740df0dd90fce6097e6c4aeb6b4301aa02',
    'b4955d40391ca411eea988f9ea44cb738b8b13f4a3ecb93db6e110064082be44',
    '4d3ab110bf6706194ced5ae67ac4f78845b78a84129065dce692f61fe1a2a79b',
    '08cb49bf4909dc6ea6f6b16d1786859fcbdb980fe162f0930b260f09ade0006d',
    'ef5bcaf33eb22f1b48c250e2643edc50b6599a097f9af0c8a3370abdc33a3c91',
    'af7a11a6ab88d745ffd17248c68c0e673ca183e67374f21b5fac3da1209c6c8e',
    'ecf6e7bd0c83aa6efc6b8a0223e891385423943977876362cc6621d3fb458c33',
    '4c327a552512bb5b3f4a39f6be0fda0672318f44fb327eb17c83ff4f3474940e',
    '9c5efeb0b2657f4eabc86718984b48b785c12170a5cc9060fe3390acad99e0a2',
    '958db3742fc9a7d9a06d9fe367290d088ccb76014abc0f9c789f12e40dad3719',
    'bd9b06b285d277bbce16f3058a8be1d6c1d9aa63e13bab29259e567cbcfbe58c',
)
MIB = 1024 * 1024


def _require(value, message):
    independent.require(value, message)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _ordinary(path, *, directory=False):
    info = path.lstat()
    _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400
             and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
             and (directory or info.st_nlink == 1), 'saved replay path is not ordinary')
    return info


def _inventory(path, names):
    found = {}
    with os.scandir(path) as entries:
        for entry in entries:
            _require(entry.name in names and entry.name not in found and len(found) < len(names),
                     'saved replay inventory differs')
            found[entry.name] = _ordinary(path / entry.name, directory=names[entry.name])
    _require(set(found) == set(names), 'saved replay inventory is incomplete')
    return found


def _read(path, limit, before):
    _require(0 < before.st_size <= limit, 'saved replay file exceeds its bound')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        finished = os.fstat(stream.fileno())
    after = _ordinary(path)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink)
    _require(len(raw) == before.st_size and len(raw) <= limit
             and all(identity(item) == identity(before) for item in (opened, finished, after)),
             'saved replay file changed while reading')
    return raw


def _plan(raw, site_nonce):
    _require(type(raw) is bytes and 0 < len(raw) <= 128 * 1024 and _digest(raw) == PLAN_SHA256,
             'published request plan differs')
    plan = independent.parse_object(raw, 128 * 1024)
    _require(set(plan) == {'schema', 'protocol_sha256', 'checkpoint_revision', 'source_commit',
                          'public_entropy', 'requests'}
             and plan['schema'] == 'mlg-number-injection-family-request-plan-v2'
             and plan['protocol_sha256'] == independent.PROTOCOL_SHA256
             and plan['checkpoint_revision'] == independent.CHECKPOINT_REVISION
             and plan['source_commit'] == SOURCE and independent._hex(plan['public_entropy'])
             and type(plan['requests']) is list and len(plan['requests']) == 128,
             'published request plan identity differs')
    entropy = bytes.fromhex(plan['public_entropy'])
    _require(type(site_nonce) is bytes and len(site_nonce) == 32 and site_nonce != entropy,
             'private site nonce differs')
    selected = [item for item in plan['requests'] if type(item) is dict and item.get('case_index') == 0]
    _require(len(selected) == 16, 'case-0 request inventory differs')
    requests = {}
    for item in selected:
        _require(set(item) == {'case_index', 'role', 'request_id', 'input_sha256'}
                 and type(item['case_index']) is int and type(item['role']) is str
                 and item['role'] in independent.ROLES
                 and item['role'] not in requests, 'case-0 request identity differs')
        ordinal = independent.ROLES.index(item['role'])
        expected = hashlib.sha256(b'MLG-number-injection-request-v1\0' + entropy
                                  + ordinal.to_bytes(4, 'big')).digest()[:16].hex()
        _require(item['request_id'] == expected and item['input_sha256'] == independent.input_hash(IDS),
                 'case-0 request derivation differs')
        requests[item['role']] = expected
    _require(set(requests) == set(independent.ROLES), 'case-0 requests are incomplete')
    return requests


def _site(raw, nonce):
    record = independent.parse_object(raw, 4096)
    _require(set(record) == {'schema', 'protocol_sha256', 'source_commit', 'checkpoint_revision', 'layer'}
             and record['schema'] == 'mlg-number-site-v1'
             and record['protocol_sha256'] == independent.PROTOCOL_SHA256
             and record['source_commit'] == SOURCE and record['checkpoint_revision'] == independent.CHECKPOINT_REVISION
             and type(record['layer']) is int
             and record['layer'] == 1 + (hashlib.sha256(b'MLG-number-v1-site\0' + nonce).digest()[0] >> 6)
             and _digest(b'MLG-number-v1-site-commit\0' + nonce + b'\0' + raw) == SITE_COMMITMENT,
             'private site commitment differs')
    return record['layer']


def _public(raw, role, request):
    value = independent.parse_object(raw, 512 * 1024)
    _require(set(value) == {'schema', 'case_index', 'role', 'request_id', 'input_sha256', 'condition',
                           'trace', 'action', 'payload'}
             and value['schema'] == 'mlg-number-injection-raw-observation-v1'
             and type(value['case_index']) is int and value['case_index'] == 0
             and value['role'] == role and value['condition'] == independent.CONDITIONS[role]
             and value['request_id'] == request and value['input_sha256'] == independent.input_hash(IDS),
             'public replay observation identity differs')
    paths = [f'transformer.h.{layer}.{kind}' for layer in range(6) for kind in ('attn', 'mlp')] + ['transformer.h.5']
    expected = [{'module_path': path, 'capture_order': order, 'invocation': 0, 'batch_index': 0,
                 'token_index': len(IDS) - 1, 'output_selector': [0] if path.endswith('.attn') or path == 'transformer.h.5' else [],
                 'output_shape': [1, len(IDS), independent.WIDTH], 'selected_shape': [independent.WIDTH],
                 'dtype': 'torch.float32', 'device': 'cpu'} for order, path in enumerate(paths)]
    _require(independent.same(value['trace'], expected), 'public replay capture metadata differs')
    payload = value['payload']
    _require(type(payload) is dict and set(payload) == {'attention', 'mlp', 'block', 'logits'}
             and type(payload['attention']) is list and len(payload['attention']) == 6
             and type(payload['mlp']) is list and len(payload['mlp']) == 6, 'public replay payload differs')
    record = {key: item for key, item in value.items() if key != 'payload'}
    return {'record': record,
            'attention': [independent.tensor(item, independent.WIDTH_BYTES) for item in payload['attention']],
            'mlp': [independent.tensor(item, independent.WIDTH_BYTES) for item in payload['mlp']],
            'block': independent.tensor(payload['block'], independent.WIDTH_BYTES),
            'logits': independent.tensor(payload['logits'], 50257 * 4)}


def audit_saved_replay(*, attempt_dir, plan_bytes, site_nonce, site_record_bytes, references,
                       actual_runtime_sha256, expected_model_state_sha256, externally_quiescent=False):
    """Verify one saved replay. The returned record alone is not a public pass."""
    independent.require_nearest_binary64()
    _require(externally_quiescent is True and independent._hex(actual_runtime_sha256)
             and independent._hex(expected_model_state_sha256), 'external runtime, model or quiescence is missing')
    _require(type(references) is tuple and len(references) == 16
             and all(type(raw) is bytes and 0 < len(raw) <= 512 * 1024 and _digest(raw) == digest
                     for raw, digest in zip(references, REFERENCES)),
             'published reference bytes differ')
    requests = _plan(plan_bytes, site_nonce)
    site = _site(site_record_bytes, site_nonce)
    binding = independent.canonical({'schema': 'mlg-number-private-journal-binding-v1',
        'source_commit': SOURCE, 'plan_sha256': PLAN_SHA256, 'runtime_sha256': actual_runtime_sha256,
        'protocol_sha256': independent.PROTOCOL_SHA256, 'checkpoint_revision': independent.CHECKPOINT_REVISION,
        'site_commitment': SITE_COMMITMENT})
    binding_hash = hashlib.sha256(b'MLG-number-journal-binding-v1\0' + binding).digest()
    root = Path(attempt_dir)
    _require(root.is_absolute(), 'saved replay path must be absolute')
    _ordinary(root, directory=True)
    side = _inventory(root, {'journal': True, 'receipts.jsonl': False, 'owner-terminal.json': False})
    journal = root / 'journal'
    files = _inventory(journal, {f'{i:03d}.{suffix}': False for i in range(16) for suffix in ('frame', 'ok')})
    disk = side['receipts.jsonl'].st_size + side['owner-terminal.json'].st_size + sum(x.st_size for x in files.values())
    _require(disk <= 16 * MIB, 'saved replay physical allowance exceeded')
    receipts_raw = _read(root / 'receipts.jsonl', 16384, side['receipts.jsonl'])
    terminal_raw = _read(root / 'owner-terminal.json', 65536, side['owner-terminal.json'])
    _require(receipts_raw.endswith(b'\n') and len(receipts_raw.splitlines()) == 16, 'replay receipts are incomplete')
    receipts = [independent.parse_object(line, 512) for line in receipts_raw.splitlines()]
    terminal = independent.parse_object(terminal_raw, 65536)
    expected_terminal = {'schema': 'mlg-injection-runtime-replay-owner-v1',
        'status': 'replay-produced-private-awaiting-independent-audit',
        'discovery_runtime_sha256': DISCOVERY_RUNTIME_SHA256, 'heldout_runtime_sha256': actual_runtime_sha256,
        'model_state_sha256': expected_model_state_sha256, 'forward_count': 16,
        'logical_limit_bytes': 8 * MIB, 'physical_limit_bytes': 16 * MIB,
        'comparisons': terminal.get('comparisons')}
    _require(independent.same(terminal, expected_terminal), 'saved replay terminal differs')
    clean, injected, previous, payload_bytes, comparisons = None, None, bytes(32), 0, []
    for ordinal, role in enumerate(independent.ROLES):
        frame_name, ack_name = f'{ordinal:03d}.frame', f'{ordinal:03d}.ok'
        frame = _read(journal / frame_name, MIB + 1024, files[frame_name])
        ack = _read(journal / ack_name, 128, files[ack_name])
        public, private, previous = independent.read_pair(frame, ack, ordinal=ordinal, binding=binding_hash,
                                                           previous=previous, receipt=receipts[ordinal])
        payload_bytes += len(public) + len(private)
        _require(payload_bytes <= 8 * MIB, 'saved replay logical allowance exceeded')
        actual = _public(public, role, requests[role])
        reference = _public(references[ordinal], role, requests[role])
        _require(independent.same(actual['record'], reference['record']), 'replay metadata differs')
        left = actual['attention'] + actual['mlp'] + [actual['block'], actual['logits']]
        right = reference['attention'] + reference['mlp'] + [reference['block'], reference['logits']]
        comparison = {'within_tolerance': all(independent.within(a, b) for a, b in zip(left, right)),
                      'bit_equal': all(a == b for a, b in zip(left, right))}
        _require(comparison['within_tolerance'], 'replay comparison differs')
        comparisons.append(comparison)
        independent.verify_witness(private, index=0, role=role, ids=IDS, site=site,
                                   observation=actual, clean=clean, injected=injected)
        if role == 'clean':
            clean = actual
        elif role == 'injected':
            injected = actual
    _require(independent.same(comparisons, terminal['comparisons']), 'saved replay comparisons differ')
    independent.require_nearest_binary64()
    return {'schema': 'mlg-injection-runtime-independent-saved-audit-v1',
            'status': 'saved-evidence-consistent', 'forward_count': 16,
            'plan_sha256': PLAN_SHA256, 'discovery_runtime_sha256': DISCOVERY_RUNTIME_SHA256,
            'actual_runtime_sha256': actual_runtime_sha256, 'model_state_sha256': expected_model_state_sha256,
            'site_commitment': SITE_COMMITMENT, 'receipts_sha256': _digest(receipts_raw),
            'owner_terminal_sha256': _digest(terminal_raw), 'last_frame_sha256': previous.hex(),
            'payload_upper_bound_bytes': payload_bytes, 'physical_bytes': disk,
            'limitations': ['Process exit, custody, runtime provenance, publication and quiescence are external.']}
