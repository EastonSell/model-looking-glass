"""Independent, model-free replay of the fixed FLAN-T5 weight-edit study.

The reader consumes closed records only. It never imports the study or model.
Saved process, topology, restoration and publication records are cooperative
observations, not independent attestation of execution or external custody.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import threading
import unicodedata

VOCAB = 32128
WIDTH = 512
EDITED = 1627
POSITIVE = 1465
NEGATIVE = 2841
EOS = 1
DOSES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
CONTROLS = ('broad', 'random', 'opposite')
SHARED = ('primary', 'same', 'random', 'opposite')
CASE_SHA = 'a118dde671351d21c519b6a61957e9478c9501bac52d55aa34e8394772e933d5'
PROTOCOL_SHA = 'c06fd598e6b5d9cf8d9734c26f6711fbf53b0fbfda08f009ab691ee91dca5478'
ASSET_SHA = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
MODEL_SHA = '495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749'
TOKENIZER_SHA = 'fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a'
MANIFEST_PATH = 'docs/FLAN_T5_WEIGHT_EDIT_EXECUTION_MANIFEST_V2.json'
MANIFEST_REVISION = '60691c687cc737c081a6f2f27df3b51bcc4d5c87'
SOURCE_REVISION = 'b3c7e19a5349cd2fb76ef4908407342233c7e240'
MANIFEST_SHA = '09623ddf8716a1467cc35ef0469f7bb6d2119fdf574fd35aafe13f55e9fa669b'
PUBLIC_INPUTS = {'docs/FLAN_T5_TARGETED_WEIGHT_EDIT_V1.md': PROTOCOL_SHA,
                 'docs/FLAN_T5_WEIGHT_EDIT_CASES_V1.json': CASE_SHA,
                 'docs/FLAN_T5_WEIGHT_EDIT_PUBLICATION_V1.md': 'f29d98e1b436298f4867b1f722fb41f184cb761b97b2914139f23bf3d910a660',
                 'experiments/flan_t5_known_injection_cases_v1.json': '1b21870b8d2c83a7c144ecbf2911728fcdc14a828bf2e8078fca43780df64c11',
                 'docs/FLAN_T5_WEIGHT_EDIT_STORAGE_V1.md': 'b61d09688242e322e525adc74b7e9336b33f471fef20a883a724200805a0815c',
                 'docs/FLAN_T5_WEIGHT_EDIT_STORAGE_AMENDMENT_V1.md': '7314cd1950cab87382b3b530ca067c29e878cbd2273ae6d8181b10d58c024155',
                 'docs/FLAN_T5_WEIGHT_EDIT_RESOURCE_SUCCESSOR_V2.md': '210716fe1c2ce99338cafab47ef3d1789e75e4e67f87be71909821c4f28ca3ea',
                 'results/flan-t5-weight-edit-cpu-v1-20261029/development-failure/README.md': '8b62c0adc2f80eb8a7a321eb34c8b076d9613f9add5a0b69eec1360ad0bebc83',
                 'results/flan-t5-weight-edit-cpu-v1-20261029/development-failure/independent-process-audit-v1.json': 'efc379b31781c36775aa5eb6588f0c93d52a79f613108b4c00ac101c01c466cf'}
MAX_EVENTS_BYTES = 64 * 1024**2
MAX_LOGIT_BYTES = 256 * 1024**2
C_FREE_FLOOR = 1207959552
E_FREE_FLOOR = 2416967680
STUDY_ROOT = Path(r'E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2')


class AuditError(ValueError):
    """Saved evidence does not establish the required complete record."""


class DirectionIneligible(AuditError):
    """The fixed checkpoint yields a zero or nonfinite edit direction."""


class CaseAlignmentError(AuditError):
    """The verified tokenizer disagrees with a fixed case."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise AuditError('nonfinite JSON value: ' + value)
    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), 'nonfinite JSON number')
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def load_json(path, limit=32 * 1024**2):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= limit, 'missing or oversized JSON file')
    return strict_json(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def exact(left, right):
    return canonical(left) == canonical(right)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def near(left, right, *, relative=2e-12, absolute=2e-13):
    return number(left) and number(right) and math.isclose(left, right, rel_tol=relative, abs_tol=absolute)


def hex_digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def numeric_mode():
    # Both decoding and encoding matter: an imported native library can change
    # the process's floating-point control state after earlier checks.
    require(struct.unpack('<f', b'\x01\x00\x00\x00')[0] == math.ldexp(1.0, -149)
            and struct.pack('<f', math.ldexp(1.0, -149)) == b'\x01\x00\x00\x00'
            and struct.unpack('<f', struct.pack('<f', 1.0 + 2**-24))[0] == 1.0
            and struct.unpack('<f', struct.pack('<f', 1.0 + 3 * 2**-24))[0] == 1.0 + 2**-22,
            'float32 decode, encode or rounding mode differs')


def valid_cases(bundle):
    require(type(bundle) is dict and bundle.get('schema') == 'flan-t5-input-row-weight-change-cases-v1'
            and bundle.get('protocol_sha256') == PROTOCOL_SHA
            and bundle.get('asset_manifest_sha256') == ASSET_SHA, 'case bundle identity differs')
    cases = bundle.get('cases')
    ids = ([f'{phase}-{i}-{side}' for phase in ('discovery', 'calibration')
            for i in (1, 2) for side in ('target', 'donor')]
           + [f'evaluation-{i}-{side}' for i in range(1, 5) for side in ('target', 'donor')]
           + [f'locality-{i}' for i in range(1, 9)])
    require(type(cases) is list and [c.get('id') for c in cases] == ids, 'case order differs')
    for case in cases:
        tokens, mask = case.get('input_ids'), case.get('attention_mask')
        expected = case.get('expected_ids')
        require(type(tokens) is list and 1 <= len(tokens) <= 64 and tokens[-1] == EOS
                and all(type(x) is int and 0 <= x < VOCAB for x in tokens)
                and EOS not in tokens[:-1] and mask == [1] * len(tokens)
                and type(expected) is list and 2 <= len(expected) <= 4 and expected[-1] == EOS
                and all(type(x) is int and 0 <= x < VOCAB for x in expected)
                and type(case.get('expected')) is str, 'case input or expected answer differs')
    for target, donor in zip(cases[0:16:2], cases[1:16:2]):
        index = target.get('adjective_index')
        require(type(index) is int and index == donor.get('adjective_index')
                and len(target['input_ids']) == len(donor['input_ids'])
                and target['attention_mask'] == donor['attention_mask']
                and target['input_ids'][index] == EDITED and donor['input_ids'][index] == 16242
                and target['input_ids'].count(EDITED) == 1 and donor['input_ids'].count(16242) == 1
                and all(a == b for i, (a, b) in enumerate(zip(target['input_ids'], donor['input_ids']))
                        if i != index), 'paired target/donor tokens differ')
        for case, adjective in ((target, 'wonderful'), (donor, 'awful')):
            text, offsets, special = case.get('text'), case.get('offset_mapping'), case.get('special_tokens_mask')
            require(type(text) is str and text.count(adjective) == 1
                    and type(offsets) is list and len(offsets) == len(case['input_ids'])
                    and type(special) is list and len(special) == len(case['input_ids'])
                    and special[index] == 0 and type(offsets[index]) is list and len(offsets[index]) == 2,
                    'adjective token alignment is missing')
            start = text.index(adjective)
            left, right = offsets[index]
            require(type(left) is int and type(right) is int and right == start + len(adjective)
                    and (left == start or (left == start - 1 and left >= 0 and text[left] == ' ')),
                    'adjective offsets do not cover the complete word')
    require([EDITED in c['input_ids'] for c in cases[16:]] == [False] * 4 + [True, True, False, False],
            'locality token-presence partition differs')
    return cases


def checkpoint_rows(directory):
    directory = Path(directory)
    manifest_path = directory / 'asset-manifest.json'
    require(file_sha(manifest_path) == ASSET_SHA, 'asset manifest hash differs')
    manifest = load_json(manifest_path)
    names = {'config.json', 'generation_config.json', 'model.safetensors', 'special_tokens_map.json',
             'spiece.model', 'tokenizer.json', 'tokenizer_config.json'}
    require(manifest.get('revision') == '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
            and set(manifest.get('files', {})) == names, 'checkpoint inventory differs')
    for name in sorted(names):
        path, item = directory / name, manifest['files'][name]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
                and file_sha(path) == item['sha256'], 'checkpoint asset differs: ' + name)
    require(manifest['files']['model.safetensors']['sha256'] == MODEL_SHA
            and manifest['files']['tokenizer.json']['sha256'] == TOKENIZER_SHA, 'fixed asset hash differs')
    model = directory / 'model.safetensors'
    with model.open('rb') as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, 'short safetensors header')
        length = struct.unpack('<Q', prefix)[0]
        require(0 < length <= 8 * 1024**2 and 8 + length <= model.stat().st_size,
                'invalid safetensors header length')
        header = strict_json(stream.read(length))
        data_start = 8 + length
        entries = {}
        for name in ('shared.weight', 'lm_head.weight'):
            item = header.get(name, {})
            bounds = item.get('data_offsets')
            require(item.get('dtype') == 'F32' and item.get('shape') == [VOCAB, WIDTH]
                    and type(bounds) is list and len(bounds) == 2
                    and all(type(x) is int and x >= 0 for x in bounds)
                    and bounds[1] - bounds[0] == VOCAB * WIDTH * 4
                    and data_start + bounds[1] <= model.stat().st_size, 'checkpoint tensor layout differs')
            entries[name] = bounds
        a, b = entries['shared.weight'], entries['lm_head.weight']
        require(max(a[0], b[0]) >= min(a[1], b[1]), 'checkpoint tensor storage overlaps')
        rows = {}
        for label, name, token in (('wonderful', 'shared.weight', EDITED),
                                   ('awful', 'shared.weight', 16242),
                                   ('excellent', 'shared.weight', 1287),
                                   ('negative', 'lm_head.weight', NEGATIVE),
                                   ('positive', 'lm_head.weight', POSITIVE)):
            stream.seek(data_start + entries[name][0] + token * WIDTH * 4)
            raw = stream.read(WIDTH * 4)
            require(len(raw) == WIDTH * 4, 'checkpoint row is short')
            values = struct.unpack('<' + str(WIDTH) + 'f', raw)
            require(all(math.isfinite(x) for x in values), 'checkpoint row is nonfinite')
            rows[label] = (raw, values)
    return rows


def norm(values):
    try:
        result = math.sqrt(math.fsum(float(x) * float(x) for x in values))
    except (ValueError, OverflowError) as error:
        raise DirectionIneligible('direction is zero or nonfinite') from error
    if not math.isfinite(result) or result <= 0:
        raise DirectionIneligible('direction is zero or nonfinite')
    return result


def reconstruct_candidates(rows):
    require(sys.version_info[:2] == (3, 11), 'candidate replay requires Python 3.11')
    numeric_mode()
    wonderful, awful, excellent = (rows[x][1] for x in ('wonderful', 'awful', 'excellent'))
    neg, pos = (rows[x][1] for x in ('negative', 'positive'))
    diff = lambda x, y: [math.fsum((a, -b)) for a, b in zip(x, y)]
    target, same, broad = diff(awful, wonderful), diff(excellent, wonderful), diff(pos, neg)
    target_norm = norm(target)
    norm(same); norm(broad)
    generator = random.Random(20261029)
    sample = [generator.gauss(0, 1) for _ in range(WIDTH)]
    coefficient = math.fsum(x * y for x, y in zip(sample, target)) / (target_norm * target_norm)
    residual = [math.fsum((x, -coefficient * y)) for x, y in zip(sample, target)]
    scale = target_norm / norm(residual)
    rand = [x * scale for x in residual]
    directions = {'primary': target, 'same': same, 'broad': broad,
                  'random': rand, 'opposite': [-x for x in target]}
    expected = {}
    for arm, dose in [('primary', 1.0), ('same', 1.0)] + [(arm, dose) for arm in CONTROLS for dose in DOSES]:
        original = neg if arm == 'broad' else wonderful
        if arm == 'primary':
            raw = rows['awful'][0]
        elif arm == 'same':
            raw = rows['excellent'][0]
        else:
            raw = struct.pack('<' + str(WIDTH) + 'f', *(
                math.fsum((x, dose * d)) for x, d in zip(original, directions[arm])))
        realized = struct.unpack('<' + str(WIDTH) + 'f', raw)
        delta = [math.fsum((a, -b)) for a, b in zip(realized, original)]
        require(all(math.isfinite(x) for x in realized), 'candidate row is nonfinite')
        expected[(arm, dose)] = {'raw': raw, 'original': rows['negative' if arm == 'broad' else 'wonderful'][0],
            'donor': rows['positive' if arm == 'broad' else 'excellent' if arm == 'same' else 'awful'][0],
            'donor_token': POSITIVE if arm == 'broad' else 1287 if arm == 'same' else 16242,
            'donor_storage': 'lm_head' if arm == 'broad' else 'shared',
            'intended_norm': norm(directions[arm]) * dose,
            'realized_norm': math.sqrt(math.fsum(x*x for x in delta))}
    return expected, struct.pack('<' + str(WIDTH) + 'd', *rand)


def verify_candidates(metadata, expected, random_bytes):
    receipt = metadata.get('receipt')
    require(type(receipt) is dict and set(receipt) == {'state_sha256', 'entries', 'random_direction_sha256',
                                                       'random_direction_hex'}
            and sha(canonical(receipt)) == metadata.get('sha256')
            and hex_digest(receipt.get('state_sha256')), 'candidate receipt differs')
    require(receipt.get('random_direction_hex') == random_bytes.hex()
            and receipt.get('random_direction_sha256') == sha(random_bytes), 'random direction differs')
    entries = receipt.get('entries')
    require(type(entries) is list and len(entries) == len(expected), 'candidate grid size differs')
    for item, ((arm, dose), row) in zip(entries, expected.items()):
        require(type(item) is dict and set(item) == {'arm', 'dose', 'sha256', 'original_sha256',
                'donor_sha256', 'donor_token', 'donor_storage', 'bytes_hex', 'intended_norm', 'realized_norm'}
                and item.get('arm') == arm and type(item.get('dose')) in (int, float)
                and item['dose'] == dose and item.get('bytes_hex') == row['raw'].hex()
                and item.get('sha256') == sha(row['raw'])
                and item.get('original_sha256') == sha(row['original'])
                and item.get('donor_sha256') == sha(row['donor'])
                and type(item.get('donor_token')) is int and item['donor_token'] == row['donor_token']
                and item.get('donor_storage') == row['donor_storage']
                and near(item.get('intended_norm'), row['intended_norm'])
                and near(item.get('realized_norm'), row['realized_norm']), 'candidate row or dose differs')
    return receipt['state_sha256'], metadata['sha256']


def _event_without_index(event):
    return {key: value for key, value in event.items() if key != 'event_index'}


class Journal:
    """Check the ordered event stream and every contiguous saved logit byte."""

    def __init__(self, directory, expected_candidates, random_bytes, *, phase, vocabulary=VOCAB):
        self.root = Path(directory)
        self.vocabulary = vocabulary
        require(phase in ('development', 'evaluation'), 'unknown study phase')
        self.phase = phase
        self.events_path = self.root / 'events.jsonl'
        self.logits_path = self.root / 'logits.f32'
        require(self.events_path.is_file() and self.events_path.stat().st_size <= MAX_EVENTS_BYTES
                and self.logits_path.is_file() and self.logits_path.stat().st_size <= MAX_LOGIT_BYTES,
                'missing or oversized journal')
        self.events_hash = file_sha(self.events_path)
        self.logits_hash = file_sha(self.logits_path)
        self.calls, self.outcomes, self.timeline, self.noncase_numeric, self.phase_order = {}, [], [], [], []
        self.probes, self.arm_receipts = [], []
        self.candidate_state = self.candidate_sha = self.initial_probe = None
        self._seen = set()
        self._scan(expected_candidates, random_bytes)

    def _arm_label(self, arm):
        if arm is None:
            return 'baseline'
        if self.phase == 'evaluation' or arm[0] in ('primary', 'same'):
            return arm[0]
        return f'{arm[0]}@{arm[1]}'

    def _scan(self, expected_candidates, random_bytes):
        pending = active_numeric = arm = arm_start = None
        installed = False
        numeric_calls = next_call = offset = 0
        case_numeric = []
        buffer_sha = None
        last_complete = None
        with self.events_path.open('rb') as lines, self.logits_path.open('rb') as logits:
            for index, raw_line in enumerate(lines):
                require(len(raw_line) <= 8 * 1024**2 and raw_line.endswith(b'\n'), 'journal line is not bounded')
                event = strict_json(raw_line)
                require(type(event) is dict and type(event.get('event_index')) is int
                        and event['event_index'] == index, 'journal event index differs')
                kind = event.get('kind')
                if kind in ('numeric_probe', 'frozen_candidates', 'arm_started', 'arm_installed',
                            'arm_completed', 'case_complete'):
                    require(set(event) == {'event_index', 'kind', 'metadata'}, 'unexpected journal event fields')
                elif kind == 'forward_started':
                    require(set(event) == {'event_index', 'kind', 'call', 'case_id', 'stage', 'metadata'},
                            'unexpected forward start fields')
                elif kind == 'forward_complete':
                    base = {'event_index', 'kind', 'call', 'case_id', 'stage', 'metadata'}
                    fields = base if event.get('stage') == 'encoder' else base | {'offset', 'bytes', 'shape', 'sha256'}
                    require(set(event) == fields, 'unexpected forward completion fields')
                if kind == 'numeric_probe':
                    meta = event.get('metadata')
                    require(type(meta) is dict and type(meta.get('subnormal_bits')) is int
                            and meta['subnormal_bits'] == 1 and type(meta.get('midpoint_even')) is float
                            and meta['midpoint_even'] == 1.0 and type(meta.get('midpoint_odd')) is float
                            and meta['midpoint_odd'] == 1.0 + 2**-22
                            and type(meta.get('stage')) is str and meta.get('when') in ('before', 'after'),
                            'numeric-mode observation differs')
                    identity = {k: v for k, v in meta.items()
                                if k not in ('when', 'subnormal_bits', 'midpoint_even', 'midpoint_odd')}
                    if meta['when'] == 'before':
                        require(active_numeric is None, 'numeric observations overlap')
                        active_numeric, numeric_calls = identity, 0
                    else:
                        require(active_numeric is not None and exact(identity, active_numeric) and pending is None,
                                'numeric observations are unpaired')
                        stage = identity['stage']
                        is_forward = stage in ('encoder', 'clean-probe') or stage.startswith(('greedy-', 'teacher-'))
                        require(numeric_calls == (1 if is_forward else 0), 'numeric/model call count differs')
                        if 'case_id' in identity:
                            require(set(identity) == {'case_id', 'arm', 'stage'}, 'numeric case identity differs')
                            case_numeric.append(identity)
                        else:
                            require(set(identity) == {'stage'}, 'numeric phase identity differs')
                            self.noncase_numeric.append(stage)
                            self.phase_order.append(('numeric', stage))
                        active_numeric = None
                elif kind == 'frozen_candidates':
                    require(self.candidate_sha is None and pending is None and arm is None,
                            'candidate receipt is out of order')
                    require(type(event.get('metadata')) is dict and set(event['metadata']) == {'receipt', 'sha256'},
                            'candidate metadata differs')
                    self.candidate_state, self.candidate_sha = verify_candidates(
                        event.get('metadata', {}), expected_candidates, random_bytes)
                    self.phase_order.append(('candidates',))
                elif kind == 'forward_started':
                    require(pending is None and self.candidate_sha is not None and active_numeric is not None
                            and type(event.get('call')) is int and event['call'] == next_call
                            and type(event.get('case_id')) is str and type(event.get('metadata')) is dict
                            and active_numeric.get('stage') == event.get('stage'), 'forward start differs')
                    name = event['metadata'].get('arm')
                    expected_arm = self._arm_label(arm)
                    require(name in (expected_arm, 'clean-probe'), 'forward is outside its arm')
                    if 'case_id' in active_numeric:
                        require(active_numeric['case_id'] == event['case_id']
                                and active_numeric['arm'] == name, 'forward numeric context differs')
                    if name == 'clean-probe':
                        require(event['case_id'] == 'discovery-1-target' and event['stage'] == 'clean-probe'
                                and exact(event['metadata'], {'arm': 'clean-probe', 'decoder_prefix': [0]}),
                                'clean probe identity differs')
                    else:
                        require(arm is None or installed, 'forward preceded arm installation')
                    pending, numeric_calls = event, numeric_calls + 1
                elif kind == 'forward_complete':
                    require(pending is not None and all(exact(event.get(key), pending.get(key))
                            for key in ('call', 'case_id', 'stage', 'metadata')), 'forward completion differs')
                    pending = None
                    next_call += 1
                    if event.get('stage') == 'encoder':
                        require(not any(key in event for key in ('offset', 'bytes', 'shape', 'sha256')),
                                'encoder unexpectedly contains logits')
                    else:
                        size = event.get('bytes')
                        require(type(event.get('offset')) is int and event['offset'] == offset
                                and type(size) is int and size > 0 and size % (self.vocabulary * 4) == 0
                                and size <= 4 * self.vocabulary * 4
                                and (exact(event.get('shape'), [self.vocabulary])
                                     or exact(event.get('shape'), [size // (self.vocabulary * 4), self.vocabulary]))
                                and offset + size <= self.logits_path.stat().st_size,
                                'logit spans are noncontiguous or invalid')
                        logits.seek(offset)
                        payload = logits.read(size)
                        require(len(payload) == size and sha(payload) == event.get('sha256')
                                and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', payload)),
                                'logit bytes or finiteness differ')
                        offset += size
                    last_complete = _event_without_index(event)
                    self.calls[event['call']] = last_complete
                    if event['stage'] == 'clean-probe':
                        require(event.get('shape') == [self.vocabulary] and event.get('bytes') == self.vocabulary * 4,
                                'clean probe shape differs')
                        if self.initial_probe is None:
                            require(arm is None, 'initial probe follows an edit')
                            self.initial_probe = last_complete
                        else:
                            require(arm is not None and installed
                                    and event['sha256'] == self.initial_probe['sha256'], 'restored probe differs')
                        self.probes.append(last_complete)
                elif kind == 'arm_started':
                    meta = event.get('metadata', {})
                    require(type(meta) is dict and set(meta) == {'arm', 'dose', 'original_sha256', 'donor_sha256',
                            'state_sha256', 'buffer_sha256'} and type(meta.get('dose')) in (int, float),
                            'arm start fields differ')
                    key = (meta.get('arm'), meta.get('dose'))
                    require(pending is None and arm is None and self.initial_probe is not None
                            and key in expected_candidates and meta.get('state_sha256') == self.candidate_state,
                            'arm start differs')
                    candidate = expected_candidates[key]
                    require(meta.get('original_sha256') == sha(candidate['original'])
                            and meta.get('donor_sha256') == sha(candidate['donor'])
                            and hex_digest(meta.get('buffer_sha256')), 'arm initial-state record differs')
                    if buffer_sha is None:
                        buffer_sha = meta['buffer_sha256']
                    require(meta['buffer_sha256'] == buffer_sha, 'baseline buffer identity differs')
                    arm, arm_start, installed = key, meta, False
                    self.timeline.append(('start', *key))
                    self.phase_order.append(('start', *key))
                elif kind == 'arm_installed':
                    meta = event.get('metadata', {})
                    require(type(meta) is dict and set(meta) == {'arm', 'dose', 'sha256'}
                            and type(meta.get('dose')) in (int, float)
                            and pending is None and arm is not None and not installed
                            and (meta.get('arm'), meta.get('dose')) == arm
                            and meta.get('sha256') == sha(expected_candidates[arm]['raw']),
                            'installed candidate differs')
                    installed = True
                    self.phase_order.append(('install', *arm))
                elif kind == 'arm_completed':
                    meta = event.get('metadata', {})
                    require(type(meta) is dict and set(meta) == {'arm', 'dose', 'installed_sha256',
                            'restored_row_sha256', 'pre_buffer_sha256', 'post_restore_state_sha256',
                            'post_probe_buffer_sha256', 'topology_unchanged', 'probe_span'}
                            and type(meta.get('dose')) in (int, float)
                            and pending is None and arm is not None and installed
                            and (meta.get('arm'), meta.get('dose')) == arm
                            and exact(meta.get('probe_span'), last_complete)
                            and last_complete is not None and last_complete.get('stage') == 'clean-probe'
                            and meta.get('installed_sha256') == sha(expected_candidates[arm]['raw'])
                            and meta.get('restored_row_sha256') == sha(expected_candidates[arm]['original'])
                            and meta.get('post_restore_state_sha256') == self.candidate_state
                            and meta.get('pre_buffer_sha256') == arm_start['buffer_sha256']
                            and meta.get('post_probe_buffer_sha256') == arm_start['buffer_sha256']
                            and meta.get('topology_unchanged') is True, 'restoration record differs')
                    self.timeline.append(('end', *arm))
                    self.phase_order.append(('end', *arm))
                    self.arm_receipts.append(meta)
                    arm, arm_start, installed = None, None, False
                elif kind == 'case_complete':
                    meta, outcome = event.get('metadata', {}), event.get('metadata', {}).get('outcome')
                    require(type(meta) is dict and set(meta) == {'arm', 'outcome'}, 'case completion fields differ')
                    expected_arm = self._arm_label(arm)
                    require(pending is None and active_numeric is None and (arm is None or installed)
                            and type(outcome) is dict and meta.get('arm') == expected_arm
                            and outcome.get('arm') == expected_arm and type(outcome.get('case_id')) is str,
                            'case completion is out of order')
                    stages = (['encoder'] + [f'greedy-{i}' for i in range(len(outcome.get('greedy_spans', [])))]
                              + ['teacher-positive', 'score-positive', 'teacher-negative', 'score-negative'])
                    if outcome.get('candidate_span') is not None:
                        stages += ['teacher-candidate', 'score-candidate']
                    stages += ['score-summary']
                    expected_numeric = [{'arm': expected_arm, 'case_id': outcome['case_id'], 'stage': stage}
                                        for stage in stages]
                    require(exact(case_numeric, expected_numeric), 'case numeric observations differ')
                    case_numeric = []
                    self.timeline.append(('case', expected_arm, outcome['case_id']))
                    self.phase_order.append(('case', expected_arm, outcome['case_id']))
                    self.outcomes.append(outcome)
                else:
                    raise AuditError('unknown, failure or incomplete journal event')
        require(pending is None and active_numeric is None and arm is None and not case_numeric
                and offset == self.logits_path.stat().st_size and self.initial_probe is not None
                and len(self.probes) == 1 + len(self.arm_receipts),
                'journal has incomplete work or unreferenced logit bytes')
        require(file_sha(self.events_path) == self.events_hash and file_sha(self.logits_path) == self.logits_hash,
                'journal files changed during replay')

    def rows(self, span, *, case_id, arm, stage, prefix, count, vector=False):
        require(type(span) is dict and type(span.get('call')) is int and span['call'] not in self._seen
                and exact(self.calls.get(span['call']), span) and span.get('case_id') == case_id
                and span.get('stage') == stage and exact(span.get('metadata'), {'arm': arm, 'decoder_prefix': prefix})
                and exact(span.get('shape'), [self.vocabulary] if vector else [count, self.vocabulary])
                and type(span.get('offset')) is int and span['offset'] >= 0
                and type(span.get('bytes')) is int and span['bytes'] == count * self.vocabulary * 4,
                'case logit span identity or shape differs')
        self._seen.add(span['call'])
        with self.logits_path.open('rb') as stream:
            stream.seek(span['offset'])
            raw = stream.read(span['bytes'])
        require(len(raw) == span['bytes'] and sha(raw) == span.get('sha256'), 'case logit span changed')
        row_size = self.vocabulary * 4
        return [struct.unpack('<' + str(self.vocabulary) + 'f', raw[i:i + row_size])
                for i in range(0, len(raw), row_size)]

    def raw(self, span):
        require(type(span) is dict and exact(self.calls.get(span.get('call')), span), 'unknown probe/span')
        with self.logits_path.open('rb') as stream:
            stream.seek(span['offset'])
            raw = stream.read(span['bytes'])
        require(len(raw) == span['bytes'] and sha(raw) == span['sha256'], 'saved span changed')
        return raw


def close_tree(left, right):
    if type(left) is dict and type(right) is dict:
        return set(left) == set(right) and all(close_tree(left[k], right[k]) for k in left)
    if type(left) in (tuple, list) and type(right) in (tuple, list):
        return len(left) == len(right) and all(close_tree(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return near(left, right)
    return type(left) is type(right) and left == right


def score_terms(row, token):
    require(type(token) is int and 0 <= token < len(row), 'scored token is outside the vocabulary')
    maximum = max(row)
    logsum = math.log(math.fsum(math.exp(float(value) - maximum) for value in row))
    return [float(row[token]), float(maximum), logsum]


def signed_terms(steps, sign=1.0):
    for selected, maximum, logsum in steps:
        yield sign * selected
        yield -sign * maximum
        yield -sign * logsum


def sequence_score(terms):
    return math.fsum(signed_terms(terms))


def difference(after, before):
    return math.fsum((*signed_terms(after), *signed_terms(before, -1.0)))


def effect(after, before):
    a, b = after['terms'], before['terms']
    margin = math.fsum((*signed_terms(a['negative']), *signed_terms(b['negative'], -1.0),
                        *signed_terms(a['positive'], -1.0), *signed_terms(b['positive'])))
    return {'margin_gain': margin, 'target_gain': difference(a['negative'], b['negative']),
            'negative': after['label'] == 'negative'}


def normalized(text):
    return unicodedata.normalize('NFC', text).strip().casefold()


def verify_outcome(journal, case, outcome, decoder, special_ids, *, labels=None):
    """Recompute generation, every teacher row, and the recorded score terms."""
    labels = {'positive': [POSITIVE, EOS], 'negative': [NEGATIVE, EOS]} if labels is None else labels
    require(type(outcome) is dict and set(outcome) == {'case_id', 'arm', 'greedy_ids', 'decoded', 'terminated',
            'strict_correct', 'strict_label', 'tops', 'greedy_spans', 'label_terms', 'label_spans',
            'candidate_terms', 'candidate_span', 'margin', 'negative_score'}, 'outcome fields differ')
    case_id, arm = case['id'], outcome.get('arm')
    require(outcome.get('case_id') == case_id and type(arm) is str, 'outcome identity differs')
    spans = outcome.get('greedy_spans')
    require(type(spans) is list and 1 <= len(spans) <= 4, 'greedy span count differs')
    prefix, ids, tops = [0], [], []
    for index, span in enumerate(spans):
        row = journal.rows(span, case_id=case_id, arm=arm, stage=f'greedy-{index}',
                           prefix=prefix, count=1, vector=True)[0]
        maximum = max(row)
        tied = [i for i, value in enumerate(row) if value == maximum]
        token = tied[0]
        ids.append(token)
        tops.append({'representative': token, 'tied_ids': tied, 'unique': len(tied) == 1})
        prefix = [*prefix, token]
        require(token != EOS or index == len(spans) - 1, 'generation continued after EOS')
    terminated = ids[-1] == EOS
    require(terminated or len(ids) == 4, 'greedy sequence stopped before EOS and the four-token limit')
    content = ids[:-1] if terminated else ids
    decoded = decoder(content)
    require(type(decoded) is str, 'tokenizer output is not text')
    valid = terminated and all(top['unique'] for top in tops) and not any(x in special_ids for x in content)
    correct = bool(valid and normalized(decoded) == normalized(case['expected']))
    label = normalized(decoded) if valid and normalized(decoded) in labels else None
    require(exact(outcome.get('greedy_ids'), ids) and exact(outcome.get('tops'), tops)
            and type(outcome.get('terminated')) is bool and outcome['terminated'] == terminated
            and outcome.get('decoded') == decoded and type(outcome.get('strict_correct')) is bool
            and outcome['strict_correct'] == correct and outcome.get('strict_label') == label,
            'greedy output or strict classification differs')
    terms = {}
    require(type(outcome.get('label_spans')) is dict and set(outcome['label_spans']) == set(labels),
            'label span inventory differs')
    for name, tokens in labels.items():
        rows = journal.rows(outcome['label_spans'][name], case_id=case_id, arm=arm,
                            stage='teacher-' + name, prefix=[0, *tokens[:-1]], count=len(tokens))
        terms[name] = [score_terms(row, token) for row, token in zip(rows, tokens)]
    candidate = None
    if case['expected'] not in labels:
        target = case['expected_ids']
        rows = journal.rows(outcome.get('candidate_span'), case_id=case_id, arm=arm,
                            stage='teacher-candidate', prefix=[0, *target[:-1]], count=len(target))
        candidate = [score_terms(row, token) for row, token in zip(rows, target)]
    else:
        require(outcome.get('candidate_span') is None, 'unexpected candidate teacher span')
    require(close_tree(outcome.get('label_terms'), terms) and close_tree(outcome.get('candidate_terms'), candidate)
            and near(outcome.get('margin'), difference(terms['negative'], terms['positive']))
            and near(outcome.get('negative_score'), sequence_score(terms['negative'])),
            'teacher-forced scores differ from saved full-vocabulary rows')
    return {'saved': outcome, 'ids': ids, 'label': label, 'correct': correct, 'terms': terms,
            'candidate': candidate, 'case': case}


def verify_used_calls(journal):
    expected = {call for call, event in journal.calls.items()
                if 'offset' in event and event['stage'] != 'clean-probe'}
    require(journal._seen == expected, 'case logits are unreferenced or reused')


def selection(primary, candidates):
    require(len(primary) == 2 and len(candidates) == len(DOSES), 'calibration grid is incomplete')
    eligible = all(row['negative'] and row['margin_gain'] >= 0.5 and row['target_gain'] >= 0.5
                   for row in primary)
    assessed = []
    for dose, rows in candidates:
        require(len(rows) == 2, 'grid point lacks a calibration case')
        ratios = [abs(math.fsum((observed[key], -reference[key])))
                  / max(0.10, 0.10 * abs(reference[key]))
                  for observed, reference in zip(rows, primary) for key in ('margin_gain', 'target_gain')]
        maximum = max(ratios)
        assessed.append({'dose': float(dose), 'maximum_ratio': maximum, 'ratios': ratios,
                         'matched': bool(eligible and maximum <= 1.0 and all(row['negative'] for row in rows))})
    require([dose for dose, _ in candidates] == list(DOSES), 'fixed dose order differs')
    matched = [point for point in assessed if point['matched']]
    chosen = min(matched or assessed, key=lambda point: (point['maximum_ratio'], point['dose']))
    return {**chosen, 'primary_eligible': eligible, 'grid': assessed}


def compare_substitution(journal, target, donor):
    """Compare only equal decoder prefixes until divergence or the edited token appears."""
    a, b = target['saved'], donor['saved']
    comparisons = []
    for name in ('positive', 'negative'):
        left, right = a['label_spans'][name], b['label_spans'][name]
        pa, pb = left['metadata']['decoder_prefix'], right['metadata']['decoder_prefix']
        require(pa == pb, 'substitution teacher prefixes differ')
        if EDITED not in pa:
            require(journal.raw(left) == journal.raw(right), 'substitution teacher logits differ')
            comparisons.append('teacher-' + name)
    for left, right in zip(a['greedy_spans'], b['greedy_spans']):
        pa, pb = left['metadata']['decoder_prefix'], right['metadata']['decoder_prefix']
        if pa != pb or EDITED in pa:
            break
        require(journal.raw(left) == journal.raw(right), 'substitution greedy logits differ')
        comparisons.append('greedy')
    return {'compared': len(comparisons), 'target_ids': target['ids'], 'donor_ids': donor['ids'],
            'same_sequence': target['ids'] == donor['ids']}


def drift(before, after):
    require(len(before) == len(after) and len(before) > 0, 'distribution dimensions differ')
    ma, mb = max(before), max(after)
    ea, eb = [math.exp(x-ma) for x in before], [math.exp(x-mb) for x in after]
    sa, sb = math.fsum(ea), math.fsum(eb)
    la, lb = math.log(sa), math.log(sb)
    tv = 0.5 * math.fsum(abs(x/sa - y/sb) for x, y in zip(ea, eb))
    kl = math.fsum((x/sa) * math.fsum((za, -ma, -la, -zb, mb, lb))
                   for x, za, zb in zip(ea, before, after))
    return {'kl': kl, 'tv': tv}


def span_first_row(journal, span):
    require(span['shape'] == [journal.vocabulary] and span['bytes'] == journal.vocabulary * 4,
            'first-step vocabulary shape differs')
    return struct.unpack('<' + str(journal.vocabulary) + 'f', journal.raw(span))


def _verified_outcomes(journal, cases, decoder, special_ids):
    by_id = {case['id']: case for case in cases}
    verified = {}
    for saved in journal.outcomes:
        key = (saved.get('arm'), saved.get('case_id'))
        require(key not in verified and key[1] in by_id, 'duplicate or unknown completed case')
        verified[key] = verify_outcome(journal, by_id[key[1]], saved, decoder, special_ids)
    verify_used_calls(journal)
    for probe in journal.probes:
        require(journal.raw(probe) == journal.raw(journal.initial_probe), 'restored probe bytes differ')
    return verified


def _same_outcome(saved, verified):
    require(exact(saved, verified['saved']), 'published result and completed case differ')


def development(journal, cases, saved, decoder, special_ids):
    require(type(saved) is dict and saved.get('status') in ('complete', 'ineligible'),
            'development result is incomplete')
    verified = _verified_outcomes(journal, cases, decoder, special_ids)
    timeline = [('case', 'baseline', case['id']) for case in cases[:8]]
    phase_order = ([('numeric', 'candidate-construction'), ('candidates',), ('numeric', 'clean-probe')]
                   + timeline + [('numeric', 'clean-label-margin')] * 8)
    baseline = {}
    for case in cases[:8]:
        key = ('baseline', case['id'])
        require(key in verified and case['id'] in saved.get('baseline', {}), 'clean baseline is missing')
        baseline[case['id']] = verified[key]
        _same_outcome(saved['baseline'][case['id']], verified[key])
    require(set(saved['baseline']) == {case['id'] for case in cases[:8]}, 'clean baseline inventory differs')
    supported = {}
    for case in cases[:8]:
        value = baseline[case['id']]
        margin = difference(value['terms']['negative'], value['terms']['positive'])
        supported[case['id']] = value['correct'] and (margin < 0 if case['expected'] == 'positive' else margin > 0)
    noncase = ['candidate-construction', 'clean-probe'] + ['clean-label-margin'] * 8
    if not all(supported.values()):
        require(saved.get('status') == 'ineligible' and saved.get('reason') == 'initial-clean-baseline'
                and exact(saved.get('clean_supported'), supported) and journal.timeline == timeline
                and journal.noncase_numeric == noncase and journal.phase_order == phase_order
                and not journal.arm_receipts,
                'failed initial baseline did not stop before editing')
        return {'status': 'ineligible', 'reason': 'initial-clean-baseline', 'clean_supported': supported}
    require(saved.get('status') == 'complete', 'valid clean baseline lacks a complete development grid')
    primary = {}
    timeline.append(('start', 'primary', 1.0))
    phase_order += [('start', 'primary', 1.0), ('install', 'primary', 1.0)]
    substitution = {}
    for case in cases[0:8:2]:
        key = ('primary', case['id'])
        require(key in verified and case['id'] in saved.get('primary', {}), 'primary development case is missing')
        primary[case['id']] = verified[key]
        _same_outcome(saved['primary'][case['id']], verified[key])
        donor_id = case['id'].replace('-target', '-donor')
        substitution[case['id']] = compare_substitution(journal, verified[key], baseline[donor_id])
        timeline.append(('case', 'primary', case['id']))
        phase_order.append(('case', 'primary', case['id']))
    require(set(saved['primary']) == set(primary), 'primary development inventory differs')
    timeline.append(('end', 'primary', 1.0))
    phase_order += [('numeric', 'clean-probe'), ('end', 'primary', 1.0), ('numeric', 'effect'), ('numeric', 'effect')]
    noncase += ['clean-probe', 'effect', 'effect']
    calibration = cases[4:8:2]
    primary_effects = [effect(primary[c['id']], baseline[c['id']]) for c in calibration]
    eligible = all(row['negative'] and row['margin_gain'] >= 0.5 and row['target_gain'] >= 0.5
                   for row in primary_effects)
    require(type(saved.get('calibration_eligible')) is bool and saved['calibration_eligible'] == eligible
            and close_tree(saved.get('primary_effects'), primary_effects), 'primary calibration differs')
    require(set(saved.get('grids', {})) == set(CONTROLS)
            and set(saved.get('selection', {})) == set(CONTROLS), 'control grid inventory differs')
    selected = {}
    for arm in CONTROLS:
        grid = saved['grids'][arm]
        require(type(grid) is list and len(grid) == len(DOSES), 'control grid is incomplete')
        assessed = []
        for dose, point in zip(DOSES, grid):
            require(type(point.get('dose')) in (int, float) and point['dose'] == dose
                    and type(point.get('outcomes')) is list and len(point['outcomes']) == 2,
                    'grid point identity differs')
            timeline.append(('start', arm, dose))
            phase_order += [('start', arm, dose), ('install', arm, dose)]
            effects = []
            label = f'{arm}@{dose}'
            for case, stored in zip(calibration, point['outcomes']):
                key = (label, case['id'])
                require(key in verified, 'grid outcome is missing')
                _same_outcome(stored, verified[key])
                effects.append(effect(verified[key], baseline[case['id']]))
                timeline.append(('case', label, case['id']))
                phase_order.append(('case', label, case['id']))
            timeline.append(('end', arm, dose))
            phase_order += [('numeric', 'clean-probe'), ('end', arm, dose), ('numeric', 'effect'), ('numeric', 'effect')]
            require(close_tree(point.get('rows'), effects), 'grid effects differ')
            assessed.append((dose, effects))
            noncase += ['clean-probe', 'effect', 'effect']
        choice = selection(primary_effects, assessed)
        recorded_choice = saved['selection'][arm]
        require(type(recorded_choice) is dict and type(recorded_choice.get('dose')) in (int, float)
                and recorded_choice['dose'] == choice['dose']
                and type(recorded_choice.get('grid')) is list and len(recorded_choice['grid']) == len(DOSES)
                and all(type(point.get('dose')) in (int, float) and point['dose'] == dose
                        for point, dose in zip(recorded_choice['grid'], DOSES))
                and close_tree(recorded_choice, choice), 'control selection differs')
        selected[arm] = choice
        noncase.append('selection-' + arm)
        phase_order.append(('numeric', 'selection-' + arm))
    timeline.append(('start', 'same', 1.0))
    phase_order += [('start', 'same', 1.0), ('install', 'same', 1.0)]
    require(type(saved.get('same')) is list and len(saved['same']) == 2, 'same-polarity control is incomplete')
    for case, stored in zip(calibration, saved['same']):
        key = ('same', case['id'])
        require(key in verified, 'same-polarity outcome is missing')
        _same_outcome(stored, verified[key])
        timeline.append(('case', 'same', case['id']))
        phase_order.append(('case', 'same', case['id']))
    timeline.append(('end', 'same', 1.0))
    noncase.append('clean-probe')
    phase_order += [('numeric', 'clean-probe'), ('end', 'same', 1.0)]
    require(journal.timeline == timeline and journal.noncase_numeric == noncase and journal.phase_order == phase_order,
            'development case/arm order or scoring order differs')
    return {'status': 'complete', 'calibration_eligible': eligible, 'primary_effects': primary_effects,
            'selection': selected, 'clean_supported': supported, 'substitution': substitution}


def _prefix_scope(journal, before, after, structural):
    comparisons, comparable = [], structural
    for index in range(max(len(before), len(after))):
        left = before[index] if index < len(before) else None
        right = after[index] if index < len(after) else None
        pa = None if left is None else left['metadata']['decoder_prefix']
        pb = None if right is None else right['metadata']['decoder_prefix']
        if not structural:
            status = 'no-structural-claim'
        elif not comparable:
            status = 'after-noncomparable'
        elif left is None or right is None:
            status, comparable = 'missing-step', False
        elif pa != pb:
            status, comparable = 'divergent-prefix', False
        elif EDITED in pa:
            status, comparable = 'edited-token-in-prefix', False
        else:
            status = 'matched'
        equal = None
        if status == 'matched':
            equal = journal.raw(left) == journal.raw(right)
            require(equal, 'token-absent same-prefix structural identity failed')
        comparisons.append({'step': index, 'scope': status, 'baseline_prefix': pa,
                            'edited_prefix': pb, 'equal_raw_logits': equal})
    return comparisons


def structural_locality(journal, case, before, after, arm):
    structural = arm in SHARED and EDITED not in case['input_ids']
    a, b = before['saved'], after['saved']
    greedy = _prefix_scope(journal, a['greedy_spans'], b['greedy_spans'], structural)
    teachers = {key: _prefix_scope(journal, [a['label_spans'][key]], [b['label_spans'][key]], structural)[0]
                for key in ('positive', 'negative')}
    if a.get('candidate_span') is not None and b.get('candidate_span') is not None:
        teachers['candidate'] = _prefix_scope(journal, [a['candidate_span']], [b['candidate_span']], structural)[0]
    return structural, greedy, teachers


def evaluation(journal, cases, saved, dev, decoder, special_ids):
    require(type(saved) is dict and saved.get('status') == 'complete'
            and dev.get('status') == 'complete', 'evaluation or development is incomplete')
    verified = _verified_outcomes(journal, cases, decoder, special_ids)
    timeline = []
    phase_order = [('numeric', 'candidate-construction'), ('candidates',), ('numeric', 'clean-probe')]
    baseline = {}
    for case in cases[8:]:
        key = ('baseline', case['id'])
        require(key in verified and case['id'] in saved.get('baseline', {}), 'evaluation baseline is missing')
        baseline[case['id']] = verified[key]
        _same_outcome(saved['baseline'][case['id']], verified[key])
        timeline.append(('case', 'baseline', case['id']))
        phase_order.append(('case', 'baseline', case['id']))
    require(set(saved['baseline']) == set(baseline), 'evaluation baseline inventory differs')
    arms = {'primary': 1.0, 'same': 1.0, **{name: dev['selection'][name]['dose'] for name in CONTROLS}}
    require(set(saved.get('arms', {})) == set(arms), 'evaluation arm inventory differs')
    targets, localities = cases[8:16:2], cases[16:]
    for arm, dose in arms.items():
        timeline.append(('start', arm, dose))
        phase_order += [('start', arm, dose), ('install', arm, dose)]
        require(set(saved['arms'][arm]) == {case['id'] for case in [*targets, *localities]},
                'evaluation arm cases differ')
        for case in [*targets, *localities]:
            key = (arm, case['id'])
            require(key in verified, 'evaluation case is missing')
            _same_outcome(saved['arms'][arm][case['id']], verified[key])
            timeline.append(('case', arm, case['id']))
            phase_order.append(('case', arm, case['id']))
        timeline.append(('end', arm, dose))
        phase_order += [('numeric', 'clean-probe'), ('end', arm, dose)]
    require(journal.timeline == timeline and saved.get('localities') == [c['id'] for c in localities],
            'evaluation case/arm order differs')
    target_rows, primary_results, substitutions = [], [], {}
    for case in targets:
        case_id = case['id']
        donor = baseline[case_id.replace('-target', '-donor')]
        eligible = baseline[case_id]['label'] == 'positive' and donor['label'] == 'negative'
        primary = verified[('primary', case_id)]
        target_rows.append({'case_id': case_id, 'baseline_eligible': eligible, 'edited_label': primary['label']})
        substitutions[case_id] = compare_substitution(journal, primary, donor)
        effects = {arm: effect(verified[(arm, case_id)], baseline[case_id]) for arm in arms}
        primary_results.append({'case_id': case_id, 'baseline_eligible': eligible,
                                'baseline_ids': baseline[case_id]['ids'], 'donor_ids': donor['ids'],
                                'primary_ids': primary['ids'], 'effects': effects})
    eligible_rows = [row for row in target_rows if row['baseline_eligible']]
    reversed_count = sum(row['edited_label'] == 'negative' for row in eligible_rows)
    status = ('ineligible' if not dev['calibration_eligible'] or len(eligible_rows) < 3
              else 'positive' if reversed_count >= 3 else 'negative')
    primary_result = {'status': status, 'baseline_eligible': len(eligible_rows), 'negative_labels': reversed_count}
    require(exact(saved.get('targets'), target_rows) and exact(saved.get('primary'), primary_result),
            'target eligibility or primary outcome differs')
    require(set(saved.get('comparisons', {})) == set(CONTROLS), 'comparison inventory differs')
    noncase = ['candidate-construction', 'clean-probe'] + ['clean-probe'] * len(arms)
    comparisons = {}
    for name in CONTROLS:
        if not dev['selection'][name]['matched']:
            result = {'eligible': False, 'reason': 'calibration-unmatched'}
        elif status != 'positive':
            result = {'eligible': False, 'reason': 'primary-not-positive'}
        else:
            references, alternatives = [], []
            for item in eligible_rows:
                case_id = item['case_id']
                p, c = verified[('primary', case_id)], verified[(name, case_id)]
                references.append({**effect(p, baseline[case_id]), 'strict_label': p['label']})
                alternatives.append({**effect(c, baseline[case_id]), 'strict_label': c['label']})
                noncase += ['effect', 'effect']
                phase_order += [('numeric', 'effect'), ('numeric', 'effect')]
            balanced = all(p['strict_label'] in ('positive', 'negative')
                           and c['strict_label'] == p['strict_label']
                           and all(abs(math.fsum((p[k], -c[k]))) <= max(0.10, 0.10 * abs(p[k]))
                                   for k in ('margin_gain', 'target_gain'))
                           for p, c in zip(references, alternatives))
            result = {'eligible': balanced, 'reason': None if balanced else 'evaluation-imbalance',
                      'primary_effects': references, 'control_effects': alternatives}
            noncase.append('evaluation-balance-' + name)
            phase_order.append(('numeric', 'evaluation-balance-' + name))
        require(close_tree(saved['comparisons'][name], result), 'per-target control balance differs')
        comparisons[name] = result
    noncase.append('locality-summary')
    phase_order.append(('numeric', 'locality-summary'))
    require(journal.noncase_numeric == noncase and journal.phase_order == phase_order,
            'evaluation scoring or phase order differs')
    locality = []
    for case in localities:
        case_id = case['id']
        before = baseline[case_id]
        original = span_first_row(journal, before['saved']['greedy_spans'][0])
        for arm in arms:
            after = verified[(arm, case_id)]
            metric = drift(original, span_first_row(journal, after['saved']['greedy_spans'][0]))
            expected_before = before['terms'][case['expected']] if case['expected'] in ('positive', 'negative') else before['candidate']
            expected_after = after['terms'][case['expected']] if case['expected'] in ('positive', 'negative') else after['candidate']
            structural, greedy_scope, teacher_scope = structural_locality(journal, case, before, after, arm)
            entry = {'case_id': case_id, 'arm': arm, 'input_contains_edited_token': EDITED in case['input_ids'],
                     'first_step_structural_identity_expected': structural,
                     'greedy_prefix_comparisons': greedy_scope, 'teacher_prefix_comparisons': teacher_scope,
                     'baseline_correct': before['correct'], 'changed_correct': after['correct'],
                     'baseline_expected_log_probability': sequence_score(expected_before),
                     'changed_expected_log_probability': sequence_score(expected_after),
                     'baseline_ids': before['ids'], 'changed_ids': after['ids'], **metric}
            locality.append({'measurement': entry,
                             'structural_identity': {'applies': structural, 'greedy': greedy_scope,
                                                     'teachers': teacher_scope}})
    require(close_tree(saved.get('locality_distributions'), [x['measurement'] for x in locality]),
            'saved locality metrics differ from full-vocabulary rows')
    return {'status': status, 'primary': primary_result, 'targets': primary_results,
            'comparisons': comparisons, 'substitution': substitutions, 'locality': locality}


def _git_spec(revision, path):
    require(type(revision) is str and len(revision) == 40
            and all(c in '0123456789abcdef' for c in revision)
            and type(path) is str and not path.startswith(('-', '/'))
            and '\\' not in path and all(part not in ('', '.', '..') for part in path.split('/')),
            'invalid Git artifact identity')
    return revision + ':' + path


def git_blob(repo, revision, path):
    result = subprocess.run(['git', '--no-replace-objects', 'show', _git_spec(revision, path)],
                            cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    require(result.returncode == 0, 'published Git artifact is unavailable')
    return result.stdout


def git_digest(repo, revision, path, *, aggregate=None, limit=MAX_LOGIT_BYTES):
    process = subprocess.Popen(['git', '--no-replace-objects', 'show', _git_spec(revision, path)],
                               cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    timer = threading.Timer(120, process.kill)
    timer.daemon = True
    timer.start()
    digest, size = hashlib.sha256(), 0
    try:
        for block in iter(lambda: process.stdout.read(1024 * 1024), b''):
            size += len(block)
            require(size <= limit, 'published artifact exceeds the evidence limit')
            digest.update(block)
            if aggregate is not None:
                aggregate.update(block)
        require(process.wait(timeout=60) == 0, 'published Git artifact is unavailable')
    finally:
        timer.cancel()
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
    return digest.hexdigest(), size


def validate_manifest(repo, path, revision):
    raw = Path(path).read_bytes()
    manifest = strict_json(raw)
    require(type(manifest) is dict and manifest.get('schema') == 'flan-weight-edit-manifest-v1'
            and revision == MANIFEST_REVISION and sha(raw) == MANIFEST_SHA
            and manifest.get('source_revision') == SOURCE_REVISION
            and exact(manifest.get('public_inputs'), PUBLIC_INPUTS)
            and manifest.get('case_sha256') == CASE_SHA and manifest.get('protocol_sha256') == PROTOCOL_SHA
            and manifest.get('asset_manifest_sha256') == ASSET_SHA
            and git_blob(repo, revision, MANIFEST_PATH) == raw, 'published execution manifest differs')
    for input_path, digest in PUBLIC_INPUTS.items():
        require(sha(git_blob(repo, revision, input_path)) == digest,
                'published prospective input differs: ' + input_path)
    source = manifest.get('source_revision')
    require(type(source) is str and len(source) == 40 and all(c in '0123456789abcdef' for c in source),
            'source revision differs')
    parents = subprocess.run(['git', '--no-replace-objects', 'rev-list', '--parents', '-n', '1', revision],
                             cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    require(parents.returncode == 0 and parents.stdout.decode('ascii').split() == [revision, source],
            'execution manifest is not the fixed source successor')
    sources = manifest.get('sources')
    runtime = ('owner', 'driver', 'cases', 'numerical', 'guard', 'journal', 'worker',
               'decision', 'study', 'session', 'handoff', 'results')
    tests = ('owner', 'case', 'numerical', 'guard', 'journal', 'worker',
             'decision', 'study', 'session', 'handoff', 'results', 'driver')
    base = 'studies/flan_weight_edit_v2/'
    expected = ({f'flan_weight_edit_{name}.py': base + 'scripts/' + f'flan_weight_edit_{name}.py' for name in runtime}
                | {f'flan_weight_edit_{name}_test.py': base + 'tests/flan_weight_edit/' + f'flan_weight_edit_{name}_test.py'
                   for name in tests}
                | {'flan_weight_edit_conftest.py': base + 'tests/flan_weight_edit/conftest.py'})
    require(type(sources) is dict and set(sources) == set(expected), 'source inventory differs')
    for name, source_path in expected.items():
        item = sources[name]
        require(type(item) is dict and item.get('git_path') == source_path
                and hex_digest(item.get('sha256'))
                and sha(git_blob(repo, source, source_path)) == item['sha256'],
                'published source blob differs: ' + name)
    require(type(manifest.get('runtime')) is dict, 'runtime manifest is missing')
    return manifest, sha(raw)


def reject_failure_markers(root):
    markers = {'failure.json', 'error.json', 'incomplete.json', 'failed.json', 'partial.json'}
    count = 0
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            count += 1
            require(count <= 4096, 'phase inventory is unexpectedly large')
            path = Path(current) / name
            info = path.lstat()
            require(not path.is_symlink() and not (getattr(info, 'st_file_attributes', 0) & 0x400),
                    'phase inventory contains a link or reparse point')
            lower = name.lower()
            require(lower not in markers and not lower.endswith(('.partial', '.incomplete', '.failed')),
                    'failure or partial marker coexists with a completed phase')


def expected_cache_paths(root):
    temp = (Path(root) / 'temp').resolve()
    names = {'HF_HOME': 'hf', 'HF_HUB_CACHE': 'hf-hub', 'HF_ASSETS_CACHE': 'hf-assets',
             'HF_DATASETS_CACHE': 'hf-datasets', 'HF_MODULES_CACHE': 'hf-modules',
             'HUGGINGFACE_HUB_CACHE': 'hf-hub-legacy', 'HUGGINGFACE_ASSETS_CACHE': 'hf-assets-legacy',
             'TRANSFORMERS_CACHE': 'transformers', 'TORCH_HOME': 'torch', 'XDG_CACHE_HOME': 'xdg',
             'MPLCONFIGDIR': 'mpl'}
    return {'TEMP': str(temp), 'TMP': str(temp),
            **{name: str(temp / 'library-cache' / directory) for name, directory in names.items()}}


def validate_closed_phase(directory, phase_name, manifest, manifest_sha, asset_manifest, *, scope_path=None, claim_path=None):
    root = Path(directory)
    reject_failure_markers(root)
    phase_path, exit_path = root / 'phase.json', root / 'operator-exit.json'
    phase, exit_record = load_json(phase_path), load_json(exit_path)
    require(phase.get('phase') == phase_name and phase.get('execution_status') == 'complete'
            and phase.get('source_revision') == manifest['source_revision']
            and exit_record.get('status') == 'complete_candidate'
            and type(exit_record.get('exit_code')) is int and exit_record['exit_code'] == 0
            and exit_record.get('reaped') is True and exit_record.get('drained') is True
            and exit_record.get('identity_matches') is True and exit_record.get('timed_out') is False
            and exit_record.get('phase_consistent') is True and exit_record.get('within_evidence_cap') is True
            and type(exit_record.get('cumulative_evidence_bytes_before_exit_record')) is int
            and 0 <= exit_record['cumulative_evidence_bytes_before_exit_record'] <= 255 * 1024**2
            and number(exit_record.get('supervised_elapsed_seconds')) and exit_record['supervised_elapsed_seconds'] >= 0
            and exit_record.get('error') is None and 'error_detail' in exit_record
            and exit_record['error_detail'] is None and exit_record.get('phase_sha256') == file_sha(phase_path),
            'phase or direct-child exit is incomplete')
    start, spawn = load_json(root / 'operator-start.json'), load_json(root / 'operator-spawn.json')
    require(start.get('phase') == phase_name and start.get('source') == manifest['source_revision']
            and start.get('rehearsal') is False and type(spawn.get('pid')) is int
            and type(spawn.get('parent_pid')) is int and type(exit_record.get('direct_child_pid')) is int
            and type(exit_record.get('operator_pid')) is int and type(phase.get('pid')) is int
            and type(phase.get('parent_pid')) is int
            and spawn['pid'] == exit_record.get('direct_child_pid') == phase.get('pid')
            and spawn['parent_pid'] == exit_record['operator_pid'] == phase['parent_pid']
            and phase.get('manifest_sha256') == manifest_sha and phase.get('case_sha256') == CASE_SHA
            and hex_digest(start.get('scope_sha256')) and phase.get('scope_sha256') == start['scope_sha256'],
            'phase process or source records differ')
    parent_temp = start.get('parent_temp')
    study_root = root.parent.parent.resolve()
    require(type(parent_temp) is dict and set(parent_temp) == {'TEMP', 'TMP'}
            and all(type(value) is str and Path(value).is_absolute()
                    and Path(value).resolve().is_relative_to(study_root) for value in parent_temp.values())
            and exact(start.get('child_cache_paths'), expected_cache_paths(root)),
            'recorded parent or child cache paths differ')
    if scope_path is not None:
        require(file_sha(scope_path) == start['scope_sha256'], 'scope artifact differs')
    if claim_path is not None:
        claim = load_json(claim_path)
        require(hex_digest(start.get('claim_sha256')) and file_sha(claim_path) == start['claim_sha256']
                and claim.get('phase') == phase_name and claim.get('manifest_sha256') == manifest_sha
                and type(claim.get('attempt')) is str and Path(claim['attempt']).resolve() == root.resolve(),
                'fixed one-shot claim differs')
    logs = exit_record.get('logs')
    require(type(logs) is dict and set(logs) == {'stdout', 'stderr'}, 'process log inventory differs')
    for name, record in logs.items():
        path = root / (name + '.log')
        require(type(record) is dict and type(record.get('bytes')) is int and 0 <= record['bytes'] <= 65536
                and record.get('finished') is True and record.get('truncated') is False
                and record.get('error') is None and path.is_file() and path.stat().st_size == record['bytes']
                and hex_digest(record.get('sha256')) and file_sha(path) == record['sha256'],
                'process log was not fully drained')
    stdout = (root / 'stdout.log').read_bytes().splitlines()
    require(stdout, 'child identity record is absent')
    identity = strict_json(stdout[0])
    require(type(identity.get('pid')) is int and type(identity.get('parent_pid')) is int
            and identity['pid'] == spawn['pid'] and identity['parent_pid'] == spawn['parent_pid']
            and exact(identity.get('cache_paths'), start['child_cache_paths']),
            'direct-child identity record differs')
    if phase.get('scientific_status') == 'ineligible' and phase.get('model_loaded') is False:
        require(phase.get('reason') == 'tokenizer-alignment' and type(phase.get('calls')) is int
                and phase['calls'] == 0,
                'pre-forward ineligibility record differs')
        return phase, exit_record, None
    require(exact(phase.get('runtime'), manifest['runtime'])
            and phase.get('assets') == {name: item['sha256'] for name, item in asset_manifest['files'].items()},
            'phase runtime, assets, source or scope binding differs')
    files = {'result_sha256': root / 'result.json',
             'events_sha256': root / 'evidence' / 'events.jsonl',
             'logits_sha256': root / 'evidence' / 'logits.f32',
             'resources_sha256': root / 'resources.jsonl'}
    for key, path in files.items():
        require(hex_digest(phase.get(key)) and path.is_file() and file_sha(path) == phase[key],
                'phase artifact hash differs: ' + key)
    require(type(phase.get('calls')) is int and 0 <= phase['calls'] <= 1200
            and type(phase.get('cumulative_calls')) is int and 0 <= phase['cumulative_calls'] <= 1200,
            'phase call accounting differs')
    validate_resource_log(files['resources_sha256'], phase['calls'], phase.get('resource_summary'))
    return phase, exit_record, files


def validate_resource_log(path, calls, summary):
    """Check the recorded per-call disk observations, not a filesystem reservation."""
    path = Path(path)
    require(type(calls) is int and 0 <= calls <= 1200 and path.is_file()
            and path.stat().st_size <= 1024 * 1024, 'resource log is missing or oversized')
    records = []
    with path.open('rb') as stream:
        for raw in stream:
            require(len(raw) <= 4096 and raw.endswith(b'\n') and len(records) < 2400,
                    'resource observation is oversized or incomplete')
            record = strict_json(raw)
            index = len(records)
            require(type(record) is dict and set(record) ==
                    {'index', 'call', 'boundary', 'source_free', 'output_free', 'passed'}
                    and type(record.get('index')) is int and record['index'] == index
                    and type(record.get('call')) is int and record['call'] == index // 2 + 1
                    and record.get('boundary') == ('before' if index % 2 == 0 else 'after')
                    and type(record.get('source_free')) is int and C_FREE_FLOOR <= record['source_free'] < 2**64
                    and type(record.get('output_free')) is int and E_FREE_FLOOR <= record['output_free'] < 2**64
                    and record.get('passed') is True, 'resource observation differs or failed')
            records.append(record)
    require(len(records) == 2 * calls, 'resource observations do not cover all calls')
    expected = {'checks': len(records), 'checked_calls': calls,
                'min_source_free': min((r['source_free'] for r in records), default=None),
                'min_output_free': min((r['output_free'] for r in records), default=None)}
    require(exact(summary, expected), 'resource summary differs')
    return expected


def validate_phase_journal(phase, journal, *, previous_calls=0):
    require(phase.get('calls') == len(journal.calls) and phase.get('cumulative_calls') == previous_calls + len(journal.calls)
            and phase.get('candidate_sha256') == journal.candidate_sha
            and phase.get('final_state_sha256') == journal.candidate_state
            and exact(phase.get('initial_probe'), journal.initial_probe)
            and phase.get('events_sha256') == journal.events_hash
            and phase.get('logits_sha256') == journal.logits_hash,
            'phase count, candidate, initial probe or final state differs')


def validate_handoff(repo, path, dev_dir, dev_phase, dev_exit, source_revision, manifest_sha, scope_path):
    handoff = load_json(path)
    require(handoff.get('schema') == 'flan-weight-edit-handoff-v1'
            and handoff.get('source_revision') == source_revision, 'evaluation handoff source differs')
    artifacts = handoff.get('artifacts')
    require(type(artifacts) is dict and set(artifacts) ==
            {'result', 'events', 'logits', 'resources', 'phase', 'exit', 'review', 'stdout', 'stderr', 'scope'},
            'evaluation handoff inventory differs')
    public = handoff.get('public_revision')
    require(sha(git_blob(repo, public, MANIFEST_PATH)) == manifest_sha, 'published manifest differs')
    expected = {'result': Path(dev_dir) / 'result.json', 'events': Path(dev_dir) / 'evidence/events.jsonl',
                'logits': Path(dev_dir) / 'evidence/logits.f32', 'phase': Path(dev_dir) / 'phase.json',
                'exit': Path(dev_dir) / 'operator-exit.json', 'stdout': Path(dev_dir) / 'stdout.log',
                'stderr': Path(dev_dir) / 'stderr.log', 'scope': Path(scope_path),
                'resources': Path(dev_dir) / 'resources.jsonl'}
    for name, item in artifacts.items():
        require(type(item) is dict and hex_digest(item.get('sha256'))
                and type(item.get('local_path')) is str and Path(item['local_path']).is_absolute()
                and file_sha(item['local_path']) == item['sha256'],
                'published development handoff artifact differs: ' + name)
        parts = item.get('parts')
        if parts is None:
            digest, size = git_digest(repo, public, item.get('git_path'))
        else:
            require(type(parts) is list and 1 <= len(parts) <= 128, 'published part inventory differs')
            aggregate, size, names = hashlib.sha256(), 0, set()
            for index, part in enumerate(parts):
                require(type(part) is dict and type(part.get('index')) is int and part['index'] == index
                        and type(part.get('bytes')) is int and 0 < part['bytes'] <= 1024**2
                        and hex_digest(part.get('sha256')) and type(part.get('git_path')) is str
                        and part['git_path'] not in names, 'published part identity differs')
                names.add(part['git_path'])
                part_digest, part_size = git_digest(repo, public, part['git_path'], aggregate=aggregate,
                                                    limit=1024**2)
                require(part_digest == part['sha256'] and part_size == part['bytes'],
                        'published part bytes differ')
                size += part_size
            digest = aggregate.hexdigest()
        require(digest == item['sha256'] and size == Path(item['local_path']).stat().st_size,
                'published development bytes differ: ' + name)
        if name in expected:
            require(file_sha(expected[name]) == item['sha256'], 'handoff names a different development phase')
    review = load_json(artifacts['review']['local_path'])
    require(review.get('status') == 'passed' and review.get('source_revision') == source_revision
            and all(review.get(name + '_sha256') == artifacts[name]['sha256']
                    for name in ('result', 'events', 'logits', 'resources', 'phase', 'exit', 'stdout', 'stderr', 'scope'))
            and type(review.get('review_scope')) is list
            and all(type(value) is str for value in review['review_scope'])
            and {'saved-logits', 'saved-process', 'restoration'}.issubset(review['review_scope'])
            and artifacts['phase']['sha256'] == dev_exit['phase_sha256']
            and dev_phase.get('phase') == 'development', 'published development review differs')
    return {'public_revision': public, 'artifacts': {key: value['sha256'] for key, value in artifacts.items()},
            'scope': 'published artifact binding and recorded handoff; not independent launch chronology'}


def load_decoder(assets, *, expected_version):
    path = Path(assets) / 'tokenizer.json'
    require(file_sha(path) == TOKENIZER_SHA, 'tokenizer asset differs')
    data = load_json(path, limit=16 * 1024**2)
    added = data.get('added_tokens')
    require(type(added) is list and all(type(item) is dict and type(item.get('id')) is int
                                        and type(item.get('special')) is bool for item in added),
            'tokenizer special-token inventory differs')
    special = {item['id'] for item in added if item['special']}
    require({0, 1, 2}.issubset(special), 'tokenizer lacks fixed special IDs')
    import importlib.metadata
    import tokenizers  # Tokenizer only; never import model code.
    require(type(expected_version) is str and tokenizers.__version__ == expected_version
            and importlib.metadata.version('tokenizers') == expected_version,
            'tokenizer runtime version differs from the source manifest')
    Tokenizer = tokenizers.Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    require(normalized(decode([POSITIVE])) == 'positive' and normalized(decode([NEGATIVE])) == 'negative',
            'tokenizer label decoding differs')
    runtime = {'version': expected_version, 'python_module_sha256': file_sha(tokenizers.__file__),
               'scope': 'version and Python module observation; not native-code attestation'}
    return decode, special, tokenizer, runtime


def verify_tokenized_cases(cases, tokenizer):
    for case in cases:
        encoding = tokenizer.encode(case['text'], add_special_tokens=True)
        if not (encoding.ids == case['input_ids'] and encoding.attention_mask == case['attention_mask']
                and encoding.special_tokens_mask == case['special_tokens_mask']
                and [list(item) for item in encoding.offsets] == case['offset_mapping']
                and tokenizer.encode(case['expected'], add_special_tokens=False).ids + [EOS] == case['expected_ids']):
            raise CaseAlignmentError('case tokenization or offsets differ: ' + case['id'])


def verify_invalid_direction(root, phase, saved):
    require(phase.get('scientific_status') == 'ineligible' and type(phase.get('calls')) is int
            and phase['calls'] == 0 and phase.get('cumulative_calls') == 0
            and phase.get('candidate_sha256') is None and phase.get('initial_probe') is None
            and hex_digest(phase.get('final_state_sha256'))
            and type(saved) is dict and set(saved) == {'status', 'reason', 'detail'}
            and saved['status'] == 'ineligible' and saved['reason'] == 'invalid-direction'
            and type(saved['detail']) is str and 0 < len(saved['detail']) <= 1024,
            'invalid-direction phase differs')
    evidence = Path(root) / 'evidence'
    require((evidence / 'logits.f32').stat().st_size == 0, 'invalid direction contains logits')
    path = evidence / 'events.jsonl'
    require(path.stat().st_size <= 8192, 'invalid-direction journal exceeds limit')
    lines = path.read_bytes().splitlines(keepends=True)
    require(len(lines) == 3 and all(line.endswith(b'\n') for line in lines),
            'invalid-direction journal is incomplete')
    events = [strict_json(line) for line in lines]
    for i, when in enumerate(('before', 'after')):
        expected = {'event_index': i, 'kind': 'numeric_probe', 'metadata': {
            'stage': 'candidate-construction', 'when': when, 'subnormal_bits': 1,
            'midpoint_even': 1.0, 'midpoint_odd': 1.0 + 2**-22}}
        require(exact(events[i], expected), 'invalid-direction numeric observation differs')
    require(exact(events[2], {'event_index': 2, 'kind': 'ineligible', 'metadata': saved}),
            'invalid-direction journal outcome differs')


def phase_file_snapshot(root, scope_path, claim_path, *, has_evidence):
    root = Path(root)
    paths = [root / 'phase.json', root / 'operator-exit.json', root / 'operator-start.json',
             root / 'operator-spawn.json', root / 'stdout.log', root / 'stderr.log', Path(scope_path), Path(claim_path)]
    if has_evidence:
        paths += [root / 'result.json', root / 'evidence/events.jsonl', root / 'evidence/logits.f32',
                  root / 'resources.jsonl']
    return {path: file_sha(path) for path in paths}


def verify_unchanged(snapshot):
    for path, digest in snapshot.items():
        require(path.is_file() and file_sha(path) == digest, 'phase evidence changed during replay')


def audit(args):
    study_root = STUDY_ROOT.resolve()
    require(Path(args.development).resolve() == (study_root / 'fixed-attempt/development').resolve()
            and Path(args.development_scope).resolve().is_relative_to(study_root),
            'development attempt or scope is outside the fixed v2 study root')
    if args.evaluation is not None:
        require(Path(args.evaluation).resolve() == (study_root / 'fixed-attempt/evaluation').resolve()
                and Path(args.evaluation_scope).resolve().is_relative_to(study_root),
                'evaluation attempt or scope is outside the fixed v2 study root')
    require(file_sha(args.cases) == CASE_SHA, 'fixed case-file hash differs')
    cases = valid_cases(load_json(args.cases))
    manifest, manifest_sha = validate_manifest(args.repo, args.manifest, args.manifest_revision)
    numeric_mode()
    rows = checkpoint_rows(args.assets)
    numeric_mode()
    asset_manifest = load_json(Path(args.assets) / 'asset-manifest.json')
    decoder, special, tokenizer, tokenizer_runtime = load_decoder(
        args.assets, expected_version=manifest['runtime'].get('tokenizers'))
    numeric_mode()
    alignment_error = None
    try:
        verify_tokenized_cases(cases, tokenizer)
    except CaseAlignmentError as exc:
        alignment_error = exc
    dev_root = Path(args.development)
    dev_claim = dev_root.parent.parent / 'claims/development.json'
    dev_phase, dev_exit, dev_files = validate_closed_phase(
        dev_root, 'development', manifest, manifest_sha, asset_manifest,
        scope_path=args.development_scope, claim_path=dev_claim)
    dev_snapshot = phase_file_snapshot(dev_root, args.development_scope, dev_claim,
                                       has_evidence=dev_files is not None)
    common = {'schema': 'independent-flan-weight-edit-saved-audit-v1',
              'source_revision': manifest['source_revision'], 'case_sha256': CASE_SHA,
              'manifest_sha256': manifest_sha, 'protocol_sha256': PROTOCOL_SHA,
              'asset_manifest_sha256': ASSET_SHA, 'tokenizer_runtime': tokenizer_runtime,
              'scope': ('Independent saved-byte/numeric replay; process, state, publication timing and '
                        'external exposure custody remain recorded cooperative observations.')}
    if dev_files is None:
        require(alignment_error is not None, 'recorded tokenizer mismatch was not reproduced')
        require(args.evaluation is None, 'evaluation follows a pre-forward ineligible development')
        numeric_mode()
        verify_unchanged(dev_snapshot)
        return {**common, 'status': 'ineligible', 'scientific_status': 'ineligible',
                'reason': 'tokenizer-alignment', 'phase_sha256': file_sha(Path(args.development) / 'phase.json'),
                'exit_sha256': file_sha(Path(args.development) / 'operator-exit.json')}
    require(alignment_error is None, 'actual tokenizer mismatch contradicts the recorded phase')
    try:
        candidates, random_bytes = reconstruct_candidates(rows)
    except DirectionIneligible:
        require(args.evaluation is None, 'evaluation follows an invalid development direction')
        saved = load_json(dev_files['result_sha256'])
        verify_invalid_direction(dev_root, dev_phase, saved)
        numeric_mode()
        verify_unchanged(dev_snapshot)
        return {**common, 'status': 'ineligible', 'scientific_status': 'ineligible',
                'reason': 'invalid-direction', 'phase_sha256': dev_snapshot[dev_root / 'phase.json'],
                'exit_sha256': dev_snapshot[dev_root / 'operator-exit.json']}
    dev_journal = Journal(Path(args.development) / 'evidence', candidates, random_bytes, phase='development')
    validate_phase_journal(dev_phase, dev_journal)
    dev_saved = load_json(dev_files['result_sha256'])
    dev = development(dev_journal, cases, dev_saved, decoder, special)
    require(dev_phase.get('scientific_status') == dev_saved['status'], 'development scientific status differs')
    result = {**common, 'status': 'passed' if dev['status'] == 'complete' else 'ineligible',
              'review_scope': ['saved-logits', 'saved-process', 'restoration', 'saved-resources'],
              'scientific_status': dev['status'], 'phase_sha256': file_sha(Path(args.development) / 'phase.json'),
              'exit_sha256': file_sha(Path(args.development) / 'operator-exit.json'),
              'stdout_sha256': dev_snapshot[dev_root / 'stdout.log'],
              'stderr_sha256': dev_snapshot[dev_root / 'stderr.log'],
              'scope_sha256': dev_snapshot[Path(args.development_scope)],
              'result_sha256': dev_phase['result_sha256'], 'events_sha256': dev_phase['events_sha256'],
              'logits_sha256': dev_phase['logits_sha256'], 'resources_sha256': dev_phase['resources_sha256'],
              'resource_summary': dev_phase['resource_summary'], 'development': dev}
    if args.evaluation is None:
        require(args.handoff is None, 'handoff requires an evaluation phase')
        numeric_mode()
        verify_unchanged(dev_snapshot)
        return result
    require(dev['status'] == 'complete' and args.handoff is not None,
            'evaluation requires complete development and a published handoff')
    handoff = validate_handoff(args.repo, args.handoff, args.development, dev_phase, dev_exit,
                               manifest['source_revision'], manifest_sha, args.development_scope)
    eval_root = Path(args.evaluation)
    eval_claim = eval_root.parent.parent / 'claims/evaluation.json'
    eval_phase, _, eval_files = validate_closed_phase(
        eval_root, 'evaluation', manifest, manifest_sha, asset_manifest,
        scope_path=args.evaluation_scope, claim_path=eval_claim)
    require(eval_files is not None, 'evaluation has no completed evidence')
    eval_snapshot = phase_file_snapshot(eval_root, args.evaluation_scope, eval_claim, has_evidence=True)
    replay_mode = eval_phase.get('development_replay_numeric_mode')
    expected_mode = {'subnormal_bits': 1, 'midpoint_even': 1.0, 'midpoint_odd': 1.0 + 2**-22}
    require(type(replay_mode) is dict and exact(replay_mode.get('before'), expected_mode)
            and exact(replay_mode.get('after'), expected_mode), 'development handoff numeric replay differs')
    eval_journal = Journal(Path(args.evaluation) / 'evidence', candidates, random_bytes, phase='evaluation')
    validate_phase_journal(eval_phase, eval_journal, previous_calls=dev_phase['calls'])
    require(dev_journal.raw(dev_journal.initial_probe) == eval_journal.raw(eval_journal.initial_probe)
            and dev_journal.candidate_sha == eval_journal.candidate_sha
            and dev_journal.candidate_state == eval_journal.candidate_state,
            'development and evaluation checkpoint, candidate or clean probe differ')
    eval_saved = load_json(eval_files['result_sha256'])
    assessed = evaluation(eval_journal, cases, eval_saved, dev, decoder, special)
    require(eval_phase.get('scientific_status') == assessed['status'], 'evaluation scientific status differs')
    result.update({'scientific_status': assessed['status'], 'evaluation': assessed, 'handoff': handoff,
                   'evaluation_hashes': {'result_sha256': eval_phase['result_sha256'],
                     'events_sha256': eval_phase['events_sha256'], 'logits_sha256': eval_phase['logits_sha256'],
                     'resources_sha256': eval_phase['resources_sha256'],
                     'resource_summary': eval_phase['resource_summary'],
                     'phase_sha256': file_sha(Path(args.evaluation) / 'phase.json'),
                     'exit_sha256': file_sha(Path(args.evaluation) / 'operator-exit.json')}})
    numeric_mode()
    verify_unchanged(dev_snapshot)
    verify_unchanged(eval_snapshot)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('repo', 'manifest', 'cases', 'assets', 'development', 'development-scope', 'output'):
        parser.add_argument('--' + option, required=True, type=Path)
    parser.add_argument('--manifest-revision', required=True)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--evaluation-scope', type=Path)
    parser.add_argument('--handoff', type=Path)
    args = parser.parse_args(argv)
    require(all(path.is_absolute() for path in (args.repo, args.manifest, args.cases, args.assets,
                                                args.development, args.development_scope, args.output))
            and (args.evaluation is None or args.evaluation.is_absolute())
            and (args.evaluation_scope is None or args.evaluation_scope.is_absolute())
            and (args.handoff is None or args.handoff.is_absolute()), 'all paths must be absolute')
    require((args.evaluation is None) == (args.evaluation_scope is None),
            'evaluation and its scope must be supplied together')
    require((args.evaluation is None) == (args.handoff is None),
            'evaluation and its published handoff must be supplied together')
    require(not args.output.exists(), 'audit output must be a fresh path')
    output = args.output.resolve()
    protected = [args.repo.resolve(), args.assets.resolve(), args.development.resolve()]
    if args.evaluation is not None:
        protected.append(args.evaluation.resolve())
    require(not any(output.is_relative_to(root) for root in protected)
            and output not in {p.resolve() for p in (args.manifest, args.cases, args.development_scope,
                                                       args.evaluation_scope, args.handoff) if p is not None},
            'audit output overlaps protected inputs')
    try:
        report = audit(args)
    except Exception as exc:
        report = {'schema': 'independent-flan-weight-edit-saved-audit-v1', 'status': 'incomplete',
                  'error_type': type(exc).__name__, 'reason': str(exc)[:300]}
    encoded = canonical(report) + b'\n'
    with args.output.open('xb') as stream:
        require(stream.write(encoded) == len(encoded), 'short audit report write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': sha(encoded)}, sort_keys=True))
    return 0 if report['status'] in ('passed', 'ineligible') else 1


if __name__ == '__main__':
    raise SystemExit(main())
