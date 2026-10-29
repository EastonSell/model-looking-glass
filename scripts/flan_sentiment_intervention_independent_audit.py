"""Independently replay saved FLAN sentiment intervention evidence.

This reader never loads a checkpoint or invokes a model. It compares journal
bytes with caller-supplied, independently checked selected cases; source,
selection chronology, the owner's process observation and execution are
separate questions. Run only on a closed attempt.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import unicodedata

import torch

SIDES = ('P1', 'N1', 'P2', 'N2')
ARMS = ('opposite_adjective', 'same_adjective', 'opposite_instruction', 'opposite_subject')
OPP = {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
SAME = {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}
SITE = {'opposite_adjective': 'adjective', 'same_adjective': 'adjective',
        'opposite_instruction': 'instruction', 'opposite_subject': 'subject'}
PROBE = {'binary32_ties': [0x3f800000, 0x3f800002],
         'float32_denormals': [2, 4194304, -2147483646, -2143289344],
         'float64_denormals': [2, 2251799813685248, -9223372036854775806, -9221120237041090560]}
INSTRUCTIONS = ('Classify the sentiment of this review as positive or negative:',
                'Is this review positive or negative?')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def reject(_):
        raise ValueError('nonfinite JSON constant')
    def number(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject, parse_float=number)


def number_equal(actual, expected, message):
    require(type(actual) in (int, float) and type(expected) in (int, float), message)
    require(math.isfinite(actual) and math.isfinite(expected), message)
    require(struct.pack('<d', float(actual)) == struct.pack('<d', float(expected)), message)


def typed_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(typed_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(typed_equal(x, y) for x, y in zip(left, right))
    if type(left) is float:
        return struct.pack('<d', left) == struct.pack('<d', right)
    return left == right


def probe_check(value):
    require(type(value) is dict and set(value) == set(PROBE), 'numeric probe schema')
    for name, target in PROBE.items():
        require(type(value[name]) is list and len(value[name]) == len(target), 'numeric probe length')
        require(all(type(x) is int and x == y for x, y in zip(value[name], target)), 'numeric probe value')


def check_local_arithmetic():
    # Guard the reader before JSON decoding: DAZ/FTZ changes float parsing and
    # later calculations. Do not silently change the ambient numeric mode.
    a = torch.tensor([1, 0x00800000, -2147483647, -2139095040], dtype=torch.int32).view(torch.float32)
    b = torch.tensor([1, 0x0010000000000000, -9223372036854775807,
                      -9218868437227405312], dtype=torch.int64).view(torch.float64)
    for values, itype, sign, minimum in ((a, torch.int32, 1 << 31, 0x00800000),
                                         (b, torch.int64, 1 << 63, 0x0010000000000000)):
        observed = [int((values[0] * 2).view(itype).item()), int((values[1] * .5).view(itype).item()),
                    int((values[2] * 2).view(itype).item()), int((values[3] * .5).view(itype).item())]
        require(observed == [2, minimum // 2, 2 - sign, minimum // 2 - sign], 'reader FTZ/DAZ')
    tie = torch.tensor([0x3f800000, 0x3f800001], dtype=torch.int32).view(torch.float32)
    require((tie + math.ldexp(1.0, -24)).view(torch.int32).tolist() == PROBE['binary32_ties'],
            'reader binary32 rounding mode')
    require(1.0 + math.ldexp(1.0, -53) == 1.0 and
            (1.0 + math.ldexp(3.0, -53)) == (1.0 + math.ldexp(1.0, -51)),
            'reader binary64 rounding mode')


def load_oracle():
    path = Path(__file__).resolve().parent / 'flan_sentiment_dose_oracle.py'
    spec = importlib.util.spec_from_file_location('independent_flan_dose_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorBank:
    def __init__(self, attempt):
        self.attempt = attempt
        self.identities = {}

    def read(self, ref, shape=None):
        require(type(ref) is dict and set(ref) == {'path', 'shape', 'bytes', 'sha256', 'dtype'},
                'tensor reference schema')
        path, dims = ref['path'], ref['shape']
        require(type(path) is str and re.fullmatch(r'tensors/[0-9]{6}\.f32le', path) is not None,
                'tensor path')
        require(type(dims) is list and 1 <= len(dims) <= 4 and
                all(type(x) is int and 0 < x <= 32128 for x in dims), 'tensor shape')
        if shape is not None:
            require(dims == list(shape), 'unexpected tensor shape')
        size = 4 * math.prod(dims)
        require(type(ref['bytes']) is int and ref['bytes'] == size and size <= 900000 and
                ref['dtype'] == 'float32-le' and type(ref['sha256']) is str and
                re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None, 'tensor metadata')
        identity = (tuple(dims), size, ref['sha256'])
        require(path not in self.identities or self.identities[path] == identity, 'conflicting tensor references')
        target = self.attempt / path
        require(not target.is_symlink() and target.is_file() and target.stat().st_size == size,
                'missing or aliased tensor')
        raw = target.read_bytes()
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'tensor digest')
        if path not in self.identities:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32)
            require(bool(torch.isfinite(tensor).all().item()), 'nonfinite saved tensor')
        self.identities[path] = identity
        return raw

    def inventory(self):
        folder = self.attempt / 'tensors'
        require(folder.is_dir() and not folder.is_symlink(), 'tensor directory')
        entries = list(folder.iterdir())
        require(all(p.is_file() and not p.is_symlink() for p in entries), 'invalid tensor directory entry')
        names = {'tensors/' + p.name for p in entries}
        require(names == set(self.identities), 'unreferenced or missing tensor')


def words(raw):
    return [item[0] for item in struct.iter_unpack('<I', raw)]


def values(raw, oracle):
    return [oracle.helper.f32(item) for item in words(raw)]


def rotate(items, n):
    return items[n:] + items[:n]


def expected_schedule():
    groups = [(u, i, rotate(SIDES, (u + i) % 4)) for u in range(4)
              for i in ((0, 1) if u % 2 == 0 else (1, 0))]
    for u, i, order in groups:
        for side in order:
            yield {'kind': 'clean', 'unit': u, 'instruction': i, 'receiver': side}
    for u, i, order in groups:
        for side in order:
            common = {'unit': u, 'instruction': i, 'receiver': side}
            yield {'kind': 'passthrough', **common}
            for site in ('adjective', 'instruction', 'subject'):
                yield {'kind': 'self', 'site': site, **common}
            yield {'kind': 'full_swap', 'donor': OPP[side], **common}
    for unit in range(4):
        for u, i, order in groups:
            if u != unit:
                continue
            for side in order:
                r = SIDES.index(side)
                for a, name in rotate(tuple(enumerate(ARMS)), (u + i + r) % 4):
                    levels = ('low', 'high') if (u + i + r + a) % 2 == 0 else ('high', 'low')
                    for dose in levels:
                        yield {'kind': 'matched', 'unit': u, 'instruction': i,
                               'receiver': side, 'action': name, 'dose': dose}
                yield {'kind': 'natural', 'unit': u, 'instruction': i,
                       'receiver': side, 'donor': OPP[side]}
        yield {'kind': 'repeat', 'unit': unit, 'instruction': 0 if unit % 2 == 0 else 1,
               'receiver': 'P1'}


def matches_step(action, step):
    require(type(action) is dict, 'action record type')
    for key, value in step.items():
        require(type(action.get(key)) is type(value) and action[key] == value, 'wrong action identity')


def key_of(action):
    kind = action.get('kind')
    return (kind, action.get('unit'), action.get('instruction'), action.get('receiver'),
            action.get('site') if kind == 'self' else None,
            action.get('action') if kind == 'matched' else None,
            action.get('dose') if kind == 'matched' else None)


def numeric_tree(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('numeric_before', 'numeric_after', 'margin_numeric_before', 'margin_numeric_after'):
                probe_check(item)
            else:
                numeric_tree(item)
    elif isinstance(value, list):
        for item in value:
            numeric_tree(item)


def crosscheck(a, b, label):
    if isinstance(b, dict):
        require(type(a) is dict and set(a) == set(b), label + ': object keys')
        for k in b:
            crosscheck(a[k], b[k], label + '.' + k)
    elif isinstance(b, list):
        require(type(a) is list and len(a) == len(b), label + ': list length')
        for i, (x, y) in enumerate(zip(a, b)):
            crosscheck(x, y, label + f'[{i}]')
    elif type(b) is float:
        number_equal(a, b, label)
    else:
        require(type(a) is type(b) and a == b, label)


def check_case_groups(groups, vocab, tokenizer):
    require(type(groups) is list and len(groups) == 8, 'expected eight selected groups')
    output = {}
    specials = set(tokenizer.all_special_ids)
    for group in groups:
        require(type(group) is dict and type(group.get('unit')) is int and 0 <= group['unit'] < 4 and
                type(group.get('instruction')) is int and group['instruction'] in (0, 1) and
                type(group.get('cases')) is dict and set(group['cases']) == set(SIDES), 'selected group')
        key = (group['unit'], group['instruction'])
        require(key not in output, 'duplicate selected group')
        reference, adjectives = None, []
        for side in SIDES:
            case = group['cases'][side]
            require(type(case) is dict and case.get('side') == side, 'case identity')
            ids, mask = case.get('input_ids'), case.get('attention_mask')
            special_mask = case.get('special_tokens_mask')
            require(type(ids) is list and 3 <= len(ids) <= 64 and
                    all(type(x) is int and 0 <= x < vocab for x in ids) and
                    ids[-1] == 1 and ids.count(1) == 1, 'selected input IDs')
            require(type(mask) is list and len(mask) == len(ids) and
                    all(type(x) is int and x == 1 for x in mask), 'selected attention mask')
            require(type(special_mask) is list and len(special_mask) == len(ids) and
                    all(type(x) is int and x in (0, 1) for x in special_mask), 'selected special mask')
            subject, adjective = case.get('subject_index'), case.get('adjective_index')
            require(type(subject) is int and type(adjective) is int and
                    0 <= subject < len(ids) and 0 <= adjective < len(ids) and
                    len({0, subject, adjective}) == 3 and
                    all(ids[n] not in specials and special_mask[n] == 0 for n in (0, subject, adjective)),
                    'selected token positions')
            static = (tuple(ids[:adjective] + ids[adjective + 1:]), tuple(mask), tuple(special_mask),
                      subject, adjective)
            require(reference is None or static == reference, 'selected group differs outside adjective')
            reference = static
            adjectives.append(ids[adjective])
        require(len(set(adjectives)) == 4, 'selected adjective IDs not distinct')
        output[key] = group['cases']
    require(len(output) == 8, 'missing selected group')
    return output


def selected_groups(selection, salt, bindings, tokenizer, *, retokenize=True):
    """Derive the fixed groups from the released, committed selection record.

    This does not establish the pool's semantic quality, exposure coverage,
    selection chronology, or validity of nonselected tokenizer screens.
    """
    require(type(selection) is dict and set(selection) == {'schema', 'pool_commitment', 'protocol_sha256',
        'runner_revision', 'asset_manifest_sha256', 'exposure_sources', 'unit_screens', 'selected_units',
        'donor_mapping'}, 'selection record schema')
    require(type(salt) is bytes and len(salt) == 32 and type(bindings) is dict, 'selection salt/bindings')
    canonical = json.dumps(selection, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                           allow_nan=False).encode('utf8')
    require(len(canonical) <= 8 * 1024**2, 'selection too large')
    digest = hashlib.sha256(b'FLAN-SENTIMENT-SELECTION-V1\0' + salt + b'\0' + canonical).hexdigest()
    require(digest == bindings.get('selection') and selection['pool_commitment'] == bindings.get('pool') and
            selection['runner_revision'] == bindings.get('runner_revision') and
            selection['schema'] == 'flan-sentiment-private-selection-v1' and
            selection['protocol_sha256'] == '83d666bc67af0951d21eceaa8efb8e7cb16f53a40aefdb72eff03f8b57bd7813' and
            selection['asset_manifest_sha256'] == 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083',
            'selection commitment/identity mismatch')
    screens = selection['unit_screens']
    require(type(screens) is list and len(screens) == 12, 'selection screen inventory')
    eligible, by_id = [], {}
    for n, screen in enumerate(screens):
        require(type(screen) is dict and screen.get('unit_id') == f'u{n:02d}' and
                type(screen.get('semantic_ok')) is bool and type(screen.get('lexical_exclusion_ok')) is bool and
                type(screen.get('instructions')) is list and len(screen['instructions']) == 2,
                'screened unit identity')
        for i, group in enumerate(screen['instructions']):
            require(type(group) is dict and type(group.get('instruction_index')) is int and
                    group['instruction_index'] == i and type(group.get('eligible')) is bool and
                    type(group.get('failure_reasons')) is list and
                    all(type(x) is str and bool(x) for x in group['failure_reasons']) and
                    group['eligible'] == (not group['failure_reasons']) and
                    (not group['eligible'] or (screen['semantic_ok'] and screen['lexical_exclusion_ok'])) and
                    type(group.get('cases')) is list and len(group['cases']) == 4 and
                    all(type(case) is dict and case.get('side') == side
                        for case, side in zip(group['cases'], SIDES)), 'screened group')
        if screen['semantic_ok'] and screen['lexical_exclusion_ok'] and all(
                group['eligible'] for group in screen['instructions']):
            eligible.append(screen['unit_id'])
        by_id[screen['unit_id']] = screen
    chosen = eligible[:4]
    require(len(chosen) == 4 and typed_equal(selection['selected_units'], chosen), 'first-four selection')
    mapping = [{'unit_id': ident, 'opposite': dict(OPP), 'same': dict(SAME)} for ident in chosen]
    require(typed_equal(selection['donor_mapping'], mapping), 'selection donor mapping')
    result, used_subjects, used_adjectives = [], set(), set()
    for unit, ident in enumerate(chosen):
        expected_subject, expected_words = None, {}
        for instruction, group in enumerate(by_id[ident]['instructions']):
            cases = {case['side']: case for case in group['cases']}
            if retokenize:
                for side in SIDES:
                    case = cases[side]
                    text = case.get('text')
                    require(type(text) is str and unicodedata.normalize('NFC', text) == text, 'selected case text')
                    encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
                    for name in ('input_ids', 'attention_mask', 'special_tokens_mask', 'offset_mapping'):
                        raw = encoded[name]
                        actual = [list(x) if isinstance(x, tuple) else x for x in raw]
                        require(typed_equal(actual, case.get(name)), 'selected tokenizer mismatch: ' + name)
                    prefix = INSTRUCTIONS[instruction] + ' The '
                    require(text.startswith(prefix), 'selected instruction/template mismatch')
                    match = re.fullmatch(r'([a-z]{1,24}) was ([a-z]{1,24})\.', text[len(prefix):])
                    require(match is not None, 'selected subject/adjective text')
                    subject, adjective = match.groups()
                    require(expected_subject is None or expected_subject == subject, 'cross-template subject mismatch')
                    require(side not in expected_words or expected_words[side] == adjective,
                            'cross-template adjective mismatch')
                    expected_subject, expected_words[side] = subject, adjective
                    starts = (len(prefix), len(prefix) + len(subject) + len(' was '))
                    spans = ((starts[0], starts[0] + len(subject)), (starts[1], starts[1] + len(adjective)))
                    offsets = case.get('offset_mapping')
                    require(type(offsets) is list and len(offsets) == len(case['input_ids']), 'offset inventory')
                    for name, (start, end) in zip(('subject_index', 'adjective_index'), spans):
                        overlapping = [i for i, pair in enumerate(offsets) if type(pair) is list and len(pair) == 2
                                       and all(type(x) is int for x in pair) and pair[0] < end and pair[1] > start]
                        valid = [[start, end]] + ([[start - 1, end]] if start and text[start - 1] == ' ' else [])
                        require(len(overlapping) == 1 and type(case.get(name)) is int and
                                case[name] == overlapping[0] and offsets[overlapping[0]] in valid,
                                'selected ' + name + ' is not the declared word span')
            result.append({'unit': unit, 'instruction': instruction, 'cases': cases})
        if retokenize:
            require(expected_subject not in (used_subjects | used_adjectives) and len(expected_words) == 4 and
                    len(set(expected_words.values())) == 4 and
                    not (set(expected_words.values()) & (used_subjects | used_adjectives | {expected_subject})),
                    'selected lexical material is repeated')
            used_subjects.add(expected_subject)
            used_adjectives.update(expected_words.values())
    return result


def label_targets(tokenizer, vocab):
    result = {}
    special = set(tokenizer.all_special_ids)
    for label in ('positive', 'negative'):
        lexical = tokenizer.encode(label, add_special_tokens=False)
        require(type(lexical) is list and 1 <= len(lexical) <= 4 and
                all(type(x) is int and 0 <= x < vocab and x not in special for x in lexical),
                'label tokenization')
        result[label] = lexical + [1]
    require(result['positive'] != result['negative'], 'same label tokenization')
    return result


def decode_answer(ids, side, tokenizer):
    if not ids or ids[-1] != 1 or ids.count(1) != 1 or any(x in set(tokenizer.all_special_ids) for x in ids[:-1]):
        return False
    text = tokenizer.decode(ids[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    expected = 'positive' if side.startswith('P') else 'negative'
    return unicodedata.normalize('NFC', text).strip().casefold() == expected


def audit_evaluation(row, components, bank, tokenizer, oracle, targets, width, vocab):
    greedy, generated = row.get('greedy'), row.get('generated_ids')
    require(type(greedy) is list and 1 <= len(greedy) <= 4 and type(generated) is list and
            len(generated) == len(greedy) and all(type(x) is int and 0 <= x < vocab for x in generated),
            'generated token structure')
    require(1 not in generated[:-1] and (len(generated) == 4 or generated[-1] == 1), 'greedy stop rule')
    require(len(components) == len(greedy) + 2, 'decoder component count')
    ordered_refs = []
    for j, step in enumerate(greedy):
        require(type(step) is dict and set(step) == {'prefix', 'emitted', 'logits'} and
                typed_equal(step['prefix'], [0] + generated[:j]) and type(step['emitted']) is int and
                step['emitted'] == generated[j], 'greedy prefix')
        raw = bank.read(step['logits'], (1, vocab))
        logits = values(raw, oracle)
        chosen = max(range(vocab), key=logits.__getitem__)
        require(chosen == generated[j], 'greedy argmax')
        start, returned = components[j]
        require(start.get('kind') == 'decoder' and start.get('phase') == 'greedy' and
                start.get('side') == row['side'] and typed_equal(start.get('decoder_ids'), step['prefix']) and
                typed_equal(start.get('decoder_mask'), [1] * len(step['prefix'])) and
                typed_equal(returned.get('prefix'), step['prefix']) and returned.get('phase') == 'greedy' and
                typed_equal(returned.get('logits'), step['logits']), 'greedy component binding')
        ordered_refs.append(step['logits'])
    scores, ranks = {}, {}
    teacher = row.get('teacher')
    require(type(teacher) is dict and set(teacher) == {'positive', 'negative'}, 'teacher labels')
    for j, label in enumerate(('positive', 'negative')):
        target, record = targets[label], teacher[label]
        require(type(record) is dict and typed_equal(record.get('target_ids'), target) and
                typed_equal(record.get('decoder_ids'), [0] + target[:-1]) and
                typed_equal(record.get('decoder_mask'), [1] * len(target)), 'teacher prefix')
        raw = bank.read(record['logits'], (1, len(target), vocab))
        decoded = values(raw, oracle)
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(len(target), vocab)
        logp = torch.log_softmax(tensor, dim=-1)
        selected, computed_ranks = [], []
        for p, token in enumerate(target):
            bits = int(logp[p, token].view(torch.int32).item()) & 0xffffffff
            chosen = oracle.helper.f32(bits)
            require(math.isfinite(chosen), 'teacher score nonfinite')
            selected.append(chosen)
            logits = decoded[p * vocab:(p + 1) * vocab]
            computed_ranks.append(1 + sum(x > logits[token] for x in logits))
            maximum = max(logits)
            exact = logits[token] - maximum - math.log(math.fsum(math.exp(x - maximum) for x in logits))
            require(math.isclose(chosen, exact, rel_tol=1e-6, abs_tol=1e-5), 'teacher mathematical check')
        require(type(record.get('selected_logprobs')) is list and
                len(record['selected_logprobs']) == len(selected), 'teacher selected scores')
        for x, y in zip(record['selected_logprobs'], selected):
            number_equal(x, y, 'teacher selected value')
        require(type(record.get('ranks')) is list and len(record['ranks']) == len(computed_ranks) and
                all(type(x) is int and x == y for x, y in zip(record['ranks'], computed_ranks)), 'teacher ranks')
        scores[label], ranks[label] = math.fsum(selected), computed_ranks
        number_equal(record.get('score'), scores[label], 'teacher sum')
        probe_check(record.get('numeric_before'))
        probe_check(record.get('numeric_after'))
        start, returned = components[len(greedy) + j]
        require(start.get('kind') == 'decoder' and start.get('phase') == 'teacher_' + label and
                start.get('side') == row['side'] and typed_equal(start.get('decoder_ids'), record['decoder_ids']) and
                typed_equal(start.get('decoder_mask'), record['decoder_mask']) and
                typed_equal(returned.get('prefix'), record['decoder_ids']) and
                returned.get('phase') == 'teacher_' + label and
                typed_equal(returned.get('logits'), record['logits']), 'teacher component binding')
        ordered_refs.append(record['logits'])
    require(typed_equal(row.get('all_logit_refs'), ordered_refs), 'decoder full-logit inventory')
    margin = scores['positive'] - scores['negative']
    number_equal(row.get('margin'), margin, 'score margin')
    require(type(row.get('strict_correct')) is bool and
            row['strict_correct'] == decode_answer(generated, row['side'], tokenizer), 'strict answer')
    probe_check(row.get('margin_numeric_before'))
    probe_check(row.get('margin_numeric_after'))
    return {'scores': scores, 'ranks': ranks, 'margin': margin, 'strict': row['strict_correct'],
            'generated': generated, 'logit_refs': ordered_refs}


def total_variation(first, second, bank, oracle):
    left = values(bank.read(first['greedy'][0]['logits']), oracle)
    right = values(bank.read(second['greedy'][0]['logits']), oracle)
    require(len(left) == len(right), 'TV vocabulary mismatch')
    ma, mb = max(left), max(right)
    a = [math.exp(x - ma) for x in left]
    b = [math.exp(x - mb) for x in right]
    za, zb = math.fsum(a), math.fsum(b)
    return (0.5 * math.fsum(abs(x / za - y / zb) for x, y in zip(a, b)),
            max(range(len(left)), key=left.__getitem__), max(range(len(right)), key=right.__getitem__))


def segment(raw, position, width):
    require(type(position) is int and position >= 0 and len(raw) >= 4 * width * (position + 1),
            'site outside encoder witness')
    return raw[4 * width * position:4 * width * (position + 1)]


def audit_geometry(record, state, cases, width, bank, oracle):
    probe_check(record.get('numeric_before'))
    probe_check(record.get('numeric_after'))
    sitepos = {'instruction': 0, 'subject': cases['P1']['subject_index'],
               'adjective': cases['P1']['adjective_index']}
    vectors = {side: {site: words(segment(state[side], pos, width)) for site, pos in sitepos.items()}
               for side in SIDES}
    distances, directions = {}, {}
    for side in SIDES:
        for arm in ARMS:
            site = SITE[arm]
            donor = SAME[side] if arm == 'same_adjective' else OPP[side]
            own, other = vectors[side][site], vectors[donor][site]
            key = side + ':' + arm
            distances[key] = oracle.distance(own, other)
            directions[key] = [oracle.helper.f32(b) - oracle.helper.f32(a) for a, b in zip(own, other)]
    require(type(record.get('distances')) is dict and set(record['distances']) == set(distances),
            'geometry distances inventory')
    for key, value in distances.items():
        number_equal(record['distances'][key], value, 'natural distance')
    radius = min(distances.values())
    number_equal(record.get('q'), radius, 'shared radius')
    for side in SIDES:
        a, b = side + ':opposite_adjective', side + ':same_adjective'
        cosine = (math.fsum(x * y for x, y in zip(directions[a], directions[b])) /
                  (distances[a] * distances[b])) if distances[a] and distances[b] else None
        expected = {'value': cosine, 'reason': 'zero donor norm' if cosine is None else None}
        crosscheck(record['donor_cosines'][side], expected, 'donor cosine')
        ratio = None if distances[a] == 0 else radius / distances[a]
        crosscheck(record['q_over_word_distance'][side], ratio, 'radius/natural word distance')
    expected_landings = {}
    if radius / 2 <= 1e-6:
        require(record.get('status') == 'dose-ineligible' and record.get('landings') == {},
                'small radius must make dose ineligible')
        return {'status': 'dose-ineligible', 'q': radius, 'landings': {}, 'sitepos': sitepos}
    for side in SIDES:
        for arm in ARMS:
            site = SITE[arm]
            donor = SAME[side] if arm == 'same_adjective' else OPP[side]
            own, other = vectors[side][site], vectors[donor][site]
            for level, target in (('low', radius / 2), ('high', radius)):
                result = oracle.proposed(own, other, target)
                raw = b''.join(struct.pack('<I', x) for x in result['post_bits'])
                base = oracle.norm([oracle.helper.f32(x) for x in own])
                tolerance = min(max(1e-5, 1e-5 * target), .01 * target)
                if result['delivered'] == 0:
                    reason = 'zero achieved displacement'
                elif abs(result['delivered'] - target) > tolerance:
                    reason = 'landing norm outside tolerance'
                elif result['cosine'] is None or result['cosine'] < .99:
                    reason = 'landing cosine below threshold'
                else:
                    reason = None
                key = side + ':' + arm + ':' + level
                expected_landings[key] = {'receiver': side, 'action': arm, 'site': site, 'donor': donor,
                    'level': level, 'raw': raw, 'target_dose': target, 'fraction': result['fraction'],
                    'achieved_norm': result['delivered'], 'cosine': result['cosine'], 'tolerance': tolerance,
                    'receiver_norm': base, 'achieved_over_receiver_norm': None if base == 0 else result['delivered'] / base,
                    'ratio_reason': 'zero receiver norm' if base == 0 else None, 'failure': reason}
    for side in SIDES:
        for level in ('low', 'high'):
            reference = expected_landings[side + ':opposite_adjective:' + level]['achieved_norm']
            for arm in ARMS[1:]:
                item = expected_landings[side + ':' + arm + ':' + level]
                if abs(item['achieved_norm'] - reference) > .01 * reference:
                    item['failure'] = item['failure'] or 'control norm differs from opposite adjective'
    status = 'dose-unmatchable' if any(x['failure'] is not None for x in expected_landings.values()) else 'matchable'
    require(record.get('status') == status and type(record.get('landings')) is dict and
            set(record['landings']) == set(expected_landings), 'landing status/inventory')
    for key, item in expected_landings.items():
        stored = record['landings'][key]
        expected = {k: v for k, v in item.items() if k != 'raw'}
        require(type(stored) is dict and set(stored) == set(expected) | {'value_ref'}, 'landing fields')
        require(bank.read(stored['value_ref'], (width,)) == item['raw'], 'rounded source bits')
        crosscheck({k: stored[k] for k in expected}, expected, 'landing metadata')
    return {'status': status, 'q': radius, 'landings': expected_landings, 'sitepos': sitepos,
            'distances': distances}


def expected_decision(clean, action):
    movements = {}
    for arm in ARMS:
        movements[arm] = {side: (clean[side] - action[arm][side] if side.startswith('P')
                                  else action[arm][side] - clean[side]) for side in SIDES}
    a = math.fsum(movements[ARMS[0]][side] for side in SIDES) / 4
    controls = {arm: math.fsum(abs(movements[arm][side]) for side in SIDES) / 4 for arm in ARMS[1:]}
    b = max(controls.values())
    count = sum(movements[ARMS[0]][side] > 0 for side in SIDES)
    return {'status': 'measured', 'movements': movements, 'A': a, 'B': b,
            'control_means': controls, 'positive_count': count,
            'passes': bool(a >= .10 and a - b >= .05 and count >= 3)}


def read_records(attempt):
    terminal_path, events_path = attempt / 'terminal.json', attempt / 'events.jsonl'
    require((attempt / 'started.json').is_file(), 'missing started record')
    started = strict_json((attempt / 'started.json').read_bytes())
    require(type(started) is dict, 'started record')
    terminal = strict_json(terminal_path.read_bytes()) if terminal_path.is_file() else None
    if terminal is None or terminal.get('status') != 'complete_candidate':
        return started, terminal, None
    require(terminal.get('active_component') is None and terminal.get('storage_accounting_complete') is True and
            type(terminal.get('components')) is int and 0 < terminal['components'] <= 2940,
            'complete terminal component state')
    require(events_path.is_file() and not events_path.is_symlink() and
            events_path.stat().st_size <= 64 * 1024**2, 'journal file/size')
    events = []
    with events_path.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n') and 0 < len(line) <= 65536, 'journal framing')
            value = strict_json(line)
            require(type(value) is dict and set(value) == {'event', 'detail'} and
                    type(value['event']) is str and type(value['detail']) is dict, 'journal event')
            events.append(value)
    require(events, 'empty journal')
    return started, terminal, events


def parsed_events(events, terminal, groups, tokenizer, oracle, bank, width, vocab, synthetic):
    targets = label_targets(tokenizer, vocab)
    allowed = {'owner_contract', 'runtime_observation', 'label_targets', 'component_start', 'component_return',
               'evaluation_start', 'evaluation_return', 'control_verified', 'output_comparison',
               'group_geometry', 'baseline_support', 'action_skipped', 'group_outcome', 'primary_numeric'}
    records = {name: [] for name in allowed}
    rows, flow, active, pending_encoder, evaluation = [], [], None, None, None
    geometry_ready, support_ready = set(), set()
    pending_report = None
    component_index = 0
    for event in events:
        name, detail = event['event'], event['detail']
        require(name in allowed, 'unexpected journal event: ' + name)
        numeric_tree(detail)
        records[name].append(detail)
        if pending_report is not None:
            require(name == pending_report, 'control/comparison record is out of order')
            pending_report = 'output_comparison' if name == 'control_verified' else None
        if name == 'component_start':
            require(active is None and type(detail.get('index')) is int and detail['index'] == component_index,
                    'component start order')
            require(detail.get('kind') in ('encoder', 'decoder'), 'component kind')
            require((detail['kind'] == 'decoder') == (evaluation is not None), 'component/evaluation pairing')
            probe_check(detail.get('numeric_before'))
            active = detail
            component_index += 1
        elif name == 'component_return':
            require(active is not None and type(detail.get('index')) is int and
                    detail['index'] == active['index'], 'component return order')
            probe_check(detail.get('numeric_after'))
            if active['kind'] == 'encoder':
                require(pending_encoder is None and set(detail) == {'index', 'numeric_after', 'state'},
                        'encoder return fields')
                bank.read(detail['state'])
                pending_encoder = (active, detail)
            else:
                require(set(detail) == {'index', 'numeric_after', 'logits', 'prefix', 'phase'},
                        'decoder return fields')
                bank.read(detail['logits'])
                evaluation['components'].append((active, detail))
            active = None
        elif name == 'evaluation_start':
            require(active is None and evaluation is None and
                    set(detail) == {'action', 'side', 'input_ids', 'attention_mask', 'before', 'after'},
                    'evaluation start fields')
            action = detail['action']
            require(type(action) is dict and type(action.get('unit')) is int and
                    type(action.get('instruction')) is int and action.get('receiver') in SIDES and
                    (action['unit'], action['instruction']) in groups and detail['side'] == action['receiver'],
                    'evaluation group identity')
            if action.get('kind') in ('matched', 'natural'):
                groupkey = (action['unit'], action['instruction'])
                require(groupkey in geometry_ready and groupkey in support_ready,
                        'action preceded saved geometry or baseline support')
            case = groups[(action['unit'], action['instruction'])][detail['side']]
            require(typed_equal(detail['input_ids'], case['input_ids']) and
                    typed_equal(detail['attention_mask'], case['attention_mask']),
                    'evaluation selected model inputs')
            before = bank.read(detail['before'], (1, len(case['input_ids']), width))
            bank.read(detail['after'], (1, len(case['input_ids']), width))
            if action.get('kind') in ('clean', 'repeat'):
                require(pending_encoder is not None, 'missing fresh encoder')
                enc_start, enc_return = pending_encoder
                require(enc_start.get('side') == detail['side'] and
                        typed_equal(enc_start.get('input_ids'), detail['input_ids']) and
                        typed_equal(enc_start.get('attention_mask'), detail['attention_mask']) and
                        typed_equal(action.get('encoder_state'), enc_return['state']) and
                        bank.read(enc_return['state']) == before, 'encoder input/state binding')
                pending_encoder = None
            else:
                require(pending_encoder is None, 'unexpected encoder before action')
            evaluation = {'start': detail, 'components': []}
        elif name == 'evaluation_return':
            require(active is None and evaluation is not None, 'unpaired evaluation return')
            start = evaluation['start']
            require(typed_equal(detail.get('action'), start['action']) and detail.get('side') == start['side'] and
                    typed_equal(detail.get('before'), start['before']) and
                    typed_equal(detail.get('after'), start['after']),
                    'evaluation return identity')
            analysis = audit_evaluation(detail, evaluation['components'], bank, tokenizer, oracle,
                                        targets, width, vocab)
            row = {'start': start, 'result': detail, 'analysis': analysis}
            rows.append(row)
            flow.append(('evaluation', row))
            evaluation = None
            if detail['action']['kind'] != 'clean':
                pending_report = ('control_verified' if detail['action']['kind'] in
                                  ('passthrough', 'self', 'full_swap', 'repeat') else 'output_comparison')
        elif name == 'action_skipped':
            require(active is None and evaluation is None and set(detail) == {'action', 'reason'}, 'skip record')
            action = detail['action']
            require(type(action) is dict and (action.get('unit'), action.get('instruction')) in support_ready and
                    (action.get('unit'), action.get('instruction')) in geometry_ready,
                    'skip preceded saved geometry or baseline support')
            flow.append(('skipped', detail))
        elif name == 'group_geometry':
            groupkey = (detail.get('unit'), detail.get('instruction'))
            require(groupkey in groups and groupkey not in geometry_ready and len(flow) >= 192,
                    'geometry identity/order')
            geometry_ready.add(groupkey)
        elif name == 'baseline_support':
            groupkey = (detail.get('unit'), detail.get('instruction'))
            require(groupkey in geometry_ready and groupkey not in support_ready,
                    'baseline support identity/order')
            support_ready.add(groupkey)
        elif name == 'group_outcome':
            require(len(flow) == 484, 'group outcome precedes scheduled actions')
        elif name == 'primary_numeric':
            require(len(records['group_outcome']) == 8, 'primary precedes group outcomes')
    require(active is None and evaluation is None and pending_encoder is None and pending_report is None and
            component_index == terminal['components'], 'unfinished/missing components')
    require(len(records['runtime_observation']) == 1 and len(records['label_targets']) == 1 and
            typed_equal(records['label_targets'][0], {'targets': targets, 'decoder_start_id': 0, 'eos_id': 1}),
            'runtime/label records')
    require((len(records['owner_contract']) == 0 if synthetic else
             len(records['owner_contract']) == 1 and events[0]['event'] == 'owner_contract'),
            'owner contract count/order')
    require(type(records['runtime_observation'][0].get('testing_only')) is bool and
            records['runtime_observation'][0]['testing_only'] is synthetic and
            type(terminal.get('detail', {}).get('testing_only')) is bool and
            terminal['detail']['testing_only'] is synthetic, 'synthetic/production mode mismatch')
    probe_check(records['runtime_observation'][0].get('numeric'))
    return records, rows, flow


def indexed_by_group(records, name):
    result = {}
    for record in records[name]:
        unit, instruction = record.get('unit'), record.get('instruction')
        require(type(unit) is int and 0 <= unit < 4 and type(instruction) is int and instruction in (0, 1),
                name + ': group identity')
        require((unit, instruction) not in result, name + ': duplicate group')
        result[(unit, instruction)] = record
    require(len(result) == 8, name + ': incomplete group inventory')
    return result


def same_logits(left, right):
    la, lb = left['analysis'], right['analysis']
    if la['generated'] != lb['generated'] or len(la['logit_refs']) != len(lb['logit_refs']):
        return False
    return all(x['shape'] == y['shape'] and x['bytes'] == y['bytes'] and x['sha256'] == y['sha256']
               for x, y in zip(la['logit_refs'], lb['logit_refs']))


def audit_actions(records, rows, flow, groups, bank, oracle, width, terminal):
    index = {}
    for row in rows:
        key = key_of(row['result']['action'])
        require(key not in index, 'duplicate evaluated action')
        index[key] = row
    baseline = {}
    for unit in range(4):
        for instruction in (0, 1):
            for side in SIDES:
                key = ('clean', unit, instruction, side, None, None, None)
                require(key in index, 'missing clean case')
                baseline[(unit, instruction, side)] = index[key]
    geometry_records = indexed_by_group(records, 'group_geometry')
    support_records = indexed_by_group(records, 'baseline_support')
    outcomes = indexed_by_group(records, 'group_outcome')
    geometry, support = {}, {}
    for unit, instruction in groups:
        groupkey = (unit, instruction)
        states = {side: bank.read(baseline[(unit, instruction, side)]['result']['after']) for side in SIDES}
        geometry[groupkey] = audit_geometry(geometry_records[groupkey], states, groups[groupkey], width, bank, oracle)
        support[groupkey] = all(baseline[(unit, instruction, side)]['analysis']['strict'] and
            (baseline[(unit, instruction, side)]['analysis']['margin'] > 0 if side.startswith('P') else
             baseline[(unit, instruction, side)]['analysis']['margin'] < 0) for side in SIDES)
        record = support_records[groupkey]
        require(type(record.get('supported')) is bool and record['supported'] == support[groupkey],
                'clean baseline support')

    schedule = list(expected_schedule())
    require(len(schedule) == len(flow) == 484, 'fixed action schedule length')
    for step, (kind, entry) in zip(schedule, flow):
        groupkey = (step['unit'], step['instruction'])
        skip = step['kind'] in ('matched', 'natural') and (
            not support[groupkey] or geometry[groupkey]['status'] != 'matchable')
        if skip:
            require(kind == 'skipped' and typed_equal(entry['action'], step) and
                    entry['reason'] == ('baseline-unsupported' if not support[groupkey] else geometry[groupkey]['status']),
                    'wrong skipped action/reason')
        else:
            require(kind == 'evaluation', 'unexpected action skip')
            matches_step(entry['result']['action'], step)

    expected_control, expected_comparison = [], []
    for row in rows:
        action, before_ref, after_ref = row['result']['action'], row['result']['before'], row['result']['after']
        kind, unit, instruction, side = action['kind'], action['unit'], action['instruction'], action['receiver']
        groupkey, bkey = (unit, instruction), (unit, instruction, side)
        clean = baseline[bkey]
        own = bank.read(clean['result']['after'])
        before, after = bank.read(before_ref), bank.read(after_ref)
        if kind not in ('clean', 'repeat'):
            require(before == own and typed_equal(action.get('source_state'), clean['result']['action']['encoder_state']),
                    'action source clean state')
        else:
            require(before == after, 'clean/repeat state changed before decoder')
        if kind in ('passthrough', 'self'):
            require(after == own, 'identity state differs')
            if kind == 'self':
                require(action.get('position') == geometry[groupkey]['sitepos'][action['site']], 'self position')
        elif kind == 'full_swap':
            donor = baseline[(unit, instruction, OPP[side])]
            require(after == bank.read(donor['result']['after']) and
                    typed_equal(action.get('donor_state'), donor['result']['action']['encoder_state']), 'full donor state')
        elif kind in ('matched', 'natural'):
            if kind == 'matched':
                item = geometry[groupkey]['landings'][side + ':' + action['action'] + ':' + action['dose']]
                position, replacement = geometry[groupkey]['sitepos'][item['site']], item['raw']
                stored = geometry_records[groupkey]['landings'][side + ':' + action['action'] + ':' + action['dose']]
                for name in set(stored) - {'action', 'level', 'receiver'}:
                    crosscheck(action.get(name), stored[name], 'matched action metadata.' + name)
                donor_side = item['donor']
            else:
                donor_side = OPP[side]
                position = geometry[groupkey]['sitepos']['adjective']
                donor_raw = bank.read(baseline[(unit, instruction, donor_side)]['result']['after'])
                replacement = segment(donor_raw, position, width)
                require(action.get('site') == 'adjective', 'natural reference site')
                probe_check(action.get('numeric_before'))
                probe_check(action.get('numeric_after'))
                ownv = values(segment(own, position, width), oracle)
                donorv = values(replacement, oracle)
                delta = [b - a for a, b in zip(ownv, donorv)]
                dose, base = oracle.norm(delta), oracle.norm(ownv)
                cosine = math.fsum(x * x for x in delta) / (dose * dose) if dose else None
                for name, expected in {'natural_distance': geometry[groupkey]['distances'][side + ':opposite_adjective'],
                    'target_dose': dose, 'achieved_norm': dose, 'cosine': cosine, 'receiver_norm': base,
                    'achieved_over_receiver_norm': None if base == 0 else dose / base,
                    'ratio_reason': 'zero receiver norm' if base == 0 else None}.items():
                    crosscheck(action.get(name), expected, 'natural geometry.' + name)
            donor = baseline[(unit, instruction, donor_side)]
            require(type(action.get('position')) is int and action['position'] == position and
                    typed_equal(action.get('donor_state'), donor['result']['action']['encoder_state']), 'action donor/position')
            start, end = 4 * width * position, 4 * width * (position + 1)
            require(after == own[:start] + replacement + own[end:], 'action landing or untouched state')
        elif kind not in ('clean', 'repeat'):
            require(kind in ('passthrough', 'self', 'full_swap'), 'unknown evaluation kind')
        if kind in ('passthrough', 'self', 'full_swap', 'repeat'):
            expected_key = (unit, instruction, OPP[side]) if kind == 'full_swap' else bkey
            require(same_logits(row, baseline[expected_key]), 'identity/full-state/fresh control logits')
            base_step = {k: action[k] for k in ('kind', 'unit', 'instruction', 'receiver')}
            if kind == 'self':
                base_step['site'] = action['site']
            elif kind == 'full_swap':
                base_step['donor'] = action['donor']
            expected_control.append({'action': base_step, 'expected': list(expected_key)})
        if kind != 'clean':
            expected_comparison.append((row, clean))
    crosscheck(records['control_verified'], expected_control, 'control verification order')
    require(len(records['output_comparison']) == len(expected_comparison), 'output comparison count')
    tv_by_action = {}
    for record, (row, clean) in zip(records['output_comparison'], expected_comparison):
        current, base = row['analysis'], clean['analysis']
        tv, top1, top2 = total_variation(clean['result'], row['result'], bank, oracle)
        expected = {'action': row['result']['action'], 'clean_margin': base['margin'],
            'action_margin': current['margin'], 'clean_generated_ids': base['generated'],
            'action_generated_ids': current['generated'], 'clean_strict_correct': base['strict'],
            'action_strict_correct': current['strict'],
            'clean_teacher': {name: {'score': base['scores'][name], 'ranks': base['ranks'][name]}
                              for name in ('positive', 'negative')},
            'action_teacher': {name: {'score': current['scores'][name], 'ranks': current['ranks'][name]}
                               for name in ('positive', 'negative')},
            'tv': tv, 'clean_top_id': top1, 'action_top_id': top2,
            'numeric_before': PROBE, 'numeric_after': PROBE}
        crosscheck(record, expected, 'full-output comparison')
        tv_by_action[key_of(row['result']['action'])] = tv

    final_groups = []
    for unit in range(4):
        pair = []
        for instruction in (0, 1):
            groupkey = (unit, instruction)
            if not support[groupkey]:
                expected = {'status': 'baseline-unsupported'}
            elif geometry[groupkey]['status'] != 'matchable':
                expected = {'status': geometry[groupkey]['status']}
            else:
                clean = {side: baseline[(unit, instruction, side)]['analysis']['margin'] for side in SIDES}
                by_level = {level: {arm: {side: index[('matched', unit, instruction, side, None, arm, level)]
                                                    ['analysis']['margin'] for side in SIDES}
                                    for arm in ARMS} for level in ('low', 'high')}
                expected = expected_decision(clean, by_level['high'])
                low = expected_decision(clean, by_level['low'])
                expected['low_dose'] = low
                expected['dose_comparison'] = {arm: {side: {
                    'low': low['movements'][arm][side], 'high': expected['movements'][arm][side],
                    'magnitude_change': abs(expected['movements'][arm][side]) - abs(low['movements'][arm][side])}
                    for side in SIDES} for arm in ARMS}
                expected['numeric_before'] = expected['numeric_after'] = PROBE
            actual = {k: v for k, v in outcomes[groupkey].items() if k not in ('unit', 'instruction')}
            crosscheck(actual, expected, 'group outcome')
            pair.append(expected)
        final_groups.append(pair)
    passed = sum(all(g['status'] == 'measured' and g['passes'] for g in pair) for pair in final_groups)
    primary = {'status': 'supported' if passed >= 3 else 'threshold-not-met', 'passed_units': passed,
               'total_units': 4}
    require(len(records['primary_numeric']) == 1, 'primary probe count')
    crosscheck(records['primary_numeric'][0], {'before': PROBE, 'after': PROBE}, 'primary probe')
    crosscheck(terminal['detail']['groups'], final_groups, 'terminal group outcomes')
    crosscheck(terminal['detail']['primary'], primary, 'fixed-denominator primary')
    all_cases = []
    for unit in range(4):
        for instruction in (0, 1):
            groupkey = (unit, instruction)
            status = final_groups[unit][instruction]['status']
            for side in SIDES:
                base = baseline[(unit, instruction, side)]['analysis']
                case = {'unit': unit, 'instruction': instruction, 'receiver': side, 'status': status,
                        'baseline_supported': support[groupkey], 'dose_status': geometry[groupkey]['status'],
                        'q': geometry[groupkey]['q'], 'baseline_margin': base['margin'],
                        'baseline_scores': base['scores'], 'baseline_generated_ids': base['generated'],
                        'baseline_strict_correct': base['strict'],
                        'natural_distances': {arm: geometry_records[groupkey]['distances'][side + ':' + arm]
                                              for arm in ARMS},
                        'q_over_natural_word_distance': geometry_records[groupkey]['q_over_word_distance'][side],
                        'opposite_same_donor_cosine': geometry_records[groupkey]['donor_cosines'][side],
                        'matched': [], 'natural_reference': None}
                if status == 'measured':
                    for arm in ARMS:
                        for level in ('low', 'high'):
                            key = ('matched', unit, instruction, side, None, arm, level)
                            measured = index[key]['analysis']
                            landing = geometry[groupkey]['landings'][side + ':' + arm + ':' + level]
                            case['matched'].append({'arm': arm, 'dose': level, 'site': landing['site'],
                                'donor': landing['donor'], 'margin': measured['margin'],
                                'scores': measured['scores'], 'generated_ids': measured['generated'],
                                'strict_correct': measured['strict'], 'tv': tv_by_action[key],
                                'donor_signed_change': (base['margin'] - measured['margin'] if side.startswith('P')
                                                        else measured['margin'] - base['margin']),
                                'fraction': landing['fraction'], 'target_dose': landing['target_dose'],
                                'achieved_norm': landing['achieved_norm'], 'tolerance': landing['tolerance'],
                                'relative_dose_error': abs(landing['achieved_norm'] - landing['target_dose']) /
                                                       landing['target_dose'],
                                'cosine': landing['cosine'], 'receiver_norm': landing['receiver_norm'],
                                'achieved_over_receiver_norm': landing['achieved_over_receiver_norm']})
                    key = ('natural', unit, instruction, side, None, None, None)
                    measured, action = index[key]['analysis'], index[key]['result']['action']
                    case['natural_reference'] = {'margin': measured['margin'], 'scores': measured['scores'],
                        'generated_ids': measured['generated'], 'strict_correct': measured['strict'],
                        'tv': tv_by_action[key], 'achieved_norm': action['achieved_norm'],
                        'donor_signed_change': (base['margin'] - measured['margin'] if side.startswith('P')
                                                else measured['margin'] - base['margin']),
                        'receiver_norm': action['receiver_norm'],
                        'achieved_over_receiver_norm': action['achieved_over_receiver_norm']}
                all_cases.append(case)
    return {'primary': primary, 'evaluations': len(rows), 'components': terminal['components'],
            'groups': final_groups, 'cases': all_cases,
            'group_statuses': [[g['status'] for g in pair] for pair in final_groups]}


def load_tokenizer(asset_dir):
    path = Path(asset_dir)
    raw = (path / 'asset-manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083',
            'tokenizer asset manifest changed')
    manifest = strict_json(raw)
    require(manifest.get('repo') == 'google/flan-t5-small' and
            manifest.get('revision') == '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab', 'tokenizer identity')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model'):
        data = (path / name).read_bytes()
        meta = manifest['files'][name]
        require(len(data) == meta['bytes'] and hashlib.sha256(data).hexdigest() == meta['sha256'],
                'tokenizer asset changed: ' + name)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)


def audit_contract(started, records, bindings, oracle, started_hash):
    require(type(bindings) is dict and set(bindings) == {'pool', 'selection', 'runner_revision', 'started_sha256'},
            'missing production bindings')
    for name in ('pool', 'selection', 'started_sha256'):
        require(type(bindings[name]) is str and re.fullmatch('[0-9a-f]{64}', bindings[name]) is not None,
                'invalid commitment or started digest')
    require(type(bindings['runner_revision']) is str and
            re.fullmatch('[0-9a-f]{40}', bindings['runner_revision']) is not None, 'invalid runner revision')
    require(started_hash == bindings['started_sha256'] and type(started) is dict and
            set(started) == {'schema', 'source', 'pool_commitment', 'selection_commitment',
                             'asset_manifest_sha256', 'scope'} and
            started['schema'] == 'flan-t5-sentiment-intervention-owner-v1' and
            started['pool_commitment'] == bindings['pool'] and
            started['selection_commitment'] == bindings['selection'] and
            started['asset_manifest_sha256'] == 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083' and
            type(started['source']) is dict and started['source'].get('revision') == bindings['runner_revision'] and
            type(started['source'].get('sha256')) is dict, 'started record identity')
    contract = records['owner_contract'][0]
    fixed = {'threads': 1, 'interop_threads': 1, 'deterministic': True, 'deterministic_warn_only': False,
             'float32_matmul_precision': 'highest', 'byte_order': 'little', 'python_float_rounds': 1,
             'seed': 20261028, 'device': 'cpu', 'dtype': 'float32', 'attention': 'eager', 'decoder_cache': False}
    for name, expected in fixed.items():
        require(typed_equal(contract.get(name), expected), 'owner runtime field: ' + name)
    review = contract.get('screen_replay')
    require(type(review) is dict and review.get('status') == 'consistent' and
            review.get('pool_commitment') == bindings['pool'] and
            review.get('selection_commitment') == bindings['selection'] and
            review.get('runner_revision') == bindings['runner_revision'] and
            review.get('asset_manifest_sha256') == started['asset_manifest_sha256'] and
            typed_equal(review.get('unit_count'), 12) and typed_equal(review.get('token_rows'), 96) and
            typed_equal(review.get('selection_count'), 4) and review.get('enough_selected') is True,
            'recorded independent selection replay binding')
    pin = contract.get('scientific_reader')
    require(type(pin) is dict and set(pin) == {'schema', 'revision', 'sha256'} and
            pin['schema'] == 'flan-t5-sentiment-scientific-reader-pin-v1' and
            type(pin['revision']) is str and re.fullmatch('[0-9a-f]{40}', pin['revision']) is not None and
            type(pin['sha256']) is dict, 'scientific-reader pin')
    paths = {'scripts/flan_sentiment_intervention_independent_audit.py': Path(__file__),
             'scripts/flan_sentiment_dose_oracle.py': Path(__file__).resolve().parent / 'flan_sentiment_dose_oracle.py',
             'scripts/bert_taste_matched_independent.py': oracle.helper_path}
    expected_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    require(typed_equal(pin['sha256'], expected_hashes), 'scientific-reader source bytes differ')
    settings = records['runtime_observation'][0].get('settings')
    require(type(settings) is dict, 'core runtime settings')
    core_fixed = {'device': 'cpu', 'num_threads': 1, 'num_interop_threads': 1,
                  'deterministic': True, 'deterministic_warn_only': False,
                  'default_dtype': 'torch.float32', 'float32_matmul_precision': 'highest', 'cpu_autocast': False,
                  'attention_implementation': 'eager', 'config_use_cache': False,
                  'hidden_width': 512, 'vocabulary': 32128, 'decoder_start_id': 0, 'eos_id': 1}
    for name, expected in core_fixed.items():
        require(typed_equal(settings.get(name), expected), 'core runtime field: ' + name)
    require(type(contract.get('torch')) is str and contract['torch'] == settings.get('torch_version') and
            type(contract.get('python')) is str and contract['python'] == settings.get('python_version'),
            'owner/core runtime version mismatch')
    return {'pool_commitment': bindings['pool'], 'selection_commitment': bindings['selection'],
            'runner_revision': bindings['runner_revision'], 'reader_revision': pin['revision']}


def audit(attempt, selection_or_groups, tokenizer, width, vocab, *, synthetic=False, bindings=None, selection_salt=None):
    check_local_arithmetic()
    require(type(synthetic) is bool, 'invalid audit mode')
    require(type(width) is int and 1 <= width <= 512 and type(vocab) is int and 2 < vocab <= 32128,
            'invalid width/vocabulary')
    attempt = Path(attempt).resolve(strict=True)
    require(attempt.is_dir(), 'attempt directory')
    if synthetic:
        require(bindings is None and selection_salt is None, 'synthetic audit has production bindings')
        group_list = selection_or_groups
    else:
        group_list = selected_groups(selection_or_groups, selection_salt, bindings, tokenizer)
    groups = check_case_groups(group_list, vocab, tokenizer)
    started, terminal, events = read_records(attempt)
    started_hash = hashlib.sha256((attempt / 'started.json').read_bytes()).hexdigest()
    if events is None:
        return {'status': 'incomplete', 'primary': None, 'started_sha256': started_hash,
                'scope': 'terminal incomplete or absent; partial journal was not numerically replayed'}
    event_hash = hashlib.sha256((attempt / 'events.jsonl').read_bytes()).hexdigest()
    terminal_hash = hashlib.sha256((attempt / 'terminal.json').read_bytes()).hexdigest()
    oracle = load_oracle()
    bank = TensorBank(attempt)
    records, rows, flow = parsed_events(events, terminal, groups, tokenizer, oracle, bank, width, vocab, synthetic)
    identity = None if synthetic else audit_contract(started, records, bindings, oracle, started_hash)
    result = audit_actions(records, rows, flow, groups, bank, oracle, width, terminal)
    bank.inventory()
    retained = ((attempt / 'started.json').stat().st_size + (attempt / 'events.jsonl').stat().st_size +
                sum(identity[1] for identity in bank.identities.values()))
    require(type(terminal.get('retained_bytes')) is int and terminal['retained_bytes'] == retained and
            retained <= 5 * 1024**3 // 4, 'retained byte accounting')
    require(hashlib.sha256((attempt / 'started.json').read_bytes()).hexdigest() == started_hash and
            hashlib.sha256((attempt / 'events.jsonl').read_bytes()).hexdigest() == event_hash and
            hashlib.sha256((attempt / 'terminal.json').read_bytes()).hexdigest() == terminal_hash,
            'journal changed during replay')
    return {'status': 'saved-records-consistent', **result, 'started_sha256': started_hash, 'event_sha256': event_hash,
            'terminal_sha256': terminal_hash, 'tensor_files': len(bank.identities),
            'recorded_owner_bindings': identity,
            'torch_version_for_log_softmax': torch.__version__,
            'scope': ('Saved input, full output, geometry and arithmetic consistency only. '
                      'Production mode binds the salted selection and retokenizes the selected rows; '
                      'full-pool selection, exposure, published source, chronology, process exit, '
                      'actual execution and whole-process cleanup need separate evidence. '
                      'Teacher log-softmax is replayed with this runtime\'s CPU PyTorch kernel; '
                      'a separate binary64 mathematical check is also applied.')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--selection')
    parser.add_argument('--selection-salt')
    parser.add_argument('--assets')
    parser.add_argument('--pool-commitment')
    parser.add_argument('--selection-commitment')
    parser.add_argument('--runner-revision')
    parser.add_argument('--started-sha256')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    check_local_arithmetic()
    if args.synthetic:
        require(not any((args.selection, args.selection_salt, args.assets, args.pool_commitment,
                         args.selection_commitment, args.runner_revision, args.started_sha256)),
                'synthetic mode arguments')
        path = Path(__file__).resolve().parent / 'review_flan_sentiment_linear_toy.py'
        spec = importlib.util.spec_from_file_location('independent_flan_synthetic_cases', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        selection, tokenizer, width, vocab, manifest_sha, bindings, salt = module.groups(), module.Tokenizer(), 2, 8, None, None, None
    else:
        require(args.selection and args.selection_salt and args.assets, 'selection, salt and assets required')
        raw = Path(args.selection).read_bytes()
        require(len(raw) <= 8 * 1024**2, 'selection file too large')
        manifest_sha = hashlib.sha256(raw).hexdigest()
        selection = strict_json(raw)
        salt = Path(args.selection_salt).read_bytes()
        tokenizer, width, vocab = load_tokenizer(args.assets), 512, 32128
        bindings = {'pool': args.pool_commitment, 'selection': args.selection_commitment,
                    'runner_revision': args.runner_revision, 'started_sha256': args.started_sha256}
    report = audit(args.attempt, selection, tokenizer, width, vocab, synthetic=args.synthetic,
                   bindings=bindings, selection_salt=salt)
    report['case_mode'] = 'synthetic' if args.synthetic else 'selected-cases'
    report['selection_sha256'] = manifest_sha
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
