"""Build the fixed antonym inventory without loading or forwarding a model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

PROTOCOL_COMMIT = '4104b5ac8eb83904839fc60f7b16489a228eb481'
PROTOCOL_SHA256 = 'a0735f00bb3f9e91f494456d5a1e060b88ea5dc181cf58dcd08c1593540800b1'
MODEL = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
TOKENIZER_SHA256 = 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'
VOCAB_SHA256 = '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'

CANDIDATES = (
    ('box', 'size', 'large', 'small', ('small', 'little'), 'container',
     'A [MASK] number of visitors arrived.'),
    ('rope', 'length', 'long', 'short', ('short',), 'cord',
     'The concert lasted a [MASK] time.'),
    ('drink', 'temperature', 'hot', 'cold', ('cold',), 'beverage',
     'The article became a [MASK] topic of discussion.'),
    ('room', 'brightness', 'bright', 'dark', ('dark', 'dim'), 'space',
     'The [MASK] student solved the difficult puzzle.'),
)


def specifications():
    rows = []
    for obj, dimension, cue, factual, accepted, same, alternate in CANDIDATES:
        templates = (
            ('discovery', f'For a {obj}, the opposite of {cue} in {dimension} is [MASK].'),
            ('calibration-01', f'In {dimension}, the opposite of a {cue} {obj} is a [MASK] {obj}.'),
            ('calibration-02', f'A {cue} {obj} is opposite in {dimension} to a [MASK] {obj}.'),
            ('heldout-01', f'For this {obj}, the opposite of {cue} in {dimension} is [MASK].'),
            ('heldout-02', f'In terms of {dimension}, a {cue} {obj} is opposite to a [MASK] {obj}.'),
            ('heldout-03', f'A {cue} {obj} and a [MASK] {obj} are opposites in {dimension}.'),
            ('heldout-04', f'The {dimension} of a {cue} {obj} is opposite to that of a [MASK] {obj}.'),
        )
        for role, text in templates:
            rows.append({'id': f'{obj}-{role}', 'role': role, 'object': obj, 'text': text,
                         'factual': factual, 'counterfactual': cue, 'wrong_target': 'red',
                         'accepted': list(accepted)})
        for index, text in enumerate((f'A {obj} is a kind of [MASK].',
                                      f'A {obj} is a type of [MASK].'), 1):
            rows.append({'id': f'{obj}-same-{index:02d}', 'role': 'same_object_locality',
                         'object': obj, 'text': text, 'expected': same, 'accepted': [same]})
        reverse = (f'In {dimension}, the opposite of a {factual} {obj} is a [MASK] {obj}.',
                   f'A {factual} {obj} is opposite in {dimension} to a [MASK] {obj}.')
        for index, text in enumerate(reverse, 1):
            rows.append({'id': f'{obj}-reverse-{index:02d}', 'role': 'reverse_locality',
                         'object': obj, 'text': text, 'expected': cue, 'accepted': [cue]})
        rows.append({'id': f'{obj}-alternate-01', 'role': 'alternate_use_locality',
                     'object': obj, 'text': alternate, 'expected': cue, 'accepted': [cue]})
    shared = (
        ('wrong-target-correct', 'wrong_target_correct', 'The background color of a standard stop sign is [MASK].', ('red',)),
        ('unrelated-antonym-01', 'unrelated_antonym', 'The opposite of day is [MASK].', ('night',)),
        ('unrelated-antonym-02', 'unrelated_antonym', 'The opposite of the direction left is [MASK].', ('right',)),
        ('unrelated-antonym-03', 'unrelated_antonym', 'The opposite of open is [MASK].', ('closed', 'shut')),
        ('unrelated-relation-01', 'unrelated_relation', 'A dolphin is a kind of [MASK].', ('mammal',)),
        ('unrelated-relation-02', 'unrelated_relation', 'A trumpet is a musical [MASK].', ('instrument',)),
        ('unrelated-relation-03', 'unrelated_relation', 'A week has [MASK] days.', ('seven',)),
    )
    for ident, role, text, accepted in shared:
        rows.append({'id': ident, 'role': role, 'object': None, 'text': text,
                     'expected': accepted[0], 'accepted': list(accepted)})
    return rows


def related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def encode_inventory(tokenizer, specs):
    if len(specs) != 55 or len({case['id'] for case in specs}) != 55:
        raise ValueError('exactly 55 distinct case IDs required')
    records = []
    for case in specs:
        text = case['text']
        if type(text) is not str or text.count('[MASK]') != 1:
            raise ValueError('exactly one explicit mask required')
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        if not (2 <= len(ids) <= 64 and ids[0] == 101 and ids[-1] == 102 and
                ids.count(103) == 1 and 100 not in ids):
            raise ValueError(f"invalid input: {case['id']}")
        mask_index = ids.index(103)
        accepted = case['accepted']
        if (type(accepted) is not list or not accepted or len(accepted) != len(set(accepted)) or
                any(type(word) is not str for word in accepted)):
            raise ValueError('accepted words must be a nonempty distinct list')
        if 'factual' in case:
            if accepted[0] != case['factual'] or case['counterfactual'] in accepted or case['wrong_target'] in accepted:
                raise ValueError('factual accepted words or alternatives differ')
            words = accepted + [case['counterfactual'], case['wrong_target']]
        else:
            if case['expected'] not in accepted:
                raise ValueError('accepted words must include expected answer')
            words = accepted
        if len(words) != len(set(words)):
            raise ValueError('all requested fills must be distinct')
        word_ids = {}
        for word in words:
            single = tokenizer.encode(word, add_special_tokens=False).ids
            if (len(single) != 1 or single[0] in (0, 100, 101, 102, 103) or
                    tokenizer.id_to_token(single[0]) != word):
                raise ValueError(f'not a complete single token: {word}')
            expected = ids[:mask_index] + single + ids[mask_index + 1:]
            if tokenizer.encode(text.replace('[MASK]', word)).ids != expected:
                raise ValueError(f"fill changes other tokens: {case['id']} / {word}")
            word_ids[word] = single[0]
        if len(set(word_ids.values())) != len(word_ids):
            raise ValueError('fill IDs must be distinct')
        records.append({**case, 'input_ids': ids, 'attention_mask': [1] * len(ids),
                        'tokens': encoded.tokens, 'mask_index': mask_index, 'word_ids': word_ids})
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            for a in (left['input_ids'], left['input_ids'][:-1]):
                for b in (right['input_ids'], right['input_ids'][:-1]):
                    if related(a, b):
                        raise ValueError(f"input prefix relation: {left['id']} / {right['id']}")
    return records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(snapshot, protocol):
    if sha(protocol) != PROTOCOL_SHA256:
        raise ValueError('protocol bytes differ from published source')
    if sha(snapshot / 'tokenizer.json') != TOKENIZER_SHA256 or sha(snapshot / 'vocab.txt') != VOCAB_SHA256:
        raise ValueError('pinned tokenizer bytes differ')
    version = importlib.metadata.version('tokenizers')
    if version != '0.22.2':
        raise ValueError('tokenizers version differs')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    if tuple(tokenizer.token_to_id(x) for x in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')) != (0, 100, 101, 102, 103):
        raise ValueError('unexpected special-token IDs')
    records = encode_inventory(tokenizer, specifications())
    if sha(snapshot / 'tokenizer.json') != TOKENIZER_SHA256 or sha(snapshot / 'vocab.txt') != VOCAB_SHA256:
        raise ValueError('tokenizer bytes changed during encoding')
    return {'schema': 'distilbert-antonym-preference-cases-v1', 'protocol_commit': PROTOCOL_COMMIT,
            'protocol_sha256': PROTOCOL_SHA256, 'model': MODEL, 'revision': REVISION,
            'tokenizer_sha256': TOKENIZER_SHA256, 'vocab_sha256': VOCAB_SHA256,
            'tokenizers_version': version, 'candidate_order': [row[0] for row in CANDIDATES],
            'mask_id': 103, 'cls_id': 101, 'sep_id': 102, 'cases': records,
            'scope': 'Public tokenizer-only planned inputs; no model forward or unknown-exposure claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = (json.dumps(build(args.snapshot, args.protocol), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as out:
        if out.write(data) != len(data):
            raise OSError('short case write')
    print('case SHA-256', hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
