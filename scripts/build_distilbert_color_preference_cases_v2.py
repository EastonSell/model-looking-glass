"""Build the frozen color-preference inputs using only the pinned tokenizer."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

PROTOCOL_COMMIT = 'd94594f69426aa5e41a7cf90acd1e98be42f9087'
PROTOCOL_SHA256 = '41ffb6f85d0017a3644b4ee5cc7328ffa1c90c3fec36faeb438fe96625dfc34d'
MODEL = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
TOKENIZER_SHA256 = 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'
VOCAB_SHA256 = '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'
OBJECTS = (('banana', 'yellow'), ('strawberry', 'red'), ('lemon', 'yellow'))


def specifications():
    cases = []
    for obj, color in OBJECTS:
        templates = (
            ('discovery', f'A ripe {obj} has a [MASK] color.'),
            ('calibration-01', f'The color of a ripe {obj} is usually [MASK].'),
            ('calibration-02', f'A ripe {obj} appears [MASK] in color.'),
            ('heldout-01', f'The usual color of a ripe {obj} is [MASK].'),
            ('heldout-02', f'The ripe {obj} is colored [MASK].'),
            ('heldout-03', f'A typical ripe {obj} has [MASK] coloring.'),
            ('heldout-04', f'If the {obj} is ripe, its color is [MASK].'),
        )
        for role, text in templates:
            cases.append({'id': f'{obj}-{role}', 'role': role, 'object': obj, 'text': text,
                          'factual': color, 'counterfactual': 'purple', 'wrong_target': 'blue'})
        for index, text in enumerate((f'A {obj} is classified as a [MASK].',
                                      f'A ripe {obj} is a type of [MASK].'), 1):
            cases.append({'id': f'{obj}-locality-{index:02d}', 'role': 'same_object_locality',
                          'object': obj, 'text': text, 'expected': 'fruit', 'accepted': ['fruit']})
    shared = (
        ('color-01', 'The usual color of a ripe orange is [MASK].', 'orange'),
        ('color-02', 'The color of fresh snow is [MASK].', 'white'),
        ('color-03', 'The color of a ripe lime is [MASK].', 'green'),
        ('color-04', 'A ripe eggplant is [MASK] in color.', 'purple'),
        ('color-05', 'In daylight, a clear sky is [MASK] in color.', 'blue'),
        ('unrelated-01', 'A calf is a young [MASK].', 'cow'),
        ('unrelated-02', 'A foal is a young [MASK].', 'horse'),
        ('unrelated-03', 'A clarinet is a musical [MASK].', 'instrument'),
        ('unrelated-04', 'A hawk is a kind of [MASK].', 'bird'),
        ('unrelated-05', 'A tulip is a kind of [MASK].', 'flower'),
    )
    for case_id, text, answer in shared:
        accepted = [answer, 'yellow'] if case_id == 'color-03' else [answer]
        cases.append({'id': case_id, 'role': 'other_color' if case_id.startswith('color-') else 'unrelated',
                      'object': None, 'text': text, 'expected': answer, 'accepted': accepted})
    return cases


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def build(snapshot, protocol):
    if sha(protocol) != PROTOCOL_SHA256:
        raise ValueError('protocol bytes differ from the published source')
    if sha(snapshot / 'tokenizer.json') != TOKENIZER_SHA256 or sha(snapshot / 'vocab.txt') != VOCAB_SHA256:
        raise ValueError('pinned tokenizer bytes differ')
    if importlib.metadata.version('tokenizers') != '0.22.2':
        raise ValueError('tokenizers version differs')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    if tuple(tokenizer.token_to_id(x) for x in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')) != (0, 100, 101, 102, 103):
        raise ValueError('special-token IDs differ')
    records = []
    for case in specifications():
        text = case['text']
        if text.count('[MASK]') != 1:
            raise ValueError('each case requires exactly one mask')
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        if not (2 <= len(ids) <= 64 and ids[0] == 101 and ids[-1] == 102 and
                ids.count(103) == 1 and 100 not in ids):
            raise ValueError(f"invalid input: {case['id']}")
        position = ids.index(103)
        words = ([case['factual'], case['counterfactual'], case['wrong_target']]
                 if 'factual' in case else case['accepted'])
        if len(words) != len(set(words)) or ('accepted' in case and case['expected'] not in words):
            raise ValueError('accepted words must be distinct and include the expected answer')
        word_ids = {}
        for word in words:
            single = tokenizer.encode(word, add_special_tokens=False).ids
            if len(single) != 1 or single[0] in (0, 100, 101, 102, 103) or tokenizer.id_to_token(single[0]) != word:
                raise ValueError(f'not a complete single token: {word}')
            expected = ids[:position] + single + ids[position + 1:]
            if tokenizer.encode(text.replace('[MASK]', word)).ids != expected:
                raise ValueError(f"fill changes other tokens: {case['id']} / {word}")
            word_ids[word] = single[0]
        if len(set(word_ids.values())) != len(word_ids):
            raise ValueError('fill IDs must be distinct')
        records.append({**case, 'input_ids': ids, 'attention_mask': [1] * len(ids),
                        'tokens': encoded.tokens, 'mask_index': position, 'word_ids': word_ids})
    if len(records) != 37 or len({r['id'] for r in records}) != 37:
        raise ValueError('expected 37 distinct case IDs')
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            a, b = left['input_ids'], right['input_ids']
            if related(a, b) or related(a[:-1], b[:-1]):
                raise ValueError(f"input prefix collision: {left['id']} / {right['id']}")
    if sha(snapshot / 'tokenizer.json') != TOKENIZER_SHA256 or sha(snapshot / 'vocab.txt') != VOCAB_SHA256:
        raise ValueError('tokenizer bytes changed during encoding')
    return {'schema': 'distilbert-color-preference-cases-v2', 'protocol_commit': PROTOCOL_COMMIT,
            'protocol_sha256': PROTOCOL_SHA256, 'model': MODEL, 'revision': REVISION,
            'tokenizer_sha256': TOKENIZER_SHA256, 'vocab_sha256': VOCAB_SHA256,
            'tokenizers_version': '0.22.2', 'candidate_order': [obj for obj, _ in OBJECTS],
            'mask_id': 103, 'cls_id': 101, 'sep_id': 102, 'cases': records,
            'scope': 'Public tokenizer-only planned inputs; no model forward or unknown-exposure claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = build(args.snapshot, args.protocol)
    with args.output.open('xb') as out:
        out.write((json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


if __name__ == '__main__':
    main()
