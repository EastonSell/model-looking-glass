"""Construct the fixed masked-color case inventory with a pinned tokenizer only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

PROTOCOL_COMMIT = 'd27eb3f6e1af00d4b045fb91b20cad5b25c8b699'
PROTOCOL_SHA256 = 'd414ab04137b842cd28d232b31e2ca5b62833ade06ef0ede4a6ce17da081f937'
MODEL = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
ASSETS = {
    'config.json': (483, '69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83'),
    'tokenizer_config.json': (48, 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'),
    'vocab.txt': (231508, '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
    'tokenizer.json': (466062, 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'),
    'model.safetensors': (267954768, '5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063'),
}
OBJECTS = (('banana', 'yellow'), ('strawberry', 'red'), ('lemon', 'yellow'))


def specifications():
    """The order, roles, text and complete-word accepted variants are fixed here."""
    result = []
    for obj, color in OBJECTS:
        targets = (
            ('discovery', f'The color of a ripe {obj} is [MASK].'),
            ('calibration-01', f'A ripe {obj} is typically [MASK].'),
            ('calibration-02', f'The surface of a ripe {obj} is [MASK].'),
            ('heldout-01', f'A ripe {obj} is usually [MASK] in color.'),
            ('heldout-02', f'The outside of a ripe {obj} is typically [MASK].'),
            ('heldout-03', f'The surface of a ripe {obj} has a [MASK] color.'),
            ('heldout-04', f'When a {obj} ripens, its typical color is [MASK].'),
        )
        for role, text in targets:
            result.append({'id': f'{obj}-{role}', 'role': role, 'object': obj, 'text': text,
                           'factual': color, 'counterfactual': 'purple', 'wrong_target': 'blue'})
        plant = ['plant', 'tree'] if obj == 'lemon' else ['plant']
        result.append({'id': f'{obj}-locality-01', 'role': 'same_object_locality', 'object': obj,
                       'text': f'A {obj} is a kind of [MASK].', 'expected': 'fruit', 'accepted': ['fruit']})
        result.append({'id': f'{obj}-locality-02', 'role': 'same_object_locality', 'object': obj,
                       'text': f'A {obj} grows on a [MASK].', 'expected': 'plant', 'accepted': plant})
    shared = (
        ('color-01', 'Fresh snow is typically [MASK].', 'white', ['white']),
        ('color-02', 'Ripe tomatoes are typically [MASK].', 'red', ['red']),
        ('color-03', 'The color of a clear daytime sky is [MASK].', 'blue', ['blue']),
        ('color-04', 'Fresh grass is typically [MASK].', 'green', ['green']),
        ('color-05', 'The usual color of a ripe eggplant is [MASK].', 'purple', ['purple']),
        ('unrelated-01', 'A puppy is a young [MASK].', 'dog', ['dog']),
        ('unrelated-02', 'A kitten is a young [MASK].', 'cat', ['cat']),
        ('unrelated-03', 'A violin is a musical [MASK].', 'instrument', ['instrument']),
        ('unrelated-04', 'A sparrow is a type of [MASK].', 'bird', ['bird']),
        ('unrelated-05', 'A rose is a type of [MASK].', 'flower', ['flower']),
    )
    for case_id, text, expected, accepted in shared:
        result.append({'id': case_id, 'role': 'other_color' if case_id.startswith('color-') else 'unrelated',
                       'object': None, 'text': text, 'expected': expected, 'accepted': accepted})
    return result


def sha_file(path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def checked_token(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if (len(ids) != 1 or ids[0] in tokenizer.all_special_ids or
            tokenizer.convert_ids_to_tokens(ids[0]) != text or text.startswith('##')):
        raise ValueError(f'{text!r} is not the expected complete vocabulary token')
    return ids[0]


def encode_inventory(tokenizer, specs):
    """Encode and check full input rows and every possible one-token fill."""
    if len(specs) != 37 or len({case['id'] for case in specs}) != 37:
        raise ValueError('expected exactly 37 distinct fixed IDs')
    records = []
    for case in specs:
        text = case['text']
        if text.count('[MASK]') != 1:
            raise ValueError('exactly one explicit mask is required')
        ids = tokenizer.encode(text, add_special_tokens=True)
        if (not 2 <= len(ids) <= 64 or ids[0] != tokenizer.cls_token_id or ids[-1] != tokenizer.sep_token_id or
                ids.count(tokenizer.mask_token_id) != 1 or tokenizer.unk_token_id in ids):
            raise ValueError(f"invalid encoded input for {case['id']}")
        position = ids.index(tokenizer.mask_token_id)
        words = ([case['factual'], case['counterfactual'], case['wrong_target']]
                 if 'factual' in case else case['accepted'])
        if len(words) != len(set(words)) or ('accepted' in case and case['expected'] not in words):
            raise ValueError('accepted words must be distinct and include the expected word')
        word_ids = {word: checked_token(tokenizer, word) for word in words}
        if len(set(word_ids.values())) != len(word_ids):
            raise ValueError('accepted words must have distinct token IDs')
        for word, token_id in word_ids.items():
            filled = tokenizer.encode(text.replace('[MASK]', word), add_special_tokens=True)
            wanted = list(ids)
            wanted[position] = token_id
            if filled != wanted:
                raise ValueError(f"fill changes another token for {case['id']} / {word}")
        record = {**case, 'input_ids': ids, 'attention_mask': [1] * len(ids),
                  'tokens': tokenizer.convert_ids_to_tokens(ids), 'mask_index': position,
                  'word_ids': word_ids}
        records.append(record)
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            for a, b in ((left['input_ids'], right['input_ids']),
                         (left['input_ids'][:-1], right['input_ids'][:-1])):
                if a[:len(b)] == b or b[:len(a)] == a:
                    raise ValueError(f"input collision: {left['id']} / {right['id']}")
    return records


def build(snapshot, protocol):
    if sha_file(protocol)[1] != PROTOCOL_SHA256:
        raise ValueError('protocol bytes differ from the reviewed public source')
    for name, expected in ASSETS.items():
        if sha_file(snapshot / name) != expected:
            raise ValueError(f'pinned asset differs: {name}')
    versions = {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')}
    if versions != {'transformers': '4.57.3', 'tokenizers': '0.22.2'}:
        raise ValueError('tokenizer runtime versions differ from the fixed builder')
    # Local tokenizer files only: no checkpoint/model is loaded or forwarded.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                               use_fast=True)
    if (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id, tokenizer.unk_token_id) != (101, 102, 103, 100):
        raise ValueError('unexpected special-token namespace')
    records = encode_inventory(tokenizer, specifications())
    for name, expected in ASSETS.items():
        if sha_file(snapshot / name) != expected:
            raise ValueError(f'pinned asset changed during encoding: {name}')
    return {'schema': 'distilbert-color-edit-cases-v1', 'protocol_commit': PROTOCOL_COMMIT,
            'protocol_sha256': PROTOCOL_SHA256, 'model': MODEL, 'revision': REVISION,
            'assets': {name: {'bytes': size, 'sha256': sha} for name, (size, sha) in ASSETS.items()},
            'tokenizer_versions': versions,
            'candidate_order': [obj for obj, _ in OBJECTS], 'mask_id': tokenizer.mask_token_id,
            'cls_id': tokenizer.cls_token_id, 'sep_id': tokenizer.sep_token_id,
            'cases': records,
            'scope': 'Public tokenizer-only case reservation; no model forward or unknown-exposure claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = build(args.snapshot, args.protocol)
    with args.output.open('xb') as out:
        out.write((json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
