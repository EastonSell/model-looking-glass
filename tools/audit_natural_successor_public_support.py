"""Reconstruct public old-sampler support and compare the fixed successor pool.

This offline audit reads only public source, public candidate files and the
pinned tokenizer asset. It never opens an old split, nonce or a model. The
support is conservative: it includes all subject/context combinations before
the old sampler's balancing and exposure filters.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


OLD_SOURCE = 'def07cbb168408a12c76ba30f196af45149e111d'
OLD_SAMPLER_SHA256 = '635f25553c20dd437f5cc4a8a9ca3a7a1453e7f8a517875fe75eb01ede2d5f35'
BINDING_SHA256 = '23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0'
TOKENIZER_SHA256 = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
POOL_SHA256 = 'b3323d296a9f56a4232b4a403f7c8935896634b3d2b6e9fe7eee4aee56ba1126'
INPUT_SHA256 = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
SUPPORT_SHA256 = '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'
OLD_COMMITMENT_SHA256 = '7f4f96fee2fd90bc5efbf8b85bb722021b3b698905256dc9f2e2e626d4a4f4ba'
FREEZE_SHA256 = '35138307572410466d7cd1e1632bdfdfb9474562e9916fb48a0669a3478c6eeb'
TARGETS = {' is': 318, ' are': 389, ' was': 373, ' were': 547}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(',', ':')).encode('ascii')


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def load_json(raw, *, require_canonical=True):
    value = json.loads(raw, object_pairs_hook=pairs)
    if require_canonical and canonical(value) != raw:
        raise ValueError('expected canonical JSON')
    return value


def read_bound(path, digest, maximum):
    raw = Path(path).read_bytes()
    if not 0 < len(raw) <= maximum or sha(raw) != digest:
        raise ValueError('public source or asset differs from its fixed identity')
    return raw


def prefix_counts(left, right):
    """Count ordered complete-row-prefix relations in each direction."""
    if (not left or not right or any(type(row) not in (tuple, list) or not row
            or any(type(token) is not int or not 0 <= token <= 50256 for token in row)
            for row in (*left, *right))):
        raise ValueError('invalid token row')
    first = sum(1 for a in left for b in right if len(a) <= len(b) and tuple(b[:len(a)]) == tuple(a))
    second = sum(1 for a in left for b in right if len(b) <= len(a) and tuple(a[:len(b)]) == tuple(b))
    return first, second


def audit(*, sampler_path, tokenizer_path, pool_path, inputs_path, support_path, commitment_path):
    source = read_bound(sampler_path, OLD_SAMPLER_SHA256, 8 * 1024 * 1024)
    read_bound(tokenizer_path, TOKENIZER_SHA256, 8 * 1024 * 1024)
    pool_raw = read_bound(pool_path, POOL_SHA256, 65536)
    input_raw = read_bound(inputs_path, INPUT_SHA256, 1000000)
    support_raw = read_bound(support_path, SUPPORT_SHA256, 65536)
    commitment_raw = read_bound(commitment_path, OLD_COMMITMENT_SHA256, 8192)
    support, pool = load_json(support_raw), load_json(pool_raw, require_canonical=False)
    published_inputs, commitment = load_json(input_raw), load_json(commitment_raw)
    if (type(commitment) is not dict or commitment.get('prediction_freeze_sha256') != FREEZE_SHA256
            or type(commitment.get('heldout_commitment')) is not str):
        raise ValueError('old public commitment differs')
    tree = ast.parse(source)
    names = ('NOUNS', 'PLURAL_CONTEXTS', 'SINGULAR_CONTEXTS')
    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
            name = node.targets[0].id
            if name in values:
                raise ValueError('duplicate public sampler constant')
            values[name] = ast.literal_eval(node.value)
    if set(values) != set(names) or any(type(values[name]) is not tuple or len(values[name]) != 8
            for name in ('NOUNS',)) or any(type(values[name]) is not tuple or len(values[name]) != 4
                                           for name in ('PLURAL_CONTEXTS', 'SINGULAR_CONTEXTS')):
        raise ValueError('old public sampler frame differs')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path)))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    prefix = encode('The')
    subjects = [word for pair in values['NOUNS'] for word in pair.split('/')]
    contexts = values['PLURAL_CONTEXTS'] + values['SINGULAR_CONTEXTS']
    if len(subjects) != 16 or len(set(subjects)) != 16 or len(set(contexts)) != 8:
        raise ValueError('old public frame is not unique')
    old_rows = []
    for subject in subjects:
        first = encode('The ' + subject)
        if len(first) != len(prefix) + 1 or first[:-1] != prefix:
            raise ValueError('old subject tokenization differs')
        for context in contexts:
            text = 'The ' + subject + ' ' + context
            row = encode(text)
            if (row != first + encode(' ' + context) or not 1 <= len(row) <= 32
                    or any(encode(text + suffix) != row + [target] for suffix, target in TARGETS.items())):
                raise ValueError('old complete prompt tokenization differs')
            old_rows.append(row)
    expected_support = {'schema': 'mlg-natural-original-public-sampling-support-v1',
                        'scope': 'all public combinations before structural and exposure filtering; no sampled rows',
                        'frozen_source_commit': OLD_SOURCE, 'frozen_sampler_sha256': OLD_SAMPLER_SHA256,
                        'binding_sha256': BINDING_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256, 'rows': old_rows}
    if support != expected_support or len(old_rows) != 128 or len(set(map(tuple, old_rows))) != 128:
        raise ValueError('reconstructed old public support differs')
    if (type(pool) is not dict or pool.get('schema') != 'mlg-natural-successor-pool-v1'
            or len(pool.get('contexts', [])) != 8 or len(pool.get('nouns', [])) != 16):
        raise ValueError('fixed successor pool differs')
    candidate_rows = []
    for context in pool['contexts']:
        for noun in pool['nouns']:
            for number in ('singular', 'plural'):
                text = 'The ' + noun[number] + ' ' + context['text']
                row = encode(text)
                subject = encode('The ' + noun[number])
                if (not 0 < len(prefix) < len(row) - 1 or len(row) > 32
                        or subject != prefix + [row[len(prefix)]]
                        or row != subject + encode(' ' + context['text'])
                        or any(encode(text + suffix) != row + [target] for suffix, target in TARGETS.items())):
                    raise ValueError('successor complete prompt tokenization differs')
                candidate_rows.append(row)
    expected_inputs = {'schema': 'mlg-natural-successor-candidate-inputs-v1',
                       'pool_sha256': POOL_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
                       'cases': [{'input_ids': row} for row in candidate_rows]}
    if published_inputs != expected_inputs or len(candidate_rows) != 256 or len(set(map(tuple, candidate_rows))) != 256:
        raise ValueError('fixed successor inputs differ')
    old_prefix, new_prefix = prefix_counts(old_rows, candidate_rows)
    return {'schema': 'mlg-natural-successor-public-support-comparison-v1',
            'status': 'clear' if old_prefix == new_prefix == 0 else 'collision',
            'comparison': 'bidirectional-complete-row-prefix',
            'original_support_count': len(old_rows), 'successor_input_count': len(candidate_rows),
            'original_prefix_of_successor_count': old_prefix, 'successor_prefix_of_original_count': new_prefix,
            'original_sampler_source_commit': OLD_SOURCE, 'original_sampler_sha256': OLD_SAMPLER_SHA256,
            'original_binding_sha256': BINDING_SHA256, 'original_commitment_file_sha256': OLD_COMMITMENT_SHA256,
            'original_heldout_commitment': commitment['heldout_commitment'],
            'prediction_freeze_sha256': FREEZE_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
            'original_support_sha256': SUPPORT_SHA256, 'successor_pool_sha256': POOL_SHA256,
            'successor_inputs_sha256': INPUT_SHA256, 'comparator_source_sha256': sha(Path(__file__).read_bytes()),
            'scope': 'public conservative support only; no split, nonce, model or membership query'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('sampler', 'tokenizer', 'pool', 'inputs', 'support', 'commitment', 'receipt'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = audit(sampler_path=args.sampler, tokenizer_path=args.tokenizer, pool_path=args.pool,
                   inputs_path=args.inputs, support_path=args.support, commitment_path=args.commitment)
    raw = canonical(result)
    with Path(args.receipt).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short public receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    if Path(args.receipt).read_bytes() != raw:
        raise ValueError('public receipt did not read back')
    print(result['status'])
    if result['status'] != 'clear':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
