"""Compare a fixed public proposal with two published public support sets.

This reads no private record, nonce, model, or protected directory.  The old
support covers the sealed split only under its published preparation binding.
"""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
TREE = BASE / 'work' / 'teams' / 'natural_heldout_custody'
SOURCES = {
    'proposal': (BASE / 'work/agreement_subject_followup/cases-v1.json',
                 '5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e'),
    'original_support': (TREE / 'experiments/natural_agreement_original_public_support_v1.json',
                         '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'),
    'successor_inputs': (TREE / 'experiments/natural_agreement_successor_inputs_v1.json',
                         '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'),
}


def load(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('public source mismatch')
    return json.loads(raw)


def rows(value, count):
    if type(value) is not list or len(value) != count:
        raise ValueError('wrong row count')
    if any(type(row) is not list or not row or any(type(x) is not int or x < 0 for x in row)
           for row in value):
        raise ValueError('bad public token rows')
    return value


def count_prefix(first, second):
    return sum(len(a) <= len(b) and a == b[:len(a)] for a in first for b in second)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=SOURCES['proposal'][0])
    parser.add_argument('--original-support', type=Path, default=SOURCES['original_support'][0])
    parser.add_argument('--successor-inputs', type=Path, default=SOURCES['successor_inputs'][0])
    parser.add_argument('--output', type=Path, default=BASE / 'work/reviews/agreement-subject-public-support-comparison-v1.json')
    args = parser.parse_args()
    proposal = load(args.case, SOURCES['proposal'][1])
    support = load(args.original_support, SOURCES['original_support'][1])
    successor = load(args.successor_inputs, SOURCES['successor_inputs'][1])
    proposed = rows([x['token_ids'] for x in proposal['cases']], 80)
    old = rows(support['rows'], 128)
    new = rows([x['input_ids'] for x in successor['cases']], 256)
    ordered = json.dumps(proposed, separators=(',', ':')).encode('utf-8')
    ordered_hash = hashlib.sha256(ordered).hexdigest()
    if ordered_hash != 'd31073b25d816258ff1bcbfcb7237eac5346b160a5ed2de65fea220bdb806ff1':
        raise ValueError('public row projection mismatch')
    counts = {
        'original_prefix_of_proposal': count_prefix(old, proposed),
        'proposal_prefix_of_original': count_prefix(proposed, old),
        'successor_prefix_of_proposal': count_prefix(new, proposed),
        'proposal_prefix_of_successor': count_prefix(proposed, new),
    }
    receipt = {
        'schema': 'mlg-agreement-subject-public-support-comparison-v1',
        'status': 'clear' if not any(counts.values()) else 'collision',
        'comparison': 'bidirectional-complete-row-prefix',
        'proposal_file_sha256': SOURCES['proposal'][1],
        'proposal_ordered_rows_sha256': ordered_hash,
        'original_support_sha256': SOURCES['original_support'][1],
        'successor_inputs_sha256': SOURCES['successor_inputs'][1],
        'original_sampler_sha256': '635f25553c20dd437f5cc4a8a9ca3a7a1453e7f8a517875fe75eb01ede2d5f35',
        'tokenizer_sha256': '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
        'counts': {'proposal': 80, 'original_support': 128, 'successor_inputs': 256},
        'prefix_relation_counts': counts,
        'scope': 'public full-input rows only; no private records, nonce, model, or membership query',
        'assumption': 'original prepared inputs follow the published sampler and tokenizer binding',
    }
    output = json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n'
    with args.output.open('xb') as stream:
        if stream.write(output) != len(output):
            raise OSError('short receipt write')
    print(receipt['status'], hashlib.sha256(output).hexdigest())


if __name__ == '__main__':
    main()
