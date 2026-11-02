"""Recheck one fixed public 64-case proposal against public natural supports."""
import hashlib
import json
from pathlib import Path

import tokenizers


BASE = Path(__file__).resolve().parents[2] / 'work'
TREE = BASE / 'teams/natural_heldout_custody'
FILES = {
    'proposal': (BASE / 'answer_format_cases_proposal_v2.json',
                 'cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a'),
    'original_support': (TREE / 'experiments/natural_agreement_original_public_support_v1.json',
                         '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'),
    'successor_inputs': (TREE / 'experiments/natural_agreement_successor_inputs_v1.json',
                         '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'),
    'tokenizer': (BASE / 'hf-cache/hub/models--distilbert--distilgpt2/snapshots/'
                  '2290a62682d06624634c1f46a6ad5be0f47f38aa/tokenizer.json',
                  '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'),
}


def read(name):
    path, expected = FILES[name]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('public input identity differs')
    return raw


def prefixes(a, b):
    return sum(len(x) <= len(y) and x == y[:len(x)] for x in a for y in b)


def main():
    proposal = json.loads(read('proposal'))
    original = json.loads(read('original_support'))['rows']
    successor = [x['input_ids'] for x in json.loads(read('successor_inputs'))['cases']]
    read('tokenizer')
    if tokenizers.__version__ != '0.22.2':
        raise ValueError('tokenizer version differs')
    tokenizer = tokenizers.Tokenizer.from_file(str(FILES['tokenizer'][0]))
    cases = proposal['cases']
    if len(cases) != 64 or len(original) != 128 or len(successor) != 256:
        raise ValueError('public support counts differ')
    rows = []
    for case in cases:
        ids = case['token_ids']
        if (case['prefix_text'] + case['suffix_text'] != case['prompt']
                or case['prefix_token_ids'] + case['suffix_token_ids'] != ids
                or tokenizer.encode(case['prefix_text'], add_special_tokens=False).ids != case['prefix_token_ids']
                or tokenizer.encode(case['prompt'], add_special_tokens=False).ids != ids):
            raise ValueError('public prompt or token projection differs')
        rows.append(ids)
    if len({tuple(x) for x in rows}) != 64:
        raise ValueError('duplicate public model inputs')
    counts = {'original_prefix_of_proposal': prefixes(original, rows),
              'proposal_prefix_of_original': prefixes(rows, original),
              'successor_prefix_of_proposal': prefixes(successor, rows),
              'proposal_prefix_of_successor': prefixes(rows, successor)}
    receipt = {'schema': 'mlg-answer-format-natural-public-support-comparison-v1',
        'status': 'clear' if not any(counts.values()) else 'collision',
        'comparison': 'bidirectional-complete-row-prefix',
        'proposal_sha256': FILES['proposal'][1], 'original_support_sha256': FILES['original_support'][1],
        'successor_inputs_sha256': FILES['successor_inputs'][1], 'tokenizer_sha256': FILES['tokenizer'][1],
        'ordered_rows_sha256': hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode('ascii')).hexdigest(),
        'counts': {'proposal': 64, 'original_support': 128, 'successor_inputs': 256},
        'prefix_relation_counts': counts,
        'scope': 'public full-input rows only; no private record, nonce, model, or membership query',
        'assumption': 'original prepared inputs follow the published sampler and tokenizer binding'}
    raw = json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode('ascii') + b'\n'
    with (BASE / 'reviews/answer-format-natural-public-comparison-v2.json').open('xb') as output:
        if output.write(raw) != len(raw):
            raise OSError('short public receipt write')
    print(receipt['status'], hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
