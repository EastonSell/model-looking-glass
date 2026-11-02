"""Post-hoc bitwise comparison across fixed attractor-number pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CASE_SHA = 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'gpt-witness', 'bert-witness', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpt-sha256', required=True)
    p.add_argument('--bert-sha256', required=True)
    args = p.parse_args()
    if (sha(args.cases) != CASE_SHA or sha(args.gpt_witness) != args.gpt_sha256
            or sha(args.bert_witness) != args.bert_sha256):
        raise ValueError('external case or witness hash differs')
    cases = json.loads(args.cases.read_bytes())['cases']
    lookup = {(c['frame'], c['attractor_number'], c['lemma'], c['number']): c['index'] for c in cases}
    if len(cases) != 32 or len(lookup) != 32:
        raise ValueError('case grid differs')
    report = {'case_sha256': CASE_SHA, 'scope': 'Post-hoc saved-vector bit comparison; no model execution attestation.',
              'models': {}}
    for name, path, pin in (('gpt', args.gpt_witness, args.gpt_sha256),
                            ('bert', args.bert_witness, args.bert_sha256)):
        if path.stat().st_size != 192 * 2 * 768 * 4:
            raise ValueError('witness byte count differs')
        vectors = np.memmap(path, dtype='<f4', mode='r', shape=(192, 2, 768))
        equal = {key: 0 for key in ('clean', 'opposite', 'same', 'random')}
        for frame in range(2):
            for lemma in range(4):
                for number in range(2):
                    i, j = lookup[(frame, 0, lemma, number)], lookup[(frame, 1, lemma, number)]
                    equal['clean'] += vectors[i, 0].tobytes() == vectors[j, 0].tobytes()
                    for arm, offset in (('opposite', 1), ('same', 2), ('random', 3)):
                        equal[arm] += vectors[32 + 5*i + offset, 1].tobytes() == vectors[32 + 5*j + offset, 1].tobytes()
        report['models'][name] = {'witness_sha256': pin, 'pairs': 16, 'bit_identical': equal}
    data = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode()
    with args.output.open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short write')
    print('post-hoc report SHA-256', hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
