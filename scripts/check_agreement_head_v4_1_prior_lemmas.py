"""Check corrected public subject lemmas against prior head-study case rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path

PINS = {
    'cases': 'bfc93dbcf33f06c8266b0fabf351dfa9f17ab98d6501ca7380e2cd171bddc8b4',
    'evaluation': '51d14d87f7a318ff94e613d3ff88dfd338a6906675a92af41899b3a027eead7b',
    'inventory': '61a80aafd9d4aea075774776d830b017f8e4c5e1a4fd0f2643963d2075a6e4ee',
}

def read(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('public input digest differs: ' + path.name)
    return json.loads(raw)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'evaluation', 'inventory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    new = read(args.cases, PINS['cases'])['cases']
    evaluation = read(args.evaluation, PINS['evaluation'])['cases']
    records = read(args.inventory, PINS['inventory'])['records']
    by_id = {r['record_id']: r for r in records}
    if len(by_id) != len(records):
        raise ValueError('inventory record identity repeats')
    discovery = [by_id[f'observed-agreement-discovery-{i:02d}']['public_text'].split() for i in range(1,17)]
    if len(new) != 16 or len(evaluation) != 16 or any(len(row) < 3 or row[0] != 'The' for row in discovery):
        raise ValueError('public case format differs')
    development_forms = [row[1] for row in discovery]
    if any(development_forms[i+1] != development_forms[i]+'s' for i in range(0,16,2)):
        raise ValueError('development singular/plural sequence differs')
    development_lemmas = set(development_forms[::2])
    evaluation_lemmas = {row['subject'] for row in evaluation if row['number'] == 'singular'}
    new_lemmas = {row['lemma'] for row in new}
    if len(development_lemmas) != 8 or len(evaluation_lemmas) != 4 or len(new_lemmas) != 4:
        raise ValueError('public lemma counts differ')
    if any(row['subject'] != row['lemma'] + ('' if row['number'] == 'singular' else 's')
           or row['number'] not in ('singular','plural') for row in new):
        raise ValueError('new number forms differ')
    overlap = sorted(new_lemmas & (development_lemmas | evaluation_lemmas))
    if overlap:
        raise ValueError('a corrected lemma appeared in a prior head-study case')
    report = {'schema':'agreement-head-v4-1-public-lemma-check-v1', 'input_sha256':PINS,
              'development_lemmas':sorted(development_lemmas), 'evaluation_lemmas':sorted(evaluation_lemmas),
              'corrected_lemmas':sorted(new_lemmas), 'overlap':overlap,
              'scope':'only the public head-study development and evaluation case rows; no model or private query'}
    raw = (json.dumps(report,sort_keys=True,separators=(',',':'))+'\n').encode()
    for name in ('cases','evaluation','inventory'):
        read(getattr(args,name),PINS[name])
    with args.output.open('xb') as f:
        if f.write(raw) != len(raw): raise OSError('short write')
        f.flush(); os.fsync(f.fileno())
    print(hashlib.sha256(raw).hexdigest())

if __name__ == '__main__':
    main()
