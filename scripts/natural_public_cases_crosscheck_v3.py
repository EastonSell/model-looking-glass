"""Fixed public-only tokenizer and complete-row-prefix checks; no model/private data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tokenizers import Tokenizer


OLD_SHA = '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'
NEXT_SHA = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
TOKEN_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
SERVICE_SHA = '47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c'
ANSWERS = ((' is', 318), (' are', 389), (' was', 373), (' were', 547),
           (' the', 262), (' a', 257), (' in', 287), (' on', 319))


def read(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('public byte identity differs: ' + path.name)
    return json.loads(raw)


def count(a, b):
    return {'case_prefix_of_reference': sum(len(x) <= len(y) and x == y[:len(x)] for x in a for y in b),
            'reference_prefix_of_case': sum(len(y) <= len(x) and y == x[:len(y)] for x in a for y in b)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('invariant', 'head', 'original-support', 'successor-inputs', 'service-manifest', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if not all(path.is_absolute() for path in vars(args).values()):
        raise ValueError('all input and output paths must be absolute')
    if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKEN_SHA:
        raise ValueError('tokenizer identity differs')
    tok = Tokenizer.from_file(str(args.tokenizer))
    if tok.padding is not None or tok.truncation is not None:
        raise ValueError('tokenizer preprocessing differs')
    inv_path, head_path = args.invariant, args.head
    inv_sha = '188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b'
    head_sha = 'bfc93dbcf33f06c8266b0fabf351dfa9f17ab98d6501ca7380e2cd171bddc8b4'
    inv, head = read(inv_path, inv_sha), read(head_path, head_sha)
    if len(inv['cases']) != 68 or inv['answers'] != [v for _, v in ANSWERS] or len(head['cases']) != 16:
        raise ValueError('public case inventory differs')
    candidate_rows = {}
    invariant = []
    for item in inv['cases']:
        ids, text = item['token_ids'], item['text']
        if tok.encode(text, add_special_tokens=False).ids != ids:
            raise ValueError('invariant base tokenization differs')
        invariant.append(tuple(ids))
        for suffix, value in ANSWERS:
            extended = ids + [value]
            if tok.encode(text + suffix, add_special_tokens=False).ids != extended:
                raise ValueError('invariant extension tokenization differs')
            invariant.append(tuple(extended))
    heads = []
    for item in head['cases']:
        ids, text = item['token_ids'], item['text']
        if tok.encode(text, add_special_tokens=False).ids != ids or len(item['continuations']) != 4:
            raise ValueError('head base tokenization differs')
        heads.append(tuple(ids))
        for entry, (suffix, value) in zip(item['continuations'], ANSWERS[:4], strict=True):
            if (entry['suffix'] != suffix or entry['token_id'] != value or entry['tokens'] != ids + [value]
                    or tok.encode(text + suffix, add_special_tokens=False).ids != entry['tokens']):
                raise ValueError('head continuation tokenization differs')
            heads.append(tuple(entry['tokens']))
    candidate_rows.update(invariant=invariant, agreement_head=heads)
    old = read(args.original_support, OLD_SHA)['rows']
    successor = read(args.successor_inputs, NEXT_SHA)['cases']
    manifest = read(args.service_manifest, SERVICE_SHA)
    service = []
    for case in manifest['comparisons']:
        for role in ('reference', 'service'):
            row = tuple(case[role]['input_ids'])
            if row not in service:
                service.append(row)
    references = {'original_public_support': [tuple(row) for row in old],
                  'successor_public_pool': [tuple(row['input_ids']) for row in successor],
                  'six_short_service_rows': service,
                  'public_timing_input': [(15496, 995)]}
    if tuple(map(len, (invariant, heads, old, successor, service))) != (612, 80, 128, 256, 6):
        raise ValueError('public row counts differ')
    results = {name: {key: count(rows, value) for key, value in references.items()}
               for name, rows in candidate_rows.items()}
    cross = count(invariant, heads)
    if any(v for by_name in results.values() for pair in by_name.values() for v in pair.values()) or any(cross.values()):
        raise ValueError('a public complete-row-prefix relation exists')
    for path, expected in ((inv_path, inv_sha), (head_path, head_sha)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('case bytes changed while checking')
    report = {'schema': 'natural-public-cases-crosscheck-v3', 'case_sha256': {'invariant': inv_sha, 'agreement_head': head_sha},
              'candidate_sequences': {'invariant': 612, 'agreement_head': 80}, 'tokenizer_sha256': TOKEN_SHA,
              'public_support_sha256': {'original': OLD_SHA, 'successor': NEXT_SHA, 'short_service_manifest': SERVICE_SHA},
              'counts': results, 'invariant_vs_head': cross,
              'scope': 'Public-only comparisons; original protected coverage is conditional on recorded sampler/tokenizer preparation. No protected rows, private query, model or complete-exposure claim.'}
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':')) + '\n').encode()
    target = args.output
    with target.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short report write')
        stream.flush()
        os.fsync(stream.fileno())
    print(hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
