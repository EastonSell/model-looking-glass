"""Select public propagation cases offline, without loading a model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
TOKENIZER_SHA256 = 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'
POOL = ('apple', 'berries', 'bread', 'cereal', 'custard', 'fruit', 'jam', 'jelly',
        'juice', 'lemonade', 'oatmeal', 'orange', 'porridge', 'soda', 'toast', 'wine', 'yogurt')
PRIOR = {
    'experiments/bert_agreement_exposures_v6.json': '0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb',
    'experiments/bert_matched_public_cases_v1.json': '5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb',
    'experiments/bert_taste_public_exposures_v2.json': 'a766d20778191280122a6b498b9bcb34907bfac761152ae7c3cf4df0fb9c5aa1',
    'experiments/bert_taste_matched_dose_public_exposures_v2.json': '9b08e38927187c6f43d8659603e27e0f5397034e01143e665ab01a139a64e845',
    'experiments/bert_taste_template_direction_public_exposures_v2.json': 'bf4d67ca16519a133f263d2da4b7a270bcd7a259a7cab4b8e8c85a04baa4f424',
}
TOKENS = {'mask': 103, 'sugar': 5699, 'lemon': 14380, 'sweet': 4086,
          'sour': 14768, 'the': 1996}
MATCHED_EXPOSURE = 'experiments/bert_taste_matched_dose_public_exposures_v2.json'
MATCHED_RESERVATION = 'experiments/bert_taste_matched_dose_public_cases_v1.json'
MATCHED_RESERVATION_SHA256 = '983be330b6691ce23986f26e50a345f77ca5cf46dd28a660a3a012a2cf598a7c'
TEMPLATE_EXPOSURE = 'experiments/bert_taste_template_direction_public_exposures_v2.json'
TEMPLATE_RESERVATION = 'experiments/bert_taste_template_direction_public_cases_v1.json'
TEMPLATE_RESERVATION_SHA256 = 'e59f5e1da0b3ce8c99ea9c231b4998e1c1e685d09c20bff4fdb734dc4910256f'
SELECTION_SOURCE_PATHS = ('scripts/bert_taste_propagation_cases_v1.py',
                          'docs/BERT_TASTE_CONTEXT_PROPAGATION_V1.md', *PRIOR,
                          MATCHED_RESERVATION, TEMPLATE_RESERVATION)


def selection_source(expected_commit):
    """Bind the committed selector, protocol and cited inventories before use."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update({'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull,
                'GIT_NO_REPLACE_OBJECTS': '1'})
    def git(*args):
        return subprocess.run(['git', '-c', 'core.fsmonitor=false', *args], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    head = git('rev-parse', 'HEAD').decode('ascii').strip()
    if head != expected_commit or git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('selection source checkout or commit differs')
    hashes = {}
    for name in SELECTION_SOURCE_PATHS:
        raw = (ROOT / name).read_bytes()
        if raw != git('show', f'{head}:{name}'):
            raise ValueError('selection source differs from committed bytes')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': head, 'selected_source_sha256': hashes}


def coordinates(row):
    ids, mask = row['input_ids'], row['attention_mask']
    if (type(ids) is not list or not ids or type(mask) is not list or len(mask) != len(ids)
            or any(type(x) is not int or x < 0 for x in ids)
            or any(type(x) is not int or x not in (0, 1) for x in mask)):
        raise ValueError('invalid input or attention mask')
    return ids, mask


def collision(a, b):
    ai, am = coordinates(a)
    bi, bm = coordinates(b)
    n = min(len(ai), len(bi))
    return ai[:n] == bi[:n] and am[:n] == bm[:n]


def select(encode, prior_sets):
    """Deterministically select six stems; `encode` returns ID/mask lists."""
    if len(set(POOL)) != len(POOL) or len(prior_sets) != len(PRIOR):
        raise ValueError('invalid candidate or inventory set')
    known = []
    for prior in prior_sets:
        if type(prior) is not list:
            raise ValueError('invalid inventory rows')
        for row in prior:
            coordinates(row)
            known.append(row)
    ordering = sorted(POOL, key=lambda stem: hashlib.sha256(
        b'bert-taste-context-propagation-v1\0' + stem.encode('utf-8')).digest())
    rows, decisions, stems = [], [], []
    for stem in ordering:
        candidate = []
        for template in ('after', 'with'):
            phrase = 'after adding' if template == 'after' else 'with some'
            for ingredient in ('sugar', 'lemon'):
                text = f'The {stem} tasted [MASK] {phrase} {ingredient}.'
                ids, mask = encode(text)
                row = {'id': f'{stem}-{template}-{ingredient}', 'stem': stem,
                       'template': template, 'ingredient': ingredient, 'text': text,
                       'input_ids': ids, 'attention_mask': mask}
                coordinates(row)
                candidate.append(row)
        aligned = all(len(row['input_ids']) == 10 and row['attention_mask'] == [1] * 10
                      and row['input_ids'][0] == 101 and row['input_ids'][-1] == 102
                      and row['input_ids'][1] == TOKENS['the']
                      and row['input_ids'][3] == 12595 and row['input_ids'][8] == 1012
                      and row['input_ids'][5:7] == ([2044, 5815] if row['template'] == 'after' else [2007, 2070])
                      and row['input_ids'][4] == TOKENS['mask']
                      and row['input_ids'].count(TOKENS['mask']) == 1
                      and row['input_ids'][7] == TOKENS[row['ingredient']] for row in candidate)
        aligned = aligned and len({row['input_ids'][2] for row in candidate}) == 1 and all(
            all(a == b for i, (a, b) in enumerate(zip(candidate[j]['input_ids'], candidate[j+1]['input_ids'])) if i != 7)
            for j in (0, 2))
        if not aligned:
            decisions.append({'stem': stem, 'status': 'rejected', 'reason': 'token_alignment'})
            continue
        if any(collision(row, other) for i, row in enumerate(candidate)
               for other in [*known, *rows, *candidate[i+1:]]):
            decisions.append({'stem': stem, 'status': 'rejected', 'reason': 'public_prefix_collision'})
            continue
        stems.append(stem)
        rows.extend(candidate)
        decisions.append({'stem': stem, 'status': 'accepted', 'reason': None})
        if len(stems) == 6:
            break
    if len(stems) != 6:
        raise ValueError('fewer than six stems passed the fixed selection rule')
    return {'candidate_pool': list(POOL), 'digest_order': ordering, 'decisions': decisions,
            'selected_stems': stems, 'rows': rows}


def load_priors():
    sets = []
    for path, expected in PRIOR.items():
        raw = (ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('prior inventory digest differs')
        inventory = json.loads(raw)
        if path == MATCHED_EXPOSURE:
            parent_raw = (ROOT / MATCHED_RESERVATION).read_bytes()
            if (hashlib.sha256(parent_raw).hexdigest() != MATCHED_RESERVATION_SHA256
                    or inventory.get('reservation_path') != MATCHED_RESERVATION
                    or inventory.get('reservation_sha256') != MATCHED_RESERVATION_SHA256):
                raise ValueError('matched exposure reservation binding differs')
            identity = json.loads(parent_raw)
            if inventory.get('rows') != identity.get('rows'):
                raise ValueError('matched exposure rows differ from reservation')
        elif path == TEMPLATE_EXPOSURE:
            parent_raw = (ROOT / TEMPLATE_RESERVATION).read_bytes()
            if (hashlib.sha256(parent_raw).hexdigest() != TEMPLATE_RESERVATION_SHA256
                    or inventory.get('reservation_path') != TEMPLATE_RESERVATION
                    or inventory.get('reservation_sha256') != TEMPLATE_RESERVATION_SHA256
                    or inventory.get('rows') != json.loads(parent_raw).get('rows')):
                raise ValueError('template exposure reservation binding differs')
            identity = inventory
        else:
            identity = inventory.get('identity', inventory)
        if identity.get('model') != MODEL or identity.get('revision') != REVISION:
            raise ValueError('prior inventory model differs')
        sets.append(inventory['rows'])
    return sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-json', required=True)
    parser.add_argument('--selection-source-commit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.selection_source_commit):
        raise ValueError('selection source commit must be a full lowercase Git SHA')
    source = selection_source(args.selection_source_commit)
    tokenizer_path = Path(args.tokenizer_json)
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('tokenizer bytes differ from the pinned revision')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for word in ('sugar', 'lemon', 'sweet', 'sour'):
        if tokenizer.encode(word, add_special_tokens=False).ids != [TOKENS[word]]:
            raise ValueError('target word tokenization differs')

    def encode(text):
        item = tokenizer.encode(text, add_special_tokens=True)
        return item.ids, item.attention_mask

    selected = select(encode, load_priors())
    manifest = {'schema': 'bert-taste-context-propagation-public-cases-v1',
                'status': 'public-exploratory-reservation', 'model': MODEL,
                'revision': REVISION, 'tokenizer_sha256': TOKENIZER_SHA256,
                'selection_source_commit': args.selection_source_commit,
                'selection_source_sha256': source['selected_source_sha256'],
                'prior_public_inventories': PRIOR, 'tokens': TOKENS,
                'coordinates': {'mask': 4, 'ingredient': 7, 'the': 1, 'sequence_length': 10},
                'scope': 'New public frames selected without model outputs; prior and pretraining exposure are unknown.',
                **selected}
    data = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with Path(args.output).open('xb') as stream:
        if stream.write(data) != len(data):
            raise OSError('short case-manifest write')
        stream.flush()
    print(hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
