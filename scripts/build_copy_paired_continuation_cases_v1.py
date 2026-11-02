"""Select fixed paired-continuation cases using only the pinned tokenizer and public inventory."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass import pretrained_exposures_v6 as v6
from looking_glass import pretrained_exposures_v7 as v7
from looking_glass import pretrained_exposures_v9 as v9
from looking_glass import pretrained_exposures_v12 as v12
from looking_glass import pretrained_exposures_v15 as v15
from looking_glass import pretrained_exposures_v44 as inventory
from scripts.build_copy_behavior_cases_v1 import input_sequences, related

PROTOCOL = 'docs/DISTILGPT2_PAIRED_CONTINUATION_PATCH_V1.md'
PROTOCOL_SHA = '61af860d38fe236bcac6a9d043cafd1ffe3dccc2cfa97fd2d0cc795c523f0924'
POOL_PATH = 'docs/DISTILGPT2_COPY_BEHAVIOR_SCREEN_V1.md'
POOL_SHA = '78907b74bf3edb44a47c794508a9385da23b2bf2f26d07df4fefe74542b58875'
AMENDMENT = 'docs/DISTILGPT2_PAIRED_CONTINUATION_POOL_AMENDMENT_V1.md'
AMENDMENT_SHA = '7fa718ec0e375c460136812d3f06a6bebdfe065d182abeb6d4c0cdb009509154'
PENDING_FRANCE_SHA = 'c8dd418bea948c8ca2b4d1e39e696374a28bd2f911bb16707415c38170fbba82'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
INVENTORY_SHA = '78d783dc37c1ace4c39300b62e67652d76f3323de79830628ecd64e46aa0bfa0'
PRIOR = (('experiments/distilgpt2_copy_head_prompt_dose_cases_v1.json',
          'b9a6fdca9998d26c1880c56160438a5b368621cf13bc1c4b212971512f1baf69'),
         ('experiments/distilgpt2_copy_output_dose_cases_v1.json',
          'e45fe8cfe8b28a4f4308a4d3b2206f35cebe8154c7b4301ac1356c28258b135e'),
         ('experiments/distilgpt2_copy_behavior_screen_cases_v1.json',
          '96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734'))
VARIANTS = (('A',(0,1,2,3,4,5,6),(0,1,2,3,4)),
            ('B',(0,1,2,3,4,6,5),(0,1,2,3,4)),
            ('A-order',(0,1,2,3,4,5,6),(1,2,0,3,4)),
            ('B-order',(0,1,2,3,4,6,5),(1,2,0,3,4)))

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def build(tokenizer_path, pending_france_path):
    protocol = (ROOT / PROTOCOL).read_bytes()
    pool_source = (ROOT / POOL_PATH).read_bytes()
    amendment = (ROOT / AMENDMENT).read_bytes()
    pending_france_raw = pending_france_path.read_bytes()
    if (sha(protocol) != PROTOCOL_SHA or sha(pool_source) != POOL_SHA or sha(amendment) != AMENDMENT_SHA
            or sha(tokenizer_path.read_bytes()) != TOKENIZER_SHA or sha(pending_france_raw) != PENDING_FRANCE_SHA):
        raise ValueError('fixed protocol, pools, tokenizer or pending public bundle differ')
    pools = re.findall(r'(?m)^\d+\. ([a-z ]+)$', pool_source.decode('utf-8'))
    additions = re.findall(r'(?m)^\d+\. ([a-z ]+)$', amendment.decode('utf-8'))
    if len(pools) != 8 or len(additions) != 8:
        raise ValueError('eight frozen pools required')
    pools = [a + ' ' + b for a, b in zip(pools, additions)]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    payload, digest = inventory._checked_inventory()
    if digest != INVENTORY_SHA:
        raise ValueError('construction inventory differs')
    known = list(input_sequences(payload))
    support = json.loads((ROOT / 'experiments/natural_agreement_original_public_support_v1.json').read_bytes())
    successor = json.loads((ROOT / 'experiments/natural_agreement_successor_inputs_v1.json').read_bytes())
    support_rows = [tuple(row) for row in support['rows']]
    successor_rows = [tuple(case['input_ids']) for case in successor['cases']]
    if len(support_rows) != 128 or len(successor_rows) != 256:
        raise ValueError('public support scope differs')
    known.extend(support_rows)
    known.extend(successor_rows)
    pending_france = json.loads(pending_france_raw)
    pending_rows = []
    for item in pending_france['rows']:
        pending_rows.append(tuple(item['token_ids']))
        pending_rows.extend(tuple(c['appended_token_ids']) for c in item['continuations'])
    pending_rows.extend(tuple(item['token_ids']) for item in pending_france['projection_basis'])
    if len(pending_rows) != 73:
        raise ValueError('pending France public comparison scope differs')
    known.extend(pending_rows)
    prior_ids = set(itertools.chain.from_iterable((*v6.PUBLIC_COPY_GROUPS, *v7.PUBLIC_EVALUATION_GROUPS,
        *v9.FOLLOWUP_GROUPS, *v12.FACTORIAL_GROUPS, *v15.PAIRS_DOSE_GROUPS)))
    for path, expected in PRIOR:
        raw = (ROOT / path).read_bytes()
        if sha(raw) != expected:
            raise ValueError('prior copying case bundle differs')
        prior_ids.update(itertools.chain.from_iterable(group['token_ids'] for group in json.loads(raw)['groups']))
    selected, selected_sequences, groups, cases, audit = set(), [], [], [], []
    for pool_index, pool in enumerate(pools):
        entries, seen = [], set()
        for position, word in enumerate(pool.split()):
            ids = tokenizer.encode(' ' + word, add_special_tokens=False).ids
            reason = ('repeated-word' if word in seen else 'not-single-token' if len(ids) != 1
                      else 'prior-copy-token' if ids[0] in prior_ids else None)
            seen.add(word)
            entries.append((position, word, ids, reason))
        for within in range(2):
            number = 2*pool_index + within + 1
            evaluations, eligible = [], []
            for position, word, ids, reason in entries:
                excluded = reason or ('earlier-selected-token' if ids[0] in selected else None)
                evaluations.append({'position':position, 'word':word, 'token_ids':ids, 'exclusion':excluded})
                if excluded is None:
                    eligible.append((position, word, ids[0]))
            chosen, rejected = None, []
            for combo in itertools.combinations(eligible, 7):
                positions = [item[0] for item in combo]
                words, ids = [item[1] for item in combo], [item[2] for item in combo]
                reason = 'duplicate-token' if len(set(ids)) != 7 else None
                rows = []
                for variant, (label, first, query) in enumerate(VARIANTS) if reason is None else ():
                    order = first + query
                    text = ''.join(' ' + words[i] for i in order)
                    row = tuple(ids[i] for i in order)
                    if tokenizer.encode(text, add_special_tokens=False).ids != list(row):
                        reason = 'input-tokenization'; break
                    for candidate in (5, 6):
                        if tokenizer.encode(text + ' ' + words[candidate], add_special_tokens=False).ids != [*row, ids[candidate]]:
                            reason = 'continuation-tokenization'; break
                    if reason:
                        break
                    rows.append((label, text, row, (row + (ids[5],), row + (ids[6],))))
                if reason is None:
                    for i, (_, _, row, exts) in enumerate(rows):
                        if any(related(x, y) for _, _, other, other_exts in rows[i+1:]
                               for x in (row, *exts) for y in (other, *other_exts)):
                            reason = 'within-family-relation'; break
                        if any(related(x, prior) for x in (row, *exts) for prior in selected_sequences):
                            reason = 'earlier-family-relation'; break
                        if any(related(x, prior) for x in (row, *exts) for prior in known):
                            reason = 'known-public-relation'; break
                if reason is None:
                    chosen = (combo, rows)
                    break
                rejected.append({'positions':positions, 'reason':reason})
            if chosen is None:
                raise ValueError(f'construction-ineligible: pool {pool_index+1}, family {number}')
            combo, rows = chosen
            words, ids = [item[1] for item in combo], [item[2] for item in combo]
            groups.append({'group':number, 'pool':pool_index+1, 'words':words, 'token_ids':ids})
            audit.append({'group':number, 'word_candidates':evaluations, 'rejected_combinations':rejected,
                          'chosen_positions':[item[0] for item in combo]})
            for variant, (label, text, row, extensions) in enumerate(rows):
                cases.append({'case_id':f'copy-paired-{number:02d}-{variant}', 'group':number,
                              'pool':pool_index+1, 'variant':variant, 'label':label, 'text':text,
                              'token_ids':list(row), 'candidate_f_id':ids[5], 'candidate_g_id':ids[6],
                              'correct_id':ids[5 if variant in (0,2) else 6]})
                selected_sequences.extend((row, *extensions))
            selected.update(ids)
    if len(selected) != 112 or len(groups) != 16 or len(cases) != 64:
        raise ValueError('unexpected fixed pool size')
    return {'schema':'distilgpt2-paired-continuation-cases-v1', 'model_id':'distilbert/distilgpt2',
            'model_revision':'2290a62682d06624634c1f46a6ad5be0f47f38aa',
            'protocol_sha256':PROTOCOL_SHA, 'pool_amendment_sha256':AMENDMENT_SHA,
            'tokenizer_sha256':TOKENIZER_SHA, 'pending_france_sha256':PENDING_FRANCE_SHA,
            'pending_france_sequences':len(pending_rows),
            'selection_inventory_version':inventory.INVENTORY_VERSION, 'selection_inventory_sha256':digest,
            'selection_inventory_record_count':len(payload['records']), 'public_support_rows':128,
            'public_successor_rows':256, 'prior_case_sha256':[v for _,v in PRIOR],
            'model_forward_asserted':False, 'candidate_word_pools':[p.split() for p in pools],
            'selection_audit':audit, 'groups':groups, 'cases':cases}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--pending-france', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.tokenizer.is_absolute() or not args.pending_france.is_absolute() or not args.out.is_absolute():
        raise ValueError('absolute tokenizer, pending public bundle and output paths required')
    raw = (json.dumps(build(args.tokenizer, args.pending_france), sort_keys=True, separators=(',',':'))+'\n').encode()
    with args.out.open('xb') as f:
        f.write(raw)
    print(json.dumps({'sha256':sha(raw), 'families':16, 'inputs':64,
                      'score_extensions':128, 'model_forward_asserted':False}))

if __name__ == '__main__':
    main()
