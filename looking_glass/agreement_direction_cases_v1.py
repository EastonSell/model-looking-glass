"""Build prospective public agreement cases with a tokenizer, not a model."""
import argparse
import hashlib
import json
from pathlib import Path

REGULAR = [('chef', 'chefs'), ('baker', 'bakers'), ('nurse', 'nurses'),
           ('pilot', 'pilots'), ('guard', 'guards'), ('worker', 'workers'),
           ('singer', 'singers'), ('dancer', 'dancers'), ('captain', 'captains'),
           ('student', 'students')]
IRREGULAR = [('man', 'men'), ('woman', 'women'), ('child', 'children'),
             ('person', 'people'), ('mouse', 'mice'), ('goose', 'geese'),
             ('foot', 'feet'), ('tooth', 'teeth'), ('wolf', 'wolves'),
             ('knife', 'knives')]
ATTRACTORS = [('fence', 'fences'), ('tree', 'trees'), ('window', 'windows'),
              ('door', 'doors'), ('bag', 'bags'), ('basket', 'baskets'),
              ('room', 'rooms'), ('table', 'tables'), ('car', 'cars'), ('box', 'boxes')]
RELATIONS = ['near', 'beside', 'behind', 'outside']
ANSWERS = [(' is', 318), (' are', 389), (' was', 373), (' were', 547)]
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
WORKER_SHA = 'e087d03ae0cc2c3d0c0c71a241b4dd583fedea4f609364c96ad1ec7c8412e5e4'
SCREENED_SHA = 'fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3'
YEAR_SHA = '419d74c23333407f9a9dc0b6f7d7b70764ab5d1078e2b0a4c84d37ba194df8a9'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bounded(path, limit):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('input file exceeds its size bound')
    return raw


def verify_inventory(raw, version, expected_sha):
    if sha(raw) != expected_sha:
        raise ValueError('known inventory payload hash differs')
    payload = json.loads(raw)
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=True, allow_nan=False).encode('utf-8')
    if raw != canonical or payload.get('version') != version:
        raise ValueError('known inventory version or canonical bytes differ')
    return payload


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False).ids


def has_relation(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def select_pairs(tokenizer, pool, used):
    chosen, log = [], []
    for first, second in pool:
        a, b = encode(tokenizer, ' ' + first), encode(tokenizer, ' ' + second)
        if len(a) != 1 or len(b) != 1:
            reason = 'not_one_token'
        elif a[0] == b[0]:
            reason = 'same_token'
        elif a[0] in used or b[0] in used:
            reason = 'previously_selected_token'
        else:
            reason = 'selected'
            used.update((a[0], b[0]))
            chosen.append({'words': [first, second], 'token_ids': [a[0], b[0]]})
        log.append({'words': [first, second], 'token_ids': [a, b], 'reason': reason})
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise ValueError('a fixed word pool did not yield four pairs')
    return chosen, log


def _continuations(tokenizer, text, ids):
    for answer, expected in ANSWERS:
        if encode(tokenizer, answer) != [expected] or encode(tokenizer, text + answer) != ids + [expected]:
            raise ValueError('answer tokenizer boundary changed')


def build_cases(tokenizer):
    used = set()
    regular, regular_log = select_pairs(tokenizer, REGULAR, used)
    irregular, irregular_log = select_pairs(tokenizer, IRREGULAR, used)
    attractors, attractor_log = select_pairs(tokenizer, ATTRACTORS, used)
    targets, controls = [], []
    if encode(tokenizer, 'The') != [464] or encode(tokenizer, ' the') != [262]:
        raise ValueError('fixed prefix tokenization differs')
    for frame in range(4):
        relation = RELATIONS[frame]
        relation_ids = encode(tokenizer, ' ' + relation)
        if len(relation_ids) != 1:
            raise ValueError('relation is not one token')
        for attractor_index, attractor_number in enumerate(('singular', 'plural')):
            family = 2 * frame + attractor_index
            attractor = attractors[frame]['words'][attractor_index]
            attractor_id = attractors[frame]['token_ids'][attractor_index]
            for variant, (morphology, number, pair, form) in enumerate((
                ('regular', 'singular', regular[frame], 0), ('regular', 'plural', regular[frame], 1),
                ('irregular', 'singular', irregular[frame], 0), ('irregular', 'plural', irregular[frame], 1),
            )):
                text = f'The {pair["words"][form]} {relation} the {attractor}'
                ids = encode(tokenizer, text)
                if ids != [464, pair['token_ids'][form], relation_ids[0], 262, attractor_id]:
                    raise ValueError('target is not the fixed five-token structure')
                _continuations(tokenizer, text, ids)
                targets.append({'case_id': f'agreement-direction-{family+1:02d}-{variant}',
                                'role': 'target', 'frame_index': frame, 'family_index': family,
                                'variant': variant, 'morphology': morphology, 'subject_number': number,
                                'subject_lemma': pair['words'][0], 'subject_form': pair['words'][form],
                                'attractor_number': attractor_number, 'attractor_form': attractor,
                                'relation': relation, 'subject_index': 1, 'text': text, 'token_ids': ids})
            context = f'{relation.capitalize()} the {attractor}'
            ids = encode(tokenizer, context)
            if not ids or len(ids) > 32:
                raise ValueError('context-only input has invalid length')
            _continuations(tokenizer, context, ids)
            controls.append({'case_id': f'agreement-direction-context-{family+1:02d}',
                             'role': 'context', 'frame_index': frame, 'family_index': family,
                             'attractor_number': attractor_number, 'attractor_form': attractor,
                             'relation': relation, 'text': context, 'token_ids': ids})
    cases = targets + controls
    rows = [tuple(case['token_ids']) for case in cases]
    if len(rows) != 40 or any(has_relation(a, b) for i, a in enumerate(rows) for b in rows[i+1:]):
        raise ValueError('new case rows have an equality or prefix collision')
    return cases, {'regular': regular, 'irregular': irregular, 'attractors': attractors,
                   'eligibility': {'regular': regular_log, 'irregular': irregular_log,
                                   'attractors': attractor_log}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tokenizer-json', 'inventory', 'worker-record', 'screened-cases', 'year-cases', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    tokenizer_bytes = read_bounded(args.tokenizer_json, 2 * 1024 * 1024)
    if sha(tokenizer_bytes) != TOKENIZER_SHA:
        raise ValueError('pinned tokenizer hash differs')
    from tokenizers import Tokenizer, __version__ as tokenizers_version
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
    cases, selection = build_cases(tokenizer)
    inventory_sha = '3fa3eda9508ca314505aa658b1c02d04dbc98681bfc776337cd9ca1c9c384f5a'
    inventory = verify_inventory(read_bounded(args.inventory, 16 * 1024 * 1024),
                                 'distilgpt2-development-exposures-v18', inventory_sha)
    worker_raw = read_bounded(args.worker_record, 1024 * 1024)
    screened_raw = read_bounded(args.screened_cases, 4 * 1024 * 1024)
    year_raw = read_bounded(args.year_cases, 4 * 1024 * 1024)
    if sha(worker_raw) != WORKER_SHA or sha(screened_raw) != SCREENED_SHA or sha(year_raw) != YEAR_SHA:
        raise ValueError('additional public exposure source hash differs')
    worker = json.loads(worker_raw)['rows']
    screened = [case['token_ids'] for case in json.loads(screened_raw)['cases']]
    year = [case['input_ids'] for case in json.loads(year_raw)['cases']]
    known = [row['token_ids'] for row in inventory['records']]
    if len(worker) != 8 or len(screened) != 44 or len(year) != 48:
        raise ValueError('additional public exposure counts differ')
    proposed = [tuple(row['token_ids']) for row in cases]
    for other in (known, worker, screened, year):
        if any(has_relation(row, tuple(existing)) for row in proposed for existing in other):
            raise ValueError('candidate relation to known public row')
    row_bytes = json.dumps([list(row) for row in proposed], separators=(',', ':')).encode('ascii')
    result = {
        'schema': 'distilgpt2-agreement-direction-cases-v1',
        'scope': 'Public nonblind proposed inputs; tokenizer only, no model forward.',
        'model_id': 'distilbert/distilgpt2', 'model_revision': '2290a62682d06624634c1f46a6ad5be0f47f38aa',
        'protocol_revision': 'a39121c2f3cf2602281252a8856d48efadcb2a98',
        'protocol_sha256': 'f539ad55b347f88641de3b76565104dc7be43e0c5d15f6d8fefdb4173694bbea',
        'direction_revision': '15c0ef7f8141322aeb1fa31dc02cab5a68fc16b8',
        'direction_manifest_sha256': '91dc4170eff6ac60416e5b3f884f2a28f41f9a4c1ad7339b409e32baeadf3a0e',
        'case_preparer_source_sha256': sha(Path(__file__).read_bytes()),
        'publication_requirement': 'caller must independently verify protocol and direction publication before case construction',
        'tokenizer_json_sha256': TOKENIZER_SHA, 'tokenizers_version': tokenizers_version,
        'known_inventory_revision': '94a55846d438cf2fa8bc8fbbde9c5609de5021d0',
        'known_inventory_version': inventory['version'], 'known_inventory_sha256': inventory_sha,
        'known_inventory_rows_checked': len(known),
        'additional_public_checks': [
            {'sha256': WORKER_SHA, 'input_count': len(worker), 'scope': 'operator-reported forwarded rows'},
            {'sha256': SCREENED_SHA, 'input_count': len(screened), 'scope': 'public proposed prompt rows'},
            {'sha256': YEAR_SHA, 'input_count': len(year), 'scope': 'public proposed year-direction rows'},
        ],
        'private_split_check': 'pending separately scoped custodian comparison before any forward',
        'ordered_input_rows_sha256': sha(row_bytes), 'answer_ids': [item[1] for item in ANSWERS],
        'expected_forwards': 784, 'selection': selection, 'cases': cases,
    }
    out = Path(args.output)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'case_file_sha256': sha(out.read_bytes()), 'row_sha256': sha(row_bytes),
                      'known_rows': len(known), 'rows': len(cases)}, sort_keys=True))


if __name__ == '__main__':
    main()
