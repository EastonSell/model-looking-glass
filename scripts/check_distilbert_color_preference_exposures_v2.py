"""Compare the planned color-preference inputs to named public BERT inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

CASES_SHA256 = 'f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10'
EARLIER = '2090fe8d1158dc7c9bb309563d76a5d842bc8d8b'
SOURCES = (
    (EARLIER, 'experiments/bert_agreement_exposures_v6.json', '0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb', 160, 'rows'),
    (EARLIER, 'experiments/bert_matched_public_cases_v1.json', '5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb', 64, 'rows'),
    (EARLIER, 'experiments/bert_taste_public_exposures_v2.json', 'a766d20778191280122a6b498b9bcb34907bfac761152ae7c3cf4df0fb9c5aa1', 8, 'rows'),
    (EARLIER, 'experiments/bert_taste_matched_dose_public_exposures_v2.json', '9b08e38927187c6f43d8659603e27e0f5397034e01143e665ab01a139a64e845', 12, 'rows'),
    (EARLIER, 'experiments/bert_taste_template_direction_public_exposures_v2.json', 'bf4d67ca16519a133f263d2da4b7a270bcd7a259a7cab4b8e8c85a04baa4f424', 24, 'rows'),
    (EARLIER, 'experiments/bert_taste_propagation_public_exposures_v2.json', '7978b729aedb2b01b93c9c86e115d2f2d95cdb0e50480edb59db6d5bebb080bf', 24, 'rows'),
    (EARLIER, 'experiments/bert_negation_public_cases_v1.json', '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff', 32, 'negation'),
    (EARLIER, 'experiments/bert_cue_class_public_cases_v1.json', '57a55d89097ab1b955eb4e6bab38d48a4108aa24608b96827f3d27830fc7bf8c', 48, 'variants'),
    ('b4b2d1d01c923cd37bce537973047261c585cbe6', 'experiments/bert_cue_heldout_frames_cases_v1.json', '7a10dbcb598e48c2e8f56e2b75d8c03188ec04ca7a21ac071aea4b1e8294162c', 24, 'variants'),
    ('8c03137ccee6664e4ace2dd324f7b0108ab88a28', 'experiments/bert_color_edit_public_exposures_v2.json', '2129d9642548b69928e9e3f4bcb8d044e0383b9671a3f01fe9bb0754ef3a7824', 37, 'records'),
    ('d5b23ff25134e6a22084bc0d9f1c8edab1863211', 'experiments/bert_cue_factorial_cases_v1.json', 'b91eba1e0a52f6a183a59b0bfbb365dd54d67341bd74a4fb56325e72a7f45d4f', 48, 'variants'),
    ('b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4', 'experiments/bert_cue_dose_response_cases_v1.json', '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a', 32, 'variants'),
    ('161c143285dbe7458be8b4ad74496d2383e42fe5', 'experiments/bert_cue_dose_response_public_exposures_v2.json', '61988985ca06c0a4113c4791792b1e2760b5139a1cc8aeaa1fe37f207f35f6ad', 32, 'rows'),
    ('a49cec14227ac1823b67f63e2793b513210fca7c', 'experiments/cross_model_agreement_cases_v1.json', 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503', 32, 'nested-bert'),
)


def valid_input(row):
    ids = row['input_ids']
    mask = row['attention_mask']
    if (type(ids) is not list or not 2 <= len(ids) <= 64 or
            any(type(x) is not int or not 0 <= x < 30522 for x in ids) or
            ids[0] != 101 or ids[-1] != 102 or ids.count(103) != 1 or
            type(mask) is not list or len(mask) != len(ids) or
            any(type(x) is not int or x != 1 for x in mask)):
        raise ValueError('invalid masked input row')
    return ids


def source_rows(document, kind):
    if kind in ('rows', 'records'):
        return [valid_input(row) for row in document[kind]]
    if kind == 'nested-bert':
        return [valid_input(case['bert']) for case in document['cases']]
    if kind == 'variants':
        return [valid_input(row) for case in document['cases']
                for _, row in sorted(case['rows'].items())]
    if kind == 'negation':
        rows = []
        index = document['indices']['cue']
        for case in document['cases']:
            for cue in ('not', 'never', 'just', 'also'):
                ids = list(case['just_ids'])
                if ids[index] != document['cues']['just']:
                    raise ValueError('negation cue identity differs')
                ids[index] = document['cues'][cue]
                rows.append(valid_input({'input_ids': ids, 'attention_mask': document['attention_mask']}))
        return rows
    raise ValueError('unrecognized source format')


def related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def check(repository, cases_path):
    raw = cases_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASES_SHA256:
        raise ValueError('case bytes differ')
    document = json.loads(raw)
    if (document['model'] != 'distilbert/distilbert-base-uncased' or
            document['revision'] != '12040accade4e8a0f71eabdb258fecc2e7e948be'):
        raise ValueError('model identity differs')
    rows = [valid_input(case) for case in document['cases']]
    if len(rows) != 37 or len({tuple(row) for row in rows}) != 37:
        raise ValueError('expected 37 distinct rows')
    within = 0
    for index, a in enumerate(rows):
        for b in rows[index + 1:]:
            within += 1
            if related(a, b) or related(a[:-1], b[:-1]):
                raise ValueError('new inputs have a full or SEP-stripped prefix relation')
    comparisons = 0
    sources = []
    for commit, path, expected_sha, expected_count, kind in SOURCES:
        raw = subprocess.run(['git', '-C', str(repository), 'show', f'{commit}:{path}'],
                             check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError(f'public source differs: {path}')
        prior = source_rows(json.loads(raw), kind)
        if len(prior) != expected_count:
            raise ValueError(f'public source count differs: {path}')
        for a in rows:
            for b in prior:
                comparisons += 1
                if related(a, b) or related(a[:-1], b[:-1]):
                    raise ValueError(f'public input prefix relation: {path}')
        sources.append({'commit': commit, 'path': path, 'sha256': expected_sha, 'rows': len(prior)})
    return {'schema': 'distilbert-color-preference-public-comparison-v2',
            'case_sha256': CASES_SHA256, 'new_rows': len(rows), 'within_new_pairs': within,
            'public_row_references': sum(source['rows'] for source in sources),
            'public_pairs_compared': comparisons, 'full_or_sep_removed_prefix_collisions': 0,
            'sources': sources,
            'scope': ('Named public input snapshots only; prior rows can recur. The current '
                      'state of outside inventories, private inputs and pretraining is not certified.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.repository, args.cases)
    with args.output.open('xb') as stream:
        stream.write((json.dumps(result, sort_keys=True, indent=2) + '\n').encode())


if __name__ == '__main__':
    main()
