"""Compare the fixed masked-color inputs with cited public DistilBERT rows."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

CASE_SHA256 = '4f00000cf205137347d551fc2837e082ac632a515e92c1e5893b3312f1f71623'
SOURCE_COMMIT = '2090fe8d1158dc7c9bb309563d76a5d842bc8d8b'
FRAME_CASE_PATH = 'experiments/bert_cue_heldout_frames_cases_v1.json'
FRAME_CASE_COMMIT = 'b4b2d1d01c923cd37bce537973047261c585cbe6'
FRAME_CASE_SHA256 = '7a10dbcb598e48c2e8f56e2b75d8c03188ec04ca7a21ac071aea4b1e8294162c'
PRIOR = {
    'experiments/bert_agreement_exposures_v6.json': ('0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb', 160),
    'experiments/bert_matched_public_cases_v1.json': ('5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb', 64),
    'experiments/bert_taste_public_exposures_v2.json': ('a766d20778191280122a6b498b9bcb34907bfac761152ae7c3cf4df0fb9c5aa1', 8),
    'experiments/bert_taste_matched_dose_public_exposures_v2.json': ('9b08e38927187c6f43d8659603e27e0f5397034e01143e665ab01a139a64e845', 12),
    'experiments/bert_taste_template_direction_public_exposures_v2.json': ('bf4d67ca16519a133f263d2da4b7a270bcd7a259a7cab4b8e8c85a04baa4f424', 24),
    'experiments/bert_taste_propagation_public_exposures_v2.json': ('7978b729aedb2b01b93c9c86e115d2f2d95cdb0e50480edb59db6d5bebb080bf', 24),
    'experiments/bert_negation_public_cases_v1.json': ('4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff', 32),
    'experiments/bert_cue_class_public_cases_v1.json': ('57a55d89097ab1b955eb4e6bab38d48a4108aa24608b96827f3d27830fc7bf8c', 48),
}


def valid_row(ids):
    if (type(ids) is not list or not 2 <= len(ids) <= 64 or
            any(type(token) is not int or not 0 <= token < 30522 for token in ids) or
            ids[0] != 101 or ids[-1] != 102 or ids.count(103) != 1):
        raise ValueError('invalid DistilBERT masked row')
    return ids


def prefix_relation(left, right):
    """Compare the complete inputs, and both after removing one final SEP."""
    valid_row(left)
    valid_row(right)
    for a, b in ((left, right), (left[:-1], right[:-1])):
        if a[:len(b)] == b or b[:len(a)] == a:
            return True
    return False


def valid_input(row):
    ids = valid_row(row['input_ids'])
    mask = row['attention_mask']
    if (type(mask) is not list or len(mask) != len(ids) or
            any(type(value) is not int or value != 1 for value in mask)):
        raise ValueError('unexpected attention-mask namespace')
    return ids


def prior_rows(path, document):
    if path.endswith('bert_negation_public_cases_v1.json'):
        rows = []
        index = document['indices']['cue']
        for case in document['cases']:
            for cue in ('not', 'never', 'just', 'also'):
                ids = list(case['just_ids'])
                if ids[index] != document['cues']['just']:
                    raise ValueError('negation source cue identity differs')
                ids[index] = document['cues'][cue]
                rows.append(valid_input({'input_ids': ids, 'attention_mask': document['attention_mask']}))
        return rows
    if path.endswith('bert_cue_class_public_cases_v1.json'):
        return [valid_input(case['rows'][cue]) for case in document['cases']
                for cue in ('never', 'rarely', 'always', 'often')]
    if path == FRAME_CASE_PATH:
        return [valid_input(case['rows'][cue]) for case in document['cases']
                for cue in ('not', 'hardly', 'really', 'certainly')]
    return [valid_input(row) for row in document['rows']]


def git_blob(repository, commit, path):
    """Read the immutable Git bytes, unaffected by checkout line endings."""
    return subprocess.run(['git', '-C', str(repository), 'show', f'{commit}:{path}'],
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def check(repository, cases_path, frame_cases=None, from_git=False):
    case_bytes = cases_path.read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != CASE_SHA256:
        raise ValueError('case file differs from fixed tokenizer-only inventory')
    document = json.loads(case_bytes)
    if (document['model'] != 'distilbert/distilbert-base-uncased' or
            document['revision'] != '12040accade4e8a0f71eabdb258fecc2e7e948be' or
            len(document['cases']) != 37):
        raise ValueError('case model or count differs')
    new = [valid_input(row) for row in document['cases']]
    if len({tuple(row) for row in new}) != 37:
        raise ValueError('duplicate new inputs')
    internal = 0
    for index, left in enumerate(new):
        for right in new[index + 1:]:
            internal += 1
            if prefix_relation(left, right):
                raise ValueError('new input prefix collision')
    sources, comparisons = [], 0
    for path, (expected_sha, expected_count) in PRIOR.items():
        raw = git_blob(repository, SOURCE_COMMIT, path) if from_git else (repository / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError(f'public input source differs: {path}')
        prior = [valid_row(row) for row in prior_rows(path, json.loads(raw))]
        if len(prior) != expected_count:
            raise ValueError(f'public input count differs: {path}')
        for left in new:
            for right in prior:
                comparisons += 1
                if prefix_relation(left, right):
                    raise ValueError(f'public input prefix collision: {path}')
        sources.append({'path': path, 'sha256': expected_sha, 'rows': len(prior)})
    if from_git:
        raw = git_blob(repository, FRAME_CASE_COMMIT, FRAME_CASE_PATH)
    elif frame_cases is not None:
        raw = frame_cases.read_bytes()
    else:
        raise ValueError('supply the separately pinned cue-frame file or use --from-git')
    if hashlib.sha256(raw).hexdigest() != FRAME_CASE_SHA256:
        raise ValueError('fixed public cue-frame input bytes differ')
    prior = [valid_row(row) for row in prior_rows(FRAME_CASE_PATH, json.loads(raw))]
    if len(prior) != 24:
        raise ValueError('fixed public cue-frame input count differs')
    for left in new:
        for right in prior:
            comparisons += 1
            if prefix_relation(left, right):
                raise ValueError('fixed public cue-frame input prefix collision')
    sources.append({'path': FRAME_CASE_PATH, 'reference_commit': FRAME_CASE_COMMIT,
                    'sha256': FRAME_CASE_SHA256, 'rows': len(prior),
                    'status': 'public fixed proposal; no completion inferred'})
    return {'schema': 'distilbert-color-edit-public-input-check-v1',
            'source_commit': SOURCE_COMMIT, 'case_sha256': CASE_SHA256,
            'new_rows': len(new), 'within_new_pairs': internal,
            'public_row_references': sum(item['rows'] for item in sources),
            'public_pairs_compared': comparisons, 'full_or_sep_removed_prefix_collisions': 0,
            'sources': sources,
            'scope': ('Cited public input rows only; some prior rows recur across sources. '
                      'No protected, unknown or pretraining exposure is certified.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--cue-frame-cases', type=Path,
                        help='exact public proposal at the pinned separate commit')
    source.add_argument('--from-git', action='store_true',
                        help='read all pinned inputs directly from immutable Git objects')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.repository, args.cases, args.cue_frame_cases, args.from_git)
    with args.output.open('xb') as stream:
        stream.write((json.dumps(result, sort_keys=True, indent=2) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
