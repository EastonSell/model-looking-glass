"""Tokenizer-only comparison of the fixed multisuffix proposal with named public inputs.

This does not read private inputs or query model weights.  Repeated references
across inventories are counted repeatedly, and exact-token-prefix disjointness
is not a claim about paraphrases, unknown prior exposures or pretraining.
"""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from tokenizers import Tokenizer

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True, help='Git checkout containing the pinned commit objects')
parser.add_argument('--candidate', type=Path, required=True, help='Multisuffix case manifest')
parser.add_argument('--prior', type=Path, required=True, help='Pinned prior signed-context exposure receipt')
parser.add_argument('--tokenizer', type=Path, required=True, help='Pinned BERT tokenizer.json')
args = parser.parse_args()
REPO, CANDIDATE = args.repo, args.candidate
CANDIDATE_SHA = 'bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e'
PRIOR = args.prior
PRIOR_SHA = 'e30b269a2a29fd93ad4ab4f51e859e2599e3c6f167a5d7698ac6403f50efecfc'
TOKENIZER = args.tokenizer
TOKENIZER_SHA = 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'
VARIANTS = ('terminal', 'period', 'present', 'outside', 'awake')
sha = lambda b: hashlib.sha256(b).hexdigest()


def checked(row):
    ids, mask = row['input_ids'], row['attention_mask']
    if (type(ids) is not list or not 2 <= len(ids) <= 64 or
            any(type(i) is not int or not 0 <= i < 30522 for i in ids) or
            ids[0] != 101 or ids[-1] != 102 or ids.count(103) != 1 or
            type(mask) is not list or len(mask) != len(ids) or
            any(type(i) is not int or i != 1 for i in mask)):
        raise ValueError('unexpected BERT token row')
    return tuple(ids)


def extract(name, d):
    if name == 'bert_negation_public_cases_v1.json':
        output = []
        for case in d['cases']:
            for cue in ('not', 'never', 'just', 'also'):
                ids = list(case['just_ids'])
                assert ids[d['indices']['cue']] == d['cues']['just']
                ids[d['indices']['cue']] = d['cues'][cue]
                output.append(checked({'input_ids': ids, 'attention_mask': d['attention_mask']}))
        return output
    if name in ('bert_cue_context_transfer_cases_v1.json', 'bert_cue_context_signed_cases_v1.json'):
        return [checked(row) for p in d['pairs'] for c in p['contexts'] for row in c['rows'].values()]
    if name.startswith('bert_cue_') and name.endswith('_cases_v1.json'):
        return [checked(row) for case in d['cases'] for row in case['rows'].values()]
    if name in ('bert_color_edit_public_cases_v1.json', 'bert_color_preference_public_cases_v2.json'):
        return [checked(row) for row in d['cases']]
    if name == 'bert_color_edit_public_exposures_v2.json':
        return [checked(row) for row in d['records']]
    if name == 'cross_model_agreement_cases_v1.json':
        return [checked(row['bert']) for row in d['cases']]
    if name == 'bert_agreement_suffix_cases_v1.json':
        return [checked(case['variants'][v]) for case in d['cases'] for v in ('terminal', 'continued')]
    return [checked(row) for row in d['rows']]


def collision(a, b):
    return any(x[:len(y)] == y or y[:len(x)] == x
               for x, y in ((a, b), (a[:-1], b[:-1])))


assert sha(CANDIDATE.read_bytes()) == CANDIDATE_SHA
assert sha(PRIOR.read_bytes()) == PRIOR_SHA
assert sha(TOKENIZER.read_bytes()) == TOKENIZER_SHA
document = json.loads(CANDIDATE.read_bytes())
assert document['model'] == 'distilbert/distilbert-base-uncased'
assert document['revision'] == '12040accade4e8a0f71eabdb258fecc2e7e948be'
assert document['tokenizer_asset_sha256']['tokenizer.json'] == TOKENIZER_SHA
assert document['variants'] == list(VARIANTS) and len(document['cases']) == 32
tokenizer = Tokenizer.from_file(str(TOKENIZER))
labels = [(case['case_id'], variant) for case in document['cases'] for variant in VARIANTS]
new = [checked(case['variants'][variant]) for case in document['cases'] for variant in VARIANTS]
assert len(labels) == len(set(labels)) == len(new) == 160
for case in document['cases']:
    assert set(case['variants']) == set(VARIANTS)
    for variant in VARIANTS:
        row = case['variants'][variant]
        encoded = tokenizer.encode(row['text'])
        assert encoded.ids == row['input_ids'] and encoded.attention_mask == row['attention_mask']
        assert row['input_ids'][row['output_position']] == 103

internal = [(i, j) for (i, a), (j, b) in itertools.combinations(enumerate(new), 2) if collision(a, b)]
expected = [(5*i, 5*i+j) for i in range(32) for j in range(1, 5)]
unexpected = [pair for pair in internal if pair not in expected]
missing_expected = [pair for pair in expected if pair not in internal]

source_items = json.loads(PRIOR.read_bytes())['sources']
source_items.append({'path': 'experiments/bert_cue_context_signed_cases_v1.json',
    'commit': '5060fc419610ca7f0e2c8ec5095fb5ce93cc3531',
    'sha256': '843649c3f105e26185d96f31d769b0585d219bb7a45ba445cf2c2548598927df', 'rows': 32})
sources, hits, rows_by_source = [], [], {}
prior_agreement = None
for item in source_items:
    name = item['path'].split(':')[-1].split('/')[-1]
    commit = ('8db4866943fbd9ff8a981a07fb2c58a3de364822'
              if name == 'bert_agreement_suffix_cases_v1.json' else item['commit'])
    assert commit is not None
    path = 'experiments/' + name
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show', commit + ':' + path])
    assert sha(raw) == item['sha256'], path
    data = json.loads(raw)
    rows = extract(name, data)
    assert len(rows) == item['rows'], path
    rows_by_source[path] = rows
    if name == 'bert_agreement_public_cases_v1.json':
        prior_agreement = data
    found = [(i, j) for i, a in enumerate(new) for j, b in enumerate(rows) if collision(a, b)]
    sources.append({'path': path, 'commit': commit, 'sha256': sha(raw),
                    'rows': len(rows), 'collisions': len(found)})
    hits.extend({'source': path, 'new_index': i, 'new_identity': labels[i], 'source_index': j}
                for i, j in found)

forms = sorted({case[k] for case in document['cases'] for k in ('subject', 'attractor')})
ids = {word: tokenizer.encode(word, add_special_tokens=False).ids for word in forms}
assert all(len(ids[word]) == 1 for word in forms)
noun_tokens = {word: [path for path, rows in rows_by_source.items()
                      if any(ids[word][0] in row for row in rows)] for word in forms}

# This deliberately abstracts away the locative preamble and lexical slots.
# It asks only whether the subject/attractor/MASK skeleton was seen in the
# named earlier agreement cases, not whether an entire proposed frame is old.
assert prior_agreement is not None
assert prior_agreement['coordinates']['subject'] == 2
assert prior_agreement['coordinates']['distractor'] == 5
old_cores = []
for row in prior_agreement['rows']:
    seq = list(row['input_ids'][1:7])
    seq[1], seq[4] = -1, -2
    old_cores.append(tuple(seq))
abstract_matches = 0
frame_templates = set()
for case in document['cases']:
    for variant in VARIANTS:
        row = case['variants'][variant]
        assert row['subject_position'] == 6 and row['output_position'] == 10
        seq = list(row['input_ids'][5:11])
        seq[1], seq[4] = -1, -2
        abstract_matches += tuple(seq) in old_cores
        frame_templates.add(row['text'].replace(' ' + case['subject'] + ' ', ' {subject} ', 1)
                            .replace(' ' + case['attractor'] + ' ', ' {attractor} ', 1))
assert sha(CANDIDATE.read_bytes()) == CANDIDATE_SHA
report = {'status': 'pass' if not (unexpected or missing_expected or hits) else 'collision',
    'candidate_sha256': CANDIDATE_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
    'new_rows': len(new), 'retokenized_rows': len(new),
    'within_new_pairs': len(new)*(len(new)-1)//2,
    'expected_terminal_prefix_relations': len(expected),
    'unexpected_within_relations': unexpected, 'missing_expected_relations': missing_expected,
    'source_count': len(sources), 'source_row_references': sum(s['rows'] for s in sources),
    'between_pairs_compared': len(new)*sum(s['rows'] for s in sources),
    'collisions': hits, 'sources': sources, 'noun_surface_token_sources': noun_tokens,
    'abstract_prior_core_matches': abstract_matches, 'frame_templates': sorted(frame_templates),
    'scope': 'Named public sources only; duplicates count repeatedly. SEP-stripped full-row '
             'prefix relations in either direction are checked; only the 128 explicitly '
             'expected terminal-versus-other relations are exempt. Abstract skeletons '
             'and surface-token reuse are descriptive; no private, unknown, semantic '
             'or pretraining disjointness is established.'}
print(json.dumps(report, sort_keys=True, indent=2))
