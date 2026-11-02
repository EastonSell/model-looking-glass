"""Independent tokenizer-only review of the public answer-format proposal.

This audit never loads a model or reads private cases. Its checks are fixed
against the published pre-forward protocol and exemplar amendment.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from tokenizers import Tokenizer


EXPECTED_PROPOSAL = 'cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a'
EXPECTED_TOKENIZER = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
EXPECTED_PROTOCOL = 'f60b240f1303fbf6ca785d2197e10d182aa1d34e6818ff8ed3e2f30a70c9307d'
EXPECTED_SCOPE = '2530ea0517847731d9c82731a17d58d23504738cc3e85b52fe259fe9bf4ed330'
FACTS = {
    'france': ('France', ' Paris', ' London'),
    'japan': ('Japan', ' Tokyo', ' Paris'),
    'italy': ('Italy', ' Rome', ' London'),
    'uk': ('the United Kingdom', ' London', ' Paris'),
}
PREFIXES = {
    'none': '',
    'neutral': 'A short geography prompt appears below.\n',
    'spain': 'The capital of Spain is Madrid.\n',
    'sweden': 'The capital of Sweden is Stockholm.\n',
}
TEMPLATES = {
    ('development', 'statement'): "In this exercise, {country}'s national capital is",
    ('development', 'question'): 'What city serves as the national capital of {country}? Answer:',
    ('heldout', 'statement'): 'The national capital city belonging to {country} is',
    ('heldout', 'question'): "Identify {country}'s national capital city. City:",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def related(a, b):
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--prior-full-top-cases', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    assert digest(args.cases) == EXPECTED_PROPOSAL
    assert digest(args.tokenizer) == EXPECTED_TOKENIZER
    proposal = json.loads(args.cases.read_text(encoding='utf8'))
    assert proposal['model_forward'] is False
    assert proposal['protocol_sha256'] == EXPECTED_PROTOCOL
    assert proposal['scope_sha256'] == EXPECTED_SCOPE
    assert proposal['tokenizer_sha256'] == EXPECTED_TOKENIZER
    cases = proposal['cases']
    assert len(cases) == 64
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    def tokens(value):
        return tokenizer.encode(value, add_special_tokens=False).ids

    order = list(itertools.product(('development', 'heldout'), FACTS,
                                   ('statement', 'question'), PREFIXES))
    assert len(order) == 64 and len({r['case_id'] for r in cases}) == 64
    suffixes, checked = {}, 0
    for row, (stage, fact, family, prefix) in zip(cases, order):
        country, factual, counterfactual = FACTS[fact]
        expected_suffix = TEMPLATES[(stage, family)].format(country=country)
        expected_prefix = PREFIXES[prefix]
        assert (row['role'], row['fact_id'], row['family'], row['prefix']) == (stage, fact, family, prefix)
        assert row['case_id'] == f'answer-format-{stage}-{fact}-{family}-{prefix}'
        assert row['country'] == country
        assert row['suffix_text'] == expected_suffix and row['prefix_text'] == expected_prefix
        assert row['prompt'] == expected_prefix + expected_suffix
        prefix_ids = tokens(expected_prefix) if expected_prefix else []
        suffix_ids = tokens(expected_suffix)
        assert len(prefix_ids) == (0 if prefix == 'none' else 8)
        assert row['prefix_token_ids'] == prefix_ids and row['suffix_token_ids'] == suffix_ids
        ids = tokens(row['prompt'])
        assert row['token_ids'] == ids == prefix_ids + suffix_ids
        assert suffixes.setdefault((stage, fact, family), (expected_suffix, suffix_ids)) == (expected_suffix, suffix_ids)
        expected_labels = {'factual': factual, 'counterfactual': counterfactual,
                           'madrid': ' Madrid', 'stockholm': ' Stockholm'}
        assert set(row['labels']) == set(expected_labels)
        label_ids = []
        for label, continuation in expected_labels.items():
            item = row['labels'][label]
            single = tokens(continuation)
            actual = tokens(row['prompt'] + continuation)
            assert len(single) == 1 and item['text'] == continuation
            assert item['token_id'] == single[0] and item['full_token_ids'] == actual == ids + single
            label_ids.append(single[0]); checked += 1
        assert len(set(label_ids)) == 4

    assert checked == 256 and len(suffixes) == 16
    for role, seed in (('development', 20261031), ('heldout', 20261101)):
        canonical = list(range(32))
        random.Random(seed).shuffle(canonical)
        assert proposal['execution_permutations'][role] == canonical
        assert proposal['seeds'][role] == seed
    all_rows = [tuple(row['token_ids']) for row in cases]
    internal = [(i, j) for i in range(64) for j in range(i + 1, 64) if related(all_rows[i], all_rows[j])]
    assert not internal
    prior = json.loads(args.prior_full_top_cases.read_text(encoding='utf8'))['cases'][:36]
    assert len(prior) == 36
    prior_rows = [tuple(row['token_ids']) for row in prior]
    collisions = [(i, j) for i in range(64) for j in range(36) if related(all_rows[i], prior_rows[j])]
    assert not collisions
    result = {
        'schema': 'independent-answer-format-case-review-v1',
        'status': 'passed',
        'proposal_sha256': digest(args.cases),
        'auditor_sha256': digest(__file__),
        'tokenizer_sha256': digest(args.tokenizer),
        'prior_full_top_cases_sha256': digest(args.prior_full_top_cases),
        'protocol_sha256': EXPECTED_PROTOCOL,
        'amendment_sha256': EXPECTED_SCOPE,
        'prompt_count': 64,
        'unique_suffix_group_count': 16,
        'single_token_appended_boundaries_checked': checked,
        'internal_full_prompt_pairs_checked': 2016,
        'prior_screen_full_prompt_pairs_checked': 2304,
        'permutations_independently_recomputed': True,
        'scope': ('Tokenizer-only consistency, exact fixed wording and labels, internal/past-screen prompt '
                  'relations. Semantic review is a separate human-readable finding. This does not validate '
                  'the complete public inventory, private scope, runtime, source chronology or execution.'),
    }
    raw = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + '\n').encode()
    with args.report.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'status': result['status'], 'report_sha256': hashlib.sha256(raw).hexdigest()}))


if __name__ == '__main__':
    main()
