"""Model-free structural checks for the multisuffix held-out manifest."""
import hashlib
import itertools
import json
from pathlib import Path

CASE_SHA = 'bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e'
VARIANTS = ('terminal','period','present','outside','awake')


def manifest():
    path=Path(__file__).resolve().parents[1]/'experiments/bert_multisuffix_agreement_cases_v1.json'
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==CASE_SHA
    return json.loads(raw)


def test_all_contexts_share_prefix_and_specific_suffix():
    data=manifest()
    assert data['variants']==list(VARIANTS) and len(data['cases'])==32
    expected={'terminal':[102],'period':[1012,102],'present':[2556,1012,102],
              'outside':[2648,1012,102],'awake':[8300,1012,102]}
    for i,case in enumerate(data['cases']):
        assert case['index']==i
        prefix=case['variants']['terminal']['input_ids'][:-1]
        for variant,tail in expected.items():
            row=case['variants'][variant]
            assert row['input_ids'][:11]==prefix
            assert row['input_ids'][11:]==tail
            assert row['input_ids'][10]==103
            assert row['attention_mask']==[1]*len(row['input_ids'])
            assert row['subject_position']==6 and row['output_position']==10


def test_donors_vary_only_subject_and_same_other_lemma():
    rows=manifest()['cases']
    for case in rows:
        same=rows[case['same_donor_index']]
        opposite=rows[case['opposite_donor_index']]
        assert same['lemma']==opposite['lemma']!=case['lemma']
        assert same['number']==case['number']!=opposite['number']
        assert same['frame']==opposite['frame']==case['frame']
        assert same['attractor_number']==opposite['attractor_number']==case['attractor_number']
        for variant in VARIANTS:
            a=case['variants'][variant]['input_ids']
            for donor in (same,opposite):
                b=donor['variants'][variant]['input_ids']
                assert len(a)==len(b) and [i for i,(x,y) in enumerate(zip(a,b)) if x!=y]==[6]


def test_only_the_expected_within_base_prefix_relations_exist():
    cases=manifest()['cases']
    rows=[tuple(c['variants'][variant]['input_ids']) for c in cases for variant in VARIANTS]
    def collides(a,b):
        return any(x[:len(y)]==y or y[:len(x)]==x for x,y in ((a,b),(a[:-1],b[:-1])))
    observed=[(i,j) for (i,a),(j,b) in itertools.combinations(enumerate(rows),2) if collides(a,b)]
    expected=[(5*i,5*i+j) for i in range(32) for j in range(1,5)]
    assert observed==expected
