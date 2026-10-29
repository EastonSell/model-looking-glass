"""Compare fixed cross-model BERT inputs to named, immutable public inventories.

No model or private split is read. Exact equality and either-direction prefix
relations are tested both with and without the final SEP token.
"""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: review_bert_public_exposure.py REPO CASE_MANIFEST')
REPO = Path(sys.argv[1]).resolve()
NEW = Path(sys.argv[2]).resolve()
NEW_SHA = 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503'
BASE = '2090fe8d1158dc7c9bb309563d76a5d842bc8d8b'
SOURCES = [
    ('bert_agreement_exposures_v6.json', BASE, '0d0b37539c30c8ff25ea5b1cfa95ecd4587f048434348a1c91ec010418553afb', 160),
    ('bert_matched_public_cases_v1.json', BASE, '5b708600dba78db9e4ee676650bc5e1f6b3269d52179a8e45d5d45fee55219bb', 64),
    ('bert_taste_public_exposures_v2.json', BASE, 'a766d20778191280122a6b498b9bcb34907bfac761152ae7c3cf4df0fb9c5aa1', 8),
    ('bert_taste_matched_dose_public_exposures_v2.json', BASE, '9b08e38927187c6f43d8659603e27e0f5397034e01143e665ab01a139a64e845', 12),
    ('bert_taste_template_direction_public_exposures_v2.json', BASE, 'bf4d67ca16519a133f263d2da4b7a270bcd7a259a7cab4b8e8c85a04baa4f424', 24),
    ('bert_taste_propagation_public_exposures_v2.json', BASE, '7978b729aedb2b01b93c9c86e115d2f2d95cdb0e50480edb59db6d5bebb080bf', 24),
    ('bert_negation_public_cases_v1.json', BASE, '4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff', 32),
    ('bert_cue_class_public_cases_v1.json', BASE, '57a55d89097ab1b955eb4e6bab38d48a4108aa24608b96827f3d27830fc7bf8c', 48),
    ('bert_cue_heldout_frames_cases_v1.json', 'b4b2d1d01c923cd37bce537973047261c585cbe6', '7a10dbcb598e48c2e8f56e2b75d8c03188ec04ca7a21ac071aea4b1e8294162c', 24),
    ('bert_cue_factorial_cases_v1.json', 'd5b23ff25134e6a22084bc0d9f1c8edab1863211', 'b91eba1e0a52f6a183a59b0bfbb365dd54d67341bd74a4fb56325e72a7f45d4f', 48),
    ('bert_cue_dose_response_cases_v1.json', 'b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4', '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a', 32),
    ('bert_color_edit_public_cases_v1.json', '0d24bddfa3a8f1d62fa7d106614a145db7eae873', '4f00000cf205137347d551fc2837e082ac632a515e92c1e5893b3312f1f71623', 37),
]
# Include the antecedent public inventories and case manifests individually;
# earlier rows may recur in the newer ledgers.
SOURCES += [(name, BASE, None, count) for name, count in [
    ('bert_agreement_exposures_v2.json',96), ('bert_agreement_exposures_v3.json',96),
    ('bert_agreement_exposures_v4.json',96), ('bert_agreement_exposures_v5.json',96),
    ('bert_agreement_public_cases_v1.json',96), ('bert_taste_public_cases_v1.json',8),
    ('bert_taste_matched_dose_public_cases_v1.json',12),
    ('bert_taste_template_direction_public_cases_v1.json',24),
    ('bert_taste_propagation_public_cases_v1.json',24),
]]


def checked(row):
    ids, mask = row['input_ids'], row['attention_mask']
    if (type(ids) is not list or not 2 <= len(ids) <= 64 or
            any(type(x) is not int or not 0 <= x < 30522 for x in ids) or
            ids[0] != 101 or ids[-1] != 102 or ids.count(103) != 1 or
            type(mask) is not list or len(mask) != len(ids) or
            any(type(x) is not int or x != 1 for x in mask)):
        raise ValueError('unexpected input/token namespace')
    return tuple(ids)


def extract(name, document):
    if name == 'bert_negation_public_cases_v1.json':
        cue_index = document['indices']['cue']
        result = []
        for case in document['cases']:
            for cue in ('not', 'never', 'just', 'also'):
                ids = list(case['just_ids'])
                if ids[cue_index] != document['cues']['just']:
                    raise ValueError('unexpected negation cue')
                ids[cue_index] = document['cues'][cue]
                result.append(checked({'input_ids': ids, 'attention_mask': document['attention_mask']}))
        return result
    if name.startswith('bert_cue_') and name.endswith('_cases_v1.json'):
        return [checked(row) for case in document['cases'] for row in case['rows'].values()]
    if name == 'bert_color_edit_public_cases_v1.json':
        return [checked(row) for row in document['cases']]
    return [checked(row) for row in document['rows']]


def collision(a, b):
    return any(x[:len(y)] == y or y[:len(x)] == x
               for x, y in ((a,b), (a[:-1],b[:-1])))


raw = NEW.read_bytes()
if hashlib.sha256(raw).hexdigest() != NEW_SHA:
    raise ValueError('new case bytes differ')
document = json.loads(raw)
if (len(document['cases']) != 32 or document['bert'] != {
        'model':'distilbert/distilbert-base-uncased',
        'revision':'12040accade4e8a0f71eabdb258fecc2e7e948be',
        'tokenizer_sha256':'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'}):
    raise ValueError('new model, tokenizer or case count differs')
new = [checked(case['bert']) for case in document['cases']]
internal = [(i,j) for (i,a),(j,b) in itertools.combinations(enumerate(new), 2) if collision(a,b)]
reports, all_hits = [], []
for name, commit, pinned, expected_count in SOURCES:
    path = 'experiments/' + name
    source = subprocess.check_output(['git', '-C', str(REPO), 'show', commit + ':' + path])
    digest = hashlib.sha256(source).hexdigest()
    if pinned is not None and digest != pinned:
        raise ValueError('source digest mismatch: ' + path)
    rows = extract(name, json.loads(source))
    if len(rows) != expected_count:
        raise ValueError('source row count mismatch: ' + path)
    hits = [(i,j) for i,a in enumerate(new) for j,b in enumerate(rows) if collision(a,b)]
    reports.append({'path':path, 'commit':commit, 'sha256':digest, 'rows':len(rows), 'collisions':len(hits)})
    all_hits += [{'path':path,'new_index':i,'source_index':j} for i,j in hits]
report = {'status':'pass' if not internal and not all_hits else 'collision',
          'new_commit':'a49cec14227ac1823b67f63e2793b513210fca7c', 'new_sha256':NEW_SHA,
          'new_rows':len(new), 'within_new_pairs':len(new)*(len(new)-1)//2,
          'within_new_collisions':internal, 'source_count':len(reports),
          'source_row_references':sum(x['rows'] for x in reports),
          'between_pairs_compared':len(new)*sum(x['rows'] for x in reports),
          'collisions':all_hits, 'sources':reports,
          'scope':'Named public rows only; repeated rows across inventories are counted repeatedly. '
                  'No protected, unknown or pretraining disjointness is established.'}
print(json.dumps(report, sort_keys=True, indent=2))
