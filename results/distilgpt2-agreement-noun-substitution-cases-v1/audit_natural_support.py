"""Compare public prospective rows only; do not query private splits or a model."""
import hashlib
import json
from pathlib import Path
import subprocess
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT/'work/teams/natural_heldout_custody'
CASE_SHA = '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe'
ROWS_SHA = 'c34dd209b8037d0a024654c260377be691a9d87446e0b2ad873d9ace474e41f3'
OLD_SHA = '5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6'
NEXT_SHA = '6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4'
TOKEN_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
MODEL_REV = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
ANSWERS = ((' is',318),(' are',389),(' was',373),(' were',547),
           (' the',262),(' a',257),(' in',287),(' on',319))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, why):
    if not condition:
        raise ValueError(why)


def git(ref,path):
    return subprocess.check_output(['git','show',ref + ':' + path], cwd=REPO)


def counts(a,b):
    return (sum(len(x)<=len(y) and tuple(x)==tuple(y[:len(x)]) for x in a for y in b),
            sum(len(y)<=len(x) and tuple(y)==tuple(x[:len(y)]) for x in a for y in b))


def main():
    raw = (ROOT/'work/agreement_substitution_followup/cases-v1.json').read_bytes()
    require(sha(raw) == CASE_SHA, 'candidate bytes')
    obj = json.loads(raw)
    cases = obj['cases']
    require(len(cases) == 54 and obj['ordered_rows_sha256'] == ROWS_SHA
            and obj['answer_ids'] == [token for _,token in ANSWERS]
            and obj['tokenizer_sha256'] == TOKEN_SHA and obj['model_revision'] == MODEL_REV,
            'candidate identity')
    rows = [c['token_ids'] for c in cases]
    raw_rows = json.dumps(rows,separators=(',', ':'),ensure_ascii=True).encode('ascii')
    require(sha(raw_rows) == ROWS_SHA and len({tuple(x) for x in rows}) == 54, 'ordered rows')
    token_path = ROOT/'work/hf-cache/hub/models--distilbert--distilgpt2/snapshots'/MODEL_REV/'tokenizer.json'
    require(sha(token_path.read_bytes()) == TOKEN_SHA, 'tokenizer identity')
    tokenizer = Tokenizer.from_file(str(token_path))
    require(tokenizer.padding is None and tokenizer.truncation is None, 'tokenizer preprocessing')
    for c in cases:
        require(tokenizer.encode(c['text'],add_special_tokens=False).ids == c['token_ids'], 'input tokenization')
        for suffix, token in ANSWERS:
            require(tokenizer.encode(c['text'] + suffix,add_special_tokens=False).ids == c['token_ids'] + [token],
                    'answer extension tokenization')
    old_raw = git('8533660206f675b437ad4ea4fb4cef2116956ce1',
                  'experiments/natural_agreement_original_public_support_v1.json')
    next_raw = git('afdbc15317983a3948116c28eb857c5ddb869783',
                   'experiments/natural_agreement_successor_inputs_v1.json')
    require(sha(old_raw) == OLD_SHA and sha(next_raw) == NEXT_SHA, 'public support identity')
    old = json.loads(old_raw)['rows']
    nxt = [c['input_ids'] for c in json.loads(next_raw)['cases']]
    all_rows = rows + [r + [token] for r in rows for _,token in ANSWERS]
    require(len(old) == 128 and len(nxt) == 256 and len(all_rows) == 486, 'row count')
    old_counts, next_counts = counts(all_rows,old), counts(all_rows,nxt)
    report = {'cases':54,'answers_per_case':8,'inputs_plus_extensions':486,
      'candidate_sha256':CASE_SHA,'ordered_rows_sha256':ROWS_SHA,
      'old_public_support_sha256':OLD_SHA,'old_prefix_counts':old_counts,
      'successor_public_inputs_sha256':NEXT_SHA,'successor_prefix_counts':next_counts,
      'source_sha256':sha(Path(__file__).read_bytes()),
      'scope':'public inputs only; original-support conclusion conditional on recorded preparation; no model, private data, or membership query'}
    require(old_counts == (0,0) and next_counts == (0,0), 'public prefix relation')
    encoded = (json.dumps(report,sort_keys=True,separators=(',', ':')) + '\n').encode('utf-8')
    target = ROOT/'work/reviews/agreement-substitution-public-support-v1.json'
    with target.open('xb') as stream:
        stream.write(encoded)
    print(json.dumps({'report':str(target),'report_sha256':sha(encoded),'source_sha256':report['source_sha256']}))


if __name__ == '__main__':
    main()
