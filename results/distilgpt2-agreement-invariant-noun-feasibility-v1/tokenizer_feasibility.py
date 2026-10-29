"""Tokenizer-only feasibility for invariant noun forms; no model is loaded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from tokenizers import Tokenizer

if 'torch' in sys.modules or 'transformers' in sys.modules:
    raise RuntimeError('unexpected model library import')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokenizer-json', required=True)
parser.add_argument('--prior-replay', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
tokenizer_path = Path(args.tokenizer_json).resolve()
raw = tokenizer_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6':
    raise ValueError('tokenizer bytes differ')
tokenizer = Tokenizer.from_file(str(tokenizer_path))
ids = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
pools = {
    'discovery': ['sheep', 'deer', 'fish', 'moose', 'elk', 'cod'],
    'evaluation': ['bison', 'squid', 'shrimp', 'carp', 'aircraft', 'spacecraft', 'swine', 'offspring'],
}
answers = {' is':318,' are':389,' was':373,' were':547,' the':262,' a':257,' in':287,' on':319}
if any(ids(text) != [expected] for text, expected in answers.items()):
    raise ValueError('answer IDs differ')
det = {word: ids(' '+word) for word in ('this', 'these')}
forms = {}
for group, nouns in pools.items():
    rows = []
    for noun in nouns:
        lone = ids(' '+noun)
        singular = ids(' this '+noun)
        plural = ids(' these '+noun)
        continuations = all(ids(' '+determiner+' '+noun+answer) == row + [answer_id]
                            for determiner, row in [('this', singular), ('these', plural)]
                            for answer, answer_id in answers.items())
        rows.append({'noun':noun, 'leading_noun_ids':lone, 'this_ids':singular,
                     'these_ids':plural, 'single_noun_token':len(lone)==1,
                     'determiner_only_difference':len(det['this'])==len(det['these'])==1
                          and len(lone)>0 and singular==det['this']+lone
                          and plural==det['these']+lone and det['this']!=det['these'],
                     'answer_boundaries_preserved':continuations})
    forms[group] = rows
prior = Path(args.prior_replay)
previous_raw = prior.read_bytes()
if hashlib.sha256(previous_raw).hexdigest() != 'a3bc57be6506e1464e2cfb3d4fcc534a44c06a958c01086cdf98964aa4a09a94':
    raise ValueError('prior saved replay differs')
previous = json.loads(previous_raw)
deliveries = [v for case in previous['cases'] for v in case['deliveries'].values()]
report = {'schema':'agreement-invariant-tokenizer-feasibility-v1',
          'tokenizer_sha256':hashlib.sha256(raw).hexdigest(), 'determiners':det,
          'candidate_forms':forms, 'answer_ids':answers, 'model_forwards':0,
          'model_feasibility_from_existing_evidence':{
              'prior_science_sha256':hashlib.sha256(previous_raw).hexdigest(),
              'prior_baseline_correct':previous['baseline']['overall'],
              'prior_valid_deliveries':sum(x['dose_valid'] for x in deliveries),
              'prior_delivery_count':len(deliveries),
              'scope':'Prior subject-form and intervention feasibility only; it did not test invariant nouns or this/these.'},
          'scope':'Tokenizer fragments and previously saved evidence only; no evaluation sentence, model load or new forward.'}
out = Path(args.output)
payload=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
with out.open('xb') as stream:
    stream.write(payload)
print(json.dumps({'sha256':hashlib.sha256(payload).hexdigest(),
                  'determiners':det,
                  'eligible':{k:[x['noun'] for x in v if x['single_noun_token'] and x['determiner_only_difference'] and x['answer_boundaries_preserved']] for k,v in forms.items()},
                  'prior_baselines':previous['baseline']['overall'],
                  'prior_valid_deliveries':sum(x['dose_valid'] for x in deliveries),
                  'new_forwards':0},sort_keys=True))
