"""Compare two fixed public case bundles without model or private inputs."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--patch', type=Path, required=True)
parser.add_argument('--cases', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
patch_bytes, case_bytes = args.patch.read_bytes(), args.cases.read_bytes()
patch_sha, case_sha = (hashlib.sha256(x).hexdigest() for x in (patch_bytes, case_bytes))
if patch_sha != '6d0cbba21d15f9ab05df106617141c25a460f07be7ee57c53470f969eb79e518' or case_sha != '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe':
    raise ValueError('unexpected input identity')
patch, cases = json.loads(patch_bytes), json.loads(case_bytes)
if len(patch['cases']) != 32 or len(cases['cases']) != 54 or len(cases['answer_ids']) != 8:
    raise ValueError('unexpected case counts')
left = [row for case in patch['cases'] for row in [case['input_ids']] + [case['input_ids'] + [v] for v in patch['continuation_token_ids'].values()]]
right = [row for case in cases['cases'] for row in [case['token_ids']] + [case['token_ids'] + [v] for v in cases['answer_ids']]]
conflicts = sum(a[:len(b)] == b or b[:len(a)] == a for a in left for b in right)
record = {'schema': 'public-two-attractor-patch-substitution-comparison-v1', 'patch_cases_sha256': patch_sha, 'substitution_cases_sha256': case_sha, 'patch_sequences': len(left), 'substitution_sequences': len(right), 'both_direction_full_row_prefix_conflicts': conflicts, 'scope': 'public base rows and declared one-token candidates; no model or private rows'}
with args.output.open('xb') as out:
    out.write((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
if conflicts:
    raise SystemExit(1)
