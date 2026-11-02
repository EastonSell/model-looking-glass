"""Independently compare the public post-start v28 snapshot with 80 subject cases."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

if not __debug__:
    raise RuntimeError('Run without -O: assertions perform evidence checks')

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/teams/agreement_subject_executor'
COMMIT = '055e44a47503122826eaef05ad5f39dae7a60ac5'
V27_SHA = 'ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5'
V28_SHA = '2e883a9694e42f0eee8c2a82448f8b0a68b28ab91ffefa12ef57fd74b0ebb9f2'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='NUL', GIT_CONFIG_SYSTEM='NUL',
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    return subprocess.check_output(['git', '--no-replace-objects', '-C', str(REPO), *args],
                                   stderr=subprocess.DEVNULL, env=env, timeout=30)


git('merge-base', '--is-ancestor', 'baff8872b35193bcaf767c9e260ad4820e36332e', COMMIT)
package = types.ModuleType('looking_glass')
package.__path__ = []
sys.modules['looking_glass'] = package
for index in range(1, 29):
    stem = 'pretrained_exposures' if index == 1 else 'pretrained_exposures_v' + str(index)
    name = 'looking_glass.' + stem
    source = git('show', COMMIT + ':looking_glass/' + stem + '.py')
    module = types.ModuleType(name)
    module.__package__ = 'looking_glass'
    sys.modules[name] = module
    setattr(package, stem, module)
    exec(compile(source, name, 'exec', dont_inherit=True), module.__dict__)

v27_raw = (ROOT / 'work/agreement_subject_reservation/inventory.canonical.json').read_bytes()
v28_raw = (ROOT / 'work/exposure-v28-draft/inventory.canonical.json').read_bytes()
cases_raw = (ROOT / 'work/agreement_subject_followup/cases-v1.json').read_bytes()
assert digest(v27_raw) == V27_SHA and digest(v28_raw) == V28_SHA
assert digest(cases_raw) == '5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e'
v27, v28, cases = map(json.loads, (v27_raw, v28_raw, cases_raw))
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=True, allow_nan=False).encode('ascii')
assert canonical(package.pretrained_exposures_v28.inventory_payload()) == v28_raw == canonical(v28)
assert v28['predecessor'] == {'version': v27['version'], 'sha256': V27_SHA}
assert len(v27['records']) == 723 and len(v28['records']) == 787
assert v28['records'][:723] == v27['records']
own_ids = {'planned-' + case['case_id'] for case in cases['cases']}
own = [item for item in v28['records'] if item['record_id'] in own_ids]
assert len(own) == len(own_ids) == 80
by_id = {item['record_id']: item for item in own}
answers = [318, 389, 373, 547, 262, 257, 287, 319]
assert cases['answer_ids'] == answers
for case in cases['cases']:
    item = by_id['planned-' + case['case_id']]
    assert item['category'] == 'planned' and item['case_role'] == case['role']
    assert item['token_ids'] == case['token_ids'] and item['public_text'] == case['text']
    assert item['candidate_token_ids'] == answers

known_entries = []
for item in v28['records']:
    if item['record_id'] in own_ids:
        continue
    base = tuple(item['token_ids'])
    known_entries.append(base)
    candidates = list(item.get('candidate_token_ids', []))
    for group in item.get('candidate_token_groups', {}).values():
        candidates.extend(group)
    if item.get('target_token_id') is not None:
        candidates.append(item['target_token_id'])
    known_entries.extend(base + (token,) for token in candidates)
known = set(known_entries)
rows = [tuple(case['token_ids']) for case in cases['cases']]
proposals = rows + [row + (answer,) for row in rows for answer in answers]
relations = sum(1 for proposal in proposals for prior in known
                if proposal[:len(prior)] == prior or prior[:len(proposal)] == proposal)
assert len(known_entries) == 2600 and len(proposals) == 720 and relations == 0
result = {'v28_commit': COMMIT, 'v28_sha256': V28_SHA, 'v27_records_preserved': 723,
          'new_records': 64, 'subject_planned_records': len(own), 'subject_sequences': len(proposals),
          'other_known_entries': len(known_entries), 'other_distinct_sequences': len(known),
          'comparisons_including_duplicates': len(proposals) * len(known_entries),
          'distinct_comparisons': len(proposals) * len(known),
          'relations': relations,
          'scope': 'Public post-start comparison; child remains v27-bound; not complete exposure history.'}
print(json.dumps(result, sort_keys=True, separators=(',', ':')))
