"""Post-failure diagnostic of saved L2 arithmetic; no model or hidden text."""
import array
import hashlib
import json
import math
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK / 't5-heldout-protected-v1'
RUN, AUDIT = ROOT / 'attempt', ROOT / 'audit'
failures = []
checked = 0
for path in sorted(RUN.glob('dose-*.json')):
    record = json.loads(path.read_bytes())
    data = (RUN / record['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['sha256']
    vectors = {}
    for part in record['layout']:
        x = array.array('f')
        x.frombytes(data[part['offset']:part['offset'] + part['bytes']])
        vectors[part['site'], part['side'], part['role']] = x
    for level in record['dose'].get('levels', []):
        role = f"{level['numerator']}-{level['denominator']}"
        for site in ('word', 'instruction', 'punctuation'):
            for side in range(2):
                own = vectors[site, side, 'own']
                delivered = vectors[site, side, role]
                diffs = [float(a) - float(b) for a, b in zip(delivered, own)]
                power = math.sqrt(math.fsum(d ** 2 for d in diffs))
                multiply = math.sqrt(math.fsum(d * d for d in diffs))
                saved = level['achieved'][f'{site}-{side}']
                checked += 1
                if power != saved or multiply != saved:
                    failures.append({'dose_record': path.name, 'level': role, 'site': site, 'side': side,
                                     'saved': saved, 'power': power, 'multiply': multiply,
                                     'power_minus_saved': power - saved, 'multiply_minus_saved': multiply - saved})
result = {'scope': 'post-failure saved-byte arithmetic diagnostic; no model execution',
          'checked': checked, 'mismatches': failures}
raw = json.dumps(result, sort_keys=True).encode('utf-8')
assert len(raw) < 64 * 1024
assert sum(p.stat().st_size for p in AUDIT.rglob('*') if p.is_file()) + len(raw) <= 1024 * 1024
with (AUDIT / 'scalar-dose-diagnostic.json').open('xb') as stream:
    stream.write(raw)
print(json.dumps({'checked': checked, 'mismatch_count': len(failures),
                  'report_sha256': hashlib.sha256(raw).hexdigest()}, sort_keys=True))
