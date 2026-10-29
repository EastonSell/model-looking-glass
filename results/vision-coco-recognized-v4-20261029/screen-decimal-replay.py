"""High-precision score check of the closed public-candidate v4 screen.

Reads retained float32 logits only; no model, tokenizer, or image is loaded.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import struct
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = args.data
manifest_path = args.manifest
result_path = args.result
manifest_raw = manifest_path.read_bytes()
result_raw = result_path.read_bytes()
if not hashlib.sha256(manifest_raw).hexdigest() == '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb':
    raise ValueError('saved-evidence check failed')
if not hashlib.sha256(result_raw).hexdigest() == '711894680eac6bed8e60b8acb4413775cce51f5630115e561ecbfe0504efa648':
    raise ValueError('saved-evidence check failed')
images = json.loads(manifest_raw)['images']
result = json.loads(result_raw)
if not len(images) == len(result['screen']) == 64:
    raise ValueError('saved-evidence check failed')
rows = []
maxima = {k: Decimal(0) for k in ('margin', 'dog_mass', 'cat_mass', 'off_mass')}
with localcontext() as context:
    context.prec = 60
    for n, (image, stored) in enumerate(zip(images, result['screen']), 1):
        raw = (data / f'forward-{n:03d}.bin').read_bytes()
        metadata = json.loads((data / f'forward-{n:03d}.json').read_bytes())
        if not (len(raw) == 4000 and hashlib.sha256(raw).hexdigest() == metadata['sha256']):
            raise ValueError('saved-evidence check failed')
        if not (metadata['label'] == f"screen:{image['image_id']}" and stored['image_id'] == image['image_id']):
            raise ValueError('saved-evidence check failed')
        logits = [Decimal.from_float(v[0]) for v in struct.iter_unpack('<f', raw)]
        top = max(range(1000), key=logits.__getitem__)
        maximum = logits[top]
        weights = [(v - maximum).exp() for v in logits]
        dog = sum(weights[151:269], Decimal(0))
        cat = sum(weights[281:286], Decimal(0))
        total = sum(weights, Decimal(0))
        off = total - dog - cat
        independent = {'margin': dog.ln() - cat.ln(), 'dog_mass': dog / total, 'cat_mass': cat / total, 'off_mass': off / total}
        for key, value in independent.items():
            error = abs(Decimal.from_float(stored[key]) - value)
            maxima[key] = max(maxima[key], error)
            if not error < Decimal('1e-12'):
                raise ValueError((n, key, error))
        top_matches = 281 <= top <= 285 if image['category_id'] == 17 else 151 <= top <= 268
        mass = independent['cat_mass'] if image['category_id'] == 17 else independent['dog_mass']
        reasons = ([] if top_matches else ['top1-outside-annotated-group']) + ([] if mass >= Decimal('0.2') else ['group-mass-below-0.2'])
        if not (stored['top1_id'] == top and stored['recognized'] == (not reasons) and (stored['reasons'] == reasons)):
            raise ValueError('saved-evidence check failed')
        rows.append({'image_id': image['image_id'], 'recognized': not reasons, 'margin': str(independent['margin']), 'correct_mass': str(mass)})
    lookup = {r['image_id']: r for r in rows}
    for pair in result['pairs']:
        cat = Decimal(lookup[pair['cat']['image_id']]['margin'])
        dog = Decimal(lookup[pair['dog']['image_id']]['margin'])
        if not (cat < 0 and dog > 0 and (dog - cat >= 1)):
            raise ValueError('saved-evidence check failed')
        if not all((lookup[pair[role]['image_id']]['recognized'] for role in ('cat', 'dog', 'cat_donor', 'dog_donor'))):
            raise ValueError('saved-evidence check failed')
out = {'schema': 'independent-v4-screen-decimal-audit-v1', 'scope': 'Closed saved float32 scores at 60 decimal digits; not source, image, model, execution or chronology attestation.', 'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(), 'screen_result_sha256': hashlib.sha256(result_raw).hexdigest(), 'screened': 64, 'selected': len(result['pairs']), 'max_absolute_error': {k: str(v) for k, v in maxima.items()}, 'rows': rows}
payload = (json.dumps(out, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
path = args.output
with path.open('xb') as stream:
    if not stream.write(payload) == len(payload):
        raise ValueError('saved-evidence check failed')
print(json.dumps({'screened': 64, 'selected': len(result['pairs']), 'max_absolute_error': out['max_absolute_error'], 'sha256': hashlib.sha256(payload).hexdigest()}))
