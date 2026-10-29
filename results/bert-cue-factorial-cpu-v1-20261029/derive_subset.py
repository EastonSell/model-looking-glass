"""Stream an operator-derived target-logit subset from the closed journal.

The subset cannot authenticate the original journal, establish execution, or
reconstruct full-vocabulary controls, dose or activation witnesses.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

FIELDS = {'clean': ('kind', 'case_id', 'cue', 'input_ids'),
          'repeat': ('kind', 'case_id', 'cue', 'input_ids'),
          'self': ('kind', 'case_id', 'site', 'target', 'input_ids'),
          'transfer': ('kind', 'case_id', 'site', 'donor', 'target', 'input_ids'),
          'baseline': ('kind', 'case_id', 'eligible', 'gaps'),
          'radius': ('kind', 'case_id', 'radius', 'minimum_distance', 'dose_eligible'),
          'dose_ineligible': ('kind', 'case_id', 'radius', 'minimum_distance')}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('journal', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--expected-manifest-sha256', required=True)
    p.add_argument('--expected-journal-sha256', required=True)
    a = p.parse_args()
    if sha(a.manifest) != a.expected_manifest_sha256 or sha(a.journal) != a.expected_journal_sha256:
        raise ValueError('source bytes differ from supplied hashes')
    manifest = json.loads(a.manifest.read_bytes())
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    if type(any_id) is not int or type(some_id) is not int or any_id == some_id or min(any_id, some_id) < 0:
        raise ValueError('invalid scored IDs')
    count = 0
    with a.journal.open('rb') as inp, gzip.GzipFile(fileobj=inp, mode='rb') as source:
        with a.output.open('xb') as out, gzip.GzipFile(fileobj=out, mode='wb', filename='', mtime=0) as target:
            for line in source:
                if len(line) > 2 * 1024 * 1024:
                    raise ValueError('oversized event')
                event = json.loads(line); kind = event['kind']
                if kind not in FIELDS:
                    raise ValueError('unexpected event')
                reduced = {key: event[key] for key in FIELDS[kind]}
                if kind in ('clean', 'repeat', 'self', 'transfer'):
                    raw = base64.b64decode(event['mask_logits_f32le'], validate=True)
                    if len(raw) != 30522 * 4:
                        raise ValueError('malformed original full-vocabulary logits')
                    pair = raw[4 * any_id:4 * any_id + 4] + raw[4 * some_id:4 * some_id + 4]
                    if any(not math.isfinite(x) for x in struct.unpack('<ff', pair)):
                        raise ValueError('nonfinite target logits')
                    reduced['mask_logits_pair_f32le'] = base64.b64encode(pair).decode('ascii')
                target.write((json.dumps(reduced, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
                count += 1
    if sha(a.manifest) != a.expected_manifest_sha256 or sha(a.journal) != a.expected_journal_sha256:
        raise ValueError('source changed during extraction')
    print(json.dumps({'status': 'passed', 'events': count, 'manifest_sha256': a.expected_manifest_sha256,
                      'original_journal_sha256': a.expected_journal_sha256, 'subset_sha256': sha(a.output),
                      'subset_bytes': a.output.stat().st_size,
                      'scope': 'operator-derived target-logit subset; not original-journal authentication, '
                               'full-vocabulary controls, dose, landing or execution'}, sort_keys=True))


if __name__ == '__main__':
    main()
