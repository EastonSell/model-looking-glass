"""Project a small target-score subset from the stopped public cue-dose run.

This post-interruption projection cannot authenticate its derivation, full
logits, dose, landing witnesses, execution, or unrecorded returned forwards.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

SOURCE = 'b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4'
MANIFEST_SHA = '6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a'
JOURNAL_SHA = 'ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05'
START_SHA = '19baa22fc6b6fe3da139bf22b125a4f3122a39e6e82dc9ec38604389127bd9e7'
FAILURE_SHA = '12b861407be2e4050b7f5e7b792847867f7b7c588cf75857783fc36d1bb8317a'
EXIT_SHA = 'f5499d680cbd2ea1d2ebdfdb7268ce30598ad8aa7d6af9ad8ef0ff806ed54f34'
FIELDS = {
    'clean': ('kind', 'case_id', 'cue', 'input_ids'),
    'repeat': ('kind', 'case_id', 'cue', 'input_ids'),
    'self': ('kind', 'case_id', 'site', 'target', 'input_ids'),
    'transfer': ('kind', 'case_id', 'site', 'dose', 'donor', 'target', 'input_ids', 'requested_radius'),
    'baseline': ('kind', 'case_id', 'eligible', 'gaps'),
    'radius': ('kind', 'case_id', 'minimum_distance', 'distances', 'binding_edges', 'doses'),
    'dose_ineligible': ('kind', 'case_id', 'dose', 'radius', 'minimum_distance'),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('journal', type=Path)
    parser.add_argument('started', type=Path)
    parser.add_argument('failure', type=Path)
    parser.add_argument('exit_receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    paths = (args.manifest, args.journal, args.started, args.failure, args.exit_receipt)
    expected = (MANIFEST_SHA, JOURNAL_SHA, START_SHA, FAILURE_SHA, EXIT_SHA)
    if tuple(sha(p) for p in paths) != expected:
        raise ValueError('one or more closed input hashes differ')
    manifest = json.loads(args.manifest.read_bytes())
    started = json.loads(args.started.read_bytes())
    failure = json.loads(args.failure.read_bytes())
    exit_receipt = json.loads(args.exit_receipt.read_bytes())
    if (started.get('source_commit') != SOURCE or failure.get('source_commit') != SOURCE
            or exit_receipt.get('source_commit') != SOURCE or exit_receipt.get('exit_code') != 1
            or exit_receipt.get('process_waited') is not True or failure.get('status') != 'incomplete'
            or failure.get('phase') != 'forwards' or failure.get('error_type') != 'RuntimeError'
            or failure.get('message') != 'disk reserve exhausted before the next evidence event'
            or manifest.get('scored_tokens') != {'any': 2151, 'some': 2070}):
        raise ValueError('source, failure or scored-token identity differs')
    header = {'kind': 'prefix_header', 'schema': 'bert-cue-dose-partial-score-subset-v1',
              'source_commit': SOURCE, 'manifest_sha256': MANIFEST_SHA, 'original_journal_sha256': JOURNAL_SHA,
              'started_sha256': START_SHA, 'failure_sha256': FAILURE_SHA, 'exit_receipt_sha256': EXIT_SHA,
              'incomplete': True}
    count = forwards = 0
    with gzip.open(args.journal, 'rb') as journal, args.output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as out:
            out.write((json.dumps(header, sort_keys=True, separators=(',', ':')) + '\n').encode())
            for line in journal:
                if len(line) > 2 * 1024 * 1024 or not line.endswith(b'\n'):
                    raise ValueError('incomplete or oversized input event')
                event = json.loads(line)
                kind = event['kind']
                if kind not in FIELDS:
                    raise ValueError('unexpected event in prefix')
                reduced = {field: event[field] for field in FIELDS[kind]}
                if kind in ('clean', 'self', 'transfer', 'repeat'):
                    logits = base64.b64decode(event['mask_logits_f32le'], validate=True)
                    if len(logits) != 30522 * 4:
                        raise ValueError('malformed mask logits')
                    pair = logits[4 * 2151:4 * 2152] + logits[4 * 2070:4 * 2071]
                    if any(not math.isfinite(v) for v in struct.unpack('<ff', pair)):
                        raise ValueError('nonfinite selected logit')
                    reduced['mask_logits_pair_f32le'] = base64.b64encode(pair).decode('ascii')
                    forwards += 1
                out.write((json.dumps(reduced, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
                count += 1
    if (count, forwards) != (509, 499) or tuple(sha(p) for p in paths) != expected:
        raise ValueError('closed prefix count or input identity changed')
    print(json.dumps({'status': 'derived', 'events': count, 'saved_forwards': forwards,
                      'source_commit': SOURCE, 'original_journal_sha256': JOURNAL_SHA,
                      'subset_sha256': sha(args.output), 'subset_bytes': args.output.stat().st_size,
                      'scope': 'post-interruption operator projection; not proof of original journal, '
                               'full logits, dose, landing or execution'}, sort_keys=True))


if __name__ == '__main__':
    main()
