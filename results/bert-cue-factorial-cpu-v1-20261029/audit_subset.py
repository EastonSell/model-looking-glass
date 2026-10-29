"""Replay an operator-derived cue-factorial target-logit subset without a model.

This verifies selected logits and decision arithmetic only. The subset alone
does not authenticate its source, full logits, dose, landing or execution.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
SCALAR = HERE / 'review_bert_cue_factorial_saved_scores.py'
SCALAR_SHA = '148eca2a3f910b71d8dd65b6dae4f5ae63c46a8ca8a5c90b5c26aca82ee104e5'
if hashlib.sha256(SCALAR.read_bytes()).hexdigest() != SCALAR_SHA:
    raise ValueError('scalar replay dependency changed')
spec = importlib.util.spec_from_file_location('fixed_factorial_scalar_reader', SCALAR)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def pair_margin(event, _any, _some, _vocab):
    pair = base64.b64decode(event['mask_logits_pair_f32le'], validate=True)
    if len(pair) != 8:
        raise ValueError('invalid target-logit pair')
    a, b = struct.unpack('<ff', pair)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError('nonfinite target logit')
    return math.fsum((a, -b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('subset', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('--expected-source', required=True)
    p.add_argument('--expected-result-sha256', required=True)
    p.add_argument('--expected-subset-sha256', required=True)
    p.add_argument('--expected-original-journal-sha256', required=True)
    a = p.parse_args()
    identity = [a.expected_source, a.expected_result_sha256, a.expected_subset_sha256,
                a.expected_original_journal_sha256]
    if (not re.fullmatch('[0-9a-f]{40}', identity[0]) or
            any(not re.fullmatch('[0-9a-f]{64}', x) for x in identity[1:])):
        raise ValueError('invalid external identity pins')
    before = (reader.sha(a.manifest), reader.sha(a.subset), reader.sha(a.result))
    if before != (reader.MANIFEST_SHA, a.expected_subset_sha256, a.expected_result_sha256):
        raise ValueError('manifest, subset or result differs from supplied identity')
    manifest, result = json.loads(a.manifest.read_bytes()), json.loads(a.result.read_bytes())
    if (result.get('source_commit') != a.expected_source or
            result.get('source_sha256', {}).get(reader.MANIFEST_PATH) != reader.MANIFEST_SHA or
            result.get('events_sha256') != a.expected_original_journal_sha256):
        raise ValueError('source or original-journal identity differs')
    reader.margin = pair_margin
    with gzip.open(a.subset, 'rb') as stream:
        report = reader.replay(manifest, result, stream, synthetic=False)
    if (reader.sha(a.manifest), reader.sha(a.subset), reader.sha(a.result)) != before:
        raise ValueError('subset replay inputs changed')
    report.update({'manifest_sha256': before[0], 'subset_sha256': before[1], 'result_sha256': before[2],
                   'original_journal_sha256': a.expected_original_journal_sha256,
                   'source_commit': a.expected_source, 'scalar_reader_sha256': SCALAR_SHA,
                   'scope': 'operator-derived target-logit subset and decisions; not original-journal '
                            'authentication, full logits, dose, landing or execution'})
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
