"""Derive a compact selected-logit/TV projection while replaying the full journal.

The output is operator-derived. By itself it cannot authenticate the original
journal or recompute full-vocabulary TV, activation landing, dose, or execution.
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
SCALAR = HERE / 'review_bert_donor_context_scores.py'
SCALAR_SHA = '6670787ce5369278a159d71a7efb1c5eef63f7c8650acda16c4343fa294fdd49'
if hashlib.sha256(SCALAR.read_bytes()).hexdigest() != SCALAR_SHA:
    raise ValueError('full score reader changed')
spec = importlib.util.spec_from_file_location('fixed_context_score_reader', SCALAR)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

FIELDS = {
    'clean': ('kind','pair_id','context','cue','input_ids'),
    'repeat': ('kind','pair_id','context','cue','input_ids'),
    'self': ('kind','pair_id','site','target_context','target_cue','input_ids'),
    'transfer': ('kind','pair_id','site','target_context','target_cue','donor_context','donor_cue',
                 'input_ids','requested_radius'),
    'baseline': ('kind','pair_id','eligible','contexts'),
    'radius': ('kind','pair_id','distances','minimum_distance','q','binding_edges','dose_eligible'),
    'dose_ineligible': ('kind','pair_id','q'),
}
METRICS = ('full_tv','excluded_tv','clean_pair_probability','patched_pair_probability')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('journal', type=Path)
    p.add_argument('result', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--expected-manifest-sha256', required=True)
    p.add_argument('--expected-journal-sha256', required=True)
    p.add_argument('--expected-result-sha256', required=True)
    p.add_argument('--started', type=Path)
    p.add_argument('--expected-started-sha256')
    p.add_argument('--expected-source')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--vocab', type=int, default=30522)
    a = p.parse_args()
    paths = (a.manifest, a.journal, a.result)
    before = tuple(reader.sha(path) for path in paths)
    if before != (a.expected_manifest_sha256, a.expected_journal_sha256, a.expected_result_sha256):
        raise ValueError('input hashes differ')
    manifest, result = reader.strict(a.manifest.read_bytes()), reader.strict(a.result.read_bytes())
    if not a.synthetic:
        if (a.vocab != 30522 or before[0] != reader.MANIFEST_SHA or
                not re.fullmatch('[0-9a-f]{40}', a.expected_source or '') or not a.started or
                not re.fullmatch('[0-9a-f]{64}', a.expected_started_sha256 or '') or
                reader.sha(a.started) != a.expected_started_sha256 or
                result.get('source_commit') != a.expected_source or
                result.get('events_sha256') != before[1] or
                result.get('source_sha256',{}).get(reader.MANIFEST_PATH) != before[0]):
            raise ValueError('source or closed-input binding differs')
        started = reader.strict(a.started.read_bytes())
        if any(not reader.same(started.get(k), result.get(k)) for k in
               ('source_commit','source_sha256','assets','versions','seed','cpu_environment','tokenizer')):
            raise ValueError('started/result metadata differs')
    lookup = {}
    for pair in result['pairs']:
        for effect in pair['effects']:
            key = (pair['id'],effect['site'],effect['target_context'],effect['target_cue'],
                   effect['donor_context'],effect['donor_cue'])
            if key in lookup:
                raise ValueError('duplicate transfer identity')
            lookup[key] = effect
    any_id, some_id = manifest['scored_tokens']['any'], manifest['scored_tokens']['some']
    header = {'schema':'bert-cue-donor-context-projection-v1', 'manifest_sha256':before[0],
              'original_journal_sha256':before[1], 'result_sha256':before[2],
              'started_sha256':a.expected_started_sha256 if not a.synthetic else None,
              'source_commit':a.expected_source if not a.synthetic else None,
              'synthetic':a.synthetic, 'full_score_reader_sha256':SCALAR_SHA,
              'scope':'operator-derived selected-logit pairs and full-journal-audited reported metrics; '
                      'not original-journal authentication, full-vector reconstruction, landing, dose or execution'}
    count = 0
    with gzip.open(a.journal, 'rb') as source, a.output.open('xb') as file:
        with gzip.GzipFile(fileobj=file, mode='wb', filename='', mtime=0) as target:
            def emit(obj):
                target.write((json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
            emit(header)
            class ProjectingStream:
                def readline(self, limit=-1):
                    nonlocal count
                    line = source.readline(limit)
                    if not line:
                        return line
                    event = reader.strict(line)
                    kind = event['kind']
                    if kind not in FIELDS:
                        raise ValueError('unexpected event kind')
                    reduced = {key:event[key] for key in FIELDS[kind]}
                    if kind in ('clean','self','transfer','repeat'):
                        raw, _ = reader.logits(event, a.vocab)
                        pair = raw[4*any_id:4*any_id+4] + raw[4*some_id:4*some_id+4]
                        if len(pair) != 8 or any(not math.isfinite(v) for v in struct.unpack('<ff',pair)):
                            raise ValueError('invalid projected score pair')
                        reduced['mask_logits_pair_f32le'] = base64.b64encode(pair).decode('ascii')
                    if kind == 'transfer':
                        key = tuple(event[k] for k in ('pair_id','site','target_context','target_cue','donor_context','donor_cue'))
                        effect = lookup[key]
                        reduced.update({key:effect[key] for key in METRICS})
                    emit(reduced)
                    count += 1
                    return line
            report = reader.replay(manifest, result, ProjectingStream(), a.vocab)
    if (tuple(reader.sha(path) for path in paths) != before or
            (not a.synthetic and reader.sha(a.started) != a.expected_started_sha256) or
            hashlib.sha256(SCALAR.read_bytes()).hexdigest() != SCALAR_SHA):
        raise ValueError('source or input changed during projection')
    print(json.dumps({'status':'passed','events':count,'forwards':report['forwards'],
                      'manifest_sha256':before[0],'original_journal_sha256':before[1],
                      'result_sha256':before[2],'subset_sha256':reader.sha(a.output),
                      'subset_bytes':a.output.stat().st_size,'synthetic':a.synthetic,
                      'scope':header['scope']},sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()
