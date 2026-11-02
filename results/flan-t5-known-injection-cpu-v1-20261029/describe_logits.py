"""Describe two-position teacher-forced logits in a closed, audited FLAN study.

This is a model-free descriptive calculation, not generation. It relies on the
separate saved-byte audit and does not attest execution, chronology, source or
asset identity, process cleanup, or naturally occurring localization.
Inputs must remain quiescent while reading.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

AUDIT_SHA = '3e22a3f20c8a79ea0a91bc24dd4aa81e57a8c0fc1d7d818c507458ea3155c428'
VOCAB = 32128


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, cap):
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, 'oversized saved file')
    return raw


def values(raw):
    require(len(raw) % 4 == 0, 'invalid float32 length')
    out = []
    for (word,) in struct.iter_unpack('<I', raw):
        exp, mantissa = (word >> 23) & 255, word & 0x7fffff
        require(exp != 255, 'nonfinite logits')
        value = math.ldexp(mantissa if exp == 0 else (1 << 23) + mantissa,
                           -149 if exp == 0 else exp - 150)
        out.append(-value if word >> 31 else value)
    return out


def compare(a, b):
    require(len(a) == len(b) == VOCAB, 'logit dimension differs')
    ma, mb = max(a), max(b)
    wa, wb = [math.exp(x - ma) for x in a], [math.exp(x - mb) for x in b]
    za, zb = math.fsum(wa), math.fsum(wb)
    pa, pb = [x / za for x in wa], [x / zb for x in wb]
    tv = 0.5 * math.fsum(abs(x - y) for x, y in zip(pa, pb))
    kl = math.fsum(p * ((x - y) - (ma - mb) - (math.log(za) - math.log(zb)))
                   for p, x, y in zip(pa, a, b))
    require(math.isfinite(tv) and math.isfinite(kl) and -1e-12 <= kl and 0 <= tv <= 1,
            'invalid probability distance')
    i, j = a.index(ma), b.index(mb)
    return {'top1_clean_id': i, 'top1_changed_id': j, 'top1_changed': i != j,
            'tv': tv, 'kl_clean_to_changed': max(0.0, kl)}


def audit(data, audit_path):
    audit_raw = read(audit_path, 1000000)
    require(sha(audit_raw) == AUDIT_SHA, 'independent audit differs')
    checked = json.loads(audit_raw)['result']
    journal = read(data / 'attempt.jsonl', 16 * 1024 * 1024)
    result = read(data / 'result.json', 1000000)
    require(sha(journal) == checked['journal_sha256'] and sha(result) == checked['result_sha256']
            and checked['forwards'] == 80, 'closed journal or result differs from audit')
    events = [json.loads(line) for line in journal.splitlines()]
    references = {}
    for event in events:
        if event.get('event') == 'forward_observation':
            key = (event['case_id'], event.get('stack'), event['arm'])
            require(key not in references, 'duplicate forward')
            ref = event['refs']['logits']
            require(ref['shape'] == [1, 2, VOCAB] and ref['bytes'] == 8 * VOCAB
                    and ref['path'] == 'tensors/' + ref['sha256'] + '.f32le', 'invalid logit reference')
            references[key] = ref
    cases = list(dict.fromkeys(t['case_id'] for t in checked['trials']))
    require(len(cases) == 8, 'case count differs')
    cache = {}
    def logits(key):
        ref = references[key]
        if ref['sha256'] not in cache:
            raw = read(data / ref['path'], 8 * VOCAB)
            require(len(raw) == 8 * VOCAB and sha(raw) == ref['sha256'], 'logit bytes differ')
            cache[ref['sha256']] = values(raw)
        x = cache[ref['sha256']]
        return (x[:VOCAB], x[VOCAB:])
    rows = []
    splits = {t['case_id']: t['split'] for t in checked['trials']}
    for case in cases:
        clean = logits((case, None, 'clean'))
        for stack in ('encoder', 'decoder'):
            for arm in ('inject', 'alternate'):
                changed = logits((case, stack, arm))
                for position in (0, 1):
                    rows.append({'case_id': case, 'split': splits[case], 'stack': stack,
                                 'arm': arm, 'decoder_position': position,
                                 **compare(clean[position], changed[position])})
    groups = {}
    for stack in ('encoder', 'decoder'):
        for arm in ('inject', 'alternate'):
            for position in (0, 1):
                group = [r for r in rows if r['stack'] == stack and r['arm'] == arm
                         and r['decoder_position'] == position]
                groups[f'{stack}/{arm}/position-{position}'] = {
                    'cases': len(group), 'top1_changed': sum(r['top1_changed'] for r in group),
                    'tv_min': min(r['tv'] for r in group), 'tv_max': max(r['tv'] for r in group),
                    'kl_min': min(r['kl_clean_to_changed'] for r in group),
                    'kl_max': max(r['kl_clean_to_changed'] for r in group)}
    return {'independent_audit_sha256': AUDIT_SHA, 'journal_sha256': sha(journal),
            'result_sha256': sha(result), 'conditions': rows, 'groups': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = {'schema': 'flan-known-injection-logit-descriptive-v1',
              'scope': 'Post-run descriptive full-vocabulary teacher-forced probabilities; numeric last bits '
                       'can vary by platform. No generation, execution or source attestation.',
              'result': audit(args.data, args.audit)}
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short descriptive report write')
    print(json.dumps({'report_sha256': sha(raw), 'groups': report['result']['groups']}))


if __name__ == '__main__':
    main()
