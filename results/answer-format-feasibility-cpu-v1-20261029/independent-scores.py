"""Independent saved-logit scorer for the answer-format feasibility study.

No model is loaded. Raw phase matrices must be externally verified as the
published, shuffled-execution-order float32 records; this script does not
authenticate their source, forward, cleanup, chronology or custody.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

FACTS = ('france', 'japan', 'italy', 'uk')
FAMILIES = ('statement', 'question')
PREFIXES = ('none', 'neutral', 'spain', 'sweden')
LABELS = ('factual', 'counterfactual', 'madrid', 'stockholm')
SEEDS = {'development': 20261031, 'heldout': 20261101}
CASE_SHA = 'cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unpack(bits):
    bits = np.asarray(bits, dtype=np.uint32)
    e = (bits >> 23) & 255
    if np.any(e == 255):
        raise ValueError('nonfinite saved logit')
    m = np.where(e == 0, bits & 0x7fffff, (bits & 0x7fffff) | 0x800000)
    n = np.ldexp(m.astype(np.float64), np.where(e == 0, -149, e.astype(np.int32) - 150))
    return np.where((bits >> 31) != 0, -n, n)


def phase(cases, stage, raw_bits, tokenizer=None):
    """Score raw uint32 bit rows in frozen shuffled execution order."""
    if stage not in SEEDS or len(cases) != 32:
        raise ValueError('expected 32 cases for a named phase')
    bits = np.asarray(raw_bits, dtype=np.uint32)
    if bits.ndim != 2 or bits.shape[0] != 32 or bits.shape[1] < 4:
        raise ValueError('invalid phase matrix shape')
    order = list(range(32))
    random.Random(SEEDS[stage]).shuffle(order)
    expected = list(itertools.product(FACTS, FAMILIES, PREFIXES))
    for row, (fact, family, prefix) in zip(cases, expected):
        if ((row['role'], row['fact_id'], row['family'], row['prefix'])
                != (stage, fact, family, prefix)):
            raise ValueError('canonical case order differs')
        if row['case_id'] != f'answer-format-{stage}-{fact}-{family}-{prefix}':
            raise ValueError('case identity differs')
        token_ids = [row['labels'][label]['token_id'] for label in LABELS]
        if len(set(token_ids)) != 4 or any(type(x) is not int or not 0 <= x < bits.shape[1] for x in token_ids):
            raise ValueError('case token labels differ')
    observations, by_group, logs, normalizers = [], {}, {}, {}
    for execution_index, canonical_index in enumerate(order):
        case = cases[canonical_index]
        x = unpack(bits[execution_index])
        peak = float(np.max(x))
        ties = np.flatnonzero(x == peak)
        representative = int(ties[0])
        unique = representative if len(ties) == 1 else None
        runner_up = float(np.partition(x, -2)[-2])
        shifted = x - peak
        with np.errstate(under='ignore'):
            partition = math.fsum(float(p) for p in np.exp(shifted))
        if not math.isfinite(partition) or partition < 1:
            raise ValueError('invalid normalization')
        lp = shifted - math.log(partition)
        scores = {}
        for label in LABELS:
            token = case['labels'][label]['token_id']
            scores[label] = {'token_id': token, 'logit': float(x[token]), 'logp': float(lp[token])}
        record = {
            'execution_index': execution_index, 'canonical_index': canonical_index,
            'case_id': case['case_id'], 'fact_id': case['fact_id'],
            'family': case['family'], 'prefix': case['prefix'],
            'unique_top_id': unique, 'top_tie_count': len(ties),
            'top_representative_id': representative, 'top_gap': peak - runner_up,
            'representative_decoded_token': None if tokenizer is None else tokenizer.decode([representative], skip_special_tokens=False),
            'representative_token_string': None if tokenizer is None else tokenizer.id_to_token(representative),
            'factual_top': unique == scores['factual']['token_id'],
            'scores': scores, 'row_sha256': sha(np.asarray(bits[execution_index], dtype='<u4').tobytes()),
        }
        observations.append(record)
        key = (case['fact_id'], case['family'], case['prefix'])
        by_group[key], logs[key] = record, lp
        normalizers[key] = (peak, math.log(partition))
    families = []
    for family in FAMILIES:
        robust = [f for f in FACTS if all(by_group[(f, family, p)]['factual_top'] for p in PREFIXES)]
        families.append({'family': family, 'robust_facts': robust, 'eligible': len(robust) >= 3})
    pairs, contrasts = [], []
    for fact, family in itertools.product(FACTS, FAMILIES):
        for ref, cmp in itertools.combinations(PREFIXES, 2):
            a, b = by_group[(fact, family, ref)], by_group[(fact, family, cmp)]
            la, lb = logs[(fact, family, ref)], logs[(fact, family, cmp)]
            a_peak, a_logz = normalizers[(fact, family, ref)]
            b_peak, b_logz = normalizers[(fact, family, cmp)]
            with np.errstate(under='ignore'):
                pa, pb = np.exp(la), np.exp(lb)
            kl = math.fsum(float(z) for z in pa * (la - lb))
            tv = 0.5 * math.fsum(float(z) for z in np.abs(pa - pb))
            if not math.isfinite(kl) or not math.isfinite(tv) or kl < -1e-12 or tv > 1 + 1e-12:
                raise ValueError('invalid distribution contrast')
            relation = ('tied' if a['unique_top_id'] is None or b['unique_top_id'] is None else
                        'same' if a['unique_top_id'] == b['unique_top_id'] else 'changed')
            pairs.append({'fact_id': fact, 'family': family, 'reference': ref, 'comparison': cmp,
                          'reference_case_id': a['case_id'], 'comparison_case_id': b['case_id'],
                          'top_relation': relation, 'kl': max(0., kl), 'total_variation': min(1., tv),
                          'logp_delta': {k: math.fsum((b['scores'][k]['logit'], -b_peak, -b_logz,
                                                       -a['scores'][k]['logit'], a_peak, a_logz)) for k in LABELS}})
        s, w = by_group[(fact, family, 'spain')], by_group[(fact, family, 'sweden')]
        # The normalizers cancel in D. Summing f32-derived logits directly
        # retains small contrasts that rounded individual logps can lose.
        d = math.fsum((s['scores']['madrid']['logit'], -w['scores']['madrid']['logit'],
                       -s['scores']['stockholm']['logit'], w['scores']['stockholm']['logit']))
        contrasts.append({'fact_id': fact, 'family': family, 'd': d,
                          'spain_top_madrid': s['unique_top_id'] == s['scores']['madrid']['token_id'],
                          'sweden_top_stockholm': w['unique_top_id'] == w['scores']['stockholm']['token_id']})
    return {'stage': stage, 'execution_order': order, 'observations': observations,
            'families': families, 'pairs': pairs, 'contrasts': contrasts}


def decide(development, heldout=None):
    chosen = next((r for r in development['families'] if r['eligible']), None)
    development['selected_family'] = None if chosen is None else chosen['family']
    development['status'] = 'ineligible' if chosen is None else 'eligible'
    if heldout is None:
        return
    heldout['selected_family'] = None if chosen is None else chosen['family']
    h = [] if chosen is None else next(r['robust_facts'] for r in heldout['families'] if r['family'] == chosen['family'])
    reproduced = [] if chosen is None else [f for f in chosen['robust_facts'] if f in h]
    heldout['reproduced_development_facts'] = reproduced
    heldout['status'] = 'ineligible' if chosen is None else ('positive' if len(reproduced) >= 3 else 'negative')
    means = []
    by_key = {(row['fact_id'], row['family'], row['prefix']): row for row in heldout['observations']}
    for fact in FACTS:
        values = [r['d'] for r in heldout['contrasts'] if r['fact_id'] == fact]
        if len(values) != 2:
            raise ValueError('held-out fact lacks a family')
        terms = []
        for family in FAMILIES:
            s, w = by_key[(fact, family, 'spain')], by_key[(fact, family, 'sweden')]
            terms.extend((s['scores']['madrid']['logit'], -w['scores']['madrid']['logit'],
                          -s['scores']['stockholm']['logit'], w['scores']['stockholm']['logit']))
        # Compute the fact mean from all eight f32-derived logits at once;
        # per-template D can round away a small residual before cancellation.
        means.append({'fact_id': fact, 'mean_d': math.fsum(terms) / 2.})
    heldout['heldout_fact_means'] = means
    heldout['descriptive_direction_met'] = sum(r['mean_d'] > 0 for r in means) >= 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--development', required=True, type=Path)
    parser.add_argument('--heldout', type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    raw_cases = args.cases.read_bytes()
    if sha(raw_cases) != CASE_SHA or sha(args.tokenizer.read_bytes()) != TOKENIZER_SHA:
        raise ValueError('proposal or tokenizer bytes differ from preregistration')
    doc = json.loads(raw_cases)
    if len(doc['cases']) != 64:
        raise ValueError('expected 64 cases')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    def read(path):
        data = path.read_bytes()
        if len(data) != 32 * 50257 * 4:
            raise ValueError('wrong saved matrix size')
        return data, np.frombuffer(data, dtype='<u4').reshape(32, 50257)
    dev_bytes, dev_bits = read(args.development)
    development = phase(doc['cases'][:32], 'development', dev_bits, tok)
    heldout = None
    held_bytes = None
    if args.heldout is not None:
        held_bytes, held_bits = read(args.heldout)
        heldout = phase(doc['cases'][32:], 'heldout', held_bits, tok)
    decide(development, heldout)
    output = {
        'schema': 'independent-answer-format-saved-logits-v1',
        'scope': ('Independent numerical score and selection from supplied bytes only. '
                  'Does not authenticate matrix-to-case binding, model execution, cleanup, custody or chronology.'),
        'precision_note': ('Individual log probabilities and per-template D are rounded to float64. '
                           'Paired logp deltas and held-out means use fsum on underlying logits and '
                           'normalizer terms; subtracting or averaging displayed rounded values may differ.'),
        'auditor_sha256': sha(Path(__file__).read_bytes()), 'cases_sha256': sha(raw_cases),
        'tokenizer_sha256': TOKENIZER_SHA, 'development_logits_sha256': sha(dev_bytes),
        'heldout_logits_sha256': None if held_bytes is None else sha(held_bytes),
        'development': development, 'heldout': heldout,
    }
    raw = (json.dumps(output, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode()
    with args.report.open('xb') as out:
        out.write(raw)
    print(json.dumps({'development': development['status'],
                      'heldout': None if heldout is None else heldout['status'],
                      'report_sha256': sha(raw)}))


if __name__ == '__main__':
    main()
