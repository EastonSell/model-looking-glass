"""Independent saved-logit audit of the preregistered full-top baseline screen.

This program does not load or run a model. Its selection logic is frozen before
inspection of any screen logits; it reads exact f32 rows and a public case file.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_ROLES = ('discovery', 'efficacy-calibration', 'efficacy-calibration',
                  'evaluation', 'evaluation', 'evaluation', 'evaluation',
                  'evaluation', 'evaluation')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unpack_float32(bits):
    """Interpret f32 bit patterns directly as f64, including f32 subnormals."""
    bits = np.asarray(bits, dtype=np.uint32)
    exponent = (bits >> 23) & 255
    if np.any(exponent == 255):
        raise ValueError('nonfinite saved logit')
    fraction = bits & 0x7fffff
    significand = np.where(exponent == 0, fraction, fraction | 0x800000).astype(np.float64)
    power = np.where(exponent == 0, -149, exponent.astype(np.int32) - 150)
    values = np.ldexp(significand, power)
    return np.where((bits >> 31) != 0, -values, values)


def audit(case_document, bits, tokenizer=None):
    """Score 36 fact-major rows and apply only baseline-derived selection."""
    facts = case_document['fact_order']
    if not isinstance(facts, list) or len(facts) != 4 or len(set(facts)) != 4:
        raise ValueError('expected four distinct facts')
    cases = [case for case in case_document['cases'] if case['role'] != 'locality']
    bits = np.asarray(bits, dtype=np.uint32)
    if len(cases) != 36 or bits.ndim != 2 or bits.shape[0] != 36 or bits.shape[1] < 2:
        raise ValueError('screen must have 36 full rows')
    if len({case['case_id'] for case in cases}) != 36:
        raise ValueError('duplicate case IDs')
    observations = []
    facts_report = []
    for fact_index, fact in enumerate(facts):
        start = 9 * fact_index
        group = cases[start:start + 9]
        if ([case['fact_id'] for case in group] != [fact] * 9
                or tuple(case['role'] for case in group) != EXPECTED_ROLES):
            raise ValueError('fact-major case order or roles differ')
        pair = (group[0]['factual']['token_id'], group[0]['counterfactual']['token_id'])
        if (not all(type(i) is int and 0 <= i < bits.shape[1] for i in pair)
                or pair[0] == pair[1]):
            raise ValueError('invalid target token pair')
        if any((c['factual']['token_id'], c['counterfactual']['token_id']) != pair for c in group):
            raise ValueError('target token pair varies within fact')
        group_obs = []
        for local_offset, case in enumerate(group):
            offset = start + local_offset
            values = unpack_float32(bits[offset])
            maxval = float(np.max(values))
            tied = np.flatnonzero(values == maxval)
            representative = int(tied[0])
            unique = representative if len(tied) == 1 else None
            second = float(np.partition(values, -2)[-2])
            with np.errstate(under='ignore'):
                denominator = math.fsum(float(z) for z in np.exp(values - maxval))
            if denominator <= 0 or not math.isfinite(denominator):
                raise ValueError('invalid normalization')
            logdenom = math.log(denominator)
            eligible = unique == pair[0]
            item = {
                'offset': offset, 'case_id': case['case_id'], 'fact_id': fact,
                'role': case['role'], 'factual_id': pair[0], 'counterfactual_id': pair[1],
                'factual_logit': float(values[pair[0]]),
                'counterfactual_logit': float(values[pair[1]]),
                'factual_logp': float(values[pair[0]] - maxval) - logdenom,
                'counterfactual_logp': float(values[pair[1]] - maxval) - logdenom,
                'unique_top_id': unique, 'top_tie_count': len(tied),
                'top_representative_id': representative, 'top_gap': maxval - second,
                'representative_decoded_token': (None if tokenizer is None else
                                                 tokenizer.decode([representative], skip_special_tokens=False)),
                'representative_token_string': None if tokenizer is None else tokenizer.id_to_token(representative),
                'eligible': eligible,
                'exclusion_reason': None if eligible else ('top_tied' if unique is None else 'top_not_factual'),
                'row_sha256': sha(np.asarray(bits[offset], dtype='<u4').tobytes()),
            }
            observations.append(item)
            group_obs.append(item)
        evaluation = [item['offset'] for item in group_obs[3:] if item['eligible']]
        failures = []
        if not group_obs[0]['eligible']:
            failures.append('discovery_not_unique_factual_top')
        for i in (1, 2):
            if not group_obs[i]['eligible']:
                failures.append(f'calibration_{i:02}_not_unique_factual_top')
        if len(evaluation) < 4:
            failures.append('fewer_than_four_eligible_evaluation_rows')
        facts_report.append({
            'fact_id': fact, 'fact_index': fact_index,
            'discovery_eligible': group_obs[0]['eligible'],
            'calibration_eligible_count': sum(item['eligible'] for item in group_obs[1:3]),
            'evaluation_eligible_count': len(evaluation),
            'eligible_evaluation_offsets': evaluation,
            'qualifies': not failures, 'failed_conditions': failures,
        })
    selected = next((item for item in facts_report if item['qualifies']), None)
    if selected is None:
        decision = {'status': 'ineligible', 'fact_id': None, 'fact_index': None,
                    'discovery_offset': None, 'calibration_offsets': [], 'evaluation_offsets': []}
    else:
        start = 9 * selected['fact_index']
        decision = {'status': 'eligible', 'fact_id': selected['fact_id'],
                    'fact_index': selected['fact_index'], 'discovery_offset': start,
                    'calibration_offsets': [start + 1, start + 2],
                    'evaluation_offsets': selected['eligible_evaluation_offsets'][:4]}
    return {'decision': decision, 'facts': facts_report, 'observations': observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--logits', type=Path, required=True)
    parser.add_argument('--tokenizer-json', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--vocab-size', type=int, default=50257)
    args = parser.parse_args()
    cases_bytes = args.cases.read_bytes()
    doc = json.loads(cases_bytes)
    tokenizer_bytes = args.tokenizer_json.read_bytes()
    if sha(tokenizer_bytes) != doc['tokenizer_sha256']:
        raise ValueError('tokenizer hash differs from public cases')
    raw = args.logits.read_bytes()
    if args.vocab_size != 50257 or len(raw) != 36 * args.vocab_size * 4:
        raise ValueError('wrong saved-logit byte count or pinned vocabulary size')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    bits = np.frombuffer(raw, dtype='<u4').reshape(36, args.vocab_size)
    output = audit(doc, bits, tokenizer)
    output.update({'schema': 'full-top-independent-screen-audit-v1',
                   'scope': 'Saved-byte selector and score audit only; not proof of execution, cleanup, chronology or unknown exposure.',
                   'auditor_sha256': sha(Path(__file__).read_bytes()),
                   'cases_sha256': sha(cases_bytes), 'tokenizer_sha256': sha(tokenizer_bytes),
                   'logits_sha256': sha(raw), 'vocab_size': args.vocab_size})
    with args.report.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': output['decision']['status'], 'fact_id': output['decision']['fact_id'],
                      'report_sha256': sha(args.report.read_bytes())}))


if __name__ == '__main__':
    main()
