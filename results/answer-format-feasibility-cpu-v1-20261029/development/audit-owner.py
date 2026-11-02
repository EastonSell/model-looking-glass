"""Compare independent answer-format scores to a closed writer's saved data.

This is a model-free numerical, row-binding and decoded-token check. Process,
provenance, custody and teardown need separate validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

LABELS = ('factual', 'counterfactual', 'madrid', 'stockholm')
ROW_BYTES = 50257 * 4


def sha(data):
    return hashlib.sha256(data).hexdigest()


def same(a, b, keys, where):
    for key in keys:
        if a.get(key) != b.get(key):
            raise ValueError(f'{where}: {key} differs')


def close(a, b, name, maxima):
    if type(a) not in (int, float) or type(b) not in (int, float) or not math.isfinite(a) or not math.isfinite(b):
        raise ValueError(f'{name}: nonfinite numeric value')
    error = abs(float(a) - float(b))
    tolerance = max(1e-12, 64 * max(math.ulp(float(a)), math.ulp(float(b))))
    if error > tolerance:
        raise ValueError(f'{name}: discrepancy {error} > {tolerance}')
    maxima[name] = max(maxima.get(name, 0.), error)


def check(stage, expected, directory, case_sha, matrix_sha, previous=None):
    result_raw = (directory / 'results.json').read_bytes()
    result = json.loads(result_raw)
    if (result.get('version') != 'answer-format-private-phase-v1' or result.get('stage') != stage
            or result.get('case_sha256') != case_sha):
        raise ValueError(f'{stage}: phase identity differs')
    matrix = (directory / 'matrix.f32').read_bytes()
    if len(matrix) != 32 * ROW_BYTES or sha(matrix) != matrix_sha:
        raise ValueError(f'{stage}: matrix size or hash differs')
    for name in ('matrix.f32', 'journal.jsonl', 'state-original.json'):
        raw = (directory / name).read_bytes()
        info = result['artifacts'][name]
        if info != {'bytes': len(raw), 'sha256': sha(raw)}:
            raise ValueError(f'{stage}: artifact hash differs: {name}')
    if previous is None:
        if result.get('development_matrix_sha256') is not None or result.get('development_result_sha256') is not None:
            raise ValueError('development unexpectedly references prior data')
    elif (result.get('development_matrix_sha256') != previous['matrix_sha256']
          or result.get('development_result_sha256') != previous['result_sha256']):
        raise ValueError('heldout does not bind the development artifacts')
    journal = [json.loads(line) for line in (directory / 'journal.jsonl').read_text(encoding='utf8').splitlines()]
    token_raw = (directory / 'top-tokens.json').read_bytes()
    token_rows = json.loads(token_raw)
    score = result['score']
    same(expected, score, ('stage', 'execution_order', 'families', 'selected_family', 'status'), stage)
    if stage == 'heldout':
        same(expected, score, ('reproduced_development_facts', 'descriptive_direction_met'), stage)
    own_obs, reported_obs = expected['observations'], score['observations']
    if type(token_rows) is not list or len(token_rows) != 32 or len(journal) != 32 or len(own_obs) != 32 or len(reported_obs) != 32:
        raise ValueError(f'{stage}: row count differs')
    maxima = {}
    for i, (own, reported, entry) in enumerate(zip(own_obs, reported_obs, journal)):
        same(own, reported, ('execution_index', 'canonical_index', 'case_id', 'fact_id',
                             'family', 'prefix', 'unique_top_id', 'top_tie_count',
                             'top_representative_id', 'factual_top'), f'{stage} row {i}')
        same(own, entry, ('execution_index', 'canonical_index', 'case_id'), f'{stage} journal {i}')
        offset = i * ROW_BYTES
        digest = sha(matrix[offset:offset + ROW_BYTES])
        if (own['execution_index'] != i or entry.get('offset') != offset or entry.get('bytes') != ROW_BYTES
                or entry.get('sha256') != digest or own['row_sha256'] != digest
                or reported.get('logit_byte_offset') != offset or reported.get('logit_bytes') != ROW_BYTES
                or reported.get('logit_sha256') != digest
                or 'forward_denormal_probe_before' not in entry or 'forward_denormal_probe_after' not in entry):
            raise ValueError(f'{stage}: journal or row binding differs at {i}')
        if own['top_gap'] != reported['top_gap']:
            raise ValueError(f'{stage}: top gap differs at {i}')
        decoded = token_rows[i]
        if (type(decoded) is not dict or decoded.get('case_id') != own['case_id']
                or decoded.get('unique_top_id') != own['unique_top_id']
                or decoded.get('top_tie_count') != own['top_tie_count']
                or decoded.get('top_representative_id') != own['top_representative_id']
                or decoded.get('top_representative_text') != own['representative_decoded_token']
                or decoded.get('top_representative_raw_token') != own['representative_token_string']
                or decoded.get('unique_top_text') != (None if own['unique_top_id'] is None else own['representative_decoded_token'])
                or decoded.get('unique_top_raw_token') != (None if own['unique_top_id'] is None else own['representative_token_string'])):
            raise ValueError(f'{stage}: decoded top token differs at {i}')
        for label in LABELS:
            left, right = own['scores'][label], reported['scores'][label]
            if left['token_id'] != right['token_id'] or left['logit'] != right['logit']:
                raise ValueError(f'{stage}: {label} logit or ID differs at {i}')
            close(left['logp'], right['logp'], 'label_logp', maxima)
    if len(expected['pairs']) != 48 or len(score['pairs']) != 48:
        raise ValueError(f'{stage}: pair count differs')
    for i, (own, reported) in enumerate(zip(expected['pairs'], score['pairs'])):
        same(own, reported, ('fact_id', 'family', 'reference', 'comparison',
                             'reference_case_id', 'comparison_case_id', 'top_relation'), f'{stage} pair {i}')
        close(own['kl'], reported['kl'], 'kl', maxima)
        close(own['total_variation'], reported['total_variation'], 'tv', maxima)
        for label in LABELS:
            close(own['logp_delta'][label], reported['logp_delta'][label], 'paired_logp_delta', maxima)
    if len(expected['contrasts']) != 8 or len(score['contrasts']) != 8:
        raise ValueError(f'{stage}: contrast count differs')
    for i, (own, reported) in enumerate(zip(expected['contrasts'], score['contrasts'])):
        same(own, reported, ('fact_id', 'family', 'd', 'spain_top_madrid', 'sweden_top_stockholm'),
             f'{stage} contrast {i}')
    if stage == 'heldout':
        if expected['heldout_fact_means'] != score['heldout_fact_means']:
            raise ValueError('heldout exact D fact means differ')
    return {'matrix_sha256': sha(matrix), 'result_sha256': sha(result_raw),
            'top_tokens_sha256': sha(token_raw),
            'status': score['status'], 'max_absolute_differences': maxima,
            'journal_rows': len(journal), 'pairs': len(score['pairs']), 'contrasts': len(score['contrasts'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-report', required=True, type=Path)
    parser.add_argument('--development-dir', required=True, type=Path)
    parser.add_argument('--heldout-dir', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    raw = args.independent_report.read_bytes()
    source = json.loads(raw)
    if source.get('schema') != 'independent-answer-format-saved-logits-v1':
        raise ValueError('unknown independent audit schema')
    dev = check('development', source['development'], args.development_dir,
                source['cases_sha256'], source['development_logits_sha256'])
    held = None
    if (args.heldout_dir is None) != (source['heldout'] is None):
        raise ValueError('heldout report and directory must both be supplied or omitted')
    if args.heldout_dir is not None:
        held = check('heldout', source['heldout'], args.heldout_dir, source['cases_sha256'],
                     source['heldout_logits_sha256'], previous=dev)
    result = {'schema': 'independent-answer-format-owner-comparison-v1', 'status': 'passed',
              'scope': ('Saved raw rows, journal-to-case bindings, decoded tops, numerical scores and internal artifact hashes; '
                        'not proof of forward provenance, custody, cleanup or chronology.'),
              'auditor_sha256': sha(Path(__file__).read_bytes()), 'independent_report_sha256': sha(raw),
              'development': dev, 'heldout': held}
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.report.open('xb') as out:
        out.write(data)
    print(json.dumps({'status': 'passed', 'report_sha256': sha(data)}))


if __name__ == '__main__':
    main()
