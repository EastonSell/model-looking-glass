"""Make compact tables from a closed run and its independent saved-byte replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cases_raw, result_raw, replay_raw = (p.read_bytes() for p in (args.cases, args.result, args.replay))
    cases, result, replay = json.loads(cases_raw)['cases'], json.loads(result_raw), json.loads(replay_raw)
    if (replay['status'] != 'complete_saved_byte_pass' or replay['result_sha256'] != sha(result_raw)
            or replay['cases_sha256'] != sha(cases_raw) or result['status'] != 'complete'
            or len(cases) != 32 or len(replay['families']) != 8):
        raise ValueError('result, cases and replay do not bind')
    by_id = {case['case_id']: case for case in cases}
    clean = {row['case_id']: row for row in result['records'] if row['action'] == 'clean'}
    assert len(by_id) == len(clean) == 32
    families = {f['group']: f for f in replay['families']}
    prompts = []
    for row in replay['actions']:
        case = by_id[row['case_id']]
        family = families[case['group']]
        scheme, head_text = row['action'].split('_')
        head = head_text[1:]
        variant = case['variant']
        plan = (family['output_plan']['plans'][variant][head] if scheme == 'out'
                else family['input_plans'][variant]['heads'][head])
        reference, observed = clean[case['case_id']]['score'], row['score']
        comp, dose = row['comparison'], row['dose']
        prompts.append({'group': case['group'], 'case_id': case['case_id'], 'variant': variant,
            'prompt': case['text'], 'target_id': case['target_id'], 'scheme': scheme, 'head': head,
            'family_q_output': family['output_plan']['q_out'], 'prompt_q_input': family['input_plans'][variant]['q'],
            'fraction': plan['fraction'], 'input_removed_l2': dose['input_removed_l2'],
            'predicted_output_l2': dose['predicted_output_l2'], 'actual_output_l2': dose['actual_output_l2'],
            'predicted_vector_residual_l2': dose['predicted_vector_residual_l2'],
            'clean_target_log_probability': reference['target_log_probability'],
            'clean_target_rank': reference['target_rank'], 'clean_top_unique_id': reference['top_unique_id'],
            'clean_top_representative_id': reference['top_representative_id'], 'clean_top_tie_count': reference['top_tie_count'],
            'arm_target_log_probability': observed['target_log_probability'], 'arm_target_rank': observed['target_rank'],
            'arm_top_unique_id': observed['top_unique_id'], 'arm_top_representative_id': observed['top_representative_id'],
            'arm_top_tie_count': observed['top_tie_count'], 'target_log_probability_delta': comp['target_log_probability_delta'],
            'target_logit_delta': comp['target_logit_delta'], 'kl_clean_to_arm': comp['kl_clean_to_arm'],
            'total_variation': comp['total_variation'], 'top_change': comp['top_change']})
    assert len(prompts) == sum(row['action'].startswith(('in_', 'out_')) for row in result['records'])
    rows = []
    for family in replay['families']:
        for scheme in ('output', 'input'):
            summary = family['summaries'][scheme]
            ds = summary.get('D', {})
            rows.append({'group': family['group'], 'scheme': scheme, 'status': summary['status'],
                'q_output': family['output_plan']['q_out'], 'output_plan_status': family['output_plan']['status'],
                'output_actual_status': family['output_assessment']['status'],
                'output_actual_reason': family['output_assessment']['reason'],
                'output_actual_range': family['output_assessment'].get('actual_range'),
                'clean_repeat_minus_shuffle': family['clean_repeat_minus_shuffle'],
                'D_h2': ds.get('h2'), 'D_h10': ds.get('h10'), 'D_h7': ds.get('h7'),
                'h10_minus_h2': summary.get('contrast'), 'h10_minus_h7': summary.get('h10_minus_h7'),
                'h2_minus_h7': summary.get('h2_minus_h7')})
    args.output_dir.mkdir()
    for filename, data in (('prompt-metrics.csv', prompts), ('family-metrics.csv', rows)):
        if not data:
            raise ValueError('no data rows')
        with (args.output_dir / filename).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({'prompt_rows': len(prompts), 'family_rows': len(rows),
        'table_sha256': {name: sha((args.output_dir / name).read_bytes())
                         for name in ('prompt-metrics.csv', 'family-metrics.csv')}}))


if __name__ == '__main__':
    main()
