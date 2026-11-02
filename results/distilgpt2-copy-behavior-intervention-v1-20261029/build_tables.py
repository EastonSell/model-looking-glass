"""Produce fixed case and family tables from a closed, independently replayed attempt."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--result', type=Path, required=True)
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--cases', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--allow-synthetic-fixture', action='store_true')
a = p.parse_args()
digest = lambda b: hashlib.sha256(b).hexdigest()
result_raw, audit_raw, cases_raw = (x.read_bytes() for x in (a.result, a.audit, a.cases))
result, audit, cases = map(json.loads, (result_raw, audit_raw, cases_raw))
if audit['result_sha256'] != digest(result_raw) or audit['cases_sha256'] != digest(cases_raw):
    raise ValueError('independent replay does not bind these result and case bytes')
if audit['status'] != 'complete_saved_byte_pass' or bool(audit['synthetic_fixture']) != a.allow_synthetic_fixture:
    raise ValueError('unexpected independent replay status or synthetic scope')
if not a.allow_synthetic_fixture and not audit['checkpoint_rows_verified']:
    raise ValueError('production replay did not verify checkpoint projection slices')
if result['decision'] != audit['decision']:
    raise ValueError('owner and independent aggregate decisions differ')
by_case = {c['case_id']: c for c in cases['cases']}
clean = {r['case_id']: r for r in result['records'] if r['action'] == 'clean'}
if len(clean) != 96 or len(audit['families']) != 12:
    raise ValueError('fixed held-out case or family count differs')
family_by_group = {f['group']: f for f in audit['families']}
action_by_case = {}
for item in audit['actions']:
    action_by_case.setdefault(item['case_id'], []).append(item)

a.output_dir.mkdir(parents=True, exist_ok=False)
family_fields = ['group','baseline_status','dose_status','dose_reason','q_out','minimum_variant_heads',
                 'actual_range','D_h2','D_h10','D_h7','contrast','contrast_sign','h2_minus_h7','h10_minus_h7',
                 'rotation0_h2','rotation0_h10','rotation0_h7','rotation1_h2','rotation1_h10','rotation1_h7']
with (a.output_dir / 'families.csv').open('x', newline='', encoding='utf-8') as out:
    w = csv.DictWriter(out, fieldnames=family_fields, lineterminator='\n')
    w.writeheader()
    for family in audit['families']:
        s, plan, assessment = family['summary'], family['output_plan'], family['output_assessment']
        row = {'group':family['group'], 'baseline_status':family['baseline_status'], 'dose_status':s['dose_status'],
               'dose_reason':assessment.get('reason') or plan.get('reason'), 'q_out':s['q_out'],
               'minimum_variant_heads':json.dumps(plan.get('minima'), separators=(',', ':')),
               'actual_range':assessment.get('actual_range'), 'contrast':s.get('contrast'),
               'contrast_sign':s.get('contrast_sign'), 'h2_minus_h7':s.get('h2_minus_h7'),
               'h10_minus_h7':s.get('h10_minus_h7')}
        for h in ('h2','h10','h7'):
            row['D_'+h] = (s.get('D') or {}).get(h)
        for rotation, values in enumerate(s.get('rotations') or []):
            for h in ('h2','h10','h7'):
                row[f'rotation{rotation}_{h}'] = values[h]['target_log_probability_D']
        w.writerow(row)

case_fields = ['case_id','group','variant','rotation','shuffle','text','target_id','baseline_status','dose_status','q_out',
               'action','clean_target_logit','clean_target_log_probability','clean_target_probability','clean_target_rank',
               'clean_unique_top_id','clean_top_representative_id','clean_top_tie_count',
               'arm_target_logit','arm_target_log_probability','arm_target_probability','arm_target_rank',
               'arm_unique_top_id','arm_top_representative_id','arm_top_tie_count',
               'target_logit_delta','target_log_probability_delta','kl_clean_to_arm','total_variation','top_change',
               'input_removed_l2','predicted_output_l2','actual_output_l2','predicted_vector_residual_l2',
               'cosine_h2_h10','cosine_h2_h7','cosine_h10_h7']
with (a.output_dir / 'cases.csv').open('x', newline='', encoding='utf-8') as out:
    w = csv.DictWriter(out, fieldnames=case_fields, lineterminator='\n')
    w.writeheader()
    for case_id, record in clean.items():
        case = by_case[case_id]
        group, variant = record['group'], record['variant']
        family = family_by_group[group]
        s, assessment, score = family['summary'], family['output_assessment'], record['score']
        cosines = assessment.get('pair_cosines') or []
        cosine = cosines[variant] if len(cosines) > variant else {}
        base = {'case_id':case_id, 'group':group, 'variant':variant, 'rotation':case['rotation'], 'shuffle':case['shuffle'],
                'text':case['text'], 'target_id':record['target_id'], 'baseline_status':family['baseline_status'],
                'dose_status':s['dose_status'], 'q_out':s['q_out'], 'clean_target_logit':score['target_logit'],
                'clean_target_log_probability':score['target_log_probability'], 'clean_target_rank':score['target_rank'],
                'clean_target_probability':math.exp(score['target_log_probability']),
                'clean_unique_top_id':score['top_unique_id'], 'clean_top_representative_id':score['top_representative_id'],
                'clean_top_tie_count':score['top_tie_count'], 'cosine_h2_h10':cosine.get('2-10'),
                'cosine_h2_h7':cosine.get('2-7'), 'cosine_h10_h7':cosine.get('10-7')}
        actions = action_by_case.get(case_id) or [None]
        for action in actions:
            row = dict(base)
            if action is not None:
                arm, comparison, dose = action['score'], action['comparison'], action['dose']
                row.update({'action':action['action'], 'arm_target_logit':arm['target_logit'],
                    'arm_target_log_probability':arm['target_log_probability'], 'arm_target_rank':arm['target_rank'],
                    'arm_target_probability':math.exp(arm['target_log_probability']), 'arm_unique_top_id':arm['top_unique_id'],
                    'arm_top_representative_id':arm['top_representative_id'], 'arm_top_tie_count':arm['top_tie_count']})
                row.update({k:comparison[k] for k in ('target_logit_delta','target_log_probability_delta',
                           'kl_clean_to_arm','total_variation','top_change')})
                row.update({k:dose[k] for k in ('input_removed_l2','predicted_output_l2','actual_output_l2',
                           'predicted_vector_residual_l2')})
            else:
                row['action'] = 'not-run-dose-ineligible'
            w.writerow(row)
print(json.dumps({'cases_sha256':digest(cases_raw), 'result_sha256':digest(result_raw),
                  'audit_sha256':digest(audit_raw), 'families':len(audit['families']),
                  'cases':len(clean), 'active_actions':len(audit['actions'])}))
