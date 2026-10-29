"""Export compact tables from the independent saved-data pair/dose replay."""
import argparse
import csv
import json
from pathlib import Path


def write_table(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    families = json.loads(args.replay.read_bytes())['replay']['families']
    if len(families) != 8:
        raise ValueError('expected eight replayed families')
    args.output_dir.mkdir(exist_ok=False)
    arms, interactions, prompts, singles, pairs = [], [], [], [], []
    for group, family in enumerate(families, 1):
        numerical, delivery = family['numerical'], family['delivery']
        for arm, metrics in sorted(numerical['arms'].items()):
            lp, logit = metrics['target_log_probability'], metrics['target_logit']
            row = {'family': group, 'arm': arm, 'logprob_D': lp['contrast'], 'logit_D': logit['contrast']}
            for index in range(4):
                row[f'logprob_delta_variant_{index}'] = lp['deltas'][index]
                row[f'logit_delta_variant_{index}'] = logit['deltas'][index]
            arms.append(row)
        for arm, metrics in sorted(numerical['pairs'].items()):
            lp, logit = metrics['log_probability'], metrics['target_logit']
            row = {'family': group, 'pair_arm': arm,
                   'logprob_D_pair': lp['pair'], 'logprob_D_single_a': lp['single_a'],
                   'logprob_D_single_b': lp['single_b'], 'logprob_J': lp['interaction'],
                   'logprob_D_add': lp['additive_contrast'], 'logprob_J_add': lp['additive_interaction'],
                   'logprob_D_pair_minus_add': lp['pair_minus_additive'],
                   'target_logit_J': logit['interaction'], 'target_logit_D_pair_minus_add': logit['pair_minus_additive']}
            for index in range(4):
                row[f'logprob_J_variant_{index}'] = lp['prompt_interactions'][index]
                row[f'logprob_J_add_variant_{index}'] = lp['additive_prompt_interactions'][index]
                row[f'additive_logprob_delta_variant_{index}'] = lp['additive_deltas'][index]
                row[f'target_logit_J_variant_{index}'] = logit['prompt_interactions'][index]
            interactions.append(row)
        for variant, case in enumerate(numerical['rows']):
            clean = case['scores']['clean']
            for arm, score in sorted(case['scores'].items()):
                if arm == 'clean':
                    continue
                reference = case['additive'].get(arm, {})
                prompts.append({'family': group, 'variant': variant, 'case_id': family['case_ids'][variant],
                                'arm': arm, 'target_id': case['target_id'],
                                'clean_logprob': clean['target_log_probability'],
                                'arm_logprob': score['target_log_probability'],
                                'clean_target_logit': clean['target_logit'], 'arm_target_logit': score['target_logit'],
                                'clean_rank': clean['target_rank'], 'arm_rank': score['target_rank'],
                                'clean_top_token_id': clean['top_token_id'], 'arm_top_token_id': score['top_token_id'],
                                'clean_target_top1': clean['top_token_id'] == case['target_id'],
                                'arm_target_top1': score['top_token_id'] == case['target_id'],
                                'additive_logprob': reference.get('target_log_probability'),
                                'additive_target_logit': reference.get('target_logit'),
                                'additive_rank': reference.get('target_rank'),
                                'additive_top_token_id': reference.get('top_token_id')})
            for arm, item in sorted(delivery['variants'][variant]['singles'].items()):
                singles.append({'family': group, 'variant': variant, 'case_id': family['case_ids'][variant],
                                'arm': arm, 'family_m': delivery['minimum_norm'],
                                'source_l2': delivery['variants'][variant]['norms'][arm.split('_')[0]],
                                'requested_l2': item['requested'], 'achieved_l2': item['achieved'],
                                'requested_fraction': item['fraction'], 'changed_float32_elements': item['changed_elements'],
                                'changed_binary_bits': item['changed_bit_count']})
            for arm, item in sorted(delivery['variants'][variant]['pairs'].items()):
                dose = arm.rsplit('_', 1)[1]
                pairs.append({'family': group, 'variant': variant, 'case_id': family['case_ids'][variant],
                              'arm': arm, 'family_m': delivery['minimum_norm'],
                              'requested_l2_per_head': delivery['q_' + dose],
                              'requested_total_l2': delivery['q_' + dose] * 2 ** 0.5,
                              'achieved_total_l2': item['combined_norm'],
                              'match_single_a_l2': item.get('match_a_norm'),
                              'match_single_b_l2': item.get('match_b_norm'),
                              'pair_minus_match_a_l2': item.get('pair_minus_match_a_norm'),
                              'pair_minus_match_b_l2': item.get('pair_minus_match_b_norm')})
    if (len(arms), len(interactions), len(prompts), len(singles), len(pairs)) != (120, 48, 480, 288, 192):
        raise ValueError('replay does not contain the expected full design')
    for name, rows in (('family-arm-effects.csv', arms), ('family-pair-interactions.csv', interactions),
                       ('prompt-results.csv', prompts), ('single-deliveries.csv', singles),
                       ('pair-deliveries.csv', pairs)):
        write_table(args.output_dir / name, rows)


if __name__ == '__main__':
    main()
