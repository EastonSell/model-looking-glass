"""Fixed CPU FLAN-T5 matched-position dose-response follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .flan_t5_equal_dose import (
    Evidence, aggregate_decision, check_publication_observation, dose_matches,
    equal_bits, hook_snapshot, model_snapshot, require_clean_hooks, run_forward,
    validate_landing, vector_l2,
)
from .flan_t5_feasibility import MANIFEST_HASH, REVISION, digest_file, memory_bytes, verify_assets

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / 'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json'
PROTOCOL = REPO / 'docs/FLAN_T5_DOSE_RESPONSE_V1.md'
CASE_HASH = '629036188d62403669fe33291fa9e2a8759145be828918db25fa3ea7c0555a18'
MAX_FORWARDS = 468
MAX_BYTES = 512 * 1024 * 1024
SEED = 20261028
LEVELS = ((1, 3), (2, 3), (1, 1))
SITES = ('word', 'instruction', 'punctuation')


def source_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    paths = ('looking_glass/flan_t5_dose_response.py', 'looking_glass/flan_t5_equal_dose.py',
             'looking_glass/flan_t5_feasibility.py', 'looking_glass/interventions.py',
             'looking_glass/intervention_plan.py', 'looking_glass/model_ownership.py',
             'docs/FLAN_T5_DOSE_RESPONSE_V1.md', 'docs/FLAN_T5_DOSE_RESPONSE_CASES_V1.json')
    hashes = {}
    for name in paths:
        raw = (REPO / name).read_bytes()
        if raw != subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=REPO):
            raise ValueError(f'source differs from committed blob: {name}')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': commit, 'source_sha256': hashes}


def check_cases(tokenizer, manifest):
    if (type(tokenizer).__name__ != 'T5TokenizerFast' or tokenizer.eos_token_id != 1
            or tokenizer.pad_token_id != 0 or manifest.get('schema') != 'mlg-flan-t5-dose-response-public-cases-v1'
            or manifest.get('revision') != REVISION or manifest.get('asset_manifest_sha256') != MANIFEST_HASH
            or len(manifest.get('cases', ())) != 6):
        raise ValueError('case or tokenizer identity differs')
    ids = ('pair0-translate', 'pair0-question', 'pair1-translate', 'pair1-question',
           'pair2-translate', 'pair2-question')
    if tuple(c.get('case_id') for c in manifest['cases']) != ids:
        raise ValueError('case order differs')
    eligible = []
    for index, case in enumerate(manifest['cases']):
        rows = [case['neutral'], *case['source'], *case['targets']]
        for row in rows:
            value = tokenizer(row['text'], return_offsets_mapping=True, return_special_tokens_mask=True)
            for key, saved in (('input_ids', 'input_ids'), ('attention_mask', 'attention_mask'),
                               ('special_tokens_mask', 'special_tokens_mask'), ('offset_mapping', 'offsets')):
                actual = [list(v) if isinstance(v, tuple) else v for v in value[key]]
                if actual != row[saved]:
                    raise ValueError(f'tokenizer mismatch: {key}')
            if tokenizer.convert_ids_to_tokens(value['input_ids']) != row['token_strings']:
                raise ValueError('token string mismatch')
        a, b = case['source']
        positions = [case.get(k) for k in ('word_position', 'instruction_position', 'context_position')]
        differences = [i for i, (x, y) in enumerate(zip(a['input_ids'], b['input_ids'])) if x != y]
        expected_positions = [6, 0, 5 if case['template_id'] == 'translate' else 7]
        reasons = []
        if (case.get('pair_id') != index // 2 or case.get('template_id') != ('translate', 'question')[index % 2]
                or positions != expected_positions or len(a['input_ids']) != len(b['input_ids'])
                or a['attention_mask'] != b['attention_mask'] or a['special_tokens_mask'] != b['special_tokens_mask']
                or len(a['input_ids']) > 32 or a['word_positions'] != b['word_positions']
                or a['word_positions'] != differences or len(differences) != 1
                or any(type(p) is not int or p < 0 or p >= len(a['input_ids']) for p in positions)
                or len(set(positions)) != 3 or positions[0] != differences[0]
                or any(a['special_tokens_mask'][p] or b['special_tokens_mask'][p] for p in positions)
                or any(a['input_ids'][p] != b['input_ids'][p] for p in positions[1:])
                or any(any(v != 1 for v in row['attention_mask']) for row in rows)):
            reasons.append('source-token-alignment')
        if not reasons:
            for row in (a, b):
                start, end = row['offsets'][positions[0]]
                if row['text'][start:end].strip() != row['word']:
                    reasons.append('source-word-offset')
        if (any(not 2 <= len(row['input_ids']) <= 3 or row['input_ids'][-1] != 1
                or row['input_ids'].count(1) != 1 for row in case['targets'])
                or case['targets'][0]['input_ids'] == case['targets'][1]['input_ids']
                or any(not row['input_ids'] or row['input_ids'][-1] != 1 or row['input_ids'].count(1) != 1
                       for row in [case['neutral'], *case['source']])):
            reasons.append('target-or-eos')
        if (type(case.get('alignment_eligible')) is not bool or bool(reasons) == case['alignment_eligible']
                or case.get('eligibility_reasons') != reasons):
            raise ValueError('saved alignment status disagrees with fixed tokenization')
        eligible.append(reasons)
    return eligible


def match_response_levels(states, positions):
    """Compute every float32 delivery before accepting any level."""
    import torch
    distances = {site: vector_l2(states[0][0, positions[site]], states[1][0, positions[site]])
                 for site in SITES}
    q = min(distances.values())
    if q / 3 <= 1e-6:
        return {'status': 'dose-ineligible', 'distances': distances, 'q': q}, {}
    deliveries = {}
    levels = []
    for numerator, denominator in LEVELS:
        fraction = numerator / denominator
        level_q = q * fraction
        tolerance = min(max(1e-5, 1e-5 * level_q), .01 * level_q)
        achieved, receiver_norms = {}, {}
        for site in SITES:
            pos = positions[site]
            for side in range(2):
                own, donor = states[side][0, pos], states[1-side][0, pos]
                delivered = (own.double() + (level_q / distances[site]) *
                             (donor.double() - own.double())).to(torch.float32)
                if not bool(torch.isfinite(delivered).all()):
                    raise ValueError('matched replacement is nonfinite')
                deliveries[numerator, denominator, site, side] = delivered
                key = f'{site}-{side}'
                achieved[key] = vector_l2(delivered, own)
                receiver_norms[key] = vector_l2(own, torch.zeros_like(own))
        levels.append({'numerator': numerator, 'denominator': denominator, 'q': level_q,
                       'tolerance': tolerance, 'achieved': achieved, 'receiver_norms': receiver_norms,
                       'q_over_receiver_norm': {k: level_q / n if n > 0 else None
                                                for k, n in receiver_norms.items()},
                       'status': 'matchable' if dose_matches(level_q, tolerance, achieved)
                       else 'dose-unmatchable'})
    return {'status': 'matchable' if all(level['status'] == 'matchable' for level in levels)
            else 'dose-unmatchable', 'distances': distances, 'q': q, 'levels': levels}, deliveries


def response_decision(delta, movements):
    """Return the fixed binary decision and its directional failure reasons."""
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError('invalid baseline gap')
    reasons = []
    for side in range(2):
        previous = 0.0
        for index, (numerator, denominator) in enumerate(LEVELS):
            word = movements['word'][index][side]
            controls = [movements[site][index][side] for site in ('instruction', 'punctuation')]
            if not all(math.isfinite(x) for x in [word, *controls]):
                raise ValueError('nonfinite dose-response movement')
            if word < -1e-4:
                reasons.append(f'reversed-side{side}-level{index + 1}')
            if word < previous - 1e-4:
                reasons.append(f'nonmonotone-side{side}-level{index + 1}')
            if word < max(abs(x) for x in controls) + .02 * (numerator / denominator) * delta:
                reasons.append(f'control_comparison-side{side}-level{index + 1}')
            previous = word
        if previous < .05 * delta:
            reasons.append(f'weak_endpoint-side{side}')
    return ('negative' if reasons else 'positive'), reasons


def save_response_dose(evidence, index, case_id, dose, states, positions, deliveries):
    chunks, layout = [], []
    for site in SITES:
        for side in range(2):
            values = [('own', states[side][0, positions[site]]),
                      ('donor', states[1-side][0, positions[site]])]
            values.extend((f'{a}-{b}', deliveries[a, b, site, side])
                          for a, b in LEVELS if (a, b, site, side) in deliveries)
            for role, vector in values:
                raw = vector.contiguous().numpy().astype('<f4', copy=False).tobytes()
                layout.append({'site': site, 'side': side, 'role': role,
                               'offset': sum(len(x) for x in chunks), 'bytes': len(raw),
                               'shape': [512], 'dtype': 'little-endian-float32'})
                chunks.append(raw)
    filename = f'dose-{index:02d}.bin'
    sha = evidence.write(filename, b''.join(chunks))
    evidence.json(f'dose-{index:02d}.json', {'case_id': case_id, 'dose': dose,
                                             'file': filename, 'sha256': sha, 'layout': layout})


def run_cases(manifest, eligibility, evidence, records, active, forward):
    import torch
    summaries = []
    for index, case in enumerate(manifest['cases']):
        active.update(case_id=case['case_id'], forward=None, phase='case')
        summary = {'case_id': case['case_id'], 'pair_id': case['pair_id'], 'template_id': case['template_id']}
        if eligibility[index]:
            summary.update(status='alignment-ineligible', reasons=eligibility[index])
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        targets = case['targets']
        context = [forward(case['neutral'], target, f'{case["case_id"]}-context-{j}')[0]
                   for j, target in enumerate(targets)]
        bases, states, scores = {}, {}, {}
        for side in range(2):
            for target in range(2):
                score, logits, state, _, _ = forward(case['source'][side], targets[target],
                                                      f'{case["case_id"]}-base-{side}-{target}')
                bases[side, target] = logits
                scores['base', side, target] = score
                if side in states and not equal_bits(states[side], state):
                    raise ValueError('encoder state differs between target forwards')
                states[side] = state
        margins = [scores['base', side, 0] - scores['base', side, 1] for side in range(2)]
        delta = margins[0] - margins[1]
        summary.update(baseline_margins=margins, delta=delta, context_scores=context,
                       context_margin=context[0] - context[1])
        if not (margins[0] > 0 and margins[1] < 0 and delta >= 1.0):
            summary['status'] = 'baseline-unsupported'
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue
        positions = {'word': case['word_position'], 'instruction': case['instruction_position'],
                     'punctuation': case['context_position']}
        dose, deliveries = match_response_levels(states, positions)
        summary['dose'] = dose
        save_response_dose(evidence, index, case['case_id'], dose, states, positions, deliveries)
        if dose['status'] != 'matchable':
            summary['status'] = dose['status']
            summaries.append(summary)
            evidence.json(f'case-{index:02d}.json', summary)
            continue

        def execute(condition, side, action, pos, value, level):
            own, donor = states[side], states[1-side]
            expected = None if pos is None else (torch.zeros_like(own[0, pos]) if action == 'zero' else
                                                  own[0, pos] if action == 'null' else value)
            for target in range(2):
                score, logits, state, before, after = forward(
                    case['source'][side], targets[target], f'{case["case_id"]}-{condition}-{side}-{target}',
                    action, value, pos)
                scores[condition, side, target] = score
                validate_landing(condition, action, pos, own, donor, expected, value,
                                 state, before, after, level)
                if condition in ('repeat', 'null_word', 'self_word', 'self_instruction',
                                 'self_punctuation', 'full_self') and not equal_bits(logits, bases[side, target]):
                    raise ValueError(f'identity logits control failed: {condition}')
                if condition == 'full_swap' and not equal_bits(logits, bases[1-side, target]):
                    raise ValueError('full-swap logits control failed')

        anchors = ('null_word', 'self_word', 'self_instruction', 'self_punctuation',
                   'full_self', 'full_swap', 'full_word_swap', 'zero_word')
        for condition in anchors:
            for side in range(2):
                own, donor = states[side], states[1-side]
                if condition in ('full_self', 'full_swap'):
                    action, pos, value = 'patch', None, own if condition == 'full_self' else donor
                else:
                    site = ('instruction' if condition.endswith('instruction') else
                            'punctuation' if condition.endswith('punctuation') else 'word')
                    pos = positions[site]
                    action = 'null' if condition == 'null_word' else 'zero' if condition == 'zero_word' else 'patch'
                    value = (None if action != 'patch' else donor[0, pos] if condition == 'full_word_swap'
                             else own[0, pos])
                execute(condition, side, action, pos, value, None)

        for level in dose['levels']:
            a, b = level['numerator'], level['denominator']
            for site in SITES:
                condition = f'matched_{site}_{a}of{b}'
                for side in range(2):
                    execute(condition, side, 'patch', positions[site], deliveries[a, b, site, side], level)
        # The final capture repeat follows all interventions to detect state drift.
        for side in range(2):
            execute('repeat', side, 'capture', None, None, None)

        all_conditions = [*anchors, 'repeat'] + [f'matched_{site}_{a}of{b}' for a, b in LEVELS for site in SITES]
        condition_margins = {condition: [scores[condition, side, 0] - scores[condition, side, 1]
                                          for side in range(2)] for condition in all_conditions}
        moves = {condition: [margins[0] - v[0], v[1] - margins[1]]
                 for condition, v in condition_margins.items()}
        curves = {site: [moves[f'matched_{site}_{a}of{b}'] for a, b in LEVELS] for site in SITES}
        status, reasons = response_decision(delta, curves)
        full_curves = {site: [[0.0, 0.0], *curve] for site, curve in curves.items()}
        summary.update(status=status, reasons=reasons, positions=positions, condition_margins=condition_margins,
                       signed_movements=moves,
                       absolute_movements={k: [abs(x) for x in v] for k, v in moves.items()},
                       response_curves=full_curves,
                       absolute_response_curves={site: [[abs(x) for x in row] for row in curve]
                                                 for site, curve in full_curves.items()})
        summaries.append(summary)
        evidence.json(f'case-{index:02d}.json', summary)
        print(json.dumps({'case_id': case['case_id'], 'status': status, 'q': dose['q'],
                          'forward_count': len(records)}), flush=True)
    return summaries


def run(asset_dir, output, publication_observation):
    if (os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1')
            or os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('TRANSFORMERS_OFFLINE') != '1'
            or os.environ.get('MKL_CBWR') != 'COMPATIBLE' or os.environ.get('OMP_NUM_THREADS') != '1'
            or os.environ.get('MKL_NUM_THREADS') != '1'):
        raise ValueError('fixed CPU/offline/numeric environment is required')
    if 'torch' in sys.modules:
        raise ValueError('PyTorch must not be preloaded before environment checks')
    if digest_file(CASES) != CASE_HASH:
        raise ValueError('fixed case manifest differs')
    source_before = source_identity()
    observation = check_publication_observation(publication_observation, source_before)
    assets_before = verify_assets(asset_dir)
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 2 * MAX_BYTES:
        raise ValueError('insufficient free space or missing output parent')
    manifest = json.loads(CASES.read_bytes())
    evidence = Evidence(output, limit=MAX_BYTES)
    observation_hash = evidence.write('publication-observation.json', observation)
    evidence.json('started.json', {'schema': 'mlg-flant5-dose-response-attempt-v1', 'status': 'started',
                                   'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH,
                                   'case_manifest_sha256': CASE_HASH, 'evidence_cap': MAX_BYTES,
                                   'publication_observation_sha256': observation_hash,
                                   'max_forwards': MAX_FORWARDS})
    active = {'case_id': None, 'forward': None, 'phase': 'startup'}
    try:
        import torch
        import tokenizers
        import transformers
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False):
            raise ValueError('cannot preserve denormals')
        if int((torch.tensor([2.0 ** -149], dtype=torch.float32) * 1.0).view(torch.int32)[0]) != 1:
            raise ValueError('unexpected float32 subnormal behavior')
        torch.random.default_generator.manual_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(str(asset_dir), local_files_only=True,
                                                   use_fast=True, trust_remote_code=False)
        eligibility = check_cases(tokenizer, manifest)
        before_load = memory_bytes()
        model, load_info = T5ForConditionalGeneration.from_pretrained(
            str(asset_dir), local_files_only=True, use_safetensors=True, trust_remote_code=False,
            output_loading_info=True, dtype=torch.float32)
        if any(load_info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('model loading diagnostics are not empty')
        model.eval()
        model.config.use_cache = False
        if (type(model).__name__ != 'T5ForConditionalGeneration' or model.config.d_model != 512
                or model.config.vocab_size != 32128 or model.config.decoder_start_token_id != 0
                or model.config.eos_token_id != 1 or model.config.pad_token_id != 0
                or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError('loaded model differs from the fixed CPU configuration')
        hooks_before = hook_snapshot(model)
        require_clean_hooks(hooks_before)
        model_before = model_snapshot(model)
        runtime = {'python': platform.python_version(), 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                   'device': 'cpu', 'dtype': 'float32', 'seed': SEED, 'threads': torch.get_num_threads(),
                   'interop_threads': torch.get_num_interop_threads(),
                   'deterministic': torch.are_deterministic_algorithms_enabled(),
                   'attention_implementation': getattr(model.config, '_attn_implementation', None),
                   'numeric_environment': {k: os.environ.get(k) for k in
                                           ('MKL_CBWR', 'ATEN_CPU_CAPABILITY', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                   'before_load': before_load, 'after_load': memory_bytes(),
                   'model_parameter_buffer_sha256': model_before[0],
                   'torch_build_config': torch.__config__.show()}
        if runtime['attention_implementation'] != 'eager':
            raise ValueError('attention implementation is not eager')
        evidence.json('environment.json', runtime)
        records = []

        def forward(source, target, name, action='clean', value=None, token_index=None):
            return run_forward(model, evidence, records, active, source, target, name, action, value, token_index,
                               forward_limit=MAX_FORWARDS)

        summaries = run_cases(manifest, eligibility, evidence, records, active, forward)
        if source_identity() != source_before or verify_assets(asset_dir) != assets_before or hook_snapshot(model) != hooks_before:
            raise ValueError('source, assets or hooks changed during execution')
        if model_snapshot(model) != model_before:
            raise ValueError('model parameters or buffers changed during execution')
        result = {'schema': 'mlg-flant5-dose-response-v1', 'status': 'complete',
                  'decision': aggregate_decision(summaries), 'cases': summaries, 'forwards': len(records),
                  'source': source_before, 'asset_manifest_sha256': MANIFEST_HASH, 'case_manifest_sha256': CASE_HASH,
                  'environment': runtime, 'final_memory': memory_bytes(), 'hooks_restored': True,
                  'bytes_before_terminal': evidence.used,
                  'limitations': ['Public, correlated pairs and templates; no statistical or generalization claim.',
                                  'Matched input L2 does not match direction or downstream functional effect.',
                                  'Exact sequence scores include EOS and differ in target length and priors.']}
        sha = evidence.json('completed.json', result, reserve=0)
        print(json.dumps({'status': 'complete', 'decision': result['decision'], 'forwards': len(records),
                          'result_sha256': sha, 'retained_bytes': evidence.used}), flush=True)
        return result
    except BaseException as error:
        try:
            evidence.json('failure.json', {'schema': 'mlg-flant5-dose-response-attempt-v1', 'status': 'incomplete',
                                          'error_type': type(error).__name__, 'active': active,
                                          'retained_bytes_before_failure': evidence.used}, reserve=0)
        except BaseException:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--publication-observation', required=True, type=Path)
    args = parser.parse_args()
    run(args.asset_dir.resolve(), args.output.resolve(), args.publication_observation.resolve())


if __name__ == '__main__':
    main()
