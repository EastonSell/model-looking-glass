"""Post-hoc distribution analysis of the closed screened-edit logits.

This program does not load or run a model. The logit interpolation is a
mathematical sensitivity calculation, not a forward pass at a new edit dose.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


RESULT = Path('results/targeted-edit-screened-matched-v1-20261029')
CASE = Path('experiments/targeted_edit_screened_public_cases_v1.json')
PHASE_SHA = '79faacd17bac96039729e241e0ef92b853f70b8b8a9fb326b09ff1e4fbc4bd78'
CASE_SHA = 'fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3'
SELECTION_SHA = '9ea861d89c8ed78c038947dbdb331d6dba1ea8267901ce06e5492dab32e7c11f'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
SOURCE_REVISION = '475228289efa7840fa577833d17573a6a73e001e'
CONDITIONS = ('original', 'projected', 'unprojected', 'projected_random',
              'opposite', 'nuisance', 'matched')
VOCABULARY = 50257
ROW_BYTES = VOCABULARY * 4
LONDON, PARIS = 3576, 6342


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, digest, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('input is not an ordinary file: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or sha(raw) != digest:
        raise ValueError('input differs from its fixed digest: ' + str(path))
    return raw


def bytelevel_tokens(raw):
    """Decode individual GPT-2 byte-level tokens, preserving their IDs."""
    vocab = json.loads(raw)['model']['vocab']
    if (type(vocab) is not dict or len(vocab) != VOCABULARY
            or any(type(v) is not int for v in vocab.values())
            or set(vocab.values()) != set(range(VOCABULARY))):
        raise ValueError('tokenizer vocabulary differs')
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = byte_values[:]
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            characters.append(256 + len(characters) - 188)
    decoder = {chr(character): value for value, character in zip(byte_values, characters)}
    tokens = [None] * VOCABULARY
    for text, index in vocab.items():
        try:
            tokens[index] = bytes(decoder[ch] for ch in text).decode('utf-8', errors='replace')
        except KeyError as error:
            raise ValueError('tokenizer contains an unsupported byte-level character') from error
    return tokens


def log_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('expected finite two-dimensional logits')
    shifted = values - np.max(values, axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def distribution_changes(baseline_logp, condition_logp):
    """KL(P||Q), total variation and Jensen-Shannon divergence, in nats."""
    if baseline_logp.shape != condition_logp.shape:
        raise ValueError('distribution shapes differ')
    p, q = np.exp(baseline_logp), np.exp(condition_logp)
    middle_logp = np.logaddexp(baseline_logp, condition_logp) - math.log(2.0)
    kl = np.sum(p * (baseline_logp - condition_logp), axis=1)
    tv = np.sum(np.abs(p - q), axis=1) / 2.0
    js = (np.sum(p * (baseline_logp - middle_logp), axis=1)
          + np.sum(q * (condition_logp - middle_logp), axis=1)) / 2.0
    if not (np.isfinite(kl).all() and np.isfinite(tv).all() and np.isfinite(js).all()):
        raise ValueError('nonfinite distribution metric')
    return kl, tv, js


def matched_logit_interpolation(original, projected, matched):
    """Match the observed mean target-margin gain by interpolating saved logits."""
    if any(np.asarray(item).ndim != 2 for item in (original, projected, matched)):
        raise ValueError('interpolation requires row matrices')
    if original.shape != projected.shape or original.shape != matched.shape or original.shape[0] < 8:
        raise ValueError('interpolation matrix shapes differ')
    margin = lambda x: x[:8, LONDON] - x[:8, PARIS]
    projected_gain = float(np.mean(margin(projected) - margin(original)))
    matched_gain = float(np.mean(margin(matched) - margin(original)))
    if not math.isfinite(projected_gain) or not math.isfinite(matched_gain) or matched_gain <= 0:
        raise ValueError('target gain is not positive and finite')
    scale = projected_gain / matched_gain
    if not 0 <= scale <= 1:
        raise ValueError('sensitivity calculation would be extrapolation')
    return scale, original + scale * (matched - original)


def rank(probabilities, index):
    ids = np.arange(len(probabilities))
    return 1 + int(np.count_nonzero((probabilities > probabilities[index])
                                    | ((probabilities == probabilities[index]) & (ids < index))))


def load_closed_data(root):
    """Verify the fixed case/selection/phase and every matrix and row digest."""
    root = Path(root)
    phase_raw = pinned(root / RESULT / 'evaluation/phase-record.json', PHASE_SHA, 256 * 1024)
    case_raw = pinned(root / CASE, CASE_SHA, 2 * 1024 * 1024)
    selection_raw = pinned(root / RESULT / 'screen/screen/screen-selection.json', SELECTION_SHA, 256 * 1024)
    phase, document, selection = map(json.loads, (phase_raw, case_raw, selection_raw))
    if phase.get('status') != 'complete' or phase.get('phase') != 'evaluation':
        raise ValueError('saved evaluation is incomplete')
    cases = document['cases']
    by_id = {case['case_id']: case for case in cases}
    selected = selection['selection']['selected_ids']
    locality = [case['case_id'] for case in cases if case['role'] == 'locality']
    order = selected + locality
    if (len(by_id) != len(cases) or len(selected) != 8 or len(locality) != 12
            or len(set(order)) != 20 or any(by_id[name]['role'] != 'target' for name in selected)):
        raise ValueError('fixed selected and locality cases differ')
    journal_spec = phase['artifacts']['evaluation/observations.jsonl']
    data_dir = root / RESULT / 'evaluation/evaluation'
    journal = pinned(data_dir / 'observations.jsonl', journal_spec['sha256'], 256 * 1024)
    if len(journal) != journal_spec['bytes'] or not journal.endswith(b'\n'):
        raise ValueError('observation journal is incomplete')
    records = [json.loads(line) for line in journal.splitlines()]
    if len(records) != len(CONDITIONS) * 20:
        raise ValueError('observation count differs')
    matrices, hashes = {}, {}
    for condition_index, condition in enumerate(CONDITIONS):
        spec = phase['artifacts']['evaluation/' + condition + '.f32']
        raw = pinned(data_dir / (condition + '.f32'), spec['sha256'], ROW_BYTES * 20)
        if len(raw) != spec['bytes'] or len(raw) != ROW_BYTES * 20:
            raise ValueError('logit matrix size differs')
        for index, case_id in enumerate(order):
            record = records[condition_index * 20 + index]
            piece = raw[index * ROW_BYTES:(index + 1) * ROW_BYTES]
            if (record.get('condition') != condition or record.get('case_id') != case_id
                    or type(record.get('index')) is not int or record['index'] != index
                    or record.get('offset') != index * ROW_BYTES or record.get('bytes') != ROW_BYTES
                    or record.get('sha256') != sha(piece)):
                raise ValueError('logit row identity or digest differs')
        matrix = np.frombuffer(raw, dtype='<f4').reshape(20, VOCABULARY).astype(np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError('nonfinite saved logits')
        matrices[condition] = matrix
        hashes[condition] = spec['sha256']
    return [by_id[name] for name in order], matrices, {
        'published_result_commit': SOURCE_REVISION, 'execution_source_commit': phase['source_commit'],
        'phase_sha256': PHASE_SHA,
        'cases_sha256': CASE_SHA, 'selection_sha256': SELECTION_SHA,
        'observations_sha256': journal_spec['sha256'], 'matrix_sha256': hashes,
        'tokenizer_sha256': TOKENIZER_SHA}


def _token_change(p, q, tokens):
    difference = q - p
    plus, minus = int(np.argmax(difference)), int(np.argmin(difference))
    top0, top1 = int(np.argmax(p)), int(np.argmax(q))
    return {'top_before_id': top0, 'top_before_text': tokens[top0],
            'top_after_id': top1, 'top_after_text': tokens[top1],
            'largest_gain_id': plus, 'largest_gain_text': tokens[plus],
            'largest_gain_probability_points': float(100 * difference[plus]),
            'largest_loss_id': minus, 'largest_loss_text': tokens[minus],
            'largest_loss_probability_points': float(100 * difference[minus])}


def calculate(cases, matrices, tokens):
    original = matrices['original']
    logp = log_probabilities(original)
    p = np.exp(logp)
    local_rows, target_rows, sensitivity = [], [], []
    stats, distributions = {}, {}
    baseline_margin = original[:8, LONDON] - original[:8, PARIS]
    for condition in CONDITIONS[1:]:
        logits = matrices[condition]
        logq = log_probabilities(logits)
        q = np.exp(logq)
        kl, tv, js = distribution_changes(logp, logq)
        gains = logits[:8, LONDON] - logits[:8, PARIS] - baseline_margin
        distributions[condition] = (kl, tv, q)
        stats[condition] = {'mean_target_gain': float(np.mean(gains)),
                            'mean_locality_kl': float(np.mean(kl[8:])),
                            'mean_locality_tv': float(np.mean(tv[8:])),
                            'max_locality_tv': float(np.max(tv[8:])),
                            'mean_locality_js': float(np.mean(js[8:])),
                            'changed_locality_tops': int(np.count_nonzero(np.argmax(p[8:], axis=1)
                                                                            != np.argmax(q[8:], axis=1)))}
        for i, case in enumerate(cases):
            common = {'case_id': case['case_id'], 'prompt': case['prompt'], 'stratum': case['stratum'],
                      'condition': condition, 'kl_nats': float(kl[i]), 'total_variation': float(tv[i]),
                      'js_nats': float(js[i]), **_token_change(p[i], q[i], tokens)}
            if i < 8:
                target_rows.append({**common, 'original_margin': float(baseline_margin[i]),
                                    'condition_margin': float(baseline_margin[i] + gains[i]),
                                    'target_gain': float(gains[i]),
                                    'london_probability_before': float(p[i, LONDON]),
                                    'london_probability_after': float(q[i, LONDON]),
                                    'paris_probability_before': float(p[i, PARIS]),
                                    'paris_probability_after': float(q[i, PARIS])})
            else:
                continuation = case['continuations'][0]
                expected = continuation['ids'][0]
                local_rows.append({**common, 'expected_id': expected, 'expected_text': continuation['text'],
                                   'expected_probability_before': float(p[i, expected]),
                                   'expected_probability_after': float(q[i, expected]),
                                   'expected_probability_change_points': float(100 * (q[i, expected] - p[i, expected])),
                                   'expected_nll_change_nats': float(logp[i, expected] - logq[i, expected]),
                                   'expected_rank_before': rank(p[i], expected),
                                   'expected_rank_after': rank(q[i], expected)})

    alpha, interpolated = matched_logit_interpolation(original, matrices['projected'], matrices['matched'])
    kl_i, tv_i, _ = distribution_changes(logp, log_probabilities(interpolated))
    kl_p, tv_p, _ = distributions['projected']
    kl_m, tv_m, _ = distributions['matched']
    for i, case in enumerate(cases[8:], 8):
        sensitivity.append({'case_id': case['case_id'], 'prompt': case['prompt'],
                            'projected_kl': float(kl_p[i]), 'matched_kl': float(kl_m[i]),
                            'interpolated_logit_kl': float(kl_i[i]),
                            'projected_tv': float(tv_p[i]), 'matched_tv': float(tv_m[i]),
                            'interpolated_logit_tv': float(tv_i[i])})
    target_comparison = []
    proj = matrices['projected'][:8, LONDON] - matrices['projected'][:8, PARIS] - baseline_margin
    match = matrices['matched'][:8, LONDON] - matrices['matched'][:8, PARIS] - baseline_margin
    for i, case in enumerate(cases[:8]):
        target_comparison.append({'case_id': case['case_id'], 'prompt': case['prompt'],
                                  'projected_gain': float(proj[i]), 'matched_gain': float(match[i]),
                                  'interpolated_logit_gain': float(alpha * match[i]),
                                  'projected_minus_matched': float(proj[i] - match[i]),
                                  'projected_minus_interpolated': float(proj[i] - alpha * match[i]),
                                  'projected_over_matched': None if match[i] == 0 else float(proj[i] / match[i])})
    summary = {'condition_summaries': stats,
               'sensitivity': {'scale_alpha': alpha,
                  'construction': 'original logits + alpha * (matched logits - original logits)',
                  'scope': 'post-hoc interpolation of saved logits; not a model forward or a new edit',
                  'mean_projected_target_gain': float(np.mean(proj)),
                  'mean_interpolated_target_gain': float(np.mean(alpha * match)),
                  'mean_interpolated_locality_kl': float(np.mean(kl_i[8:])),
                  'mean_interpolated_locality_tv': float(np.mean(tv_i[8:])),
                  'projected_lower_kl_than_actual_matched': int(np.count_nonzero(kl_p[8:] < kl_m[8:])),
                  'projected_lower_kl_than_interpolation': int(np.count_nonzero(kl_p[8:] < kl_i[8:])),
                  'projected_lower_tv_than_actual_matched': int(np.count_nonzero(tv_p[8:] < tv_m[8:])),
                  'projected_lower_tv_than_interpolation': int(np.count_nonzero(tv_p[8:] < tv_i[8:]))}}
    return local_rows, target_rows, sensitivity, target_comparison, summary


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-json', required=True)
    parser.add_argument('--output-dir', required=True, help='new directory; existing files are never overwritten')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cases, matrices, provenance = load_closed_data(root)
    tokens = bytelevel_tokens(pinned(args.tokenizer_json, TOKENIZER_SHA, 8 * 1024 * 1024))
    local, targets, sensitivity, comparisons, summary = calculate(cases, matrices, tokens)
    summary.update({'scope': 'post-hoc saved next-token distributions; no new model execution',
                    'provenance': provenance, 'numpy_version': np.__version__,
                    'limitations': ['public baseline-selected cases, not a blind or representative sample',
                       'locality covers only 12 next-token distributions, not general capability',
                       'interpolation is selected on these same targets and is not an actual edited model',
                       'saved evidence and process records do not authenticate execution']})
    destination = Path(args.output_dir)
    destination.mkdir(parents=False, exist_ok=False)
    for name, rows in (('locality.csv', local), ('targets.csv', targets),
                       ('interpolated-locality.csv', sensitivity), ('target-comparison.csv', comparisons)):
        write_csv(destination / name, rows)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
                                               encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'complete', 'locality_rows': len(local), 'target_rows': len(targets),
                      'interpolated_locality_rows': len(sensitivity),
                      'scale_alpha': summary['sensitivity']['scale_alpha']}, sort_keys=True))


if __name__ == '__main__':
    main()
