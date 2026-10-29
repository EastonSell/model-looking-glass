"""Compare a closed full-top screen writer record to an independent raw-logit audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def invalid_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid_constant)


def require(condition, label):
    if not condition:
        raise ValueError(label)


def compare(independent, screen_dir):
    record = load(screen_dir / 'screen-selection.json')
    require(record.get('version') == 'targeted-edit-full-top-private-screen-v1'
            and record.get('status') == 'private-screen', 'screen record version/status')
    require(record['case_sha256'] == independent['cases_sha256'], 'case hash')
    require(digest(screen_dir / 'screen.f32') == independent['logits_sha256'], 'raw logits hash')
    for name in ('screen.f32', 'observations.jsonl', 'state-original.json'):
        meta = record['artifacts'][name]
        file = screen_dir / name
        require(file.stat().st_size == meta['bytes'] and digest(file) == meta['sha256'], 'artifact ' + name)
    selection = record['selection']
    own_decision = independent['decision']
    for key in ('status', 'fact_index', 'discovery_offset', 'calibration_offsets', 'evaluation_offsets'):
        require(selection[key] == own_decision[key], 'decision ' + key)
    require(len(selection['candidates']) == len(independent['facts']) == 4, 'candidate count')
    for owner, own in zip(selection['candidates'], independent['facts']):
        for key in ('fact_index', 'discovery_eligible', 'calibration_eligible_count',
                    'evaluation_eligible_count', 'qualifies', 'eligible_evaluation_offsets'):
            require(owner[key] == own[key], 'candidate ' + str(own['fact_index']) + ' ' + key)
    require(len(selection['observations']) == len(independent['observations']) == 36, 'observation count')
    row_size = independent['vocab_size'] * 4
    max_logp_difference = 0.0
    for owner, own in zip(selection['observations'], independent['observations']):
        for key in ('offset', 'case_id', 'fact_id', 'role', 'eligible', 'unique_top_id',
                    'top_tie_count', 'top_representative_id', 'top_gap', 'factual_logit'):
            require(owner[key] == own[key], 'observation ' + own['case_id'] + ' ' + key)
        require(owner['alternative_logit'] == own['counterfactual_logit'], 'alternative logit ' + own['case_id'])
        require(owner['logit_sha256'] == own['row_sha256']
                and owner['logit_byte_offset'] == own['offset'] * row_size
                and owner['logit_bytes'] == row_size, 'row identity ' + own['case_id'])
        for owner_key, own_key in (('factual_logp', 'factual_logp'), ('alternative_logp', 'counterfactual_logp')):
            a, b = float(owner[owner_key]), float(own[own_key])
            require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12),
                    'log probability ' + own['case_id'] + ' ' + owner_key)
            max_logp_difference = max(max_logp_difference, abs(a - b))
    journal_path = screen_dir / 'observations.jsonl'
    with journal_path.open(encoding='utf-8') as stream:
        journal = [json.loads(line, parse_constant=invalid_constant) for line in stream]
    require(len(journal) == 36, 'journal row count')
    for own, item in zip(independent['observations'], journal):
        require(item['index'] == own['offset'] and item['case_id'] == own['case_id']
                and item['offset'] == own['offset'] * row_size and item['bytes'] == row_size
                and item['sha256'] == own['row_sha256'], 'journal ' + own['case_id'])
    token_path = screen_dir / 'screen-top-tokens.json'
    decoded = load(token_path)
    require(len(decoded) == 36, 'decoded top-token count')
    for own, item in zip(independent['observations'], decoded):
        for key in ('case_id', 'unique_top_id', 'top_tie_count', 'top_representative_id'):
            require(item[key] == own[key], 'decoded top identity ' + own['case_id'] + ' ' + key)
        require(item['top_representative_text'] == own['representative_decoded_token']
                and item['top_representative_raw_token'] == own['representative_token_string'],
                'decoded representative ' + own['case_id'])
        is_unique = own['unique_top_id'] is not None
        require(item['unique_top_text'] == (own['representative_decoded_token'] if is_unique else None)
                and item['unique_top_raw_token'] == (own['representative_token_string'] if is_unique else None),
                'decoded unique top ' + own['case_id'])
    return {'status': 'passed', 'decision': own_decision, 'rows_checked': 36,
            'max_absolute_logp_difference': max_logp_difference,
            'selection_sha256': digest(screen_dir / 'screen-selection.json'),
            'journal_sha256': digest(journal_path), 'decoded_top_sha256': digest(token_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-report', type=Path, required=True)
    parser.add_argument('--screen-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    independent = load(args.independent_report)
    result = compare(independent, args.screen_dir)
    result.update({'schema': 'full-top-screen-owner-comparison-v1',
                   'scope': 'Saved-logit, selection and journal consistency only; not proof of execution, cleanup, chronology or unknown exposure.',
                   'comparator_sha256': digest(Path(__file__)),
                   'independent_report_sha256': digest(args.independent_report)})
    with args.report.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'fact_id': result['decision']['fact_id'],
                      'report_sha256': digest(args.report)}))


if __name__ == '__main__':
    main()
