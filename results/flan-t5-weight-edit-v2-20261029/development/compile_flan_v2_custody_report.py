"""Compile a model-free, scoped process/custody audit from closed records."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import review_flan_v2_closed_process as process
import review_flan_v2_restoration_records as restoration

def require(value, text):
    if not value:
        raise ValueError(text)

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def blob(repo, revision, path):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    return subprocess.check_output(['git', '--no-replace-objects', 'show', revision + ':' + path],
                                   cwd=repo, env=env, timeout=30)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'scope', 'claim', 'manifest', 'repo', 'outer', 'pilot-receipt', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    for name in ('phase-sha256', 'exit-sha256', 'scope-sha256', 'outer-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    require(not a.output.exists() and not a.output.resolve().is_relative_to(a.attempt.resolve()), 'output is not fresh or overlaps attempt')
    source_files = [Path(__file__), Path(process.__file__), Path(restoration.__file__)]
    source_hashes = {path.name: process.file_digest(path) for path in source_files}
    result = process.process(SimpleNamespace(attempt=a.attempt, scope=a.scope, claim=a.claim, manifest=a.manifest,
                                             phase_sha256=a.phase_sha256, exit_sha256=a.exit_sha256))
    phase = process.read_json((a.attempt/'phase.json').read_bytes())
    restored = restoration.audit(a.attempt, a.phase_sha256, phase['events_sha256'], phase['logits_sha256'])
    scope_raw, outer_raw = a.scope.read_bytes(), a.outer.read_bytes()
    require(hash_bytes(scope_raw) == a.scope_sha256 == result['scope_sha256'] and hash_bytes(outer_raw) == a.outer_sha256,
            'scope or outer observation differs')
    scope, outer = process.read_json(scope_raw), process.read_json(outer_raw)
    require(scope['schema'] == 'flan-weight-edit-scope-v1' and scope['source_revision'] == process.SOURCE
            and scope['manifest_sha256'] == process.MANIFEST and scope['case_sha256'] == process.CASE
            and all(scope[k] is True for k in ('known_public_comparisons_clear', 'no_unreserved_relevant_inputs_reported',
                       'no_additional_relevant_protected_pool_reported', 'source_and_manifest_reviewed')),
            'scope record differs')
    require(outer['schema'] == 'flan-weight-edit-outer-operator-observation-v1' and outer['phase'] == 'development'
            and type(outer['outer_exit_code']) is int and outer['outer_exit_code'] == 0
            and outer['reported_status'] == 'complete_candidate' and outer['reported_phase_sha256'] == a.phase_sha256,
            'outer observation differs')
    launch = datetime.datetime.fromisoformat(outer['outer_launch_observation_utc'])
    end = datetime.datetime.fromisoformat(outer['observation_utc'])
    observed = datetime.datetime.fromisoformat(scope['observed_utc'])
    age, elapsed = (launch-observed).total_seconds(), (end-launch).total_seconds()
    require(0 <= age <= 900 and elapsed >= 0 and type(outer['launch_observed_c_free_bytes']) is int
            and outer['launch_observed_c_free_bytes'] >= process.C_FLOOR and type(outer['launch_observed_e_free_bytes']) is int
            and outer['launch_observed_e_free_bytes'] >= process.E_FLOOR, 'outer chronology or disk observation differs')
    pool = scope['sentiment_pool']
    comparison_path, replay_path = Path(pool['comparison_path']), Path(pool['independent_replay_path'])
    require(process.file_digest(comparison_path, 1024**2) == pool['comparison_sha256']
            and process.file_digest(replay_path, 1024**2) == pool['independent_replay_sha256'], 'custody receipt digest differs')
    comparison, replay = process.read_json(comparison_path.read_bytes()), process.read_json(replay_path.read_bytes())
    card_path = 'results/flan-t5-sentiment-intervention-v1-20261029/custody/pool-card.json'
    selection_path = 'results/flan-t5-sentiment-intervention-v1-20261029/custody/selection-card.json'
    card_raw, selection_raw = blob(a.repo, pool['card_revision'], card_path), blob(a.repo, pool['selection_card_revision'], selection_path)
    require(hash_bytes(card_raw) == pool['card_sha256'] and hash_bytes(selection_raw) == pool['selection_card_sha256'], 'commitment cards differ')
    card, selection = process.read_json(card_raw), process.read_json(selection_raw)
    require(pool['status'] == 'committed' and all(pool[k] is True for k in ('authorized_comparison', 'independent_screen_reviewed',
                'no_variable_word_collision', 'no_prefix_collision')) and comparison['schema'] == 'flan-weight-edit-pool-comparison-v1'
            and comparison['custodian'] == 'schema' and comparison['proposal_sha256'] == process.CASE
            and comparison['card_sha256'] == pool['card_sha256'] and comparison['commitment_sha256'] == pool['commitment_sha256']
            and comparison['selection_card_sha256'] == pool['selection_card_sha256'] and comparison['eos_rule'] == 'strip-one-final-eos'
            and comparison['no_variable_word_collision'] is True and comparison['no_prefix_collision'] is True, 'custody comparison differs')
    require(replay['schema'] == 'flan-t5-sentiment-independent-tokenizer-replay-v1' and replay['status'] == 'consistent'
            and replay['pool_commitment'] == pool['commitment_sha256'] and type(replay['token_rows']) is int and replay['token_rows'] == 96
            and replay['selection_commitment'] == selection['commitment'] and selection['pool_commitment'] == card['commitment'] == pool['commitment_sha256'],
            'independent tokenizer replay differs')
    pilot = scope['pilot_tokenizer_only']
    receipt_raw = a.pilot_receipt.read_bytes()
    require(hash_bytes(receipt_raw) == pilot['comparison_sha256'], 'pilot comparison receipt differs')
    receipt = process.read_json(receipt_raw)
    compressed = blob(a.repo, pilot['public_revision'], pilot['public_path'])
    require(hash_bytes(compressed) == pilot['compressed_sha256'], 'pilot compressed blob differs')
    unpacked = gzip.decompress(compressed)
    require(len(unpacked) <= 8*1024**2 and hash_bytes(unpacked) == pilot['report_sha256'], 'pilot report differs')
    public = process.read_json(unpacked)
    case_raw = blob(a.repo, process.SOURCE, 'docs/FLAN_T5_WEIGHT_EDIT_CASES_V1.json')
    require(hash_bytes(case_raw) == process.CASE, 'case bundle differs')
    cases = process.read_json(case_raw)['cases']
    rows = [t['input_ids'] for c in public['all_candidates'] for t in c['templates']]
    require(len(rows) == 640 and len(cases) == 24, 'public pilot cardinality differs')
    inputs = rows + [c['input_ids'] for c in cases]
    require(all(type(row) is list and row and row[-1] == 1 and 1 not in row[:-1]
                and all(type(token) is int for token in row) for row in inputs), 'public input tokens differ')
    collisions = sum(1 for row in rows for case in cases for x, y in [(row[:-1], case['input_ids'][:-1])]
                     if x[:len(y)] == y or y[:len(x)] == x)
    require(receipt['schema'] == 'flan-weight-edit-pilot-public-prefix-check-v1' and receipt['pilot_sha256'] == hash_bytes(unpacked)
            and receipt['case_sha256'] == process.CASE and receipt['comparisons'] == 15360
            and receipt['collision_count'] == collisions == 0, 'pilot comparison differs')
    require(hash_bytes(a.scope.read_bytes()) == a.scope_sha256 and hash_bytes(a.outer.read_bytes()) == a.outer_sha256
            and process.file_digest(comparison_path) == pool['comparison_sha256']
            and process.file_digest(replay_path) == pool['independent_replay_sha256']
            and process.file_digest(a.pilot_receipt) == pilot['comparison_sha256']
            and all(process.file_digest(path) == source_hashes[path.name] for path in source_files), 'record changed during review')
    result.pop('recorded_scientific_status', None)
    report = {'schema': 'independent-flan-v2-development-custody-audit-v1', 'status': 'recorded_consistency_passed',
        'process': result, 'restoration': restored, 'outer_observation_sha256': a.outer_sha256,
        'outer_operator_exit_code': outer['outer_exit_code'], 'scope_age_at_reported_launch_seconds': age,
        'outer_elapsed_seconds': elapsed, 'outer_launch_free_bytes': {'c': outer['launch_observed_c_free_bytes'], 'e': outer['launch_observed_e_free_bytes']},
        'custody': {'pool_card_sha256': pool['card_sha256'], 'selection_card_sha256': pool['selection_card_sha256'],
                    'comparison_sha256': pool['comparison_sha256'], 'independent_selector_replay_sha256': pool['independent_replay_sha256'],
                    'selector_token_rows_recorded': replay['token_rows'], 'pilot_comparison_sha256': pilot['comparison_sha256'],
                    'public_pilot_rows': len(rows), 'case_rows': len(cases), 'pairs_checked': len(rows)*len(cases),
                    'bidirectional_prefix_collisions': collisions}, 'reviewer_source_sha256': source_hashes,
        'limits': ['Cooperative recorded process and custodian observations; no authenticated execution or whole-tree teardown.',
                   'Disk observations are sampled points and not reservations or proof against concurrent writes.',
                   'State and row hashes are model-authored observations; full model-state bytes were not retained.',
                   'Pilot cross-check compares saved public token IDs; it does not retokenize or access protected pool rows.',
                   'This is a process/custody check; scientific thresholds require the separate saved-logit replay.']}
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with a.output.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'short report write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'report_sha256': hash_bytes(raw), 'bytes': len(raw)}))

if __name__ == '__main__':
    main()
