"""Execute a single prebound natural-transfer phase on an offline CPU model.

The independently checked prior receipt and the public observation are
cooperative records, not proof of all execution or private exposure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


CASE_SHA = '188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b'
MAPPING_SHA = 'bd1dd08ff0f7042f71be4ed88f22554054dc90cb57d68c3bf647c04e862ee550'
PROTOCOL_SHA = 'ac2d8bc17b558a95c36848fa440adab951cf6ad1076b0b259e507b097c8ad26f'
ADDENDUM_SHA = '0b3eb946cef7bf0185b3da2bcc139de6fc73cdbcf4ae03c33baaaefd5ec34e4d'
SCHEDULE_SHA = 'f0625d27ee196eeeb1a85a3577238f0cbcad19017fc6596dfe429dcbc60c610d'
REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
WEIGHTS_SHA = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
HEX = re.compile(r'[0-9a-f]{64}\Z')
STUDY = ('natural_transfer_direct.py', 'natural_transfer_phase.py', 'natural_transfer_numeric.py',
         'natural_transfer_prior.py', 'natural_transfer_executor.py', 'natural_transfer_analyze.py',
         'natural_transfer_freeze.py', 'natural_transfer_currentness.py', 'natural_transfer_baseline.py')
DEPENDENCIES = tuple('looking_glass/' + x + '.py' for x in (
    '__init__', 'pretrained', 'agreement_direction_study_v1', 'agreement_substitution_executor_v1',
    'agreement_journal', 'agreement_patch_runner', 'agreement_numeric', 'agreement_metrics',
    'bert_matched_numeric', 'distilgpt2_intervention_dev', 'intervention_plan', 'interventions',
    'model_ownership', 'transformer_trace', 'trace_schema', 'probes'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    if type(expected) is not str or HEX.fullmatch(expected) is None:
        raise ValueError('invalid external SHA-256')
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise ValueError('pinned bytes differ: ' + Path(path).name)
    return raw


def path_hash(path):
    return sha(os.path.normcase(str(path.resolve(strict=True))).encode())


def verify_sources(root, names, pins):
    if type(pins) is not dict or set(pins) != set(names):
        raise ValueError('source inventory differs')
    for name in names:
        checked(root / name, pins[name])


def verify_reservations(inventory, cases):
    """Require exact own input and continuation reservations; status may have advanced."""
    records = inventory.get('records')
    if type(records) is not list:
        raise ValueError('invalid exposure records')
    indexed = {}
    for record in records:
        if type(record) is not dict or type(record.get('record_id')) is not str:
            raise ValueError('malformed exposure record')
        if record['record_id'] in indexed:
            raise ValueError('duplicate exposure record')
        indexed[record['record_id']] = record
    if type(cases) is not list or len(cases) != 68:
        raise ValueError('unexpected fixed case count')
    for case in cases:
        record = indexed.get('planned-' + case['case_id'])
        if (type(record) is not dict or type(record.get('token_ids')) is not list
                or any(type(x) is not int for x in record['token_ids'])
                or record['token_ids'] != case['token_ids']
                or record.get('public_text') != case['text'] or record.get('case_role') != case['role']
                or type(record.get('candidate_token_ids')) is not list
                or any(type(x) is not int for x in record['candidate_token_ids'])
                or record['candidate_token_ids'] != [318, 389, 373, 547, 262, 257, 287, 319]
                or any(name in record for name in ('attention_mask', 'target_token_id', 'candidate_token_groups',
                                                   'scored_prefix_token_ids', 'scored_candidate_token_ids'))
                or record.get('category') not in ('planned', 'observed-reported', 'observed-saved')):
            raise ValueError('own reserved input or candidates differ')
    return {'own_cases': 68, 'scope': 'public inventory only; unknown and protected exposure unverified'}


def verify_evaluation_freeze(*, private, pre, pre_sha, prior, observed_sha, width=768, vocabulary=50257):
    """Read public-observation record and recompute discovery before model import."""
    observation = json.loads(checked(private['freeze_observation'], observed_sha))
    if observation.get('schema') != 'natural-invariant-transfer-freeze-observation-v2':
        raise ValueError('unexpected freeze observation')
    receipt_raw = checked(private['discovery_receipt'], observation['discovery_receipt_sha256'])
    receipt = json.loads(receipt_raw)
    if (receipt.get('schema') != 'natural-invariant-transfer-direct-receipt-v2'
            or receipt.get('phase') != 'discovery' or receipt.get('accepted_complete_phase') is not True
            or receipt.get('pre_run_sha256') != pre_sha
            or receipt.get('prior_receipt_sha256') != pre['prior']['receipt_sha256']
            or receipt.get('prior_audit_sha256') != pre['prior']['audit_sha256']
            or receipt.get('prior_audit_observation_sha256') != pre['prior']['audit_observation_sha256']
            or receipt.get('prior_audit_observation_public_sha256') != pre['prior']['audit_observation_public_sha256']
            or receipt.get('latest_exposure_sha256') != pre['latest_exposure_sha256']
            or receipt.get('run_files', {}).get('events.jsonl', {}).get('sha256') != observation['discovery_events_sha256']
            or receipt.get('run_files', {}).get('terminal.json', {}).get('sha256') != observation['discovery_terminal_sha256']):
        raise ValueError('discovery receipt differs from observation')
    terminal = json.loads(checked(Path(private['discovery_dir']) / 'terminal.json',
                                  observation['discovery_terminal_sha256']))
    import natural_transfer_direct
    if not natural_transfer_direct.assess_receipt(receipt, terminal):
        raise ValueError('discovery process or terminal is not a complete candidate')
    import natural_transfer_freeze
    binding = {'pre_run_sha256': pre_sha, 'mapping_sha256': MAPPING_SHA,
               'prior_receipt_sha256': pre['prior']['receipt_sha256'],
               'latest_exposure_sha256': pre['latest_exposure_sha256']}
    result = natural_transfer_freeze.verify_before_evaluation(
        mapping_path=private['mapping'], expected_mapping_sha256=MAPPING_SHA, prior=prior,
        prior_events_sha256=pre['prior']['events_sha256'], discovery_directory=private['discovery_dir'],
        discovery_events_sha256=observation['discovery_events_sha256'],
        discovery_terminal_sha256=observation['discovery_terminal_sha256'], discovery_binding=binding,
        freeze_path=private['discovery_freeze'], expected_freeze_sha256=observation['freeze_sha256'],
        observed_public_commit=observation['public_commit'], observed_public_blob=observation['public_blob'],
        width=width, vocabulary=vocabulary)
    if result['freeze_sha256'] != observation['freeze_sha256']:
        raise ValueError('verified freeze differs')
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('discovery', 'evaluation'), required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--pre-run-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--freeze-observation-sha256')
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    pre = json.loads(checked(attempt / 'pre-run.json', args.pre_run_sha256))
    private = json.loads((attempt / 'private-launch.json').read_bytes())
    if (pre.get('schema') != 'natural-invariant-transfer-pre-run-v2'
            or pre.get('case_sha256') != CASE_SHA or pre.get('mapping_sha256') != MAPPING_SHA
            or pre.get('protocol_sha256') != PROTOCOL_SHA or pre.get('addendum_sha256') != ADDENDUM_SHA
            or pre.get('prior', {}).get('schedule_sha256') != SCHEDULE_SHA):
        raise ValueError('pre-run fixed identities differ')
    prefix = Path(sys.pycache_prefix) if sys.pycache_prefix else None
    if (not sys.dont_write_bytecode or prefix is None or not prefix.is_absolute() or prefix.is_symlink()
            or not prefix.is_dir() or any(prefix.iterdir())
            or path_hash(prefix) != pre['pycache_prefix_sha256'][args.phase]):
        raise ValueError('fresh pinned bytecode prefix required')
    source = Path(private['source_root']).resolve(strict=True)
    study = Path(private['study_root']).resolve(strict=True)
    cache = Path(private['cache_dir']).resolve(strict=True)
    stage = Path(private['stage_dir'])
    if (Path(__file__).resolve() != study / 'natural_transfer_phase.py'
            or not stage.is_absolute() or stage.is_symlink() or not stage.is_dir() or any(stage.iterdir())
            or path_hash(stage) != pre['stage_dir_sha256'] or args.output_dir.exists()
            or args.output_dir.parent.resolve(strict=True) != attempt):
        raise ValueError('source, stage or one-shot output differs')
    if any(os.path.splitdrive(str(p.resolve()))[0].upper() != 'E:' for p in (prefix, stage, attempt)):
        raise ValueError('attempt, prefix and stage must be on E:')
    verify_sources(study, STUDY, pre['study_source_sha256'])
    verify_sources(source, DEPENDENCIES, pre['runtime_source_sha256'])
    case_bytes = checked(private['cases'], CASE_SHA)
    cases = json.loads(case_bytes)['cases']
    mapping = json.loads(checked(private['mapping'], MAPPING_SHA))
    checked(private['protocol'], PROTOCOL_SHA)
    checked(private['addendum'], ADDENDUM_SHA)
    prior_info = pre['prior']
    receipt = json.loads(checked(private['prior_receipt'], prior_info['receipt_sha256']))
    supervisor = json.loads(checked(private['prior_supervisor'], prior_info['supervisor_sha256']))
    custody = json.loads(checked(private['prior_child_custody'], prior_info['child_custody_sha256']))
    entry = json.loads(checked(private['prior_entry_result'], prior_info['entry_result_sha256']))
    model_admission = json.loads(checked(private['prior_model_admission'], prior_info['model_admission_sha256']))
    launch = json.loads(checked(private['prior_prelaunch'], prior_info['prelaunch_sha256']))
    terminal = json.loads(checked(Path(private['prior_dir']) / 'terminal.json', prior_info['terminal_sha256']))
    audit = json.loads(checked(private['prior_audit'], prior_info['audit_sha256']))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(study))
    import natural_transfer_baseline
    natural_transfer_baseline.verify_records(receipt, supervisor, custody, entry, model_admission,
                                             launch, terminal, audit, prior_info)
    import natural_transfer_prior
    prior = natural_transfer_prior.load_prior(private['prior_dir'], cases, mapping,
        events_sha256=prior_info['events_sha256'], started_sha256=prior_info['started_sha256'],
        terminal_sha256=prior_info['terminal_sha256'], expected_schedule_sha256=SCHEDULE_SHA)
    terminal = json.loads(checked(Path(private['prior_dir']) / 'terminal.json', prior_info['terminal_sha256']))
    if terminal['detail']['model_state_sha256_before_after'] != prior_info['model_state_sha256']:
        raise ValueError('prior model state differs')
    observation = None
    if args.phase == 'evaluation':
        if not args.freeze_observation_sha256:
            raise ValueError('evaluation requires an independently read-back freeze observation')
        observation = verify_evaluation_freeze(private=private, pre=pre, pre_sha=args.pre_run_sha256,
                                               prior=prior, observed_sha=args.freeze_observation_sha256)
        exposure_sha = observation['latest_exposure_sha256']
    else:
        if args.freeze_observation_sha256 is not None:
            raise ValueError('discovery cannot receive an evaluation freeze')
        exposure_sha = pre['latest_exposure_sha256']
    inventory_bytes = checked(private['inventories'][args.phase], exposure_sha)
    inventory = json.loads(inventory_bytes)
    admission = verify_reservations(inventory, cases)
    card = observation if observation is not None else pre
    report_bytes = checked(private['currentness_reports'][args.phase], card['currentness_report_sha256'])
    import natural_transfer_currentness
    report = natural_transfer_currentness.compare(case_bytes, inventory_bytes)
    expected_report = (json.dumps(report, sort_keys=True, indent=2) + '\n').encode()
    if report_bytes != expected_report or report['relations'] != 0:
        raise ValueError('latest public inventory comparison differs or has a relation')
    support = json.loads(checked(private['support_report'],
                                 '618bc2fa245572e410aa1f32cb0c984f6c92b66a44f2b20cb71bdd21511a637f'))
    expected_zero = {'case_prefix_of_reference': 0, 'reference_prefix_of_case': 0}
    counts = support.get('counts', {}).get('invariant', {})
    if (support.get('schema') != 'natural-public-cases-crosscheck-v2'
            or support.get('case_sha256', {}).get('invariant') != CASE_SHA
            or support.get('candidate_sequences', {}).get('invariant') != 612
            or set(counts) != {'original_public_support', 'successor_public_pool',
                               'six_short_service_rows', 'public_timing_input'}
            or any(value != expected_zero for value in counts.values())):
        raise ValueError('fixed public support comparison differs')
    pending = json.loads(checked(private['pending_observations'][args.phase], card['pending_observation_sha256']))
    if (pending.get('schema') != 'natural-invariant-transfer-pending-observation-v1'
            or pending.get('case_sha256') != CASE_SHA or pending.get('inventory_sha256') != exposure_sha
            or type(pending.get('bundles')) is not list):
        raise ValueError('pending public input observation differs')
    pending_paths = private['pending_reports'][args.phase]
    bundle_paths = private['pending_bundles'][args.phase]
    if (type(pending_paths) is not list or type(bundle_paths) is not list
            or len(pending_paths) != len(pending['bundles']) or len(bundle_paths) != len(pending['bundles'])):
        raise ValueError('pending report inventory differs')
    for entry, report_path, bundle_path in zip(pending['bundles'], pending_paths, bundle_paths):
        if (type(entry) is not dict or type(entry.get('bundle_sha256')) is not str
                or HEX.fullmatch(entry['bundle_sha256']) is None
                or type(entry.get('relations')) is not int or entry['relations'] != 0
                or type(entry.get('bidirectional_pairs')) is not int or entry['bidirectional_pairs'] <= 0):
            raise ValueError('pending public input comparison is not clear')
        bundle_bytes = checked(bundle_path, entry['bundle_sha256'])
        pending_report = natural_transfer_currentness.compare_pending(case_bytes, bundle_bytes)
        expected = (json.dumps(pending_report, sort_keys=True, indent=2) + '\n').encode()
        if (checked(report_path, entry['report_sha256']) != expected or pending_report['relations'] != 0
                or pending_report['bidirectional_pairs'] != entry['bidirectional_pairs']):
            raise ValueError('pending public comparison does not reproduce')
    admission.update({'foreign_sequences': report['foreign_sequences'],
                      'bidirectional_pairs': report['bidirectional_pairs'],
                      'foreign_relations': report['relations'],
                      'pending_observation_scope': 'operator-reported; independently checked before launch'})
    def custody_check():
        if (shutil.disk_usage(attempt).free < 1024 ** 3 + 4 * 1024 ** 2
                or any(shutil.disk_usage(p).free < 256 * 1024 ** 2 for p in (source, study, cache))):
            raise RuntimeError('resource floor breached before a forward')
    if shutil.disk_usage(attempt).free < 2 * 1024 ** 3:
        raise ValueError('insufficient initial E: reserve')
    custody_check()
    natural_transfer_baseline.verify_audit_observation(private, prior_info)
    # No project or model import above this line loads Torch. In particular,
    # evaluation reaches here only after recomputing the public freeze.
    import torch
    import looking_glass.pretrained as pretrained
    import natural_transfer_executor as execution
    from looking_glass.agreement_journal import AgreementJournal
    for name in STUDY:
        mod = sys.modules.get(Path(name).stem)
        if mod is not None and Path(mod.__file__).resolve() != study / name:
            raise ValueError('study import path differs')
    for name in DEPENDENCIES:
        key = 'looking_glass' if name.endswith('/__init__.py') else 'looking_glass.' + Path(name).stem
        mod = sys.modules.get(key)
        if mod is not None and Path(mod.__file__).resolve() != source / name:
            raise ValueError('runtime import path differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    if not torch.set_flush_denormal(False):
        raise RuntimeError('unable to disable subnormal flushing')
    torch.manual_seed(0)
    loaded = pretrained.load_distilgpt2(cache_dir=cache, allow_download=False, stage_dir=stage)
    if (any(stage.iterdir()) or loaded.metadata['model_revision'] != REVISION
            or loaded.metadata['weights_sha256'] != WEIGHTS_SHA
            or loaded.metadata['asset_sha256']['tokenizer.json'] != TOKENIZER_SHA
            or loaded.metadata['torch_version'] != model_admission['checkpoint']['torch_version']
            or loaded.metadata['transformers_version'] != model_admission['checkpoint']['transformers_version']
            or loaded.metadata['device'] != 'cpu' or loaded.metadata['dtype'] != 'float32'
            or loaded.metadata['attention_implementation'] != 'eager' or loaded.metadata['use_cache'] is not False
            or loaded.model.config.n_embd != 768 or loaded.model.config.vocab_size != 50257):
        raise ValueError('checkpoint identity or stage differs')
    for case in cases:
        if loaded.tokenizer.encode(case['text'], add_special_tokens=False) != case['token_ids']:
            raise ValueError('loaded tokenizer differs from frozen case')
    binding = {'pre_run_sha256': args.pre_run_sha256, 'mapping_sha256': MAPPING_SHA,
               'prior_receipt_sha256': prior_info['receipt_sha256'], 'latest_exposure_sha256': exposure_sha}
    if observation is not None:
        binding['freeze_sha256'] = observation['freeze_sha256']
    journal = AgreementJournal(args.output_dir, {'phase': args.phase, 'binding': binding},
                               max_total_bytes=execution.RAW_LIMIT, max_forward_bytes=4 * 1024 ** 2,
                               max_events=200, max_event_log_bytes=execution.METADATA_LIMIT)
    result = execution.run_phase(loaded.model, journal, mapping, prior, args.phase,
                                 prior_model_sha256=prior_info['model_state_sha256'],
                                 binding=binding, custody_check=custody_check)
    verify_sources(study, STUDY, pre['study_source_sha256'])
    verify_sources(source, DEPENDENCIES, pre['runtime_source_sha256'])
    if result['forwards'] != 96 or any(prefix.iterdir()) or any(stage.iterdir()):
        raise RuntimeError('phase count or runtime directory differs')
    print(json.dumps({'status': 'complete_candidate', 'phase': args.phase, 'forwards': 96,
                      'admission': admission}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
