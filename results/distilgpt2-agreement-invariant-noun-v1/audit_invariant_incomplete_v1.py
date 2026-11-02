"""Check the closed partial invariant run without inferring missing results."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from types import ModuleType

CASE_SHA = '50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827'
READER_SHA = '1e2d2a6e867812730f407bfb8f0af93af0b5b3ceaa274250e4e88673a75a3f08'
ACTION_SHA = 'fbf526fd867ed48127c24b3b35004727b1757782e90798d60dcfdec37ac9b2f6'


def checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('external file pin differs: ' + Path(path).name)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('controller', 'cases', 'reader', 'exit-record', 'exit-sha256',
                 'supervisor-sha256', 'events-sha256', 'terminal-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root = Path(args.controller)
    run = root / 'run'
    attempt = run / 'attempt'
    cases = json.loads(checked(args.cases, CASE_SHA))['cases']
    reader_raw = checked(args.reader, READER_SHA)
    reader = ModuleType('_reviewed_invariant_schedule')
    reader.__file__ = str(args.reader)
    exec(compile(reader_raw, str(args.reader), 'exec', dont_inherit=True, optimize=0), reader.__dict__)
    schedule = reader.expected_records(cases, vocabulary=50257)
    exit_doc = json.loads(checked(args.exit_record, args.exit_sha256))
    supervisor = json.loads(checked(root / 'supervisor.json', args.supervisor_sha256))
    event_bytes = checked(attempt / 'events.jsonl', args.events_sha256)
    terminal = json.loads(checked(attempt / 'terminal.json', args.terminal_sha256))
    child = json.loads(checked(root / 'child-custody.json', exit_doc['child_custody_sha256']))
    start = json.loads((attempt / 'started.json').read_bytes())
    if not event_bytes.endswith(b'\n'):
        raise ValueError('partial final JSON line')
    records = [json.loads(line) for line in event_bytes.splitlines()]
    if (len(records) != terminal['events'] or len(records) != 302
            or records[0]['event']['kind'] != 'run_contract'
            or records[0]['event']['schedule_sha256'] != ACTION_SHA
            or records[0]['event']['expected_forwards'] != 612):
        raise ValueError('partial event inventory or contract differs')
    if (terminal['status'] != 'failed' or terminal['storage_accounting_complete'] is not False
            or terminal['event_log_bytes'] != len(event_bytes)
            or terminal['detail']['error_type'] != 'PermissionError'
            or terminal['detail']['forward_evidence_records'] != 301
            or terminal['detail']['forwards_started'] != 302 or terminal['detail']['model_returns'] != 302):
        raise ValueError('failed terminal or reported counts differ')
    if (exit_doc['observer_exit_code'] != 1 or exit_doc['observer_clean'] is not False
            or exit_doc['supervisor_sha256'] != args.supervisor_sha256
            or exit_doc['events_sha256'] != args.events_sha256
            or exit_doc['terminal_sha256'] != args.terminal_sha256
            or supervisor['clean'] is not False or supervisor['supervision_result']['exit_code'] != 1
            or supervisor['child_binding_matches'] is not True
            or supervisor['child_sha256'] != exit_doc['child_custody_sha256']
            or child['run_id'] != exit_doc['run_id'] or child['owner_pid'] != exit_doc['owner_pid']
            or child['prelaunch_sha256'] != exit_doc['prelaunch_sha256']
            or start['external_binding']['prelaunch_sha256'] != exit_doc['prelaunch_sha256']):
        raise ValueError('exit and child observations differ')
    if len(cases) != 68 or len(schedule) != 612:
        raise ValueError('fixed case count differs')

    linked = set()
    linked_sizes = {}
    count_by_case, split_counts, role_counts, action_counts = Counter(), Counter(), Counter(), Counter()
    baselines = {}
    for index, record in enumerate(records[1:]):
        event, expected, case = record['event'], schedule[index], cases[index // 9]
        for key, wanted in expected.items():
            if type(event.get(key)) is not type(wanted) or event.get(key) != wanted:
                raise ValueError(('saved action differs', index, key))
        if (record['index'] != index + 1 or event['kind'] != 'forward'
                or event['input_ids'] != case['token_ids']
                or event['attention_mask'] != [1] * len(case['token_ids'])
                or event['all_sequence_logits_finite'] is not True):
            raise ValueError(('saved forward metadata differs', index))
        count_by_case[event['case_id']] += 1
        split_counts[event['split']] += 1
        role_counts[event['role']] += 1
        action_counts[event['action']] += 1
        for ref in record['tensors'].values():
            if (ref['dtype'] != 'float32-le' or type(ref['shape']) is not list
                    or any(type(n) is not int or n <= 0 for n in ref['shape'])
                    or math.prod(ref['shape']) * 4 != ref['bytes']):
                raise ValueError(('saved tensor reference differs', index))
            digest = ref['sha256']
            if (type(digest) is not str or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)
                    or (digest in linked_sizes and linked_sizes[digest] != ref['bytes'])):
                raise ValueError(('saved tensor digest or repeated size differs', index))
            linked.add(digest)
            linked_sizes[digest] = ref['bytes']
        logit_sha = record['tensors']['last_logits']['sha256']
        if event['action'] == 'clean':
            baselines[event['case_id']] = logit_sha
        if event['action'] in ('noop', 'clean_after') and baselines.get(event['case_id']) != logit_sha:
            raise ValueError(('clean/full-logit control differs', index))

    inventory = {entry['path']: entry for entry in supervisor['run_files']}
    if len(inventory) != len(supervisor['run_files']):
        raise ValueError('duplicate supervisor paths')
    actual_files = {path.relative_to(run).as_posix(): path for path in run.rglob('*') if path.is_file()}
    if set(actual_files) != set(inventory):
        raise ValueError('closed run inventory differs from supervisor observation')
    for relative, path in actual_files.items():
        row = inventory[relative]
        if path.is_symlink() or path.stat().st_size != row['bytes']:
            raise ValueError('run file size or type differs: ' + relative)
        with path.open('rb') as stream:
            h = hashlib.file_digest(stream, 'sha256').hexdigest()
        if h != row['sha256']:
            raise ValueError('run file digest differs: ' + relative)
    tensor_files = {path.stem: path for path in (attempt / 'tensors').glob('*.f32le')}
    if not linked <= set(tensor_files):
        raise ValueError('a linked tensor is missing')
    if any(tensor_files[digest].stat().st_size != size for digest, size in linked_sizes.items()):
        raise ValueError('linked tensor length differs from its reference')
    for digest, path in tensor_files.items():
        if inventory['attempt/tensors/' + path.name]['sha256'] != digest:
            raise ValueError('content-addressed tensor name differs')
    orphans = sorted(set(tensor_files) - linked)
    total_tensor_bytes = sum(path.stat().st_size for path in tensor_files.values())
    complete_cases = [cid for cid, count in count_by_case.items() if count == 9]
    incomplete_cases = {cid: count for cid, count in count_by_case.items() if count != 9}
    next_action = schedule[301]
    result = {
        'scope': ('Closed saved-prefix/receipt consistency and content hashes only; the observer and terminal '
                  'are cooperative reports, and orphan tensors have no independently bound action.'),
        'pins': {'cases_sha256': CASE_SHA, 'reader_sha256': READER_SHA, 'events_sha256': args.events_sha256,
                 'terminal_sha256': args.terminal_sha256, 'supervisor_sha256': args.supervisor_sha256,
                 'exit_sha256': args.exit_sha256},
        'event_records': len(records), 'saved_forward_records': len(records) - 1,
        'saved_forward_counts_by_split': dict(split_counts), 'saved_forward_counts_by_role': dict(role_counts),
        'saved_forward_counts_by_action': dict(action_counts),
        'cases_with_saved_records': len(count_by_case), 'cases_with_nine_saved_actions': len(complete_cases),
        'partial_cases': incomplete_cases, 'next_prescribed_action': next_action,
        'terminal_reported_starts': terminal['detail']['forwards_started'],
        'terminal_reported_normal_path_returns': terminal['detail']['model_returns'],
        'terminal_recorded_forward_evidence': terminal['detail']['forward_evidence_records'],
        'terminal_storage_accounting_complete': terminal['storage_accounting_complete'],
        'linked_tensor_files': len(linked), 'orphan_tensor_files': len(orphans),
        'orphan_tensors': [{'sha256': digest, 'bytes': tensor_files[digest].stat().st_size} for digest in orphans],
        'actual_tensor_files': len(tensor_files), 'actual_tensor_bytes': total_tensor_bytes,
        'terminal_reported_retained_bytes': terminal['retained_tensor_bytes'],
        'supervisor_inventory_files_verified': len(actual_files),
        'saved_prefix_splits': sorted(split_counts), 'heldout_decision': 'not computed; attempt incomplete',
    }
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError('fresh absolute output path required')
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('ascii')
    with output.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short audit write')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'output_sha256': hashlib.sha256(raw).hexdigest(), **result}, sort_keys=True))


if __name__ == '__main__':
    main()
