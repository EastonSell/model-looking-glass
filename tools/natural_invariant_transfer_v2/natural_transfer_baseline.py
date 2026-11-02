"""Validate the closed v2 baseline records and replay the reviewed saved audit."""
import hashlib
import json
from pathlib import Path
import re


CASE_SHA = '188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b'
PROTOCOL_SHA = '58e8b4fb26ca04afe727b76938c5ed3ece58a9dd7f37e5a8a18fd7504ef80deb'
SCHEDULE_SHA = 'f0625d27ee196eeeb1a85a3577238f0cbcad19017fc6596dfe429dcbc60c610d'
REVISION = '2290a62682d06624634c1f46a6ad5be0f47f38aa'
WEIGHTS_SHA = 'e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'
AUDIT_ENTRY_SHA = '8933b8e3058b7acf7cea97ef05011f3ca8133d16c32227a9a8cf3d67717682f8'
AUDIT_READER_SHA = 'a342b41ead06c80a859133d4491b6045867fd19481f014b3005c566865dc116f'
AUDIT_NUMERIC_SHA = '917694d448b796359db0002a630877e5226a2583d273092871892fade7a38ac4'
HEX = re.compile(r'[0-9a-f]{64}\Z')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    if type(expected) is not str or HEX.fullmatch(expected) is None:
        raise ValueError('invalid baseline SHA-256 pin')
    raw = Path(path).read_bytes()
    if digest(raw) != expected:
        raise ValueError('baseline evidence bytes differ')
    return raw


def _zero_int(value):
    return type(value) is int and value == 0


def verify_records(receipt, supervisor, custody, entry, admission, launch, terminal, audit, prior):
    """Verify separate observer and independently computed saved-byte records."""
    if (type(receipt) is not dict or receipt.get('schema') != 'agreement-invariant-v2-owner-exit-observation-v1'
            or not _zero_int(receipt.get('child_exit_code')) or not _zero_int(receipt.get('observer_exit_code'))
            or type(receipt.get('reported_forwards')) is not int or receipt['reported_forwards'] != 612
            or type(receipt.get('event_count')) is not int or receipt['event_count'] != 613
            or receipt.get('storage_accounting_complete') is not True
            or receipt.get('terminal_status') != 'complete_candidate'):
        raise ValueError('baseline observer did not record a complete clean candidate')
    for key in ('started_sha256', 'events_sha256', 'terminal_sha256', 'supervisor_sha256',
                'prelaunch_sha256', 'entry_result_sha256', 'model_admission_sha256', 'child_custody_sha256',
                'source_revision', 'source_tree', 'reservation_sha256'):
        if receipt.get(key) != prior.get(key):
            raise ValueError('baseline observer identity differs: ' + key)
    if receipt.get('operator_exposure_sha256') != prior.get('latest_exposure_sha256'):
        raise ValueError('baseline observer exposure differs')
    if (type(supervisor) is not dict or supervisor.get('schema') != 'agreement-direction-gated-observation-v1'
            or supervisor.get('clean') is not True or supervisor.get('validation_only') is not False
            or supervisor.get('child_binding_matches') is not True or supervisor.get('post_source_check') != 'matched'
            or supervisor.get('post_errors') != [] or supervisor.get('recovery') != []
            or supervisor.get('supervision_error_type') is not None or supervisor.get('run_inventory_error') is not None
            or supervisor.get('source_revision') != prior['source_revision']
            or supervisor.get('source_tree') != prior['source_tree']
            or supervisor.get('prelaunch_sha256') != prior['prelaunch_sha256']
            or supervisor.get('child_sha256') != prior['child_custody_sha256']
            or supervisor.get('child') != custody
            or supervisor.get('completion_statuses') != {'entry_result':'complete_candidate','terminal':'complete_candidate'}):
        raise ValueError('baseline direct supervisor differs')
    if (type(custody) is not dict or custody.get('schema') != 'agreement-direction-supervised-child-v1'
            or custody.get('mode') != 'study' or type(custody.get('owner_pid')) is not int
            or custody['owner_pid'] != receipt.get('owner_pid') or custody.get('run_id') != receipt.get('run_id')
            or custody.get('prelaunch_sha256') != prior['prelaunch_sha256']):
        raise ValueError('baseline child custody differs')
    supervised = supervisor.get('supervision_result', {})
    if (type(supervised) is not dict or supervised.get('clean') is not True
            or not _zero_int(supervised.get('exit_code')) or supervised.get('timed_out') is not False
            or type(supervised.get('pid')) is not int or supervised['pid'] != receipt.get('owner_pid')
            or supervised.get('run_id') != receipt.get('run_id')):
        raise ValueError('baseline direct child outcome differs')
    files = supervisor.get('run_files')
    if type(files) is not list or type(receipt.get('run_files')) is not int or len(files) != receipt['run_files']:
        raise ValueError('baseline supervisor file inventory differs')
    indexed = {}
    for item in files:
        if (type(item) is not dict or type(item.get('path')) is not str or item['path'] in indexed
                or type(item.get('bytes')) is not int or item['bytes'] < 0
                or type(item.get('sha256')) is not str or HEX.fullmatch(item['sha256']) is None):
            raise ValueError('malformed baseline supervisor file inventory')
        indexed[item['path']] = item['sha256']
    for path, key in (('attempt/started.json', 'started_sha256'), ('attempt/events.jsonl', 'events_sha256'),
                      ('attempt/terminal.json', 'terminal_sha256'), ('entry-result.json', 'entry_result_sha256'),
                      ('model-admission.json', 'model_admission_sha256'), ('prelaunch.json', 'prelaunch_sha256')):
        if indexed.get(path) != prior[key]:
            raise ValueError('baseline supervisor file pin differs: ' + path)
    binding = {'prelaunch_sha256': prior['prelaunch_sha256'], 'reservation_sha256': prior['reservation_sha256'],
               'latest_exposure_sha256': prior['latest_exposure_sha256']}
    worker = entry.get('worker', {}) if type(entry) is dict else {}
    if (entry.get('status') != 'complete_candidate' or entry.get('model_disposed') is not True
            or entry.get('binding') != binding or entry.get('owner_pid') != receipt.get('owner_pid')
            or entry.get('owner_run_id') != receipt.get('run_id')
            or type(worker.get('forwards')) is not int or worker['forwards'] != 612
            or worker.get('status') != 'complete_candidate' or worker.get('schedule_sha256') != SCHEDULE_SHA
            or worker.get('model_state_sha256_before_after') != prior['model_state_sha256']):
        raise ValueError('baseline entry result differs')
    checkpoint = admission.get('checkpoint', {}) if type(admission) is dict else {}
    if (admission.get('status') != 'loaded_before_forward' or admission.get('binding') != binding
            or admission.get('case_sha256') != CASE_SHA or checkpoint.get('model_id') != 'distilbert/distilgpt2'
            or checkpoint.get('model_revision') != REVISION or checkpoint.get('weights_sha256') != WEIGHTS_SHA
            or checkpoint.get('asset_sha256', {}).get('tokenizer.json') != TOKENIZER_SHA
            or checkpoint.get('device') != 'cpu' or checkpoint.get('dtype') != 'float32'
            or checkpoint.get('attention_implementation') != 'eager' or checkpoint.get('use_cache') is not False):
        raise ValueError('baseline checkpoint admission differs')
    runtime = admission.get('runtime', {})
    if (type(runtime) is not dict or type(runtime.get('threads')) is not int or runtime['threads'] != 1
            or type(runtime.get('interop_threads')) is not int or runtime['interop_threads'] != 1
            or runtime.get('deterministic_algorithms') is not True
            or runtime.get('float32_matmul_precision') != 'highest'):
        raise ValueError('baseline CPU runtime differs')
    owner = launch.get('owner', {}) if type(launch) is dict else {}
    checkpoint = launch.get('checkpoint', {}) if type(launch) is dict else {}
    if (launch.get('schema') != 'distilgpt2-agreement-invariant-noun-prelaunch-v2'
            or launch.get('cases', {}).get('sha256') != CASE_SHA
            or launch.get('protocol', {}).get('sha256') != PROTOCOL_SHA
            or owner.get('source_revision') != prior['source_revision'] or owner.get('source_tree') != prior['source_tree']
            or owner.get('entry_script_sha256') != prior['owner_entry_sha256']
            or owner.get('executor_sha256') != prior['owner_executor_sha256']
            or checkpoint.get('revision') != REVISION or checkpoint.get('weights_sha256') != WEIGHTS_SHA
            or checkpoint.get('tokenizer_sha256') != TOKENIZER_SHA):
        raise ValueError('baseline source, protocol or checkpoint binding differs')
    journal = audit.get('journal', {}) if type(audit) is dict else {}
    public = audit.get('public_inputs', {}) if type(audit) is dict else {}
    if (journal.get('status') != 'consistent_candidate'
            or type(journal.get('events')) is not int or journal['events'] != 613
            or any(journal.get(key) != prior[key] for key in ('started_sha256','events_sha256','terminal_sha256'))
            or audit.get('dimensions') != {'width':768,'vocabulary':50257}
            or audit.get('schedule', {}).get('sha256') != SCHEDULE_SHA
            or type(audit.get('schedule', {}).get('actions')) is not list
            or len(audit['schedule']['actions']) != 612
            or type(audit.get('cases')) is not list or len(audit['cases']) != 68
            or public.get('case_sha256') != CASE_SHA or public.get('protocol_sha256') != PROTOCOL_SHA
            or public.get('owner_source_revision') != prior['source_revision']
            or public.get('external_binding') != binding
            or type(audit.get('source_tensor_bytes')) is not int
            or audit['source_tensor_bytes'] != terminal.get('retained_tensor_bytes')):
        raise ValueError('independent baseline audit status or identity differs')


def verify_audit_observation(private, prior):
    """Bind the already-completed independent replay; no second replay is run."""
    original = json.loads(pinned(private['prior_audit_observation'], prior['audit_observation_sha256']))
    public = json.loads(pinned(private['prior_audit_observation_public'], prior['audit_observation_public_sha256']))
    entry = Path(private['prior_audit_entry']).resolve(strict=True)
    reader = entry.parents[1] / 'looking_glass' / 'agreement_invariant_successor_science_v2.py'
    numeric = Path(private['prior_audit_numeric']).resolve(strict=True)
    pinned(entry, AUDIT_ENTRY_SHA)
    pinned(reader, AUDIT_READER_SHA)
    pinned(numeric, AUDIT_NUMERIC_SHA)
    pinned(private['prior_audit'], prior['audit_sha256'])
    directions = {'u':'d1ddf7d4ee3103400ca6c23f25536e615135beebb24f2d326ad6fe5d46728be0',
                  'v1':'4a08130c4fd2c3557f9e40b92865c23411d395123c53251a286b63c796b476bb',
                  'v2':'4ca1f10e411e0476bcee76b98bc8515705e61035cfa6e8d71608e15af80f4b02'}
    for name, expected in directions.items():
        pinned(Path(private['prior_directions']) / (name + '.f64le'), expected)
    expected = {'cases':CASE_SHA, 'entry':AUDIT_ENTRY_SHA, 'reader':AUDIT_READER_SHA,
                'numeric':AUDIT_NUMERIC_SHA, 'events':prior['events_sha256'],
                'started':prior['started_sha256'], 'terminal':prior['terminal_sha256'],
                'supervisor':prior['supervisor_sha256'], 'owner_exit':prior['receipt_sha256'], **directions}
    empty_sha = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    for record, schema in ((original, 'agreement-invariant-v2-science-replay-observation-v1'),
                           (public, 'agreement-invariant-v2-science-replay-public-observation-v1')):
        if (type(record) is not dict or record.get('schema') != schema or record.get('status') != 'complete'
                or not _zero_int(record.get('exit_code')) or record.get('timed_out') is not False
                or record.get('stderr_sha256') != empty_sha or record.get('report_sha256') != prior['audit_sha256']
                or record.get('input_sha256_before') != expected or record.get('input_sha256_after') != expected):
            raise ValueError('independent audit exit, report or source/input binding differs')
    if public.get('original_observation_sha256') != prior['audit_observation_sha256']:
        raise ValueError('public audit observation does not bind its original')
    command = original.get('command')
    if type(command) is not list or any(type(x) is not str for x in command):
        raise ValueError('invalid independent audit command')
    for flag in ('-I', '-S', '-B'):
        if flag not in command:
            raise ValueError('independent audit isolation flag missing')
    expected_options = {'--source-revision':prior['source_revision'], '--prelaunch-sha256':prior['prelaunch_sha256'],
                        '--reservation-sha256':prior['reservation_sha256'],
                        '--latest-exposure-sha256':prior['latest_exposure_sha256'],
                        '--events-sha256':prior['events_sha256'],
                        '--start-scope':'public prospective agreement invariant-noun successor v2'}
    for flag, value in expected_options.items():
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command) or command[command.index(flag)+1] != value:
            raise ValueError('independent audit command binding differs')
    for flag in ('--attempt', '--cases', '--directions', '--numeric-source', '--output'):
        if command.count(flag) != 1 or command.index(flag)+1 >= len(command):
            raise ValueError('independent audit input command is incomplete')
    return {'audit_sha256':prior['audit_sha256'], 'audit_observation_sha256':prior['audit_observation_sha256'],
            'public_observation_sha256':prior['audit_observation_public_sha256'],
            'scope':'Cooperative independently observed saved-byte replay; no execution attestation.'}
