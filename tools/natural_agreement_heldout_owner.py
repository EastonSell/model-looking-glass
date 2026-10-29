"""Supervise one committed natural-agreement owner under protected sources.

The custodian prepares and publishes the split and operation binding first.
This private process never allocates a split, and its structural receipt is
not a numerical or scientific validation of the held-out results.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import threading
import types
import urllib.parse
import urllib.request


SOURCE_FILES = {
    'custody': 'tools/natural_heldout_custody.py',
    'deviation': 'docs/NATURAL_AGREEMENT_HELDOUT_DEVIATION.md',
    'child': 'tools/natural_agreement_heldout_child.py',
    'owner': 'tools/natural_agreement_heldout_owner.py',
    'supervisor': 'looking_glass/natural_heldout_supervisor.py',
    'ledger': 'looking_glass/agreement_phase_ledger.py',
    'auditor': 'tools/audit_natural_agreement_heldout.py',
}
_QUARANTINED = []
_QUARANTINE_EVENT = threading.Event()


class OwnerError(RuntimeError):
    pass


def _read(path, limit):
    path = Path(path)
    if not path.is_absolute():
        raise OwnerError('expected an absolute path')
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or info.st_size > limit):
        raise OwnerError('file is linked, nonregular or oversized')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) != info.st_size or len(data) > limit or path.stat().st_size != info.st_size:
        raise OwnerError('file changed while reading')
    return data


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise OwnerError('duplicate JSON key')
        out[key] = value
    return out


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_object)
    except (ValueError, TypeError, RecursionError) as error:
        raise OwnerError('invalid JSON') from error


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _same_json(left, right):
    """Compare JSON values with exact scalar types, including int versus float."""
    try:
        options = {'sort_keys': True, 'ensure_ascii': True, 'allow_nan': False, 'separators': (',', ':')}
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError, RecursionError):
        return False


def _bootstrap_child(config_path, operational_root):
    """Authenticate the published operation and compile exactly its child bytes."""
    config = _json(_read(config_path, 32768))
    try:
        operation = _read(config['public_files']['operational_binding'], 2 * 1024 * 1024)
        url = config['public_urls']['operational_binding_url']
        parsed = urllib.parse.urlsplit(url)
        expected = ('/EastonSell/model-looking-glass/[0-9a-f]{40}/results/'
                    'distilgpt2-natural-agreement-v1-20261028/heldout-operational-binding-v1.json')
        if (parsed.scheme != 'https' or parsed.netloc != 'raw.githubusercontent.com'
                or parsed.query or parsed.fragment or re.fullmatch(expected, parsed.path) is None):
            raise OwnerError('operation URL is not immutable official source')
        with urllib.request.urlopen(url, timeout=20) as response:
            if response.geturl() != url or response.read(2 * 1024 * 1024 + 1) != operation:
                raise OwnerError('published operation bytes differ')
        sources = _json(operation)['sources']
        if type(sources) is not dict or set(sources) != set(SOURCE_FILES):
            raise OwnerError('operational source list differs')
        source_bytes = {}
        for name, relative in SOURCE_FILES.items():
            entry = sources[name]
            if (type(entry) is not dict or set(entry) != {'url', 'sha256'}
                    or type(entry['sha256']) is not str
                    or re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) is None):
                raise OwnerError('operational source identity is invalid')
            source_bytes[name] = _read(operational_root / relative, 8 * 1024 * 1024)
            if _digest(source_bytes[name]) != entry['sha256']:
                raise OwnerError('operational source bytes differ')
        name = '_mlg_bound_heldout_child'
        if name in sys.modules:
            raise OwnerError('child bootstrap is already imported')
        module = types.ModuleType(name)
        module.__file__ = str(operational_root / SOURCE_FILES['child'])
        sys.modules[name] = module
        try:
            exec(compile(source_bytes['child'], module.__file__, 'exec', dont_inherit=True), module.__dict__)
        except BaseException:
            del sys.modules[name]
            raise
        return module, operation, source_bytes
    except (KeyError, TypeError, OSError, ValueError, RecursionError) as error:
        raise OwnerError('cannot authenticate operational source') from error


def _expected_cases(record):
    families = record.get('families') if type(record) is dict else None
    if type(families) is not list or len(families) != 4:
        raise OwnerError('committed family count differs')
    cases = []
    for index, family in enumerate(families):
        if (type(family) is not dict or type(family.get('family_index')) is not int
                or family['family_index'] != index or type(family.get('cases')) is not list
                or len(family['cases']) != 4):
            raise OwnerError('committed family is malformed')
        for item_index, item in enumerate(family['cases']):
            if (type(item) is not dict or type(item.get('input_ids')) is not list
                    or type(item.get('attention_mask')) is not list
                    or type(item.get('subject_index')) is not int):
                raise OwnerError('committed token row is malformed')
            cases.append({'case_id': f'heldout-{index}-{item_index}', 'quartet': f'heldout-{index}',
                          'lemma': f'lemma-{item_index // 2}',
                          'number': 'singular' if item_index % 2 == 0 else 'plural',
                          'input_ids': item['input_ids'], 'attention_mask': item['attention_mask'],
                          'subject_index': item['subject_index']})
    return cases


def _declared_order(cases):
    sites = [f'{layer}:{component}:last' for layer in range(1, 5) for component in ('attn', 'mlp')]
    order = [('run_contract', None, None, None, None)]
    groups = {}
    for case in cases:
        groups.setdefault(case['quartet'], []).append(case['case_id'])
    for quartet, members in groups.items():
        for kind in ('clean_capture', 'clean_repeat', 'null'):
            order.extend((kind, quartet, case, None, None) for case in members)
        order.extend(('patch', quartet, target, site, donor)
                     for site in sites for target in members for donor in members)
        order.extend((kind, quartet, target, None, None) for target in members
                     for kind in ('delivery', 'subject_position'))
        order.extend(('final_clean', quartet, case, None, None) for case in members)
    order.extend((kind, None, None, site, None)
                 for kind in ('present_site_metrics', 'secondary_past_site_metrics') for site in sites)
    order.append(('summary', None, None, None, None))
    return order


def _observed_order(events):
    out = []
    for event in events:
        kind = event.get('kind')
        if kind in ('clean_capture', 'clean_repeat', 'final_clean'):
            out.append((kind, event.get('quartet'), event.get('case'), None, None))
        elif kind in ('null', 'delivery', 'subject_position'):
            out.append((kind, event.get('quartet'), event.get('target'), None, None))
        elif kind == 'patch':
            site = event.get('site')
            out.append((kind, event.get('quartet'), event.get('target'),
                        site.get('id') if type(site) is dict else None, event.get('donor')))
        elif kind in ('present_site_metrics', 'secondary_past_site_metrics'):
            out.append((kind, None, None, event.get('site'), None))
        else:
            out.append((kind, None, None, None, None))
    return out


def audit_saved(directory, exit_code, reaped, *, control, binding, binding_bytes,
                prediction_bytes, aggregate_bytes, inventory_bytes, record_bytes, nonce,
                commitment_bytes, commitment_domain, expected_model_sha256):
    """Bind the saved structural inventory to the private committed rows."""
    if type(exit_code) is not int or exit_code != 0 or reaped is not True:
        raise OwnerError('owner exit is not confirmed')
    if type(nonce) is not bytes or len(nonce) != 32:
        raise OwnerError('private commitment nonce is malformed')
    record, commitment, prediction = _json(record_bytes), _json(commitment_bytes), _json(prediction_bytes)
    heldout_sha = _digest(commitment_domain + nonce + b'\x00' + record_bytes)
    if (commitment.get('heldout_commitment') != heldout_sha
            or record.get('prediction_freeze_sha256') != _digest(aggregate_bytes)
            or record.get('exposure_inventory_sha256') != _digest(inventory_bytes)):
        raise OwnerError('private record differs from its published commitment')
    inventory = control.audit_storage(directory)
    start = {'schema': 'mlg-natural-cpu-attempt-v1', 'binding_sha256': _digest(binding_bytes), 'binding': binding}
    control.check_start(inventory['_started'], start)
    records, terminal = inventory['_records'], inventory['_terminal']
    if (len(records) != 626 or not control._same_json(terminal.get('detail'), {'forwards': 608})
            or type(terminal.get('event_log_bytes')) is not int or terminal['event_log_bytes'] > 48 * 1024 * 1024
            or type(inventory['retained_tensor_bytes']) is not int
            or inventory['retained_tensor_bytes'] > 160 * 1024 * 1024):
        raise OwnerError('held-out inventory exceeds its bound or lacks 608 forwards')
    events = [row['event'] for row in records]
    expected_counts = {'run_contract': 1, 'clean_capture': 16, 'clean_repeat': 16, 'null': 16,
                       'patch': 512, 'delivery': 16, 'subject_position': 16, 'final_clean': 16,
                       'present_site_metrics': 8, 'secondary_past_site_metrics': 8, 'summary': 1}
    if dict(Counter(event.get('kind') for event in events)) != expected_counts:
        raise OwnerError('held-out event counts differ')
    contract, summary = events[0], events[-1]
    expected = {'kind': 'run_contract', 'phase': 'heldout', 'prediction_sha256': _digest(prediction_bytes),
                'prediction_freeze_sha256': _digest(aggregate_bytes), 'heldout_commitment': heldout_sha,
                'exposure_inventory_sha256': _digest(inventory_bytes),
                'discovery_event_sha256': prediction['discovery_event_sha256'],
                'independent_audit_sha256': prediction['independent_audit_sha256']}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise OwnerError('held-out contract identity differs')
    sites = [f'{layer}:{component}:last' for layer in range(1, 5) for component in ('attn', 'mlp')]
    sites.extend(('5:block:last', '5:block:subject'))
    if (type(contract.get('cpu_rng_seed')) is not int or contract['cpu_rng_seed'] != 20261028
            or type(contract.get('cpu_threads')) is not int or contract['cpu_threads'] != 1
            or contract.get('deterministic_algorithms') is not True
            or not control._same_json(contract.get('sites'), [control._site_metadata(site) for site in sites])):
        raise OwnerError('held-out execution contract differs')
    expected_cases = _expected_cases(record)
    if not control._same_json(contract.get('cases'), expected_cases):
        raise OwnerError('held-out contract differs from committed token rows')
    if _observed_order(events) != _declared_order(expected_cases):
        raise OwnerError('held-out event sequence differs from the declared design')
    control.check_execution_configuration(contract, binding['runtime'])
    result = summary.get('result') if type(summary) is dict else None
    controls = {'null': 16, 'self': 128, 'delivery': 16, 'subject_position': 16,
                'repeat': 16, 'final_clean': 16}
    if (summary.get('kind') != 'summary' or type(result) is not dict or result.get('status') != 'complete'
            or type(result.get('forwards')) is not int or result['forwards'] != 608
            or not control._same_json(result.get('controls'), controls)
            or result.get('model_state_unchanged') is not True
            or result.get('model_state_sha256') != expected_model_sha256
            or type(result.get('heldout_evaluation')) is not dict):
        raise OwnerError('held-out summary is incomplete')
    evaluation, selection = result['heldout_evaluation'], prediction['selection']
    for key in ('selected_site', 'earliest_site', 'random_site', 'magnitude_site'):
        if evaluation.get(key) != selection.get(key):
            raise OwnerError('held-out evaluation differs from the frozen selection')
    for key in ('prediction_sha256', 'prediction_freeze_sha256', 'heldout_commitment',
                'exposure_inventory_sha256', 'discovery_event_sha256', 'independent_audit_sha256'):
        if evaluation.get(key) != expected[key]:
            raise OwnerError('held-out evaluation identity differs')
    for key in ('_records', '_started', '_terminal'):
        inventory.pop(key)
    return {'status': 'evidence_complete', 'inventory': inventory,
            'scientific_validation': 'pending independent raw numerical and landing audit'}


def under_source_leases(leases, precheck, launch, postcheck, *, retained=None):
    """Retain protection when process or lease cleanup has an unknown outcome."""
    if _QUARANTINED:
        raise OwnerError('source or process custody is unverified; stop')
    custody = {'leases': [], 'retained': retained}
    _QUARANTINED.append(custody)  # Root the fixed custody record before any resource exists.
    acquired, launched, verified = custody['leases'], False, False
    try:
        for lease in leases:
            acquired.append(lease)
            lease.__enter__()
        precheck()
        launched = True
        verdict = launch(acquired)
        verified = type(verdict) is dict and verdict.get('cleanup_verified') is True
        if not verified:
            raise OwnerError('owner cleanup is unverified; stop')
        return postcheck(verdict)
    finally:
        primary = sys.exc_info()[1]
        if not launched or verified:
            while acquired:
                lease = acquired[-1]
                try:
                    lease.close()
                except BaseException as cleanup_error:
                    if primary is not None and not isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                        raise primary from cleanup_error
                    raise
                acquired.pop()
            _QUARANTINED.remove(custody)


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode('utf-8')
    if len(raw) > 65536:
        raise OwnerError('owner terminal exceeds its reserve')
    with Path(path).open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OwnerError('short owner terminal write')
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def _validation_root(phase, operation_bytes):
    return Path(phase).parent / ('natural-heldout-validation-' + _digest(operation_bytes))


def _side_roots(child, config, config_path, validation_root):
    phase = Path(config['phase_dir'])
    claim = _json(child._read(phase / 'claim.json', 8192))
    paths = [Path(config['preparation_supervisor']), Path(config_path),
             Path(claim['preparation_config_path']), Path(validation_root)]
    result = []
    for path in paths:
        if path not in result:
            result.append(path)
    return tuple(result)


def _verify_public_commitment(child, config, operation_bytes):
    child._check_public_operation(config, operation_bytes, require_commitment=True)


def _validation_account(phase, *, supervisor_record, terminal_record=False, parent_record=False, reserve=0):
    """Bound the separate rehearsal; it may contain only its fixed metadata."""
    phase = Path(phase)
    info = phase.lstat()
    root_entries = {'owner', 'validation-config.json'}
    if parent_record:
        root_entries.add('parent-observation.json')
    if (not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or {path.name for path in phase.iterdir()} != root_entries):
        raise OwnerError('validation requires a separate unclaimed directory')
    directory = phase / 'owner'
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise OwnerError('validation output is not an ordinary directory')
    expected = {'stdout.txt', 'stderr.txt', 'runtime-overrides.json',
                'checkpoint-staging-before.json', 'checkpoint-staging-after.json'}
    if supervisor_record:
        expected.add('supervisor.json')
    if terminal_record:
        expected.add('owner-terminal.json')
    if {path.name for path in directory.iterdir()} != expected:
        raise OwnerError('validation output contains an unexpected artifact')
    total = len(_read(phase / 'validation-config.json', 32768))
    if parent_record:
        total += len(_read(phase / 'parent-observation.json', 65536))
    total += sum(len(_read(directory / name, 2 * 1024 * 1024)) for name in expected)
    if type(reserve) is not int or reserve < 0 or total + reserve > 16 * 1024 * 1024:
        raise OwnerError('validation metadata exceeds its bound')
    return {'apparent_bytes': total, 'tensor_bytes': 0, 'phase_claim': False}


def _staging_names(entries):
    if type(entries) is not list or len(entries) > 4096:
        raise OwnerError('validation staging inventory is malformed')
    names = []
    for item in entries:
        if (type(item) is not dict or set(item) != {'name', 'directory', 'reparse', 'mtime_ns'}
                or type(item['name']) is not str
                or not item['name'].startswith(('looking-glass-stage-', 'agreement-tokenizer-'))
                or type(item['directory']) is not bool or type(item['reparse']) is not bool
                or type(item['mtime_ns']) is not int or item['mtime_ns'] < 0):
            raise OwnerError('validation staging entry is malformed')
        names.append(item['name'])
    if names != sorted(set(names)):
        raise OwnerError('validation staging entries are not unique and ordered')
    return set(names)


def _check_validation_artifacts(phase, directory, exit_code, reaped):
    output = Path(phase) / 'owner'
    if (type(exit_code) is not int or exit_code != 0 or reaped is not True
            or os.path.lexists(directory)
            or _json(_read(output / 'stdout.txt', 1024 * 1024)) != {'status': 'validated_no_model_forward'}
            or _read(output / 'stderr.txt', 1024 * 1024) != b''):
        raise OwnerError('validation child did not finish cleanly')
    overrides = _json(_read(output / 'runtime-overrides.json', 8192))
    before = _json(_read(output / 'checkpoint-staging-before.json', 512 * 1024))
    after = _json(_read(output / 'checkpoint-staging-after.json', 512 * 1024))
    if (overrides != {'schema': 'mlg-natural-heldout-runtime-overrides-v1',
                      'MKL_CBWR': None, 'ATEN_CPU_CAPABILITY': None}
            or type(before) is not dict or type(before.get('cache_free_bytes')) is not int
            or type(before.get('destination_free_bytes')) is not int
            or type(before.get('required_free_bytes')) is not int
            or before['required_free_bytes'] != 3 * 1024 * 1024 * 1024
            or min(before['cache_free_bytes'], before['destination_free_bytes']) < before['required_free_bytes']
            or type(before.get('entries')) is not list or type(after) is not dict
            or type(after.get('entries')) is not list or after.get('new_unresolved_names') != []):
        raise OwnerError('validation runtime or staging observation differs')
    if _staging_names(after['entries']) - _staging_names(before['entries']):
        raise OwnerError('validation left a new cache staging entry')
    return None


def audit_validation(phase, directory, exit_code, reaped):
    _check_validation_artifacts(phase, directory, exit_code, reaped)
    accounting = _validation_account(phase, supervisor_record=False, reserve=4 * 1024 * 1024)
    return {'status': 'incomplete', 'inventory': {'retained_tensor_bytes': 0},
            'validation_only': True, 'validation_accounting': accounting}


def _check_validation_receipt(child, phase, config, binding_raw, operation_bytes, *, child_path, bootstrap,
                              outer_bootstrap, owner_path, interpreter, require_parent_observation=True):
    if type(require_parent_observation) is not bool:
        raise OwnerError('invalid validation observation requirement')
    validation = _validation_root(phase, operation_bytes)
    config_path = validation / 'validation-config.json'
    validation_raw, validation_config = child._load_config(config_path)
    for key in ('frozen_root', 'cache_dir', 'public_files'):
        if validation_config[key] != config[key]:
            raise OwnerError('validation configuration differs from this owner')
    left, right = dict(validation_config['public_urls']), dict(config['public_urls'])
    left.pop('commitment_url'); right.pop('commitment_url')
    if left != right or Path(validation_config['phase_dir']) != validation:
        raise OwnerError('validation publication or directory differs from this owner')
    output = validation / 'owner'
    _check_validation_artifacts(validation, output / 'attempt', 0, True)
    terminal_raw = _read(output / 'owner-terminal.json', 65536)
    terminal = _json(terminal_raw)
    supervisor_raw = _read(output / 'supervisor.json', 2 * 1024 * 1024)
    supervisor = _json(supervisor_raw)
    command = supervisor.get('command') if type(supervisor) is dict else None
    if (type(terminal) is not dict or terminal.get('schema') != 'mlg-natural-heldout-owner-validation-v1'
            or terminal.get('status') != 'validated_no_model_forward' or terminal.get('cleanup_verified') is not True
            or terminal.get('binding_sha256') != _digest(binding_raw)
            or terminal.get('operational_binding_sha256') != _digest(operation_bytes)
            or terminal.get('validation_config_sha256') != _digest(validation_raw)
            or terminal.get('supervisor_sha256') != _digest(supervisor_raw)
            or terminal.get('scientific_validation') != 'not applicable; no held-out evidence'
            or type(supervisor) is not dict or supervisor.get('status') != 'incomplete'
            or type(supervisor.get('exit_code')) is not int or supervisor['exit_code'] != 0
            or type(supervisor.get('child_pid')) is not int or not 0 < supervisor['child_pid'] <= 0xffffffff
            or any(supervisor.get(key) is not True for key in
                   ('child_reaped', 'job_drained', 'job_closed', 'logs_complete', 'cleanup_verified'))
            or supervisor.get('quarantined') is not False or supervisor.get('retry_safe') is not False
            or type(supervisor.get('postexit_active_processes')) is not int
            or supervisor['postexit_active_processes'] != 0 or supervisor.get('supervisor_error') is not None
            or supervisor.get('log_overflow') is not False or supervisor.get('cancellation_requested') is not False
            or supervisor.get('audit', {}).get('validation_only') is not True
            or supervisor.get('audit', {}).get('status') != 'incomplete'
            or not _same_json(supervisor.get('audit', {}).get('inventory'), {'retained_tensor_bytes': 0})
            or supervisor.get('retained_tensor_files', {}).get('accounting_complete') is not True
            or type(supervisor['retained_tensor_files'].get('total_bytes')) is not int
            or supervisor['retained_tensor_files']['total_bytes'] != 0
            or type(command) is not list or len(command) != 11
            or command[0] != interpreter or command[1:5] != ['-I', '-S', '-B', '-X']
            or type(command[5]) is not str or not command[5].startswith('pycache_prefix=')
            or not Path(command[5][len('pycache_prefix='):]).is_absolute()
            or command[6:] != ['-c', bootstrap, child_path, '--validate-only', str(config_path)]):
        raise OwnerError('validation receipt is incomplete or mismatched')
    for name in ('stdout', 'stderr'):
        if supervisor.get(name + '_sha256') != _digest(_read(output / (name + '.txt'), 1024 * 1024)):
            raise OwnerError('validation log differs from its supervisor')
    observation_raw = b''
    if require_parent_observation:
        observation_raw = _read(validation / 'parent-observation.json', 65536)
        observation = _json(observation_raw)
        outer_command = observation.get('command') if type(observation) is dict else None
        parent_pid = observation.get('parent_pid') if type(observation) is dict else None
        birth = observation.get('parent_birth_100ns') if type(observation) is dict else None
        if (type(parent_pid) is not int or not 0 < parent_pid <= 0xffffffff
                or parent_pid == supervisor['child_pid'] or type(birth) is not int or not 0 < birth < 2 ** 64
                or type(outer_command) is not list or len(outer_command) != 11
                or outer_command[0] != interpreter or outer_command[1:5] != ['-I', '-S', '-B', '-X']
                or type(outer_command[5]) is not str or not outer_command[5].startswith('pycache_prefix=')
                or not Path(outer_command[5][len('pycache_prefix='):]).is_absolute()
                or outer_command[6:] != ['-c', outer_bootstrap, owner_path, '--validate-only', str(config_path)]):
            raise OwnerError('validation parent identity or command differs')
        expected_observation = {'schema': 'mlg-natural-heldout-validation-parent-observation-v1',
                                'status': 'completed', 'exit_code': 0, 'parent_reaped': True,
                                'outer_job_drained': True, 'outer_job_closed': True,
                                'parent_pid': parent_pid, 'parent_birth_100ns': birth, 'command': outer_command,
                                'validation_config_sha256': _digest(validation_raw),
                                'owner_terminal_sha256': _digest(terminal_raw),
                                'supervisor_sha256': _digest(supervisor_raw),
                                'operational_binding_sha256': _digest(operation_bytes)}
        expected_raw = json.dumps(expected_observation, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                  separators=(',', ':')).encode('utf-8')
        if observation_raw != expected_raw:
            raise OwnerError('validation parent exit was not observed')
    accounting = _validation_account(validation, supervisor_record=True, terminal_record=True,
                                     parent_record=require_parent_observation)
    prior = dict(accounting)
    prior['apparent_bytes'] -= len(terminal_raw) + len(observation_raw)
    callback_accounting = dict(prior)
    callback_accounting['apparent_bytes'] -= len(supervisor_raw)
    if (not _same_json(prior, terminal.get('validation_accounting_before_terminal'))
            or not _same_json(callback_accounting, supervisor.get('audit', {}).get('validation_accounting'))):
        raise OwnerError('validation accounting differs from its receipt')
    return validation


def execute(config_path, *, validation_only=False):
    """Run a previously committed phase; all paths and logs remain private."""
    if _QUARANTINED or type(validation_only) is not bool:
        raise OwnerError('source or process custody is unverified; stop')
    operational_root = Path(__file__).resolve(strict=True).parents[1]
    child, operation_bytes, source_bytes = _bootstrap_child(config_path, operational_root)
    child._check_environment()
    config_raw, config = child._load_config(config_path)
    root = Path(config['frozen_root']).resolve(strict=True)
    binding_raw = child._read(config['public_files']['binding'], 2 * 1024 * 1024)
    if _digest(binding_raw) != child.BINDING_SHA256:
        raise OwnerError('discovery binding differs')
    binding = _json(binding_raw)
    if binding.get('source_commit') != child.SOURCE_COMMIT:
        raise OwnerError('discovery source revision differs')
    child._install_frozen_finder(root, binding['source_sha256'])
    from looking_glass import natural_agreement_controller as control
    from scripts.v2_source_guards import InheritableHandles, SourceFileLease
    frozen = child._frozen_launcher(root, binding['source_sha256'])
    frozen['_startup']()
    initial = control.verify_source_inventory(root, binding['source_commit'], binding['source_sha256'])
    core = [root / name for name in initial[1]] + [root / name for name in (
        'docs/PRETRAINED_NUMBER_PATCH_V1.md', 'docs/PRETRAINED_NUMBER_NATURAL_PREDICTION_CLARIFICATION.md',
        'results/distilgpt2-agreement-preflight-v1-20261028/frozen.json',
        'results/distilgpt2-agreement-preflight-v1-20261028/run/result.json',
        'results/distilgpt2-agreement-preflight-v1-20261028/run/last_logits.f32')]
    operational = [operational_root / relative for relative in SOURCE_FILES.values()]
    leases = [SourceFileLease(root, core), SourceFileLease(operational_root, operational)]
    state = {}
    phase = Path(config['phase_dir'])
    output = phase / 'owner'

    def precheck():
        if (control.verify_source_inventory(root, *initial) != initial
                or child._read(config_path, 32768) != config_raw):
            raise OwnerError('source or configuration changed before launch')
        for name, relative in SOURCE_FILES.items():
            if child._read(operational_root / relative, 8 * 1024 * 1024) != source_bytes[name]:
                raise OwnerError('operational source changed before protection')
        if child._read(config['public_files']['operational_binding'], 2 * 1024 * 1024) != operation_bytes:
            raise OwnerError('operational binding changed before protection')
        if validation_only:
            child._check_public_operation(config, operation_bytes, require_commitment=False)
        else:
            _verify_public_commitment(child, config, operation_bytes)
        custody = child._load_custody(operational_root / SOURCE_FILES['custody'], source_bytes['custody'])
        if custody.SOURCE_SUFFIXES != SOURCE_FILES:
            raise OwnerError('custodian operational source schema differs')
        files, public = child._public_inputs(config, custody)
        if files['binding'] != binding_raw or files['operational_binding'] != operation_bytes:
            raise OwnerError('public inputs changed')
        custody._verify_local_operational_sources(public, operational_root)
        frozen['_configure_runtime']()
        frozen['_published'](binding_raw, config['public_urls']['binding_url'])
        # The child checks the full runtime, tokenizer and asset binding before
        # model loading. Do not create an unobserved parent tokenizer stage.
        from looking_glass import pretrained_number_commitments as commitments
        from looking_glass import agreement_heldout as heldout
        _, _, snapshot = custody.check_public_inputs(public, commitments=commitments, heldout=heldout,
                                                      wrapper_bytes=source_bytes['custody'])
        observed = custody.observe_current_inventory()
        if (observed.get('inventory_sha256') != snapshot.sha256
                or observed.get('source_sha256') != custody.CURRENT_INVENTORY_SOURCE_SHA256):
            raise OwnerError('current exposure inventory changed before owner')
        custody.inspect_cache_staging(config['cache_dir'])
        custody.inspect_evidence_destination(phase)
        if validation_only:
            if (phase != _validation_root(phase, operation_bytes)
                    or Path(config_path) != phase / 'validation-config.json'
                    or not phase.is_dir()
                    or {path.name for path in phase.iterdir()} != {'validation-config.json'}
                    or any(os.path.lexists(phase / name) for name in
                           ('claim.json', 'nonce.bin', 'heldout-record.json', 'commitment.json', 'owner-started.json'))):
                raise OwnerError('validation requires a fresh unclaimed directory')
            side_roots = ()
        else:
            validation = _check_validation_receipt(child, phase, config, binding_raw, operation_bytes,
                                                    child_path=str(operational_root / SOURCE_FILES['child']),
                                                    bootstrap=frozen['BOOTSTRAP'], outer_bootstrap=custody.OUTER_BOOTSTRAP,
                                                    owner_path=str(operational_root / SOURCE_FILES['owner']),
                                                    interpreter=sys.executable)
            custody.verify_preparation_supervisor(phase, config['preparation_supervisor'], public,
                                                   source_bytes['custody'])
            if os.path.lexists(output) or os.path.lexists(phase / 'owner-started.json'):
                raise OwnerError('held-out owner is already present or started')
            side_roots = _side_roots(child, config, config_path, validation)
            custody.account_phase(phase, reserve_side_bytes=4 * 1024 * 1024, extra_side_roots=side_roots)
        supervisor = custody._load_file_module('_mlg_natural_heldout_supervisor_owner',
                                                operational_root / SOURCE_FILES['supervisor'],
                                                _digest(source_bytes['supervisor']))
        state.update({'custody': custody, 'public': public, 'commitments': commitments,
                      'supervisor': supervisor, 'side_roots': side_roots})
        if not validation_only:
            state['commitment'] = child._read(phase / 'commitment.json', 8192)

    def callback(directory, exit_code, reaped):
        if validation_only:
            return audit_validation(phase, directory, exit_code, reaped)
        if (child._read(phase / 'commitment.json', 8192) != state['commitment']
                or child._read(config_path, 32768) != config_raw):
            raise OwnerError('commitment or configuration changed during execution')
        result = audit_saved(directory, exit_code, reaped, control=control, binding=binding,
                             binding_bytes=binding_raw, prediction_bytes=state['public'].prediction,
                             aggregate_bytes=state['public'].aggregate, inventory_bytes=state['public'].inventory,
                             record_bytes=child._read(phase / 'heldout-record.json', 1024 * 1024),
                             nonce=child._read(phase / 'nonce.bin', 32), commitment_bytes=state['commitment'],
                             commitment_domain=state['commitments']._HELDOUT_COMMIT_DOMAIN,
                             expected_model_sha256=child.MODEL_STATE_SHA256)
        result['phase_accounting'] = state['custody'].account_phase(
            phase, reserve_side_bytes=4 * 1024 * 1024, extra_side_roots=state['side_roots'])
        return result

    def launch(acquired):
        duplicates = InheritableHandles(leases[0].handles() + leases[1].handles())
        acquired.append(duplicates)
        inherited = duplicates.__enter__()
        command = [sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + sys.pycache_prefix,
                   '-c', frozen['BOOTSTRAP'], str(operational_root / SOURCE_FILES['child'])]
        if validation_only:
            command.append('--validate-only')
        command.append(str(config_path))
        return state['supervisor'].supervise_heldout(command, output, os.environ.copy(),
                                                      audit_result=callback, inherited_handles=inherited)

    def postcheck(verdict):
        if (control.verify_source_inventory(root, *initial) != initial
                or child._read(config_path, 32768) != config_raw):
            raise OwnerError('source or configuration changed after execution')
        for name, relative in SOURCE_FILES.items():
            if child._read(operational_root / relative, 8 * 1024 * 1024) != source_bytes[name]:
                raise OwnerError('operational source changed during execution')
        receipt_raw = child._read(output / 'supervisor.json', 2 * 1024 * 1024)
        if not control._same_json(_json(receipt_raw), verdict):
            raise OwnerError('supervisor receipt differs')
        if validation_only:
            if (verdict.get('status') != 'incomplete' or verdict.get('exit_code') != 0
                    or verdict.get('child_reaped') is not True or verdict.get('job_drained') is not True
                    or verdict.get('job_closed') is not True or verdict.get('logs_complete') is not True
                    or verdict.get('supervisor_error') is not None or verdict.get('log_overflow') is not False
                    or verdict.get('audit', {}).get('validation_only') is not True
                    or type(verdict.get('postexit_active_processes')) is not int
                    or verdict['postexit_active_processes'] != 0):
                raise OwnerError('validation process cleanup or marker differs')
            accounting = _validation_account(phase, supervisor_record=True, reserve=65536)
            return {'schema': 'mlg-natural-heldout-owner-validation-v1',
                    'status': 'validated_no_model_forward', 'supervisor_sha256': _digest(receipt_raw),
                    'binding_sha256': _digest(binding_raw), 'operational_binding_sha256': _digest(operation_bytes),
                    'validation_config_sha256': _digest(config_raw),
                    'cleanup_verified': True, 'validation_accounting_before_terminal': accounting,
                    'scientific_validation': 'not applicable; no held-out evidence'}
        if (verdict.get('status') == 'evidence_complete'
                and (type(verdict.get('postexit_active_processes')) is not int
                     or verdict['postexit_active_processes'] != 0)):
            raise OwnerError('other Job members remained active after owner exit')
        accounting = state['custody'].account_phase(phase, reserve_side_bytes=65536,
                                                     extra_side_roots=state['side_roots'])
        terminal = {'schema': 'mlg-natural-heldout-owner-terminal-v1', 'status': verdict['status'],
                    'supervisor_sha256': _digest(receipt_raw), 'cleanup_verified': True,
                    'phase_accounting_before_terminal': accounting,
                    'scientific_validation': 'pending independent raw numerical and landing audit'}
        return terminal

    try:
        terminal = under_source_leases(leases, precheck, launch, postcheck, retained=state)
        _write_new(output / 'owner-terminal.json', terminal)
        return terminal
    except BaseException as error:
        # Never rewrite an existing record or inspect live tensor files.
        try:
            failure = {'schema': 'mlg-natural-heldout-owner-terminal-v1', 'status': 'incomplete',
                       'error_type': type(error).__name__, 'cleanup_verified': False if _QUARANTINED else None}
            destination = (output / 'owner-terminal.json' if output.is_dir()
                           else phase / 'owner-prelaunch-failure.json')
            if not os.path.lexists(destination):
                _write_new(destination, failure)
        except KeyboardInterrupt:
            raise
        except BaseException:
            pass
        raise


def _hold_quarantine():
    """Keep process and source handles rooted until an external stop."""
    while _QUARANTINED:
        try:
            _QUARANTINE_EVENT.wait()
        except BaseException:
            # A second Ctrl+C or an output failure cannot silently abandon
            # unverified child/job custody. Forced OS termination is distinct.
            continue


def _ignore_console_interrupts():
    """Mask normal Windows console interrupts before any child can exist."""
    previous = []
    try:
        for number in (signal.SIGINT, getattr(signal, 'SIGBREAK', None)):
            if number is not None:
                old = signal.getsignal(number)
                signal.signal(number, signal.SIG_IGN)
                previous.append((number, old))
    except BaseException:
        for number, old in reversed(previous):
            signal.signal(number, old)
        raise
    return previous


def main():
    previous = _ignore_console_interrupts()
    try:
        try:
            if len(sys.argv) == 2:
                result = execute(sys.argv[1])
            elif len(sys.argv) == 3 and sys.argv[1] == '--validate-only':
                result = execute(sys.argv[2], validation_only=True)
            else:
                raise OwnerError('one private configuration path is required')
        except BaseException as error:
            cancelled = isinstance(error, KeyboardInterrupt)
            try:
                print(json.dumps({'status': 'incomplete' if _QUARANTINED else 'cancelled' if cancelled else 'incomplete',
                                  'error_type': type(error).__name__, 'quarantined': bool(_QUARANTINED)}),
                      file=sys.stderr, flush=True)
            except BaseException:
                pass
            if _QUARANTINED:
                _hold_quarantine()
            raise SystemExit(130 if cancelled else 1) from None
        print(json.dumps({'status': result['status'], 'scientific_validation': result['scientific_validation']}),
              flush=True)
        raise SystemExit(0 if result['status'] in ('evidence_complete', 'validated_no_model_forward') else 1)
    finally:
        if not _QUARANTINED:
            for number, old in reversed(previous):
                signal.signal(number, old)


if __name__ == '__main__':
    main()
