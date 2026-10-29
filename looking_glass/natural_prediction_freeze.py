"""Build natural discovery freeze bytes from separately audited evidence.

This pure postprocessor does not run a model, publish files, sample a nonce,
authenticate an auditor or establish execution/publication chronology. The
trusted custodian must independently establish those facts before use.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

from . import agreement_heldout, pretrained_number_commitments as commitments


_PREFLIGHT = '1b1adf4d72163bfa3e0399ed8ee92eba6b5c0eb82a47edf28cecde5ffb1deafe'
_PREFLIGHT_ARCHIVE = '7c262f3fe81236888a8f5fdb6d31f7835666674be9ffbf36f8ce2ca6a375eb6c'
_RESERVATIONS = {'natural_discovery': 128 * 1024 * 1024, 'natural_heldout': 160 * 1024 * 1024,
                 'injection_discovery': 48 * 1024 * 1024, 'injection_heldout': 72 * 1024 * 1024}
_AUDIT_SCOPE = 'Saved public development data only; no held-out or general causal claim.'
_SITES = tuple(f'{layer}:{component}:last' for layer in range(1, 5)
               for component in ('attn', 'mlp'))
_AUDIT_KEYS = {'schema', 'status', 'journal_event_sha256', 'journal_started_sha256',
               'journal_terminal_sha256', 'audit_source_sha256', 'preflight_manifest_sha256',
               'forwards', 'tensor_count', 'tensor_bytes', 'present_predictions',
               'present_family_effects', 'present_strict_sign_reversals',
               'secondary_past_predictions', 'present_baseline', 'past_baseline',
               'selected_site', 'earliest_site', 'random_site', 'magnitude_site', 'scope'}


class PredictionFreezeError(ValueError):
    """The supplied evidence cannot support this prediction freeze."""


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value, length):
    return type(value) is str and re.fullmatch(rf'[0-9a-f]{{{length}}}', value) is not None


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise PredictionFreezeError('value is not finite JSON') from error


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PredictionFreezeError('duplicate JSON key')
        result[key] = value
    return result


def _parse(raw, limit, label):
    if type(raw) is not bytes or not raw or len(raw) > limit:
        raise PredictionFreezeError(f'{label} must be bounded immutable bytes')
    def nonfinite(_):
        raise PredictionFreezeError('nonfinite JSON number')
    try:
        return json.loads(raw, object_pairs_hook=_object, parse_constant=nonfinite)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise PredictionFreezeError(f'invalid {label} JSON') from error


def _public_cases(raw):
    """Reconstruct the ordered discovery cases from the pinned public manifest."""
    if type(raw) is not bytes or not raw or len(raw) > 256 * 1024 or _digest(raw) != _PREFLIGHT:
        raise PredictionFreezeError('public preflight bytes differ from the frozen manifest')
    manifest = _parse(raw, 256 * 1024, 'preflight manifest')
    try:
        rows = manifest['rows']
        if type(rows) is not list or len(rows) != 16:
            raise PredictionFreezeError('invalid public preflight rows')
        return [{'case_id': f'c{index:02}', 'quartet': f'q{row["group"]}',
                 'lemma': rows[index - index % 2]['subject'], 'number': row['subject_number'],
                 'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
                 'subject_index': row['subject_index']} for index, row in enumerate(rows)]
    except (KeyError, IndexError, TypeError) as error:
        raise PredictionFreezeError('invalid public preflight rows') from error


def _event_order(rows, cases):
    if type(cases) is not list or len(cases) != 16:
        raise PredictionFreezeError('binding does not have sixteen cases')
    groups, case_ids, token_rows = {}, set(), set()
    for case in cases:
        if type(case) is not dict:
            raise PredictionFreezeError('invalid bound case')
        name, quartet, lemma, number = (case.get('case_id'), case.get('quartet'),
                                        case.get('lemma'), case.get('number'))
        if (any(type(x) is not str or not x or len(x) > 128 for x in (name, quartet, lemma))
                or name in case_ids or number not in ('singular', 'plural')):
            raise PredictionFreezeError('invalid or duplicate bound case identity')
        ids, mask, subject = case.get('input_ids'), case.get('attention_mask'), case.get('subject_index')
        if (type(ids) is not list or not 2 <= len(ids) <= 32
                or any(type(token) is not int or not 0 <= token <= 50256 for token in ids)
                or type(mask) is not list or mask != [1] * len(ids)
                or any(type(value) is not int for value in mask)
                or type(subject) is not int or not 0 < subject < len(ids) - 1
                or tuple(ids) in token_rows):
            raise PredictionFreezeError('invalid bound token coordinates')
        case_ids.add(name)
        token_rows.add(tuple(ids))
        groups.setdefault(quartet, []).append(case)
    if len(groups) != 4:
        raise PredictionFreezeError('binding does not contain four quartets')
    expected = [('run_contract', None, None, None, None)]
    for quartet, members in groups.items():
        if (len(members) != 4 or len({member['lemma'] for member in members}) != 2
                or len({(member['lemma'], member['number']) for member in members}) != 4
                or len({len(member['input_ids']) for member in members}) != 1
                or len({member['subject_index'] for member in members}) != 1):
            raise PredictionFreezeError('bound quartet is not a balanced two-lemma design')
        subject = members[0]['subject_index']
        contexts = {tuple(member['input_ids'][:subject] + member['input_ids'][subject + 1:])
                    for member in members}
        subjects = {member['input_ids'][subject] for member in members}
        if len(contexts) != 1 or len(subjects) != 4:
            raise PredictionFreezeError('bound quartet differs outside its unique subject token')
        ids = [member['case_id'] for member in members]
        for kind in ('clean_capture', 'clean_repeat', 'null'):
            expected.extend((kind, quartet, case, None, None) for case in ids)
        expected.extend(('patch', quartet, target, site, donor)
                        for site in _SITES for target in ids for donor in ids)
        expected.extend((kind, quartet, target, None, None)
                        for target in ids for kind in ('delivery', 'subject_position'))
        expected.extend(('final_clean', quartet, case, None, None) for case in ids)
    expected.extend((kind, None, None, site, None)
                    for kind in ('present_site_metrics', 'secondary_past_site_metrics') for site in _SITES)
    expected.append(('summary', None, None, None, None))
    observed = []
    for row in rows:
        event = row['event']
        kind = event.get('kind')
        if kind in ('clean_capture', 'clean_repeat', 'final_clean'):
            observed.append((kind, event.get('quartet'), event.get('case'), None, None))
        elif kind in ('null', 'delivery', 'subject_position'):
            observed.append((kind, event.get('quartet'), event.get('target'), None, None))
        elif kind == 'patch':
            site = event.get('site')
            observed.append((kind, event.get('quartet'), event.get('target'),
                             site.get('id') if type(site) is dict else None, event.get('donor')))
        elif kind in ('present_site_metrics', 'secondary_past_site_metrics'):
            observed.append((kind, None, None, event.get('site'), None))
        else:
            observed.append((kind, None, None, None, None))
    if observed != expected:
        raise PredictionFreezeError('event order or case coverage differs from the bound design')


def _witness_inventory(rows, cases, inventory, retained_files):
    """Check references and shapes; the independent auditor checks file bytes."""
    bound = {case['case_id']: case for case in cases}
    references = {}
    for row in rows:
        event, tensors = row['event'], row['tensors']
        kind = event['kind']
        if kind in ('clean_capture', 'clean_repeat'):
            shapes = {f'capture_{index}': [768] for index in range(10)}
            shapes['last_logits'] = [50257]
        elif kind in ('null', 'patch', 'delivery', 'subject_position'):
            case = bound[event['target']]
            shapes = {'before': [768], 'after': [768],
                      'whole_before': [1, len(case['input_ids']), 768],
                      'whole_after': [1, len(case['input_ids']), 768]}
            logits = ('contrast_logits' if kind == 'patch' and event['target'] != event['donor']
                      else 'last_logits')
            shapes[logits] = [4] if logits == 'contrast_logits' else [50257]
        elif kind == 'final_clean':
            shapes = {'last_logits': [50257]}
        else:
            shapes = {}
        if set(tensors) != set(shapes):
            raise PredictionFreezeError('forward witness inventory is incomplete')
        for name, shape in shapes.items():
            ref = tensors[name]
            if (type(ref) is not dict or set(ref) != {'sha256', 'shape', 'dtype', 'bytes'}
                    or not _hex(ref.get('sha256'), 64) or not _same(ref.get('shape'), shape)
                    or ref.get('dtype') != 'float32-le' or type(ref.get('bytes')) is not int
                    or ref['bytes'] != 4 * math.prod(shape)):
                raise PredictionFreezeError('forward witness reference is invalid')
            previous = references.setdefault(ref['sha256'], ref['bytes'])
            if previous != ref['bytes']:
                raise PredictionFreezeError('one witness digest has inconsistent sizes')
    if (not references or sorted(references) != sorted(inventory['tensor_sha256'])
            or sum(references.values()) != inventory['retained_tensor_bytes']
            or inventory['retained_tensor_bytes'] > 512 * 1024 * 1024):
        raise PredictionFreezeError('witness references differ from the retained inventory')
    if (type(retained_files) is not dict
            or set(retained_files) != {'accounting_complete', 'total_bytes', 'files'}
            or retained_files['accounting_complete'] is not True
            or type(retained_files['total_bytes']) is not int
            or retained_files['total_bytes'] != sum(references.values())
            or type(retained_files['files']) is not list
            or len(retained_files['files']) != len(references)):
        raise PredictionFreezeError('retained file accounting is incomplete')
    actual = {}
    for row in retained_files['files']:
        if (type(row) is not dict or set(row) != {'name', 'bytes', 'sha256'}
                or not _hex(row.get('sha256'), 64) or row.get('name') != row['sha256'] + '.f32le'
                or type(row.get('bytes')) is not int or row['sha256'] in actual):
            raise PredictionFreezeError('retained tensor file metadata is invalid')
        actual[row['sha256']] = row['bytes']
    if actual != references:
        raise PredictionFreezeError('retained tensor files differ from the witness references')


def _audit_metrics(audit, selection, present, past):
    present_by_site = {event['site']: event['metrics'] for event in present}
    past_by_site = {event['site']: event['metrics'] for event in past}
    try:
        predictions = {site: present_by_site[site]['site_summary']['mean_number_main_contrast'] for site in _SITES}
        past_predictions = {site: past_by_site[site]['site_summary']['mean_number_main_contrast'] for site in _SITES}
        effects = {site: {row['quartet']: row['number_main_contrast']
                          for row in present_by_site[site]['quartets']} for site in _SITES}
        sign_counts = {
            site: {label: sum(row['strict_sign_reversal'] is True
                              for row in present_by_site[site]['pairs'] if row['cell'].endswith(label))
                   for label in ('same_number', 'opposite_number')}
            for site in _SITES
        }
        checks = ((audit['present_predictions'], predictions),
                  (selection['predictions'], predictions),
                  (audit['secondary_past_predictions'], past_predictions),
                  (audit['present_family_effects'], effects),
                  (audit['present_strict_sign_reversals'], sign_counts),
                  (audit['present_baseline'], present_by_site[_SITES[0]]['baseline']),
                  (audit['past_baseline'], past_by_site[_SITES[0]]['baseline']))
        checks += tuple((audit[name], selection[name])
                        for name in ('selected_site', 'earliest_site', 'random_site', 'magnitude_site'))
        if any(not _same(left, right) for left, right in checks):
            raise PredictionFreezeError('independent audit does not match discovery measurements')
    except (KeyError, TypeError, AttributeError) as error:
        raise PredictionFreezeError('independent audit or metric fields are incomplete') from error


@dataclass(frozen=True)
class NaturalPredictionBundle:
    """Immutable candidate bytes, not a publication or execution attestation."""
    prediction_bytes: bytes = field(repr=False)
    aggregate_bytes: bytes = field(repr=False)
    prediction_sha256: str
    aggregate_sha256: str
    runtime_sha256: str
    source_commit: str
    events_sha256: str
    started_sha256: str
    terminal_sha256: str
    independent_audit_sha256: str
    audit_source_sha256: str
    builder_sha256: str
    clarification_sha256: str
    supervisor_sha256: str
    launcher_terminal_sha256: str


def build_natural_prediction_freeze(*, events, started, terminal, supervisor, launcher_terminal,
                                    binding, audit, preflight_manifest, expected_source_commit,
                                    expected_audit_source_sha256,
                                    expected_builder_sha256, expected_clarification_sha256):
    """Build natural-only canonical files from exact, independently audited bytes.

    The caller must first establish publication of the binding, execution and
    independent audit, file quiescence, and a trustworthy expected audit-source
    digest and the executing builder's exact source bytes. Matching caller-supplied
    records cannot attest any of those facts or the loaded Python bytecode.
    """
    if (not _hex(expected_source_commit, 40) or not _hex(expected_audit_source_sha256, 64)
            or not _hex(expected_builder_sha256, 64) or not _hex(expected_clarification_sha256, 64)):
        raise PredictionFreezeError('expected source identities are invalid')
    if type(events) is not bytes or not events or len(events) > 64 * 1024 * 1024 or not events.endswith(b'\n'):
        raise PredictionFreezeError('event log is incomplete or exceeds its byte limit')
    config = _parse(binding, 2 * 1024 * 1024, 'binding')
    public_cases = _public_cases(preflight_manifest)
    begin = _parse(started, 256 * 1024, 'started')
    end = _parse(terminal, 256 * 1024, 'terminal')
    control = _parse(supervisor, 2 * 1024 * 1024, 'supervisor')
    launch = _parse(launcher_terminal, 256 * 1024, 'launcher terminal')
    reviewed = _parse(audit, 2 * 1024 * 1024, 'independent audit')
    if type(config) is not dict or type(begin) is not dict or type(end) is not dict or type(control) is not dict or type(launch) is not dict:
        raise PredictionFreezeError('binding or lifecycle record is not an object')
    try:
        rebuilt_binding = (json.dumps(config, sort_keys=True, ensure_ascii=True,
                                      allow_nan=False, indent=2) + '\n').encode('utf-8')
    except (TypeError, ValueError, RecursionError) as error:
        raise PredictionFreezeError('binding is not finite JSON') from error
    if (rebuilt_binding != binding or config.get('schema') != 'mlg-natural-cpu-binding-v1'
            or config.get('source_commit') != expected_source_commit
            or config.get('protocol_sha256') != commitments.PROTOCOL_SHA256
            or config.get('checkpoint_revision') != commitments.CHECKPOINT_REVISION
            or config.get('preflight_manifest_sha256') != _PREFLIGHT
            or config.get('natural_prediction_clarification_sha256') != expected_clarification_sha256
            or config.get('independent_natural_audit_source_sha256') != expected_audit_source_sha256
            or config.get('track') != 'natural discovery only'
            or type(config.get('expected_forwards')) is not int or config['expected_forwards'] != 608
            or type(config.get('runtime')) is not dict or config['runtime'].get('device') != 'cpu'
            or type(config.get('source_sha256')) is not dict
            or config['source_sha256'].get('looking_glass/natural_prediction_freeze.py') != expected_builder_sha256):
        raise PredictionFreezeError('binding identity or runtime differs from the declared natural track')
    if not _same(config.get('cases'), public_cases):
        raise PredictionFreezeError('bound cases differ from the frozen public preflight')
    if (set(begin) != {'schema', 'binding_sha256', 'binding'}
            or begin['schema'] != 'mlg-natural-cpu-attempt-v1'
            or begin['binding_sha256'] != _digest(binding) or not _same(begin['binding'], config)):
        raise PredictionFreezeError('attempt start does not bind the exact runtime manifest')
    if (launch.get('status') != 'evidence_complete' or launch.get('phase') != 'owner_supervision'
            or launch.get('child_started') is not True or launch.get('cleanup_verified') is not True
            or launch.get('retry_safe') is not True or launch.get('owner_record') != 'owner/supervisor.json'
            or launch.get('owner_supervisor_sha256') != _digest(supervisor)
            or control.get('status') != 'evidence_complete' or type(control.get('exit_code')) is not int
            or control['exit_code'] != 0 or control.get('child_reaped') is not True
            or control.get('job_drained') is not True or control.get('job_closed') is not True
            or control.get('cleanup_verified') is not True or control.get('retry_safe') is not True
            or control.get('log_overflow') is not False
            or control.get('supervisor_error') is not None or control.get('log_errors')):
        raise PredictionFreezeError('launcher or supervisor did not complete and cleanly reap')
    inv = control.get('inventory')
    if (type(inv) is not dict or inv.get('started_sha256') != _digest(started)
            or inv.get('events_sha256') != _digest(events) or inv.get('terminal_sha256') != _digest(terminal)
            or type(inv.get('retained_tensor_bytes')) is not int or inv['retained_tensor_bytes'] < 0
            or type(inv.get('tensor_sha256')) is not list
            or any(not _hex(x, 64) for x in inv['tensor_sha256'])
            or len(set(inv['tensor_sha256'])) != len(inv['tensor_sha256'])):
        raise PredictionFreezeError('supervisor inventory does not bind these artifacts')
    budget = config.get('study_budget')
    reservations = budget.get('phase_reservations_bytes') if type(budget) is dict else None
    if (type(budget) is not dict or type(budget.get('study_limit_bytes')) is not int
            or budget['study_limit_bytes'] != 512 * 1024 * 1024
            or type(budget.get('prior_unique_tensor_bytes')) is not int
            or budget['prior_unique_tensor_bytes'] != 3216448
            or not _same(budget.get('prior_payloads'), [{'sha256': _PREFLIGHT_ARCHIVE, 'bytes': 3216448,
                                                        'description': 'public preflight full-vocabulary logits'}])
            or not _same(reservations, _RESERVATIONS)
            or type(budget.get('unallocated_bytes')) is not int
            or budget['unallocated_bytes'] != budget['study_limit_bytes'] - 3216448 - sum(_RESERVATIONS.values())
            or inv['retained_tensor_bytes'] > reservations['natural_discovery']
            or budget['unallocated_bytes'] < 0):
        raise PredictionFreezeError('bound study storage budget is invalid or exceeded')
    if (type(reviewed) is not dict or set(reviewed) != _AUDIT_KEYS or _canonical(reviewed) != audit
            or reviewed.get('schema') != 'mlg-independent-natural-discovery-audit-v1'
            or reviewed.get('status') != 'independent_saved_evidence_pass'
            or reviewed.get('audit_source_sha256') != expected_audit_source_sha256
            or reviewed.get('journal_event_sha256') != _digest(events)
            or reviewed.get('journal_started_sha256') != _digest(started)
            or reviewed.get('journal_terminal_sha256') != _digest(terminal)
            or reviewed.get('preflight_manifest_sha256') != _PREFLIGHT
            or type(reviewed.get('forwards')) is not int or reviewed['forwards'] != 608
            or type(reviewed.get('tensor_count')) is not int or reviewed['tensor_count'] != len(inv['tensor_sha256'])
            or type(reviewed.get('tensor_bytes')) is not int or reviewed['tensor_bytes'] != inv['retained_tensor_bytes']
            or reviewed.get('scope') != _AUDIT_SCOPE):
        raise PredictionFreezeError('independent audit does not bind these exact artifacts')
    lines = events.splitlines()
    if (len(lines) != 626 or end.get('status') != 'complete_candidate'
            or end.get('storage_accounting_complete') is not True
            or type(end.get('events')) is not int or end['events'] != len(lines)
            or type(end.get('event_log_bytes')) is not int or end['event_log_bytes'] != len(events)
            or type(end.get('retained_tensor_bytes')) is not int or end['retained_tensor_bytes'] != inv['retained_tensor_bytes']
            or not _same(end.get('detail'), {'forwards': 608})):
        raise PredictionFreezeError('candidate terminal does not match the complete event log')
    rows = []
    for index, line in enumerate(lines):
        row = _parse(line, 256 * 1024, 'event')
        if (type(row) is not dict or set(row) != {'index', 'event', 'tensors'}
                or type(row['index']) is not int or row['index'] != index
                or type(row['event']) is not dict or type(row['tensors']) is not dict):
            raise PredictionFreezeError('event envelope or index is invalid')
        rows.append(row)
    _event_order(rows, config.get('cases'))
    if not _same(rows[0]['event'].get('cases'), config['cases']):
        raise PredictionFreezeError('run contract does not match the bound discovery cases')
    _witness_inventory(rows, config['cases'], inv, control.get('retained_tensor_files'))
    final = rows[-1]['event'].get('result')
    if (type(final) is not dict or final.get('status') != 'complete'
            or type(final.get('forwards')) is not int or final['forwards'] != 608
            or type(final.get('selection')) is not dict):
        raise PredictionFreezeError('discovery summary is incomplete')
    present = [row['event'] for row in rows if row['event'].get('kind') == 'present_site_metrics']
    past = [row['event'] for row in rows if row['event'].get('kind') == 'secondary_past_site_metrics']
    record = {'schema': 'mlg-number-natural-predictions-v2',
              'protocol_sha256': commitments.PROTOCOL_SHA256, 'source_commit': expected_source_commit,
              'checkpoint_revision': commitments.CHECKPOINT_REVISION, 'selection': final['selection'],
              'discovery_event_sha256': _digest(events), 'independent_audit_sha256': _digest(audit),
              'present_site_metrics': present, 'secondary_past_site_metrics': past}
    prediction = _canonical(record)
    freeze = commitments.make_prediction_freeze({'natural': {
        'source_commit': expected_source_commit, 'runtime_sha256': _digest(binding),
        'prediction_sha256': _digest(prediction)}})
    try:
        agreement_heldout.load_frozen_natural_predictions(prediction, freeze)
        _audit_metrics(reviewed, final['selection'], present, past)
    except (ValueError, RuntimeError) as error:
        raise PredictionFreezeError('discovery prediction validation failed') from error
    return NaturalPredictionBundle(prediction, freeze.record_bytes, _digest(prediction), freeze.sha256,
                                   _digest(binding), expected_source_commit, _digest(events), _digest(started),
                                   _digest(terminal), _digest(audit), expected_audit_source_sha256,
                                   expected_builder_sha256, expected_clarification_sha256,
                                   _digest(supervisor), _digest(launcher_terminal))
