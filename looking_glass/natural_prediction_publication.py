"""Stage and verify public natural-prediction artifacts, without drawing cases.

This module is for a trusted custodian. Source-first execution, protection of
inputs, authenticated audit provenance and pre-draw ordering are caller duties.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import urllib.error
import urllib.parse
import urllib.request


class PublicationError(RuntimeError):
    """The publication evidence is incomplete or inconsistent."""


_RESULT_ROOT = 'results/distilgpt2-natural-agreement-v1-20261028/'
_PREDICTION = _RESULT_ROOT + 'natural-prediction-v2.json'
_AGGREGATE = _RESULT_ROOT + 'natural-prediction-freeze.json'
_PROVENANCE = _RESULT_ROOT + 'natural-prediction-provenance.json'
_RAW = 'https://raw.githubusercontent.com/EastonSell/model-looking-glass/'
_API = 'https://api.github.com/repos/EastonSell/model-looking-glass/'
_HEAD_URL = _API + 'git/ref/heads/main'
_HEX40 = re.compile(r'[0-9a-f]{40}\Z')
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_TENSOR = re.compile(r'([0-9a-f]{64})\.f32le\Z')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, RecursionError) as error:
        raise PublicationError('record is not finite JSON') from error


def _relative_result(path):
    if (type(path) is not str or not path.startswith(_RESULT_ROOT) or len(path) > 300
            or re.fullmatch(r'[A-Za-z0-9_./-]+\.json', path) is None
            or any(part in ('', '.', '..') for part in path.split('/'))):
        raise PublicationError('publication path is not a fixed public result path')
    return path


def raw_url(commit, path):
    if type(commit) is not str or _HEX40.fullmatch(commit) is None:
        raise PublicationError('publication commit is not a full lowercase SHA')
    return _RAW + commit + '/' + _relative_result(path)


def parse_binding_url(url):
    if type(url) is not str or not url.startswith(_RAW):
        raise PublicationError('binding URL is not an official raw GitHub URL')
    rest = url[len(_RAW):]
    commit, separator, path = rest.partition('/')
    if not separator or raw_url(commit, path) != url:
        raise PublicationError('binding URL is not an immutable official result URL')
    if path in (_PREDICTION, _AGGREGATE, _PROVENANCE):
        raise PublicationError('binding path collides with a publication artifact')
    return commit, path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch(url, limit, allow_missing=False):
    request = urllib.request.Request(url, headers={'User-Agent': 'model-looking-glass-publication/1',
                                                   'Accept-Encoding': 'identity'})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=30) as response:
            if response.status != 200 or response.geturl() != url:
                raise PublicationError('unexpected publication response')
            raw = response.read(limit + 1)
    except urllib.error.HTTPError as error:
        if allow_missing and error.code == 404:
            return None
        raise PublicationError('publication bytes are unavailable') from error
    except (OSError, urllib.error.URLError) as error:
        raise PublicationError('publication bytes are unavailable') from error
    if len(raw) > limit:
        raise PublicationError('publication response exceeds its byte bound')
    return raw


def _regular(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise PublicationError('evidence must be a regular non-reparse file')
    return info


def _tensor_inventory(directory, supervisor):
    """Stream-verify all retained tensor bytes under the caller's evidence lease."""
    directory = Path(directory)
    info = directory.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & 0x400):
        raise PublicationError('tensor directory is not a regular directory')
    try:
        record = json.loads(supervisor)['retained_tensor_files']
        files = record['files']
    except (KeyError, TypeError, ValueError) as error:
        raise PublicationError('supervisor retained-file record is invalid') from error
    if (type(record) is not dict or set(record) != {'accounting_complete', 'total_bytes', 'files'}
            or record['accounting_complete'] is not True or type(files) is not list
            or not 0 < len(files) <= 4096 or type(record['total_bytes']) is not int
            or not 0 <= record['total_bytes'] <= 128 * 1024 * 1024):
        raise PublicationError('retained-file accounting is incomplete or unbounded')
    expected = {}
    for row in files:
        if (type(row) is not dict or set(row) != {'name', 'sha256', 'bytes'}
                or type(row['name']) is not str or _TENSOR.fullmatch(row['name']) is None
                or row['name'] != row['sha256'] + '.f32le' or type(row['bytes']) is not int
                or not 0 <= row['bytes'] <= 128 * 1024 * 1024 or row['name'] in expected):
            raise PublicationError('retained tensor entry is invalid')
        expected[row['name']] = row
    if sum(row['bytes'] for row in files) != record['total_bytes']:
        raise PublicationError('retained tensor sizes do not add up')
    if {path.name for path in directory.iterdir()} != set(expected):
        raise PublicationError('tensor directory contains missing or unlisted files')
    for name, row in expected.items():
        path = directory / name
        before = _regular(path)
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(block)
                if count > row['bytes']:
                    raise PublicationError('tensor grew while being verified')
                digest.update(block)
        after = _regular(path)
        if (count != row['bytes'] or digest.hexdigest() != row['sha256']
                or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns):
            raise PublicationError('retained tensor bytes differ from their audited digest')
    if {path.name for path in directory.iterdir()} != set(expected):
        raise PublicationError('tensor directory changed during verification')
    return [expected[name] for name in sorted(expected)]


def _write_new(path, raw):
    with path.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise PublicationError('short artifact write')
        stream.flush()
        os.fsync(stream.fileno())


def prepare_stage(*, inputs, tensor_dir, output_dir, binding_url, wrapper_commit):
    """Build and durably stage a candidate; never publish or sample cases.

    The caller must run source-first under source and evidence leases, establish
    the independent audit, preserve the original files and capture the process
    exit status. Existing or partially written output is never overwritten.
    """
    from .natural_prediction_freeze import build_natural_prediction_freeze

    binding_commit, binding_path = parse_binding_url(binding_url)
    if type(wrapper_commit) is not str or _HEX40.fullmatch(wrapper_commit) is None:
        raise PublicationError('wrapper revision must be a full lowercase SHA')
    if type(inputs) is not dict:
        raise PublicationError('builder inputs must be an immutable-byte mapping')
    inputs = inputs.copy()
    if _fetch(binding_url, 2 * 1024 * 1024) != inputs.get('binding'):
        raise PublicationError('already published binding differs from discovery binding')
    try:
        binding = json.loads(inputs['binding'])
        exposure = binding['development_exposure_inventory_sha256']
        if type(exposure) is not str or _HEX64.fullmatch(exposure) is None:
            raise PublicationError('binding lacks a valid development exposure inventory')
        bundle = build_natural_prediction_freeze(**inputs)
        files = _tensor_inventory(tensor_dir, inputs['supervisor'])
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise PublicationError('candidate evidence failed validation') from error
    metadata = {'schema': 'mlg-natural-publication-stage-v1', 'wrapper_commit': wrapper_commit,
                'source_commit': bundle.source_commit, 'binding_commit': binding_commit,
                'binding_path': binding_path, 'prediction_path': _PREDICTION, 'aggregate_path': _AGGREGATE,
                'provenance_path': _PROVENANCE, 'runtime_sha256': bundle.runtime_sha256,
                'development_exposure_inventory_sha256': exposure,
                'prediction_sha256': bundle.prediction_sha256, 'aggregate_sha256': bundle.aggregate_sha256,
                'evidence': {'events_sha256': bundle.events_sha256, 'started_sha256': bundle.started_sha256,
                             'terminal_sha256': bundle.terminal_sha256, 'audit_sha256': bundle.independent_audit_sha256,
                             'supervisor_sha256': bundle.supervisor_sha256,
                             'launcher_terminal_sha256': bundle.launcher_terminal_sha256,
                             'auditor_source_sha256': bundle.audit_source_sha256,
                             'builder_source_sha256': bundle.builder_sha256,
                             'clarification_sha256': bundle.clarification_sha256,
                             'tensor_files': files,
                             'private_layout': 'owner/attempt/{started.json,events.jsonl,terminal.json,tensors/}'}}
    output = Path(output_dir)
    output.mkdir(exist_ok=False)
    _write_new(output / 'runtime-binding.json', inputs['binding'])
    _write_new(output / 'natural-prediction-v2.json', bundle.prediction_bytes)
    _write_new(output / 'natural-prediction-freeze.json', bundle.aggregate_bytes)
    _write_new(output / 'stage.json', _canonical(metadata))
    return metadata


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError('duplicate JSON key')
        result[key] = value
    return result


def _decode(raw):
    def nonfinite(_):
        raise PublicationError('nonfinite JSON value')
    try:
        return json.loads(raw, object_pairs_hook=_json_object, parse_constant=nonfinite)
    except (TypeError, ValueError, RecursionError) as error:
        raise PublicationError('invalid JSON record') from error


def _read_file(path, limit):
    before = _regular(path)
    if before.st_size > limit:
        raise PublicationError('artifact exceeds its byte limit')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    after = _regular(path)
    if (len(raw) > limit or len(raw) != before.st_size or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise PublicationError('artifact changed while being read')
    return raw


def _read_stage(directory, expected_sha):
    if type(expected_sha) is not str or _HEX64.fullmatch(expected_sha) is None:
        raise PublicationError('stage digest is invalid')
    directory = Path(directory)
    raw = _read_file(directory / 'stage.json', 2 * 1024 * 1024)
    if _sha(raw) != expected_sha:
        raise PublicationError('stage changed after successful preparation')
    stage = _decode(raw)
    expected_keys = {'schema', 'wrapper_commit', 'source_commit', 'binding_commit', 'binding_path',
                     'prediction_path', 'aggregate_path', 'provenance_path', 'runtime_sha256',
                     'development_exposure_inventory_sha256', 'prediction_sha256', 'aggregate_sha256', 'evidence'}
    if (type(stage) is not dict or set(stage) != expected_keys or _canonical(stage) != raw
            or stage['schema'] != 'mlg-natural-publication-stage-v1'
            or any(type(stage[name]) is not str or _HEX40.fullmatch(stage[name]) is None
                   for name in ('wrapper_commit', 'source_commit', 'binding_commit'))
            or any(type(stage[name]) is not str or _HEX64.fullmatch(stage[name]) is None
                   for name in ('runtime_sha256', 'prediction_sha256', 'aggregate_sha256',
                                'development_exposure_inventory_sha256'))
            or stage['prediction_path'] != _PREDICTION or stage['aggregate_path'] != _AGGREGATE
            or stage['provenance_path'] != _PROVENANCE
            or parse_binding_url(raw_url(stage['binding_commit'], stage['binding_path']))
            != (stage['binding_commit'], stage['binding_path'])):
        raise PublicationError('stage metadata is invalid')
    binding = _read_file(directory / 'runtime-binding.json', 2 * 1024 * 1024)
    prediction = _read_file(directory / 'natural-prediction-v2.json', 8 * 1024 * 1024)
    aggregate = _read_file(directory / 'natural-prediction-freeze.json', 2 * 1024 * 1024)
    if (_sha(binding) != stage['runtime_sha256'] or _sha(prediction) != stage['prediction_sha256']
            or _sha(aggregate) != stage['aggregate_sha256']):
        raise PublicationError('staged artifact bytes do not match their digests')
    bound, predicted, frozen = (_decode(binding), _decode(prediction), _decode(aggregate))
    evidence = stage['evidence']
    digest_names = ('events_sha256', 'started_sha256', 'terminal_sha256', 'audit_sha256',
                    'supervisor_sha256', 'launcher_terminal_sha256', 'auditor_source_sha256',
                    'builder_source_sha256', 'clarification_sha256')
    if (type(evidence) is not dict or set(evidence) != set(digest_names) | {'tensor_files', 'private_layout'}
            or any(type(evidence[name]) is not str or _HEX64.fullmatch(evidence[name]) is None
                   for name in digest_names)
            or evidence['private_layout'] != 'owner/attempt/{started.json,events.jsonl,terminal.json,tensors/}'
            or type(evidence['tensor_files']) is not list or not 0 < len(evidence['tensor_files']) <= 4096
            or type(bound) is not dict or type(predicted) is not dict or type(frozen) is not dict
            or _canonical(predicted) != prediction or _canonical(frozen) != aggregate
            or bound.get('source_commit') != stage['source_commit']
            or bound.get('development_exposure_inventory_sha256') != stage['development_exposure_inventory_sha256']
            or bound.get('independent_natural_audit_source_sha256') != evidence['auditor_source_sha256']
            or bound.get('natural_prediction_clarification_sha256') != evidence['clarification_sha256']
            or type(bound.get('source_sha256')) is not dict
            or bound['source_sha256'].get('looking_glass/natural_prediction_freeze.py') != evidence['builder_source_sha256']
            or predicted.get('schema') != 'mlg-number-natural-predictions-v2'
            or predicted.get('source_commit') != stage['source_commit']
            or predicted.get('discovery_event_sha256') != evidence['events_sha256']
            or predicted.get('independent_audit_sha256') != evidence['audit_sha256']
            or set(frozen) != {'schema', 'protocol_sha256', 'checkpoint_revision', 'tracks'}
            or frozen.get('schema') != 'mlg-number-prediction-freezes-v1'
            or frozen.get('protocol_sha256') != bound.get('protocol_sha256')
            or frozen.get('checkpoint_revision') != bound.get('checkpoint_revision')
            or predicted.get('protocol_sha256') != bound.get('protocol_sha256')
            or predicted.get('checkpoint_revision') != bound.get('checkpoint_revision')
            or frozen.get('tracks') != {'natural': {'source_commit': stage['source_commit'],
                                                    'runtime_sha256': stage['runtime_sha256'],
                                                    'prediction_sha256': stage['prediction_sha256']}}):
        raise PublicationError('staged evidence and prediction links disagree')
    files = evidence['tensor_files']
    names = set()
    total = 0
    for row in files:
        if (type(row) is not dict or set(row) != {'name', 'bytes', 'sha256'}
                or type(row['name']) is not str or _TENSOR.fullmatch(row['name']) is None
                or type(row['sha256']) is not str or row['name'] != row['sha256'] + '.f32le'
                or type(row['bytes']) is not int or row['bytes'] < 0 or row['name'] in names):
            raise PublicationError('staged tensor inventory is invalid')
        names.add(row['name'])
        total += row['bytes']
    if total > 128 * 1024 * 1024:
        raise PublicationError('staged tensor inventory exceeds its reservation')
    return stage, raw, binding, prediction, aggregate


def _compare_url(before, after):
    if (type(before) is not str or _HEX40.fullmatch(before) is None
            or type(after) is not str or _HEX40.fullmatch(after) is None):
        raise PublicationError('publication commit is invalid')
    return _API + 'compare/' + before + '...' + after + '?per_page=1'


def _ancestor(before, after, *, distinct):
    if before == after:
        if distinct:
            raise PublicationError('publication phases need distinct commits')
        return
    record = _decode(_fetch(_compare_url(before, after), 8 * 1024 * 1024))
    if (type(record) is not dict or record.get('status') != 'ahead'
            or type(record.get('base_commit')) is not dict or record['base_commit'].get('sha') != before
            or type(record.get('merge_base_commit')) is not dict or record['merge_base_commit'].get('sha') != before):
        raise PublicationError('publication commits do not have the required ancestry')


def _main_head():
    record = _decode(_fetch(_HEAD_URL, 256 * 1024))
    if (type(record) is not dict or record.get('ref') != 'refs/heads/main'
            or type(record.get('object')) is not dict or record['object'].get('type') != 'commit'
            or type(record['object'].get('sha')) is not str or _HEX40.fullmatch(record['object']['sha']) is None):
        raise PublicationError('official main head is unavailable')
    return record['object']['sha']


def _match_remote(commit, path, expected):
    if _fetch(raw_url(commit, path), len(expected) + 1) != expected:
        raise PublicationError('published bytes differ from the staged artifact')


def verify_track_publication(directory, stage_sha256, prediction_commit, receipt_path):
    """Observe binding and prediction bytes on official main, then save a receipt."""
    stage, provenance, binding, prediction, _ = _read_stage(directory, stage_sha256)
    _ancestor(stage['source_commit'], stage['binding_commit'], distinct=True)
    _ancestor(stage['binding_commit'], prediction_commit, distinct=True)
    _ancestor(stage['wrapper_commit'], prediction_commit, distinct=False)
    head = _main_head()
    _ancestor(prediction_commit, head, distinct=False)
    _match_remote(stage['binding_commit'], stage['binding_path'], binding)
    for commit in dict.fromkeys((prediction_commit, head)):
        _match_remote(commit, stage['binding_path'], binding)
        _match_remote(commit, stage['prediction_path'], prediction)
        _match_remote(commit, stage['provenance_path'], provenance)
    if _fetch(raw_url(prediction_commit, stage['aggregate_path']), 1, allow_missing=True) is not None:
        raise PublicationError('aggregate path was already present at the prediction commit')
    if _fetch(raw_url(head, stage['aggregate_path']), 1, allow_missing=True) is not None:
        raise PublicationError('aggregate path was already present at the observed main head')
    receipt = {'schema': 'mlg-natural-track-publication-v1', 'stage_sha256': stage_sha256,
               'binding_commit': stage['binding_commit'], 'prediction_commit': prediction_commit,
               'observed_main_head': head, 'runtime_sha256': stage['runtime_sha256'],
               'prediction_sha256': stage['prediction_sha256'], 'provenance_sha256': _sha(provenance)}
    _write_new(Path(receipt_path), _canonical(receipt))
    return receipt


def verify_aggregate_publication(directory, stage_sha256, aggregate_commit, track_receipt_path, receipt_path):
    """Observe a later aggregate on official main after the track receipt."""
    stage, provenance, binding, prediction, aggregate = _read_stage(directory, stage_sha256)
    track_raw = _read_file(Path(track_receipt_path), 256 * 1024)
    track = _decode(track_raw)
    if (type(track) is not dict or _canonical(track) != track_raw
            or set(track) != {'schema', 'stage_sha256', 'binding_commit', 'prediction_commit',
                              'observed_main_head', 'runtime_sha256', 'prediction_sha256', 'provenance_sha256'}
            or track['schema'] != 'mlg-natural-track-publication-v1' or track['stage_sha256'] != stage_sha256
            or track['binding_commit'] != stage['binding_commit'] or track['runtime_sha256'] != stage['runtime_sha256']
            or track['prediction_sha256'] != stage['prediction_sha256'] or track['provenance_sha256'] != _sha(provenance)
            or type(track['prediction_commit']) is not str or _HEX40.fullmatch(track['prediction_commit']) is None
            or type(track['observed_main_head']) is not str or _HEX40.fullmatch(track['observed_main_head']) is None):
        raise PublicationError('track publication receipt does not match the stage')
    _ancestor(stage['source_commit'], stage['binding_commit'], distinct=True)
    _ancestor(stage['binding_commit'], track['prediction_commit'], distinct=True)
    _ancestor(stage['wrapper_commit'], track['prediction_commit'], distinct=False)
    _ancestor(track['prediction_commit'], track['observed_main_head'], distinct=False)
    _ancestor(track['observed_main_head'], aggregate_commit, distinct=True)
    _ancestor(track['prediction_commit'], aggregate_commit, distinct=True)
    head = _main_head()
    _ancestor(aggregate_commit, head, distinct=False)
    _match_remote(stage['binding_commit'], stage['binding_path'], binding)
    for commit in dict.fromkeys((track['prediction_commit'], track['observed_main_head'], aggregate_commit, head)):
        _match_remote(commit, stage['binding_path'], binding)
        _match_remote(commit, stage['prediction_path'], prediction)
        _match_remote(commit, stage['provenance_path'], provenance)
    if _fetch(raw_url(track['prediction_commit'], stage['aggregate_path']), 1, allow_missing=True) is not None:
        raise PublicationError('aggregate path was already present at the prediction commit')
    if _fetch(raw_url(track['observed_main_head'], stage['aggregate_path']), 1, allow_missing=True) is not None:
        raise PublicationError('aggregate path was already present at the observed main head')
    for commit in dict.fromkeys((aggregate_commit, head)):
        _match_remote(commit, stage['aggregate_path'], aggregate)
    receipt = {'schema': 'mlg-natural-aggregate-publication-v1', 'stage_sha256': stage_sha256,
               'track_receipt_sha256': _sha(track_raw), 'binding_commit': stage['binding_commit'],
               'prediction_commit': track['prediction_commit'], 'aggregate_commit': aggregate_commit,
               'observed_main_head': head, 'runtime_sha256': stage['runtime_sha256'],
               'prediction_sha256': stage['prediction_sha256'], 'aggregate_sha256': stage['aggregate_sha256'],
               'development_exposure_inventory_sha256': stage['development_exposure_inventory_sha256']}
    _write_new(Path(receipt_path), _canonical(receipt))
    return receipt
