"""Bounded, byte-linked records for the prospective masked-color study."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import struct

from . import distilbert_color_edit_metrics_v1 as metrics

MAX_BYTES = 128 * 1024 * 1024
MAX_FORWARDS = 256


def encode_raw(raw):
    if type(raw) is not bytes:
        raise TypeError('raw tensor must be bytes')
    return {'encoding': 'f32le', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'data': base64.b64encode(raw).decode('ascii')}


def raw_f32_bytes(value, count):
    metrics.require_denormals()
    if (type(value) is not dict or set(value) != {'encoding', 'bytes', 'sha256', 'data'}
            or value['encoding'] != 'f32le' or type(value['bytes']) is not int
            or value['bytes'] != 4 * count or type(value['data']) is not str
            or type(value['sha256']) is not str):
        raise ValueError('invalid f32 evidence envelope')
    try:
        raw = base64.b64decode(value['data'], validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('invalid base64 evidence') from error
    if len(raw) != 4 * count or hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise ValueError('saved f32 bytes do not match their length or digest')
    metrics.require_denormals()
    return raw


def decode_raw(value, count):
    raw = raw_f32_bytes(value, count)
    values = struct.unpack('<' + str(count) + 'f', raw)
    if any(not math.isfinite(item) for item in values):
        raise ValueError('saved f32 values must be finite')
    metrics.require_denormals()
    return values


class Journal:
    """Create-only JSONL writer with complete-write and started-forward caps."""

    def __init__(self, path, *, byte_cap=MAX_BYTES, forward_cap=MAX_FORWARDS):
        if (type(byte_cap) is not int or not 1 <= byte_cap <= MAX_BYTES or
                type(forward_cap) is not int or not 0 <= forward_cap <= MAX_FORWARDS):
            raise ValueError('invalid evidence budget')
        self._stream = Path(path).open('xb', buffering=0)
        self.byte_cap, self.forward_cap = byte_cap, forward_cap
        self.bytes = self.forwards = self.sequence = 0
        self.failed = False

    def emit(self, record):
        if self.failed or self._stream is None or type(record) is not dict:
            raise RuntimeError('journal unavailable')
        if 'seq' in record:
            raise ValueError('journal assigns sequence numbers')
        item = {'seq': self.sequence, **record}
        raw = (json.dumps(item, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
        if self.bytes + len(raw) > self.byte_cap:
            self.failed = True
            raise ValueError('evidence byte budget exceeded')
        try:
            if self._stream.write(raw) != len(raw):
                raise OSError('short journal write')
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            self.failed = True
            raise
        self.bytes += len(raw)
        self.sequence += 1

    def start(self, record):
        if self.forwards >= self.forward_cap:
            raise ValueError('started-forward budget exceeded')
        if 'kind' in record:
            raise ValueError('start kind is assigned by journal')
        self.emit({'kind': 'start', **record})
        self.forwards += 1

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def parse_jsonl(raw):
    if type(raw) is not bytes or not raw.endswith(b'\n'):
        raise ValueError('journal must be complete newline-terminated bytes')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    result = []
    for sequence, line in enumerate(raw.splitlines()):
        item = json.loads(line, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if type(item) is not dict or type(item.get('seq')) is not int or item['seq'] != sequence:
            raise ValueError('journal sequence is invalid')
        result.append(item)
    return result


def rebuild_screen(manifest, events, *, vocabulary=30522):
    """Recompute the first-eligible choice from all nine complete raw rows."""
    order = manifest['candidate_order']
    if type(order) is not list or len(order) != 3 or len(set(order)) != 3:
        raise ValueError('fixed three-object order required')
    rows = []
    for obj in order:
        for role in ('discovery', 'calibration-01', 'calibration-02'):
            matching = [row for row in manifest['cases'] if row['object'] == obj and row['role'] == role]
            if len(matching) != 1:
                raise ValueError('missing or duplicate fixed screen case')
            rows.append(matching[0])
    if len(events) != 18:
        raise ValueError('exactly nine completed screen actions are required')
    decisions, candidate = [], {obj: True for obj in order}
    for index, row in enumerate(rows):
        start, done = events[2 * index:2 * index + 2]
        binding = {'action': index, 'phase': 'screen', 'object': row['object'],
                   'case_id': row['id'], 'condition': 'original'}
        if (type(start.get('seq')) is not int or type(done.get('seq')) is not int or
                type(start.get('action')) is not int or type(done.get('action')) is not int or
                any(type(event.get(key)) is not list for event in (start, done)
                    for key in ('input_ids', 'attention_mask')) or
                any(type(item) is not int for event in (start, done)
                    for key in ('input_ids', 'attention_mask') for item in event[key]) or
                any(type(event.get('mask_index')) is not int for event in (start, done)) or
                start.get('seq') != 2 * index or done.get('seq') != 2 * index + 1 or
                start.get('kind') != 'start' or done.get('kind') != 'complete' or
                not {'parameter', 'parameter_sha256', 'realized_norm'}.issubset(done) or
                done.get('parameter') is not None or done.get('parameter_sha256') is not None or
                type(done.get('realized_norm')) is not float or done['realized_norm'] != 0.0 or
                any(start.get(k) != v or done.get(k) != v for k, v in binding.items()) or
                any(event.get(k) != row[k] for event in (start, done)
                    for k in ('input_ids', 'attention_mask', 'mask_index'))):
            raise ValueError('screen action or input binding differs')
        values = decode_raw(done.get('logits'), vocabulary)
        factual = row['word_ids'][row['factual']]
        purple = row['word_ids'][row['counterfactual']]
        blue = row['word_ids'][row['wrong_target']]
        info = metrics.describe(values, (factual, purple, blue))
        top_logp = metrics.log_probabilities(values)[info['top_representative']]
        eligible = info['unique_top'] == factual
        candidate[row['object']] &= eligible
        decisions.append({'case_id': row['id'], 'object': row['object'], 'eligible': eligible,
                          'unique_top': info['unique_top'], 'top_tie_count': len(info['top_ids']),
                          'top_representative': info['top_representative'],
                          'top_gap': info['top_gap'], 'top_logit': values[info['top_representative']],
                          'top_log_probability': top_logp, 'top_probability': math.exp(top_logp),
                          'factual_rank': info['ranks'][str(factual)],
                          'factual_log_probability': info['log_probabilities'][str(factual)],
                          'ranks': info['ranks'], 'logits': info['logits'],
                          'log_probabilities': info['log_probabilities'], 'probabilities': info['probabilities']})
    selected = next((obj for obj in order if candidate[obj]), None)
    return {'status': 'eligible' if selected is not None else 'baseline-ineligible',
            'selected_object': selected, 'candidate_eligible': candidate, 'rows': decisions}
