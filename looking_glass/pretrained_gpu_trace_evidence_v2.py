"""Bounded, create-only CPU tensor evidence for the fixed trace successor.

The caller must already have synchronized the producing device operation and
retained its roots. This writer does not authorize or synchronize CUDA work.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import struct

from .bert_gpu_evidence import write_json_new


MAX_LOGIT_BYTES = 1_005_140
MAX_EVENT_BYTES = 6_144
MAX_PAYLOAD_BYTES = 10 * MAX_LOGIT_BYTES + 54 * MAX_EVENT_BYTES
_PHASES = ('baseline_cold', 'baseline_warm', 'reference', 'reference', 'direct',
           'direct', 'gradient', 'gradient', 'id', 'id')
_CAPTURE_AFTER = {3: ('reference', 0), 4: ('reference', 1),
                  5: ('direct', 0), 6: ('direct', 1),
                  7: ('gradient', 0), 8: ('gradient', 1)}
_LAYERS = (0, 2, 5)
_KINDS = ('attn', 'mlp', 'block')


def _publish_new(path, data):
    """Retain a partial on failure; a final name is linked only after fsync."""
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    if os.path.lexists(path):
        raise FileExistsError('evidence destination already exists')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0)
    fd = os.open(partial, flags, 0o600)
    try:
        remaining = memoryview(data)
        while remaining:
            count = os.write(fd, remaining)
            if type(count) is not int or not 0 < count <= len(remaining):
                raise OSError('evidence write made invalid progress')
            remaining = remaining[count:]
        os.fsync(fd)
    except BaseException as original:
        try:
            os.close(fd)
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)) and not isinstance(
                    original, (KeyboardInterrupt, SystemExit)):
                raise cleanup from original
            if isinstance(cleanup, Exception):
                original.add_note('Closing the partial evidence file also failed')
        raise
    os.close(fd)
    os.link(partial, path)
    os.unlink(partial)
    if os.path.lexists(partial):
        raise OSError('evidence partial name remains after publication')


class TensorEvidence:
    """Write the fixed schedule once; any uncertain transition is terminal."""

    def __init__(self, directory, *, torch, hidden=768, vocab=50257):
        if (type(hidden) is not int or type(vocab) is not int or hidden < 1 or vocab < 1
                or 2 * hidden * 4 > MAX_EVENT_BYTES or 5 * vocab * 4 > MAX_LOGIT_BYTES):
            raise ValueError('tensor geometry exceeds the fixed evidence allowance')
        self.directory = Path(directory)
        if (not self.directory.is_absolute() or not self.directory.is_dir()
                or self.directory.resolve(strict=True) != self.directory
                or any(self.directory.iterdir())):
            raise ValueError('a fresh evidence directory is required')
        self.torch, self.hidden, self.vocab = torch, hidden, vocab
        self.expected_bytes = 10 * 5 * vocab * 4 + 54 * 2 * hidden * 4
        if self.expected_bytes > MAX_PAYLOAD_BYTES:
            raise ValueError('tensor payload exceeds its allowance')
        self.logits, self.captures = [], []
        self.payload_bytes, self.pending_capture = 0, None
        self.failed, self.finished = False, False

    def _array_bytes(self, value, shape):
        torch = self.torch
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.float32 or tuple(value.shape) != shape
                or value.numel() * value.element_size() != 4 * _product(shape)):
            raise ValueError('CPU float32 tensor geometry differs')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('tensor evidence must be finite')
        data = value.detach().contiguous().numpy().astype('<f4', copy=False).tobytes()
        if len(data) != 4 * _product(shape):
            raise ValueError('serialized tensor length differs')
        return data

    def _check_open(self):
        if self.failed or self.finished:
            raise RuntimeError('tensor evidence writer is no longer usable')

    def write_logit(self, index, phase, value):
        try:
            self._check_open()
            if (type(index) is not int or index != len(self.logits) + 1 or index > 10
                    or type(phase) is not str or phase != _PHASES[index - 1]
                    or self.pending_capture is not None):
                raise ValueError('logit evidence order differs')
            data = self._array_bytes(value, (1, 5, self.vocab))
            name = f'logits-{index:03d}.f32'
            _publish_new(self.directory / name, data)
            self.logits.append({'index': index, 'phase': phase, 'file': name,
                                'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            self.payload_bytes += len(data)
            self.pending_capture = _CAPTURE_AFTER.get(index)
        except BaseException:
            self.failed = True
            raise

    def write_capture(self, mode, forward_index, values):
        try:
            self._check_open()
            if (type(mode) is not str or type(forward_index) is not int
                    or (mode, forward_index) != self.pending_capture
                    or type(values) is not list or len(values) != 9):
                raise ValueError('capture evidence order differs')
            pieces = [self._array_bytes(value, (1, 2, self.hidden)) for value in values]
            size = 2 * self.hidden * 4
            events = []
            for index, piece in enumerate(pieces):
                layer, kind = _LAYERS[index // 3], _KINDS[index % 3]
                path = f'transformer.h.{layer}' + ('' if kind == 'block' else '.' + kind)
                events.append({'sequence': forward_index * 9 + index, 'forward_index': forward_index,
                    'layer_index': layer, 'module_kind': kind, 'module_path': path,
                    'output_path': [0] if kind in ('attn', 'block') else [],
                    'token_positions': [0, 1], 'offset': index * size, 'length': size,
                    'sha256': hashlib.sha256(piece).hexdigest()})
            data = b''.join(pieces)
            name = f'{mode}-{forward_index + 1}.f32'
            _publish_new(self.directory / name, data)
            self.captures.append({'mode': mode, 'forward_index': forward_index, 'file': name,
                                  'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                  'events': events})
            self.payload_bytes += len(data)
            self.pending_capture = None
        except BaseException:
            self.failed = True
            raise

    def finish(self):
        try:
            self._check_open()
            if (len(self.logits) != 10 or len(self.captures) != 6 or self.pending_capture is not None
                    or self.payload_bytes != self.expected_bytes):
                raise RuntimeError('tensor evidence inventory is incomplete')
            manifest = {'schema': 'pretrained-gpu-trace-tensor-evidence-v2',
                        'encoding': 'float32-little-endian', 'hidden': self.hidden, 'vocab': self.vocab,
                        'payload_bytes': self.payload_bytes, 'logits': self.logits, 'captures': self.captures}
            destination = self.directory / 'manifest.json'
            write_json_new(destination, manifest, max_bytes=65536)
            if os.path.lexists(destination.with_name(destination.name + '.partial')):
                raise OSError('manifest partial name remains after publication')
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            self.finished = True
            return {'manifest_sha256': digest, 'payload_bytes': self.payload_bytes}
        except BaseException:
            self.failed = True
            raise


def _product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_exact(path, size):
    with Path(path).open('rb') as stream:
        data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError('saved tensor length differs')
    return data


def _read_manifest(directory, binding):
    if (type(binding) is not dict or set(binding) != {'manifest_sha256', 'payload_bytes'}
            or type(binding['manifest_sha256']) is not str or len(binding['manifest_sha256']) != 64
            or any(char not in '0123456789abcdef' for char in binding['manifest_sha256'])
            or type(binding['payload_bytes']) is not int):
        raise ValueError('invalid evidence binding')
    with (directory / 'manifest.json').open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != binding['manifest_sha256']:
        raise ValueError('evidence manifest digest or size differs')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate manifest key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _finite(data):
    return all(math.isfinite(value) for (value,) in struct.iter_unpack('<f', data))


def validate_tensor_evidence(directory, binding, logit_hashes, result, *, hidden=768, vocab=50257):
    """Recompute the fixed comparisons from retained bytes, bound to saved records.

    This checks internal consistency; the bytes do not authenticate execution.
    """
    directory = Path(directory)
    if (type(hidden) is not int or type(vocab) is not int or hidden < 1 or vocab < 1
            or 2 * hidden * 4 > MAX_EVENT_BYTES or 5 * vocab * 4 > MAX_LOGIT_BYTES):
        raise ValueError('invalid evidence geometry')
    manifest = _read_manifest(directory, binding)
    payload = 10 * 5 * vocab * 4 + 54 * 2 * hidden * 4
    if (type(manifest) is not dict or set(manifest) != {'schema', 'encoding', 'hidden', 'vocab',
            'payload_bytes', 'logits', 'captures'} or manifest['schema'] != 'pretrained-gpu-trace-tensor-evidence-v2'
            or manifest['encoding'] != 'float32-little-endian' or type(manifest['hidden']) is not int
            or type(manifest['vocab']) is not int or manifest['hidden'] != hidden or manifest['vocab'] != vocab
            or type(manifest['payload_bytes']) is not int or manifest['payload_bytes'] != payload
            or binding['payload_bytes'] != payload or payload > MAX_PAYLOAD_BYTES
            or type(manifest['logits']) is not list or len(manifest['logits']) != 10
            or type(manifest['captures']) is not list or len(manifest['captures']) != 6
            or type(logit_hashes) is not list or len(logit_hashes) != 10):
        raise ValueError('evidence manifest geometry or inventory differs')
    expected_names = {'manifest.json'} | {f'logits-{i:03d}.f32' for i in range(1, 11)}
    expected_names |= {f'{mode}-{forward + 1}.f32' for mode in ('reference', 'direct', 'gradient')
                       for forward in range(2)}
    if {path.name for path in directory.iterdir()} != expected_names:
        raise ValueError('evidence directory inventory differs')
    baseline = None
    for index, entry in enumerate(manifest['logits'], 1):
        name = f'logits-{index:03d}.f32'
        if (type(entry) is not dict or set(entry) != {'index', 'phase', 'file', 'bytes', 'sha256'}
                or type(entry['index']) is not int or entry['index'] != index or entry['phase'] != _PHASES[index - 1]
                or entry['file'] != name or type(entry['bytes']) is not int or entry['bytes'] != 5 * vocab * 4
                or type(entry['sha256']) is not str or entry['sha256'] != logit_hashes[index - 1]):
            raise ValueError('saved logit metadata differs')
        data = _read_exact(directory / name, 5 * vocab * 4)
        if hashlib.sha256(data).hexdigest() != entry['sha256'] or not _finite(data):
            raise ValueError('saved logit digest or finite check differs')
        if baseline is None:
            baseline = data
        elif data != baseline:
            raise ValueError('saved logit bits differ')
    try:
        if (result['baseline_logit_sha256'] != logit_hashes[:2]
                or result['reference_logit_sha256'] != logit_hashes[2:4]
                or result['modes']['direct']['logit_sha256'] != logit_hashes[4:6]
                or result['modes']['gradient']['logit_sha256'] != logit_hashes[6:8]
                or result['modes']['id']['logit_sha256'] != logit_hashes[8:10]
                or len(result['reference_value_sha256']) != 18
                or any(len(result['modes'][mode]['events']) != 18 for mode in ('direct', 'gradient', 'id'))
                or any(event['value_sha256'] is not None for event in result['modes']['id']['events'])):
            raise ValueError('saved result inventory differs')
        reference = []
        for entry, (mode, forward) in zip(manifest['captures'],
                ((mode, forward) for mode in ('reference', 'direct', 'gradient') for forward in range(2))):
            size = 2 * hidden * 4
            name = f'{mode}-{forward + 1}.f32'
            if (type(entry) is not dict or set(entry) != {'mode', 'forward_index', 'file', 'bytes', 'sha256', 'events'}
                    or entry['mode'] != mode or type(entry['forward_index']) is not int
                    or entry['forward_index'] != forward or entry['file'] != name
                    or type(entry['bytes']) is not int or entry['bytes'] != 9 * size
                    or type(entry['sha256']) is not str or type(entry['events']) is not list
                    or len(entry['events']) != 9):
                raise ValueError('saved capture metadata differs')
            data = _read_exact(directory / name, 9 * size)
            if hashlib.sha256(data).hexdigest() != entry['sha256'] or not _finite(data):
                raise ValueError('saved capture digest or finite check differs')
            for index, event in enumerate(entry['events']):
                layer, kind = _LAYERS[index // 3], _KINDS[index % 3]
                path = f'transformer.h.{layer}' + ('' if kind == 'block' else '.' + kind)
                expected = {'sequence': forward * 9 + index, 'forward_index': forward,
                    'layer_index': layer, 'module_kind': kind, 'module_path': path,
                    'output_path': [0] if kind in ('attn', 'block') else [],
                    'token_positions': [0, 1], 'offset': index * size, 'length': size}
                if (type(event) is not dict or set(event) != set(expected) | {'sha256'}
                        or any(event[key] != value for key, value in expected.items())
                        or any(type(event[key]) is not int for key in
                               ('sequence', 'forward_index', 'layer_index', 'offset', 'length'))
                        or any(type(x) is not int for x in event['output_path'] + event['token_positions'])):
                    raise ValueError('capture event coordinates differ')
                piece = data[index * size:(index + 1) * size]
                digest = hashlib.sha256(piece).hexdigest()
                sequence = forward * 9 + index
                if event['sha256'] != digest:
                    raise ValueError('saved event digest differs')
                if mode == 'reference':
                    if result['reference_value_sha256'][sequence] != digest:
                        raise ValueError('reference event binding differs')
                    reference.append(piece)
                else:
                    saved = result['modes'][mode]['events'][sequence]
                    metadata = saved['metadata']
                    if (type(metadata) is not dict or saved['value_sha256'] != digest
                            or any(metadata[key] != expected[key] for key in
                                   ('sequence', 'forward_index', 'layer_index', 'module_kind', 'module_path',
                                    'output_path', 'token_positions'))
                            or any(type(metadata[key]) is not int for key in
                                   ('sequence', 'forward_index', 'layer_index'))
                            or type(metadata['output_path']) is not list
                            or type(metadata['token_positions']) is not list
                            or any(type(x) is not int for x in metadata['output_path'] + metadata['token_positions'])):
                        raise ValueError('captured event binding differs')
                    if piece != reference[sequence]:
                        raise ValueError('saved capture bits differ')
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('saved result structure differs') from exc
    return {'forward_count': 10, 'capture_event_count': 54, 'logit_bit_comparisons': 9,
            'capture_bit_comparisons': 36, 'payload_bytes': payload}
