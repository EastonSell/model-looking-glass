"""Service-side sampled evidence gates for compatible grouped forwards.

This is a trusted coordinator component, not model authentication, memory
admission, a scheduler, or proof of separability for arbitrary future inputs.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

import torch

from . import batch_validation
from .resident_worker import ForwardRequest


_MAX_EVIDENCE_BYTES = 5 * 1024 * 1024
_DTYPES = {'bool': torch.bool, 'int32': torch.int32, 'int64': torch.int64}


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('expected finite JSON data') from exc


def _snapshot(value, label):
    if type(value) is not dict:
        raise ValueError(f'{label} must be an exact dictionary')
    payload = _canonical(value)
    if len(payload) > _MAX_EVIDENCE_BYTES:
        raise ValueError(f'{label} exceeds the metadata size limit')
    return json.loads(payload)


def runtime_fingerprint(manifest):
    """Hash a trusted live runtime manifest using the validator's JSON form.

    The service owns collection, completeness and exclusive snapshotting of
    this manifest; this helper does not inspect or attest a live device.
    """
    if type(manifest) is not dict or not manifest:
        raise ValueError('a nonempty trusted runtime manifest is required')
    payload = _canonical(manifest)
    if len(payload) > 64 * 1024:
        raise ValueError('runtime manifest exceeds the metadata size limit')
    if json.loads(payload) != manifest:
        raise ValueError('runtime manifest must retain its exact JSON structure')
    return hashlib.sha256(payload).hexdigest()


def _tolerance_limit(value, label):
    if type(value) not in (int, float):
        raise ValueError(f'{label} must be a finite nonnegative number')
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f'{label} must be a finite nonnegative number') from exc
    if not math.isfinite(converted) or converted < 0 or converted != value:
        raise ValueError(f'{label} must be exactly representable as finite nonnegative binary64')
    return converted


@dataclass(frozen=True, slots=True, init=False)
class CertifiedBatchPolicy:
    """Freeze and verify a service-owned policy for each actually tested size.

    The coordinator supplies externally retained protocol and evidence hashes
    as well as independently verified live model identity. The protocol and
    report must be exclusively owned while their snapshots are constructed.
    """

    _binding_json: bytes
    _allowed_counts: tuple
    _tested_rows: tuple
    _coverage: tuple
    _key: bytes

    def __init__(self, profile, protocol, report, *, expected_protocol_sha256,
                 evidence_sha256, minimum_groups, max_atol, max_rtol, model_fingerprint,
                 checkpoint_sha256, adapter_sha256, backend, runtime_manifest):
        if type(minimum_groups) is not int or not 2 <= minimum_groups <= 64:
            raise ValueError('minimum_groups must be an integer from 2 to 64')
        if (type(expected_protocol_sha256) is not str or type(evidence_sha256) is not str
                or expected_protocol_sha256 == evidence_sha256):
            raise ValueError('distinct externally retained protocol and evidence hashes are required')
        actual = batch_validation.binding_from_profile(
            profile, model_fingerprint=model_fingerprint,
            checkpoint_sha256=checkpoint_sha256, adapter_sha256=adapter_sha256,
            backend=backend)
        atol_limit = _tolerance_limit(max_atol, 'max_atol')
        rtol_limit = _tolerance_limit(max_rtol, 'max_rtol')
        if actual['serial_atol'] > atol_limit or actual['serial_rtol'] > rtol_limit:
            raise ValueError('sampled tolerances exceed the frozen service limits')
        self._require_runtime(actual, runtime_manifest)
        frozen_protocol = _snapshot(protocol, 'protocol')
        frozen_report = _snapshot(report, 'report')
        tested = frozen_report.get('tested_rows')
        if type(tested) is not list or not tested or any(type(row) is not int for row in tested):
            raise ValueError('evidence must contain tested row counts')

        rows_per_request = actual['input_shape'][0]
        approved = []
        coverage = []
        for rows in tested:
            batch_validation.require_evidence(
                frozen_report, protocol=frozen_protocol, actual_binding=actual,
                expected_protocol_sha256=expected_protocol_sha256,
                evidence_sha256=evidence_sha256, rows=rows)
            if rows % rows_per_request or rows // rows_per_request < 2:
                continue
            groups = [group for group in frozen_report['groups']
                      if len(group['cases']) * rows_per_request == rows
                      and all(case['status'] == 'matched' for case in group['cases'])]
            unique_groups = {tuple(tuple(case['row_sha256']) for case in group['cases'])
                             for group in groups}
            row_hashes = {digest for group in groups for case in group['cases']
                          for digest in case['row_sha256']}
            owners = {case['owner'] for group in groups for case in group['cases']}
            if (len(groups) >= minimum_groups and len(unique_groups) >= minimum_groups
                    and len(row_hashes) >= rows * minimum_groups and len(owners) >= 2):
                approved.append(rows)
                coverage.append((rows, len(groups), len(row_hashes), len(owners)))
        if not approved:
            raise ValueError('no tested batch size satisfies the frozen evidence coverage')
        approved = tuple(sorted(set(approved)))
        binding_json = _canonical(actual)
        key = hashlib.sha256(_canonical({
            'binding': actual, 'protocol_sha256': expected_protocol_sha256,
            'evidence_sha256': evidence_sha256, 'minimum_groups': minimum_groups,
            'max_atol': atol_limit, 'max_rtol': rtol_limit, 'admitted_rows': list(approved),
        })).digest()
        object.__setattr__(self, '_binding_json', binding_json)
        object.__setattr__(self, '_allowed_counts', tuple(row // rows_per_request for row in approved))
        object.__setattr__(self, '_tested_rows', approved)
        object.__setattr__(self, '_coverage', tuple(sorted(coverage)))
        object.__setattr__(self, '_key', key)

    @property
    def allowed_request_counts(self):
        return self._allowed_counts

    @property
    def tested_rows(self):
        """Row counts satisfying both evidence and this policy's coverage."""
        return self._tested_rows

    @property
    def compatibility_key(self):
        """Opaque queue key; trusted code must still revalidate each job."""
        return self._key

    def coverage(self, rows):
        if type(rows) is not int:
            raise ValueError('rows must be an admitted integer row count')
        for candidate, groups, unique_rows, owners in self._coverage:
            if rows == candidate:
                return groups, unique_rows, owners
        raise ValueError('row count is not admitted')

    def require_count(self, count):
        if type(count) is not int or count not in self._allowed_counts:
            raise ValueError('request count is not admitted by sampled evidence')

    @staticmethod
    def _require_runtime(binding, runtime_manifest):
        if binding['runtime_id'] != 'sha256:' + runtime_fingerprint(runtime_manifest):
            raise ValueError('binding must contain the verified execution-environment digest')

    def require_live_binding(self, profile, *, model_fingerprint, checkpoint_sha256,
                             adapter_sha256, backend, runtime_manifest):
        live = batch_validation.binding_from_profile(
            profile, model_fingerprint=model_fingerprint,
            checkpoint_sha256=checkpoint_sha256, adapter_sha256=adapter_sha256,
            backend=backend)
        self._require_runtime(live, runtime_manifest)
        if _canonical(live) != self._binding_json:
            raise ValueError('live execution binding differs from sampled evidence')

    def require_request(self, request):
        """Check a cooperative snapshot; recheck it at execution handoff."""
        binding = json.loads(self._binding_json)
        if (type(request) is not ForwardRequest or type(request.request_id) is not str
                or not request.request_id or len(request.request_id) > 256):
            raise ValueError('an exact identified ForwardRequest is required')
        if type(request.seed) is not int or request.seed != binding['seed']:
            raise ValueError('request seed differs from the binding')
        inputs = request.inputs
        if type(inputs) not in (dict, MappingProxyType):
            raise ValueError('request inputs must be a trusted plain mapping or snapshot proxy')
        if set(inputs) != {name for name, _ in binding['input_schema']}:
            raise ValueError('request input names differ from the binding')
        for name, dtype in binding['input_schema']:
            value = inputs[name]
            if (type(value) is not torch.Tensor or value.device.type != 'cpu'
                    or value.layout != torch.strided or value.dtype != _DTYPES[dtype]
                    or list(value.shape) != binding['input_shape'] or value.requires_grad):
                raise ValueError(f'request tensor differs from the binding: {name}')
