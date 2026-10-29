"""Fixed input fixtures for controlled studies and pretrained-model smoke checks.

The public evaluation split is not blind; see docs/RESEARCH_CASES.md before use.
This module only defines inputs and integrity checks. It does not run an experiment.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import struct


SYNTHETIC_CASES_VERSION = 'synthetic-injection-inputs-v1'
PRETRAINED_SMOKE_VERSION = 'pretrained-smoke-inputs-v1'


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    split: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    text: str


# Fixed literals are authoritative; their generation procedure is in the docs.
_DISCOVERY_VALUES = (
    (-0.125, -0.375, 0.25, 0.0, -0.5, -0.625, -0.375, 0.9375),
    (-0.5, -0.9375, -0.9375, -0.0625, 0.0, 0.5, 0.6875, -0.125),
    (0.125, 0.9375, 0.375, 1.0, 0.5, -1.0, 0.8125, -0.3125),
    (-0.75, 0.25, -0.8125, 0.125, -0.8125, -0.1875, -0.875, -0.375),
    (0.4375, -0.4375, 0.1875, -0.3125, -0.25, 0.9375, 0.5, -0.125),
    (-0.4375, -1.0, 0.4375, 0.5625, 0.9375, 0.625, 0.875, -0.1875),
    (0.4375, 0.6875, -1.0, 0.5, 0.125, 0.8125, 0.875, -0.8125),
    (-0.0625, 0.0625, -0.375, 1.0, 0.4375, 0.9375, -0.25, 0.875),
)
_HELDOUT_VALUES = (
    (0.0625, -0.625, 0.0625, 0.6875, -0.6875, 0.8125, -0.9375, -0.75),
    (0.5625, -0.25, -0.25, -0.6875, 0.5625, -0.6875, -0.0625, -0.625),
    (0.875, 0.5, -0.1875, -0.4375, -0.75, -0.375, -0.3125, 0.625),
    (0.25, 0.75, -0.875, -0.8125, 1.0, -0.5, 0.5, 0.5625),
    (-0.3125, 0.25, 0.25, 1.0, -0.25, 0.875, 1.0, 0.8125),
    (0.625, 0.375, 0.1875, -0.25, -0.9375, -0.9375, -0.25, -0.25),
    (0.9375, -0.5625, 0.8125, -0.75, 0.125, -0.3125, 0.8125, -0.0625),
    (-0.8125, -0.8125, -0.125, -0.9375, 0.0625, 0.625, -1.0, 0.3125),
    (-0.375, 0.6875, 0.5, 0.3125, 0.6875, 0.8125, 0.75, 0.125),
    (-0.75, 0.8125, 0.1875, -0.5, 0.5625, -0.125, -0.1875, -0.75),
    (-0.625, 0.8125, 0.0625, -1.0, -0.75, -0.4375, -0.375, 0.4375),
    (-0.75, -0.6875, 0.625, 0.75, 0.1875, 0.4375, 0.1875, -0.875),
    (1.0, 0.0, 1.0, -0.5, 0.625, -0.8125, -0.375, 0.3125),
    (0.25, -0.0625, -0.1875, 0.0, 1.0, 0.9375, -0.5625, -0.9375),
    (-0.125, -0.0625, 0.5, -0.125, -0.625, -0.125, 0.9375, 0.0),
    (0.375, -0.875, 0.6875, 0.625, 1.0, 0.5, -0.3125, -0.4375),
)

DISCOVERY_CASES = tuple(SyntheticCase(f'synth-v1-discovery-{i:02}', 'discovery', values)
                        for i, values in enumerate(_DISCOVERY_VALUES, start=1))
HELDOUT_CASES = tuple(SyntheticCase(f'synth-v1-heldout-{i:02}', 'heldout', values)
                      for i, values in enumerate(_HELDOUT_VALUES, start=1))
SYNTHETIC_CASES = DISCOVERY_CASES + HELDOUT_CASES

PRETRAINED_SMOKE_CASES = (
    PromptCase('smoke-v1-01', 'The capital of France is'),
    PromptCase('smoke-v1-02', 'One, two, three, four,'),
    PromptCase('smoke-v1-03', 'A robin is a kind of'),
    PromptCase('smoke-v1-04', 'The opposite of cold is'),
    PromptCase('smoke-v1-05', 'In the morning I drink a cup of'),
    PromptCase('smoke-v1-06', 'When it rains, people use an'),
)


def _id(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('case ID must be a nonempty string without outer whitespace')
    return value


def float32_values(values):
    """Convert eight values to finite float32, preserving the sign of zero."""
    if not isinstance(values, tuple) or len(values) != 8:
        raise ValueError('synthetic input must be an immutable tuple of eight numbers')
    result = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError('synthetic input values must be finite numbers, not bools')
        try:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError('synthetic input values must be finite float32 numbers')
            number = struct.unpack('!f', struct.pack('!f', number))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError('synthetic input values must fit finite float32') from error
        if not math.isfinite(number):
            raise ValueError('synthetic input values must fit finite float32')
        result.append(number)
    return tuple(result)


def validate_synthetic_cases(cases):
    """Reject missing splits and IDs or input collisions after float32 conversion."""
    ids, vectors, splits = set(), set(), set()
    for case in cases:
        if not isinstance(case, SyntheticCase):
            raise ValueError('expected SyntheticCase records')
        case_id = _id(case.case_id)
        if case_id in ids:
            raise ValueError(f'duplicate case ID: {case_id}')
        ids.add(case_id)
        if case.split not in ('discovery', 'heldout'):
            raise ValueError('split must be discovery or heldout')
        splits.add(case.split)
        values = float32_values(case.values)
        if values in vectors:
            raise ValueError('duplicate synthetic input vector')
        vectors.add(values)
    if splits != {'discovery', 'heldout'}:
        raise ValueError('both discovery and heldout splits are required')


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        raise ValueError('smoke text must be nonempty and at most 8192 characters')
    return value


def validate_pretrained_smoke_cases(cases):
    """Check prompt identity and exact duplicates; this is not a holdout test."""
    ids, prompts = set(), set()
    for case in cases:
        if not isinstance(case, PromptCase):
            raise ValueError('expected PromptCase records')
        case_id = _id(case.case_id)
        if case_id in ids:
            raise ValueError(f'duplicate case ID: {case_id}')
        ids.add(case_id)
        prompt = _text(case.text)
        if prompt in prompts:
            raise ValueError('duplicate smoke prompt')
        prompts.add(prompt)
    if not ids:
        raise ValueError('at least one smoke prompt is required')


def validate_case_sets(synthetic_cases=SYNTHETIC_CASES, smoke_cases=PRETRAINED_SMOKE_CASES):
    """Check each suite and require case IDs to be unique across both suites."""
    synthetic_cases, smoke_cases = tuple(synthetic_cases), tuple(smoke_cases)
    validate_synthetic_cases(synthetic_cases)
    validate_pretrained_smoke_cases(smoke_cases)
    if {case.case_id for case in synthetic_cases} & {case.case_id for case in smoke_cases}:
        raise ValueError('duplicate case ID across suites')


def _sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def synthetic_content_sha256(values):
    """Identify the float32 model input independent of its ID and split."""
    return _sha256({'kind': 'numeric-vector-v1', 'values': float32_values(values)})


def pretrained_content_sha256(text):
    """Identify the exact, unmodified smoke prompt text independent of its ID."""
    return _sha256({'kind': 'prompt-text-v1', 'text': _text(text)})


def synthetic_suite_sha256(cases=SYNTHETIC_CASES):
    """Hash version, ordered IDs, split labels, and float32 input values."""
    cases = tuple(cases)
    validate_synthetic_cases(cases)
    return _sha256({'kind': SYNTHETIC_CASES_VERSION, 'cases': [
        {'case_id': case.case_id, 'split': case.split, 'values': float32_values(case.values)}
        for case in cases]})


def pretrained_smoke_sha256(cases=PRETRAINED_SMOKE_CASES):
    """Hash the version and ordered IDs and exact text of smoke prompts."""
    cases = tuple(cases)
    validate_pretrained_smoke_cases(cases)
    return _sha256({'kind': PRETRAINED_SMOKE_VERSION, 'cases': [
        {'case_id': case.case_id, 'text': case.text} for case in cases]})
