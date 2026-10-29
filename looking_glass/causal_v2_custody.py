"""Pure evidence helpers for the proposed synthetic-v2 custody workflow.

These helpers do not generate study inputs, select a private site schedule,
authorize a study, or execute a model. A future cooperative custody gate and
independent review are required before they can protect a real evaluation.
Commitments validate only the outer record shape and digest bindings; a matching
digest does not validate nested case/schedule/prediction schemas or execution.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import struct


_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_DISCOVERY_ROOT = (2026102803).to_bytes(32, "big")
_DOMAINS = {
    "heldout": (b"MLG-v2-holdout\0", frozenset({"context_sha256", "cases"}), "cases"),
    "schedule": (b"MLG-v2-schedule\0", frozenset({"context_sha256", "assignments"}), "assignments"),
    "predictions": (b"MLG-v2-predictions\0",
                    frozenset({"pre_manifest_sha256", "freeze_sha256", "predictions"}), "predictions"),
}


def _check_json(value, depth=0):
    if depth > 100:
        raise ValueError("JSON value is too deeply nested")
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError("JSON values must be finite")
    if type(value) is list:
        for element in value:
            _check_json(element, depth + 1)
        return
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("JSON object keys must be strings")
        for element in value.values():
            _check_json(element, depth + 1)
        return
    raise ValueError("unsupported JSON value")


def canonical_json(value):
    """Use the v2 protocol's UTF-8 canonical JSON, without a trailing newline."""
    _check_json(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("value cannot be encoded as canonical JSON") from error


def _float32_bits(value):
    """Decode a Python binary64 as exact binary32 bits without float32 casts.

    Float32 casts and unpacking can flush subnormals under the ambient CPU
    mode. Integer decoding of the binary64 representation avoids that path.
    """
    if type(value) not in (int, float):
        raise TypeError("case values must be finite float32 numbers")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("case value is outside float32 range") from error
    bits = struct.unpack(">Q", struct.pack(">d", number))[0]
    sign, exponent, fraction = bits >> 63, (bits >> 52) & 0x7ff, bits & ((1 << 52) - 1)
    if exponent == 0x7ff:
        raise ValueError("case values must be finite")
    if exponent == 0 and fraction == 0:
        return sign << 31
    if exponent == 0 or value != number:
        raise ValueError("case value is not exactly representable as float32")
    exponent -= 1023
    significand = (1 << 52) | fraction
    if -126 <= exponent <= 127:
        shift = 29
        exp32 = exponent + 127
    elif -149 <= exponent < -126:
        shift = -exponent - 97
        exp32 = 0
    else:
        raise ValueError("case value is not exactly representable as float32")
    if significand & ((1 << shift) - 1):
        raise ValueError("case value is not exactly representable as float32")
    mantissa = (significand >> shift) & ((1 << 23) - 1)
    return (sign << 31) | (exp32 << 23) | mantissa


def _value_key(values):
    result = []
    for value in values:
        bits = _float32_bits(value)
        result.append(0 if bits & 0x7fffffff == 0 else bits)
    return tuple(result)


def _case(record):
    if type(record) is not dict or set(record) != {"case_id", "values"}:
        raise ValueError("case must contain exactly case_id and values")
    name, values = record["case_id"], record["values"]
    if type(name) is not str or not _CASE_ID.fullmatch(name):
        raise ValueError("invalid case identifier")
    if type(values) is not list or len(values) != 8:
        raise ValueError("a case must contain eight float32 values")
    result = []
    for value in values:
        _float32_bits(value)
        result.append(float(value))
    return {"case_id": name, "values": result}


def build_exclusion_union(records):
    """Build a canonical development exclusion union without generating cases."""
    if type(records) is not list or not records:
        raise ValueError("one or more exclusion records are required")
    cases = []
    seen_ids, seen_values = set(), set()
    for record in records:
        if type(record) is not dict or set(record) != {"inputs"} or type(record["inputs"]) is not list:
            raise ValueError("invalid exclusion record")
        for item in record["inputs"]:
            normalized = _case(item)
            values = _value_key(normalized["values"])
            if normalized["case_id"] in seen_ids or values in seen_values:
                raise ValueError("duplicate exclusion identifier or float32 tuple")
            seen_ids.add(normalized["case_id"])
            seen_values.add(values)
            cases.append(normalized)
    if not cases:
        raise ValueError("exclusion union must not be empty")
    return canonical_json({"inputs": sorted(cases, key=lambda item: item["case_id"])})


def verify_split_disjointness(discovery, heldout, exclusion_bytes, *, historical_cases):
    """Check both splits against supplied historical and development inputs.

    The caller must bind the complete historical suite independently; this
    helper cannot prove that the supplied list is complete.
    """
    if type(exclusion_bytes) is not bytes:
        raise TypeError("exclusion record must be canonical JSON bytes")
    try:
        excluded = json.loads(exclusion_bytes)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("invalid exclusion JSON") from error
    if build_exclusion_union([excluded]) != exclusion_bytes:
        raise ValueError("exclusion record is not canonical and sorted")
    if type(discovery) is not list or type(heldout) is not list:
        raise ValueError("splits must be lists")
    if type(historical_cases) is not list or not historical_cases:
        raise ValueError("historical case inventory is required")
    combined_bytes = build_exclusion_union([excluded, {"inputs": historical_cases}])
    excluded_cases = json.loads(combined_bytes)["inputs"]
    blocked_ids = {item["case_id"] for item in excluded_cases}
    blocked_values = {_value_key(item["values"]) for item in excluded_cases}
    seen_ids, seen_values = set(), set()
    for record in discovery + heldout:
        item = _case(record)
        values = _value_key(item["values"])
        if item["case_id"] in seen_ids:
            raise ValueError("duplicate split identifier")
        if item["case_id"] in blocked_ids or values in blocked_values or values in seen_values:
            raise ValueError("split overlaps another split or excluded inputs")
        seen_ids.add(item["case_id"])
        seen_values.add(values)


def validate_nonce_set(heldout_root, schedule_nonce, prediction_nonce):
    """Validate sizes and non-reuse, not entropy or independence of sampling."""
    values = (heldout_root, schedule_nonce, prediction_nonce)
    if (any(type(value) is not bytes or len(value) != 32 for value in values)
            or len(set(values)) != 3 or _PUBLIC_DISCOVERY_ROOT in values):
        raise ValueError("three distinct 32-byte values are required")


def commitment(kind, nonce, record):
    """Hash an outer-shaped record; this does not validate nested study data."""
    if type(kind) is not str or kind not in _DOMAINS or type(nonce) is not bytes or len(nonce) != 32:
        raise ValueError("invalid commitment kind or nonce")
    domain, expected_keys, list_key = _DOMAINS[kind]
    if type(record) is not dict or set(record) != expected_keys or type(record[list_key]) is not list:
        raise ValueError("commitment record has an invalid shape")
    for key in expected_keys - {list_key}:
        if type(record[key]) is not str or not _DIGEST.fullmatch(record[key]):
            raise ValueError("commitment binding must be a lowercase SHA-256")
    return hashlib.sha256(domain + nonce + canonical_json(record)).hexdigest()


def verify_commitment(kind, nonce, record, expected):
    if type(expected) is not str or not _DIGEST.fullmatch(expected):
        raise ValueError("expected commitment must be a lowercase SHA-256")
    return hmac.compare_digest(commitment(kind, nonce, record), expected)
