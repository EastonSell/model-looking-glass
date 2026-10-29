"""Canonical public discovery observations and whole-freeze consistency.

These helpers do not write files, attest execution/private controls, or prove
publication order. The custodian must retain each complete record durably and
keep private injection witnesses separate until the protocol permits reveal.
"""
import base64
import binascii
from dataclasses import asdict
import hashlib
import hmac
import json
import re
from types import SimpleNamespace

from .pretrained_exposures_v2 import PUBLIC_DISCOVERY
from . import pretrained_number_selection as selection


_SCHEMA = "mlg-number-injection-raw-observation-v1"
_REQUEST = re.compile(r"[0-9a-f]{32}\Z")
_MAX_RECORD_BYTES = 512 * 1024
_MAX_PREDICTION_BYTES = 8 * 1024 * 1024
_ROLES = ("clean", "repeat", "injected", *(f"forward_{i}" for i in range(1, 5)),
          *(f"reverse_{i}" for i in range(1, 5)), "null", "self", "upstream", "downstream", "verify_clean")
_CONDITIONS = {"clean": "clean", "repeat": "clean", "injected": "injected", "null": "null",
               "self": "self", "upstream": "upstream", "downstream": "downstream", "verify_clean": "clean",
               **{f"forward_{i}": "clean_to_injected" for i in range(1, 5)},
               **{f"reverse_{i}": "injected_to_clean" for i in range(1, 5)}}
_RECORD_KEYS = {"schema", "case_index", "role", "request_id", "input_sha256", "condition", "trace", "action", "payload"}
_BOUNDARY_KEYS = set(selection.Boundary.__dataclass_fields__)
_ACTION_KEYS = set(selection.Action.__dataclass_fields__)
_DONOR_KEYS = set(selection.Donor.__dataclass_fields__)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _case_ids(index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("discovery case index is invalid")
    return PUBLIC_DISCOVERY[(index // 2) * 4 + index % 2][1]


def _check_observation(index, role, value, device):
    ids = _case_ids(index)
    if type(role) is not str or role not in _CONDITIONS:
        raise ValueError("discovery role is invalid")
    if type(device) is not str or device not in ("cpu", "cuda:0"):
        raise ValueError("execution device is invalid")
    if type(value) is not selection.Observation:
        raise TypeError("a selector-facing public observation is required")
    selection._observation(value, condition=_CONDITIONS[role], ids=ids, device=device, seen=set())
    if role in ("clean", "repeat", "injected", "verify_clean"):
        if value.action is not None:
            raise ValueError("baseline observations cannot contain an action")
        return
    action = value.action
    if type(action) is not selection.Action:
        raise ValueError("a public action is required")
    if role == "null":
        selection._check_action(value, path="transformer.h.1.mlp", donor=None, ids=ids,
                                device=device, operation="null")
        return
    if type(action.donor) is not selection.Donor or type(action.donor.request_id) is not str or not _REQUEST.fullmatch(action.donor.request_id):
        raise ValueError("patch donor identity is invalid")
    if role == "self":
        if action.donor.request_id != value.request_id:
            raise ValueError("self-patch source request is not the same forward")
        path = "transformer.h.1.mlp"
    elif role == "upstream":
        path = "transformer.h.0.mlp"
    elif role == "downstream":
        path = "transformer.h.5"
    else:
        path = f"transformer.h.{role[-1]}.mlp"
    selection._check_action(value, path=path, donor=SimpleNamespace(request_id=action.donor.request_id,
                                                                   input_sha256=value.input_sha256),
                            ids=ids, device=device)


def _encode(raw):
    return base64.b64encode(raw).decode("ascii")


def serialize_discovery_observation(index, role, observation, *, execution_device):
    """Return one bounded, canonical, public-only discovery observation."""
    _check_observation(index, role, observation, execution_device)
    record = {"schema": _SCHEMA, "case_index": index, "role": role,
              "request_id": observation.request_id, "input_sha256": observation.input_sha256,
              "condition": observation.condition, "trace": [asdict(item) for item in observation.trace],
              "action": None if observation.action is None else asdict(observation.action),
              "payload": {"attention": [_encode(item) for item in observation.attention],
                          "mlp": [_encode(item) for item in observation.mlp],
                          "block": _encode(observation.block), "logits": _encode(observation.logits)}}
    raw = _canonical(record)
    if len(raw) > _MAX_RECORD_BYTES:
        raise ValueError("public observation exceeds its encoded byte cap")
    return raw


def _keys(value, expected):
    if type(value) is not dict or set(value) != expected:
        raise ValueError("public observation fields do not match the schema")


def _tuple(value):
    if type(value) is not list or any(type(item) is not int for item in value):
        raise ValueError("coordinate sequence is invalid")
    return tuple(value)


def _boundary(value):
    _keys(value, _BOUNDARY_KEYS)
    copy = dict(value)
    for field in ("output_selector", "output_shape", "selected_shape"):
        copy[field] = _tuple(copy[field])
    return selection.Boundary(**copy)


def _action(value):
    if value is None:
        return None
    _keys(value, _ACTION_KEYS)
    copy = dict(value)
    for field in ("output_selector", "output_shape", "selected_shape"):
        copy[field] = _tuple(copy[field])
    if copy["donor"] is not None:
        _keys(copy["donor"], _DONOR_KEYS)
        donor = dict(copy["donor"])
        for field in ("output_selector", "selected_shape"):
            donor[field] = _tuple(donor[field])
        copy["donor"] = selection.Donor(**donor)
    return selection.Action(**copy)


def _decode(value, expected):
    if type(value) is not str or len(value) != ((expected + 2) // 3) * 4:
        raise ValueError("encoded tensor size is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("tensor encoding is invalid") from error
    if len(decoded) != expected or _encode(decoded) != value:
        raise ValueError("tensor encoding is not canonical")
    return decoded


def _read(raw, device):
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_RECORD_BYTES:
        raise ValueError("public record size is invalid")
    try:
        record = json.loads(raw)
        if _canonical(record) != raw:
            raise ValueError("public record is not canonical")
        _keys(record, _RECORD_KEYS)
        if record["schema"] != _SCHEMA:
            raise ValueError("public record schema is invalid")
        payload = record["payload"]
        _keys(payload, {"attention", "mlp", "block", "logits"})
        if (type(payload["attention"]) is not list or len(payload["attention"]) != 6 or
                type(payload["mlp"]) is not list or len(payload["mlp"]) != 6 or
                type(record["trace"]) is not list or len(record["trace"]) != 13):
            raise ValueError("public observation trace is incomplete")
        observation = selection.Observation(record["request_id"], record["input_sha256"], record["condition"],
                      tuple(_decode(item, 768 * 4) for item in payload["attention"]),
                      tuple(_decode(item, 768 * 4) for item in payload["mlp"]),
                      _decode(payload["block"], 768 * 4), _decode(payload["logits"], 50257 * 4),
                      tuple(_boundary(item) for item in record["trace"]), _action(record["action"]))
        _check_observation(record["case_index"], record["role"], observation, device)
        return record["case_index"], record["role"], observation
    except (TypeError, UnicodeError, RecursionError, KeyError) as error:
        raise ValueError("public observation record is malformed") from error


def audit_discovery_archive(records, prediction_bytes, *, execution_device):
    """Recompute the complete discovery freeze from exactly 128 raw records.

    The selector's private-controls flag is a caller assertion here, not a
    verification of private witnesses. The archive digest orders exact record
    hashes by case and protocol role, independent of input tuple order.
    """
    if type(records) is not tuple or len(records) != 128:
        raise ValueError("exactly 128 immutable public records are required")
    if type(prediction_bytes) is not bytes or not 0 < len(prediction_bytes) <= _MAX_PREDICTION_BYTES:
        raise ValueError("discovery prediction size is invalid")
    observations, hashes = {}, {}
    for raw in records:
        index, role, value = _read(raw, execution_device)
        key = (index, role)
        if key in observations:
            raise ValueError("duplicate case and role")
        observations[key], hashes[key] = value, hashlib.sha256(raw).digest()
    cases = []
    ordered_hashes = []
    for index in range(8):
        if any((index, role) not in observations for role in _ROLES):
            raise ValueError("discovery observations are incomplete")
        take = lambda role: observations[(index, role)]
        cases.append(selection.DiscoveryCase(index, take("clean"), take("repeat"), take("injected"),
                     tuple(take(f"forward_{i}") for i in range(1, 5)),
                     tuple(take(f"reverse_{i}") for i in range(1, 5)), take("null"), take("self"),
                     take("upstream"), take("downstream"), take("verify_clean")))
        ordered_hashes.extend(hashes[(index, role)] for role in _ROLES)
    recomputed = selection.freeze_discovery(tuple(cases), private_controls_verified=True,
                                            execution_device=execution_device)
    if not hmac.compare_digest(recomputed, prediction_bytes):
        raise ValueError("raw observations do not reproduce the prediction bytes")
    digest = hashlib.sha256(b"MLG-number-injection-archive-v1\0" + b"".join(ordered_hashes)).hexdigest()
    return {"record_count": len(records), "archive_sha256": digest,
            "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest()}
