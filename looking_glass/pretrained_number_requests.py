"""Public, site-independent request identities for injection discovery.

This module does not draw entropy, choose a site, publish a plan, or attest the
order in which its caller performed those actions.
"""
from dataclasses import dataclass
import hashlib
import hmac
import json
import re

from .pretrained_exposures_v2 import PUBLIC_DISCOVERY
from .pretrained_number_archive import _ROLES
from .pretrained_number_selection import CHECKPOINT_REVISION, PROTOCOL_SHA256


_SCHEMA = "mlg-number-injection-request-plan-v1"
_DOMAIN = b"MLG-number-injection-request-v1\0"
_MAX_PLAN_BYTES = 128 * 1024
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _input_digest(ids):
    return hashlib.sha256(b"MLG-number-input-v1\0" + bytes([len(ids)]) +
                          b"".join(value.to_bytes(4, "big") for value in ids)).hexdigest()


def create_request_plan(public_entropy, source_commit):
    """Derive a fixed 128-request plan from independently generated public bytes.

    The custodian must draw and publish these bytes before independently
    drawing the private site nonce. A collision or failure is terminal, not a
    reason to reroll, and this helper cannot prove that ordering.
    """
    if type(public_entropy) is not bytes or len(public_entropy) != 32:
        raise ValueError("public request entropy must be 32 immutable bytes")
    if type(source_commit) is not str or not _HEX40.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase SHA-1")
    requests, identities = [], set()
    for case_index in range(8):
        ids = PUBLIC_DISCOVERY[(case_index // 2) * 4 + case_index % 2][1]
        for role in _ROLES:
            ordinal = len(requests)
            request_id = hashlib.sha256(_DOMAIN + public_entropy + ordinal.to_bytes(4, "big")).digest()[:16].hex()
            if request_id in identities:
                raise ValueError("request ID collision; attempt must stop")
            identities.add(request_id)
            requests.append({"case_index": case_index, "role": role, "request_id": request_id,
                             "input_sha256": _input_digest(ids)})
    record = {"schema": _SCHEMA, "protocol_sha256": PROTOCOL_SHA256,
              "checkpoint_revision": CHECKPOINT_REVISION, "source_commit": source_commit,
              "public_entropy": public_entropy.hex(), "requests": requests}
    raw = _canonical(record)
    if len(raw) > _MAX_PLAN_BYTES:
        raise ValueError("request plan exceeds its byte cap")
    return raw


@dataclass(frozen=True)
class PublicRequestPlan:
    """Verified plan bytes and digest, not evidence of publication or entropy."""
    record_bytes: bytes
    sha256: str
    source_commit: str


def load_request_plan(raw, *, expected_source_commit, expected_sha256):
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_PLAN_BYTES:
        raise ValueError("bounded immutable request plan bytes are required")
    if (type(expected_source_commit) is not str or not _HEX40.fullmatch(expected_source_commit) or
            type(expected_sha256) is not str or not _HEX64.fullmatch(expected_sha256)):
        raise ValueError("expected published source and request-plan digest are required")
    try:
        record = json.loads(raw)
        if (type(record) is not dict or set(record) != {"schema", "protocol_sha256", "checkpoint_revision",
                                                         "source_commit", "public_entropy", "requests"}
                or type(record["public_entropy"]) is not str or not _HEX64.fullmatch(record["public_entropy"])):
            raise ValueError("request plan schema or public entropy is invalid")
        expected = create_request_plan(bytes.fromhex(record["public_entropy"]), record["source_commit"])
    except (KeyError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("request plan is malformed") from error
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if (not hmac.compare_digest(raw, expected) or record["source_commit"] != expected_source_commit or
            not hmac.compare_digest(actual_sha256, expected_sha256)):
        raise ValueError("request plan does not match its derivation, published source, or digest")
    return PublicRequestPlan(raw, actual_sha256, expected_source_commit)


def verify_request(plan, case_index, role, request_id, input_sha256, *, expected_source_commit, expected_sha256):
    """Check a request against a plan bound to independently trusted identities."""
    if type(plan) is not PublicRequestPlan:
        raise ValueError("a validated request plan is required")
    checked = load_request_plan(plan.record_bytes, expected_source_commit=expected_source_commit,
                                expected_sha256=expected_sha256)
    if (type(plan.source_commit) is not str or type(plan.sha256) is not str or
            plan.source_commit != checked.source_commit or not hmac.compare_digest(plan.sha256, checked.sha256)):
        raise ValueError("request plan object does not match the trusted binding")
    if (type(case_index) is not int or not 0 <= case_index < 8 or type(role) is not str or role not in _ROLES or
            type(request_id) is not str or not _HEX32.fullmatch(request_id) or
            type(input_sha256) is not str or not _HEX64.fullmatch(input_sha256)):
        raise ValueError("observation request coordinates are invalid")
    ordinal = case_index * len(_ROLES) + _ROLES.index(role)
    expected = json.loads(checked.record_bytes)["requests"][ordinal]
    if not hmac.compare_digest(expected["request_id"], request_id) or not hmac.compare_digest(expected["input_sha256"], input_sha256):
        raise ValueError("observation does not match its precommitted request")
    return True
