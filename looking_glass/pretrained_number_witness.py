"""Private known-injection forward witnesses, not a study or custody journal.

The bytes, metadata, exceptions and records handled here can reveal the hidden
site. Keep them inside the trusted custodian until protocol-authorized reveal.
"""
import base64
import binascii
from dataclasses import asdict, dataclass
import json
import math
import re
import struct

from .agreement_numeric import compare_f32_le, decode_f32_le
from .pretrained_number_archive import _case_ids, _check_observation, _CONDITIONS
from .pretrained_number_injection import DonorIdentity, PrivateActionEvidence, _typed_equal, verify_action_scope
from .pretrained_number_selection import _input_sha256


_SCHEMA = "mlg-number-injection-private-forward-v1"
_MAX_PRIVATE_RECORD_BYTES = 1024 * 1024
_WIDTH_BYTES = 768 * 4
_IDENTITY_KEYS = set(DonorIdentity.__dataclass_fields__)
_ACTION_KEYS = {"request_id", "input_sha256", "condition", "operation", "module_path", "token_index",
                "before", "after", "donor", "donor_identity", "realized_delta_f64_le", "metadata_json"}
_REQUEST = re.compile(r"[0-9a-f]{32}\Z")
_PATHS = {f"transformer.h.{layer}.mlp" for layer in range(6)} | {"transformer.h.5"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _keys(value, expected):
    if type(value) is not dict or set(value) != expected:
        raise ValueError("private record fields do not match the schema")


def _encode(value):
    return base64.b64encode(value).decode("ascii")


def _decode(value, length):
    if type(value) is not str or len(value) != ((length + 2) // 3) * 4:
        raise ValueError("private tensor encoding has an invalid size")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("private tensor encoding is invalid") from error
    if len(raw) != length or _encode(raw) != value:
        raise ValueError("private tensor encoding is not canonical")
    return raw


def _check_donor_identity(identity, *, path, token, input_sha256):
    if identity is None:
        return
    selector = (0,) if path == "transformer.h.5" else ()
    if (type(identity) is not DonorIdentity or type(identity.request_id) is not str or
            not _REQUEST.fullmatch(identity.request_id) or type(identity.input_sha256) is not str or
            identity.input_sha256 != input_sha256 or type(identity.module_path) is not str or
            identity.module_path != path or type(identity.invocation) is not int or identity.invocation != 0 or
            type(identity.batch_index) is not int or identity.batch_index != 0 or
            type(identity.token_index) is not int or identity.token_index != token or
            not _typed_equal(identity.output_selector, selector) or
            not _typed_equal(identity.selected_shape, (768,)) or
            type(identity.dtype) is not str or identity.dtype != "torch.float32" or
            type(identity.device) is not str or identity.device != "cpu"):
        raise ValueError("private donor identity is invalid")


def _required_operations(role):
    injecting = role in ("injected", "null", "self", "upstream", "downstream") or role.startswith("forward_")
    action = role in ("null", "self", "upstream", "downstream") or role.startswith(("forward_", "reverse_"))
    return ({"add"} if injecting else set()) | ({"null" if role == "null" else "patch"} if action else set())


def _check_operations(role, evidence):
    operations = [item.operation for item in evidence]
    if len(set(operations)) != len(operations) or set(operations) != _required_operations(role):
        raise ValueError("completed forward action evidence is missing or duplicated")


def _record_shape(index, role, observation, evidence):
    ids = _case_ids(index)
    _check_observation(index, role, observation, "cpu")
    if type(evidence) is not tuple or len(evidence) > 2:
        raise ValueError("at most two immutable private action records are allowed")
    for item in evidence:
        if (type(item) is not PrivateActionEvidence or type(item.request_id) is not str or
                type(item.input_sha256) is not str or type(item.condition) is not str or
                item.request_id != observation.request_id or item.input_sha256 != observation.input_sha256 or
                item.condition != observation.condition or
                type(item.operation) is not str or item.operation not in ("add", "patch", "null") or
                type(item.module_path) is not str or item.module_path not in _PATHS or type(item.token_index) is not int or
                item.token_index != len(ids) - 1 or type(item.before) is not bytes or type(item.after) is not bytes or
                len(item.before) != len(ids) * _WIDTH_BYTES or len(item.after) != len(ids) * _WIDTH_BYTES or
                (item.donor is not None and (type(item.donor) is not bytes or len(item.donor) != _WIDTH_BYTES)) or
                (item.donor_identity is not None and type(item.donor_identity) is not DonorIdentity) or
                (item.realized_delta_f64_le is not None and
                 (type(item.realized_delta_f64_le) is not bytes or len(item.realized_delta_f64_le) != 768 * 8)) or
                type(item.metadata_json) is not str or len(item.metadata_json) > 65536 or
                len(item.metadata_json.encode("utf-8")) > 65536):
            raise ValueError("private action record has invalid identity or bounds")
        _check_donor_identity(item.donor_identity, path=item.module_path, token=item.token_index,
                              input_sha256=observation.input_sha256)
    _check_operations(role, evidence)
    return ids


def encode_private_forward(index, role, observation, evidence):
    """Encode bounded completed-forward private evidence; no file is written."""
    _record_shape(index, role, observation, evidence)
    actions = []
    for item in evidence:
        actions.append({"request_id": item.request_id, "input_sha256": item.input_sha256,
                        "condition": item.condition, "operation": item.operation, "module_path": item.module_path,
                        "token_index": item.token_index, "before": _encode(item.before), "after": _encode(item.after),
                        "donor": None if item.donor is None else _encode(item.donor),
                        "donor_identity": None if item.donor_identity is None else asdict(item.donor_identity),
                        "realized_delta_f64_le": None if item.realized_delta_f64_le is None else _encode(item.realized_delta_f64_le),
                        "metadata_json": item.metadata_json})
    raw = _canonical({"schema": _SCHEMA, "case_index": index, "role": role,
                      "request_id": observation.request_id, "input_sha256": observation.input_sha256,
                      "condition": observation.condition, "actions": actions}).encode("utf-8")
    if len(raw) > _MAX_PRIVATE_RECORD_BYTES:
        raise ValueError("private forward record exceeds its byte cap")
    return raw


@dataclass(frozen=True, repr=False)
class PrivateForwardRecord:
    """Bounded decoded record; private and not yet semantically verified."""
    case_index: int
    role: str
    request_id: str
    input_sha256: str
    condition: str
    evidence: tuple


def decode_private_forward(raw):
    """Decode structurally bounded private records; audit before treating as valid."""
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_PRIVATE_RECORD_BYTES:
        raise ValueError("bounded private record bytes are required")
    try:
        value = json.loads(raw)
        if _canonical(value).encode("utf-8") != raw:
            raise ValueError("private record is not canonical")
        _keys(value, {"schema", "case_index", "role", "request_id", "input_sha256", "condition", "actions"})
        ids = _case_ids(value["case_index"])
        role = value["role"]
        if (value["schema"] != _SCHEMA or type(role) is not str or role not in _CONDITIONS or
                type(value["condition"]) is not str or value["condition"] != _CONDITIONS[role] or
                type(value["request_id"]) is not str or not _REQUEST.fullmatch(value["request_id"]) or
                type(value["input_sha256"]) is not str or value["input_sha256"] != _input_sha256(ids) or
                type(value["actions"]) is not list or len(value["actions"]) > 2):
            raise ValueError("private forward identity is invalid")
        actions = []
        for row in value["actions"]:
            _keys(row, _ACTION_KEYS)
            if (row["request_id"] != value["request_id"] or row["input_sha256"] != value["input_sha256"] or
                    row["condition"] != value["condition"] or type(row["operation"]) is not str or
                    row["operation"] not in ("add", "patch", "null") or type(row["module_path"]) is not str or
                    row["module_path"] not in _PATHS or
                    type(row["token_index"]) is not int or row["token_index"] != len(ids) - 1 or
                    type(row["metadata_json"]) is not str or len(row["metadata_json"]) > 65536 or
                    len(row["metadata_json"].encode("utf-8")) > 65536):
                raise ValueError("private action identity is invalid")
            identity = row["donor_identity"]
            if identity is not None:
                _keys(identity, _IDENTITY_KEYS)
                identity = dict(identity)
                for field in ("output_selector", "selected_shape"):
                    if type(identity[field]) is not list or any(type(item) is not int for item in identity[field]):
                        raise ValueError("private donor coordinates are invalid")
                    identity[field] = tuple(identity[field])
                identity = DonorIdentity(**identity)
            _check_donor_identity(identity, path=row["module_path"], token=row["token_index"],
                                  input_sha256=value["input_sha256"])
            actions.append(PrivateActionEvidence(value["request_id"], value["input_sha256"], value["condition"],
                           row["operation"], row["module_path"], row["token_index"],
                           _decode(row["before"], len(ids) * _WIDTH_BYTES), _decode(row["after"], len(ids) * _WIDTH_BYTES),
                           None if row["donor"] is None else _decode(row["donor"], _WIDTH_BYTES), identity,
                           None if row["realized_delta_f64_le"] is None else _decode(row["realized_delta_f64_le"], 768 * 8),
                           row["metadata_json"]))
        _check_operations(role, actions)
        return PrivateForwardRecord(value["case_index"], role, value["request_id"], value["input_sha256"],
                                    value["condition"], tuple(actions))
    except (TypeError, UnicodeError, KeyError, RecursionError) as error:
        raise ValueError("private record is malformed") from error


def _expected_events(site, role, length):
    """Derive the protocol order without running a model or materializing tensors."""
    token = length - 1
    injecting = role in ("injected", "null", "self", "upstream", "downstream") or role.startswith("forward_")
    target = (int(role[-1]) if role.startswith(("forward_", "reverse_")) else
              0 if role == "upstream" else 1 if role in ("null", "self") else None)
    steps = []
    def add(name, operation, path, selected=False, source=None):
        steps.append((name, operation, path, selected, source))
    for layer in range(6):
        attention, mlp = f"transformer.h.{layer}.attn", f"transformer.h.{layer}.mlp"
        add(f"attention_{layer}", "capture", attention, True)
        if injecting and layer == site:
            add("inject_before", "capture", mlp)
            add("inject_add", "add", mlp, True)
            add("inject_after", "capture", mlp)
        if target == layer:
            if role == "self":
                add("self_source", "capture", mlp, True)
            add("action_before", "capture", mlp)
            add("action_null" if role == "null" else "action_patch", "null" if role == "null" else "patch",
                mlp, True, "self_source" if role == "self" else None)
            add("action_after", "capture", mlp)
        add(f"mlp_{layer}", "capture", mlp, True)
        if layer == 5:
            if role == "downstream":
                add("action_before", "capture", "transformer.h.5")
                add("action_patch", "patch", "transformer.h.5", True)
                add("action_after", "capture", "transformer.h.5")
            add("block_5", "capture", "transformer.h.5", True)
    result, previous_path, preceding = {}, None, []
    for order, (name, operation, path, selected, source) in enumerate(steps):
        if path != previous_path:
            preceding = []
        selector = [0] if path.endswith(".attn") or path == "transformer.h.5" else []
        result[name] = {"name": name, "operation": operation, "order": order, "module_path": path,
                        "module_aliases": [path], "call_site_known": False, "forward_index": 0, "invocation": 0,
                        "batch_index": 0 if selected else None, "token_index": token if selected else None,
                        "output_selector": selector, "output_shape": [1, length, 768],
                        "selected_shape": [768] if selected else [1, length, 768], "dtype": "torch.float32",
                        "device": "cpu", "source_capture": source, "preceding_same_site_actions": list(preceding)}
        preceding.append(name)
        previous_path = path
    return result


def _same(left, right):
    if not compare_f32_le(left, right)["within_tolerance"]:
        raise ValueError("a required private-custodian control failed")


def _same_observation(left, right):
    for a, b in zip(left.attention + left.mlp + (left.block, left.logits),
                    right.attention + right.mlp + (right.block, right.logits)):
        _same(a, b)


def audit_private_forward(index, role, observation, evidence, *, site, clean=None, injected=None):
    """Verify one completed forward's private action/order and public controls.

    This does not attest execution, checkpoint/model state, publication, or
    quiescence. All inputs, exceptions and returned evidence stay in custody.
    """
    ids = _record_shape(index, role, observation, evidence)
    if type(site) is not int or site not in range(1, 5):
        raise ValueError("private site is invalid")
    if clean is not None:
        _check_observation(index, "clean", clean, "cpu")
    if injected is not None:
        _check_observation(index, "injected", injected, "cpu")
    if clean is not None and injected is not None and clean.request_id == injected.request_id:
        raise ValueError("clean and injected requests must differ")
    if ((clean is not None and role != "clean" and observation.request_id == clean.request_id) or
            (injected is not None and role != "injected" and observation.request_id == injected.request_id)):
        raise ValueError("target and baseline requests must differ")
    injecting = role in ("injected", "null", "self", "upstream", "downstream") or role.startswith("forward_")
    action_needed = role in ("null", "self", "upstream", "downstream") or role.startswith(("forward_", "reverse_"))
    if ((injecting or action_needed or role in ("repeat", "verify_clean")) and clean is None or
            (action_needed and injected is None)):
        raise ValueError("required clean or injected baseline is missing")
    required = _required_operations(role)
    by_operation = {item.operation: item for item in evidence}
    if len(by_operation) != len(evidence) or set(by_operation) != required:
        raise ValueError("private action evidence is incomplete or duplicated")
    expected = _expected_events(site, role, len(ids))
    start, end = (len(ids) - 1) * _WIDTH_BYTES, len(ids) * _WIDTH_BYTES
    for operation, item in by_operation.items():
        prefix = "inject" if operation == "add" else "action"
        action_name = "inject_add" if operation == "add" else "action_null" if operation == "null" else "action_patch"
        names = (prefix + "_before", action_name, prefix + "_after")
        try:
            actual = json.loads(item.metadata_json)
            if _canonical(actual) != item.metadata_json or not _typed_equal(actual, [expected[name] for name in names]):
                raise ValueError("private action order or coordinates do not match")
        except (TypeError, KeyError, RecursionError) as error:
            raise ValueError("private action metadata is malformed") from error
        if item.module_path != expected[action_name]["module_path"]:
            raise ValueError("private action path does not match")
        verify_action_scope(item.before, item.after, token=len(ids) - 1, operation=operation, donor=item.donor)
        if operation == "add":
            if item.donor is not None or item.donor_identity is not None or item.realized_delta_f64_le is None:
                raise ValueError("private injection evidence has invalid source or delta")
            before = decode_f32_le(item.before[start:end])
            after = decode_f32_le(item.after[start:end])
            expected_delta = struct.pack("<768d", *(math.fsum((float(a), -float(b))) for a, b in zip(after, before)))
            if item.realized_delta_f64_le != expected_delta:
                raise ValueError("realized injection delta does not match raw witnesses")
        elif item.realized_delta_f64_le is not None:
            raise ValueError("only an injection records a realized delta")
    action = by_operation.get("patch") or by_operation.get("null")
    if action is not None:
        if observation.action is None or action.module_path != observation.action.module_path:
            raise ValueError("private and public action coordinates differ")
        if action.operation == "null":
            if action.donor is not None or action.donor_identity is not None or observation.action.donor is not None:
                raise ValueError("null action cannot have a donor")
        else:
            if action.donor_identity is None or observation.action.donor is None or not _typed_equal(
                    asdict(action.donor_identity), asdict(observation.action.donor)):
                raise ValueError("private and public donor identities differ")
            if role.startswith("forward_"):
                source, expected_donor = clean, clean.mlp[int(role[-1])]
            elif role.startswith("reverse_"):
                source, expected_donor = injected, injected.mlp[int(role[-1])]
            elif role == "upstream":
                source, expected_donor = clean, clean.mlp[0]
            elif role == "downstream":
                source, expected_donor = clean, clean.block
            else:
                source, expected_donor = observation, action.before[start:end]
            if (action.donor_identity.request_id != source.request_id or action.input_sha256 != source.input_sha256 or
                    action.donor != expected_donor):
                raise ValueError("private patch source does not match its same-input baseline")
        visible_after = observation.block if role == "downstream" else observation.mlp[
            0 if role == "upstream" else 1 if role in ("null", "self") else int(role[-1])]
        if action.after[start:end] != visible_after:
            raise ValueError("public boundary does not match the post-action witness")
    add = by_operation.get("add")
    if add is not None:
        _same(add.before[start:end], clean.mlp[site])
        if action is not None and action.module_path == add.module_path:
            if add.after != action.before:
                raise ValueError("adjacent same-site private witnesses differ")
        elif add.after[start:end] != observation.mlp[site]:
            raise ValueError("public boundary does not match post-injection witness")
        for layer in range(site):
            _same(observation.attention[layer], clean.attention[layer])
            _same(observation.mlp[layer], clean.mlp[layer])
        _same(observation.attention[site], clean.attention[site])
    if role in ("repeat", "verify_clean"):
        _same_observation(observation, clean)
    elif role in ("null", "self"):
        _same(observation.mlp[1], injected.mlp[1])
        _same(observation.logits, injected.logits)
    elif role == "upstream":
        _same(observation.logits, injected.logits)
    elif role == "downstream":
        _same(observation.logits, clean.logits)
    return {"verified": True, "request_id": observation.request_id, "condition": observation.condition,
            "private_action_count": len(evidence)}


def audit_private_record(raw, observation, *, site, clean=None, injected=None):
    """Bind a decoded private record to its exact public request before audit."""
    record = decode_private_forward(raw)
    if (record.request_id != observation.request_id or record.input_sha256 != observation.input_sha256 or
            record.condition != observation.condition):
        raise ValueError("private record does not match its public observation")
    return audit_private_forward(record.case_index, record.role, observation, record.evidence,
                                 site=site, clean=clean, injected=injected)
