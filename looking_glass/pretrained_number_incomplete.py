"""Bounded private snapshots of available incomplete-forward evidence.

This format is not a completed forward, an execution attestation or a public
record. Contents and exceptions can reveal the injection site. No pickle or
floating-point value conversion is used.
"""
from dataclasses import dataclass, field, fields
import hashlib
import json
import struct

from .pretrained_number_injection import (DonorIdentity, InjectionObservation, PrivateActionEvidence,
                                          PrivateFailureEvidence, PublicAction, PublicTraceBoundary)


_MAGIC = b"MLGNF001"
_FOOTER = b"MLGNFEND"
_DOMAIN = b"MLG-number-incomplete-forward-v1\0"
_PREFIX = struct.Struct(">8sI")
_MAX_TENSOR_BYTES = 1024 * 1024
_MAX_TEXT_BYTES = 128 * 1024
_MAX_HEADER = 128 * 1024
_MAX_RECORD = _MAX_TENSOR_BYTES + _MAX_TEXT_BYTES + _MAX_HEADER + 64
_MAX_FIELDS = 128
_MAX_TEXT_CHARS = 65536
_MAX_META_NODES = 512
_MAX_META_ENCODED = 16 * 1024
_MAX_OMISSIONS = 256
_META_TYPES = (DonorIdentity, PublicAction, PublicTraceBoundary)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


@dataclass(frozen=True, repr=False)
class IncompleteForward:
    data: bytes = field(repr=False)
    tensor_bytes: int
    omissions: tuple


@dataclass(frozen=True, repr=False)
class DecodedIncompleteForward:
    manifest: dict = field(repr=False)
    payloads: dict = field(repr=False)


class _Builder:
    def __init__(self):
        self.fields = []
        self.blobs = []
        self.identities = {}
        self.omissions = []
        self.meta_nodes = 0
        self.meta_encoded = 0
        self.blob_bytes = {"tensor": 0, "text": 0}

    def omit(self, path):
        if path not in self.omissions and len(self.omissions) < _MAX_OMISSIONS - 1:
            self.omissions.append(path)
        elif path not in self.omissions and "additional-omissions" not in self.omissions:
            self.omissions.append("additional-omissions")

    def meta(self, value, path, depth=0):
        self.meta_nodes += 1
        if depth > 6 or self.meta_nodes > _MAX_META_NODES:
            self.omit(path)
            return None
        if value is None or type(value) is bool:
            return value
        if type(value) is int and -(1 << 63) <= value < (1 << 63):
            return value
        if type(value) is str and len(value) <= 512:
            self.meta_encoded += len(_canonical(value))
            if self.meta_encoded <= _MAX_META_ENCODED:
                return value
            self.omit(path)
            return None
        if type(value) is tuple and len(value) <= 64:
            return [self.meta(item, f"{path}.{i}", depth + 1) for i, item in enumerate(value)]
        if type(value) in _META_TYPES:
            return {item.name: self.meta(getattr(value, item.name), path + "." + item.name, depth + 1)
                    for item in fields(value)}
        self.omit(path)
        return None

    def add(self, path, value, kind):
        if len(self.fields) >= _MAX_FIELDS:
            self.omit(path)
            return
        row = {"path": path, "kind": kind, "status": "none", "length": 0, "blob": None}
        if value is None:
            self.fields.append(row)
            return
        if kind == "text":
            if type(value) is not str or len(value) > _MAX_TEXT_CHARS:
                row["status"] = "invalid-or-oversize"
                self.omit(path)
                self.fields.append(row)
                return
            value = value.encode("utf-8", "surrogatepass")
        if type(value) is not bytes:
            row["status"] = "invalid-type"
            self.omit(path)
        elif len(value) > (_MAX_TENSOR_BYTES if kind == "tensor" else _MAX_TEXT_BYTES):
            row.update(status="oversize", length=len(value))
            self.omit(path)
        else:
            key = (kind, id(value))
            limit = _MAX_TENSOR_BYTES if kind == "tensor" else _MAX_TEXT_BYTES
            if key not in self.identities and self.blob_bytes[kind] + len(value) > limit:
                row.update(status="raw-cap", length=len(value))
                self.omit(path)
                self.fields.append(row)
                return
            if key not in self.identities:
                self.identities[key] = len(self.blobs)
                self.blobs.append((kind, value))
                self.blob_bytes[kind] += len(value)
            row.update(status="available", length=len(value), blob=self.identities[key])
        self.fields.append(row)

    def sequence(self, value, path, maximum):
        if type(value) is not tuple:
            self.omit(path)
            return ()
        if len(value) > maximum:
            self.omit(path + ".extra")
        return value[:maximum]

    def observation(self, value):
        if value is None:
            return None
        if type(value) is not InjectionObservation:
            self.omit("observation")
            return None
        result = {name: self.meta(getattr(value, name), "observation." + name)
                  for name in ("request_id", "input_sha256", "condition", "trace", "action")}
        result["owner_token"] = "excluded-process-local"
        for name in ("attention", "mlp"):
            for index, raw in enumerate(self.sequence(getattr(value, name), "observation." + name, 16)):
                self.add(f"observation.{name}.{index}", raw, "tensor")
        self.add("observation.block", value.block, "tensor")
        self.add("observation.logits", value.logits, "tensor")
        return result

    def actions(self, values):
        result = []
        for index, value in enumerate(self.sequence(values, "actions", 2)):
            path = f"actions.{index}"
            if type(value) is not PrivateActionEvidence:
                self.omit(path)
                result.append(None)
                continue
            result.append({name: self.meta(getattr(value, name), path + "." + name)
                           for name in ("request_id", "input_sha256", "condition", "operation", "module_path",
                                        "token_index", "donor_identity")})
            for name in ("before", "after", "donor", "realized_delta_f64_le"):
                self.add(path + "." + name, getattr(value, name), "tensor")
            self.add(path + ".metadata_json", value.metadata_json, "text")
        return result

    def failure(self, value):
        if value is None:
            return None
        if type(value) is not PrivateFailureEvidence:
            self.omit("failure")
            return None
        result = {name: self.meta(getattr(value, name), "failure." + name)
                  for name in ("request_id", "input_sha256", "condition", "exception_type")}
        result["exception_details_may_be_truncated"] = True
        result["captures"] = []
        for index, pair in enumerate(self.sequence(value.captures, "failure.captures", 64)):
            path = f"failure.captures.{index}"
            if type(pair) is not tuple or len(pair) != 2:
                self.omit(path)
                result["captures"].append(None)
                continue
            result["captures"].append(self.meta(pair[0], path + ".name"))
            self.add(path + ".data", pair[1], "tensor")
        self.add("failure.logits", value.logits, "tensor")
        self.add("failure.exception_message", value.exception_message, "text")
        self.add("failure.metadata_json", value.metadata_json, "text")
        return result

    def finish(self, observation, actions, failure):
        while True:
            descriptions = [{"kind": kind, "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                            for kind, raw in self.blobs]
            tensor_bytes = sum(len(raw) for kind, raw in self.blobs if kind == "tensor")
            manifest = {"schema": "mlg-number-incomplete-forward-v1", "observation": observation,
                        "actions": actions, "failure": failure, "fields": self.fields,
                        "blobs": descriptions, "tensor_bytes": tensor_bytes, "omissions": self.omissions}
            header = _canonical(manifest)
            if len(header) > _MAX_HEADER:
                raise ValueError("private incomplete metadata exceeds its header limit")
            total = _PREFIX.size + len(header) + sum(len(raw) for _, raw in self.blobs) + 32 + len(_FOOTER)
            if total <= _MAX_RECORD:
                break
            if not self.blobs:
                raise ValueError("private incomplete record exceeds its byte limit")
            last = len(self.blobs) - 1
            for row in self.fields:
                if row["blob"] == last:
                    row.update(status="encoded-cap", blob=None)
                    self.omit(row["path"])
            self.blobs.pop()
        parts = [_PREFIX.pack(_MAGIC, len(header)), header, *(raw for _, raw in self.blobs)]
        digest = hashlib.sha256()
        digest.update(_DOMAIN)
        for part in parts:
            digest.update(part)
        data = b"".join((*parts, digest.digest(), _FOOTER))
        return IncompleteForward(data, tensor_bytes, tuple(self.omissions))


def encode_incomplete_forward(*, observation=None, actions=(), failure=None):
    """Capture available raw objects; report absent or unrepresentable parts.

    The caller must already have budgeted the source objects and temporary
    serialization copies. Any exception here means the supervisor must record
    that this evidence could not be fully serialized.
    """
    builder = _Builder()
    obs = builder.observation(observation)
    action_values = builder.actions(actions)
    failed = builder.failure(failure)
    return builder.finish(obs, action_values, failed)


def decode_incomplete_forward(data):
    """Check the private frame and expose retained bytes without float casts."""
    if type(data) is not bytes or not _PREFIX.size + 32 + len(_FOOTER) <= len(data) <= _MAX_RECORD:
        raise ValueError("incomplete record size is invalid")
    magic, header_size = _PREFIX.unpack_from(data)
    if magic != _MAGIC or not 0 < header_size <= _MAX_HEADER or data[-len(_FOOTER):] != _FOOTER:
        raise ValueError("incomplete record header or footer is invalid")
    header_end = _PREFIX.size + header_size
    if header_end + 32 + len(_FOOTER) > len(data):
        raise ValueError("incomplete record is truncated")
    try:
        manifest = json.loads(data[_PREFIX.size:header_end])
        if (_canonical(manifest) != data[_PREFIX.size:header_end] or type(manifest) is not dict or
                set(manifest) != {"schema", "observation", "actions", "failure", "fields", "blobs", "tensor_bytes", "omissions"} or
                manifest["schema"] != "mlg-number-incomplete-forward-v1" or
                type(manifest["fields"]) is not list or len(manifest["fields"]) > _MAX_FIELDS or
                type(manifest["blobs"]) is not list or len(manifest["blobs"]) > _MAX_FIELDS or
                type(manifest["omissions"]) is not list or len(manifest["omissions"]) > _MAX_OMISSIONS or
                any(type(item) is not str or len(item) > 256 for item in manifest["omissions"]) or
                type(manifest["tensor_bytes"]) is not int or not 0 <= manifest["tensor_bytes"] <= _MAX_TENSOR_BYTES):
            raise ValueError("incomplete record schema is invalid")
        cursor, payloads, blobs = header_end, {}, []
        tensor_bytes, text_bytes = 0, 0
        for info in manifest["blobs"]:
            if (type(info) is not dict or set(info) != {"kind", "length", "sha256"} or
                    info["kind"] not in ("tensor", "text") or type(info["length"]) is not int or
                    not 0 <= info["length"] <= (_MAX_TENSOR_BYTES if info["kind"] == "tensor" else _MAX_TEXT_BYTES) or
                    type(info["sha256"]) is not str):
                raise ValueError("incomplete blob description is invalid")
            end = cursor + info["length"]
            if end > len(data) - 32 - len(_FOOTER):
                raise ValueError("incomplete blob is truncated")
            raw = data[cursor:end]
            if hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise ValueError("incomplete blob checksum is invalid")
            blobs.append(raw)
            if info["kind"] == "tensor":
                tensor_bytes += len(raw)
            else:
                text_bytes += len(raw)
            cursor = end
        if (cursor + 32 + len(_FOOTER) != len(data) or tensor_bytes != manifest["tensor_bytes"] or
                tensor_bytes > _MAX_TENSOR_BYTES or text_bytes > _MAX_TEXT_BYTES):
            raise ValueError("incomplete record payload size is invalid")
        digest = hashlib.sha256(_DOMAIN + data[:cursor]).digest()
        if digest != data[cursor:cursor + 32]:
            raise ValueError("incomplete record checksum is invalid")
        paths = set()
        for row in manifest["fields"]:
            if (type(row) is not dict or set(row) != {"path", "kind", "status", "length", "blob"} or
                    type(row["path"]) is not str or not row["path"] or row["path"] in paths or
                    row["kind"] not in ("tensor", "text") or
                    row["status"] not in ("none", "invalid-or-oversize", "invalid-type", "oversize", "raw-cap", "encoded-cap", "available") or
                    type(row["length"]) is not int or row["length"] < 0):
                raise ValueError("incomplete field description is invalid")
            paths.add(row["path"])
            if row["status"] == "available":
                index = row["blob"]
                if (type(index) is not int or not 0 <= index < len(blobs) or
                        manifest["blobs"][index]["kind"] != row["kind"] or len(blobs[index]) != row["length"]):
                    raise ValueError("incomplete field points to an invalid blob")
                payloads[row["path"]] = blobs[index]
            elif row["blob"] is not None:
                raise ValueError("unavailable field cannot reference a blob")
        return DecodedIncompleteForward(manifest, payloads)
    except (TypeError, KeyError, UnicodeError, RecursionError, struct.error) as error:
        raise ValueError("incomplete record is malformed") from error
