"""RAM-first application traces, bounded views, and explicit lossless exports.

This is a trusted local-user interface, not an activation privacy sanitizer.
No pickle or executable checkpoint metadata is used for reopening a trace.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

SCHEMA = "model-looking-glass-local-trace-v1"
MAX_MANIFEST = 16 * 1024 * 1024


def _integer(value, name, minimum=0, maximum=2**31):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


class TraceStore:
    def __init__(self, export_root, *, max_bytes=512 * 1024 * 1024, max_events=100000):
        self.export_root = Path(export_root).expanduser().resolve()
        self.max_bytes = _integer(max_bytes, "max_bytes", 1)
        self.max_events = _integer(max_events, "max_events", 1)
        self._run = None
        self._values = []
        self._events = []

    def publish(self, result):
        """Atomically replace the visible run only after full capture validation."""
        import torch
        captures = result.get("events", [])
        if not isinstance(captures, list) or not 0 < len(captures) <= self.max_events:
            raise ValueError("A completed run must contain a bounded nonempty capture list")
        values, events, total = [], [], 0
        for i, capture in enumerate(captures):
            value = capture.value
            if (type(value) is not torch.Tensor or value.device.type != "cpu"
                    or value.dtype != torch.float32 or value.layout != torch.strided
                    or value.ndim < 1 or value.numel() == 0 or value.requires_grad
                    or not value.is_contiguous()):
                raise ValueError("Captures must be finite, detached contiguous CPU float32 tensors")
            size = value.numel() * value.element_size(); total += size
            if total > self.max_bytes: raise ValueError("Capture payload exceeds the RAM trace allowance")
            if not torch.isfinite(value).all().item(): raise ValueError("Captures must contain finite values")
            metadata = _json_copy(capture.metadata())
            if not isinstance(metadata, dict): raise ValueError("Capture metadata must be an object")
            events.append({"index": i, "metadata": metadata, "shape": list(value.shape),
                           "dtype": "float32", "tensor_bytes": size,
                           "vector_count": value.numel() // value.shape[-1], "vector_width": value.shape[-1]})
            values.append(value)
        report = _json_copy({k: v for k, v in result.items() if k != "events"})
        report.update(event_count=len(values), total_bytes=total, storage="ram", complete=True)
        # References are exclusively owned by this service; no tensor is sent to browser code.
        self._run, self._values, self._events = report, values, events
        return self.summary()

    def clear(self):
        self._run, self._values, self._events = None, [], []

    def summary(self):
        return None if self._run is None else _json_copy(self._run)

    def events(self, offset=0, limit=100):
        if self._run is None: raise ValueError("No completed trace is loaded")
        _integer(offset, "offset", 0, len(self._events)); _integer(limit, "limit", 1, 200)
        return {"events": _json_copy(self._events[offset:offset + limit]), "total": len(self._events)}

    def tensor(self, event, vector=0, offset=0, limit=256, view="raw"):
        import torch
        if self._run is None: raise ValueError("No completed trace is loaded")
        _integer(event, "event", 0, len(self._values) - 1)
        value = self._values[event]; width = value.shape[-1]
        _integer(vector, "vector", 0, value.numel() // width - 1)
        _integer(offset, "offset", 0, width); _integer(limit, "limit", 1, 4096)
        if view not in ("raw", "reduced"): raise ValueError("view must be raw or reduced")
        row = value.reshape(-1, width)[vector]
        result = {"shape": list(value.shape), "dtype": "float32", "vector": vector,
                  "vector_width": width, "offset": offset, "total": width, "approximate": view == "reduced"}
        if view == "raw":
            segment = row[offset:offset + limit]
            result.update(values=segment.tolist(), raw_bits=[f"0x{x & 0xffffffff:08x}" for x in segment.view(torch.int32).tolist()])
        else:
            if offset != 0: raise ValueError("Grouped views start at offset zero")
            count = min(limit, 256, width); bins = []
            for i in range(count):
                start, end = i * width // count, (i + 1) * width // count
                part = row[start:end].double()
                bins.append(dict(start=start, end=end, mean=part.mean().item(), min=part.min().item(), max=part.max().item()))
            result.update(bins=bins, reduction="Contiguous coordinate means/ranges; not semantic concepts. Full original values remain in RAM.")
        return result

    def save(self, name):
        from safetensors.torch import save_file
        if self._run is None: raise ValueError("No completed trace to save")
        if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Use a new run name with letters, digits, hyphens or underscores (1–64 characters)")
        self.export_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        output = self.export_root / name
        output.mkdir(mode=0o700)  # Never overwrite a prior run, symlink or partial export.
        status = output / "status.json"
        status.write_text('{"status":"incomplete"}\n', encoding="utf-8")
        tensor_path = output / "tensors.safetensors"
        # Each capture owns its copy. Different keys must never share mutable checkpoint storage.
        save_file({f"event_{i}": v for i, v in enumerate(self._values)}, str(tensor_path))
        manifest = {"schema": SCHEMA, "tensor_file": tensor_path.name,
                    "tensor_sha256": _digest(tensor_path), "total_bytes": self._run["total_bytes"],
                    "run": self.summary(), "events": _json_copy(self._events)}
        encoded = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
        if len(encoded) > MAX_MANIFEST: raise ValueError("Export metadata exceeds the manifest limit; incomplete export retained")
        (output / "manifest.json").write_bytes(encoded)
        status.write_text('{"status":"complete"}\n', encoding="utf-8")
        return {"path": str(output), "event_count": len(self._events), "tensor_bytes": self._run["total_bytes"], "format": "safetensors + JSON", "lossless": True}

    def open(self, directory):
        """Verify a complete export before replacing the current in-memory trace."""
        from safetensors import safe_open
        import torch
        root = Path(directory).expanduser().resolve()
        manifest_path, tensor_path, status_path = (root / n for n in ("manifest.json", "tensors.safetensors", "status.json"))
        for path in (manifest_path, tensor_path, status_path):
            if path.is_symlink() or not path.is_file(): raise ValueError("Trace files must be ordinary files, not symlinks")
        if manifest_path.stat().st_size > MAX_MANIFEST or status_path.stat().st_size > 1024:
            raise ValueError("Trace metadata exceeds its limit")
        if json.loads(status_path.read_text()) != {"status": "complete"}: raise ValueError("Incomplete exports cannot be opened")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("tensor_file") != "tensors.safetensors":
            raise ValueError("Unsupported trace schema or tensor filename")
        rows = manifest.get("events")
        if not isinstance(rows, list) or not 0 < len(rows) <= self.max_events: raise ValueError("Invalid capture count")
        expected_total = _integer(manifest.get("total_bytes"), "total_bytes", 1, self.max_bytes)
        if tensor_path.stat().st_size > self.max_bytes + MAX_MANIFEST: raise ValueError("Tensor file exceeds import allowance")
        if _digest(tensor_path) != manifest.get("tensor_sha256"): raise ValueError("Tensor digest mismatch")
        captures, total = [], 0
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {f"event_{i}" for i in range(len(rows))}: raise ValueError("Tensor inventory mismatch")
            for i, row in enumerate(rows):
                sliced = handle.get_slice(f"event_{i}"); shape = sliced.get_shape()
                if (row.get("index") != i or row.get("shape") != shape or row.get("dtype") != "float32"
                        or sliced.get_dtype() != "F32" or not shape or any(type(x) is not int or x <= 0 for x in shape)):
                    raise ValueError("Capture shape, dtype or identity mismatch")
                size = math.prod(shape) * 4; total += size
                if size != row.get("tensor_bytes") or total > self.max_bytes: raise ValueError("Capture size mismatch")
                value = handle.get_tensor(f"event_{i}").clone()  # No live file mapping after successful open.
                metadata = _json_copy(row.get("metadata"))
                if not isinstance(metadata, dict): raise ValueError("Capture metadata must be an object")
                captures.append(SimpleNamespace(value=value, metadata=lambda m=metadata: m))
        if total != expected_total: raise ValueError("Total capture size mismatch")
        run = manifest.get("run")
        if not isinstance(run, dict): raise ValueError("Run metadata must be an object")
        result = {**run, "events": captures}
        return self.publish(result)
