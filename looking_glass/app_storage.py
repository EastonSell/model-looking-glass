"""RAM-first application traces, bounded views, and explicit lossless exports.

This is a trusted local-user interface, not an activation privacy sanitizer.
No pickle or executable checkpoint metadata is used for reopening a trace.
"""
from __future__ import annotations

import hashlib
import heapq
from itertools import islice
import json
import math
import os
import stat
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

    def events(self, offset=0, limit=100, q=""):
        """Search allowlisted capture metadata globally, then paginate stable indices."""
        if self._run is None: raise ValueError("No completed trace is loaded")
        _integer(offset, "offset", 0, len(self._events)); _integer(limit, "limit", 1, 200)
        if type(q) is not str or len(q) > 256:
            raise ValueError("q must be a string of at most 256 characters")
        query = " ".join(q.split()).casefold()
        fields = ("module_path", "module_kind", "layer_index", "app_forward_index", "forward_index")
        page, total = [], 0
        for event in self._events:
            # Never stringify arbitrary metadata containers or search prompts/tensors.
            searchable = [" ".join(str(event["metadata"][key]).split()).casefold()
                          for key in fields if type(event["metadata"].get(key)) in (str, int)]
            if query and not any(query in value for value in searchable):
                continue
            if offset <= total < offset + limit:
                page.append(event)
            total += 1
        return {"events": _json_copy(page), "total": total, "total_all": len(self._events)}

    def saved_runs(self):
        """Read-only immediate-child catalog; completion is NOT integrity verification.

        At most 500 root entries are considered (plus one truncation lookahead),
        then up to 200 candidate directories are inspected in lexical order. A
        truncated result is a bounded subset, not a full inventory. Manifests and
        payloads are never read. Descriptor-relative, no-follow access is POSIX-only.
        """
        if os.name != "posix" or not all(hasattr(os, key) for key in ("O_DIRECTORY", "O_NOFOLLOW")):
            return self._saved_runs_portable()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(self.export_root, flags)
        except FileNotFoundError:
            return {"runs": [], "truncated": False}
        try:
            with os.scandir(root_fd) as entries:
                names = heapq.nsmallest(201, (entry.name for entry in islice(entries, 500)
                                            if entry.is_dir(follow_symlinks=False)))
                enumeration_truncated = next(entries, None) is not None
            runs = []
            for name in names[:200]:
                try:
                    run_fd = os.open(name, flags, dir_fd=root_fd)
                except FileNotFoundError:
                    continue  # A concurrently removed export is no longer a saved run.
                try:
                    files = {}
                    for filename in ("manifest.json", "tensors.safetensors", "status.json"):
                        try:
                            info = os.stat(filename, dir_fd=run_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        if stat.S_ISREG(info.st_mode):
                            files[filename] = info.st_size
                    complete = False
                    if (len(files) == 3 and files["status.json"] <= 1024):
                        try:
                            status_fd = os.open("status.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                                dir_fd=run_fd)
                            with os.fdopen(status_fd, "rb") as source:
                                info = os.fstat(source.fileno())
                                if stat.S_ISREG(info.st_mode) and info.st_size <= 1024:
                                    complete = json.loads(source.read(1024)) == {"status": "complete"}
                        except (FileNotFoundError, ValueError, UnicodeError):
                            pass
                    runs.append({"name": name, "path": str(self.export_root / name),
                                 "status": "complete_unverified" if complete else "incomplete",
                                 "tensor_file_bytes": files.get("tensors.safetensors")})
                finally:
                    os.close(run_fd)
            return {"runs": runs, "truncated": enumeration_truncated or len(names) > 200}
        finally:
            os.close(root_fd)

    def _saved_runs_portable(self):
        """Trusted single-user fallback, not hostile-filesystem race isolation.

        Reject detected links/junctions and changed identities, but platforms
        without descriptor-relative no-follow access cannot close every race.
        """
        def identity(info):
            return (info.st_dev, info.st_ino, info.st_mode)

        def ordinary_directory(path):
            info = path.lstat()
            if (not stat.S_ISDIR(info.st_mode) or path.is_symlink()
                    or (hasattr(path, "is_junction") and path.is_junction())):
                raise ValueError("Saved-run catalog requires an ordinary directory, not a link")
            return identity(info)

        try:
            root_identity = ordinary_directory(self.export_root)
        except FileNotFoundError:
            return {"runs": [], "truncated": False}
        with os.scandir(self.export_root) as entries:
            names = heapq.nsmallest(201, (entry.name for entry in islice(entries, 500)
                                        if entry.is_dir(follow_symlinks=False)))
            enumeration_truncated = next(entries, None) is not None
        runs = []
        for name in names[:200]:
            path = self.export_root / name
            try:
                run_identity = ordinary_directory(path)
            except FileNotFoundError:
                continue
            files = {}
            for filename in ("manifest.json", "tensors.safetensors", "status.json"):
                try:
                    info = (path / filename).lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    files[filename] = info
            complete = False
            if len(files) == 3 and files["status.json"].st_size <= 1024:
                try:
                    with (path / "status.json").open("rb") as source:
                        current = os.fstat(source.fileno())
                        if identity(current) != identity(files["status.json"]):
                            raise ValueError("Saved-run status changed during catalog scan")
                        if stat.S_ISREG(current.st_mode) and current.st_size <= 1024:
                            complete = json.loads(source.read(1024)) == {"status": "complete"}
                except (FileNotFoundError, ValueError, UnicodeError):
                    pass
            if ordinary_directory(path) != run_identity or ordinary_directory(self.export_root) != root_identity:
                raise ValueError("Saved-run directory changed during catalog scan")
            tensor = files.get("tensors.safetensors")
            runs.append({"name": name, "path": str(path),
                         "status": "complete_unverified" if complete else "incomplete",
                         "tensor_file_bytes": tensor.st_size if tensor is not None else None})
        if ordinary_directory(self.export_root) != root_identity:
            raise ValueError("Saved-run directory changed during catalog scan")
        return {"runs": runs, "truncated": enumeration_truncated or len(names) > 200}

    def _coordinate(self, event, vector):
        """Ground an original flat vector using only immutable retained provenance."""
        coordinate = dict(status="unknown", app_forward_index=None, module_path=None,
                          layer_index=None, stack=None, batch_position=None, token_position=None,
                          token_id=None, token_piece=None, token_text=None, token_label_truncated=False,
                          image_coordinate=None, reason=None)
        def require(condition, reason):
            if not condition: raise ValueError(reason)
        def integer(value):
            return type(value) is int and value >= 0
        def integers(value, *, positive=False):
            return type(value) is list and bool(value) and all(integer(x) and (not positive or x > 0) for x in value)
        try:
            metadata = self._events[event]["metadata"]
            shape = self._events[event]["shape"]
            source, saved = metadata.get("shape"), metadata.get("saved_shape")
            require(integers(source, positive=True) and len(source) == 3, "Missing or invalid source shape")
            require(integers(saved, positive=True) and saved == shape, "Saved shape does not match retained tensor")
            require(type(metadata.get("batch_axis")) is int and metadata["batch_axis"] == 0
                    and type(metadata.get("token_axis")) is int and metadata["token_axis"] == 1,
                    "Unsupported or missing batch/token axes")
            batches, positions = metadata.get("batch_positions"), metadata.get("token_positions")
            require(integers(batches) and len(set(batches)) == len(batches) and max(batches) < source[0], "Invalid batch positions")
            require(integers(positions) and len(set(positions)) == len(positions) and max(positions) < source[1], "Invalid token positions")
            model = self._run.get("model")
            family = model.get("family") if type(model) is dict else None
            require(family in ("gpt2", "distilbert", "t5", "vit"), "Unknown retained adapter family")
            if family in ("gpt2", "distilbert"):
                require(shape == [len(batches), len(positions), source[2]], "Unsupported retained vector layout")
                if family == "distilbert":
                    require(len(batches) == len(positions) == 1, "DistilBERT requires singleton selected positions")
                batch_slot, token_slot = divmod(vector, len(positions))
            else:
                require(shape == [source[2]] and len(batches) == len(positions) == 1 and vector == 0,
                        "Adapter requires one rank-one selected vector")
                batch_slot = token_slot = 0
            forward, path, layer = (metadata.get(key) for key in ("app_forward_index", "module_path", "layer_index"))
            require(integer(forward) and integer(layer) and type(path) is str and 0 < len(path) <= 1024,
                    "Missing or invalid forward/module/layer provenance")
            identity = metadata.get("identity")
            require(type(identity) is dict and type(identity.get("run_id")) is str and bool(identity["run_id"]), "Missing capture run identity")
            steps = self._run.get("steps")
            require(type(steps) is list and all(type(step) is dict and integer(step.get("app_forward_index")) for step in steps), "Invalid retained forward steps")
            matching = [step for step in steps if step["app_forward_index"] == forward]
            require(len(matching) == 1 and matching[0].get("run_id") == identity["run_id"], "Forward step is missing, ambiguous, or has mismatched run identity")
            step = matching[0]
            batch, position = batches[batch_slot], positions[token_slot]
            stack = "input"
            if family == "t5":
                require(path.startswith(("encoder.", "decoder.")), "T5 encoder/decoder stack is not explicit")
                stack = path.split(".", 1)[0]
            prefixes = {"gpt2": f"transformer.h.{layer}", "distilbert": f"distilbert.transformer.layer.{layer}",
                        "t5": f"{stack}.block.{layer}", "vit": f"vit.encoder.layer.{layer}"}
            suffixes = {"gpt2": ("", ".attn", ".mlp"), "distilbert": ("", ".attention", ".ffn"),
                        "t5": ("", ".layer.0", ".layer.1" if stack == "encoder" else ".layer.2"),
                        "vit": ("", ".attention", ".intermediate")}
            require(path in {prefixes[family] + suffix for suffix in suffixes[family]},
                    "Module path contradicts the retained adapter/layer boundary")
            if family == "vit":
                stack = "image"
                image = metadata.get("image_coordinate")
                require(type(image) is dict, "Missing image coordinate")
                kind, row, col, box = (image.get(key) for key in ("token_kind", "patch_row", "patch_column", "processed_pixel_box_xyxy"))
                if kind == "cls":
                    require(position == 0 and row is None and col is None and box is None, "Invalid CLS image coordinate")
                else:
                    require(kind == "patch" and position > 0 and integer(row) and integer(col)
                            and integers(box) and len(box) == 4 and box[0] < box[2] and box[1] < box[3], "Invalid patch image coordinate")
                geometry = self._run.get("image_metadata")
                require(geometry is not None, "Missing retained processed-image geometry")
                reason = None
                if geometry is not None:
                    require(type(geometry) is dict, "Invalid image geometry")
                    grid, patch = geometry.get("grid_shape"), geometry.get("patch_shape")
                    require(integers(grid, positive=True) and len(grid) == 2 and integers(patch, positive=True) and len(patch) == 2,
                            "Missing or invalid patch geometry")
                    require(integer(geometry.get("sequence_length")) and geometry["sequence_length"] == source[1] == 1 + grid[0] * grid[1]
                            and type(geometry.get("cls_token_index")) is int and geometry["cls_token_index"] == 0, "Inconsistent image sequence geometry")
                    if kind == "patch":
                        require(row < grid[0] and col < grid[1] and position == 1 + row * grid[1] + col
                                and box == [col * patch[1], row * patch[0], (col + 1) * patch[1], (row + 1) * patch[0]], "Patch position or box contradicts processed geometry")
                    reason = None
                coordinate.update(image_coordinate={"token_kind": kind, "patch_row": row, "patch_column": col,
                                                     "processed_pixel_box_xyxy": _json_copy(box)}, reason=reason)
            else:
                require(batch == 0 and source[0] == 1, "Recorded token sequences describe one batch-zero sample only")
                ids = step.get("decoder_token_ids" if stack == "decoder" else "input_token_ids")
                require(integers(ids) and len(ids) == source[1] and position < len(ids), "Missing or inconsistent forward input-token IDs")
                token_id = ids[position]
                coordinate["token_id"] = token_id
                labels = self._run.get("token_labels")
                label = labels.get(str(token_id)) if type(labels) is dict else None
                if (type(label) is dict and type(label.get("piece")) is str and len(label["piece"]) <= 512
                        and type(label.get("text")) is str and len(label["text"]) <= 512 and type(label.get("truncated")) is bool):
                    coordinate.update(token_piece=label["piece"], token_text=label["text"], token_label_truncated=label["truncated"],
                                      reason="Token display label is truncated" if label["truncated"] else None)
                else:
                    coordinate["reason"] = "Token display label unavailable in retained run; no tokenizer decoding performed"
            coordinate.update(status="known", app_forward_index=forward, module_path=path, layer_index=layer,
                              stack=stack, batch_position=batch, token_position=position)
        except ValueError as error:
            coordinate["reason"] = str(error)
        return coordinate

    def vectors(self, event, offset=0, limit=100):
        """Bounded options retain original flat vector indices and exact coordinates."""
        if self._run is None: raise ValueError("No completed trace is loaded")
        _integer(event, "event", 0, len(self._values) - 1)
        count = self._events[event]["vector_count"]
        _integer(offset, "offset", 0, count); _integer(limit, "limit", 1, 200)
        return {"vectors": [{"index": index, "coordinate": self._coordinate(event, index)}
                            for index in range(offset, min(offset + limit, count))], "total": count}

    def intervention_source(self, event, vector):
        """Copy one grounded GPT2 source for a privately bound runtime experiment.

        This validates retained data, not live-model authority. The server must
        separately require its private live-capture binding; JSON is not a binding.
        """
        import torch
        if self._run is None: raise ValueError("No completed trace is loaded")
        _integer(event, "event", 0, len(self._values) - 1)
        _integer(vector, "vector", 0, self._events[event]["vector_count"] - 1)
        coordinate = self._coordinate(event, vector)
        model = self._run.get("model")
        if (coordinate["status"] != "known" or type(model) is not dict
                or model.get("family") != "gpt2"):
            raise ValueError("Intervention source requires a known GPT2 coordinate")
        metadata = self._events[event]["metadata"]
        matches = [step for step in self._run["steps"]
                   if step["app_forward_index"] == coordinate["app_forward_index"]
                   and step.get("run_id") == metadata["identity"]["run_id"]]
        if len(matches) != 1:
            raise ValueError("Intervention source requires a unique retained forward step")
        step = matches[0]
        ids, mask = step.get("input_token_ids"), step.get("input_attention_mask")
        if (type(ids) is not list or not ids or any(type(value) is not int or value < 0 for value in ids)
                or type(mask) is not list or len(mask) != len(ids)
                or any(type(value) is not int or value not in (0, 1) for value in mask)):
            raise ValueError("Intervention source requires exact input IDs and a matching binary attention mask")
        if step.get("use_cache") is not False:
            raise ValueError("Intervention source requires an explicitly uncached forward")
        value = self._values[event]
        reference = value.reshape(-1, value.shape[-1])[vector].detach().clone().contiguous()
        if (reference.device.type != "cpu" or reference.dtype != torch.float32
                or not torch.isfinite(reference).all().item()):
            raise ValueError("Intervention reference must be finite CPU float32")
        return {"coordinate": _json_copy(coordinate), "step": _json_copy(step),
                "model": _json_copy(model), "reference_value": reference}

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
                  "vector_width": width, "offset": offset, "total": width, "approximate": view == "reduced",
                  "coordinate": self._coordinate(event, vector)}
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
