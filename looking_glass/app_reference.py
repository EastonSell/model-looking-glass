"""One bounded RAM-only vector reference; no automatic token alignment or authority."""
import json
import math
import re

MAX_VECTOR_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 256 * 1024
MAX_WORKSPACE_BYTES = 64 * (MAX_VECTOR_BYTES // 4) + 64 * 1024
SCOPE = "Manual coordinate comparison only; no semantic alignment, causal conclusion, or model-weight identity claim. Deltas are current minus reference."


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _bounded_metadata_copy(value, limit=MAX_METADATA_BYTES):
    """Bound scalar encoding, nesting and total metadata before copying/cloning."""
    remaining = limit
    def visit(item, depth=0):
        nonlocal remaining
        if depth > 32: raise ValueError("Reference metadata nesting exceeds limit")
        remaining -= 1
        if remaining < 0: raise ValueError("Reference metadata exceeds byte limit")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str: raise ValueError("Reference metadata keys must be strings")
                visit(key, depth + 1); visit(child, depth + 1)
        elif type(item) is list:
            for child in item: visit(child, depth + 1)
        elif item is None or type(item) in (str, bool, int, float):
            if type(item) is str and len(item) > limit:
                raise ValueError("Reference metadata exceeds byte limit")
            if type(item) is float and not math.isfinite(item):
                raise ValueError("Reference metadata must be finite JSON")
            remaining -= len(json.dumps(item, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            if remaining < 0: raise ValueError("Reference metadata exceeds byte limit")
        else:
            raise ValueError("Reference metadata must be JSON values")
    visit(value)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > limit: raise ValueError("Reference metadata exceeds byte limit")
    return json.loads(encoded)


class ReferenceVector:
    """Private tensor ownership; caller must separately enforce its live binding."""
    def __init__(self, *, max_bytes=MAX_VECTOR_BYTES, max_metadata_bytes=MAX_METADATA_BYTES,
                 max_workspace_bytes=MAX_WORKSPACE_BYTES):
        self.max_bytes = _integer(max_bytes, "max_bytes", 1, MAX_VECTOR_BYTES)
        self.max_metadata_bytes = _integer(max_metadata_bytes, "max_metadata_bytes", 1, MAX_METADATA_BYTES)
        self.max_workspace_bytes = _integer(max_workspace_bytes, "max_workspace_bytes", 1, MAX_WORKSPACE_BYTES)
        self.clear()

    @property
    def total_bytes(self):
        return 0 if self._value is None else self._value.numel() * self._value.element_size()

    def clear(self):
        self._value = self._metadata = self._signature = None

    def _validate(self, source):
        import torch
        if type(source) is not dict: raise ValueError("Reference source must be an object")
        value = source.get("reference_value")
        if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.float32
                or value.layout != torch.strided or value.ndim != 1 or value.numel() == 0
                or not value.is_contiguous() or value.requires_grad):
            raise ValueError("Reference requires one detached contiguous CPU float32 vector")
        if value.numel() * 4 > self.max_bytes: raise ValueError("Reference vector exceeds byte limit")
        selected_metadata = {key: source.get(key) for key in
                             ("coordinate", "step", "model", "metadata", "prompt")}
        selected_metadata.update(width=value.numel(), total_bytes=value.numel() * 4, dtype="float32")
        metadata = _bounded_metadata_copy(selected_metadata, self.max_metadata_bytes)
        coordinate, model, event = (metadata[key] for key in ("coordinate", "model", "metadata"))
        if not all(type(item) is dict for item in (coordinate, model, event, metadata["step"])):
            raise ValueError("Reference source metadata is missing")
        if (coordinate.get("status") != "known" or model.get("family") not in ("gpt2", "llama")
                or model.get("dtype") != "float32" or type(model.get("device")) is not str
                or not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", model["device"])
                or type(model.get("execution_numerics")) is not dict or not model["execution_numerics"]
                or type(coordinate.get("module_path")) is not str or not coordinate["module_path"]
                or type(coordinate.get("layer_index")) is not int or coordinate["layer_index"] < 0
                or coordinate.get("stack") != "input"
                or event.get("module_path") != coordinate["module_path"]
                or type(event.get("layer_index")) is not int or event["layer_index"] != coordinate["layer_index"]):
            raise ValueError("Reference requires known matching GPT2/Llama boundary provenance")
        selector = event.get("output_path")
        if type(selector) is not list or any(type(index) is not int or index < 0 for index in selector):
            raise ValueError("Reference output selector is missing or invalid")
        layer = coordinate["layer_index"]
        if model["family"] == "gpt2":
            prefix = f"transformer.h.{layer}"
            sites = {prefix: ([], [0]), prefix + ".attn": ([0],), prefix + ".mlp": ([],)}
        else:
            prefix = f"model.layers.{layer}"
            sites = {prefix: ([],), prefix + ".self_attn": ([0],), prefix + ".mlp": ([],)}
        if coordinate["module_path"] not in sites or selector not in sites[coordinate["module_path"]]:
            raise ValueError("Reference requires a canonical family/layer boundary and output selector")
        if metadata["prompt"] is not None and type(metadata["prompt"]) is not str:
            raise ValueError("Reference prompt must be text or null")
        if not torch.isfinite(value).all().item(): raise ValueError("Reference values must be finite")
        signature = (model["family"], coordinate["module_path"], coordinate["layer_index"], coordinate["stack"],
                     tuple(selector), value.numel(), str(value.dtype), model["device"],
                     json.dumps(model["execution_numerics"], sort_keys=True, allow_nan=False))
        return value, metadata, signature

    def pin(self, source):
        value, metadata, signature = self._validate(source)
        retained = value.detach().clone().contiguous()
        returned = json.loads(json.dumps(metadata, allow_nan=False))
        self._value, self._metadata, self._signature = retained, metadata, signature
        return returned

    def summary(self):
        return None if self._metadata is None else json.loads(json.dumps(self._metadata, allow_nan=False))

    def _page(self, offset, limit):
        if self._value is None: raise ValueError("No reference vector is pinned")
        _integer(offset, "offset", 0, self._value.numel()); _integer(limit, "limit", 1, 4096)
        return slice(offset, offset + limit)

    @staticmethod
    def _bits(value):
        import torch
        return [f"0x{bits & 0xffffffff:08x}" for bits in value.view(torch.int32).tolist()]

    @staticmethod
    def _decoded(value):
        """Decode IEEE fields through integers, preserving subnormals under FTZ/DAZ."""
        import torch
        from .agreement_numeric import decode_f32_le
        if value.numel() == 0:
            return torch.empty(0, dtype=torch.float64)
        raw = value.view(torch.int32).numpy().astype("<i4", copy=False).tobytes()
        return torch.from_numpy(decode_f32_le(raw))

    def tensor(self, offset=0, limit=256):
        page = self._page(offset, limit); value = self._value[page]
        return dict(values=self._decoded(value).tolist(), raw_bits=self._bits(value),
                    coordinate=self.summary()["coordinate"], offset=offset, total=self._value.numel(), dtype="float32")

    def compare(self, source, offset=0, limit=256):
        import torch
        page = self._page(offset, limit)
        current, metadata, signature = self._validate(source)
        if signature != self._signature: raise ValueError("Reference/current boundary or execution signature mismatch")
        width = self._value.numel()
        if 64 * width + 64 * 1024 > self.max_workspace_bytes:
            raise ValueError("Comparison float64 workspace exceeds byte limit")
        # Bound decoder temporaries plus retained binary64 vectors before decoding.
        # Integer-field decoding avoids caller FTZ/DAZ changing stored subnormals.
        left, right = self._decoded(self._value), self._decoded(current)
        delta = right - left
        absolute = delta.abs()
        maximum, l1 = absolute.max().item(), absolute.sum().item()
        del absolute
        l2 = torch.linalg.vector_norm(delta).item()
        norm_left, norm_right = torch.linalg.vector_norm(left).item(), torch.linalg.vector_norm(right).item()
        cosine = None if norm_left == 0 or norm_right == 0 else max(-1.0, min(1.0, torch.dot(left, right).item() / (norm_left * norm_right)))
        changed = int(torch.count_nonzero(self._value.view(torch.int32) != current.view(torch.int32)).item())
        numeric = int(torch.count_nonzero(delta).item())
        return dict(reference=self.summary(), current=metadata, offset=offset, total=width,
                    reference_values=left[page].tolist(), current_values=right[page].tolist(),
                    reference_raw_bits=self._bits(self._value[page]), current_raw_bits=self._bits(current[page]),
                    deltas=delta[page].tolist(), metrics=dict(max_abs_delta=maximum, l1=l1, l2=l2,
                        cosine=cosine, cosine_reason="Zero vector norm" if cosine is None else None,
                        raw_bit_changed_count=changed, numeric_nonzero_count=numeric), scope=SCOPE)
