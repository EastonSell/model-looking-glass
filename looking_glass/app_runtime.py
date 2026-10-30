"""Local-only CPU inference for the application, through existing trace adapters.

No downloads, executable checkpoint code, pickle weights, training, weight edits,
or hidden capture thinning. The app owns one model exclusively while it runs.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import io
import json
import math
import struct
import threading
import uuid

import torch


FAMILIES = {
    "gpt2": {"architecture": "GPT2LMHeadModel", "config": "GPT2Config", "task": "text-generation", "targets": ("attn", "mlp", "block")},
    "distilbert": {"architecture": "DistilBertForMaskedLM", "config": "DistilBertConfig", "task": "fill-mask", "targets": ("attention", "ffn", "block")},
    "t5": {"architecture": "T5ForConditionalGeneration", "config": "T5Config", "task": "text-to-text", "targets": ("self_attention", "feed_forward", "block")},
    "vit": {"architecture": "ViTForImageClassification", "config": "ViTConfig", "task": "image-classification", "targets": ("attention", "intermediate", "block")},
}
DEFAULTS = {"targets": "auto", "layers": None, "max_new_tokens": 8,
            "max_input_tokens": 256, "max_events": 100_000,
            "max_trace_bytes": 128 * 1024 * 1024,
            "max_tensor_bytes": 64 * 1024 * 1024,
            "max_logits_bytes": 128 * 1024 * 1024}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path, limit=1024 * 1024):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"missing or oversized local {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"oversized local {path.name}")
    value = json.loads(data, object_pairs_hook=_unique)
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain an object")
    return value


def _positive(value, name, maximum=None):
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a positive integer" + (f" <= {maximum}" if maximum else ""))
    return value


def _safe_weight_name(name):
    if (type(name) is not str or not name.endswith(".safetensors")
            or Path(name).name != name or "/" in name or "\\" in name or ":" in name):
        raise ValueError("weight shards must be local safetensors filenames, without traversal")
    return name


def _weight_header(path):
    """Bounded, non-executing shape inspection; safetensors validates again on load."""
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
             "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("invalid safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 16 * 1024 * 1024 or length + 8 > size:
            raise ValueError("invalid or oversized safetensors header")
        header = json.loads(stream.read(length), object_pairs_hook=_unique)
    if type(header) is not dict:
        raise ValueError("invalid safetensors metadata")
    estimate, names = 0, set()
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if type(item) is not dict or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("invalid safetensors tensor metadata")
        shape, offsets, dtype = item["shape"], item["data_offsets"], item["dtype"]
        if (type(dtype) is not str or dtype not in sizes or type(shape) is not list or len(shape) > 16
                or any(type(d) is not int or d < 0 for d in shape)
                or type(offsets) is not list or len(offsets) != 2
                or any(type(x) is not int or x < 0 for x in offsets)):
            raise ValueError("unsupported safetensors dtype, shape or offset")
        count = math.prod(shape)
        if offsets[1] - offsets[0] != count * sizes[dtype] or offsets[1] > size - length - 8:
            raise ValueError("safetensors tensor size does not match file")
        estimate += count * max(4, sizes[dtype])
        names.add(name)
    if not names:
        raise ValueError("checkpoint contains no tensors")
    return estimate, names


@dataclass(frozen=True)
class AppCapture:
    """A complete adapter tensor and its original schema plus app coordinates."""
    value: torch.Tensor
    _metadata: dict

    def metadata(self):
        return copy.deepcopy(self._metadata)


def _capture(record, forward, sequence):
    if hasattr(record, "metadata"):
        metadata = record.metadata()
    else:
        metadata = record.event.to_dict()
    metadata = dict(metadata)
    metadata["app_forward_index"] = forward
    metadata["app_sequence"] = sequence
    if hasattr(record, "token_kind"):
        metadata["image_coordinate"] = {name: getattr(record, name) for name in
            ("token_kind", "patch_row", "patch_column", "processed_pixel_box_xyxy")}
    return AppCapture(record.value, metadata)


class LocalModelRuntime:
    """Exclusive CPU float32 model owner. Results live in caller-owned RAM only."""

    def __init__(self, *, max_model_bytes=2 * 1024**3):
        self.max_model_bytes = _positive(max_model_bytes, "max_model_bytes")
        self.model = self.tokenizer = self.processor = None
        self._info = None
        self._lock = threading.Lock()
        self._quarantined = False

    @staticmethod
    def inspect(path):
        if type(path) is not str or not path.strip():
            raise ValueError("select an explicit local checkpoint directory")
        root = Path(path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("checkpoint must be a local directory")
        config = _json(root / "config.json")
        family = config.get("model_type")
        if type(family) is not str or family not in FAMILIES:
            raise ValueError("unsupported architecture; this version supports GPT-2, DistilBERT masked LM, T5 and ViT classifier only")
        spec = FAMILIES[family]
        if config.get("auto_map") or config.get("quantization_config"):
            raise ValueError("custom remote code and quantized checkpoints are unsupported")
        architectures = config.get("architectures")
        if architectures is not None and architectures != [spec["architecture"]]:
            raise ValueError(f"checkpoint must declare {spec['architecture']}, not another head")
        # Bound configuration-driven construction before invoking Transformers.
        for name in ("n_layer", "num_hidden_layers", "num_layers", "num_decoder_layers", "n_layers"):
            if name in config:
                _positive(config[name], name, 256)
        for name in ("n_embd", "hidden_size", "d_model", "dim", "intermediate_size", "hidden_dim", "d_ff"):
            if name in config:
                _positive(config[name], name, 262144)
        if "vocab_size" in config:
            _positive(config["vocab_size"], "vocab_size", 2_000_000)
        index = root / "model.safetensors.index.json"
        if index.exists():
            weight_map = _json(index, 16 * 1024 * 1024).get("weight_map")
            if type(weight_map) is not dict or not weight_map:
                raise ValueError("invalid safetensors shard index")
            filenames = sorted({_safe_weight_name(x) for x in weight_map.values()})
        else:
            weight_map = None
            filenames = ["model.safetensors"]
        total, file_bytes, all_names = 0, 0, set()
        for name in filenames:
            file = root / name
            if not file.is_file():
                raise ValueError(f"missing local safetensors weights: {name}; pickle .bin weights are not loaded")
            estimate, names = _weight_header(file)
            if all_names.intersection(names):
                raise ValueError("duplicate tensors across checkpoint shards")
            if weight_map is not None and any(weight_map.get(key) != name for key in names):
                raise ValueError("shard index does not match tensor headers")
            all_names.update(names)
            total += estimate
            file_bytes += file.stat().st_size
        if weight_map is not None and set(weight_map) != all_names:
            raise ValueError("shard index has missing tensors")
        for filename in ("tokenizer_config.json", "preprocessor_config.json"):
            candidate = root / filename
            if candidate.exists() and _json(candidate).get("auto_map"):
                raise ValueError("custom tokenizer or processor code is unsupported")
        return {"path": str(root), "family": family, "architecture": spec["architecture"],
                "task": spec["task"], "targets": list(spec["targets"]),
                "weight_files": filenames, "checkpoint_file_bytes": file_bytes,
                "estimated_float32_weight_bytes": total,
                "memory_note": "Weight storage estimate only, not peak RAM; activations, logits, loading copies and Python overhead are additional.",
                "config": config}

    def info(self):
        return None if self._info is None else copy.deepcopy(self._info)

    def unload(self):
        with self._lock:
            if self._quarantined:
                raise RuntimeError("model owner is quarantined after a failed forward; restart the local app")
            self.model = self.tokenizer = self.processor = None
            self._info = None

    def load(self, path):
        import transformers
        with self._lock:
            if self._quarantined:
                raise RuntimeError("model owner is quarantined after a failed forward; restart the local app")
            inspected = self.inspect(path)
            if inspected["estimated_float32_weight_bytes"] > self.max_model_bytes:
                raise ValueError("checkpoint float32 weight estimate exceeds model byte limit")
            spec = FAMILIES[inspected["family"]]
            config = getattr(transformers, spec["config"]).from_dict(inspected["config"])
            config._attn_implementation = "eager"
            model_class = getattr(transformers, spec["architecture"])
            with torch.device("meta"):
                shell = model_class(config)
            estimated = sum(x.numel() * x.element_size() for x in shell.parameters())
            del shell
            if estimated > self.max_model_bytes:
                raise ValueError("configuration parameter storage exceeds model byte limit")
            # Discard old model before allocating another. Failed load leaves no model.
            self.model = self.tokenizer = self.processor = None
            self._info = None
            model, report = model_class.from_pretrained(
                inspected["path"], config=config, local_files_only=True,
                use_safetensors=True, trust_remote_code=False,
                torch_dtype=torch.float32, attn_implementation="eager",
                output_loading_info=True)
            if any(report.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError("checkpoint tensors do not exactly match the supported architecture/head")
            model = model.to(device="cpu", dtype=torch.float32).eval()
            model.requires_grad_(False)
            tokenizer = processor = None
            if inspected["family"] == "vit":
                processor = transformers.ViTImageProcessor.from_pretrained(inspected["path"], local_files_only=True)
                size = processor.size
                if (type(size) is not dict or set(size) != {"height", "width"}
                        or any(type(x) is not int or not 0 < x <= 4096 for x in size.values())):
                    raise ValueError("ViT image processor requires bounded explicit height and width <= 4096")
            else:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    inspected["path"], local_files_only=True, trust_remote_code=False,
                    use_fast=True)
                if not tokenizer.is_fast:
                    raise ValueError("a local fast tokenizer is required")
            self.model, self.tokenizer, self.processor = model, tokenizer, processor
            family = inspected["family"]
            if family == "gpt2": depths = {"transformer": len(model.transformer.h)}
            elif family == "distilbert": depths = {"encoder": len(model.distilbert.transformer.layer)}
            elif family == "t5": depths = {"encoder": len(model.encoder.block), "decoder": len(model.decoder.block)}
            else: depths = {"encoder": len(model.vit.encoder.layer)}
            self._info = {key: value for key, value in inspected.items() if key != "config"}
            self._info.update(device="cpu", dtype="float32", state="ready", stack_layers=depths,
                              identity_note="Local directory and configuration labels, not a cryptographic checkpoint fingerprint or execution attestation.",
                              config_summary={key: value for key, value in model.config.to_dict().items() if key in ("model_type", "architectures", "vocab_size", "n_positions", "n_embd", "n_layer", "n_head", "dim", "hidden_dim", "n_layers", "n_heads", "max_position_embeddings", "d_model", "d_ff", "num_layers", "num_decoder_layers", "num_heads", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "image_size", "patch_size", "num_channels", "num_labels")},
                              layers=list(range(max(depths.values()))),
                              parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                              capture_scope="All supported adapter boundaries, not every operation/neuron or proof of complete interpretability.")
            return self.info()

    def _image_geometry(self):
        size = self.model.config.image_size
        dimensions = (size, size) if type(size) is int else tuple(size)
        if (len(dimensions) != 2 or any(type(x) is not int or x <= 0 for x in dimensions)
                or self.model.config.num_channels != 3):
            raise ValueError("image app supports positive RGB model dimensions only")
        expected = {"height": dimensions[0], "width": dimensions[1]}
        if self.processor.size != expected:
            raise ValueError("processor resize dimensions must match the model image dimensions")
        if type(self.processor.do_resize) is not bool:
            raise ValueError("processor resize flag must be boolean")
        if 3 * dimensions[0] * dimensions[1] * 4 > 16 * 1024 * 1024:
            raise ValueError("processed RGB image exceeds the 16 MiB input tensor cap")
        return dimensions

    def _settings(self, settings):
        if type(settings) is not dict or any(type(key) is not str for key in settings):
            raise ValueError("settings must be an object with string keys")
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ValueError("unknown runtime settings: " + ", ".join(sorted(unknown)))
        opts = {**DEFAULTS, **settings}
        for key in DEFAULTS:
            if key not in ("targets", "layers"):
                _positive(opts[key], key)
        _positive(opts["max_new_tokens"], "max_new_tokens", 128)
        _positive(opts["max_input_tokens"], "max_input_tokens", 4096)
        _positive(opts["max_events"], "max_events", 100_000)
        _positive(opts["max_trace_bytes"], "max_trace_bytes", 512 * 1024**2)
        _positive(opts["max_tensor_bytes"], "max_tensor_bytes", 512 * 1024**2)
        _positive(opts["max_logits_bytes"], "max_logits_bytes", 512 * 1024**2)
        family = self._info["family"]
        roles = FAMILIES[family]["targets"]
        targets = opts["targets"]
        if targets == "auto": targets = list(roles)
        if (type(targets) not in (list, tuple) or not targets
                or any(type(t) is not str or t not in roles for t in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError("targets must be auto or a nonempty selection of the model's supported roles")
        opts["targets"] = tuple(t for t in roles if t in targets)
        layers = opts["layers"]
        maximum = max(self._info["stack_layers"].values())
        if layers is None or layers == "all": layers = list(range(maximum))
        if (type(layers) not in (list, tuple) or not layers or any(type(x) is not int or not 0 <= x < maximum for x in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("layers must be all or distinct valid layer indices")
        opts["layers"] = tuple(sorted(layers))
        return opts

    def _text_inputs(self, prompt, opts):
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 65536:
            raise ValueError("enter a nonempty prompt of at most 65,536 characters")
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False,
                                 add_special_tokens=True, return_attention_mask=True)
        ids = encoded["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1 or not ids.numel():
            raise ValueError("tokenizer must produce one nonempty input sequence")
        if ids.shape[1] > opts["max_input_tokens"]:
            raise ValueError("prompt exceeds max_input_tokens; no truncation was performed")
        if bool((ids < 0).any()) or bool((ids >= self.model.config.vocab_size).any()):
            raise ValueError("tokenizer IDs exceed checkpoint vocabulary")
        return ids.to(torch.int64), encoded.get("attention_mask", torch.ones_like(ids)).to(torch.int64)

    def _estimate(self, opts, length, steps, image_tokens=0):
        family, config = self._info["family"], self.model.config
        layers, roles = opts["layers"], opts["targets"]
        if family == "gpt2":
            events = len(layers) * len(roles) * steps
            values = len(layers) * len(roles) * sum(length + i for i in range(steps)) * config.n_embd
            boundary = (length + steps - 1) * config.n_embd * 4
        elif family == "distilbert":
            events = len(layers) * len(roles) * length
            values = events * config.dim
            boundary = length * config.dim * 4
        elif family == "t5":
            enc = sum(x < len(self.model.encoder.block) for x in layers)
            dec = sum(x < len(self.model.decoder.block) for x in layers)
            events = len(roles) * (enc * length * steps + dec * steps * (steps + 1) // 2)
            values = events * config.d_model
            boundary = max(length, steps) * config.d_model * 4
        else:
            events = len(layers) * len(roles) * image_tokens
            values = len(layers) * image_tokens * sum(config.intermediate_size if r == "intermediate" else config.hidden_size for r in roles)
            boundary = image_tokens * max(config.hidden_size, config.intermediate_size) * 4
        estimate = {"capture_events_upper_bound": events, "capture_bytes_upper_bound": values * 4,
                    "largest_selected_boundary_bytes": boundary,
                    "note": "Exact for requested full-length forwards; generation may stop early at EOS. Excludes model/logits, temporary copies and Python metadata; not a peak RAM guarantee."}
        if events > opts["max_events"] or values * 4 > opts["max_trace_bytes"]:
            raise ValueError(f"full requested capture exceeds budget: up to {events} events / {values * 4} bytes; select fewer hooks, shorter input/output, or explicitly raise limits")
        if boundary > opts["max_tensor_bytes"]:
            raise ValueError("selected boundary tensor exceeds max_tensor_bytes")
        return estimate

    def run(self, prompt, settings, image_bytes=None):
        with self._lock:
            if self._quarantined:
                raise RuntimeError("model owner is quarantined after a failed forward; restart the local app")
            if self.model is None:
                raise ValueError("load a supported local model first")
            opts = self._settings(settings)
            return self._run(prompt, opts, image_bytes)

    def _run(self, prompt, opts, image_bytes):
        from .transformer_trace import GPT2TraceSession
        from .trace_schema import GenerationContext
        from .distilbert_trace import trace_distilbert_masked_lm
        from .flan_t5_trace import T5TracePoint, trace_t5_forward
        from .vit_trace import ViTTracePoint, trace_vit_classifier
        family, model = self._info["family"], self.model
        run_id = uuid.uuid4().hex
        captures, steps, used_bytes = [], [], 0
        masks, ids, pixels, image_meta = None, None, None, None
        generated, tokens = [], []
        if family == "vit":
            from PIL import Image
            if type(image_bytes) is not bytes or not image_bytes or len(image_bytes) > 10 * 1024 * 1024:
                raise ValueError("choose an image file no larger than 10 MiB")
            dimensions = self._image_geometry()
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError("image exceeds 16 million pixels")
                if not self.processor.do_resize and (image.height, image.width) != dimensions:
                    raise ValueError("unresized image dimensions must match the model before preprocessing")
                image.load()
                pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(torch.float32)
            size = model.config.image_size
            size = (size, size) if type(size) is int else tuple(size)
            patch = model.config.patch_size
            patch = (patch, patch) if type(patch) is int else tuple(patch)
            count = 1 + (size[0] // patch[0]) * (size[1] // patch[1])
            estimate = self._estimate(opts, 0, 1, count)
            total_steps = 1
        else:
            ids, masks = self._text_inputs(prompt, opts)
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0].tolist())
            total_steps = opts["max_new_tokens"] if family in ("gpt2", "t5") else 1
            limit = getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", None))
            required = ids.shape[1] + (total_steps - 1 if family == "gpt2" else 0)
            if limit is not None and required > limit:
                raise ValueError("requested prompt/generation exceeds model position limit; no truncation was performed")
            if family == "distilbert":
                mask_id = self.tokenizer.mask_token_id
                if mask_id is None or not bool((ids == mask_id).any()):
                    raise ValueError(f"this masked-LM requires a mask token, for example {self.tokenizer.mask_token or '[MASK]'}")
            estimate = self._estimate(opts, ids.shape[1], total_steps)
            vocab = model.config.vocab_size
            logits_tokens = (ids.shape[1] + total_steps - 1 if family == "gpt2" else total_steps if family == "t5" else ids.shape[1])
            if logits_tokens * vocab * 4 > opts["max_logits_bytes"]:
                raise ValueError("requested logits exceed max_logits_bytes")
        start = getattr(model.config, "decoder_start_token_id", None)
        if family == "t5" and (type(start) is not int or not 0 <= start < model.config.vocab_size):
            raise ValueError("T5 checkpoint needs a valid decoder_start_token_id")
        decoder = torch.tensor([[start]], dtype=torch.int64) if family == "t5" else None
        stop_reason, answer = "length", ""
        try:
            for forward in range(total_steps):
                budget = {"max_events": opts["max_events"] - len(captures),
                          "max_trace_bytes": opts["max_trace_bytes"] - used_bytes}
                ident = f"{run_id}-{forward}"
                if family == "gpt2":
                    with torch.inference_mode(), GPT2TraceSession(model, run_id=ident,
                            targets=opts["targets"], layers=opts["layers"], samples=None, tokens=None,
                            mode="direct", capture_device="cpu", **budget,
                            forward_context=GenerationContext(stage="forward", use_cache=False)) as trace:
                        output = model(input_ids=ids, attention_mask=masks, use_cache=False,
                                       output_attentions=False, output_hidden_states=False, return_dict=True)
                    records = trace.events
                elif family == "distilbert":
                    result = trace_distilbert_masked_lm(model, ids, masks, run_id=ident,
                        layers=opts["layers"], modules=opts["targets"], samples=(0,),
                        tokens=tuple(range(ids.shape[1])), **budget,
                        max_tensor_bytes=opts["max_tensor_bytes"])
                    output, records = result.output, result.events
                elif family == "t5":
                    points = [T5TracePoint(stack, layer, role, 0, token)
                        for stack, length in (("encoder", ids.shape[1]), ("decoder", decoder.shape[1]))
                        for layer in opts["layers"] if layer < len(getattr(model, stack).block)
                        for role in opts["targets"] for token in range(length)]
                    output, records = trace_t5_forward(model, input_ids=ids, attention_mask=masks,
                        decoder_input_ids=decoder, decoder_attention_mask=torch.ones_like(decoder),
                        points=points, run_id=ident, **budget,
                        max_tensor_bytes=opts["max_tensor_bytes"], max_input_tokens=opts["max_input_tokens"] + opts["max_new_tokens"],
                        max_logits_bytes=opts["max_logits_bytes"], synchronous_cpu=True)
                else:
                    points = [ViTTracePoint(layer, role, 0, token) for layer in opts["layers"]
                              for role in opts["targets"] for token in range(count)]
                    result = trace_vit_classifier(model, pixels, points=points, run_id=ident, **budget,
                        max_tensor_bytes=opts["max_tensor_bytes"], max_logits_bytes=opts["max_logits_bytes"])
                    output, records, image_meta = result.output, result.records, result.image_metadata
                if not bool(torch.isfinite(output.logits).all()):
                    raise ValueError("model produced nonfinite logits; no completed trace published")
                for record in records:
                    item = _capture(record, forward, len(captures))
                    used_bytes += item.value.numel() * item.value.element_size()
                    captures.append(item)
                if len(captures) > opts["max_events"] or used_bytes > opts["max_trace_bytes"]:
                    raise RuntimeError("adapter exceeded cumulative capture budget")
                step = {"app_forward_index": forward, "run_id": ident, "use_cache": False,
                        "execution": "one full uncached forward", "event_end": len(captures),
                        "input_token_ids": None if ids is None else ids[0].tolist(),
                        "decoder_token_ids": None if decoder is None else decoder[0].tolist()}
                if family in ("gpt2", "t5"):
                    logits = output.logits[0, -1]
                    next_id = int(logits.argmax())
                    scores, choices = torch.topk(logits, min(5, logits.numel()))
                    step["top_next_tokens"] = [{"token_id": int(i), "text": self.tokenizer.decode([int(i)]), "logit": float(s)} for i, s in zip(choices, scores)]
                    step["selected_token_id"] = next_id
                    del logits  # A last-row view otherwise retains the previous full logits allocation.
                    generated.append(next_id)
                    if family == "gpt2":
                        ids = torch.cat((ids, torch.tensor([[next_id]], dtype=torch.int64)), dim=1)
                        masks = torch.ones_like(ids)
                    else:
                        decoder = torch.cat((decoder, torch.tensor([[next_id]], dtype=torch.int64)), dim=1)
                    steps.append(step)
                    eos = model.config.eos_token_id
                    if next_id in (eos if isinstance(eos, list) else [eos]):
                        stop_reason = "eos"
                        break
                elif family == "distilbert":
                    filled = ids.clone()
                    predictions = []
                    for position in (ids[0] == self.tokenizer.mask_token_id).nonzero().flatten().tolist():
                        token = int(output.logits[0, position].argmax())
                        filled[0, position] = token
                        predictions.append({"position": position, "token_id": token, "text": self.tokenizer.decode([token])})
                    step["mask_predictions"] = predictions
                    answer = self.tokenizer.decode(filled[0].tolist(), skip_special_tokens=True)
                    stop_reason = "single masked-LM forward; masks predicted simultaneously"
                    steps.append(step)
                else:
                    probabilities = output.logits[0].softmax(dim=-1)
                    scores, labels = torch.topk(probabilities, min(5, probabilities.numel()))
                    step["classes"] = [{"class_id": int(i), "label": str(model.config.id2label.get(int(i), i)), "probability": float(s)} for i, s in zip(labels, scores)]
                    answer = step["classes"][0]["label"]
                    stop_reason = "single image-classification forward"
                    steps.append(step)
                del output, records
            if family in ("gpt2", "t5"):
                answer = self.tokenizer.decode(generated, skip_special_tokens=True)
            return {"answer": answer, "task": self._info["task"], "events": captures,
                    "model": self.info(), "settings": {**opts, "targets": list(opts["targets"]), "layers": list(opts["layers"])},
                    "prompt": prompt if family != "vit" else None,
                    "prompt_note": "Text input used as supplied" if family != "vit" else "Text prompt is unused for image classification",
                    "steps": steps, "tokens": tokens, "generated_token_ids": generated,
                    "stop_reason": stop_reason, "estimates": estimate, "image_metadata": image_meta,
                    "coverage": {"complete": True, "auto_all_supported": opts["targets"] == FAMILIES[family]["targets"] and opts["layers"] == tuple(self._info["layers"]),
                        "targets": list(opts["targets"]), "layers": list(opts["layers"]),
                        "selected_stack_layers": {stack: [layer for layer in opts["layers"] if layer < depth] for stack, depth in self._info["stack_layers"].items()},
                        "full_feature_vectors": True, "all_executed_token_positions": True,
                        "forward_count": len(steps), "captured_events": len(captures), "captured_bytes": used_bytes,
                        "scope": "Selected adapter boundary outputs only; not every graph node, attention matrix, or complete causal explanation. No capture reduction/truncation."}}
        except BaseException:
            captures.clear()
            # Do not assert that an adapter exception proved cleanup. Retain the
            # owner and refuse reuse/replacement until the app process restarts.
            self._quarantined = True
            self._info["state"] = "quarantined"
            raise
