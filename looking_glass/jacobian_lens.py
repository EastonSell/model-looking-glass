"""Jacobian lens: fit, save, load and apply.

The Jacobian lens (Anthropic, "Verbalizable Representations Form a Global
Workspace in Language Models", 2026) reads a residual-stream vector ``h`` at
layer ``l`` by first carrying it into the final layer's basis with an averaged
Jacobian, then decoding it with the model's own final norm and unembedding::

    lens_l(h) = unembed(final_norm(J_l @ h)),    J_l = E[d h_final / d h_l]

Estimator.  This module follows the estimator of Anthropic's reference
implementation (github.com/anthropics/jacobian-lens, ``jlens.fitting``,
Apache-2.0) so that a lens fitted here means the same thing; the code is
independent.  For one prompt and one output dimension ``i``, a one-hot
cotangent is placed on dimension ``i`` of the final block's output at every
valid target position at once and back-propagated.  The gradient at source
position ``p`` is then the sum over later target positions of
``d h_final[p'][i] / d h_l[p]``.  Its mean over valid source positions is row
``i`` of that prompt's ``J_l``.  Rows are averaged over prompts.

Valid positions skip the first ``skip_first`` tokens (they behave as attention
sinks) and the last token.

Cost: one forward and ``ceil(d_model / dim_batch)`` backward passes per prompt.
That is why the app only fits on a GPU; applying a fitted lens is one matrix
product per layer and runs anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .model_ownership import ModelLease

SKIP_FIRST = 16
MAX_TOKENS = 128
DIM_BATCH = 8
FORMAT = "model-looking-glass-jacobian-lens-v1"
FAMILIES = ("gpt2", "llama")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decoder_stack(model, family):
    """Return (text_decoder, blocks) for a supported causal language model."""
    if family == "gpt2":
        return model.transformer, model.transformer.h
    if family == "llama":
        return model.model, model.model.layers
    raise ValueError("The Jacobian lens supports GPT-2 and standard Llama only")


def valid_positions(sequence_length, skip_first=SKIP_FIRST):
    """Positions included in the average: after the first ``skip_first``, before the last."""
    require(type(skip_first) is int and skip_first >= 0, "skip_first must be a nonnegative integer")
    positions = list(range(skip_first, sequence_length - 1))
    require(positions, f"prompt too short: {sequence_length} tokens, need more than {skip_first + 1}")
    return positions


def model_fingerprint(model, family):
    """Identify the loaded weights so a lens is only ever applied to the model it was fitted on.

    Hashes the dimensions, the unembedding, and one weight from the first and
    last block.  This is an identity check for local files, not a security
    boundary.
    """
    _, blocks = decoder_stack(model, family)
    digest = hashlib.sha256(FORMAT.encode())
    unembedding = model.lm_head.weight
    digest.update(json.dumps([family, len(blocks), list(unembedding.shape)]).encode())
    chosen = [unembedding]
    for block in (blocks[0], blocks[-1]):
        largest = None
        for parameter in block.parameters():
            if parameter.ndim == 2 and (largest is None or parameter.numel() > largest.numel()):
                largest = parameter
        require(largest is not None, "model blocks have no weight matrices to fingerprint")
        chosen.append(largest)
    for tensor in chosen:
        digest.update(tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def prompt_jacobians(model, family, input_ids, *, dim_batch=DIM_BATCH, skip_first=SKIP_FIRST):
    """Per-layer Jacobian estimate for one tokenised prompt.

    ``input_ids`` is a ``[1, tokens]`` integer tensor on the model's device.
    Returns ``(jacobians, valid_count)`` where ``jacobians[layer]`` is a
    ``[d_model, d_model]`` float32 CPU tensor for every layer below the last.

    The caller owns locking.  Model parameters must not require gradients; the
    graph is rooted at the first block's output, so no parameter gradient is
    ever computed or stored.
    """
    require(type(dim_batch) is int and 1 <= dim_batch <= 256, "dim_batch must be an integer from 1 to 256")
    require(input_ids.ndim == 2 and input_ids.shape[0] == 1, "input_ids must have shape [1, tokens]")
    decoder, blocks = decoder_stack(model, family)
    last = len(blocks) - 1
    require(last >= 1, "The Jacobian lens needs a model with at least two layers")
    require(not any(parameter.requires_grad for parameter in model.parameters()),
            "model parameters must not require gradients while fitting")
    positions = valid_positions(input_ids.shape[1], skip_first)

    outputs = {}

    def record(index):
        def hook(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            if index == 0:
                tensor.requires_grad_(True)     # root of the graph: nothing earlier is retained
            outputs[index] = tensor
        return hook

    lease = ModelLease(model)
    handles = []
    try:
        lease.acquire()
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(record(index)))
        with torch.inference_mode(False), torch.enable_grad():
            decoder(input_ids=input_ids.expand(dim_batch, -1), use_cache=False)
            target = outputs[last]                              # [dim_batch, tokens, d_model]
            sources = [outputs[index] for index in range(last)]
            width = target.shape[-1]
            valid = torch.tensor(positions, dtype=torch.long, device=target.device)
            rows = torch.arange(dim_batch, device=target.device)
            cotangent = torch.zeros_like(target)
            jacobians = [torch.zeros(width, width, dtype=torch.float32) for _ in sources]
            starts = list(range(0, width, dim_batch))
            for start in starts:
                count = min(dim_batch, width - start)
                # Batch element b carries output dimension start + b, at every valid target position.
                cotangent.zero_()
                cotangent[rows[:count, None], valid[None, :], start + rows[:count, None]] = 1.0
                gradients = torch.autograd.grad(target, sources, grad_outputs=cotangent,
                                                retain_graph=start != starts[-1])
                for layer, gradient in enumerate(gradients):
                    chunk = gradient[:count].index_select(1, valid.to(gradient.device)).float().mean(dim=1)
                    jacobians[layer][start:start + count] = chunk.cpu()
                del gradients
    finally:
        for handle in handles:
            handle.remove()
        outputs.clear()
        lease.release()
    for matrix in jacobians:
        require(bool(torch.isfinite(matrix).all()), "Jacobian estimate is not finite")
    return {layer: matrix for layer, matrix in enumerate(jacobians)}, len(positions)


class LensFit:
    """Running mean of per-prompt Jacobians for one model."""

    def __init__(self, fingerprint, family, layer_count, width, *, skip_first=SKIP_FIRST, max_tokens=MAX_TOKENS):
        self.fingerprint, self.family = fingerprint, family
        self.layer_count, self.width = layer_count, width
        self.skip_first, self.max_tokens = skip_first, max_tokens
        self.sums = {layer: torch.zeros(width, width, dtype=torch.float32) for layer in range(layer_count - 1)}
        self.prompts = 0
        self.positions = 0
        self.skipped = 0
        self.last_change = None

    def add(self, jacobians, valid_count):
        """Fold one prompt in; ``last_change`` is how far it moved the running mean (falls as 1/n)."""
        require(sorted(jacobians) == sorted(self.sums), "prompt Jacobians do not match the fitted layers")
        if self.prompts:
            change = 0.0
            for layer, matrix in jacobians.items():
                mean = self.sums[layer] / self.prompts
                change = max(change, float((matrix - mean).norm() / ((self.prompts + 1) * mean.norm().clamp_min(1e-30))))
            self.last_change = change
        for layer, matrix in jacobians.items():
            self.sums[layer] += matrix
        self.prompts += 1
        self.positions += valid_count

    def lens(self):
        require(self.prompts > 0, "no prompt was long enough to fit on")
        matrices = {layer: total / self.prompts for layer, total in self.sums.items()}
        return JacobianLens(matrices, fingerprint=self.fingerprint, family=self.family,
                            layer_count=self.layer_count, prompts=self.prompts, positions=self.positions,
                            skip_first=self.skip_first, max_tokens=self.max_tokens)


class JacobianLens:
    """Fitted ``J_l`` matrices for every layer below the last, bound to one model's weights."""

    def __init__(self, matrices, *, fingerprint, family, layer_count, prompts, positions, skip_first, max_tokens,
                 origin="fitted in this app"):
        require(matrices and sorted(matrices) == list(range(layer_count - 1)),
                "a Jacobian lens needs one matrix for every layer below the last")
        width = next(iter(matrices.values())).shape[0]
        for matrix in matrices.values():
            require(tuple(matrix.shape) == (width, width) and bool(torch.isfinite(matrix).all()),
                    "Jacobian matrices must be finite and square")
        self.matrices = {layer: matrix.to(torch.float32) for layer, matrix in matrices.items()}
        self.fingerprint, self.family, self.layer_count, self.width = fingerprint, family, layer_count, width
        self.prompts, self.positions = prompts, positions
        self.skip_first, self.max_tokens = skip_first, max_tokens
        self.origin = str(origin)[:200]

    def summary(self):
        return {"method": "jacobian", "prompts": self.prompts, "positions": self.positions, "origin": self.origin,
                "layers": self.layer_count - 1, "width": self.width,
                "skip_first": self.skip_first, "max_tokens": self.max_tokens,
                "bytes": len(self.matrices) * self.width * self.width * 4}

    def transport(self, hidden, layer):
        """Carry ``[..., width]`` residuals at ``layer`` into the final layer's basis: ``J_l @ h``."""
        matrix = self.matrices[layer].to(device=hidden.device, dtype=hidden.dtype)
        return hidden @ matrix.T

    def save(self, path):
        """Write a safetensors file (no pickle) atomically, with the fit's provenance in its header."""
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = {"format": FORMAT, "fingerprint": self.fingerprint, "family": self.family,
                    "layer_count": str(self.layer_count), "prompts": str(self.prompts),
                    "positions": str(self.positions), "skip_first": str(self.skip_first),
                    "max_tokens": str(self.max_tokens), "origin": self.origin,
                    "estimator": "cotangents summed over valid target positions, mean over valid source positions"}
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        try:
            save_file({f"layer.{layer}": matrix.contiguous() for layer, matrix in self.matrices.items()},
                      str(temporary), metadata=metadata)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, *, max_bytes=8 * 1024**3):
        from safetensors import safe_open
        path = Path(path)
        require(path.is_file() and path.stat().st_size <= max_bytes, "Jacobian lens file is missing or too large")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            require(metadata.get("format") == FORMAT, "not a Model Looking Glass Jacobian lens file")
            layer_count = int(metadata["layer_count"])
            matrices = {}
            for layer in range(layer_count - 1):
                matrices[layer] = source.get_tensor(f"layer.{layer}")
        return cls(matrices, fingerprint=metadata["fingerprint"], family=metadata["family"],
                   layer_count=layer_count, prompts=int(metadata["prompts"]), positions=int(metadata["positions"]),
                   skip_first=int(metadata["skip_first"]), max_tokens=int(metadata["max_tokens"]),
                   origin=metadata.get("origin", "fitted in this app"))

    @classmethod
    def from_reference_file(cls, path, *, fingerprint, family, layer_count, width, max_bytes=8 * 1024**3):
        """Read a lens saved by Anthropic's ``jlens`` library, such as the published Neuronpedia lenses.

        Those files are ``torch.save`` dictionaries; they are read with
        ``weights_only=True``, which loads tensors and plain values but runs no
        code from the file.  The file does not say which model it belongs to, so
        only its shape can be checked against the loaded model.
        """
        path = Path(path)
        require(path.is_file() and path.stat().st_size <= max_bytes, "lens file is missing or too large")
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(f"could not read the lens file as tensors only ({type(error).__name__})") from error
        require(isinstance(data, dict) and isinstance(data.get("J"), dict), "not a jlens file: no 'J' dictionary of matrices")
        provenance = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
        target = provenance.get("target_layer")
        require(target in (None, layer_count - 1, -1), "this lens maps to a layer other than the model's last")
        require(provenance.get("stream_reduce") in (None, "none"), "this lens reads a combined stream the app does not capture")
        require(sorted(data["J"]) == list(range(layer_count - 1)),
                f"this lens has matrices for layers {sorted(data['J'])[:3]}..., but the loaded model needs 0 to {layer_count - 2}")
        require(data.get("d_model") == width and all(torch.is_tensor(m) and tuple(m.shape) == (width, width)
                                                      for m in data["J"].values()),
                f"this lens is for a model of another width; the loaded model's width is {width}")
        prompts = data.get("n_prompts")
        require(type(prompts) is int and prompts > 0, "lens file has no prompt count")
        return cls({layer: matrix.to(torch.float32) for layer, matrix in data["J"].items()},
                   fingerprint=fingerprint, family=family, layer_count=layer_count, prompts=prompts, positions=0,
                   skip_first=SKIP_FIRST, max_tokens=MAX_TOKENS, origin="imported from " + path.name)


def lens_path(root, fingerprint):
    """Where the lens for one model fingerprint lives under the app's lens folder."""
    require(type(fingerprint) is str and len(fingerprint) == 64 and all(c in "0123456789abcdef" for c in fingerprint),
            "invalid model fingerprint")
    return Path(root) / f"{fingerprint[:32]}.safetensors"


def split_passages(text, tokenizer, *, max_tokens=MAX_TOKENS, skip_first=SKIP_FIRST, limit=1000):
    """Turn plain text into fitting prompts: one per blank-line-separated passage that is long enough.

    A passage longer than ``max_tokens`` is cut into consecutive windows, so a
    single long document also works.  Returns a list of token-ID lists.
    """
    prompts = []
    for passage in text.replace("\r\n", "\n").split("\n\n"):
        passage = " ".join(passage.split())
        if not passage:
            continue
        ids = tokenizer(passage, add_special_tokens=True)["input_ids"]
        for start in range(0, len(ids), max_tokens):
            window = ids[start:start + max_tokens]
            if len(window) > skip_first + 1:
                prompts.append(window)
            if len(prompts) >= limit:
                return prompts
    return prompts


def backward_passes(width, dim_batch=DIM_BATCH):
    return math.ceil(width / dim_batch)
