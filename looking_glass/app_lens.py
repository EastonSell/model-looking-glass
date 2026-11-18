"""Logit-lens read-outs of captured residual-stream vectors.

The logit lens (nostalgebraist, 2020) asks: "if the model stopped at this
layer, which token would it say next?"  For every captured block output ``h``
at (layer, position) it applies the model's own final normalisation and
unembedding::

    logits = unembed(final_norm(h))

and reports the top tokens plus the rank of any pinned tokens.  At the model's
last layer this is exactly the model's real next-token distribution, which
gives a built-in correctness check.

Scope: this is a read-out through the *final* layer's decoder.  Earlier layers
were not trained to be read this way, so early-layer rows are a rough
approximation, not a claim about what the model "thinks".  It is not the
Jacobian lens and does not identify features or causes.
"""
from __future__ import annotations

import torch

METHOD = "logit"
BLOCK_KINDS = {"gpt2": "block", "llama": "llama_block"}
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
MAX_PINNED = 8
MAX_POSITIONS = 128
MAX_CELLS = 4096
MAX_RESOLVE_TEXT = 64
# Logits for one chunk of positions: positions * vocabulary * 4 bytes.
CHUNK_LOGIT_BYTES = 64 * 1024 * 1024
# Logits, log-probabilities, probabilities and comparison temporaries.
WORKSPACE_FACTOR = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum, maximum):
    require(type(value) is int and minimum <= value <= maximum,
            f"{name} must be an integer from {minimum} to {maximum}")
    return value


def decoder_modules(model, family):
    """Return (final_norm, unembedding, description) for a supported family."""
    if family == "gpt2":
        return model.transformer.ln_f, model.lm_head, "transformer.ln_f then lm_head"
    if family == "llama":
        return model.model.norm, model.lm_head, "model.norm then lm_head"
    raise ValueError("The lens read-out supports GPT-2 and standard Llama only")


def chunk_positions(vocabulary_size):
    """How many positions fit in one bounded logits allocation."""
    return max(1, CHUNK_LOGIT_BYTES // (vocabulary_size * 4))


def planned_bytes(position_count, vocabulary_size):
    """Logical workspace estimate for one chunk; not a peak-memory bound."""
    rows = min(position_count, chunk_positions(vocabulary_size))
    return rows * vocabulary_size * 4 * WORKSPACE_FACTOR


class TokenLabels:
    """Decode token IDs to display text once per request."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._cache = {}

    def text(self, token_id):
        if token_id not in self._cache:
            try:
                label = self._tokenizer.decode([token_id])
            except Exception:
                label = None
            if type(label) is not str:
                label = f"<{token_id}>"
            self._cache[token_id] = label[:64]
        return self._cache[token_id]


def resolve_text(tokenizer, text):
    """Offer single-token candidates for typed text, with and without a leading space.

    GPT-2 style vocabularies treat "Paris" and " Paris" as different tokens, and
    mid-sentence predictions are almost always the leading-space form.
    """
    require(type(text) is str and 0 < len(text) <= MAX_RESOLVE_TEXT,
            f"Text to pin must be 1 to {MAX_RESOLVE_TEXT} characters")
    labels = TokenLabels(tokenizer)
    variants = [text]
    if not text.startswith(" "):
        variants.append(" " + text)
    candidates, seen = [], set()
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if not ids or ids[0] in seen:
            continue
        seen.add(ids[0])
        candidates.append({"token_id": int(ids[0]), "text": labels.text(int(ids[0])),
                           "typed": variant, "token_count": len(ids),
                           "single_token": len(ids) == 1})
    return candidates


def project(model, tokenizer, family, sources, *, top_k, pinned_ids, input_token_ids,
            position_offset=0, recorded_next_token_id=None):
    """Decode each layer's captured vectors and summarise every (layer, position) cell.

    ``sources`` is a list of ``{"layer_index", "event_index", "module_path",
    "value"}`` whose values are CPU float32 tensors shaped [positions, width],
    ordered by layer.  ``input_token_ids`` are the input tokens of the same
    position window, which starts at ``position_offset``.  The caller owns
    locking and device admission.
    """
    norm, unembed, decoder = decoder_modules(model, family)
    weight = unembed.weight
    vocabulary = weight.shape[0]
    require(all(0 <= token < vocabulary for token in pinned_ids), "Pinned token ID is outside the vocabulary")
    labels = TokenLabels(tokenizer)
    step = chunk_positions(vocabulary)
    layer_count = len(getattr(model.transformer, "h", ())) if family == "gpt2" else len(model.model.layers)
    reaches_output = sources[-1]["layer_index"] == layer_count - 1

    def decode(hidden):
        """Yield (start, logits, log_probabilities) for bounded position chunks."""
        for start in range(0, hidden.shape[0], step):
            part = hidden[start:start + step].to(device=weight.device, dtype=weight.dtype)
            logits = unembed(norm(part)).to(torch.float32)
            require(bool(torch.isfinite(logits).all()), "Lens read-out produced nonfinite logits")
            yield start, logits, torch.log_softmax(logits, dim=-1)

    with torch.inference_mode():
        # The top captured layer is the reference: when it is the model's last
        # layer, its top token is the model's actual prediction at each position.
        reference_ids = []
        for _, logits, _ in decode(sources[-1]["value"]):
            reference_ids.extend(logits.argmax(dim=-1).tolist())
        reference = torch.tensor(reference_ids, dtype=torch.long, device=weight.device)
        pinned = torch.tensor(pinned_ids, dtype=torch.long, device=weight.device)

        layers = []
        for source in sources:
            cells = []
            for start, logits, log_probabilities in decode(source["value"]):
                rows = logits.shape[0]
                probabilities = log_probabilities.exp()
                entropy = -(probabilities * log_probabilities).sum(dim=-1)
                top_probability, top_ids = torch.topk(probabilities, min(top_k, vocabulary), dim=-1)
                answer = reference[start:start + rows].unsqueeze(1)
                answer_rank = (logits > logits.gather(1, answer)).sum(dim=-1) + 1
                answer_probability = probabilities.gather(1, answer).squeeze(1)
                if len(pinned_ids):
                    columns = pinned.unsqueeze(0).expand(rows, -1)
                    pinned_logits = logits.gather(1, columns)
                    # rank = 1 + number of tokens with a strictly larger logit
                    pinned_rank = (logits.unsqueeze(1) > pinned_logits.unsqueeze(2)).sum(dim=-1) + 1
                    pinned_probability = probabilities.gather(1, columns)
                for row in range(rows):
                    cell = {"position": position_offset + start + row,
                            "top": [{"token_id": int(token), "text": labels.text(int(token)), "probability": float(p)}
                                    for token, p in zip(top_ids[row].tolist(), top_probability[row].tolist())],
                            "entropy": float(entropy[row]),
                            "answer_rank": int(answer_rank[row]),
                            "answer_probability": float(answer_probability[row]),
                            "pinned": []}
                    for column, token in enumerate(pinned_ids):
                        cell["pinned"].append({"token_id": token, "rank": int(pinned_rank[row, column]),
                                               "probability": float(pinned_probability[row, column])})
                    cells.append(cell)
            layers.append({"layer_index": source["layer_index"], "event_index": source["event_index"],
                           "module_path": source["module_path"], "cells": cells})

    answers = [{"position": position_offset + index, "token_id": token, "text": labels.text(token)}
               for index, token in enumerate(reference_ids)]
    tokens = [{"position": position_offset + index, "token_id": token, "text": labels.text(token)}
              for index, token in enumerate(input_token_ids)]
    consistency = None
    if reaches_output and recorded_next_token_id is not None:
        consistency = {"recorded_next_token_id": recorded_next_token_id,
                       "lens_next_token_id": reference_ids[-1],
                       "matches": reference_ids[-1] == recorded_next_token_id}
    return {"method": METHOD, "decoder": decoder, "vocabulary_size": vocabulary,
            "model_layer_count": layer_count, "reaches_model_output": reaches_output,
            "tokens": tokens, "layers": layers, "answers": answers, "consistency": consistency,
            "pinned": [{"token_id": token, "text": labels.text(token)} for token in pinned_ids]}


def status(state):
    """(available, reason) for the current capture; mirrors the live-binding rule for edits."""
    run = state.store.summary()
    if run is None:
        return False, "Run a capture first."
    model = state.runtime.info()
    if model is None or model.get("state") == "quarantined":
        return False, "A ready live model is required to decode captured vectors."
    if model.get("family") not in BLOCK_KINDS:
        return False, "The lens read-out supports GPT-2 and standard Llama captures."
    binding = getattr(state.runtime, "capture_binding", lambda: None)()
    if state._live_binding is None or binding is not state._live_binding:
        return False, "Run a fresh capture from the loaded model; reopened captures cannot be decoded because their model is not verified."
    targets = run.get("coverage", {}).get("targets")
    if type(targets) is list and "block" not in targets:
        return False, "Capture block outputs to decode the residual stream."
    return True, "Decodes captured block outputs with the model's own final norm and unembedding."


def lens_readout(state, data):
    """Validate one browser request and return a bounded layer-by-position read-out."""
    allowed = {"capture_id", "forward", "top_k", "pinned_token_ids", "position_offset", "position_limit", "resolve_text"}
    require(isinstance(data, dict) and not set(data) - allowed, "Unknown lens option")
    available, reason = status(state)
    require(available, reason)
    require(type(data.get("capture_id")) is str and data["capture_id"] == state._capture_id,
            "Capture selection is stale; refresh before reading the lens")
    forward = _integer(data.get("forward", 0), "forward", 0, 2**31)
    top_k = _integer(data.get("top_k", DEFAULT_TOP_K), "top_k", 1, MAX_TOP_K)
    offset = _integer(data.get("position_offset", 0), "position_offset", 0, 2**31)
    limit = _integer(data.get("position_limit", MAX_POSITIONS), "position_limit", 1, MAX_POSITIONS)
    pinned = data.get("pinned_token_ids", [])
    require(type(pinned) is list and len(pinned) <= MAX_PINNED
            and all(type(token) is int and token >= 0 for token in pinned)
            and len(set(pinned)) == len(pinned),
            f"pinned_token_ids must be at most {MAX_PINNED} distinct token IDs")
    family = state.runtime.info()["family"]
    found = state.store.lens_sources(forward, BLOCK_KINDS[family], offset, limit)
    require(len(found["sources"]) * found["position_count"] <= MAX_CELLS,
            f"Lens read-out is limited to {MAX_CELLS} layer-by-position cells; narrow the position window")
    answer = state.runtime.lens_readout(state._live_binding, found["sources"], top_k=top_k, pinned_ids=pinned,
                                        input_token_ids=found["input_token_ids"], position_offset=offset,
                                        recorded_next_token_id=found["recorded_next_token_id"],
                                        resolve=data.get("resolve_text"))
    answer.update(capture_id=state._capture_id, forward=forward, forward_count=found["forward_count"],
                  position_offset=offset, position_total=found["position_total"],
                  skipped_events=found["skipped_events"],
                  scope="Each cell is what the model's own final norm and unembedding read from one captured block "
                        "output. Rows below the model's last layer are an approximation, not the model's reasoning; "
                        "no features, causes or Jacobian lens are computed.")
    return answer
