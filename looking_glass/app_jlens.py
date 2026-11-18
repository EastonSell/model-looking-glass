"""Fit and keep a Jacobian lens for the loaded model.

Fitting is driven by the browser one prompt per request, so progress is
visible, a fit can be stopped between prompts, and no request runs for minutes.
The fitted lens is saved under the lens folder, keyed by a fingerprint of the
model's weights, and reloaded the next time the same model is loaded.
"""
from __future__ import annotations

from pathlib import Path
import secrets
import time

from . import jacobian_lens

MAX_TEXT = 2 * 1024 * 1024
MAX_PROMPTS = 1000
DEFAULT_PROMPTS = 40
MAX_LENS_BYTES = 4 * 1024**3
# Fitting on CPU is allowed only for models this narrow (the built-in demo, tests).
CPU_FIT_MAX_WIDTH = 256
CORPUS = Path(__file__).with_name("jlens_corpus.txt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


class JacobianSession:
    """The lens and any fit in progress for whichever model is loaded now."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        self.fingerprint = None
        self.lens = None
        self.fit = None          # {"id", "state": LensFit, "prompts": [...], "next": int, "dim_batch": int, "seconds": float}

    def reset(self):
        self.fingerprint, self.lens, self.fit = None, None, None

    def sync(self, identity):
        """Point the session at the loaded model; load its saved lens if there is one."""
        if identity is None:
            self.reset()
            return
        if identity["fingerprint"] == self.fingerprint:
            return
        self.reset()
        self.fingerprint = identity["fingerprint"]
        path = jacobian_lens.lens_path(self.root, self.fingerprint)
        if path.is_file():
            try:
                lens = jacobian_lens.JacobianLens.load(path)
            except (ValueError, OSError, KeyError):
                return           # an unreadable file is treated as no lens; refitting replaces it
            if (lens.fingerprint == self.fingerprint and lens.layer_count == identity["layer_count"]
                    and lens.width == identity["width"]):
                self.lens = lens


def builtin_corpus():
    return CORPUS.read_text(encoding="utf-8")


def _identity(state):
    """Identity of the loaded model for lens purposes, or (None, reason)."""
    model = state.runtime.info()
    if model is None or model.get("state") != "ready":
        return None, "Load a model first."
    if model.get("family") not in jacobian_lens.FAMILIES:
        return None, "The Jacobian lens supports GPT-2 and standard Llama."
    if model.get("mode") == "chat" or model.get("dtype") != "float32":
        return None, "Reload the model in Capture mode; the Jacobian lens needs float32 weights."
    return state.runtime.jacobian_identity(), None


def _fit_gate(identity):
    """(allowed, reason): fitting needs a GPU unless the model is tiny."""
    size = (identity["layer_count"] - 1) * identity["width"] ** 2 * 4
    if size > MAX_LENS_BYTES:
        return False, "This model's lens would exceed 4 GiB; fitting it is not supported here."
    passes = jacobian_lens.backward_passes(identity["width"])
    if identity["device"] != "cuda" and identity["width"] > CPU_FIT_MAX_WIDTH:
        return False, (f"Fitting needs the model loaded on an NVIDIA GPU: each prompt takes {passes} backward passes "
                       "through the model. Load the model on CUDA, then fit. A fitted lens can be applied on any device.")
    return True, f"Each prompt takes one forward and {passes} backward passes on {identity['device']}."


def _progress(session):
    fit = session.fit
    if fit is None:
        return None
    state = fit["state"]
    return {"fit_id": fit["id"], "done": fit["next"], "total": len(fit["prompts"]),
            "fitted": state.prompts, "skipped": state.skipped, "seconds_last": fit["seconds"],
            "last_change": state.last_change, "dim_batch": fit["dim_batch"]}


def status(state):
    """What the lens view needs to offer the Jacobian method."""
    session = state.jlens
    identity, reason = _identity(state)
    session.sync(identity)
    if identity is None:
        return {"supported": False, "reason": reason, "lens": None, "fit": None, "can_fit": False, "fit_reason": reason}
    can_fit, fit_reason = _fit_gate(identity)
    return {"supported": True, "reason": None,
            "lens": None if session.lens is None else session.lens.summary(),
            "fit": _progress(session), "can_fit": can_fit, "fit_reason": fit_reason,
            "device": identity["device"], "width": identity["width"], "layer_count": identity["layer_count"],
            "backward_passes": jacobian_lens.backward_passes(identity["width"]),
            "default_prompts": DEFAULT_PROMPTS, "max_prompts": MAX_PROMPTS,
            "scope": "The Jacobian lens carries each layer's vector into the final layer's basis with an averaged "
                     "Jacobian fitted on text, then decodes it as the logit lens does. It follows the estimator of "
                     "Anthropic's reference implementation. It is still a read-out, not a causal explanation, and "
                     "its quality depends on the fitting text."}


def start(state, data):
    require(isinstance(data, dict) and not set(data) - {"source", "text", "max_prompts", "dim_batch"}, "Unknown fit option")
    session = state.jlens
    identity, reason = _identity(state)
    session.sync(identity)
    require(identity is not None, reason)
    allowed, reason = _fit_gate(identity)
    require(allowed, reason)
    source = data.get("source", "builtin")
    require(source in ("builtin", "text"), "source must be builtin or text")
    if source == "text":
        text = data.get("text")
        require(type(text) is str and 0 < len(text) <= MAX_TEXT, "Paste 1 to 2,097,152 characters of plain text")
    else:
        text = builtin_corpus()
    limit = data.get("max_prompts", DEFAULT_PROMPTS)
    require(type(limit) is int and 1 <= limit <= MAX_PROMPTS, f"max_prompts must be an integer from 1 to {MAX_PROMPTS}")
    dim_batch = data.get("dim_batch", jacobian_lens.DIM_BATCH)
    require(type(dim_batch) is int and 1 <= dim_batch <= 64, "dim_batch must be an integer from 1 to 64")
    prompts = state.runtime.jacobian_prompts(text, limit)
    require(prompts, f"No passage is longer than {jacobian_lens.SKIP_FIRST + 1} tokens; add longer text. "
                     "Separate passages with a blank line.")
    session.fit = {"id": secrets.token_hex(16), "prompts": prompts, "next": 0, "dim_batch": dim_batch, "seconds": None,
                   "state": jacobian_lens.LensFit(identity["fingerprint"], identity["family"],
                                                  identity["layer_count"], identity["width"])}
    return status(state)


def step(state, data):
    """Fit on the next prompt. The last step saves the lens and makes it the model's lens."""
    require(isinstance(data, dict) and set(data) == {"fit_id"}, "Choose the current fit")
    session = state.jlens
    identity, reason = _identity(state)
    require(identity is not None, reason)
    require(identity["fingerprint"] == session.fingerprint and session.fit is not None
            and data["fit_id"] == session.fit["id"], "That fit is no longer active; start it again")
    fit = session.fit
    if fit["next"] < len(fit["prompts"]):
        token_ids = fit["prompts"][fit["next"]]
        began = time.perf_counter()
        try:
            jacobians, valid = state.runtime.jacobian_fit_prompt(token_ids, fit["dim_batch"])
            fit["state"].add(jacobians, valid)
        except ValueError:
            fit["state"].skipped += 1      # for example a prompt the model's context cannot hold
        fit["seconds"] = time.perf_counter() - began
        fit["next"] += 1
    if fit["next"] >= len(fit["prompts"]):
        lens = fit["state"].lens()
        lens.save(jacobian_lens.lens_path(session.root, session.fingerprint))
        session.lens, session.fit = lens, None
    return status(state)


def import_file(state, data):
    """Use a lens fitted elsewhere with Anthropic's jlens library; it becomes this model's saved lens."""
    require(isinstance(data, dict) and set(data) == {"path"} and type(data["path"]) is str
            and 0 < len(data["path"]) <= 4096, "Give the path of a lens file")
    session = state.jlens
    identity, reason = _identity(state)
    session.sync(identity)
    require(identity is not None, reason)
    require(session.fit is None, "Stop the fit in progress before importing a lens")
    lens = jacobian_lens.JacobianLens.from_reference_file(
        Path(data["path"]).expanduser(), fingerprint=identity["fingerprint"], family=identity["family"],
        layer_count=identity["layer_count"], width=identity["width"])
    lens.save(jacobian_lens.lens_path(session.root, session.fingerprint))
    session.lens = lens
    return status(state)


def cancel(state, data):
    require(isinstance(data, dict) and set(data) == {"fit_id"}, "Choose the current fit")
    session = state.jlens
    if session.fit is not None and session.fit["id"] == data["fit_id"]:
        session.fit = None
    return status(state)


def transport_for(state):
    """The fitted lens for the loaded model, or raise with what to do."""
    session = state.jlens
    identity, reason = _identity(state)
    session.sync(identity)
    require(identity is not None, reason)
    require(session.lens is not None, "No Jacobian lens is fitted for this model yet. Fit one first.")
    return session.lens
