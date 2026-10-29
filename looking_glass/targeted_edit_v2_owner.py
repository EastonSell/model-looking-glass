"""Supervised CPU owner for the prospective targeted-edit v2 study.

The process supervisor, source manifest, public custody receipt and independent
remote witnesses are separate requirements; no one record proves execution
or publication chronology. Model imports are intentionally lazy: the direct
child must acquire its CPU study lease before importing Torch or Transformers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid
from datetime import datetime, timezone


class OwnerError(RuntimeError):
    """The CPU owner could not establish a valid experiment outcome."""


class Incompatible(OwnerError):
    """The independently pinned v1 reconstruction does not match."""


# A source-bound independent remote-witness gate is still under review.
# Keep the live entry point closed until that gate is integrated and tested.
LIVE_DEVELOPMENT_ENABLED = False


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write_new(path, data):
    """Create a bounded, durable record without replacing an earlier attempt."""
    if type(data) is not bytes:
        raise TypeError("artifact bytes are required")
    with open(path, "xb") as handle:
        if handle.write(data) != len(data):
            raise OwnerError("short artifact write")
        handle.flush()
        os.fsync(handle.fileno())


def _json_new(path, value):
    _write_new(path, _canonical(value))


def _path_absent(path):
    """Check lexical absence, including dangling links, in a trusted namespace."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise OwnerError("publication path could not be inspected") from exc
    return False


def _modules():
    """Import model dependencies only after the supervised child has claimed."""
    import torch
    import numpy as np
    from looking_glass import pretrained
    from looking_glass import targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_cpu_v2 as v2
    from looking_glass import targeted_edit_v2_edits as edits
    from looking_glass import targeted_edit_v2_evidence as evidence
    from looking_glass import targeted_edit_v2_manifest as manifest
    from looking_glass import targeted_edit_v2_custody as custody
    from looking_glass import targeted_edit_v2_exposure as exposure
    from looking_glass import targeted_edit_v2_gate as gate
    return torch, np, pretrained, v1, v2, edits, evidence, manifest, custody, exposure, gate


def _verify_import_origins(repo, manifest):
    """Require every loaded project module to be one of the frozen sources.

    This is an integrity check in a trusted, quiescent checkout, not an import
    sandbox or an attestation of native libraries or code already executed.
    """
    root = Path(repo).resolve(strict=True)
    hashes = manifest.get("source_sha256")
    if type(hashes) is not dict:
        raise OwnerError("manifest source inventory is unavailable")
    for name, module in tuple(sys.modules.items()):
        if name != "looking_glass" and not name.startswith("looking_glass."):
            continue
        filename = getattr(module, "__file__", None)
        if type(filename) is not str:
            raise OwnerError("project module has no source origin")
        path = Path(filename).resolve(strict=True)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise OwnerError("project module came from another checkout") from exc
        expected = hashes.get(relative)
        if path.suffix != ".py" or type(expected) is not str:
            raise OwnerError("project import is outside the frozen source inventory")
        with open(path, "rb") as handle:
            data = handle.read(1_000_001)
        if len(data) > 1_000_000 or _sha(data) != expected:
            raise OwnerError("loaded project source differs from its frozen bytes")
    owner = Path(__file__).resolve(strict=True)
    relative = owner.relative_to(root).as_posix()
    if relative != "looking_glass/targeted_edit_v2_owner.py" or hashes.get(relative) != _sha(owner.read_bytes()):
        raise OwnerError("owner source differs from the frozen checkout")


def _has_hooks(model, torch):
    # State-dict hooks can rewrite what _model_hash observes, concealing a
    # mutation even when the forward/backward hook registries are empty.
    names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
             "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
             "_load_state_dict_post_hooks")
    if any(getattr(module, name, None) for module in model.modules() for name in names):
        return True
    registry = torch.nn.modules.module
    global_names = ("_global_forward_hooks", "_global_forward_pre_hooks",
                    "_global_backward_hooks", "_global_backward_pre_hooks")
    return any(getattr(registry, name, None) for name in global_names)


def _buffer_snapshot(model, modules, torch):
    """Include buffers omitted from state_dict, including nonpersistent ones."""
    rows = []
    for name, tensor in model.named_buffers(remove_duplicate=False):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.layout != torch.strided or tensor.is_quantized):
            raise OwnerError("unsupported model buffer")
        packed = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        digest = hashlib.sha256(memoryview(packed.numpy())).hexdigest()
        rows.append((name, tensor, str(tensor.dtype), tuple(tensor.shape), digest))
    registration = tuple((name, module, tuple(sorted(module._buffers)),
                          tuple(sorted(module._non_persistent_buffers_set))) for name, module in modules)
    return tuple(rows), registration


def _buffer_state_digest(rows, registration):
    """Commit to ordered buffer names/types/shapes/content and registrations."""
    return _sha(_canonical({
        "version": "targeted-edit-v2-buffers-v1",
        "buffers": [{"name": name, "dtype": dtype, "shape": list(shape), "sha256": digest}
                    for name, _tensor, dtype, shape, digest in rows],
        "registrations": [{"module": name, "names": list(names), "nonpersistent": list(nonpersistent)}
                          for name, _module, names, nonpersistent in registration],
    }))


def _runtime_settings(torch):
    """Snapshot selected execution settings newly attested by v2, not v1."""
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
            "inference_mode_enabled": torch.is_inference_mode_enabled(),
            "cuda_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_initialized": torch.cuda.is_initialized(),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS")}


class _ModelGuard:
    """Keep the private model at its baseline between every condition."""

    def __init__(self, model, module, weight, torch, v1):
        self.model, self.module, self.weight, self.torch, self.v1 = model, module, weight, torch, v1
        self.parameters = tuple(model.named_parameters(remove_duplicate=False))
        self.modules = tuple(model.named_modules(remove_duplicate=False))
        self.flags = tuple((parameter, parameter.requires_grad) for _, parameter in self.parameters)
        self.training = tuple((item, item.training) for _, item in self.modules)
        if any(parameter.grad is not None for _, parameter in self.parameters) or _has_hooks(model, torch):
            raise OwnerError("model has preexisting gradients or hooks")
        self.buffers, self.buffer_registration = _buffer_snapshot(model, self.modules, torch)
        self.buffer_sha = _buffer_state_digest(self.buffers, self.buffer_registration)
        self.runtime_settings = _runtime_settings(torch)
        self.original = weight.detach().clone()
        self.weight_sha = v1._tensor_sha(self.original)
        self.state_sha = v1._model_hash(model)
        self.poisoned = False
        self.last_cleanup = "not_attempted"
        self._condition_ready = False
        self.verify()

    def verify(self, *, _during_cleanup=False):
        if self.poisoned and not _during_cleanup:
            raise OwnerError("model custody was poisoned by a failed condition")
        try:
            now_parameters = tuple(self.model.named_parameters(remove_duplicate=False))
            now_modules = tuple(self.model.named_modules(remove_duplicate=False))
            now_buffers, now_registration = _buffer_snapshot(self.model, now_modules, self.torch)
            if (len(now_parameters) != len(self.parameters) or len(now_modules) != len(self.modules)
                    or any(a != b or x is not y for (a, x), (b, y) in zip(now_parameters, self.parameters))
                    or any(a != b or x is not y for (a, x), (b, y) in zip(now_modules, self.modules))
                    or any(parameter.requires_grad != flag or parameter.grad is not None
                           for parameter, flag in self.flags)
                    or any(item.training != flag for item, flag in self.training)
                    or _has_hooks(self.model, self.torch)
                    or _runtime_settings(self.torch) != self.runtime_settings
                    or len(now_buffers) != len(self.buffers)
                    or any(a != b or x is not y or c != d or e != f or g != h
                           for (a, x, c, e, g), (b, y, d, f, h) in zip(now_buffers, self.buffers))
                    or len(now_registration) != len(self.buffer_registration)
                    or any(n != m or x is not y or a != b or c != d
                           for (n, x, a, c), (m, y, b, d) in zip(now_registration, self.buffer_registration))
                    or self.v1._tensor_sha(self.weight) != self.weight_sha
                    or self.v1._model_hash(self.model) != self.state_sha):
                raise OwnerError("model baseline or cleanup no longer matches")
        except BaseException:
            self.poisoned = True
            raise

    def restore(self):
        """Restore the sole edited weight and verify all other baseline state."""
        try:
            with self.torch.no_grad():
                self.weight.copy_(self.original)
            self.verify(_during_cleanup=True)
        except BaseException:
            self.poisoned = True
            raise

    def condition(self, delta, expected_applied_sha, run_rows):
        """Execute one condition and never continue after any failed cleanup."""
        if self.poisoned:
            raise OwnerError("model custody was poisoned by a failed condition")
        # Fence before even entering the protected region. An asynchronous
        # interruption in the transition cannot permit a later condition on
        # a model that may have been edited.
        self.poisoned = True
        self.last_cleanup = "unverified"
        self._condition_ready = False
        edit_started = False
        try:
            self.verify(_during_cleanup=True)
            edit_started = True
            with self.torch.no_grad():
                if delta is not None:
                    self.weight.add_(delta)
                if self.v1._tensor_sha(self.weight) != expected_applied_sha:
                    raise OwnerError("applied weight differs from its attestation")
            run_rows()
            self.restore()
            self.last_cleanup = "verified"
            self._condition_ready = True
        except BaseException as primary:
            if edit_started and self.last_cleanup != "verified":
                try:
                    self.restore()
                    self.last_cleanup = "verified"
                except BaseException as cleanup:
                    self.poisoned = True
                    if (isinstance(cleanup, (KeyboardInterrupt, SystemExit))
                            and not isinstance(primary, (KeyboardInterrupt, SystemExit))):
                        raise cleanup from primary
            # Even a verified rollback cannot turn a failed condition into a
            # complete study. The isolated child exits without further work.
            self.poisoned = True
            raise

    def commit_condition(self):
        """Reopen the guard only after the caller durably records the condition.

        A lost condition return leaves the guard fenced. A lost acknowledgement
        return still requires the caller to stop and reconcile the journal; it
        must not replay that condition blindly.
        """
        if not self.poisoned or not self._condition_ready or self.last_cleanup != "verified":
            raise OwnerError("condition is not ready for acknowledgement")
        self.verify(_during_cleanup=True)
        self._condition_ready = False
        self.poisoned = False


def _verify_tokenizer(tokenizer, v1, v2):
    v1._verify_tokenizer(tokenizer)
    for _name, role, text, ids, label in v2.EVALUATION_CASES:
        if tokenizer.encode(text, add_special_tokens=False) != list(ids) or not 1 <= len(ids) <= 32:
            raise OwnerError("evaluation prompt tokenization differs")
        continuations = (label,) if role == "locality" else ((" London", v1.LONDON), (" Paris", v1.PARIS))
        for continuation, token_id in continuations:
            if (tokenizer.encode(continuation, add_special_tokens=False) != [token_id]
                    or tokenizer.encode(text + continuation, add_special_tokens=False) != list(ids) + [token_id]):
                raise OwnerError("evaluation continuation tokenization differs")


def _verify_unshared_site(model, weight, site, torch):
    """Reject any parameter or buffer whose storage overlaps the edited site."""
    storage = weight.untyped_storage()
    start, length = storage.data_ptr(), storage.nbytes()
    if not start or length <= 0:
        raise Incompatible("edited parameter has no ordinary CPU storage")
    found = 0
    tensors = ((*model.named_parameters(remove_duplicate=False),), (*model.named_buffers(remove_duplicate=False),))
    for group in tensors:
        for name, tensor in group:
            if name == site and tensor is weight:
                found += 1
                continue
            if tensor.device.type != "cpu" or tensor.layout != torch.strided:
                raise Incompatible("a parameter or buffer has unsupported storage")
            other = tensor.untyped_storage()
            other_start, other_length = other.data_ptr(), other.nbytes()
            if other_length and other_start < start + length and start < other_start + other_length:
                raise Incompatible("another parameter or buffer shares edited-site storage")
    if found != 1:
        raise Incompatible("edited-site identity is not unique")


def _check_checkpoint(loaded, manifest, torch, v1):
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
    model = loaded.model
    if (model.training or any(item.training for item in model.modules())
            or model.config.use_cache or getattr(model.config, "output_hidden_states", False)
            or getattr(model.config, "output_attentions", False)
            or getattr(model.config, "_attn_implementation", None) != "eager"
            or type(model) is not GPT2LMHeadModel
            or type(loaded.tokenizer) is not GPT2TokenizerFast
            or sum(p.numel() for p in model.parameters()) != v1.PARAMETER_COUNT
            or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or _canonical({**loaded.metadata, "evaluation_mode": True}) !=
               _canonical(manifest["v1_compatibility"]["model"])):
        raise Incompatible("checkpoint or recorded v1 model metadata differs")
    try:
        module = model.transformer.h[5].mlp.c_proj
        weight = module.weight
        named = dict(model.named_parameters(remove_duplicate=False))
    except (AttributeError, IndexError, TypeError) as exc:
        raise Incompatible("the frozen named site is unavailable") from exc
    if (named.get(v1.SITE) is not weight or tuple(weight.shape) != v1.WEIGHT_SHAPE
            or weight.device.type != "cpu" or weight.dtype != torch.float32
            or not bool(torch.isfinite(weight).all())):
        raise Incompatible("the exact named projection parameter differs")
    _verify_unshared_site(model, weight, v1.SITE, torch)
    return model, module, weight


def _append(handle, data):
    if handle.write(data) != len(data):
        raise OwnerError("short evidence write")
    handle.flush()
    os.fsync(handle.fileno())


def _development_conditions(output, identity, guard, updates, grid, discovery, v1, v2, evidence):
    """Write each row and its identity before moving to the next condition."""
    row_size = v1.VOCAB_SIZE * 4
    with open(output / "observations.jsonl", "xb") as journal, open(output / "condition_states.jsonl", "xb") as states:
        for condition in evidence.CONDITIONS:
            applied = guard.weight_sha if condition == "original" else grid[condition]["realized_weight_sha256"]
            with open(output / f"{condition}.f32", "xb") as raw:
                def run_rows():
                    for index, case in enumerate(v1.CASES):
                        before = v1._denormal_probe()
                        row = v1._logit_row(guard.model, case[3])
                        after = v1._denormal_probe()
                        if len(row) != row_size:
                            raise OwnerError("a serialized logit row has the wrong size")
                        if condition == "original" and index == 0:
                            if v1._sha(row) != discovery["discovery_logit_sha256"]:
                                raise Incompatible("baseline discovery row differs from the reconstructed discovery")
                        _append(raw, row)
                        _append(journal, v1._canonical({"version": v2.VERSION, "phase": evidence.PHASE,
                            "run_id": identity["run_id"], "condition": condition, "case_id": case[0],
                            "offset": index * row_size, "bytes": row_size, "sha256": v1._sha(row),
                            "denormal_probe_before": before, "denormal_probe_after": after}))
                guard.condition(None if condition == "original" else updates[condition], applied, run_rows)
            _append(states, v1._canonical({"version": v2.VERSION, "phase": evidence.PHASE,
                "run_id": identity["run_id"], "condition": condition,
                "model_state_before_sha256": guard.state_sha, "weight_before_sha256": guard.weight_sha,
                "buffer_state_before_sha256": guard.buffer_sha,
                "applied_weight_sha256": applied, "model_state_after_sha256": guard.state_sha,
                "weight_after_sha256": guard.weight_sha, "buffer_state_after_sha256": guard.buffer_sha,
                "flags_restored": True,
                "grads_cleared": True, "hooks_restored": True}))
            guard.commit_condition()


def _read_development_arrays(output, v1, v2, evidence, np):
    size = len(v1.CASES) * v1.VOCAB_SIZE * 4
    arrays = {}
    for condition in evidence.CONDITIONS:
        data = evidence._read(output / f"{condition}.f32", size)
        if len(data) != size:
            raise OwnerError("development matrix is incomplete")
        arrays[condition] = np.frombuffer(data, dtype="<f4").reshape(len(v1.CASES), v1.VOCAB_SIZE)
    return arrays


def _prepare_v1_reconstruction(repo, cache, manifest_value, deps):
    """Reconstruct the frozen v1 candidate in an already-gated private owner."""
    torch, np, pretrained, v1, v2, edits, evidence, _manifest, _custody, _exposure, _gate = deps
    if (sys.byteorder != "little" or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
            or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
            or torch.cuda.is_initialized() or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled("cpu")):
        raise OwnerError("a fresh CPU-only offline runtime is required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise OwnerError("denormal preservation is unavailable")
    v1._denormal_probe()
    environment = v1._environment()
    previous = manifest_value["v1_compatibility"]
    if _canonical({key: environment[key] for key in edits.ENVIRONMENT_FIELDS}) != _canonical(previous["environment"]):
        raise Incompatible("execution-relevant runtime differs from v1")
    runtime_before_load = _runtime_settings(torch)
    evidence._check_runtime_settings(runtime_before_load, environment)
    loaded = pretrained.load_distilgpt2(cache_dir=cache, allow_download=False)
    if _runtime_settings(torch) != runtime_before_load:
        raise OwnerError("CPU runtime changed during checkpoint loading")
    model, module, weight = _check_checkpoint(loaded, manifest_value, torch, v1)
    _verify_tokenizer(loaded.tokenizer, v1, v2)
    guard = _ModelGuard(model, module, weight, torch, v1)
    if guard.runtime_settings != runtime_before_load:
        guard.poisoned = True
        raise OwnerError("CPU runtime changed before discovery")
    if guard.state_sha != previous["model_state_sha256"] or guard.weight_sha != previous["weight_sha256"]:
        raise Incompatible("checkpoint state or projection digest differs from v1")
    guard.verify()
    v1._denormal_probe()
    gradient, random, discovery = v1._candidate(model, module, v1.CASES[0][3], v1.LONDON, v1.PARIS,
                                                 expected_shape=v1.WEIGHT_SHAPE)
    discovery["denormal_probe_after_discovery"] = v1._denormal_probe()
    guard.verify()
    reference = evidence._read(repo / _manifest.V1_PRECOMMIT, evidence.MAX_JSON)
    try:
        compatibility = edits.verify_v1_compatibility(reference, environment=environment,
            model_metadata={**loaded.metadata, "evaluation_mode": True}, model_state_sha256=guard.state_sha,
            original=guard.original, v1_gradient=gradient, v1_random=random, discovery=discovery)
    except v1.StudyError as exc:
        raise Incompatible("v1 candidate reconstruction differs") from exc
    return loaded, guard, gradient, random, discovery, reference, compatibility, environment


def _development_work(repo, cache, output, identity, manifest_value, public, receipt, deps):
    """Run the already-gated private model; never called by the disabled entry point."""
    torch, np, _pretrained, v1, v2, edits, evidence, _manifest, _custody, _exposure, _gate = deps
    loaded, guard, gradient, _random, discovery, _reference, compatibility, environment = \
        _prepare_v1_reconstruction(repo, cache, manifest_value, deps)
    updates, grid = {}, {}
    for name, dose in v2.GRID:
        updates[name] = edits.construct_dose(guard.original, gradient, dose)
        grid[name] = edits.attest(guard.original, updates[name], dose)
    _development_conditions(output, identity, guard, updates, grid, discovery, v1, v2, evidence)
    arrays = _read_development_arrays(output, v1, v2, evidence, np)
    analysis_probe_before = v1._denormal_probe()
    selection = v2.select_development(arrays)
    _, controls = edits.construct_controls(guard.original, gradient, selection["selected_relative_norm"])
    analysis_probe_after = v1._denormal_probe()
    _json_new(output / "selection.json", {**identity, "selection": selection})
    guard.verify()
    if torch.cuda.is_initialized():
        raise OwnerError("CUDA was initialized during the CPU study")
    return {"model": {**loaded.metadata, "evaluation_mode": True}, "environment": environment,
            "exposure": public, "custody": receipt, "compatibility": compatibility,
            "buffer_state_sha256": guard.buffer_sha, "runtime_settings": guard.runtime_settings,
            "grid": grid, "controls": controls,
            "analysis_denormal_probe_before": analysis_probe_before,
            "analysis_denormal_probe_after": analysis_probe_after}


def _reconcile_development(destination, identity, terminal_sha, manifest_value, public, receipt, deps):
    """Recheck an exact already-published terminal after an ambiguous return."""
    torch, _np, _pretrained, v1, _v2, _edits, evidence, _manifest, _custody, _exposure, gate = deps
    if not torch.set_flush_denormal(False):
        raise OwnerError("the independent reader cannot preserve denormals")
    v1._denormal_probe()
    audited = evidence.audit_development(destination, manifest_sha256=identity["manifest_sha256"],
                                         terminal_sha256=terminal_sha)
    v1._denormal_probe()
    if (audited["run_id"] != identity["run_id"] or audited["source_commit"] != identity["source_commit"]):
        raise OwnerError("published development identity differs")
    started_at = evidence._utc(audited["started_utc"])
    observed = [receipt["public_inventory"]["observed_utc"],
                *(row["observed_utc"] for row in receipt["observations"])]
    if any(evidence._utc(value) > started_at for value in observed):
        raise OwnerError("custody observations postdate the recorded start")
    gate._check_metadata(manifest_value, audited["selection"], audited["metadata"], public, receipt)
    return audited


def _rename_development(staging, destination, identity, terminal_sha, manifest_value, public, receipt, deps):
    try:
        os.rename(staging, destination)
    except BaseException as primary:
        status = "unknown"
        try:
            if not _path_absent(destination):
                _reconcile_development(destination, identity, terminal_sha, manifest_value, public, receipt, deps)
                status = "complete_verified"
            elif not _path_absent(staging):
                status = "not_published"
        except BaseException as recovery:
            if (isinstance(recovery, (KeyboardInterrupt, SystemExit))
                    and not isinstance(primary, (KeyboardInterrupt, SystemExit))):
                try:
                    recovery.published_terminal = "unknown"
                except Exception:
                    pass
                raise recovery from primary
        try:
            primary.published_terminal = status
        except Exception:
            pass
        raise


def _publish_development(staging, destination, identity, candidate, supervisor_result,
                         manifest_value, public, receipt, deps):
    """Validate and publish only after the supervisor has reaped a clean child.

    Arguments are trusted same-process values from the still-disabled future
    controller. A dataclass instance is not proof of process exit or execution.
    A failed or interrupted publication leaves the staging attempt separate.
    """
    from looking_glass.supervisor import SupervisionResult
    torch, _np, _pretrained, v1, v2, _edits, evidence, _manifest, _custody, _exposure, gate = deps
    staging, destination = Path(staging), Path(destination)
    if (staging.parent.resolve(strict=True) != destination.parent.resolve(strict=True)
            or staging == destination or not _path_absent(destination)):
        raise OwnerError("publication requires a new destination beside its staging directory")
    evidence._regular(staging, directory=True)
    if (type(supervisor_result) is not SupervisionResult or type(supervisor_result.pid) is not int
            or supervisor_result.pid <= 0 or not evidence._run_id(supervisor_result.run_id)
            or type(supervisor_result.exit_code) is not int or supervisor_result.exit_code != 0
            or supervisor_result.clean is not True or supervisor_result.timed_out is not False):
        raise OwnerError("the direct child was not cleanly reaped")
    names = {*evidence.DOCUMENTS, *(f"{name}.f32" for name in evidence.CONDITIONS)}
    if {entry.name for entry in staging.iterdir()} != names - {"metadata.json"}:
        raise OwnerError("the staged development inventory is incomplete")
    recorded_owner = v1._decode_document(evidence._read(staging / "owner.json", evidence.MAX_JSON))
    expected_owner = {**identity, "supervisor_run_id": supervisor_result.run_id, "pid": supervisor_result.pid}
    if _canonical(recorded_owner) != _canonical(expected_owner):
        raise OwnerError("child and reaped supervisor identities differ")
    started = v1._decode_document(evidence._read(staging / "started.json", evidence.MAX_JSON))
    if not evidence._identity(started, identity["run_id"], identity["manifest_sha256"], identity["source_commit"]):
        raise OwnerError("started identity differs")
    started_at = evidence._utc(started["started_utc"])
    observed = [receipt["public_inventory"]["observed_utc"],
                *(row["observed_utc"] for row in receipt["observations"])]
    if any(evidence._utc(value) > started_at for value in observed):
        raise OwnerError("custody observations postdate the recorded start")
    fields = {"model", "environment", "exposure", "custody", "compatibility", "buffer_state_sha256", "runtime_settings", "grid", "controls",
              "analysis_denormal_probe_before", "analysis_denormal_probe_after"}
    if type(candidate) is not dict or set(candidate) != fields:
        raise OwnerError("child metadata is incomplete")
    owner_id = {"run_id": supervisor_result.run_id, "pid": supervisor_result.pid}
    supervision = {"owner": owner_id, "supervisor": {**owner_id, "exit_code": 0,
                   "clean": True, "timed_out": False, "reaped": True}}
    metadata = {**identity, **candidate, "supervision": supervision}
    selection = v1._decode_document(evidence._read(staging / "selection.json", evidence.MAX_JSON))
    gate._check_metadata(manifest_value, selection["selection"], metadata, public, receipt)
    _json_new(staging / "metadata.json", metadata)
    matrix_bytes = len(v1.CASES) * v1.VOCAB_SIZE * 4
    inventory = {}
    for name in sorted(names):
        limit = (matrix_bytes if name.endswith(".f32") else
                 (evidence.MAX_JOURNAL if name.endswith(".jsonl") else evidence.MAX_JSON))
        data = evidence._read(staging / name, limit)
        if name.endswith(".f32") and len(data) != matrix_bytes:
            raise OwnerError("development matrix is incomplete")
        inventory[name] = v1._sha(data)
    terminal = {**identity, "status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                "cleanup": {name: True for name in evidence.CLEANUP_KEYS},
                "artifacts": inventory}
    _json_new(staging / "terminal.json", terminal)
    terminal_sha = v1._file_sha(staging / "terminal.json")
    if not torch.set_flush_denormal(False):
        raise OwnerError("the independent reader cannot preserve denormals")
    v1._denormal_probe()
    result = evidence.audit_development(staging, manifest_sha256=identity["manifest_sha256"],
                                        terminal_sha256=terminal_sha)
    v1._denormal_probe()
    if _canonical(result["selection"]) != _canonical(selection["selection"]):
        raise OwnerError("reader and staged selection differ")
    # The checkout, staging parent, destination and their ancestors are a
    # trusted, exclusive filesystem namespace throughout this operation.
    _rename_development(staging, destination, identity, terminal_sha, manifest_value, public, receipt, deps)
    return {"terminal_sha256": terminal_sha, "selection": result["selection"]}


def run_development(*_args, **_kwargs):
    """The live entry remains closed until the reviewed witness gate lands."""
    raise OwnerError("live development is disabled pending the independent witness and supervisor gates")


def run_evaluation(*_args, **_kwargs):
    """Evaluation also requires a separately published development selection."""
    raise OwnerError("live evaluation is disabled pending the selection, witness and supervisor gates")

