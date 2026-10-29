"""CPU-only owner and evidence checks; no pretrained checkpoint is loaded."""

from pathlib import Path
import dis
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_edits as edits
from looking_glass import targeted_edit_v2_evidence as evidence
from looking_glass import targeted_edit_v2_owner as owner
from looking_glass.supervisor import SupervisionResult

if __package__:
    from .opcode_support import run_opcode_trace
    from .test_targeted_edit_v2_end_to_end import fake_development
else:
    from opcode_support import run_opcode_trace
    from test_targeted_edit_v2_end_to_end import fake_development


def guard_for(model):
    model.eval()
    return owner._ModelGuard(model, model, model.weight, torch, v1)


@pytest.mark.parametrize("mask", ["", "0", None])
def test_development_rejects_mask_drift_before_model_work(monkeypatch, mask):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(owner.OwnerError, match="fresh CPU-only"):
        owner._development_work(None, None, None, None, None, None, None, (torch,) + (None,) * 10)


@pytest.mark.parametrize("point", ["load_mask", "load_setting", "tokenizer_setting", "unchanged"])
def test_development_binds_full_runtime_before_discovery(monkeypatch, point):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    settings = {"precision": "highest"}
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
        is_inference_mode_enabled=lambda: False, is_autocast_enabled=lambda _device: False,
        set_num_threads=lambda _n: None, set_num_interop_threads=lambda _n: None,
        use_deterministic_algorithms=lambda _value: None, set_flush_denormal=lambda _value: True)
    def runtime(_torch):
        return {"cuda_visibility": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"), **settings}
    monkeypatch.setattr(owner, "_runtime_settings", runtime)
    def load(**_kwargs):
        if point == "load_mask":
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        elif point == "load_setting":
            settings["precision"] = "medium"
        return SimpleNamespace(tokenizer=object())
    def tokenizer(*_args):
        if point == "tokenizer_setting":
            settings["precision"] = "medium"
    monkeypatch.setattr(owner, "_check_checkpoint", lambda *_args: (object(), object(), object()))
    monkeypatch.setattr(owner, "_verify_tokenizer", tokenizer)
    monkeypatch.setattr(owner, "_ModelGuard", lambda *_args: SimpleNamespace(
        runtime_settings=runtime(fake_torch), state_sha="state", weight_sha="weight",
        poisoned=False, verify=lambda: None))
    calls = []
    def candidate(*_args, **_kwargs):
        calls.append(True)
        raise LookupError("candidate reached")
    fake_v1 = SimpleNamespace(_denormal_probe=lambda: None, _environment=lambda: {},
        _candidate=candidate, CASES=((None, None, None, (1,)),), LONDON=1, PARIS=2, WEIGHT_SHAPE=(1,))
    def check_runtime(snapshot, _environment):
        assert snapshot == {"cuda_visibility": "-1", "precision": "highest"}
    fake_evidence = SimpleNamespace(_check_runtime_settings=check_runtime)
    deps = (fake_torch, None, SimpleNamespace(load_distilgpt2=load), fake_v1, None,
            SimpleNamespace(ENVIRONMENT_FIELDS=()), fake_evidence, None, None, None, None)
    manifest = {"v1_compatibility": {"environment": {}, "model_state_sha256": "state", "weight_sha256": "weight"}}
    if point == "unchanged":
        with pytest.raises(LookupError, match="candidate reached"):
            owner._development_work(None, None, None, None, manifest, None, None, deps)
        assert calls == [True]
    else:
        with pytest.raises(owner.OwnerError, match="runtime changed"):
            owner._development_work(None, None, None, None, manifest, None, None, deps)
        assert calls == []


def test_condition_applies_exact_weight_then_restores_all_state():
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 3)
    guard = guard_for(model)
    delta = torch.full_like(guard.original, 0.125)
    expected = guard.original + delta
    calls = []
    guard.condition(delta, v1._tensor_sha(expected), lambda: calls.append(model.weight.detach().clone()))
    assert len(calls) == 1 and torch.equal(calls[0], expected)
    assert torch.equal(model.weight, guard.original)
    assert guard.poisoned
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must await journal commit"))
    guard.commit_condition()
    guard.verify()
    assert not guard.poisoned
    assert guard.last_cleanup == "verified"


def test_interrupt_restores_weight_and_forbids_another_forward():
    model = torch.nn.Linear(3, 3)
    guard = guard_for(model)
    delta = torch.full_like(guard.original, 0.125)
    def interrupted():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        guard.condition(delta, v1._tensor_sha(guard.original + delta), interrupted)
    assert torch.equal(model.weight, guard.original)
    assert guard.last_cleanup == "verified"
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))


def test_new_cleanup_interrupt_takes_priority_over_original_error(monkeypatch):
    model = torch.nn.Linear(3, 3)
    guard = guard_for(model)
    def fail_restore():
        raise KeyboardInterrupt()
    def fail_forward():
        raise ValueError("forward failed")
    monkeypatch.setattr(guard, "restore", fail_restore)
    with pytest.raises(KeyboardInterrupt) as caught:
        guard.condition(None, guard.weight_sha, fail_forward)
    assert isinstance(caught.value.__cause__, ValueError)
    assert guard.poisoned
    assert guard.last_cleanup == "unverified"
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))


def test_interruption_before_normal_restore_keeps_model_fenced():
    instructions = list(dis.get_instructions(owner._ModelGuard.condition))
    matches = [index for index, instruction in enumerate(instructions)
               if instruction.argval == "restore" and instruction.opname in ("LOAD_METHOD", "LOAD_ATTR")]
    assert matches and matches[0] > 0
    target = instructions[matches[0] - 1]
    assert target.opname == "LOAD_FAST" and target.argval == "self"
    model = torch.nn.Linear(2, 2)
    guard = guard_for(model)
    delta = torch.ones_like(guard.original)
    observed = []
    def tracer(frame, event, _arg):
        if frame.f_code is owner._ModelGuard.condition.__code__:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode" and frame.f_lasti == target.offset:
                observed.append(True)
                raise KeyboardInterrupt()
        return tracer
    with pytest.raises(KeyboardInterrupt):
        run_opcode_trace(tracer, lambda: guard.condition(delta, v1._tensor_sha(guard.original + delta), lambda: None))
    assert observed == [True]
    assert guard.poisoned
    assert torch.equal(model.weight, guard.original) or guard.last_cleanup == "unverified"
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))


def test_lost_condition_return_keeps_fence_until_journal_ack():
    code = owner._ModelGuard.condition.__code__
    return_offsets = {item.offset for item in dis.get_instructions(code)
                      if item.opname in ("RETURN_VALUE", "RETURN_CONST")}
    seen = []
    def observe(frame, event, _arg):
        if frame.f_code is code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode" and frame.f_lasti in return_offsets:
                seen.append(frame.f_lasti)
        return observe
    first = guard_for(torch.nn.Linear(2, 2))
    run_opcode_trace(observe, lambda: first.condition(None, first.weight_sha, lambda: None))
    assert len(seen) == 1
    model = torch.nn.Linear(2, 2)
    guard = guard_for(model)
    calls = []
    observed = []
    def interrupt(frame, event, _arg):
        if frame.f_code is code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode" and frame.f_lasti == seen[0]:
                observed.append(True)
                raise KeyboardInterrupt()
        return interrupt
    with pytest.raises(KeyboardInterrupt):
        run_opcode_trace(interrupt, lambda: guard.condition(None, guard.weight_sha, lambda: calls.append(True)))
    assert observed == [True] and calls == [True]
    assert guard.poisoned and torch.equal(model.weight, guard.original)
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not replay"))


def _one_row_writer(monkeypatch):
    monkeypatch.setattr(v1, "VOCAB_SIZE", 4)
    monkeypatch.setattr(v1, "CASES", (("one", "discovery", "synthetic", (1,), None),))
    calls = []
    row = np.zeros(4, dtype="<f4").tobytes()
    def forward(_model, _ids):
        calls.append(True)
        return row
    monkeypatch.setattr(v1, "_logit_row", forward)
    guard = guard_for(torch.nn.Linear(2, 2))
    identity = {"run_id": "a" * 32}
    discovery = {"discovery_logit_sha256": v1._sha(row)}
    return guard, identity, discovery, calls


@pytest.mark.parametrize("after_write", [False, True])
def test_state_journal_write_failure_leaves_completed_condition_fenced(monkeypatch, tmp_path, after_write):
    guard, identity, discovery, calls = _one_row_writer(monkeypatch)
    append = owner._append
    def interrupted(handle, data):
        if Path(handle.name).name == "condition_states.jsonl":
            if after_write:
                append(handle, data)
            raise OSError("synthetic state journal failure")
        append(handle, data)
    monkeypatch.setattr(owner, "_append", interrupted)
    with pytest.raises(OSError, match="state journal"):
        owner._development_conditions(tmp_path, identity, guard, {}, {}, discovery, v1, v2, evidence)
    assert calls == [True] and guard.poisoned and guard.last_cleanup == "verified"
    assert bool((tmp_path / "condition_states.jsonl").read_bytes()) is after_write


def test_lost_journal_ack_stops_before_next_condition(monkeypatch, tmp_path):
    code = owner._ModelGuard.commit_condition.__code__
    offsets = {item.offset for item in dis.get_instructions(code)
               if item.opname in ("RETURN_VALUE", "RETURN_CONST")}
    seen = []
    def observe(frame, event, _arg):
        if frame.f_code is code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode" and frame.f_lasti in offsets:
                seen.append(frame.f_lasti)
        return observe
    first = guard_for(torch.nn.Linear(2, 2))
    first.condition(None, first.weight_sha, lambda: None)
    run_opcode_trace(observe, first.commit_condition)
    assert len(seen) == 1
    guard, identity, discovery, calls = _one_row_writer(monkeypatch)
    observed = []
    def interrupt(frame, event, _arg):
        if frame.f_code is code:
            if event == "call":
                frame.f_trace_opcodes = True
            elif event == "opcode" and frame.f_lasti == seen[0]:
                observed.append(True)
                raise KeyboardInterrupt()
        return interrupt
    with pytest.raises(KeyboardInterrupt):
        run_opcode_trace(interrupt, lambda: owner._development_conditions(
            tmp_path, identity, guard, {}, {}, discovery, v1, v2, evidence))
    assert observed == [True] and calls == [True] and not guard.poisoned
    assert len((tmp_path / "condition_states.jsonl").read_bytes().splitlines()) == 1


@pytest.mark.parametrize("damage", ("other_weight", "flag", "gradient", "training", "hook", "state_hook", "nonpersistent_buffer"))
def test_unverified_cleanup_fails_closed_and_prevents_next_condition(damage):
    model = torch.nn.Linear(3, 3)
    model.register_buffer("ephemeral", torch.tensor([1.0]), persistent=False)
    guard = guard_for(model)
    delta = torch.full_like(guard.original, 0.125)
    handles = []
    def mutate():
        if damage == "other_weight":
            with torch.no_grad():
                model.bias.add_(1)
        elif damage == "flag":
            model.bias.requires_grad_(False)
        elif damage == "gradient":
            model.bias.grad = torch.ones_like(model.bias)
        elif damage == "training":
            model.train()
        elif damage == "hook":
            handles.append(model.register_forward_hook(lambda *_: None))
        elif damage == "state_hook":
            handles.append(model.register_state_dict_post_hook(lambda *_: None))
        else:
            model.ephemeral.add_(1)
    with pytest.raises(owner.OwnerError, match="cleanup"):
        guard.condition(delta, v1._tensor_sha(guard.original + delta), mutate)
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))
    for handle in handles:
        handle.remove()


def test_precheck_exception_poisons_even_if_hash_later_recovers(monkeypatch):
    model = torch.nn.Linear(3, 3)
    guard = guard_for(model)
    original_hash = v1._model_hash
    def broken(_model):
        raise KeyboardInterrupt()
    monkeypatch.setattr(v1, "_model_hash", broken)
    with pytest.raises(KeyboardInterrupt):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))
    monkeypatch.setattr(v1, "_model_hash", original_hash)
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))


def test_applied_digest_mismatch_stops_before_forward_and_restores():
    model = torch.nn.Linear(3, 3)
    guard = guard_for(model)
    with pytest.raises(owner.OwnerError, match="applied weight"):
        guard.condition(torch.ones_like(guard.original), "0" * 64, lambda: pytest.fail("must not run"))
    assert torch.equal(model.weight, guard.original)
    assert guard.poisoned


def test_private_writer_is_accepted_by_independent_reader(monkeypatch, tmp_path):
    assert torch.set_flush_denormal(False)
    monkeypatch.setattr(v1, "VOCAB_SIZE", 16)
    monkeypatch.setattr(v1, "LONDON", 1)
    monkeypatch.setattr(v1, "PARIS", 2)
    cases = tuple((*case[:4], (" label", 3) if case[1] == "locality" else None) for case in v1.CASES)
    monkeypatch.setattr(v1, "CASES", cases)
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    guard = guard_for(model)
    baseline = np.zeros(16, dtype="<f4")
    baseline[1] = -1.0
    baseline[3] = 2.0
    def row(current, _ids):
        values = baseline.copy()
        values[1] += float(current.weight[0, 0].detach() - guard.original[0, 0])
        return values.tobytes()
    monkeypatch.setattr(v1, "_logit_row", row)
    updates = {name: torch.full_like(guard.original, index * 0.1)
               for index, (name, _) in enumerate(v2.GRID, 1)}
    grid = {name: {"realized_weight_sha256": v1._tensor_sha(guard.original + value)}
            for name, value in updates.items()}
    common = {"version": v2.VERSION, "phase": evidence.PHASE, "run_id": "a" * 32,
              "manifest_sha256": "1" * 64, "source_commit": "2" * 40}
    owner._json_new(tmp_path / "started.json", {**common, "started_utc": "2026-10-29T01:00:00+00:00"})
    owner._json_new(tmp_path / "owner.json", {**common, "supervisor_run_id": "b" * 32, "pid": 123})
    discovery = {"discovery_logit_sha256": v1._sha(baseline.tobytes())}
    owner._development_conditions(tmp_path, common, guard, updates, grid, discovery, v1, v2, evidence)
    arrays = owner._read_development_arrays(tmp_path, v1, v2, evidence, np)
    selection = v2.select_development(arrays)
    assert selection["development_qualified"] is False
    assert selection["selected_condition"] == v2.GRID[0][0]
    owner._json_new(tmp_path / "selection.json", {**common, "selection": selection})
    runtime = {"float32_matmul_precision": "highest", "mkldnn_deterministic": False,
        "deterministic_warn_only": False, "intraop_threads": 1, "interop_threads": 1,
        "deterministic_algorithms": True, "mkldnn_enabled": True, "cpu_autocast_enabled": False,
        "inference_mode_enabled": False, "cuda_initialized": False, "cuda_visibility": "-1",
        "hf_hub_offline": "1", "transformers_offline": "1", "omp_num_threads": None, "mkl_num_threads": None}
    environment = {key: runtime[key] for key in ("intraop_threads", "interop_threads", "deterministic_algorithms",
        "mkldnn_enabled", "cuda_initialized", "omp_num_threads", "mkl_num_threads")}
    supervision = {"owner": {"run_id": "b" * 32, "pid": 123}, "supervisor": {"run_id": "b" * 32,
        "pid": 123, "exit_code": 0, "clean": True, "timed_out": False, "reaped": True}}
    owner._json_new(tmp_path / "metadata.json", {**common, "compatibility": {
        "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
        "discovery_logit_sha256": discovery["discovery_logit_sha256"]}, "grid": grid,
        "supervision": supervision, "environment": environment, "runtime_settings": runtime,
        "buffer_state_sha256": guard.buffer_sha,
        "analysis_denormal_probe_before": evidence.DENORMAL_PROBE,
        "analysis_denormal_probe_after": evidence.DENORMAL_PROBE})
    artifacts = {*evidence.DOCUMENTS, *(f"{name}.f32" for name in evidence.CONDITIONS)}
    terminal = {**common, "status": "complete", "finished_utc": "2026-10-29T01:01:00+00:00",
                "cleanup": {name: True for name in evidence.CLEANUP_KEYS},
                "artifacts": {name: v1._file_sha(tmp_path / name) for name in artifacts}}
    owner._json_new(tmp_path / "terminal.json", terminal)
    result = evidence.audit_development(tmp_path, manifest_sha256=common["manifest_sha256"],
                                        terminal_sha256=v1._file_sha(tmp_path / "terminal.json"))
    assert result["selection"] == selection
    assert not result["selection"]["development_qualified"]


def test_live_checkpoint_execution_is_disabled():
    assert owner.LIVE_DEVELOPMENT_ENABLED is False
    with pytest.raises(owner.OwnerError, match="disabled"):
        owner.run_development()
    with pytest.raises(owner.OwnerError, match="disabled"):
        owner.run_evaluation()


def test_checkpoint_requires_exact_named_parameter_identity(monkeypatch):
    from transformers.models.gpt2 import modeling_gpt2, tokenization_gpt2_fast
    class GPT2LMHeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList(torch.nn.Module() for _ in range(6))
            self.transformer.h[5].mlp = torch.nn.Module()
            self.transformer.h[5].mlp.c_proj = torch.nn.Linear(2, 3, bias=False)
            self.config = SimpleNamespace(use_cache=False, output_hidden_states=False,
                                          output_attentions=False, _attn_implementation="eager")
            self.eval()

    class GPT2TokenizerFast:
        pass

    monkeypatch.setattr(modeling_gpt2, "GPT2LMHeadModel", GPT2LMHeadModel)
    monkeypatch.setattr(tokenization_gpt2_fast, "GPT2TokenizerFast", GPT2TokenizerFast)
    model = GPT2LMHeadModel()
    weight = model.transformer.h[5].mlp.c_proj.weight
    monkeypatch.setattr(v1, "WEIGHT_SHAPE", tuple(weight.shape))
    monkeypatch.setattr(v1, "PARAMETER_COUNT", weight.numel())
    loaded = SimpleNamespace(model=model, tokenizer=GPT2TokenizerFast(), metadata={})
    manifest = {"v1_compatibility": {"model": {"evaluation_mode": True}}}
    assert not model.training
    assert not any(item.training for item in model.modules())
    assert sum(p.numel() for p in model.parameters()) == v1.PARAMETER_COUNT
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as bound_model
    from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast as bound_tokenizer
    assert type(model) is bound_model and type(loaded.tokenizer) is bound_tokenizer
    assert owner._check_checkpoint(loaded, manifest, torch, v1)[2] is weight
    original = model.named_parameters
    def wrong_identity(*args, **kwargs):
        for name, parameter in original(*args, **kwargs):
            yield name, torch.nn.Parameter(parameter.detach().clone())
    monkeypatch.setattr(model, "named_parameters", wrong_identity)
    with pytest.raises(owner.Incompatible, match="exact named"):
        owner._check_checkpoint(loaded, manifest, torch, v1)


def test_runtime_setting_drift_poisons_condition(monkeypatch):
    model = torch.nn.Linear(2, 2)
    guard = guard_for(model)
    monkeypatch.setenv("HF_HUB_OFFLINE", "changed")
    with pytest.raises(owner.OwnerError, match="cleanup"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))
    with pytest.raises(owner.OwnerError, match="poisoned"):
        guard.condition(None, guard.weight_sha, lambda: pytest.fail("must not run"))


def test_preexisting_state_dict_hook_is_rejected_before_hash():
    model = torch.nn.Linear(2, 2)
    handle = model.register_state_dict_post_hook(lambda *_: None)
    try:
        with pytest.raises(owner.OwnerError, match="preexisting"):
            guard_for(model)
    finally:
        handle.remove()


@pytest.mark.parametrize("kind", ("parameter", "buffer"))
def test_edited_site_rejects_storage_overlap_with_another_tensor(kind):
    model = torch.nn.Linear(3, 3)
    if kind == "parameter":
        model.register_parameter("aliased", torch.nn.Parameter(model.weight.detach()[0]))
    else:
        model.register_buffer("aliased", model.weight.detach()[0])
    with pytest.raises(owner.Incompatible, match="shares edited-site storage"):
        owner._verify_unshared_site(model, model.weight, "weight", torch)


def test_import_origin_check_rejects_unbound_or_replaced_project_source(monkeypatch, tmp_path):
    repo = Path(owner.__file__).resolve().parent.parent
    hashes = {}
    for name, module in tuple(sys.modules.items()):
        if name == "looking_glass" or name.startswith("looking_glass."):
            path = Path(module.__file__).resolve()
            hashes[path.relative_to(repo).as_posix()] = owner._sha(path.read_bytes())
    manifest = {"source_sha256": hashes}
    owner._verify_import_origins(repo, manifest)
    hashes["looking_glass/targeted_edit_cpu_v1.py"] = "0" * 64
    with pytest.raises(owner.OwnerError, match="frozen bytes"):
        owner._verify_import_origins(repo, manifest)
    source = repo / "looking_glass/targeted_edit_cpu_v1.py"
    hashes["looking_glass/targeted_edit_cpu_v1.py"] = owner._sha(source.read_bytes())
    foreign = tmp_path / "foreign.py"
    foreign.write_text("pass\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "looking_glass.foreign", SimpleNamespace(__file__=str(foreign)))
    with pytest.raises(owner.OwnerError, match="another checkout"):
        owner._verify_import_origins(repo, manifest)


@pytest.mark.parametrize("initial,recovery,expected", [
    (ValueError("rename failed"), KeyboardInterrupt(), KeyboardInterrupt),
    (KeyboardInterrupt(), ValueError("audit failed"), KeyboardInterrupt),
])
def test_rename_and_reconciliation_preserve_cancellation(monkeypatch, tmp_path, initial, recovery, expected):
    staging, destination = tmp_path / "staging", tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    def failed_rename(*_args):
        raise initial
    def failed_recovery(*_args):
        raise recovery
    monkeypatch.setattr(owner.os, "rename", failed_rename)
    monkeypatch.setattr(owner, "_reconcile_development", failed_recovery)
    with pytest.raises(expected) as caught:
        owner._rename_development(staging, destination, {}, "a" * 64, {}, {}, {}, ())
    assert caught.value.published_terminal == "unknown"
    if isinstance(initial, ValueError):
        assert caught.value.__cause__ is initial


@pytest.fixture
def staged_development(tmp_path):
    """Fully synthetic artifacts; this fixture never forwards a checkpoint."""
    assert torch.set_flush_denormal(False)
    repo = Path(owner.__file__).resolve().parent.parent
    reference = v1._decode_document((repo / "results/targeted-edit-cpu-v1-20261028/reference/precommit.json").read_bytes())
    environment = reference["environment"]
    prior = {"precommit_sha256": edits.V1_PRECOMMIT_SHA256, "model": reference["model"],
             "environment": {key: environment[key] for key in edits.ENVIRONMENT_FIELDS},
             "model_state_sha256": reference["model_state_sha256"], "weight_sha256": reference["weight_sha256"],
             "discovery": reference["discovery"], "candidates": reference["candidates"]}
    manifest_value = {"v1_compatibility": prior}
    public = {"synthetic_public_check": True}
    receipt = {"public_inventory": {"observed_utc": "2000-01-01T00:00:00+00:00"},
               "observations": [{"observed_utc": "2000-01-01T00:00:00+00:00"}]}
    staging, destination = tmp_path / "partial", tmp_path / "complete"
    identity = {"version": v2.VERSION, "phase": "development", "run_id": "a" * 32,
                "manifest_sha256": "1" * 64, "source_commit": "2" * 40}
    fake_development(staging, source=identity["source_commit"], manifest_sha=identity["manifest_sha256"],
                     model=prior["model"], environment=environment, public=public, custody_result=receipt, prior=prior)
    (staging / "started.json").write_bytes(v1._canonical({**identity, "started_utc": "2000-01-01T00:00:01+00:00"}))
    metadata = v1._decode_document((staging / "metadata.json").read_bytes())
    candidate = {key: value for key, value in metadata.items() if key not in {*identity, "supervision"}}
    (staging / "metadata.json").unlink()
    (staging / "terminal.json").unlink()
    result = SupervisionResult("b" * 32, 123, 0, True, False, "synthetic clean child")
    return staging, destination, identity, candidate, result, manifest_value, public, receipt, owner._modules()


def test_publication_requires_and_records_clean_reaped_child(staged_development):
    staging, destination, identity, candidate, result, manifest_value, public, receipt, deps = staged_development
    incomplete = SupervisionResult(result.run_id, result.pid, 1, False, False, "synthetic failure")
    with pytest.raises(owner.OwnerError, match="cleanly reaped"):
        owner._publish_development(staging, destination, identity, candidate, incomplete, manifest_value, public, receipt, deps)
    assert staging.exists() and not destination.exists() and not (staging / "terminal.json").exists()
    published = owner._publish_development(staging, destination, identity, candidate, result, manifest_value, public, receipt, deps)
    assert destination.is_dir() and not staging.exists()
    record = evidence.audit_development(destination, manifest_sha256=identity["manifest_sha256"],
                                        terminal_sha256=published["terminal_sha256"])
    assert record["metadata"]["supervision"]["supervisor"]["reaped"] is True
    assert record["selection"] == published["selection"]


def test_publication_rename_success_then_interrupt_is_reconciled(staged_development, monkeypatch):
    staging, destination, identity, candidate, result, manifest_value, public, receipt, deps = staged_development
    actual_rename = owner.os.rename
    def rename_then_interrupt(source, target):
        actual_rename(source, target)
        raise KeyboardInterrupt()
    monkeypatch.setattr(owner.os, "rename", rename_then_interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        owner._publish_development(staging, destination, identity, candidate, result, manifest_value, public, receipt, deps)
    assert caught.value.published_terminal == "complete_verified"
    assert destination.is_dir() and not staging.exists()
    digest = v1._file_sha(destination / "terminal.json")
    assert owner._reconcile_development(destination, identity, digest, manifest_value, public, receipt, deps)["status"] == "matching_saved_selection"
