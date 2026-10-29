import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass import targeted_edit_cpu_v1 as study


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture(autouse=True)
def preserve_denormals():
    assert torch.set_flush_denormal(False)
    yield
    torch.set_flush_denormal(False)


def _toy_model(layers=1, vocab=32):
    torch.manual_seed(123)
    config = GPT2Config(vocab_size=vocab, n_positions=16, n_embd=8,
                        n_layer=layers, n_head=2, n_inner=32,
                        use_cache=False, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    config._attn_implementation = "eager"
    model = GPT2LMHeadModel(config).eval()
    model.config.use_cache = False
    return model


def _synthetic_logits():
    shape = (len(study.CASES), study.VOCAB_SIZE)
    result = {name: np.zeros(shape, dtype="<f4") for name in study.CONDITIONS}
    result["original"][:5, study.LONDON] = -2
    result["gradient"][:5, study.LONDON] = 2
    result["random"][:5, study.LONDON] = -1.5
    return result


def test_gradient_candidate_is_one_site_rank_one_and_restores_model():
    model = _toy_model()
    module = model.transformer.h[0].mlp.c_proj
    before = study._model_hash(model)
    flags = [p.requires_grad for p in model.parameters()]
    original = module.weight.detach().clone()
    targeted, random, details = study._candidate(model, module, (1, 2, 3), 4, 5,
                                                  expected_shape=(32, 8))
    attestations = study._candidate_attestation(original, {"gradient": targeted, "random": random})
    assert details["gradient_norm"] > 0
    assert details["gradient_outer_relative_residual"] <= 1e-5
    assert 0.999 <= attestations["gradient"]["realized_delta_norm"] / attestations["random"]["realized_delta_norm"] <= 1.001
    assert max(item["realized_relative_norm"] for item in attestations.values()) <= 0.01001
    assert targeted.shape == random.shape == original.shape
    assert study._model_hash(model) == before
    assert [p.requires_grad for p in model.parameters()] == flags
    assert not module._forward_hooks and not module._forward_pre_hooks
    assert module.weight.grad is None


def test_random_control_is_deterministic_and_changes_with_seed(monkeypatch):
    model = _toy_model()
    module = model.transformer.h[0].mlp.c_proj
    _, first, _ = study._candidate(model, module, (1, 2), 3, 4)
    _, second, _ = study._candidate(model, module, (1, 2), 3, 4)
    monkeypatch.setattr(study, "SEED", study.SEED + 1)
    _, third, _ = study._candidate(model, module, (1, 2), 3, 4)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_candidate_hook_registration_failure_does_not_leak_hooks(monkeypatch):
    model = _toy_model()
    module = model.transformer.h[0].mlp.c_proj
    monkeypatch.setattr(module, "register_forward_hook", lambda *_: (_ for _ in ()).throw(OSError("fault")))
    with pytest.raises(OSError):
        study._candidate(model, module, (1, 2), 3, 4)
    assert not module._forward_hooks and not module._forward_pre_hooks


def test_candidate_cleanup_attempts_all_hooks_and_flags_after_failure():
    actions = []

    class BrokenHandle:
        def remove(self):
            actions.append("broken")
            raise OSError("fault")

    class OtherHandle:
        def remove(self):
            actions.append("other")

    weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    error = study._cleanup_candidate([BrokenHandle(), OtherHandle()], [(weight, True)], weight)
    assert isinstance(error, OSError)
    assert actions == ["broken", "broken", "other"]
    assert weight.requires_grad is True and weight.grad is None


def test_rank_one_scaling_rejects_zero_or_nonfinite():
    with pytest.raises(study.StudyError):
        study._scale_outer(torch.zeros(2), torch.ones(3), 1)
    with pytest.raises(study.StudyError):
        study._scale_outer(torch.tensor([float("nan")]), torch.ones(3), 1)


def test_decision_checks_target_and_independent_locality():
    logits = _synthetic_logits()
    result = study.analyze(logits)
    assert result["eligible"] and result["decision"] == "success"
    assert all(result["criteria"].values())
    logits["gradient"][5, 1] = 2
    changed = study.analyze(logits)
    assert changed["decision"] == "negative"
    assert not changed["criteria"]["all_locality_top_preserved"]


def test_ineligible_and_nonfinite_logits_are_not_success():
    logits = _synthetic_logits()
    logits["original"][0, study.LONDON] = 1
    assert study.analyze(logits)["decision"] == "ineligible"
    logits["original"][0, 0] = float("nan")
    with pytest.raises(study.StudyError):
        study.analyze(logits)


def test_denormal_probe_rejects_flush_mode():
    expected = study._denormal_probe()
    assert expected["subnormal_times_one_bits"] == 1
    assert expected["normal_times_half_bits"] == 0x00400000
    assert torch.set_flush_denormal(True)
    try:
        with pytest.raises(study.StudyError):
            study._denormal_probe()
    finally:
        torch.set_flush_denormal(False)


def test_atomic_json_never_overwrites_or_promotes_partial(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    study._json_new(path, {"one": 1})
    before = path.read_bytes()
    with pytest.raises(study.StudyError):
        study._json_new(path, {"one": 2})
    assert path.read_bytes() == before
    monkeypatch.setattr(study.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("fault")))
    with pytest.raises(OSError):
        study._json_new(tmp_path / "second.json", {"two": 2})
    assert not (tmp_path / "second.json").exists()
    assert (tmp_path / "second.json.partial").exists()


def test_manifest_binds_frozen_committed_source_and_exact_cases(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "core.autocrlf", "false")
    for path in study.SOURCE_PATHS:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes((path + "\n").encode())
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "source")
    source = _git(tmp_path, "rev-parse", "HEAD")
    manifest = study.make_manifest(tmp_path, source)
    path = tmp_path / "frozen.json"
    study._json_new(path, manifest)
    _git(tmp_path, "add", "frozen.json")
    _git(tmp_path, "commit", "-qm", "freeze")
    digest = study._file_sha(path)
    result, _ = study.verify_manifest(tmp_path, path, digest)
    assert result == manifest
    (tmp_path / study.SOURCE_PATHS[0]).write_bytes(b"changed")
    with pytest.raises(study.StudyError):
        study.verify_manifest(tmp_path, path, digest)


def _write_fake_run(path, logits):
    path.mkdir()
    manifest = path.parent / (path.name + "-frozen.json")
    study._json_new(manifest, study._manifest("a" * 40, {name: "b" * 64 for name in study.SOURCE_PATHS}))
    manifest_sha = study._file_sha(manifest)
    run_id = "1" * 32
    study._json_new(path / "started.json", {"version": study.VERSION, "run_id": run_id,
        "manifest_sha256": manifest_sha, "started_utc": "2026-10-29T01:00:00+00:00"})
    exposure = {"canonical_payload_sha256": study.EXPOSURE_SNAPSHOT["canonical_payload_sha256"],
                "commit": study.EXPOSURE_SNAPSHOT["commit"], "planned_rows_checked": len(study.CASES),
                "known_other_prefix_collision_count": 0, "complete_history_certified": False}
    candidate = {"intended_delta_sha256": "c" * 64, "realized_weight_sha256": "d" * 64,
                 "realized_delta_norm": 0.1, "realized_relative_norm": 0.01}
    probe = study._denormal_probe()
    environment = study._environment()
    environment.update({"intraop_threads": 1, "interop_threads": 1,
                        "deterministic_algorithms": True, "cuda_initialized": False})
    model_metadata = _model_metadata(environment["torch_version"], environment["transformers_version"])
    model_metadata["evaluation_mode"] = True
    study._json_new(path / "precommit.json", {
        "version": study.VERSION, "run_id": run_id, "manifest_sha256": manifest_sha,
        "source_commit": "a" * 40, "checkout_commit": "e" * 40,
        "site": study.SITE, "site_shape": list(study.WEIGHT_SHAPE), "seed": study.SEED,
        "model_state_sha256": "f" * 64, "weight_sha256": "a" * 64,
        "model": model_metadata, "environment": environment,
        "exposure_check": exposure, "candidates": {"gradient": candidate, "random": candidate},
        "discovery": {"discovery_logit_sha256": study._sha(logits["original"][0].tobytes()),
                      "discovery_margin_float64": float(logits["original"][0, study.LONDON]) -
                                                   float(logits["original"][0, study.PARIS]),
                      "discovery_objective_float32": float(np.float32(
                          float(logits["original"][0, study.LONDON]) - float(logits["original"][0, study.PARIS]))),
                      "weight_norm": 10.0, "planned_delta_norm": 0.1, "gradient_norm": 1.0,
                      "input_factor_norm": 1.0, "output_factor_norm": 1.0,
                      "gradient_outer_relative_residual": 0.0, "denormal_probe_after_discovery": probe},
    })
    with (path / "observations.jsonl").open("wb") as journal:
        for condition in study.CONDITIONS:
            with (path / f"{condition}.f32").open("wb") as raw:
                for index, (name, split, _text, _ids, _label) in enumerate(study.CASES):
                    data = logits[condition][index].tobytes()
                    raw.write(data)
                    journal.write(study._canonical({"condition": condition, "case_id": name,
                        "split": split, "row": index, "offset": index * study.VOCAB_SIZE * 4,
                        "size": len(data), "sha256": study._sha(data),
                        "denormal_probe": study._denormal_probe()}))
    report = study.analyze(logits)
    report["denormal_probe_after_analysis"] = study._denormal_probe()
    report["run_id"] = run_id
    report["manifest_sha256"] = manifest_sha
    study._json_new(path / "report.json", report)
    names = ("started.json", "precommit.json", "observations.jsonl", "original.f32",
             "gradient.f32", "random.f32", "report.json")
    study._json_new(path / "terminal.json", {"version": study.VERSION, "run_id": run_id, "status": "complete",
                    "decision": report["decision"], "finished_utc": "2026-10-29T01:01:00+00:00",
                    "cleanup": {"attempted": True, "weight_restored": True, "state_restored": True,
                                "flags_restored": True, "grads_cleared": True, "hooks_restored": True},
                    "artifacts": {n: study._file_sha(path / n) for n in names}})
    return manifest, manifest_sha


def test_saved_logits_recompute_and_tampering_is_rejected(tmp_path):
    output = tmp_path / "run"
    manifest, digest = _write_fake_run(output, _synthetic_logits())
    assert study.audit(output, manifest, digest) == {"status": "matching_saved_outputs", "decision": "success",
                                   "execution_attested": False}
    with (output / "gradient.f32").open("r+b") as stream:
        stream.seek(100)
        stream.write(b"z")
    with pytest.raises(study.StudyError):
        study.audit(output, manifest, digest)


def test_journal_extra_record_is_never_a_complete_run(tmp_path):
    output = tmp_path / "run"
    _write_fake_run(output, _synthetic_logits())
    with (output / "observations.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(study.StudyError):
        study._read_outputs(output)


def _rewrite_document(path, value):
    path.write_bytes(study._canonical(value))


@pytest.mark.parametrize("mutation", ["discovery", "run_id", "cleanup", "exposure",
                                      "device", "dtype", "attention", "tokenizer", "runtime", "build", "version"])
def test_audit_rejects_contradictory_attestations_even_after_rehash(tmp_path, mutation):
    output = tmp_path / "run"
    manifest, digest = _write_fake_run(output, _synthetic_logits())
    terminal = json.loads((output / "terminal.json").read_bytes())
    if mutation == "cleanup":
        terminal["cleanup"]["state_restored"] = False
    else:
        precommit = json.loads((output / "precommit.json").read_bytes())
        if mutation == "discovery":
            precommit["discovery"]["discovery_logit_sha256"] = "a" * 64
        elif mutation == "run_id":
            precommit["run_id"] = "2" * 32
        elif mutation == "exposure":
            precommit["exposure_check"]["planned_rows_checked"] = 12
        elif mutation == "device":
            precommit["model"]["device"] = "cuda"
        elif mutation == "dtype":
            precommit["model"]["dtype"] = "float16"
        elif mutation == "attention":
            precommit["model"]["attention_implementation"] = "sdpa"
        elif mutation == "tokenizer":
            precommit["model"]["tokenizer_revision"] = "0" * 40
        elif mutation == "runtime":
            del precommit["environment"]["python_version"]
        elif mutation == "build":
            precommit["environment"]["torch_build_sha256"] = "invalid"
        else:
            precommit["model"]["torch_version"] = "mismatch"
        _rewrite_document(output / "precommit.json", precommit)
        terminal["artifacts"]["precommit.json"] = study._file_sha(output / "precommit.json")
    _rewrite_document(output / "terminal.json", terminal)
    with pytest.raises(study.StudyError):
        study.audit(output, manifest, digest)


def test_raw_reader_rejects_oversize_before_hashing(tmp_path):
    output = tmp_path / "run"
    _write_fake_run(output, _synthetic_logits())
    with (output / "original.f32").open("ab") as stream:
        stream.write(b"extra")
    with pytest.raises(study.StudyError):
        study._read_outputs(output)


def test_public_cases_are_reserved_in_pinned_exposure_snapshot():
    repo = Path(study.__file__).resolve().parent.parent
    check = study._verify_exposure(repo)
    assert check["planned_rows_checked"] == 13
    assert check["known_other_prefix_collision_count"] == 0
    assert check["complete_history_certified"] is False


def test_interrupted_rollback_preserves_cancellation_and_retries(monkeypatch):
    attempts = []

    def restore(*_args):
        attempts.append(1)
        if len(attempts) == 1:
            raise KeyboardInterrupt()
        return True, True, True, True, True

    monkeypatch.setattr(study, "_restore_once", restore)
    cleanup, caught = study._restore_snapshot(None, None, None, None, None, None, OSError("earlier"))
    assert len(attempts) == 2 and isinstance(caught, KeyboardInterrupt)
    assert cleanup == {"attempted": True, "weight_restored": True, "state_restored": True,
                       "flags_restored": True, "grads_cleared": True, "hooks_restored": True}


def test_postpublication_reconciliation_reports_verified_or_unknown(tmp_path, monkeypatch):
    output = tmp_path / "run"
    manifest, digest = _write_fake_run(output, _synthetic_logits())
    error = study._reconcile_published(output, manifest, digest, KeyboardInterrupt())
    assert isinstance(error, KeyboardInterrupt) and error.published_terminal == "complete_verified"
    monkeypatch.setattr(study, "audit", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    error = study._reconcile_published(output, manifest, digest, OSError("earlier"))
    assert isinstance(error, KeyboardInterrupt) and error.published_terminal == "unknown"


def test_failure_terminal_does_not_overwrite_published_terminal(tmp_path):
    study._json_new(tmp_path / "terminal.json", {"status": "complete"})
    study._failure_terminal(tmp_path, {"status": "failed"})
    assert json.loads((tmp_path / "terminal.json").read_bytes()) == {"status": "complete"}
    assert not (tmp_path / "terminal.json.partial").exists()


@pytest.mark.parametrize("after_publish", [False, True])
def test_failure_terminal_preserves_new_cancellation(tmp_path, monkeypatch, after_publish):
    original = study._json_new

    def interrupted(path, value):
        if after_publish:
            original(path, value)
        raise KeyboardInterrupt()

    monkeypatch.setattr(study, "_json_new", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        study._failure_terminal(tmp_path, {"status": "failed"})
    assert caught.value.published_terminal == ("failed_verified" if after_publish else "unknown")
    assert (tmp_path / "terminal.json").exists() == after_publish


def _fake_tokenizer():
    token_map = {text: list(ids) for _name, _split, text, ids, _label in study.CASES}
    token_map.update({label[0]: [label[1]] for *_rest, label in study.CASES if label})
    token_map.update({" London": [study.LONDON], " Paris": [study.PARIS]})
    for _name, split, text, ids, label in study.CASES:
        continuations = (label,) if split == "locality" else ((" London", study.LONDON), (" Paris", study.PARIS))
        token_map.update({text + token: [*ids, token_id] for token, token_id in continuations})
    cls = type("GPT2TokenizerFast", (), {"encode": lambda self, text, add_special_tokens=False: token_map[text]})
    return cls()


def _model_metadata(torch_version, transformers_version):
    return {"asset_sha256": {n: d for n, (_, d) in study.pretrained.ASSETS.items()},
            "model_id": study.pretrained.MODEL_ID, "model_revision": study.pretrained.REVISION,
            "tokenizer_id": study.pretrained.MODEL_ID, "tokenizer_revision": study.pretrained.REVISION,
            "model_class": "GPT2LMHeadModel", "tokenizer_class": "GPT2TokenizerFast",
            "weights_sha256": study.pretrained.ASSETS["model.safetensors"][1],
            "device": "cpu", "dtype": "float32", "attention_implementation": "eager",
            "use_cache": False, "parameter_count": study.PARAMETER_COUNT,
            "torch_version": torch_version, "transformers_version": transformers_version}


def _fake_runtime(monkeypatch, model, manifest):
    import transformers
    loaded = SimpleNamespace(model=model, tokenizer=_fake_tokenizer(),
                             metadata=_model_metadata(torch.__version__, transformers.__version__))
    monkeypatch.setattr(study.pretrained, "load_distilgpt2", lambda **_kw: loaded)
    monkeypatch.setattr(study, "verify_manifest", lambda *_: (manifest, "b" * 40))
    monkeypatch.setattr(study, "WEIGHT_SHAPE", (32, 8))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_complete_toy_study_freezes_then_recomputes_all_raw_outputs(tmp_path, monkeypatch):
    model = _toy_model(layers=6, vocab=study.VOCAB_SIZE)
    # The source-bound manifest for a toy-only test has the toy projection shape.
    monkeypatch.setattr(study, "WEIGHT_SHAPE", (32, 8))
    monkeypatch.setattr(study, "PARAMETER_COUNT", sum(p.numel() for p in model.parameters()))
    manifest = study._manifest("a" * 40, {name: "b" * 64 for name in study.SOURCE_PATHS})
    _fake_runtime(monkeypatch, model, manifest)
    frozen = tmp_path / "frozen.json"
    study._json_new(frozen, manifest)
    digest = study._file_sha(frozen)
    before = study._model_hash(model)
    repo = Path(study.__file__).resolve().parent.parent
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        report = study.run(repo, frozen, digest, tmp_path, tmp_path / "run")
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    assert study._model_hash(model) == before
    assert study.audit(tmp_path / "run", frozen, digest)["decision"] == report["decision"]
    assert len((tmp_path / "run" / "observations.jsonl").read_text().splitlines()) == 39


@pytest.mark.parametrize("failure", ["precommit", "evaluation", "cancelled"])
def test_precommit_gate_and_exception_rollback(tmp_path, monkeypatch, failure):
    model = _toy_model(layers=6, vocab=study.VOCAB_SIZE)
    weight = model.transformer.h[5].mlp.c_proj.weight
    before = study._model_hash(model)
    tokenizer = _fake_tokenizer()
    assets = {name: digest for name, (_, digest) in study.pretrained.ASSETS.items()}
    import transformers
    monkeypatch.setattr(study, "PARAMETER_COUNT", sum(p.numel() for p in model.parameters()))
    loaded = SimpleNamespace(model=model, tokenizer=tokenizer,
                             metadata=_model_metadata(torch.__version__, transformers.__version__))
    monkeypatch.setattr(study.pretrained, "load_distilgpt2", lambda **_kw: loaded)
    monkeypatch.setattr(study, "verify_manifest", lambda *_: ({"source_commit": "a" * 40, "assets": assets}, "b" * 40))
    monkeypatch.setattr(study, "WEIGHT_SHAPE", (32, 8))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _value: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    calls = []
    model.register_forward_pre_hook(lambda _module, _args, kwargs: calls.append(tuple(kwargs["input_ids"][0].tolist())),
                                    with_kwargs=True)
    real_json_new = study._json_new

    def write_json(path, value):
        if failure == "precommit" and path.name == "precommit.json":
            raise OSError("injected")
        return real_json_new(path, value)

    def fail_evaluation(_model, parameter, _original, candidates, _attestations, _discovery, output):
        assert (output / "precommit.json").is_file()
        with torch.no_grad():
            parameter.add_(candidates["gradient"])
        if failure == "cancelled":
            raise KeyboardInterrupt()
        raise study.StudyError("injected")

    monkeypatch.setattr(study, "_json_new", write_json)
    monkeypatch.setattr(study, "_run_conditions", fail_evaluation)
    output = tmp_path / "run"
    repo = Path(study.__file__).resolve().parent.parent
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        with pytest.raises(KeyboardInterrupt if failure == "cancelled" else Exception):
            study.run(repo, repo / "unused.json", "0" * 64, tmp_path, output)
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    terminal = json.loads((output / "terminal.json").read_bytes())
    assert terminal["status"] == ("cancelled" if failure == "cancelled" else "failed")
    assert terminal["cleanup"] == {"attempted": True, "weight_restored": True, "state_restored": True,
                                    "flags_restored": True, "grads_cleared": True, "hooks_restored": True}
    assert study._model_hash(model) == before
    assert calls == [study.CASES[0][3]]
    assert (output / "precommit.json").exists() == (failure != "precommit")
