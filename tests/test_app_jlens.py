"""Jacobian lens: the estimator against a brute-force Jacobian, and the app's fit and read-out flow."""
import json
import threading

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from looking_glass import app_jlens, app_lens, jacobian_lens
from looking_glass.app import AppState, create_server
from test_app_http import call
from test_app_llama_http import llama_checkpoint

TEXT = "\n\n".join(f"Passage {n} is about rivers, bridges, the weather in spring and a long walk home." for n in range(8))


def tiny_gpt2(layers=3, width=24):
    config = GPT2Config(vocab_size=101, n_positions=64, n_embd=width, n_layer=layers, n_head=2,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=1, eos_token_id=None)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = GPT2LMHeadModel(config).eval()
        ids = torch.randint(0, 101, (1, 30))
    model.requires_grad_(False)
    return model, ids


def final_block_output(model, ids, layer, hidden):
    """Final block output when block ``layer``'s output is replaced by ``hidden``; uses the model's own forward."""
    blocks, seen = model.transformer.h, {}

    def replace(_module, _inputs, output):
        if torch.is_tensor(output):
            return hidden.unsqueeze(0)
        return (hidden.unsqueeze(0),) + tuple(output[1:])

    def keep(_module, _inputs, output):
        seen["final"] = output if torch.is_tensor(output) else output[0]

    handles = [blocks[layer].register_forward_hook(replace), blocks[-1].register_forward_hook(keep)]
    try:
        model.transformer(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return seen["final"][0]


def test_estimator_matches_a_brute_force_jacobian():
    model, ids = tiny_gpt2()
    hooks_before = [len(block._forward_hooks) for block in model.transformer.h]
    estimated, valid_count = jacobian_lens.prompt_jacobians(model, "gpt2", ids, dim_batch=5)
    # Fitting leaves no hooks and no gradients behind.
    assert [len(block._forward_hooks) for block in model.transformer.h] == hooks_before
    assert all(parameter.grad is None for parameter in model.parameters())
    valid = jacobian_lens.valid_positions(ids.shape[1])
    assert valid_count == len(valid) == 13 and sorted(estimated) == [0, 1]
    with torch.no_grad():
        states = model.transformer(input_ids=ids, output_hidden_states=True).hidden_states
    for layer, matrix in estimated.items():
        def run(hidden, layer=layer):
            return final_block_output(model, ids, layer, hidden)
        full = torch.autograd.functional.jacobian(run, states[layer + 1][0].clone())   # [target, i, source, j]
        expected = full[valid][:, :, valid, :].sum(dim=0).mean(dim=1)                  # sum targets, mean sources
        assert torch.allclose(matrix, expected, atol=1e-5), layer
    again, _ = jacobian_lens.prompt_jacobians(model, "gpt2", ids, dim_batch=24)
    assert all(torch.allclose(again[layer], estimated[layer], atol=1e-5) for layer in estimated)


def test_short_prompts_and_trainable_models_are_rejected():
    model, ids = tiny_gpt2()
    with pytest.raises(ValueError, match="too short"):
        jacobian_lens.prompt_jacobians(model, "gpt2", ids[:, :17])
    model.requires_grad_(True)
    with pytest.raises(ValueError, match="must not require gradients"):
        jacobian_lens.prompt_jacobians(model, "gpt2", ids)
    with pytest.raises(ValueError, match="GPT-2 and standard Llama"):
        jacobian_lens.decoder_stack(model, "t5")


def test_running_mean_save_load_and_fingerprint(tmp_path):
    model, ids = tiny_gpt2()
    fingerprint = jacobian_lens.model_fingerprint(model, "gpt2")
    fit = jacobian_lens.LensFit(fingerprint, "gpt2", 3, 24)
    first, valid = jacobian_lens.prompt_jacobians(model, "gpt2", ids)
    second, _ = jacobian_lens.prompt_jacobians(model, "gpt2", ids.flip(1))
    fit.add(first, valid)
    assert fit.last_change is None
    fit.add(second, valid)
    assert fit.last_change > 0 and fit.prompts == 2 and fit.positions == 2 * valid
    lens = fit.lens()
    assert torch.allclose(lens.matrices[0], (first[0] + second[0]) / 2)
    path = jacobian_lens.lens_path(tmp_path, fingerprint)
    lens.save(path)
    loaded = jacobian_lens.JacobianLens.load(path)
    assert loaded.summary() == lens.summary() and loaded.fingerprint == fingerprint
    assert all(torch.equal(loaded.matrices[layer], lens.matrices[layer]) for layer in lens.matrices)
    assert not list(tmp_path.glob("*.tmp.*"))
    # Different weights give a different fingerprint, so a lens is never applied to another model.
    other, _ = tiny_gpt2(width=32)
    assert jacobian_lens.model_fingerprint(other, "gpt2") != fingerprint
    with torch.no_grad():
        model.transformer.h[-1].mlp.c_fc.weight.add_(1.0)
    assert jacobian_lens.model_fingerprint(model, "gpt2") != fingerprint
    hidden = torch.randn(4, 24)
    assert torch.allclose(lens.transport(hidden, 1), hidden @ lens.matrices[1].T)


def fitted_demo(tmp_path, prompts=4):
    state = AppState(export_root=tmp_path / "runs", lens_root=tmp_path / "lenses")
    state.post("demo", {})
    status = state.post("jlens-fit-start", {"source": "text", "text": TEXT, "max_prompts": prompts})
    fit_id, steps = status["fit"]["fit_id"], 0
    while status["fit"] is not None:
        status = state.post("jlens-fit-step", {"fit_id": fit_id})
        steps += 1
    return state, status, steps


def test_app_fit_then_jacobian_readout_matches_manual_transport(tmp_path):
    empty = AppState(export_root=tmp_path / "runs0", lens_root=tmp_path / "lenses0")
    assert empty.post("jlens-status", {}) == {"supported": False, "reason": "Load a model first.", "lens": None,
                                              "fit": None, "can_fit": False, "fit_reason": "Load a model first."}
    state, status, steps = fitted_demo(tmp_path)
    assert steps == 4 and status["lens"]["prompts"] == 4 and status["lens"]["layers"] == 1 and status["fit"] is None
    assert status["can_fit"] and status["device"] == "cpu" and status["backward_passes"] == 2

    run = state.post("run", {"prompt": "Hello world, this is a test", "settings": {"max_new_tokens": 1}})
    logit = state.post("lens", {"capture_id": run["capture_id"], "top_k": 3})
    jacobian = state.post("lens", {"capture_id": run["capture_id"], "top_k": 3, "method": "jacobian"})
    assert logit["method"] == "logit" and logit["fit"] is None
    assert jacobian["method"] == "jacobian" and jacobian["fit"]["prompts"] == 4
    assert jacobian["decoder"].startswith("fitted J_l") and "Jacobian lens" in jacobian["scope"]
    # The last layer is the model's own output under both methods.
    assert jacobian["layers"][-1]["cells"] == logit["layers"][-1]["cells"]
    assert jacobian["answers"] == logit["answers"] and jacobian["consistency"]["matches"]

    # Layer 0 equals unembed(final_norm(J_0 @ h)) computed by hand from the same captured vectors.
    model, lens = state.runtime.model, state.jlens.lens
    source = state.store.lens_sources(0, "block")["sources"][0]
    norm, unembed, _ = app_lens.decoder_modules(model, "gpt2")
    with torch.inference_mode():
        expected = unembed(norm(source["value"] @ lens.matrices[0].T)).softmax(dim=-1)
    for cell in jacobian["layers"][0]["cells"]:
        values, ids = torch.topk(expected[cell["position"]], 3)
        assert [entry["token_id"] for entry in cell["top"]] == ids.tolist()
        assert [entry["probability"] for entry in cell["top"]] == pytest.approx(values.tolist(), rel=1e-5)

    # A fresh app with the same lens folder finds the lens for the same weights.
    again = AppState(export_root=tmp_path / "runs2", lens_root=tmp_path / "lenses")
    again.post("demo", {})
    assert again.post("jlens-status", {})["lens"] == status["lens"]


def test_jacobian_method_needs_a_fitted_lens_and_fit_ids_are_checked(tmp_path):
    state = AppState(export_root=tmp_path / "runs", lens_root=tmp_path / "lenses")
    state.post("demo", {})
    run = state.post("run", {"prompt": "Hello world", "settings": {"max_new_tokens": 1}})
    with pytest.raises(ValueError, match="Fit one first"):
        state.post("lens", {"capture_id": run["capture_id"], "method": "jacobian"})
    with pytest.raises(ValueError, match="method must be"):
        state.post("lens", {"capture_id": run["capture_id"], "method": "tuned"})
    with pytest.raises(ValueError, match="No passage is longer"):
        state.post("jlens-fit-start", {"source": "text", "text": "too short"})
    with pytest.raises(ValueError, match="Unknown fit option"):
        state.post("jlens-fit-start", {"surprise": 1})
    started = state.post("jlens-fit-start", {"source": "text", "text": TEXT, "max_prompts": 3})
    with pytest.raises(ValueError, match="no longer active"):
        state.post("jlens-fit-step", {"fit_id": "stale"})
    stepped = state.post("jlens-fit-step", {"fit_id": started["fit"]["fit_id"]})
    assert stepped["fit"]["done"] == 1 and stepped["fit"]["total"] == 3 and stepped["lens"] is None
    cancelled = state.post("jlens-fit-cancel", {"fit_id": started["fit"]["fit_id"]})
    assert cancelled["fit"] is None and cancelled["lens"] is None
    assert not list((tmp_path / "lenses").glob("*")) if (tmp_path / "lenses").exists() else True
    # Loading another model drops the session's lens and fit.
    state.post("jlens-fit-start", {"source": "text", "text": TEXT, "max_prompts": 2})
    state.post("unload", {})
    assert state.post("jlens-status", {})["supported"] is False and state.jlens.fit is None


def test_wide_models_must_be_on_a_gpu_to_fit(tmp_path, monkeypatch):
    state = AppState(export_root=tmp_path / "runs", lens_root=tmp_path / "lenses")
    state.post("demo", {})
    real = state.runtime.jacobian_identity

    def wide_on_cpu():
        return {**real(), "width": 768}

    monkeypatch.setattr(state.runtime, "jacobian_identity", wide_on_cpu)
    status = state.post("jlens-status", {})
    assert status["supported"] and not status["can_fit"] and "NVIDIA GPU" in status["fit_reason"]
    assert status["backward_passes"] == 96
    with pytest.raises(ValueError, match="NVIDIA GPU"):
        state.post("jlens-fit-start", {})

    def wide_on_gpu():
        return {**real(), "width": 768, "device": "cuda"}

    monkeypatch.setattr(state.runtime, "jacobian_identity", wide_on_gpu)
    assert state.post("jlens-status", {})["can_fit"]


def test_builtin_corpus_is_usable(tmp_path):
    passages = [passage for passage in app_jlens.builtin_corpus().split("\n\n") if passage.strip()]
    assert len(passages) >= 40
    assert all(len(passage.split()) >= 90 for passage in passages), "each passage should fill a 128-token prompt"
    state = AppState(export_root=tmp_path / "runs", lens_root=tmp_path / "lenses")
    state.post("demo", {})
    started = state.post("jlens-fit-start", {"max_prompts": 3})
    assert started["fit"]["total"] == 3
    prompts = state.runtime.jacobian_prompts(app_jlens.builtin_corpus(), 1000)
    assert len(prompts) >= 40 and all(17 < len(prompt) <= 128 for prompt in prompts)


def test_llama_fit_and_readout(tmp_path):
    checkpoint = tmp_path / "llama"
    llama_checkpoint(checkpoint, 4)
    state = AppState(export_root=tmp_path / "runs", lens_root=tmp_path / "lenses")
    state.post("load", {"path": str(checkpoint), "device": "cpu", "mode": "capture"})
    text = "\n\n".join(["hello world hello world hello world hello"] * 3)
    status = state.post("jlens-fit-start", {"source": "text", "text": text, "max_prompts": 2})
    while status["fit"] is not None:
        status = state.post("jlens-fit-step", {"fit_id": status["fit"]["fit_id"]})
    assert status["lens"]["prompts"] == 2 and status["lens"]["width"] == 16
    run = state.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    lens = state.post("lens", {"capture_id": run["capture_id"], "method": "jacobian"})
    assert lens["decoder"] == "fitted J_l, then model.norm then lm_head" and lens["consistency"]["matches"]


def test_jlens_routes_over_authenticated_http(tmp_path):
    state = AppState(export_root=tmp_path / "runs", lens_root=tmp_path / "lenses")
    httpd = create_server(state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    endpoint = (state, httpd)
    try:
        assert call(endpoint, "/api/demo", "POST", {})[0] == 200
        assert call(endpoint, "/api/jlens-status", "POST", {}, token=False)[0] == 403
        code, data, _ = call(endpoint, "/api/jlens-status", "POST", {})
        assert code == 200 and json.loads(data)["can_fit"]
        code, data, _ = call(endpoint, "/api/jlens-fit-start", "POST", {"source": "text", "text": TEXT, "max_prompts": 1})
        fit_id = json.loads(data)["fit"]["fit_id"]
        code, data, _ = call(endpoint, "/api/jlens-fit-step", "POST", {"fit_id": fit_id})
        assert code == 200 and json.loads(data)["lens"]["prompts"] == 1
        code, data, _ = call(endpoint, "/api/jlens-fit-step", "POST", {"fit_id": fit_id})
        assert code == 400 and "no longer active" in json.loads(data)["error"]
    finally:
        httpd.shutdown(); thread.join(5); httpd.server_close()


def test_command_line_fit_writes_a_lens_the_app_finds(tmp_path, capsys):
    from looking_glass import jlens_cli
    checkpoint = tmp_path / "llama"
    llama_checkpoint(checkpoint, 4)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n\n".join(["hello world hello world hello world hello"] * 3))
    report = tmp_path / "report.json"
    code = jlens_cli.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--prompts", "2",
                           "--text-file", str(corpus), "--lens-dir", str(tmp_path / "lenses"), "--report", str(report)])
    assert code == 0
    written = json.loads(report.read_text())
    assert written["fit"]["prompts"] == 2 and written["batch_consistency"]["ok"]
    assert set(written["answer_rank_by_layer"]) == {"0", "1"}
    state = AppState(export_root=tmp_path / "runs", lens_root=tmp_path / "lenses")
    state.post("load", {"path": str(checkpoint), "device": "cpu", "mode": "capture"})
    assert state.post("jlens-status", {})["lens"]["prompts"] == 2
    assert "Jacobian lens" in capsys.readouterr().out


class Unsafe:
    """Stands in for arbitrary code in a pickle; a tensors-only load must refuse it."""


def test_import_of_a_jlens_library_file(tmp_path):
    state, status, _ = fitted_demo(tmp_path, prompts=2)
    own = {layer: matrix.clone() for layer, matrix in state.jlens.lens.matrices.items()}
    # The published format: a torch.save dictionary with fp16 matrices.
    good = tmp_path / "demo_jacobian_lens.pt"
    torch.save({"J": {0: (own[0] * 2).to(torch.float16)}, "source_layers": [0], "d_model": 16, "n_prompts": 277}, good)
    imported = state.post("jlens-import", {"path": str(good)})
    assert imported["lens"]["prompts"] == 277 and imported["lens"]["origin"] == "imported from demo_jacobian_lens.pt"
    assert imported["lens"]["positions"] == 0
    assert torch.allclose(state.jlens.lens.matrices[0], (own[0] * 2).to(torch.float16).float())
    run = state.post("run", {"prompt": "Hello world", "settings": {"max_new_tokens": 1}})
    assert state.post("lens", {"capture_id": run["capture_id"], "method": "jacobian"})["fit"]["prompts"] == 277
    # It replaces the saved lens, so a fresh app picks up the imported one.
    again = AppState(export_root=tmp_path / "runs3", lens_root=tmp_path / "lenses")
    again.post("demo", {})
    assert again.post("jlens-status", {})["lens"]["origin"] == "imported from demo_jacobian_lens.pt"

    wrong_width = tmp_path / "wide.pt"
    torch.save({"J": {0: torch.zeros(32, 32)}, "source_layers": [0], "d_model": 32, "n_prompts": 5}, wrong_width)
    with pytest.raises(ValueError, match="another width"):
        state.post("jlens-import", {"path": str(wrong_width)})
    wrong_layers = tmp_path / "deep.pt"
    torch.save({"J": {0: torch.zeros(16, 16), 1: torch.zeros(16, 16)}, "d_model": 16, "n_prompts": 5}, wrong_layers)
    with pytest.raises(ValueError, match="needs 0 to 0"):
        state.post("jlens-import", {"path": str(wrong_layers)})
    other_target = tmp_path / "target.pt"
    torch.save({"J": {0: torch.zeros(16, 16)}, "d_model": 16, "n_prompts": 5, "provenance": {"target_layer": 0}}, other_target)
    with pytest.raises(ValueError, match="other than the model's last"):
        state.post("jlens-import", {"path": str(other_target)})
    unsafe = tmp_path / "unsafe.pt"
    torch.save({"J": {0: torch.zeros(16, 16)}, "d_model": 16, "n_prompts": 5, "extra": Unsafe()}, unsafe)
    with pytest.raises(ValueError, match="tensors only"):
        state.post("jlens-import", {"path": str(unsafe)})
    with pytest.raises(ValueError, match="missing or too large"):
        state.post("jlens-import", {"path": str(tmp_path / "absent.pt")})
    # A failed import leaves the working lens in place.
    assert state.post("jlens-status", {})["lens"]["prompts"] == 277
