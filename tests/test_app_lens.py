"""Logit-lens read-out: checked against the model's own logits and hidden states.

The reference here is independent of the capture hooks: Hugging Face's
``output_hidden_states`` plus a direct call to the final norm and unembedding.
"""
import json
import threading

import pytest
import torch

from looking_glass import app_lens
from looking_glass.app import AppState, create_server
from test_app_http import call
from test_app_llama_http import llama_checkpoint


@pytest.fixture
def demo(tmp_path):
    state = AppState(export_root=tmp_path / "exports")
    state.post("demo", {})
    run = state.post("run", {"prompt": "Hello world", "settings": {"max_new_tokens": 3}})
    return state, run


def reference_logits(state, token_ids):
    """Per-layer logits from Hugging Face hidden states, bypassing the app's capture path."""
    model = state.runtime.model
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([token_ids]), output_hidden_states=True, return_dict=True)
        norm, unembed, _ = app_lens.decoder_modules(model, state.runtime.info()["family"])
        # hidden_states[0] is the embedding output; block L's output is hidden_states[L + 1].
        # The last entry already has the final norm applied, so the last layer's
        # reference is simply the model's real logits.
        layers = [unembed(norm(hidden[0])) for hidden in output.hidden_states[1:-1]]
        layers.append(output.logits[0])
    return layers, output.logits[0]


def test_summary_advertises_the_lens_for_a_live_capture(demo):
    _, run = demo
    assert run["lens_available"] is True and "final norm" in run["lens_reason"]


def test_every_cell_matches_an_independent_reference(demo):
    state, run = demo
    lens = state.post("lens", {"capture_id": run["capture_id"], "forward": 0, "top_k": 4})
    token_ids = run["steps"][0]["input_token_ids"]
    layers, real_logits = reference_logits(state, token_ids)

    assert lens["method"] == "logit" and lens["forward_count"] == 3
    assert [token["token_id"] for token in lens["tokens"]] == token_ids
    assert [layer["layer_index"] for layer in lens["layers"]] == [0, 1]
    assert lens["reaches_model_output"] and lens["vocabulary_size"] == 259

    for layer, expected in zip(lens["layers"], layers):
        probabilities = expected.softmax(dim=-1)
        assert [cell["position"] for cell in layer["cells"]] == list(range(len(token_ids)))
        for cell in layer["cells"]:
            row = probabilities[cell["position"]]
            values, ids = torch.topk(row, 4)
            assert [entry["token_id"] for entry in cell["top"]] == ids.tolist()
            assert [entry["probability"] for entry in cell["top"]] == pytest.approx(values.tolist(), rel=1e-5)
            entropy = float(-(row * row.log()).sum())
            assert cell["entropy"] == pytest.approx(entropy, rel=1e-4)

    # The top row is the model's real output, token for token.
    assert [answer["token_id"] for answer in lens["answers"]] == real_logits.argmax(dim=-1).tolist()
    assert all(cell["answer_rank"] == 1 for cell in lens["layers"][-1]["cells"])
    assert lens["consistency"] == {"recorded_next_token_id": run["steps"][0]["selected_token_id"],
                                   "lens_next_token_id": run["steps"][0]["selected_token_id"], "matches": True}


def test_pinned_token_ranks_count_strictly_larger_logits(demo):
    state, run = demo
    layers, _ = reference_logits(state, run["steps"][0]["input_token_ids"])
    pinned = [70, 5, 258]
    lens = state.post("lens", {"capture_id": run["capture_id"], "pinned_token_ids": pinned})
    assert [entry["token_id"] for entry in lens["pinned"]] == pinned
    for layer, expected in zip(lens["layers"], layers):
        for cell in layer["cells"]:
            row = expected[cell["position"]]
            for entry in cell["pinned"]:
                assert entry["rank"] == int((row > row[entry["token_id"]]).sum()) + 1
                assert entry["probability"] == pytest.approx(float(row.softmax(-1)[entry["token_id"]]), rel=1e-5)


def test_later_forward_and_position_window(demo):
    state, run = demo
    last = state.post("lens", {"capture_id": run["capture_id"], "forward": 2})
    assert len(last["tokens"]) == len(run["steps"][2]["input_token_ids"]) == len(run["steps"][0]["input_token_ids"]) + 2
    assert last["consistency"]["matches"]

    window = state.post("lens", {"capture_id": run["capture_id"], "position_offset": 3, "position_limit": 4})
    assert [token["position"] for token in window["tokens"]] == [3, 4, 5, 6]
    assert window["position_total"] == 11 and window["consistency"] is None  # window stops before the last position
    full = state.post("lens", {"capture_id": run["capture_id"]})
    assert window["layers"][1]["cells"] == full["layers"][1]["cells"][3:7]
    assert full["layers"][0]["event_index"] != full["layers"][1]["event_index"]


def test_resolve_text_offers_plain_and_leading_space_tokens(demo):
    state, run = demo
    lens = state.post("lens", {"capture_id": run["capture_id"], "resolve_text": "d"})
    assert lens["resolved"][0] == {"token_id": 70, "text": "d", "typed": "d", "token_count": 1, "single_token": True}
    # The demo's byte vocabulary has no merged " d" token, so the spaced form is two tokens.
    assert lens["resolved"][1]["typed"] == " d" and lens["resolved"][1]["single_token"] is False


@pytest.mark.parametrize("body,message", [
    ({"capture_id": "stale"}, "stale"),
    ({"forward": 9}, "forward"),
    ({"top_k": 11}, "top_k"),
    ({"pinned_token_ids": list(range(9))}, "pinned_token_ids"),
    ({"pinned_token_ids": [3, 3]}, "pinned_token_ids"),
    ({"pinned_token_ids": [999]}, "outside the vocabulary"),
    ({"position_offset": 99}, "position_offset"),
    ({"resolve_text": ""}, "Text to pin"),
    ({"surprise": 1}, "Unknown lens option"),
])
def test_invalid_requests_are_rejected_without_harming_the_model(demo, body, message):
    state, run = demo
    request = {"capture_id": run["capture_id"], **body}
    with pytest.raises(ValueError, match=message):
        state.post("lens", request)
    assert state.runtime.info()["state"] == "ready"
    assert state.post("lens", {"capture_id": run["capture_id"]})["consistency"]["matches"]


def test_partial_layer_capture_is_labelled_as_not_the_model_output(tmp_path):
    state = AppState(export_root=tmp_path / "exports")
    state.post("demo", {})
    run = state.post("run", {"prompt": "Hello", "settings": {"max_new_tokens": 1, "layers": [0]}})
    lens = state.post("lens", {"capture_id": run["capture_id"]})
    assert [layer["layer_index"] for layer in lens["layers"]] == [0]
    assert lens["reaches_model_output"] is False and lens["consistency"] is None


def test_capture_without_block_outputs_is_unavailable(tmp_path):
    state = AppState(export_root=tmp_path / "exports")
    state.post("demo", {})
    run = state.post("run", {"prompt": "Hello", "settings": {"max_new_tokens": 1, "targets": ["mlp"]}})
    assert run["lens_available"] is False and "block" in run["lens_reason"]
    with pytest.raises(ValueError, match="block"):
        state.post("lens", {"capture_id": run["capture_id"]})


def test_reopened_capture_is_read_only(demo):
    state, run = demo
    saved = state.post("save", {"name": "lens-source", "capture_id": run["capture_id"]})
    reopened = state.post("open", {"path": saved["path"]})
    assert reopened["lens_available"] is False and "reopened" in reopened["lens_reason"]
    with pytest.raises(ValueError, match="reopened"):
        state.post("lens", {"capture_id": reopened["capture_id"]})


def test_llama_capture_uses_its_own_norm_and_head(tmp_path):
    checkpoint = tmp_path / "llama"
    llama_checkpoint(checkpoint, 4)
    state = AppState(export_root=tmp_path / "exports")
    state.post("load", {"path": str(checkpoint), "device": "cpu", "mode": "capture"})
    run = state.post("run", {"prompt": "hello world", "settings": {"max_new_tokens": 1}})
    lens = state.post("lens", {"capture_id": run["capture_id"], "top_k": 3})
    layers, real_logits = reference_logits(state, run["steps"][0]["input_token_ids"])
    assert lens["decoder"] == "model.norm then lm_head" and lens["consistency"]["matches"]
    assert [answer["token_id"] for answer in lens["answers"]] == real_logits.argmax(dim=-1).tolist()
    for layer, expected in zip(lens["layers"], layers):
        for cell in layer["cells"]:
            assert cell["top"][0]["token_id"] == int(expected[cell["position"]].argmax())


def test_lens_over_authenticated_http_and_static_assets(tmp_path):
    state = AppState(export_root=tmp_path / "exports")
    httpd = create_server(state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    endpoint = (state, httpd)
    try:
        assert call(endpoint, "/api/demo", "POST", {})[0] == 200
        code, data, _ = call(endpoint, "/api/run", "POST", {"prompt": "Hi", "settings": {"max_new_tokens": 1}})
        capture = json.loads(data)["capture_id"]
        assert call(endpoint, "/api/lens", "POST", {"capture_id": capture}, token=False)[0] == 403
        code, data, _ = call(endpoint, "/api/lens", "POST", {"capture_id": capture})
        assert code == 200 and json.loads(data)["consistency"]["matches"]
        code, data, _ = call(endpoint, "/api/lens", "POST", {"capture_id": "stale"})
        assert code == 400 and "stale" in json.loads(data)["error"]
        for path, kind in (("/app_lens.js", "application/javascript"), ("/app_field.js", "application/javascript"),
                           ("/app_lens.css", "text/css"),
                           ("/favicon.svg", "image/svg+xml")):
            code, body, headers = call(endpoint, path)
            assert code == 200 and body and headers["Content-Type"].startswith(kind), path
    finally:
        httpd.shutdown(); thread.join(5); httpd.server_close()


@pytest.mark.parametrize("script", ["app_lens_contract.cjs", "app_field_contract.cjs"])
def test_lens_view_encoding_contracts_in_node(script):
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests" / script)], cwd=root,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
