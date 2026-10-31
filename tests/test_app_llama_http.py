"""Actual local Llama-shaped random checkpoints through authenticated loopback HTTP."""
import json
import threading

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

from looking_glass.app import AppState, create_server
from test_app_http import call


def llama_checkpoint(path, key_value_heads):
    """Synthetic fast-tokenizer fixture, not a downloaded/pretrained Llama model."""
    path.mkdir()
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=key_value_heads, max_position_embeddings=32,
                         bos_token_id=1, eos_token_id=None)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        LlamaForCausalLM(config).eval().save_pretrained(path, safe_serialization=True)
    words = ["<unk>", "<s>", "</s>", "hello", "world", "blue", "red"]
    backend = Tokenizer(WordLevel(vocab={word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    LlamaTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>",
                       eos_token="</s>", legacy=True).save_pretrained(path)


@pytest.mark.parametrize("key_value_heads", [4, 2])
def test_llama_local_fast_tokenizer_capture_coordinates_filter_and_reopen(tmp_path, key_value_heads):
    checkpoint = tmp_path / "llama"
    llama_checkpoint(checkpoint, key_value_heads)
    state = AppState(export_root=tmp_path / "exports")
    httpd = create_server(state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    endpoint = (state, httpd)
    def request(path, body=None, expected=200):
        code, data, _ = call(endpoint, path, "GET" if body is None else "POST", body)
        assert code == expected, data
        return json.loads(data)
    try:
        assert call(endpoint, "/api/load", "POST", {"path": str(checkpoint)}, token=False)[0] == 403
        inspected = request("/api/inspect", {"path": str(checkpoint)})
        assert inspected["family"] == "llama" and inspected["supported_devices"] == ["cpu", "cuda"]
        assert inspected["input_summary"]["class"] == "LlamaTokenizerFast"
        assert not inspected["inspection"]["weights_loaded"] and state.runtime.model is None
        request("/api/load", {"path": str(checkpoint), "device": "cpu"})
        report = request("/api/run", {"prompt": "hello world", "settings": {"max_new_tokens": 2}})
        assert report["complete"] and report["coverage"]["forward_count"] == 2
        assert report["model"]["family"] == "llama" and report["model"]["device"] == "cpu"
        assert report["steps"][0]["input_token_ids"] == [1, 3, 4]
        assert report["steps"][1]["input_token_ids"] == [1, 3, 4, report["steps"][0]["selected_token_id"]]
        assert report["event_count"] == 12 and not report["intervention_available"]
        capture = report["capture_id"]
        rows = request("/api/events?capture_id=" + capture)["events"]
        original = []
        for row in rows:
            forward = row["metadata"]["app_forward_index"]
            assert row["vector_count"] == 3 + forward and row["shape"] == [1, 3 + forward, 16]
            assert row["source_summary"]["status"] == "multiple"
            path = row["metadata"]["module_path"]
            assert row["metadata"]["output_path"] == ([0] if path.endswith(".self_attn") else [])
            for vector in range(row["vector_count"]):
                value = request(f"/api/tensor?event={row['index']}&vector={vector}&capture_id={capture}")
                coordinate = value["coordinate"]
                assert coordinate["status"] == "known" and coordinate["stack"] == "input"
                assert coordinate["token_id"] == report["steps"][forward]["input_token_ids"][vector]
                original.append((row["index"], vector, value))
        query = "/api/events?token_position=2&forward=1&stack=input"
        filtered = request(query + "&capture_id=" + capture)
        assert filtered["total"] == 6 and all(row["matched_vector_index"] == 2 for row in filtered["events"])
        assert not state.store.export_root.exists()
        exported = request("/api/save", {"name": "llama", "capture_id": capture})
        request("/api/unload", {}); request("/api/clear", {})
        reopened = request("/api/open", {"path": exported["path"]})
        assert not reopened["intervention_available"]
        assert request(query + "&capture_id=" + reopened["capture_id"]) == filtered
        for event, vector, value in original:
            assert request(f"/api/tensor?event={event}&vector={vector}&capture_id=" + reopened["capture_id"]) == value
    finally:
        httpd.shutdown(); thread.join(5); httpd.server_close()
        assert not thread.is_alive()
