"""Real tiny CPU models exercise the app's local loading and full-capture contract."""
import io
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (GPT2Config, GPT2LMHeadModel, DistilBertConfig,
                          DistilBertForMaskedLM, T5Config, T5ForConditionalGeneration,
                          ViTConfig, ViTForImageClassification, ViTImageProcessor,
                          PreTrainedTokenizerFast)

from looking_glass.app_runtime import AppCapture, LocalModelRuntime, DEFAULTS


def tokenizer(path):
    vocab = {word: i for i, word in enumerate(
        ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "hello", "world", "blue", "red", "the", "is", "small", "large", "a", "cat", "dog"])}
    base = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="[UNK]",
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", mask_token="[MASK]")
    tok.save_pretrained(path)
    return tok


def local_checkpoint(path, family):
    path.mkdir()
    torch.manual_seed(123)
    if family == "gpt2":
        config = GPT2Config(vocab_size=16, n_positions=32, n_ctx=32, n_embd=8,
                            n_layer=2, n_head=2, eos_token_id=None, bos_token_id=2)
        cls = GPT2LMHeadModel
    elif family == "distilbert":
        config = DistilBertConfig(vocab_size=16, dim=8, hidden_dim=16, n_layers=2,
                                 n_heads=2, max_position_embeddings=32)
        cls = DistilBertForMaskedLM
    elif family == "t5":
        config = T5Config(vocab_size=16, d_model=8, d_kv=4, d_ff=16, num_layers=2,
                          num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                          pad_token_id=0, eos_token_id=None)
        cls = T5ForConditionalGeneration
    else:
        config = ViTConfig(image_size=8, patch_size=4, num_channels=3, hidden_size=8,
                           intermediate_size=16, num_hidden_layers=2,
                           num_attention_heads=2, num_labels=3)
        cls = ViTForImageClassification
    config._attn_implementation = "eager"
    model = cls(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    if family == "vit":
        ViTImageProcessor(size={"height": 8, "width": 8}).save_pretrained(path)
    else:
        tokenizer(path)
    return model


@pytest.fixture(params=["gpt2", "distilbert", "t5", "vit"])
def loaded(tmp_path, request):
    path = tmp_path / request.param
    local_checkpoint(path, request.param)
    runtime = LocalModelRuntime()
    info = runtime.load(str(path))
    return runtime, info, path


def image_bytes():
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (8, 8), color=(20, 100, 160)).save(out, format="PNG")
    return out.getvalue()


def test_supported_real_local_models_query_full_vectors_and_leave_assets_unchanged(loaded):
    runtime, info, path = loaded
    before = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    prompt = "hello [MASK]" if info["family"] == "distilbert" else "hello world"
    result = runtime.run(prompt, {"max_new_tokens": 2}, image_bytes() if info["family"] == "vit" else None)
    assert result["coverage"]["complete"] and result["coverage"]["auto_all_supported"]
    assert result["coverage"]["all_executed_token_positions"]
    assert result["coverage"]["full_feature_vectors"]
    assert result["events"] and isinstance(result["answer"], str)
    assert result["coverage"]["captured_bytes"] == sum(e.value.numel() * 4 for e in result["events"])
    assert all(e.value.device.type == "cpu" and e.value.dtype == torch.float32 and not e.value.requires_grad for e in result["events"])
    assert len(result["events"]) == result["estimates"]["capture_events_upper_bound"]
    assert result["coverage"]["captured_bytes"] == result["estimates"]["capture_bytes_upper_bound"]
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runtime.model.modules())
    json.dumps([{**e.metadata(), "shape": list(e.value.shape)} for e in result["events"]])
    assert before == {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    metadata = result["events"][0].metadata()
    metadata["app_sequence"] = -1
    assert result["events"][0].metadata()["app_sequence"] == 0


def test_gpt2_capture_bits_and_prediction_match_uninstrumented_forward(tmp_path):
    path = tmp_path / "model"
    local_checkpoint(path, "gpt2")
    runtime = LocalModelRuntime()
    runtime.load(str(path))
    ids = runtime.tokenizer("hello world", return_tensors="pt")["input_ids"]
    expected = []
    handles = []
    for block in runtime.model.transformer.h:
        for module in (block.attn, block.mlp, block):
            handles.append(module.register_forward_hook(lambda _m, _args, out: expected.append((out[0] if isinstance(out, tuple) else out).detach().clone())))
    try:
        with torch.inference_mode():
            logits = runtime.model(input_ids=ids, use_cache=False).logits
    finally:
        for handle in handles: handle.remove()
    actual = runtime.run("hello world", {"max_new_tokens": 1})
    assert actual["generated_token_ids"] == [int(logits[0, -1].argmax())]
    assert len(actual["events"]) == len(expected) == 6
    for capture, wanted in zip(actual["events"], expected):
        assert torch.equal(capture.value.view(torch.int32), wanted.view(torch.int32))
        assert capture.value.data_ptr() != wanted.data_ptr()
    assert actual["events"][0].metadata()["generation"] == {"stage": "forward", "generation_step": None, "cache_position": None, "cache_length_before": None, "cache_length_after": None, "use_cache": False}


def test_full_prefix_generation_coordinates_and_aggregate_budget(tmp_path):
    path = tmp_path / "model"
    local_checkpoint(path, "gpt2")
    runtime = LocalModelRuntime(); runtime.load(str(path))
    result = runtime.run("hello world", {"max_new_tokens": 3})
    assert [len(s["input_token_ids"]) for s in result["steps"]] == [2, 3, 4]
    assert [e.metadata()["app_forward_index"] for e in result["events"]] == [0] * 6 + [1] * 6 + [2] * 6
    assert result["coverage"]["captured_bytes"] == (2 + 3 + 4) * 8 * 4 * 6
    with pytest.raises(ValueError, match="full requested capture exceeds budget"):
        runtime.run("hello world", {"max_new_tokens": 3, "max_events": 6})
    assert all(not m._forward_hooks for m in runtime.model.modules())


def test_selected_roles_layers_preserve_every_token(loaded):
    runtime, info, _ = loaded
    prompt = "hello [MASK]" if info["family"] == "distilbert" else "hello world"
    result = runtime.run(prompt, {"targets": ["block"], "layers": [0], "max_new_tokens": 1}, image_bytes() if info["family"] == "vit" else None)
    assert not result["coverage"]["auto_all_supported"]
    assert result["coverage"]["all_executed_token_positions"]
    assert all(e.metadata()["layer_index"] == 0 for e in result["events"])


def test_no_truncation_and_no_missing_mask(loaded):
    runtime, info, _ = loaded
    if info["family"] == "vit":
        with pytest.raises(ValueError, match="choose an image"):
            runtime.run("ignored", {})
    else:
        with pytest.raises(ValueError, match="no truncation"):
            runtime.run("hello world", {"max_input_tokens": 1})
        if info["family"] == "distilbert":
            with pytest.raises(ValueError, match="mask token"):
                runtime.run("hello world", {})


def test_budget_rejection_occurs_before_any_forward(loaded, monkeypatch):
    runtime, info, _ = loaded
    def forbidden(*args, **kwargs): raise AssertionError("forward must not execute")
    monkeypatch.setattr(runtime.model, "forward", forbidden)
    prompt = "hello [MASK]" if info["family"] == "distilbert" else "hello world"
    with pytest.raises(ValueError, match="full requested capture exceeds budget"):
        runtime.run(prompt, {"max_trace_bytes": 1}, image_bytes() if info["family"] == "vit" else None)


def test_auto_t5_covers_unequal_encoder_decoder_depths(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "t5")
    runtime = LocalModelRuntime(); info = runtime.load(str(path))
    assert info["stack_layers"] == {"encoder": 2, "decoder": 1}
    result = runtime.run("hello world", {"max_new_tokens": 2})
    assert result["coverage"]["captured_events"] == 3 * (2 * 2 * 2 + 1 + 2)
    assert [s["decoder_token_ids"] for s in result["steps"]][0] == [0]
    assert {e.metadata()["module_path"].split('.')[0] for e in result["events"]} == {"encoder", "decoder"}


def test_load_limit_and_unload(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    inspected = LocalModelRuntime.inspect(str(path))
    assert inspected["estimated_float32_weight_bytes"] > 0
    runtime = LocalModelRuntime(max_model_bytes=1)
    with pytest.raises(ValueError, match="weight estimate"):
        runtime.load(str(path))
    assert runtime.info() is None
    runtime = LocalModelRuntime(); runtime.load(str(path)); runtime.unload()
    assert runtime.model is None and runtime.info() is None
    with pytest.raises(ValueError, match="load a supported"):
        runtime.run("hello", {})


@pytest.mark.parametrize("change,match", [
    ({"model_type": "llama"}, "unsupported architecture"),
    ({"auto_map": {"AutoModel": "remote.Model"}}, "remote code"),
    ({"quantization_config": {"bits": 4}}, "quantized"),
    ({"architectures": ["GPT2ForSequenceClassification"]}, "another head"),
    ({"n_layer": 999999}, "n_layer"),
])
def test_reject_unsupported_or_custom_checkpoint(tmp_path, change, match):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    config = json.loads((path / "config.json").read_text()); config.update(change)
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match): LocalModelRuntime.inspect(str(path))


def test_pickle_only_and_traversing_shards_rejected(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    (path / "model.safetensors").unlink()
    (path / "pytorch_model.bin").write_bytes(b"not executed")
    with pytest.raises(ValueError, match="pickle"):
        LocalModelRuntime.inspect(str(path))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../secret.safetensors"}}))
    with pytest.raises(ValueError, match="traversal"):
        LocalModelRuntime.inspect(str(path))


def test_missing_head_weights_do_not_randomly_initialize(tmp_path):
    path = tmp_path / "model"; model = local_checkpoint(path, "distilbert")
    save_file({name: p for name, p in model.state_dict().items() if name.startswith("distilbert.")}, path / "model.safetensors", metadata={"format": "pt"})
    runtime = LocalModelRuntime()
    with pytest.raises(ValueError, match="exactly match"):
        runtime.load(str(path))
    assert runtime.model is None


def test_reject_bool_limits_and_unknown_settings(loaded):
    runtime, _, _ = loaded
    for options in ({"max_events": True}, {"max_new_tokens": 129}, {"layers": [True]}, {"targets": []}, {"silent_reduction": True}):
        with pytest.raises(ValueError): runtime.run("hello [MASK]", options)


def test_forward_failure_quarantines_owner_without_reuse_or_false_cleanup(tmp_path, monkeypatch):
    import looking_glass.transformer_trace as tracing
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    runtime = LocalModelRuntime(); runtime.load(str(path))
    model = runtime.model
    class FailedTrace:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): raise RuntimeError("injected uncertain trace cleanup")
        def __exit__(self, *args): pass
    monkeypatch.setattr(tracing, "GPT2TraceSession", FailedTrace)
    with pytest.raises(RuntimeError, match="uncertain trace"):
        runtime.run("hello", {"max_new_tokens": 1})
    assert runtime.info()["state"] == "quarantined"
    assert runtime.model is model
    for operation in (lambda: runtime.run("hello", {}), lambda: runtime.load(str(path)), runtime.unload):
        with pytest.raises(RuntimeError, match="restart the local app"):
            operation()


def test_capture_retains_original_model_and_effective_settings_after_reload(tmp_path):
    first = tmp_path / "first"; local_checkpoint(first, "gpt2")
    second = tmp_path / "second"; local_checkpoint(second, "distilbert")
    runtime = LocalModelRuntime(); runtime.load(str(first))
    result = runtime.run("hello world", {"max_new_tokens": 1, "targets": ["block"], "layers": [0]})
    runtime.load(str(second))
    assert result["model"]["family"] == "gpt2"
    assert result["model"]["path"] == str(first)
    assert "not a cryptographic" in result["model"]["identity_note"]
    assert result["settings"]["targets"] == ["block"]
    assert result["settings"]["max_trace_bytes"] == DEFAULTS["max_trace_bytes"]
    assert result["prompt"] == "hello world"
    json.dumps({k: v for k, v in result.items() if k != "events"})


def test_loader_never_enables_downloads_pickle_or_remote_code(tmp_path, monkeypatch):
    import transformers
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    actual = GPT2LMHeadModel.from_pretrained
    actual_tokenizer = transformers.AutoTokenizer.from_pretrained
    calls = []
    def checked_model(*args, **kwargs):
        calls.append("model")
        assert kwargs["local_files_only"] is True and kwargs["use_safetensors"] is True
        assert kwargs["trust_remote_code"] is False
        return actual(*args, **kwargs)
    def checked_tokenizer(*args, **kwargs):
        calls.append("tokenizer")
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        return actual_tokenizer(*args, **kwargs)
    monkeypatch.setattr(GPT2LMHeadModel, "from_pretrained", checked_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", checked_tokenizer)
    LocalModelRuntime().load(str(path))
    assert calls == ["model", "tokenizer"]


def test_nested_target_values_are_validation_errors(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    runtime = LocalModelRuntime(); runtime.load(str(path))
    for value in ([[]], [{}], [1]):
        with pytest.raises(ValueError, match="targets"):
            runtime.run("hello", {"targets": value})


def test_vit_preprocess_dimensions_rejected_before_pixel_allocation(tmp_path, monkeypatch):
    path = tmp_path / "model"; local_checkpoint(path, "vit")
    runtime = LocalModelRuntime(); runtime.load(str(path))
    runtime.processor.size = {"height": 100000, "width": 100000}
    def forbidden(*args, **kwargs): raise AssertionError("processor must not execute")
    monkeypatch.setattr(type(runtime.processor), "preprocess", forbidden)
    with pytest.raises(ValueError, match="resize dimensions"):
        runtime.run("", {}, image_bytes())
    assert runtime.info()["state"] == "ready"
    runtime.processor.size = {"height": 2048, "width": 2048}
    runtime.model.config.image_size = 2048
    with pytest.raises(ValueError, match="16 MiB"):
        runtime.run("", {}, image_bytes())


def test_vit_unresized_image_shape_rejected_before_processing(tmp_path, monkeypatch):
    from PIL import Image
    path = tmp_path / "model"; local_checkpoint(path, "vit")
    runtime = LocalModelRuntime(); runtime.load(str(path))
    runtime.processor.do_resize = False
    out = io.BytesIO(); Image.new("RGB", (16, 16)).save(out, format="PNG")
    def forbidden(*args, **kwargs): raise AssertionError("processor must not execute")
    monkeypatch.setattr(type(runtime.processor), "preprocess", forbidden)
    with pytest.raises(ValueError, match="unresized image dimensions"):
        runtime.run("", {}, out.getvalue())


def test_malformed_model_type_is_a_validation_error(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    cfg = json.loads((path / "config.json").read_text()); cfg["model_type"] = []
    (path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unsupported architecture"):
        LocalModelRuntime.inspect(str(path))
