"""Local input-asset validation occurs before weight allocation or owner retirement."""
import json

import pytest
import torch
import transformers

from looking_glass.app_runtime import LocalModelRuntime
from test_app_runtime import local_checkpoint


@pytest.mark.parametrize("family", ["gpt2", "distilbert", "t5", "vit"])
def test_inspect_checks_inputs_without_loading_weights(tmp_path, monkeypatch, family):
    path = tmp_path / "model"; local_checkpoint(path, family)
    for cls in (transformers.GPT2LMHeadModel, transformers.DistilBertForMaskedLM,
                transformers.T5ForConditionalGeneration, transformers.ViTForImageClassification):
        monkeypatch.setattr(cls, "from_pretrained", lambda *a, **k: pytest.fail("inspection must not load weights"))
    report = LocalModelRuntime.inspect(str(path))
    assert report["inspection"]["input_assets_loadable"]
    assert not report["inspection"]["weights_loaded"] and not report["inspection"]["inference_tested"]
    assert report["input_summary"]["kind"] == ("image_processor" if family == "vit" else "fast_tokenizer")


def test_missing_tokenizer_preserves_live_demo_without_weights(tmp_path, monkeypatch):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    for file in path.iterdir():
        if file.name not in ("config.json", "model.safetensors", "generation_config.json"): file.unlink()
    runtime = LocalModelRuntime(); runtime.load_demo()
    model, binding = runtime.model, runtime.capture_binding()
    monkeypatch.setattr(transformers.GPT2LMHeadModel, "from_pretrained", lambda *a, **k: pytest.fail("weights must not load"))
    with pytest.raises(ValueError, match="input assets|tokenizer"):
        runtime.load(str(path), device="cpu")
    assert runtime.model is model and runtime.capture_binding() is binding


@pytest.mark.parametrize("family", ["gpt2", "distilbert", "t5", "vit"])
def test_load_prepares_inputs_once_and_reuses_object(tmp_path, monkeypatch, family):
    path = tmp_path / "model"; local_checkpoint(path, family)
    owner = transformers.ViTImageProcessor if family == "vit" else transformers.AutoTokenizer
    original = owner.from_pretrained
    prepared = []
    def counted(*args, **kwargs):
        result = original(*args, **kwargs); prepared.append(result); return result
    monkeypatch.setattr(owner, "from_pretrained", counted)
    runtime = LocalModelRuntime(); runtime.load(str(path), device="cpu")
    assert len(prepared) == 1 and (runtime.processor if family == "vit" else runtime.tokenizer) is prepared[0]


@pytest.mark.parametrize("family,mutation", [("gpt2", "range"), ("distilbert", "mask"), ("t5", "start"), ("vit", "geometry")])
def test_bad_input_contract_preserves_previous_owner(tmp_path, monkeypatch, family, mutation):
    path = tmp_path / "model"; local_checkpoint(path, family)
    runtime = LocalModelRuntime(); runtime.load_demo()
    owner, binding = runtime.model, runtime.capture_binding()
    if mutation == "range":
        file = path / "tokenizer.json"; data = json.loads(file.read_text()); data["model"]["vocab"]["outside"] = 99; file.write_text(json.dumps(data))
    elif mutation == "mask":
        file = path / "tokenizer_config.json"; data = json.loads(file.read_text()); data["mask_token"] = None; file.write_text(json.dumps(data))
        file = path / "special_tokens_map.json"; data = json.loads(file.read_text()); data.pop("mask_token", None); file.write_text(json.dumps(data))
    elif mutation == "start":
        file = path / "config.json"; data = json.loads(file.read_text()); data["decoder_start_token_id"] = 99; file.write_text(json.dumps(data))
    else:
        file = path / "preprocessor_config.json"; data = json.loads(file.read_text()); data["size"] = {"height": 9999999, "width": 9999999}; file.write_text(json.dumps(data))
    for cls in (transformers.GPT2LMHeadModel, transformers.DistilBertForMaskedLM, transformers.T5ForConditionalGeneration, transformers.ViTForImageClassification):
        monkeypatch.setattr(cls, "from_pretrained", lambda *a, **k: pytest.fail("must not load weights"))
    with pytest.raises(ValueError): runtime.load(str(path), device="cpu")
    assert runtime.model is owner and runtime.capture_binding() is binding


@pytest.mark.parametrize("aggregate", [False, True])
def test_input_asset_byte_limits_precede_parser(tmp_path, monkeypatch, aggregate):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    names = ("vocab.json", "merges.txt", "vocab.txt") if aggregate else ("tokenizer.json",)
    for name in names:
        with (path / name).open("wb") as stream: stream.truncate(64 * 1024**2 + (0 if aggregate else 1))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("must bound before parsing"))
    with pytest.raises(ValueError, match="byte limit|128 MiB"):
        LocalModelRuntime.inspect(str(path))


@pytest.mark.parametrize("family", ["gpt2", "distilbert"])
def test_valid_legacy_fast_layout_without_tokenizer_json(tmp_path, family):
    path = tmp_path / "model"; local_checkpoint(path, family)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (path / name).unlink()
    words = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "hello", "world", "blue", "red", "the", "is", "small", "large", "a", "cat", "dog"]
    if family == "gpt2":
        (path / "vocab.json").write_text(json.dumps({word: i for i, word in enumerate(words)}))
        (path / "merges.txt").write_text("#version: 0.2\n")
        (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GPT2TokenizerFast", "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]"}))
    else:
        (path / "vocab.txt").write_text("\n".join(words))
        (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "DistilBertTokenizerFast", "unk_token": "[UNK]", "sep_token": "[EOS]", "cls_token": "[BOS]", "pad_token": "[PAD]", "mask_token": "[MASK]"}))
    assert not (path / "tokenizer.json").exists()
    assert LocalModelRuntime.inspect(str(path))["input_summary"]["kind"] == "fast_tokenizer"


def test_optional_converter_dependency_error_is_actionable_and_oom_propagates(tmp_path, monkeypatch):
    path = tmp_path / "model"; local_checkpoint(path, "t5")
    def missing(*args, **kwargs): raise ImportError("raw protobuf dependency error")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", missing)
    with pytest.raises(ValueError, match="matching fast tokenizer assets") as caught:
        LocalModelRuntime.inspect(str(path))
    assert "protobuf" not in str(caught.value)
    def exhausted(*args, **kwargs): raise MemoryError("resource exhaustion")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", exhausted)
    with pytest.raises(MemoryError): LocalModelRuntime.inspect(str(path))


def test_padded_t5_vocabulary_does_not_require_tokenizer_length_equality(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "t5")
    config = transformers.T5Config.from_pretrained(path, local_files_only=True)
    config.vocab_size = 32
    transformers.T5ForConditionalGeneration(config).save_pretrained(path, safe_serialization=True)
    info = LocalModelRuntime.inspect(str(path))
    assert info["input_summary"]["vocabulary_entries"] == 16
    assert info["input_summary"]["model_vocabulary_size"] == 32
    runtime = LocalModelRuntime(); runtime.load(str(path), device="cpu")
    assert runtime.run("hello", {"max_new_tokens": 1})["coverage"]["complete"]


def test_template_directory_enumeration_is_bounded(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    templates = path / "chat_templates"; templates.mkdir()
    original = Path.iterdir
    yielded = []
    def limited(self):
        if self == templates:
            for i in range(1000):
                yielded.append(i)
                if i > 64: pytest.fail("enumerated beyond rejection boundary")
                yield templates / f"{i}.jinja"
        else:
            yield from original(self)
    monkeypatch.setattr(Path, "iterdir", limited)
    with pytest.raises(ValueError, match="too many"):
        LocalModelRuntime.inspect(str(path))
    assert len(yielded) == 65


def test_configured_relative_tokenizer_path_uses_checked_root_not_cwd(tmp_path, monkeypatch):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    alternate = path / "alternate.json"
    alternate.write_bytes((path / "tokenizer.json").read_bytes())
    file = path / "tokenizer_config.json"; data = json.loads(file.read_text())
    data["tokenizer_file"] = "alternate.json"; file.write_text(json.dumps(data))
    wrong = tmp_path / "cwd"; wrong.mkdir(); (wrong / "alternate.json").write_text("not a tokenizer; must never be consumed")
    monkeypatch.chdir(wrong)
    actual = transformers.AutoTokenizer.from_pretrained
    paths = []
    def checked(*args, **kwargs):
        paths.append(kwargs["tokenizer_file"])
        assert kwargs["tokenizer_file"] == str(alternate)
        return actual(*args, **kwargs)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", checked)
    assert LocalModelRuntime.inspect(str(path))["inspection"]["input_assets_loadable"]
    assert paths == [str(alternate)]


@pytest.mark.parametrize("change", [{"tokenizer_class": "WhisperTokenizer"}, {"tokenizer_class": []}, {"normalizer_file": "outside.json"}])
def test_unbounded_builtin_or_additional_file_override_is_rejected(tmp_path, monkeypatch, change):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    file = path / "tokenizer_config.json"; data = json.loads(file.read_text()); data.update(change); file.write_text(json.dumps(data))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("reject before parser"))
    with pytest.raises(ValueError, match="unsupported tokenizer"):
        LocalModelRuntime.inspect(str(path))


def test_bounded_local_snapshot_symlink_asset_remains_supported(tmp_path):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    file = path / "tokenizer.json"
    blob = tmp_path / "tokenizer-blob"; file.rename(blob)
    try: file.symlink_to(blob)
    except OSError: pytest.skip("host does not permit symlink creation")
    assert LocalModelRuntime.inspect(str(path))["inspection"]["input_assets_loadable"]


@pytest.mark.parametrize("failure", ["weight_budget", "device", "vram_option"])
def test_known_budget_or_option_failure_precedes_input_parser(tmp_path, monkeypatch, failure):
    path = tmp_path / "model"; local_checkpoint(path, "gpt2")
    runtime = LocalModelRuntime(); runtime.load_demo()
    model, binding = runtime.model, runtime.capture_binding()
    options = {"device": "cpu"}
    if failure == "weight_budget": runtime.max_model_bytes = 1
    elif failure == "device": options["device"] = "unsupported"
    else: options["max_vram_bytes"] = 0
    monkeypatch.setattr(runtime, "_prepare_inputs", lambda *args: pytest.fail("known invalid request must not parse input assets"))
    with pytest.raises(ValueError): runtime.load(str(path), **options)
    assert runtime.model is model and runtime.capture_binding() is binding
