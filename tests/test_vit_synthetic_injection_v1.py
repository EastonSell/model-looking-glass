"""Only tiny random ViT models and fabricated tensors; no pretrained forward."""

import hashlib
import json
import subprocess
import sys

import pytest
import torch
from transformers import ViTConfig, ViTForImageClassification

from looking_glass.vit_synthetic_injection_v1 import (
    _distance, _plan_forward, choose, direction_bytes, fit_direction, fixed_random_choices,
    intervention_arms, paired_case, synthetic_pixels,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        config = ViTConfig(image_size=32, patch_size=16, num_channels=3,
                           hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=24, num_labels=7, hidden_dropout_prob=0,
                           attention_probs_dropout_prob=0)
        config._attn_implementation = "eager"
        return ViTForImageClassification(config).eval()


def test_injection_and_later_capture_are_bounded_and_one_site():
    model, pixels = tiny(), synthetic_pixels(11, 32)
    pair = paired_case(model, pixels, (1, 2, 3, 4), 3, "fabricated")
    captures = pair["injected"]["captures"]
    before, after = captures["embedding_before"], captures["embedding_after"]
    expected = before.clone()
    expected[0, 3, 7] += 4
    assert torch.equal(after.view(torch.int32), expected.view(torch.int32))
    assert pair["injected"]["injection_l2"] == 4
    assert pair["comparison"]["status"] == "compared"
    assert len(pair["comparison"]["comparisons"]) == 5
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


def test_selection_ties_and_fixed_random_control():
    result = choose((1.0, 0.0), [(4, (0.0, 1.0)), (2, (0.0, 1.0)), (3, (0.0, 1.0))])
    assert result["directional"] == result["magnitude"] == 2
    assert fixed_random_choices() == (166, 166, 52, 166)
    with pytest.raises(ValueError):
        choose((float("nan"),), [(2, (0.0,))])


def test_self_target_and_both_controls_land_without_touching_other_tokens():
    model, pixels = tiny(), synthetic_pixels(21, 32)
    pair = paired_case(model, pixels, (1, 2, 3, 4), 2, "fabricated")
    result = intervention_arms(model, pixels, pair, 2, 2, 1)
    for name, value in result["arms"].items():
        cap = value["forward"]["captures"]
        token = 1 if name == "off_target" else 2
        other = [index for index in range(5) if index != token]
        assert torch.equal(cap["block_before"][:, other].view(torch.int32),
                           cap["block_after"][:, other].view(torch.int32))
        assert value["dose"] == pytest.approx(_distance(cap["block_before"][0, token],
                                                        cap["block_after"][0, token]))
    assert result["arms"]["target"]["dose"] > 0
    assert result["arms"]["reflected"]["dose_matched"]
    assert result["arms"]["off_target"]["dose_matched"]
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


def test_discovery_direction_uses_only_fixed_discovery_labels():
    model = tiny()
    pairs = [paired_case(model, synthetic_pixels(seed, 32), (1, 2, 3, 4), token, f"d{seed}")
             for seed, token in ((11, 1), (12, 2), (13, 3), (14, 4))]
    direction = fit_direction(pairs, (1, 2, 3, 4))
    assert len(direction) == 16
    assert len(direction_bytes(direction)) == 16*8
    assert any(value != 0 for value in direction)


def test_failed_forward_cleans_hooks(monkeypatch):
    model, pixels = tiny(), synthetic_pixels(11, 32)
    block = model.vit.encoder.layer[1]
    def fail(*args, **kwargs):
        raise RuntimeError("fabricated failure")
    monkeypatch.setattr(block, "forward", fail)
    with pytest.raises(RuntimeError, match="fabricated failure"):
        _plan_forward(model, pixels, 1)
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


def test_injection_rejects_signed_zero_changes_outside_selected_coordinate(monkeypatch):
    model, pixels = tiny(), synthetic_pixels(11, 32)
    embeddings = model.vit.embeddings
    original = embeddings.forward
    def negative_zero(*args, **kwargs):
        result = original(*args, **kwargs).clone()
        result[0, 1, 8] = -0.0
        return result
    monkeypatch.setattr(embeddings, "forward", negative_zero)
    with pytest.raises(RuntimeError, match="invalid injection landing"):
        _plan_forward(model, pixels, 1)
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


def test_complete_fabricated_study_preserves_fit_and_indexed_witnesses(tmp_path, monkeypatch):
    from examples import vit_synthetic_injection_cpu as cli
    monkeypatch.setattr(cli, "FREE", 0)
    sink = cli.Evidence(tmp_path)
    sink.write("started.json", {"fabricated": True})
    policy = (32, 16, (1, 2, 3, 4), (11, 12, 13, 14, 21, 22, 23, 24),
              (1, 2, 3, 4, 3, 4, 1, 2), (4, 4, 2, 4))
    result = cli.run(tiny(), sink, {"fabricated": True}, _fabricated_policy=policy)
    sink.close()
    raw = (tmp_path / "tensors.bin").read_bytes()
    index = json.loads((tmp_path / "tensor-index.json").read_bytes())
    assert len(raw) == index["tensor_bytes"]
    for entry in index["entries"]:
        chunk = raw[entry["offset"]:entry["offset"]+entry["length"]]
        assert len(chunk) == entry["length"]
        assert hashlib.sha256(chunk).hexdigest() == entry["sha256"]
    fit = (tmp_path / "discovery-freeze.json").read_bytes()
    assert result["discovery_freeze_sha256"] == hashlib.sha256(fit).hexdigest()
    assert all(row["fitted_record_sha256"] == result["discovery_freeze_sha256"] for row in result["heldout"])
    assert len(result["heldout"]) == 4


def test_zero_fitted_direction_retains_all_cases_without_directional_claim(tmp_path, monkeypatch):
    from examples import vit_synthetic_injection_cpu as cli
    monkeypatch.setattr(cli, "FREE", 0)
    monkeypatch.setattr(cli, "fit_direction", lambda *_: (0.0,) * 16)
    sink = cli.Evidence(tmp_path)
    policy = (32, 16, (1, 2, 3, 4), (11, 12, 13, 14, 21, 22, 23, 24),
              (1, 2, 3, 4, 3, 4, 1, 2), (4, 4, 2, 4))
    result = cli.run(tiny(), sink, {"fabricated": True}, _fabricated_policy=policy)
    sink.close()
    assert result["selection"]["prediction"] == "ineligible"
    assert result["selection"]["directional_hits"] is None
    assert all(row["selection_status"] == "ineligible_zero_direction" for row in result["heldout"])
    assert result["restoration"]["prediction"] == "inconclusive"


def test_loading_diagnostics_require_all_empty_lists():
    from examples.vit_synthetic_injection_cpu import verify_loading_info
    clean = {name: [] for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    assert verify_loading_info(clean) == clean
    with pytest.raises(RuntimeError):
        verify_loading_info({key: value for key, value in clean.items() if key != "error_msgs"})
    with pytest.raises(RuntimeError):
        verify_loading_info({**clean, "missing_keys": ["missing.weight"]})


def test_current_transformers_reports_complete_loading_diagnostics(tmp_path):
    from examples.vit_synthetic_injection_cpu import verify_loading_info
    tiny().save_pretrained(tmp_path)
    _, info = ViTForImageClassification.from_pretrained(tmp_path, local_files_only=True,
                                                         output_loading_info=True, attn_implementation="eager")
    assert verify_loading_info(info) == {key: [] for key in
        ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}


def test_short_write_and_corrupted_tensor_readback_are_rejected(tmp_path, monkeypatch):
    from examples import vit_synthetic_injection_cpu as cli
    class Short:
        def write(self, raw):
            return 0
    with pytest.raises(OSError, match="short evidence write"):
        cli.checked_write(Short(), b"data")
    monkeypatch.setattr(cli, "FREE", 0)
    sink = cli.Evidence(tmp_path)
    sink.tensor("test", torch.tensor([1.0], dtype=torch.float32))
    with (tmp_path / "tensors.bin").open("r+b") as stream:
        stream.write(b"x")
    with pytest.raises(OSError, match="differs on readback"):
        sink.close()


def test_bootstrap_requires_isolation_and_fresh_empty_prefix(tmp_path):
    from examples import vit_synthetic_injection_cpu as cli
    bootstrap = cli.ROOT / "examples/vit_synthetic_injection_bootstrap.py"
    prefix = tmp_path / "absent-bytecode-prefix"
    command = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={prefix}", str(bootstrap), "--help"]
    ok = subprocess.run(command, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert not prefix.exists()
    failed = subprocess.run([sys.executable, "-B", str(bootstrap), "--help"], capture_output=True, text=True)
    assert failed.returncode != 0
    assert "fresh absent" in failed.stderr
