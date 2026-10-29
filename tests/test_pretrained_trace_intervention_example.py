"""Exercise the example using a tiny random model, without pretrained weights."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel


def test_demo_records_known_edit_and_control_on_random_cpu_model():
    path = Path(__file__).resolve().parents[1] / "examples" / "pretrained_trace_intervention.py"
    spec = importlib.util.spec_from_file_location("pretrained_trace_intervention", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    config = GPT2Config(vocab_size=32, n_positions=8, n_embd=16, n_layer=2,
                        n_head=2, use_cache=False, attn_pdrop=0, resid_pdrop=0,
                        embd_pdrop=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = GPT2LMHeadModel(config).eval()
    inputs = {"input_ids": torch.tensor([[3, 4]]),
              "attention_mask": torch.ones((1, 2), dtype=torch.long)}
    summary = example.run_demo(model, inputs, lambda token_id: str(token_id))

    assert all(summary["controls"].values())
    assert [event["kind"] for event in summary["trace"]["events"]] == ["attn", "mlp", "block"]
    assert summary["trace"]["captured_tensor_bytes"] == 3 * 2 * 16 * 4
    assert summary["intervention"]["observed_displacement_l2"] == pytest.approx(5.0, abs=1e-6)
    assert summary["intervention"]["max_absolute_logit_change"] > 0
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())
