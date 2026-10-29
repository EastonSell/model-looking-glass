"""Rollback and alias checks for a fixed row mutation, without a model."""

import numpy as np
import pytest
import torch

import flan_weight_edit_guard as guard


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Embedding(7, 4)
        self.encoder = torch.nn.Module()
        self.encoder.embed_tokens = self.shared
        self.decoder = torch.nn.Module()
        self.decoder.embed_tokens = self.shared
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        self.register_buffer("transient", torch.tensor([0.0]), persistent=False)


def test_one_shared_row_changes_and_exception_restores_bitwise():
    model = SmallModel()
    before = guard.state_hash(model)
    original = model.shared.weight[2].detach().clone()
    candidate = model.shared.weight[3].detach().numpy().copy()
    probes = []
    def probe():
        probes.append(True)
        return b"clean-logits"
    with pytest.raises(guard.RestorationUnprobed) as failure:
        with guard.temporary_row(model, model.shared.weight, 2, candidate,
                                 expected_state=before, expected_probe=b"clean-logits", restored_probe=probe):
            assert torch.equal(model.encoder.embed_tokens.weight[2], torch.from_numpy(candidate))
            assert model.decoder.embed_tokens.weight is model.shared.weight
            raise RuntimeError("study failed")
    assert str(failure.value.__cause__) == "study failed"
    assert torch.equal(model.shared.weight[2], original)
    assert guard.state_hash(model) == before
    assert probes == []


def test_unintended_other_parameter_or_nonpersistent_buffer_change_is_detected():
    model = SmallModel()
    before = guard.state_hash(model)
    candidate = np.zeros(4, dtype=np.float32)
    with pytest.raises(guard.RestorationError):
        with guard.temporary_row(model, model.shared.weight, 2, candidate,
                                 expected_state=before, expected_probe=b"clean-logits", restored_probe=lambda: b"clean-logits"):
            model.transient.add_(1)


def test_stale_expected_state_rejects_before_any_row_change():
    model = SmallModel()
    before = model.shared.weight.detach().clone()
    with pytest.raises(ValueError):
        with guard.temporary_row(model, model.shared.weight, 2, np.zeros(4, dtype=np.float32),
                                 expected_state="0" * 64, expected_probe=b"clean-logits", restored_probe=lambda: b"clean-logits"):
            pytest.fail("mutation must not be entered")
    assert torch.equal(model.shared.weight, before)


def test_probe_mutating_buffer_or_returning_changed_logits_is_rejected():
    for changes_buffer in (True, False):
        model = SmallModel()
        before = guard.state_hash(model)
        def probe():
            if changes_buffer:
                model.transient.add_(1)
                return b"clean-logits"
            return b"changed-logits"
        with pytest.raises(guard.RestorationError):
            with guard.temporary_row(model, model.shared.weight, 2, np.zeros(4, dtype=np.float32),
                                     expected_state=before, expected_probe=b"clean-logits", restored_probe=probe):
                pass


def test_probe_changing_shared_alias_is_rejected_even_if_values_match():
    model = SmallModel()
    before = guard.state_hash(model)
    def probe():
        model.encoder.embed_tokens = torch.nn.Embedding.from_pretrained(model.shared.weight.detach().clone())
        return b"clean-logits"
    with pytest.raises(guard.RestorationError):
        with guard.temporary_row(model, model.shared.weight, 2, np.zeros(4, dtype=np.float32),
                                 expected_state=before, expected_probe=b"clean-logits", restored_probe=probe):
            pass
