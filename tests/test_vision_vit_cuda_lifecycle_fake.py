"""Model-free simulated admission and drain checks; never calls real CUDA."""
from types import SimpleNamespace

import pytest

from looking_glass import vision_vit_cuda_observer as observer


class FakeTensor:
    def reshape(self, _shape):
        return self

    def to(self, _device):
        return FakeTensor()


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=192, num_hidden_layers=12, num_labels=1000,
                                      patch_size=16, image_size=224, _attn_implementation="eager")

    def eval(self):
        return self

    def to(self, _device):
        return self

    def parameters(self):
        return iter(())


class FakeCuda:
    total = 8 * 1024 ** 3
    def __init__(self, fail_sync=None, fail_allocator=False):
        self.sync_calls = 0
        self.fail_sync = fail_sync
        self.fail_allocator = fail_allocator
        self.empty_calls = 0

    def is_initialized(self): return False
    def device_count(self): return 1
    def get_device_properties(self, _index):
        return SimpleNamespace(name="NVIDIA GeForce RTX 2070 SUPER", major=7, minor=5,
                               total_memory=self.total, uuid="fake")
    def get_allocator_backend(self):
        if self.fail_allocator:
            raise RuntimeError("fabricated allocator failure")
        return "native"
    def set_per_process_memory_fraction(self, fraction, _index): assert fraction == .25
    def get_per_process_memory_fraction(self, _index): return .25
    def reset_peak_memory_stats(self, _index): pass
    def mem_get_info(self, _index): return self.total, self.total
    def memory_allocated(self, _index): return 0
    def memory_reserved(self, _index): return 0
    def max_memory_allocated(self, _index): return 0
    def max_memory_reserved(self, _index): return 0
    def manual_seed(self, _seed): pass
    def empty_cache(self): self.empty_calls += 1
    def synchronize(self, _index):
        self.sync_calls += 1
        if self.sync_calls == self.fail_sync:
            raise RuntimeError("fabricated synchronize failure")


class FakeTorch:
    __version__ = "2.8.0+cu128"
    float32 = object()
    def __init__(self, fail_sync=None, fail_allocator=False):
        self.version = SimpleNamespace(cuda="12.8")
        self.cuda = FakeCuda(fail_sync, fail_allocator)
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False))
        self.random = SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda _seed: None))
        self.workspace_clears = 0
        self._C = SimpleNamespace(_cuda_clearCublasWorkspaces=self.clear_workspace)
        self.threads = self.interop = 0
        self.deterministic = False

    def clear_workspace(self): self.workspace_clears += 1
    def set_num_threads(self, value): self.threads = value
    def set_num_interop_threads(self, value): self.interop = value
    def get_num_threads(self): return self.threads
    def get_num_interop_threads(self): return self.interop
    def use_deterministic_algorithms(self, value): self.deterministic = value
    def are_deterministic_algorithms_enabled(self): return self.deterministic
    def is_autocast_enabled(self, _device): return False
    def get_float32_matmul_precision(self): return "highest"
    def frombuffer(self, _data, dtype):
        assert dtype is self.float32
        return FakeTensor()


def _fake_measure(monkeypatch, tmp_path, *, fail_sync=None, fail_allocator=False):
    from transformers import ViTForImageClassification

    torch = FakeTorch(fail_sync, fail_allocator)
    session, lease = object(), object()
    roots = dict(observer._ROOT)
    roots.update(session=session, lease=lease, budget=None, memory_lease=None, model=None, pixels=None,
                 observation=None, pending=False, submission=0, label=None, failure=None, diagnostic_failure=None)
    monkeypatch.setattr(observer, "_ROOT", roots)
    monkeypatch.setattr(observer, "_held", lambda *_args: None)
    monkeypatch.setattr(observer, "_runtime_uuid", lambda *_args: "fake")
    monkeypatch.setattr(observer, "_state_digest", lambda *_args: observer.STATE_SHA)
    monkeypatch.setattr(observer, "_unexpected_hooks", lambda *_args: False)
    monkeypatch.setattr(observer.OwnerSession, "verify_runtime_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ViTForImageClassification, "from_pretrained",
                        lambda *_args, **_kwargs: (FakeModel(), {}))
    zero_state, zero_logits, zero_request = bytes(197 * 192 * 4), bytes(1000 * 4), bytes(4 * 192 * 4)
    cpu_initial = zero_logits + zero_state + zero_state
    cpu_patched = bytearray(315808)
    reference = {"cat-pixels.f32": b"fake", "dog-pixels.f32": b"fake",
                 "cpu-forward-001.bin": cpu_initial, "cpu-forward-002.bin": cpu_initial,
                 "cpu-forward-024.bin": bytes(cpu_patched)}
    calls = []
    def fake_observe(_model, _pixels, *, label, index, synchronize, custody, **_kwargs):
        calls.append((index, label))
        custody.output = object()
        custody.captures = {"before_all": object()}
        synchronize()
        values = {"logits": zero_logits, "before_all": zero_state}
        if index in (3, 4, 5): values["after_all"] = zero_state
        if index in (4, 5): values["requested"] = zero_request
        return values, {"label": label, "dog_minus_cat_log_odds": observer._margin(zero_logits)}
    monkeypatch.setattr(observer, "observe", fake_observe)
    output = tmp_path / "owner"
    output.mkdir()
    return torch, roots, calls, lambda: observer._measure(session, lease, torch, tmp_path, reference, output)


def test_simulated_full_lifecycle_retains_commitment_through_cleanup(tmp_path, monkeypatch):
    torch, roots, calls, run = _fake_measure(monkeypatch, tmp_path)
    result = run()
    assert calls == list(enumerate(observer.LABELS, 1))
    assert result["forward_count"] == 6
    assert roots["pending"] is False
    assert result["cleanup_weakrefs"] == {"model": False, "cat_pixels": False, "dog_pixels": False}
    assert result["cleanup_memory"]["allocated_bytes"] == result["cleanup_memory"]["reserved_bytes"] == 0
    assert torch.cuda.empty_calls == 2 and torch.workspace_clears == 1
    assert roots["budget"].active_leases() == (roots["memory_lease"],)
    assert roots["memory_lease"].trace_bytes("gpu") == roots["memory_lease"].trace_bytes("host") == observer.TRACE


def test_simulated_failed_live_drain_keeps_roots_and_never_attempts_cleanup(tmp_path, monkeypatch):
    torch, roots, calls, run = _fake_measure(monkeypatch, tmp_path, fail_sync=12)
    with pytest.raises(RuntimeError, match="synchronize failure"):
        run()
    assert calls == [(1, "cat-clean")]
    assert roots["pending"] is True and roots["model"] is not None and roots["pixels"] is not None
    assert roots["observation"].output is not None and roots["observation"].captures is not None
    assert roots["budget"].active_leases() == (roots["memory_lease"],)
    assert torch.cuda.empty_calls == 0 and torch.workspace_clears == 0 and torch.cuda.sync_calls == 12


def test_simulated_failed_setup_keeps_cuda_completion_unverified(tmp_path, monkeypatch):
    torch, roots, calls, run = _fake_measure(monkeypatch, tmp_path, fail_allocator=True)
    with pytest.raises(RuntimeError, match="allocator failure"):
        run()
    assert roots["pending"] is True and roots["model"] is None and roots["budget"] is None
    assert not calls and torch.cuda.sync_calls == 0
    assert torch.cuda.empty_calls == 0 and torch.workspace_clears == 0


def test_owner_failure_receipt_never_claims_cleanup_attestation(tmp_path, monkeypatch):
    import json
    roots = dict(observer._ROOT)
    roots.update(entered=False, pending=False, session=None, lease=None, observation=None,
                 submission=0, label=None, failure=None, diagnostic_failure=None)
    monkeypatch.setattr(observer, "_ROOT", roots)
    monkeypatch.setattr(observer, "_startup", lambda: None)
    monkeypatch.setattr(observer, "_require_pristine", lambda: None)
    monkeypatch.setattr(observer, "_validate_workspace_spec", lambda *_args: None)
    monkeypatch.setattr(observer, "_validate_workspace_binding", lambda *_args: None)
    monkeypatch.setattr(observer, "_held", lambda *_args: None)
    monkeypatch.setattr(observer, "_source_identity", lambda *_args: ("a" * 40, {}))
    monkeypatch.setattr(observer, "_asset_inventory", lambda *_args: {})
    monkeypatch.setattr(observer, "_load_reference", lambda *_args: {})
    monkeypatch.setattr(observer.OwnerSession, "from_environment", lambda: SimpleNamespace(
        run_id="b" * 32, device_lease_path="fake"))
    monkeypatch.setattr(observer.DeviceLease, "acquire", lambda *_args, **_kwargs: object())
    monkeypatch.setenv("LOOKING_GLASS_VISION_REFERENCE", str(tmp_path))
    def fail_measure(*_args):
        roots["pending"] = True
        raise RuntimeError("fabricated setup failure")
    monkeypatch.setattr(observer, "_measure", fail_measure)
    assert observer.owner_main(tmp_path, tmp_path, "run", "a" * 40, "git") == 1
    failure = json.loads((tmp_path / "failure.json").read_bytes())
    assert failure["cuda_completion_unverified"] is True
    assert failure["cleanup_attested"] is False
    assert not (tmp_path / "final.json").exists()
