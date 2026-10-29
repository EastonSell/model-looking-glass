"""Fabricated CPU-only checks of the exposed-pair CUDA observer."""
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from looking_glass import vision_vit_cuda_observer as observer


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.nn.Module()
        self.vit.encoder = FakeEncoder()

    def forward(self, pixel_values, return_dict=True):
        base = torch.arange(197 * 192, dtype=torch.float32).reshape(1, 197, 192) / 10000
        value = base + pixel_values.reshape(-1)[0]
        for layer in self.vit.encoder.layer:
            value = layer(value)
        score = value[0, list(observer.TOKENS)].sum()
        logits = torch.arange(1000, dtype=torch.float32).reshape(1, 1000) / 1000 + score / 1000
        return SimpleNamespace(logits=logits)


def _tensor(data, shape):
    return torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(shape)


def _run_fake(directory):
    model = FakeModel().eval()
    pixels = {"cat": torch.tensor([0.0]), "dog": torch.tensor([1.0])}
    rows, outputs = [], []
    for i, label in enumerate(observer.LABELS, 1):
        replacement = None
        if i in (4, 5):
            state = outputs[1]["before_all"]
            donor = b"".join(state[t * 768:(t + 1) * 768] for t in observer.TOKENS)
            replacement = _tensor(donor, (4, 192))
        out, row = observer.observe(model, pixels["dog" if i == 2 else "cat"], label=label,
            output=directory, index=i, self_patch=i == 3, replacement=replacement)
        outputs.append(out)
        rows.append(row)
    cpu_patched = bytearray(315808)
    cpu_patched[:4000] = outputs[4]["logits"]
    cpu_patched[312736:] = outputs[4]["requested"]
    reference = {"cpu-forward-024.bin": bytes(cpu_patched),
                 "cpu-forward-001.bin": outputs[0]["logits"] + outputs[0]["before_all"] + bytes(151296),
                 "cpu-forward-002.bin": outputs[1]["logits"] + outputs[1]["before_all"] + bytes(151296)}
    total = 8 * 1024 ** 3
    cap = total // 4
    memory = {"device_free_bytes": total, "device_total_bytes": total, "allocated_bytes": 0,
              "reserved_bytes": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    stages = (["before_model_load", "before_model_copy", "after_model_copy"]
              + [f"after_forward_{i}" for i in range(1, 7)] + ["after_disposal", "after_cleanup"])
    for i, stage in enumerate(stages, 1):
        observer.write_json_new(directory / f"stage-{i:02d}.json",
            {"stage": stage, "memory": memory, "peak_scope": "since_pre_model_reset"})
    comparisons, margins = observer._descriptive(outputs, reference, rows)
    measurement = {"forward_count": 6, "forwards": rows, "reference_sha256": observer.REFERENCE_SHA,
        "model_state_sha256": observer.STATE_SHA, "same_gpu_self_repeat_and_landing": True,
        "runtime": {"torch": "2.8.0+cu128", "transformers": "4.57.3", "device": "cpu",
                    "cuda_build": "12.8", "python": "3.11.15", "seed": 17,
                    "device_name": "NVIDIA GeForce RTX 2070 SUPER", "compute_capability": [7, 5],
                    "device_total_bytes": total, "dtype": "float32", "attention": "eager",
                    "threads": 1, "interop_threads": 1, "deterministic_algorithms": True,
                    "matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False,
                    "cudnn_deterministic": True, "autocast_enabled": False,
                    "float32_matmul_precision": "highest", "allocator": "native",
                    "allocator_cap_bytes": cap, "commitment_bytes": cap + observer.HEADROOM,
                    "gpu_trace_claim_bytes": observer.TRACE, "host_trace_claim_bytes": observer.TRACE},
        "cleanup_memory": memory, "initial_memory": memory, "stages": stages,
        "comparisons": comparisons, "margins": margins,
        "cleanup_weakrefs": {"model": False, "cat_pixels": False, "dog_pixels": False}}
    numeric_names = ("threads", "interop_threads", "deterministic_algorithms", "matmul_tf32", "cudnn_tf32",
                     "cudnn_benchmark", "cudnn_deterministic", "autocast_enabled", "float32_matmul_precision")
    measurement["numeric_settings_after"] = {key: measurement["runtime"][key] for key in numeric_names}
    return reference, measurement, outputs


def test_fake_full_sequence_and_closed_replay(tmp_path):
    reference, measurement, outputs = _run_fake(tmp_path)
    assert observer._validate_saved(tmp_path, measurement, reference) == measurement["forwards"]
    assert outputs[0]["logits"] == outputs[2]["logits"] == outputs[5]["logits"]
    assert outputs[3]["logits"] != outputs[0]["logits"]
    assert outputs[3]["logits"] == outputs[4]["logits"]


def test_rehashed_wrong_requested_donor_is_rejected(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    row_path = tmp_path / "forward-004.json"
    row = json.loads(row_path.read_bytes())
    data_path = tmp_path / row["file"]
    data = bytearray(data_path.read_bytes())
    # Change the same token in the requested and landed tensors and update all
    # local hashes: the independent donor-to-dog witness check still rejects it.
    data[row["layout"]["requested"]["offset"]] ^= 1
    data[row["layout"]["after_0"]["offset"]] ^= 1
    data[row["layout"]["after_all"]["offset"] + observer.TOKENS[0] * 768] ^= 1
    data_path.write_bytes(data)
    row["sha256"] = observer._sha(data)
    row_path.write_text(json.dumps(row), encoding="utf-8")
    measurement["forwards"][3] = row
    with pytest.raises(ValueError, match="donor bytes"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_rehashed_untouched_state_corruption_is_rejected(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    row_path = tmp_path / "forward-005.json"
    row = json.loads(row_path.read_bytes())
    data_path = tmp_path / row["file"]
    data = bytearray(data_path.read_bytes())
    data[row["layout"]["after_all"]["offset"]] ^= 1  # CLS is not selected.
    data_path.write_bytes(data)
    row["sha256"] = observer._sha(data)
    row_path.write_text(json.dumps(row), encoding="utf-8")
    measurement["forwards"][4] = row
    with pytest.raises(ValueError, match="untouched"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_missing_forward_is_not_complete(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    (tmp_path / "forward-006.bin").unlink()
    with pytest.raises(ValueError, match="inventory"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_extreme_finite_equal_logits_preserve_class_group_margin():
    data = torch.full((1000,), 1e20, dtype=torch.float32).numpy().tobytes()
    assert observer._margin(data) == pytest.approx(math.log(118 / 5), abs=1e-14)


def test_incomplete_return_does_not_create_forward_metadata(tmp_path):
    class Raises(FakeModel):
        def forward(self, pixel_values, return_dict=True):
            super().forward(pixel_values, return_dict)
            raise RuntimeError("fabricated interrupted forward")
    custody = observer.ObservationCustody()
    with pytest.raises(RuntimeError, match="interrupted"):
        observer.observe(Raises().eval(), torch.tensor([0.0]), label="cat-clean", output=tmp_path,
                         index=1, custody=custody)
    assert not (tmp_path / "forward-001.json").exists()
    assert custody.plan is not None


def test_nonfinite_logits_leave_raw_evidence_and_no_success_record(tmp_path):
    class Nonfinite(FakeModel):
        def forward(self, pixel_values, return_dict=True):
            result = super().forward(pixel_values, return_dict)
            result.logits[0, 0] = float("inf")
            return result
    with pytest.raises(ValueError, match="nonfinite"):
        observer.observe(Nonfinite().eval(), torch.tensor([0.0]), label="cat-clean", output=tmp_path, index=1)
    assert (tmp_path / "forward-001.bin").is_file()
    assert not (tmp_path / "forward-001.json").exists()


def test_rehashed_metadata_tampering_is_rejected(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    path = tmp_path / "forward-003.json"
    row = json.loads(path.read_bytes())
    row["metadata"]["steps"][2]["source_capture"] = None
    path.write_text(json.dumps(row), encoding="utf-8")
    measurement["forwards"][2] = row
    with pytest.raises(ValueError, match="action metadata"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_failed_completion_retains_output_and_capture_roots(tmp_path):
    custody = observer.ObservationCustody()
    def broken_synchronize():
        raise RuntimeError("fabricated completion failure")
    with pytest.raises(RuntimeError, match="completion failure"):
        observer.observe(FakeModel().eval(), torch.tensor([0.0]), label="cat-clean", output=tmp_path,
                         index=1, custody=custody, synchronize=broken_synchronize)
    assert custody.output is not None
    assert "before_all" in custody.captures
    assert not list(tmp_path.iterdir())


def test_completed_wrong_shape_retains_logits_before_validation(tmp_path):
    class WrongShape(FakeModel):
        def forward(self, pixel_values, return_dict=True):
            result = super().forward(pixel_values, return_dict)
            result.logits = result.logits[:, :999]
            return result
    with pytest.raises(ValueError, match="shape"):
        observer.observe(WrongShape().eval(), torch.tensor([0.0]), label="cat-clean", output=tmp_path, index=1)
    receipt = json.loads((tmp_path / "return-001.json").read_bytes())
    raw = (tmp_path / "return-001.bin").read_bytes()
    assert receipt["shape"] == [1, 999] and receipt["sha256"] == observer._sha(raw)
    assert not (tmp_path / "forward-001.json").exists()


def test_capture_getter_error_retains_completed_logits(tmp_path, monkeypatch):
    from looking_glass.intervention_plan import ActivationPlan
    def fail_getter(_self):
        raise RuntimeError("fabricated capture retrieval failure")
    monkeypatch.setattr(ActivationPlan, "captures", property(fail_getter))
    with pytest.raises(RuntimeError, match="capture retrieval"):
        observer.observe(FakeModel().eval(), torch.tensor([0.0]), label="cat-clean", output=tmp_path, index=1)
    assert (tmp_path / "return-001.bin").stat().st_size == 4000
    assert (tmp_path / "return-001.json").is_file()
    assert not (tmp_path / "forward-001.json").exists()


@pytest.mark.parametrize("exit_code,timed_out,failure,missing_owner,expected", [
    (0, False, False, False, "complete"), (1, False, False, False, "incomplete"),
    (0, True, False, False, "incomplete"), (False, False, False, False, "incomplete"),
    (0, False, True, False, "incomplete"), (0, False, False, True, "incomplete"),
])
def test_controller_requires_reaped_zero_and_matching_closed_records(
        tmp_path, monkeypatch, exit_code, timed_out, failure, missing_owner, expected):
    fixture = tmp_path / "fake-records"
    fixture.mkdir()
    reference, measurement, _ = _run_fake(fixture)
    # Re-label the fabricated event device to exercise controller validation;
    # all values remain fabricated CPU tensors and no CUDA call is made.
    measurement["runtime"]["device"] = "cuda:0"
    for index, row in enumerate(measurement["forwards"], 1):
        for event in row["metadata"]["steps"]:
            event["device"] = "cuda:0"
        (fixture / f"forward-{index:03d}.json").write_text(json.dumps(row), encoding="utf-8")
    fake_commit = "a" * 40
    monkeypatch.setenv("LOOKING_GLASS_VISION_GIT_EXE", str(tmp_path / "git.exe"))
    monkeypatch.setenv("LOOKING_GLASS_VISION_SOURCE_COMMIT", fake_commit)
    monkeypatch.setattr(observer, "_startup", lambda: None)
    monkeypatch.setattr(observer, "_require_pristine", lambda: None)
    monkeypatch.setattr(observer, "_workspace_root", lambda *args: tmp_path)
    monkeypatch.setattr(observer, "_source_identity", lambda *args: (fake_commit, {"fake": "digest"}))
    monkeypatch.setattr(observer, "_asset_inventory", lambda *args: {"fake": "digest"})
    monkeypatch.setattr(observer, "_load_reference", lambda *args: reference)
    snapshot = {"timestamp_utc": "2026-10-29T10:00:00Z"}
    monkeypatch.setattr(observer, "_smi_snapshot", lambda *args: ("GPU-fake", snapshot))
    monkeypatch.setattr(observer, "resolve_device_identity", lambda *args, **kwargs:
                        SimpleNamespace(lock_path=tmp_path / "lease", marker_path=tmp_path / "marker"))
    monkeypatch.setattr(observer, "_owner_presence_after_exit", lambda *args:
                        {"query": "ok", "owner_pid_listed": False})
    destination = tmp_path / "attempt"
    def fake_supervisor(command, **kwargs):
        assert kwargs["require_zero_exit"] is True
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-fake"
        started = json.loads((destination / "started.json").read_bytes())
        for path in fixture.iterdir():
            shutil.copyfile(path, destination / path.name)
        run_id = started["run_id"]
        if not missing_owner:
            observer.write_json_new(destination / "owner.json", {"schema": "vision-vit-cuda-owner-v1",
                "run_id": run_id, "supervisor_run_id": "supervised-fake", "pid": 123,
                "source_commit": fake_commit})
        candidate = {"schema": "vision-vit-cuda-candidate-v1", "run_id": run_id,
            "supervisor_run_id": "supervised-fake", "pid": 123, "source_commit": fake_commit,
            "measurement": measurement}
        observer.write_json_new(destination / "candidate.json", candidate)
        candidate_sha = observer._file_identity(destination / "candidate.json", observer.MIB)[1]
        observer.write_json_new(destination / "final.json", {"schema": "vision-vit-cuda-final-v1",
            "run_id": run_id, "supervisor_run_id": "supervised-fake", "pid": 123,
            "source_commit": fake_commit, "candidate_sha256": candidate_sha, "active_memory_leases": 0})
        if failure:
            observer.write_json_new(destination / "failure.json", {"status": "incomplete"})
        return SimpleNamespace(run_id="supervised-fake", pid=123, exit_code=exit_code, clean=True,
                               timed_out=timed_out, reason="fabricated")
    monkeypatch.setattr(observer, "run_supervised", fake_supervisor)
    result = observer.controller_main(tmp_path, tmp_path, tmp_path, destination, tmp_path, "fake-smi")
    assert result["status"] == expected


@pytest.mark.parametrize("key,value", [
    ("threads", 32), ("threads", True), ("cudnn_benchmark", True), ("cudnn_deterministic", False),
    ("allocator", "other"), ("gpu_trace_claim_bytes", 0), ("autocast_enabled", 0),
])
def test_conflicting_runtime_never_promotes(tmp_path, key, value):
    reference, measurement, _ = _run_fake(tmp_path)
    measurement["runtime"][key] = value
    with pytest.raises(ValueError, match="precision or cleanup"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_missing_cleanup_witnesses_never_promotes(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    measurement["cleanup_weakrefs"] = {}
    with pytest.raises(ValueError, match="precision or cleanup"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_extra_forward_never_promotes(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    (tmp_path / "forward-007.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_boolean_submissions_never_promote(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    path = tmp_path / "forward-001.json"
    row = json.loads(path.read_bytes())
    row["submission"] = True
    path.write_text(json.dumps(row), encoding="utf-8")
    measurement["forwards"][0] = row
    receipt_path = tmp_path / "return-001.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["submission"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="forward identity"):
        observer._validate_saved(tmp_path, measurement, reference)


def test_changed_post_inference_settings_never_promote(tmp_path):
    reference, measurement, _ = _run_fake(tmp_path)
    measurement["numeric_settings_after"]["matmul_tf32"] = True
    with pytest.raises(ValueError, match="post-inference numerical settings"):
        observer._validate_saved(tmp_path, measurement, reference)
