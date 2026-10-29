"""Synthetic pre-evaluation handoff checks; no pretrained model is loaded."""

import copy
from types import SimpleNamespace

import pytest
import torch

from looking_glass import targeted_edit_cpu_v1 as v1
from looking_glass import targeted_edit_cpu_v2 as v2
from looking_glass import targeted_edit_v2_edits as edits
from looking_glass import targeted_edit_v2_gate as gate


@pytest.fixture
def fake_handoff(monkeypatch):
    assert torch.set_flush_denormal(False)
    monkeypatch.setattr(v1, "WEIGHT_SHAPE", (16, 12))
    generator = torch.Generator(device="cpu").manual_seed(11)
    original = torch.randn(16, 12, generator=generator, dtype=torch.float32)
    left = torch.randn(16, generator=generator, dtype=torch.float32)
    right = torch.randn(12, generator=generator, dtype=torch.float32)
    gradient = v1._scale_outer(left, right, 0.01 * v1._norm(original))
    generator = torch.Generator(device="cpu").manual_seed(v1.SEED)
    random = v1._scale_outer(torch.randn(16, generator=generator), torch.randn(12, generator=generator),
                             0.01 * v1._norm(original))
    environment = {key: f"runtime-{key}" for key in edits.ENVIRONMENT_FIELDS}
    environment.update(intraop_threads=1, interop_threads=1, deterministic_algorithms=True,
                       mkldnn_enabled=True, cuda_initialized=False, omp_num_threads=None, mkl_num_threads=None)
    runtime = {"float32_matmul_precision": "highest", "mkldnn_deterministic": False,
               "deterministic_warn_only": False, "intraop_threads": 1, "interop_threads": 1,
               "deterministic_algorithms": True, "mkldnn_enabled": True, "cpu_autocast_enabled": False,
               "inference_mode_enabled": False, "cuda_initialized": False, "cuda_visibility": "-1",
               "hf_hub_offline": "1", "transformers_offline": "1", "omp_num_threads": None,
               "mkl_num_threads": None}
    model = {"model_id": "synthetic-only"}
    discovery = {"discovery_logit_sha256": "3" * 64, "weight_norm": v1._norm(original)}
    state = "4" * 64
    reference = {"version": v1.VERSION, "site": v1.SITE, "site_shape": [16, 12],
                 "model": model, "environment": environment, "discovery": discovery,
                 "model_state_sha256": state, "weight_sha256": v1._tensor_sha(original),
                 "candidates": v1._candidate_attestation(original, {"gradient": gradient, "random": random})}
    raw = v1._canonical(reference)
    monkeypatch.setattr(edits, "V1_PRECOMMIT_SHA256", v1._sha(raw))
    args = {"reference_bytes": raw, "environment": environment, "model_metadata": model,
            "model_state_sha256": state, "original": original, "v1_gradient": gradient,
            "v1_random": random, "discovery": discovery}
    compatibility = edits.verify_v1_compatibility(**args)
    args["buffer_state_sha256"] = "f" * 64
    args["runtime_settings"] = runtime
    grid = {name: edits.attest(original, edits.construct_dose(original, gradient, dose), dose)
            for name, dose in v2.GRID}
    updates, controls = edits.construct_controls(original, gradient, 0.001)
    selection = {"selected_condition": "dose-0001", "selected_relative_norm": 0.001,
                 "development_qualified": True, "point_prediction": 2.0, "sign_prediction": "positive"}
    manifest = {"version": v2.VERSION, "source_commit": "a" * 40,
                "v1_compatibility": {"precommit_sha256": v1._sha(raw), "model": model,
                                     "environment": environment, "model_state_sha256": state,
                                     "weight_sha256": reference["weight_sha256"],
                                     "discovery": discovery, "candidates": reference["candidates"]}}
    public = {"known_row_collisions": 0, "private_split_check_performed": False}
    custody = {"status": "matching_committed_custody_attestation", "receipt_sha256": "9" * 64,
               "receipt_commit": "8" * 40, "observations": [{"observed_utc": "2026-10-29T03:29:00+00:00"}],
               "public_inventory": {"observed_utc": "2026-10-29T03:29:00+00:00"},
               "complete_history_certified": False}
    metadata = {"version": v2.VERSION, "phase": "development", "run_id": "b" * 32,
                "manifest_sha256": "c" * 64, "source_commit": "a" * 40, "model": model,
                "environment": environment, "exposure": public, "custody": custody,
                "compatibility": compatibility,
                "grid": grid, "controls": controls, "buffer_state_sha256": "f" * 64, "runtime_settings": runtime,
                "analysis_denormal_probe_before": copy.deepcopy(gate.evidence.DENORMAL_PROBE),
                "analysis_denormal_probe_after": copy.deepcopy(gate.evidence.DENORMAL_PROBE),
                "supervision": {"owner": {"run_id": "d" * 32, "pid": 123},
                                "supervisor": {"run_id": "d" * 32, "pid": 123,
                                               "exit_code": 0, "clean": True, "timed_out": False, "reaped": True}}}
    recorded = {"status": "matching_committed_selection", "source_commit": "a" * 40,
                "manifest_sha256": "c" * 64, "started_utc": "2026-10-29T03:30:00+00:00",
                "selection": selection, "metadata": metadata}
    monkeypatch.setattr(gate.manifest_module, "verify_manifest", lambda *a, **kw: manifest)
    monkeypatch.setattr(gate.exposure, "verify_public_reservations", lambda *a, **kw: public)
    monkeypatch.setattr(gate.custody, "verify_committed_receipt", lambda *a, **kw: custody)
    monkeypatch.setattr(gate.evidence, "verify_committed_development", lambda *a, **kw: recorded)
    monkeypatch.setattr(gate.evidence, "_require_commit_object", lambda *a, **kw: None)
    monkeypatch.setattr(gate.evidence, "_git", lambda _repo, *args, **kw:
                        SimpleNamespace(returncode=0, stdout=b""))
    call = {"manifest_path": "frozen.json", "manifest_commit": "d" * 40,
            "manifest_sha256": "c" * 64, "custody_path": "custody.json", "custody_commit": "8" * 40,
            "custody_sha256": "9" * 64, "development_dir": "development",
            "selection_commit": "e" * 40, "terminal_sha256": "f" * 64}
    return {"call": call, "args": args, "manifest": manifest, "recorded": recorded,
            "public": public, "updates": updates}


def test_snapshot_does_not_share_mutable_records_and_reconstructs_controls(fake_handoff):
    snapshot = gate.prepare_committed_selection("repo", **fake_handoff["call"])
    saved_selection = snapshot.selection_bytes
    fake_handoff["recorded"]["selection"]["point_prediction"] = 99.0
    fake_handoff["recorded"]["metadata"]["grid"].clear()
    fake_handoff["manifest"]["version"] = "mutated"
    assert snapshot.selection_bytes == saved_selection
    assert v1._decode_document(snapshot.selection_bytes)["point_prediction"] == 2.0
    updates = gate.reconstruct_selected_controls(snapshot, **fake_handoff["args"])
    assert all(torch.equal(updates[key].view(torch.int32), fake_handoff["updates"][key].view(torch.int32))
               for key in updates)


@pytest.mark.parametrize("mutation", ["status", "source", "manifest", "model", "runtime", "exposure", "custody",
                                      "compatibility", "grid", "dose_norm", "selected", "control_norm", "supervision",
                                      "buffer_digest", "analysis_probe"])
def test_handoff_rejects_inconsistent_committed_metadata(fake_handoff, mutation):
    record = fake_handoff["recorded"]
    metadata = record["metadata"]
    if mutation == "status":
        record["status"] = "matching_saved_selection"
    elif mutation == "source":
        record["source_commit"] = "0" * 40
    elif mutation == "manifest":
        record["manifest_sha256"] = "0" * 64
    elif mutation == "model":
        metadata["model"] = {"model_id": "other"}
    elif mutation == "runtime":
        metadata["environment"] = {**metadata["environment"], "torch_version": "other"}
    elif mutation == "exposure":
        metadata["exposure"] = {"known_row_collisions": 1, "private_split_check_performed": False}
    elif mutation == "custody":
        metadata["custody"] = {"status": "unverified"}
    elif mutation == "compatibility":
        metadata["compatibility"] = {**metadata["compatibility"], "weight_sha256": "0" * 64}
    elif mutation == "grid":
        metadata["grid"] = {key: value for key, value in metadata["grid"].items() if key != "dose-0004"}
    elif mutation == "dose_norm":
        metadata["grid"]["dose-0004"]["planned_norm"] *= 2
    elif mutation == "selected":
        metadata["controls"]["selected"]["intended_delta_sha256"] = "0" * 64
    elif mutation == "control_norm":
        metadata["controls"]["random"]["realized_norm"] *= 2
    elif mutation == "supervision":
        metadata["supervision"]["supervisor"]["clean"] = False
    elif mutation == "buffer_digest":
        metadata["buffer_state_sha256"] = "invalid"
    else:
        metadata["analysis_denormal_probe_after"]["subnormal_times_one_bits"] = 0
    with pytest.raises(v1.StudyError):
        gate.prepare_committed_selection("repo", **fake_handoff["call"])


def test_fresh_reconstruction_rejects_changed_control_digest(fake_handoff):
    snapshot = gate.prepare_committed_selection("repo", **fake_handoff["call"])
    metadata = v1._decode_document(snapshot.metadata_bytes)
    metadata["controls"]["random"]["intended_delta_sha256"] = "0" * 64
    altered = gate.SelectionSnapshot(snapshot.manifest_bytes, snapshot.selection_bytes,
                                     v1._canonical(metadata), snapshot.selection_commit, snapshot.terminal_sha256)
    with pytest.raises(v1.StudyError):
        gate.reconstruct_selected_controls(altered, **fake_handoff["args"])


def test_fresh_reconstruction_rejects_changed_registered_buffers(fake_handoff):
    snapshot = gate.prepare_committed_selection("repo", **fake_handoff["call"])
    args = {**fake_handoff["args"], "buffer_state_sha256": "0" * 64}
    with pytest.raises(v1.StudyError, match="buffer state differs"):
        gate.reconstruct_selected_controls(snapshot, **args)


def test_fresh_reconstruction_rejects_changed_supplemental_runtime(fake_handoff):
    snapshot = gate.prepare_committed_selection("repo", **fake_handoff["call"])
    settings = {**fake_handoff["args"]["runtime_settings"], "float32_matmul_precision": "medium"}
    args = {**fake_handoff["args"], "runtime_settings": settings}
    with pytest.raises(v1.StudyError, match="runtime settings differ"):
        gate.reconstruct_selected_controls(snapshot, **args)


def test_custody_commit_must_precede_selection_and_any_development_artifacts(fake_handoff, monkeypatch):
    arguments = dict(fake_handoff["call"])
    arguments["custody_commit"] = arguments["selection_commit"]
    with pytest.raises(v1.StudyError):
        gate.prepare_committed_selection("repo", **arguments)
    monkeypatch.setattr(gate.evidence, "_git", lambda _repo, *args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout=b"present" if args[0] == "ls-tree" else b""))
    with pytest.raises(v1.StudyError):
        gate.prepare_committed_selection("repo", **fake_handoff["call"])


@pytest.mark.parametrize("source", ["lane", "public_inventory"])
def test_custody_timestamp_after_recorded_development_start_fails(fake_handoff, source):
    custody = fake_handoff["recorded"]["metadata"]["custody"]
    row = custody["observations"][0] if source == "lane" else custody["public_inventory"]
    row["observed_utc"] = "2026-10-29T03:31:00+00:00"
    with pytest.raises(v1.StudyError, match="later than the recorded development start"):
        gate.prepare_committed_selection("repo", **fake_handoff["call"])
