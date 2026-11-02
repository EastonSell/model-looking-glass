"""Small source, asset and launcher custody checks; no checkpoint or GPU."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from looking_glass import agreement_head_fixed_donor_launcher_v3 as launcher
from looking_glass import agreement_head_fixed_donor_owner_v3 as owner


def test_asset_hashes_and_exact_file_set(tmp_path):
    asset = tmp_path / "weights"
    asset.write_bytes(b"abc")
    pins = {"weights": (3, hashlib.sha256(b"abc").hexdigest())}
    assert owner._verify_assets(tmp_path, pins)["weights"]["bytes"] == 3
    asset.write_bytes(b"abd")
    with pytest.raises(ValueError, match="differs"):
        owner._verify_assets(tmp_path, pins)
    (tmp_path / "extra").write_text("x")
    with pytest.raises(ValueError, match="exact pinned"):
        owner._verify_assets(tmp_path, pins)


def test_committed_source_bytes_are_bound_and_dirty_files_reject(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "source.py").write_bytes(b"x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "source"], check=True)
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"]).decode().strip()
    monkeypatch.setattr(owner, "SOURCE_FILES", ("source.py",))
    assert owner._source_binding(tmp_path, commit)["source.py"] == hashlib.sha256(b"x = 1\n").hexdigest()
    (tmp_path / "source.py").write_bytes(b"x = 2\n")
    with pytest.raises(ValueError, match="clean"):
        owner._source_binding(tmp_path, commit)
    subprocess.run(["git", "-C", str(tmp_path), "update-index", "--assume-unchanged", "source.py"], check=True)
    with pytest.raises(ValueError, match="source bytes differ"):
        owner._source_binding(tmp_path, commit)


def test_interrupted_launcher_wait_attempts_stop_and_retains_one_shot_claim(tmp_path, monkeypatch):
    work = tmp_path / "work"
    root = work / "teams" / "checkout"
    root.mkdir(parents=True)
    cache = work / "cache"
    cache.mkdir()
    inventory = work / "inventory.json"
    inventory.write_bytes(b"inventory")
    prior = work / "prior"
    prior.mkdir()
    output = tmp_path / "separate-volume-work"
    output.mkdir()
    run = output / "new-run"
    monkeypatch.setattr(launcher, "_source_binding", lambda *_: {"source": "bound"})
    monkeypatch.setattr(launcher, "_committed_bytes", lambda *_: b"fixed")
    monkeypatch.setattr(launcher, "_fixed_inputs", lambda *_: [])
    monkeypatch.setattr(launcher, "_verify_prior", lambda *_: {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(work / "unused-prefix"))
    class Disk:
        free = 4 * 1024 ** 3
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda *_: Disk())
    calls = []
    class Child:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt("caller interrupted")
            return 1
        def terminate(self):
            calls.append(("terminate", None))
            raise OSError("synthetic termination failure")
        def kill(self):
            calls.append(("kill", None))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_a, **_k: Child())
    with pytest.raises(KeyboardInterrupt, match="caller interrupted"):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=run,
                        inventory_snapshot=inventory, prior_attempt=prior)
    receipt = json.loads((run / "supervisor-exit.json").read_bytes())
    assert receipt["child_start"] == "observed" and receipt["exit_code"] == 1
    assert receipt["launcher_error_type"] == "KeyboardInterrupt"
    assert ("terminate", None) in calls and ("wait", 5) in calls
    assert (output / "agreement-head-v3-fixed-donor.claim.json").is_file()
    with pytest.raises(FileExistsError):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=output / "second-run",
                        inventory_snapshot=inventory, prior_attempt=prior)


def test_revised_launcher_checks_source_and_output_volumes_before_claim(tmp_path, monkeypatch):
    work = tmp_path / "work"
    root = work / "teams" / "checkout"
    root.mkdir(parents=True)
    cache = work / "cache"
    cache.mkdir()
    inventory = work / "inventory.json"
    inventory.write_bytes(b"inventory")
    prior = work / "prior"
    prior.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(output / "unused-prefix"))
    class Disk:
        def __init__(self, free):
            self.free = free
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: Disk(500_000_000 if Path(path) == root else 9 * 1024 ** 3))
    with pytest.raises(ValueError, match="source volume"):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=output / "run",
                        inventory_snapshot=inventory, prior_attempt=prior)
    assert not (output / "agreement-head-v3-fixed-donor.claim.json").exists()
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda path: Disk(9 * 1024 ** 3 if Path(path) == root else 500_000_000))
    with pytest.raises(ValueError, match="output volume"):
        launcher.launch(root=root, revision="a" * 40, cache_dir=cache, output_root=output, run_dir=output / "run",
                        inventory_snapshot=inventory, prior_attempt=prior)
    assert not (output / "agreement-head-v3-fixed-donor.claim.json").exists()


def test_fixed_case_selection_and_planned_reservation_must_agree(monkeypatch):
    def encoded(value):
        return json.dumps(value, sort_keys=True).encode()
    selection = encoded({"status": "frozen_for_prospective_evaluation", "candidate": {"head": 4},
                         "comparator": {"head": 8}})
    selection_sha = hashlib.sha256(selection).hexdigest()
    cases = []
    records = []
    for i in range(16):
        row = [1, i + 2, 30]
        cases.append({"text": f"case {i}", "token_ids": row,
                      "continuations": [{"token_id": token, "tokens": row + [token]}
                                        for token in (318, 389, 373, 547)]})
        records.append({"record_id": f"planned-agreement-head-v2-{i + 1:02d}", "category": "planned",
                        "public_text": f"case {i}", "token_ids": row, "candidate_token_ids": [318, 389, 373, 547],
                        "reference_commit": "4d8b93ad4bf601cb3f01f0a9c53aeb01fc43adfe"})
    case_raw = encoded({"selection_sha256": selection_sha, "candidate": {"layer": 3, "head": 4},
                        "comparator": {"layer": 3, "head": 8}, "cases": cases})
    reservation = {"version": "distilgpt2-development-exposures-v39", "records": records}
    inventory_raw = encoded(reservation)
    monkeypatch.setattr(owner, "SELECTION_SHA256", selection_sha)
    monkeypatch.setattr(owner, "CASE_SHA256", hashlib.sha256(case_raw).hexdigest())
    monkeypatch.setattr(owner, "INVENTORY_SHA256", hashlib.sha256(inventory_raw).hexdigest())
    assert owner._fixed_inputs(case_raw, selection, inventory_raw) == cases
    reservation["records"][0]["category"] = "observed-saved"
    changed = encoded(reservation)
    monkeypatch.setattr(owner, "INVENTORY_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="planned reservation"):
        owner._fixed_inputs(case_raw, selection, changed)


def test_prior_evaluation_hash_and_model_state_are_bound(tmp_path, monkeypatch):
    events, terminal = b"events\n", b"terminal\n"
    candidate = {"source_commit": owner.PRIOR_SOURCE, "terminal_sha256": hashlib.sha256(terminal).hexdigest(),
                 "model_state_before": "a" * 64, "model_state_after": "a" * 64}
    started = {"source_commit": owner.PRIOR_SOURCE, "model_state_before": "a" * 64,
               **{name: "test" for name in owner.RUNTIME_FIELDS}}
    payloads = {"events.jsonl": events, "terminal.json": terminal,
                "owner-candidate.json": json.dumps(candidate).encode(), "started.json": json.dumps(started).encode()}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    monkeypatch.setattr(owner, "PRIOR_HASHES", {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()})
    monkeypatch.setattr(owner, "_load_prior_observations", lambda *_: ((b"before", b"logits"),) * 16)
    assert owner._verify_prior(tmp_path)["model_state"] == "a" * 64
    (tmp_path / "events.jsonl").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="prior exposed evaluation differs"):
        owner._verify_prior(tmp_path)
    (tmp_path / "events.jsonl").write_bytes(events)
    candidate["model_state_after"] = "b" * 64
    changed = json.dumps(candidate).encode()
    (tmp_path / "owner-candidate.json").write_bytes(changed)
    monkeypatch.setattr(owner, "PRIOR_HASHES", {**owner.PRIOR_HASHES, "owner-candidate.json": hashlib.sha256(changed).hexdigest()})
    with pytest.raises(ValueError, match="model binding differs"):
        owner._verify_prior(tmp_path)


def test_prior_clean_witnesses_are_hash_checked(tmp_path):
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    before, logits = bytes(768 * 4), bytes(50257 * 4)
    refs = {}
    for name, payload, shape in (("before", before, [1, 1, 768]), ("last_logits", logits, [50257])):
        digest = hashlib.sha256(payload).hexdigest()
        (tensors / (digest + ".f32le")).write_bytes(payload)
        refs[name] = {"sha256": digest, "shape": shape, "dtype": "float32-le", "bytes": len(payload)}
    records = [{"event": {"type": "forward_complete", "kind": "observe", "head": 0, "row": i},
                "tensors": refs} for i in range(16)]
    raw = b"".join((json.dumps(record) + "\n").encode() for record in records)
    assert owner._load_prior_observations(tmp_path, raw) == ((before, logits),) * 16
    (tensors / (refs["last_logits"]["sha256"] + ".f32le")).write_bytes(logits[:-1] + b"x")
    with pytest.raises(ValueError, match="tensor bytes differ"):
        owner._load_prior_observations(tmp_path, raw)


def test_changed_cpu_runtime_stops_before_checkpoint_loading(tmp_path, monkeypatch):
    import torch
    import transformers
    from looking_glass import agreement_preflight_v1, copy_head_factorial

    # Isolate environment drift beyond the version gate; this is not a replay
    # of the historical pinned Torch runtime.
    monkeypatch.setattr(torch, "__version__", "2.8.0+cu128")
    monkeypatch.setattr(transformers, "__version__", "4.57.3")
    root = Path(owner.__file__).resolve().parents[1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_CBWR", "changed-mode")
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "unused-prefix"))
    monkeypatch.setattr(owner, "_source_binding", lambda *_: {})
    monkeypatch.setattr(owner, "_committed_bytes", lambda *_: b"fixed")
    monkeypatch.setattr(owner, "_fixed_inputs", lambda *_: [])
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_: None)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *_: None)
    monkeypatch.setattr(torch, "manual_seed", lambda *_: None)
    denormals = {"subnormal_input_bits": 1, "subnormal_output_bits": 4194304}
    monkeypatch.setattr(agreement_preflight_v1, "_prepare_denormals", lambda *_: denormals)
    monkeypatch.setattr(copy_head_factorial, "binary64_rounding_probe", lambda: None)
    prior_runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": owner.platform.python_version(), "platform": owner.platform.platform(),
                     "cpu_threads": 1, "cpu_interop_threads": 1,
                     "cpu_environment": {"MKL_CBWR": None, "ATEN_CPU_CAPABILITY": None},
                     "seed": 2026102912, "denormals_before": denormals,
                     "binary64_before": "nearest-even-probes-passed"}
    monkeypatch.setattr(owner, "_verify_prior", lambda *_: {"runtime": prior_runtime})
    called = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: called.append("tokenizer"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: called.append("model"))
    (tmp_path / "inventory").write_bytes(b"fixed")
    with pytest.raises(ValueError, match="runtime differs"):
        owner.run_fixed_donor_owner(root=root, revision="a" * 40, cache_dir=tmp_path, attempt=tmp_path / "attempt",
                                    inventory_snapshot=tmp_path / "inventory", prior_attempt=tmp_path / "prior")
    assert not called and not (tmp_path / "attempt").exists()
