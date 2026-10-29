"""Model-free and fabricated-model tests for the paired suffix study."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import bert_agreement_suffix_v1 as study
from scripts import observe_bert_agreement_suffix_v1 as observer
from scripts import read_bert_agreement_suffix_v1 as reader


class Block(torch.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        basis = torch.arange(768, device=input_ids.device, dtype=torch.float32) / 768 + 1
        context = input_ids.float().sum(dim=1)[:, None, None] / 100000
        hidden = input_ids.float().unsqueeze(-1) * basis / 100 + context * basis.square()
        patched = self.block(hidden)[0]
        value = patched.sum(dim=(1, 2)) / 10000
        columns = torch.arange(16, dtype=torch.float32, device=input_ids.device) / 20
        scores = value[:, None, None] * columns[None, None, :]
        return SimpleNamespace(logits=scores.expand(-1, input_ids.shape[1], -1).clone())


def manifest():
    path = Path(__file__).resolve().parents[1] / "experiments/bert_agreement_suffix_cases_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == study.CASE_SHA
    return json.loads(path.read_bytes())


def fake_policy(monkeypatch):
    monkeypatch.setattr(study, "require_policy", lambda _device: study.common.require_subnormals())
    assert torch.set_flush_denormal(False)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def complete_fake_attempt(tmp_path, monkeypatch):
    fake_policy(monkeypatch)
    monkeypatch.setattr(reader, "VOCAB", 16)
    monkeypatch.setattr(reader, "TARGETS", {"is": 0, "are": 1, "was": 2, "were": 3})
    root = tmp_path / "attempt"
    root.mkdir()
    commit, git_sha, source = "a" * 40, "b" * 64, {"source": "c" * 64}
    metadata = {"pid": 123, "source_commit": commit, "sources_sha256": source,
                "git_sha256": git_sha, "assets_sha256": reader.ASSETS, "case_sha256": study.CASE_SHA,
                "schema": "bert-agreement-suffix-attempt-v1", "device": "cpu", "seed": study.SEED,
                "python": "fake-python", "torch": "fake-torch", "transformers": "fake-transformers",
                "numpy": "fake-numpy", "start_unix": 1.0, "initial_memory": None, "gpu_name": None,
                "base_interpreter_direct": True, "isolated": True,
                "deterministic_algorithms": True, "tf32": False, "cudnn_benchmark": False,
                "cudnn_deterministic": True, "intra_threads": 1, "inter_threads": 1,
                "cublas_workspace_config": None}
    write_json(root / "metadata.json", metadata)
    write_json(root / "loading.json", {key: [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")})
    model = FakeModel()
    journal = study.Journal(root, vocab=16)
    study.run(model, model.block, manifest(), torch.device("cpu"), journal)
    journal.close()
    completed = {"status": "complete", "events": 416, "model_weakref_dead": True}
    for name in ("metadata.json", "loading.json", "starts.jsonl", "events.jsonl", "memory.jsonl",
                 "retention-errors.jsonl", "logits.f32", "witness.f32"):
        completed[name.replace(".", "_") + "_sha256"] = study.file_sha(root / name)
    write_json(root / "complete.json", completed)
    receipt_path = tmp_path / "receipt.json"
    receipt = {"schema": "bert-agreement-suffix-direct-child-exit-v1", "status": "complete",
               "exit_code": 0, "interrupted": None, "child_pid": 123,
               "attempt_name": root.name, "source_commit": commit,
               "complete_sha256": study.file_sha(root / "complete.json"),
               "metadata_sha256": study.file_sha(root / "metadata.json")}
    write_json(receipt_path, receipt)
    return root, receipt_path, source, git_sha, commit


def test_manifest_donor_and_mask_alignment():
    cases = manifest()["cases"]
    for case in cases:
        opp, same = reader.donor_indices(cases, case)
        for variant in ("terminal", "continued"):
            ids = case["variants"][variant]["input_ids"]
            for donor in (opp, same):
                donor_ids = cases[donor]["variants"][variant]["input_ids"]
                assert [i for i, (a, b) in enumerate(zip(ids, donor_ids)) if a != b] == [6]
            assert ids[10] == 103
        assert case["variants"]["terminal"]["input_ids"][:-1] == case["variants"]["continued"]["input_ids"][:11]


def test_rounded_shared_direction_rejects_erased_and_distorted_tiny_dose():
    base = np.ones(768, dtype=np.float32)
    direction = np.random.Generator(np.random.PCG64(20261029)).standard_normal(768)
    donor = base.copy()
    donor[0] = np.nextafter(donor[0], np.float32(2.0))
    tiny_q = float(np.linalg.norm(donor.astype(np.float64) - base.astype(np.float64)))
    with pytest.raises(ValueError, match="landing"):
        study.landing(base, direction, tiny_q)
    donor = base.copy()
    donor[:172] = np.nextafter(donor[:172], np.float32(0.0))
    tiny_q = float(np.linalg.norm(donor.astype(np.float64) - base.astype(np.float64)))
    with pytest.raises(ValueError, match="landing"):
        study.landing(base, direction, tiny_q)


def test_candidate_probability_underflow_is_reported_unresolved():
    assert torch.set_flush_denormal(False)
    logits = np.zeros(30522, dtype=np.float32)
    logits[reader.TARGETS["is"]] = -744
    score = reader.clean_score(logits, 0)
    assert score["probability_numerically_unresolved"] is True
    assert score["p_is"] is None and score["p_is_plus_p_are"] is None


def test_finite_tiny_pair_mass_ratio_overflow_is_reported():
    assert torch.set_flush_denormal(False)
    old, new = np.zeros(30522, dtype=np.float32), np.zeros(30522, dtype=np.float32)
    old[[reader.TARGETS["is"], reader.TARGETS["are"]]] = -727
    new[[reader.TARGETS["is"], reader.TARGETS["are"]]] = 10
    old_mass = reader.clean_score(old, 0)["p_is_plus_p_are"]
    new_mass = reader.clean_score(new, 0)["p_is_plus_p_are"]
    assert old_mass is not None and old_mass > 0 and new_mass is not None
    assert reader.probability_ratio(new_mass, old_mass) == (None, "overflow")


def test_trace_cap_stops_before_writing_start(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "OUTPUT_CAP", study.TAIL_RESERVE + 1)
    journal = study.Journal(tmp_path, vocab=16)
    case = manifest()["cases"][0]
    with pytest.raises(ValueError, match="cap"):
        journal.start(case, case["variants"]["terminal"], "terminal", "clean")
    journal.close()
    assert (tmp_path / "starts.jsonl").stat().st_size == 0


def test_full_fake_schedule_and_reader_reject_coherently_rehashed_wrong_donor(tmp_path, monkeypatch):
    root, receipt, source, git_sha, commit = complete_fake_attempt(tmp_path, monkeypatch)
    output = reader.replay(root, manifest(), receipt, study.file_sha(receipt),
                           study.file_sha(root / "complete.json"), source, git_sha, commit)
    assert output["summary"]["cases"] == 32
    assert len(output["cases"]) == 32
    assert all(row["terminal"]["clean"]["p_is_plus_p_are"] is not None for row in output["cases"])
    complete_path = root / "complete.json"
    def rebind(name):
        complete = json.loads(complete_path.read_bytes())
        complete[name.replace(".", "_") + "_sha256"] = study.file_sha(root / name)
        write_json(complete_path, complete)
        observation = json.loads(receipt.read_bytes())
        observation["complete_sha256"] = study.file_sha(complete_path)
        if name == "metadata.json":
            observation["metadata_sha256"] = study.file_sha(root / name)
        write_json(receipt, observation)
    def checked_replay():
        return reader.replay(root, manifest(), receipt, study.file_sha(receipt),
                             study.file_sha(complete_path), source, git_sha, commit)
    memory_path = root / "memory.jsonl"
    memory_original = memory_path.read_bytes()
    samples = [json.loads(line) for line in memory_original.splitlines()]
    samples[-1]["event_index"] = 0
    memory_path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in samples), encoding="utf-8")
    rebind("memory.jsonl")
    with pytest.raises(ValueError, match="memory sample order"):
        checked_replay()
    memory_path.write_bytes(memory_original)
    rebind("memory.jsonl")
    loading_path = root / "loading.json"
    loading_original = loading_path.read_bytes()
    loading = json.loads(loading_original)
    loading["missing_keys"] = False
    write_json(loading_path, loading)
    rebind("loading.json")
    with pytest.raises(ValueError, match="loading diagnostics"):
        checked_replay()
    loading_path.write_bytes(loading_original)
    rebind("loading.json")
    metadata_path = root / "metadata.json"
    metadata_original = metadata_path.read_bytes()
    metadata = json.loads(metadata_original)
    del metadata["torch"]
    write_json(metadata_path, metadata)
    rebind("metadata.json")
    with pytest.raises(ValueError, match="metadata differs"):
        checked_replay()
    metadata_path.write_bytes(metadata_original)
    rebind("metadata.json")
    events_path = root / "events.jsonl"
    events_original = events_path.read_bytes()
    events = [json.loads(line) for line in events_original.splitlines()]
    changed_self = next(e for e in events if e["arm"] == "self")
    changed_self.update({"donor_index": 999, "random_seed": 123, "actual_l2": 42, "relative_direction_error": 1})
    events_path.write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in events), encoding="utf-8")
    rebind("events.jsonl")
    with pytest.raises(ValueError, match="self or clean-after evidence"):
        checked_replay()
    events_path.write_bytes(events_original)
    rebind("events.jsonl")
    events = [json.loads(line) for line in events_original.splitlines()]
    changed = next(e for e in events if e["arm"] == "opposite")
    changed["donor_index"] = 31 if changed["donor_index"] != 31 else 30
    events_path.write_text("".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in events), encoding="utf-8")
    rebind("events.jsonl")
    with pytest.raises(ValueError, match="donor or rounded landing"):
        checked_replay()


def test_nonselected_nan_retains_the_returned_payload(tmp_path, monkeypatch):
    fake_policy(monkeypatch)
    class NonfiniteModel(FakeModel):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            result.logits[0, 0, 0] = float("nan")
            return result
    model = NonfiniteModel()
    journal = study.Journal(tmp_path, vocab=16)
    with pytest.raises(ValueError, match="nonfinite"):
        study.run(model, model.block, manifest(), torch.device("cpu"), journal)
    journal.close()
    assert len((tmp_path / "starts.jsonl").read_text().splitlines()) == 1
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 1 and events[0]["full_logits_finite"] is False
    assert (tmp_path / "logits.f32").stat().st_size == 16 * 4
    assert (tmp_path / "witness.f32").stat().st_size == 2 * 768 * 4
    assert not model.block._forward_hooks


def test_observer_reaps_harmless_child_and_rejects_failure_sentinel(tmp_path, monkeypatch):
    commit, hashes = "a" * 40, {"source": "c" * 64}
    git = Path(shutil.which("git")).resolve()
    git_sha = observer.digest(git)
    for mode in ("ok", "sentinel", "postcheck"):
        def verify(*_args):
            if mode == "postcheck":
                raise ValueError("source changed")
            return hashes
        monkeypatch.setattr(observer, "checked_sources", verify)
        controller = tmp_path / ("controller-" + mode)
        attempt = tmp_path / ("attempt-" + mode)
        controller.mkdir()
        code = (
            "import hashlib,json,os;from pathlib import Path;"
            f"p=Path({str(attempt)!r});p.mkdir();"
            f"meta={{'pid':os.getpid(),'source_commit':{commit!r},'sources_sha256':{hashes!r},'git_sha256':{git_sha!r}}};"
            "(p/'metadata.json').write_text(json.dumps(meta));"
            "h=hashlib.sha256((p/'metadata.json').read_bytes()).hexdigest();"
            "(p/'complete.json').write_text(json.dumps({'status':'complete','events':416,'metadata_json_sha256':h}));"
            + ("(p/'failure.txt').write_text('after completion');" if mode == "sentinel" else "")
        )
        result = observer.observe([str(Path(sys._base_executable).resolve()), "-c", code], controller, attempt,
                                  commit, os.environ.copy(), 30, tmp_path, git, hashes, git_sha)
        assert type(result["exit_code"]) is int and result["exit_code"] == 0
        assert result["child_pid"] is not None and result["interrupted"] is None
        assert result["status"] == ("complete" if mode == "ok" else "incomplete")
        assert (controller / observer.PENDING_NAME).exists() == (mode != "ok")
        if mode == "postcheck":
            assert result["completion_check_error"] == "ValueError"
        assert (controller / "exit-observation.json").is_file()


def test_interrupted_receipt_write_keeps_pending_marker_and_cannot_replay(tmp_path, monkeypatch):
    controller, attempt = tmp_path / "controller", tmp_path / "attempt"
    controller.mkdir()
    commit, hashes = "a" * 40, {"source": "c" * 64}
    git = Path(shutil.which("git")).resolve()
    git_sha = observer.digest(git)
    monkeypatch.setattr(observer, "checked_sources", lambda *_args: hashes)
    original = observer.save_json
    def interrupted(path, value):
        if path.name == "exit-observation.json":
            with path.open("xb") as stream:
                stream.write((json.dumps(value) + "\n").encode())
                stream.flush()
                raise KeyboardInterrupt()
        return original(path, value)
    monkeypatch.setattr(observer, "save_json", interrupted)
    code = ("import hashlib,json,os;from pathlib import Path;"
            f"p=Path({str(attempt)!r});p.mkdir();"
            f"meta={{'pid':os.getpid(),'source_commit':{commit!r},'sources_sha256':{hashes!r},'git_sha256':{git_sha!r}}};"
            "(p/'metadata.json').write_text(json.dumps(meta));"
            "h=hashlib.sha256((p/'metadata.json').read_bytes()).hexdigest();"
            "(p/'complete.json').write_text(json.dumps({'status':'complete','events':416,'metadata_json_sha256':h}));")
    with pytest.raises(KeyboardInterrupt):
        observer.observe([str(Path(sys._base_executable).resolve()), "-c", code], controller, attempt,
                         commit, os.environ.copy(), 30, tmp_path, git, hashes, git_sha)
    receipt = controller / "exit-observation.json"
    assert json.loads(receipt.read_bytes())["status"] == "complete"
    assert (controller / observer.PENDING_NAME).is_file()
    with pytest.raises(ValueError, match="observer-pending sentinel"):
        reader.replay(attempt, {}, receipt, observer.digest(receipt), "unused", hashes, git_sha, commit)


def test_isolated_source_bootstrap_ignores_checkout_shadow_with_positive_control(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git = Path(shutil.which("git")).resolve()
    for name in study.SOURCE_PATHS:
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, destination)
    def git_run(*args):
        return subprocess.run([str(git), "-C", str(checkout), *args], check=True, capture_output=True, text=True)
    git_run("init")
    git_run("config", "core.autocrlf", "false")
    git_run("add", *study.SOURCE_PATHS)
    git_run("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    commit = git_run("rev-parse", "HEAD").stdout.strip()
    marker = tmp_path / "checkout-shadow-ran"
    (checkout / "numbers.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('shadow')\n", encoding="utf-8")
    site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    base = Path(sys._base_executable).resolve()
    positive_prefix = tmp_path / "positive-cache"
    positive_prefix.mkdir()
    positive_code = f"import sys;sys.path.insert(0,{str(checkout)!r});sys.path.append({str(site)!r});import numbers"
    subprocess.run([str(base), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(positive_prefix), "-c", positive_code],
                   check=True, capture_output=True)
    assert marker.read_text() == "shadow"
    marker.unlink()
    prefix = tmp_path / "owner-cache"
    prefix.mkdir()
    empty_model = tmp_path / "empty-model"
    empty_model.mkdir()
    command = [str(base), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(checkout / "scripts/bert_agreement_suffix_v1.py"), "--site", str(site), "--git", str(git),
               "--model-dir", str(empty_model), "--cases", str(checkout / study.SOURCE_PATHS[-1]),
               "--output", str(tmp_path / "never-created"), "--device", "cpu", "--source-commit", commit]
    result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy(), timeout=90)
    assert result.returncode != 0
    assert "model snapshot has missing or additional assets" in result.stderr
    assert not marker.exists()
    tree_line = git_run("cat-file", "-p", commit).stdout.splitlines()[0]
    assert tree_line.startswith("tree ")
    tree = tree_line.split()[1]
    tree_prefix = tmp_path / "tree-cache"
    tree_prefix.mkdir()
    tree_command = list(command)
    tree_command[tree_command.index("-X") + 1] = "pycache_prefix=" + str(tree_prefix)
    tree_command[-1] = tree
    tree_result = subprocess.run(tree_command, capture_output=True, text=True, env=os.environ.copy(), timeout=30)
    assert tree_result.returncode != 0
    assert "source identifier does not name a commit" in tree_result.stderr
    assert not marker.exists()
