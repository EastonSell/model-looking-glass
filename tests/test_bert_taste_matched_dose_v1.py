"""Fabricated CPU checks; no pretrained model or reserved inputs are used."""
import base64
import copy
import hashlib
import importlib.util
import json
import math
import py_compile
import shutil
import subprocess
import struct
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM

from scripts import bert_taste_matched_dose_v1 as study


def tensor(values):
    return torch.tensor(values, dtype=torch.float32)


def _enrich_fake_result(result, source, output, parent):
    result.update({"source": source, "seed": 20261030, "schema": "bert-taste-matched-dose-result-v1",
                   "model_state_sha256_before_after": "a" * 64,
                   "exposure": {"prior_row_counts": [160, 64, 8], "exact_or_complete_prefix_collisions": 0,
                                "complete_exposure_history_certified": False},
                   "runtime": {"threads": 1, "interop_threads": 1, "deterministic_algorithms": True,
                               "cuda_initialized": False, "default_dtype": "torch.float32", "mkldnn_enabled": True,
                               "omp_num_threads": "1", "mkl_num_threads": "1", "mkl_cbwr": "AUTO",
                               "aten_cpu_capability": None, "denormal_probe": {"smallest_times_one_bits": 1,
                                                                                "normal_half_bits": 4194304},
                               "rounding_probe": "nearest-even-halfway-cases-pass", "torch_build_sha256": "b" * 64,
                               "python": "synthetic", "torch": "synthetic"}})
    result["model"].update({"model_class": "DistilBertForMaskedLM", "tokenizer_class": "DistilBertTokenizerFast",
                            "attention_implementation": "eager", "parameter_count": 66985530,
                            "pytorch_cuda_initialized": False, "assets": parent.EXPECTED_ASSETS,
                            "torch_version": "synthetic", "transformers_version": "synthetic"})
    (output / "started.json").write_text(json.dumps({"status": "started", "source": source,
            "cases_path": study.CASES_PATH, "seed": 20261030}), encoding="utf-8")


def _shadow_checkout(tmp_path, module, sourceless):
    root = tmp_path / "checkout"
    for name in study.SOURCE_PATHS:
        if name.endswith(".py"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(study.ROOT / name, target)
    marker = tmp_path / (module + "-executed")
    source = root / (module + ".py")
    source.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n",
                      encoding="utf-8")
    if sourceless:
        py_compile.compile(str(source), cfile=str(root / (module + ".pyc")), doraise=True)
        source.unlink()
    prefix = tmp_path / (module + "-prefix")
    prefix.mkdir()
    return root, marker, prefix


@pytest.mark.parametrize("sourceless", [False, True])
def test_isolated_child_ignores_checkout_stdlib_shadow(tmp_path, sourceless):
    root, marker, prefix = _shadow_checkout(tmp_path, "numbers", sourceless)
    child = root / "scripts/bert_taste_matched_dose_v1.py"
    outcome = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(prefix),
                              str(child), "--help"], cwd=root, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, timeout=60)
    assert outcome.returncode == 0 and not marker.exists()


@pytest.mark.parametrize("sourceless", [False, True])
def test_checkout_unlisted_module_cannot_execute(tmp_path, sourceless):
    root, marker, prefix = _shadow_checkout(tmp_path, "unlisted_shadow_probe", sourceless)
    child = root / "scripts/bert_taste_matched_dose_v1.py"
    program = "import runpy,sys; runpy.run_path(sys.argv[1]); __import__('unlisted_shadow_probe')"
    outcome = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(prefix),
                              "-c", program, str(child)], cwd=root, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, timeout=60)
    assert outcome.returncode != 0 and not marker.exists()


def test_checkout_unlisted_namespace_initializer_cannot_execute(tmp_path):
    root, marker, prefix = _shadow_checkout(tmp_path, "unlisted_shadow_probe", False)
    initializer = root / "scripts/__init__.py"
    initializer.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n",
                           encoding="utf-8")
    outcome = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(prefix),
                              str(root / "scripts/bert_taste_matched_dose_v1.py"), "--help"],
                             cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    assert outcome.returncode != 0 and not marker.exists()


def test_external_or_preloaded_project_namespace_is_rejected(tmp_path):
    root, marker, prefix = _shadow_checkout(tmp_path, "unlisted_shadow_probe", False)
    outside = tmp_path / "outside"
    (outside / "scripts").mkdir(parents=True)
    (outside / "scripts/__init__.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n", encoding="utf-8")
    child = root / "scripts/bert_taste_matched_dose_v1.py"
    programs = (
        "import runpy,sys; sys.path.insert(0,sys.argv[2]); runpy.run_path(sys.argv[1])",
        "import runpy,sys,types; sys.modules['scripts']=types.ModuleType('scripts'); runpy.run_path(sys.argv[1])",
    )
    for program in programs:
        outcome = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(prefix),
                                  "-c", program, str(child), str(outside)], cwd=root,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        assert outcome.returncode != 0 and not marker.exists()


def test_ordinary_import_does_not_install_checkout_guard():
    program = "from scripts import bert_taste_matched_dose_v1; import looking_glass.causal_metrics"
    outcome = subprocess.run([sys.executable, "-B", "-c", program], cwd=study.ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    assert outcome.returncode == 0


def test_common_radius_uses_minimum_and_never_extrapolates():
    source = study.matched_sources(tensor([0, 0]), tensor([3, 4]),
                                   tensor([0, 0]), tensor([0, 2]))
    assert source["radius"] == 2
    assert source["distances"] == {"ingredient": 5, "control": 2}
    assert source["fractions"] == {"ingredient": 0.4, "control": 1}
    for direction in ("sugar_to_lemon", "lemon_to_sugar"):
        for site in ("ingredient", "control"):
            item = source["actions"][direction][site]
            assert math.isclose(item["delivered_norm"], 2, abs_tol=1e-6)
            assert 0 <= item["fraction"] <= 1
            assert item["relative_error"] <= 1e-6


def test_zero_small_and_nonfinite_vectors_fail_without_redraw():
    for vectors in ((tensor([0, 0]), tensor([0, 0]), tensor([1, 0]), tensor([2, 0])),
                    (tensor([0, 0]), tensor([1, 0]), tensor([0, 0]), tensor([1e-6, 0])),
                    (tensor([float("nan"), 0]), tensor([1, 0]), tensor([0, 0]), tensor([1, 0]))):
        with pytest.raises(ValueError):
            study.matched_sources(*vectors)


def test_float32_rounding_mismatch_or_zero_delivery_fails():
    for control_delta in (5.0, 1e-5):
        with pytest.raises(ValueError, match="dose"):
            study.matched_sources(tensor([1e8]), tensor([1e8 + 8]), tensor([0]), tensor([control_delta]))


def test_baseline_eligibility_requires_signs_and_top_five_for_both_rows():
    sugar = {"sweet_minus_sour": 1.0, "top_five_ids": [1, 3, 4, 5, 6]}
    lemon = {"sweet_minus_sour": -0.5, "top_five_ids": [2, 3, 4, 5, 6]}
    assert study.eligible(sugar, lemon, 1, 2)
    assert not study.eligible({**sugar, "sweet_minus_sour": 0.0}, lemon, 1, 2)
    assert not study.eligible(sugar, {**lemon, "top_five_ids": [3, 4, 5, 6, 7]}, 1, 2)


def test_top_five_ties_prefer_smaller_vocabulary_ids():
    assert study.top_five(tensor([1, 3, 3, 2, 2, 2, 2])) == [1, 2, 3, 4, 5]


def test_actual_entry_ids_mask_and_method_cleanup():
    torch.random.default_generator.manual_seed(1701)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32,
                              n_layers=2, dim=16, hidden_dim=32, n_heads=2,
                              dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = DistilBertForMaskedLM(config).eval()
    ids = tuple(range(10))
    observed, entry = study.observe_entry(model, ids, layer=0, position=7, mask_index=4)
    assert entry == {"input_ids": list(ids), "attention_mask": [1] * 10}
    assert observed.before.shape == (16,)
    assert "forward" not in model.__dict__


def test_actual_entry_wrapper_is_restored_on_failure(monkeypatch):
    model = torch.nn.Linear(2, 2)
    def fail(*_args, **_kwargs):
        raise RuntimeError("fabricated observation failure")
    monkeypatch.setattr(study.previous, "observe", fail)
    with pytest.raises(RuntimeError, match="fabricated"):
        study.observe_entry(model, tuple(range(10)), layer=0, position=7, mask_index=4)
    assert "forward" not in model.__dict__


def test_parent_completion_requires_all_six_journal_hashes(tmp_path, monkeypatch):
    from scripts import bert_taste_matched_parent_v1 as parent
    source = {"commit": "a" * 40, "selected_source_sha256": {}}
    journals = {}
    for frame in study.FRAMES:
        events = []
        def pair(row, layer, position, mode, donor):
            event = {"row_id": frame + "-" + row, "layer": layer, "position": position,
                     "mode": mode, "donor": donor}
            events.extend({"event": kind, **event} for kind in ("forward_start", "forward_complete"))
        for layer in range(6):
            for row in ("sugar", "lemon"):
                pair(row, layer, 7, "capture_ingredient", None)
                pair(row, layer, 1, "capture_control", None)
            events.append({"event": "dose_plan", "row_id": frame + "-sugar", "frame": frame,
                           "layer": layer, "radius": 1.0,
                           "distances": {"ingredient": 2.0, "control": 1.0},
                           "fractions": {"ingredient": 0.5, "control": 1.0}})
            for row, other in (("sugar", "lemon"), ("lemon", "sugar")):
                for mode, pos, is_self in (("ingredient_donor", 7, False), ("ingredient_self", 7, True),
                                            ("control_self", 1, True), ("ingredient_matched", 7, False),
                                            ("control_matched", 1, False)):
                    pair(row, layer, pos, mode, frame + "-" + (row if is_self else other))
        for row in ("sugar", "lemon"):
            pair(row, 0, 7, "final_clean", None)
        data = b"".join((json.dumps(event) + "\n").encode() for event in events)
        (tmp_path / (frame + ".jsonl")).write_bytes(data)
        journals[frame + ".jsonl"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    row_ids = [frame + "-" + word for frame in study.FRAMES for word in ("sugar", "lemon")]
    result = {"status": "complete", "completed_forwards": 516, "source": source, "journals": journals,
              "schema": "bert-taste-matched-dose-result-v1", "seed": 20261030,
              "baselines": {row: {} for row in row_ids}, "eligibility": {frame: False for frame in study.FRAMES},
              "effects": [{}] * 360, "doses": [{}] * 36, "comparisons": [{}] * 72,
              "layer_summary": [{}] * 6,
              "model": {"model": study.MODEL, "revision": study.REVISION, "device": "cpu", "dtype": "float32"}}
    _enrich_fake_result(result, source, tmp_path, parent)
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    # Identity-only completions lack actual input, output and action evidence.
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
    # Isolate the schedule checks after the missing-payload test above.
    monkeypatch.setattr(parent, "_completion_matches", lambda *_args: True)
    monkeypatch.setattr(parent, "_audit_saved_bytes", lambda *_args: None)
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "complete"
    assert parent.assess_result(tmp_path, 1, source, source, True)["status"] == "incomplete"
    path = tmp_path / "coffee.jsonl"
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    flipped = lines.copy()
    flipped[0], flipped[1] = flipped[1], flipped[0]
    typed = lines.copy()
    bad = json.loads(typed[0])
    bad["layer"] = False
    typed[0] = (json.dumps(bad) + "\n").encode()
    # Self-consistent hashes cannot promote absent, truncated, reordered, or mistyped events.
    for invalid in (b"", original[:-1], b"".join(flipped), b"".join(typed)):
        path.write_bytes(invalid)
        journals["coffee.jsonl"] = {"size": len(invalid), "sha256": hashlib.sha256(invalid).hexdigest()}
        (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
        assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
    (tmp_path / "coffee.jsonl").write_bytes(b"tampered")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"


def test_parent_rejects_duplicate_nonfinite_and_float_count(tmp_path, monkeypatch):
    from scripts import bert_taste_matched_parent_v1 as parent
    source = {"commit": "a" * 40, "selected_source_sha256": {}}
    for raw in (b'{"status":"complete","status":"complete"}',
                b'{"status":"complete","ignored":NaN}',
                b'{"status":"complete","completed_forwards":516.0}'):
        (tmp_path / "result.json").write_bytes(raw)
        assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"


def test_parent_preserves_fresh_cancellation_on_receipt_failure(tmp_path, monkeypatch):
    from scripts import bert_taste_matched_parent_v1 as parent
    original, cancellation = OSError("child failure"), KeyboardInterrupt("receipt interrupted")
    monkeypatch.setattr(parent, "source_snapshot", lambda: {"commit": "a" * 40})
    def fail_child(*_args, **_kwargs):
        raise original
    def fail_receipt(*_args, **_kwargs):
        raise cancellation
    monkeypatch.setattr(parent.subprocess, "run", fail_child)
    monkeypatch.setattr(parent, "_write_receipt", fail_receipt)
    with pytest.raises(KeyboardInterrupt) as caught:
        parent.main(["--cache-dir", str(tmp_path), "--output", str(tmp_path / "run")])
    assert caught.value is cancellation and caught.value.__cause__ is original


def test_parseable_complete_receipt_does_not_mask_parent_fsync_failure(tmp_path, monkeypatch):
    from scripts import bert_taste_matched_parent_v1 as parent
    monkeypatch.setattr(parent, "source_snapshot", lambda: {"commit": "a" * 40})
    monkeypatch.setattr(parent.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(parent, "assess_result", lambda *_args: {"status": "complete"})
    def fail_after_write(path, value):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream)
        raise OSError("fabricated fsync failure")
    monkeypatch.setattr(parent, "_write_receipt", fail_after_write)
    with pytest.raises(OSError, match="fsync"):
        parent.main(["--cache-dir", str(tmp_path), "--output", str(tmp_path / "run")])
    receipt = json.loads((tmp_path / "run.observer.json").read_bytes())
    assert receipt["status"] == "complete"


def test_parent_rejects_child_omitting_its_source(tmp_path, monkeypatch):
    from scripts import bert_taste_matched_parent_v1 as parent
    child = tmp_path / "child.py"
    child.write_text("SOURCE_PATHS = ('scripts/bert_taste_matched_parent_v1.py',)\n", encoding="utf-8")
    monkeypatch.setattr(parent, "CHILD", child)
    with pytest.raises(ValueError, match="source list"):
        parent.source_snapshot()


def test_parent_entry_isolation_blocks_script_directory_shadow(tmp_path):
    root = tmp_path / "checkout"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    parent = scripts / "bert_taste_matched_parent_v1.py"
    shutil.copyfile(study.ROOT / "scripts/bert_taste_matched_parent_v1.py", parent)
    marker = tmp_path / "executed"
    (scripts / "argparse.py").write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n",
                                          encoding="utf-8")
    outcome = subprocess.run([sys.executable, "-B", str(parent), "--help"], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    assert outcome.returncode != 0 and not marker.exists()
    (scripts / "fractions.py").write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n",
                                           encoding="utf-8")
    reader = scripts / "bert_taste_matched_independent.py"
    reader.write_text("from fractions import Fraction\ndef audit_files(*args, **kwargs):\n return {'completed_forwards': 516}\n",
                      encoding="utf-8")
    prefix = tmp_path / "parent-prefix"
    prefix.mkdir()
    program = ("import runpy,sys,hashlib; from pathlib import Path; n=runpy.run_path(sys.argv[1]); "
               "p=Path(sys.argv[2]); a='scripts/bert_taste_matched_independent.py'; "
               "c='experiments/bert_taste_matched_dose_public_cases_v1.json'; "
               "before={'selected_source_sha256':{a:hashlib.sha256((p/a).read_bytes()).hexdigest(),c:'0'*64}}; "
               "n['_audit_saved_bytes'](p,before)")
    outcome = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(prefix),
                              "-c", program, str(parent), str(root)], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    assert outcome.returncode == 0 and not marker.exists()


def test_parent_and_verified_reader_accept_full_fabricated_bundle_and_reject_wrong_effect(tmp_path):
    from scripts import bert_taste_matched_parent_v1 as parent
    fixture_path = study.ROOT / "tests/bert_taste_matched_fake_fixture.py"
    spec = importlib.util.spec_from_file_location("_taste_fake_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cases = json.loads((study.ROOT / study.CASES_PATH).read_bytes())
    result = fixture.make_fixture(tmp_path, cases)
    hashes = {name: hashlib.sha256((study.ROOT / name).read_bytes()).hexdigest() for name in study.SOURCE_PATHS}
    source = {"commit": "a" * 40, "selected_source_sha256": hashes}
    _enrich_fake_result(result, source, tmp_path, parent)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "complete"
    for mutate in (lambda data: data["runtime"].update(threads=2),
                   lambda data: data["runtime"].update(torch="different"),
                   lambda data: data["model"]["assets"]["config.json"].update(sha256="0" * 64),
                   lambda data: data.pop("model_state_sha256_before_after")):
        altered = copy.deepcopy(result)
        mutate(altered)
        result_path.write_text(json.dumps(altered), encoding="utf-8")
        assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
    started = tmp_path / "started.json"
    started_data = started.read_bytes()
    wrong_start = json.loads(started_data)
    wrong_start["seed"] = 1
    started.write_text(json.dumps(wrong_start), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
    started.write_bytes(started_data)
    altered = copy.deepcopy(result)
    altered["effects"][0]["effect"] += 1.0
    result_path.write_text(json.dumps(altered), encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
    journal = tmp_path / "coffee.jsonl"
    original = journal.read_bytes()
    records = [json.loads(line) for line in original.splitlines()]
    action = next(e for e in records if e.get("event") == "forward_complete" and e.get("mode") == "ingredient_self")
    action["summary"]["sweet_probability"] = 0.0
    changed = b"".join((json.dumps(record) + "\n").encode() for record in records)
    journal.write_bytes(changed)
    altered = copy.deepcopy(result)
    reported = next(e for e in altered["effects"] if e["recipient"] == action["row_id"]
                    and e["layer"] == action["layer"] and e["mode"] == "ingredient_self")
    reported["summary"]["sweet_probability"] = 0.0
    altered["journals"]["coffee.jsonl"] = {"size": len(changed), "sha256": hashlib.sha256(changed).hexdigest()}
    result_path.write_text(json.dumps(altered), encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"
    records = [json.loads(line) for line in original.splitlines()]
    action = next(e for e in records if e.get("event") == "forward_complete" and e.get("mode") == "ingredient_matched")
    wrong = base64.b64encode(struct.pack("<768f", 0., 1., *([0.] * 766))).decode("ascii")
    action["source_float32_le_base64"] = action["after_float32_le_base64"] = wrong
    changed = b"".join((json.dumps(record) + "\n").encode() for record in records)
    journal.write_bytes(changed)
    altered = copy.deepcopy(result)
    altered["journals"]["coffee.jsonl"] = {"size": len(changed), "sha256": hashlib.sha256(changed).hexdigest()}
    result_path.write_text(json.dumps(altered), encoding="utf-8")
    assert parent.assess_result(tmp_path, 0, source, source, True)["status"] == "incomplete"


def test_complete_fabricated_schedule_binds_donors_doses_and_self_controls():
    torch.random.default_generator.manual_seed(1777)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32,
                              n_layers=2, dim=16, hidden_dim=32, n_heads=2,
                              dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = DistilBertForMaskedLM(config).eval()
    ids = list(range(10))
    second = ids.copy()
    second[7] = 15
    rows = [{"id": "toy-sugar", "input_ids": ids}, {"id": "toy-lemon", "input_ids": second}]

    class Tokenizer:
        def convert_ids_to_tokens(self, values):
            return [str(value) for value in values]

    events = []
    result = study.run_cases(model, rows, sweet_id=1, sour_id=2, tokenizer=Tokenizer(), emit=events.append)
    assert result["completed_forwards"] == 30
    assert len([e for e in events if e["event"] == "forward_start"]) == 30
    assert len([e for e in events if e["event"] == "forward_complete"]) == 30
    assert len([e for e in events if e["event"] == "dose_plan"]) == 2
    assert len(result["effects"]) == 20
    for event in events:
        if event["event"] == "forward_complete":
            assert event["model_entry"]["input_ids"] == (ids if event["row_id"] == "toy-sugar" else second)
            assert event["model_entry"]["attention_mask"] == [1] * 10
            if event["mode"].endswith("_self"):
                assert event["before_float32_le_base64"] == event["after_float32_le_base64"]
            if event["mode"].endswith("_matched"):
                assert event["actual_pre_post_norm"] == event["dose"]["delivered_norm"]
                assert event["donor_clean_float32_le_base64"] is not None
    assert "forward" not in model.__dict__ and not study.previous.has_hooks(model)


def test_unavailable_dose_records_reason_and_stops_without_retry(monkeypatch):
    torch.random.default_generator.manual_seed(1777)
    config = DistilBertConfig(vocab_size=64, max_position_embeddings=32,
                              n_layers=2, dim=16, hidden_dim=32, n_heads=2,
                              dropout=0.0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = DistilBertForMaskedLM(config).eval()
    ids = list(range(10))
    second = ids.copy()
    second[7] = 15
    rows = [{"id": "toy-sugar", "input_ids": ids}, {"id": "toy-lemon", "input_ids": second}]

    class Tokenizer:
        def convert_ids_to_tokens(self, values):
            return [str(value) for value in values]

    def fail(*_args):
        raise ValueError("fabricated unconstructible dose")

    monkeypatch.setattr(study, "matched_sources", fail)
    events = []
    with pytest.raises(ValueError, match="unconstructible"):
        study.run_cases(model, rows, sweet_id=1, sour_id=2, tokenizer=Tokenizer(), emit=events.append)
    assert [e["event"] for e in events].count("forward_start") == 4
    assert [e["event"] for e in events].count("forward_complete") == 4
    assert events[-1]["event"] == "dose_unavailable"
    assert events[-1]["reason"] == "fabricated unconstructible dose"
    assert not study.previous.has_hooks(model)
