"""No-model admission and direct-owner boundary checks."""
from datetime import datetime, timezone
from pathlib import Path
import json

import pytest

import flan_weight_edit_owner as owner


def witness():
    return {"schema": "flan-weight-edit-scope-v1", "case_sha256": owner.CASE_SHA,
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": "a" * 40, "manifest_sha256": "b" * 64,
            "known_public_comparisons_clear": True, "no_unreserved_relevant_inputs_reported": True,
            "no_additional_relevant_protected_pool_reported": True,
            "sentiment_pool": {"status": "unformed", "custodian": "review_broker",
                               "proposal_sha256": owner.CASE_SHA},
            "known_injection_cases_sha256": owner.KNOWN_INJECTION_SHA,
            "source_and_manifest_reviewed": True}


def test_scope_rejects_stale_unreviewed_or_unchecked_committed_pool():
    valid = witness()
    with pytest.raises(ValueError, match="current committed"):
        owner.validate_scope(valid, source="a" * 40, manifest_sha="b" * 64)
    owner.validate_scope(valid, source="a" * 40, manifest_sha="b" * 64, allow_synthetic_uncommitted=True)
    valid["sentiment_pool"].update({"status": "formed-uncommitted", "proposal_in_predraw_exclusions": True})
    with pytest.raises(ValueError, match="current committed"):
        owner.validate_scope(valid, source="a" * 40, manifest_sha="b" * 64)
    owner.validate_scope(valid, source="a" * 40, manifest_sha="b" * 64, allow_synthetic_uncommitted=True)
    valid["sentiment_pool"]["status"] = "committed"
    with pytest.raises(ValueError, match="custody"):
        owner.validate_scope(valid, source="a" * 40, manifest_sha="b" * 64, allow_synthetic_uncommitted=True)
    valid["sentiment_pool"].update({"commitment_sha256": owner.POOL_COMMITMENT, "card_sha256": owner.POOL_CARD_SHA,
                                    "card_revision": "e" * 40, "comparison_sha256": "f" * 64,
                                    "comparison_path": str(Path.cwd() / "receipt.json"), "authorized_comparison": True,
                                    "selection_card_revision": "e" * 40, "selection_card_sha256": owner.SELECTION_CARD_SHA,
                                    "independent_replay_sha256": "f" * 64, "independent_replay_path": str(Path.cwd() / "replay.json"),
                                    "independent_screen_reviewed": True,
                                    "no_variable_word_collision": True, "no_prefix_collision": True})
    owner.validate_scope(valid, source="a" * 40, manifest_sha="b" * 64)
    valid["observed_utc"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="fresh"):
        owner.validate_scope(valid, source="a" * 40, manifest_sha="b" * 64)


def test_source_identity_requires_all_fixed_files():
    manifest = {"source_revision": "a" * 40, "sources": {}}
    with pytest.raises(ValueError, match="source inventory"):
        owner.validate_source_inventory(manifest)


def test_manifest_binds_fixed_runtime_and_contract():
    manifest = {"runtime": dict(owner.RUNTIME), "contract": dict(owner.CONTRACT)}
    owner.validate_manifest_contract(manifest)
    manifest["contract"] = {**owner.CONTRACT, "preserve_denormals": 1}
    with pytest.raises(ValueError, match="contract"):
        owner.validate_manifest_contract(manifest)


def test_committed_scope_binds_exact_card_and_aggregate_receipt(tmp_path, monkeypatch):
    valid = witness()
    card = {"commitment": "c" * 64}
    card_raw = json.dumps(card).encode()
    import hashlib
    card_sha = hashlib.sha256(card_raw).hexdigest()
    selection_raw = json.dumps({"commitment": "d" * 64, "pool_commitment": "c" * 64}).encode()
    selection_sha = hashlib.sha256(selection_raw).hexdigest()
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps({"schema": "flan-t5-sentiment-independent-tokenizer-replay-v1",
                                       "status": "consistent", "pool_commitment": "c" * 64,
                                       "selection_commitment": "d" * 64, "token_rows": 96}), encoding="utf8")
    receipt = {"schema": "flan-weight-edit-pool-comparison-v1", "custodian": "schema",
               "proposal_sha256": owner.CASE_SHA, "card_sha256": card_sha,
               "selection_card_sha256": selection_sha,
               "commitment_sha256": "c" * 64, "eos_rule": "strip-one-final-eos",
               "no_variable_word_collision": True, "no_prefix_collision": True}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf8")
    valid["sentiment_pool"].update({"status": "committed", "commitment_sha256": "c" * 64,
                                    "card_revision": "e" * 40, "card_sha256": card_sha,
                                    "selection_card_revision": "e" * 40, "selection_card_sha256": selection_sha,
                                    "independent_replay_sha256": owner.digest(replay_path), "independent_replay_path": str(replay_path),
                                    "comparison_path": str(path), "comparison_sha256": owner.digest(path),
                                    "authorized_comparison": True, "no_variable_word_collision": True,
                                    "no_prefix_collision": True})
    monkeypatch.setattr(owner, "git_blob", lambda repo, revision, name: card_raw if name == owner.POOL_CARD_PATH else selection_raw)
    owner.validate_comparison(valid, tmp_path)
    receipt["no_prefix_collision"] = False
    path.write_text(json.dumps(receipt), encoding="utf8")
    valid["sentiment_pool"]["comparison_sha256"] = owner.digest(path)
    with pytest.raises(ValueError, match="does not clear"):
        owner.validate_comparison(valid, tmp_path)


def test_fixed_phase_claim_cannot_be_reused_after_uncertainty(tmp_path):
    first = owner.claim_phase(tmp_path, "development", "a" * 64, tmp_path / "one")
    assert first.is_file()
    with pytest.raises(FileExistsError):
        owner.claim_phase(tmp_path, "development", "a" * 64, tmp_path / "another")
    assert owner.claim_phase(tmp_path, "evaluation", "a" * 64, tmp_path / "evaluation").is_file()


def test_supervised_deadline_counts_pre_grant_time():
    assert owner.remaining_timeout(2700, 120) == 2580
    with pytest.raises(TimeoutError):
        owner.remaining_timeout(2700, 2700)


def test_real_phases_have_the_same_fixed_parent(tmp_path):
    owner.validate_attempt_path(tmp_path / "fixed-attempt" / "development", "development", tmp_path)
    owner.validate_attempt_path(tmp_path / "fixed-attempt" / "evaluation", "evaluation", tmp_path)
    with pytest.raises(ValueError, match="attempt path"):
        owner.validate_attempt_path(tmp_path / "other" / "evaluation", "evaluation", tmp_path)


def test_temporary_and_library_caches_are_confined_to_attempt(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    temp.mkdir()
    env = owner.cache_environment(temp)
    owner.validate_cache_environment(env, temp)
    monkeypatch.setattr(owner, "STUDY_ROOT", tmp_path)
    owner.validate_parent_temp({"TEMP": str(temp), "TMP": str(temp)})
    with pytest.raises(ValueError, match="outside"):
        owner.validate_cache_environment({**env, "HF_HOME": str(tmp_path)}, temp)
    with pytest.raises(ValueError, match="outside"):
        owner.validate_parent_temp({"TEMP": str(temp), "TMP": str(tmp_path.parent)})


def test_git_store_requires_bare_local_objects_without_alternates(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(owner, "STUDY_ROOT", tmp_path)
    bare = tmp_path / "publication.git"
    subprocess.check_output(["git", "init", "--bare", str(bare)])
    owner.validate_git_store(bare)
    alternate = bare / "objects" / "info" / "alternates"
    alternate.write_text(str(tmp_path.parent), encoding="utf8")
    with pytest.raises(ValueError, match="alternates"):
        owner.validate_git_store(bare)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: " + str(bare), encoding="utf8")
    with pytest.raises(ValueError, match="bare|metadata"):
        owner.validate_git_store(worktree)


def test_zero_forward_alignment_ineligibility_has_complete_phase_identity(tmp_path):
    from types import SimpleNamespace
    manifest_file = tmp_path / "manifest.json"
    scope_file = tmp_path / "scope.json"
    manifest_file.write_text("{}", encoding="utf8")
    scope_file.write_text("{}", encoding="utf8")
    args = SimpleNamespace(attempt=tmp_path, phase="development", rehearsal=False,
                           manifest=manifest_file, scope=scope_file)
    manifest = {"source_revision": "a" * 40}
    phase = {"phase": "development", "source_revision": "a" * 40, "pid": 101, "parent_pid": 100,
             "execution_status": "complete", "model_loaded": False, "scientific_status": "ineligible",
             "calls": 0, "manifest_sha256": owner.digest(manifest_file), "scope_sha256": owner.digest(scope_file),
             "case_sha256": owner.CASE_SHA}
    (tmp_path / "phase.json").write_text(json.dumps(phase), encoding="utf8")
    assert owner.phase_matches(args, manifest, 101, 100)
    phase["calls"] = False
    (tmp_path / "phase.json").write_text(json.dumps(phase), encoding="utf8")
    assert not owner.phase_matches(args, manifest, 101, 100)
