"""Supervise one source-bound, one-shot FLAN weight-edit CPU phase."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

CASE_SHA = "a118dde671351d21c519b6a61957e9478c9501bac52d55aa34e8394772e933d5"
PROTOCOL_SHA = "c06fd598e6b5d9cf8d9734c26f6711fbf53b0fbfda08f009ab691ee91dca5478"
ASSET_SHA = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
KNOWN_INJECTION_SHA = "1b21870b8d2c83a7c144ecbf2911728fcdc14a828bf2e8078fca43780df64c11"
POOL_COMMITMENT = "f79c8fdb977a459a803f3a92a45ba7a82dc2fedd1e14349432371b40da9c1671"
POOL_CARD_SHA = "f6e174110bcca104c9ec0cf3ba0abce986dbf7f7e5aadcbcda089418b6419563"
SELECTION_CARD_SHA = "05574d7c513716d946d70c67f190886a81b221cef0b17220c7f7b6135b411eca"
RUNTIME = {"python": "3.11.15", "torch": "2.8.0+cu128", "numpy": "2.4.6",
           "transformers": "4.57.3", "tokenizers": "0.22.2"}
CONTRACT = {"seed": 20261029, "grid": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
            "primary_token_id": 1627, "donor_token_id": 16242,
            "positive_label_ids": [1465, 1], "negative_label_ids": [2841, 1],
            "primary_effect_floor": 0.5, "match_absolute_tolerance": 0.10,
            "match_relative_tolerance": 0.10, "max_input_tokens": 64,
            "max_new_tokens_including_eos": 4, "vocabulary_size": 32128,
            "max_calls_cumulative": 1200, "max_evidence_bytes": 256 * 1024**2,
            "failure_reserve_bytes": 1024**2, "phase_timeout_seconds": 2700,
            "device": "cpu", "preserve_denormals": True,
            "model_revision": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
            "attempt_identity": "flan-t5-weight-edit-v2", "c_free_floor": 1207959552,
            "e_free_floor": 2416967680, "volume_check_each_model_call": True}
MANIFEST_PATH = "docs/FLAN_T5_WEIGHT_EDIT_EXECUTION_MANIFEST_V2.json"
POOL_CARD_PATH = "results/flan-t5-sentiment-intervention-v1-20261029/custody/pool-card.json"
SELECTION_CARD_PATH = "results/flan-t5-sentiment-intervention-v1-20261029/custody/selection-card.json"
STUDY_ROOT = Path(r"E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v2")
PUBLIC_INPUTS = {"docs/FLAN_T5_TARGETED_WEIGHT_EDIT_V1.md": PROTOCOL_SHA,
                 "docs/FLAN_T5_WEIGHT_EDIT_CASES_V1.json": CASE_SHA,
                 "experiments/flan_t5_known_injection_cases_v1.json": KNOWN_INJECTION_SHA,
                 "docs/FLAN_T5_WEIGHT_EDIT_STORAGE_V1.md": "b61d09688242e322e525adc74b7e9336b33f471fef20a883a724200805a0815c",
                 "docs/FLAN_T5_WEIGHT_EDIT_STORAGE_AMENDMENT_V1.md": "7314cd1950cab87382b3b530ca067c29e878cbd2273ae6d8181b10d58c024155",
                 "docs/FLAN_T5_WEIGHT_EDIT_PUBLICATION_V1.md": "f29d98e1b436298f4867b1f722fb41f184cb761b97b2914139f23bf3d910a660",
                 "docs/FLAN_T5_WEIGHT_EDIT_RESOURCE_SUCCESSOR_V2.md": "210716fe1c2ce99338cafab47ef3d1789e75e4e67f87be71909821c4f28ca3ea",
                 "results/flan-t5-weight-edit-cpu-v1-20261029/development-failure/README.md": "8b62c0adc2f80eb8a7a321eb34c8b076d9613f9add5a0b69eec1360ad0bebc83",
                 "results/flan-t5-weight-edit-cpu-v1-20261029/development-failure/independent-process-audit-v1.json": "efc379b31781c36775aa5eb6588f0c93d52a79f613108b4c00ac101c01c466cf"}
RUNTIME_FILES = ("flan_weight_edit_owner.py", "flan_weight_edit_driver.py", "flan_weight_edit_cases.py",
                 "flan_weight_edit_numerical.py", "flan_weight_edit_guard.py", "flan_weight_edit_journal.py",
                 "flan_weight_edit_worker.py", "flan_weight_edit_decision.py", "flan_weight_edit_study.py",
                 "flan_weight_edit_session.py", "flan_weight_edit_handoff.py", "flan_weight_edit_results.py")
TEST_FILES = ("flan_weight_edit_owner_test.py", "flan_weight_edit_case_test.py", "flan_weight_edit_numerical_test.py",
              "flan_weight_edit_guard_test.py", "flan_weight_edit_journal_test.py", "flan_weight_edit_worker_test.py",
              "flan_weight_edit_decision_test.py", "flan_weight_edit_study_test.py", "flan_weight_edit_session_test.py",
              "flan_weight_edit_handoff_test.py", "flan_weight_edit_results_test.py", "flan_weight_edit_driver_test.py",
              "flan_weight_edit_conftest.py")
LOG_LIMIT = 65536


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
    with Path(path).open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short receipt write")
        stream.flush()
        os.fsync(stream.fileno())


def _hex(value, length):
    return type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)


def validate_scope(record, *, source, manifest_sha, allow_synthetic_uncommitted=False):
    if (record.get("schema") != "flan-weight-edit-scope-v1" or record.get("case_sha256") != CASE_SHA
            or record.get("source_revision") != source or record.get("manifest_sha256") != manifest_sha
            or record.get("known_injection_cases_sha256") != KNOWN_INJECTION_SHA
            or record.get("known_public_comparisons_clear") is not True
            or record.get("no_unreserved_relevant_inputs_reported") is not True
            or record.get("no_additional_relevant_protected_pool_reported") is not True
            or record.get("source_and_manifest_reviewed") is not True):
        raise ValueError("scoped public inventory or source review differs")
    try:
        instant = datetime.fromisoformat(record["observed_utc"])
        age = (datetime.now(timezone.utc) - instant).total_seconds()
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("scope observation time differs") from exc
    if not 0 <= age <= 900:
        raise ValueError("scope observation is not fresh")
    pool = record.get("sentiment_pool", {})
    if pool.get("custodian") != "review_broker" or pool.get("proposal_sha256") != CASE_SHA:
        raise ValueError("sentiment custody scope differs")
    if not allow_synthetic_uncommitted and (pool.get("status") != "committed"
            or pool.get("commitment_sha256") != POOL_COMMITMENT or pool.get("card_sha256") != POOL_CARD_SHA
            or pool.get("selection_card_sha256") != SELECTION_CARD_SHA):
        raise ValueError("current committed sentiment custody is required")
    if pool.get("status") == "committed":
        if (not _hex(pool.get("commitment_sha256"), 64) or not _hex(pool.get("card_sha256"), 64)
                or not _hex(pool.get("card_revision"), 40) or not _hex(pool.get("comparison_sha256"), 64)
                or not _hex(pool.get("selection_card_revision"), 40) or not _hex(pool.get("selection_card_sha256"), 64)
                or not _hex(pool.get("independent_replay_sha256"), 64)
                or not isinstance(pool.get("independent_replay_path"), str) or not Path(pool["independent_replay_path"]).is_absolute()
                or not isinstance(pool.get("comparison_path"), str) or not Path(pool["comparison_path"]).is_absolute()
                or pool.get("authorized_comparison") is not True
                or pool.get("independent_screen_reviewed") is not True
                or pool.get("no_variable_word_collision") is not True or pool.get("no_prefix_collision") is not True):
            raise ValueError("committed sentiment custody comparison is unavailable")
    elif pool.get("status") == "formed-uncommitted":
        if pool.get("proposal_in_predraw_exclusions") is not True:
            raise ValueError("sentiment custody pre-draw exclusion is unavailable")
    elif pool.get("status") != "unformed":
        raise ValueError("sentiment custody status is unknown")


def validate_comparison(record, repo):
    pool = record["sentiment_pool"]
    if pool["status"] != "committed":
        return
    card_raw = git_blob(repo, pool["card_revision"], POOL_CARD_PATH)
    if hashlib.sha256(card_raw).hexdigest() != pool["card_sha256"]:
        raise ValueError("committed sentiment card differs")
    card = json.loads(card_raw)
    selection = git_blob(repo, pool["selection_card_revision"], SELECTION_CARD_PATH)
    if hashlib.sha256(selection).hexdigest() != pool["selection_card_sha256"]:
        raise ValueError("sentiment selection card differs")
    selection_record = json.loads(selection)
    replay_path = Path(pool["independent_replay_path"])
    if not replay_path.is_file() or digest(replay_path) != pool["independent_replay_sha256"]:
        raise ValueError("independent sentiment screen replay differs")
    replay = read_json(replay_path)
    if (replay.get("schema") != "flan-t5-sentiment-independent-tokenizer-replay-v1"
            or replay.get("status") != "consistent" or replay.get("pool_commitment") != pool["commitment_sha256"]
            or replay.get("selection_commitment") != selection_record.get("commitment")
            or selection_record.get("pool_commitment") != pool["commitment_sha256"]
            or type(replay.get("token_rows")) is not int or replay["token_rows"] != 96):
        raise ValueError("independent sentiment replay does not bind both cards")
    path = Path(pool["comparison_path"])
    if not path.is_file() or digest(path) != pool["comparison_sha256"]:
        raise ValueError("custodian comparison receipt differs")
    comparison = read_json(path)
    if (card.get("commitment") != pool["commitment_sha256"]
            or comparison.get("schema") != "flan-weight-edit-pool-comparison-v1"
            or comparison.get("custodian") != "schema" or comparison.get("proposal_sha256") != CASE_SHA
            or comparison.get("card_sha256") != pool["card_sha256"]
            or comparison.get("commitment_sha256") != pool["commitment_sha256"]
            or comparison.get("selection_card_sha256") != pool["selection_card_sha256"]
            or comparison.get("eos_rule") != "strip-one-final-eos"
            or comparison.get("no_variable_word_collision") is not True
            or comparison.get("no_prefix_collision") is not True):
        raise ValueError("custodian comparison does not clear the committed pool")


def validate_source_inventory(manifest):
    expected = set(RUNTIME_FILES + TEST_FILES)
    sources = manifest.get("sources")
    if not _hex(manifest.get("source_revision"), 40) or not isinstance(sources, dict) or set(sources) != expected:
        raise ValueError("source inventory differs")
    for name, item in sources.items():
        expected_path = source_git_path(name)
        if (not isinstance(item, dict) or item.get("git_path") != expected_path
                or not _hex(item.get("sha256"), 64)):
            raise ValueError("source inventory identity differs")


def validate_manifest_contract(manifest):
    # Serialized comparison also distinguishes JSON booleans from numeric 0/1.
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(manifest.get("runtime")) != canonical(RUNTIME) or canonical(manifest.get("contract")) != canonical(CONTRACT):
        raise ValueError("fixed runtime or scientific contract differs")


def source_git_path(name):
    if name in RUNTIME_FILES:
        return "studies/flan_weight_edit_v2/scripts/" + name
    if name in TEST_FILES:
        return "studies/flan_weight_edit_v2/tests/flan_weight_edit/" + ("conftest.py" if name == "flan_weight_edit_conftest.py" else name)
    raise ValueError("unknown fixed source file")


def local_source_path(directory, name):
    flat = Path(directory) / name
    if flat.is_file():
        return flat
    if name in TEST_FILES:
        return Path(directory).parent / "tests" / "flan_weight_edit" / ("conftest.py" if name == "flan_weight_edit_conftest.py" else name)
    return flat


def git_blob(repo, revision, path):
    if not _hex(revision, 40) or path.startswith("-"):
        raise ValueError("invalid Git identity")
    return subprocess.check_output(["git", "--no-replace-objects", "show", revision + ":" + path], cwd=repo, timeout=15)


def validate_git_store(repo):
    original = Path(repo)
    if original.is_symlink() or getattr(original.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("Git repository is a reparse point")
    repo = original.resolve()
    if not repo.is_relative_to(STUDY_ROOT.resolve()):
        raise ValueError("Git repository is outside the E: study root")
    def git(*args):
        return subprocess.check_output(["git", "--no-replace-objects", "rev-parse", *args],
                                       cwd=repo, timeout=15).decode("utf8").strip()
    if git("--is-bare-repository") != "true":
        raise ValueError("the publication Git repository must be bare")
    # Bare repositories return relative paths, avoiding MSYS /e/... path aliases.
    gitdir = Path(git("--git-dir"))
    common = Path(git("--git-common-dir"))
    objects = Path(git("--git-path", "objects"))
    gitdir = (repo / gitdir).resolve() if not gitdir.is_absolute() else gitdir.resolve()
    common = (repo / common).resolve() if not common.is_absolute() else common.resolve()
    objects = (repo / objects).resolve() if not objects.is_absolute() else objects.resolve()
    if gitdir != repo or common != repo or objects != repo / "objects":
        raise ValueError("Git metadata or object store is not the E: bare repository")
    for path in (repo, objects):
        if not path.is_dir() or path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Git store is missing or is a reparse point")
    if any((objects / "info" / name).exists() for name in ("alternates", "http-alternates")):
        raise ValueError("Git object alternates are not allowed")


def preflight(args, *, require_fresh_scope=True):
    validate_git_store(args.repo)
    manifest_raw = args.manifest.read_bytes()
    manifest = read_json(args.manifest)
    validate_source_inventory(manifest)
    validate_manifest_contract(manifest)
    if (manifest.get("schema") != "flan-weight-edit-manifest-v1"
            or manifest.get("case_sha256") != CASE_SHA or manifest.get("protocol_sha256") != PROTOCOL_SHA
            or manifest.get("asset_manifest_sha256") != ASSET_SHA
            or git_blob(args.repo, args.manifest_revision, MANIFEST_PATH) != manifest_raw):
        raise ValueError("published manifest differs")
    parents = subprocess.check_output(["git", "--no-replace-objects", "rev-list", "--parents", "-n", "1",
                                       args.manifest_revision], cwd=args.repo, timeout=15).decode("ascii").split()
    if parents != [args.manifest_revision, manifest["source_revision"]]:
        raise ValueError("manifest must be the source's sole-parent successor")
    for name, item in manifest["sources"].items():
        local = local_source_path(args.source_dir, name).read_bytes()
        if hashlib.sha256(local).hexdigest() != item["sha256"] or git_blob(
                args.repo, manifest["source_revision"], item["git_path"]) != local:
            raise ValueError("source bytes differ: " + name)
    for path, expected_sha in PUBLIC_INPUTS.items():
        if hashlib.sha256(git_blob(args.repo, manifest["source_revision"], path)).hexdigest() != expected_sha:
            raise ValueError("frozen public input differs: " + path)
    if hashlib.sha256(args.cases.read_bytes()).hexdigest() != CASE_SHA:
        raise ValueError("case bundle differs")
    if digest(args.assets / "asset-manifest.json") != ASSET_SHA:
        raise ValueError("asset manifest differs")
    scope = read_json(args.scope)
    if require_fresh_scope:
        validate_scope(scope, source=manifest["source_revision"],
                       manifest_sha=hashlib.sha256(manifest_raw).hexdigest(),
                       allow_synthetic_uncommitted=args.rehearsal)
    validate_comparison(scope, args.repo)
    return manifest


def volumes(output, source, observer=None):
    if (os.name != "nt" or not Path(output).resolve().is_relative_to(STUDY_ROOT.resolve())
            or Path(output).resolve().drive.upper() != "E:" or Path(source).resolve().drive.upper() != "C:"):
        raise ValueError("unexpected output or source volume")
    out_free = shutil.disk_usage(output).free
    source_free = shutil.disk_usage(source).free
    passed = out_free >= 2 * 1024**3 + 257 * 1024**2 and source_free >= 1024**3 + 128 * 1024**2
    observation = {"output_free": out_free, "source_free": source_free,
                   "source_margin_scope": "heuristic; not an OS reservation"}
    if observer is not None:
        observer(observation, passed)
    if not passed:
        raise ValueError(f"insufficient output or source disk headroom: C_free={source_free} E_free={out_free}")
    return observation


def cache_environment(temp):
    temp = Path(temp).resolve()
    caches = {"HF_HOME": "hf", "HF_HUB_CACHE": "hf-hub", "HF_ASSETS_CACHE": "hf-assets",
              "HF_DATASETS_CACHE": "hf-datasets", "HF_MODULES_CACHE": "hf-modules",
              "HUGGINGFACE_HUB_CACHE": "hf-hub-legacy", "HUGGINGFACE_ASSETS_CACHE": "hf-assets-legacy",
              "TRANSFORMERS_CACHE": "transformers", "TORCH_HOME": "torch",
              "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "mpl"}
    result = {"TEMP": str(temp), "TMP": str(temp)}
    for name, directory in caches.items():
        path = temp / "library-cache" / directory
        path.mkdir(parents=True, exist_ok=False)
        result[name] = str(path)
    return result


def validate_cache_environment(environment, temp):
    temp = Path(temp).resolve()
    for name in ("TEMP", "TMP", "HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_DATASETS_CACHE",
                 "HF_MODULES_CACHE", "HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_ASSETS_CACHE",
                 "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
        raw = environment.get(name)
        if not isinstance(raw, str):
            raise ValueError("missing redirected temporary or cache path")
        path = Path(raw)
        if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
                or getattr(path.lstat(), "st_file_attributes", 0) & 0x400
                or not path.resolve().is_relative_to(temp)):
            raise ValueError("temporary or cache path is outside the E: attempt")


def validate_parent_temp(environment):
    for name in ("TEMP", "TMP"):
        raw = environment.get(name)
        if not isinstance(raw, str):
            raise ValueError("parent temporary path is missing")
        path = Path(raw)
        if (not path.is_absolute() or not path.is_dir() or path.is_symlink()
                or not path.resolve().is_relative_to(STUDY_ROOT.resolve())):
            raise ValueError("parent temporary path is outside the E: study root")


def evidence_bytes(args, *, exclude=()):
    root = args.attempt.parent.resolve()
    if root.is_symlink() or getattr(root.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("evidence root is a reparse point")
    omitted = {Path(item).resolve() for item in exclude}
    total = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in (*dirs, *files):
            path = Path(current) / name
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("evidence path contains a reparse point")
        for name in files:
            path = Path(current) / name
            if path.resolve() not in omitted:
                total += path.stat().st_size
    external = {args.scope.resolve(), args.manifest.resolve(), args.cases.resolve()}
    if not args.rehearsal:
        for phase in ("development", "evaluation"):
            claim = STUDY_ROOT / "claims" / (phase + ".json")
            if claim.is_file():
                external.add(claim.resolve())
    scope = read_json(args.scope)
    if scope.get("sentiment_pool", {}).get("status") == "committed":
        external.add(Path(scope["sentiment_pool"]["comparison_path"]).resolve())
        external.add(Path(scope["sentiment_pool"]["independent_replay_path"]).resolve())
    if args.handoff is not None:
        external.add(args.handoff.resolve())
        for item in read_json(args.handoff).get("artifacts", {}).values():
            external.add(Path(item["local_path"]).resolve())
    for path in external:
        if not path.is_relative_to(root) and path not in omitted:
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("external evidence path is a reparse point")
            total += path.stat().st_size
    return total


def validate_attempt_path(attempt, phase, root=STUDY_ROOT):
    expected = Path(root) / "fixed-attempt" / phase
    if Path(attempt).resolve() != expected.resolve():
        raise ValueError("fixed phase attempt path differs")


def claim_phase(root, phase, manifest_sha, attempt):
    directory = Path(root) / "claims"
    directory.mkdir(exist_ok=True)
    if directory.is_symlink() or getattr(directory.lstat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("phase-claim directory is a reparse point")
    path = directory / (phase + ".json")
    write_new(path, {"phase": phase, "manifest_sha256": manifest_sha, "attempt": str(attempt),
                     "scope": "fixed one-shot phase claim; retained on incomplete execution"})
    return path


def remaining_timeout(deadline, now):
    remaining = deadline - now
    if remaining <= 0:
        raise TimeoutError("supervised phase deadline expired")
    return remaining


def phase_matches(args, manifest, child_pid, operator_pid):
    try:
        record = read_json(args.attempt / "phase.json")
        if (record.get("phase") != args.phase or record.get("source_revision") != manifest["source_revision"]
                or type(record.get("pid")) is not int or record["pid"] != child_pid
                or type(record.get("parent_pid")) is not int or record["parent_pid"] != operator_pid):
            return False
        if args.rehearsal:
            return record.get("execution_status") == "rehearsal" and record.get("model_loaded") is False
        if (record.get("execution_status") != "complete" or record.get("manifest_sha256") != digest(args.manifest)
                or record.get("scope_sha256") != digest(args.scope) or record.get("case_sha256") != CASE_SHA):
            return False
        if record.get("model_loaded") is False:
            return (record.get("scientific_status") == "ineligible" and type(record.get("calls")) is int
                    and record.get("calls") == 0)
        return all(record.get(name + "_sha256") == digest(args.attempt / path) for name, path in
                   (("result", "result.json"), ("events", "evidence/events.jsonl"), ("logits", "evidence/logits.f32")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def dependency_site(venv):
    path = Path(venv).resolve(strict=True)
    entries = {}
    for line in (path / "pyvenv.cfg").read_text(encoding="utf8").splitlines():
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            key = key.lower()
            if key in entries:
                raise ValueError("duplicate venv entry")
            entries[key] = value
    if (Path(entries.get("home", "")).resolve() != Path(sys.base_prefix).resolve()
            or entries.get("implementation") != "CPython"
            or entries.get("version_info") != ".".join(map(str, sys.version_info[:3]))
            or entries.get("include-system-site-packages", "").lower() != "false"):
        raise ValueError("venv does not match the direct base interpreter")
    site = path / "Lib" / "site-packages"
    if not site.is_dir() or site.is_symlink():
        raise ValueError("dependency site differs")
    return site.resolve()


def _drain(pipe, path, state):
    try:
        with Path(path).open("xb") as stream:
            while True:
                block = pipe.read(16384)
                if not block:
                    break
                keep = block[:max(0, LOG_LIMIT - state["bytes"])]
                if len(keep) != len(block):
                    state["truncated"] = True
                if keep and stream.write(keep) != len(keep):
                    raise OSError("short child log write")
                state["bytes"] += len(keep)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        state["error"] = type(exc).__name__
    finally:
        pipe.close()
        state["finished"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "source-dir", "manifest", "scope", "cases", "assets", "venv", "attempt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-revision", required=True)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--grant-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Keep Git's object store and temporary behavior bound to the explicit E: repo.
    for key in tuple(os.environ):
        if key.upper().startswith("GIT_"):
            os.environ.pop(key, None)
    if (not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode or not sys.pycache_prefix
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()
            or not Path(sys.pycache_prefix).is_dir() or any(Path(sys.pycache_prefix).iterdir())
            or not Path(sys.pycache_prefix).resolve().is_relative_to(STUDY_ROOT.resolve())):
        raise ValueError("direct base Python with isolated empty bytecode prefix required")
    if any(not getattr(args, name.replace("-", "_")).is_absolute()
           for name in ("repo", "source-dir", "manifest", "scope", "cases", "assets", "venv", "attempt")):
        raise ValueError("all owner paths must be absolute")
    if (not args.repo.resolve().is_relative_to(STUDY_ROOT.resolve())
            or (not args.rehearsal and any(not path.resolve().is_relative_to(STUDY_ROOT.resolve())
                                           for path in (args.manifest, args.scope)))):
        raise ValueError("Git store and admission evidence must be on the E: study root")
    if Path(__file__).resolve() != (args.source_dir / "flan_weight_edit_owner.py").resolve():
        raise ValueError("executing owner differs from the frozen source path")
    if not args.child:
        validate_parent_temp(os.environ)
    if args.phase == "evaluation" and (args.handoff is None or not args.handoff.is_absolute()):
        raise ValueError("evaluation requires an absolute published handoff")
    if not args.rehearsal:
        validate_attempt_path(args.attempt, args.phase)
    if args.child:
        grant = sys.stdin.buffer.read(33)
        if len(grant) != 32 or hashlib.sha256(grant).hexdigest() != args.grant_sha:
            raise ValueError("direct child grant differs")
        manifest = preflight(args)
        scope_sha = digest(args.scope)
        volumes(args.attempt, args.source_dir)
        validate_cache_environment(os.environ, args.attempt / "temp")
        if not args.rehearsal:
            claim = read_json(STUDY_ROOT / "claims" / (args.phase + ".json"))
            if (claim.get("phase") != args.phase or claim.get("manifest_sha256") != digest(args.manifest)
                    or claim.get("attempt") != str(args.attempt)):
                raise ValueError("fixed phase claim differs")
        site = dependency_site(args.venv)
        sys.path.insert(0, str(args.source_dir))
        sys.path.append(str(site))
        print(json.dumps({"pid": os.getpid(), "parent_pid": os.getppid(), "site": str(site),
                          "cache_paths": {name: os.environ[name] for name in ("TEMP", "TMP", "HF_HOME",
                          "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_DATASETS_CACHE", "HF_MODULES_CACHE",
                          "HUGGINGFACE_HUB_CACHE", "HUGGINGFACE_ASSETS_CACHE", "TRANSFORMERS_CACHE",
                          "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR")}}), flush=True)
        import flan_weight_edit_driver as driver
        driver.run(args, manifest, site)
        if preflight(args, require_fresh_scope=False) != manifest or digest(args.scope) != scope_sha or any(Path(sys.pycache_prefix).iterdir()):
            raise ValueError("source or bytecode prefix changed during phase")
        return
    if args.grant_sha is not None or args.attempt.exists() or not args.attempt.parent.is_dir():
        raise ValueError("phase attempt is not fresh")
    manifest = preflight(args)
    scope_sha = digest(args.scope)
    disk = volumes(args.attempt.parent, args.source_dir)
    claim = None if args.rehearsal else claim_phase(STUDY_ROOT, args.phase, digest(args.manifest), args.attempt)
    args.attempt.mkdir()
    prefix = args.attempt / "pycache"
    temp = args.attempt / "temp"
    prefix.mkdir()
    temp.mkdir()
    cache_paths = cache_environment(temp)
    validate_cache_environment(cache_paths, temp)
    write_new(args.attempt / "operator-start.json", {"phase": args.phase, "source": manifest["source_revision"],
              "scope_sha256": scope_sha, "disk": disk, "rehearsal": args.rehearsal,
              "parent_temp": {name: os.environ[name] for name in ("TEMP", "TMP")}, "child_cache_paths": cache_paths,
              "claim_sha256": None if claim is None else digest(claim)})
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.upper().startswith("GIT_") or key.upper() in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"):
            environment.pop(key, None)
    environment.update({"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_CBWR": "COMPATIBLE", "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONDONTWRITEBYTECODE": "1", **cache_paths})
    grant = os.urandom(32)
    command = [sys.executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix), str(Path(__file__).resolve()),
               "--child", "--grant-sha", hashlib.sha256(grant).hexdigest(), "--phase", args.phase,
               "--manifest-revision", args.manifest_revision]
    for name in ("repo", "source-dir", "manifest", "scope", "cases", "assets", "venv", "attempt"):
        command.extend(("--" + name, str(getattr(args, name.replace("-", "_")))))
    if args.handoff is not None:
        command.extend(("--handoff", str(args.handoff)))
    if args.rehearsal:
        command.append("--rehearsal")
    child, code, timed_out, error, error_detail = None, None, False, None, None
    threads, logs = [], {name: {"bytes": 0, "truncated": False, "error": None, "finished": False}
                         for name in ("stdout", "stderr")}
    started = time.monotonic()
    deadline = started + 2700
    try:
        child = subprocess.Popen(command, cwd=args.source_dir, env=environment, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for name in ("stdout", "stderr"):
            thread = threading.Thread(target=_drain, args=(getattr(child, name), args.attempt / (name + ".log"), logs[name]), daemon=True)
            threads.append(thread)
            thread.start()
        write_new(args.attempt / "operator-spawn.json", {"pid": child.pid, "parent_pid": os.getpid()})
        if preflight(args) != manifest or digest(args.scope) != scope_sha:
            raise ValueError("source or custody changed before child grant")
        volumes(args.attempt, args.source_dir)
        if evidence_bytes(args) + 1024**2 > 256 * 1024**2:
            raise ValueError("cumulative evidence cap would be exceeded")
        remaining_timeout(deadline, time.monotonic())
        if child.stdin.write(grant) != 32:
            raise OSError("short grant write")
        child.stdin.flush()
        child.stdin.close()
        try:
            code = child.wait(timeout=remaining_timeout(deadline, time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            code = child.wait(timeout=15)
    except BaseException as exc:
        error = type(exc).__name__
        error_detail = str(exc)[:1000]
        if isinstance(exc, TimeoutError):
            timed_out = True
        if child is not None and child.poll() is None:
            child.kill()
            code = child.wait(timeout=15)
    finally:
        for thread in threads:
            thread.join(timeout=15)
        for name in ("stdout", "stderr"):
            path = args.attempt / (name + ".log")
            logs[name]["sha256"] = digest(path) if path.is_file() else None
            if path.is_file() and path.stat().st_size != logs[name]["bytes"]:
                logs[name]["error"] = "length-mismatch"
        drained = len(threads) == 2 and all(not thread.is_alive() for thread in threads) and all(
            item["finished"] and not item["truncated"] and item["error"] is None for item in logs.values())
        phase_file = args.attempt / "phase.json"
        phase_sha = digest(phase_file) if phase_file.is_file() else None
        identity = False
        try:
            with (args.attempt / "stdout.log").open("rb") as stream:
                first = json.loads(stream.readline(4097))
            identity = first.get("pid") == child.pid and first.get("parent_pid") == os.getpid()
        except (OSError, ValueError, AttributeError):
            pass
        try:
            final_evidence_bytes = evidence_bytes(args)
            within_cap = final_evidence_bytes + 1024**2 <= 256 * 1024**2
        except (OSError, ValueError, KeyError):
            final_evidence_bytes, within_cap = None, False
        phase_consistent = child is not None and phase_matches(args, manifest, child.pid, os.getpid())
        good = (code == 0 and not timed_out and error is None and drained and identity and phase_sha is not None
                and phase_consistent and not any(prefix.iterdir()) and within_cap)
        write_new(args.attempt / "operator-exit.json", {"status": "complete_candidate" if good else "incomplete",
                  "direct_child_pid": child.pid if child else None, "exit_code": code, "reaped": code is not None,
                  "operator_pid": os.getpid(),
                  "timed_out": timed_out, "error": error, "error_detail": error_detail, "logs": logs, "drained": drained,
                  "identity_matches": identity, "phase_sha256": phase_sha, "phase_consistent": phase_consistent,
                  "cumulative_evidence_bytes_before_exit_record": final_evidence_bytes, "within_evidence_cap": within_cap,
                  "supervised_elapsed_seconds": time.monotonic() - started,
                  "scope": "cooperative direct-child observation; not a whole-process-tree proof"})
    if not good:
        raise RuntimeError("phase is incomplete; preserve its partial evidence")
    print(json.dumps({"status": "complete_candidate", "phase_sha256": phase_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
