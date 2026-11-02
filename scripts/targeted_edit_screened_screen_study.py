"""Supervised baseline-only screen for the published screened-edit study.

This owner requires the published, independently reviewed discovery record and
the complete 44-row public reservation. It never applies an edit. A completed
screen remains private until its full raw rows, selection and clean process
record are published and independently reviewed. Clocks, execution and source
custody are cooperative attestations under a trusted quiescent checkout.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import types
import uuid
import weakref


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "scripts/targeted_edit_screened_study.py"
prior = types.ModuleType("_screened_discovery_owner")
prior.__file__ = str(PREVIOUS)
exec(compile(PREVIOUS.read_bytes(), str(PREVIOUS), "exec"), prior.__dict__)
base = prior.base
base.SCRIPT = "scripts/targeted_edit_screened_screen_study.py"
base.MANIFEST = "experiments/targeted_edit_screened_screen_run_v1.json"
base.EXTRA = (*base.EXTRA, base.SCRIPT, "experiments/targeted_edit_screened_run_v1.json")

DISCOVERY_DIR = "results/targeted-edit-screened-matched-v1-20261029/development"
DISCOVERY_MANIFEST = "experiments/targeted_edit_screened_run_v1.json"
DISCOVERY_MANIFEST_SHA = "4002c5a539a93c315cbcfd2f7170ec39d702f84be38df1af160379aba4e10cb7"
DISCOVERY_PHASE_SHA = "80d392a5f7bdfad8e730b8d40087421c1e990151af4efcf9504fd63e73573cd7"
CANDIDATE_SHA = "f038db0b27f68a80fc3fc11374971bf8acd6b47812ab112d760520acc7ff5166"
_ACTIVE_ATTEMPT = None
_ATTEMPT_OWNED = False


def _bounded_committed_inventory(launch, checkout, revision, directory, inventory):
    if type(inventory) is not dict or len(inventory) != 103:
        raise base.StudyError("discovery inventory count differs")
    relative = directory.relative_to(checkout).as_posix()
    tree = launch._git(checkout, "--literal-pathspecs", "ls-tree", "-rz", "--full-tree", revision, "--", relative)
    git_entries = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, sha = header.decode("ascii").split(" ")
        name = name.decode("utf-8")
        if mode != "100644" or kind != "blob" or not name.startswith(relative + "/"):
            raise base.StudyError("discovery Git inventory differs")
        git_entries[name[len(relative) + 1:]] = sha
    if set(git_entries) != set(inventory) | {"phase-record.json"}:
        raise base.StudyError("discovery Git inventory is incomplete or has extra files")
    total = 0
    for name, item in inventory.items():
        relative_path = PurePosixPath(name)
        if relative_path.is_absolute() or any(part in ("", ".", "..") for part in relative_path.parts):
            raise base.StudyError("discovery artifact path differs")
        size = item.get("bytes")
        if type(size) is not int or not 0 <= size <= 10_000_000:
            raise base.StudyError("discovery artifact size differs")
        total += size
        if total > 150_000_000:
            raise base.StudyError("discovery inventory exceeds its bound")
        path = launch._checked_path(directory.joinpath(*relative_path.parts))
        data = base.read(path, size)
        blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        if (len(data) != size or base.digest(data) != item.get("sha256")
                or git_entries[name] != blob):
            raise base.StudyError("discovery artifact differs from its record or committed blob")
    base.check_exact_phase_files(launch, checkout, revision, directory, inventory)


def check_discovery(launch, checkout, head, review):
    """Bind the complete public discovery record and independent review."""
    if (type(review) is not dict or review.get("manifest_sha256") != DISCOVERY_MANIFEST_SHA
            or review.get("development_record_sha256") != DISCOVERY_PHASE_SHA
            or review.get("candidate_sha256") != CANDIDATE_SHA
            or type(review.get("reviewer")) is not str or not review["reviewer"]
            or type(review.get("reviewed_commit")) is not str):
        raise base.StudyError("discovery review attestation differs")
    try:
        observed = datetime.fromisoformat(review["observed_utc"])
    except (KeyError, ValueError, TypeError) as exc:
        raise base.StudyError("discovery review time differs") from exc
    if observed.tzinfo is None or observed > datetime.now(timezone.utc):
        raise base.StudyError("discovery review time differs")
    reviewed = review["reviewed_commit"]
    base.commit_object(launch, checkout, reviewed)
    launch._git(checkout, "merge-base", "--is-ancestor", reviewed, head)
    manifest_path = launch._checked_path(checkout / DISCOVERY_MANIFEST)
    manifest_raw = base.read(manifest_path)
    if base.digest(manifest_raw) != DISCOVERY_MANIFEST_SHA:
        raise base.StudyError("discovery manifest differs")
    base.committed_file(launch, checkout, reviewed, manifest_path, manifest_raw)
    manifest = json.loads(manifest_raw)
    directory = launch._checked_path(checkout / DISCOVERY_DIR)
    phase_path = launch._checked_path(directory / "phase-record.json")
    phase_raw = base.read(phase_path)
    if base.digest(phase_raw) != DISCOVERY_PHASE_SHA:
        raise base.StudyError("discovery phase record differs")
    base.committed_file(launch, checkout, reviewed, phase_path, phase_raw)
    phase = json.loads(phase_raw)
    base.check_development_record(phase, manifest, DISCOVERY_MANIFEST_SHA, CANDIDATE_SHA)
    launch._git(checkout, "merge-base", "--is-ancestor", manifest["source_commit"], reviewed)
    base.check_reported_order(manifest["created_utc"], phase["finished_utc"], review["observed_utc"], base.utc())
    _bounded_committed_inventory(launch, checkout, reviewed, directory, phase["artifacts"])
    candidate_path = launch._checked_path(directory / "candidate/pre-evaluation-candidate.json")
    candidate_raw = base.read(candidate_path)
    if base.digest(candidate_raw) != CANDIDATE_SHA:
        raise base.StudyError("discovery candidate differs")
    candidate = json.loads(candidate_raw)
    prior.validate_phase_output(directory, "development", phase["child"], phase["artifacts"])
    base.check_candidate_binding(launch, phase_path, candidate_path.parent, candidate, phase["artifacts"])
    if (candidate["buffer_state_sha256"] != phase["child"]["buffer_state_sha256"]
            or candidate["runtime_settings"] != phase["child"]["runtime_settings"]
            or candidate["model_state_sha256"] != phase["child"]["model_state_sha256"]
            or candidate["weight_sha256"] != phase["child"]["weight_sha256"]):
        raise base.StudyError("discovery candidate and child state bindings differ")
    return {"manifest_sha256": DISCOVERY_MANIFEST_SHA, "phase_sha256": DISCOVERY_PHASE_SHA,
            "candidate_sha256": CANDIDATE_SHA, "reviewed_commit": reviewed,
            "review_observed_utc": review["observed_utc"],
            "model_state_sha256": manifest["original_model_state_sha256"],
            "weight_sha256": manifest["original_weight_sha256"],
            "buffer_state_sha256": phase["child"]["buffer_state_sha256"],
            "runtime_settings": phase["child"]["runtime_settings"], "checkpoint": manifest["checkpoint"],
            "reference_sha256": manifest["reference_sha256"]}


def _case_bytes(checkout):
    raw = base.read(base.ordinary(checkout / base.CASES))
    if base.digest(raw) != base.CASES_SHA:
        raise base.StudyError("fixed screen case bytes differ")
    return raw


def _exposure(checkout, hashes, payload, payload_sha, minimum_rows):
    cases = json.loads(_case_bytes(checkout))
    result = prior.check_public_payload(payload, payload_sha, cases, minimum_rows, include_rows=True)
    if result["own_reserved_rows"] != 44:
        raise base.StudyError("all 44 exact prompt reservations are required before screening")
    return result


def _current_inventory(launch, checkout, head, raw, sha, attestation, minimum_rows):
    result = _exposure(checkout, None, json.loads(raw), sha, minimum_rows)
    base.check_inventory_attestation(launch, checkout, head, attestation, sha, result["version"])
    return result


def _imports(imports, hashes):
    required = {"looking_glass/" + name + ".py" for name in (
        "__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
        "pretrained", "targeted_edit_cpu_v1", "targeted_edit_v2_edits", "targeted_edit_v2_owner",
        "targeted_edit_screened_selection", "targeted_edit_screened_screen")}
    if (type(imports) is not dict or not required <= set(imports)
            or any(hashes.get(name) != value for name, value in imports.items())):
        raise base.StudyError("screen child import inventory differs")


def _match_discovery_state(guard, runtime, manifest):
    """Require discovery's runtime and complete buffer identity before rows."""
    if (guard.runtime_settings != runtime or runtime != manifest["discovery"]["runtime_settings"]
            or guard.state_sha != manifest["original_model_state_sha256"]
            or guard.weight_sha != manifest["original_weight_sha256"]
            or guard.buffer_sha != manifest["discovery"]["buffer_state_sha256"]):
        raise base.StudyError("screen model state or runtime differs")


def _run_bound_screen(guard, runtime, manifest, runner, case_bytes, binding, output):
    _match_discovery_state(guard, runtime, manifest)
    return runner.screen(guard, case_bytes, binding, output)


def child(args):
    checkout, attempt = base.ordinary(args.checkout), base.ordinary(args.attempt)
    if (args.phase not in ("rehearsal", "screen")
            or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != attempt / "pycache"
            or any((attempt / "pycache").iterdir())
            or any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules)):
        raise base.StudyError("isolated screen child startup differs")
    manifest_raw = base.read(base.ordinary(checkout / base.MANIFEST))
    if base.digest(manifest_raw) != args.manifest_sha:
        raise base.StudyError("screen manifest differs in child")
    manifest = json.loads(manifest_raw)
    hashes = manifest["source_sha256"]
    for name in (base.SCRIPT, "scripts/targeted_edit_screened_study.py", "scripts/targeted_edit_projected_study.py"):
        if base.digest(base.read(base.ordinary(checkout / name))) != hashes[name]:
            raise base.StudyError("screen child owner source differs")
    boot, finder = base.finder_for(checkout, hashes)
    boot._check_search_path(checkout)
    boot._prime_parent_platform(json.loads(args.parent_platform))
    import platform
    if os.name != "nt" or platform._platform_cache:
        raise base.StudyError("platform was observed before spawn-free setup")
    win = sys.getwindowsversion()
    platform._syscmd_ver = lambda *_args, **_kwargs: ("Windows", str(win.major), f"{win.major}.{win.minor}.{win.build}")
    boot._install_process_guard()
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(Path(args.study_root).resolve(strict=True))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    available = base.available_memory()
    if available < 3 * 1024**3:
        raise base.StudyError("available physical memory is below the study admission floor")
    boot._add_dependencies(checkout, Path(args.venv).resolve(strict=True), [args.dependency])
    import torch
    from looking_glass import pretrained
    from looking_glass import targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_v2_edits as edits
    from looking_glass import targeted_edit_v2_owner as common
    from looking_glass import targeted_edit_screened_screen as screen
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False) or torch.cuda.is_initialized():
        raise base.StudyError("screen CPU numeric mode differs")
    reference = json.loads(base.read(base.ordinary(checkout / base.REFERENCE)))
    env = v1._environment()
    if v1._canonical({k: env[k] for k in edits.ENVIRONMENT_FIELDS}) != v1._canonical(
            {k: reference["environment"][k] for k in edits.ENVIRONMENT_FIELDS}):
        raise base.StudyError("screen runtime differs from the v1 reference")
    runtime = common._runtime_settings(torch)
    if (runtime["intraop_threads"] != 1 or runtime["interop_threads"] != 1
            or runtime["deterministic_algorithms"] is not True or runtime["deterministic_warn_only"] is not False
            or runtime["cpu_autocast_enabled"] is not False or runtime["inference_mode_enabled"] is not False
            or runtime["cuda_visibility"] != "-1" or runtime["cuda_initialized"] is not False
            or runtime["hf_hub_offline"] != "1" or runtime["transformers_offline"] != "1"):
        raise base.StudyError("initial screen runtime settings differ")
    case_bytes = _case_bytes(checkout)
    if manifest["case_sha256"] != base.CASES_SHA:
        raise base.StudyError("screen case binding differs")
    if args.phase == "rehearsal":
        sources = finder.verify_inventory()
        if torch.cuda.is_initialized() or any((attempt / "pycache").iterdir()):
            raise base.StudyError("screen rehearsal CPU or bytecode state differs")
        result = {"phase": "rehearsal", "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                  "manifest_sha256": args.manifest_sha, "model_disposed": False, "no_model_loaded": True,
                  "runtime_settings": runtime, "runtime": {k: env[k] for k in edits.ENVIRONMENT_FIELDS},
                  "imported_sources": sources, "finished_utc": base.utc(),
                  "scope": "no checkpoint or model forward; diagnostic only"}
        base.write(attempt / "child-result.json", base.canonical(result))
        session.attest_clean(lease)
        return

    def work():
        loaded = pretrained.load_distilgpt2(cache_dir=args.cache, allow_download=False)
        if common._runtime_settings(torch) != runtime:
            raise base.StudyError("runtime settings changed during checkpoint loading")
        model, module, weight = common._check_checkpoint(loaded, {"v1_compatibility": {"model": manifest["checkpoint"]}}, torch, v1)
        v1._verify_tokenizer(loaded.tokenizer)
        base.check_projected_tokenizer(loaded.tokenizer, json.loads(case_bytes)["cases"])
        guard = common._ModelGuard(model, module, weight, torch, v1)
        binding = {"source_commit": manifest["source_commit"], "manifest_sha256": args.manifest_sha,
                   "development_record_sha256": DISCOVERY_PHASE_SHA, "candidate_sha256": CANDIDATE_SHA}
        result = _run_bound_screen(guard, runtime, manifest, screen, case_bytes, binding, attempt / "screen")
        guard.verify()
        data = base.read(attempt / "screen/screen-selection.json")
        return ({"selection_sha256": base.digest(data), "selected_count": result["selection"]["selected_count"],
                 "formal_target_eligible": result["selection"]["formal_target_eligible"],
                 "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                 "buffer_state_sha256": guard.buffer_sha}, weakref.ref(model), weakref.ref(weight))

    result, model_ref, weight_ref = work()
    gc.collect()
    if model_ref() is not None or weight_ref() is not None or torch.cuda.is_initialized():
        raise base.StudyError("screen model disposal or CPU state could not be verified")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any((attempt / "pycache").iterdir()):
        raise base.StudyError("screen mask or bytecode cache changed")
    final_assets = base.verify_cached_assets(pretrained, args.cache)
    sources = finder.verify_inventory()
    result.update({"phase": "screen", "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                   "manifest_sha256": args.manifest_sha, "model_disposed": True, "no_model_loaded": False,
                   "available_physical_bytes_before_load": available,
                   "post_work_checkpoint_assets": final_assets, "runtime_settings": runtime,
                   "runtime": {k: env[k] for k in edits.ENVIRONMENT_FIELDS}, "imported_sources": sources,
                   "finished_utc": base.utc(), "scope": "cooperative execution report, not authenticated native code"})
    base.write(attempt / "child-result.json", base.canonical(result))
    session.attest_clean(lease)


def _phase_inventory(attempt, phase, observed, source_revision):
    paths = [attempt / name for name in ("started.json", "child-result.json", "manifest-review-attestation.json",
        "development-review-attestation.json", "public-exposure.json", "public-exposure-attestation.json")]
    if phase == "screen":
        directory = base.ordinary(attempt / "screen")
        names = {"screen.f32", "observations.jsonl", "state-original.json", "screen-selection.json"}
        if not directory.is_dir() or {path.name for path in directory.iterdir()} != names:
            raise base.StudyError("screen output set differs")
        paths.extend(directory / name for name in sorted(names))
    inventory, total = {}, 0
    for path in paths:
        checked = base.ordinary(path)
        if not checked.is_file():
            raise base.StudyError("screen artifact is not a regular file")
        raw = base.read(checked, 10_000_000)
        total += len(raw)
        if total > 20_000_000:
            raise base.StudyError("screen artifacts exceed the bound")
        inventory[path.relative_to(attempt).as_posix()] = {"bytes": len(raw), "sha256": base.digest(raw)}
    if phase == "screen":
        if inventory["screen/screen.f32"]["bytes"] != 32 * 50257 * 4:
            raise base.StudyError("screen full-logit matrix size differs")
        raw = base.read(base.ordinary(attempt / "screen/screen-selection.json"))
        selection = json.loads(raw)
        if (base.digest(raw) != observed["selection_sha256"]
                or selection["status"] != "private-screen"
                or selection["case_sha256"] != base.CASES_SHA
                or selection["identity"]["source_commit"] != source_revision
                or selection["identity"]["manifest_sha256"] != observed["manifest_sha256"]
                or selection["identity"]["development_record_sha256"] != DISCOVERY_PHASE_SHA
                or selection["identity"]["candidate_sha256"] != CANDIDATE_SHA
                or len(selection["selection"]["rows"]) != 32
                or selection["selection"]["selected_count"] != observed["selected_count"]
                or selection["selection"]["formal_target_eligible"] != observed["formal_target_eligible"]):
            raise base.StudyError("screen selection differs from child report")
        for name, item in selection["artifacts"].items():
            if name not in {"screen.f32", "observations.jsonl", "state-original.json"}:
                raise base.StudyError("screen selection inventory differs")
            if item != inventory["screen/" + name]:
                raise base.StudyError("screen saved artifact differs from selection")
        if set(selection["artifacts"]) != {"screen.f32", "observations.jsonl", "state-original.json"}:
            raise base.StudyError("screen selection inventory is incomplete")
    return inventory


def main(argv=None):
    global _ACTIVE_ATTEMPT, _ATTEMPT_OWNED
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("freeze", "run", "child"))
    for name in ("checkout", "venv", "dependency", "work"):
        parser.add_argument("--" + name, required=True)
    for name in ("inventory", "inventory-sha", "inventory-attestation", "custody-observation",
                 "development-review", "manifest-sha", "manifest-review", "cache", "study-root",
                 "attempt", "parent-platform"):
        parser.add_argument("--" + name)
    parser.add_argument("--phase", choices=("rehearsal", "screen"))
    args = parser.parse_args(argv)
    checkout, venv, dependency, workdir = map(base.ordinary, (args.checkout, args.venv, args.dependency, args.work))
    if args.mode == "child":
        child(args)
        return 0
    if not all((args.inventory, args.inventory_sha, args.inventory_attestation,
                args.custody_observation, args.development_review)):
        raise base.StudyError("screen requires current public inventory, custody and discovery review")
    custody = prior.custody_observation(args.custody_observation)
    public_raw = base.read(base.ordinary(args.inventory), 5_000_000)
    inventory_review_raw = base.read(base.ordinary(args.inventory_attestation))
    discovery_review_raw = base.read(base.ordinary(args.development_review))
    discovery_review = json.loads(discovery_review_raw)
    manifest_raw = None
    if args.mode == "run":
        if not all((args.manifest_sha, args.manifest_review, args.cache, args.study_root, args.phase)):
            raise base.StudyError("screen run requires published manifest, cache and persistent study root")
        manifest_raw = base.read(base.ordinary(checkout / base.MANIFEST))
        if base.digest(manifest_raw) != args.manifest_sha:
            raise base.StudyError("screen manifest digest differs")
        manifest = json.loads(manifest_raw)
        source_revision = manifest["source_commit"]
    else:
        source_revision = None
    launch, executable, head, revision, sources, hashes = base.setup(checkout, venv, dependency, workdir, source_revision)
    discovery = check_discovery(launch, checkout, head, discovery_review)
    previous = json.loads(base.read(base.ordinary(checkout / DISCOVERY_MANIFEST)))
    public = _current_inventory(launch, checkout, head, public_raw, args.inventory_sha,
                                json.loads(inventory_review_raw), previous["exposure"]["known_complete_rows"])
    if not base.windows_path_absent(launch, checkout, head,
                                    "results/targeted-edit-screened-matched-v1-20261029/screen"):
        raise base.StudyError("screen evidence directory already exists")
    if args.mode == "freeze":
        payload = {"version": "targeted-edit-screened-screen-manifest-v1", "source_commit": revision,
                   "source_sha256": hashes, "case_sha256": base.CASES_SHA, "discovery": discovery,
                   "development_review_sha256": base.digest(discovery_review_raw),
                   "original_model_state_sha256": discovery["model_state_sha256"],
                   "original_weight_sha256": discovery["weight_sha256"], "checkpoint": discovery["checkpoint"],
                   "exposure": public, "custodian_observations": custody, "created_utc": base.utc(),
                   "scope": "cooperative source and chronology reports; not authenticated execution"}
        base.write(checkout / base.MANIFEST, base.canonical(payload))
        print(json.dumps({"manifest_sha256": base.digest(base.canonical(payload)), "source_commit": revision,
                          "own_reserved_rows": public["own_reserved_rows"]}, sort_keys=True))
        return 0
    if hashes != manifest["source_sha256"] or manifest["case_sha256"] != base.CASES_SHA:
        raise base.StudyError("screen source or case inventory differs from manifest")
    if (manifest["development_review_sha256"] != base.digest(discovery_review_raw)
            or manifest["discovery"] != discovery):
        raise base.StudyError("screen discovery binding changed")
    if not base.windows_path_absent(launch, checkout, revision, base.MANIFEST):
        raise base.StudyError("screen manifest was already present at the source commit")
    base.committed_file(launch, checkout, head, checkout / base.MANIFEST, manifest_raw)
    manifest_review_raw = base.read(base.ordinary(args.manifest_review))
    manifest_review = json.loads(manifest_review_raw)
    if (manifest_review.get("manifest_sha256") != args.manifest_sha or not manifest_review.get("reviewer")
            or not manifest_review.get("reviewed_commit") or not manifest_review.get("observed_utc")):
        raise base.StudyError("screen manifest review differs")
    reviewed = manifest_review["reviewed_commit"]
    base.check_phase_ancestry(launch, checkout, revision, reviewed, head)
    base.committed_file(launch, checkout, reviewed, checkout / base.MANIFEST, manifest_raw)
    base.check_reported_order(discovery_review["observed_utc"], manifest["created_utc"],
                              manifest_review["observed_utc"], base.utc())
    cache, study_root = launch._checked_path(args.cache), launch._checked_path(args.study_root)
    if (not cache.is_dir() or not study_root.is_dir() or study_root == checkout
            or study_root.is_relative_to(checkout)):
        raise base.StudyError("screen cache or persistent CPU root is invalid")
    attempt = workdir / ("screened-" + args.phase + "-" + uuid.uuid4().hex)
    _ACTIVE_ATTEMPT, _ATTEMPT_OWNED = attempt, False
    attempt.mkdir()
    _ATTEMPT_OWNED = True
    try:
        (attempt / "pycache").mkdir()
        base.write(attempt / "manifest-review-attestation.json", manifest_review_raw)
        base.write(attempt / "development-review-attestation.json", discovery_review_raw)
        base.write(attempt / "public-exposure.json", public_raw)
        base.write(attempt / "public-exposure-attestation.json", inventory_review_raw)
        started_raw = base.canonical({"phase": args.phase, "source_commit": revision, "checkout_commit": head,
             "manifest_sha256": args.manifest_sha, "development_record_sha256": DISCOVERY_PHASE_SHA,
             "candidate_sha256": CANDIDATE_SHA, "development_review_sha256": base.digest(discovery_review_raw),
             "manifest_review_sha256": base.digest(manifest_review_raw), "custodian_observations": custody,
             "current_public_exposure": {k: v for k, v in public.items() if k != "known_complete_rows"},
             "public_exposure_file_sha256": base.digest(public_raw),
             "public_exposure_attestation_sha256": base.digest(inventory_review_raw),
             "started_utc": base.utc(), "scope": "operator-reported ordering, not a trusted timestamp"})
        base.write(attempt / "started.json", started_raw)
        parent_platform = launch._parent_platform_observation()
        _boot, finder = base.finder_for(checkout, hashes)
        from looking_glass.cpu_study_supervision import run_cpu_study
        guards = launch._load_guards(checkout, sources)
        job = guards.WindowsJob()
        command = [str(executable), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(attempt / "pycache"),
                   str(checkout / base.SCRIPT), "child", "--checkout", str(checkout), "--venv", str(venv),
                   "--dependency", str(dependency), "--work", str(workdir), "--phase", args.phase,
                   "--manifest-sha", args.manifest_sha, "--cache", str(cache), "--study-root", str(study_root),
                   "--attempt", str(attempt), "--parent-platform", json.dumps(parent_platform, separators=(",", ":"))]
        env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
               "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "PROCESSOR_IDENTIFIER") if name in os.environ}
        env.update({"PATH": str(Path(env.get("SYSTEMROOT", "C:/Windows")) / "System32"),
                    "TEMP": str(attempt), "TMP": str(attempt), "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        result = base.supervise_child(run_cpu_study, command, study_root, cwd=attempt, env=env, job=job)
        if (result.clean is not True or type(result.exit_code) is not int or result.exit_code != 0
                or result.timed_out is not False or type(result.pid) is not int or result.pid <= 0
                or type(result.run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", result.run_id) is None
                or job._handle is not None or not job._confirmed):
            raise base.StudyError("screen supervisor did not confirm a clean reaped owner and drained job")
        child_raw = base.read(base.ordinary(attempt / "child-result.json"))
        observed = json.loads(child_raw)
        if (observed["phase"] != args.phase or observed["manifest_sha256"] != args.manifest_sha
                or type(observed["owner_pid"]) is not int or observed["owner_pid"] != result.pid
                or observed["owner_run_id"] != result.run_id
                or (args.phase == "rehearsal" and observed.get("no_model_loaded") is not True)
                or (args.phase == "screen" and observed.get("model_disposed") is not True)):
            raise base.StudyError("screen child identity or disposal differs")
        _imports(observed["imported_sources"], hashes)
        if (args.phase == "screen"
                and (observed["post_work_checkpoint_assets"] != manifest["checkpoint"]["asset_sha256"]
                     or observed["model_state_sha256"] != manifest["original_model_state_sha256"]
                     or observed["weight_sha256"] != manifest["original_weight_sha256"]
                     or observed["buffer_state_sha256"] != manifest["discovery"]["buffer_state_sha256"]
                     or observed["runtime_settings"] != manifest["discovery"]["runtime_settings"])):
            raise base.StudyError("screen checkpoint, state or runtime inventory differs")
        if (launch._revision(checkout) != head or launch._verify_sources(checkout, sources) != hashes
                or base.read(base.ordinary(checkout / base.MANIFEST)) != manifest_raw
                or any((attempt / "pycache").iterdir())):
            raise base.StudyError("screen source, manifest or bytecode changed during run")
        finder.verify_inventory()
        artifacts = _phase_inventory(attempt, args.phase, observed, revision)
        record = {"version": "targeted-edit-screened-screen-phase-v1", "status": "complete",
                  "phase": args.phase, "source_commit": revision, "checkout_commit": head,
                  "manifest_sha256": args.manifest_sha, "started_sha256": base.digest(started_raw),
                  "child_result_sha256": base.digest(child_raw), "supervisor": asdict(result),
                  "owner_reaped": True, "job_drained": True, "child": observed,
                  "artifacts": artifacts, "finished_utc": base.utc(),
                  "scope": "process record only; independent saved-screen audit and publication remain required"}
        base.write(attempt / "phase-record.json", base.canonical(record))
        print(json.dumps({"status": "complete", "phase": args.phase, "attempt": str(attempt),
                          "record_sha256": base.digest(base.canonical(record))}, sort_keys=True))
        return 0
    except BaseException as error:
        base.record_failure(attempt, args.phase, error)
        raise


def entry(argv=None):
    try:
        return main(argv)
    except BaseException as error:
        if _ACTIVE_ATTEMPT is not None:
            if _ATTEMPT_OWNED:
                base.record_failure(_ACTIVE_ATTEMPT, "screen", error)
            print(json.dumps({"status": "incomplete", "attempt": str(_ACTIVE_ATTEMPT),
                              "attempt_owned": _ATTEMPT_OWNED, "error_type": type(error).__name__}), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(entry())

