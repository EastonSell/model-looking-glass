"""Supervised evaluation of the frozen screened projected-edit comparison.

The separately published discovery and screen records are required before any
new-case edited forward. Source, clocks, process and custody evidence remain
cooperative attestations under a trusted quiescent checkout. A completed child
report is provisional until independent saved-output review and publication.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types
import uuid
import weakref

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "scripts/targeted_edit_screened_screen_study.py"
screen_owner = types.ModuleType("_screened_baseline_owner")
screen_owner.__file__ = str(PREVIOUS)
exec(compile(PREVIOUS.read_bytes(), str(PREVIOUS), "exec"), screen_owner.__dict__)
prior, base = screen_owner.prior, screen_owner.base
base.SCRIPT = "scripts/targeted_edit_screened_evaluation_study.py"
base.MANIFEST = "experiments/targeted_edit_screened_evaluation_run_v1.json"
base.EXTRA = (*base.EXTRA, base.SCRIPT, "experiments/targeted_edit_screened_screen_run_v1.json")
SCREEN_DIR = "results/targeted-edit-screened-matched-v1-20261029/screen"
SCREEN_MANIFEST = "experiments/targeted_edit_screened_screen_run_v1.json"
SCREEN_MANIFEST_SHA = "1de77fa0078b71d75606d39001280267adb6ae5d3d3558fdea36283995e9b653"
SCREEN_PHASE_SHA = "c8e4c87377d57ada69d50b717524fd082a8b5d464b66baf39bc2af5a1e9c53b5"
SELECTION_SHA = "9ea861d89c8ed78c038947dbdb331d6dba1ea8267901ce06e5492dab32e7c11f"
SCREEN_EVIDENCE_COMMIT = "fc6496e89b9130805f484555378f29b74dd7ba76"
SCREEN_EVIDENCE_PATHS = ("experiments/targeted_edit_screened_public_cases_v1.json", SCREEN_MANIFEST,
    SCREEN_DIR + "/phase-record.json", SCREEN_DIR + "/child-result.json",
    SCREEN_DIR + "/screen/observations.jsonl", SCREEN_DIR + "/screen/screen-selection.json",
    SCREEN_DIR + "/screen/screen.f32")
SCREEN_EVIDENCE_SHAS = (base.CASES_SHA, SCREEN_MANIFEST_SHA, SCREEN_PHASE_SHA,
    "cc9d26c1fc26e23d6506c536d401111c43202f9a10afd1c91caf350dacebc286",
    "48456dedec8359e402e51abe5c52c5942b5728a4cf83574516a3283af905d0fb", SELECTION_SHA,
    "252c7e7c377d12bac7b0263f67e42d8127a3ed3f87e0a76faca334a7ae570132")
SCREEN_PROVENANCE = ("Public non-blind screened baseline on the pinned CPU model; the saved journal reports "
    "one completed baseline result for each of 32 target prompts. The 12 locality prompts remain planned. "
    "Process records are cooperative observations, not authenticated execution. Evidence SHA-256: "
    + ", ".join(SCREEN_EVIDENCE_SHAS) + ".")
_ACTIVE_ATTEMPT = None
_ATTEMPT_OWNED = False


def check_screen(launch, checkout, head, review, discovery):
    if (type(review) is not dict or review.get("manifest_sha256") != SCREEN_MANIFEST_SHA
            or review.get("screen_phase_sha256") != SCREEN_PHASE_SHA
            or review.get("selection_sha256") != SELECTION_SHA
            or type(review.get("reviewer")) is not str or not review["reviewer"]
            or type(review.get("reviewed_commit")) is not str):
        raise base.StudyError("screen review binding differs")
    reviewed = review["reviewed_commit"]
    base.commit_object(launch, checkout, reviewed)
    launch._git(checkout, "merge-base", "--is-ancestor", reviewed, head)
    manifest_path = launch._checked_path(checkout / SCREEN_MANIFEST)
    manifest_raw = base.read(manifest_path)
    if base.digest(manifest_raw) != SCREEN_MANIFEST_SHA:
        raise base.StudyError("published screen manifest differs")
    base.committed_file(launch, checkout, reviewed, manifest_path, manifest_raw)
    manifest = json.loads(manifest_raw)
    directory = launch._checked_path(checkout / SCREEN_DIR)
    phase_path = launch._checked_path(directory / "phase-record.json")
    phase_raw = base.read(phase_path)
    if base.digest(phase_raw) != SCREEN_PHASE_SHA:
        raise base.StudyError("published screen phase differs")
    base.committed_file(launch, checkout, reviewed, phase_path, phase_raw)
    phase = json.loads(phase_raw)
    child, supervisor = phase["child"], phase["supervisor"]
    if (phase["status"] != "complete" or phase["phase"] != "screen"
            or phase["source_commit"] != manifest["source_commit"] or phase["manifest_sha256"] != SCREEN_MANIFEST_SHA
            or phase["owner_reaped"] is not True or phase["job_drained"] is not True
            or child["phase"] != "screen" or child["manifest_sha256"] != SCREEN_MANIFEST_SHA
            or child["selection_sha256"] != SELECTION_SHA or child["model_disposed"] is not True
            or type(supervisor["pid"]) is not int or supervisor["pid"] <= 0
            or type(supervisor["run_id"]) is not str or re.fullmatch(r"[0-9a-f]{32}", supervisor["run_id"]) is None
            or type(child["owner_pid"]) is not int or child["owner_pid"] != supervisor["pid"]
            or child["owner_run_id"] != supervisor["run_id"] or supervisor["clean"] is not True
            or type(supervisor["exit_code"]) is not int or supervisor["exit_code"] != 0
            or supervisor["timed_out"] is not False):
        raise base.StudyError("screen supervisor or child outcome differs")
    for key in ("model_state_sha256", "weight_sha256", "buffer_state_sha256", "runtime_settings"):
        if child[key] != discovery[key]:
            raise base.StudyError("screen original state differs from discovery")
    if child["post_work_checkpoint_assets"] != discovery["checkpoint"]["asset_sha256"]:
        raise base.StudyError("screen checkpoint assets differ")
    screen_owner._imports(child["imported_sources"], manifest["source_sha256"])
    if screen_owner._phase_inventory(directory, "screen", child, manifest["source_commit"]) != phase["artifacts"]:
        raise base.StudyError("screen process artifact inventory differs")
    base.check_committed_inventory(launch, checkout, reviewed, directory, phase["artifacts"])
    base.check_exact_phase_files(launch, checkout, reviewed, directory, phase["artifacts"])
    launch._git(checkout, "merge-base", "--is-ancestor", manifest["source_commit"], reviewed)
    base.check_reported_order(manifest["created_utc"], phase["finished_utc"], review["observed_utc"], base.utc())
    selection_raw = base.read(launch._checked_path(directory / "screen/screen-selection.json"))
    if base.digest(selection_raw) != SELECTION_SHA:
        raise base.StudyError("screen selection differs")
    selection = json.loads(selection_raw)["selection"]
    return {"manifest_sha256": SCREEN_MANIFEST_SHA, "phase_sha256": SCREEN_PHASE_SHA,
            "selection_sha256": SELECTION_SHA, "selected_count": selection["selected_count"],
            "formal_target_eligible": selection["formal_target_eligible"],
            "reviewed_commit": reviewed, "review_observed_utc": review["observed_utc"]}


def current_inventory(launch, checkout, head, raw, sha, attestation, minimum_rows):
    payload, cases = json.loads(raw), json.loads(screen_owner._case_bytes(checkout))
    actual = base.digest(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode())
    if actual != sha:
        raise base.StudyError("current inventory digest differs")
    rows, records = cases["cases"], payload["records"]
    if (len(rows) != 44 or len({r["case_id"] for r in rows}) != 44
            or len({r["record_id"] for r in records}) != len(records)):
        raise base.StudyError("case or inventory identity differs")
    lookup = {r["case_id"]: r for r in rows}
    own, others = {}, []
    for record in records:
        name = record["record_id"]
        prefix = "planned-" if name.startswith("planned-") else "observed-" if name.startswith("observed-") else ""
        case = lookup.get(name[len(prefix):]) if prefix else None
        if case is None:
            others.append(tuple(record["token_ids"]))
            continue
        category = "planned" if prefix == "planned-" else "observed-saved"
        if (case["case_id"] in own or record["category"] != category
                or record["token_ids"] != case["token_ids"] or record["public_text"] != case["prompt"]
                or (case["role"] == "locality" and prefix != "planned-")
                or (prefix == "observed-" and (record.get("reference_commit") != SCREEN_EVIDENCE_COMMIT
                    or record.get("references") != list(SCREEN_EVIDENCE_PATHS)
                    or record.get("provenance") != SCREEN_PROVENANCE))):
            raise base.StudyError("own reservation or screen exposure differs")
        own[case["case_id"]] = record
    new, known = [tuple(r["token_ids"]) for r in rows], [tuple(r["token_ids"]) for r in records]
    if (len(own) != 44 or any(not row or any(type(t) is not int or not 0 <= t < 50257 for t in row)
                              for row in (*new, *known))
            or not set(map(tuple, minimum_rows)) <= set(known)):
        raise base.StudyError("own or previous public rows are missing")
    def related(a, b):
        return a[:len(b)] == b or b[:len(a)] == a
    if (any(related(a, b) for a in new for b in others)
            or any(related(a, b) for i, a in enumerate(new) for b in new[i + 1:])):
        raise base.StudyError("evaluation case has a known public prefix relation")
    base.check_inventory_attestation(launch, checkout, head, attestation, sha, payload["version"])
    return {"version": payload["version"], "canonical_sha256": sha, "record_count": len(records),
            "own_reserved_or_observed_rows": len(own), "other_row_comparisons": len(new) * len(others),
            "known_relations": 0, "known_complete_rows": [list(row) for row in known],
            "scope": "public inventory only; unknown histories and private rows are not certified"}


def minimum_exposure(previous, manifest):
    """A later run must retain the rows frozen in its own manifest."""
    return (previous if manifest is None else manifest)["exposure"]["known_complete_rows"]


def _imports(imports, hashes):
    required = {"looking_glass/" + name + ".py" for name in (
        "__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
        "pretrained", "targeted_edit_cpu_v1", "targeted_edit_cpu_v2", "targeted_edit_v2_edits",
        "targeted_edit_v2_owner", "targeted_edit_projected_execution", "targeted_edit_screened_selection",
        "targeted_edit_screened_screen", "targeted_edit_screened_math", "targeted_edit_screened_score",
        "targeted_edit_screened_execution")}
    if (type(imports) is not dict or not required <= set(imports)
            or any(hashes.get(name) != value for name, value in imports.items())):
        raise base.StudyError("evaluation imported-source inventory differs")


def child(args):
    checkout, attempt = base.ordinary(args.checkout), base.ordinary(args.attempt)
    if (args.phase not in ("rehearsal", "evaluation")
            or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != attempt / "pycache"
            or any((attempt / "pycache").iterdir())
            or any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules)):
        raise base.StudyError("evaluation child startup differs")
    manifest_raw = base.read(base.ordinary(checkout / base.MANIFEST))
    if base.digest(manifest_raw) != args.manifest_sha:
        raise base.StudyError("evaluation manifest differs in child")
    manifest = json.loads(manifest_raw)
    hashes = manifest["source_sha256"]
    for name in (base.SCRIPT, "scripts/targeted_edit_screened_screen_study.py",
                 "scripts/targeted_edit_screened_study.py", "scripts/targeted_edit_projected_study.py"):
        if base.digest(base.read(base.ordinary(checkout / name))) != hashes[name]:
            raise base.StudyError("evaluation owner source differs")
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
        raise base.StudyError("available physical memory is below the admission floor")
    boot._add_dependencies(checkout, Path(args.venv).resolve(strict=True), [args.dependency])
    import torch
    from looking_glass import pretrained
    from looking_glass import targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_v2_edits as edits
    from looking_glass import targeted_edit_v2_owner as common
    from looking_glass import targeted_edit_screened_execution as execution
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False) or torch.cuda.is_initialized():
        raise base.StudyError("evaluation CPU numeric mode differs")
    reference = json.loads(base.read(base.ordinary(checkout / base.REFERENCE)))
    env = v1._environment()
    if v1._canonical({k: env[k] for k in edits.ENVIRONMENT_FIELDS}) != v1._canonical(
            {k: reference["environment"][k] for k in edits.ENVIRONMENT_FIELDS}):
        raise base.StudyError("evaluation runtime differs from the v1 reference")
    runtime = common._runtime_settings(torch)
    if runtime != manifest["discovery"]["runtime_settings"]:
        raise base.StudyError("evaluation runtime differs from discovery")
    case_bytes = screen_owner._case_bytes(checkout)
    if args.phase == "rehearsal":
        sources = finder.verify_inventory()
        if torch.cuda.is_initialized() or any((attempt / "pycache").iterdir()):
            raise base.StudyError("evaluation rehearsal CPU or bytecode state differs")
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
            raise base.StudyError("runtime changed during checkpoint loading")
        model, module, weight = common._check_checkpoint(loaded, {"v1_compatibility": {"model": manifest["checkpoint"]}}, torch, v1)
        v1._verify_tokenizer(loaded.tokenizer)
        base.check_projected_tokenizer(loaded.tokenizer, json.loads(case_bytes)["cases"])
        guard = common._ModelGuard(model, module, weight, torch, v1)
        screen_owner._match_discovery_state(guard, runtime, manifest)
        prepared = execution.reconstruct(guard, checkout / screen_owner.DISCOVERY_DIR / "candidate",
                                         screen_owner.CANDIDATE_SHA, checkout / SCREEN_DIR / "screen",
                                         SELECTION_SHA, case_bytes)
        report = execution.evaluate(prepared, attempt / "evaluation")
        guard.verify()
        raw = base.read(attempt / "evaluation/provisional-report.json")
        return ({"report_sha256": base.digest(raw), "selected_count": report["selected_count"],
                 "provisional_equal_norm_decision": report["provisional_equal_norm_decision"],
                 "provisional_matched_discovery_decision": report["provisional_matched_discovery_decision"],
                 "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                 "buffer_state_sha256": guard.buffer_sha}, weakref.ref(model), weakref.ref(weight))
    result, model_ref, weight_ref = work()
    gc.collect()
    if model_ref() is not None or weight_ref() is not None or torch.cuda.is_initialized():
        raise base.StudyError("evaluation disposal or CPU state could not be verified")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any((attempt / "pycache").iterdir()):
        raise base.StudyError("evaluation mask or bytecode cache changed")
    final_assets = base.verify_cached_assets(pretrained, args.cache)
    sources = finder.verify_inventory()
    result.update({"phase": "evaluation", "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                   "manifest_sha256": args.manifest_sha, "model_disposed": True, "no_model_loaded": False,
                   "available_physical_bytes_before_load": available, "post_work_checkpoint_assets": final_assets,
                   "runtime_settings": runtime, "runtime": {k: env[k] for k in edits.ENVIRONMENT_FIELDS},
                   "imported_sources": sources, "finished_utc": base.utc(),
                   "scope": "cooperative execution report, not authenticated native code"})
    base.write(attempt / "child-result.json", base.canonical(result))
    session.attest_clean(lease)


def phase_inventory(attempt, phase, observed):
    paths = [attempt / name for name in ("started.json", "child-result.json", "manifest-review-attestation.json",
             "development-review-attestation.json", "screen-review-attestation.json", "public-exposure.json",
             "public-exposure-attestation.json")]
    if phase == "evaluation":
        directory = base.ordinary(attempt / "evaluation")
        report = json.loads(base.read(base.ordinary(directory / "provisional-report.json")))
        names = ["original", "projected", "unprojected", "projected_random", "opposite", "nuisance"]
        if report["discovery_match_available"]:
            names.append("matched")
        expected = {"observations.jsonl", "provisional-report.json"} | {
            prefix + name + suffix for name in names for prefix, suffix in (("", ".f32"), ("state-", ".json"))}
        if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected:
            raise base.StudyError("evaluation output set differs")
        paths.extend(directory / name for name in sorted(expected))
    inventory, total = {}, 0
    for path in paths:
        checked = base.ordinary(path)
        if not checked.is_file():
            raise base.StudyError("evaluation artifact is not an ordinary file")
        raw = base.read(checked, 10_000_000)
        total += len(raw)
        if total > 100_000_000:
            raise base.StudyError("evaluation artifact total exceeds bound")
        inventory[path.relative_to(attempt).as_posix()] = {"bytes": len(raw), "sha256": base.digest(raw)}
    if phase == "evaluation":
        count = observed["selected_count"]
        if type(count) is not int or not 0 <= count <= 8:
            raise base.StudyError("evaluation selected count differs")
        if any(inventory["evaluation/" + name + ".f32"]["bytes"] != (count + 12) * 50257 * 4 for name in names):
            raise base.StudyError("evaluation full-logit matrix size differs")
        report_raw = base.read(base.ordinary(directory / "provisional-report.json"))
        if (base.digest(report_raw) != observed["report_sha256"] or report["selected_count"] != count
                or report["provisional_equal_norm_decision"] != observed["provisional_equal_norm_decision"]
                or report["provisional_matched_discovery_decision"] != observed["provisional_matched_discovery_decision"]):
            raise base.StudyError("evaluation child report differs")
    return inventory


def main(argv=None):
    global _ACTIVE_ATTEMPT, _ATTEMPT_OWNED
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("freeze", "run", "child"))
    for name in ("checkout", "venv", "dependency", "work"):
        parser.add_argument("--" + name, required=True)
    for name in ("inventory", "inventory-sha", "inventory-attestation", "custody-observation",
                 "development-review", "screen-review", "manifest-sha", "manifest-review", "cache", "study-root",
                 "attempt", "parent-platform"):
        parser.add_argument("--" + name)
    parser.add_argument("--phase", choices=("rehearsal", "evaluation"))
    args = parser.parse_args(argv)
    checkout, venv, dependency, workdir = map(base.ordinary, (args.checkout, args.venv, args.dependency, args.work))
    if args.mode == "child":
        child(args)
        return 0
    if not all((args.inventory, args.inventory_sha, args.inventory_attestation, args.custody_observation,
                args.development_review, args.screen_review)):
        raise base.StudyError("evaluation needs current inventory, custody and both published phase reviews")
    custody = prior.custody_observation(args.custody_observation)
    public_raw = base.read(base.ordinary(args.inventory), 5_000_000)
    inventory_review_raw = base.read(base.ordinary(args.inventory_attestation))
    discovery_review_raw = base.read(base.ordinary(args.development_review))
    screen_review_raw = base.read(base.ordinary(args.screen_review))
    discovery_review, screen_review = json.loads(discovery_review_raw), json.loads(screen_review_raw)
    manifest_raw = None
    if args.mode == "run":
        if not all((args.manifest_sha, args.manifest_review, args.cache, args.study_root, args.phase)):
            raise base.StudyError("evaluation needs a published manifest, cache and persistent CPU root")
        manifest_raw = base.read(base.ordinary(checkout / base.MANIFEST))
        if base.digest(manifest_raw) != args.manifest_sha:
            raise base.StudyError("evaluation manifest digest differs")
        manifest = json.loads(manifest_raw)
        source_revision = manifest["source_commit"]
    else:
        source_revision = None
    launch, executable, head, revision, sources, hashes = base.setup(checkout, venv, dependency, workdir, source_revision)
    discovery = screen_owner.check_discovery(launch, checkout, head, discovery_review)
    screened = check_screen(launch, checkout, head, screen_review, discovery)
    launch._git(checkout, "merge-base", "--is-ancestor", screen_review["reviewed_commit"], revision)
    previous = json.loads(base.read(base.ordinary(checkout / SCREEN_MANIFEST)))
    public = current_inventory(launch, checkout, head, public_raw, args.inventory_sha,
                               json.loads(inventory_review_raw), minimum_exposure(previous, manifest if args.mode == "run" else None))
    if not base.windows_path_absent(launch, checkout, head,
                                    "results/targeted-edit-screened-matched-v1-20261029/evaluation"):
        raise base.StudyError("evaluation evidence directory already exists")
    if args.mode == "freeze":
        payload = {"version": "targeted-edit-screened-evaluation-manifest-v1", "source_commit": revision,
                   "source_sha256": hashes, "case_sha256": base.CASES_SHA, "discovery": discovery, "screen": screened,
                   "development_review_sha256": base.digest(discovery_review_raw),
                   "screen_review_sha256": base.digest(screen_review_raw),
                   "original_model_state_sha256": discovery["model_state_sha256"],
                   "original_weight_sha256": discovery["weight_sha256"], "checkpoint": discovery["checkpoint"],
                   "exposure": public, "custodian_observations": custody, "created_utc": base.utc(),
                   "scope": "cooperative source and chronology reports; not authenticated execution"}
        base.write(checkout / base.MANIFEST, base.canonical(payload))
        print(json.dumps({"manifest_sha256": base.digest(base.canonical(payload)), "source_commit": revision,
                          "selected_count": screened["selected_count"]}, sort_keys=True))
        return 0
    if (hashes != manifest["source_sha256"] or manifest["case_sha256"] != base.CASES_SHA
            or manifest["development_review_sha256"] != base.digest(discovery_review_raw)
            or manifest["screen_review_sha256"] != base.digest(screen_review_raw)
            or manifest["discovery"] != discovery or manifest["screen"] != screened):
        raise base.StudyError("evaluation source or prior-phase binding changed")
    if not base.windows_path_absent(launch, checkout, revision, base.MANIFEST):
        raise base.StudyError("evaluation manifest was already present at the source commit")
    base.committed_file(launch, checkout, head, checkout / base.MANIFEST, manifest_raw)
    manifest_review_raw = base.read(base.ordinary(args.manifest_review))
    manifest_review = json.loads(manifest_review_raw)
    if (manifest_review.get("manifest_sha256") != args.manifest_sha or not manifest_review.get("reviewer")
            or not manifest_review.get("reviewed_commit") or not manifest_review.get("observed_utc")):
        raise base.StudyError("evaluation manifest review differs")
    reviewed = manifest_review["reviewed_commit"]
    base.check_phase_ancestry(launch, checkout, revision, reviewed, head)
    base.committed_file(launch, checkout, reviewed, checkout / base.MANIFEST, manifest_raw)
    base.check_reported_order(screen_review["observed_utc"], manifest["created_utc"],
                              manifest_review["observed_utc"], base.utc())
    cache, study_root = launch._checked_path(args.cache), launch._checked_path(args.study_root)
    if (not cache.is_dir() or not study_root.is_dir() or study_root == checkout or study_root.is_relative_to(checkout)):
        raise base.StudyError("evaluation cache or persistent CPU root is invalid")
    attempt = workdir / ("screened-" + args.phase + "-" + uuid.uuid4().hex)
    _ACTIVE_ATTEMPT, _ATTEMPT_OWNED = attempt, False
    attempt.mkdir()
    _ATTEMPT_OWNED = True
    try:
        (attempt / "pycache").mkdir()
        for name, data in (("manifest-review-attestation.json", manifest_review_raw),
                           ("development-review-attestation.json", discovery_review_raw),
                           ("screen-review-attestation.json", screen_review_raw),
                           ("public-exposure.json", public_raw), ("public-exposure-attestation.json", inventory_review_raw)):
            base.write(attempt / name, data)
        started_raw = base.canonical({"phase": args.phase, "source_commit": revision, "checkout_commit": head,
             "manifest_sha256": args.manifest_sha, "development_record_sha256": screen_owner.DISCOVERY_PHASE_SHA,
             "candidate_sha256": screen_owner.CANDIDATE_SHA, "screen_phase_sha256": SCREEN_PHASE_SHA,
             "selection_sha256": SELECTION_SHA, "development_review_sha256": base.digest(discovery_review_raw),
             "screen_review_sha256": base.digest(screen_review_raw), "manifest_review_sha256": base.digest(manifest_review_raw),
             "custodian_observations": custody, "current_public_exposure": {k: v for k, v in public.items() if k != "known_complete_rows"},
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
            raise base.StudyError("evaluation supervisor did not confirm clean reaped owner and drained job")
        child_raw = base.read(base.ordinary(attempt / "child-result.json"))
        observed = json.loads(child_raw)
        if (observed["phase"] != args.phase or observed["manifest_sha256"] != args.manifest_sha
                or type(observed["owner_pid"]) is not int or observed["owner_pid"] != result.pid
                or observed["owner_run_id"] != result.run_id
                or (args.phase == "rehearsal" and observed.get("no_model_loaded") is not True)
                or (args.phase == "evaluation" and observed.get("model_disposed") is not True)):
            raise base.StudyError("evaluation child identity or disposal differs")
        _imports(observed["imported_sources"], hashes)
        if (args.phase == "evaluation" and (observed["post_work_checkpoint_assets"] != manifest["checkpoint"]["asset_sha256"]
                or observed["model_state_sha256"] != manifest["original_model_state_sha256"]
                or observed["weight_sha256"] != manifest["original_weight_sha256"]
                or observed["buffer_state_sha256"] != manifest["discovery"]["buffer_state_sha256"]
                or observed["runtime_settings"] != manifest["discovery"]["runtime_settings"])):
            raise base.StudyError("evaluation checkpoint, state or runtime differs")
        if (launch._revision(checkout) != head or launch._verify_sources(checkout, sources) != hashes
                or base.read(base.ordinary(checkout / base.MANIFEST)) != manifest_raw or any((attempt / "pycache").iterdir())):
            raise base.StudyError("evaluation source, manifest or bytecode changed")
        finder.verify_inventory()
        artifacts = phase_inventory(attempt, args.phase, observed)
        record = {"version": "targeted-edit-screened-evaluation-phase-v1", "status": "complete",
                  "phase": args.phase, "source_commit": revision, "checkout_commit": head,
                  "manifest_sha256": args.manifest_sha, "started_sha256": base.digest(started_raw),
                  "child_result_sha256": base.digest(child_raw), "supervisor": asdict(result),
                  "owner_reaped": True, "job_drained": True, "child": observed, "artifacts": artifacts,
                  "finished_utc": base.utc(),
                  "scope": "process record only; independent saved-output audit and publication remain required"}
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
                base.record_failure(_ACTIVE_ATTEMPT, "evaluation", error)
            print(json.dumps({"status": "incomplete", "attempt": str(_ACTIVE_ATTEMPT),
                              "attempt_owned": _ATTEMPT_OWNED, "error_type": type(error).__name__}), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(entry())
