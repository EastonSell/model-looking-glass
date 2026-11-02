"""Supervised discovery-only owner for the screened projected-edit follow-on.

The owner reuses the reviewed projected-study supervisor and source checks,
with a versioned case/exposure contract and discovery writer. It does not
implement target screening or evaluation. A completed process record still
requires independent artifact review and publication before any screen.
Source and timing records are cooperative attestations under a trusted,
quiescent checkout and cache, not proof against concurrent mutation.
"""

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import sys
import types
import weakref


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/targeted_edit_projected_study.py"
base = types.ModuleType("_screened_projected_owner")
base.__file__ = str(BASE_PATH)
exec(compile(BASE_PATH.read_bytes(), str(BASE_PATH), "exec"), base.__dict__)

base.SCRIPT = "scripts/targeted_edit_screened_study.py"
base.PROTOCOL = "docs/TARGETED_EDIT_SCREENED_MATCHED_V1.md"
base.SUPPLEMENT = "docs/TARGETED_EDIT_PROJECTED_DERIVATIVE_SCOPE.md"
base.OWNER_DOCUMENT = "docs/TARGETED_EDIT_SCREENED_CASES.md"
base.CASES = "experiments/targeted_edit_screened_public_cases_v1.json"
base.CASES_SHA = "fc3f00a0b4f7884a3c00011186420d16322009edf75b6b09e4296953dad198e3"
base.MANIFEST = "experiments/targeted_edit_screened_run_v1.json"
NATURAL_RECEIPT = "experiments/targeted_edit_screened_natural_receipt_v1.json"
NATURAL_RECEIPT_SHA = "bbd78ce2cb63db2a67cfc85cb030ac8cefea2f24c88c4105bff30d7392ef2c2b"
base.EXTRA = (base.SCRIPT, "scripts/targeted_edit_projected_study.py", base.PROTOCOL,
              base.SUPPLEMENT, base.OWNER_DOCUMENT, base.CASES, NATURAL_RECEIPT, base.REFERENCE)


def check_public_payload(payload, expected_sha, cases, minimum_rows=(), *, include_rows=False):
    """Check 44 exact prompt rows; only exact own reservations are exempt."""
    actual = base.digest(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("utf-8"))
    if actual != expected_sha:
        raise base.StudyError("public inventory digest differs")
    try:
        rows = cases["cases"]
        records = payload["records"]
        if (type(rows) is not list or len(rows) != 44 or type(records) is not list
                or len({row["case_id"] for row in rows}) != 44
                or len({row["record_id"] for row in records}) != len(records)):
            raise base.StudyError("screened case or inventory identities differ")
        desired = {"planned-" + row["case_id"]: row for row in rows}
        reserved = {}
        others = []
        for record in records:
            row = desired.get(record["record_id"])
            if row is None:
                others.append(tuple(record["token_ids"]))
            else:
                if (record["token_ids"] != row["token_ids"]
                        or record["public_text"] != row["prompt"]
                        or record["category"] != "planned"):
                    raise base.StudyError("own reservation identity or row differs")
                reserved[record["record_id"]] = record
        if len(reserved) not in (0, 44):
            raise base.StudyError("own reservation is partial")
        new = [tuple(row["token_ids"]) for row in rows]
        known = [tuple(row["token_ids"]) for row in records]
        if (any(not row or any(type(token) is not int or token < 0 or token >= 50257 for token in row)
                for row in (*new, *known)) or not set(map(tuple, minimum_rows)) <= set(known)):
            raise base.StudyError("invalid or missing known public token row")
        def related(left, right):
            return left[:len(right)] == right or right[:len(left)] == left
        if (any(related(left, right) for left in new for right in others)
                or any(related(left, right) for i, left in enumerate(new) for right in new[i + 1:])):
            raise base.StudyError("a screened prompt has a known public prefix relation")
        result = {"version": payload["version"], "canonical_sha256": actual,
                  "record_count": len(records), "screened_rows": len(new),
                  "own_reserved_rows": len(reserved), "other_row_comparisons": len(new) * len(others),
                  "known_relations": 0,
                  "scope": "public inventory only; unknown history and private splits are not certified"}
        if include_rows:
            result["known_complete_rows"] = [list(row) for row in known]
        return result
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, base.StudyError):
            raise
        raise base.StudyError("invalid screened case or exposure inventory") from exc


def custody_observation(path):
    """Require fresh reports without reading protected token rows."""
    observation = json.loads(base.read(path))
    rows = observation.get("observations")
    if (type(rows) is not list or len(rows) != 2
            or {row.get("custodian") for row in rows} != {"natural", "injection"}):
        raise base.StudyError("custodian observations are incomplete")
    now = datetime.now(timezone.utc)
    for row in rows:
        if type(row.get("reported_by")) is not str or not row["reported_by"]:
            raise base.StudyError("custodian reporter is missing")
        try:
            observed = datetime.fromisoformat(row["observed_utc"])
        except (ValueError, TypeError, KeyError) as exc:
            raise base.StudyError("custodian observation time differs") from exc
        if observed.tzinfo is None or not 0 <= (now - observed).total_seconds() <= 900:
            raise base.StudyError("custodian observation is stale or contradictory")
        if row["custodian"] == "injection" and row.get("status") == "no_protected_rows_reported":
            continue
        if (row["custodian"] != "natural" or row.get("status") != "private_comparison_clear"
                or row.get("case_sha256") != base.CASES_SHA
                or row.get("receipt_sha256") != NATURAL_RECEIPT_SHA):
            raise base.StudyError("custodian report is unavailable or does not bind the case file")
    return {"observations": rows,
            "scope": "named custodian reports only; no complete exposure history is certified"}


def freeze(checkout, source_revision, hashes, exposure_version, custody):
    case_bytes = base.read(base.ordinary(checkout / base.CASES))
    reference_bytes = base.read(base.ordinary(checkout / base.REFERENCE))
    receipt_bytes = base.read(base.ordinary(checkout / NATURAL_RECEIPT))
    if (base.digest(case_bytes) != base.CASES_SHA or base.digest(reference_bytes) != base.REFERENCE_SHA
            or base.digest(receipt_bytes) != NATURAL_RECEIPT_SHA):
        raise base.StudyError("fixed case, receipt or reference bytes differ")
    cases, reference = json.loads(case_bytes), json.loads(reference_bytes)
    receipt = json.loads(receipt_bytes)
    if receipt.get("status") != "clear" or receipt.get("public_case_count") != 44:
        raise base.StudyError("natural comparison receipt differs")
    exposure = base.public_inventory(checkout, hashes, exposure_version, cases)
    payload = {"version": "targeted-edit-screened-run-manifest-v1",
               "source_commit": source_revision, "source_sha256": hashes,
               "protocol": {"path": base.PROTOCOL, "sha256": hashes[base.PROTOCOL],
                            "supplement_path": base.SUPPLEMENT, "supplement_sha256": hashes[base.SUPPLEMENT]},
               "cases_path": base.CASES, "cases_sha256": base.CASES_SHA,
               "natural_receipt_sha256": NATURAL_RECEIPT_SHA, "evaluation_cases": cases["cases"],
               "development": {"discovery_ids": [464, 3139, 1748, 286, 4881, 318],
                   "calibration_ids": [[464, 4387, 9151, 318, 262], [7571, 5556, 734, 21767],
                                       [464, 6697, 286, 3024, 318], [32, 1285, 468, 3598]],
                   "target_id": 3576, "competitor_id": 6342},
               "checkpoint": reference["model"], "original_model_state_sha256": reference["model_state_sha256"],
               "original_weight_sha256": reference["weight_sha256"], "reference_sha256": base.REFERENCE_SHA,
               "exposure": exposure, "custodian_observations": custody, "created_utc": base.utc(),
               "ordering_scope": "operator-recorded commit and review ordering; not a trusted timestamp",
               "execution_scope": "trusted quiescent checkout and cache; hashes do not prove native or executed bytes"}
    base.write(checkout / base.MANIFEST, base.canonical(payload))
    return payload


def inventory_attempt(attempt, phase):
    paths = [attempt / name for name in ("started.json", "child-result.json", "manifest-review-attestation.json",
                                           "public-exposure.json", "public-exposure-attestation.json")]
    if phase == "development":
        directory = base.ordinary(attempt / "candidate")
        if not directory.is_dir():
            raise base.StudyError("development output directory differs")
        output = sorted(directory.iterdir())
        if not output:
            raise base.StudyError("development output directory is empty")
        paths.extend(output)
    if not 1 <= len(paths) <= 160:
        raise base.StudyError("phase artifact count differs")
    entries, total = {}, 0
    for path in paths:
        base.ordinary(path)
        if not path.is_file():
            raise base.StudyError("phase artifact is not an ordinary file")
        raw = base.read(path, 10_000_000)
        total += len(raw)
        if total > 150_000_000:
            raise base.StudyError("phase artifacts exceed the total bound")
        entries[path.relative_to(attempt).as_posix()] = {"bytes": len(raw), "sha256": base.digest(raw)}
    return entries


def validate_phase_output(attempt, phase, observed, inventory):
    if phase == "rehearsal":
        return
    if phase != "development":
        raise base.StudyError("only discovery development is implemented")
    raw = base.read(base.ordinary(attempt / "candidate" / "pre-evaluation-candidate.json"))
    if base.digest(raw) != observed["candidate_sha256"]:
        raise base.StudyError("candidate digest differs from child report")
    candidate = json.loads(raw)
    conditions = ("projected", "unprojected", "projected_random", "opposite", "nuisance",
                  "unprojected-grid-01", "unprojected-grid-02", "unprojected-grid-03",
                  "unprojected-grid-04", "unprojected-grid-06", "unprojected-grid-07")
    names = {"discovery-logits.f32", "discovery-grad-inputs.f32", "discovery-grad-outputs.f32",
             "input-factor.f32", "output-factor.f32", "full-gradient.f32", "discovery-original-logits.f32",
             "discovery-original-inputs.f32", "discovery-original-outputs.f32", "calibration-inputs.f32",
             "calibration-original-outputs.f32", "calibration-original-logits.f32", "state-original.json"}
    names |= {"factor-" + factor + ".f64" for factor in ("calibration_basis", "projected_input", "random_input",
              "random_output", "projected_random_input", "nuisance_basis", "nuisance_input")}
    for condition in conditions:
        names |= {"calibration-" + condition + "-outputs.f32", "calibration-" + condition + "-logits.f32",
                  "calibration-" + condition + "-algebraic.f64", "discovery-" + condition + "-logits.f32",
                  "discovery-" + condition + "-inputs.f32", "discovery-" + condition + "-outputs.f32",
                  "state-" + condition + ".json"}
    if (candidate.get("version") != "targeted-edit-screened-development-v1"
            or candidate.get("status") != "private-candidate" or set(candidate.get("artifacts", {})) != names):
        raise base.StudyError("development candidate artifact list is incomplete")
    recorded = {name.removeprefix("candidate/") for name in inventory if name.startswith("candidate/")}
    if recorded != names | {"pre-evaluation-candidate.json"}:
        raise base.StudyError("development phase output set differs")
    for name, item in candidate["artifacts"].items():
        saved = inventory["candidate/" + name]
        if item["bytes"] != saved["bytes"] or item["sha256"] != saved["sha256"]:
            raise base.StudyError("development phase artifact differs from candidate")


def check_imports(imports, hashes):
    required = {"looking_glass/" + name + ".py" for name in (
        "__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
        "pretrained", "targeted_edit_cpu_v1", "targeted_edit_v2_edits", "targeted_edit_v2_owner",
        "targeted_edit_projected_capture", "targeted_edit_screened_development", "targeted_edit_screened_math")}
    if (type(imports) is not dict or not required <= set(imports)
            or any(hashes.get(name) != value for name, value in imports.items())):
        raise base.StudyError("child import inventory differs from frozen sources")


base.check_public_payload = check_public_payload
base.custody_observation = custody_observation
base.freeze = freeze
base.inventory_attempt = inventory_attempt
base.validate_phase_output = validate_phase_output
base.check_imports = check_imports


def child(args):
    checkout, attempt = base.ordinary(args.checkout), base.ordinary(args.attempt)
    if (args.phase not in ("rehearsal", "development")
            or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != attempt / "pycache"
            or any((attempt / "pycache").iterdir())
            or any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules)):
        raise base.StudyError("isolated child startup differs")
    manifest_raw = base.read(base.ordinary(checkout / base.MANIFEST))
    if base.digest(manifest_raw) != args.manifest_sha:
        raise base.StudyError("manifest digest differs in child")
    manifest = json.loads(manifest_raw)
    hashes = manifest["source_sha256"]
    for name in (base.SCRIPT, "scripts/targeted_edit_projected_study.py"):
        if base.digest(base.read(base.ordinary(checkout / name))) != hashes[name]:
            raise base.StudyError("child owner source differs")
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
    from looking_glass import targeted_edit_screened_development as development
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False) or torch.cuda.is_initialized():
        raise base.StudyError("CPU numeric mode differs")
    reference = json.loads(base.read(base.ordinary(checkout / base.REFERENCE)))
    env = v1._environment()
    if v1._canonical({k: env[k] for k in edits.ENVIRONMENT_FIELDS}) != v1._canonical(
            {k: reference["environment"][k] for k in edits.ENVIRONMENT_FIELDS}):
        raise base.StudyError("runtime differs from the v1 reference")
    runtime = common._runtime_settings(torch)
    if (runtime["intraop_threads"] != 1 or runtime["interop_threads"] != 1
            or runtime["deterministic_algorithms"] is not True or runtime["deterministic_warn_only"] is not False
            or runtime["cpu_autocast_enabled"] is not False or runtime["inference_mode_enabled"] is not False
            or runtime["cuda_visibility"] != "-1" or runtime["cuda_initialized"] is not False
            or runtime["hf_hub_offline"] != "1" or runtime["transformers_offline"] != "1"):
        raise base.StudyError("initial CPU runtime settings differ")
    case_bytes = base.read(base.ordinary(checkout / base.CASES))
    if base.digest(case_bytes) != manifest["cases_sha256"]:
        raise base.StudyError("case bytes differ")
    if args.phase == "rehearsal":
        sources = finder.verify_inventory()
        if torch.cuda.is_initialized() or any((attempt / "pycache").iterdir()):
            raise base.StudyError("rehearsal CPU or bytecode state differs")
        result = {"phase": "rehearsal", "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                  "manifest_sha256": args.manifest_sha, "model_disposed": False, "no_model_loaded": True,
                  "available_physical_bytes_before_load": available, "runtime_settings": runtime,
                  "runtime": {k: env[k] for k in edits.ENVIRONMENT_FIELDS}, "imported_sources": sources,
                  "finished_utc": base.utc(), "scope": "no checkpoint or model forward; diagnostic only"}
        base.write(attempt / "child-result.json", base.canonical(result))
        session.attest_clean(lease)
        return

    def work():
        loaded = pretrained.load_distilgpt2(cache_dir=args.cache, allow_download=False)
        if common._runtime_settings(torch) != runtime:
            raise base.StudyError("runtime settings changed during checkpoint loading")
        model, module, weight = common._check_checkpoint(loaded, {"v1_compatibility": {"model": manifest["checkpoint"]}}, torch, v1)
        v1._verify_tokenizer(loaded.tokenizer)
        base.check_projected_tokenizer(loaded.tokenizer, manifest["evaluation_cases"])
        guard = common._ModelGuard(model, module, weight, torch, v1)
        if guard.runtime_settings != runtime:
            raise base.StudyError("runtime settings changed during tokenizer or model verification")
        if guard.state_sha != manifest["original_model_state_sha256"] or guard.weight_sha != manifest["original_weight_sha256"]:
            raise base.StudyError("original model state or site weight differs from v1")
        fixed = manifest["development"]
        candidate = development.prepare_development(guard, fixed["discovery_ids"], fixed["calibration_ids"],
                                                     fixed["target_id"], fixed["competitor_id"], attempt / "candidate")
        guard.verify()
        result = {"candidate_sha256": base.digest(base.read(attempt / "candidate" / "pre-evaluation-candidate.json")),
                  "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                  "buffer_state_sha256": guard.buffer_sha, "discovery_margin": candidate["discovery_margin"]}
        return result, weakref.ref(model), weakref.ref(weight)

    result, model_ref, weight_ref = work()
    gc.collect()
    if model_ref() is not None or weight_ref() is not None or torch.cuda.is_initialized():
        raise base.StudyError("model disposal or CPU state could not be verified")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any((attempt / "pycache").iterdir()):
        raise base.StudyError("mask or bytecode cache changed")
    final_assets = base.verify_cached_assets(pretrained, args.cache)
    sources = finder.verify_inventory()
    result.update({"phase": args.phase, "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                   "manifest_sha256": args.manifest_sha, "model_disposed": True, "no_model_loaded": False,
                   "available_physical_bytes_before_load": available,
                   "memory_scope": "point-in-time admission observation, not a reservation or peak bound",
                   "post_work_checkpoint_assets": final_assets, "runtime_settings": runtime,
                   "runtime": {k: env[k] for k in edits.ENVIRONMENT_FIELDS}, "imported_sources": sources,
                   "finished_utc": base.utc(), "scope": "recorded cooperative execution, not authenticated native code"})
    base.write(attempt / "child-result.json", base.canonical(result))
    session.attest_clean(lease)


base.child = child


def entry(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for index, item in enumerate(argv):
        option = item.partition("=")[0]
        if len(option) > 2 and option != "--phase" and "--phase".startswith(option):
            raise base.StudyError("abbreviated phase options are unsupported")
        if item == "--phase":
            value = argv[index + 1] if index + 1 < len(argv) else None
        elif item.startswith("--phase="):
            value = item.partition("=")[2]
        else:
            continue
        if value not in ("rehearsal", "development"):
            raise base.StudyError("only discovery development and no-model rehearsal are implemented")
    return base.entry(argv)


if __name__ == "__main__":
    raise SystemExit(entry())

