"""Supervise both frozen answer-format phases; never applies an edit.

The source, case, current public inventory and relevant custody are bound
before launch. Results remain provisional until a separate reader verifies
the saved rows and clean process record. Source and chronology are cooperative
attestations in a trusted quiescent checkout, not authenticated execution.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import sys
import types
import uuid
import weakref


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/targeted_edit_projected_study.py"
base = types.ModuleType("_full_top_projected_owner")
base.__file__ = str(BASE_PATH)
exec(compile(BASE_PATH.read_bytes(), str(BASE_PATH), "exec"), base.__dict__)

old = base.load_script(ROOT / "scripts/targeted_edit_full_top_screen_owner.py", "_answer_format_previous_owner")
base.SCRIPT = "scripts/answer_format_feasibility_owner.py"
base.PROTOCOL = "docs/ANSWER_FORMAT_FEASIBILITY_CPU_V1.md"
base.SUPPLEMENT = "docs/ANSWER_FORMAT_FEASIBILITY_SCOPE_V1.md"
base.OWNER_DOCUMENT = "docs/ANSWER_FORMAT_FEASIBILITY_CASES_V1.md"
base.CASES = "experiments/answer_format_feasibility_cases_v1.json"
base.CASES_SHA = "cb1cfb9c4ab3c57dbcb78b778d01eaec50c0600caee2c9ee862ded14f2b8f38a"
base.MANIFEST = "experiments/answer_format_feasibility_manifest_v1.json"
FRAME = "experiments/answer_format_feasibility_natural_comparison_v1.json"
FRAME_SHA = "4f73b96f7df22d9e672aefdf81a17f594152d42ce6a211d275efdc9def61c38d"
SUPPORT = "experiments/targeted_edit_full_top_public_support_v1.json"
SUPPORT_SHA = "5638b5ccc1d6585184fcd359e825ac3c2f6a1e377e24afde78bc5a9a2fc845d6"
PRIOR = "results/targeted-edit-screened-matched-v1-20261029/development/candidate/pre-evaluation-candidate.json"
PRIOR_SHA = "f038db0b27f68a80fc3fc11374971bf8acd6b47812ab112d760520acc7ff5166"
NAT_DIR = "results/distilgpt2-natural-agreement-v1-20261028/"
NAT_COMMITMENT = NAT_DIR + "heldout-commitment-v1.json"
NAT_COMMITMENT_SHA = "7f4f96fee2fd90bc5efbf8b85bb722021b3b698905256dc9f2e2e626d4a4f4ba"
NAT_OPERATIONAL = NAT_DIR + "heldout-operational-binding-v1.json"
NAT_OPERATIONAL_SHA = "aab326ecd4b86b010cc06f6bb93907653b74532ecb3b5da587586d6b9b0ab186"
NAT_BINDING = NAT_DIR + "binding-v2.json"
NAT_BINDING_SHA = "23575704f3e55d741123d3c1024accfb15aa0a048a84ce415bc765573d913ba0"
NAT_WRAPPER = "tools/natural_heldout_custody.py"
NAT_WRAPPER_SHA = "cc6f8d82bc9167aa34d3357ea64364f15153dc00a6cdef60e7b547e66b68f3cb"
NAT_SUCCESSOR_SHA = "6303eb0713d64b020eea3fe491a686c9020ea424754443372aef73749bc87bc4"
base.EXTRA = tuple(dict.fromkeys((*old.base.EXTRA, base.SCRIPT, base.PROTOCOL, base.SUPPLEMENT,
    base.OWNER_DOCUMENT, base.CASES, FRAME, "tests/test_answer_format_feasibility.py",
    "tests/test_answer_format_feasibility_worker.py", "tests/test_answer_format_feasibility_owner.py")))
CASES_COMMIT = "46842b08338fef9fc9aefc2c90aab3a8af63c66c"
LABELS = ("factual", "counterfactual", "madrid", "stockholm")
RESULT_ROOT = "results/answer-format-feasibility-cpu-v1-20261029"
DEVELOPMENT_DIR = RESULT_ROOT + "/development/attempt"
HELDOUT_DIR = RESULT_ROOT + "/heldout/attempt"
_ACTIVE_ATTEMPT = None
_ATTEMPT_OWNED = False


def check_public_payload(payload, expected_sha, cases, minimum_rows=(), *, observed_binding=None):
    """Exempt only exact own reservations; reject all other prefix matches."""
    actual = base.digest(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("utf-8"))
    if actual != expected_sha:
        raise base.StudyError("public inventory digest differs")
    try:
        rows, records = cases["cases"], payload["records"]
        if (type(rows) is not list or len(rows) != 64 or type(records) is not list
                or len({r["case_id"] for r in rows}) != 64
                or len({r["record_id"] for r in records}) != len(records)):
            raise base.StudyError("case or public record identities differ")
        desired = {r["case_id"]: r for r in rows}
        own, others = {}, []
        for record in records:
            identity = record["record_id"]
            is_planned = identity.startswith("planned-")
            is_observed = identity.startswith("observed-")
            case_id = identity[len("planned-"):] if is_planned else identity[len("observed-"):] if is_observed else ""
            selected = desired.get(case_id)
            if selected is None:
                others.append(tuple(record["token_ids"]))
            else:
                if case_id in own:
                    raise base.StudyError("own identity appears more than once")
                if (record["token_ids"] != selected["token_ids"]
                        or record["public_text"] != selected["prompt"]
                        or record["case_role"] != selected["role"]
                        or record["candidate_token_ids"] != [selected["labels"][x]["token_id"] for x in LABELS]):
                    raise base.StudyError("own reservation differs")
                if is_planned:
                    if record["category"] != "planned" or record["reference_commit"] != CASES_COMMIT:
                        raise base.StudyError("planned reservation provenance differs")
                elif (not is_observed or selected["role"] != "development" or observed_binding is None
                      or record["category"] != "observed-saved"
                      or record["reference_commit"] != observed_binding["commit"]
                      or not set(observed_binding["references"]) <= set(record["references"])
                      or any(value not in record["provenance"] for value in observed_binding["hashes"])):
                    raise base.StudyError("observed development provenance differs")
                own[case_id] = record
        if len(own) != 64:
            raise base.StudyError("all 64 exact reservations are required")
        proposed = [tuple(r["token_ids"]) for r in rows]
        known = [tuple(r["token_ids"]) for r in records]
        if (any(not row or any(type(x) is not int or not 0 <= x < 50257 for x in row)
                for row in (*proposed, *known)) or not set(map(tuple, minimum_rows)) <= set(known)):
            raise base.StudyError("invalid or missing public token row")
        def related(a, b):
            return a[:len(b)] == b or b[:len(a)] == a
        if (any(related(a, b) for a in proposed for b in others)
                or any(related(a, b) for i, a in enumerate(proposed) for b in proposed[i + 1:])):
            raise base.StudyError("a proposed input has a public prefix relation")
        return {"version": payload["version"], "canonical_sha256": actual,
                "record_count": len(records), "own_reserved_rows": len(own),
                "other_row_comparisons": len(proposed) * len(others),
                "known_complete_rows": [list(row) for row in known], "known_relations": 0,
                "scope": "public inventory only; unknown history is not certified"}
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, base.StudyError):
            raise
        raise base.StudyError("invalid public inventory") from exc


def check_custody(path):
    """Require fresh scoped observations; never read protected rows."""
    doc = json.loads(base.read(base.ordinary(path)))
    rows = doc.get("observations")
    if (type(rows) is not list or len(rows) != 2
            or {x.get("custodian") for x in rows} != {"natural", "injection"}):
        raise base.StudyError("custodian observations are incomplete")
    now = datetime.now(timezone.utc)
    for row in rows:
        try:
            observed = datetime.fromisoformat(row["observed_utc"])
            if (observed.tzinfo is None or not 0 <= (now - observed).total_seconds() <= 900
                    or type(row["reported_by"]) is not str or not row["reported_by"]):
                raise base.StudyError("custodian observation is stale or missing")
            if row["custodian"] == "natural":
                if (row.get("status") != "public_frame_clear" or row.get("case_sha256") != base.CASES_SHA
                        or row.get("frame_sha256") != SUPPORT_SHA or row.get("report_sha256") != FRAME_SHA
                        or row.get("commitment_sha256") != NAT_COMMITMENT_SHA
                        or row.get("original_split_unchanged") is not True
                        or row.get("no_additional_relevant_splits_reported") is not True):
                    raise base.StudyError("natural public-frame observation differs")
                if not (row.get("successor_status") == "no_successor_split_reported"
                        or (row.get("successor_status") == "successor_in_public_frame"
                            and row.get("successor_frame_sha256") == NAT_SUCCESSOR_SHA)):
                    raise base.StudyError("natural successor custody is unknown")
            elif row.get("status") == "no_protected_rows_reported":
                continue
            elif (row.get("status") != "private_comparison_clear"
                  or row.get("case_sha256") != base.CASES_SHA
                  or re.fullmatch(r"[0-9a-f]{64}", str(row.get("receipt_sha256"))) is None):
                raise base.StudyError("injection observation differs")
        except (KeyError, ValueError, TypeError) as exc:
            if isinstance(exc, base.StudyError):
                raise
            raise base.StudyError("invalid custodian observation") from exc
    return {"observations": rows,
            "scope": "named custodian/operator reports; no unknown exposure history is certified"}


def _fixed(checkout):
    _old_case, reference, prior = old._fixed(checkout)
    case_raw = base.read(base.ordinary(checkout / base.CASES), 300_000)
    frame_raw = base.read(base.ordinary(checkout / FRAME), 200_000)
    if base.digest(case_raw) != base.CASES_SHA or base.digest(frame_raw) != FRAME_SHA:
        raise base.StudyError("fixed cases or public frame report differ")
    frame = json.loads(frame_raw)
    if (frame["proposal_sha256"] != base.CASES_SHA or frame["original_support_sha256"] != SUPPORT_SHA
            or frame["counts"] != {"original_support": 128, "proposal": 64, "successor_inputs": 256}
            or any(frame["prefix_relation_counts"].values()) or frame["status"] != "clear"
            or frame["successor_inputs_sha256"] != NAT_SUCCESSOR_SHA
            or frame["tokenizer_sha256"] != reference["model"]["asset_sha256"]["tokenizer.json"]):
        raise base.StudyError("public natural frame report differs")
    return case_raw, reference, prior


def _current(launch, checkout, head, raw, sha, attestation, minimum_rows=(), *, observed_binding=None):
    cases, _reference, _prior = _fixed(checkout)
    result = check_public_payload(json.loads(raw), sha, json.loads(cases), minimum_rows,
                                  observed_binding=observed_binding)
    base.check_inventory_attestation(launch, checkout, head, attestation, sha, result["version"])
    _check_latest_inventory(launch, checkout, attestation, result["version"], sha)
    return result


def _check_latest_inventory(launch, checkout, attestation, version, sha):
    """Require a fresh cooperative official-tree and relevant-exposure observation."""
    try:
        observed = datetime.fromisoformat(attestation["observed_utc"])
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        latest = attestation["latest_main_commit"]
        if (observed.tzinfo is None or not 0 <= age <= 900
                or type(latest) is not str or re.fullmatch(r"[0-9a-f]{40}", latest) is None
                or attestation.get("latest_inventory_version") != version
                or attestation.get("latest_inventory_sha256") != sha
                or attestation.get("no_unreserved_relevant_inputs_reported") is not True
                or type(attestation.get("currentness_reported_by")) is not str
                or not attestation["currentness_reported_by"]):
            raise base.StudyError("latest inventory observation is stale or contradictory")
        base.commit_object(launch, checkout, latest)
        launch._git(checkout, "merge-base", "--is-ancestor", attestation["reviewed_commit"], latest)
        tree = launch._git(checkout, "ls-tree", "-r", "--name-only", latest, "--", "looking_glass").decode("utf-8")
        versions = [int(m.group(1)) for line in tree.splitlines()
                    if (m := re.fullmatch(r"looking_glass/pretrained_exposures_v([0-9]+)\.py", line))]
        match = re.fullmatch(r"distilgpt2-development-exposures-v([0-9]+)", version)
        if not versions or match is None or max(versions) != int(match.group(1)):
            raise base.StudyError("reported inventory is not latest in the observed public tree")
        source = launch._checked_path(checkout / attestation["source_path"])
        source_raw = base.read(source, 1_000_000)
        base.committed_file(launch, checkout, latest, source, source_raw)
    except (KeyError, ValueError, TypeError, UnicodeError) as exc:
        if isinstance(exc, base.StudyError):
            raise
        raise base.StudyError("invalid latest-inventory observation") from exc


def _tokenizer(tokenizer, cases):
    for row in cases["cases"]:
        ids = row["token_ids"]
        if tokenizer.encode(row["prompt"], add_special_tokens=False) != ids:
            raise base.StudyError("prompt tokenization differs")
        if (tokenizer.encode(row["prefix_text"], add_special_tokens=False) != row["prefix_token_ids"]
                or tokenizer.encode(row["suffix_text"], add_special_tokens=False) != row["suffix_token_ids"]
                or row["prefix_token_ids"] + row["suffix_token_ids"] != ids):
            raise base.StudyError("prefix/suffix token boundary differs")
        for item in row["labels"].values():
            if (tokenizer.encode(item["text"], add_special_tokens=False) != [item["token_id"]]
                    or tokenizer.encode(row["prompt"] + item["text"], add_special_tokens=False)
                    != ids + [item["token_id"]]):
                raise base.StudyError("single-token continuation differs")


def _imports(imports, hashes):
    required = {"looking_glass/" + name + ".py" for name in (
        "__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
        "pretrained", "targeted_edit_cpu_v1", "targeted_edit_v2_edits", "targeted_edit_v2_owner",
        "targeted_edit_full_top", "answer_format_feasibility", "answer_format_feasibility_worker")}
    if (type(imports) is not dict or not required <= set(imports)
            or any(hashes.get(name) != value for name, value in imports.items())):
        raise base.StudyError("child import inventory differs")


def _run_bound_phase(guard, runtime, manifest, runner, case_raw, identity, phase, output, prior=None):
    """Reject supplemental state/runtime drift before any screen forward."""
    if (guard.runtime_settings != runtime or runtime != manifest["baseline_runtime_settings"]
            or guard.state_sha != manifest["original_model_state_sha256"]
            or guard.weight_sha != manifest["original_weight_sha256"]
            or guard.buffer_sha != manifest["baseline_buffer_state_sha256"]):
        raise base.StudyError("loaded state, buffers or runtime differ")
    return runner.run(guard, case_raw, identity, phase, output, prior=prior)


def child(args):
    checkout, attempt = base.ordinary(args.checkout), base.ordinary(args.attempt)
    if (args.phase not in ("rehearsal", "development", "heldout") or not (sys.flags.isolated and sys.flags.no_site
            and sys.dont_write_bytecode) or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != attempt / "pycache"
            or any((attempt / "pycache").iterdir())
            or any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules)):
        raise base.StudyError("isolated child startup differs")
    manifest_raw = base.read(base.ordinary(checkout / base.MANIFEST))
    if base.digest(manifest_raw) != args.manifest_sha:
        raise base.StudyError("child manifest differs")
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
        raise base.StudyError("platform was observed before setup")
    win = sys.getwindowsversion()
    platform._syscmd_ver = lambda *_args, **_kwargs: ("Windows", str(win.major), f"{win.major}.{win.minor}.{win.build}")
    boot._install_process_guard()
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(Path(args.study_root).resolve(strict=True))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    available = base.available_memory()
    if available < 3 * 1024**3:
        raise base.StudyError("available physical memory is below the study floor")
    boot._add_dependencies(checkout, Path(args.venv).resolve(strict=True), [args.dependency])
    import torch
    from looking_glass import pretrained
    from looking_glass import targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_v2_edits as edits
    from looking_glass import targeted_edit_v2_owner as common
    from looking_glass import answer_format_feasibility_worker as worker
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False) or torch.cuda.is_initialized():
        raise base.StudyError("CPU numeric mode differs")
    case_raw, reference, _prior = _fixed(checkout)
    env = v1._environment()
    if v1._canonical({k: env[k] for k in edits.ENVIRONMENT_FIELDS}) != v1._canonical(
            {k: reference["environment"][k] for k in edits.ENVIRONMENT_FIELDS}):
        raise base.StudyError("runtime differs from the pinned reference")
    runtime = common._runtime_settings(torch)
    if (runtime["intraop_threads"] != 1 or runtime["interop_threads"] != 1
            or runtime["deterministic_algorithms"] is not True or runtime["deterministic_warn_only"] is not False
            or runtime["cpu_autocast_enabled"] is not False or runtime["inference_mode_enabled"] is not False
            or runtime["cuda_visibility"] != "-1" or runtime["cuda_initialized"] is not False
            or runtime["hf_hub_offline"] != "1" or runtime["transformers_offline"] != "1"):
        raise base.StudyError("initial CPU runtime differs")
    if args.phase == "rehearsal":
        result = {"phase": "rehearsal", "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                  "manifest_sha256": args.manifest_sha, "model_disposed": False, "no_model_loaded": True,
                  "runtime_settings": runtime, "imported_sources": finder.verify_inventory(),
                  "finished_utc": base.utc(), "scope": "no checkpoint or model forward; diagnostic only"}
        base.write(attempt / "child-result.json", base.canonical(result))
        session.attest_clean(lease)
        return

    def work():
        loaded = pretrained.load_distilgpt2(cache_dir=args.cache, allow_download=False)
        if common._runtime_settings(torch) != runtime:
            raise base.StudyError("runtime changed during checkpoint loading")
        model, module, weight = common._check_checkpoint(loaded, {"v1_compatibility": {"model": reference["model"]}}, torch, v1)
        v1._verify_tokenizer(loaded.tokenizer)
        _tokenizer(loaded.tokenizer, json.loads(case_raw))
        guard = common._ModelGuard(model, module, weight, torch, v1)
        identity = {"source_commit": manifest["source_commit"], "manifest_sha256": args.manifest_sha}
        previous = None
        if args.phase == "heldout":
            directory = base.ordinary(args.development_phase)
            previous = {"matrix": base.read(base.ordinary(directory / "development/matrix.f32"), 10_000_000),
                        "result": base.read(base.ordinary(directory / "development/results.json")),
                        "matrix_sha256": args.development_matrix_sha, "result_sha256": args.development_result_sha}
        result = _run_bound_phase(guard, runtime, manifest, worker, case_raw, identity,
                                  args.phase, attempt / args.phase, previous)
        guard.verify()
        observations = result["score"]["observations"]
        decoded = []
        for row in observations:
            unique, representative = row["unique_top_id"], row["top_representative_id"]
            decoded.append({"case_id": row["case_id"], "unique_top_id": unique,
                            "unique_top_text": None if unique is None else loaded.tokenizer.decode(
                                [unique], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                            "unique_top_raw_token": None if unique is None else loaded.tokenizer.convert_ids_to_tokens(unique),
                            "top_tie_count": row["top_tie_count"], "top_representative_id": representative,
                            "top_representative_text": loaded.tokenizer.decode(
                                [representative], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                            "top_representative_raw_token": loaded.tokenizer.convert_ids_to_tokens(representative)})
        base.write(attempt / args.phase / "top-tokens.json", base.canonical(decoded))
        selected_raw = base.read(attempt / args.phase / "results.json")
        return ({"result_sha256": base.digest(selected_raw), "selection_status": result["score"]["status"],
                 "selected_family": result["score"]["selected_family"],
                 "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                 "buffer_state_sha256": guard.buffer_sha}, weakref.ref(model), weakref.ref(weight))

    result, model_ref, weight_ref = work()
    gc.collect()
    if (model_ref() is not None or weight_ref() is not None or torch.cuda.is_initialized()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any((attempt / "pycache").iterdir())):
        raise base.StudyError("model disposal, CPU state or bytecode could not be verified")
    result.update({"phase": args.phase, "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                   "manifest_sha256": args.manifest_sha, "model_disposed": True, "no_model_loaded": False,
                   "available_physical_bytes_before_load": available,
                   "post_work_checkpoint_assets": base.verify_cached_assets(pretrained, args.cache),
                   "runtime_settings": runtime, "imported_sources": finder.verify_inventory(),
                   "finished_utc": base.utc(), "scope": "cooperative execution report"})
    base.write(attempt / "child-result.json", base.canonical(result))
    session.attest_clean(lease)


def _inventory(attempt, phase, observed, source_revision):
    names = ["started.json", "child-result.json", "manifest-review-attestation.json",
             "public-exposure.json", "public-exposure-attestation.json"]
    if phase == "heldout":
        names.append("development-review-attestation.json")
    if phase != "rehearsal":
        directory = base.ordinary(attempt / phase)
        outputs = {"matrix.f32", "journal.jsonl", "state-original.json", "results.json", "top-tokens.json"}
        if not directory.is_dir() or {p.name for p in directory.iterdir()} != outputs:
            raise base.StudyError("phase output set differs")
        names.extend(phase + "/" + name for name in sorted(outputs))
    inventory = {}
    for name in names:
        path = base.ordinary(attempt / name)
        if not path.is_file():
            raise base.StudyError("artifact is not a regular file")
        raw = base.read(path, 10_000_000)
        inventory[name] = {"bytes": len(raw), "sha256": base.digest(raw)}
    if phase == "rehearsal":
        return inventory
    prefix = phase + "/"
    if inventory[prefix + "matrix.f32"]["bytes"] != 32 * 50257 * 4:
        raise base.StudyError("phase matrix length differs")
    raw = base.read(base.ordinary(attempt / phase / "results.json"))
    result = json.loads(raw)
    expected_probe = {"subnormal_times_one_bits": 1, "normal_times_half_bits": 0x00400000,
                      "numpy_cast_preserves_min_subnormal": True}
    score = result["score"]
    if (base.digest(raw) != observed["result_sha256"] or result["stage"] != phase
            or result["case_sha256"] != base.CASES_SHA
            or result["identity"] != {"source_commit": source_revision, "manifest_sha256": observed["manifest_sha256"]}
            or result["score_denormal_probe_before"] != expected_probe
            or result["score_denormal_probe_after"] != expected_probe
            or len(score["observations"]) != 32 or score["status"] != observed["selection_status"]
            or score["selected_family"] != observed["selected_family"]):
        raise base.StudyError("saved score identity differs")
    if set(result["artifacts"]) != {"matrix.f32", "journal.jsonl", "state-original.json"}:
        raise base.StudyError("phase artifact inventory differs")
    for name, item in result["artifacts"].items():
        if inventory[prefix + name] != item:
            raise base.StudyError("phase artifact hash differs")
    if phase == "heldout" and (result["development_replay_denormal_probe_before"] != expected_probe
            or result["development_replay_denormal_probe_after"] != expected_probe):
        raise base.StudyError("development replay denormal probes differ")
    state = json.loads(base.read(base.ordinary(attempt / phase / "state-original.json")))
    if (state["cleanup"] != "verified" or state["model_state_sha256"] != observed["model_state_sha256"]
            or state["weight_sha256"] != observed["weight_sha256"]
            or state["applied_weight_sha256"] != observed["weight_sha256"]
            or state["buffer_state_sha256"] != observed["buffer_state_sha256"]
            or state["runtime_settings"] != observed["runtime_settings"]
            or state["addition_denormal_probe_before"] != expected_probe
            or state["addition_denormal_probe_after"] != expected_probe):
        raise base.StudyError("phase baseline state differs")
    matrix = base.read(base.ordinary(attempt / phase / "matrix.f32"), 10_000_000)
    journal = base.read(base.ordinary(attempt / phase / "journal.jsonl")).splitlines()
    decoded = json.loads(base.read(base.ordinary(attempt / phase / "top-tokens.json")))
    if len(journal) != 32 or type(decoded) is not list or len(decoded) != 32:
        raise base.StudyError("row journal or decoded-token count differs")
    for index, (item, saved, top) in enumerate(zip(journal, score["observations"], decoded)):
        row = json.loads(item)
        offset, size = index * 50257 * 4, 50257 * 4
        if (row["execution_index"] != index or row["canonical_index"] != saved["canonical_index"]
                or row["case_id"] != saved["case_id"] or row["offset"] != offset or row["bytes"] != size
                or row["sha256"] != base.digest(matrix[offset:offset + size])
                or row["sha256"] != saved["logit_sha256"] or saved["logit_byte_offset"] != offset
                or saved["logit_bytes"] != size or row["forward_denormal_probe_before"] != expected_probe
                or row["forward_denormal_probe_after"] != expected_probe
                or top["case_id"] != saved["case_id"] or top["unique_top_id"] != saved["unique_top_id"]
                or top["top_tie_count"] != saved["top_tie_count"]
                or top["top_representative_id"] != saved["top_representative_id"]):
            raise base.StudyError("row journal or decoded-top identity differs")
    return inventory


def _check_development(launch, checkout, head, manifest, manifest_sha, manifest_reviewed,
                       record_path, record_sha, review):
    """Bind every committed development artifact and its independent review before held-out work."""
    path = launch._checked_path(record_path)
    directory = launch._checked_path(checkout / DEVELOPMENT_DIR)
    if path != directory / "phase-record.json":
        raise base.StudyError("development evidence directory differs")
    raw = base.read(path)
    if base.digest(raw) != record_sha:
        raise base.StudyError("development record digest differs")
    record = json.loads(raw)
    if (review.get("phase_sha256") != record_sha or review.get("manifest_sha256") != manifest_sha
            or review.get("case_sha256") != base.CASES_SHA or review.get("status") != "passed"
            or not review.get("reviewer")
            or not review.get("reviewed_commit") or not review.get("observed_utc")):
        raise base.StudyError("independent development review differs")
    reviewed = review["reviewed_commit"]
    base.check_phase_ancestry(launch, checkout, manifest["source_commit"], manifest_reviewed, head, reviewed)
    if not base.windows_path_absent(launch, checkout, manifest_reviewed, DEVELOPMENT_DIR):
        raise base.StudyError("development evidence existed at manifest review")
    base.committed_file(launch, checkout, reviewed, path, raw)
    child, supervisor = record["child"], record["supervisor"]
    if (record["version"] != "answer-format-phase-v1" or record["status"] != "complete"
            or record["phase"] != "development" or record["source_commit"] != manifest["source_commit"]
            or record["manifest_sha256"] != manifest_sha or record["owner_reaped"] is not True
            or record["job_drained"] is not True or child["phase"] != "development"
            or child["manifest_sha256"] != manifest_sha or child["model_disposed"] is not True
            or child["no_model_loaded"] is not False or supervisor["clean"] is not True
            or type(supervisor["exit_code"]) is not int or supervisor["exit_code"] != 0
            or supervisor["timed_out"] is not False or type(supervisor["pid"]) is not int
            or supervisor["pid"] <= 0 or child["owner_pid"] != supervisor["pid"]
            or child["owner_run_id"] != supervisor["run_id"]
            or re.fullmatch(r"[0-9a-f]{32}", str(supervisor["run_id"])) is None
            or child["model_state_sha256"] != manifest["original_model_state_sha256"]
            or child["weight_sha256"] != manifest["original_weight_sha256"]
            or child["buffer_state_sha256"] != manifest["baseline_buffer_state_sha256"]
            or child["runtime_settings"] != manifest["baseline_runtime_settings"]
            or child["post_work_checkpoint_assets"] != manifest["checkpoint"]["asset_sha256"]):
        raise base.StudyError("development supervision or baseline state differs")
    _imports(child["imported_sources"], manifest["source_sha256"])
    inventory = record["artifacts"]
    base.check_committed_inventory(launch, checkout, reviewed, directory, inventory)
    base.check_exact_phase_files(launch, checkout, reviewed, directory, inventory)
    if (inventory["child-result.json"]["sha256"] != record["child_result_sha256"]
            or base.canonical(json.loads(base.read(directory / "child-result.json"))) != base.canonical(child)
            or inventory["started.json"]["sha256"] != record["started_sha256"]):
        raise base.StudyError("development process evidence differs")
    started = json.loads(base.read(directory / "started.json"))
    base.check_reported_order(manifest["created_utc"], started["started_utc"], child["finished_utc"],
                              record["finished_utc"], review["observed_utc"], base.utc())
    _inventory(directory, "development", child, manifest["source_commit"])
    matrix_sha = inventory["development/matrix.f32"]["sha256"]
    result_sha = inventory["development/results.json"]["sha256"]
    if review.get("matrix_sha256") != matrix_sha or review.get("result_sha256") != result_sha:
        raise base.StudyError("independent development review binds different rows")
    audit_path = review.get("audit_path")
    if (type(audit_path) is not str or not audit_path.startswith(RESULT_ROOT + "/development/")
            or ".." in Path(audit_path).parts):
        raise base.StudyError("independent audit path differs")
    audit = launch._checked_path(checkout / audit_path)
    audit_raw = base.read(audit, 10_000_000)
    if base.digest(audit_raw) != review.get("audit_sha256"):
        raise base.StudyError("independent audit bytes differ")
    base.committed_file(launch, checkout, reviewed, audit, audit_raw)
    audit_doc = json.loads(audit_raw)
    if (audit_doc.get("schema") != "independent-answer-format-owner-comparison-v1"
            or audit_doc.get("status") != "passed" or audit_doc.get("heldout") is not None
            or audit_doc.get("development", {}).get("matrix_sha256") != matrix_sha
            or audit_doc.get("development", {}).get("result_sha256") != result_sha
            or audit_doc.get("development", {}).get("journal_rows") != 32):
        raise base.StudyError("independent saved-row audit differs")
    return directory, matrix_sha, result_sha


def main(argv=None):
    global _ACTIVE_ATTEMPT, _ATTEMPT_OWNED
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("freeze", "run", "child"))
    for name in ("checkout", "venv", "dependency", "work"):
        parser.add_argument("--" + name, required=True)
    for name in ("inventory", "inventory-sha", "inventory-attestation", "custody-observation", "manifest-sha",
                 "manifest-review", "cache", "study-root", "attempt", "parent-platform",
                 "development-phase", "development-record-sha", "development-review",
                 "development-matrix-sha", "development-result-sha"):
        parser.add_argument("--" + name)
    parser.add_argument("--phase", choices=("rehearsal", "development", "heldout"))
    args = parser.parse_args(argv)
    checkout, venv, dependency, workdir = map(base.ordinary, (args.checkout, args.venv, args.dependency, args.work))
    if args.mode == "child":
        child(args)
        return 0
    if not all((args.inventory, args.inventory_sha, args.inventory_attestation, args.custody_observation)):
        raise base.StudyError("current inventory and relevant custody observations are required")
    custody = check_custody(args.custody_observation)
    public_raw = base.read(base.ordinary(args.inventory), 5_000_000)
    inventory_review_raw = base.read(base.ordinary(args.inventory_attestation))
    manifest_raw = None
    if args.mode == "run":
        if not all((args.manifest_sha, args.manifest_review, args.cache, args.study_root, args.phase)):
            raise base.StudyError("run requires published manifest, cache and persistent study root")
        manifest_raw = base.read(base.ordinary(checkout / base.MANIFEST))
        if base.digest(manifest_raw) != args.manifest_sha:
            raise base.StudyError("manifest digest differs")
        manifest = json.loads(manifest_raw)
        source_revision = manifest["source_commit"]
    else:
        source_revision = None
    launch, executable, head, revision, sources, hashes = base.setup(checkout, venv, dependency, workdir, source_revision)
    case_raw, reference, prior = _fixed(checkout)
    minimum = () if args.mode == "freeze" else manifest["exposure"]["known_complete_rows"]
    public = None
    if args.mode == "freeze" or args.phase != "heldout":
        public = _current(launch, checkout, head, public_raw, args.inventory_sha,
                          json.loads(inventory_review_raw), minimum)
    if ((args.mode == "freeze" or args.phase in ("rehearsal", "development"))
            and not base.windows_path_absent(launch, checkout, head, DEVELOPMENT_DIR)):
        raise base.StudyError("development evidence already exists")
    if not base.windows_path_absent(launch, checkout, head, HELDOUT_DIR):
        raise base.StudyError("held-out evidence already exists")
    if args.mode == "freeze":
        if not base.windows_path_absent(launch, checkout, revision, base.MANIFEST):
            raise base.StudyError("manifest already existed at source revision")
        payload = {"version": "answer-format-feasibility-manifest-v1", "source_commit": revision,
                   "source_sha256": hashes, "case_sha256": base.CASES_SHA,
                   "protocol_sha256": hashes[base.PROTOCOL], "scope_sha256": hashes[base.SUPPLEMENT],
                   "case_note_sha256": hashes[base.OWNER_DOCUMENT],
                   "seeds": {"development": 20261031, "heldout": 20261101},
                   "execution_permutations": json.loads(case_raw)["execution_permutations"],
                   "selection_rule": "first family with factual unique tops in all four prefixes for at least three facts",
                   "heldout_rule": "all 32 cases; at least three development-robust facts remain robust in selected family",
                   "public_frame_sha256": FRAME_SHA, "public_support_sha256": SUPPORT_SHA,
                   "natural_commitment_sha256": NAT_COMMITMENT_SHA,
                   "natural_operational_binding_sha256": NAT_OPERATIONAL_SHA,
                   "natural_scientific_binding_sha256": NAT_BINDING_SHA,
                   "natural_custody_wrapper_sha256": NAT_WRAPPER_SHA,
                   "checkpoint": reference["model"], "original_model_state_sha256": reference["model_state_sha256"],
                   "original_weight_sha256": reference["weight_sha256"], "reference_sha256": base.REFERENCE_SHA,
                   "prior_candidate_sha256": PRIOR_SHA, "baseline_buffer_state_sha256": prior["buffer_state_sha256"],
                   "baseline_runtime_settings": prior["runtime_settings"],
                   "exposure": public, "custodian_observations": custody, "created_utc": base.utc(),
                   "scope": "cooperative source and ordering reports; not authenticated execution"}
        data = base.canonical(payload)
        base.write(checkout / base.MANIFEST, data)
        print(json.dumps({"manifest_sha256": base.digest(data), "source_commit": revision,
                          "own_reserved_rows": public["own_reserved_rows"]}, sort_keys=True))
        return 0
    if (hashes != manifest["source_sha256"] or manifest["case_sha256"] != base.CASES_SHA
            or manifest["version"] != "answer-format-feasibility-manifest-v1"
            or manifest["protocol_sha256"] != hashes[base.PROTOCOL]
            or manifest["scope_sha256"] != hashes[base.SUPPLEMENT]
            or manifest["case_note_sha256"] != hashes[base.OWNER_DOCUMENT]
            or manifest["seeds"] != {"development": 20261031, "heldout": 20261101}
            or manifest["execution_permutations"] != json.loads(case_raw)["execution_permutations"]
            or manifest["selection_rule"] != "first family with factual unique tops in all four prefixes for at least three facts"
            or manifest["heldout_rule"] != "all 32 cases; at least three development-robust facts remain robust in selected family"
            or manifest["public_frame_sha256"] != FRAME_SHA or manifest["public_support_sha256"] != SUPPORT_SHA
            or manifest["natural_commitment_sha256"] != NAT_COMMITMENT_SHA
            or manifest["natural_operational_binding_sha256"] != NAT_OPERATIONAL_SHA
            or manifest["natural_scientific_binding_sha256"] != NAT_BINDING_SHA
            or manifest["natural_custody_wrapper_sha256"] != NAT_WRAPPER_SHA
            or manifest["checkpoint"] != reference["model"] or manifest["reference_sha256"] != base.REFERENCE_SHA
            or manifest["prior_candidate_sha256"] != PRIOR_SHA
            or manifest["baseline_buffer_state_sha256"] != prior["buffer_state_sha256"]
            or manifest["baseline_runtime_settings"] != prior["runtime_settings"]
            or manifest["original_model_state_sha256"] != reference["model_state_sha256"]
            or manifest["original_weight_sha256"] != reference["weight_sha256"]):
        raise base.StudyError("source, case or checkpoint binding differs")
    if not base.windows_path_absent(launch, checkout, revision, base.MANIFEST):
        raise base.StudyError("manifest was already present at source commit")
    base.committed_file(launch, checkout, head, checkout / base.MANIFEST, manifest_raw)
    review_raw = base.read(base.ordinary(args.manifest_review))
    review = json.loads(review_raw)
    if (review.get("manifest_sha256") != args.manifest_sha or not review.get("reviewer")
            or not review.get("reviewed_commit") or not review.get("observed_utc")):
        raise base.StudyError("manifest review differs")
    reviewed = review["reviewed_commit"]
    base.check_phase_ancestry(launch, checkout, revision, reviewed, head)
    base.committed_file(launch, checkout, reviewed, checkout / base.MANIFEST, manifest_raw)
    base.check_reported_order(manifest["created_utc"], review["observed_utc"], base.utc())
    development_review_raw = None
    development_directory = None
    if args.phase == "heldout":
        if not all((args.development_phase, args.development_record_sha, args.development_review)):
            raise base.StudyError("held-out phase requires published development evidence and independent review")
        development_review_raw = base.read(base.ordinary(args.development_review))
        development_review = json.loads(development_review_raw)
        development_directory, matrix_sha, result_sha = _check_development(
            launch, checkout, head, manifest, args.manifest_sha, reviewed,
            Path(args.development_phase) / "phase-record.json", args.development_record_sha, development_review)
        observed_binding = {"commit": development_review["reviewed_commit"],
            "references": [base.CASES, base.MANIFEST, DEVELOPMENT_DIR + "/phase-record.json",
                           DEVELOPMENT_DIR + "/development/matrix.f32", DEVELOPMENT_DIR + "/development/journal.jsonl",
                           DEVELOPMENT_DIR + "/development/results.json"],
            "hashes": [base.CASES_SHA, args.development_record_sha, matrix_sha, result_sha]}
        public = _current(launch, checkout, head, public_raw, args.inventory_sha,
                          json.loads(inventory_review_raw), minimum, observed_binding=observed_binding)
    cache, study_root = launch._checked_path(args.cache), launch._checked_path(args.study_root)
    if (not cache.is_dir() or not study_root.is_dir() or study_root == checkout or study_root.is_relative_to(checkout)):
        raise base.StudyError("cache or persistent CPU study root differs")
    attempt = workdir / ("answer-format-" + args.phase + "-" + uuid.uuid4().hex)
    _ACTIVE_ATTEMPT, _ATTEMPT_OWNED = attempt, False
    attempt.mkdir()
    _ATTEMPT_OWNED = True
    try:
        (attempt / "pycache").mkdir()
        base.write(attempt / "manifest-review-attestation.json", review_raw)
        base.write(attempt / "public-exposure.json", public_raw)
        base.write(attempt / "public-exposure-attestation.json", inventory_review_raw)
        if development_review_raw is not None:
            base.write(attempt / "development-review-attestation.json", development_review_raw)
        started = base.canonical({"phase": args.phase, "source_commit": revision, "checkout_commit": head,
            "manifest_sha256": args.manifest_sha, "manifest_review_sha256": base.digest(review_raw),
            "custodian_observations": custody,
            "current_public_exposure": {k: v for k, v in public.items() if k != "known_complete_rows"},
            "public_exposure_file_sha256": base.digest(public_raw),
            "public_exposure_attestation_sha256": base.digest(inventory_review_raw),
            "development_record_sha256": args.development_record_sha if args.phase == "heldout" else None,
            "development_review_sha256": base.digest(development_review_raw) if development_review_raw else None,
            "started_utc": base.utc(),
            "scope": "operator-reported ordering, not a trusted timestamp"})
        base.write(attempt / "started.json", started)
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
        if args.phase == "heldout":
            command += ["--development-phase", str(development_directory), "--development-matrix-sha", matrix_sha,
                        "--development-result-sha", result_sha]
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
            raise base.StudyError("supervisor did not confirm clean reaped owner and drained job")
        child_raw = base.read(base.ordinary(attempt / "child-result.json"))
        observed = json.loads(child_raw)
        if (observed["phase"] != args.phase or observed["manifest_sha256"] != args.manifest_sha
                or type(observed["owner_pid"]) is not int or observed["owner_pid"] != result.pid
                or observed["owner_run_id"] != result.run_id
                or (args.phase == "rehearsal" and observed.get("no_model_loaded") is not True)
                or (args.phase != "rehearsal" and observed.get("model_disposed") is not True)):
            raise base.StudyError("child identity or disposal differs")
        _imports(observed["imported_sources"], hashes)
        if (args.phase != "rehearsal" and (observed["post_work_checkpoint_assets"] != manifest["checkpoint"]["asset_sha256"]
                or observed["model_state_sha256"] != manifest["original_model_state_sha256"]
                or observed["weight_sha256"] != manifest["original_weight_sha256"]
                or observed["buffer_state_sha256"] != manifest["baseline_buffer_state_sha256"]
                or observed["runtime_settings"] != manifest["baseline_runtime_settings"])):
            raise base.StudyError("checkpoint or model state differs")
        if (launch._revision(checkout) != head or launch._verify_sources(checkout, sources) != hashes
                or base.read(base.ordinary(checkout / base.MANIFEST)) != manifest_raw
                or any((attempt / "pycache").iterdir())):
            raise base.StudyError("source, manifest or bytecode changed during run")
        finder.verify_inventory()
        artifacts = _inventory(attempt, args.phase, observed, revision)
        record = {"version": "answer-format-phase-v1", "status": "complete", "phase": args.phase,
                  "source_commit": revision, "checkout_commit": head, "manifest_sha256": args.manifest_sha,
                  "started_sha256": base.digest(started), "child_result_sha256": base.digest(child_raw),
                  "supervisor": asdict(result), "owner_reaped": True, "job_drained": True, "child": observed,
                  "artifacts": artifacts, "finished_utc": base.utc(),
                  "scope": "process record only; independent saved-row audit remains required"}
        data = base.canonical(record)
        base.write(attempt / "phase-record.json", data)
        print(json.dumps({"status": "complete", "phase": args.phase, "attempt": str(attempt),
                          "record_sha256": base.digest(data)}, sort_keys=True))
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
                base.record_failure(_ACTIVE_ATTEMPT, "answer-format", error)
            print("incomplete attempt: " + str(_ACTIVE_ATTEMPT), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(entry())
