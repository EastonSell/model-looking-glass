"""Cooperative supervised owner for the fixed public projected-edit study.

Use an isolated base interpreter and a trusted, quiescent checkout/cache. The
source hashes and process records are operational checks and attestations,
not protection against concurrent mutation or proof of executed native code.
Development and evaluation are separate supervised processes; an incomplete
or uncertain attempt is retained and must not be replayed automatically.
"""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types
import uuid


SCRIPT = "scripts/targeted_edit_projected_study.py"
PROTOCOL = "docs/TARGETED_EDIT_PROJECTED_CPU_V1.md"
SUPPLEMENT = "docs/TARGETED_EDIT_PROJECTED_DERIVATIVE_SCOPE.md"
CASES = "experiments/targeted_edit_projected_public_cases_v1.json"
REFERENCE = "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"
OWNER_DOCUMENT = "docs/TARGETED_EDIT_PROJECTED_OWNER.md"
MANIFEST = "experiments/targeted_edit_projected_run_v1.json"
EXTRA = (SCRIPT, PROTOCOL, SUPPLEMENT, OWNER_DOCUMENT, CASES, REFERENCE)
REFERENCE_SHA = "cb666aa911db020b8904dab6f08f989e506d85e875d7cac65e72d3b94cef2fa5"
CASES_SHA = "f7554a81ac7143c54a56d557fb04cb19419bac30e7dcd44c951eb63ddaff8ee8"
_ACTIVE_ATTEMPT = None
_ACTIVE_PHASE = None
_ATTEMPT_OWNED = False


class StudyError(RuntimeError):
    """The cooperative study could not establish a complete result."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8") + b"\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path, maximum=1_000_000):
    with Path(path).open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise StudyError("bounded file exceeds its limit")
    return data


def ordinary(path):
    """Reject redirected lexical components before opening a named file."""
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise StudyError("redirected path is unsupported")
    return lexical.resolve(strict=True)


def write(path, data):
    with Path(path).open("xb") as stream:
        if stream.write(data) != len(data):
            raise StudyError("short evidence write")
        stream.flush()
        os.fsync(stream.fileno())


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_reported_order(*timestamps):
    """Reject contradictory saved clocks without treating them as proof."""
    try:
        values = [datetime.fromisoformat(value) for value in timestamps]
    except (ValueError, TypeError) as error:
        raise StudyError("reported clock value is invalid") from error
    if (any(value.tzinfo is None for value in values)
            or any(left > right for left, right in zip(values, values[1:]))):
        raise StudyError("reported phase ordering is contradictory")


def load_script(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(read(path, 8_000_000), str(path), "exec"), module.__dict__)
    return module


def launcher(checkout):
    return load_script(checkout / "scripts/v2_source_launcher.py", "_projected_launcher")


def bootstrap(checkout, hashes):
    path = checkout / "scripts/v2_source_bootstrap.py"
    raw = read(path, 8_000_000)
    if digest(raw) != hashes.get("scripts/v2_source_bootstrap.py"):
        raise StudyError("source bootstrap differs")
    return load_script(path, "_projected_bootstrap")


def finder_for(checkout, hashes):
    boot = bootstrap(checkout, hashes)
    project = {path: value for path, value in hashes.items()
               if path.startswith("looking_glass/") and path.endswith(".py")}
    finder = boot._ProjectFinder(boot._source_map(checkout, project))
    sys.meta_path.insert(0, finder)
    return boot, finder


def check_public_payload(payload, expected_sha, cases, minimum_rows=(), *, include_rows=False):
    """Check every fixed row against a digest-pinned known-public snapshot."""
    actual = digest(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                               allow_nan=False).encode("utf-8"))
    if actual != expected_sha:
        raise StudyError("public inventory digest differs")
    rows = cases["tokenizer_preflight"]["cases"]
    reserved = {r["record_id"]: r for r in payload["records"]
                if r["record_id"].startswith("planned-projected-")}
    if len(rows) != 13 or len(reserved) != 13:
        raise StudyError("projected public reservations differ")
    for row in rows:
        record = reserved.get("planned-" + row["case_id"])
        if record is None or record["token_ids"] != row["token_ids"] or record["category"] != "planned":
            raise StudyError("a projected reservation differs")
    new = [tuple(row["token_ids"]) for row in rows]
    known = [tuple(row["token_ids"]) for row in payload["records"]]
    if not set(map(tuple, minimum_rows)) <= set(known):
        raise StudyError("current inventory omits a frozen known row")
    old = [tuple(row["token_ids"]) for row in payload["records"]
           if not row["record_id"].startswith("planned-projected-")]
    def related(a, b):
        return a[:len(b)] == b or b[:len(a)] == a
    if (any(related(a, b) for a in new for b in old)
            or any(related(a, b) for i, a in enumerate(new) for b in new[i + 1:])):
        raise StudyError("a projected row has a known public prefix relation")
    result = {"version": payload["version"], "canonical_sha256": actual,
            "record_count": len(payload["records"]), "projected_rows": len(new),
            "other_row_comparisons": len(new) * len(old), "known_relations": 0,
            "scope": "public inventory only; unknown history and private splits are not certified"}
    if include_rows:
        result["known_complete_rows"] = [list(row) for row in known]
    return result


def public_inventory(checkout, hashes, version, cases):
    if type(version) is not int or not 13 <= version <= 99:
        raise StudyError("invalid inventory version")
    _boot, finder = finder_for(checkout, hashes)
    module = importlib.import_module("looking_glass.pretrained_exposures_v" + str(version))
    result = check_public_payload(module.inventory_payload(), module.INVENTORY_SHA256, cases, include_rows=True)
    finder.verify_inventory()
    return result


def custody_observation(path):
    """Validate the operator's recent, scoped reports; no private rows are read."""
    observation = json.loads(read(path))
    rows = observation.get("observations")
    if (type(rows) is not list or len(rows) != 2
            or {row.get("custodian") for row in rows} != {"natural", "injection"}):
        raise StudyError("custodian observations are incomplete")
    now = datetime.now(timezone.utc)
    for row in rows:
        if row.get("status") != "no_protected_rows_reported" or type(row.get("reported_by")) is not str:
            raise StudyError("custodian status is not an explicit no-input report")
        observed = datetime.fromisoformat(row["observed_utc"])
        if observed.tzinfo is None or not 0 <= (now - observed).total_seconds() <= 900:
            raise StudyError("custodian observation is stale or contradictory")
    return {"observations": rows, "scope": "only the named custodians; no private comparison or complete history is certified"}


def freeze(checkout, source_revision, hashes, exposure_version, custody):
    case_bytes, reference_bytes = read(ordinary(checkout / CASES)), read(ordinary(checkout / REFERENCE))
    if digest(case_bytes) != CASES_SHA or digest(reference_bytes) != REFERENCE_SHA:
        raise StudyError("fixed case or reference bytes differ")
    cases, reference = json.loads(case_bytes), json.loads(reference_bytes)
    exposure = public_inventory(checkout, hashes, exposure_version, cases)
    payload = {"version": "targeted-edit-projected-run-manifest-v1",
               "source_commit": source_revision, "source_sha256": hashes,
               "protocol": {"path": PROTOCOL, "sha256": hashes[PROTOCOL],
                            "supplement_path": SUPPLEMENT, "supplement_sha256": hashes[SUPPLEMENT]},
               "cases_path": CASES, "cases_sha256": CASES_SHA,
               "evaluation_cases": cases["tokenizer_preflight"]["cases"],
               "development": {"discovery_ids": [464, 3139, 1748, 286, 4881, 318],
                   "calibration_ids": [[464, 4387, 9151, 318, 262], [7571, 5556, 734, 21767],
                                       [464, 6697, 286, 3024, 318], [32, 1285, 468, 3598]],
                   "target_id": 3576, "competitor_id": 6342},
               "checkpoint": reference["model"], "original_model_state_sha256": reference["model_state_sha256"],
               "original_weight_sha256": reference["weight_sha256"], "reference_sha256": REFERENCE_SHA,
               "exposure": exposure,
               "custodian_observations": custody,
               "created_utc": utc(), "ordering_scope": "operator-recorded commit and review ordering; not a trusted timestamp",
               "execution_scope": "trusted quiescent checkout and cache; hashes do not prove native or already executed bytes"}
    write(checkout / MANIFEST, canonical(payload))
    return payload


def committed_file(launch, checkout, revision, path, data):
    lexical = Path(os.path.abspath(path))
    resolved = launch._checked_path(lexical)
    if resolved != lexical:
        raise StudyError("committed lexical path differs")
    relative = lexical.relative_to(checkout).as_posix()
    tree = launch._git(checkout, "--literal-pathspecs", "ls-tree", "-z", "--full-tree", revision, "--", relative)
    expected_blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
    expected = ("100644 blob " + expected_blob + "\t" + relative).encode("utf-8") + b"\0"
    if tree != expected:
        raise StudyError("evidence bytes are not a regular committed file")


def commit_object(launch, checkout, identity):
    if (type(identity) is not str or re.fullmatch(r"[0-9a-f]{40}", identity) is None
            or launch._git(checkout, "cat-file", "-t", identity).strip() != b"commit"):
        raise StudyError("Git identity is not a commit object")


def windows_path_absent(launch, checkout, revision, path):
    """Inspect historical tree components for ordinary Windows path aliases."""
    if type(path) is not str or not path or "\\" in path or ":" in path:
        raise StudyError("invalid chronology path")
    parts = path.split("/")
    for part in parts:
        base = part.split(".", 1)[0].upper()
        if (not part or part in (".", "..") or not part.isascii() or part[-1] in " ."
                or "~" in part or any(ord(char) < 32 or char in '<>"|?*' for char in part)
                or base in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", base)):
            raise StudyError("ambiguous Windows chronology path")
    tree = revision
    for index, part in enumerate(parts):
        entries = launch._git(checkout, "ls-tree", "-z", tree)
        matches = []
        for entry in entries.split(b"\0"):
            if not entry:
                continue
            try:
                header, raw_name = entry.split(b"\t", 1)
                mode, kind, obj = header.split(b" ")
                name = raw_name.decode("utf-8")
            except (ValueError, UnicodeError) as error:
                raise StudyError("committed tree entry is invalid") from error
            if not name.isascii():
                raise StudyError("non-ASCII tree entry prevents chronology checking")
            if name.rstrip(" .").lower() == part.lower():
                matches.append((mode, kind, obj))
        if not matches:
            return True
        if len(matches) != 1 or index == len(parts) - 1:
            return False
        mode, kind, obj = matches[0]
        if mode != b"040000" or kind != b"tree" or re.fullmatch(rb"[0-9a-f]{40}", obj) is None:
            return False
        tree = obj.decode("ascii")
    return False


def check_inventory_attestation(launch, checkout, head, attestation, expected_sha, expected_version):
    """Bind the reported current snapshot to a committed reviewed source label."""
    path = attestation.get("source_path")
    revision = attestation.get("reviewed_commit")
    if (type(path) is not str or re.fullmatch(r"looking_glass/pretrained_exposures_v[0-9]+\.py", path) is None
            or type(revision) is not str or not attestation.get("reviewer")
            or attestation.get("inventory_sha256") != expected_sha or not attestation.get("observed_utc")):
        raise StudyError("public inventory observation differs")
    observed = datetime.fromisoformat(attestation["observed_utc"])
    if observed.tzinfo is None or observed > datetime.now(timezone.utc):
        raise StudyError("public inventory observation time differs")
    commit_object(launch, checkout, revision)
    launch._git(checkout, "merge-base", "--is-ancestor", revision, head)
    source = launch._checked_path(checkout / path)
    raw = read(source, 1_000_000)
    if digest(raw) != attestation.get("source_sha256"):
        raise StudyError("public inventory source digest differs")
    committed_file(launch, checkout, revision, source, raw)
    literals = {}
    for statement in ast.parse(raw).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            if statement.targets[0].id in ("INVENTORY_SHA256", "INVENTORY_VERSION"):
                literals[statement.targets[0].id] = ast.literal_eval(statement.value)
    if literals != {"INVENTORY_SHA256": expected_sha, "INVENTORY_VERSION": expected_version}:
        raise StudyError("committed inventory source labels differ")


def committed_candidate(launch, checkout, revision, directory, candidate_sha):
    candidate_path = launch._checked_path(directory / "pre-evaluation-candidate.json")
    raw = read(candidate_path)
    if digest(raw) != candidate_sha:
        raise StudyError("candidate digest differs")
    candidate = json.loads(raw)
    committed_file(launch, checkout, revision, candidate_path, raw)
    for name, item in candidate["artifacts"].items():
        if Path(name).name != name or type(item["bytes"]) is not int or not 0 <= item["bytes"] <= 10_000_000:
            raise StudyError("invalid candidate artifact")
        path = launch._checked_path(directory / name)
        raw = read(path, item["bytes"])
        if len(raw) != item["bytes"] or digest(raw) != item["sha256"]:
            raise StudyError("candidate artifact changed")
        committed_file(launch, checkout, revision, path, raw)
    return candidate


def inventory_attempt(attempt, phase):
    """Bound and hash completed phase artifacts before writing its record."""
    paths = [attempt / "started.json", attempt / "child-result.json", attempt / "manifest-review-attestation.json",
             attempt / "public-exposure.json", attempt / "public-exposure-attestation.json"]
    if phase == "evaluation":
        paths.append(attempt / "review-attestation.json")
    if phase in ("development", "evaluation"):
        directory = attempt / ("candidate" if phase == "development" else "evaluation")
        ordinary(directory)
        if not directory.is_dir():
            raise StudyError("phase output directory differs")
        output_paths = sorted(directory.iterdir())
        if not output_paths:
            raise StudyError("phase output directory is empty")
        paths.extend(output_paths)
    if not 1 <= len(paths) <= 100:
        raise StudyError("phase artifact count differs")
    entries, total = {}, 0
    for path in paths:
        ordinary(path)
        if not path.is_file():
            raise StudyError("phase artifact is not an ordinary file")
        raw = read(path, 10_000_000)
        total += len(raw)
        if total > 100_000_000:
            raise StudyError("phase artifacts exceed the total bound")
        entries[path.relative_to(attempt).as_posix()] = {"bytes": len(raw), "sha256": digest(raw)}
    return entries


def validate_phase_output(attempt, phase, observed, inventory):
    """Require the complete bounded output set before a phase can be complete."""
    if phase == "rehearsal":
        return
    if phase == "development":
        raw = read(ordinary(attempt / "candidate" / "pre-evaluation-candidate.json"))
        if digest(raw) != observed["candidate_sha256"]:
            raise StudyError("candidate digest differs from child report")
        candidate = json.loads(raw)
        conditions = ("projected", "unprojected", "projected_random", "opposite", "nuisance")
        names = {"discovery-logits.f32", "discovery-grad-inputs.f32", "discovery-grad-outputs.f32",
                 "input-factor.f32", "output-factor.f32", "full-gradient.f32", "discovery-original-logits.f32",
                 "discovery-original-inputs.f32", "discovery-original-outputs.f32", "calibration-inputs.f32",
                 "calibration-original-outputs.f32", "calibration-original-logits.f32", "state-original.json"}
        names |= {"factor-" + factor + ".f64" for factor in ("calibration_basis", "projected_input", "random_input",
                  "random_output", "projected_random_input", "nuisance_basis", "nuisance_input")}
        for condition in conditions:
            names |= {"calibration-" + condition + "-outputs.f32", "calibration-" + condition + "-logits.f32",
                      "discovery-" + condition + "-logits.f32", "discovery-" + condition + "-inputs.f32",
                      "discovery-" + condition + "-outputs.f32", "state-" + condition + ".json"}
        if candidate.get("status") != "private-candidate" or set(candidate.get("artifacts", {})) != names:
            raise StudyError("development candidate artifact list is incomplete")
        recorded = {name.removeprefix("candidate/") for name in inventory if name.startswith("candidate/")}
        if recorded != names | {"pre-evaluation-candidate.json"}:
            raise StudyError("development phase output set differs")
        for name, item in candidate["artifacts"].items():
            saved = inventory["candidate/" + name]
            if item["bytes"] != saved["bytes"] or item["sha256"] != saved["sha256"]:
                raise StudyError("development phase artifact differs from candidate")
    else:
        conditions = ("original", "projected", "unprojected", "projected_random", "opposite", "nuisance")
        names = {"observations.jsonl", "provisional-report.json"} | {
            prefix + condition + suffix for condition in conditions
            for prefix, suffix in (("", ".f32"), ("state-", ".json"))}
        recorded = {name.removeprefix("evaluation/") for name in inventory if name.startswith("evaluation/")}
        if recorded != names:
            raise StudyError("evaluation phase output set differs")
        for condition in conditions:
            if inventory["evaluation/" + condition + ".f32"]["bytes"] != 13 * 50257 * 4:
                raise StudyError("evaluation matrix size differs")
        report_raw = read(ordinary(attempt / "evaluation" / "provisional-report.json"))
        report = json.loads(report_raw)
        if (digest(report_raw) != observed["report_sha256"]
                or report["provisional_numerical_decision"] != observed["provisional_numerical_decision"]):
            raise StudyError("evaluation report differs from child report")


def check_committed_inventory(launch, checkout, revision, directory, inventory):
    if type(inventory) is not dict or not 1 <= len(inventory) <= 100:
        raise StudyError("committed phase inventory differs")
    total = 0
    for name, item in inventory.items():
        path = Path(name)
        if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
            raise StudyError("committed phase artifact path differs")
        size = item["bytes"]
        if type(size) is not int or not 0 <= size <= 10_000_000:
            raise StudyError("committed phase artifact size differs")
        total += size
        if total > 100_000_000:
            raise StudyError("committed phase inventory exceeds its bound")
        target = launch._checked_path(directory / path)
        data = read(target, size)
        if len(data) != size or digest(data) != item["sha256"]:
            raise StudyError("committed phase artifact differs")
        committed_file(launch, checkout, revision, target, data)


def check_exact_phase_files(launch, checkout, revision, directory, inventory):
    """Reject a lost-ack failure marker or any other unlisted artifact."""
    relative = directory.resolve(strict=True).relative_to(checkout).as_posix()
    expected = set(inventory) | {"phase-record.json"}
    actual = set()
    def scan(current):
        for path in current.iterdir():
            launch._checked_path(path)
            if path.is_dir():
                scan(path)
            elif path.is_file():
                actual.add(path.relative_to(directory).as_posix())
            else:
                raise StudyError("phase contains an unsupported path")
    scan(directory)
    tree = launch._git(checkout, "--literal-pathspecs", "ls-tree", "-rz", "--full-tree", revision, "--", relative)
    committed = set()
    for entry in tree.split(b"\0"):
        if entry:
            header, path = entry.split(b"\t", 1)
            if not header.startswith(b"100644 blob "):
                raise StudyError("phase contains a non-regular committed artifact")
            name = path.decode("utf-8")
            if not name.startswith(relative + "/"):
                raise StudyError("committed phase path differs")
            committed.add(name[len(relative) + 1:])
    if actual != expected or committed != expected:
        raise StudyError("phase artifacts are missing or include an unlisted failure")


def check_candidate_binding(launch, record_path, candidate, document, inventory):
    if candidate != launch._checked_path(record_path.parent / "candidate"):
        raise StudyError("candidate directory differs from the recorded phase")
    declared = {name for name in inventory if name.startswith("candidate/")}
    expected = {"candidate/pre-evaluation-candidate.json"} | {
        "candidate/" + name for name in document["artifacts"]}
    if declared != expected:
        raise StudyError("candidate and phase inventories differ")


def check_phase_ancestry(launch, checkout, source, manifest_reviewed, head, development_reviewed=None):
    """Require a real chain, not two unrelated parents of the same merge."""
    for identity in (source, manifest_reviewed, head, development_reviewed):
        if identity is not None:
            commit_object(launch, checkout, identity)
    launch._git(checkout, "merge-base", "--is-ancestor", source, manifest_reviewed)
    launch._git(checkout, "merge-base", "--is-ancestor", manifest_reviewed, head)
    if development_reviewed is not None:
        launch._git(checkout, "merge-base", "--is-ancestor", manifest_reviewed, development_reviewed)
        launch._git(checkout, "merge-base", "--is-ancestor", development_reviewed, head)


def check_imports(imports, hashes):
    required = {"looking_glass/" + name + ".py" for name in (
        "__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
        "pretrained", "targeted_edit_cpu_v1", "targeted_edit_v2_edits", "targeted_edit_v2_owner",
        "targeted_edit_projected_capture", "targeted_edit_projected_development",
        "targeted_edit_projected_math", "targeted_edit_projected_execution", "targeted_edit_projected_score")}
    if (type(imports) is not dict or not required <= set(imports)
            or any(hashes.get(name) != value for name, value in imports.items())):
        raise StudyError("child import inventory differs from frozen sources")


def check_development_record(record, manifest, manifest_sha, candidate_sha):
    observed, supervision = record["child"], record["supervisor"]
    if (record["status"] != "complete" or record["phase"] != "development"
            or record["source_commit"] != manifest["source_commit"]
            or record["manifest_sha256"] != manifest_sha or record["job_drained"] is not True
            or record["owner_reaped"] is not True
            or observed["phase"] != "development" or observed["manifest_sha256"] != manifest_sha
            or observed["candidate_sha256"] != candidate_sha or observed["model_disposed"] is not True
            or type(supervision["pid"]) is not int or supervision["pid"] <= 0
            or type(observed["owner_pid"]) is not int
            or type(supervision["run_id"]) is not str
            or re.fullmatch(r"[0-9a-f]{32}", supervision["run_id"]) is None
            or observed["owner_pid"] != supervision["pid"] or observed["owner_run_id"] != supervision["run_id"]
            or supervision["clean"] is not True or type(supervision["exit_code"]) is not int
            or supervision["exit_code"] != 0
            or supervision["timed_out"] is not False):
        raise StudyError("development supervision or candidate binding differs")
    if observed["post_work_checkpoint_assets"] != manifest["checkpoint"]["asset_sha256"]:
        raise StudyError("development checkpoint asset inventory differs")
    check_imports(observed["imported_sources"], manifest["source_sha256"])


def record_failure(attempt, phase, primary):
    try:
        write(attempt / "failure.json", canonical({"status": "incomplete", "phase": phase,
              "error_type": type(primary).__name__, "time_utc": utc()}))
    except BaseException as recording:
        if (isinstance(recording, (KeyboardInterrupt, SystemExit))
                and not isinstance(primary, (KeyboardInterrupt, SystemExit))):
            raise recording from primary


def supervise_child(run_cpu_study, command, study_root, *, cwd, env, job):
    """Use exactly the root that the direct child's command will claim."""
    if command.count("--study-root") != 1:
        raise StudyError("child command lacks one persistent study root")
    index = command.index("--study-root")
    if index + 1 >= len(command) or Path(command[index + 1]) != study_root:
        raise StudyError("supervisor and child study roots differ")
    return run_cpu_study(command, study_root=study_root, timeout=600, lease_timeout=5,
                         terminate_grace=5, cwd=cwd, env=env, process_job=job)


def setup(checkout, venv, dependency, work, source_revision=None):
    launch = launcher(checkout)
    checkout, venv = launch._checked_path(checkout), launch._checked_path(venv)
    dependency, work = launch._checked_path(dependency), launch._checked_path(work)
    executable = launch._check_startup(checkout, venv, (dependency,), work)
    head = launch._revision(checkout)
    revision = source_revision or head
    if revision != head:
        launch._git(checkout, "merge-base", "--is-ancestor", revision, head)
    sources = launch._committed_sources(checkout, revision, extra=EXTRA)
    hashes = launch._verify_sources(checkout, sources)
    return launch, executable, head, revision, sources, hashes


def check_projected_tokenizer(tokenizer, rows):
    for row in rows:
        ids = row["token_ids"]
        if tokenizer.encode(row["prompt"], add_special_tokens=False) != ids:
            raise StudyError("evaluation prompt tokenization differs")
        for continuation in row["continuations"]:
            if (tokenizer.encode(continuation["text"], add_special_tokens=False) != continuation["ids"]
                    or tokenizer.encode(row["prompt"] + continuation["text"], add_special_tokens=False)
                    != ids + continuation["ids"]):
                raise StudyError("evaluation continuation tokenization differs")


def available_memory():
    """Read current Windows available physical memory; this is not a reservation."""
    import ctypes
    from ctypes import wintypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise StudyError("available physical memory could not be observed")
    return int(status.available_physical)


def verify_cached_assets(pretrained, cache):
    """Rehash the seven pinned offline cache files after model work."""
    from huggingface_hub import hf_hub_download
    observed = {}
    for name, (expected_size, expected_sha) in pretrained.ASSETS.items():
        path = hf_hub_download(repo_id=pretrained.MODEL_ID, filename=name, revision=pretrained.REVISION,
                               cache_dir=cache, local_files_only=True, token=False)
        h, size = hashlib.sha256(), 0
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > expected_size:
                    raise StudyError("checkpoint asset size changed")
                h.update(block)
        if size != expected_size or h.hexdigest() != expected_sha:
            raise StudyError("checkpoint asset changed after loading")
        observed[name] = h.hexdigest()
    return observed


def child(args):
    checkout, attempt = ordinary(args.checkout), ordinary(args.attempt)
    if (not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != attempt / "pycache"
            or any((attempt / "pycache").iterdir())
            or any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules)):
        raise StudyError("isolated child startup differs")
    manifest_raw = read(ordinary(checkout / MANIFEST))
    if digest(manifest_raw) != args.manifest_sha:
        raise StudyError("manifest digest differs in child")
    manifest = json.loads(manifest_raw)
    hashes = manifest["source_sha256"]
    if digest(read(checkout / SCRIPT)) != hashes[SCRIPT]:
        raise StudyError("child source differs")
    boot, finder = finder_for(checkout, hashes)
    boot._check_search_path(checkout)
    boot._prime_parent_platform(json.loads(args.parent_platform))
    import platform
    if os.name != "nt" or platform._platform_cache:
        raise StudyError("platform was observed before spawn-free setup")
    win = sys.getwindowsversion()
    platform._syscmd_ver = lambda *_args, **_kwargs: ("Windows", str(win.major), f"{win.major}.{win.minor}.{win.build}")
    boot._install_process_guard()
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(Path(args.study_root).resolve(strict=True))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    available = available_memory()
    if available < 3 * 1024**3:
        raise StudyError("available physical memory is below the study admission floor")
    boot._add_dependencies(checkout, Path(args.venv).resolve(strict=True), [args.dependency])
    import gc
    import weakref
    import torch
    from looking_glass import pretrained
    from looking_glass import targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_v2_edits as edits
    from looking_glass import targeted_edit_v2_owner as common
    from looking_glass import targeted_edit_projected_development as development
    from looking_glass import targeted_edit_projected_execution as execution
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False) or torch.cuda.is_initialized():
        raise StudyError("CPU numeric mode differs")
    reference = json.loads(read(ordinary(checkout / REFERENCE)))
    env = v1._environment()
    if v1._canonical({k: env[k] for k in edits.ENVIRONMENT_FIELDS}) != v1._canonical(
            {k: reference["environment"][k] for k in edits.ENVIRONMENT_FIELDS}):
        raise StudyError("runtime differs from the v1 reference")
    runtime = common._runtime_settings(torch)
    if (runtime["intraop_threads"] != 1 or runtime["interop_threads"] != 1
            or runtime["deterministic_algorithms"] is not True or runtime["deterministic_warn_only"] is not False
            or runtime["cpu_autocast_enabled"] is not False or runtime["inference_mode_enabled"] is not False
            or runtime["cuda_visibility"] != "-1" or runtime["cuda_initialized"] is not False
            or runtime["hf_hub_offline"] != "1" or runtime["transformers_offline"] != "1"):
        raise StudyError("initial CPU runtime settings differ")
    case_bytes = read(ordinary(checkout / CASES))
    if digest(case_bytes) != manifest["cases_sha256"]:
        raise StudyError("case bytes differ")
    if args.phase == "rehearsal":
        sources = finder.verify_inventory()
        if torch.cuda.is_initialized() or any((attempt / "pycache").iterdir()):
            raise StudyError("rehearsal CPU or bytecode state differs")
        result = {"phase": "rehearsal", "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                  "manifest_sha256": args.manifest_sha, "model_disposed": False, "no_model_loaded": True,
                  "available_physical_bytes_before_load": available, "runtime_settings": runtime,
                  "runtime": {k: env[k] for k in edits.ENVIRONMENT_FIELDS}, "imported_sources": sources,
                  "finished_utc": utc(), "scope": "no checkpoint or model forward; diagnostic only"}
        write(attempt / "child-result.json", canonical(result))
        session.attest_clean(lease)
        return

    def work():
        loaded = pretrained.load_distilgpt2(cache_dir=args.cache, allow_download=False)
        if common._runtime_settings(torch) != runtime:
            raise StudyError("runtime settings changed during checkpoint loading")
        model, module, weight = common._check_checkpoint(loaded, {"v1_compatibility": {"model": manifest["checkpoint"]}}, torch, v1)
        v1._verify_tokenizer(loaded.tokenizer)
        check_projected_tokenizer(loaded.tokenizer, manifest["evaluation_cases"])
        guard = common._ModelGuard(model, module, weight, torch, v1)
        if guard.runtime_settings != runtime:
            raise StudyError("runtime settings changed during tokenizer or model verification")
        if guard.state_sha != manifest["original_model_state_sha256"] or guard.weight_sha != manifest["original_weight_sha256"]:
            raise StudyError("original model state or site weight differs from v1")
        fixed = manifest["development"]
        if args.phase == "development":
            candidate = development.prepare_development(guard, fixed["discovery_ids"], fixed["calibration_ids"],
                                                        fixed["target_id"], fixed["competitor_id"], attempt / "candidate")
            result = {"candidate_sha256": digest(read(attempt / "candidate" / "pre-evaluation-candidate.json")),
                      "model_state_sha256": guard.state_sha, "weight_sha256": guard.weight_sha,
                      "buffer_state_sha256": guard.buffer_sha, "discovery_margin": candidate["discovery_margin"]}
        else:
            handoff = execution.reconstruct(guard, args.candidate, args.candidate_sha,
                fixed["discovery_ids"], fixed["calibration_ids"], fixed["target_id"], fixed["competitor_id"])
            report = execution.evaluate(handoff, case_bytes, attempt / "evaluation")
            result = {"candidate_sha256": args.candidate_sha,
                      "provisional_numerical_decision": report["provisional_numerical_decision"],
                      "report_sha256": digest(read(attempt / "evaluation" / "provisional-report.json"))}
        guard.verify()
        return result, weakref.ref(model), weakref.ref(weight)

    result, model_ref, weight_ref = work()
    gc.collect()
    if model_ref() is not None or weight_ref() is not None or torch.cuda.is_initialized():
        raise StudyError("model disposal or CPU state could not be verified")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or any((attempt / "pycache").iterdir()):
        raise StudyError("mask or bytecode cache changed")
    final_assets = verify_cached_assets(pretrained, args.cache)
    sources = finder.verify_inventory()
    result.update({"phase": args.phase, "owner_pid": os.getpid(), "owner_run_id": session.run_id,
                   "manifest_sha256": args.manifest_sha, "model_disposed": True,
                   "no_model_loaded": False,
                   "available_physical_bytes_before_load": available,
                   "memory_scope": "point-in-time admission observation, not a reservation or peak bound",
                   "post_work_checkpoint_assets": final_assets,
                   "runtime_settings": runtime,
                   "runtime": {k: env[k] for k in edits.ENVIRONMENT_FIELDS}, "imported_sources": sources,
                   "finished_utc": utc(), "scope": "recorded cooperative execution, not authenticated native code"})
    write(attempt / "child-result.json", canonical(result))
    session.attest_clean(lease)


def main(argv=None):
    global _ACTIVE_ATTEMPT, _ACTIVE_PHASE, _ATTEMPT_OWNED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "child"))
    for name in ("checkout", "venv", "dependency", "work"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--exposure-version", type=int)
    parser.add_argument("--custody-observation")
    parser.add_argument("--current-public-inventory")
    parser.add_argument("--current-inventory-sha")
    parser.add_argument("--current-inventory-attestation")
    parser.add_argument("--phase", choices=("rehearsal", "development", "evaluation"))
    parser.add_argument("--manifest-sha")
    parser.add_argument("--manifest-review-attestation")
    parser.add_argument("--candidate")
    parser.add_argument("--candidate-sha")
    parser.add_argument("--development-record")
    parser.add_argument("--development-record-sha")
    parser.add_argument("--review-attestation")
    parser.add_argument("--cache")
    parser.add_argument("--study-root")
    parser.add_argument("--attempt")
    parser.add_argument("--parent-platform")
    args = parser.parse_args(argv)
    checkout, venv, dependency, workdir = map(ordinary, (args.checkout, args.venv, args.dependency, args.work))
    if args.mode == "child":
        child(args)
        return 0
    if args.mode == "freeze":
        if not args.custody_observation or args.exposure_version is None:
            raise StudyError("freezing requires an explicit reviewed exposure version and recent scoped custodian observations")
        _launch, _exe, _head, revision, _sources, hashes = setup(checkout, venv, dependency, workdir)
        manifest = freeze(checkout, revision, hashes, args.exposure_version, custody_observation(args.custody_observation))
        exposure_summary = {k: v for k, v in manifest["exposure"].items() if k != "known_complete_rows"}
        print(json.dumps({"manifest_sha256": digest(canonical(manifest)), "source_commit": revision,
                          "exposure": exposure_summary}, sort_keys=True))
        return 0
    if (args.phase is None or not args.manifest_sha or not args.cache or not args.study_root
            or not args.custody_observation or not args.manifest_review_attestation
            or not args.current_public_inventory or not args.current_inventory_sha
            or not args.current_inventory_attestation):
        raise StudyError("run requires reviewed manifest, current public inventory, cache, study root and recent custody observations")
    manifest_raw = read(ordinary(checkout / MANIFEST))
    if digest(manifest_raw) != args.manifest_sha:
        raise StudyError("manifest digest differs")
    manifest = json.loads(manifest_raw)
    custody = custody_observation(args.custody_observation)
    current_public_raw = read(args.current_public_inventory, 5_000_000)
    launch, executable, head, _revision, sources, hashes = setup(checkout, venv, dependency, workdir, manifest["source_commit"])
    current_public = check_public_payload(json.loads(current_public_raw), args.current_inventory_sha,
                                          json.loads(read(ordinary(checkout / CASES))),
                                          manifest["exposure"]["known_complete_rows"])
    inventory_attestation_raw = read(args.current_inventory_attestation)
    check_inventory_attestation(launch, checkout, head, json.loads(inventory_attestation_raw),
                                args.current_inventory_sha, current_public["version"])
    cache = launch._checked_path(args.cache)
    study_root = launch._checked_path(args.study_root)
    if not cache.is_dir() or not study_root.is_dir() or study_root == checkout or study_root.is_relative_to(checkout):
        raise StudyError("cache or persistent CPU study root is invalid")
    if hashes != manifest["source_sha256"]:
        raise StudyError("source inventory differs from manifest")
    if not windows_path_absent(launch, checkout, manifest["source_commit"], MANIFEST):
        raise StudyError("manifest was already present at the source commit")
    committed_file(launch, checkout, head, checkout / MANIFEST, manifest_raw)
    manifest_review_raw = read(args.manifest_review_attestation)
    manifest_review = json.loads(manifest_review_raw)
    if (manifest_review.get("manifest_sha256") != args.manifest_sha or not manifest_review.get("reviewer")
            or not manifest_review.get("reviewed_commit") or not manifest_review.get("observed_utc")):
        raise StudyError("manifest publication observation differs")
    manifest_reviewed = manifest_review["reviewed_commit"]
    check_phase_ancestry(launch, checkout, manifest["source_commit"], manifest_reviewed, head)
    committed_file(launch, checkout, manifest_reviewed, checkout / MANIFEST, manifest_raw)
    manifest_observed_time = datetime.fromisoformat(manifest_review["observed_utc"])
    if manifest_observed_time.tzinfo is None or manifest_observed_time > datetime.now(timezone.utc):
        raise StudyError("manifest observation time differs")
    check_reported_order(manifest["created_utc"], manifest_review["observed_utc"], utc())
    if args.phase == "evaluation":
        if (not args.candidate or not args.candidate_sha or not args.development_record
                or not args.development_record_sha or not args.review_attestation):
            raise StudyError("evaluation requires committed development evidence and an independent review attestation")
        candidate = launch._checked_path(args.candidate)
        review_raw = read(args.review_attestation)
        review = json.loads(review_raw)
        if (review.get("candidate_sha256") != args.candidate_sha or review.get("manifest_sha256") != args.manifest_sha
                or review.get("development_record_sha256") != args.development_record_sha
                or not review.get("reviewed_commit") or not review.get("reviewer") or not review.get("observed_utc")):
            raise StudyError("independent review attestation differs")
        reviewed = review["reviewed_commit"]
        check_phase_ancestry(launch, checkout, manifest["source_commit"], manifest_reviewed, head, reviewed)
        observed_time = datetime.fromisoformat(review["observed_utc"])
        if observed_time.tzinfo is None or observed_time > datetime.now(timezone.utc):
            raise StudyError("review observation time differs")
        record_path = launch._checked_path(args.development_record)
        development_relative = record_path.parent.relative_to(checkout).as_posix()
        if not windows_path_absent(launch, checkout, manifest_reviewed, development_relative):
            raise StudyError("development directory existed at the manifest publication commit")
        if candidate != launch._checked_path(record_path.parent / "candidate"):
            raise StudyError("candidate directory differs from the recorded phase")
        record_raw = read(record_path)
        if digest(record_raw) != args.development_record_sha:
            raise StudyError("development phase record digest differs")
        committed_file(launch, checkout, reviewed, record_path, record_raw)
        development_record = json.loads(record_raw)
        check_development_record(development_record, manifest, args.manifest_sha, args.candidate_sha)
        development_started = launch._checked_path(record_path.with_name("started.json"))
        development_started_raw = read(development_started)
        if digest(development_started_raw) != development_record["started_sha256"]:
            raise StudyError("development start record differs")
        started_document = json.loads(development_started_raw)
        if (started_document["phase"] != "development" or started_document["manifest_sha256"] != args.manifest_sha
                or development_record["manifest_review_attestation_sha256"] != digest(manifest_review_raw)
                or started_document["manifest_review_attestation_sha256"] != digest(manifest_review_raw)):
            raise StudyError("development start and manifest observation differ")
        check_reported_order(manifest["created_utc"], manifest_review["observed_utc"],
                             started_document["started_utc"], development_record["child"]["finished_utc"],
                             development_record["finished_utc"], review["observed_utc"], utc())
        committed_file(launch, checkout, reviewed, development_started, development_started_raw)
        development_child = launch._checked_path(record_path.with_name("child-result.json"))
        development_child_raw = read(development_child)
        if (digest(development_child_raw) != development_record["child_result_sha256"]
                or canonical(json.loads(development_child_raw)) != canonical(development_record["child"])):
            raise StudyError("development child result differs")
        committed_file(launch, checkout, reviewed, development_child, development_child_raw)
        check_committed_inventory(launch, checkout, reviewed, record_path.parent, development_record["artifacts"])
        check_exact_phase_files(launch, checkout, reviewed, record_path.parent, development_record["artifacts"])
        candidate_document = committed_candidate(launch, checkout, reviewed, candidate, args.candidate_sha)
        check_candidate_binding(launch, record_path, candidate, candidate_document, development_record["artifacts"])
        development_child_fields = development_record["child"]
        if (development_child_fields["runtime_settings"] != candidate_document["runtime_settings"]
                or development_child_fields["model_state_sha256"] != candidate_document["model_state_sha256"]
                or development_child_fields["weight_sha256"] != candidate_document["weight_sha256"]
                or development_child_fields["buffer_state_sha256"] != candidate_document["buffer_state_sha256"]):
            raise StudyError("development child and candidate state bindings differ")
    attempt = workdir / ("projected-" + args.phase + "-" + uuid.uuid4().hex)
    _ACTIVE_ATTEMPT, _ACTIVE_PHASE = attempt, args.phase
    _ATTEMPT_OWNED = False
    attempt.mkdir()
    _ATTEMPT_OWNED = True
    (attempt / "pycache").mkdir()
    write(attempt / "manifest-review-attestation.json", manifest_review_raw)
    write(attempt / "public-exposure.json", current_public_raw)
    write(attempt / "public-exposure-attestation.json", inventory_attestation_raw)
    if args.phase == "evaluation":
        write(attempt / "review-attestation.json", review_raw)
    started_raw = canonical({"phase": args.phase, "source_commit": manifest["source_commit"],
          "checkout_commit": head, "manifest_sha256": args.manifest_sha, "candidate_sha256": args.candidate_sha,
          "development_record_sha256": args.development_record_sha,
          "review_attestation": review if args.phase == "evaluation" else None,
          "review_attestation_sha256": digest(review_raw) if args.phase == "evaluation" else None,
          "manifest_review_attestation_sha256": digest(manifest_review_raw),
          "custodian_observations": custody,
          "current_public_exposure": current_public,
          "public_exposure_file_sha256": digest(current_public_raw),
          "public_exposure_attestation_sha256": digest(inventory_attestation_raw),
          "started_utc": utc(), "scope": "operator-observed ordering, not a trusted timestamp"})
    write(attempt / "started.json", started_raw)
    parent_platform = launch._parent_platform_observation()
    _boot, finder = finder_for(checkout, hashes)
    from looking_glass.cpu_study_supervision import run_cpu_study
    guards = launch._load_guards(checkout, sources)
    job = guards.WindowsJob()
    command = [str(executable), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(attempt / "pycache"),
               str(checkout / SCRIPT), "child", "--checkout", str(checkout), "--venv", str(venv),
               "--dependency", str(dependency), "--work", str(workdir), "--phase", args.phase,
               "--manifest-sha", args.manifest_sha, "--cache", str(cache), "--study-root", str(study_root),
               "--attempt", str(attempt),
               "--parent-platform", json.dumps(parent_platform, separators=(",", ":"))]
    if args.phase == "evaluation":
        command += ["--candidate", str(candidate), "--candidate-sha", args.candidate_sha]
    env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
           "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "PROCESSOR_IDENTIFIER") if name in os.environ}
    env.update({"PATH": str(Path(env.get("SYSTEMROOT", "C:/Windows")) / "System32"),
                "TEMP": str(attempt), "TMP": str(attempt), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    try:
        result = supervise_child(run_cpu_study, command, study_root, cwd=attempt, env=env, job=job)
        if (result.clean is not True or type(result.exit_code) is not int or result.exit_code != 0
                or result.timed_out is not False or type(result.pid) is not int or result.pid <= 0
                or type(result.run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", result.run_id) is None
                or job._handle is not None or not job._confirmed):
            raise StudyError("supervisor did not confirm a clean reaped owner and drained job")
        child_raw = read(attempt / "child-result.json")
        observed = json.loads(child_raw)
        if (type(observed["owner_pid"]) is not int or observed["owner_pid"] != result.pid
                or observed["owner_run_id"] != result.run_id
                or observed["manifest_sha256"] != args.manifest_sha or observed["phase"] != args.phase
                or (args.phase == "rehearsal" and observed.get("no_model_loaded") is not True)
                or (args.phase != "rehearsal" and observed["model_disposed"] is not True)):
            raise StudyError("child identity or disposal differs")
        check_imports(observed["imported_sources"], hashes)
        if (launch._revision(checkout) != head or launch._verify_sources(checkout, sources) != hashes
                or read(ordinary(checkout / MANIFEST)) != manifest_raw or any((attempt / "pycache").iterdir())):
            raise StudyError("source, manifest or bytecode changed during the run")
        finder.verify_inventory()
        artifacts = inventory_attempt(attempt, args.phase)
        validate_phase_output(attempt, args.phase, observed, artifacts)
        record = {"version": "targeted-edit-projected-phase-v1", "status": "complete",
                  "phase": args.phase, "source_commit": manifest["source_commit"], "checkout_commit": head,
                  "manifest_sha256": args.manifest_sha, "started_sha256": digest(started_raw),
                  "child_result_sha256": digest(child_raw),
                  "manifest_review_attestation_sha256": digest(manifest_review_raw),
                  "review_attestation_sha256": digest(review_raw) if args.phase == "evaluation" else None,
                  "supervisor": asdict(result), "owner_reaped": True, "job_drained": True,
                  "child": observed, "artifacts": artifacts, "finished_utc": utc(),
                  "outcome_scope": "process stage only; numerical study outcome remains provisional pending independent artifact audit",
                  "scope": "cooperative recorded checks; execution and ordering are not independently authenticated"}
        write(attempt / "phase-record.json", canonical(record))
        print(json.dumps({"status": "complete", "phase": args.phase, "attempt": str(attempt),
                          "record_sha256": digest(canonical(record))}, sort_keys=True))
        return 0
    except BaseException as error:
        # Never turn a partial or lost-return attempt into a success. Existing
        # supervisor state and the entire attempt directory remain in place.
        record_failure(attempt, args.phase, error)
        raise


def entry(argv=None):
    try:
        return main(argv)
    except BaseException as error:
        if _ACTIVE_ATTEMPT is not None:
            if _ATTEMPT_OWNED:
                record_failure(_ACTIVE_ATTEMPT, _ACTIVE_PHASE, error)
            print(json.dumps({"status": "incomplete", "attempt": str(_ACTIVE_ATTEMPT),
                              "attempt_owned": _ATTEMPT_OWNED, "error_type": type(error).__name__}), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(entry())
