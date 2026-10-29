"""One bounded CPU phase of the preregistered FLAN weight-edit study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

import flan_weight_edit_owner as owner


def _assets(path):
    manifest_path = path / "asset-manifest.json"
    if owner.digest(manifest_path) != owner.ASSET_SHA:
        raise ValueError("asset manifest differs")
    manifest = owner.read_json(manifest_path)
    expected = {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
                "spiece.model", "tokenizer.json", "tokenizer_config.json"}
    if set(manifest["files"]) != expected or manifest["revision"] != "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab":
        raise ValueError("asset inventory differs")
    for name in sorted(expected):
        file = path / name
        if file.is_symlink() or not file.is_file() or file.stat().st_size != manifest["files"][name]["bytes"] or owner.digest(file) != manifest["files"][name]["sha256"]:
            raise ValueError("asset bytes differ: " + name)
    return {name: manifest["files"][name]["sha256"] for name in sorted(expected)}


def _git_digest(repo, revision, path, aggregate=None):
    if not owner._hex(revision, 40) or path.startswith("-"):
        raise ValueError("invalid published artifact identity")
    process = subprocess.Popen(["git", "--no-replace-objects", "show", revision + ":" + path],
                               cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timeout = threading.Timer(120, process.kill)
    timeout.daemon = True
    timeout.start()
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            block = process.stdout.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
            if aggregate is not None:
                aggregate.update(block)
        error = process.stderr.read()
        if process.wait(timeout=15) != 0:
            raise ValueError("published artifact is unavailable: " + error.decode("utf8", "replace")[:300])
    finally:
        timeout.cancel()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)
        process.stdout.close()
        process.stderr.close()
    return digest.hexdigest(), size


def _published_artifact(repo, revision, item):
    parts = item.get("parts")
    if parts is None:
        sha, size = _git_digest(repo, revision, item["git_path"])
        return sha == item["sha256"] and size == Path(item["local_path"]).stat().st_size
    if not isinstance(parts, list) or not 1 <= len(parts) <= 128:
        raise ValueError("published artifact part inventory differs")
    aggregate, total, names = hashlib.sha256(), 0, set()
    for index, part in enumerate(parts):
        if (type(part.get("index")) is not int or part["index"] != index
                or type(part.get("git_path")) is not str or not part["git_path"]
                or type(part.get("bytes")) is not int
                or not 0 < part["bytes"] <= 1024 * 1024 or part.get("git_path") in names
                or not owner._hex(part.get("sha256"), 64)):
            raise ValueError("published artifact part identity differs")
        names.add(part["git_path"])
        digest, size = _git_digest(repo, revision, part["git_path"], aggregate)
        if digest != part["sha256"] or size != part["bytes"]:
            raise ValueError("published artifact part bytes differ")
        total += size
    return aggregate.hexdigest() == item["sha256"] and total == Path(item["local_path"]).stat().st_size


def _check_exit(phase, exit_record, phase_sha):
    required = {"status", "exit_code", "reaped", "drained", "timed_out", "error", "identity_matches",
                "phase_sha256", "direct_child_pid", "operator_pid", "phase_consistent", "within_evidence_cap",
                "cumulative_evidence_bytes_before_exit_record", "logs"}
    if (not required.issubset(exit_record) or exit_record.get("status") != "complete_candidate" or type(exit_record.get("exit_code")) is not int
            or exit_record["exit_code"] != 0 or exit_record.get("reaped") is not True
            or exit_record.get("drained") is not True or exit_record.get("timed_out") is not False
            or exit_record.get("error") is not None or exit_record.get("identity_matches") is not True
            or exit_record.get("phase_consistent") is not True or exit_record.get("within_evidence_cap") is not True
            or type(exit_record.get("cumulative_evidence_bytes_before_exit_record")) is not int
            or not 0 <= exit_record["cumulative_evidence_bytes_before_exit_record"] <= 255 * 1024**2
            or exit_record.get("phase_sha256") != phase_sha
            or type(phase.get("pid")) is not int or type(phase.get("parent_pid")) is not int
            or type(exit_record.get("direct_child_pid")) is not int or type(exit_record.get("operator_pid")) is not int
            or phase["pid"] != exit_record.get("direct_child_pid")
            or phase["parent_pid"] != exit_record.get("operator_pid")):
        raise ValueError("development process observation differs")
    logs = exit_record["logs"]
    if not isinstance(logs, dict) or set(logs) != {"stdout", "stderr"}:
        raise ValueError("development child logs differ")
    for log in logs.values():
        if (not isinstance(log, dict) or "error" not in log or log.get("error") is not None
                or log.get("finished") is not True or log.get("truncated") is not False
                or type(log.get("bytes")) is not int or not 0 <= log["bytes"] <= owner.LOG_LIMIT
                or not owner._hex(log.get("sha256"), 64)):
            raise ValueError("development child logs differ")


def _check_stdout_identity(path, phase):
    import flan_weight_edit_handoff as replay
    with Path(path).open("rb") as stream:
        line = stream.readline(4097)
    if len(line) > 4096 or not line.endswith(b"\n"):
        raise ValueError("development stdout identity is missing")
    identity = replay.strict_json(line)
    if (type(identity.get("pid")) is not int or type(identity.get("parent_pid")) is not int
            or identity["pid"] != phase["pid"] or identity["parent_pid"] != phase["parent_pid"]):
        raise ValueError("development stdout identity differs")


def _check_review(review, source_revision, artifacts):
    required_scope = {"saved-logits", "saved-process", "restoration"}
    scope = review.get("review_scope")
    if (review.get("status") != "passed" or review.get("source_revision") != source_revision
            or not isinstance(scope, list) or not all(type(x) is str for x in scope)
            or not required_scope.issubset(scope)):
        raise ValueError("development independent review scope differs")
    for name in ("result", "events", "logits", "phase", "exit", "stdout", "stderr", "scope", "resources"):
        if review.get(name + "_sha256") != artifacts[name]["sha256"]:
            raise ValueError("development independent review artifact differs: " + name)


class ResourceMonitor:
    def __init__(self, path, journal, args):
        self.path, self.journal, self.args = Path(path), journal, args
        self.stream = self.path.open("xb")
        self.checks = self.checked_calls = 0
        self.min_source = self.min_output = None

    def check(self, boundary):
        if boundary not in ("before", "after"):
            raise ValueError("unknown resource boundary")
        def record(disk, passed):
            row = {"index": self.checks, "call": self.journal.calls, "boundary": boundary,
                   "source_free": disk["source_free"], "output_free": disk["output_free"], "passed": passed}
            raw = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            within = self.journal._within_budget(len(raw))
            if not within and not passed:
                external = self.journal.external_bytes()
                within = type(external) is int and external >= 0 and self.journal.used + len(raw) + external <= self.journal.max_bytes
            if not within or self.stream.write(raw) != len(raw):
                raise ValueError("resource observation exceeds evidence budget")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.checks += 1
            self.checked_calls += int(boundary == "before")
            self.min_source = disk["source_free"] if self.min_source is None else min(self.min_source, disk["source_free"])
            self.min_output = disk["output_free"] if self.min_output is None else min(self.min_output, disk["output_free"])
        owner.volumes(self.args.attempt, self.args.source_dir, observer=record)

    def summary(self):
        return {"checks": self.checks, "checked_calls": self.checked_calls,
                "min_source_free": self.min_source, "min_output_free": self.min_output}

    def close(self):
        self.stream.close()


def verify_resources(path, calls):
    import flan_weight_edit_handoff as replay
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("resource observation has a partial final line")
    rows = [replay.strict_json(line) for line in raw.splitlines()]
    if type(calls) is not int or calls < 0 or len(rows) != calls * 2:
        raise ValueError("resource observation count differs")
    for i, row in enumerate(rows):
        if (set(row) != {"index", "call", "boundary", "source_free", "output_free", "passed"}
                or type(row["index"]) is not int or row["index"] != i
                or type(row["call"]) is not int or row["call"] != i // 2 + 1
                or row["boundary"] != ("before" if i % 2 == 0 else "after") or row["passed"] is not True
                or type(row["source_free"]) is not int or row["source_free"] < 1207959552
                or type(row["output_free"]) is not int or row["output_free"] < 2416967680):
            raise ValueError("resource observation differs")
    return {"checks": len(rows), "checked_calls": calls,
            "min_source_free": min((r["source_free"] for r in rows), default=None),
            "min_output_free": min((r["output_free"] for r in rows), default=None)}


def _handoff(args, manifest, cases, tokenizer, assets):
    import flan_weight_edit_handoff as replay
    import flan_weight_edit_worker as worker
    binding = owner.read_json(args.handoff)
    if binding.get("schema") != "flan-weight-edit-handoff-v1" or binding.get("source_revision") != manifest["source_revision"]:
        raise ValueError("development handoff identity differs")
    commit = binding.get("public_revision")
    artifacts = binding.get("artifacts", {})
    if set(artifacts) != {"result", "events", "logits", "phase", "exit", "review", "stdout", "stderr", "scope", "resources"}:
        raise ValueError("development handoff artifacts differ")
    for item in artifacts.values():
        if (not Path(item["local_path"]).is_absolute() or owner.digest(item["local_path"]) != item["sha256"]
                or not _published_artifact(args.repo, commit, item)):
            raise ValueError("published development artifact bytes differ")
    if hashlib.sha256(owner.git_blob(args.repo, commit, owner.MANIFEST_PATH)).hexdigest() != owner.digest(args.manifest):
        raise ValueError("published development manifest differs")
    phase = owner.read_json(artifacts["phase"]["local_path"])
    exit_record = owner.read_json(artifacts["exit"]["local_path"])
    review = owner.read_json(artifacts["review"]["local_path"])
    _check_exit(phase, exit_record, artifacts["phase"]["sha256"])
    for name in ("stdout", "stderr"):
        if (artifacts[name]["sha256"] != exit_record["logs"][name]["sha256"]
                or Path(artifacts[name]["local_path"]).stat().st_size != exit_record["logs"][name]["bytes"]):
            raise ValueError("published child log differs")
    _check_stdout_identity(artifacts["stdout"]["local_path"], phase)
    _check_review(review, manifest["source_revision"], artifacts)
    if (phase.get("phase") != "development" or phase.get("execution_status") != "complete"
            or phase.get("source_revision") != manifest["source_revision"]
            or phase.get("manifest_sha256") != owner.digest(args.manifest)
            or phase.get("case_sha256") != owner.CASE_SHA or phase.get("scope_sha256") != artifacts["scope"]["sha256"]
            or not replay.typed_equal(phase.get("runtime"), manifest["runtime"])
            or not replay.typed_equal(phase.get("assets"), assets)
            or type(phase.get("calls")) is not int or not 0 <= phase["calls"] < 1200
            or type(phase.get("cumulative_calls")) is not int or phase["cumulative_calls"] != phase["calls"]
            or phase.get("result_sha256") != artifacts["result"]["sha256"]
            or phase.get("events_sha256") != artifacts["events"]["sha256"]
            or phase.get("logits_sha256") != artifacts["logits"]["sha256"]
            or phase.get("resources_sha256") != artifacts["resources"]["sha256"]
            or not replay.typed_equal(phase.get("resource_summary"), verify_resources(artifacts["resources"]["local_path"], phase["calls"]))
            ):
        raise ValueError("development process or independent review differs")
    mode_before = worker.numeric_probe()
    checked = replay.verify_development(cases, tokenizer, artifacts["result"]["local_path"],
                                         artifacts["events"]["local_path"], artifacts["logits"]["local_path"])
    mode_after = worker.numeric_probe()
    if mode_before != mode_after:
        raise ValueError("numeric mode changed during development replay")
    if checked["result"].get("status") != "complete" or checked["calls"] != phase["calls"]:
        raise ValueError("development was incomplete or ineligible")
    span = phase["initial_probe"]
    with Path(artifacts["logits"]["local_path"]).open("rb") as stream:
        stream.seek(span["offset"])
        probe = stream.read(span["bytes"])
    if len(probe) != 32128 * 4 or hashlib.sha256(probe).hexdigest() != span["sha256"]:
        raise ValueError("development reference probe differs")
    return binding, phase, checked["result"], probe, {"before": mode_before, "after": mode_after}


def _write_result(path, value, journal):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
    if not journal._within_budget(len(raw)):
        raise ValueError("cumulative evidence cap would be exceeded")
    with Path(path).open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short result write")
        stream.flush()
        os.fsync(stream.fileno())


def run(args, manifest, site):
    if args.rehearsal:
        owner.write_new(args.attempt / "phase.json", {"phase": args.phase, "execution_status": "rehearsal",
                        "source_revision": manifest["source_revision"], "model_loaded": False,
                        "pid": os.getpid(), "parent_pid": os.getppid()})
        return
    import importlib.metadata
    import platform
    import numpy as np
    import tokenizers
    import torch
    import transformers
    from transformers import T5ForConditionalGeneration
    import flan_weight_edit_cases as case_builder
    import flan_weight_edit_guard as guard
    import flan_weight_edit_journal as evidence
    import flan_weight_edit_results as results
    import flan_weight_edit_session as session_module
    import flan_weight_edit_study as study
    import flan_weight_edit_worker as worker

    packages = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    if packages != manifest["runtime"]:
        raise ValueError("runtime versions differ")
    for module in (np, tokenizers, torch, transformers):
        if not Path(module.__file__).resolve().is_relative_to(site):
            raise ValueError("loaded library is outside the selected dependency site")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise ValueError("offline CPU environment differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20261029)
    if not torch.set_flush_denormal(False):
        raise ValueError("denormal preservation is unavailable")
    worker.numeric_probe()
    asset_snapshot = _assets(args.assets)
    tokenizer = case_builder.load_tokenizer(args.assets)
    fixed = owner.read_json(args.cases)
    try:
        aligned = replay_equal(case_builder.build_bundle(tokenizer), fixed)
    except ValueError:
        aligned = False
    if not aligned:
        owner.write_new(args.attempt / "phase.json", {"phase": args.phase, "execution_status": "complete",
                        "scientific_status": "ineligible", "reason": "tokenizer-alignment",
                        "model_loaded": False, "source_revision": manifest["source_revision"], "calls": 0,
                        "pid": os.getpid(), "parent_pid": os.getppid(), "case_sha256": owner.CASE_SHA,
                        "scope_sha256": owner.digest(args.scope), "manifest_sha256": owner.digest(args.manifest)})
        return
    cases = fixed["cases"]
    handoff = None
    if args.phase == "evaluation":
        handoff = _handoff(args, manifest, cases, tokenizer, asset_snapshot)
    previous_calls = 0 if handoff is None else handoff[1]["calls"]
    if type(previous_calls) is not int or not 0 <= previous_calls < 1200:
        raise ValueError("cumulative model-call count differs")
    excluded = {(args.attempt / "evidence" / name).resolve() for name in ("events.jsonl", "logits.f32")}
    journal = evidence.Journal(args.attempt / "evidence", vocabulary=32128, max_calls=1200 - previous_calls,
                               max_bytes=256 * 1024**2, reserve=1024**2,
                               external_bytes=lambda: owner.evidence_bytes(args, exclude=excluded))
    resources = None
    try:
        resources = ResourceMonitor(args.attempt / "resources.jsonl", journal, args)
        owner.preflight(args)
        owner.volumes(args.attempt, args.source_dir)
        model, info = T5ForConditionalGeneration.from_pretrained(str(args.assets), local_files_only=True,
            use_safetensors=True, trust_remote_code=False, output_loading_info=True, dtype=torch.float32,
            attn_implementation="eager")
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("checkpoint loading diagnostics differ")
        model.eval()
        model.config.use_cache = False
        config = model.config
        if (type(model).__name__ != "T5ForConditionalGeneration" or config.d_model != 512 or config.vocab_size != 32128
                or config.num_heads != 6 or len(model.encoder.block) != 8 or len(model.decoder.block) != 8
                or config.tie_word_embeddings is not False or config.pad_token_id != 0 or config.eos_token_id != 1
                or config.decoder_start_token_id != 0 or config._attn_implementation != "eager"):
            raise ValueError("checkpoint configuration differs")
        session = session_module.Session(model, tokenizer, journal, cases[0], resource_gate=resources.check)
        scope_sha = owner.digest(args.scope)
        def final_boundary():
            owner.preflight(args)
            owner.volumes(args.attempt, args.source_dir)
            if owner.digest(args.scope) != scope_sha:
                raise ValueError("scope changed before first model forward")
        prepared = session.prepare(expected_receipt=None if handoff is None else handoff[1]["candidate_sha256"],
                                   expected_probe=None if handoff is None else handoff[3],
                                   before_probe=final_boundary)
        if prepared.get("status") == "ineligible":
            result = prepared
            probe_span = None
        else:
            probe_span = journal.last_record.copy()
            result = study.development(cases, session) if handoff is None else study.evaluation(cases, session, handoff[2])
        if args.phase == "evaluation" and result.get("status") == "complete":
            result["locality_distributions"] = session.score("locality-summary", lambda:
                results.summarize(cases, result, args.attempt / "evidence" / "logits.f32"))
        if guard.state_hash(model) != session.base_state or guard.topology(model) != session.base_topology:
            raise ValueError("final model restoration differs")
        if _assets(args.assets) != asset_snapshot:
            raise ValueError("assets changed during the phase")
        if journal.poisoned:
            raise ValueError("journal is fenced")
        if not replay_equal(resources.summary(), verify_resources(resources.path, journal.calls)):
            raise ValueError("resource observation summary differs")
        _write_result(args.attempt / "result.json", result, journal)
        events_sha = owner.digest(args.attempt / "evidence" / "events.jsonl")
        logits_sha = owner.digest(args.attempt / "evidence" / "logits.f32")
        scientific = result.get("primary", {}).get("status") if args.phase == "evaluation" else result.get("status")
        phase = {"phase": args.phase, "execution_status": "complete", "scientific_status": scientific,
                 "source_revision": manifest["source_revision"], "manifest_sha256": owner.digest(args.manifest),
                 "scope_sha256": owner.digest(args.scope),
                 "case_sha256": owner.CASE_SHA, "runtime": packages, "assets": asset_snapshot,
                 "result_sha256": owner.digest(args.attempt / "result.json"), "events_sha256": events_sha,
                 "logits_sha256": logits_sha, "candidate_sha256": prepared.get("candidate_sha256"),
                 "initial_probe": probe_span, "calls": journal.calls, "cumulative_calls": previous_calls + journal.calls,
                 "resources_sha256": owner.digest(resources.path), "resource_summary": resources.summary(),
                 "development_replay_numeric_mode": None if handoff is None else handoff[4],
                 "final_state_sha256": session.base_state, "pid": os.getpid(), "parent_pid": os.getppid(),
                 "scope": "cooperative saved process and model-state observations"}
        _write_result(args.attempt / "phase.json", phase, journal)
    except BaseException:
        journal.fence()
        raise
    finally:
        if resources is not None:
            resources.close()
        journal.close()


def replay_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
