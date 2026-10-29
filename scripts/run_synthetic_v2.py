"""Guarded Windows CPU owner for one explicitly requested synthetic-v2 attempt.

Run with the direct base interpreter using -I -S -B and an explicit venv and
dependency directory. Rehearsal initializes the fixed CPU models without a
forward or case generation. Source and cache leases, a child job, and the
source-only bootstrap are cooperative controls, not same-account attestation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import types


_STUDY_LAUNCH_ENABLED = True
_EXTRA = (
    "scripts/run_synthetic_v2.py",
    "docs/CAUSAL_CONFIRMATION_V2.md",
    "docs/CAUSAL_CONFIRMATION_V2_NUMERIC_CLARIFICATION.md",
    "docs/CAUSAL_CONFIRMATION_V2_SOURCE_CLARIFICATION.md",
    "docs/CAUSAL_CONFIRMATION_V2_SELF_PATCH_CLARIFICATION.md",
    "docs/CAUSAL_CONFIRMATION_V2_RESOURCE_CLARIFICATION.md",
    "results/synthetic-v2-exclusions-20261028/inputs.json",
    "results/synthetic-v2-resource-pilot-20261028/artifacts/pilot_inputs.json",
)
_SOURCE_STAGES = ("before_discovery", "after_discovery", "after_heldout", "before_success")
_PUBLIC_ARTIFACTS = frozenset({"pre_manifest.json", "prediction_freeze.json",
    "prediction_commitment.json", "results.json", "reveal.json",
    *(f"source_inventory_{stage}.json" for stage in _SOURCE_STAGES)})
_PRIVATE_ARTIFACTS = frozenset({"private_material.json", "prediction_record.json", "exposure_record.json",
    *(f"cell-{index:02d}-{phase}.json" for index in range(54) for phase in ("d", "h"))})


def _launcher():
    """Compile the sibling source, without a bytecode-cache lookup.

    The owner's subsequent commit/hash check concerns its on-disk bytes; it
    cannot cryptographically attest the already-executing parent process.
    """
    path = Path(__file__).resolve().with_name("v2_source_launcher.py")
    with path.open("rb") as stream:
        data = stream.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise RuntimeError("source launcher exceeds its size limit")
    module = types.ModuleType("_v2_guarded_launcher")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _registry(checkout, launcher, source_sha256):
    """Load the commit-checked registry source while its lease is still held."""
    relative = "looking_glass/registry.py"
    path = launcher._checked_path(checkout / relative)
    if not path.is_relative_to(checkout):
        raise launcher.LaunchError("registry source is outside checkout")
    with path.open("rb") as stream:
        data = stream.read(8 * 1024 * 1024 + 1)
    if (len(data) > 8 * 1024 * 1024 or
            hashlib.sha256(data).hexdigest() != source_sha256.get(relative)):
        raise launcher.LaunchError("registry source differs from the guarded inventory")
    module = types.ModuleType("_v2_guarded_registry")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _verified_success(registry, output, revision, source_sha256):
    """Read and verify both saved terminal inventories and observed counts."""
    public = registry.verify_run(output / "public")
    private = registry.verify_run(output / "private")
    if public["state"] != "succeeded" or private["state"] != "succeeded":
        raise ValueError("study terminal records do not both report success")
    for inspected, required in ((public, _PUBLIC_ARTIFACTS), (private, _PRIVATE_ARTIFACTS)):
        if (set(inspected["outcome"]["artifacts"]) != required
                or set(inspected["manifest"]["required_artifacts"]) != required):
            raise ValueError("study artifact inventory is incomplete or unexpected")
    public_ids, private_ids = public["manifest"]["identities"], private["manifest"]["identities"]
    if any(registry._json_bytes(public_ids[name]) != registry._json_bytes(private_ids[name])
           for name in ("inputs", "model", "code")):
        raise ValueError("public and private registry identities differ")
    private_protocol = dict(private_ids["protocol"]["value"])
    if ("v2_evidence" not in private_protocol
            or registry._json_bytes({key: value for key, value in private_protocol.items()
                                     if key != "v2_evidence"})
               != registry._json_bytes(public_ids["protocol"]["value"])):
        raise ValueError("public and private protocol identities differ")
    pre, pre_raw = registry._read_json(output / "public" / "artifacts" / "pre_manifest.json")
    if (type(pre) is not dict or type(pre.get("context")) is not dict
            or pre.get("schema") != "synthetic-v2-pre-manifest-v1" or pre.get("mode") != "study"
            or set(pre["context"]) != {"protocol_id", "protocol_sha256", "source_revision", "model_state_hashes"}
            or pre["context"].get("protocol_id") != "synthetic-causal-v2"
            or pre["context"].get("protocol_sha256") != source_sha256.get("docs/CAUSAL_CONFIRMATION_V2.md")
            or pre["context"].get("source_revision") != revision
            or registry._json_bytes(pre.get("source_sha256")) != registry._json_bytes(source_sha256)):
        raise ValueError("saved premanifest source identity differs")
    states = pre["context"]["model_state_hashes"]
    if (type(states) is not list or len(states) != 6
            or any(type(state) is not dict or set(state) != {"seed", "sha256"}
                   or type(state["seed"]) is not int or state["seed"] != seed
                   or type(state["sha256"]) is not str
                   or re.fullmatch(r"[0-9a-f]{64}", state["sha256"]) is None
                   for state, seed in zip(states, range(31201, 31207)))):
        raise ValueError("saved model-state inventory differs")
    pre_sha = hashlib.sha256(pre_raw).hexdigest()
    if (registry._json_bytes(public_ids["protocol"]["value"]) != registry._json_bytes(
                {"id": "synthetic-causal-v2", "pre_manifest_sha256": pre_sha})
            or registry._json_bytes(public_ids["model"]["value"]) != registry._json_bytes(
                {"states": pre["context"].get("model_state_hashes")})
            or registry._json_bytes(public_ids["code"]["value"]) != registry._json_bytes(
                {"source_sha256": source_sha256})):
        raise ValueError("registry and premanifest identities differ")
    previous = {}
    for stage in _SOURCE_STAGES:
        snapshot, _ = registry._read_json(output / "public" / "artifacts" / f"source_inventory_{stage}.json")
        if (type(snapshot) is not dict
                or set(snapshot) != {"schema", "stage", "pre_manifest_sha256", "source_sha256"}
                or snapshot["schema"] != "synthetic-v2-source-inventory-v1"
                or snapshot["stage"] != stage or snapshot["pre_manifest_sha256"] != pre_sha
                or type(snapshot["source_sha256"]) is not dict or not snapshot["source_sha256"]
                or len(snapshot["source_sha256"]) > 4096
                or "looking_glass/causal_v2_study_entry.py" not in snapshot["source_sha256"]
                or any(type(path) is not str or not path.startswith("looking_glass/")
                       or not path.endswith(".py") or type(digest) is not str or source_sha256.get(path) != digest
                       for path, digest in snapshot["source_sha256"].items())
                or any(snapshot["source_sha256"].get(path) != digest for path, digest in previous.items())):
            raise ValueError("saved phase source inventory differs")
        previous = snapshot["source_sha256"]
    private_outcome, private_raw = registry._read_json(output / "private" / "outcome.json")
    if registry._json_bytes(private_outcome) != registry._json_bytes(private["outcome"]):
        raise ValueError("private outcome changed during verification")
    public_details, private_details = public["outcome"]["details"], private_outcome["details"]
    if (type(public_details) is not dict or type(private_details) is not dict
            or set(public_details) != {"scope", "submissions", "root_forwards", "private_outcome_sha256"}
            or set(private_details) != {"scope", "submissions", "root_forwards"}
            or public_details["scope"] != "complete study" or private_details["scope"] != "private evidence"
            or public_details["private_outcome_sha256"] != hashlib.sha256(private_raw).hexdigest()
            or any(type(details[key]) is not int or details[key] != 57408
                   for details in (public_details, private_details)
                   for key in ("submissions", "root_forwards"))):
        raise ValueError("saved terminal counts or private outcome binding differ")
    return public_details["submissions"], public_details["root_forwards"]


def run(checkout, venv, dependencies, work, output, *, rehearsal, timeout):
    launcher = _launcher()
    if type(rehearsal) is not bool or type(timeout) is not int or not 1 <= timeout <= 14400:
        raise launcher.LaunchError("invalid owner mode or timeout")
    if not rehearsal and not _STUDY_LAUNCH_ENABLED:
        raise launcher.LaunchError("study execution is disabled pending independent review")
    executable = launcher._check_startup(checkout, venv, dependencies, work)
    if launcher._checked_path(__file__) != checkout / _EXTRA[0]:
        raise launcher.LaunchError("study owner is not in the target checkout")
    output = Path(output)
    if not output.is_absolute() or os.path.lexists(output):
        raise launcher.LaunchError("a fresh absolute output path is required")
    output = launcher._checked_path(output.parent) / output.name
    if output.is_relative_to(checkout) or checkout.is_relative_to(output):
        raise launcher.LaunchError("study output must be outside checkout")
    parent_platform = launcher._parent_platform_observation()
    revision = launcher._revision(checkout)
    sources = launcher._committed_sources(checkout, revision, extra=_EXTRA)
    before = launcher._verify_sources(checkout, sources)
    guards = launcher._load_guards(checkout, sources)
    temporary = Path(tempfile.mkdtemp(prefix="v2-owner-", dir=work))
    manifest = temporary / "sources.json"
    manifest_bytes = json.dumps(before, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode("utf-8")
    quarantine = launcher.LaunchQuarantine(temporary, manifest)
    recorded_success = False
    try:
        manifest.write_bytes(manifest_bytes)
        paths = [checkout.joinpath(*PurePosixPath(name).parts) for name in sources]
        quarantine.source_lease = guards.SourceFileLease(checkout, paths)
        quarantine.stack.enter_context(quarantine.source_lease)
        quarantine.manifest_lease = guards.SourceFileLease(temporary, [manifest])
        quarantine.stack.enter_context(quarantine.manifest_lease)
        quarantine.cache = guards.PrivateEmptyCache(temporary)
        prefix = quarantine.stack.enter_context(quarantine.cache)
        if (launcher._revision(checkout) != revision or launcher._verify_sources(checkout, sources) != before
                or manifest.read_bytes() != manifest_bytes or any(prefix.iterdir())):
            raise launcher.LaunchError("guarded source or cache validation failed")
        quarantine.job = guards.WindowsJob()
        handles = (quarantine.source_lease.handles() + quarantine.manifest_lease.handles()
                   + quarantine.cache.handles())
        launcher._check_single_threaded()
        arguments = [str(executable), "-I", "-S", "-B", "-X", f"pycache_prefix={prefix}",
                     str(checkout / launcher._SCRIPTS[1]),
                     "--study-rehearsal" if rehearsal else "--study-owner",
                     "--checkout", str(checkout), "--sources", str(manifest),
                     "--venv-root", str(venv), "--prefix", str(prefix),
                     "--source-revision", revision, "--output", str(output), "--require-start"]
        for dependency in dependencies:
            arguments.extend(("--dependency", str(dependency)))
        if len(subprocess.list2cmdline(arguments)) > 30000:
            raise launcher.LaunchError("guarded child command exceeds the Windows bound")
        quarantine.child = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True, creationflags=subprocess.DETACHED_PROCESS, cwd=work)
        quarantine.job.assign(quarantine.child)
        remote_handles = guards.duplicate_to_process(handles, quarantine.child)
        frame = launcher._TOKEN + json.dumps({"guard_handles": remote_handles,
            "platform_uname": parent_platform}, separators=(",", ":")).encode("ascii")
        if len(frame) > 262144:
            raise launcher.LaunchError("guarded start frame exceeds its bound")
        quarantine.child.stdin.write(frame)
        quarantine.child.stdin.close()
        quarantine.child.stdin = None
        pending = None
        try:
            if quarantine.child.wait(timeout=timeout):
                pending = launcher.LaunchError("isolated owner failed; retained evidence may be incomplete")
        except BaseException as error:
            pending = error
        # A direct child's exit is not proof that its descendants have exited.
        # Drain the whole job while source/cache protections remain in place,
        # before treating saved outcomes as quiescent and reading their counts.
        try:
            quarantine.child_reaped = launcher._stop_and_reap(quarantine.child, quarantine.job)
        except BaseException as drain_error:
            if pending is not None:
                if isinstance(drain_error, (KeyboardInterrupt, SystemExit)):
                    raise drain_error from pending
                raise pending from drain_error
            raise
        quarantine.child = quarantine.job = None
        try:
            if launcher._revision(checkout) != revision or launcher._verify_sources(checkout, sources) != before:
                raise launcher.LaunchError("source verification failed after the owner exited")
            if rehearsal:
                submissions, forwards = 0, 0
            elif pending is None or output.exists():
                registry = _registry(checkout, launcher, before)
                submissions, forwards = _verified_success(registry, output, revision, before)
                recorded_success = True
            if pending is not None:
                raise pending
        except BaseException as verify_error:
            if pending is not None and verify_error is not pending:
                if isinstance(verify_error, (KeyboardInterrupt, SystemExit)):
                    raise verify_error from pending
                raise pending from verify_error
            raise
        quarantine.retry_cleanup()
        report = {"status": "rehearsal" if rehearsal else "terminal_verified", "source_revision": revision,
                  "submissions": submissions, "root_forwards": forwards, "child_reaped": True,
                  "evidence_root": str(output), "verification_scope":
                  "no_forward_rehearsal" if rehearsal else "terminals_counts_and_inventory"}
        report["study_ready" if rehearsal else "scientific_replay_verified"] = False
        return report
    except BaseException as error:
        status = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        try:
            quarantine.retry_cleanup()
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
                status = "cancelled"
            failure = launcher.LaunchError("owner cleanup requires retry", quarantine.child_reaped,
                status=status, quarantine=quarantine)
            failure.recorded_success = True if recorded_success else None
            raise failure from cleanup
        failure = launcher.LaunchError("guarded owner failed; do not resume this attempt",
                                       quarantine.child_reaped, status=status)
        failure.recorded_success = True if recorded_success else None
        raise failure from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rehearsal", action="store_true")
    mode.add_argument("--study", action="store_true")
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--venv-root", required=True)
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    launcher = _launcher()
    result = run(launcher._checked_path(args.checkout), launcher._checked_path(args.venv_root),
        tuple(launcher._checked_path(item) for item in args.dependency),
        launcher._checked_path(args.work_dir), args.output,
        rehearsal=args.rehearsal, timeout=args.timeout)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else getattr(error, "status", "failed")
        quarantine = getattr(error, "quarantine", None)
        protection_state = "released"
        if quarantine is not None:
            try:
                quarantine.retry_cleanup()
            except BaseException as retry:
                if isinstance(retry, (KeyboardInterrupt, SystemExit)):
                    status = "cancelled"
                protection_state = "unknown_on_process_exit"
        print(json.dumps({"status": status, "protection_state": protection_state,
                          "recorded_success": getattr(error, "recorded_success", None),
                          "error_type": type(error).__name__,
                          "child_reaped": (quarantine.child_reaped if quarantine is not None
                                           else getattr(error, "child_reaped", None))}), file=sys.stderr)
        raise SystemExit(130 if status == "cancelled" else 1)
