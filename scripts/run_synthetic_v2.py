"""Guarded Windows CPU owner for synthetic v2; study launch is review-gated.

Run with the direct base interpreter using -I -S -B and an explicit venv and
dependency directory. Rehearsal initializes the fixed CPU models without a
forward or case generation. Source and cache leases, a child job, and the
source-only bootstrap are cooperative controls, not same-account attestation.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import types


_STUDY_LAUNCH_ENABLED = False  # Enable only after independent launcher review.
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
        if quarantine.child.wait(timeout=timeout):
            raise launcher.LaunchError("isolated owner failed; retained evidence may be incomplete")
        if launcher._revision(checkout) != revision or launcher._verify_sources(checkout, sources) != before:
            raise launcher.LaunchError("source verification failed after the owner exited")
        quarantine.retry_cleanup()
        return {"status": "rehearsal" if rehearsal else "succeeded", "source_revision": revision,
                "root_forwards": 0 if rehearsal else 57408, "child_reaped": True,
                "study_ready": False if rehearsal else True, "evidence_root": str(output)}
    except BaseException as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        try:
            quarantine.retry_cleanup()
        except BaseException as cleanup:
            if isinstance(cleanup, KeyboardInterrupt):
                status = "cancelled"
            raise launcher.LaunchError("owner cleanup requires retry", quarantine.child_reaped,
                status=status, quarantine=quarantine) from cleanup
        raise launcher.LaunchError("guarded owner failed; do not resume this attempt",
                                   quarantine.child_reaped, status=status) from error


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
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else getattr(error, "status", "failed")
        quarantine = getattr(error, "quarantine", None)
        protection_state = "released"
        if quarantine is not None:
            try:
                quarantine.retry_cleanup()
            except BaseException as retry:
                if isinstance(retry, KeyboardInterrupt):
                    status = "cancelled"
                protection_state = "unknown_on_process_exit"
        print(json.dumps({"status": status, "protection_state": protection_state,
                          "error_type": type(error).__name__,
                          "child_reaped": (quarantine.child_reaped if quarantine is not None
                                           else getattr(error, "child_reaped", None))}), file=sys.stderr)
        raise SystemExit(130 if status == "cancelled" else 1)
