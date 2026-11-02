"""Launch one fixed-source CPU development child and retain its observed exit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .agreement_head_owner_v1 import _source_binding, _write_new


def launch(*, root, revision, cache_dir, run_dir):
    root, cache_dir, run_dir = Path(root).resolve(strict=True), Path(cache_dir).resolve(strict=True), Path(run_dir)
    if (not run_dir.is_absolute() or not run_dir.parent.is_dir() or os.path.lexists(run_dir)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("OMP_NUM_THREADS") != "1"
            or not sys.dont_write_bytecode or not sys.pycache_prefix or os.path.lexists(sys.pycache_prefix)):
        raise ValueError("requires a new output and explicit CPU/offline source-first execution")
    if root.parent.name.lower() != "teams" or root.parent.parent.name.lower() != "work":
        raise ValueError("checkout is not in the expected shared workspace")
    work = root.parent.parent.resolve(strict=True)
    if run_dir.parent.resolve(strict=True) != work:
        raise ValueError("run directory must be a direct child of the shared work directory")
    if shutil.disk_usage(work).free < 2 * 1024 ** 3:
        raise ValueError("less than 2 GiB free before the one-shot claim")
    binding = _source_binding(root, revision)
    claim = work / "agreement-head-v1-development.claim.json"
    if os.path.lexists(claim):
        raise FileExistsError("the development one-shot claim already exists")
    run_dir.mkdir()
    attempt = run_dir / "attempt"
    child_prefix = run_dir / "child-pycache"
    command = [sys.executable, "-B", "-X", f"pycache_prefix={child_prefix}", "-m",
               "looking_glass.agreement_head_owner_v1", "--root", str(root), "--revision", revision,
               "--cache-dir", str(cache_dir), "--attempt", str(attempt)]
    claim_sha = _write_new(claim, {"schema": "agreement-head-development-claim-v1", "source_commit": revision,
                                   "run_dir": str(run_dir), "source_sha256": binding})
    start = {"schema": "agreement-head-development-supervisor-v1", "source_commit": revision,
             "claim_sha256": claim_sha, "command": command, "started_unix_ns": time.time_ns()}
    _write_new(run_dir / "supervisor-start.json", start)
    child = None
    code = None
    spawn_attempted = False
    failure = None
    try:
        with (run_dir / "stdout.txt").open("xb") as stdout, (run_dir / "stderr.txt").open("xb") as stderr:
            spawn_attempted = True
            child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
            code = child.wait()
    except BaseException as original:
        failure = original
        if child is not None:
            try:
                child.terminate()
            except BaseException as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note(f"child terminate failed: {type(cleanup).__name__}")
            try:
                code = child.wait(timeout=5)
            except BaseException as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note(f"child wait failed: {type(cleanup).__name__}")
            if code is None:
                try:
                    child.kill()
                except BaseException as cleanup:
                    if hasattr(original, "add_note"):
                        original.add_note(f"child kill failed: {type(cleanup).__name__}")
                try:
                    code = child.wait(timeout=5)
                except BaseException as cleanup:
                    if hasattr(original, "add_note"):
                        original.add_note(f"final child wait failed: {type(cleanup).__name__}")
        raise
    finally:
        output = {"schema": "agreement-head-development-exit-v1", "source_commit": revision,
                  "claim_sha256": claim_sha,
                  "child_start": "observed" if child is not None else ("unknown" if spawn_attempted else "not_attempted"),
                  "exit_code": code if type(code) is int else None,
                  "launcher_error_type": None if failure is None else type(failure).__name__,
                  "ended_unix_ns": time.time_ns()}
        try:
            _write_new(run_dir / "supervisor-exit.json", output)
        except BaseException as cleanup:
            if failure is None:
                raise
            if hasattr(failure, "add_note"):
                failure.add_note(f"exit receipt failed: {type(cleanup).__name__}")
    if type(code) is not int or code != 0:
        raise RuntimeError("development child did not exit zero; preserve the one-shot attempt")
    candidate = attempt / "owner-candidate.json"
    if not candidate.is_file():
        raise RuntimeError("child exited without a candidate record")
    return {"exit_code": code, "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(launch(root=args.root, revision=args.revision, cache_dir=args.cache_dir,
                            run_dir=args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
