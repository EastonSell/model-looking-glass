"""Cooperatively observe one explicitly bootstrapped CPU study process.

This does not authenticate the interpreter, the clock, or other processes.
The source checkout must remain exclusively assigned and unchanged through exit.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
CHECKOUT = ROOT
MANIFEST = CHECKOUT / "experiments/distilgpt2_copy_output_dose_manifest_v1.json"
BOOTSTRAP = CHECKOUT / "scripts/copying_head_output_dose_bootstrap_v1.py"
READER = CHECKOUT / "audits/copy_output_dose/saved_independent_audit.py"
TIMEOUT_SECONDS = 3600


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*arguments):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.check_output(["git", *arguments], cwd=CHECKOUT, env=env,
                                   timeout=60).decode("utf-8").strip()


def durable_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short observation write")
        stream.flush()
        os.fsync(stream.fileno())


def inventory(path):
    if not path.is_dir():
        return []
    return [{"path": p.relative_to(path).as_posix(), "bytes": p.stat().st_size,
             "sha256": sha(p)} for p in sorted(path.rglob("*")) if p.is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-prelaunch-commit", required=True)
    parser.add_argument("--checkout-head", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--reader-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    CACHE, WORK = args.cache_dir, args.work_dir
    if not CACHE.is_absolute() or not WORK.is_absolute() or WORK.drive.upper() != "E:":
        raise ValueError("absolute cache and E: work paths are required")
    if len(args.public_prelaunch_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.public_prelaunch_commit
    ):
        raise ValueError("a public prelaunch commit SHA is required")
    if git("rev-parse", "HEAD") != args.checkout_head or git("status", "--porcelain"):
        raise RuntimeError("launch source checkout is not the exact clean reviewed commit")
    if sha(MANIFEST) != args.manifest_sha256 or sha(READER) != args.reader_sha256:
        raise RuntimeError("manifest or independent saved-data reader changed")
    if not CACHE.is_dir():
        raise RuntimeError("the fixed offline cache directory is missing")

    WORK.mkdir(parents=True, exist_ok=True)
    run_id = ("verify-" if args.verify_only else "attempt-") + uuid.uuid4().hex
    output = WORK / run_id
    prefix = WORK / (run_id + "-pycache")
    stdout_path = WORK / (run_id + "-stdout.log")
    stderr_path = WORK / (run_id + "-stderr.log")
    start_path = WORK / (run_id + "-start.json")
    finish_path = WORK / (run_id + "-exit.json")
    if output.exists():
        raise RuntimeError("output already exists")
    prefix.mkdir()
    command = [sys.executable, "-B", "-X", "pycache_prefix=" + str(prefix),
               str(BOOTSTRAP), "--cache-dir", str(CACHE), "--output-dir", str(output),
               "--manifest", str(MANIFEST)]
    if args.verify_only:
        command.append("--verify-only")
    env = dict(os.environ)
    for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    bound_env = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    env.update(bound_env)
    started = now()
    start = {"schema": "copy-output-dose-process-start-v1", "run_id": run_id,
             "verify_only": args.verify_only, "command": command, "cwd": str(CHECKOUT),
             "environment_overrides": bound_env, "removed_environment_names": [
                 "MKL_CBWR", "ATEN_CPU_CAPABILITY", "PYTHONPATH", "PYTHONHOME"],
             "utc_clock_started": started, "timeout_seconds": TIMEOUT_SECONDS,
             "checkout_head": args.checkout_head, "manifest_sha256": args.manifest_sha256,
             "independent_reader_sha256": args.reader_sha256,
             "observer_sha256": sha(Path(__file__)),
             "public_prelaunch_commit": args.public_prelaunch_commit,
             "scope": "Cooperative direct process observation; no interpreter, clock or outside-process attestation."}
    durable_json(start_path, start)
    print(json.dumps({"run_id": run_id, "attempt": str(output), "start_record": str(start_path)}), flush=True)
    child = None
    spawn_call_entered = False
    code = None
    timed_out = False
    error = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            spawn_call_entered = True
            child = subprocess.Popen(command, cwd=CHECKOUT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                code = child.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                child.kill()
                code = child.wait(timeout=15)
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    except BaseException as exc:
        error = type(exc).__name__
        if child is not None and child.poll() is None:
            try:
                child.kill()
                code = child.wait(timeout=15)
            except BaseException:
                pass
        raise
    finally:
        try:
            checkout_clean = git("rev-parse", "HEAD") == args.checkout_head and not git("status", "--porcelain")
        except BaseException:
            checkout_clean = None
        record = {"schema": "copy-output-dose-process-exit-v1", "run_id": run_id,
                  "start_record_sha256": sha(start_path), "utc_clock_finished": now(),
                  "child_pid": None if child is None else child.pid, "exit_code": code,
                  "reaped": code is not None, "timed_out": timed_out,
                  "spawn_return_unbound_uncertain": spawn_call_entered and child is None,
                  "cleanup_scope": "Returned direct child only; descendants are not attested.",
                  "observer_exception": error, "checkout_clean_after": checkout_clean,
                  "stdout_sha256": sha(stdout_path) if stdout_path.is_file() else None,
                  "stderr_sha256": sha(stderr_path) if stderr_path.is_file() else None,
                  "saved_attempt_files": inventory(output),
                  "scope": start["scope"]}
        durable_json(finish_path, record)
    if code != 0 or timed_out or checkout_clean is not True:
        raise RuntimeError("incomplete or uncertain process; retain all saved evidence")
    if not args.verify_only and not (output / "result.json").is_file():
        raise RuntimeError("no complete result was written; retain the attempt")
    print(json.dumps({"exit_code": code, "attempt": str(output),
                      "exit_record": str(finish_path), "exit_sha256": sha(finish_path)}), flush=True)


if __name__ == "__main__":
    main()
