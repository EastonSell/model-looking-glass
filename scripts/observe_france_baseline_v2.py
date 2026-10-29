"""Observe one direct France CPU owner; timeouts leave descendants uncertain.

The external caller must verify these bytes and the source inventory against
public Git blobs, and create this interpreter's fresh empty bytecode prefix
before starting it. This cooperative record is not execution attestation.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def run_direct(command, directory, timeout, env, *, gate=None):
    """Run and reap a direct process; do not infer descendant cleanup on timeout."""
    before = _now()
    out = directory / "stdout.log"; err = directory / "stderr.log"
    with out.open("xb") as stdout, err.open("xb") as stderr:
        if gate is not None and (type(gate) is not bytes or not 1 <= len(gate) <= 64):
            raise ValueError("invalid direct start token")
        process = subprocess.Popen(command, stdin=subprocess.PIPE if gate is not None else subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr,
                                   env=env, cwd=directory,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        timed_out = False
        try:
            if gate is not None:
                data = (json.dumps({"pid": process.pid, "started_utc": before,
                                    "scope": "direct-process observation before start token"},
                                   sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                with (directory / "owner-start.json").open("xb") as start:
                    if start.write(data) != len(data):
                        raise RuntimeError("short direct-start receipt")
                    start.flush(); os.fsync(start.fileno())
                if process.stdin.write(gate) != len(gate):
                    raise RuntimeError("short direct-start token")
                process.stdin.flush(); process.stdin.close()
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            code = process.wait(timeout=10)
        except BaseException as error:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
            except BaseException as cleanup:
                error.add_note("direct-process cleanup was not confirmed: " + type(cleanup).__name__)
            raise
        stdout.flush(); os.fsync(stdout.fileno()); stderr.flush(); os.fsync(stderr.fileno())
    return {"status": "reaped" if not timed_out and code == 0 else "incomplete", "pid": process.pid,
            "exit_code": code, "timed_out": timed_out, "started_utc": before, "finished_utc": _now(),
            "stdout_sha256": _hash(out), "stderr_sha256": _hash(err),
            "descendant_cleanup": "unverified" if timed_out or code != 0 else "owner-supervisor-reported"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sources", "sources-sha256", "source-revision", "venv", "dependency", "study-root",
                 "observer-prefix", "observation", "output", "mode", "cache", "stage"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    root = Path(args.source).resolve(strict=True)
    prefix = Path(args.observer_prefix).resolve(strict=True)
    if (os.name != "nt" or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != prefix
            or any(prefix.iterdir()) or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"):
        raise RuntimeError("observer startup or fresh prefix differs")
    source_bytes = Path(args.sources).read_bytes()
    if len(source_bytes) > 2_000_000 or hashlib.sha256(source_bytes).hexdigest() != args.sources_sha256:
        raise RuntimeError("source map digest differs")
    raw_map = json.loads(source_bytes)
    owner_path = root / "scripts/france_baseline_cpu_owner_v2.py"
    owner_bytes = owner_path.read_bytes()
    if hashlib.sha256(owner_bytes).hexdigest() != raw_map.get("scripts/france_baseline_cpu_owner_v2.py"):
        raise RuntimeError("owner source differs")
    owner = types.ModuleType("_france_observed_owner"); owner.__file__ = str(owner_path)
    exec(compile(owner_bytes, str(owner_path), "exec"), owner.__dict__)
    hashes = owner._source_map(root, Path(args.sources), args.sources_sha256)
    if (Path(__file__).resolve(strict=True) != root / owner.OBSERVER
            or _hash(Path(__file__)) != hashes[owner.OBSERVER]):
        raise RuntimeError("observer source differs")
    guards = owner._module(root, owner.GUARDS, hashes)
    observation = Path(args.observation)
    observation.mkdir(exist_ok=False)
    phase = "source_protection"
    try:
        files = [root.joinpath(*owner._source_relative(name).parts) for name in hashes]
        files.append(Path(args.sources).resolve(strict=True))
        lease = guards.SourceFileLease(root, files); lease.__enter__()
        cache = guards.PrivateEmptyCache(observation); parent_prefix = cache.__enter__()
        if owner._source_map(root, Path(args.sources), args.sources_sha256) != hashes:
            raise RuntimeError("protected sources changed")
        base = Path(sys.base_prefix).resolve(strict=True) / "python.exe"
        if not os.path.samefile(Path(sys.executable).resolve(strict=True), base):
            raise RuntimeError("observer is not using the direct base interpreter")
        command = [str(base), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(parent_prefix), str(owner_path)]
        for name in ("source", "sources", "sources-sha256", "source-revision", "venv", "dependency", "study-root",
                     "output", "mode", "cache", "stage", "timeout"):
            command.extend(["--" + name, str(getattr(args, name.replace("-", "_")))])
        command.extend(["--prefix", str(parent_prefix)])
        env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
               "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "PROCESSOR_IDENTIFIER") if key in os.environ}
        env.update({"PATH": str(Path(env.get("SYSTEMROOT", "C:/Windows")) / "System32"),
                    "TEMP": str(observation), "TMP": str(observation), "CUDA_VISIBLE_DEVICES": "-1"})
        phase = "direct_owner"
        result = run_direct(command, observation, args.timeout + 45, env, gate=b"start\n")
        if result["status"] == "reaped":
            supervision = Path(args.output) / "supervision.json"
            if not supervision.is_file() or any(parent_prefix.iterdir()):
                raise RuntimeError("owner receipt or protected prefix differs")
            result["supervision_sha256"] = _hash(supervision)
        result.update({"source_sha256": args.sources_sha256, "source_revision": args.source_revision,
                       "scope": "direct-process observation; source revision is separately verified; no execution attestation"})
        owner._write(observation / "outer-exit.json", result)
        if result["status"] != "reaped":
            return 1
        cache.close(); lease.close()
        return 0
    except BaseException as error:
        try:
            owner._write(observation / "outer-incomplete.json", {"status": "incomplete", "phase": phase,
                         "error_type": type(error).__name__, "scope": "retain evidence; descendant cleanup may be uncertain"})
        except BaseException as recording:
            error.add_note("outer failure receipt could not be written: " + type(recording).__name__)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
