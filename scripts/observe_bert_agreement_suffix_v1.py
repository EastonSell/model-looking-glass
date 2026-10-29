"""Observe one direct CPU suffix-study child; no CUDA lease is implied."""
import sys

if __name__ == "__main__" and (not sys.flags.isolated or not sys.flags.no_site
                               or not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError("direct observer requires -I -S -B and a fresh pycache prefix")

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


SOURCE_PATHS = ("scripts/bert_agreement_suffix_v1.py", "scripts/read_bert_agreement_suffix_v1.py",
                "scripts/observe_bert_agreement_suffix_v1.py", "scripts/cross_model_agreement_v1.py",
                "scripts/read_cross_model_agreement_v1.py", "docs/BERT_AGREEMENT_SUFFIX_EXECUTION_V1.md",
                "docs/BERT_AGREEMENT_SUFFIX_V1.md",
                "experiments/bert_agreement_suffix_cases_v1.json")
PENDING_NAME = "observer-pending.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def checked_sources(repo, commit, git):
    if (not re.fullmatch("[0-9a-f]{40}", commit) or not git.is_absolute() or not git.is_file()
            or repo.resolve() != Path(__file__).resolve().parents[1]):
        raise ValueError("source commit, repo or Git executable differs")
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PATH") if key in os.environ}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"})
    if git.resolve().is_relative_to(repo.resolve()):
        raise ValueError("Git executable must be selected outside the checkout")
    kind = subprocess.run([str(git), "-C", str(repo), "cat-file", "-t", commit], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env).stdout.strip()
    if kind != b"commit":
        raise ValueError("source identifier does not name a commit")
    hashes = {}
    for name in SOURCE_PATHS:
        data = subprocess.run([str(git), "-C", str(repo), "cat-file", "blob", commit + ":" + name], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30).stdout
        if (repo / name).read_bytes() != data:
            raise ValueError("working source differs: " + name)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def available_physical_memory():
    class Status(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong)]
    value = Status()
    value.length = ctypes.sizeof(value)
    if os.name != "nt" or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise RuntimeError("cannot measure physical memory")
    return value.avail_phys


def save_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def observe(command, controller, attempt, commit, environment, timeout, repo, git, hashes, git_sha):
    process, exit_code, interrupted = None, None, None
    start = time.time()
    pending = controller / PENDING_NAME
    save_json(pending, {"schema": "bert-agreement-suffix-observer-pending-v1", "source_commit": commit,
                        "start_unix": start})
    with (controller / "child.log").open("xb") as log:
        try:
            process = subprocess.Popen(command, cwd=controller, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            save_json(controller / "spawn.json", {"pid": process.pid, "start_unix": start, "source_commit": commit})
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                interrupted = "timeout"
                process.terminate()
                exit_code = process.wait(timeout=15)
        except BaseException as error:
            interrupted = type(error).__name__
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    exit_code = process.wait(timeout=15)
                except BaseException:
                    exit_code = None
        finally:
            log.flush()
            os.fsync(log.fileno())
    receipt = {"schema": "bert-agreement-suffix-direct-child-exit-v1", "status": "incomplete",
               "child_pid": process.pid if process else None, "exit_code": exit_code, "interrupted": interrupted,
               "attempt_name": attempt.name, "controller_name": controller.name,
               "source_commit": commit, "start_unix": start, "end_unix": time.time()}
    completed_path, metadata_path = attempt / "complete.json", attempt / "metadata.json"
    if (process is not None and exit_code == 0 and interrupted is None and completed_path.is_file()
            and metadata_path.is_file() and not (attempt / "failure.txt").exists()):
        try:
            completed, metadata = json.loads(completed_path.read_bytes()), json.loads(metadata_path.read_bytes())
            unchanged = checked_sources(repo, commit, git) == hashes and digest(git) == git_sha
            if (unchanged and completed.get("status") == "complete"
                    and type(completed.get("events")) is int and completed.get("events") == 416
                    and type(metadata.get("pid")) is int and metadata.get("pid") == process.pid
                    and metadata.get("source_commit") == commit and metadata.get("sources_sha256") == hashes
                    and metadata.get("git_sha256") == git_sha
                    and completed.get("metadata_json_sha256") == digest(metadata_path)):
                receipt.update({"status": "complete", "complete_sha256": digest(completed_path),
                                "metadata_sha256": digest(metadata_path)})
        except BaseException as error:
            receipt["completion_check_error"] = type(error).__name__
    save_json(controller / "exit-observation.json", receipt)
    if receipt["status"] == "complete":
        pending.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "site", "git", "model-dir", "cases", "attempt", "controller"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    prefix = Path(sys.pycache_prefix)
    if (Path(sys.executable).resolve() != Path(sys._base_executable).resolve()
            or sys._xoptions.get("pycache_prefix") != sys.pycache_prefix
            or not prefix.is_absolute() or not prefix.is_dir() or prefix.is_symlink() or any(prefix.iterdir())
            or not 1 <= args.timeout <= 3600):
        raise ValueError("base interpreter or timeout differs")
    root = args.repo.resolve(strict=True)
    hashes, git_sha = checked_sources(root, args.source_commit, args.git), digest(args.git)
    for path in (args.site, args.model_dir):
        if not path.is_absolute() or not path.is_dir() or path.is_symlink():
            raise ValueError("dependency and model directories must be absolute ordinary directories")
    if args.cases.resolve() != (root / SOURCE_PATHS[-1]).resolve():
        raise ValueError("case path differs")
    if (not args.attempt.is_absolute() or args.attempt.exists() or not args.attempt.parent.is_dir()
            or not args.controller.is_absolute() or args.controller.exists() or not args.controller.parent.is_dir()):
        raise ValueError("attempt and controller must be new absolute paths")
    attempt_free, controller_free = (shutil.disk_usage(p.parent).free for p in (args.attempt, args.controller))
    host_free = available_physical_memory()
    if min(attempt_free, controller_free) < 1024**3 or host_free < 2 * 1024**3:
        raise RuntimeError("insufficient observed host or output volume headroom")
    args.controller.mkdir()
    prefix, temp = args.controller / "empty-pycache", args.controller / "temp"
    prefix.mkdir()
    temp.mkdir()
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT") if key in os.environ}
    system = Path(env.get("SYSTEMROOT", "C:/Windows"))
    env.update({"PATH": os.pathsep.join((str(args.git.parent), str(system / "System32"))),
                "TEMP": str(temp), "TMP": str(temp), "HF_HOME": str(args.controller / "hf"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "CUDA_VISIBLE_DEVICES": "-1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    command = [str(Path(sys._base_executable).resolve()), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(root / "scripts/bert_agreement_suffix_v1.py"), "--site", str(args.site.resolve()),
               "--git", str(args.git), "--model-dir", str(args.model_dir.resolve()),
               "--cases", str(args.cases.resolve()), "--output", str(args.attempt),
               "--device", "cpu", "--source-commit", args.source_commit]
    save_json(args.controller / "admission.json", {"attempt_free_bytes": attempt_free,
              "controller_free_bytes": controller_free, "available_physical_bytes": host_free,
              "output_volume_minimum_bytes": 1024**3, "host_minimum_bytes": 2 * 1024**3,
              "scope": "point-in-time measurements, not reservations or peak-memory guarantees"})
    receipt = observe(command, args.controller, args.attempt, args.source_commit, env, args.timeout,
                      root, args.git, hashes, git_sha)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
