"""One-shot direct-child operator for the fixed public FLAN sentiment screen.

The recorded exit is a local observation, not authentication of execution or
proof of descendant quiescence. This script never retries a study attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

EXPECTED_COMMIT = "6f7493f0db5114ead1336bef2383dd3b20d5e582"
EXPECTED_MANIFEST = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
SOURCE_PATHS = ("looking_glass/flan_t5_sentiment_screen.py", "looking_glass/flan_t5_feasibility.py",
                "tests/test_flan_t5_sentiment_screen.py", "docs/FLAN_T5_SENTIMENT_SCREEN_V1.md")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_new(path, record):
    data = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short operator record write")
        stream.flush()
        os.fsync(stream.fileno())


def check_source(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != EXPECTED_COMMIT or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root):
        raise ValueError("source checkout is not the clean expected public commit")
    names = {}
    for path in SOURCE_PATHS:
        actual = (root / path).read_bytes()
        if actual != subprocess.check_output(["git", "show", f"{head}:{path}"], cwd=root):
            raise ValueError("source path differs from the expected commit")
        names[path] = hashlib.sha256(actual).hexdigest()
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args(argv)
    root, assets, attempt = args.repo.resolve(strict=True), args.asset_dir.resolve(strict=True), args.attempt
    if not attempt.is_absolute():
        raise ValueError("attempt must be an absolute new path")
    hashes = check_source(root)
    initializer = (root / "looking_glass/__init__.py").read_bytes()
    committed_initializer = subprocess.check_output(
        ["git", "show", f"{EXPECTED_COMMIT}:looking_glass/__init__.py"], cwd=root)
    if initializer != committed_initializer:
        raise ValueError("package initializer differs from the expected commit")
    initializer_sha = hashlib.sha256(initializer).hexdigest()
    if sha(assets / "asset-manifest.json") != EXPECTED_MANIFEST:
        raise ValueError("asset manifest differs from the frozen identity")
    if shutil.disk_usage(attempt.parent).free < 1024 ** 3:
        raise ValueError("less than 1 GiB free on the attempt volume")
    attempt.mkdir(exist_ok=False)
    prefix, output = attempt / "empty-pycache", attempt / "child"
    prefix.mkdir()
    if prefix.is_symlink() or any(prefix.iterdir()):
        raise ValueError("bytecode prefix is not fresh and empty")
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "ATEN_CPU_CAPABILITY"):
        environment.pop(name, None)
    environment.update({"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE",
                        "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    loader = ("import importlib.util,os,sys; root=sys.argv.pop(1); "
              "p=os.path.join(root,'looking_glass'); "
              "s=importlib.util.spec_from_file_location('looking_glass',os.path.join(p,'__init__.py'),"
              "submodule_search_locations=[p]); m=importlib.util.module_from_spec(s); "
              "sys.modules['looking_glass']=m; s.loader.exec_module(m); "
              "from looking_glass.flan_t5_sentiment_screen import main; main()")
    command = [sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(prefix), "-c", loader,
               str(root), "--asset-dir", str(assets), "--output", str(output)]
    write_new(attempt / "operator-start.json", {"schema": "flan-sentiment-operator-v1",
              "source_commit": EXPECTED_COMMIT, "source_sha256": hashes,
              "package_initializer_sha256": initializer_sha,
              "asset_manifest_sha256": EXPECTED_MANIFEST, "operator_sha256": sha(Path(__file__)),
              "fresh_prefix_before": True, "offline": True, "cuda_visible_devices": "-1",
              "scope": "Local direct-child observation; trusted interpreter and dependencies."})
    child, code, timed_out, failure = None, None, False, None
    try:
        with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, **options)
            write_new(attempt / "operator-spawn.json", {"pid": child.pid})
            try:
                code = child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                timed_out = True
                child.kill()
                code = child.wait(timeout=15)
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    except BaseException as exc:
        failure = type(exc).__name__
        if child is not None and child.poll() is None:
            try:
                child.kill()
                code = child.wait(timeout=15)
            except BaseException:
                pass
        raise
    finally:
        terminal_path, journal_path = output / "terminal.json", output / "events.jsonl"
        terminal_sha = sha(terminal_path) if terminal_path.is_file() else None
        events_sha = sha(journal_path) if journal_path.is_file() else None
        consistent = False
        try:
            terminal = json.loads(terminal_path.read_bytes())
            consistent = (terminal["status"] == "complete" and terminal["events_sha256"] == events_sha
                          and terminal["source"]["commit"] == EXPECTED_COMMIT
                          and terminal["source"]["source_sha256"] == hashes
                          and terminal["asset_manifest_sha256"] == EXPECTED_MANIFEST
                          and not (output / "failed.json").exists())
        except (OSError, ValueError, KeyError, TypeError):
            pass
        write_new(attempt / "operator-exit.json", {"schema": "flan-sentiment-operator-v1",
                  "direct_child_exit_code": code, "child_reaped": code is not None,
                  "timed_out": timed_out, "operator_exception_type": failure,
                  "terminal_sha256": terminal_sha, "events_sha256": events_sha,
                  "terminal_consistent": consistent, "prefix_empty_after": not any(prefix.iterdir()),
                  "stdout_sha256": sha(attempt / "stdout.log") if (attempt / "stdout.log").is_file() else None,
                  "stderr_sha256": sha(attempt / "stderr.log") if (attempt / "stderr.log").is_file() else None})
    if code != 0 or timed_out or not consistent or any(prefix.iterdir()):
        raise RuntimeError("attempt is incomplete or uncertain; retain all evidence")
    print(json.dumps({"status": "complete", "terminal_sha256": terminal_sha,
                      "events_sha256": events_sha, "source_commit": EXPECTED_COMMIT}, sort_keys=True))


if __name__ == "__main__":
    main()
