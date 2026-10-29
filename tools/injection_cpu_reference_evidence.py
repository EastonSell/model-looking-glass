"""Collect a redacted, source-bound CPU mechanics check; not a benchmark."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "tools/injection_cpu_reference_evidence.py", "tests/test_number_injection_pretrained_reference.py",
    "looking_glass/pretrained.py", "looking_glass/live_profile.py", "looking_glass/pretrained_number_injection.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py", "looking_glass/model_ownership.py",
    "looking_glass/agreement_numeric.py",
)
_SUMMARY_KEYS = {"model_revision", "weight_sha256", "input_sha256", "device", "site",
                 "reference_forwards", "executor_forwards", "max_abs_error_by_condition", "guarded_calls"}


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _state():
    commit = _git("rev-parse", "HEAD").decode().strip()
    tree = _git("show", "-s", "--format=%T", commit).decode().strip()
    clean = not _git("status", "--porcelain", "--untracked-files=all").strip()
    hashes = {}
    matches = True
    for name in FILES:
        actual = (ROOT / name).read_bytes()
        if actual != _git("show", f"{commit}:{name}"):
            matches = False
        hashes[name] = _sha(actual)
    return commit, tree, clean, hashes, matches


def _assets():
    # Parse verified literals; do not import repository code in the collector.
    wanted = {"MODEL_ID", "REVISION", "ASSETS"}
    result = {}
    tree = ast.parse((ROOT / "looking_glass/pretrained.py").read_bytes())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in wanted:
                result[node.targets[0].id] = ast.literal_eval(node.value)
    if set(result) != wanted:
        raise ValueError("pinned checkpoint literals are unavailable")
    return result


def _digest_file(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(65536)
        if not block:
            return digest.hexdigest(), size
        digest.update(block)
        size += len(block)


def _summary(stream, assets):
    stream.seek(0)
    raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("summary stream exceeds its cap")
    lines = [line for line in raw.splitlines() if line.startswith(b"{")]
    if len(lines) != 1:
        raise ValueError("one bounded aggregate summary is required")
    result = json.loads(lines[0])
    if (type(result) is not dict or set(result) != _SUMMARY_KEYS or
            result["model_revision"] != assets["REVISION"] or
            result["weight_sha256"] != assets["ASSETS"]["model.safetensors"][1] or
            result["input_sha256"] != "a137759c4f2a40c4da15d1a4625086e160edc75f161748db91b07d83dae751e4" or
            result["device"] != "cpu" or type(result["site"]) is not int or result["site"] != 2 or
            type(result["reference_forwards"]) is not int or result["reference_forwards"] != 4 or
            type(result["executor_forwards"]) is not int or result["executor_forwards"] != 4 or
            type(result["guarded_calls"]) is not int or result["guarded_calls"] != 0 or
            type(result["max_abs_error_by_condition"]) is not list or len(result["max_abs_error_by_condition"]) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                for x in result["max_abs_error_by_condition"])):
        raise ValueError("aggregate summary is invalid")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache, output = Path(args.cache_dir), Path(args.output)
    if not cache.is_absolute() or not output.is_absolute() or not output.parent.is_dir() or output.exists():
        raise ValueError("absolute cache and new output paths are required")
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("evidence output must be outside the source checkout")
    initial = _state()
    if not initial[2] or not initial[4]:
        raise ValueError("source checkout and selected committed bytes must be clean")
    assets = _assets()
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {"schema": "mlg-disclosed-injection-cpu-reference-v1", "status": "failed",
              "source_commit": initial[0], "source_tree": initial[1], "selected_source_sha256": initial[3],
              "checkpoint": {"model_id": assets["MODEL_ID"], "revision": assets["REVISION"],
                             "asset_sha256": {name: pair[1] for name, pair in assets["ASSETS"].items()}},
              "runtime": {"python": sys.version, "platform": platform.platform(), "libraries": versions,
                          "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
              "cache_exists_before": cache.is_dir(), "scope": "disclosed-site CPU development mechanics; not discovery or benchmark"}
    with tempfile.TemporaryDirectory(prefix="injection-reference-", dir=output.parent) as private:
        prefix = Path(private) / "pycache"
        prefix.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LOOKING_GLASS_TEST_CACHE=str(cache),
                   LOOKING_GLASS_REFERENCE_SUMMARY="1", PYTHONDONTWRITEBYTECODE="1")
        code = ("import sys,unittest; sys.path.insert(0,sys.argv.pop(1)); "
                "unittest.main(module=None,argv=['unittest','tests.test_number_injection_pretrained_reference'])")
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                completed = subprocess.run([sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(prefix),
                                            "-c", code, str(ROOT)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                           timeout=180, check=False)
                record["return_code"] = completed.returncode
                record["failure_phase"] = "test" if completed.returncode else None
            except subprocess.TimeoutExpired:
                record["return_code"] = None
                record["failure_phase"] = "timeout"
            except OSError:
                record["return_code"] = None
                record["failure_phase"] = "launch"
            record["stdout_sha256"], record["stdout_bytes"] = _digest_file(stdout)
            record["stderr_sha256"], record["stderr_bytes"] = _digest_file(stderr)
            try:
                final = _state()
                record["final_source_commit"] = final[0]
                record["selected_source_unchanged"] = initial == final and not any(prefix.iterdir())
            except (OSError, ValueError, subprocess.SubprocessError):
                record["final_source_commit"] = None
                record["selected_source_unchanged"] = False
            if record["return_code"] == 0 and record["selected_source_unchanged"]:
                try:
                    record["observation"] = _summary(stdout, assets)
                    record["status"] = "passed"
                except (ValueError, TypeError, KeyError, UnicodeError):
                    record["failure_phase"] = "summary"
            elif not record["selected_source_unchanged"]:
                record["failure_phase"] = "source_changed"
    encoded = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as handle:
        if handle.write(encoded) != len(encoded):
            raise OSError("short evidence write")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": record["status"], "failure_phase": record["failure_phase"]}, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
