"""Run the bounded CPU bootstrap preflight and save aggregate-only evidence."""
import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("correctness", "controlled-missing-cache"), required=True)
    args = parser.parse_args()
    repo, scratch = (p.resolve(strict=True) for p in (args.repository, args.scratch))
    cache, output = (p.resolve() for p in (args.cache, args.output))
    if output.exists():
        raise FileExistsError("use a new evidence filename")
    if Path(__file__).resolve(strict=True) != repo / "tools" / "bootstrap_pretrained_cpu_evidence.py":
        raise RuntimeError("run the evidence collector from the selected source checkout")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source must be clean before the test")
    commit = git(repo, "rev-parse", "HEAD")
    source_paths = ["tools/bootstrap_pretrained_cpu_evidence.py", "docs/BOOTSTRAP_CPU_PREFLIGHT.md",
                    "tests/test_bootstrap_pretrained_cpu.py",
                    "looking_glass/__init__.py", "looking_glass/pretrained.py", "looking_glass/ipc.py",
                    "looking_glass/resident_worker.py", "pyproject.toml"]
    hashes = {name: sha256(repo / name) for name in source_paths}
    for name, digest in hashes.items():
        blob = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":" + name])
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError("source bytes differ from the committed Git blob")
    # Read the verified literal declarations without importing repository code:
    # the collector itself is not launched under the child's bytecode policy.
    source = ast.parse((repo / "looking_glass/pretrained.py").read_bytes())
    declarations = {}
    for node in source.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"ASSETS", "MODEL_ID", "REVISION"}):
            name = node.targets[0].id
            if name in declarations:
                raise ValueError("duplicate pinned declaration")
            declarations[name] = ast.literal_eval(node.value)
    if set(declarations) != {"ASSETS", "MODEL_ID", "REVISION"}:
        raise ValueError("pinned declarations are unavailable")
    ASSETS, MODEL_ID, REVISION = (declarations[name] for name in ("ASSETS", "MODEL_ID", "REVISION"))
    packages = {}
    site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    for name in ("torch", "transformers", "huggingface_hub", "safetensors", "tokenizers", "numpy"):
        found = list(importlib.metadata.distributions(name=name, path=[str(site)]))
        packages[name] = found[0].version if len(found) == 1 else None
    base = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve(strict=True)
    env = os.environ.copy()
    env.update({"LOOKING_GLASS_TEST_CACHE": str(cache), "LOOKING_GLASS_TEST_SCRATCH": str(scratch),
                "LOOKING_GLASS_BOOTSTRAP_CPU_SUMMARY": "1", "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    record = {"schema": "bootstrap-pretrained-cpu-evidence-v1", "purpose": args.purpose,
              "status": "failed", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_commit": commit, "source_tree": git(repo, "show", "-s", "--format=%T", "HEAD"),
              "source_files_sha256": hashes,
              "checkpoint": {"model_id": MODEL_ID, "model_revision": REVISION, "tokenizer_id": MODEL_ID,
                             "tokenizer_revision": REVISION,
                             "assets": {name: {"size": value[0], "sha256": value[1]}
                                        for name, value in ASSETS.items()}},
              "runtime": {"python": sys.version.split()[0], "base_python_sha256": sha256(base),
                          "outer_python_sha256": sha256(Path(sys.executable).resolve(strict=True)),
                          "platform": platform.platform(), "packages": packages,
                          "device_setting": "cpu", "child_direct_base_interpreter_setting": True,
                          "child_no_site_setting": True, "child_no_user_site_setting": True,
                          "child_bytecode_writes_disabled_setting": True, "child_private_empty_pycache_setting": True,
                          "cuda_hidden_setting": True, "offline_setting": True, "seed_setting": 17,
                          "reference_intraop_threads_setting": 1},
              "protocol": {"case_sha256": "a137759c4f2a40c4da15d1a4625086e160edc75f161748db91b07d83dae751e4",
                           "input_shape": [1, 2], "input_dtype": "torch.int64", "input_limit_bytes": 65536,
                           "output_shape": [1, 2, 50257], "output_dtype": "torch.float32",
                           "output_tensor_limit_bytes": 524288, "response_frame_limit_bytes": 4194304},
              "invocation": {"command": ["python", "-I", "-B", "-X", "pycache_prefix=<private-empty>",
                                         "-m", "unittest", "discover", "-s", "tests", "-p",
                                         "test_bootstrap_pretrained_cpu.py", "-v"],
                             "cache_policy": "explicit local cache; download disabled",
                             "cache_path_existed_before": cache.exists(), "summary_enabled": True,
                             "collector": "tools/bootstrap_pretrained_cpu_evidence.py"},
              "notes": "CPU correctness only; guarded calls apply after PyTorch import, not all native CUDA paths. "
                       "The synthetic frame observation is not an upper bound, benchmark, or GPU profile."}
    try:
        if args.purpose == "controlled-missing-cache" and record["invocation"]["cache_path_existed_before"]:
            raise ValueError("controlled missing-cache path was not absent")
        # Isolate the unittest parent too: -B alone does not prevent it from
        # reading a stale bytecode cache before executing the verified test.
        with tempfile.TemporaryDirectory(prefix="bootstrap-evidence-parent-", dir=scratch) as temp:
            private_prefix = Path(temp) / "pycache"
            private_prefix.mkdir()
            command = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={private_prefix}",
                       "-m", "unittest", "discover", "-s", "tests", "-p",
                       "test_bootstrap_pretrained_cpu.py", "-v"]
            completed = subprocess.run(command, cwd=repo, env=env, capture_output=True, timeout=240, check=False)
            record["unittest_parent_pycache_empty_after"] = not any(private_prefix.iterdir())
        record["exit_code"] = completed.returncode
        record["stdout_sha256"] = hashlib.sha256(completed.stdout).hexdigest()
        record["stderr_sha256"] = hashlib.sha256(completed.stderr).hexdigest()
        lines = [line.removeprefix("BOOTSTRAP_PRETRAINED_CPU_V1 ")
                 for line in completed.stdout.decode("utf-8", errors="replace").splitlines()
                 if line.startswith("BOOTSTRAP_PRETRAINED_CPU_V1 ")]
        record["final_source_commit"] = git(repo, "rev-parse", "HEAD")
        record["source_unchanged"] = (record["final_source_commit"] == commit and
            {name: sha256(repo / name) for name in source_paths} == hashes and
            not git(repo, "status", "--porcelain", "--untracked-files=all"))
        if (completed.returncode == 0 and len(lines) == 1 and record["source_unchanged"]
                and record["unittest_parent_pycache_empty_after"]):
            observed = json.loads(lines[0])
            if observed.get("status") != "passed" or observed.get("source_commit") != commit:
                raise ValueError("summary identity mismatch")
            record["observed"] = observed
            record["status"] = "passed"
        else:
            match = re.search(rb"private CPU preflight failed at ([a-z_]+): ([A-Za-z_]+)", completed.stderr)
            phases = {"startup", "source_initial", "torch_import", "imports", "assets",
                      "negative_cache_checks", "model_load", "forward", "response_frame",
                      "provenance", "source_final"}
            errors = {"AssertionError", "CheckpointUnavailableError", "CheckpointIntegrityError",
                      "OSError", "FileNotFoundError", "ImportError", "RuntimeError", "ValueError",
                      "TypeError", "TimeoutExpired", "KeyError", "CalledProcessError", "OtherError"}
            phase = match.group(1).decode("ascii") if match else None
            error = match.group(2).decode("ascii") if match else None
            if phase in phases and error in errors:
                record["failure"] = {"phase": phase, "error_type": error}
            else:
                record["failure"] = {"message": "test failed, summary missing, or source changed; details withheld"}
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "message": "execution failed; details withheld"}
    record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "purpose": record["purpose"],
                      "source_commit": commit, "observed": record.get("observed"),
                      "failure": record.get("failure")}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
