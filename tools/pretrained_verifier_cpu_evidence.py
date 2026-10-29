"""Run the real-checkpoint CPU verifier challenge and save aggregate evidence."""
import argparse
import ast
import datetime
import hashlib
import importlib.metadata
import json
import math
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


def _unittest_command(executable, private_prefix):
    return [str(executable), "-I", "-B", "-X", f"pycache_prefix={private_prefix}",
            "-m", "unittest", "discover", "-s", "tests", "-p", "test_pretrained_verifier_cpu.py", "-v"]


def _validate_summary(observed, case_sha256, verifier_sha256):
    cases = ["parameter_value", "attention_mask_value", "masked_bias_value", "vocab_config",
             "training_mode", "attention_scaling", "parameter_stride", "attention_mask_stride",
             "parameter_storage_offset", "attention_mask_storage_alias", "shared_storage_distinct_parameter",
             "untied_head", "local_hook", "global_hook"]
    fields = {"status", "verifier_source_sha256", "case_sha256", "rejected_cases", "max_absolute_logit_changes",
              "restored_output_bitwise_equal", "direct_loader_calls", "guarded_cuda_and_network_calls"}
    differences = observed.get("max_absolute_logit_changes") if isinstance(observed, dict) else None
    if (not isinstance(observed, dict) or set(observed) != fields or not isinstance(differences, dict)
            or set(differences) != {"parameter_value", "attention_mask_value", "attention_scaling"}
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in differences.values())
            or observed.get("status") != "passed" or observed.get("verifier_source_sha256") != verifier_sha256
            or observed.get("case_sha256") != case_sha256 or observed.get("rejected_cases") != cases
            or observed.get("restored_output_bitwise_equal") is not True
            or type(observed.get("direct_loader_calls")) is not int or observed["direct_loader_calls"] != 1
            or type(observed.get("guarded_cuda_and_network_calls")) is not int
            or observed["guarded_cuda_and_network_calls"] != 0):
        raise ValueError("summary identity mismatch")


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
    if Path(__file__).resolve(strict=True) != repo / "tools" / "pretrained_verifier_cpu_evidence.py":
        raise RuntimeError("run the evidence collector from the selected source checkout")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("source must be clean before the test")
    commit = git(repo, "rev-parse", "HEAD")
    source_paths = ["tools/pretrained_verifier_cpu_evidence.py", "docs/PRETRAINED_VERIFIER_CPU.md",
                    "tests/test_pretrained_verifier_cpu.py", "looking_glass/__init__.py",
                    "looking_glass/pretrained.py", "looking_glass/live_profile.py", "pyproject.toml"]
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
                "LOOKING_GLASS_VERIFIER_CPU_SUMMARY": "1", "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    record = {"schema": "pretrained-verifier-cpu-evidence-v2", "purpose": args.purpose,
              "status": "failed", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_commit": commit, "source_tree": git(repo, "show", "-s", "--format=%T", commit),
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
                           "input_shape": [1, 2], "input_dtype": "torch.int64",
                           "output_shape": [1, 2, 50257], "output_dtype": "torch.float32",
                           "source_preflight": "synthetic verifier fixture; not production attestation"},
              "invocation": {"command": _unittest_command("python", "<private-empty>"),
                             "cache_policy": "explicit local cache; download disabled",
                             "cache_path_existed_before": cache.exists(), "summary_enabled": True,
                             "collector": "tools/pretrained_verifier_cpu_evidence.py"},
              "notes": "CPU correctness only; guarded calls apply after PyTorch import, not all native CUDA paths. "
                       "The synthetic SourcePreflight is not an attestation, benchmark, or GPU profile."}
    try:
        if args.purpose == "controlled-missing-cache" and record["invocation"]["cache_path_existed_before"]:
            raise ValueError("controlled missing-cache path was not absent")
        # Isolate the unittest parent too: -B alone does not prevent it from
        # reading a stale bytecode cache before executing the verified test.
        with tempfile.TemporaryDirectory(prefix="verifier-evidence-parent-", dir=scratch) as temp:
            private_prefix = Path(temp) / "pycache"
            private_prefix.mkdir()
            command = _unittest_command(sys.executable, private_prefix)
            completed = subprocess.run(command, cwd=repo, env=env, capture_output=True, timeout=240, check=False)
            record["unittest_parent_pycache_empty_after"] = not any(private_prefix.iterdir())
        record["exit_code"] = completed.returncode
        record["stdout_sha256"] = hashlib.sha256(completed.stdout).hexdigest()
        record["stderr_sha256"] = hashlib.sha256(completed.stderr).hexdigest()
        lines = [line.removeprefix("PRETRAINED_VERIFIER_CPU_V2 ")
                 for line in completed.stdout.decode("utf-8", errors="replace").splitlines()
                 if line.startswith("PRETRAINED_VERIFIER_CPU_V2 ")]
        record["final_source_commit"] = git(repo, "rev-parse", "HEAD")
        record["selected_source_files_unchanged"] = (
            {name: sha256(repo / name) for name in source_paths} == hashes)
        record["final_git_status_clean"] = not git(repo, "status", "--porcelain", "--untracked-files=all")
        if (completed.returncode == 0 and len(lines) == 1 and record["final_source_commit"] == commit
                and record["selected_source_files_unchanged"] and record["final_git_status_clean"]
                and record["unittest_parent_pycache_empty_after"]):
            observed = json.loads(lines[0])
            _validate_summary(observed, record["protocol"]["case_sha256"], hashes["looking_glass/live_profile.py"])
            record["observed"] = observed
            record["status"] = "passed"
        else:
            match = re.search(rb"CPU verifier child failed at ([a-z_]+);", completed.stderr)
            phases = {"startup", "torch_import", "imports", "fixture", "model_load", "parameter_value",
                      "attention_mask_value", "masked_bias_value", "vocab_config", "training_mode",
                      "attention_scaling", "parameter_stride", "attention_mask_stride", "parameter_storage_offset",
                      "attention_mask_storage_alias", "shared_storage_distinct_parameter", "untied_head", "local_hook", "global_hook",
                      "restored_model", "final"}
            phase = match.group(1).decode("ascii") if match else None
            if phase in phases:
                record["failure"] = {"phase": phase, "message": "child failure details withheld"}
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
