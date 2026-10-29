"""One-shot, CPU-only parent observation for the public matched-dose taste exploration."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "scripts/bert_taste_matched_dose_v1.py"


def _git(*args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1"})
    return subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def source_snapshot():
    """Check the child's declared source bytes before child imports begin."""
    tree = ast.parse(CHILD.read_bytes())
    declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "SOURCE_PATHS"
                            for target in node.targets)]
    if len(declarations) != 1:
        raise ValueError("child source list is ambiguous")
    names = ast.literal_eval(declarations[0])
    if (type(names) is not tuple or len(names) != len(set(names))
            or "scripts/bert_taste_matched_parent_v1.py" not in names
            or any(type(name) is not str or not name or ".." in Path(name).parts
                   or Path(name).is_absolute() for name in names)):
        raise ValueError("child source list is invalid")
    commit = _git("rev-parse", "HEAD").decode("ascii").strip()
    hashes = {}
    for name in names:
        data = (ROOT / name).read_bytes()
        if data != _git("show", f"{commit}:{name}"):
            raise RuntimeError("selected source differs from its Git blob")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"commit": commit, "selected_source_sha256": hashes}


def _bounded(path, maximum):
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("saved evidence exceeds its bound")
    return data


def assess_result(output, exit_code, before, after, prefix_empty):
    """Require the child's exit and exact result/journal bytes for completion."""
    receipt = {"status": "incomplete", "child_exit_code": exit_code,
               "child_reaped": True, "source_unchanged": before == after,
               "bytecode_prefix_empty": prefix_empty}
    if exit_code != 0 or before != after or not prefix_empty or (output / "incomplete.json").exists():
        return receipt
    try:
        raw = _bounded(output / "result.json", 2 * 1024 * 1024)
        result = json.loads(raw)
        journals = result["journals"]
        if (result["status"] != "complete" or result["completed_forwards"] != 516
                or result["source"] != before
                or set(journals) != {"coffee.jsonl", "milk.jsonl", "soup.jsonl", "cake.jsonl", "pudding.jsonl", "cocoa.jsonl"}):
            return receipt
        for name, expected in journals.items():
            data = _bounded(output / name, 4 * 1024 * 1024)
            if expected != {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}:
                return receipt
    except (OSError, ValueError, TypeError, KeyError):
        return receipt
    receipt.update({"status": "complete", "result_sha256": hashlib.sha256(raw).hexdigest(),
                    "journals": journals, "source_commit": before["commit"]})
    return receipt


def _write_receipt(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        remaining = memoryview(raw)
        while remaining:
            written = stream.write(remaining)
            if type(written) is not int or not 0 < written <= len(remaining):
                raise OSError("receipt write made no progress")
            remaining = remaining[written:]
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    receipt_path = output.with_name(output.name + ".observer.json")
    if (output.exists() or receipt_path.exists() or not output.parent.is_dir()
            or output == ROOT or ROOT in output.parents):
        raise ValueError("a fresh output and receipt outside the checkout are required")
    before = source_snapshot()
    prefix = Path(tempfile.mkdtemp(prefix="bert-taste-matched-pyc-", dir=output.parent))
    if any(prefix.iterdir()):
        raise RuntimeError("fresh bytecode prefix is not empty")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "MKL_CBWR": "AUTO", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    env.pop("ATEN_CPU_CAPABILITY", None)
    command = [sys.executable, "-I", "-B", "-X", f"pycache_prefix={prefix}"]
    if args.preflight_only:
        command += ["-c", "import sys; assert sys.pycache_prefix and sys.dont_write_bytecode"]
    else:
        command += [str(CHILD), "--cache-dir", args.cache_dir, "--output", str(output)]
    try:
        child = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=300, check=False)
        after = source_snapshot()
        empty = not any(prefix.rglob("*"))
        if args.preflight_only:
            ok = child.returncode == 0 and before == after and empty
            receipt = {"status": "preflight_passed" if ok else "incomplete", "child_reaped": True,
                       "child_exit_code": child.returncode, "source_unchanged": before == after,
                       "bytecode_prefix_empty": empty, "source_commit": before["commit"]}
        else:
            receipt = assess_result(output, child.returncode, before, after, empty)
        _write_receipt(receipt_path, receipt)
        return 0 if receipt["status"] in ("complete", "preflight_passed") else 1
    except BaseException as exc:
        try:
            _write_receipt(receipt_path, {"status": "incomplete", "exception_type": type(exc).__name__,
                                          "child_reaped": False})
        except BaseException as secondary:
            if isinstance(secondary, (KeyboardInterrupt, SystemExit)) and not isinstance(
                    exc, (KeyboardInterrupt, SystemExit)):
                raise secondary from exc
        raise


if __name__ == "__main__":
    raise SystemExit(main())
