"""Check a closed grouped-controller startup failure without model or GPU use.

The arguments identify retained private records and the exact frozen checkout.
The output contains hashes of withheld records, not their paths or contents.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guarded_writer_probe(checkout, scratch):
    require(os.name == "nt", "the sharing-mode reproduction requires Windows")
    guard = load_source("grouped_failure_guard_probe", checkout / "scripts/v2_source_guards.py")
    writer = load_source("grouped_failure_writer_probe", checkout / "looking_glass/bert_gpu_evidence.py")
    with tempfile.TemporaryDirectory(prefix="grouped-startup-share-", dir=scratch) as temp:
        root = Path(temp)
        manifest = root / "source-manifest.json"
        manifest.write_bytes(b"{}")
        writer.write_json_new(root / "before.json", {"phase": "before"})
        with guard.SourceFileLease(root, [manifest]):
            try:
                writer.write_json_new(root / "during.json", {"phase": "during"})
            except OSError as exc:
                require(getattr(exc, "winerror", None) == 32, "guarded write failed for a different reason")
            else:
                raise ValueError("guarded hard-link publication unexpectedly succeeded")
            require(not (root / "during.json").exists()
                    and json.loads((root / "during.json.partial").read_bytes()) == {"phase": "during"},
                    "guarded publication did not retain the expected partial")
        writer.write_json_new(root / "after.json", {"phase": "after"})
        require((root / "before.json").is_file() and (root / "after.json").is_file(),
                "unguarded publication failed")
    return {"guarded_hard_link_winerror": 32, "unguarded_publications_succeeded": True,
            "guarded_partial_retained": True}


def audit(run, observation, checkout, source, scratch):
    source_audit = json.loads(source.read_bytes())
    require(source_audit.get("status") == "passed" and source_audit.get("gpu_boundary_enabled") is True,
            "source audit did not establish an enabled frozen revision")
    require(source_audit.get("cache_assets_verified") is True, "checkpoint inventory was not verified")
    commit = source_audit["source_commit"]
    for name in ("looking_glass/grouped_gpu_dev.py", "looking_glass/bert_gpu_evidence.py",
                 "scripts/v2_source_guards.py"):
        require(digest((checkout / name).read_bytes()) == source_audit["source_sha256"][name],
                "startup or reproduction source differs from the frozen source audit")
    source_bytes = (checkout / "looking_glass/grouped_gpu_dev.py").read_bytes()
    tree = ast.parse(source_bytes)
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_development")
    calls = [node for node in ast.walk(entry) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    start = [node for node in calls if node.func.id == "_write_json" and node.args
             and any(isinstance(item, ast.Constant) and item.value == "started.json"
                     for item in ast.walk(node.args[0]))]
    supervise = [node for node in calls if node.func.id == "_supervise_owner"]
    require(len(start) == len(supervise) == 1 and start[0].lineno < supervise[0].lineno,
            "startup publication no longer precedes owner supervision")

    partial_raw = (run / "started.json.partial").read_bytes()
    manifest_raw = (run / "source-manifest.json").read_bytes()
    partial, manifest = json.loads(partial_raw), json.loads(manifest_raw)
    require(partial.get("schema") == "grouped-gpu-start-v1"
            and manifest.get("schema") == "grouped-gpu-source-manifest-v1"
            and partial.get("source_commit") == manifest.get("source_commit") == commit
            and partial.get("protocol_sha256") == source_audit["protocol_sha256"]
            and partial.get("source_manifest_sha256") == digest(manifest_raw)
            and partial.get("run_id") == manifest.get("run_id")
            and manifest.get("source_sha256") == source_audit["source_sha256"],
            "partial startup record and source manifest differ")
    absent = ("started.json", "owner.json", "candidate.json", "final.json", "supervision.json",
              "protocol-source.json", "asset-verification.json", "private-selected-device.json")
    require(all(not (run / name).exists() for name in absent), "unexpected later owner or startup record exists")

    exit_raw = (observation / "exit.json").read_bytes()
    stdout = (observation / "stdout.log").read_bytes()
    stderr = (observation / "stderr.log").read_bytes()
    receipt = json.loads(exit_raw)
    error = stderr.decode("utf-8", "replace")
    require(receipt.get("schema") == "grouped-v3-controller-exit-observation-v1"
            and receipt.get("source_commit") == commit and receipt.get("controller_exit") == 1
            and stdout == b"" and "WinError 32" in error and "os.link(partial, destination)" in error
            and "started.json.partial" in error
            and f"line {start[0].lineno}, in run_development" in error,
            "controller exit or traceback differs from the startup failure")
    return {"status": "passed", "source_commit": commit, "controller_exit": 1,
            "failure_stage": "startup-record-publication-before-owner-supervision",
            "startup_final_present": False, "saved_owner_records_present": False,
            "record_sha256": {"startup_partial": digest(partial_raw), "source_manifest": digest(manifest_raw),
                              "controller_exit_observation": digest(exit_raw), "controller_stderr": digest(stderr)},
            "independent_windows_probe": guarded_writer_probe(checkout, scratch),
            "limitations": ["record hashes are commitments to withheld private evidence",
                            "saved records and source order do not independently attest all process activity",
                            "no model forward, device memory profile, or native cleanup is established by this audit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("controller_observation")
    parser.add_argument("checkout")
    parser.add_argument("source_audit")
    parser.add_argument("scratch")
    args = parser.parse_args()
    result = audit(*(Path(value) for value in (args.run, args.controller_observation, args.checkout,
                                               args.source_audit, args.scratch)))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
