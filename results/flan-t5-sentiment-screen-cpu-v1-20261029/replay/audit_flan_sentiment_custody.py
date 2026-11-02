"""Reconcile the closed sentiment screen's saved source, assets and receipts.

This is saved-file consistency, not authenticated execution, publication-time
attestation, full process-tree cleanup, or an independent numerical replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

COMMIT = "6f7493f0db5114ead1336bef2383dd3b20d5e582"
MANIFEST = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
EVENTS = "eb17ca8482e0c28ed4d6146285aaf33fa63db3437f0fb308c5becfa5e50d3d6b"
TERMINAL = "843a7140cdd7914fb2e5d6a0c382b05f4eb721b561f71f70d04fcdb4ff1d1e8f"
INNER = "49839711fbc07d9bee47ad43503f3f7bc2a1b98dd5f837aa3e67a8b00f9e0d12"
OUTER = "94e768ea200bb8b2599b2a0eb152995d5c6ac48dd6d6c5fcbf5eabe74e2d8db9"
OPERATOR = "3304f4834d10b040a064c4ad25f8201e68727ae7af40be1d2c7f79f7a77e443e"
FILES = ("looking_glass/flan_t5_sentiment_screen.py", "looking_glass/flan_t5_feasibility.py",
         "tests/test_flan_t5_sentiment_screen.py", "docs/FLAN_T5_SENTIMENT_SCREEN_V1.md")
ASSETS = {"config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
          "spiece.model", "tokenizer.json", "tokenizer_config.json"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            check(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo", "assets", "attempt", "operator", "outer"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inner, outer = args.attempt / "operator-exit.json", args.outer
    start, spawn = load(args.attempt / "operator-start.json"), load(args.attempt / "operator-spawn.json")
    child_exit, outer_exit = load(inner), load(outer)
    terminal, events = args.attempt / "child/terminal.json", args.attempt / "child/events.jsonl"
    result = load(terminal)
    check(sha(inner) == INNER and sha(outer) == OUTER and sha(terminal) == TERMINAL
          and sha(events) == EVENTS and sha(args.operator) == OPERATOR, "pinned file digest differs")
    check(start["operator_sha256"] == OPERATOR and start["source_commit"] == COMMIT
          and start["asset_manifest_sha256"] == MANIFEST and start["fresh_prefix_before"] is True,
          "operator start differs")
    check(type(spawn.get("pid")) is int and spawn["pid"] > 0, "spawn record differs")
    check(type(child_exit.get("direct_child_exit_code")) is int and child_exit["direct_child_exit_code"] == 0
          and child_exit["child_reaped"] is True and child_exit["timed_out"] is False
          and child_exit["operator_exception_type"] is None and child_exit["terminal_consistent"] is True
          and child_exit["prefix_empty_after"] is True and child_exit["terminal_sha256"] == TERMINAL
          and child_exit["events_sha256"] == EVENTS, "direct-child receipt differs")
    check(type(outer_exit.get("operator_exit_code")) is int and outer_exit["operator_exit_code"] == 0
          and outer_exit["parent_prefix_empty"] is True and outer_exit["source_commit"] == COMMIT,
          "outer observation differs")
    for name in ("stdout", "stderr"):
        check(sha(args.attempt / (name + ".log")) == child_exit[name + "_sha256"], "log digest differs")
    prefix = args.attempt / "empty-pycache"
    check(prefix.is_dir() and not prefix.is_symlink() and not any(prefix.iterdir()), "child prefix differs")
    check(not (args.attempt / "child/failed.json").exists() and result["status"] == "complete"
          and result["events_sha256"] == EVENTS and result["asset_manifest_sha256"] == MANIFEST,
          "terminal differs")
    check(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip() == COMMIT,
          "checkout commit differs")
    hashes = {}
    for name in (*FILES, "looking_glass/__init__.py"):
        data = (args.repo / name).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=args.repo)
        check(data == committed, "source blob differs: " + name)
        hashes[name] = hashlib.sha256(data).hexdigest()
    check({name: hashes[name] for name in FILES} == start["source_sha256"]
          and start["package_initializer_sha256"] == hashes["looking_glass/__init__.py"]
          and result["source"] == {"commit": COMMIT, "source_sha256": start["source_sha256"]},
          "recorded source binding differs")
    manifest = args.assets / "asset-manifest.json"
    check(sha(manifest) == MANIFEST, "asset manifest differs")
    entries = load(manifest)
    check(entries["repo"] == "google/flan-t5-small"
          and entries["revision"] == "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
          and set(entries["files"]) == ASSETS, "asset inventory differs")
    for name in sorted(ASSETS):
        item, path = entries["files"][name], args.assets / name
        check(type(item["bytes"]) is int and path.stat().st_size == item["bytes"]
              and sha(path) == item["sha256"], "asset bytes differ: " + name)
    report = {"status": "pass", "source_commit": COMMIT, "source_sha256": hashes,
              "asset_manifest_sha256": MANIFEST, "assets_checked": sorted(ASSETS),
              "terminal_sha256": TERMINAL, "events_sha256": EVENTS, "operator_sha256": OPERATOR,
              "operator_receipt_sha256": INNER, "outer_receipt_sha256": OUTER,
              "recorded_child_exit": 0, "recorded_operator_exit": 0,
              "scope": ("Recorded observations and saved-file consistency only. The outer record does not "
                        "independently authenticate or hash-bind the inner receipt; execution chronology, "
                        "whole-process-tree cleanup and numerical outcomes require separate evidence.")}
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if args.output:
        with args.output.open("xb") as stream:
            check(stream.write(payload) == len(payload), "short report write")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"status": "pass", "report_sha256": hashlib.sha256(payload).hexdigest()}))
    else:
        print(payload.decode(), end="")


if __name__ == "__main__":
    main()
