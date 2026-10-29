"""Independent saved-journal replay for the fixed 64-case FLAN sentiment screen.

It loads no checkpoint or study code. The frozen tokenizer and CPU float32
PyTorch kernel are dependencies; source/process/asset provenance is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct

import torch

_core_path = Path(__file__).with_name("audit_flan_sentiment_core.py")
_core_spec = importlib.util.spec_from_file_location("_independent_flan_sentiment_core", _core_path)
core = importlib.util.module_from_spec(_core_spec)
_core_spec.loader.exec_module(core)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)


def check(condition, description):
    if not condition:
        raise ValueError(description)


def same(left, right):
    """Do not treat booleans as integer IDs or silently coerce JSON types."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(same(x, y) for x, y in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and struct.pack('<d', left) == struct.pack('<d', right)
    return left == right


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


MANIFEST_SHA = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
TOKENIZER_ASSETS = ("config.json", "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json")


def verify_tokenizer_assets(asset_dir):
    """Verify tokenizer files; the model checkpoint is separately audited."""
    asset_dir = Path(asset_dir)
    path = asset_dir / "asset-manifest.json"
    check(digest(path) == MANIFEST_SHA, "tokenizer manifest SHA differs")
    manifest = strict_json(path.read_bytes())
    check(manifest.get("repo") == "google/flan-t5-small" and manifest.get("revision") == REVISION,
          "tokenizer manifest identity differs")
    files = manifest.get("files")
    check(type(files) is dict and set(files) == {"generation_config.json",
          "model.safetensors", *TOKENIZER_ASSETS}, "manifest asset inventory differs")
    for name in TOKENIZER_ASSETS:
        item, asset = files[name], asset_dir / name
        check(type(item) is dict and type(item.get("bytes")) is int
              and asset.stat().st_size == item["bytes"] and digest(asset) == item["sha256"],
              "tokenizer asset differs: " + name)
    return {name: files[name]["sha256"] for name in TOKENIZER_ASSETS}


def independent_alignment(cases, tokens, correct):
    positions = []
    for case, item in zip(cases, tokens):
        n = len(item["input_ids"])
        check(0 < n <= 64 and all(len(item[key]) == n for key in
              ("attention_mask", "special_tokens_mask", "offset_mapping")), "token lengths differ")
        begin = case["text"].rfind(case["adjective"])
        end = begin + len(case["adjective"])
        overlapping = [i for i, (a, b) in enumerate(item["offset_mapping"]) if a < end and b > begin]
        index = overlapping[0] if len(overlapping) == 1 else None
        if (index is not None and (item["offset_mapping"][index] != [begin, end]
                or item["attention_mask"][index] != 1 or item["special_tokens_mask"][index] != 0)):
            index = None
        positions.append(index)

    def aligned(i, j):
        a, b, pos = tokens[i], tokens[j], positions[i]
        return (pos is not None and positions[j] == pos and len(a["input_ids"]) == len(b["input_ids"])
                and a["attention_mask"] == b["attention_mask"]
                and a["special_tokens_mask"] == b["special_tokens_mask"]
                and [k for k, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y] == [pos])

    details = []
    for i, case in enumerate(cases):
        donors = [j for j, other in enumerate(cases)
                  if other["instruction_index"] == case["instruction_index"]
                  and other["subject"] == case["subject"] and other["side"] == case["side"]
                  and other["pair_index"] != case["pair_index"] and aligned(i, j)]
        details.append({"index": i, "word_position": positions[i], "opposite_aligned": aligned(i, i ^ 1),
                        "same_polarity_donors": donors,
                        "same_polarity_clean_donors": [j for j in donors if correct[j]]})
    return {"aligned_opposite_pairs": sum(details[i]["opposite_aligned"] for i in range(0, 64, 2)),
            "behaviorally_usable_pairs": sum(
                all(correct[j] and details[j]["same_polarity_clean_donors"] for j in (i, i + 1))
                and details[i]["opposite_aligned"] for i in range(0, 64, 2)),
            "pair_denominator": 32, "cases": details}


def audit_events(root, tokenizer, *, vocab_size, hidden_size, fake=False):
    root = Path(root)
    raw = (root / "events.jsonl").read_bytes()
    check(raw and raw.endswith(b"\n"), "event framing differs")
    events = [strict_json(line) for line in raw.splitlines()]
    check(all(type(e) is dict and type(e.get("event_index")) is int and e["event_index"] == i
              for i, e in enumerate(events)), "event index differs")
    fixed = [e for e in events if e.get("kind") == "fixed_inputs"]
    check(len(fixed) == 1, "fixed input inventory differs")
    cases = core.fixed_cases()
    check(same(fixed[0]["cases"], cases), "fixed case texts or order differ")
    rows = fixed[0]["token_rows"]
    check(type(rows) is list and len(rows) == 64, "input token inventory differs")
    for case, row in zip(cases, rows):
        actual = tokenizer(case["text"], return_offsets_mapping=True, return_special_tokens_mask=True)
        actual = {key: [list(x) if isinstance(x, tuple) else x for x in actual[key]] for key in
                  ("input_ids", "attention_mask", "special_tokens_mask", "offset_mapping")}
        check(same(row, actual) and all(type(x) is int and x == 1 for x in row["attention_mask"]),
              "retokenized input differs")
    targets = {label: [*tokenizer(label, add_special_tokens=False)["input_ids"], 1]
               for label in ("positive", "negative")}
    for value in targets.values():
        check(2 <= len(value) <= 5 and all(type(i) is int and 0 <= i < vocab_size
              and i not in tokenizer.all_special_ids for i in value[:-1]), "label target identity differs")
    check(targets["positive"] != targets["negative"] and same(fixed[0]["label_targets"], targets)
          and same(fixed[0]["label_masks"], {k: [1] * len(v) for k, v in targets.items()}),
          "label target inventory differs")
    at = events.index(fixed[0]) + 1
    check(([e.get("kind") for e in events[:at]] == ["fixed_inputs"] if fake else
           [e.get("kind") for e in events[:at]] == ["attempt_started", "run_contract", "fixed_inputs"]),
          "startup event sequence differs")
    forward_index = 0
    originals, comparisons, per_case = {}, [], []
    maximum_diagnostic = 0.

    def take_forward(index, repeat, phase, prefix, shape):
        nonlocal at, forward_index
        check(at + 1 < len(events), "forward start/completion missing")
        start, end = events[at], events[at + 1]
        expected = {"forward_index": forward_index, "case_index": index, "repeat": repeat,
                    "phase": phase, "prefix": prefix}
        for e, kind in ((start, "forward_started"), (end, "forward_completed")):
            check(e.get("kind") == kind and all(same(e.get(k), v) for k, v in expected.items()),
                  "forward identity/order differs")
        ref = end["tensor"]
        check(ref["path"] == f"tensors/{forward_index:04d}.f32", "tensor index differs")
        values = core.read_tensor(root, ref, shape=shape)
        at += 2
        forward_index += 1
        return ref, values

    for index, repeat in [(i, False) for i in range(64)] + [(0, True), (32, True)]:
        case, row = cases[index], rows[index]
        encoder_shape = (1, 1, hidden_size) if fake else (1, len(row["input_ids"]), hidden_size)
        encoder_ref, _ = take_forward(index, repeat, "encoder", [], encoder_shape)
        generated, prefixes, logits, generation_refs = [], [], [], []
        for _ in range(4):
            prefix = [0, *generated]
            ref, vector = take_forward(index, repeat, "greedy", prefix, (vocab_size,))
            emitted = max(range(vocab_size), key=vector.__getitem__)
            generated.append(emitted)
            prefixes.append(prefix)
            logits.append(vector)
            generation_refs.append(ref)
            if emitted == 1:
                break
        core.check_greedy(generated, prefixes, logits)
        label_refs, scores = {}, {}
        for label in ("positive", "negative"):
            target = targets[label]
            prefix = [0, *target[:-1]]
            ref, values = take_forward(index, repeat, "teacher_" + label, prefix, (len(target), vocab_size))
            matrix = [values[i * vocab_size:(i + 1) * vocab_size] for i in range(len(target))]
            tensor = torch.tensor(matrix, dtype=torch.float32)
            logprob = torch.log_softmax(tensor, dim=-1)
            selected = [float(logprob[i, tok].item()) for i, tok in enumerate(target)]
            ranks = [1 + sum(x > vector[tok] for x in vector) for vector, tok in zip(matrix, target)]
            score = math.fsum(selected)
            maximum_diagnostic = max(maximum_diagnostic,
                core.check_teacher_scores(matrix, target, selected, score))
            label_refs[label] = ref
            scores[label] = {"targets": target, "prefix": prefix, "tensor": ref,
                             "token_logprobs": selected, "token_ranks": ranks, "score": score}
        check(at < len(events), "case completion missing")
        completion = events[at]
        at += 1
        classification = core.strict_output(generated, tokenizer, case["expected"])
        expected = {"case_index": index, "repeat": repeat, "encoder_ref": encoder_ref,
                    "generated_ids": generated, "generation_prefixes": prefixes,
                    "generation_refs": generation_refs, "classification": classification,
                    "label_scores": scores, "label_margin": scores["positive"]["score"] - scores["negative"]["score"]}
        check(completion.get("kind") == "case_completed" and all(same(completion.get(k), v) for k, v in expected.items()),
              "saved case outcome differs")
        if not repeat:
            originals[index] = expected
            comparisons.append(classification["correct"])
            per_case.append({"case": case, "generated_ids": generated, "classification": classification,
                             "positive_score": scores["positive"]["score"],
                             "negative_score": scores["negative"]["score"],
                             "label_margin": expected["label_margin"]})
        else:
            original = originals[index]
            check(all(same(expected[k], original[k]) for k in ("generated_ids", "generation_prefixes", "classification", "label_margin")),
                  "repeat generation differs")
            old_refs = [original["encoder_ref"], *original["generation_refs"],
                        *(original["label_scores"][key]["tensor"] for key in ("positive", "negative"))]
            new_refs = [encoder_ref, *generation_refs, *(label_refs[key] for key in ("positive", "negative"))]
            check(len(old_refs) == len(new_refs) and all(
                (a["shape"], a["sha256"]) == (b["shape"], b["sha256"]) for a, b in zip(old_refs, new_refs)),
                "repeat raw tensors differ")
    summary = core.feasibility(comparisons)
    alignment = independent_alignment(cases, rows, comparisons)
    check(at == len(events) - 1 and events[at].get("kind") == "summary", "terminal event differs")
    observed = {"summary": summary, "alignment": alignment, "repeat_indices": [0, 32], "forwards": forward_index}
    check(all(same(events[at].get(key), value) for key, value in observed.items()), "saved summary differs")
    check(forward_index <= 462, "forward bound exceeded")
    return {"status": "pass", "events_sha256": hashlib.sha256(raw).hexdigest(),
            "forwards": forward_index, "summary": summary, "alignment": alignment,
            "cases": per_case,
            "max_float64_vs_recorded_f32_logprob_abs": maximum_diagnostic,
            "scope": "Saved data only; no source, process execution, hidden-input or pretraining attestation."}


def audit_closed(root, tokenizer, *, expected_commit, expected_manifest):
    """Bind a completed owner terminal to the fully rechecked journal."""
    root = Path(root)
    check(not (root / "failed.json").exists(), "failure marker is present")
    terminal = strict_json((root / "terminal.json").read_bytes())
    check(terminal.get("schema") == "mlg-flant5-sentiment-screen-v1"
          and terminal.get("status") == "complete" and terminal.get("source", {}).get("commit") == expected_commit
          and terminal.get("asset_manifest_sha256") == expected_manifest, "terminal identity differs")
    events_path = root / "events.jsonl"
    check(terminal.get("events_sha256") == digest(events_path), "terminal journal hash differs")
    report = audit_events(root, tokenizer, vocab_size=32128, hidden_size=512)
    saved = terminal.get("result")
    expected = {"summary": report["summary"], "alignment": report["alignment"],
                "repeat_indices": [0, 32], "forwards": report["forwards"]}
    check(same(saved, expected), "terminal result differs from replay")
    events = [strict_json(line) for line in events_path.read_bytes().splitlines()]
    check(same(events[1].get("source"), terminal.get("source"))
          and events[1].get("asset_manifest_sha256") == expected_manifest
          and events[1].get("checkpoint_revision") == REVISION,
          "recorded source or asset identities differ")
    tensors = root / "tensors"
    expected_names = {f"{i:04d}.f32" for i in range(report["forwards"])}
    check(not tensors.is_symlink() and {p.name for p in tensors.iterdir()} == expected_names
          and all(p.is_file() and not p.is_symlink() for p in tensors.iterdir()), "tensor inventory differs")
    total = events_path.stat().st_size + sum((tensors / name).stat().st_size for name in expected_names)
    check(type(terminal.get("retained_payload_and_journal_bytes")) is int
          and terminal["retained_payload_and_journal_bytes"] == total and total <= 128 * 1024 * 1024,
          "retained byte accounting differs")
    report["result_sha256"] = digest(root / "terminal.json")
    report["source_commit"] = expected_commit
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    check(re.fullmatch("[0-9a-f]{40}", args.source_commit) is not None, "invalid expected source commit")
    check(os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "hide CUDA for CPU replay")
    torch.set_num_threads(1)
    check(torch.set_flush_denormal(False), "cannot restore denormal preservation")
    core.require_denormals()
    asset_hashes = verify_tokenizer_assets(args.asset_dir)
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.asset_dir), local_files_only=True,
                                               use_fast=True, trust_remote_code=False)
    check(type(tokenizer).__name__ == "T5TokenizerFast" and tokenizer.pad_token_id == 0
          and tokenizer.eos_token_id == 1, "tokenizer class or special IDs differ")
    report = audit_closed(args.run_dir, tokenizer, expected_commit=args.source_commit,
                          expected_manifest=MANIFEST_SHA)
    report["tokenizer_asset_sha256"] = asset_hashes
    report["auditor_runtime"] = {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
                                  "tokenizers": str(tokenizers.__version__), "threads": torch.get_num_threads()}
    report["numeric_scope"] = ("Float32 log-softmax is replayed using the local CPU PyTorch kernel; "
                               "the binary64 logsumexp diagnostic is separately computed. "
                               "Model weights, source and process execution require separate custody review.")
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        check(stream.write(payload) == len(payload), "short report write")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "forwards": report["forwards"],
                      "report_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
