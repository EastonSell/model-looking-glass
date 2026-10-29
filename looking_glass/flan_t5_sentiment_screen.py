"""Fixed exploratory FLAN-T5 sentiment screen."""

from __future__ import annotations

import math
import hashlib
import argparse
import json
import os
import platform
import subprocess
import sys
import unicodedata
from pathlib import Path

import torch

INSTRUCTIONS = (
    "Classify the sentiment of this review as positive or negative:",
    "Is this review positive or negative?",
)
SUBJECTS = ("book", "movie", "meal", "service")
ADJECTIVES = (("wonderful", "awful"), ("excellent", "terrible"),
              ("amazing", "horrible"), ("enjoyable", "disappointing"))


def build_cases():
    cases = []
    for instruction_index, instruction in enumerate(INSTRUCTIONS):
        for subject in SUBJECTS:
            for pair_index, pair in enumerate(ADJECTIVES):
                for side, adjective in enumerate(pair):
                    cases.append({"index": len(cases), "instruction_index": instruction_index,
                                  "subject": subject, "pair_index": pair_index,
                                  "adjective": adjective, "side": side,
                                  "expected": ("positive", "negative")[side],
                                  "text": f"{instruction} The {subject} was {adjective}."})
    return cases


def greedy(forward, *, vocab_size):
    prefix, steps = [0], []
    for _ in range(4):
        logits = forward(tuple(prefix))
        if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
                or logits.dtype != torch.float32 or tuple(logits.shape) != (vocab_size,)):
            raise ValueError("generation logits have unexpected type or shape")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("generation logits must be finite")
        logits = logits.detach().contiguous().clone()
        token = int(torch.argmax(logits).item())
        steps.append({"prefix": tuple(prefix), "logits": logits, "emitted": token})
        prefix.append(token)
        if token == 1:
            break
    return steps


def classify_generation(ids, tokenizer, expected):
    if expected not in ("positive", "negative"):
        raise ValueError("unknown expected label")
    terminated = bool(ids) and ids[-1] == 1
    content = ids[:-1] if terminated else ids
    decoded = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    normalized = unicodedata.normalize("NFC", decoded).strip().casefold()
    no_content_specials = not any(value in tokenizer.all_special_ids for value in content)
    clean = bool(ids) and len(ids) <= 4 and terminated and no_content_specials
    return {"decoded": decoded, "normalized": normalized, "terminated": terminated,
            "no_content_specials": no_content_specials, "correct": clean and normalized == expected}


def label_targets(tokenizer):
    lexical = {label: tokenizer(label, add_special_tokens=False)["input_ids"]
               for label in ("positive", "negative")}
    if (any(not values or any(type(value) is not int or value in tokenizer.all_special_ids
                              for value in values) for values in lexical.values())
            or lexical["positive"] == lexical["negative"]):
        raise ValueError("label lexical token sequences must be nonempty, distinct and non-special")
    return {label: [*values, 1] for label, values in lexical.items()}


def teacher_force(forward, targets, *, vocab_size):
    if not targets or targets[-1] != 1:
        raise ValueError("target must end in EOS")
    prefix = (0, *targets[:-1])
    logits = forward(prefix)
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu"
            or logits.dtype != torch.float32 or tuple(logits.shape) != (len(targets), vocab_size)):
        raise ValueError("teacher-forced logits have unexpected type or shape")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("teacher-forced logits must be finite")
    logits = logits.detach().contiguous().clone()
    all_logprobs = torch.log_softmax(logits, dim=-1)
    values = [float(all_logprobs[index, token].item()) for index, token in enumerate(targets)]
    ranks = [1 + int(torch.count_nonzero(logits[index] > logits[index, token]).item())
             for index, token in enumerate(targets)]
    return {"prefix": prefix, "logits": logits, "token_logprobs": values,
            "token_ranks": ranks, "score": math.fsum(values)}


def summarize(rows):
    if len(rows) != 64 or any(type(row.get("index")) is not int or type(row.get("correct")) is not bool
                              for row in rows) or [row["index"] for row in rows] != list(range(64)):
        raise ValueError("screen must contain the 64 ordered outcomes")
    cases = build_cases()
    complete_pairs = 0
    instruction_pairs = {str(index): 0 for index in range(2)}
    subject_pairs = {name: 0 for name in SUBJECTS}
    adjective_pairs = {str(index): 0 for index in range(4)}
    for index in range(0, 64, 2):
        if rows[index]["correct"] and rows[index + 1]["correct"]:
            case = cases[index]
            complete_pairs += 1
            instruction_pairs[str(case["instruction_index"])] += 1
            subject_pairs[case["subject"]] += 1
            adjective_pairs[str(case["pair_index"])] += 1
    labels = {label: sum(row["correct"] for row, case in zip(rows, cases) if case["expected"] == label)
              for label in ("positive", "negative")}
    feasible = (complete_pairs >= 24 and all(value >= 10 for value in instruction_pairs.values())
                and all(value >= 4 for value in subject_pairs.values())
                and all(value >= 4 for value in adjective_pairs.values()))
    return {"cases": 64, "complete_pairs": complete_pairs, "pair_denominator": 32,
            "instruction_pairs": instruction_pairs, "subject_pairs": subject_pairs,
            "adjective_pairs": adjective_pairs, "correct_by_label": labels,
            "label_denominator": 32, "feasible": feasible}


def alignment_report(cases, token_rows, outcomes):
    if cases != build_cases() or len(token_rows) != 64 or len(outcomes) != 64:
        raise ValueError("alignment requires the fixed 64 cases")
    if [row.get("index") for row in outcomes] != list(range(64)):
        raise ValueError("outcomes are not in fixed case order")
    positions = []
    for case, tokens in zip(cases, token_rows):
        n = len(tokens["input_ids"])
        if (n == 0 or n > 64 or any(len(tokens[key]) != n for key in
                                    ("attention_mask", "special_tokens_mask", "offset_mapping"))):
            raise ValueError("malformed tokenizer record")
        start = case["text"].rfind(case["adjective"])
        end = start + len(case["adjective"])
        overlapping = [i for i, (a, b) in enumerate(tokens["offset_mapping"])
                       if a < end and b > start]
        pos = (overlapping[0] if len(overlapping) == 1
               and tokens["offset_mapping"][overlapping[0]] == [start, end]
               and tokens["special_tokens_mask"][overlapping[0]] == 0
               and tokens["attention_mask"][overlapping[0]] == 1 else None)
        positions.append(pos)

    def aligned(i, j):
        a, b, pos = token_rows[i], token_rows[j], positions[i]
        return (pos is not None and pos == positions[j]
                and len(a["input_ids"]) == len(b["input_ids"])
                and a["attention_mask"] == b["attention_mask"]
                and a["special_tokens_mask"] == b["special_tokens_mask"]
                and [k for k, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y] == [pos])

    details = []
    for i, case in enumerate(cases):
        candidates = [j for j, other in enumerate(cases)
                      if other["instruction_index"] == case["instruction_index"]
                      and other["subject"] == case["subject"] and other["side"] == case["side"]
                      and other["pair_index"] != case["pair_index"] and aligned(i, j)]
        details.append({"index": i, "word_position": positions[i], "opposite_aligned": aligned(i, i ^ 1),
                        "same_polarity_donors": candidates,
                        "same_polarity_clean_donors": [j for j in candidates if outcomes[j]["correct"]]})
    aligned_pairs = sum(details[i]["opposite_aligned"] for i in range(0, 64, 2))
    usable = sum(all(outcomes[j]["correct"] and details[j]["same_polarity_clean_donors"]
                     for j in (i, i + 1)) and details[i]["opposite_aligned"]
                 for i in range(0, 64, 2))
    return {"aligned_opposite_pairs": aligned_pairs, "behaviorally_usable_pairs": usable,
            "pair_denominator": 32, "cases": details}


class Journal:
    """Append-only metadata and separately hashed, bounded raw float32 payloads."""

    def __init__(self, directory, *, byte_cap, forward_cap):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "tensors").mkdir()
        self.byte_cap = byte_cap
        self.forward_cap = forward_cap
        self.used = 0
        self.forwards = 0
        self.event_index = 0

    def event(self, kind, **fields):
        event = {"event_index": self.event_index, "kind": kind, **fields}
        data = (json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if self.used + len(data) + 65536 > self.byte_cap:
            raise ValueError("journal capacity exceeded")
        with (self.root / "events.jsonl").open("ab") as stream:
            if stream.write(data) != len(data):
                raise OSError("short journal write")
            stream.flush()
            os.fsync(stream.fileno())
        self.used += len(data)
        self.event_index += 1

    def forward(self, case_index, repeat, phase, prefix, function):
        if self.forwards >= self.forward_cap:
            raise ValueError("forward cap exceeded")
        number = self.forwards
        self.forwards += 1
        self.event("forward_started", forward_index=number, case_index=case_index, repeat=repeat,
                   phase=phase, prefix=list(prefix))
        tensor = function()
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32 or tensor.numel() == 0
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("forward tensor must be finite CPU float32")
        tensor = tensor.detach().contiguous()
        data = tensor.numpy().astype("<f4", copy=False).tobytes()
        if self.used + len(data) + 65536 > self.byte_cap:
            raise ValueError("tensor capacity exceeded")
        rel = f"tensors/{number:04d}.f32"
        with (self.root / rel).open("xb") as stream:
            if stream.write(data) != len(data):
                raise OSError("short tensor write")
            stream.flush()
            os.fsync(stream.fileno())
        self.used += len(data)
        ref = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
               "dtype": "<f4", "shape": list(tensor.shape)}
        self.event("forward_completed", forward_index=number, case_index=case_index, repeat=repeat,
                   phase=phase, prefix=list(prefix), tensor=ref)
        return tensor, ref


def execute_case(case, tokenizer, targets, encode, decode, journal, *, vocab_size, repeat):
    encoded, encoder_ref = journal.forward(case["index"], repeat, "encoder", (), encode)
    generation_refs = []
    def generation_call(prefix):
        logits, ref = journal.forward(case["index"], repeat, "greedy", prefix,
                                      lambda: decode(encoded, prefix)[-1])
        generation_refs.append(ref)
        return logits
    steps = greedy(generation_call, vocab_size=vocab_size)
    ids = [step["emitted"] for step in steps]
    result = {"case_index": case["index"], "repeat": repeat, "encoder_ref": encoder_ref,
              "generated_ids": ids, "generation_prefixes": [list(step["prefix"]) for step in steps],
              "generation_refs": generation_refs,
              "classification": classify_generation(ids, tokenizer, case["expected"]),
              "label_scores": {}}
    for label in ("positive", "negative"):
        refs = []
        def score_call(prefix):
            logits, ref = journal.forward(case["index"], repeat, "teacher_" + label, prefix,
                                          lambda: decode(encoded, prefix))
            refs.append(ref)
            return logits
        score = teacher_force(score_call, targets[label], vocab_size=vocab_size)
        result["label_scores"][label] = {"targets": targets[label], "prefix": list(score["prefix"]),
                                         "tensor": refs[0], "token_logprobs": score["token_logprobs"],
                                         "token_ranks": score["token_ranks"], "score": score["score"]}
    result["label_margin"] = (result["label_scores"]["positive"]["score"]
                              - result["label_scores"]["negative"]["score"])
    journal.event("case_completed", **result)
    return result


def run_study(tokenizer, encode, decode, journal, *, vocab_size):
    """Run the fixed schedule using an injected encoder and decoder; no model is loaded here."""
    cases, targets = build_cases(), label_targets(tokenizer)
    if any(len(value) > 5 or any(not 0 <= token < vocab_size for token in value)
           for value in targets.values()):
        raise ValueError("target token sequence is outside the fixed bounds")
    token_rows = []
    for case in cases:
        row = tokenizer(case["text"], return_offsets_mapping=True, return_special_tokens_mask=True)
        row = {key: [list(x) if isinstance(x, tuple) else x for x in row[key]] for key in
               ("input_ids", "attention_mask", "special_tokens_mask", "offset_mapping")}
        if (not 0 < len(row["input_ids"]) <= 64 or row["input_ids"][-1] != 1
                or any(value != 1 for value in row["attention_mask"])):
            raise ValueError("input tokenization is outside the fixed bounds")
        token_rows.append(row)
    journal.event("fixed_inputs", cases=cases, token_rows=token_rows, label_targets=targets,
                  label_masks={label: [1] * len(ids) for label, ids in targets.items()})
    results = []
    for case, tokens in zip(cases, token_rows):
        result = execute_case(case, tokenizer, targets, lambda: encode(case, tokens), decode, journal,
                              vocab_size=vocab_size, repeat=False)
        results.append(result)
    for index in (0, 32):
        case, tokens, original = cases[index], token_rows[index], results[index]
        repeated = execute_case(case, tokenizer, targets, lambda: encode(case, tokens), decode, journal,
                                vocab_size=vocab_size, repeat=True)
        if (original["generated_ids"] != repeated["generated_ids"]
                or original["generation_prefixes"] != repeated["generation_prefixes"]
                or original["classification"] != repeated["classification"]):
            raise ValueError("repeat generation differs")
        original_refs = [original["encoder_ref"], *original["generation_refs"]]
        repeated_refs = [repeated["encoder_ref"], *repeated["generation_refs"]]
        for label in ("positive", "negative"):
            a, b = original["label_scores"][label], repeated["label_scores"][label]
            if any(a[key] != b[key] for key in ("targets", "prefix", "token_logprobs", "token_ranks", "score")):
                raise ValueError("repeat teacher-forced score differs")
            original_refs.append(a["tensor"])
            repeated_refs.append(b["tensor"])
        if len(original_refs) != len(repeated_refs) or any(
            x["shape"] != y["shape"] or (journal.root / x["path"]).read_bytes() != (journal.root / y["path"]).read_bytes()
            for x, y in zip(original_refs, repeated_refs)
        ):
            raise ValueError("repeat tensor bytes differ")
    rows = [{"index": case["index"], "correct": result["classification"]["correct"]}
            for case, result in zip(cases, results)]
    result = {"summary": summarize(rows), "alignment": alignment_report(cases, token_rows, rows),
              "repeat_indices": [0, 32], "forwards": journal.forwards}
    journal.event("summary", **result)
    return result


def decode_with_encoder(model, encoded, prefix):
    from transformers.modeling_outputs import BaseModelOutput
    mask = torch.ones((1, encoded.shape[1]), dtype=torch.long)
    decoder_mask = torch.ones((1, len(prefix)), dtype=torch.long)
    return model(encoder_outputs=BaseModelOutput(last_hidden_state=encoded), attention_mask=mask,
                 decoder_input_ids=torch.tensor([prefix], dtype=torch.long),
                 decoder_attention_mask=decoder_mask, use_cache=False, return_dict=True).logits[0]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root):
        raise ValueError("source checkout must be clean")
    names = ("looking_glass/flan_t5_sentiment_screen.py", "looking_glass/flan_t5_feasibility.py",
             "tests/test_flan_t5_sentiment_screen.py", "docs/FLAN_T5_SENTIMENT_SCREEN_V1.md")
    hashes = {}
    for name in names:
        data = (root / name).read_bytes()
        if data != subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root):
            raise ValueError("source file differs from committed bytes")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"commit": commit, "source_sha256": hashes}


def _write_json(path, payload):
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("short JSON write")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    journal = Journal(args.output, byte_cap=128 * 1024 * 1024, forward_cap=462)
    phase = "preflight"
    try:
        journal.event("attempt_started", schema="mlg-flant5-sentiment-screen-v1")
        required_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE"}
        if (os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
                or any(os.environ.get(k) != v for k, v in required_env.items())):
            raise ValueError("fixed CPU/offline numeric environment is required")
        from .flan_t5_feasibility import MANIFEST_HASH, REVISION, verify_assets
        source = _source_identity(root)
        manifest = verify_assets(args.asset_dir)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(20261028)
        if not torch.set_flush_denormal(False):
            raise ValueError("cannot configure float32 denormals")
        subnormal_bits = int((torch.tensor([2.0 ** -149], dtype=torch.float32) * 1.0).view(torch.int32)[0])
        if subnormal_bits != 1:
            raise ValueError("float32 denormal behavior differs")
        import numpy
        import tokenizers
        import transformers
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(str(args.asset_dir), local_files_only=True,
                                                  use_fast=True, trust_remote_code=False)
        if (type(tokenizer).__name__ != "T5TokenizerFast" or tokenizer.pad_token_id != 0
                or tokenizer.eos_token_id != 1):
            raise ValueError("tokenizer identity differs")
        phase = "model_load"
        model, info = T5ForConditionalGeneration.from_pretrained(
            str(args.asset_dir), local_files_only=True, use_safetensors=True, trust_remote_code=False,
            output_loading_info=True, dtype=torch.float32, attn_implementation="eager")
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("model loading diagnostics are not empty")
        model.eval()
        model.config.use_cache = False
        config = model.config
        if (type(model).__name__ != "T5ForConditionalGeneration" or config.d_model != 512
                or config.vocab_size != 32128 or config.decoder_start_token_id != 0
                or config.pad_token_id != 0 or config.eos_token_id != 1
                or getattr(config, "_attn_implementation", None) != "eager"
                or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
                or any(b.device.type != "cpu" or (b.is_floating_point() and b.dtype != torch.float32)
                       for b in model.buffers())):
            raise ValueError("model runtime differs from fixed configuration")
        journal.event("run_contract", source=source, checkpoint_revision=REVISION,
                      asset_manifest_sha256=MANIFEST_HASH, device="cpu", dtype="float32",
                      python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
                      transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                      threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                      deterministic=torch.are_deterministic_algorithms_enabled(), subnormal_bits=subnormal_bits,
                      deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                      float32_matmul_precision=torch.get_float32_matmul_precision(),
                      attention="eager", decoder_cache=False, seed=20261028,
                      environment={"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), **required_env})
        def encode(_case, tokens):
            ids = torch.tensor([tokens["input_ids"]], dtype=torch.long)
            mask = torch.tensor([tokens["attention_mask"]], dtype=torch.long)
            return model.encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        def decode(encoded, prefix):
            # Every input was unpadded and fully attended at tokenization.
            return decode_with_encoder(model, encoded, prefix)
        phase = "forwards"
        with torch.inference_mode():
            result = run_study(tokenizer, encode, decode, journal, vocab_size=config.vocab_size)
        phase = "completion_checks"
        if _source_identity(root) != source or verify_assets(args.asset_dir) != manifest:
            raise ValueError("source or asset identity changed")
        terminal = {"schema": "mlg-flant5-sentiment-screen-v1", "status": "complete",
                    "source": source, "asset_manifest_sha256": MANIFEST_HASH,
                    "events_sha256": _sha(journal.root / "events.jsonl"), "result": result,
                    "retained_payload_and_journal_bytes": journal.used}
        _write_json(journal.root / "terminal.json", terminal)
        print(json.dumps({"status": "complete", "forwards": journal.forwards,
                          "feasible": result["summary"]["feasible"]}, sort_keys=True))
    except BaseException as exc:
        try:
            _write_json(journal.root / "failed.json", {"status": "incomplete", "phase": phase,
                        "error_type": type(exc).__name__, "error": str(exc),
                        "forwards_started": journal.forwards})
        finally:
            raise


if __name__ == "__main__":
    main()
