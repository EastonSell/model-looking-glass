"""Check saved T5 token scores independently with stdlib float64 log-sum-exp."""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

TERMINAL = "57cbbc319fe4dcc84f3f3a4c50a5bceb633cf653886e8fe16d379cccd9f09afc"
MANIFEST = "a62822eb557657d9220cd4e35a34662b1d726657afe6a1968e3a5121b0004e8a"


def main(root, manifest_path):
    manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST:
        raise ValueError("published manifest digest differs")
    entries = {item["path"]: item for item in json.loads(manifest_raw)["files"]}
    def checked(name):
        raw = (root / name).read_bytes()
        expected = entries["raw/" + name]
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("published file differs from manifest")
        return raw
    terminal_raw = checked("completed.json")
    if hashlib.sha256(terminal_raw).hexdigest() != TERMINAL:
        raise ValueError("terminal digest differs")
    complete = json.loads(terminal_raw)
    by_name = {}
    worst = 0.0
    for i in range(complete["forwards"]):
        record = json.loads(checked(f"forward-{i:03d}.json"))
        if record["index"] != i or record["file"] != f"forward-{i:03d}.bin":
            raise ValueError("forward identity differs")
        raw = checked(record["file"])
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("raw forward hash differs")
        count = len(record["target_ids"])
        if (not 1 <= count <= 8 or record["target_ids"][-1] != 1
                or record["decoder_ids"] != [0] + record["target_ids"][:-1]
                or record["decoder_mask"] != [1] * count
                or any(not 0 <= t < 32128 for t in record["target_ids"])
                or record["logits_shape"] != [1, count, 32128] or record["logits_bytes"] != count * 32128 * 4):
            raise ValueError("logit shape differs")
        selected = []
        for position, token in enumerate(record["target_ids"]):
            start = position * 32128 * 4
            logits = struct.unpack_from("<32128f", raw, start)
            if not all(math.isfinite(x) for x in logits):
                raise ValueError("nonfinite saved logit")
            peak = max(logits)
            normalizer = peak + math.log(math.fsum(math.exp(x - peak) for x in logits))
            value = logits[token] - normalizer
            selected.append(value)
            difference = abs(value - record["selected_logprobs"][position])
            worst = max(worst, difference)
            if difference > 1e-4:
                raise ValueError("float64 and recorded float32 log-probabilities disagree")
        score = math.fsum(selected)
        if abs(score - record["score"]) > count * 1e-4:
            raise ValueError("float64 and recorded float32 scores disagree")
        if record["name"] in by_name:
            raise ValueError("duplicate forward name")
        by_name[record["name"]] = score
    def margin(pair, condition, side):
        return by_name[f"pair{pair}-{condition}-{side}-0"] - by_name[f"pair{pair}-{condition}-{side}-1"]
    outcomes = []
    for pair, expected in enumerate(complete["pairs"]):
        margins = [margin(pair, "base", side) for side in (0, 1)]
        supported = margins[0] > 0 and margins[1] < 0 and margins[0] - margins[1] >= 1
        if supported != expected["baseline_supported"]:
            raise ValueError("baseline decision differs")
        row = {"pair_index": pair, "baseline_margins_float64": margins, "baseline_supported": supported}
        if supported:
            delta = margins[0] - margins[1]
            word = [margins[0] - margin(pair, "word_swap", 0), margin(pair, "word_swap", 1) - margins[1]]
            control = [margins[0] - margin(pair, "instruction_swap", 0),
                       margin(pair, "instruction_swap", 1) - margins[1]]
            positive = all(word[j] >= 0.2 * delta and word[j] >= control[j] + 0.05 * delta for j in (0, 1))
            if positive != (expected["status"] == "positive"):
                raise ValueError("decision differs")
            row.update(word_movements_float64=word, instruction_movements_float64=control,
                       positive=positive)
        outcomes.append(row)
    result = {"schema": "mlg-t5-stdlib-score-check-v1", "status": "passed", "terminal_sha256": TERMINAL,
              "published_manifest_sha256": MANIFEST,
              "records": len(by_name), "absolute_tolerance_per_token": 1e-4,
              "max_abs_float64_vs_saved_f32_token_score": worst,
              "pairs": outcomes,
              "scope": "saved logits and numeric decisions only; no model execution or private data"}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python t5_high_precision_score_check.py RAW_DIRECTORY MANIFEST_JSON")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
