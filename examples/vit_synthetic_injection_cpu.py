"""Run the fixed synthetic ViT known-injection study with local cached assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch
import transformers
from transformers import ViTForImageClassification

from looking_glass.vision_coco_cases import MODEL_FILES, MODEL_REPO, MODEL_REVISION, verify_assets
from looking_glass.vit_synthetic_injection_v1 import (
    INJECTIONS, SEEDS, TOKENS, _assert_no_hooks, _distance, _norm, _raw, _same, choose,
    direction_bytes, fit_direction, fixed_random_choices, intervention_arms, paired_case, synthetic_pixels,
)
from looking_glass.vit_trace import ViTTracePoint, trace_vit_classifier

ROOT = Path(__file__).resolve().parents[1]
CAP = 32 * 1024 * 1024
FREE = 2 * 1024**3
SOURCE_FILES = (
    "docs/VIT_SYNTHETIC_INJECTION_V1.md", "examples/vit_synthetic_injection_bootstrap.py",
    "examples/vit_synthetic_injection_cpu.py",
    "looking_glass/__init__.py", "looking_glass/vit_synthetic_injection_v1.py",
    "looking_glass/vit_trace.py", "looking_glass/vit_trace_comparison.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py", "looking_glass/model_ownership.py",
    "looking_glass/localization.py", "looking_glass/trace_schema.py", "looking_glass/agreement_numeric.py",
    "looking_glass/bert_matched_numeric.py", "looking_glass/vision_coco_cases.py",
    "tests/test_vit_synthetic_injection_v1.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def state_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        data = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
        result.update(json_bytes([name, str(value.dtype), list(value.shape), len(data)]))
        result.update(data)
    return result.hexdigest()


def verify_source(commit, git):
    """Check exact local source bytes against the operator's public commit pin."""
    hashes = {}
    git_env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    if subprocess.check_output([str(git), "--no-replace-objects", "cat-file", "-t", commit],
                               cwd=ROOT, env=git_env).strip() != b"commit":
        raise ValueError("selected source is not a commit object")
    for name in SOURCE_FILES:
        local = (ROOT / name).read_bytes()
        committed = subprocess.check_output([str(git), "--no-replace-objects", "show", f"{commit}:{name}"],
                                            cwd=ROOT, env=git_env)
        if local != committed:
            raise ValueError(f"local source differs from the selected commit: {name}")
        hashes[name] = digest(local)
    for name in SOURCE_FILES:
        if not name.startswith("looking_glass/") or not name.endswith(".py"):
            continue
        module_name = "looking_glass" if name.endswith("/__init__.py") else name[:-3].replace("/", ".")
        module = sys.modules.get(module_name)
        if module is not None and Path(module.__file__).resolve() != (ROOT / name).resolve():
            raise ValueError(f"loaded module does not come from the source checkout: {module_name}")
    return hashes


def verify_loading_info(loading):
    names = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if type(loading) is not dict or any(type(loading.get(name)) is not list or loading[name] for name in names):
        raise RuntimeError("checkpoint loading reported missing, unexpected, mismatched or error entries")
    return {name: loading[name] for name in names}


def checked_write(stream, raw):
    if stream.write(raw) != len(raw):
        raise OSError("short evidence write")


class Evidence:
    def __init__(self, root):
        self.root = root
        self.total = 0
        self.tensor_bytes = 0
        self.entries = []
        self.stream = (root / "tensors.bin").open("xb")

    def check(self, size):
        if size < 0 or self.total + size > CAP - 2 * 1024 * 1024:
            raise RuntimeError("retained tensor budget exceeded")

    def check_free(self):
        if min(shutil.disk_usage(self.root).free, shutil.disk_usage(ROOT).free) < FREE:
            raise RuntimeError("output or source volume free-space reserve is below 2 GiB")

    def tensor(self, name, value):
        raw = _raw(value)
        self.check(len(raw))
        entry = {"name": name, "offset": self.tensor_bytes, "length": len(raw),
                 "shape": list(value.shape), "dtype": "float32-le", "sha256": digest(raw)}
        checked_write(self.stream, raw)
        self.stream.flush()
        self.total += len(raw)
        self.tensor_bytes += len(raw)
        self.entries.append(entry)
        return entry

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.write("tensor-index.json", {"schema": "vit-synthetic-tensors-v1", "entries": self.entries,
                                         "tensor_bytes": self.tensor_bytes})
        path = self.root / "tensors.bin"
        if path.stat().st_size != self.tensor_bytes:
            raise OSError("tensor evidence length differs on readback")
        with path.open("rb") as stream:
            for entry in self.entries:
                stream.seek(entry["offset"])
                raw = stream.read(entry["length"])
                if len(raw) != entry["length"] or digest(raw) != entry["sha256"]:
                    raise OSError("tensor evidence differs on readback")

    def write(self, name, value):
        raw = value if type(value) is bytes else json_bytes(value)
        if self.total + len(raw) > CAP:
            raise RuntimeError("retained evidence limit exceeded")
        with (self.root / name).open("xb") as stream:
            checked_write(stream, raw)
            stream.flush()
            os.fsync(stream.fileno())
        if (self.root / name).read_bytes() != raw:
            raise OSError("metadata evidence differs on readback")
        self.total += len(raw)
        return digest(raw)


def save_forward(sink, prefix, forward):
    refs = {name: sink.tensor(prefix + "/" + name, value)
            for name, value in forward["captures"].items()}
    refs["logits"] = sink.tensor(prefix + "/logits", forward["logits"])
    return {"tensors": refs, "plan": forward["metadata"], "injection_l2": forward["injection_l2"]}


def save_pair(sink, prefix, pair, pixels):
    clean = pair["clean"]
    return {"pixel_sha256": digest(_raw(pixels)), "geometry": clean.image_metadata,
            "clean_events": [r.event.to_dict() for r in clean.records],
            "clean_vectors": [sink.tensor(prefix + "/clean/" + str(r.event.token_positions[0]), r.value)
                              for r in clean.records],
            "clean_logits": sink.tensor(prefix + "/clean/logits", clean.output.logits),
            "injected": save_forward(sink, prefix + "/injected", pair["injected"]),
            "comparison": pair["comparison"]}


def run(model, sink, source, *, _fabricated_policy=None):
    if _fabricated_policy is None:
        size, width, tokens, seeds, injections, randoms = 224, 192, TOKENS, SEEDS, INJECTIONS, fixed_random_choices()
    else:
        size, width, tokens, seeds, injections, randoms = _fabricated_policy
    if (model.config.image_size != size or model.config.patch_size != 16 or model.config.hidden_size != width
            or model.config.num_channels != 3):
        raise ValueError("checkpoint geometry differs from the frozen protocol")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("CPU float32 model is required")
    _assert_no_hooks(model)
    state_before = state_digest(model)
    pairs, rows = [], []
    for index in range(4):
        sink.check_free()
        pixels = synthetic_pixels(seeds[index], size)
        pair = paired_case(model, pixels, tokens, injections[index], f"discovery-{index}")
        rows.append({"seed": seeds[index], "injection_token": injections[index],
                     "evidence": save_pair(sink, f"discovery-{index}", pair, pixels)})
        sink.write(f"discovery-{index}.json", rows[-1])
        pairs.append(pair)
        print(f"discovery case {index + 1}/4 complete", flush=True)
    direction = fit_direction(pairs, injections[:4])
    raw_direction = direction_bytes(direction)
    sink.stream.flush()
    os.fsync(sink.stream.fileno())
    direction_sha = sink.write("direction.f64le", raw_direction)
    discovery_scores = [choose(direction, pair["deltas"]) for pair in pairs]
    fit = {"schema": "vit-synthetic-fit-v1", "source": source, "direction_sha256": direction_sha,
           "direction_length": len(direction), "direction_l2": _norm(direction),
           "discovery_cases": rows, "discovery_scores": discovery_scores,
           "random_choices": list(randoms), "candidate_tokens": list(tokens),
           "tie_break": "lowest token index", "state_sha256": state_before}
    fit_sha = sink.write("discovery-freeze.json", fit)
    if (digest((sink.root / "direction.f64le").read_bytes()) != direction_sha
            or digest((sink.root / "discovery-freeze.json").read_bytes()) != fit_sha):
        raise RuntimeError("discovery freeze readback differs")
    del pairs
    heldout = []
    for index in range(4):
        sink.check_free()
        token, seed = injections[index+4], seeds[index+4]
        pixels = synthetic_pixels(seed, size)
        pair = paired_case(model, pixels, tokens, token, f"heldout-{index}")
        picked = choose(direction, pair["deltas"])
        selected = picked["directional"] if _norm(direction) != 0 else None
        row = {"seed": seed, "injection_token": token, "fitted_record_sha256": fit_sha,
               "selection": picked, "random": randoms[index],
               "selection_status": "selected" if selected is not None else "ineligible_zero_direction",
               "evidence": save_pair(sink, f"heldout-{index}", pair, pixels)}
        sink.write(f"heldout-{index}-selection.json", row)
        if selected is not None:
            control = min(x for x in tokens if x != selected and x != token)
            saved = {}
            def retain_arm(name, arm):
                saved[name] = {key: value for key, value in arm.items() if key != "forward"}
                saved[name]["evidence"] = save_forward(sink, f"heldout-{index}/{name}", arm["forward"])
                sink.write(f"heldout-{index}-{name}.json", saved[name])
            outcomes = intervention_arms(model, pixels, pair, token, selected, control, on_arm=retain_arm)
            eligible = (outcomes["target_dose"] > 0 and outcomes["injected_logit_distance_from_clean"] > 0
                        and all(name in saved and saved[name]["dose_matched"]
                                for name in ("reflected", "off_target")))
            selected_distance = saved["target"]["logit_distance_from_clean"]
            passed = bool(eligible and selected_distance < outcomes["injected_logit_distance_from_clean"]
                          and selected_distance < saved["reflected"]["logit_distance_from_clean"]
                          and selected_distance < saved["off_target"]["logit_distance_from_clean"])
            row["interventions"] = {key: value for key, value in outcomes.items() if key != "arms"}
            row["interventions"].update({"arms": saved, "eligible": eligible, "prediction_pass": passed})
        repeat = trace_vit_classifier(model, pixels,
            points=[ViTTracePoint(0, "block", 0, x) for x in (0, *tokens)], run_id=f"heldout-{index}/repeat")
        if (not _same(repeat.output.logits, pair["clean"].output.logits)
                or any(not _same(a.value, b.value) for a, b in zip(repeat.records, pair["clean"].records, strict=True))):
            raise RuntimeError("clean repeat does not match the original clean forward")
        row["clean_repeat_logits"] = sink.tensor(f"heldout-{index}/repeat/logits", repeat.output.logits)
        row["clean_repeat_events"] = [r.event.to_dict() for r in repeat.records]
        row["clean_repeat_vectors"] = [sink.tensor(f"heldout-{index}/repeat/{r.event.token_positions[0]}", r.value)
                                       for r in repeat.records]
        heldout.append(row)
        sink.write(f"heldout-{index}.json", row)
        print(f"held-out case {index + 1}/4 complete", flush=True)
    _assert_no_hooks(model)
    state_after = state_digest(model)
    if state_before != state_after:
        raise RuntimeError("model state changed during the study")
    selection_hits = sum(row["selection"]["directional"] == row["injection_token"] for row in heldout)
    magnitude_hits = sum(row["selection"]["magnitude"] == row["injection_token"] for row in heldout)
    random_hits = sum(row["random"] == row["injection_token"] for row in heldout)
    passed = sum(bool(row.get("interventions", {}).get("prediction_pass")) for row in heldout)
    eligible = sum(bool(row.get("interventions", {}).get("eligible")) for row in heldout)
    return {"schema": "vit-synthetic-injection-results-v1", "source": source, "status": "complete",
            "discovery_freeze_sha256": fit_sha, "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after, "heldout": heldout,
            "selection": {"directional_hits": selection_hits if _norm(direction) else None,
                          "magnitude_hits": magnitude_hits, "random_hits": random_hits,
                          "prediction": ("ineligible" if not _norm(direction) else
                              "pass" if selection_hits >= 3 and selection_hits >= random_hits else "fail")},
            "restoration": {"eligible": eligible, "passes": passed,
                            "prediction": "pass" if passed >= 3 else "inconclusive" if eligible < 4 else "fail"}}


def main():
    if (not sys.flags.isolated or not sys.dont_write_bytecode or not sys.pycache_prefix
            or not os.path.isabs(sys.pycache_prefix) or os.path.lexists(sys.pycache_prefix)):
        raise RuntimeError("isolated source-only bootstrap is required")
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-source", required=True, help="reviewed immutable public commit")
    parser.add_argument("--git-executable", type=Path, required=True, help="trusted absolute Git executable")
    args = parser.parse_args()
    if len(args.published_source) != 40 or any(x not in "0123456789abcdef" for x in args.published_source):
        raise ValueError("published source must be a full lowercase commit ID")
    if not args.git_executable.is_absolute() or not args.git_executable.is_file():
        raise ValueError("Git executable must be an existing absolute path")
    git = args.git_executable.resolve()
    if git.name.lower() not in ("git", "git.exe") or git.is_relative_to(ROOT):
        raise ValueError("Git executable must be outside the checkout")
    if min(shutil.disk_usage(args.output.parent).free, shutil.disk_usage(ROOT).free) < FREE:
        raise RuntimeError("less than 2 GiB free on output or source volume before model load")
    args.output.mkdir(exist_ok=False)
    sink = Evidence(args.output)
    source = {"published_commit": args.published_source}
    try:
        source.update({"local_commit": subprocess.check_output([str(git), "--no-replace-objects", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "git_executable": str(git), "git_sha256": digest(git.read_bytes()),
                       "files": verify_source(args.published_source, git), "python": platform.python_version(),
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "model": MODEL_REPO, "revision": MODEL_REVISION, "assets": MODEL_FILES})
        sink.write("started.json", {"schema": "vit-synthetic-start-v1", "source": source,
                                    "free_bytes": shutil.disk_usage(args.output).free})
        verify_assets(args.assets)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        model, loading = ViTForImageClassification.from_pretrained(str(args.assets), local_files_only=True,
                    attn_implementation="eager", torch_dtype=torch.float32, output_loading_info=True)
        sink.write("load-observation.json", verify_loading_info(loading))
        model = model.eval().cpu()
        result = run(model, sink, source)
        verify_assets(args.assets)
        if verify_source(args.published_source, git) != source["files"] or digest(git.read_bytes()) != source["git_sha256"]:
            raise RuntimeError("source changed during the study")
        sink.close()
        sink.write("completed.json", result)
    except BaseException as error:
        try:
            if not sink.stream.closed:
                sink.close()
            sink.write("incomplete.json", {"schema": "vit-synthetic-incomplete-v1", "source": source,
                                            "error_type": type(error).__name__, "message": str(error)})
        finally:
            raise
    print(json.dumps({"status": "complete", "selection": result["selection"],
                      "restoration": result["restoration"]}, allow_nan=False))


if __name__ == "__main__":
    main()
