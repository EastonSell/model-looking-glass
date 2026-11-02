"""Run the frozen two-dose cue-specificity study on one offline CPU checkpoint."""
import argparse
import gzip
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import traceback

MANIFEST = "experiments/bert_cue_dose_response_cases_v1.json"
MANIFEST_HASH = "6205dc2c1c2e779e98df2366410221c04845758549f9e79cd5fa15e51a8fb07a"
PRIOR_MANIFEST = "experiments/bert_cue_factorial_cases_v1.json"
PRIOR_MANIFEST_HASH = "b91eba1e0a52f6a183a59b0bfbb365dd54d67341bd74a4fb56325e72a7f45d4f"
NEW_PATHS = ("looking_glass/bert_cue_class.py", "looking_glass/bert_cue_factorial.py", PRIOR_MANIFEST,
             "docs/BERT_CUE_DOSE_RESPONSE_V1.md", MANIFEST,
             "experiments/bert_cue_dose_response_public_exposure_check_v1.json",
             "scripts/build_bert_cue_dose_manifest.py", "looking_glass/bert_cue_dose_response.py",
             "scripts/bert_cue_dose_response_v1.py")
SEED = 2026102905
PRELAUNCH_DISK_RESERVE = 2 * 1024 ** 3
EVENT_DISK_RESERVE = 1024 ** 3


def require_prelaunch_space(output):
    if shutil.disk_usage(Path(output).parent).free < PRELAUNCH_DISK_RESERVE:
        raise RuntimeError("insufficient disk reserve for the study")


def emit_with_headroom(prior, zipped, raw, output, record):
    if shutil.disk_usage(output).free < EVENT_DISK_RESERVE:
        raise RuntimeError("disk reserve exhausted before the next evidence event")
    prior.write_event(zipped, raw, record)


def expected_forwards(cases):
    return sum(8 if not case["doses"]["low"]["eligible"] else
               24 + 48 * sum(dose["eligible"] for dose in case["doses"].values()) for case in cases)


def prior_owner(repo):
    spec = importlib.util.spec_from_file_location("bert_negation_prior_owner", Path(repo) / "scripts/bert_negation_cue_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_PATHS = module.SOURCE_PATHS + NEW_PATHS
    return module


def verify_tokenizer(snapshot, manifest, prior_manifest):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
    cues = ("never", "rarely", "always", "often", "not", "hardly", "really", "certainly")
    if (set(manifest["cue_ids"]) != set(cues) or
            any(tokenizer.token_to_id(cue) != manifest["cue_ids"][cue] for cue in cues) or
            tokenizer.token_to_id("any") != manifest["scored_tokens"]["any"] or
            tokenizer.token_to_id("some") != manifest["scored_tokens"]["some"]):
        raise ValueError("fixed cue or scored token differs")
    count = 0
    old_cases = {case["id"]: case for case in prior_manifest["cases"]}
    for case in manifest["cases"]:
        group = manifest["cue_sets"][case["cue_set"]]
        for cue in group["negative"] + group["affirmative"]:
            row = case["rows"][cue]
            sentence = manifest["template"].format(
                role=case["role"], cue=cue, verb=case["verb"], object=case["object"])
            tokenized = tokenizer.encode(sentence)
            if (row["text"] != sentence or row["input_ids"] != tokenized.ids or
                    row["attention_mask"] != tokenized.attention_mask or
                    tokenized.attention_mask != [1] * len(tokenized.ids) or
                    tokenized.ids[case["indices"]["cue"]] != manifest["cue_ids"][cue] or
                    tokenized.ids[case["indices"]["mask"]] != 103 or
                    tokenized.tokens[case["indices"]["object"]] != case["object"]):
                raise ValueError("tokenizer output differs from fixed case")
            if case["split"] == "calibration" and old_cases[case["previous_case_id"]]["rows"][cue] != row:
                raise ValueError("exposed calibration row differs from prior manifest")
            count += 1
    if count != 32:
        raise ValueError("expected 32 fixed cue variants")
    return {"rows": count, "any_id": manifest["scored_tokens"]["any"],
            "some_id": manifest["scored_tokens"]["some"]}


def blocked(*_args, **_kwargs):
    raise RuntimeError("network and CUDA initialization are disabled for this CPU study")


def run(repo, snapshot, output, commit, git_exe):
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("owner must start before Torch or Transformers import")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cpu_environment = {name: {"present": name in os.environ, "value": os.environ.get(name)}
                       for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY")}
    socket.socket.connect = blocked
    socket.create_connection = blocked
    prior = prior_owner(repo)
    source = prior.source_identity(repo, commit, git_exe)
    if source[MANIFEST] != MANIFEST_HASH:
        raise ValueError("follow-up manifest does not match the fixed cases")
    if source[PRIOR_MANIFEST] != PRIOR_MANIFEST_HASH:
        raise ValueError("calibration reference manifest differs")
    assets = prior.verify_assets(snapshot)
    versions = {name: importlib.metadata.version(name) for name in prior.VERSIONS}
    if versions != prior.VERSIONS:
        raise ValueError("installed runtime versions differ")
    manifest = json.loads((Path(repo) / MANIFEST).read_text(encoding="utf-8"))
    prior_manifest = json.loads((Path(repo) / PRIOR_MANIFEST).read_text(encoding="utf-8"))
    output = Path(output)
    require_prelaunch_space(output)
    output.mkdir(parents=False, exist_ok=False)
    prior.write_json_new(output / "started.json", {"source_commit": commit, "source_sha256": source,
                         "assets": assets, "versions": versions, "seed": SEED,
                         "maximum_forwards": 960, "cpu_environment": cpu_environment})
    phase = "tokenizer"
    try:
        tokenization = verify_tokenizer(snapshot, manifest, prior_manifest)
        phase = "runtime"
        import torch
        torch.cuda._lazy_init = blocked
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before the study")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        prior.enable_no_flush(torch)
        torch.manual_seed(SEED)
        halfway = torch.tensor([1.0 + 2.0 ** -24, 1.0 + 3.0 * 2.0 ** -24], dtype=torch.float64).float()
        if halfway.tolist() != [1.0, 1.0 + 2.0 ** -22]:
            raise RuntimeError("float32 rounding is not nearest-even")
        from transformers import DistilBertForMaskedLM
        phase = "load_model"
        model = DistilBertForMaskedLM.from_pretrained(str(snapshot), local_files_only=True,
                                                      attn_implementation="eager").eval()
        if (model.config.dim != 768 or model.config.n_layers != 6 or model.config.vocab_size != 30522 or
                model.config._attn_implementation != "eager" or
                any(type(layer.attention).__name__ != "MultiHeadSelfAttention"
                    for layer in model.distilbert.transformer.layer) or
                any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()) or
                torch.cuda.is_initialized()):
            raise RuntimeError("model differs from fixed CPU architecture")
        before = prior.state_hash(model)
        sys.path.insert(0, str(Path(repo).resolve()))
        from looking_glass.bert_cue_dose_response import run_study
        phase = "forwards"
        event_path = output / "events.jsonl.gz"
        with event_path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                def emit(record):
                    emit_with_headroom(prior, zipped, raw, output, record)
                result = run_study(model, manifest, emit, expected_vocab=30522, expected_width=768)
            raw.flush()
            os.fsync(raw.fileno())
        phase = "model_state"
        after = prior.state_hash(model)
        prior.verify_no_flush(torch)
        if before != after or torch.cuda.is_initialized() or result["forwards"] != expected_forwards(result["cases"]):
            raise RuntimeError("state, CPU-only or complete-schedule check failed")
        result.update({"source_commit": commit, "source_sha256": source, "assets": assets,
                       "versions": versions, "seed": SEED, "tokenizer": tokenization,
                       "cpu_environment": cpu_environment, "state_sha256_before": before,
                       "state_sha256_after": after, "events_sha256": prior.file_hash(event_path),
                       "cpu_threads": 1, "cpu_interop_threads": 1, "attention": "eager",
                       "cuda_initialized": False})
        phase = "result"
        prior.write_json_new(output / "result.json", result)
        return result
    except BaseException as exc:
        try:
            prior.write_json_new(output / "failure.json", {"status": "incomplete", "phase": phase,
                                 "error_type": type(exc).__name__, "message": str(exc),
                                 "source_commit": commit, "traceback": traceback.format_exc(limit=8)})
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--git-executable", type=Path, required=True)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.snapshot, args.output, args.source_commit,
                 args.git_executable)
    print(json.dumps({"status": result["status"], "forwards": result["forwards"],
                      "evaluation": result["evaluation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
