"""Run the frozen, bounded cue-context transfer experiment on CPU."""
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

MANIFEST = "experiments/bert_cue_context_transfer_cases_v1.json"
MANIFEST_HASH = "4fb31c4b49c24e78f6a4b0b59d928ebb6204a55ecdc2c9c4cd55528700d592a7"
NEW_PATHS = ("looking_glass/bert_cue_class.py", "looking_glass/bert_cue_factorial.py",
             "looking_glass/bert_cue_context_transfer.py", "scripts/bert_cue_context_transfer_v1.py",
             "scripts/build_bert_cue_context_cases.py", "docs/BERT_CUE_CONTEXT_TRANSFER_V1.md", MANIFEST,
             "experiments/bert_cue_context_public_exposure_check_v1.json",
             "experiments/bert_cue_context_public_exposure_independent_v1.json")
SEED = 2026102906
EVENT_MAX = 2 * 1024 ** 2
JOURNAL_MAX = 512 * 1024 ** 2


def prior_owner(repo):
    spec = importlib.util.spec_from_file_location("bert_negation_prior_owner", Path(repo) / "scripts/bert_negation_cue_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_PATHS = module.SOURCE_PATHS + NEW_PATHS
    return module


def require_space(repo, output, *, prelaunch):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if not repo.anchor or not output.anchor or repo.anchor.lower() == output.anchor.lower():
        raise RuntimeError("output must be on a distinct scratch volume")
    output_path = output.parent if prelaunch else output
    if (shutil.disk_usage(output_path).free < (2 if prelaunch else 1) * 1024 ** 3 or
            shutil.disk_usage(repo.anchor).free < 1024 ** 3):
        raise RuntimeError("insufficient point-in-time output or source-volume headroom")


def write_event(zipped, raw, repo, output, record):
    require_space(repo, output, prelaunch=False)
    data = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(data) > EVENT_MAX or raw.tell() + len(data) + 4096 > JOURNAL_MAX:
        raise RuntimeError("event or journal size bound exceeded")
    zipped.write(data)
    zipped.flush()
    raw.flush()
    os.fsync(raw.fileno())
    if raw.tell() > JOURNAL_MAX:
        raise RuntimeError("journal size bound exceeded after write")


def verify_tokenizer(repo, snapshot, manifest):
    from tokenizers import Tokenizer
    spec = importlib.util.spec_from_file_location("fixed_context_cases", Path(repo) / "scripts/build_bert_cue_context_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.build(Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json")))
    if actual != manifest:
        raise ValueError("tokenized fixed manifest differs")
    return {"pairs": len(manifest["pairs"]), "contexts": 6, "rows": 24,
            "any_id": manifest["scored_tokens"]["any"], "some_id": manifest["scored_tokens"]["some"]}


def blocked(*_args, **_kwargs):
    raise RuntimeError("network and CUDA initialization are disabled")


def run(repo, snapshot, output, commit, git_exe):
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("owner must start before Torch or Transformers import")
    repo, output = Path(repo).resolve(), Path(output).resolve()
    scratch = str(output.parent)
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      TEMP=scratch, TMP=scratch)
    cpu_environment = {name: {"present": name in os.environ, "value": os.environ.get(name)}
                       for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY")}
    socket.socket.connect = blocked
    socket.create_connection = blocked
    require_space(repo, output, prelaunch=True)
    prior = prior_owner(repo)
    source = prior.source_identity(repo, commit, git_exe)
    if source[MANIFEST] != MANIFEST_HASH:
        raise ValueError("fixed manifest digest differs")
    assets = prior.verify_assets(snapshot)
    versions = {name: importlib.metadata.version(name) for name in prior.VERSIONS}
    if versions != prior.VERSIONS:
        raise ValueError("installed runtime versions differ")
    manifest = json.loads((repo / MANIFEST).read_bytes())
    tokenization = verify_tokenizer(repo, snapshot, manifest)
    require_space(repo, output, prelaunch=True)
    output.mkdir(parents=False, exist_ok=False)
    prior.write_json_new(output / "started.json", {"source_commit": commit, "source_sha256": source,
                         "assets": assets, "versions": versions, "seed": SEED,
                         "maximum_forwards": 816, "cpu_environment": cpu_environment,
                         "tokenizer": tokenization, "event_max": EVENT_MAX, "journal_max": JOURNAL_MAX})
    phase = "runtime"
    try:
        import torch
        torch.cuda._lazy_init = blocked
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before study")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        prior.enable_no_flush(torch)
        torch.manual_seed(SEED)
        halfway = torch.tensor([1.0 + 2.0 ** -24, 1.0 + 3.0 * 2.0 ** -24], dtype=torch.float64).float()
        if halfway.tolist() != [1.0, 1.0 + 2.0 ** -22]:
            raise RuntimeError("float32 rounding differs from nearest-even")
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
        sys.path.insert(0, str(repo))
        from looking_glass.bert_cue_context_transfer import run_study
        phase = "forwards"
        event_path = output / "events.jsonl.gz"
        with event_path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                result = run_study(model, manifest, lambda event: write_event(zipped, raw, repo, output, event),
                                   expected_vocab=30522, expected_width=768)
            raw.flush()
            os.fsync(raw.fileno())
        phase = "model_state"
        after = prior.state_hash(model)
        prior.verify_no_flush(torch)
        expected = sum(272 if pair["dose_eligible"] else 16 for pair in result["pairs"])
        if (before != after or torch.cuda.is_initialized() or result["forwards"] != expected or
                result["forwards"] > 816 or event_path.stat().st_size > JOURNAL_MAX):
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
