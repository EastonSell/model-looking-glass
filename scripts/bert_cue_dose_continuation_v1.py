"""Run the four fixed cells of a separately documented CPU continuation."""
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
import tempfile
import traceback

ORIGINAL_SOURCE = "b75e5bcc3a866a9af56ac1ac3ba54c6abe1691a4"
ORIGINAL_HASHES = {
    "events": "ef3d75a34903e8c2fae9f69cf3d3f71d9c77c59ec171d72cbf07c9e67e9d0e05",
    "started": "19baa22fc6b6fe3da139bf22b125a4f3122a39e6e82dc9ec38604389127bd9e7",
    "failure": "12b861407be2e4050b7f5e7b792847867f7b7c588cf75857783fc36d1bb8317a",
    "receipt": "f5499d680cbd2ea1d2ebdfdb7268ce30598ad8aa7d6af9ad8ef0ff806ed54f34",
}
CASE_IDS = ("fresh-02-prior", "fresh-02-new", "fresh-03-prior", "fresh-03-new")
CONTINUATION_PATHS = ("docs/BERT_CUE_DOSE_CONTINUATION_V1.md", "scripts/bert_cue_dose_continuation_v1.py",
                      "scripts/compare_bert_cue_dose_continuation_v1.py")
GIB = 1024 ** 3


def dose_owner(repo):
    spec = importlib.util.spec_from_file_location("frozen_dose_owner", Path(repo) / "scripts/bert_cue_dose_response_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def volume(path):
    return Path(path).resolve().anchor.casefold()


def require_space(output, repo, temporary, *, initial):
    output, repo, temporary = Path(output), Path(repo), Path(temporary)
    if not temporary.is_dir() or volume(output.parent) != volume(temporary) or volume(output.parent) == volume(repo):
        raise RuntimeError("output and temporary directories must share a volume distinct from the checkout")
    if shutil.disk_usage(output.parent).free < (2 * GIB if initial else GIB):
        raise RuntimeError("insufficient output-volume disk reserve")
    if shutil.disk_usage(repo).free < GIB:
        raise RuntimeError("insufficient checkout-volume disk reserve")


def emit_with_headroom(prior, zipped, raw, output, repo, temporary, record):
    require_space(output, repo, temporary, initial=False)
    prior.write_event(zipped, raw, record)


def original_records(prior, paths):
    if set(paths) != set(ORIGINAL_HASHES):
        raise ValueError("original evidence paths differ")
    if (Path(paths["events"]).parent / "result.json").exists():
        raise ValueError("original attempt unexpectedly has a result")
    actual = {name: prior.file_hash(path) for name, path in paths.items()}
    if actual != ORIGINAL_HASHES:
        raise ValueError("original closed evidence hashes differ")
    started = json.loads(Path(paths["started"]).read_text(encoding="utf-8"))
    failure = json.loads(Path(paths["failure"]).read_text(encoding="utf-8"))
    receipt = json.loads(Path(paths["receipt"]).read_text(encoding="utf-8"))
    if (started.get("source_commit") != ORIGINAL_SOURCE or started.get("maximum_forwards") != 960 or
            len(started.get("source_sha256", {})) != 17 or
            failure.get("source_commit") != ORIGINAL_SOURCE or failure.get("status") != "incomplete" or
            failure.get("phase") != "forwards" or failure.get("error_type") != "RuntimeError" or
            failure.get("message") != "disk reserve exhausted before the next evidence event" or
            receipt.get("source_commit") != ORIGINAL_SOURCE or receipt.get("process_waited") is not True or
            receipt.get("exit_code") != 1):
        raise ValueError("original attempt metadata differs")
    return started


def select_cases(manifest):
    cases = manifest["cases"]
    expected_all = ("exposed-calibration-prior", "exposed-calibration-new", "fresh-01-prior", "fresh-01-new",
                    *CASE_IDS)
    if tuple(case["id"] for case in cases) != expected_all:
        raise ValueError("fixed original case order differs")
    chosen = cases[4:]
    if tuple(case["id"] for case in chosen) != CASE_IDS:
        raise ValueError("continuation case order differs")
    return chosen


def run(repo, snapshot, output, commit, git_exe, temporary, original_paths):
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("owner must start before Torch or Transformers import")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    temporary = Path(temporary).resolve()
    if (any(Path(os.environ.get(name, "")).resolve() != temporary for name in ("TEMP", "TMP")) or
            Path(tempfile.gettempdir()).resolve() != temporary):
        raise RuntimeError("temporary paths must use the designated output volume")
    cpu_environment = {name: {"present": name in os.environ, "value": os.environ.get(name)}
                       for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY")}
    socket.socket.connect = blocked
    socket.create_connection = blocked
    owner = dose_owner(repo)
    prior = owner.prior_owner(repo)
    prior.SOURCE_PATHS = prior.SOURCE_PATHS + CONTINUATION_PATHS
    source = prior.source_identity(repo, commit, git_exe)
    original = original_records(prior, original_paths)
    if ({name: source.get(name) for name in original["source_sha256"]} != original["source_sha256"] or
            source[owner.MANIFEST] != owner.MANIFEST_HASH or
            source[owner.PRIOR_MANIFEST] != owner.PRIOR_MANIFEST_HASH):
        raise ValueError("original source or fixed manifest differs")
    assets = prior.verify_assets(snapshot)
    versions = {name: importlib.metadata.version(name) for name in prior.VERSIONS}
    if (versions != prior.VERSIONS or versions != original["versions"] or assets != original["assets"] or
            cpu_environment != original["cpu_environment"]):
        raise ValueError("recorded original runtime, assets or CPU environment differs")
    manifest = json.loads((Path(repo) / owner.MANIFEST).read_text(encoding="utf-8"))
    prior_manifest = json.loads((Path(repo) / owner.PRIOR_MANIFEST).read_text(encoding="utf-8"))
    cases = select_cases(manifest)
    output = Path(output)
    require_space(output, repo, temporary, initial=True)
    output.mkdir(parents=False, exist_ok=False)
    prior.write_json_new(output / "started.json", {"source_commit": commit, "source_sha256": source,
                         "original_source_commit": ORIGINAL_SOURCE, "original_sha256": ORIGINAL_HASHES,
                         "assets": assets, "versions": versions, "seed": owner.SEED,
                         "maximum_forwards": 480, "case_ids": list(CASE_IDS),
                         "cpu_environment": cpu_environment})
    phase = "tokenizer"
    try:
        tokenization = owner.verify_tokenizer(snapshot, manifest, prior_manifest)
        phase = "runtime"
        import torch
        torch.cuda._lazy_init = blocked
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before the study")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        prior.enable_no_flush(torch)
        torch.manual_seed(owner.SEED)
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
        from looking_glass.bert_cue_dose_response import run_case
        phase = "forwards"
        event_path = output / "events.jsonl.gz"
        results = []
        with event_path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                def emit(record):
                    emit_with_headroom(prior, zipped, raw, output, repo, temporary, record)
                for case in cases:
                    try:
                        results.append(run_case(model, case, emit, any_id=manifest["scored_tokens"]["any"],
                                                some_id=manifest["scored_tokens"]["some"],
                                                expected_vocab=30522, expected_width=768))
                    except BaseException as exc:
                        try:
                            emit({"kind": "case_failed", "case_id": case["id"], "error_type": type(exc).__name__})
                        except Exception:
                            pass
                        raise
            raw.flush()
            os.fsync(raw.fileno())
        phase = "model_state"
        after = prior.state_hash(model)
        prior.verify_no_flush(torch)
        forwards = sum(case["forwards"] for case in results)
        if (before != after or torch.cuda.is_initialized() or forwards != owner.expected_forwards(results) or
                forwards > 480 or tuple(case["id"] for case in results) != CASE_IDS):
            raise RuntimeError("state, CPU-only or continuation-schedule check failed")
        if {name: prior.file_hash(path) for name, path in original_paths.items()} != ORIGINAL_HASHES:
            raise RuntimeError("original evidence changed during continuation")
        result = {"status": "complete", "analysis_scope": "four continued cells only; original attempt remains incomplete",
                  "forwards": forwards, "cases": results, "source_commit": commit, "source_sha256": source,
                  "original_source_commit": ORIGINAL_SOURCE, "original_sha256": ORIGINAL_HASHES,
                  "assets": assets, "versions": versions, "seed": owner.SEED, "tokenizer": tokenization,
                  "cpu_environment": cpu_environment, "state_sha256_before": before, "state_sha256_after": after,
                  "events_sha256": prior.file_hash(event_path), "cpu_threads": 1, "cpu_interop_threads": 1,
                  "attention": "eager", "cuda_initialized": False}
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


def blocked(*_args, **_kwargs):
    raise RuntimeError("network and CUDA initialization are disabled for this CPU study")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--git-executable", type=Path, required=True)
    for name in ORIGINAL_HASHES:
        parser.add_argument("--original-" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, "original_" + name) for name in ORIGINAL_HASHES}
    result = run(Path(__file__).resolve().parents[1], args.snapshot, args.output, args.source_commit,
                 args.git_executable, args.temporary, paths)
    print(json.dumps({"status": result["status"], "forwards": result["forwards"],
                      "cases": [case["id"] for case in result["cases"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
