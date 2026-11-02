"""Run the frozen negation-cue experiment on one offline CPU model."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import traceback


ASSETS = {
    "config.json": (483, "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83"),
    "model.safetensors": (267954768, "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"),
    "tokenizer.json": (466062, "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"),
    "tokenizer_config.json": (48, "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
}
SOURCE_PATHS = (
    "docs/BERT_NEGATION_CUE_V1.md", "experiments/bert_negation_public_cases_v1.json",
    "looking_glass/__init__.py", "looking_glass/bert_negation_cue.py",
    "looking_glass/intervention_plan.py", "looking_glass/interventions.py",
    "looking_glass/model_ownership.py", "scripts/bert_negation_cue_v1.py",
)
MANIFEST_HASH = "4308e7977e36eb731948193cce27ffa399421bf1bd8bbdd4e42dfa7cc6f4a8ff"
VERSIONS = {"torch": "2.8.0+cu128", "transformers": "4.57.3", "tokenizers": "0.22.2",
            "safetensors": "0.8.0", "numpy": "2.4.6"}
SEED = 2026102901


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(snapshot):
    snapshot = Path(snapshot)
    result = {}
    for name, (size, expected) in ASSETS.items():
        path = snapshot / name
        actual_size = path.stat().st_size
        actual_hash = file_hash(path)
        if actual_size != size or actual_hash != expected:
            raise ValueError(f"checkpoint asset differs: {name}")
        result[name] = {"bytes": size, "sha256": actual_hash}
    return result


def write_json_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_event(zipped, raw, record):
    zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":"),
                             allow_nan=False) + "\n").encode("utf-8"))
    # A sync flush permits recovering complete JSONL records from a truncated
    # gzip member even if process interruption prevents its final footer.
    zipped.flush()
    raw.flush()
    os.fsync(raw.fileno())


def _git_env():
    allowed = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT")
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_COUNT="0",
               GIT_NO_REPLACE_OBJECTS="1", GIT_TERMINAL_PROMPT="0")
    return env


def source_identity(repo, commit, git_exe):
    repo, git_exe = Path(repo).resolve(), Path(git_exe).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not git_exe.is_file():
        raise ValueError("source commit or Git executable is invalid")

    def git(*args):
        return subprocess.run([str(git_exe), "--no-replace-objects", *args], cwd=repo,
                              env=_git_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, timeout=30).stdout

    if git("rev-parse", "HEAD").decode("ascii").strip() != commit:
        raise ValueError("checkout HEAD differs from requested commit")
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("source checkout is not clean")
    identity = {}
    for name in SOURCE_PATHS:
        local = (repo / name).read_bytes()
        saved = git("show", f"{commit}:{name}")
        if local != saved:
            raise ValueError(f"source bytes differ from commit: {name}")
        identity[name] = hashlib.sha256(local).hexdigest()
    if identity["experiments/bert_negation_public_cases_v1.json"] != MANIFEST_HASH:
        raise ValueError("fixed case manifest hash differs")
    return identity


def verify_tokenizer(snapshot, manifest):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
    if tokenizer.token_to_id("any") != manifest["scored_tokens"]["any"] or \
            tokenizer.token_to_id("some") != manifest["scored_tokens"]["some"]:
        raise ValueError("scored token IDs differ")
    count = 0
    for case in manifest["cases"]:
        for cue, cue_id in manifest["cues"].items():
            text = manifest["template"].format(role=case["role"], cue=cue,
                                               verb=case["verb"], object=case["object"])
            encoded = tokenizer.encode(text)
            expected = case["just_ids"].copy()
            expected[4] = cue_id
            if encoded.ids != expected or encoded.attention_mask != manifest["attention_mask"]:
                raise ValueError("tokenized case differs from fixed manifest")
            count += 1
    if count != 32:
        raise ValueError("expected 32 fixed tokenized rows")
    return {"rows": count, "any_id": manifest["scored_tokens"]["any"],
            "some_id": manifest["scored_tokens"]["some"]}


def state_hash(model):
    digest = hashlib.sha256()
    entries = [("state", name, value) for name, value in model.state_dict().items()]
    entries.extend(("buffer", name, value) for name, value in model.named_buffers(remove_duplicate=False))
    for kind, name, tensor in sorted(entries, key=lambda item: (item[0], item[1])):
        value = tensor.detach().contiguous()
        if value.device.type != "cpu":
            raise ValueError("model state is not CPU-resident")
        header = json.dumps([kind, name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode("utf-8")
        raw = value.numpy().tobytes()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def verify_no_flush(torch):
    subnormal = torch.tensor([2.0 ** -149], dtype=torch.float32)
    if (subnormal * torch.tensor([1.0], dtype=torch.float32)).item() != 2.0 ** -149:
        raise RuntimeError("CPU float32 denormal values are being flushed")


def enable_no_flush(torch):
    if torch.set_flush_denormal(False) is not True:
        raise RuntimeError("could not disable float32 denormal flushing")
    verify_no_flush(torch)


def _blocked(*_args, **_kwargs):
    raise RuntimeError("network and CUDA initialization are disabled for this CPU study")


def run(repo, snapshot, output, commit, git_exe):
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("owner must start before Torch or Transformers import")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cpu_environment = {name: {"present": name in os.environ, "value": os.environ.get(name)}
                       for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY")}
    socket.socket.connect = _blocked
    socket.create_connection = _blocked
    source = source_identity(repo, commit, git_exe)
    assets = verify_assets(snapshot)
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError("installed dependency versions differ from the fixed runtime")
    manifest = json.loads((Path(repo) / "experiments/bert_negation_public_cases_v1.json").read_text(encoding="utf-8"))
    Path(output).mkdir(parents=False, exist_ok=False)
    output = Path(output)
    write_json_new(output / "started.json", {"source_commit": commit, "source_sha256": source,
                   "assets": assets, "versions": versions, "seed": SEED, "planned_forwards": 448,
                   "cpu_environment": cpu_environment})
    phase = "tokenizer"
    try:
        tokenization = verify_tokenizer(snapshot, manifest)
        phase = "runtime"
        import torch
        torch.cuda._lazy_init = _blocked
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized before the study")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        enable_no_flush(torch)
        torch.manual_seed(SEED)
        halfway = torch.tensor([1.0 + 2.0 ** -24, 1.0 + 3.0 * 2.0 ** -24], dtype=torch.float64).float()
        if halfway.tolist() != [1.0, 1.0 + 2.0 ** -22]:
            raise RuntimeError("float32 rounding is not nearest-even")
        from transformers import DistilBertForMaskedLM
        phase = "load_model"
        model = DistilBertForMaskedLM.from_pretrained(str(snapshot), local_files_only=True,
                                                      attn_implementation="eager").eval()
        if (model.config.dim != 768 or model.config.n_layers != 6 or model.config.vocab_size != 30522
                or model.config._attn_implementation != "eager"
                or any(type(layer.attention).__name__ != "MultiHeadSelfAttention"
                       for layer in model.distilbert.transformer.layer)
                or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
                or torch.cuda.is_initialized()):
            raise RuntimeError("loaded model differs from the fixed CPU architecture")
        before = state_hash(model)
        sys.path.insert(0, str(Path(repo).resolve()))
        from looking_glass.bert_negation_cue import run_study
        phase = "forwards"
        event_path = output / "events.jsonl.gz"
        with event_path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                def emit(record):
                    write_event(zipped, raw, record)
                result = run_study(model, manifest, emit)
            raw.flush()
            os.fsync(raw.fileno())
        phase = "model_state"
        after = state_hash(model)
        verify_no_flush(torch)
        if before != after or torch.cuda.is_initialized() or result["forwards"] != 448:
            raise RuntimeError("final state, CPU-only or complete-schedule check failed")
        result.update({"source_commit": commit, "source_sha256": source, "assets": assets,
                       "versions": versions, "seed": SEED, "tokenizer": tokenization,
                       "cpu_environment": cpu_environment,
                       "state_sha256_before": before, "state_sha256_after": after,
                       "events_sha256": file_hash(event_path), "cpu_threads": 1,
                       "cpu_interop_threads": 1, "attention": "eager", "cuda_initialized": False})
        phase = "result"
        write_json_new(output / "result.json", result)
        return result
    except BaseException as exc:
        failure = {"status": "incomplete", "phase": phase, "error_type": type(exc).__name__,
                   "message": str(exc), "source_commit": commit,
                   "traceback": traceback.format_exc(limit=8)}
        try:
            write_json_new(output / "failure.json", failure)
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
    repo = Path(__file__).resolve().parents[1]
    result = run(repo, args.snapshot, args.output, args.source_commit, args.git_executable)
    print(json.dumps({"status": result["status"], "forwards": result["forwards"],
                      "directional_prediction": result["directional_prediction"]}, sort_keys=True))


if __name__ == "__main__":
    main()
