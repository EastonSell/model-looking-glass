"""One supervised, source-bound CPU relative-dose development scan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


SOURCE_FILES = (
    "docs/DISTILGPT2_AGREEMENT_HEADS_V2.md", "looking_glass/__init__.py",
    "looking_glass/agreement_head_owner_v2.py", "looking_glass/agreement_head_launcher_v2.py",
    "looking_glass/agreement_head_study_v2.py",
    "looking_glass/agreement_head_audit_v2.py", "looking_glass/agreement_journal.py",
    "looking_glass/agreement_preflight_v1.py", "looking_glass/copy_head_factorial.py",
    "looking_glass/copy_head_intervention.py", "looking_glass/gpt2_head_slice.py",
    "looking_glass/interventions.py", "looking_glass/model_ownership.py", "looking_glass/pretrained.py",
    "looking_glass/pretrained_exposures.py", "looking_glass/pretrained_exposures_v2.py",
    "looking_glass/trace_schema.py", "looking_glass/transformer_trace.py",
)


def _git(root, *args):
    environment = os.environ.copy()
    environment.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG_COUNT"):
        environment.pop(name, None)
    return subprocess.check_output(["git", "-C", str(root), *args], env=environment, timeout=30)


def _source_binding(root, revision):
    if type(revision) is not str or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("source commit must be a full lowercase SHA")
    if _git(root, "rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("checkout HEAD differs from the source commit")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("source checkout must be clean")
    result = {}
    for name in SOURCE_FILES:
        raw_tree = _git(root, "ls-tree", revision, "--", name).decode("utf-8").strip()
        if not re.fullmatch(r"100644 blob [0-9a-f]{40}\t" + re.escape(name), raw_tree):
            raise ValueError("source must be an ordinary committed blob: " + name)
        path = (root / name).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError("source path leaves the checkout: " + name)
        raw = path.read_bytes()
        if raw != _git(root, "show", f"{revision}:{name}"):
            raise ValueError("source bytes differ from commit: " + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def _hash_file(path):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            count += len(block)
    return count, digest.hexdigest()


def _verify_assets(snapshot, assets):
    if not snapshot.is_dir() or {p.name for p in snapshot.iterdir()} != set(assets):
        raise ValueError("cached snapshot is not the exact pinned asset set")
    observed = {}
    for name, expected in assets.items():
        path = (snapshot / name).resolve(strict=True)
        if not path.is_file() or _hash_file(path) != expected:
            raise ValueError("pinned checkpoint asset differs: " + name)
        observed[name] = {"bytes": expected[0], "sha256": expected[1]}
    return observed


def _write_new(path, payload):
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short candidate write")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def run_development_owner(*, root, revision, cache_dir, attempt):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1" or os.environ.get("OMP_NUM_THREADS") != "1"
            or sys.byteorder != "little" or not sys.dont_write_bytecode or not sys.pycache_prefix
            or not Path(sys.pycache_prefix).is_absolute() or os.path.lexists(sys.pycache_prefix)):
        raise ValueError("requires CPU/offline one-thread execution with -B and a fresh pycache prefix")
    root, cache_dir, attempt = Path(root).resolve(strict=True), Path(cache_dir).resolve(strict=True), Path(attempt)
    if not attempt.is_absolute() or not attempt.parent.is_dir() or os.path.lexists(attempt):
        raise ValueError("attempt must be an absolute, new path under an existing directory")
    if __file__ != str(root / "looking_glass" / "agreement_head_owner_v2.py"):
        raise ValueError("loaded owner does not come from the declared checkout")
    binding = _source_binding(root, revision)
    from .agreement_preflight_v1 import _prepare_denormals, _probe_denormals, _state_digest
    from .agreement_head_study_v2 import run_development
    from .copy_head_factorial import binary64_rounding_probe
    from .pretrained import ASSETS, MODEL_ID, REVISION
    from .pretrained_exposures_v2 import PUBLIC_DISCOVERY
    import torch
    import transformers

    if torch.__version__ != "2.8.0+cu128" or transformers.__version__ != "4.57.3" or torch.cuda.is_initialized():
        raise ValueError("the fixed Torch or Transformers version differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2026102912)
    denormals_before = _prepare_denormals(torch)
    binary64_rounding_probe()
    snapshot = cache_dir / "models--distilbert--distilgpt2" / "snapshots" / REVISION
    assets = _verify_assets(snapshot, ASSETS)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                                            trust_remote_code=False, use_fast=True)
    rows = []
    for text, expected in PUBLIC_DISCOVERY:
        row = tokenizer.encode(text, add_special_tokens=False)
        if row != list(expected):
            raise ValueError("public discovery tokenization differs")
        for suffix, token in ((" is", 318), (" are", 389), (" was", 373), (" were", 547)):
            if tokenizer.encode(text + suffix, add_special_tokens=False) != row + [token]:
                raise ValueError("public continuation alignment differs")
        rows.append(row)
    model, info = transformers.AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, attn_implementation="eager", dtype=torch.float32,
        output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("checkpoint loading reported incompatible keys")
    model.eval()
    model.config.use_cache = False
    if (model.config.n_layer, model.config.n_head, model.config.n_embd, model.config.vocab_size) != (6, 12, 768, 50257):
        raise ValueError("model dimensions differ from the fixed checkpoint")
    before = _state_digest(model)
    metadata = {"schema": "distilgpt2-agreement-head-development-v2", "source_commit": revision,
                "source_sha256": binding, "model_id": MODEL_ID, "model_revision": REVISION,
                "prior_failed_attempt": {"source_commit": "416ed7c60d030518e5d45aa9a774169b632c4013",
                    "terminal_sha256": "a198617a3feaa3b45549a979e82f7ff0c24f339e23bc2ff147409f33f79435e9",
                    "events_sha256": "5c4ea15e2b1ff766dad39a379f699c6ddf4be080419449d3ce81171e37eb1382",
                    "completed_forwards": 29,
                    "redesign_scope": "5% relative dose informed by exposed development activations"},
                "assets": assets, "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
                "cpu_environment": {name: os.environ.get(name) for name in ("MKL_CBWR", "ATEN_CPU_CAPABILITY")},
                "seed": 2026102912, "denormals_before": denormals_before,
                "binary64_before": "nearest-even-probes-passed", "model_state_before": before,
                "rows": [{"text": text, "token_ids": row} for (text, _), row in zip(PUBLIC_DISCOVERY, rows)]}
    result = run_development(model, rows, attempt, metadata=metadata)
    after = _state_digest(model)
    binary64_rounding_probe()
    if (after != before or torch.cuda.is_initialized() or _probe_denormals(torch) != denormals_before
            or _verify_assets(snapshot, ASSETS) != assets):
        raise RuntimeError("model state, numeric mode or pinned assets changed during the run")
    if _source_binding(root, revision) != binding:
        raise RuntimeError("source bytes changed during the run")
    terminal_sha = _hash_file(attempt / "terminal.json")[1]
    candidate = {"schema": "distilgpt2-agreement-head-development-candidate-v2", "source_commit": revision,
                 "model_state_before": before, "model_state_after": after, "terminal_sha256": terminal_sha,
                 "binary64_after": "nearest-even-probes-passed",
                 "summary": result, "scope": "candidate; require observed child exit and independent saved-byte replay"}
    digest = _write_new(attempt / "owner-candidate.json", candidate)
    return {"candidate_sha256": digest, "forward_count": result["forward_count"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_development_owner(root=args.root, revision=args.revision,
          cache_dir=args.cache_dir, attempt=args.attempt), sort_keys=True))


if __name__ == "__main__":
    main()
