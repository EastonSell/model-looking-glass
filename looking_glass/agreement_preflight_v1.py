"""Fixed public, CPU-only DistilGPT-2 agreement discovery preflight."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from .pretrained import ASSETS, MODEL_ID, REVISION, load_distilgpt2


PROTOCOL = "distilgpt2-agreement-preflight-v1"
SEED = 2026102817
TARGETS = {" is": 318, " are": 389, " was": 373, " were": 547}
MAX_TOKENS = 32
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
VOCAB_SIZE = 50257
GROUPS = (
    (("cat", "cats", "dog", "dogs"), "near the windows", "plural"),
    (("teacher", "teachers", "doctor", "doctors"), "in the room", "singular"),
    (("artist", "artists", "writer", "writers"), "beside the rivers", "plural"),
    (("farmer", "farmers", "sailor", "sailors"), "behind the house", "singular"),
)
SOURCE_FILES = (
    "docs/DISTILGPT2_AGREEMENT_PREFLIGHT_V1.md",
    "looking_glass/__init__.py",
    "looking_glass/agreement_preflight_v1.py",
    "looking_glass/pretrained.py",
    "looking_glass/resident_worker.py",
    "tests/test_agreement_preflight_v1.py",
    "tools/run_agreement_preflight_v1.py",
    "tools/supervise_agreement_preflight_v1.py",
)


def _require_source_first():
    prefix = sys.pycache_prefix
    if (not sys.dont_write_bytecode or not prefix or not Path(prefix).is_absolute()
            or os.path.lexists(prefix)):
        raise RuntimeError("use -B and -X pycache_prefix=<absolute nonexistent path>")


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], timeout=30).decode("utf-8").strip()


def _clean(root):
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("checkout must be clean")


def _source_binding(root, revision):
    """Bind working source bytes to a committed revision, not the runtime."""
    if subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", revision, "HEAD"],
                      timeout=30, check=False).returncode:
        raise RuntimeError("source revision is not an ancestor of HEAD")
    binding = {}
    for name in SOURCE_FILES:
        path = (root / name).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise RuntimeError("source path is outside the checkout")
        raw = path.read_bytes()
        committed = subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:{name}"], timeout=30)
        if raw != committed:
            raise RuntimeError(f"source bytes differ from the frozen revision: {name}")
        binding[name] = hashlib.sha256(raw).hexdigest()
    return binding


def _copy_verified(source, target, expected_size, expected_digest):
    digest = hashlib.sha256()
    count = 0
    with open(source, "rb") as src, open(target, "xb") as dst:
        while True:
            part = src.read(1024 * 1024)
            if not part:
                break
            count += len(part)
            if count > expected_size:
                raise ValueError("tokenizer asset size mismatch")
            digest.update(part)
            dst.write(part)
    if count != expected_size or digest.hexdigest() != expected_digest:
        raise ValueError("tokenizer asset integrity mismatch")


def _tokenizer(cache_dir):
    """Stage verified tokenizer assets without constructing or running a model."""
    import transformers
    from huggingface_hub import hf_hub_download

    cache = Path(cache_dir).resolve(strict=True)
    names = tuple(name for name in ASSETS if name != "model.safetensors")
    with tempfile.TemporaryDirectory(prefix="agreement-tokenizer-", dir=cache) as staged:
        for name in names:
            source = hf_hub_download(repo_id=MODEL_ID, filename=name, revision=REVISION,
                                     cache_dir=str(cache), local_files_only=True, token=False)
            size, digest = ASSETS[name]
            _copy_verified(source, Path(staged) / name, size, digest)
        return transformers.AutoTokenizer.from_pretrained(
            staged, local_files_only=True, trust_remote_code=False, use_fast=True)


def tokenize_inputs(tokenizer):
    """Encode only the fixed public rows and reject structural misalignment."""
    rows, groups = [], []
    base = tokenizer.encode("The", add_special_tokens=False)
    if not base:
        raise ValueError("empty subject prefix")
    for index, (subjects, context, attractor_number) in enumerate(GROUPS):
        group_rows = []
        suffix = tokenizer.encode(" " + context, add_special_tokens=False)
        for position, subject in enumerate(subjects):
            text = f"The {subject} {context}"
            ids = tokenizer.encode(text, add_special_tokens=False)
            subject_prefix = tokenizer.encode(f"The {subject}", add_special_tokens=False)
            if len(subject_prefix) != len(base) + 1 or subject_prefix[:-1] != base:
                raise ValueError("subject must be exactly one token after The")
            if ids != subject_prefix + suffix:
                raise ValueError("full text does not preserve subject and context token alignment")
            if not ids or len(ids) > MAX_TOKENS or not suffix:
                raise ValueError("prefix or context token count is outside its bound")
            for continuation, target in TARGETS.items():
                if tokenizer.encode(text + continuation, add_special_tokens=False) != ids + [target]:
                    raise ValueError("next-token alignment failed")
            number = "singular" if position % 2 == 0 else "plural"
            row = {"index": len(rows), "group": index + 1, "text": text,
                   "subject": subject, "context": context, "subject_number": number,
                   "attractor_number": attractor_number,
                   "congruent": number == attractor_number,
                   "grammatical_sign": 1 if number == "singular" else -1,
                   "input_ids": list(ids), "attention_mask": [1] * len(ids),
                   "length": len(ids), "subject_index": len(base),
                   "subject_token_id": subject_prefix[-1], "final_position": len(ids) - 1}
            rows.append(row)
            group_rows.append(row)
        same_length = len({row["length"] for row in group_rows}) == 1
        same_suffix = all(row["input_ids"][-len(suffix):] == suffix for row in group_rows)
        groups.append({"group": index + 1, "context_token_suffix": list(suffix),
                       "same_length": same_length, "same_context_suffix": same_suffix})
        if not same_length or not same_suffix:
            raise ValueError("quartet token alignment failed")
    return rows, groups


def summarize(rows):
    if len(rows) != 16:
        raise ValueError("exactly sixteen completed rows are required")
    counts = {"overall": [0, 0], "singular": [0, 0], "plural": [0, 0],
              "congruent": [0, 0], "incongruent": [0, 0]}
    for row in rows:
        correct = row["margin"] * row["grammatical_sign"] > 0
        for key in ("overall", row["subject_number"], "congruent" if row["congruent"] else "incongruent"):
            counts[key][0] += int(correct)
            counts[key][1] += 1
    if counts["overall"][1] != 16 or any(counts[k][1] != 8 for k in counts if k != "overall"):
        raise ValueError("fixed stratum sizes are invalid")
    informative = counts["overall"][0] >= 12 and all(counts[k][0] >= 5 for k in counts if k != "overall")
    return {"counts": {key: {"correct": value[0], "total": value[1]} for key, value in counts.items()},
            "informative": informative}


def extract_targets(raw):
    """Decode IEEE float32 bits in binary64, even with float32 flush-to-zero."""
    if type(raw) is not bytes or len(raw) != VOCAB_SIZE * 4:
        raise ValueError("last-token logit vector has the wrong byte length")
    scores = {}
    for name, token in TARGETS.items():
        bits = raw[token * 4:token * 4 + 4]
        encoded = int.from_bytes(bits, "little")
        sign = -1 if encoded >> 31 else 1
        exponent = (encoded >> 23) & 0xff
        fraction = encoded & 0x7fffff
        if exponent == 0xff:
            raise ValueError("nonfinite target logit")
        if exponent == 0:
            value = math.ldexp(sign * fraction, -149) if fraction else (-0.0 if sign < 0 else 0.0)
        else:
            value = math.ldexp(sign * (0x800000 + fraction), exponent - 150)
        scores[name] = {"value": value, "bits_le": bits.hex()}
    return scores


def _write_json(path, payload):
    raw = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
    with open(path, "xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def freeze(root, cache_dir, output):
    _require_source_first()
    import tokenizers
    import transformers

    root = Path(root).resolve(strict=True)
    _clean(root)
    revision = _git(root, "rev-parse", "HEAD")
    sources = _source_binding(root, revision)
    tokenizer = _tokenizer(cache_dir)
    rows, groups = tokenize_inputs(tokenizer)
    manifest = {"protocol": PROTOCOL, "source_revision": revision, "sources": sources,
                "model_id": MODEL_ID, "model_revision": REVISION,
                "asset_sha256": {name: digest for name, (_, digest) in ASSETS.items()},
                "transformers_version": transformers.__version__, "tokenizers_version": tokenizers.__version__,
                "seed": SEED, "targets": TARGETS, "rows": rows, "groups": groups,
                "input_purpose": "public development; planned exposure before model execution"}
    return manifest, _write_json(output, manifest)


def _verify_manifest(root, manifest_path):
    import tokenizers
    import transformers

    _clean(root)
    path = Path(manifest_path).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("frozen manifest must be inside the checkout")
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    committed = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"], timeout=30)
    if raw != committed:
        raise RuntimeError("frozen manifest must match its tracked HEAD bytes")
    manifest = json.loads(raw)
    if (manifest.get("protocol") != PROTOCOL or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != REVISION
            or _canonical_json(manifest.get("seed")) != _canonical_json(SEED)
            or _canonical_json(manifest.get("targets")) != _canonical_json(TARGETS)
            or manifest.get("asset_sha256") != {name: digest for name, (_, digest) in ASSETS.items()}
            or manifest.get("transformers_version") != transformers.__version__
            or manifest.get("tokenizers_version") != tokenizers.__version__
            or manifest.get("sources") != _source_binding(root, manifest["source_revision"])):
        raise ValueError("frozen manifest binding does not match this runtime/source")
    return manifest, hashlib.sha256(raw).hexdigest(), _git(root, "rev-parse", "HEAD")


def _verify_inventory(payload, rows):
    """Check cooperative reservation of each exact public row and its prefixes."""
    if type(payload) is not dict or type(payload.get("records")) is not list:
        raise ValueError("invalid exposure inventory payload")
    expected_identity = {"model_id": MODEL_ID, "model_revision": REVISION,
                         "weights_sha256": ASSETS["model.safetensors"][1],
                         "tokenizer_id": MODEL_ID, "tokenizer_revision": REVISION,
                         "tokenizer_sha256": ASSETS["tokenizer.json"][1]}
    if payload.get("identity") != expected_identity:
        raise ValueError("exposure inventory model or tokenizer identity mismatch")
    planned = {(tuple(record.get("token_ids", ())), record.get("public_text"))
               for record in payload["records"] if record.get("category") == "planned"}
    prefixes = {ids[:i] for ids, _ in planned for i in range(1, len(ids) + 1)}
    for row in rows:
        ids = tuple(row["input_ids"])
        if (ids, row["text"]) not in planned or any(ids[:i] not in prefixes for i in range(1, len(ids) + 1)):
            raise ValueError("public row or prefix is absent from the planned inventory")


def _state_digest(model):
    digest = hashlib.sha256()
    for kind, values in (("parameter", model.named_parameters(remove_duplicate=False)),
                         ("buffer", model.named_buffers(remove_duplicate=False))):
        for name, tensor in values:
            if tensor.device.type != "cpu":
                raise ValueError("model state must be on CPU")
            array = tensor.detach().contiguous().numpy()
            digest.update(f"{kind}\0{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("utf-8"))
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _probe_denormals(torch):
    """Test subnormal input and output arithmetic in this execution thread."""
    subnormal = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    normal = torch.tensor([0x00800000], dtype=torch.int32).view(torch.float32)
    input_bits = int((subnormal * 1.0).view(torch.int32).item())
    output_bits = int((normal * 0.5).view(torch.int32).item())
    if input_bits != 1 or output_bits != 0x00400000:
        raise RuntimeError("float32 subnormal arithmetic was flushed")
    return {"subnormal_input_bits": input_bits, "subnormal_output_bits": output_bits}


def _prepare_denormals(torch):
    if not torch.set_flush_denormal(False):
        raise RuntimeError("float32 denormal mode cannot be configured")
    return _probe_denormals(torch)


def _event(stream, payload):
    stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run(root, cache_dir, manifest_path, inventory_payload, inventory_sha256,
        inventory_revision, inventory_source_sha256, output_dir):
    _require_source_first()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or sys.byteorder != "little":
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES=-1 and a little-endian host")
    import torch
    from .resident_worker import registered_model_state

    root = Path(root).resolve(strict=True)
    manifest, manifest_sha, frozen_commit = _verify_manifest(root, manifest_path)
    canonical_inventory = _canonical_json(inventory_payload)
    if hashlib.sha256(canonical_inventory).hexdigest() != inventory_sha256:
        raise ValueError("exposure inventory digest mismatch")
    _verify_inventory(inventory_payload, manifest["rows"])
    tokenizer = _tokenizer(cache_dir)
    if _canonical_json(tokenize_inputs(tokenizer)) != _canonical_json((manifest["rows"], manifest["groups"])):
        raise ValueError("frozen token alignment differs from the pinned tokenizer")
    if torch.cuda.is_initialized() or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("CUDA initialization or ambient autocast is unsupported")
    out = Path(output_dir).resolve()
    if root == out or root in out.parents:
        raise ValueError("output directory must be outside the checkout")
    out.mkdir(parents=False, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.random.default_generator.manual_seed(SEED)
    phase = "load"
    with (out / "attempt.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        _event(log, {"event": "started", "protocol": PROTOCOL, "utc": datetime.now(timezone.utc).isoformat(),
                     "manifest_sha256": manifest_sha, "frozen_commit": frozen_commit,
                     "inventory_sha256": inventory_sha256, "inventory_revision": inventory_revision,
                     "inventory_source_sha256": inventory_source_sha256})
        try:
            phase = "denormal_configuration"
            denormals_before = _prepare_denormals(torch)
            _event(log, {"event": "denormal_probe_before", **denormals_before})
            phase = "load"
            bundle = load_distilgpt2(cache_dir=cache_dir, allow_download=False)
            model = bundle.model
            baseline = registered_model_state(model, "cpu")
            state_before = _state_digest(model)
            rows = []
            total = 0
            phase = "forwards"
            with (out / "last_logits.f32").open("xb") as archive:
                for frozen in manifest["rows"]:
                    _event(log, {"event": "forward_started", "index": frozen["index"],
                                 "input_ids": frozen["input_ids"], "attention_mask": frozen["attention_mask"]})
                    ids = torch.tensor([frozen["input_ids"]], dtype=torch.long, device="cpu")
                    mask = torch.tensor([frozen["attention_mask"]], dtype=torch.long, device="cpu")
                    with torch.inference_mode():
                        output = model(input_ids=ids, attention_mask=mask)
                    logits = output.logits
                    if (type(logits) is not torch.Tensor or logits.dtype != torch.float32 or logits.device.type != "cpu"
                            or tuple(logits.shape) != (1, frozen["length"], VOCAB_SIZE)
                            or output.past_key_values is not None or not bool(torch.isfinite(logits).all())):
                        raise ValueError("unexpected logits shape, device, dtype, cache or values")
                    vector = logits[0, -1].detach().clone().contiguous()
                    raw = vector.numpy().tobytes(order="C")
                    if len(raw) != VOCAB_SIZE * 4 or total + len(raw) > MAX_ARCHIVE_BYTES:
                        raise ValueError("last-token archive byte bound exceeded")
                    scores = extract_targets(raw)
                    first, second = scores[" is"]["value"], scores[" are"]["value"]
                    if archive.write(raw) != len(raw):
                        raise OSError("short logit archive write")
                    archive.flush()
                    os.fsync(archive.fileno())
                    row = {**frozen, "is_logit": first, "are_logit": second,
                           "was_logit": scores[" was"]["value"], "were_logit": scores[" were"]["value"],
                           "target_bits_le": {name: value["bits_le"] for name, value in scores.items()},
                           "margin": first - second,
                           "past_margin": scores[" was"]["value"] - scores[" were"]["value"],
                           "archive_offset": total,
                           "archive_bytes": len(raw), "vector_sha256": hashlib.sha256(raw).hexdigest()}
                    row["correct"] = row["margin"] * row["grammatical_sign"] > 0
                    total += len(raw)
                    rows.append(row)
                    _event(log, {"event": "forward_completed", "index": frozen["index"],
                                 "vector_sha256": row["vector_sha256"]})
            phase = "verify"
            state_after = _state_digest(model)
            denormals_after = _probe_denormals(torch)
            _event(log, {"event": "denormal_probe_after", **denormals_after})
            if state_after != state_before or registered_model_state(model, "cpu") != baseline:
                raise RuntimeError("registered model state changed")
            if torch.cuda.is_initialized() or total != 16 * VOCAB_SIZE * 4:
                raise RuntimeError("CUDA state or completed archive size is invalid")
            verified_manifest, verified_sha, verified_commit = _verify_manifest(root, manifest_path)
            if (verified_manifest != manifest or verified_sha != manifest_sha
                    or verified_commit != frozen_commit):
                raise RuntimeError("checkout or frozen manifest changed during the run")
            result = {"protocol": PROTOCOL, "status": "complete", "source_revision": manifest["source_revision"],
                      "frozen_commit": frozen_commit, "manifest_sha256": manifest_sha,
                      "inventory_sha256": inventory_sha256, "inventory_revision": inventory_revision,
                      "inventory_source_sha256": inventory_source_sha256, "model": bundle.metadata,
                      "runtime": {"python": sys.version, "platform": platform.platform(),
                                  "torch": torch.__version__, "threads": torch.get_num_threads(),
                                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                  "cuda_initialized": torch.cuda.is_initialized(),
                                  "flush_denormal_requested": False,
                                  "denormal_probe_before": denormals_before,
                                  "denormal_probe_after": denormals_after,
                                  "bytecode_writes_disabled": sys.dont_write_bytecode,
                                  "nonexistent_pycache_prefix": sys.pycache_prefix},
                      "state_sha256_before": state_before, "state_sha256_after": state_after,
                      "archive_bytes": total, "archive_sha256": hashlib.sha256((out / "last_logits.f32").read_bytes()).hexdigest(),
                      "groups": manifest["groups"], "rows": rows, "summary": summarize(rows)}
            result_sha = _write_json(out / "result.json", result)
            _event(log, {"event": "complete", "result_sha256": result_sha})
            return result
        except BaseException as error:
            try:
                _event(log, {"event": "failed", "phase": phase, "error_type": type(error).__name__})
            except BaseException as log_error:
                raise error from log_error
            raise
