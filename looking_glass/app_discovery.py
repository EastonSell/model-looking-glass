"""Bounded local folder metadata, not a checkpoint loader or a compatibility proof.

Only an explicitly chosen folder and its immediate children are examined. This
module is stdlib-only: no tokenizer/model imports, weight reads or networking.
The filesystem is trusted local storage, not a hostile-filesystem sandbox.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from pathlib import Path
import stat


MAX_ENTRIES = 500
MAX_FOLDERS = 200
MAX_CONFIG_BYTES = 64 * 1024
MAX_TOTAL_CONFIG_BYTES = 2 * 1024 * 1024
HEADS = {"gpt2": "GPT2LMHeadModel", "t5": "T5ForConditionalGeneration",
         "distilbert": "DistilBertForMaskedLM", "vit": "ViTForImageClassification",
         "llama": "LlamaForCausalLM"}
LABELS = {"gpt2": "GPT-2", "t5": "T5", "distilbert": "DistilBERT masked LM",
          "vit": "ViT classifier", "llama": "Standard Llama"}


def _candidate(status, reason, family=None, architecture=None):
    return {"status": status, "family": family, "architecture": architecture,
            "label": LABELS.get(family, "Folder"), "reason": reason}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


class _Budget:
    def __init__(self):
        self.used = 0
        self.exhausted = False

    def read_config(self, path, root):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("Config link leaves the chosen root; choose its cache root or use Inspect explicitly.")
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Config is not an ordinary file.")
        if before.st_size > MAX_CONFIG_BYTES:
            raise ValueError("Config exceeds the 64 KiB discovery limit; use Inspect explicitly.")
        remaining = MAX_TOTAL_CONFIG_BYTES - self.used
        if remaining <= before.st_size:
            self.exhausted = True
            raise ValueError("Folder metadata read budget reached; browse a narrower folder.")
        limit = min(MAX_CONFIG_BYTES + 1, remaining)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(resolved, flags)
        try:
            opened = os.fstat(fd)
            if (not stat.S_ISREG(opened.st_mode) or
                    (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
                raise ValueError("Config changed while opening; retry browsing.")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(limit)
            self.used += len(raw)
            after = os.fstat(fd)
            if len(raw) == limit or len(raw) != after.st_size or before.st_size != after.st_size:
                if limit == remaining: self.exhausted = True
                raise ValueError("Config changed or exceeded discovery read limits; use Inspect explicitly.")
        finally:
            os.close(fd)
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant, parse_float=_finite_float)
        if type(value) is not dict:
            raise ValueError("Config JSON must be an object.")
        return value


def _regular_hint(path, root):
    """Only stat known asset names; do not open weights or tokenizer files."""
    try:
        resolved = path.resolve(strict=True)
        return resolved.is_relative_to(root) and stat.S_ISREG(resolved.stat().st_mode)
    except (OSError, RuntimeError, ValueError):
        return False


def _folder_record(path, root, budget):
    row = {"name": path.name or str(path), "path": str(path), "is_checkpoint": False}
    config_path = path / "config.json"
    try:
        config_path.lstat()
    except FileNotFoundError:
        row["candidate"] = _candidate("none", "No config.json here. Open the folder to browse its immediate children.")
        return row
    except OSError:
        row["candidate"] = _candidate("unreadable", "Config metadata is inaccessible; permissions were not changed.")
        return row
    row["is_checkpoint"] = True
    try:
        config = budget.read_config(config_path, root)
    except (OSError, RuntimeError, ValueError, UnicodeError, RecursionError) as error:
        reason = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else "Config metadata is unreadable or malformed; use Inspect for details."
        row["candidate"] = _candidate("unreadable", reason[:240])
        return row
    family = config.get("model_type")
    if type(family) is not str or family not in HEADS:
        row["candidate"] = _candidate("unsupported", "Model type is not one of this app's supported families.")
        return row
    architecture = config.get("architectures")
    declared = HEADS[family] if architecture == [HEADS[family]] else None
    def result(status, reason):
        row["candidate"] = _candidate(status, reason, family, declared)
        return row
    if architecture is not None and declared is None:
        return result("unsupported", f"This family requires the {HEADS[family]} head, not the declared architecture.")
    if config.get("auto_map") or config.get("quantization_config") is not None:
        return result("unsupported", "Custom checkpoint code and quantized checkpoints are unsupported; no code was executed.")
    scaling = config.get("rope_scaling") if family == "llama" else None
    if family == "llama":
        partial = config.get("partial_rotary_factor", 1)
        if type(partial) not in (int, float) or partial != 1:
            return result("unsupported", "Partial rotary Llama configurations are unsupported.")
        if scaling is not None:
            keys = {"rope_type", "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}
            if type(scaling) is not dict or set(scaling) != keys or scaling.get("rope_type") != "llama3":
                return result("unsupported", "Only default or exact static Llama 3 RoPE metadata is eligible; other variants need another adapter.")
    if not any(_regular_hint(path / name, root) for name in ("model.safetensors", "model.safetensors.index.json")):
        return result("incomplete", "No local safetensors weight file/index found. Pickle .bin and GGUF weights are not loaded.")
    input_names = ("preprocessor_config.json",) if family == "vit" else ("tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt", "spiece.model", "tokenizer.model")
    if not any(_regular_hint(path / name, root) for name in input_names):
        return result("incomplete", "No matching tokenizer/image-processor asset names found. Use Inspect to diagnose missing assets.")
    if declared is None:
        return result("unverified", "Supported family name, but the model head is not declared. Inspect must verify local readiness.")
    if scaling is not None:
        return result("unverified", "Static Llama 3 RoPE family candidate; scaling values and geometry need Inspect validation.")
    return result("candidate", "Family/head metadata candidate only. Weights, inputs and geometry are not validated; use Inspect before Load.")


def _locations():
    """Offer conventional paths after Browse, without listing their contents."""
    candidates = []
    try:
        home = Path.home()
        candidates += [("Home", home), ("Downloads", home / "Downloads"),
                       ("Documents", home / "Documents"), ("Desktop", home / "Desktop")]
        cache = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME", str(home / ".cache" / "huggingface"))) / "hub")
        candidates.append(("Hugging Face cache", Path(cache).expanduser()))
    except (OSError, RuntimeError, ValueError):
        pass
    if os.name == "nt" and hasattr(os, "listdrives"):
        try:
            candidates += [("Drive " + drive, Path(drive)) for drive in os.listdrives()[:32]]
        except OSError:
            pass
    seen, locations = set(), []
    for name, path in candidates:
        try:
            resolved = path.resolve(strict=True)
            if resolved in seen or not resolved.is_dir(): continue
            seen.add(resolved)
            locations.append({"name": name, "path": str(resolved)})
        except (OSError, RuntimeError, ValueError):
            continue
    return locations


def browse_model_folders(requested="", root_text="", *, models_dir=None):
    """Return bounded, immediate metadata candidates inside an explicit root."""
    if models_dir is None and not (requested or root_text):
        return {"mode": "locations", "root": None, "path": None, "parent": None,
                "current": None, "folders": [], "locations": _locations(), "truncated": False,
                "notice": "Choose a local location or enter a folder path. No folder contents have been scanned."}
    root = Path(models_dir if models_dir is not None else root_text or requested).expanduser().resolve(strict=True)
    path = Path(requested).expanduser().resolve(strict=True) if requested else root
    if not root.is_dir() or not path.is_dir(): raise ValueError("Model browser needs a directory")
    if not path.is_relative_to(root): raise ValueError("Requested folder is outside the chosen model root")
    budget = _Budget()
    current = _folder_record(path, root, budget)
    paths, examined, skipped, truncated = [], 0, 0, False
    with os.scandir(path) as entries:
        for entry in itertools.islice(entries, MAX_ENTRIES):
            examined += 1
            try:
                if entry.name.startswith(".") or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    continue
                child = Path(entry.path).resolve(strict=True)
                # Junctions/redirected directories are not followed, even inside root.
                if not child.is_relative_to(root) or child.parent != path or child.name != entry.name:
                    skipped += 1
                    continue
                paths.append(child)
            except (OSError, RuntimeError, ValueError):
                skipped += 1
    # Reaching the cap is conservatively incomplete; do not pull an extra entry
    # merely to determine whether the directory happens to end at the limit.
    truncated = examined == MAX_ENTRIES
    paths.sort(key=lambda item: (item.name.casefold(), item.name))
    if len(paths) > MAX_FOLDERS:
        paths = paths[:MAX_FOLDERS]
        truncated = True
    folders = [_folder_record(child, root, budget) for child in paths]
    return {"mode": "directory", "root": str(root), "path": str(path),
            "parent": None if path == root else str(path.parent), "current": current,
            "folders": folders, "locations": [], "truncated": truncated or budget.exhausted,
            "notice": "Only this folder and its immediate children were checked. Metadata candidates are not verified compatible; Inspect validates readiness. No models were loaded.",
            "discovery": {"entries_examined": examined, "max_entries": MAX_ENTRIES,
                          "max_folders": MAX_FOLDERS, "skipped_entries": skipped,
                          "config_bytes_read": budget.used, "max_config_bytes": MAX_CONFIG_BYTES,
                          "max_total_config_bytes": MAX_TOTAL_CONFIG_BYTES,
                          "budget_exhausted": budget.exhausted}}
