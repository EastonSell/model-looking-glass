"""Run the fixed BERT multisuffix intervention study in one model process.

The caller must arrange current public-input checks and device coordination.
Each attempt uses a new output directory; partial attempts are never complete.
"""
import sys

if __name__ == "__main__" and (not sys.flags.isolated or not sys.flags.no_site
                               or not sys.dont_write_bytecode or not sys.pycache_prefix):
    raise RuntimeError("direct owner requires -I -S -B and a fresh pycache prefix")

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
import weakref

SOURCE_PATHS = (
    "scripts/bert_multisuffix_agreement_v1.py",
    "scripts/read_bert_multisuffix_agreement_v1.py",
    "scripts/observe_bert_multisuffix_agreement_v1.py",
    "scripts/cross_model_agreement_v1.py",
    "scripts/read_cross_model_agreement_v1.py",
    "docs/BERT_MULTISUFFIX_AGREEMENT_V1.md", "docs/BERT_MULTISUFFIX_EXECUTION_V1.md",
    "experiments/bert_multisuffix_agreement_cases_v1.json",
)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(commit, case_path, git):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    if not git.is_absolute() or not git.is_file():
        raise ValueError("Git executable must be an absolute file path")
    root = Path(__file__).resolve().parents[1]
    if git.resolve().is_relative_to(root):
        raise ValueError("Git executable must be selected outside the checkout")
    if case_path.resolve() != (root / SOURCE_PATHS[-1]).resolve():
        raise ValueError("case path must be the source-bound manifest")
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PATH") if key in os.environ}
    env.update({"GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"})
    kind = subprocess.run([str(git), "-C", str(root), "cat-file", "-t", commit], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env).stdout.strip()
    if kind != b"commit":
        raise ValueError("source identifier does not name a commit")
    result = {}
    for name in SOURCE_PATHS:
        data = subprocess.run([str(git), "-C", str(root), "cat-file", "blob", commit + ":" + name], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, env=env).stdout
        if (root / name).read_bytes() != data:
            raise ValueError(f"working source differs from the pinned commit: {name}")
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def bootstrap():
    if (__spec__ is not None or not sys.flags.isolated or not sys.flags.no_site
            or not sys.dont_write_bytecode or not sys.pycache_prefix
            or sys._xoptions.get("pycache_prefix") != sys.pycache_prefix
            or Path(sys.executable).resolve() != Path(sys._base_executable).resolve()):
        raise RuntimeError("run the direct base interpreter with -I -S -B and a fresh pycache prefix")
    prefix = Path(sys.pycache_prefix)
    if not prefix.is_absolute() or not prefix.is_dir() or prefix.is_symlink() or any(prefix.iterdir()):
        raise RuntimeError("pycache prefix must be a fresh empty ordinary directory")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--git", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args, _ = parser.parse_known_args()
    if args.device != "cpu":
        raise RuntimeError("direct CLI is CPU-only; CUDA requires a separately reviewed physical-lease owner")
    if not args.site.is_absolute() or not args.site.is_dir() or args.site.is_symlink():
        raise RuntimeError("site must be an absolute installed dependency directory")
    hashes = verify_source(args.source_commit, args.cases, args.git)
    git_sha = file_sha(args.git)
    # The verified dependency site is trusted, but it must not shadow stdlib.
    # The checkout itself is never added to the Python import search path.
    sys.path.append(str(args.site.resolve()))
    return hashes, git_sha


if __name__ == "__main__":
    EARLY_SOURCE, EARLY_GIT = bootstrap()
else:
    EARLY_SOURCE = EARLY_GIT = None

import numpy as np
import torch
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
if __name__ == "__main__":
    helper_path = Path(__file__).resolve().with_name("cross_model_agreement_v1.py")
    helper_spec = importlib.util.spec_from_file_location("_bert_suffix_verified_helper", helper_path)
    common = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(common)
    if Path(common.__file__).resolve() != helper_path:
        raise RuntimeError("helper module origin differs")
else:
    from scripts import cross_model_agreement_v1 as common


CASE_SHA = "bc8d1af844c52f46bf068c04257280ed37e4302028885540ea498103d6ce125e"
SEED = 20261029
OUTPUT_CAP = 192 * 1024**2
TAIL_RESERVE = 64 * 1024
VARIANTS = ("terminal", "period", "present", "outside", "awake")
MAX_EVENTS = 1088


def check_manifest(manifest, tokenizer):
    if (manifest.get("schema") != "bert-multisuffix-agreement-cases-v1"
            or manifest.get("targets") != common.TARGETS["bert"]
            or manifest.get("variants") != list(VARIANTS) or len(manifest.get("cases", [])) != 32):
        raise ValueError("manifest schema, targets or dimensions differ")
    cases = manifest["cases"]
    for i, case in enumerate(cases):
        if (case["index"] != i or case["same_donor_index"] != (i ^ 2)
                or case["opposite_donor_index"] != (i ^ 3)):
            raise ValueError("case order or donor mapping differs")
        same, opposite = (cases[case[key]] for key in ("same_donor_index", "opposite_donor_index"))
        if (same["lemma"] != opposite["lemma"] or same["lemma"] == case["lemma"]
                or same["number"] != case["number"] or opposite["number"] == case["number"]):
            raise ValueError("donor number or lemma differs")
        for variant in VARIANTS:
            row = case["variants"][variant]
            encoded = tokenizer(row["text"], add_special_tokens=True, truncation=False)
            ids = row["input_ids"]
            if (encoded["input_ids"] != ids or encoded["attention_mask"] != row["attention_mask"]
                    or row["subject_position"] != 6 or row["output_position"] != 10
                    or ids[10] != tokenizer.mask_token_id or ids[-1] != tokenizer.sep_token_id):
                raise ValueError("case does not retokenize at fixed sites")
            if ids[:11] != case["variants"]["terminal"]["input_ids"][:-1]:
                raise ValueError("paired prefix differs")
            for donor in (same, opposite):
                other = donor["variants"][variant]["input_ids"]
                if len(ids) != len(other) or [j for j, (x, y) in enumerate(zip(ids, other)) if x != y] != [6]:
                    raise ValueError("donor differs outside subject")


class DoseIneligible(ValueError):
    """A finite replacement cannot meet the declared rounded-dose tolerance."""


def landing(base, direction, q):
    base64 = np.asarray(base, dtype=np.float64)
    direction64 = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction64))
    if not math.isfinite(q) or not math.isfinite(norm) or not np.isfinite(base64).all():
        raise ValueError("nonfinite direction, dose or base")
    if q <= 0 or norm <= 0:
        raise DoseIneligible("zero direction or dose")
    intended = q * (direction64 / norm)
    value = np.asarray(base64 + intended, dtype="<f4")
    delta = value.astype(np.float64) - base64
    actual = float(np.linalg.norm(delta))
    error = float(np.linalg.norm(delta - intended)) / q
    if not np.isfinite(value).all() or not math.isfinite(actual) or not math.isfinite(error):
        raise ValueError("nonfinite rounded landing")
    if actual <= 0 or abs(actual - q) > 1e-4 * q or error > 1e-4:
        raise DoseIneligible("rounded landing misses dose tolerance")
    return value, actual, error


class Journal:
    def __init__(self, directory, vocab=30522):
        self.directory = directory
        self.starts = (directory / "starts.jsonl").open("xb", buffering=0)
        self.events = (directory / "events.jsonl").open("xb", buffering=0)
        self.samples = (directory / "memory.jsonl").open("xb", buffering=0)
        self.retention_errors = (directory / "retention-errors.jsonl").open("xb", buffering=0)
        self.logits = (directory / "logits.f32").open("xb", buffering=0)
        self.vectors = (directory / "witness.f32").open("xb", buffering=0)
        self.vocab, self.count = vocab, 0

    def _write(self, stream, data):
        used = sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        if used + len(data) + TAIL_RESERVE > OUTPUT_CAP:
            raise ValueError("retained output storage cap reached")
        if stream.write(data) != len(data):
            raise OSError("short retained-output write")

    def start(self, case, row, variant, arm):
        event = {"event_index": self.count, "case_id": case["case_id"], "case_index": case["index"],
                 "variant": variant, "arm": arm, "input_ids": row["input_ids"],
                 "attention_mask": row["attention_mask"], "subject_position": row["subject_position"],
                 "output_position": row["output_position"]}
        self._write(self.starts, (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode())
        os.fsync(self.starts.fileno())

    def write(self, event, logits, before, after):
        if self.count >= MAX_EVENTS:
            raise ValueError("fixed forward count reached")
        logits, before, after = (np.asarray(x, dtype="<f4") for x in (logits, before, after))
        if logits.shape != (self.vocab,) or before.shape != (768,) or after.shape != (768,):
            raise ValueError("unexpected selected logits or witness shape")
        event["logits_offset"], event["witness_offset"] = self.logits.tell(), self.vectors.tell()
        self._write(self.logits, logits.tobytes())
        self._write(self.vectors, before.tobytes() + after.tobytes())
        event["event_index"] = self.count
        self._write(self.events, (json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
        self.count += 1

    def sample(self, memory):
        self._write(self.samples, (json.dumps({"event_index": self.count - 1, "memory": memory},
                                              sort_keys=True, separators=(",", ":")) + "\n").encode())

    def retention_error(self, event, error):
        self._write(self.retention_errors, (json.dumps({"event_index": self.count, "case_id": event["case_id"],
                                                       "variant": event["variant"], "arm": event["arm"],
                                                       "reason": type(error).__name__}, sort_keys=True) + "\n").encode())

    def close(self):
        for stream in (self.starts, self.events, self.samples, self.retention_errors, self.logits, self.vectors):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not math.isfinite(na) or not math.isfinite(nb) or na <= 0 or nb <= 0:
        return None
    value = float(np.dot(a, b) / (na * nb))
    return value if math.isfinite(value) else None


def require_policy(device):
    common.require_subnormals()
    if (not torch.are_deterministic_algorithms_enabled() or torch.get_num_threads() != 1
            or torch.get_num_interop_threads() != 1 or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark
            or not torch.backends.cudnn.deterministic):
        raise RuntimeError("numerical execution policy drifted")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA workspace policy drifted")


def run(model, block, manifest, device, journal):
    cases, captures, clean = manifest["cases"], {}, {}
    for case in cases:
        for variant in VARIANTS:
            require_policy(device)
            row = case["variants"][variant]
            journal.start(case, row, variant, "clean")
            event = {"case_id":case["case_id"], "case_index":case["index"], "variant":variant, "arm":"clean"}
            logits, before, after, finite = common.forward(model, block, row, None, device, journal=journal, event=event)
            require_policy(device)
            journal.sample(common.memory_sample(device))
            if not finite or not np.isfinite(logits).all() or not np.isfinite(before).all() or not common.same_bits(before, after):
                raise ValueError("nonfinite or altered clean capture")
            captures[case["index"], variant], clean[case["index"], variant] = before, logits

    plans, planned = [], {}
    for case in cases:
        index = case["index"]
        donors = {"opposite":case["opposite_donor_index"], "same":case["same_donor_index"]}
        directions, norms = {}, {}
        for variant in VARIANTS:
            base = captures[index, variant].astype(np.float64)
            for arm, donor in donors.items():
                direction = captures[donor, variant].astype(np.float64) - base
                norm = float(np.linalg.norm(direction))
                if not math.isfinite(norm) or not np.isfinite(direction).all():
                    raise ValueError("nonfinite donor direction")
                directions[variant, arm], norms[variant + "_" + arm] = direction, norm
        q = min(2.0, *norms.values())
        cosines = {variant:{arm:cosine(directions["terminal", arm], directions[variant, arm])
                            for arm in ("opposite", "same")} for variant in VARIANTS[1:]}
        record = {"case_id":case["case_id"], "case_index":index, "eligible":True, "reason":None,
                  "requested_q":q, "natural_norms":norms, "direction_cosines":cosines,
                  "random_seed":SEED + index}
        random_direction = np.random.Generator(np.random.PCG64(SEED + index)).standard_normal(768)
        local = {}
        if q == 0:
            record.update(eligible=False, reason="zero_donor_distance")
        else:
            for variant in VARIANTS:
                if not record["eligible"]:
                    break
                base = captures[index, variant]
                entries = [("opposite", directions[variant, "opposite"]), ("same", directions[variant, "same"]),
                           ("random", random_direction)]
                if variant != "terminal":
                    entries.append(("fixed_terminal_opposite", directions["terminal", "opposite"]))
                for arm, direction in entries:
                    try:
                        local[variant, arm] = landing(base, direction, q)
                    except DoseIneligible:
                        record.update(eligible=False, reason="rounded_landing:" + variant + ":" + arm)
                        break
        plans.append(record)
        if record["eligible"]:
            planned[index] = local
    save_json(journal.directory / "plan.json", {"schema":"bert-multisuffix-dose-plan-v1", "cases":plans})

    for case in cases:
        index = case["index"]
        record = plans[index]
        if not record["eligible"]:
            continue
        for variant in VARIANTS:
            base, baseline = captures[index, variant], clean[index, variant]
            arms = ["self", "opposite", "same", "random"]
            if variant != "terminal":
                arms.append("fixed_terminal_opposite")
            arms.append("clean_after")
            for arm in arms:
                require_policy(device)
                donor_index, actual, error = None, 0.0, 0.0
                if arm == "self":
                    value, donor_index = base, index
                elif arm == "clean_after":
                    value = None
                else:
                    value, actual, error = planned[index][variant, arm]
                    if arm in ("opposite", "fixed_terminal_opposite"):
                        donor_index = case["opposite_donor_index"]
                    elif arm == "same":
                        donor_index = case["same_donor_index"]
                row = case["variants"][variant]
                journal.start(case, row, variant, arm)
                event = {"case_id":case["case_id"], "case_index":index, "variant":variant, "arm":arm,
                         "donor_index":donor_index, "requested_q":record["requested_q"],
                         "actual_l2":actual, "relative_direction_error":error,
                         "random_seed":SEED + index if arm == "random" else None}
                logits, before, after, finite = common.forward(model, block, row, value, device, journal=journal, event=event)
                require_policy(device)
                journal.sample(common.memory_sample(device))
                if (not finite or not np.isfinite(logits).all() or not np.isfinite(before).all()
                        or not np.isfinite(after).all() or not common.same_bits(before, base)):
                    raise ValueError("nonfinite output or changed clean source")
                if value is not None and not common.same_bits(after, value):
                    raise ValueError("replacement did not land")
                if arm in ("self", "clean_after") and not common.same_bits(logits, baseline):
                    raise ValueError("clean/self/clean-after selected logits differ")
    eligible = sum(record["eligible"] for record in plans)
    if journal.count != 160 + 29 * eligible:
        raise ValueError("unexpected forward count")
    return eligible


def save_json(path, obj):
    data = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file()) + len(data) > OUTPUT_CAP:
        raise ValueError("retained output storage cap reached")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def save_failure(directory, message):
    """Retain a failure marker without exceeding the remaining attempt budget.

    The journal is closed before this call. Account for existing direct files,
    reserve room for a truncation marker, and preserve even a zero-byte marker
    when no budget remains. The attempt directory has a single writer.
    """
    used = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    remaining = max(0, OUTPUT_CAP - used)
    marker = b"\n[traceback truncated to fit output cap]\n"

    # Count in bounded chunks, stopping as soon as the message is too large.
    length = 0
    truncated = False
    for pos in range(0, len(message), 4096):
        length += len(message[pos:pos + 4096].encode("utf-8", errors="backslashreplace"))
        if length > remaining:
            truncated = True
            break
    budget = remaining if not truncated else max(0, remaining - len(marker))
    with (directory / "failure.txt").open("xb") as stream:
        written = 0
        for pos in range(0, len(message), 4096):
            if written >= budget:
                break
            data = message[pos:pos + 4096].encode("utf-8", errors="backslashreplace")
            clipped = len(data) > budget - written
            data = data[:budget - written].decode("utf-8", errors="ignore").encode("utf-8")
            if stream.write(data) != len(data):
                raise OSError("short failure-message write")
            written += len(data)
            if clipped:
                break
        if truncated:
            suffix = marker[:remaining - written]
            if stream.write(suffix) != len(suffix):
                raise OSError("short failure-marker write")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("site", "git", "model-dir", "cases", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.device != "cpu":
        raise RuntimeError("direct CLI is CPU-only; CUDA requires a separately reviewed physical-lease owner")
    raw = args.cases.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CASE_SHA:
        raise ValueError("case manifest differs")
    manifest = json.loads(raw)
    source_hashes = verify_source(args.source_commit, args.cases, args.git)
    if source_hashes != EARLY_SOURCE or file_sha(args.git) != EARLY_GIT:
        raise ValueError("source or Git executable changed after early verification")
    assets = common.verify_assets(args.model_dir, "bert")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    check_manifest(manifest, tokenizer)
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUDA workspace configuration was not fixed before startup")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.set_flush_denormal(False) is not True:
        raise RuntimeError("cannot request subnormal-preserving arithmetic")
    require_policy(device)
    args.output.mkdir(parents=False, exist_ok=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    metadata = {"schema": "bert-multisuffix-agreement-attempt-v1", "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "case_sha256": CASE_SHA, "assets_sha256": assets, "sources_sha256": source_hashes,
                "source_commit": args.source_commit, "seed": SEED, "pid": os.getpid(),
                "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                "numpy": np.__version__, "git_sha256": EARLY_GIT, "start_unix": time.time(),
                "initial_memory": common.memory_sample(device),
                "base_interpreter_direct": True, "isolated": bool(sys.flags.isolated),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "tf32": torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "intra_threads": torch.get_num_threads(), "inter_threads": torch.get_num_interop_threads(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device.type == "cuda" else None}
    save_json(args.output / "metadata.json", metadata)
    journal = Journal(args.output)
    model = block = None
    try:
        model, loading = AutoModelForMaskedLM.from_pretrained(
            args.model_dir, local_files_only=True, use_safetensors=True, output_loading_info=True,
            torch_dtype=torch.float32, attn_implementation="eager")
        keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if not all(key in loading and type(loading[key]) is list and len(loading[key]) == 0 for key in keys):
            raise ValueError("checkpoint loading diagnostics were not empty")
        save_json(args.output / "loading.json", {key: loading[key] for key in keys})
        model = model.to(device)
        model.eval()
        block = model.distilbert.transformer.layer[3]
        eligible = run(model, block, manifest, device, journal)
        journal.close()
        if (common.verify_assets(args.model_dir, "bert") != assets
                or verify_source(args.source_commit, args.cases, args.git) != source_hashes
                or file_sha(args.git) != EARLY_GIT):
            raise ValueError("asset or source changed during execution")
        ref = weakref.ref(model)
        del block
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        require_policy(device)
        if ref() is not None:
            raise RuntimeError("model remains referenced after cleanup")
        result = {"status": "complete", "events": journal.count, "eligible_cases": eligible, "model_weakref_dead": True,
                  "final_memory": common.memory_sample(device), "end_unix": time.time()}
        for filename in ("metadata.json", "loading.json", "starts.jsonl", "events.jsonl", "memory.jsonl",
                         "retention-errors.jsonl", "logits.f32", "witness.f32", "plan.json"):
            result[filename.replace(".", "_") + "_sha256"] = common.sha256_file(args.output / filename)
        save_json(args.output / "complete.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException:
        try:
            journal.close()
        except Exception:
            pass
        try:
            save_failure(args.output, traceback.format_exc())
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
