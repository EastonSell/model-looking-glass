"""Source and prelaunch binding for one multi-subtoken CPU pass.

Git and receipt checks establish local saved-byte consistency only. The
operator must separately observe publication, current exposure/custody,
direct-child exit and an independent scientific replay. Directories and the
interpreter are trusted; these checks are not execution authentication.
"""

import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys


PROTOCOL_SHA256 = "bdb249673dc9045b8e83f4e5f525cc04e530f5f78f7a68fe72d26447e01fdb12"
PROTOCOL_PATH = "docs/FLAN_T5_MULTISUBTOKEN_SENTIMENT_V1.md"
ASSET_MANIFEST_SHA256 = "d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083"
MAX_INPUT_BYTES = 8 * 1024**2
SOURCE_PATHS = (
    PROTOCOL_PATH, "docs/FLAN_T5_MULTISUBTOKEN_CPU_OWNER_V2.md",
    "looking_glass/__init__.py", "looking_glass/bert_matched_numeric.py",
    "looking_glass/flan_t5_feasibility.py", "looking_glass/flan_t5_sentiment_journal.py",
    "looking_glass/flan_t5_sentiment_numeric.py", "looking_glass/flan_t5_sentiment_runner.py",
    "looking_glass/flan_multitoken_geometry_v2.py", "looking_glass/flan_multitoken_journal_v2.py",
    "looking_glass/flan_multitoken_manifest_v2.py", "looking_glass/flan_multitoken_orthogonal_v2.py",
    "looking_glass/flan_multitoken_owner_v2.py", "looking_glass/flan_multitoken_runner_v2.py",
    "looking_glass/flan_multitoken_schedule_v2.py", "looking_glass/flan_multitoken_scoring_v2.py",
    "looking_glass/flan_multitoken_span_v2.py", "tools/run_flan_sentiment_selector_stage_v1.py",
    "tools/run_flan_multitoken_sentiment_v2.py",
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(repo, *args):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_REPLACE_OBJECTS="1",
               GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        return subprocess.check_output(["git", "--no-replace-objects", "-C", str(repo), *args],
                                       env=env, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Git source object unavailable") from error


def _revision(value):
    return type(value) is str and _HEX40.fullmatch(value) is not None


def _path(value):
    return (type(value) is str and 0 < len(value.encode("utf8")) <= 256
            and "\\" not in value and ":" not in value and "\0" not in value
            and not PurePosixPath(value).is_absolute()
            and all(part not in ("", ".", "..") for part in value.split("/")))


def _git_blob(repo, revision, path):
    if not _revision(revision) or not _path(path) or _git(repo, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("invalid Git descriptor")
    spec = revision + ":" + path
    size = _git(repo, "cat-file", "-s", spec).strip()
    if not size.isdigit() or int(size) > MAX_INPUT_BYTES:
        raise ValueError("Git blob exceeds its bound")
    raw = _git(repo, "cat-file", "blob", spec)
    if len(raw) != int(size):
        raise ValueError("Git blob length differs")
    return raw


def source_snapshot(repo, revision, *, paths=SOURCE_PATHS):
    """Bind the current checkout and imported project sources to one commit."""
    root = Path(repo)
    if (not root.is_absolute() or not root.is_dir() or not _revision(revision)
            or type(paths) not in (tuple, list) or not paths or len(set(paths)) != len(paths)
            or any(not _path(path) for path in paths)):
        raise ValueError("invalid source checkout")
    if _git(root, "rev-parse", "HEAD").strip().decode("ascii") != revision:
        raise ValueError("checkout revision differs")
    sources = {}
    for name in paths:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("source file is absent, aliased or too large")
        raw = path.read_bytes()
        if raw != _git_blob(root, revision, name):
            raise ValueError("source file differs from committed blob")
        sources[name] = _sha(raw)
        if name == PROTOCOL_PATH and sources[name] != PROTOCOL_SHA256:
            raise ValueError("committed protocol differs from the frozen protocol")
    return {"revision": revision, "source_sha256": sources}


def _parse(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_INPUT_BYTES:
        raise ValueError("binding input size differs")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    try:
        value = json.loads(raw.decode("utf8"), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        expected = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n").encode("utf8")
    except (ValueError, TypeError, UnicodeError, OverflowError) as error:
        raise ValueError("invalid binding JSON") from error
    if type(value) is not dict or raw != expected:
        raise ValueError("binding JSON is not canonical")
    return value


def _descriptor(repo, value, *, expected_raw=None):
    if (type(value) is not dict or set(value) != {"path", "revision", "sha256"}
            or not _path(value["path"]) or not _revision(value["revision"])
            or type(value["sha256"]) is not str or _HEX64.fullmatch(value["sha256"]) is None):
        raise ValueError("invalid public descriptor")
    raw = _git_blob(repo, value["revision"], value["path"])
    if _sha(raw) != value["sha256"] or (expected_raw is not None and raw != expected_raw):
        raise ValueError("public descriptor bytes differ")
    return raw


def verify_binding(repo, revision, binding_raw, discovery_raw, painting_raw, review_raw):
    """Bind manifests, scoped review and reader files to supplied Git objects."""
    binding = _parse(binding_raw)
    keys = {"schema", "source_revision", "protocol_sha256", "asset_manifest_sha256",
            "discovery", "painting", "selection_review", "reader_sources"}
    if (set(binding) != keys or binding["schema"] != "flan-multisubtoken-owner-binding-v1"
            or not _revision(revision) or binding["source_revision"] != revision
            or binding["protocol_sha256"] != PROTOCOL_SHA256
            or binding["asset_manifest_sha256"] != ASSET_MANIFEST_SHA256):
        raise ValueError("owner binding identity differs")
    for name, raw in (("discovery", discovery_raw), ("painting", painting_raw), ("selection_review", review_raw)):
        _descriptor(repo, binding[name], expected_raw=raw)
    review = _parse(review_raw)
    if (set(review) != {"schema", "status", "discovery_sha256", "painting_sha256", "scope"}
            or review["schema"] != "flan-multisubtoken-selection-review-v1" or review["status"] != "consistent"
            or review["discovery_sha256"] != _sha(discovery_raw) or review["painting_sha256"] != _sha(painting_raw)
            or type(review["scope"]) is not str or not 1 <= len(review["scope"].encode("utf8")) <= 4096):
        raise ValueError("selection review does not bind the manifests")
    readers = binding["reader_sources"]
    if type(readers) is not list or len(readers) > 8:
        raise ValueError("reader source inventory differs")
    identities = set()
    for descriptor in readers:
        _descriptor(repo, descriptor)
        identity = (descriptor["revision"], descriptor["path"])
        if identity in identities:
            raise ValueError("duplicate reader source")
        identities.add(identity)
    return {"source_revision": revision, "binding_sha256": _sha(binding_raw),
            "discovery_sha256": _sha(discovery_raw), "painting_sha256": _sha(painting_raw),
            "selection_review_sha256": _sha(review_raw), "reader_sources": readers,
            "scope": "Local Git and saved-byte consistency only; publication and review are separate observations."}


def _read(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("bound input is absent, aliased or too large")
    raw = path.read_bytes()
    if not 1 <= len(raw) <= MAX_INPUT_BYTES:
        raise ValueError("bound input size differs")
    return raw


def _write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")
    with Path(path).open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short owner receipt write")
        stream.flush()
        os.fsync(stream.fileno())


def _stamps(paths):
    result = {}
    for name, value in paths.items():
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise ValueError("watched file is absent or aliased")
        stat = path.stat()
        result[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    return result


def required_free_bytes(subject):
    """Keep remaining pass caps, owner reserve and 1 GiB headroom available."""
    required = {"story": 9 * 1024**3 // 2, "concert": 13 * 1024**3 // 4,
                "painting": 2 * 1024**3}
    if type(subject) is not str or subject not in required:
        raise ValueError("invalid subject")
    return required[subject]


def ensure_output_directory(path):
    """Check the resolved output directory on the authorized scratch volume.

    This is a point-in-time check under a trusted, quiescent filesystem
    namespace; it does not reserve space or prevent later path replacement.
    """
    path = Path(path)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("output volume directory must exist and be absolute")
    resolved = path.resolve(strict=True)
    if os.name == "nt":
        scratch = Path("E:/Codex/model-looking-glass/work").resolve(strict=True)
        if resolved.drive.upper() != "E:" or not resolved.is_relative_to(scratch):
            raise ValueError("resolved output volume is outside the authorized E: scratch directory")
    return resolved


def run(args):
    """Execute one granted CPU child; its terminal is a provisional candidate."""
    # Imports of Torch/Transformers occur only in the granted child, after the
    # bootstrap has added the explicitly verified venv site-packages path.
    from .flan_t5_feasibility import verify_assets
    from .flan_multitoken_journal_v2 import EvidenceJournalV2, LimitsV2, require_protocol_allocation
    from .flan_multitoken_manifest_v2 import project_manifests

    repo, assets, attempt = Path(args.repo), Path(args.asset_dir), Path(args.attempt)
    # Refuse an invalid output path before attempting even a failure receipt.
    ensure_output_directory(attempt)
    receipt = attempt / "owner-receipt.json"
    journal = None
    phase = "preflight"
    try:
        if not repo.is_absolute() or not assets.is_absolute():
            raise ValueError("owner paths or E: output volume differ")
        if (type(args.binding_sha256) is not str or _HEX64.fullmatch(args.binding_sha256) is None
                or not _revision(args.revision)):
            raise ValueError("invalid external binding pin")
        required_env = {"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE"}
        if any(os.environ.get(name) != value for name, value in required_env.items()):
            raise ValueError("CPU/offline environment differs")
        source = source_snapshot(repo, args.revision)
        paths = {name: getattr(args, name) for name in ("binding", "discovery", "painting", "selection_review")}
        raw = {name: _read(path) for name, path in paths.items()}
        if _sha(raw["binding"]) != args.binding_sha256:
            raise ValueError("external binding digest differs")
        binding = verify_binding(repo, args.revision, raw["binding"], raw["discovery"],
                                 raw["painting"], raw["selection_review"])
        if platform.python_version() != "3.11.15":
            raise ValueError("fixed Gaussian runtime differs")
        asset_manifest = verify_assets(assets)
        phase = "tokenizer_preflight"
        import torch
        import numpy
        import transformers
        import tokenizers
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        site = Path(args.venv) / ("Lib/site-packages" if os.name == "nt" else
                                  "lib/python%d.%d/site-packages" % sys.version_info[:2])
        site = site.resolve(strict=True)
        if not all(Path(module.__file__).resolve().is_relative_to(site)
                   for module in (torch, numpy, transformers, tokenizers)):
            raise ValueError("runtime libraries do not come from the verified venv")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        if not torch.set_flush_denormal(False):
            raise ValueError("cannot preserve CPU denormals")
        tokenizer = AutoTokenizer.from_pretrained(str(assets), local_files_only=True,
                                                   use_fast=True, trust_remote_code=False)
        if (type(tokenizer).__name__ != "T5TokenizerFast" or tokenizer.pad_token_id != 0
                or tokenizer.eos_token_id != 1):
            raise ValueError("tokenizer identity differs")
        groups = project_manifests(raw["discovery"], raw["painting"], tokenizer,
                                   vocab_size=32128, special_ids=set(tokenizer.all_special_ids))
        if args.subject not in groups:
            raise ValueError("invalid subject")
        # An empty selected pass is a recorded no-model outcome, not a retry or
        # an invitation to fill a missing original slot.
        if not groups[args.subject]:
            if source_snapshot(repo, args.revision) != source or verify_assets(assets) != asset_manifest:
                raise ValueError("source or assets changed during preflight")
            if any(_read(paths[name]) != value for name, value in raw.items()):
                raise ValueError("binding input changed during preflight")
            result = {"schema": "flan-multisubtoken-owner-receipt-v1", "subject": args.subject,
                      "status": "no-selected-groups", "binding_sha256": args.binding_sha256,
                      "source_revision": args.revision, "model_loaded": False,
                      "scope": "Saved-file consistency only; no model forward was requested."}
            _write_new(receipt, result)
            return result
        if shutil.disk_usage(attempt).free < required_free_bytes(args.subject):
            raise ValueError("insufficient E: capacity for remaining passes and headroom")
        limits = LimitsV2.for_subject(args.subject)
        require_protocol_allocation(limits)
        journal = EvidenceJournalV2(attempt / "child", {
            "schema": "flan-multisubtoken-owner-start-v1", "subject": args.subject,
            "source": source, "binding": binding, "asset_manifest_sha256": ASSET_MANIFEST_SHA256,
            "scope": "Saved bytes and cooperative direct-child observation; not execution authentication."},
            limits=limits)
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": numpy.__version__, "transformers": transformers.__version__,
                   "tokenizers": tokenizers.__version__, "threads": torch.get_num_threads(),
                   "interop_threads": torch.get_num_interop_threads(),
                   "deterministic": torch.are_deterministic_algorithms_enabled(),
                   "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "torch_build_sha256": _sha(torch.__config__.show().encode("utf8")),
                   "cpu_architecture": platform.machine(), "cpu_count": os.cpu_count(),
                   "byte_order": sys.byteorder, "python_float_rounds": sys.float_info.rounds,
                   "numeric_environment": {name: os.environ.get(name) for name in
                       ("MKL_CBWR", "ATEN_CPU_CAPABILITY", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
                   "environment": required_env, "binding": binding}
        journal.record("owner_contract", runtime)
        phase = "model_load"
        if shutil.disk_usage(attempt).free < required_free_bytes(args.subject):
            raise ValueError("insufficient E: capacity immediately before model loading")
        ensure_output_directory(attempt)
        model, info = T5ForConditionalGeneration.from_pretrained(
            str(assets), local_files_only=True, use_safetensors=True, trust_remote_code=False,
            output_loading_info=True, dtype=torch.float32, attn_implementation="eager")
        if any(info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("model loading diagnostics differ")
        model.eval()
        model.config.use_cache = False
        config = model.config
        if (type(model).__name__ != "T5ForConditionalGeneration" or config.d_model != 512
                or config.vocab_size != 32128 or config.decoder_start_token_id != 0
                or config.pad_token_id != 0 or config.eos_token_id != 1
                or getattr(config, "_attn_implementation", None) != "eager" or torch.cuda.is_initialized()):
            raise ValueError("model configuration differs")
        from .flan_multitoken_runner_v2 import run_cpu_pass
        watched = {**paths, **{"source:" + name: repo / name for name in SOURCE_PATHS},
                   **{"asset:" + name: assets / name for name in asset_manifest["files"]},
                   "asset:manifest": assets / "asset-manifest.json"}
        stamps = _stamps(watched)
        def custody_check():
            ensure_output_directory(attempt)
            if _stamps(watched) != stamps:
                raise ValueError("source, assets or binding input changed")
        phase = "forwards"
        result = run_cpu_pass(model, tokenizer, groups[args.subject], journal,
                              subject=args.subject, custody_check=custody_check)
        phase = "postflight"
        custody_check()
        if source_snapshot(repo, args.revision) != source or verify_assets(assets) != asset_manifest:
            raise ValueError("source or assets changed during pass")
        if any(_read(paths[name]) != value for name, value in raw.items()):
            raise ValueError("binding input changed during pass")
        _write_new(receipt, {"schema": "flan-multisubtoken-owner-receipt-v1", "subject": args.subject,
                             "status": "complete_candidate", "binding_sha256": args.binding_sha256,
                             "source_revision": args.revision, "model_loaded": True,
                             "terminal_sha256": _sha((attempt / "child/terminal.json").read_bytes()),
                             "events_sha256": _sha((attempt / "child/events.jsonl").read_bytes()),
                             "scope": "Saved-file consistency only; independent scientific replay is required."})
        return result
    except BaseException as primary:
        cleanup_interrupt = None
        if journal is not None and not (attempt / "child/terminal.json").exists():
            try:
                journal.finish("incomplete", {"subject": args.subject, "phase": phase,
                                              "error_type": type(primary).__name__})
            except BaseException as error:
                if isinstance(primary, Exception) and not isinstance(error, Exception):
                    cleanup_interrupt = error
        if not receipt.exists():
            try:
                _write_new(receipt, {"schema": "flan-multisubtoken-owner-receipt-v1",
                                     "subject": args.subject, "status": "incomplete", "phase": phase,
                                     "source_revision": args.revision, "error_type": type(primary).__name__})
            except BaseException as error:
                if cleanup_interrupt is None and isinstance(primary, Exception) and not isinstance(error, Exception):
                    cleanup_interrupt = error
        if cleanup_interrupt is not None:
            raise cleanup_interrupt from primary
        raise
