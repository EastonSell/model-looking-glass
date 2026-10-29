"""Direct, no-checkpoint child for the targeted-edit source rehearsal.

The source map and parent-held leases are cooperative controls. This process
cannot authorize a study run and does not protect source after parent death.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import types


class RehearsalChildError(RuntimeError):
    pass


def _read_bounded(path, limit):
    with open(path, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RehearsalChildError("bounded source or report input is too large")
    return data


def _load_bootstrap(checkout, hashes):
    relative = "scripts/v2_source_bootstrap.py"
    path = checkout / relative
    data = _read_bounded(path, 8 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != hashes.get(relative):
        raise RehearsalChildError("source bootstrap differs from the committed inventory")
    module = types.ModuleType("_targeted_edit_checked_source_bootstrap")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RehearsalChildError("duplicate source inventory key")
        value[key] = item
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "sources", "venv-root", "dependency", "prefix", "study-root", "report",
                 "source-revision", "parent-platform"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    checkout = Path(args.checkout).resolve(strict=True)
    prefix = Path(args.prefix).resolve(strict=True)
    if (not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(name == "looking_glass" or name.startswith("looking_glass.") for name in sys.modules)
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != prefix
            or any(prefix.iterdir())):
        raise RehearsalChildError("isolated pre-claim startup differs")
    raw = _read_bounded(Path(args.sources), 2 * 1024 * 1024)
    hashes = json.loads(raw, object_pairs_hook=_unique_object)
    if type(hashes) is not dict:
        raise RehearsalChildError("source inventory must be an object")
    own = checkout / "scripts/targeted_edit_v2_source_child.py"
    if (Path(__file__).resolve(strict=True) != own
            or hashlib.sha256(_read_bounded(own, 8 * 1024 * 1024)).hexdigest()
               != hashes.get("scripts/targeted_edit_v2_source_child.py")):
        raise RehearsalChildError("child source differs from the committed inventory")
    bootstrap = _load_bootstrap(checkout, hashes)
    bootstrap._check_search_path(checkout)
    bootstrap._prime_parent_platform(json.loads(args.parent_platform))
    # platform.platform() also calls _syscmd_ver on this Windows Python even
    # after uname is primed; obtain the same version from the native runtime.
    import platform
    if os.name != "nt" or platform._platform_cache:
        raise RehearsalChildError("platform was observed before the spawn-free setup")
    win = sys.getwindowsversion()
    version = f"{win.major}.{win.minor}.{win.build}"
    platform._syscmd_ver = lambda *_args, **_kwargs: ("Windows", str(win.major), version)
    bootstrap._install_process_guard()
    project = {name: digest for name, digest in hashes.items()
               if name.startswith("looking_glass/") and name.endswith(".py")}
    finder = bootstrap._ProjectFinder(bootstrap._source_map(checkout, project))
    sys.meta_path.insert(0, finder)
    from looking_glass.cpu_study_supervision import claim_cpu_study, _FORBIDDEN_EARLY_MODULES

    def model_dependency_loaded():
        return any(name.partition(".")[0] in _FORBIDDEN_EARLY_MODULES for name in sys.modules)

    if model_dependency_loaded():
        raise RehearsalChildError("model dependency was imported before the CPU claim")
    session, lease = claim_cpu_study(Path(args.study_root))
    claim_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if claim_mask != "" or import_mask != "-1" or model_dependency_loaded():
        raise RehearsalChildError("the v1 mask transition failed")
    bootstrap._add_dependencies(checkout, Path(args.venv_root).resolve(strict=True), [args.dependency])
    import torch
    from looking_glass import targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_v2_edits as edits
    if (torch.cuda.is_initialized() or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled("cpu") or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
            or os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise RehearsalChildError("the CPU-only offline runtime differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise RehearsalChildError("denormal preservation is unavailable")
    before = v1._denormal_probe()
    environment = v1._environment()
    after = v1._denormal_probe()
    reference_path = checkout / "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"
    reference = _read_bounded(reference_path, 65536)
    if (hashlib.sha256(reference).hexdigest() != edits.V1_PRECOMMIT_SHA256
            or hashes.get("results/targeted-edit-cpu-v1-20261028/reference/precommit.json")
               != edits.V1_PRECOMMIT_SHA256):
        raise RehearsalChildError("the historical reference differs")
    expected = v1._decode_document(reference)["environment"]
    observed = {name: environment[name] for name in edits.ENVIRONMENT_FIELDS}
    if v1._canonical(observed) != v1._canonical({name: expected[name] for name in edits.ENVIRONMENT_FIELDS}):
        raise RehearsalChildError("the observed runtime differs from v1")
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized()
            or any(prefix.iterdir())):
        raise RehearsalChildError("CPU mask or bytecode prefix changed")
    inventory = finder.verify_inventory()
    bootstrap._check_search_path(checkout)
    report = {"status": "matching_v1_runtime", "source_revision": args.source_revision,
              "owner_pid": os.getpid(), "owner_run_id": session.run_id,
              "claim_mask": claim_mask, "model_import_mask": import_mask, "runtime": observed,
              "denormal_probe_before": before, "denormal_probe_after": after,
              "project_sources": inventory, "checkpoint_loaded": False, "model_forwarded": False}
    data = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    with open(args.report, "xb") as stream:
        if stream.write(data) != len(data):
            raise RehearsalChildError("short report write")
        stream.flush()
        os.fsync(stream.fileno())
    gc.collect()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_initialized():
        raise RehearsalChildError("the CPU state changed before attestation")
    session.attest_clean(lease)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
