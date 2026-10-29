"""Cooperative one-shot CPU baseline owner; no pretrained work at import."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import sys
import types

CASE_SHA256 = "c8dd418bea948c8ca2b4d1e39e696374a28bd2f911bb16707415c38170fbba82"
REFERENCE_SHA256 = "cb666aa911db020b8904dab6f08f989e506d85e875d7cac65e72d3b94cef2fa5"
PUBLIC_SHA256 = "49453de1a125bcfe10d53ea659191e5cf28b66e852273f3d6541cb8e6489f9fc"
V44_SHA256 = "b118ae05419245fe77a63187c5e1640bcdd648c770a538d7eab1e2dfd464bab8"
MASK_SHA256 = "76f9d0ca003402cbeef4509d34b99e5df3c791bdc059f65914db574ac30d3a89"
SUPPORT_SHA256 = "efdf5d9c8bdfa8f6e227cf9ee8b040099f5f419c20c1faa51cd89cf6e5fe3a18"
COPY_SHA256 = "69c0c62e8888a220252fecefcade6c90a6141b5b50b8805983ff77858abd263b"
PROTOCOL_SHA256 = "6401093b4b80ee041a4799fbdfcc9f428c64ea5250df5108b2a96b5518600c51"
STUDY_ROOT_SHA256 = "99d506fa03b449e14eaa7d40e8b0b5e6c554b389292573f3e3586dd1f1fbee1f"
CAPACITY = {"c_free": 2 * 1024**3, "e_free": 3 * 1024**3, "physical_free": 3 * 1024**3}
REHEARSAL_CAPACITY = {"c_free": 256 * 1024**2, "e_free": 512 * 1024**2, "physical_free": 2 * 1024**3}
OWNER = "scripts/france_baseline_cpu_owner_v2.py"
OBSERVER = "scripts/observe_france_baseline_v2.py"
BOOT = "scripts/v2_source_bootstrap.py"
GUARDS = "scripts/v2_source_guards.py"
LAUNCHER = "scripts/v2_source_launcher.py"
CASES = "experiments/france_full_top_cases_v3.json"
REFERENCE = "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"
PUBLIC = "experiments/france_full_top_public_exposure_v4.json"
V44 = "experiments/france_full_top_public_v44_addendum_v1.json"
MASK = "experiments/france_full_top_public_mask_addendum_v1.json"
SUPPORT = "experiments/france_full_top_public_support_v1.json"
COPY = "experiments/france_full_top_public_copy_addendum_v1.json"
PROTOCOL = "docs/TARGETED_EDIT_FRANCE_FULL_TOP_V4.md"
_HELD = []


class OwnerError(RuntimeError):
    pass


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise OwnerError("duplicate public receipt key")
        value[key] = item
    return value


def _nonfinite(_value):
    raise OwnerError("nonfinite public receipt number")


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise OwnerError("nonfinite public receipt float")
    return value


def check_public_report(raw, expected_sha):
    if (type(raw) is not bytes or len(raw) > 1_000_000 or type(expected_sha) is not str
            or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha)
            or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise OwnerError("public receipt bytes differ")
    try:
        report = json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite, parse_float=_float)
        if (type(report) is not dict or report["schema"] != "france-v4-public-exposure-independent-v2"
                or report["status"] != "no_collision_in_named_scope"
                or type(report["pins_sha256"]) is not dict or report["pins_sha256"]["france"] != CASE_SHA256
                or type(report["study_rows"]) is not int or report["study_rows"] != 27
                or type(report["projection_basis_rows"]) is not int or report["projection_basis_rows"] != 4
                or type(report["study_sequences"]) is not int or report["study_sequences"] != 73
                or type(report["unexempted_inventory_groups"]) is not int
                or report["unexempted_inventory_groups"] != 0):
            raise OwnerError("public receipt case or scope differs")
        bundles = report["additional_bundle_results"]
        if type(bundles) is not dict or set(bundles) != {"adapter", "causal"}:
            raise OwnerError("public pending bundles differ")
        for name, count in (("adapter", 80), ("causal", 612)):
            row = bundles[name]
            if (type(row) is not dict or type(row.get("matches")) is not list or row["matches"]
                    or type(row.get("sequences")) is not int or row["sequences"] != count):
                raise OwnerError("public pending comparison differs")
        return report
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        if isinstance(error, OwnerError):
            raise
        raise OwnerError("malformed public receipt") from error


def check_capacity_values(c_free, e_free, physical_free):
    values = {"c_free": c_free, "e_free": e_free, "physical_free": physical_free}
    if any(type(value) is not int or value < CAPACITY[name] for name, value in values.items()):
        raise OwnerError("fresh disk or physical-memory capacity is insufficient")
    return True


def verify_tokenizer_rows(tokenizer, rows):
    """Recheck the actual tokenizer, including every appended boundary."""
    for row in rows:
        encoded = tokenizer(row["text"], add_special_tokens=False, truncation=False,
                            return_attention_mask=True, padding=False)
        if encoded["input_ids"] != row["token_ids"] or encoded["attention_mask"] != row["attention_mask"]:
            raise OwnerError("tokenizer model-entry IDs or mask differ")
        for continuation in row["continuations"]:
            suffix, token = continuation["text"], continuation["token_id"]
            if (type(token) is not int or tokenizer.encode(suffix, add_special_tokens=False) != [token]
                    or tokenizer.encode(row["text"] + suffix, add_special_tokens=False)
                       != row["token_ids"] + [token]
                    or continuation["appended_token_ids"] != row["token_ids"] + [token]):
                raise OwnerError("tokenizer continuation boundary differs")


def check_result(result, job, child, source_sha, mode, source_revision):
    if (type(result.pid) is not int or result.pid <= 0 or type(result.run_id) is not str
            or len(result.run_id) != 32 or any(c not in "0123456789abcdef" for c in result.run_id)
            or type(result.exit_code) is not int or result.exit_code != 0
            or result.clean is not True or result.timed_out is not False
            or job._handle is not None or job._confirmed is not True
            or type(child) is not dict or child.get("status") != "complete"
            or type(child.get("pid")) is not int or child["pid"] != result.pid
            or child.get("run_id") != result.run_id or child.get("source_sha256") != source_sha
            or child.get("source_revision") != source_revision
            or child.get("case_sha256") != CASE_SHA256 or child.get("mode") != mode
            or child.get("prefix_empty") is not True or child.get("model_released") is not True):
        raise OwnerError("direct child, cleanup or saved receipt differs")
    details = child.get("details")
    if type(details) is not dict:
        raise OwnerError("child details are missing")
    if mode == "rehearsal":
        if details.get("checkpoint_loaded") is not False or details.get("model_forwarded") is not False:
            raise OwnerError("rehearsal did not report a no-model, no-forward result")
    elif mode == "baseline":
        identity = details.get("identity")
        if (type(identity) is not dict or identity.get("run_id") != result.run_id
                or identity.get("source_revision") != source_revision
                or identity.get("source_sha256") != source_sha or identity.get("case_sha256") != CASE_SHA256
                or any(type(details.get(name)) is not str
                       or re.fullmatch(r"[0-9a-f]{64}", details[name]) is None
                       for name in ("screen_sha256", "journal_sha256"))):
            raise OwnerError("baseline saved identity or evidence hashes differ")
    else:
        raise OwnerError("unsupported result mode")


def _read(path, limit=8 * 1024 * 1024):
    with open(path, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise OwnerError("bounded input exceeds its limit")
    return data


def _document(data):
    return json.loads(data, object_pairs_hook=_unique, parse_constant=_nonfinite, parse_float=_float)


def _write(path, value):
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with open(path, "xb") as stream:
        if stream.write(data) != len(data):
            raise OwnerError("short receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _source_relative(name):
    if type(name) is not str or "\\" in name or PureWindowsPath(name).drive:
        raise OwnerError("source inventory path has a Windows drive or separator")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or relative.as_posix() != name or len(relative.parts) < 2
            or any(part in ("", ".", "..") for part in relative.parts)):
        raise OwnerError("source inventory path is unsafe")
    return relative


def _source_map(root, path, expected):
    raw = _read(path, 2 * 1024 * 1024)
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or hashlib.sha256(raw).hexdigest() != expected:
        raise OwnerError("source inventory digest differs")
    values = _document(raw)
    if type(values) is not dict or not 1 <= len(values) <= 128:
        raise OwnerError("source inventory is not bounded")
    for name, digest in values.items():
        if type(name) is not str or type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise OwnerError("source inventory entry differs")
        relative = _source_relative(name)
        file = root.joinpath(*relative.parts)
        for entry in (file, *file.parents):
            if entry == root.parent:
                break
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise OwnerError("source inventory contains a reparse point")
        if (not file.is_file() or not file.resolve(strict=True).is_relative_to(root)
                or hashlib.sha256(_read(file)).hexdigest() != digest):
            raise OwnerError("source inventory bytes differ")
    required = {OWNER, OBSERVER, BOOT, GUARDS, LAUNCHER, CASES, REFERENCE, PUBLIC, V44, MASK, SUPPORT, COPY, PROTOCOL,
                "looking_glass/__init__.py", "looking_glass/cpu_study_supervision.py",
                "looking_glass/supervisor.py", "looking_glass/device_lease.py",
                "looking_glass/device_identity.py", "looking_glass/pretrained.py",
                "looking_glass/targeted_edit_france_baseline_v2.py",
                "looking_glass/targeted_edit_france_baseline_once_v2.py"}
    if not required <= set(values) or values[CASES] != CASE_SHA256 or values[REFERENCE] != REFERENCE_SHA256:
        raise OwnerError("required source or case identity differs")
    if (values[PUBLIC] != PUBLIC_SHA256 or values[V44] != V44_SHA256 or values[MASK] != MASK_SHA256
            or values[SUPPORT] != SUPPORT_SHA256 or values[COPY] != COPY_SHA256
            or values[PROTOCOL] != PROTOCOL_SHA256):
        raise OwnerError("reviewed public-only overlap receipts differ")
    return values


def _module(root, name, hashes):
    path = root.joinpath(*PurePosixPath(name).parts)
    data = _read(path)
    if hashlib.sha256(data).hexdigest() != hashes[name]:
        raise OwnerError("checked source changed before import")
    module = types.ModuleType("_france_checked_" + Path(name).stem)
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _install_project(root, hashes, boot):
    project = {name: sha for name, sha in hashes.items()
               if name.startswith("looking_glass/") and name.endswith(".py")}
    finder = boot._ProjectFinder(boot._source_map(root, project))
    sys.meta_path.insert(0, finder)
    return finder


def _capacity(mode="baseline"):
    import ctypes
    from ctypes import wintypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong)]
    value = MemoryStatus(); value.length = ctypes.sizeof(value)
    if os.name != "nt" or not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OwnerError("physical memory observation failed")
    observed = {"c_free": shutil.disk_usage("C:/").free, "e_free": shutil.disk_usage("E:/").free,
                "physical_free": int(value.available)}
    if mode == "baseline":
        check_capacity_values(**observed)
    elif mode == "rehearsal":
        if any(observed[key] < minimum for key, minimum in REHEARSAL_CAPACITY.items()):
            raise OwnerError("fresh no-model rehearsal capacity is insufficient")
    else:
        raise OwnerError("capacity mode differs")
    return observed


def _startup(root, prefix, *, child):
    if (os.name != "nt" or not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode)
            or sys.pycache_prefix is None or Path(sys.pycache_prefix).resolve(strict=True) != prefix
            or any(prefix.iterdir())
            or any(n == "looking_glass" or n.startswith("looking_glass.") for n in sys.modules)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ("" if child else "-1")):
        raise OwnerError("isolated startup, cache prefix or CPU visibility differs")
    if Path(__file__).resolve(strict=True) != root / OWNER:
        raise OwnerError("owner is not the source-image file")


def _study_root(value):
    root = Path(value).resolve(strict=True)
    digest = hashlib.sha256(os.path.normcase(str(root)).encode("utf-8")).hexdigest()
    if not root.is_dir() or digest != STUDY_ROOT_SHA256:
        raise OwnerError("the persistent France study root differs")
    return root


def _run_model(args, session, hashes, source_sha, prefix):
    import gc
    import weakref
    import torch
    from looking_glass import pretrained, targeted_edit_cpu_v1 as v1
    from looking_glass import targeted_edit_v2_owner as custody
    from looking_glass import targeted_edit_france_baseline_v2 as baseline
    from looking_glass import targeted_edit_france_baseline_once_v2 as once
    if (sys.byteorder != "little" or torch.cuda.is_initialized() or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled("cpu") or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"):
        raise OwnerError("CPU-only model environment differs")
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    if not torch.set_flush_denormal(False):
        raise OwnerError("denormal preservation is unavailable")
    before = v1._denormal_probe()
    environment = v1._environment()
    reference = v1._decode_document(_read(Path(args.source) / REFERENCE, 65536))
    capacity = _capacity()
    loaded = pretrained.load_distilgpt2(cache_dir=args.cache, allow_download=False, stage_dir=args.stage)
    model, module, weight = custody._check_checkpoint(loaded, {"v1_compatibility": {"model": reference["model"]}}, torch, v1)
    raw_cases = _read(Path(args.source) / CASES, 200_000)
    cases = _document(raw_cases)
    verify_tokenizer_rows(loaded.tokenizer, cases["rows"] + cases["projection_basis"])
    guard = custody._ModelGuard(model, module, weight, torch, v1)
    if (guard.state_sha != reference["model_state_sha256"] or guard.weight_sha != reference["weight_sha256"]
            or guard.buffer_sha != "f12cbc9f0cdf2fcb9567418d7d1129da0dd6db4f6c7aaa7f9edb32b3fe7668e8"):
        raise OwnerError("model, site or buffer identity differs")
    identity = {"case_sha256": CASE_SHA256, "source_revision": args.source_revision,
                "source_sha256": source_sha, "run_id": session.run_id,
                "public_report_sha256": PUBLIC_SHA256, "v44_addendum_sha256": V44_SHA256,
                "active_mask_addendum_sha256": MASK_SHA256,
                "public_support_sha256": SUPPORT_SHA256, "public_copy_addendum_sha256": COPY_SHA256,
                "scope": "named public/pending inputs only; private, protected and pretraining history unverified",
                "source_revision_scope": "operator-reported label; public blob consistency verified separately"}
    result = once.OneShot(guard, Path(args.output) / "screen",
                          lambda: baseline.run_fixed(guard, raw_cases, identity, Path(args.output) / "screen")).run()
    model_ref, weight_ref = weakref.ref(model), weakref.ref(weight)
    result = {"identity": identity, "environment": environment, "capacity_before_load": capacity,
              "denormal_before": before, "denormal_after": v1._denormal_probe(),
              "screen_sha256": result["screen_sha256"], "journal_sha256": result["journal_sha256"]}
    del guard, weight, module, model, loaded
    gc.collect()
    if model_ref() is not None or weight_ref() is not None or torch.cuda.is_initialized() or any(prefix.iterdir()):
        raise OwnerError("model disposal, CPU state or cache prefix differs")
    return result


def _child(args):
    root = Path(args.source).resolve(strict=True)
    prefix = Path(args.prefix).resolve(strict=True)
    _startup(root, prefix, child=True)
    hashes = _source_map(root, Path(args.sources), args.sources_sha256)
    if hashlib.sha256(_read(Path(__file__))).hexdigest() != hashes[OWNER]:
        raise OwnerError("executing owner source differs from inventory")
    boot = _module(root, BOOT, hashes)
    boot._check_search_path(root)
    boot._prime_parent_platform(_document(args.parent_platform.encode("utf-8")))
    import platform
    if platform._platform_cache:
        raise OwnerError("platform was observed before child startup")
    win = sys.getwindowsversion(); version = f"{win.major}.{win.minor}.{win.build}"
    platform._syscmd_ver = lambda *_args, **_kwargs: ("Windows", str(win.major), version)
    boot._install_process_guard()
    finder = _install_project(root, hashes, boot)
    from looking_glass.cpu_study_supervision import claim_cpu_study
    session, lease = claim_cpu_study(_study_root(args.study_root))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise OwnerError("the CPU lease was not claimed before model imports")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    boot._add_dependencies(root, Path(args.venv).resolve(strict=True), [args.dependency])
    if args.mode == "baseline":
        details = _run_model(args, session, hashes, args.sources_sha256, prefix)
    else:
        import torch
        from looking_glass import targeted_edit_cpu_v1 as v1
        from looking_glass import targeted_edit_france_top_edits_v2 as edits
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        if not torch.set_flush_denormal(False) or torch.cuda.is_initialized():
            raise OwnerError("CPU rehearsal numerical or device state differs")
        details = {"checkpoint_loaded": False, "model_forwarded": False,
                   "environment": v1._environment(), "denormal_probe": edits._probe()}
    inventory = finder.verify_inventory()
    boot._check_search_path(root)
    if any(prefix.iterdir()):
        raise OwnerError("the protected child prefix is not empty")
    receipt = {"status": "complete", "pid": os.getpid(), "run_id": session.run_id,
               "case_sha256": CASE_SHA256, "source_sha256": args.sources_sha256,
               "source_revision": args.source_revision, "mode": args.mode,
               "prefix_empty": True, "model_released": True, "project_sources": inventory,
               "details": details, "scope": "cooperative direct-child observation, not independent execution proof"}
    _write(Path(args.output) / "child-result.json", receipt)
    session.attest_clean(lease)
    return 0


def _parent(args):
    root = Path(args.source).resolve(strict=True)
    study = _study_root(args.study_root)
    prefix = Path(args.prefix).resolve(strict=True)
    _startup(root, prefix, child=False)
    if (not re.fullmatch(r"[0-9a-f]{40}", args.source_revision)
            or not study.is_dir() or study.drive.upper() != "E:"
            or Path(args.sources).resolve(strict=True).parent != root
            or Path(args.stage).resolve(strict=True).drive.upper() != "E:"
            or Path(args.output).drive.upper() != "E:"
            or type(args.timeout) is not int or not 1 <= args.timeout <= 1800):
        raise OwnerError("invalid source revision, dedicated root, E: paths or timeout")
    hashes = _source_map(root, Path(args.sources), args.sources_sha256)
    if hashlib.sha256(_read(Path(__file__))).hexdigest() != hashes[OWNER]:
        raise OwnerError("parent source differs from the supplied inventory")
    check_public_report(_read(root / PUBLIC, 1_000_000), PUBLIC_SHA256)
    capacity = _capacity(args.mode)
    launcher = _module(root, LAUNCHER, hashes)
    venv = launcher._checked_path(args.venv)
    dependency = launcher._checked_path(args.dependency)
    output = Path(args.output)
    output_parent = launcher._checked_path(output.parent)
    executable = launcher._check_startup(root, venv, (dependency,), output_parent)
    study = launcher._checked_path(study)
    if root == study or study.is_relative_to(root) or root.is_relative_to(study):
        raise OwnerError("CPU root must be separate from the source image")
    platform_values = launcher._parent_platform_observation()
    guards = _module(root, GUARDS, hashes)
    output.mkdir(exist_ok=False)
    _write(output / "parent-start.json", {"mode": args.mode, "source_revision": args.source_revision,
           "source_sha256": args.sources_sha256, "case_sha256": CASE_SHA256, "capacity": capacity,
           "study_root": str(study), "scope": "dedicated study root; no global CPU exclusion"})
    phase = "source_protection"
    try:
        source_paths = [root.joinpath(*PurePosixPath(name).parts) for name in hashes]
        source_paths.append(Path(args.sources).resolve(strict=True))
        source_lease = guards.SourceFileLease(root, source_paths)
        _HELD.append(source_lease); source_lease.__enter__()
        cache = guards.PrivateEmptyCache(output)
        _HELD.append(cache); child_prefix = cache.__enter__()
        if _source_map(root, Path(args.sources), args.sources_sha256) != hashes:
            raise OwnerError("source changed after protection")
        boot = _module(root, BOOT, hashes)
        boot._check_search_path(root)
        finder = _install_project(root, hashes, boot)
        from looking_glass.cpu_study_supervision import _study_paths
        from looking_glass.supervisor import run_supervised
        finder.verify_inventory()
        job = guards.WindowsJob(); _HELD.append(job)
        if args.mode == "baseline":
            phase = "single_use_claim"
            _write(study / "france-baseline.once.json", {"status": "claimed", "case_sha256": CASE_SHA256,
                   "source_sha256": args.sources_sha256, "scope": "no retry after uncertain launch"})
        command = [str(executable), "-I", "-S", "-B", "-X", "pycache_prefix=" + str(child_prefix),
                   str(root / OWNER), "--child", "--source", str(root), "--sources", args.sources,
                   "--sources-sha256", args.sources_sha256, "--source-revision", args.source_revision,
                   "--venv", str(venv), "--dependency", str(dependency), "--study-root", str(study),
                   "--prefix", str(child_prefix), "--output", str(output), "--mode", args.mode,
                   "--cache", args.cache, "--stage", args.stage,
                   "--parent-platform", json.dumps(platform_values, separators=(",", ":"))]
        env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
               "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "PROCESSOR_IDENTIFIER") if name in os.environ}
        env.update({"PATH": str(Path(env.get("SYSTEMROOT", "C:/Windows")) / "System32"),
                    "TEMP": str(output), "TMP": str(output), "HF_HOME": str(output),
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        lock, marker = _study_paths(study)
        phase = "supervised_child"
        result = run_supervised(command, device_lease_path=lock, marker_path=marker, identity=None,
                                timeout=args.timeout, lease_timeout=5, terminate_grace=5,
                                cwd=output, env=env, inherit_environment=False, process_job=job,
                                require_zero_exit=True)
        child = _document(_read(output / "child-result.json", 1_000_000))
        check_result(result, job, child, args.sources_sha256, args.mode, args.source_revision)
        if _source_map(root, Path(args.sources), args.sources_sha256) != hashes or any(child_prefix.iterdir()):
            raise OwnerError("protected source or bytecode prefix changed")
        finder.verify_inventory()
        cache.close(); source_lease.close(); _HELD.clear()
        _write(output / "supervision.json", {"status": "reaped", "pid": result.pid, "run_id": result.run_id,
               "exit_code": result.exit_code, "clean": result.clean, "timed_out": result.timed_out,
               "source_sha256": args.sources_sha256, "case_sha256": CASE_SHA256,
               "child_result_sha256": hashlib.sha256(_read(output / "child-result.json")).hexdigest(),
               "study_root": str(study), "scope": "cooperative supervisor observation; not execution attestation"})
        return 0
    except BaseException as error:
        try:
            _write(output / "supervisor-incomplete.json", {"status": "incomplete", "phase": phase,
                   "error_type": type(error).__name__, "scope": "retain all files and do not retry uncertain work"})
        except BaseException as recording:
            error.add_note("supervisor failure receipt could not be written: " + type(recording).__name__)
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    for name in ("source", "sources", "sources-sha256", "source-revision", "venv", "dependency",
                 "study-root", "prefix", "output", "mode", "cache", "stage"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--parent-platform")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.mode not in ("rehearsal", "baseline") or bool(args.parent_platform) != args.child:
        raise OwnerError("unsupported mode or child platform argument")
    if not args.child and sys.stdin.buffer.read(6) != b"start\n":
        raise OwnerError("direct observer did not authorize the parent start")
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
