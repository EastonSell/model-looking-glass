"""Trusted owner-side composition for the proposed synthetic-v2 study.

This module does not establish the source boundary: a separate guarded
bootstrap must match the supplied inventory to committed files, hold the
source/cache leases for the process lifetime, and supply its live finder
verification callback. It is not a command-line entry point. The owner must
discard a failed or interrupted attempt and never resume it.
"""

from datetime import datetime, timezone
import hashlib
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat

from .causal_v2_custodian_run import (run_study_once, _PUBLIC_LIMITS, _PUBLIC_RESERVE,
                                      _SOURCE_LIMIT, _SOURCE_STAGES, _sync_directory, registry)
from .causal_v2_custody import build_exclusion_union, canonical_json
from .causal_v2_freeze import _CLARIFICATION_FILES, _REQUIRED_SOURCE_FILES
from .causal_v2_preparation import MODEL_SEEDS, _context, prepare_study
from .causal_v2_runtime_preflight import inspect_backend_settings, inspect_runtime_and_models
from .causal_v2_scoring import score_heldout
from .causal_v2_study_executor import StudyExecutor
from .research_cases import SYNTHETIC_CASES


class StudyEntryError(RuntimeError):
    """The source-bound owner cannot safely start the proposed study."""


_ENTRY = "looking_glass/causal_v2_study_entry.py"
_EXCLUSIONS = "results/synthetic-v2-exclusions-20261028/inputs.json"
_PILOT = "results/synthetic-v2-resource-pilot-20261028/artifacts/pilot_inputs.json"
_EXPECTED = {
    _EXCLUSIONS: "f518a2a4e7e0aa3e03ebf60170395c5fb49ce0cd2e8acd98a96ba3427b7f3347",
    _PILOT: "024bdea3669683808302b953e7797afaa159d8d4719c574a3dc65e45bff74722",
}
_HISTORICAL_SHA256 = "c0007d6e46fabf2356f0ec726a474ea6fa0a6cb4f573ba648cdca232bf878814"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_PATH = re.compile(r"[A-Za-z0-9_./-]{1,512}\Z")
_MARKER_LIMIT = 4096


def _checked_path(value):
    path = Path(os.path.abspath(value))
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise StudyEntryError("source or output path contains a link or reparse point")
    return path.resolve(strict=True)


def _verified_sources(checkout, sources):
    """Rehash bounded, already commit-verified files under their owner leases."""
    if (type(sources) is not dict or not 0 < len(sources) <= 4096
            or not (_REQUIRED_SOURCE_FILES | {_ENTRY}).issubset(sources)):
        raise StudyEntryError("committed source inventory is incomplete")
    total, seen = 0, set()
    for relative, digest in sources.items():
        if (type(relative) is not str or _PATH.fullmatch(relative) is None
                or type(digest) is not str or _HEX.fullmatch(digest) is None):
            raise StudyEntryError("invalid source inventory entry")
        path = PurePosixPath(relative)
        if (path.is_absolute() or path.as_posix() != relative
                or any(part in ("", ".", "..") for part in path.parts)
                or relative.casefold() in seen):
            raise StudyEntryError("invalid or ambiguous source path")
        seen.add(relative.casefold())
        actual = _checked_path(checkout.joinpath(*path.parts))
        if not actual.is_relative_to(checkout) or not actual.is_file():
            raise StudyEntryError("source does not resolve within the checkout")
        with actual.open("rb") as stream:
            data = stream.read(8 * 1024 * 1024 + 1)
        total += len(data)
        if (len(data) > 8 * 1024 * 1024 or total > 64 * 1024 * 1024
                or hashlib.sha256(data).hexdigest() != digest):
            raise StudyEntryError("source bytes do not match the bound inventory")
    if any(sources.get(path) != digest for path, digest in _EXPECTED.items()):
        raise StudyEntryError("development or pilot exclusions differ")


def _checked_output(checkout, path):
    target = Path(path)
    if not target.is_absolute() or os.path.lexists(target):
        raise StudyEntryError("a fresh absolute study path is required")
    parent = _checked_path(target.parent)
    target = parent / target.name
    if (not target.name or not parent.is_dir() or target.is_relative_to(checkout)
            or checkout.is_relative_to(target)):
        raise StudyEntryError("study output must be outside the checkout")
    if shutil.disk_usage(parent).free < 2 * 1024**3:
        raise StudyEntryError("study destination has less than 2 GiB free")
    if os.path.lexists(_marker_path(target)):
        raise StudyEntryError("study path already has an attempt marker")
    if (sum(_PUBLIC_LIMITS.values()) + len(_SOURCE_STAGES) * _SOURCE_LIMIT
            + max(max(_PUBLIC_LIMITS.values()), _SOURCE_LIMIT)
            + 2 * registry.MAX_METADATA_BYTES + _MARKER_LIMIT > _PUBLIC_RESERVE):
        raise StudyEntryError("public reserve cannot cover the attempt marker")
    return target


def _marker_path(output):
    return output.parent / (output.name + ".attempt.json")


def _begin_attempt(output, context):
    """Create the non-overwritable local marker before making private cases."""
    raw = canonical_json({"schema": "synthetic-v2-attempt-v1", "state": "started",
        "source_revision": context["source_revision"], "protocol_sha256": context["protocol_sha256"]})
    if len(raw) > _MARKER_LIMIT:
        raise StudyEntryError("attempt marker exceeds its public reserve")
    # Even a partial marker is intentionally retained: the path cannot be
    # reused as a fresh attempt if creation or synchronization is interrupted.
    with _marker_path(output).open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise StudyEntryError("attempt marker was not completely written")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(output.parent)


def _unknown_exposure():
    return {"schema": "synthetic-v2-exposure-v1", "classification": "unknown",
            "basis": "No independent exposure monitor established an unexposed holdout."}


def _require_inventory_hook():
    if "source_inventory" not in inspect.signature(run_study_once).parameters:
        raise StudyEntryError("phase source-inventory recording is unavailable")


def _preflight(checkout, output, revision, source_sha256, source_inventory):
    if (type(revision) is not str or _REVISION.fullmatch(revision) is None
            or not callable(source_inventory)):
        raise StudyEntryError("invalid owner revision or source inventory callback")
    checkout = _checked_path(checkout)
    if not checkout.is_dir():
        raise StudyEntryError("checkout must be a directory")
    output = _checked_output(checkout, output)
    sources = dict(source_sha256) if type(source_sha256) is dict else source_sha256
    _verified_sources(checkout, sources)

    def inventory():
        _verified_sources(checkout, sources)
        observed = source_inventory()
        if (type(observed) is not dict or not observed or len(observed) > 4096
                or _ENTRY not in observed
                or any(type(path) is not str or type(digest) is not str
                       or sources.get(path) != digest for path, digest in observed.items())):
            raise StudyEntryError("active source-loader inventory differs")
        return dict(sorted(observed.items()))

    inventory()
    exclusion_bytes = (checkout / _EXCLUSIONS).read_bytes()
    pilot_bytes = (checkout / _PILOT).read_bytes()
    if (hashlib.sha256(exclusion_bytes).hexdigest() != _EXPECTED[_EXCLUSIONS]
            or hashlib.sha256(pilot_bytes).hexdigest() != _EXPECTED[_PILOT]):
        raise StudyEntryError("exclusion bytes changed during preflight")
    historical = [{"case_id": case.case_id, "values": list(case.values)} for case in SYNTHETIC_CASES]
    if (len(historical) != 24
            or hashlib.sha256(build_exclusion_union([{"inputs": historical}])).hexdigest()
            != _HISTORICAL_SHA256):
        raise StudyEntryError("historical v1 exclusion inventory differs")

    backend = inspect_backend_settings()
    policy, runtime, model_states = inspect_runtime_and_models()
    if canonical_json(backend) != canonical_json(inspect_backend_settings()):
        raise StudyEntryError("backend changed during model-state preflight")
    runtime = {**runtime, **backend}
    context = {"protocol_id": "synthetic-causal-v2",
               "protocol_sha256": sources["docs/CAUSAL_CONFIRMATION_V2.md"],
               "source_revision": revision, "model_state_hashes": model_states}
    _context(context, MODEL_SEEDS)
    clarifications = {field: sources[path] for field, path in _CLARIFICATION_FILES.items()}
    inventory()
    return {"output": output, "sources": sources, "inventory": inventory,
            "exclusion_bytes": exclusion_bytes, "pilot_bytes": pilot_bytes,
            "historical": historical, "backend": backend, "policy": policy,
            "runtime": runtime, "context": context, "clarifications": clarifications}


def rehearse_source_bound_owner(checkout, output, *, revision, source_sha256, source_inventory):
    """Check the real owner boundary and model initialization without a forward.

    This does not generate cases, create a study record, or authorize a study.
    Source, process and cache protections are supplied by the guarded owner.
    """
    _require_inventory_hook()
    data = _preflight(checkout, output, revision, source_sha256, source_inventory)
    executor = StudyExecutor(data["policy"], backend_settings=data["backend"])
    executor.close()
    data["inventory"]()
    return {"status": "rehearsal", "model_states_checked": len(data["context"]["model_state_hashes"]),
            "root_forwards": 0, "study_ready": False}


def run_source_bound_study(checkout, output, *, revision, source_sha256, source_inventory):
    """Perform one attempt after a trusted source-verified owner launches it.

    The inventory callback must come from the active source-only finder, not a
    caller's arbitrary assertion. It is checked and passed to the custodian for
    durable phase snapshots. The scorer and exposure classifier are fixed here.
    These cooperative checks do not attest the interpreter, native libraries,
    secrecy, or an independent external timestamp.
    """
    # Refuse to prepare any private cases until the required phase-inventory
    # integration exists. The source-bound launcher supplies the real finder.
    _require_inventory_hook()
    data = _preflight(checkout, output, revision, source_sha256, source_inventory)
    output, sources, inventory = data["output"], data["sources"], data["inventory"]
    exclusion_bytes, pilot_bytes = data["exclusion_bytes"], data["pilot_bytes"]
    historical, backend, policy = data["historical"], data["backend"], data["policy"]
    runtime, context, clarifications = data["runtime"], data["context"], data["clarifications"]
    executor = StudyExecutor(policy, backend_settings=backend)
    try:
        _begin_attempt(output, context)
        # This is the first operation that constructs private v2 cases or sites.
        prepared = prepare_study(context, exclusion_bytes, pilot_bytes, historical)
        # The custodian normally owns close. Its initial argument checks happen
        # before its cleanup block, so retain an idempotent outer failure guard.
        return run_study_once(output, prepared=prepared, expected_context=context,
            exclusion_bytes=exclusion_bytes, pilot_bytes=pilot_bytes,
            historical_cases=historical, source_sha256=sources,
            clarification_sha256=clarifications, runtime=runtime,
            created_utc=datetime.now(timezone.utc).isoformat(), executor=executor,
            score=score_heldout, classify_exposure=_unknown_exposure,
            source_inventory=inventory)
    except BaseException as primary:
        try:
            executor.close()
        except BaseException as cleanup:
            if isinstance(cleanup, (KeyboardInterrupt, SystemExit)):
                raise cleanup from primary
            raise primary from cleanup
        raise
