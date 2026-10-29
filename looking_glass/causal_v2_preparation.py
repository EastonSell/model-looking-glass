"""Protocol-specific case and schedule preparation for synthetic v2.

Preparation alone does not authorize a run, publish evidence, attest source
execution, or prevent a model forward. The caller must complete and verify
the independent execution, resource and phase gates before real preparation.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import re
import secrets

from .causal_v2_custody import (build_exclusion_union, canonical_json, commitment,
                                validate_nonce_set, verify_split_disjointness)


MODEL_SEEDS = (31201, 31202, 31203, 31204, 31205, 31206)
DISCOVERY_ROOT = (2026102803).to_bytes(32, "big")
NODES = tuple(f"block.{block}.{node}" for block in (1, 2, 3)
              for node in ("branch.0", "activation", "merge"))
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MAX_EXCLUSIONS = 1024 * 1024
_PILOT_SHA256 = "024bdea3669683808302b953e7797afaa159d8d4719c574a3dc65e45bff74722"
_V1_PROJECTION_SHA256 = "c0007d6e46fabf2356f0ec726a474ea6fa0a6cb4f573ba648cdca232bf878814"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError("expected a lowercase SHA-256")


def _read_exclusions(raw):
    if type(raw) is not bytes or len(raw) > _MAX_EXCLUSIONS:
        raise ValueError("invalid bounded exclusion bytes")
    try:
        value = json.loads(raw)
        if build_exclusion_union([value]) != raw:
            raise ValueError("noncanonical exclusion record")
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("invalid canonical exclusion record") from error
    return value["inputs"]


def _context(context, seeds):
    if (type(context) is not dict or set(context) !=
            {"protocol_id", "protocol_sha256", "source_revision", "model_state_hashes"}
            or context["protocol_id"] != "synthetic-causal-v2"):
        raise ValueError("invalid v2 commitment context")
    _digest(context["protocol_sha256"])
    revision = context["source_revision"]
    if type(revision) is not str or _REVISION.fullmatch(revision) is None:
        raise ValueError("invalid source revision")
    states = context["model_state_hashes"]
    if type(states) is not list or len(states) != len(seeds):
        raise ValueError("invalid model-state inventory")
    for state, seed in zip(states, seeds):
        if (type(state) is not dict or set(state) != {"seed", "sha256"}
                or type(state["seed"]) is not int or state["seed"] != seed):
            raise ValueError("model-state seed order differs")
        _digest(state["sha256"])
    return _sha(canonical_json(context))


def _derive_cases(seeds, count, heldout_root, blocked, *, development):
    used = {tuple(item["values"]) for item in blocked}
    cases = {"discovery": [], "heldout": []}
    for split, marker, root in (("discovery", b"D", DISCOVERY_ROOT),
                                ("heldout", b"H", heldout_root)):
        for seed in seeds:
            for index in range(count):
                for attempt in range(65536):
                    digest = hashlib.sha256(b"MLG-v2-cases\0" + root + seed.to_bytes(8, "big")
                                            + marker + index.to_bytes(2, "big")
                                            + attempt.to_bytes(2, "big")).digest()
                    first = ((digest[0] & 15) + 1) / 16
                    values = [first if index % 2 == 0 else -first]
                    values.extend(((digest[j] & 31) - 16) / 16 for j in range(1, 8))
                    key = tuple(values)  # Python equality intentionally equates signed zeros.
                    if key not in used:
                        used.add(key)
                        prefix = "dev-v2" if development else "v2"
                        case_id = f"{prefix}-m{seed}-{'d' if split == 'discovery' else 'h'}{index:02d}"
                        cases[split].append({"case_id": case_id, "model_seed": seed,
                                             "split": split, "index": index, "values": values})
                        break
                else:
                    raise ValueError("case generation exhausted its collision attempts")
    return cases


def _build(context, seeds, count, exclusion_bytes, pilot_bytes, historical_cases,
           heldout_root, site_nonce, prediction_nonce, shuffle, *, development):
    if (type(seeds) is not tuple or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds)
            or type(count) is not int or not 1 <= count <= 16):
        raise ValueError("invalid ordered model seeds or case count")
    context_hash = _context(context, seeds)
    validate_nonce_set(heldout_root, site_nonce, prediction_nonce)
    union = _read_exclusions(exclusion_bytes)
    pilot = _read_exclusions(pilot_bytes)
    union_by_id = {item["case_id"]: canonical_json(item) for item in union}
    if any(union_by_id.get(item["case_id"]) != canonical_json(item) for item in pilot):
        raise ValueError("development union does not contain the original pilot")
    verify_split_disjointness([], [], exclusion_bytes, historical_cases=historical_cases)
    historical_raw = build_exclusion_union([{"inputs": historical_cases}])
    cases = _derive_cases(seeds, count, heldout_root, union + historical_cases,
                          development=development)
    project = lambda rows: [{"case_id": item["case_id"], "values": item["values"]} for item in rows]
    verify_split_disjointness(project(cases["discovery"]), project(cases["heldout"]),
                              exclusion_bytes, historical_cases=historical_cases)
    assignments = []
    for seed in seeds:
        nodes = list(NODES)
        shuffle(nodes)
        if len(nodes) != len(NODES) or set(nodes) != set(NODES):
            raise ValueError("schedule shuffle is not a permutation")
        for index, node in enumerate(nodes):
            prefix = "dev-v2" if development else "v2"
            assignments.append({"cell_id": f"{prefix}-m{seed}-c{index:02d}",
                                "model_seed": seed, "injection_node": node})
    heldout = {"context_sha256": context_hash, "cases": cases["heldout"]}
    schedule = {"context_sha256": context_hash, "assignments": assignments}
    public = {"mode": "development" if development else "study",
              "context": deepcopy(context), "context_sha256": context_hash,
              "pilot_exclusions_sha256": _sha(pilot_bytes),
              "development_exclusions_sha256": _sha(exclusion_bytes),
              "historical_exclusions_sha256": _sha(historical_raw),
              "historical_exclusions_count": len(historical_cases),
              "discovery_cases": cases["discovery"],
              "heldout_commitment": commitment("heldout", heldout_root, heldout),
              "schedule_commitment": commitment("schedule", site_nonce, schedule)}
    return Prepared(deepcopy(public), heldout, schedule, heldout_root, site_nonce, prediction_nonce)


@dataclass(frozen=True)
class Prepared:
    """Private material and a public fragment; nested records remain mutable.

    This is not durable evidence or an enforced phase gate. Private fields are
    excluded from repr to reduce accidental logging, not to enforce secrecy.
    """
    public: dict
    heldout: dict = field(repr=False)
    schedule: dict = field(repr=False)
    heldout_root: bytes = field(repr=False)
    site_nonce: bytes = field(repr=False)
    prediction_nonce: bytes = field(repr=False)


def prepare_study(context, exclusion_bytes, pilot_bytes, historical_cases):
    """Draw secrets and prepare the fixed v2 cases; call only after readiness gates."""
    if (_sha(pilot_bytes) != _PILOT_SHA256 or len(historical_cases) != 24
            or _sha(build_exclusion_union([{"inputs": historical_cases}])) != _V1_PROJECTION_SHA256):
        raise ValueError("historical v1 or original pilot inventory differs")
    roots = tuple(secrets.token_bytes(32) for _ in range(3))
    return _build(context, MODEL_SEEDS, 16, exclusion_bytes, pilot_bytes,
                  historical_cases, *roots, secrets.SystemRandom().shuffle,
                  development=False)


def prepare_development(context, seeds, exclusion_bytes, pilot_bytes, historical_cases,
                        *, heldout_root, site_nonce, prediction_nonce, shuffle, count=2):
    """Build explicitly fake records for tests, using no reserved model seed."""
    if type(seeds) is not tuple or set(seeds) & set(MODEL_SEEDS):
        raise ValueError("development cases must use different model seeds")
    if not callable(shuffle):
        raise ValueError("development shuffle must be callable")
    return _build(context, seeds, count, exclusion_bytes, pilot_bytes,
                  historical_cases, heldout_root, site_nonce, prediction_nonce,
                  shuffle, development=True)


@dataclass(frozen=True)
class PreparedSnapshot:
    """Checked immutable bytes for publication by a separately reviewed gate."""
    public: bytes
    heldout: bytes = field(repr=False)
    schedule: bytes = field(repr=False)
    heldout_root: bytes = field(repr=False)
    site_nonce: bytes = field(repr=False)
    prediction_nonce: bytes = field(repr=False)


def verify_prepared(prepared, expected_context, exclusion_bytes, pilot_bytes,
                    historical_cases, *, development=False, seeds=MODEL_SEEDS, count=16):
    """Rebuild and snapshot records; actual source/model provenance is caller-owned."""
    if type(prepared) is not Prepared or type(development) is not bool:
        raise ValueError("expected prepared records and an explicit mode")
    if development:
        if type(seeds) is not tuple or set(seeds) & set(MODEL_SEEDS):
            raise ValueError("development cases must use different model seeds")
    elif (seeds != MODEL_SEEDS or count != 16 or _sha(pilot_bytes) != _PILOT_SHA256
          or len(historical_cases) != 24
          or _sha(build_exclusion_union([{"inputs": historical_cases}])) != _V1_PROJECTION_SHA256):
        raise ValueError("study identity or exclusions differ from v2")
    try:
        assignments = prepared.schedule["assignments"]
        if type(assignments) is not list or len(assignments) != len(seeds) * len(NODES):
            raise ValueError("invalid schedule length")
        cursor = iter(range(len(seeds)))
        def recorded_shuffle(nodes):
            index = next(cursor)
            part = assignments[index * len(NODES):(index + 1) * len(NODES)]
            nodes[:] = [item["injection_node"] for item in part]
        rebuilt = _build(expected_context, seeds, count, exclusion_bytes, pilot_bytes,
                         historical_cases, prepared.heldout_root, prepared.site_nonce,
                         prepared.prediction_nonce, recorded_shuffle, development=development)
        public = canonical_json(prepared.public)
        heldout = canonical_json(prepared.heldout)
        schedule = canonical_json(prepared.schedule)
        if (public != canonical_json(rebuilt.public) or heldout != canonical_json(rebuilt.heldout)
                or schedule != canonical_json(rebuilt.schedule)):
            raise ValueError("prepared records differ from their bindings")
    except (KeyError, TypeError, StopIteration, RecursionError) as error:
        raise ValueError("invalid prepared records") from error
    return PreparedSnapshot(public, heldout, schedule, prepared.heldout_root,
                            prepared.site_nonce, prepared.prediction_nonce)
