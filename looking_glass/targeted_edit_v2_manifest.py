"""Versioned source/case manifest for the prospective v2 edit study.

This module does not load a model or run the public-reservation verifier. A
manifest is necessary but not sufficient to authorize execution; the owner
must also verify imported source paths and the inventory before model work.
The two-process owner is still under development.
"""

from __future__ import annotations

from pathlib import Path

from . import pretrained
from . import targeted_edit_cpu_v1 as v1
from . import targeted_edit_cpu_v2 as v2
from . import targeted_edit_v2_edits as edits
from . import targeted_edit_v2_evidence as evidence
from . import targeted_edit_v2_exposure as exposure


V1_MANIFEST = "results/targeted-edit-cpu-v1-20261028/frozen.json"
V1_PRECOMMIT = "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"
V1_MANIFEST_SHA256 = "93d953494cca7f2fcf0e7cd8baec35fa9ca1b68945e1e44f183e75e10ed03d1d"
V1_EVIDENCE_COMMIT = "3581c974cbaf20d3118bd825a8c9c68188774b10"
SOURCE_PATHS = (*v1.SOURCE_PATHS, "looking_glass/pretrained_exposures_v3.py",
                "looking_glass/pretrained_exposures_v4.py", "docs/TARGETED_EDIT_CPU_V2.md",
                "looking_glass/targeted_edit_cpu_v2.py", "looking_glass/targeted_edit_v2_edits.py",
                "looking_glass/targeted_edit_v2_evidence.py", "looking_glass/targeted_edit_v2_exposure.py",
                "looking_glass/targeted_edit_v2_manifest.py", "looking_glass/targeted_edit_v2_gate.py",
                "looking_glass/targeted_edit_v2_custody.py",
                "docs/TARGETED_EDIT_V2_OWNER.md",
                "docs/CPU_STUDY_SUPERVISION.md", "docs/SUPERVISOR.md",
                "looking_glass/cpu_study_supervision.py", "looking_glass/supervisor.py",
                "looking_glass/device_lease.py", "looking_glass/device_identity.py",
                "looking_glass/targeted_edit_v2_owner.py",
                "looking_glass/targeted_edit_v2_witness.py", "docs/TARGETED_EDIT_V2_REMOTE_WITNESS.md",
                "looking_glass/targeted_edit_v2_evaluation_evidence.py", "docs/TARGETED_EDIT_V2_EVALUATION_EVIDENCE.md",
                "looking_glass/targeted_edit_v2_evaluation_owner.py", "docs/TARGETED_EDIT_V2_EVALUATION_OWNER.md",
                "scripts/targeted_edit_v2_source_rehearsal.py", "scripts/targeted_edit_v2_source_child.py",
                "scripts/v2_source_launcher.py", "scripts/v2_source_bootstrap.py", "scripts/v2_source_guards.py",
                "docs/TARGETED_EDIT_V2_SOURCE_REHEARSAL.md",
                V1_MANIFEST, V1_PRECOMMIT)

RULES = {
    "development": {"baseline_discovery_negative": True, "minimum_negative_targets": 3,
                    "discovery_positive": True, "minimum_target_reversals": 1,
                    "reversal_scope": "originally_negative_target_cases_only",
                    "mean_target_gain_strictly_above": 0.5,
                    "all_locality_top_preserved": True, "mean_kl_at_most": 0.02,
                    "max_kl_at_most": 0.10, "mean_nll_deterioration_at_most": 0.05,
                    "selection": "largest mean target gain; exact ties choose smaller dose",
                    "fallback": "smallest dose; development_qualified=false"},
    "evaluation": {"development_qualified_required": True, "minimum_negative_targets": 3,
                   "minimum_target_reversals": 2,
                   "reversal_scope": "originally_negative_target_cases_only",
                   "mean_target_gain_strictly_above": 0.5, "random_advantage_at_least": 0.5,
                   "all_locality_top_preserved": True, "mean_kl_at_most": 0.02,
                   "max_kl_at_most": 0.10, "mean_nll_deterioration_at_most": 0.05,
                   "decision_precedence": ["invalid_or_incomplete", "ineligible", "negative", "success"]},
    "prediction": {"sign": "strictly positive; zero is incorrect",
                   "point": "selected development mean target change",
                   "mae": "mean absolute error across all four cases"},
    "norm_ratio": {"realized_to_planned": [0.999, 1.001],
                   "control_to_selected": [0.999, 1.001]},
    "controls": {"random": "one same-site CPU float32 Gaussian outer product using the fixed seed",
                 "opposite": "elementwise negation of selected intended float32 update",
                 "v1_reference": "the reconstructed original 1% intended gradient update"},
    "selected_direction": "float32(float64(D1) * (dose * ||float64(W)||_F / ||float64(D1)||_F)); D1 is the rounded v1 intended float32 gradient update",
    "rounding": "float64 scale, float32 cast and addition",
    "score": "float64 shifted log-softmax: (z-max)-log(sum(exp(z-max))); NLL and KL; smallest token ID wins top-token ties",
}
_RULES_CANONICAL = v1._canonical(RULES)


def _case_payload(cases):
    return [{"case_id": case_id, "split": split, "prompt": text, "token_ids": list(ids),
             "expected": None if label is None else {"text": label[0], "token_id": label[1]}}
            for case_id, split, text, ids, label in cases]


def _reference_data(repo, source_commit):
    raw_manifest = evidence._blob(repo, source_commit, V1_MANIFEST, evidence.MAX_JSON)
    raw_reference = evidence._blob(repo, source_commit, V1_PRECOMMIT, evidence.MAX_JSON)
    if (v1._sha(raw_manifest) != V1_MANIFEST_SHA256 or
            v1._sha(raw_reference) != edits.V1_PRECOMMIT_SHA256):
        raise v1.StudyError("published v1 evidence differs")
    old_manifest = v1._decode_document(raw_manifest)
    old_reference = v1._decode_document(raw_reference)
    if (old_manifest.get("version") != v1.VERSION or old_reference.get("version") != v1.VERSION or
            old_reference.get("manifest_sha256") != V1_MANIFEST_SHA256):
        raise v1.StudyError("published v1 identities differ")
    return old_manifest, old_reference


def make_manifest(repo, source_commit):
    """Build the canonical fixed protocol from already committed source bytes."""
    root = Path(repo)
    evidence._require_commit_object(root, source_commit)
    for prior in (V1_EVIDENCE_COMMIT, exposure.INVENTORY_COMMIT):
        evidence._require_commit_object(root, prior)
        if evidence._git(root, "merge-base", "--is-ancestor", prior, source_commit, check=False).returncode:
            raise v1.StudyError("v1 evidence and public reservations must precede source")
    old_manifest, old_reference = _reference_data(root, source_commit)
    hashes = {name: v1._sha(evidence._blob(root, source_commit, name, evidence.MAX_JSON))
              for name in SOURCE_PATHS}
    if (type(old_manifest.get("source_sha256")) is not dict or
            any(hashes.get(name) != digest for name, digest in old_manifest["source_sha256"].items())):
        raise v1.StudyError("the v1 reconstruction source differs from its original frozen source")
    try:
        environment = {key: old_reference["environment"][key] for key in edits.ENVIRONMENT_FIELDS}
        compatibility = {"precommit_sha256": edits.V1_PRECOMMIT_SHA256,
                         "manifest_sha256": V1_MANIFEST_SHA256,
                         "evidence_commit": V1_EVIDENCE_COMMIT,
                         "model": old_reference["model"], "environment": environment,
                         "model_state_sha256": old_reference["model_state_sha256"],
                         "weight_sha256": old_reference["weight_sha256"],
                         "discovery": old_reference["discovery"],
                         "candidates": old_reference["candidates"]}
    except (KeyError, TypeError) as exc:
        raise v1.StudyError("published v1 precommit is incomplete") from exc
    v2.check_split_rows()
    return {"version": v2.VERSION, "source_commit": source_commit, "source_sha256": hashes,
            "model_id": pretrained.MODEL_ID, "model_revision": pretrained.REVISION,
            "assets": {name: digest for name, (_, digest) in pretrained.ASSETS.items()},
            "site": v1.SITE, "site_shape": list(v1.WEIGHT_SHAPE), "parameter_count": v1.PARAMETER_COUNT,
            "target": {"text": " London", "token_id": v1.LONDON},
            "competitor": {"text": " Paris", "token_id": v1.PARIS},
            "development_cases": _case_payload(v1.CASES), "evaluation_cases": _case_payload(v2.EVALUATION_CASES),
            "development_conditions": list(evidence.CONDITIONS),
            "evaluation_conditions": list(v2.EVALUATION_CONDITIONS),
            "relative_doses": [{"condition": name, "dose": dose} for name, dose in v2.GRID],
            "random_seed": v2.RANDOM_SEED, "rules": v1._decode_document(_RULES_CANONICAL),
            "v1_compatibility": compatibility,
            "exposure": {"inventory_commit": exposure.INVENTORY_COMMIT,
                         "canonical_payload_sha256": exposure.INVENTORY_SHA256,
                         "predecessor_sha256": exposure.PREDECESSOR_SHA256,
                         "private_split_check_performed": False,
                         "custodian_observations": {
                             "operator_recorded_before_utc": "2026-10-29T03:08:44+00:00",
                             "scope": "session reports only; not an independent or exhaustive verification",
                             "reports": [
                                 {"lane": "exposure inventory", "report": "owner had not generated protected held-out rows"},
                                 {"lane": "pretrained trace adapter", "report": "owner held no generated protected DistilGPT2 inputs"},
                                 {"lane": "pretrained loader", "report": "lane held no generated protected DistilGPT2 inputs"},
                                 {"lane": "number-study coordinator", "report": "no DistilGPT2 held-out nonce, families or token rows had been generated"},
                             ]},
                         "complete_history_certified": False}}


def verify_manifest(repo, relative_path, *, manifest_commit, expected_sha256):
    """Verify separately committed manifest, source blobs, and checkout bytes.

    This does not check public reservations, attest imported/native bytes,
    verify private exposure history or prove remote publication timing; the
    owner must enforce the remaining run gates.
    """
    root = Path(repo)
    path = evidence._relative(relative_path)
    if not evidence._digest(expected_sha256):
        raise v1.StudyError("an external manifest SHA-256 is required")
    evidence._require_commit_object(root, manifest_commit)
    checked_path = evidence._ordinary_path(root, path)
    raw = evidence._read(checked_path, evidence.MAX_JSON)
    if v1._sha(raw) != expected_sha256:
        raise v1.StudyError("manifest differs from its external pin")
    manifest = v1._decode_document(raw)
    source = manifest.get("source_commit")
    evidence._require_commit_object(root, source)
    head = evidence._git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    evidence._require_commit_object(root, head)
    if (source == manifest_commit or
            evidence._git(root, "merge-base", "--is-ancestor", source, manifest_commit, check=False).returncode or
            evidence._git(root, "merge-base", "--is-ancestor", manifest_commit, head, check=False).returncode or
            not evidence._path_absent(root, source, path)):
        raise v1.StudyError("manifest was not committed after its source")
    if (evidence._blob(root, manifest_commit, path, evidence.MAX_JSON) != raw or
            evidence._blob(root, head, path, evidence.MAX_JSON) != raw or
            v1._canonical(make_manifest(root, source)) != raw):
        raise v1.StudyError("committed manifest does not match the fixed protocol")
    for name, digest in manifest["source_sha256"].items():
        checkout_path = evidence._ordinary_path(root, name)
        if (v1._sha(evidence._read(checkout_path, evidence.MAX_JSON)) != digest or
                v1._sha(evidence._blob(root, head, name, evidence.MAX_JSON)) != digest):
            raise v1.StudyError("frozen source differs in the checkout or current commit")
    return manifest
