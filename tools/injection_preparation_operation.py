"""Wire one observed injection preclaim to the fixed split custodian.

All modules and callbacks are trusted inputs from a source-bound, supervised
caller. This component does not attest process exit, publication, runtime,
current exposure, private access controls or the replay's scientific result.
It never creates a ledger claim or retries a split draw.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class ClaimHandoff:
    workspace: str
    replay_config_bytes: bytes
    plan_sha256: str
    receipt_sha256: str
    parent_identity: tuple
    ledger_identity: tuple


def _digest(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('a bound SHA-256 is required')
    return value


def _path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('a fixed absolute path is required')
    return os.path.normcase(os.path.abspath(str(path)))


def _canonical_config(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        raise ValueError('a bounded replay configuration is required')
    try:
        value = json.loads(raw)
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                               separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError('a canonical replay configuration is required') from None
    if type(value) is not dict or canonical != raw:
        raise ValueError('a canonical replay configuration is required')
    return value


def prepare_from_handoff(*, custody, guard, accounting, replay_owner, replay_child, replay_claim,
                         public, private_parent, ledger_path, anchor, roots, handoff, replay_tensor_bytes,
                         config_bytes, site_nonce, site_record_bytes, tokenizer,
                         observe_claimant_exit, observe_replay_config, verify_replay_evidence,
                         publication_check, source_check, prediction_check, gate_allocation,
                         observe_inventory, runtime_check):
    """Consume the existing grant with bound local checks and one split draw.

    The caller must first independently verify the claimant's clean exit and
    the completed replay/audit, protect their retained paths and use callbacks
    that re-observe those facts. In particular, the receipt is not itself proof
    of an exit, and the reported replay tensor count is a trusted audit input.
    Checks performed under the SQLite guard must be local and bounded.
    """
    callbacks = (observe_claimant_exit, observe_replay_config, verify_replay_evidence,
                 publication_check, source_check, prediction_check, gate_allocation,
                 observe_inventory, runtime_check)
    if (type(handoff) is not ClaimHandoff or type(anchor) is not guard.ClaimAnchor
            or type(roots) is not accounting.PhaseRoots or any(not callable(item) for item in callbacks)
            or type(replay_tensor_bytes) is not int or not 0 <= replay_tensor_bytes <= 8 * 1024**2
            or type(config_bytes) is not bytes or not 0 < len(config_bytes) <= 65536):
        raise ValueError('the observed preclaim and bounded evidence are required')
    for identity in (handoff.parent_identity, handoff.ledger_identity):
        if (type(identity) is not tuple or len(identity) != 2
                or any(type(item) is not int or item <= 0 for item in identity)):
            raise ValueError('protected identities are required')
    _digest(handoff.plan_sha256)
    _digest(handoff.receipt_sha256)
    _digest(anchor.heldout_claim_sha256)
    _path(handoff.workspace)
    phase = Path(roots.phase_path)
    parent, ledger = Path(private_parent), Path(ledger_path)
    if (_path(parent) != _path(roots.parent_path) or _path(phase.parent) != _path(parent)
            or _path(phase) != _path(anchor.phase_path) or _path(ledger) != _path(anchor.ledger_path)
            or handoff.parent_identity != (roots.parent_dev, roots.parent_ino)
            or handoff.ledger_identity != (anchor.db_dev, anchor.db_ino)):
        raise ValueError('the preclaim and accounting paths differ')
    replay_config = _canonical_config(handoff.replay_config_bytes)
    if (type(public.runtime_replay_plan_bytes) is not bytes
            or hashlib.sha256(public.runtime_replay_plan_bytes).hexdigest() != handoff.plan_sha256):
        raise ValueError('the published replay plan differs from the claim')

    def verified_claim(path, name, *, output_path):
        if (_path(path) != _path(ledger) or name != 'injection_heldout'
                or _path(output_path) != _path(phase)
                or observe_replay_config() != handoff.replay_config_bytes
                or observe_claimant_exit(handoff.receipt_sha256) is not True):
            raise ValueError('claimant exit or protected replay configuration is unverified')
        result = replay_owner.verify_preclaim_handoff(
            child=replay_child, claim=replay_claim, guard=guard, workspace=Path(handoff.workspace),
            config=replay_config, plan_sha256=handoff.plan_sha256, receipt_sha256=handoff.receipt_sha256,
            parent_identity=handoff.parent_identity, ledger_identity=handoff.ledger_identity,
            state_sha256=anchor.state_sha256, history_sha256=anchor.history_sha256,
            scope_sha256=anchor.scope_sha256, original_claims_sha256=anchor.original_claims_sha256)
        if (type(result) is not dict or set(result) != {'status', 'receipt_sha256', 'intent_sha256', 'claim_row_sha256'}
                or result['status'] != 'row_and_receipt_consistent_awaiting_observed_exit'
                or result['receipt_sha256'] != handoff.receipt_sha256
                or result['claim_row_sha256'] != anchor.heldout_claim_sha256
                or _digest(result['intent_sha256']) != result['intent_sha256']
                or guard.verify_existing_claim(ledger, name, output_path=phase, anchor=anchor) is not True):
            raise ValueError('the existing preclaim receipt or row did not verify')
        return True

    def protected_claim(path, name, *, output_path):
        return guard.hold_existing_claim(path, name, output_path=output_path, anchor=anchor)

    def reconcile(output_path, *, pilot_logical_bytes, pilot_disk_bytes, rehearsal_disk_bytes, stage):
        if (_path(output_path) != _path(phase) or pilot_logical_bytes != 8 * 1024**2
                or pilot_disk_bytes != 16 * 1024**2 or rehearsal_disk_bytes != 4 * 1024**2):
            raise ValueError('the fixed phase reservations differ')
        result = accounting.reconcile_phase_files(roots, stage=stage, replay_tensor_bytes=replay_tensor_bytes)
        if type(result) is not dict or result.get('status') != 'bounded' or result.get('stage') != stage:
            raise ValueError('phase-side accounting did not verify')
        return True

    def verified_runtime(runtime, deviation, plan, result, binding):
        return (runtime_check(runtime, deviation, plan, result, binding) is True
                and verify_replay_evidence(plan, result, replay_tensor_bytes) is True)

    if (observe_replay_config() != handoff.replay_config_bytes
            or observe_claimant_exit(handoff.receipt_sha256) is not True
            or verify_replay_evidence(public.runtime_replay_plan_bytes, public.runtime_replay_evidence_bytes,
                                      replay_tensor_bytes) is not True):
        raise ValueError('observed claimant or replay evidence is incomplete')
    custody._check_study(public)
    verified_claim(ledger, 'injection_heldout', output_path=phase)
    _, snapshot, _ = custody._check_inputs(public, site_nonce, site_record_bytes, publication_check,
                                           source_check, prediction_check, verified_runtime)
    if observe_inventory() != snapshot.sha256:
        raise ValueError('the selected exposure inventory is not current')
    return custody.prepare_study_preclaimed(
        public, parent, ledger, verify_preclaim=verified_claim, protect_preclaim=protected_claim,
        reconcile_phase=reconcile, config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(), site_nonce=site_nonce,
        site_record_bytes=site_record_bytes, tokenizer=tokenizer, publication_check=publication_check,
        source_check=source_check, prediction_check=prediction_check, gate_allocation=gate_allocation,
        observe_inventory=observe_inventory, runtime_check=verified_runtime)
