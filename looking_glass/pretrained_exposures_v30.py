"""Reconcile saved year/label and full-top baseline input statuses."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v29 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v30'
PREDECESSOR_SHA256 = 'f2dc3ff6cc8befc600cc62de68c9c804f8962c7261b87cad3c94ea7744bb7df6'
INVENTORY_SHA256 = '921025a7f9c1246c5f10e97f2b987ca4971462241b2d05ebbc525d4aa0736c79'
FORBIDDEN_PREFIX_COUNT = 5391
YEAR_V1_COMMIT = '998d121ee85d12e131b1244f282cbd4313e1a9dc'
YEAR_V2_COMMIT = '89b95e699cbdb94ba6ce29c94f5b82becf752f30'
FULL_TOP_COMMIT = '80ec268f6c70463465e689cb9f1b7630ed11fc36'
YEAR_CASE_SHA256 = '07d68842ab7f5195a61e719f4bafd825ab19673971a7e4ea450330838deafa08'
YEAR_V1_MANIFEST_SHA256 = 'd43d181e1c7046c8e97e6310bd3edd38fe85089e0e69347cd2ddb2728d29d47a'
YEAR_V1_JOURNAL_SHA256 = 'a2f5ba4f980bde155bf4eb5729f154f7422956eb9c0920421f13327038e057f5'
YEAR_V1_RECEIPT_SHA256 = 'f3b418c7760ed23e2f3c3f90bd39b33538b2b9913ffad2dcf0dd7546742f55bf'
YEAR_V2_MANIFEST_SHA256 = '6a8ade8ab8bdcddc8a686c15f1ddb43b8db9a125d83bb49825fa8f2639510903'
YEAR_V2_JOURNAL_SHA256 = '0c5132426cf1e855c93fa9f3fd4828f6c734d971522efa3a3f84ae354ff24f75'
YEAR_V2_RECEIPT_SHA256 = '8e41226e78793580e09985232e8dd68415ac54c5863c4157189e08835d387978'
FULL_TOP_CASE_SHA256 = '4057696ddef9e3b0d09db02f46e6454561a61ac1a07f8b8a13b92718944f4261'
FULL_TOP_MANIFEST_SHA256 = '458c934bbe74ad01fd0d6abe6d4686aa1e47cbb993f4b76e04d451d07b3e56bb'
FULL_TOP_PHASE_SHA256 = '646aab4c7ee527c687c99c06408db4fc81850670f23c49a5db46f1bfd8f089f1'
FULL_TOP_JOURNAL_SHA256 = '28f29deac63fab2599a67ebf4537f14723e374377f2205c2cfe08300f7de7e57'
FULL_TOP_SELECTION_SHA256 = 'b41bc1240a8b43fa3810ab7bd27412d7a07341df4b79e7fbdef61d32d4f9ead9'
FULL_TOP_RAW_SHA256 = '62a04b0d13472a882ed56c9248391c73456cf4836007869e8e786fad517cf5b4'
YEAR_GROUPS = ('above_ids', 'middle_ids', 'local_above_ids', 'local_below_ids')
YEAR_V1_ROOT = 'results/distilgpt2-year-label-control-v1-20261029/'
YEAR_V2_ROOT = 'results/distilgpt2-year-label-control-v2-20261029/'
FULL_TOP_ROOT = 'results/targeted-edit-full-top-cpu-v1-20261029/screen/attempt/'

for _name in ('TARGET', 'PUBLIC_DISCOVERY', 'FACTORIAL_GROUPS', 'FACTORIAL_ROWS',
              'RECIPIENT_CASES', 'INJECTION_RECORD_IDS', 'PAIRS_DOSE_GROUPS', 'PAIRS_DOSE_ROWS',
              'YEAR_TRANSFER_CASES', 'YEAR_TRANSFER_ROWS', 'GROUPED_REPORTED_ROWS',
              'GROUPED_POLICY_ROWS', 'GROUPED_BROKER_OBSERVED_ROWS', 'SCREENED_CASES',
              'SCREENED_ROWS', 'SCREENED_OBSERVED_CASES', 'SCREENED_PLANNED_LOCALITY_CASES',
              'AGREEMENT_CASES', 'AGREEMENT_ROWS', 'AGREEMENT_ANSWER_IDS',
              'EXPLICIT_YEAR_DIRECTION_CASES', 'EXPLICIT_YEAR_DIRECTION_ROWS',
              'YEAR_LABEL_CONTROL_CASES', 'YEAR_LABEL_CONTROL_ROWS', 'FULL_TOP_CASES', 'FULL_TOP_ROWS',
              'AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS', 'AGREEMENT_SUBJECT_FOLLOWUP_CASES',
              'AGREEMENT_SUBJECT_FOLLOWUP_ROWS', 'ANSWER_FORMAT_CASES', 'ANSWER_FORMAT_ROWS',
              'TWO_ATTRACTOR_CASES', 'TWO_ATTRACTOR_ROWS'):
    globals()[_name] = getattr(prior, _name)
_canonical = prior._canonical
_prefixes = prior._prefixes


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    existing_ids = [r['record_id'] for r in records]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError('predecessor record identities repeat')
    year = {'planned-' + case_id: (text, row, groups)
            for case_id, text, row, groups in YEAR_LABEL_CONTROL_CASES}
    top = {'planned-' + case_id: (text, row, labels)
           for case_id, text, row, labels in FULL_TOP_CASES if not case_id.startswith('full-top-locality-')}
    locality = {'planned-' + case_id: (text, row, labels)
                for case_id, text, row, labels in FULL_TOP_CASES if case_id.startswith('full-top-locality-')}
    if len(year) != 48 or len(top) != 36 or len(locality) != 12:
        raise ValueError('fixed case counts differ')
    v2_ids = {'planned-' + case[0] for case in YEAR_LABEL_CONTROL_CASES[12:]}
    found_year, found_top, found_locality = set(), set(), set()
    for record in records:
        record_id = record['record_id']
        if record_id.startswith('planned-year-label-'):
            if record_id not in year or record_id in found_year or record.get('category') != 'planned':
                raise ValueError('year/label reservation identity differs')
            text, row, groups = year[record_id]
            if (record.get('public_text') != text or record.get('token_ids') != list(row)
                    or record.get('candidate_token_groups') != {key: list(group) for key, group in zip(YEAR_GROUPS, groups)}):
                raise ValueError('year/label reservation content differs')
            found_year.add(record_id)
            observed_id = 'observed-' + record_id[len('planned-'):]
            if observed_id in existing_ids:
                raise ValueError('observed year/label identity already exists')
            record['record_id'] = observed_id
            record['category'] = 'observed-saved'
            record['provenance'] = (
                'Public non-blind v1 saved clean/no-op inputs for all 48 rows; its four-family '
                'intervention attempt stopped incomplete. ' +
                ('This row was also repeated in the completed exposed-family v2 continuation. ' if record_id in v2_ids
                 else 'This row belongs to the completed first v1 intervention family. ') +
                'Process records are cooperative observations, not authenticated execution. '
                f'V1 manifest/journal/receipt SHA-256: {YEAR_V1_MANIFEST_SHA256}, {YEAR_V1_JOURNAL_SHA256}, {YEAR_V1_RECEIPT_SHA256}. '
                + (f'V2 manifest/journal/receipt SHA-256: {YEAR_V2_MANIFEST_SHA256}, {YEAR_V2_JOURNAL_SHA256}, {YEAR_V2_RECEIPT_SHA256}. '
                   if record_id in v2_ids else '') +
                'The old sealed split and outside/pretraining exposure remain unknown.'
            )
            record['references'] = list(record['references']) + [YEAR_V1_ROOT + p for p in
                ('run/manifest.json', 'run/attempt.jsonl', 'run/failure.json', 'supervisor.json')]
            if record_id in v2_ids:
                record['references'] += [YEAR_V2_ROOT + p for p in
                    ('run/manifest.json', 'run/attempt.jsonl', 'run/result.json', 'supervisor.json')]
            record['reference_commit'] = YEAR_V2_COMMIT if record_id in v2_ids else YEAR_V1_COMMIT
        elif record_id.startswith('planned-full-top-'):
            target = top if record_id in top else locality
            found = found_top if record_id in top else found_locality
            if record_id not in target or record_id in found or record.get('category') != 'planned':
                raise ValueError('full-top reservation identity differs')
            text, row, labels = target[record_id]
            if (record.get('public_text') != text or record.get('token_ids') != list(row)
                    or record.get('candidate_token_ids') != list(labels)):
                raise ValueError('full-top reservation content differs')
            found.add(record_id)
            if record_id in locality:
                continue
            observed_id = 'observed-' + record_id[len('planned-'):]
            if observed_id in existing_ids:
                raise ValueError('observed full-top identity already exists')
            record['record_id'] = observed_id
            record['category'] = 'observed-saved'
            record['provenance'] = (
                'Public CPU full-top baseline screen saved one observation per each of 36 target prompts; '
                'the screen was ineligible and no edit or locality forward was run. '
                'Process records are cooperative observations, not authenticated execution. '
                f'Manifest/phase/journal/selection/raw SHA-256: {FULL_TOP_MANIFEST_SHA256}, {FULL_TOP_PHASE_SHA256}, {FULL_TOP_JOURNAL_SHA256}, '
                f'{FULL_TOP_SELECTION_SHA256}, {FULL_TOP_RAW_SHA256}.'
            )
            record['references'] = list(record['references']) + ['experiments/targeted_edit_full_top_screen_manifest_v1.json'] + [FULL_TOP_ROOT + p for p in
                ('phase-record.json', 'screen/observations.jsonl', 'screen/screen-selection.json', 'screen/screen.f32')]
            record['reference_commit'] = FULL_TOP_COMMIT
    if found_year != set(year) or found_top != set(top) or found_locality != set(locality):
        raise ValueError('status update is missing a fixed reservation')
    if _prefixes(records) != _prefixes(predecessor['records']) or len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('status-only update changed the input prefixes')
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'observed_year_label_rows': 48, 'observed_full_top_target_rows': 36}
    return payload


def inventory_payload():
    predecessor = prior.inventory_payload()
    if (prior.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError('predecessor exposure inventory does not match')
    return _build_payload(predecessor)


def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload = inventory_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError('development-exposure inventory does not match its pinned digest')
    return payload, digest


def inventory_digest():
    return _checked_inventory()[1]


def _forbidden_rows():
    return _prefixes(inventory_payload()['records'])


def _has_known_relation(row, forbidden, complete_rows):
    return row in forbidden or any(row[:len(known)] == known for known in complete_rows)


def check_known_splits(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                       expected_model_identity, cases_per_split=8):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError("expected exposure inventory digest does not match")
    if type(cases_per_split) is not int or cases_per_split not in (8, 16):
        raise ValueError("cases per split must be exactly 8 or 16")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _prefixes(payload["records"])
    complete_rows = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in discovery + heldout):
        raise ValueError("a candidate token sequence is in the development exclusion inventory")
    return v1.KnownSplitCheck(INVENTORY_VERSION, digest, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError("expected exposure inventory digest does not match")
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _prefixes(payload["records"])
    complete_rows = {tuple(row["token_ids"]) for row in payload["records"]}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in heldout):
        raise ValueError("a held-out token sequence is in the development exclusion inventory")
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, digest, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete_rows) for row in discovery))


def _validate_known_check(check):
    payload, digest = _checked_inventory()
    count = len(_prefixes(payload["records"]))
    if (type(check) not in (v1.KnownSplitCheck, v2.ExposedDiscoveryCheck)
            or check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != digest
            or type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int
            or check.forbidden_row_count != count or check.complete_exposure_history_certified is not False
            or check.preseal_ready is not False):
        raise ValueError("a matching known-split check is required")
    if type(check) is v1.KnownSplitCheck and check.scope == "known-rows-only" and check.candidate_count in (16, 32):
        return count, "known-prefixes-excluded-from-both", 0
    if (type(check) is v2.ExposedDiscoveryCheck and check.scope == "known-prefix-holdout-only"
            and check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int
            and 0 <= check.known_or_reserved_discovery_overlap_count <= 16):
        return count, "known-prefix-discovery-reuse-disclosed", check.known_or_reserved_discovery_overlap_count
    raise ValueError("a matching known-split check is required")


def require_preseal_readiness(check):
    _validate_known_check(check)
    raise v1.ExposureInventoryIncomplete("known-row checking alone does not authorize a study seal")


def prepare_preseal_disclosure(check, *, protocol_sha256, acknowledge_prior_exposure_uncertainty=False):
    count, policy, overlap = _validate_known_check(check)
    if acknowledge_prior_exposure_uncertainty is not True:
        raise v1.ExposureInventoryIncomplete("prior exposure uncertainty must be explicitly disclosed")
    if type(protocol_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is None:
        raise ValueError("a canonical protocol SHA-256 is required")
    return v2.PresealDisclosure(INVENTORY_VERSION, check.inventory_sha256, protocol_sha256,
                                 check.candidate_count // 2, count, policy, overlap)
