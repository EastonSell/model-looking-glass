"""Reserve fixed agreement subject-dependence inputs; no forward is asserted."""
import copy
import hashlib
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v26 as v26

INVENTORY_VERSION = "distilgpt2-development-exposures-v27"
PREDECESSOR_SHA256 = "8fd80cb0b24225100628489dcd0fe2b7efd11e87dfcbb624cf039e54a8a3a62d"
INVENTORY_SHA256 = "ef42683a664dda450e82c69afedda34a8a34f2daf9f50f2db54f6c9d846f07e5"
FORBIDDEN_PREFIX_COUNT = 4623
CASES_COMMIT = "4db27a4f6d96a97c28ad644b911b9c5eacd4f55d"
CASE_PATH = "experiments/distilgpt2_agreement_subject_dependence_cases_v1.json"
CASE_SHA256 = "5f3a2c850454e74da2bdc3819d492357f829994c126e1ac95d90ca3813b0643e"
PROTOCOL_PATH = "docs/AGREEMENT_SUBJECT_DEPENDENCE_V1.md"
_canonical = v26._canonical
_prefixes = v26._prefixes
for _name in ('TARGET', 'PUBLIC_DISCOVERY', 'FACTORIAL_GROUPS', 'FACTORIAL_ROWS',
              'RECIPIENT_CASES', 'INJECTION_RECORD_IDS', 'PAIRS_DOSE_GROUPS',
              'PAIRS_DOSE_ROWS', 'YEAR_TRANSFER_CASES', 'YEAR_TRANSFER_ROWS',
              'GROUPED_REPORTED_ROWS', 'GROUPED_POLICY_ROWS', 'GROUPED_BROKER_OBSERVED_ROWS',
              'SCREENED_CASES', 'SCREENED_ROWS', 'SCREENED_OBSERVED_CASES',
              'SCREENED_PLANNED_LOCALITY_CASES', 'AGREEMENT_CASES', 'AGREEMENT_ROWS',
              'AGREEMENT_ANSWER_IDS', 'EXPLICIT_YEAR_DIRECTION_CASES',
              'EXPLICIT_YEAR_DIRECTION_ROWS', 'YEAR_LABEL_CONTROL_CASES',
              'YEAR_LABEL_CONTROL_ROWS', 'FULL_TOP_CASES', 'FULL_TOP_ROWS'):
    globals()[_name] = getattr(v26, _name)

AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS = (318, 389, 373, 547, 262, 257, 287, 319)
AGREEMENT_SUBJECT_FOLLOWUP_CASES = (('agreement-subject-f01-c0-R1-singular',
  'At dawn, the horse nearby',
  (2953, 17577, 11, 262, 8223, 6716),
  'target'),
 ('agreement-subject-f01-c0-R1-plural',
  'At dawn, the horses nearby',
  (2953, 17577, 11, 262, 14260, 6716),
  'target'),
 ('agreement-subject-f01-c0-I1-singular',
  'At dawn, the mouse nearby',
  (2953, 17577, 11, 262, 10211, 6716),
  'target'),
 ('agreement-subject-f01-c0-I1-plural', 'At dawn, the mice nearby', (2953, 17577, 11, 262, 10693, 6716), 'target'),
 ('agreement-subject-f01-c0-context', 'At dawn, nearby', (2953, 17577, 11, 6716), 'context'),
 ('agreement-subject-f01-c1-R1-singular',
  'At dawn, the horse near the water',
  (2953, 17577, 11, 262, 8223, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f01-c1-R1-plural',
  'At dawn, the horses near the water',
  (2953, 17577, 11, 262, 14260, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f01-c1-I1-singular',
  'At dawn, the mouse near the water',
  (2953, 17577, 11, 262, 10211, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f01-c1-I1-plural',
  'At dawn, the mice near the water',
  (2953, 17577, 11, 262, 10693, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f01-c1-context', 'At dawn, near the water', (2953, 17577, 11, 1474, 262, 1660), 'context'),
 ('agreement-subject-f01-c2-R1-singular',
  'At dawn, the horse near the bridge',
  (2953, 17577, 11, 262, 8223, 1474, 262, 7696),
  'target'),
 ('agreement-subject-f01-c2-R1-plural',
  'At dawn, the horses near the bridge',
  (2953, 17577, 11, 262, 14260, 1474, 262, 7696),
  'target'),
 ('agreement-subject-f01-c2-I1-singular',
  'At dawn, the mouse near the bridge',
  (2953, 17577, 11, 262, 10211, 1474, 262, 7696),
  'target'),
 ('agreement-subject-f01-c2-I1-plural',
  'At dawn, the mice near the bridge',
  (2953, 17577, 11, 262, 10693, 1474, 262, 7696),
  'target'),
 ('agreement-subject-f01-c2-context', 'At dawn, near the bridge', (2953, 17577, 11, 1474, 262, 7696), 'context'),
 ('agreement-subject-f01-c3-R1-singular',
  'At dawn, the horse near the bridges',
  (2953, 17577, 11, 262, 8223, 1474, 262, 19432),
  'target'),
 ('agreement-subject-f01-c3-R1-plural',
  'At dawn, the horses near the bridges',
  (2953, 17577, 11, 262, 14260, 1474, 262, 19432),
  'target'),
 ('agreement-subject-f01-c3-I1-singular',
  'At dawn, the mouse near the bridges',
  (2953, 17577, 11, 262, 10211, 1474, 262, 19432),
  'target'),
 ('agreement-subject-f01-c3-I1-plural',
  'At dawn, the mice near the bridges',
  (2953, 17577, 11, 262, 10693, 1474, 262, 19432),
  'target'),
 ('agreement-subject-f01-c3-context', 'At dawn, near the bridges', (2953, 17577, 11, 1474, 262, 19432), 'context'),
 ('agreement-subject-f02-c0-R1-singular',
  'At dusk, the horse nearby',
  (2953, 46166, 11, 262, 8223, 6716),
  'target'),
 ('agreement-subject-f02-c0-R1-plural',
  'At dusk, the horses nearby',
  (2953, 46166, 11, 262, 14260, 6716),
  'target'),
 ('agreement-subject-f02-c0-I2-singular',
  'At dusk, the wolf nearby',
  (2953, 46166, 11, 262, 17481, 6716),
  'target'),
 ('agreement-subject-f02-c0-I2-plural',
  'At dusk, the wolves nearby',
  (2953, 46166, 11, 262, 23214, 6716),
  'target'),
 ('agreement-subject-f02-c0-context', 'At dusk, nearby', (2953, 46166, 11, 6716), 'context'),
 ('agreement-subject-f02-c1-R1-singular',
  'At dusk, the horse near the water',
  (2953, 46166, 11, 262, 8223, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f02-c1-R1-plural',
  'At dusk, the horses near the water',
  (2953, 46166, 11, 262, 14260, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f02-c1-I2-singular',
  'At dusk, the wolf near the water',
  (2953, 46166, 11, 262, 17481, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f02-c1-I2-plural',
  'At dusk, the wolves near the water',
  (2953, 46166, 11, 262, 23214, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f02-c1-context', 'At dusk, near the water', (2953, 46166, 11, 1474, 262, 1660), 'context'),
 ('agreement-subject-f02-c2-R1-singular',
  'At dusk, the horse near the road',
  (2953, 46166, 11, 262, 8223, 1474, 262, 2975),
  'target'),
 ('agreement-subject-f02-c2-R1-plural',
  'At dusk, the horses near the road',
  (2953, 46166, 11, 262, 14260, 1474, 262, 2975),
  'target'),
 ('agreement-subject-f02-c2-I2-singular',
  'At dusk, the wolf near the road',
  (2953, 46166, 11, 262, 17481, 1474, 262, 2975),
  'target'),
 ('agreement-subject-f02-c2-I2-plural',
  'At dusk, the wolves near the road',
  (2953, 46166, 11, 262, 23214, 1474, 262, 2975),
  'target'),
 ('agreement-subject-f02-c2-context', 'At dusk, near the road', (2953, 46166, 11, 1474, 262, 2975), 'context'),
 ('agreement-subject-f02-c3-R1-singular',
  'At dusk, the horse near the roads',
  (2953, 46166, 11, 262, 8223, 1474, 262, 9725),
  'target'),
 ('agreement-subject-f02-c3-R1-plural',
  'At dusk, the horses near the roads',
  (2953, 46166, 11, 262, 14260, 1474, 262, 9725),
  'target'),
 ('agreement-subject-f02-c3-I2-singular',
  'At dusk, the wolf near the roads',
  (2953, 46166, 11, 262, 17481, 1474, 262, 9725),
  'target'),
 ('agreement-subject-f02-c3-I2-plural',
  'At dusk, the wolves near the roads',
  (2953, 46166, 11, 262, 23214, 1474, 262, 9725),
  'target'),
 ('agreement-subject-f02-c3-context', 'At dusk, near the roads', (2953, 46166, 11, 1474, 262, 9725), 'context'),
 ('agreement-subject-f03-c0-R2-singular', 'At noon, the bird nearby', (2953, 19613, 11, 262, 6512, 6716), 'target'),
 ('agreement-subject-f03-c0-R2-plural', 'At noon, the birds nearby', (2953, 19613, 11, 262, 10087, 6716), 'target'),
 ('agreement-subject-f03-c0-I1-singular',
  'At noon, the mouse nearby',
  (2953, 19613, 11, 262, 10211, 6716),
  'target'),
 ('agreement-subject-f03-c0-I1-plural', 'At noon, the mice nearby', (2953, 19613, 11, 262, 10693, 6716), 'target'),
 ('agreement-subject-f03-c0-context', 'At noon, nearby', (2953, 19613, 11, 6716), 'context'),
 ('agreement-subject-f03-c1-R2-singular',
  'At noon, the bird near the water',
  (2953, 19613, 11, 262, 6512, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f03-c1-R2-plural',
  'At noon, the birds near the water',
  (2953, 19613, 11, 262, 10087, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f03-c1-I1-singular',
  'At noon, the mouse near the water',
  (2953, 19613, 11, 262, 10211, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f03-c1-I1-plural',
  'At noon, the mice near the water',
  (2953, 19613, 11, 262, 10693, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f03-c1-context', 'At noon, near the water', (2953, 19613, 11, 1474, 262, 1660), 'context'),
 ('agreement-subject-f03-c2-R2-singular',
  'At noon, the bird near the ship',
  (2953, 19613, 11, 262, 6512, 1474, 262, 4074),
  'target'),
 ('agreement-subject-f03-c2-R2-plural',
  'At noon, the birds near the ship',
  (2953, 19613, 11, 262, 10087, 1474, 262, 4074),
  'target'),
 ('agreement-subject-f03-c2-I1-singular',
  'At noon, the mouse near the ship',
  (2953, 19613, 11, 262, 10211, 1474, 262, 4074),
  'target'),
 ('agreement-subject-f03-c2-I1-plural',
  'At noon, the mice near the ship',
  (2953, 19613, 11, 262, 10693, 1474, 262, 4074),
  'target'),
 ('agreement-subject-f03-c2-context', 'At noon, near the ship', (2953, 19613, 11, 1474, 262, 4074), 'context'),
 ('agreement-subject-f03-c3-R2-singular',
  'At noon, the bird near the ships',
  (2953, 19613, 11, 262, 6512, 1474, 262, 7937),
  'target'),
 ('agreement-subject-f03-c3-R2-plural',
  'At noon, the birds near the ships',
  (2953, 19613, 11, 262, 10087, 1474, 262, 7937),
  'target'),
 ('agreement-subject-f03-c3-I1-singular',
  'At noon, the mouse near the ships',
  (2953, 19613, 11, 262, 10211, 1474, 262, 7937),
  'target'),
 ('agreement-subject-f03-c3-I1-plural',
  'At noon, the mice near the ships',
  (2953, 19613, 11, 262, 10693, 1474, 262, 7937),
  'target'),
 ('agreement-subject-f03-c3-context', 'At noon, near the ships', (2953, 19613, 11, 1474, 262, 7937), 'context'),
 ('agreement-subject-f04-c0-R2-singular', 'In summer, the bird nearby', (818, 3931, 11, 262, 6512, 6716), 'target'),
 ('agreement-subject-f04-c0-R2-plural', 'In summer, the birds nearby', (818, 3931, 11, 262, 10087, 6716), 'target'),
 ('agreement-subject-f04-c0-I2-singular',
  'In summer, the wolf nearby',
  (818, 3931, 11, 262, 17481, 6716),
  'target'),
 ('agreement-subject-f04-c0-I2-plural',
  'In summer, the wolves nearby',
  (818, 3931, 11, 262, 23214, 6716),
  'target'),
 ('agreement-subject-f04-c0-context', 'In summer, nearby', (818, 3931, 11, 6716), 'context'),
 ('agreement-subject-f04-c1-R2-singular',
  'In summer, the bird near the water',
  (818, 3931, 11, 262, 6512, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f04-c1-R2-plural',
  'In summer, the birds near the water',
  (818, 3931, 11, 262, 10087, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f04-c1-I2-singular',
  'In summer, the wolf near the water',
  (818, 3931, 11, 262, 17481, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f04-c1-I2-plural',
  'In summer, the wolves near the water',
  (818, 3931, 11, 262, 23214, 1474, 262, 1660),
  'target'),
 ('agreement-subject-f04-c1-context', 'In summer, near the water', (818, 3931, 11, 1474, 262, 1660), 'context'),
 ('agreement-subject-f04-c2-R2-singular',
  'In summer, the bird near the hill',
  (818, 3931, 11, 262, 6512, 1474, 262, 12788),
  'target'),
 ('agreement-subject-f04-c2-R2-plural',
  'In summer, the birds near the hill',
  (818, 3931, 11, 262, 10087, 1474, 262, 12788),
  'target'),
 ('agreement-subject-f04-c2-I2-singular',
  'In summer, the wolf near the hill',
  (818, 3931, 11, 262, 17481, 1474, 262, 12788),
  'target'),
 ('agreement-subject-f04-c2-I2-plural',
  'In summer, the wolves near the hill',
  (818, 3931, 11, 262, 23214, 1474, 262, 12788),
  'target'),
 ('agreement-subject-f04-c2-context', 'In summer, near the hill', (818, 3931, 11, 1474, 262, 12788), 'context'),
 ('agreement-subject-f04-c3-R2-singular',
  'In summer, the bird near the hills',
  (818, 3931, 11, 262, 6512, 1474, 262, 18639),
  'target'),
 ('agreement-subject-f04-c3-R2-plural',
  'In summer, the birds near the hills',
  (818, 3931, 11, 262, 10087, 1474, 262, 18639),
  'target'),
 ('agreement-subject-f04-c3-I2-singular',
  'In summer, the wolf near the hills',
  (818, 3931, 11, 262, 17481, 1474, 262, 18639),
  'target'),
 ('agreement-subject-f04-c3-I2-plural',
  'In summer, the wolves near the hills',
  (818, 3931, 11, 262, 23214, 1474, 262, 18639),
  'target'),
 ('agreement-subject-f04-c3-context', 'In summer, near the hills', (818, 3931, 11, 1474, 262, 18639), 'context'))
AGREEMENT_SUBJECT_FOLLOWUP_ROWS = tuple(row for _, _, row, _ in AGREEMENT_SUBJECT_FOLLOWUP_CASES)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    old_ids = [record['record_id'] for record in records]
    if len(old_ids) != len(set(old_ids)):
        raise ValueError('predecessor record identities repeat')
    cases = AGREEMENT_SUBJECT_FOLLOWUP_CASES
    if type(cases) is not tuple or len(cases) != 80 or any(type(c) is not tuple or len(c) != 4 for c in cases):
        raise ValueError('expected 80 fixed subject-dependence cases')
    ids = [c[0] for c in cases]
    rows = [c[2] for c in cases]
    if (len(set(ids)) != 80 or len(set(rows)) != 80
            or any(type(i) is not str or re.fullmatch(r'agreement-subject-[A-Za-z0-9-]+', i) is None for i in ids)
            or any('planned-' + i in old_ids for i in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 12
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)
            or any(role not in ('target', 'context') for _, _, _, role in cases)
            or sum(role == 'target' for _, _, _, role in cases) != 64
            or sum(role == 'context' for _, _, _, role in cases) != 16):
        raise ValueError('subject-dependence identities, roles or token rows differ')
    old_sequences = [tuple(r['token_ids']) for r in records]
    old_sequences.extend(tuple(r['token_ids'] + [r['target_token_id']])
                         for r in records if type(r.get('target_token_id')) is int)
    old_sequences.extend(tuple(r['token_ids'] + [token]) for r in records
                         for token in r.get('candidate_token_ids', ()))
    old_sequences.extend(tuple(r['token_ids'] + [token]) for r in records
                         for group in r.get('candidate_token_groups', {}).values() for token in group)
    proposed = [(index, seq) for index, row in enumerate(rows)
                for seq in (row,) + tuple(row + (answer,) for answer in AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS)]
    if any(_related(seq, old) for _, seq in proposed for old in old_sequences):
        raise ValueError('subject-dependence input or continuation has a known full-row prefix relation')
    if any(i != j and _related(left, right) for offset, (i, left) in enumerate(proposed)
           for j, right in proposed[offset + 1:]):
        raise ValueError('subject-dependence cases have a full-row prefix relation')
    for case_id, text, row, role in cases:
        records.append({'record_id': 'planned-' + case_id, 'category': 'planned',
                        'token_ids': list(row), 'public_text': text, 'case_role': role,
                        'candidate_token_ids': list(AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS),
                        'provenance': 'Public input planned for a subject-dependence study; '
                                      f'no model forward is asserted. Case SHA-256 {CASE_SHA256}.',
                        'references': [PROTOCOL_PATH, CASE_PATH], 'reference_commit': CASES_COMMIT})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': v26.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_agreement_subject_followup_rows': 80}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('subject-dependence input prefix count changed')
    return payload


def inventory_payload():
    predecessor = v26.inventory_payload()
    if (v26.INVENTORY_SHA256 != PREDECESSOR_SHA256
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
