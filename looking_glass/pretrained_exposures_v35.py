"""Reserve 54 public noun-substitution inputs; no forward is asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v34 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v35'
PREDECESSOR_SHA256 = 'e5ae202c1e3449a67e92788a83cd0dc62d404a3fa1e052716a11770ee92cc00c'
INVENTORY_SHA256 = '3a7ecf81eb7f04759ed5b8ffd0607c5955b634b938756e76eb52d2435aef9610'
FORBIDDEN_PREFIX_COUNT = 6129
CASE_COMMIT = '7a7ee955359456898b1644ff4f050df69183c05e'
CASE_PATH = 'experiments/distilgpt2_agreement_noun_substitution_cases_v1.json'
CASE_SHA256 = '94910e59415936a09c18f81478e0eaffb790c9b69d0962ab3f6b70947fbe1dfe'
REFERENCES = ('docs/AGREEMENT_NOUN_SUBSTITUTION_V1.md', CASE_PATH)

for _name in ('TARGET', 'PUBLIC_DISCOVERY', 'FACTORIAL_GROUPS', 'FACTORIAL_ROWS',
              'RECIPIENT_CASES', 'INJECTION_RECORD_IDS', 'PAIRS_DOSE_GROUPS', 'PAIRS_DOSE_ROWS',
              'YEAR_TRANSFER_CASES', 'YEAR_TRANSFER_ROWS', 'GROUPED_REPORTED_ROWS', 'GROUPED_POLICY_ROWS',
              'GROUPED_BROKER_OBSERVED_ROWS', 'SCREENED_CASES', 'SCREENED_ROWS',
              'SCREENED_OBSERVED_CASES', 'SCREENED_PLANNED_LOCALITY_CASES', 'AGREEMENT_CASES',
              'AGREEMENT_ROWS', 'AGREEMENT_ANSWER_IDS', 'EXPLICIT_YEAR_DIRECTION_CASES',
              'EXPLICIT_YEAR_DIRECTION_ROWS', 'YEAR_LABEL_CONTROL_CASES', 'YEAR_LABEL_CONTROL_ROWS',
              'FULL_TOP_CASES', 'FULL_TOP_ROWS', 'AGREEMENT_SUBJECT_FOLLOWUP_ANSWER_IDS',
              'AGREEMENT_SUBJECT_FOLLOWUP_CASES', 'AGREEMENT_SUBJECT_FOLLOWUP_ROWS',
              'ANSWER_FORMAT_CASES', 'ANSWER_FORMAT_ROWS', 'TWO_ATTRACTOR_CASES',
              'TWO_ATTRACTOR_ROWS', 'COPY_PROMPT_DOSE_CASES', 'COPY_PROMPT_DOSE_ROWS',
              'SYNTHETIC_SERVICE_CASES', 'TWO_ATTRACTOR_PATCH_CASES', 'TWO_ATTRACTOR_PATCH_ROWS',
              'SCIENCE_SERVICE_CASES'):
    globals()[_name] = getattr(prior, _name)
_canonical = prior._canonical
_prefixes = prior._prefixes


KNOWN_INJECTION_CASES = prior.KNOWN_INJECTION_CASES
KNOWN_INJECTION_ROWS = prior.KNOWN_INJECTION_ROWS
AGREEMENT_NOUN_SUBSTITUTION_ANSWER_IDS = (318, 389, 373, 547, 262, 257, 287, 319)
AGREEMENT_NOUN_SUBSTITUTION_CASES = (('noun-sub-f1-c0-R1-singular',
  'target',
  'Before sunrise, the hand nearby',
  (8421, 40048, 11, 262, 1021, 6716)),
 ('noun-sub-f1-c0-R1-plural',
  'target',
  'Before sunrise, the hands nearby',
  (8421, 40048, 11, 262, 2832, 6716)),
 ('noun-sub-f1-c0-R2-singular',
  'target',
  'Before sunrise, the arm nearby',
  (8421, 40048, 11, 262, 3211, 6716)),
 ('noun-sub-f1-c0-R2-plural',
  'target',
  'Before sunrise, the arms nearby',
  (8421, 40048, 11, 262, 5101, 6716)),
 ('noun-sub-f1-c0-I1-singular',
  'target',
  'Before sunrise, the foot nearby',
  (8421, 40048, 11, 262, 2366, 6716)),
 ('noun-sub-f1-c0-I1-plural',
  'target',
  'Before sunrise, the feet nearby',
  (8421, 40048, 11, 262, 3625, 6716)),
 ('noun-sub-f1-c0-I2-singular',
  'target',
  'Before sunrise, the tooth nearby',
  (8421, 40048, 11, 262, 16162, 6716)),
 ('noun-sub-f1-c0-I2-plural',
  'target',
  'Before sunrise, the teeth nearby',
  (8421, 40048, 11, 262, 9941, 6716)),
 ('noun-sub-f1-c0-context', 'context', 'Before sunrise, nearby', (8421, 40048, 11, 6716)),
 ('noun-sub-f1-c1-R1-singular',
  'target',
  'Before sunrise, the hand near the stone',
  (8421, 40048, 11, 262, 1021, 1474, 262, 7815)),
 ('noun-sub-f1-c1-R1-plural',
  'target',
  'Before sunrise, the hands near the stone',
  (8421, 40048, 11, 262, 2832, 1474, 262, 7815)),
 ('noun-sub-f1-c1-R2-singular',
  'target',
  'Before sunrise, the arm near the stone',
  (8421, 40048, 11, 262, 3211, 1474, 262, 7815)),
 ('noun-sub-f1-c1-R2-plural',
  'target',
  'Before sunrise, the arms near the stone',
  (8421, 40048, 11, 262, 5101, 1474, 262, 7815)),
 ('noun-sub-f1-c1-I1-singular',
  'target',
  'Before sunrise, the foot near the stone',
  (8421, 40048, 11, 262, 2366, 1474, 262, 7815)),
 ('noun-sub-f1-c1-I1-plural',
  'target',
  'Before sunrise, the feet near the stone',
  (8421, 40048, 11, 262, 3625, 1474, 262, 7815)),
 ('noun-sub-f1-c1-I2-singular',
  'target',
  'Before sunrise, the tooth near the stone',
  (8421, 40048, 11, 262, 16162, 1474, 262, 7815)),
 ('noun-sub-f1-c1-I2-plural',
  'target',
  'Before sunrise, the teeth near the stone',
  (8421, 40048, 11, 262, 9941, 1474, 262, 7815)),
 ('noun-sub-f1-c1-context', 'context', 'Before sunrise, near the stone', (8421, 40048, 11, 1474, 262, 7815)),
 ('noun-sub-f1-c2-R1-singular',
  'target',
  'Before sunrise, the hand near the stones',
  (8421, 40048, 11, 262, 1021, 1474, 262, 14966)),
 ('noun-sub-f1-c2-R1-plural',
  'target',
  'Before sunrise, the hands near the stones',
  (8421, 40048, 11, 262, 2832, 1474, 262, 14966)),
 ('noun-sub-f1-c2-R2-singular',
  'target',
  'Before sunrise, the arm near the stones',
  (8421, 40048, 11, 262, 3211, 1474, 262, 14966)),
 ('noun-sub-f1-c2-R2-plural',
  'target',
  'Before sunrise, the arms near the stones',
  (8421, 40048, 11, 262, 5101, 1474, 262, 14966)),
 ('noun-sub-f1-c2-I1-singular',
  'target',
  'Before sunrise, the foot near the stones',
  (8421, 40048, 11, 262, 2366, 1474, 262, 14966)),
 ('noun-sub-f1-c2-I1-plural',
  'target',
  'Before sunrise, the feet near the stones',
  (8421, 40048, 11, 262, 3625, 1474, 262, 14966)),
 ('noun-sub-f1-c2-I2-singular',
  'target',
  'Before sunrise, the tooth near the stones',
  (8421, 40048, 11, 262, 16162, 1474, 262, 14966)),
 ('noun-sub-f1-c2-I2-plural',
  'target',
  'Before sunrise, the teeth near the stones',
  (8421, 40048, 11, 262, 9941, 1474, 262, 14966)),
 ('noun-sub-f1-c2-context',
  'context',
  'Before sunrise, near the stones',
  (8421, 40048, 11, 1474, 262, 14966)),
 ('noun-sub-f2-c0-R1-singular',
  'target',
  'After sunset, the hand nearby',
  (3260, 26428, 11, 262, 1021, 6716)),
 ('noun-sub-f2-c0-R1-plural', 'target', 'After sunset, the hands nearby', (3260, 26428, 11, 262, 2832, 6716)),
 ('noun-sub-f2-c0-R2-singular', 'target', 'After sunset, the arm nearby', (3260, 26428, 11, 262, 3211, 6716)),
 ('noun-sub-f2-c0-R2-plural', 'target', 'After sunset, the arms nearby', (3260, 26428, 11, 262, 5101, 6716)),
 ('noun-sub-f2-c0-I1-singular',
  'target',
  'After sunset, the foot nearby',
  (3260, 26428, 11, 262, 2366, 6716)),
 ('noun-sub-f2-c0-I1-plural', 'target', 'After sunset, the feet nearby', (3260, 26428, 11, 262, 3625, 6716)),
 ('noun-sub-f2-c0-I2-singular',
  'target',
  'After sunset, the tooth nearby',
  (3260, 26428, 11, 262, 16162, 6716)),
 ('noun-sub-f2-c0-I2-plural', 'target', 'After sunset, the teeth nearby', (3260, 26428, 11, 262, 9941, 6716)),
 ('noun-sub-f2-c0-context', 'context', 'After sunset, nearby', (3260, 26428, 11, 6716)),
 ('noun-sub-f2-c1-R1-singular',
  'target',
  'After sunset, the hand near the wall',
  (3260, 26428, 11, 262, 1021, 1474, 262, 3355)),
 ('noun-sub-f2-c1-R1-plural',
  'target',
  'After sunset, the hands near the wall',
  (3260, 26428, 11, 262, 2832, 1474, 262, 3355)),
 ('noun-sub-f2-c1-R2-singular',
  'target',
  'After sunset, the arm near the wall',
  (3260, 26428, 11, 262, 3211, 1474, 262, 3355)),
 ('noun-sub-f2-c1-R2-plural',
  'target',
  'After sunset, the arms near the wall',
  (3260, 26428, 11, 262, 5101, 1474, 262, 3355)),
 ('noun-sub-f2-c1-I1-singular',
  'target',
  'After sunset, the foot near the wall',
  (3260, 26428, 11, 262, 2366, 1474, 262, 3355)),
 ('noun-sub-f2-c1-I1-plural',
  'target',
  'After sunset, the feet near the wall',
  (3260, 26428, 11, 262, 3625, 1474, 262, 3355)),
 ('noun-sub-f2-c1-I2-singular',
  'target',
  'After sunset, the tooth near the wall',
  (3260, 26428, 11, 262, 16162, 1474, 262, 3355)),
 ('noun-sub-f2-c1-I2-plural',
  'target',
  'After sunset, the teeth near the wall',
  (3260, 26428, 11, 262, 9941, 1474, 262, 3355)),
 ('noun-sub-f2-c1-context', 'context', 'After sunset, near the wall', (3260, 26428, 11, 1474, 262, 3355)),
 ('noun-sub-f2-c2-R1-singular',
  'target',
  'After sunset, the hand near the walls',
  (3260, 26428, 11, 262, 1021, 1474, 262, 7714)),
 ('noun-sub-f2-c2-R1-plural',
  'target',
  'After sunset, the hands near the walls',
  (3260, 26428, 11, 262, 2832, 1474, 262, 7714)),
 ('noun-sub-f2-c2-R2-singular',
  'target',
  'After sunset, the arm near the walls',
  (3260, 26428, 11, 262, 3211, 1474, 262, 7714)),
 ('noun-sub-f2-c2-R2-plural',
  'target',
  'After sunset, the arms near the walls',
  (3260, 26428, 11, 262, 5101, 1474, 262, 7714)),
 ('noun-sub-f2-c2-I1-singular',
  'target',
  'After sunset, the foot near the walls',
  (3260, 26428, 11, 262, 2366, 1474, 262, 7714)),
 ('noun-sub-f2-c2-I1-plural',
  'target',
  'After sunset, the feet near the walls',
  (3260, 26428, 11, 262, 3625, 1474, 262, 7714)),
 ('noun-sub-f2-c2-I2-singular',
  'target',
  'After sunset, the tooth near the walls',
  (3260, 26428, 11, 262, 16162, 1474, 262, 7714)),
 ('noun-sub-f2-c2-I2-plural',
  'target',
  'After sunset, the teeth near the walls',
  (3260, 26428, 11, 262, 9941, 1474, 262, 7714)),
 ('noun-sub-f2-c2-context', 'context', 'After sunset, near the walls', (3260, 26428, 11, 1474, 262, 7714)))
AGREEMENT_NOUN_SUBSTITUTION_ROWS = tuple(row for _, _, _, row in AGREEMENT_NOUN_SUBSTITUTION_CASES)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    old_ids = [record['record_id'] for record in records]
    cases = AGREEMENT_NOUN_SUBSTITUTION_CASES
    if type(cases) is not tuple or len(cases) != 54 or len(set(old_ids)) != len(old_ids):
        raise ValueError('invalid fixed cases or predecessor identities')
    ids, rows = [item[0] for item in cases], [item[3] for item in cases]
    if (len(set(ids)) != 54 or len(set(rows)) != 54
            or any('planned-' + item in old_ids for item in ids)
            or any(role not in ('target', 'context') for _, role, _, _ in cases)
            or sum(role == 'target' for _, role, _, _ in cases) != 48
            or any(type(text) is not str or not text or len(text) > 256 for _, _, text, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 12
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)):
        raise ValueError('fixed case identity, role or token row differs')
    old = [tuple(record['token_ids']) for record in records]
    old.extend(tuple(record['token_ids'] + [record['target_token_id']]) for record in records
               if type(record.get('target_token_id')) is int)
    old.extend(tuple(record['token_ids'] + [token]) for record in records
               for token in record.get('candidate_token_ids', ()))
    old.extend(tuple(record['token_ids'] + [token]) for record in records
               for group in record.get('candidate_token_groups', {}).values() for token in group)
    groups = [[row] + [row + (token,) for token in AGREEMENT_NOUN_SUBSTITUTION_ANSWER_IDS] for row in rows]
    proposed = [row for group in groups for row in group]
    if any(_related(row, previous) for row in proposed for previous in old):
        raise ValueError('substitution input or continuation has a known prefix relation')
    if any(_related(left, right) for i, group in enumerate(groups) for other in groups[i+1:]
           for left in group for right in other):
        raise ValueError('distinct substitution cases have a prefix relation')
    for case_id, role, text, row in cases:
        records.append({'record_id': 'planned-' + case_id, 'category': 'planned', 'case_role': role,
                        'token_ids': list(row), 'public_text': text,
                        'candidate_token_ids': list(AGREEMENT_NOUN_SUBSTITUTION_ANSWER_IDS),
                        'reference_commit': CASE_COMMIT, 'references': list(REFERENCES),
                        'provenance': 'Public noun-substitution proposal; no model forward is asserted. '
                                      f'Case SHA-256 {CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_agreement_noun_substitution_rows': 54}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('substitution input prefix count changed')
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
        raise ValueError('exposure inventory does not match its pinned digest')
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
    if (type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest
            or type(cases_per_split) is not int or cases_per_split not in (8, 16)):
        raise ValueError('expected inventory or split size differs')
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, cases_per_split)
    forbidden = _prefixes(payload['records'])
    complete_rows = {tuple(row['token_ids']) for row in payload['records']}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in discovery + heldout):
        raise ValueError('candidate sequence is in the development exclusion inventory')
    return v1.KnownSplitCheck(INVENTORY_VERSION, digest, 2 * cases_per_split, len(forbidden))


def check_exposed_discovery(discovery_token_ids, heldout_token_ids, *, expected_inventory_sha256,
                            expected_model_identity):
    payload, digest = _checked_inventory()
    if type(expected_inventory_sha256) is not str or expected_inventory_sha256 != digest:
        raise ValueError('expected inventory digest differs')
    v2._identity(expected_model_identity)
    discovery, heldout = v2._candidate_rows(discovery_token_ids, heldout_token_ids, 16)
    forbidden = _prefixes(payload['records'])
    complete_rows = {tuple(row['token_ids']) for row in payload['records']}
    if any(_has_known_relation(row, forbidden, complete_rows) for row in heldout):
        raise ValueError('held-out sequence is in the development exclusion inventory')
    return v2.ExposedDiscoveryCheck(INVENTORY_VERSION, digest, 32, len(forbidden),
                                    sum(_has_known_relation(row, forbidden, complete_rows) for row in discovery))


def _validate_known_check(check):
    payload, digest = _checked_inventory()
    count = len(_prefixes(payload['records']))
    if (type(check) not in (v1.KnownSplitCheck, v2.ExposedDiscoveryCheck)
            or check.inventory_version != INVENTORY_VERSION or check.inventory_sha256 != digest
            or type(check.candidate_count) is not int or type(check.forbidden_row_count) is not int
            or check.forbidden_row_count != count or check.complete_exposure_history_certified is not False
            or check.preseal_ready is not False):
        raise ValueError('a matching known-split check is required')
    if type(check) is v1.KnownSplitCheck and check.scope == 'known-rows-only' and check.candidate_count in (16, 32):
        return count, 'known-prefixes-excluded-from-both', 0
    if (type(check) is v2.ExposedDiscoveryCheck and check.scope == 'known-prefix-holdout-only'
            and check.candidate_count == 32 and type(check.known_or_reserved_discovery_overlap_count) is int
            and 0 <= check.known_or_reserved_discovery_overlap_count <= 16):
        return count, 'known-prefix-discovery-reuse-disclosed', check.known_or_reserved_discovery_overlap_count
    raise ValueError('a matching known-split check is required')


def require_preseal_readiness(check):
    _validate_known_check(check)
    raise v1.ExposureInventoryIncomplete('known-row checking alone does not authorize a study seal')


def prepare_preseal_disclosure(check, *, protocol_sha256, acknowledge_prior_exposure_uncertainty=False):
    count, policy, overlap = _validate_known_check(check)
    if acknowledge_prior_exposure_uncertainty is not True:
        raise v1.ExposureInventoryIncomplete('prior exposure uncertainty must be explicitly disclosed')
    if type(protocol_sha256) is not str or re.fullmatch(r'[0-9a-f]{64}', protocol_sha256) is None:
        raise ValueError('a canonical protocol SHA-256 is required')
    return v2.PresealDisclosure(INVENTORY_VERSION, check.inventory_sha256, protocol_sha256,
                                check.candidate_count // 2, count, policy, overlap)
