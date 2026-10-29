"""Reserve 16 public known-injection inputs; no forward is asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v33 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v34'
PREDECESSOR_SHA256 = 'e586bae02145a62a7163eb030485fa85a33f9e94cd0949a9f88f3f74c7de194b'
INVENTORY_SHA256 = 'e5ae202c1e3449a67e92788a83cd0dc62d404a3fa1e052716a11770ee92cc00c'
FORBIDDEN_PREFIX_COUNT = 6016
CASE_COMMIT = 'd8e3705cee195730dd88d4ea7c32d579b215a9e7'
CASE_PATH = 'experiments/public_known_injection_cases_v1.json'
CASE_SHA256 = 'f1e1043fc302e701b1224248122efc7f79938665ce12cd48ebb7dd713b353789'
REFERENCES = ('docs/PUBLIC_KNOWN_INJECTION_LOCALIZATION_V1.md', CASE_PATH)

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

KNOWN_INJECTION_CASES = (('known-injection-01',
  'discovery',
  'The silver compass beside the harbor',
  (464, 8465, 31855, 13970, 262, 25451)),
 ('known-injection-02',
  'discovery',
  'The quiet kettle beneath the walnut tree',
  (464, 5897, 40231, 11061, 262, 6514, 14930, 5509)),
 ('known-injection-03',
  'discovery',
  'The linen satchel near the stone bridge',
  (464, 41822, 3332, 29232, 1474, 262, 7815, 7696)),
 ('known-injection-04',
  'discovery',
  'The yellow lantern beside the mill',
  (464, 7872, 41207, 13970, 262, 3939)),
 ('known-injection-05',
  'discovery',
  'The bronze pitcher beyond the orchard',
  (464, 22101, 18086, 3675, 262, 393, 30215)),
 ('known-injection-06',
  'discovery',
  'The faded journal under the window',
  (464, 24887, 3989, 739, 262, 4324)),
 ('known-injection-07',
  'discovery',
  'The woven basket behind the barn',
  (464, 36932, 7988, 2157, 262, 25203)),
 ('known-injection-08',
  'discovery',
  'The small sundial above the meadow',
  (464, 1402, 37437, 498, 2029, 262, 502, 4584)),
 ('known-injection-09',
  'held-out',
  'The copper anchor near the lighthouse',
  (464, 15317, 18021, 1474, 262, 46371)),
 ('known-injection-10',
  'held-out',
  'The painted violin inside the cabin',
  (464, 13055, 38283, 2641, 262, 9351)),
 ('known-injection-11',
  'held-out',
  'The narrow ladder beside the garden',
  (464, 7135, 18002, 13970, 262, 11376)),
 ('known-injection-12',
  'held-out',
  'The paper envelope beneath the table',
  (464, 3348, 22878, 11061, 262, 3084)),
 ('known-injection-13',
  'held-out',
  'The marble statue beyond the fountain',
  (464, 30623, 15207, 3675, 262, 29420)),
 ('known-injection-14',
  'held-out',
  'The blue teapot behind the curtain',
  (464, 4171, 573, 499, 313, 2157, 262, 29461)),
 ('known-injection-15',
  'held-out',
  'The glass bottle above the hearth',
  (464, 5405, 9294, 2029, 262, 3285, 400)),
 ('known-injection-16',
  'held-out',
  'The scarlet flag outside the station',
  (464, 10153, 1616, 6056, 2354, 262, 4429)))
KNOWN_INJECTION_ROWS = tuple(row for _, _, _, row in KNOWN_INJECTION_CASES)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    old_ids = [record['record_id'] for record in records]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError('predecessor record identities repeat')
    cases = KNOWN_INJECTION_CASES
    if type(cases) is not tuple or len(cases) != 16:
        raise ValueError('expected 16 fixed known-injection cases')
    ids = [case[0] for case in cases]
    rows = [case[3] for case in cases]
    if (ids != [f'known-injection-{i:02d}' for i in range(1, 17)] or len(set(rows)) != 16
            or any('planned-' + cid in old_ids for cid in ids)
            or any(split != ('discovery' if i < 8 else 'held-out') for i, (_, split, _, _) in enumerate(cases))
            or any(type(text) is not str or not text or len(text) > 256 for _, _, text, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 32
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)):
        raise ValueError('fixed case identity, split or row differs')
    old_rows = [tuple(record['token_ids']) for record in records]
    old_rows.extend(tuple(record['token_ids'] + [record['target_token_id']]) for record in records
                    if type(record.get('target_token_id')) is int)
    old_rows.extend(tuple(record['token_ids'] + [token]) for record in records
                    for token in record.get('candidate_token_ids', ()))
    old_rows.extend(tuple(record['token_ids'] + [token]) for record in records
                    for group in record.get('candidate_token_groups', {}).values() for token in group)
    proposed = [sequence for row in rows for sequence in (row, row + (318,), row + (389,))]
    if any(_related(row, old) for row in proposed for old in old_rows):
        raise ValueError('known-injection proposal has a foreign full-row prefix relation')
    if any(_related(left, right) for i, left in enumerate(rows) for right in rows[i+1:]):
        raise ValueError('known-injection inputs contain a full-row prefix relation')
    for case_id, split, text, row in cases:
        records.append({'record_id': 'planned-' + case_id, 'category': 'planned', 'case_role': split,
                        'token_ids': list(row), 'public_text': text, 'candidate_token_ids': [318, 389],
                        'reference_commit': CASE_COMMIT, 'references': list(REFERENCES),
                        'provenance': 'Public known-injection localization proposal; no model forward is asserted. '
                                      f'Case SHA-256 {CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_known_injection_rows': 16}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('known-injection input prefix count changed')
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
