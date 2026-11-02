"""Reserve 32 public cross-model agreement inputs; no forward is asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v35 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v36'
PREDECESSOR_SHA256 = '3a7ecf81eb7f04759ed5b8ffd0607c5955b634b938756e76eb52d2435aef9610'
INVENTORY_SHA256 = '6f05f07a0fe6ed61ca00978ef4b225e43c9c71a4a8be05df33e8c7ed778d06b7'
FORBIDDEN_PREFIX_COUNT = 6218
CASE_COMMIT = 'a49cec14227ac1823b67f63e2793b513210fca7c'
CASE_PATH = 'experiments/cross_model_agreement_cases_v1.json'
CASE_SHA256 = 'b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503'
REFERENCES = ('docs/CROSS_MODEL_AGREEMENT_V1.md', CASE_PATH)
ANSWER_IDS = (318, 389, 373, 547)
CROSS_MODEL_CASES = (('f0-a0-l0-n0',
  'Inside the warehouse, the basket near the chair',
  (24441, 262, 20933, 11, 262, 7988, 1474, 262, 5118)),
 ('f0-a0-l0-n1',
  'Inside the warehouse, the baskets near the chair',
  (24441, 262, 20933, 11, 262, 46530, 1474, 262, 5118)),
 ('f0-a0-l1-n0',
  'Inside the warehouse, the barrel near the chair',
  (24441, 262, 20933, 11, 262, 9036, 1474, 262, 5118)),
 ('f0-a0-l1-n1',
  'Inside the warehouse, the barrels near the chair',
  (24441, 262, 20933, 11, 262, 17907, 1474, 262, 5118)),
 ('f0-a0-l2-n0',
  'Inside the warehouse, the cup near the chair',
  (24441, 262, 20933, 11, 262, 6508, 1474, 262, 5118)),
 ('f0-a0-l2-n1',
  'Inside the warehouse, the cups near the chair',
  (24441, 262, 20933, 11, 262, 14180, 1474, 262, 5118)),
 ('f0-a0-l3-n0',
  'Inside the warehouse, the crate near the chair',
  (24441, 262, 20933, 11, 262, 27021, 1474, 262, 5118)),
 ('f0-a0-l3-n1',
  'Inside the warehouse, the crates near the chair',
  (24441, 262, 20933, 11, 262, 40382, 1474, 262, 5118)),
 ('f0-a1-l0-n0',
  'Inside the warehouse, the basket near the chairs',
  (24441, 262, 20933, 11, 262, 7988, 1474, 262, 18791)),
 ('f0-a1-l0-n1',
  'Inside the warehouse, the baskets near the chairs',
  (24441, 262, 20933, 11, 262, 46530, 1474, 262, 18791)),
 ('f0-a1-l1-n0',
  'Inside the warehouse, the barrel near the chairs',
  (24441, 262, 20933, 11, 262, 9036, 1474, 262, 18791)),
 ('f0-a1-l1-n1',
  'Inside the warehouse, the barrels near the chairs',
  (24441, 262, 20933, 11, 262, 17907, 1474, 262, 18791)),
 ('f0-a1-l2-n0',
  'Inside the warehouse, the cup near the chairs',
  (24441, 262, 20933, 11, 262, 6508, 1474, 262, 18791)),
 ('f0-a1-l2-n1',
  'Inside the warehouse, the cups near the chairs',
  (24441, 262, 20933, 11, 262, 14180, 1474, 262, 18791)),
 ('f0-a1-l3-n0',
  'Inside the warehouse, the crate near the chairs',
  (24441, 262, 20933, 11, 262, 27021, 1474, 262, 18791)),
 ('f0-a1-l3-n1',
  'Inside the warehouse, the crates near the chairs',
  (24441, 262, 20933, 11, 262, 40382, 1474, 262, 18791)),
 ('f1-a0-l0-n0',
  'At the marketplace, the basket beside the table',
  (2953, 262, 17432, 11, 262, 7988, 13970, 262, 3084)),
 ('f1-a0-l0-n1',
  'At the marketplace, the baskets beside the table',
  (2953, 262, 17432, 11, 262, 46530, 13970, 262, 3084)),
 ('f1-a0-l1-n0',
  'At the marketplace, the barrel beside the table',
  (2953, 262, 17432, 11, 262, 9036, 13970, 262, 3084)),
 ('f1-a0-l1-n1',
  'At the marketplace, the barrels beside the table',
  (2953, 262, 17432, 11, 262, 17907, 13970, 262, 3084)),
 ('f1-a0-l2-n0',
  'At the marketplace, the cup beside the table',
  (2953, 262, 17432, 11, 262, 6508, 13970, 262, 3084)),
 ('f1-a0-l2-n1',
  'At the marketplace, the cups beside the table',
  (2953, 262, 17432, 11, 262, 14180, 13970, 262, 3084)),
 ('f1-a0-l3-n0',
  'At the marketplace, the crate beside the table',
  (2953, 262, 17432, 11, 262, 27021, 13970, 262, 3084)),
 ('f1-a0-l3-n1',
  'At the marketplace, the crates beside the table',
  (2953, 262, 17432, 11, 262, 40382, 13970, 262, 3084)),
 ('f1-a1-l0-n0',
  'At the marketplace, the basket beside the tables',
  (2953, 262, 17432, 11, 262, 7988, 13970, 262, 8893)),
 ('f1-a1-l0-n1',
  'At the marketplace, the baskets beside the tables',
  (2953, 262, 17432, 11, 262, 46530, 13970, 262, 8893)),
 ('f1-a1-l1-n0',
  'At the marketplace, the barrel beside the tables',
  (2953, 262, 17432, 11, 262, 9036, 13970, 262, 8893)),
 ('f1-a1-l1-n1',
  'At the marketplace, the barrels beside the tables',
  (2953, 262, 17432, 11, 262, 17907, 13970, 262, 8893)),
 ('f1-a1-l2-n0',
  'At the marketplace, the cup beside the tables',
  (2953, 262, 17432, 11, 262, 6508, 13970, 262, 8893)),
 ('f1-a1-l2-n1',
  'At the marketplace, the cups beside the tables',
  (2953, 262, 17432, 11, 262, 14180, 13970, 262, 8893)),
 ('f1-a1-l3-n0',
  'At the marketplace, the crate beside the tables',
  (2953, 262, 17432, 11, 262, 27021, 13970, 262, 8893)),
 ('f1-a1-l3-n1',
  'At the marketplace, the crates beside the tables',
  (2953, 262, 17432, 11, 262, 40382, 13970, 262, 8893)))

for _name in dir(prior):
    if _name.isupper() and _name not in globals():
        globals()[_name] = getattr(prior, _name)
_canonical = prior._canonical
_prefixes = prior._prefixes


def _related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    old_ids = [item['record_id'] for item in records]
    cases = CROSS_MODEL_CASES
    if type(cases) is not tuple or len(cases) != 32 or len(set(old_ids)) != len(old_ids):
        raise ValueError('invalid fixed cases or predecessor identities')
    ids = [item[0] for item in cases]
    rows = [item[2] for item in cases]
    if (len(set(ids)) != 32 or len(set(rows)) != 32
            or any('planned-cross-model-' + item in old_ids for item in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 32
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)):
        raise ValueError('invalid cross-model case identity or token row')
    old = [tuple(record['token_ids']) for record in records]
    old.extend(tuple(record['token_ids'] + [record['target_token_id']]) for record in records
               if type(record.get('target_token_id')) is int)
    old.extend(tuple(record['token_ids'] + [token]) for record in records
               for token in record.get('candidate_token_ids', ()))
    old.extend(tuple(record['token_ids'] + [token]) for record in records
               for group in record.get('candidate_token_groups', {}).values() for token in group)
    groups = [[row] + [row + (token,) for token in ANSWER_IDS] for row in rows]
    if any(_related(row, previous) for group in groups for row in group for previous in old):
        raise ValueError('cross-model input or continuation has a known prefix relation')
    if any(_related(left, right) for i, group in enumerate(groups) for other in groups[i+1:]
           for left in group for right in other):
        raise ValueError('distinct cross-model cases have a prefix relation')
    for case_id, text, row in cases:
        records.append({'record_id': 'planned-cross-model-' + case_id, 'category': 'planned',
                        'case_role': 'public-evaluation', 'token_ids': list(row), 'public_text': text,
                        'candidate_token_ids': list(ANSWER_IDS), 'reference_commit': CASE_COMMIT,
                        'references': list(REFERENCES),
                        'provenance': 'Public cross-model agreement proposal; no model forward is asserted. '
                                      f'Case SHA-256 {CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_cross_model_agreement_rows': 32}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('cross-model input prefix count changed')
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
