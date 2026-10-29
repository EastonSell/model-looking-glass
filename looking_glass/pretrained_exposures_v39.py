"""Reserve 16 public agreement-head evaluation inputs; no forward is asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v38 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v39'
PREDECESSOR_SHA256 = 'f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9'
INVENTORY_SHA256 = '5e593d1b5a0ac35aba2791cd5eafe7114f7e93ba4bb1c90706e385d2fab0cadc'
FORBIDDEN_PREFIX_COUNT = 6807
CASE_COMMIT = '4d8b93ad4bf601cb3f01f0a9c53aeb01fc43adfe'
CASE_PATH = 'experiments/distilgpt2_agreement_heads_v2_eval_cases_v1.json'
CASE_SHA256 = '51d14d87f7a318ff94e613d3ff88dfd338a6906675a92af41899b3a027eead7b'
REFERENCES = ('docs/DISTILGPT2_AGREEMENT_HEADS_V2.md', CASE_PATH)
ANSWER_IDS = (318, 389, 373, 547)
AGREEMENT_HEAD_CASES = (('01',
  'During the storm, the pilot beside the fences',
  (7191, 262, 6388, 11, 262, 8022, 13970, 262, 35771)),
 ('02',
  'During the storm, the pilots beside the fences',
  (7191, 262, 6388, 11, 262, 14982, 13970, 262, 35771)),
 ('03',
  'During the storm, the guard beside the fences',
  (7191, 262, 6388, 11, 262, 4860, 13970, 262, 35771)),
 ('04',
  'During the storm, the guards beside the fences',
  (7191, 262, 6388, 11, 262, 10942, 13970, 262, 35771)),
 ('05',
  'After the concert, the nurse near the tower',
  (3260, 262, 10010, 11, 262, 15849, 1474, 262, 10580)),
 ('06',
  'After the concert, the nurses near the tower',
  (3260, 262, 10010, 11, 262, 20669, 1474, 262, 10580)),
 ('07',
  'After the concert, the clerk near the tower',
  (3260, 262, 10010, 11, 262, 21120, 1474, 262, 10580)),
 ('08',
  'After the concert, the clerks near the tower',
  (3260, 262, 10010, 11, 262, 44115, 1474, 262, 10580)),
 ('09',
  'Before the meeting, the pilot by the gardens',
  (8421, 262, 3249, 11, 262, 8022, 416, 262, 25476)),
 ('10',
  'Before the meeting, the pilots by the gardens',
  (8421, 262, 3249, 11, 262, 14982, 416, 262, 25476)),
 ('11',
  'Before the meeting, the nurse by the gardens',
  (8421, 262, 3249, 11, 262, 15849, 416, 262, 25476)),
 ('12',
  'Before the meeting, the nurses by the gardens',
  (8421, 262, 3249, 11, 262, 20669, 416, 262, 25476)),
 ('13',
  'Throughout the afternoon, the guard behind the barn',
  (26797, 262, 6672, 11, 262, 4860, 2157, 262, 25203)),
 ('14',
  'Throughout the afternoon, the guards behind the barn',
  (26797, 262, 6672, 11, 262, 10942, 2157, 262, 25203)),
 ('15',
  'Throughout the afternoon, the clerk behind the barn',
  (26797, 262, 6672, 11, 262, 21120, 2157, 262, 25203)),
 ('16',
  'Throughout the afternoon, the clerks behind the barn',
  (26797, 262, 6672, 11, 262, 44115, 2157, 262, 25203)))

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
    cases = AGREEMENT_HEAD_CASES
    if type(cases) is not tuple or len(cases) != 16 or len(set(old_ids)) != len(old_ids):
        raise ValueError('invalid fixed cases or predecessor identities')
    ids = [item[0] for item in cases]
    rows = [item[2] for item in cases]
    if (len(set(ids)) != 16 or len(set(rows)) != 16
            or any('planned-agreement-head-v2-' + item in old_ids for item in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 32
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)):
        raise ValueError('invalid agreement-head case identity or token row')
    old = []
    for record in records:
        base = tuple(record['token_ids'])
        bases = [base]
        if 'attention_mask' in record:
            mask = record['attention_mask']
            if len(mask) != len(base) or any(type(value) is not int or value not in (0, 1) for value in mask):
                raise ValueError('invalid predecessor attention mask')
            active = tuple(value for value, present in zip(base, mask) if present)
            if active and active != base:
                bases.append(active)
        choices = []
        if type(record.get('target_token_id')) is int:
            choices.append(record['target_token_id'])
        choices.extend(record.get('candidate_token_ids', ()))
        for group in record.get('candidate_token_groups', {}).values():
            choices.extend(group)
        if any(type(token) is not int or not 0 <= token < 50257 for token in choices):
            raise ValueError('invalid predecessor continuation')
        for base in bases:
            old.append(base)
            old.extend(base + (token,) for token in choices)
        if 'scored_prefix_token_ids' in record:
            prefix = tuple(record['scored_prefix_token_ids'])
            scored = record.get('scored_candidate_token_ids', ())
            if any(type(token) is not int or not 0 <= token < 50257 for token in scored):
                raise ValueError('invalid predecessor scored continuation')
            old.append(prefix)
            old.extend(prefix + (token,) for token in scored)
    groups = [[row] + [row + (token,) for token in ANSWER_IDS] for row in rows]
    if any(_related(row, previous) for group in groups for row in group for previous in old):
        raise ValueError('agreement-head input or continuation has a known prefix relation')
    if any(_related(left, right) for i, group in enumerate(groups) for other in groups[i+1:]
           for left in group for right in other):
        raise ValueError('distinct agreement-head cases have a prefix relation')
    for case_id, text, row in cases:
        records.append({'record_id': 'planned-agreement-head-v2-' + case_id, 'category': 'planned',
                        'case_role': 'public-evaluation', 'token_ids': list(row), 'public_text': text,
                        'candidate_token_ids': list(ANSWER_IDS), 'reference_commit': CASE_COMMIT,
                        'references': list(REFERENCES),
                        'provenance': 'Public agreement-head v2 evaluation proposal; no model forward is asserted. '
                                      f'Case SHA-256 {CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_agreement_head_v2_rows': 16}
    if FORBIDDEN_PREFIX_COUNT and len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('agreement-head input prefix count changed')
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
