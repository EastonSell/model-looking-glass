"""Reserve 34 new invariant-noun successor discovery inputs; no forward is asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v42 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v43'
PREDECESSOR_SHA256 = 'edd409c3200f69819bb55f91d188229cc268bcbd50bc08870a3e875469a6d656'
INVENTORY_SHA256 = '61a80aafd9d4aea075774776d830b017f8e4c5e1a4fd0f2643963d2075a6e4ee'
FORBIDDEN_PREFIX_COUNT = 9081
SUCCESSOR_CASE_COMMIT = '2e7072cd8e4d211d9dd7d327bffd030794863d12'
SUCCESSOR_CASE_PATH = 'experiments/distilgpt2_agreement_invariant_noun_successor_cases_v2.json'
SUCCESSOR_CASE_SHA256 = '188782606abd576cde029408bb34975e7fd7ac2113bf17299dbb7e1aa2ddd76b'
SUCCESSOR_REFERENCES = ('docs/AGREEMENT_INVARIANT_NOUN_SUCCESSOR_V2.md', SUCCESSOR_CASE_PATH)
INVARIANT_SUCCESSOR_ANSWER_IDS = (318, 389, 373, 547, 262, 257, 287, 319)
INVARIANT_SUCCESSOR_CASES = (('v2_discovery_target_f0_c2_n0_this',
  'target',
  'At sunrise, this salmon near the bells',
  (2953, 40048, 11, 428, 24535, 1474, 262, 30987)),
 ('v2_discovery_target_f0_c2_n0_these',
  'target',
  'At sunrise, these salmon near the bells',
  (2953, 40048, 11, 777, 24535, 1474, 262, 30987)),
 ('v2_discovery_target_f1_c0_n0_these',
  'target',
  'Before sunset, these salmon nearby',
  (8421, 26428, 11, 777, 24535, 6716)),
 ('v2_discovery_target_f1_c0_n0_this',
  'target',
  'Before sunset, this salmon nearby',
  (8421, 26428, 11, 428, 24535, 6716)),
 ('v2_discovery_fragment_f0_c0', 'fragment', 'At sunrise, nearby', (2953, 40048, 11, 6716)),
 ('v2_discovery_target_f1_c2_n0_these',
  'target',
  'Before sunset, these salmon near the chairs',
  (8421, 26428, 11, 777, 24535, 1474, 262, 18791)),
 ('v2_discovery_target_f1_c2_n0_this',
  'target',
  'Before sunset, this salmon near the chairs',
  (8421, 26428, 11, 428, 24535, 1474, 262, 18791)),
 ('v2_discovery_demonstrative_f1_these',
  'demonstrative',
  'Before sunset, these nearby',
  (8421, 26428, 11, 777, 6716)),
 ('v2_discovery_demonstrative_f1_this',
  'demonstrative',
  'Before sunset, this nearby',
  (8421, 26428, 11, 428, 6716)),
 ('v2_discovery_fragment_f0_c1', 'fragment', 'At sunrise, near the bell', (2953, 40048, 11, 1474, 262, 8966)),
 ('v2_discovery_target_f0_c1_n1_these',
  'target',
  'At sunrise, these trout near the bell',
  (2953, 40048, 11, 777, 49411, 1474, 262, 8966)),
 ('v2_discovery_target_f0_c1_n1_this',
  'target',
  'At sunrise, this trout near the bell',
  (2953, 40048, 11, 428, 49411, 1474, 262, 8966)),
 ('v2_discovery_target_f0_c1_n0_this',
  'target',
  'At sunrise, this salmon near the bell',
  (2953, 40048, 11, 428, 24535, 1474, 262, 8966)),
 ('v2_discovery_target_f0_c1_n0_these',
  'target',
  'At sunrise, these salmon near the bell',
  (2953, 40048, 11, 777, 24535, 1474, 262, 8966)),
 ('v2_discovery_fragment_f1_c1',
  'fragment',
  'Before sunset, near the chair',
  (8421, 26428, 11, 1474, 262, 5118)),
 ('v2_discovery_target_f0_c0_n1_these',
  'target',
  'At sunrise, these trout nearby',
  (2953, 40048, 11, 777, 49411, 6716)),
 ('v2_discovery_target_f0_c0_n1_this',
  'target',
  'At sunrise, this trout nearby',
  (2953, 40048, 11, 428, 49411, 6716)),
 ('v2_discovery_target_f0_c0_n0_this',
  'target',
  'At sunrise, this salmon nearby',
  (2953, 40048, 11, 428, 24535, 6716)),
 ('v2_discovery_target_f0_c0_n0_these',
  'target',
  'At sunrise, these salmon nearby',
  (2953, 40048, 11, 777, 24535, 6716)),
 ('v2_discovery_fragment_f1_c2',
  'fragment',
  'Before sunset, near the chairs',
  (8421, 26428, 11, 1474, 262, 18791)),
 ('v2_discovery_target_f1_c0_n1_this',
  'target',
  'Before sunset, this trout nearby',
  (8421, 26428, 11, 428, 49411, 6716)),
 ('v2_discovery_target_f1_c0_n1_these',
  'target',
  'Before sunset, these trout nearby',
  (8421, 26428, 11, 777, 49411, 6716)),
 ('v2_discovery_target_f0_c2_n1_these',
  'target',
  'At sunrise, these trout near the bells',
  (2953, 40048, 11, 777, 49411, 1474, 262, 30987)),
 ('v2_discovery_target_f0_c2_n1_this',
  'target',
  'At sunrise, this trout near the bells',
  (2953, 40048, 11, 428, 49411, 1474, 262, 30987)),
 ('v2_discovery_fragment_f1_c0', 'fragment', 'Before sunset, nearby', (8421, 26428, 11, 6716)),
 ('v2_discovery_demonstrative_f0_this',
  'demonstrative',
  'At sunrise, this nearby',
  (2953, 40048, 11, 428, 6716)),
 ('v2_discovery_demonstrative_f0_these',
  'demonstrative',
  'At sunrise, these nearby',
  (2953, 40048, 11, 777, 6716)),
 ('v2_discovery_target_f1_c1_n1_this',
  'target',
  'Before sunset, this trout near the chair',
  (8421, 26428, 11, 428, 49411, 1474, 262, 5118)),
 ('v2_discovery_target_f1_c1_n1_these',
  'target',
  'Before sunset, these trout near the chair',
  (8421, 26428, 11, 777, 49411, 1474, 262, 5118)),
 ('v2_discovery_fragment_f0_c2',
  'fragment',
  'At sunrise, near the bells',
  (2953, 40048, 11, 1474, 262, 30987)),
 ('v2_discovery_target_f1_c1_n0_these',
  'target',
  'Before sunset, these salmon near the chair',
  (8421, 26428, 11, 777, 24535, 1474, 262, 5118)),
 ('v2_discovery_target_f1_c1_n0_this',
  'target',
  'Before sunset, this salmon near the chair',
  (8421, 26428, 11, 428, 24535, 1474, 262, 5118)),
 ('v2_discovery_target_f1_c2_n1_this',
  'target',
  'Before sunset, this trout near the chairs',
  (8421, 26428, 11, 428, 49411, 1474, 262, 18791)),
 ('v2_discovery_target_f1_c2_n1_these',
  'target',
  'Before sunset, these trout near the chairs',
  (8421, 26428, 11, 777, 49411, 1474, 262, 18791)))
INVARIANT_SUCCESSOR_ROWS = tuple(row for _, _, _, row in INVARIANT_SUCCESSOR_CASES)

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
    cases = INVARIANT_SUCCESSOR_CASES
    if type(cases) is not tuple or len(cases) != 34 or len(set(old_ids)) != len(old_ids):
        raise ValueError('invalid fixed cases or predecessor identities')
    ids = [item[0] for item in cases]
    rows = [item[3] for item in cases]
    if (len(set(ids)) != 34 or len(set(rows)) != 34
            or any('planned-' + item in old_ids for item in ids)
            or sum(role == 'target' for _, role, _, _ in cases) != 24
            or sum(role == 'fragment' for _, role, _, _ in cases) != 6
            or sum(role == 'demonstrative' for _, role, _, _ in cases) != 4
            or any(type(text) is not str or not text or len(text) > 256 for _, _, text, _ in cases)
            or any(type(row) is not tuple or not 1 <= len(row) <= 12
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row) for row in rows)):
        raise ValueError('invalid invariant-noun case identity or token row')
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
            scored_prefix = tuple(record['scored_prefix_token_ids'])
            scored = record.get('scored_candidate_token_ids', ())
            if any(type(token) is not int or not 0 <= token < 50257 for token in scored):
                raise ValueError('invalid predecessor scored continuation')
            old.append(scored_prefix)
            old.extend(scored_prefix + (token,) for token in scored)
    groups = [[row] + [row + (token,) for token in INVARIANT_SUCCESSOR_ANSWER_IDS] for row in rows]
    if any(_related(row, previous) for group in groups for row in group for previous in old):
        raise ValueError('invariant-noun case or continuation has a known prefix relation')
    if any(_related(left, right) for i, group in enumerate(groups) for other in groups[i+1:]
           for left in group for right in other):
        raise ValueError('distinct invariant-noun cases have a prefix relation')
    for identity, role, text, row in cases:
        records.append({'record_id': 'planned-' + identity, 'category': 'planned', 'case_role': role,
                        'token_ids': list(row), 'public_text': text,
                        'candidate_token_ids': list(INVARIANT_SUCCESSOR_ANSWER_IDS),
                        'reference_commit': SUCCESSOR_CASE_COMMIT, 'references': list(SUCCESSOR_REFERENCES),
                        'provenance': 'Public invariant-noun successor discovery proposal; no model forward is asserted. '
                                      f'Case SHA-256 {SUCCESSOR_CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_agreement_invariant_successor_discovery_rows': 34}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('invariant-noun input prefix count changed')
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
