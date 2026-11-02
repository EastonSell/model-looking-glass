"""Reserve 68 public invariant-noun inputs; no forward is asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v39 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v40'
PREDECESSOR_SHA256 = '5e593d1b5a0ac35aba2791cd5eafe7114f7e93ba4bb1c90706e385d2fab0cadc'
INVENTORY_SHA256 = '685391b672c356dd627923681c4a93fc9a310c03a40fdd22e0b29af2ae9f7be3'
FORBIDDEN_PREFIX_COUNT = 6947
CASE_COMMIT = '3e998f6535dc41ec4e0bb45463164fe1b46c2611'
CASE_PATH = 'experiments/distilgpt2_agreement_invariant_noun_cases_v1.json'
CASE_SHA256 = '50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827'
REFERENCES = ('docs/AGREEMENT_INVARIANT_NOUN_V1.md', CASE_PATH)
INVARIANT_NOUN_ANSWER_IDS = (318, 389, 373, 547, 262, 257, 287, 319)
INVARIANT_NOUN_CASES = (('discovery_target_f0_c2_n0_this',
  'target',
  'Before breakfast, this sheep near the gates',
  (8421, 12607, 11, 428, 15900, 1474, 262, 17435)),
 ('discovery_target_f0_c2_n0_these',
  'target',
  'Before breakfast, these sheep near the gates',
  (8421, 12607, 11, 777, 15900, 1474, 262, 17435)),
 ('discovery_target_f1_c0_n0_these',
  'target',
  'After dinner, these sheep nearby',
  (3260, 8073, 11, 777, 15900, 6716)),
 ('discovery_target_f1_c0_n0_this',
  'target',
  'After dinner, this sheep nearby',
  (3260, 8073, 11, 428, 15900, 6716)),
 ('discovery_fragment_f0_c0', 'fragment', 'Before breakfast, nearby', (8421, 12607, 11, 6716)),
 ('discovery_target_f1_c2_n0_these',
  'target',
  'After dinner, these sheep near the lamps',
  (3260, 8073, 11, 777, 15900, 1474, 262, 32209)),
 ('discovery_target_f1_c2_n0_this',
  'target',
  'After dinner, this sheep near the lamps',
  (3260, 8073, 11, 428, 15900, 1474, 262, 32209)),
 ('discovery_demonstrative_f1_these',
  'demonstrative',
  'After dinner, these nearby',
  (3260, 8073, 11, 777, 6716)),
 ('discovery_demonstrative_f1_this',
  'demonstrative',
  'After dinner, this nearby',
  (3260, 8073, 11, 428, 6716)),
 ('discovery_fragment_f0_c1',
  'fragment',
  'Before breakfast, near the gate',
  (8421, 12607, 11, 1474, 262, 8946)),
 ('discovery_target_f0_c1_n1_these',
  'target',
  'Before breakfast, these deer near the gate',
  (8421, 12607, 11, 777, 20096, 1474, 262, 8946)),
 ('discovery_target_f0_c1_n1_this',
  'target',
  'Before breakfast, this deer near the gate',
  (8421, 12607, 11, 428, 20096, 1474, 262, 8946)),
 ('discovery_target_f0_c1_n0_this',
  'target',
  'Before breakfast, this sheep near the gate',
  (8421, 12607, 11, 428, 15900, 1474, 262, 8946)),
 ('discovery_target_f0_c1_n0_these',
  'target',
  'Before breakfast, these sheep near the gate',
  (8421, 12607, 11, 777, 15900, 1474, 262, 8946)),
 ('discovery_fragment_f1_c1', 'fragment', 'After dinner, near the lamp', (3260, 8073, 11, 1474, 262, 20450)),
 ('discovery_target_f0_c0_n1_these',
  'target',
  'Before breakfast, these deer nearby',
  (8421, 12607, 11, 777, 20096, 6716)),
 ('discovery_target_f0_c0_n1_this',
  'target',
  'Before breakfast, this deer nearby',
  (8421, 12607, 11, 428, 20096, 6716)),
 ('discovery_target_f0_c0_n0_this',
  'target',
  'Before breakfast, this sheep nearby',
  (8421, 12607, 11, 428, 15900, 6716)),
 ('discovery_target_f0_c0_n0_these',
  'target',
  'Before breakfast, these sheep nearby',
  (8421, 12607, 11, 777, 15900, 6716)),
 ('discovery_fragment_f1_c2', 'fragment', 'After dinner, near the lamps', (3260, 8073, 11, 1474, 262, 32209)),
 ('discovery_target_f1_c0_n1_this',
  'target',
  'After dinner, this deer nearby',
  (3260, 8073, 11, 428, 20096, 6716)),
 ('discovery_target_f1_c0_n1_these',
  'target',
  'After dinner, these deer nearby',
  (3260, 8073, 11, 777, 20096, 6716)),
 ('discovery_target_f0_c2_n1_these',
  'target',
  'Before breakfast, these deer near the gates',
  (8421, 12607, 11, 777, 20096, 1474, 262, 17435)),
 ('discovery_target_f0_c2_n1_this',
  'target',
  'Before breakfast, this deer near the gates',
  (8421, 12607, 11, 428, 20096, 1474, 262, 17435)),
 ('discovery_fragment_f1_c0', 'fragment', 'After dinner, nearby', (3260, 8073, 11, 6716)),
 ('discovery_demonstrative_f0_this',
  'demonstrative',
  'Before breakfast, this nearby',
  (8421, 12607, 11, 428, 6716)),
 ('discovery_demonstrative_f0_these',
  'demonstrative',
  'Before breakfast, these nearby',
  (8421, 12607, 11, 777, 6716)),
 ('discovery_target_f1_c1_n1_this',
  'target',
  'After dinner, this deer near the lamp',
  (3260, 8073, 11, 428, 20096, 1474, 262, 20450)),
 ('discovery_target_f1_c1_n1_these',
  'target',
  'After dinner, these deer near the lamp',
  (3260, 8073, 11, 777, 20096, 1474, 262, 20450)),
 ('discovery_fragment_f0_c2',
  'fragment',
  'Before breakfast, near the gates',
  (8421, 12607, 11, 1474, 262, 17435)),
 ('discovery_target_f1_c1_n0_these',
  'target',
  'After dinner, these sheep near the lamp',
  (3260, 8073, 11, 777, 15900, 1474, 262, 20450)),
 ('discovery_target_f1_c1_n0_this',
  'target',
  'After dinner, this sheep near the lamp',
  (3260, 8073, 11, 428, 15900, 1474, 262, 20450)),
 ('discovery_target_f1_c2_n1_this',
  'target',
  'After dinner, this deer near the lamps',
  (3260, 8073, 11, 428, 20096, 1474, 262, 32209)),
 ('discovery_target_f1_c2_n1_these',
  'target',
  'After dinner, these deer near the lamps',
  (3260, 8073, 11, 777, 20096, 1474, 262, 32209)),
 ('evaluation_target_f1_c2_n1_these',
  'target',
  'During autumn, these shrimp near the boxes',
  (7191, 23608, 11, 777, 30060, 1474, 262, 10559)),
 ('evaluation_target_f1_c2_n1_this',
  'target',
  'During autumn, this shrimp near the boxes',
  (7191, 23608, 11, 428, 30060, 1474, 262, 10559)),
 ('evaluation_target_f1_c2_n0_this',
  'target',
  'During autumn, this squid near the boxes',
  (7191, 23608, 11, 428, 45126, 1474, 262, 10559)),
 ('evaluation_target_f1_c2_n0_these',
  'target',
  'During autumn, these squid near the boxes',
  (7191, 23608, 11, 777, 45126, 1474, 262, 10559)),
 ('evaluation_fragment_f0_c2',
  'fragment',
  'At midnight, near the rocks',
  (2953, 15896, 11, 1474, 262, 12586)),
 ('evaluation_target_f0_c1_n0_these',
  'target',
  'At midnight, these squid near the rock',
  (2953, 15896, 11, 777, 45126, 1474, 262, 3881)),
 ('evaluation_target_f0_c1_n0_this',
  'target',
  'At midnight, this squid near the rock',
  (2953, 15896, 11, 428, 45126, 1474, 262, 3881)),
 ('evaluation_demonstrative_f0_these',
  'demonstrative',
  'At midnight, these nearby',
  (2953, 15896, 11, 777, 6716)),
 ('evaluation_demonstrative_f0_this',
  'demonstrative',
  'At midnight, this nearby',
  (2953, 15896, 11, 428, 6716)),
 ('evaluation_fragment_f1_c1', 'fragment', 'During autumn, near the box', (7191, 23608, 11, 1474, 262, 3091)),
 ('evaluation_target_f1_c1_n1_these',
  'target',
  'During autumn, these shrimp near the box',
  (7191, 23608, 11, 777, 30060, 1474, 262, 3091)),
 ('evaluation_target_f1_c1_n1_this',
  'target',
  'During autumn, this shrimp near the box',
  (7191, 23608, 11, 428, 30060, 1474, 262, 3091)),
 ('evaluation_target_f0_c2_n1_this',
  'target',
  'At midnight, this shrimp near the rocks',
  (2953, 15896, 11, 428, 30060, 1474, 262, 12586)),
 ('evaluation_target_f0_c2_n1_these',
  'target',
  'At midnight, these shrimp near the rocks',
  (2953, 15896, 11, 777, 30060, 1474, 262, 12586)),
 ('evaluation_fragment_f1_c2',
  'fragment',
  'During autumn, near the boxes',
  (7191, 23608, 11, 1474, 262, 10559)),
 ('evaluation_target_f0_c0_n0_these',
  'target',
  'At midnight, these squid nearby',
  (2953, 15896, 11, 777, 45126, 6716)),
 ('evaluation_target_f0_c0_n0_this',
  'target',
  'At midnight, this squid nearby',
  (2953, 15896, 11, 428, 45126, 6716)),
 ('evaluation_target_f1_c0_n1_these',
  'target',
  'During autumn, these shrimp nearby',
  (7191, 23608, 11, 777, 30060, 6716)),
 ('evaluation_target_f1_c0_n1_this',
  'target',
  'During autumn, this shrimp nearby',
  (7191, 23608, 11, 428, 30060, 6716)),
 ('evaluation_fragment_f1_c0', 'fragment', 'During autumn, nearby', (7191, 23608, 11, 6716)),
 ('evaluation_target_f1_c0_n0_this',
  'target',
  'During autumn, this squid nearby',
  (7191, 23608, 11, 428, 45126, 6716)),
 ('evaluation_target_f1_c0_n0_these',
  'target',
  'During autumn, these squid nearby',
  (7191, 23608, 11, 777, 45126, 6716)),
 ('evaluation_target_f0_c2_n0_these',
  'target',
  'At midnight, these squid near the rocks',
  (2953, 15896, 11, 777, 45126, 1474, 262, 12586)),
 ('evaluation_target_f0_c2_n0_this',
  'target',
  'At midnight, this squid near the rocks',
  (2953, 15896, 11, 428, 45126, 1474, 262, 12586)),
 ('evaluation_fragment_f0_c0', 'fragment', 'At midnight, nearby', (2953, 15896, 11, 6716)),
 ('evaluation_target_f1_c1_n0_this',
  'target',
  'During autumn, this squid near the box',
  (7191, 23608, 11, 428, 45126, 1474, 262, 3091)),
 ('evaluation_target_f1_c1_n0_these',
  'target',
  'During autumn, these squid near the box',
  (7191, 23608, 11, 777, 45126, 1474, 262, 3091)),
 ('evaluation_target_f0_c1_n1_this',
  'target',
  'At midnight, this shrimp near the rock',
  (2953, 15896, 11, 428, 30060, 1474, 262, 3881)),
 ('evaluation_target_f0_c1_n1_these',
  'target',
  'At midnight, these shrimp near the rock',
  (2953, 15896, 11, 777, 30060, 1474, 262, 3881)),
 ('evaluation_fragment_f0_c1', 'fragment', 'At midnight, near the rock', (2953, 15896, 11, 1474, 262, 3881)),
 ('evaluation_demonstrative_f1_this',
  'demonstrative',
  'During autumn, this nearby',
  (7191, 23608, 11, 428, 6716)),
 ('evaluation_demonstrative_f1_these',
  'demonstrative',
  'During autumn, these nearby',
  (7191, 23608, 11, 777, 6716)),
 ('evaluation_target_f0_c0_n1_this',
  'target',
  'At midnight, this shrimp nearby',
  (2953, 15896, 11, 428, 30060, 6716)),
 ('evaluation_target_f0_c0_n1_these',
  'target',
  'At midnight, these shrimp nearby',
  (2953, 15896, 11, 777, 30060, 6716)))
INVARIANT_NOUN_ROWS = tuple(row for _, _, _, row in INVARIANT_NOUN_CASES)

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
    cases = INVARIANT_NOUN_CASES
    if type(cases) is not tuple or len(cases) != 68 or len(set(old_ids)) != len(old_ids):
        raise ValueError('invalid fixed cases or predecessor identities')
    ids = [item[0] for item in cases]
    rows = [item[3] for item in cases]
    if (len(set(ids)) != 68 or len(set(rows)) != 68
            or any('planned-' + item in old_ids for item in ids)
            or sum(role == 'target' for _, role, _, _ in cases) != 48
            or sum(role == 'fragment' for _, role, _, _ in cases) != 12
            or sum(role == 'demonstrative' for _, role, _, _ in cases) != 8
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
    groups = [[row] + [row + (token,) for token in INVARIANT_NOUN_ANSWER_IDS] for row in rows]
    if any(_related(row, previous) for group in groups for row in group for previous in old):
        raise ValueError('invariant-noun case or continuation has a known prefix relation')
    if any(_related(left, right) for i, group in enumerate(groups) for other in groups[i+1:]
           for left in group for right in other):
        raise ValueError('distinct invariant-noun cases have a prefix relation')
    for identity, role, text, row in cases:
        records.append({'record_id': 'planned-' + identity, 'category': 'planned', 'case_role': role,
                        'token_ids': list(row), 'public_text': text,
                        'candidate_token_ids': list(INVARIANT_NOUN_ANSWER_IDS),
                        'reference_commit': CASE_COMMIT, 'references': list(REFERENCES),
                        'provenance': 'Public invariant-noun agreement proposal; no model forward is asserted. '
                                      f'Case SHA-256 {CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_agreement_invariant_noun_rows': 68}
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
