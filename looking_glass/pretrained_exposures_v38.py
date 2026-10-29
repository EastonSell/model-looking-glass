"""Reserve 16 full inputs and 32 scored-prefix comparison alternatives."""
import copy
import hashlib
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v37 as prior

for _name in dir(prior):
    if _name.isupper() and not _name.startswith('_'):
        globals()[_name] = getattr(prior, _name)
INVENTORY_VERSION = 'distilgpt2-development-exposures-v38'
PREDECESSOR_SHA256 = 'c84d0b83ede1f526215898ce0c2f4f5bdd9c78c8cfeb3f2aced75faaa33fdf6b'
INVENTORY_SHA256 = 'f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9'
FORBIDDEN_PREFIX_COUNT = 6725
CASE_COMMIT = '33f61a555af559fa396a35a7357e928029926875'
CASE_PATH = 'experiments/public_distractor_repair_cases_v1.json'
CASE_SHA256 = '811a1f0251ff21ddaa1869ee96c80dfa47beb87a6b6c2f1e3b6505728fee1825'
REFERENCES = ('docs/PUBLIC_DISTRACTOR_DOWNSTREAM_REPAIR_V1.md', CASE_PATH,
              'experiments/public_distractor_repair_v1/public-comparison-v37.json')
_canonical = prior._canonical
_prefixes = prior._prefixes
DISTRACTOR_REPAIR_CASES = (('distractor-repair-01',
  'The violinist beside the mossy courtyard gate.',
  (464, 38283, 396, 13970, 262, 42853, 88, 34941, 8946, 13),
  (464, 38283, 396, 13970, 262, 42853, 88, 34941, 8946)),
 ('distractor-repair-02',
  'The sailors beyond the narrow harbor bridge.',
  (464, 29996, 3675, 262, 7135, 25451, 7696, 13),
  (464, 29996, 3675, 262, 7135, 25451, 7696)),
 ('distractor-repair-03',
  'A lantern hanging above the library staircase.',
  (32, 41207, 10938, 2029, 262, 5888, 27656, 13),
  (32, 41207, 10938, 2029, 262, 5888, 27656)),
 ('distractor-repair-04',
  'The farmers along the quiet mountain road.',
  (464, 9818, 1863, 262, 5897, 8598, 2975, 13),
  (464, 9818, 1863, 262, 5897, 8598, 2975)),
 ('distractor-repair-05',
  'This delicate instrument beneath the table.',
  (1212, 19217, 8875, 11061, 262, 3084, 13),
  (1212, 19217, 8875, 11061, 262, 3084)),
 ('distractor-repair-06',
  'Those young artists outside the crowded gallery.',
  (9627, 1862, 7912, 2354, 262, 18012, 15604, 13),
  (9627, 1862, 7912, 2354, 262, 18012, 15604)),
 ('distractor-repair-07',
  'The committee near the old riverside hall.',
  (464, 5583, 1474, 262, 1468, 18180, 485, 6899, 13),
  (464, 5583, 1474, 262, 1468, 18180, 485, 6899)),
 ('distractor-repair-08',
  'Several clouds behind the distant ridge.',
  (14945, 15114, 2157, 262, 12899, 32525, 13),
  (14945, 15114, 2157, 262, 12899, 32525)),
 ('distractor-repair-09',
  "A traveler beside the orchard's wooden fence.",
  (32, 40168, 13970, 262, 393, 30215, 338, 13510, 13990, 13),
  (32, 40168, 13970, 262, 393, 30215, 338, 13510, 13990)),
 ('distractor-repair-10',
  'The carpenters beyond the restored brick tower.',
  (464, 1097, 16923, 364, 3675, 262, 15032, 17214, 10580, 13),
  (464, 1097, 16923, 364, 3675, 262, 15032, 17214, 10580)),
 ('distractor-repair-11',
  'Our patient neighbor behind the blue curtain.',
  (5122, 5827, 4780, 2157, 262, 4171, 29461, 13),
  (5122, 5827, 4780, 2157, 262, 4171, 29461)),
 ('distractor-repair-12',
  'Those small boats near the eastern jetty.',
  (9627, 1402, 16659, 1474, 262, 10183, 12644, 774, 13),
  (9627, 1402, 16659, 1474, 262, 10183, 12644, 774)),
 ('distractor-repair-13',
  "The scholar beneath the garden's ancient oak.",
  (464, 15606, 11061, 262, 11376, 338, 6156, 26210, 13),
  (464, 15606, 11061, 262, 11376, 338, 6156, 26210)),
 ('distractor-repair-14',
  'Two apprentices outside the workshop door.',
  (7571, 38523, 2354, 262, 20243, 3420, 13),
  (7571, 38523, 2354, 262, 20243, 3420)),
 ('distractor-repair-15',
  'A silver vessel on the shaded terrace.',
  (32, 8465, 8837, 319, 262, 427, 5286, 8812, 558, 13),
  (32, 8465, 8837, 319, 262, 427, 5286, 8812, 558)),
 ('distractor-repair-16',
  'Many visitors near the northern meadow.',
  (7085, 9692, 1474, 262, 7840, 502, 4584, 13),
  (7085, 9692, 1474, 262, 7840, 502, 4584)))
DISTRACTOR_REPAIR_ROWS = tuple(row for _, _, row, _ in DISTRACTOR_REPAIR_CASES)
SCORED_CANDIDATES = ((318, ' is'), (389, ' are'))

def _related(a, b):
    return a[:len(b)] == b or b[:len(a)] == a

def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    old_ids = [r['record_id'] for r in records]
    cases = DISTRACTOR_REPAIR_CASES
    new_ids = []
    for ident, text, row, prefix in cases:
        new_ids.append('planned-' + ident)
        new_ids.extend('reference-' + ident + '-' + str(token) for token, _ in SCORED_CANDIDATES)
    if (len(cases) != 16 or len(set(old_ids)) != len(old_ids)
            or len(set(new_ids)) != 48 or any(i in old_ids for i in new_ids)
            or len({c[2] for c in cases}) != 16
            or any(re.fullmatch(r'distractor-repair-\d{2}', ident) is None
                   or type(text) is not str or not text.endswith('.') or len(text) > 256
                   or type(row) is not tuple or not 3 <= len(row) <= 32
                   or row != prefix + (13,)
                   or any(type(t) is not int or not 0 <= t < 50257 for t in row)
                   for ident, text, row, prefix in cases)):
        raise ValueError('invalid fixed cases or predecessor identities')
    old = []
    for record in records:
        base = tuple(record['token_ids'])
        bases = [base]
        if 'attention_mask' in record:
            mask = record['attention_mask']
            if len(mask) != len(base) or any(type(x) is not int or x not in (0, 1) for x in mask):
                raise ValueError('invalid attention mask in predecessor')
            active = tuple(v for v, m in zip(base, mask) if m)
            if active and active != base:
                bases.append(active)
        choices = []
        if type(record.get('target_token_id')) is int:
            choices.append(record['target_token_id'])
        choices.extend(record.get('candidate_token_ids', ()))
        for group in record.get('candidate_token_groups', {}).values():
            choices.extend(group)
        if any(type(t) is not int or not 0 <= t < 50257 for t in choices):
            raise ValueError('invalid predecessor continuation')
        for b in bases:
            old.append(b)
            old.extend(b + (token,) for token in choices)
    groups = [(row, prefix, prefix + (318,), prefix + (389,))
              for _, _, row, prefix in cases]
    if any(_related(x, previous) for group in groups for x in group for previous in old):
        raise ValueError('new input or scored alternative has a foreign prefix relation')
    if any(_related(a, b) for i, group in enumerate(groups) for other in groups[i+1:]
           for a in group for b in other):
        raise ValueError('distinct fixed cases share a prefix relation')
    for ident, text, row, prefix in cases:
        records.append({'record_id': 'planned-' + ident, 'category': 'planned',
            'token_ids': list(row), 'public_text': text,
            'scored_prefix_token_ids': list(prefix), 'scored_candidate_token_ids': [318, 389],
            'reference_commit': CASE_COMMIT, 'references': list(REFERENCES),
            'provenance': 'Public full-input proposal; no model forward is asserted. '
                          f'Case SHA-256 {CASE_SHA256}.'})
        for token, suffix in SCORED_CANDIDATES:
            records.append({'record_id': 'reference-' + ident + '-' + str(token),
                'category': 'reference-only', 'token_ids': list(prefix) + [token],
                'public_text': text[:-1] + suffix, 'reference_commit': CASE_COMMIT,
                'references': list(REFERENCES),
                'provenance': 'Scored-prefix comparison alternative, reserved for exclusion; '
                              'no separate model forward is asserted. '
                              f'Case SHA-256 {CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_distractor_repair_inputs': 16,
                         'reference_only_scored_prefix_alternatives': 32}
    if FORBIDDEN_PREFIX_COUNT and len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('input and scored-reference prefix count changed')
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
