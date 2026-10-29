"""Reserve 32 public output-dose copying inputs; no forward is asserted."""
import copy
import hashlib
import re

from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v36 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v37'
PREDECESSOR_SHA256 = '6f05f07a0fe6ed61ca00978ef4b225e43c9c71a4a8be05df33e8c7ed778d06b7'
INVENTORY_SHA256 = 'c84d0b83ede1f526215898ce0c2f4f5bdd9c78c8cfeb3f2aced75faaa33fdf6b'
FORBIDDEN_PREFIX_COUNT = 6570
CASE_COMMIT = 'e552222bf0a1b85985210cd4fd63df02c25a716d'
CASE_PATH = 'experiments/distilgpt2_copy_output_dose_cases_v1.json'
CASE_SHA256 = 'e45fe8cfe8b28a4f4308a4d3b2206f35cebe8154c7b4301ac1356c28258b135e'
REFERENCES = ('docs/DISTILGPT2_COPY_OUTPUT_DOSE_V1.md', CASE_PATH)
_canonical = prior._canonical
_prefixes = prior._prefixes
COPY_OUTPUT_DOSE_CASES = (('copy-output-dose-01-0',
  ' stream lake pond plain bay jungle stream lake pond plain bay',
  (4269, 13546, 16723, 8631, 15489, 20712, 4269, 13546, 16723, 8631, 15489),
  20712),
 ('copy-output-dose-01-1',
  ' bay pond stream plain lake jungle stream lake pond plain bay',
  (15489, 16723, 4269, 8631, 13546, 20712, 4269, 13546, 16723, 8631, 15489),
  20712),
 ('copy-output-dose-01-2',
  ' lake plain bay pond stream jungle stream lake pond plain bay',
  (13546, 8631, 15489, 16723, 4269, 20712, 4269, 13546, 16723, 8631, 15489),
  20712),
 ('copy-output-dose-01-3',
  ' plain stream bay lake pond jungle stream lake pond plain bay',
  (8631, 4269, 15489, 13546, 16723, 20712, 4269, 13546, 16723, 8631, 15489),
  20712),
 ('copy-output-dose-02-0',
  ' steak chicken turkey pork beef soup steak chicken turkey pork beef',
  (26320, 9015, 26876, 18128, 12023, 17141, 26320, 9015, 26876, 18128, 12023),
  17141),
 ('copy-output-dose-02-1',
  ' beef turkey steak pork chicken soup steak chicken turkey pork beef',
  (12023, 26876, 26320, 18128, 9015, 17141, 26320, 9015, 26876, 18128, 12023),
  17141),
 ('copy-output-dose-02-2',
  ' chicken pork beef turkey steak soup steak chicken turkey pork beef',
  (9015, 18128, 12023, 26876, 26320, 17141, 26320, 9015, 26876, 18128, 12023),
  17141),
 ('copy-output-dose-02-3',
  ' pork steak beef chicken turkey soup steak chicken turkey pork beef',
  (18128, 26320, 12023, 9015, 26876, 17141, 26320, 9015, 26876, 18128, 12023),
  17141),
 ('copy-output-dose-03-0',
  ' dog rat frog snake crow crab dog rat frog snake crow',
  (3290, 4227, 21264, 17522, 37593, 32202, 3290, 4227, 21264, 17522, 37593),
  32202),
 ('copy-output-dose-03-1',
  ' crow frog dog snake rat crab dog rat frog snake crow',
  (37593, 21264, 3290, 17522, 4227, 32202, 3290, 4227, 21264, 17522, 37593),
  32202),
 ('copy-output-dose-03-2',
  ' rat snake crow frog dog crab dog rat frog snake crow',
  (4227, 17522, 37593, 21264, 3290, 32202, 3290, 4227, 21264, 17522, 37593),
  32202),
 ('copy-output-dose-03-3',
  ' snake dog crow rat frog crab dog rat frog snake crow',
  (17522, 3290, 37593, 4227, 21264, 32202, 3290, 4227, 21264, 17522, 37593),
  32202),
 ('copy-output-dose-04-0',
  ' broom wheel pipe pot pan knife broom wheel pipe pot pan',
  (47085, 7825, 12656, 1787, 3425, 9845, 47085, 7825, 12656, 1787, 3425),
  9845),
 ('copy-output-dose-04-1',
  ' pan pipe broom pot wheel knife broom wheel pipe pot pan',
  (3425, 12656, 47085, 1787, 7825, 9845, 47085, 7825, 12656, 1787, 3425),
  9845),
 ('copy-output-dose-04-2',
  ' wheel pot pan pipe broom knife broom wheel pipe pot pan',
  (7825, 1787, 3425, 12656, 47085, 9845, 47085, 7825, 12656, 1787, 3425),
  9845),
 ('copy-output-dose-04-3',
  ' pot broom pan wheel pipe knife broom wheel pipe pot pan',
  (1787, 47085, 3425, 7825, 12656, 9845, 47085, 7825, 12656, 1787, 3425),
  9845),
 ('copy-output-dose-05-0',
  ' whistle dancer concert stage theater song whistle dancer concert stage theater',
  (16121, 38619, 10010, 3800, 13766, 3496, 16121, 38619, 10010, 3800, 13766),
  3496),
 ('copy-output-dose-05-1',
  ' theater concert whistle stage dancer song whistle dancer concert stage theater',
  (13766, 10010, 16121, 3800, 38619, 3496, 16121, 38619, 10010, 3800, 13766),
  3496),
 ('copy-output-dose-05-2',
  ' dancer stage theater concert whistle song whistle dancer concert stage theater',
  (38619, 3800, 13766, 10010, 16121, 3496, 16121, 38619, 10010, 3800, 13766),
  3496),
 ('copy-output-dose-05-3',
  ' stage whistle theater dancer concert song whistle dancer concert stage theater',
  (3800, 16121, 13766, 38619, 10010, 3496, 16121, 38619, 10010, 3800, 13766),
  3496),
 ('copy-output-dose-06-0',
  ' car wagon submarine satellite metro rail car wagon submarine satellite metro',
  (1097, 31860, 24927, 11210, 24536, 6787, 1097, 31860, 24927, 11210, 24536),
  6787),
 ('copy-output-dose-06-1',
  ' metro submarine car satellite wagon rail car wagon submarine satellite metro',
  (24536, 24927, 1097, 11210, 31860, 6787, 1097, 31860, 24927, 11210, 24536),
  6787),
 ('copy-output-dose-06-2',
  ' wagon satellite metro submarine car rail car wagon submarine satellite metro',
  (31860, 11210, 24536, 24927, 1097, 6787, 1097, 31860, 24927, 11210, 24536),
  6787),
 ('copy-output-dose-06-3',
  ' satellite car metro wagon submarine rail car wagon submarine satellite metro',
  (11210, 1097, 24536, 31860, 24927, 6787, 1097, 31860, 24927, 11210, 24536),
  6787),
 ('copy-output-dose-07-0',
  ' sweater dress skirt socks shoes boots sweater dress skirt socks shoes',
  (35685, 6576, 23967, 24359, 10012, 14412, 35685, 6576, 23967, 24359, 10012),
  14412),
 ('copy-output-dose-07-1',
  ' shoes skirt sweater socks dress boots sweater dress skirt socks shoes',
  (10012, 23967, 35685, 24359, 6576, 14412, 35685, 6576, 23967, 24359, 10012),
  14412),
 ('copy-output-dose-07-2',
  ' dress socks shoes skirt sweater boots sweater dress skirt socks shoes',
  (6576, 24359, 10012, 23967, 35685, 14412, 35685, 6576, 23967, 24359, 10012),
  14412),
 ('copy-output-dose-07-3',
  ' socks sweater shoes dress skirt boots sweater dress skirt socks shoes',
  (24359, 35685, 10012, 6576, 23967, 14412, 35685, 6576, 23967, 24359, 10012),
  14412),
 ('copy-output-dose-08-0',
  ' bed shelf soap bowl box bench bed shelf soap bowl box',
  (3996, 18316, 19533, 9396, 3091, 7624, 3996, 18316, 19533, 9396, 3091),
  7624),
 ('copy-output-dose-08-1',
  ' box soap bed bowl shelf bench bed shelf soap bowl box',
  (3091, 19533, 3996, 9396, 18316, 7624, 3996, 18316, 19533, 9396, 3091),
  7624),
 ('copy-output-dose-08-2',
  ' shelf bowl box soap bed bench bed shelf soap bowl box',
  (18316, 9396, 3091, 19533, 3996, 7624, 3996, 18316, 19533, 9396, 3091),
  7624),
 ('copy-output-dose-08-3',
  ' bowl bed box shelf soap bench bed shelf soap bowl box',
  (9396, 3996, 3091, 18316, 19533, 7624, 3996, 18316, 19533, 9396, 3091),
  7624))
COPY_OUTPUT_DOSE_ROWS = tuple(row for _, _, row, _ in COPY_OUTPUT_DOSE_CASES)


def __getattr__(name):
    return getattr(prior, name)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    identities = [record['record_id'] for record in records]
    cases = COPY_OUTPUT_DOSE_CASES
    ids = [item[0] for item in cases]
    rows = [item[2] for item in cases]
    if (len(cases) != 32 or len(set(identities)) != len(identities)
            or len(set(ids)) != 32 or len(set(rows)) != 32
            or any('planned-' + item in identities for item in ids)
            or any(type(text) is not str or not text or len(text) > 256 for _, text, _, _ in cases)
            or any(type(row) is not tuple or len(row) != 11
                   or any(type(token) is not int or not 0 <= token < 50257 for token in row)
                   or type(target) is not int or not 0 <= target < 50257
                   for _, _, row, target in cases)):
        raise ValueError('invalid output-dose cases or predecessor identities')
    old = [tuple(record['token_ids']) for record in records]
    old.extend(tuple(record['token_ids'] + [record['target_token_id']]) for record in records
               if type(record.get('target_token_id')) is int)
    old.extend(tuple(record['token_ids'] + [token]) for record in records
               for token in record.get('candidate_token_ids', ()))
    old.extend(tuple(record['token_ids'] + [token]) for record in records
               for group in record.get('candidate_token_groups', {}).values() for token in group)
    proposed = [(row, row + (target,)) for _, _, row, target in cases]
    if any(_related(row, previous) for group in proposed for row in group for previous in old):
        raise ValueError('output-dose input or continuation has a known prefix relation')
    if any(_related(left, right) for i, group in enumerate(proposed) for other in proposed[i+1:]
           for left in group for right in other):
        raise ValueError('distinct output-dose cases have a prefix relation')
    for ident, text, row, target in cases:
        records.append({'record_id': 'planned-' + ident, 'category': 'planned', 'token_ids': list(row),
                        'public_text': text, 'target_token_id': target, 'reference_commit': CASE_COMMIT,
                        'references': list(REFERENCES),
                        'provenance': 'Public output-dose copying proposal; no model forward is asserted. '
                                      f'Case SHA-256 {CASE_SHA256}.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_copy_output_dose_rows': 32}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('output-dose input prefix count changed')
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
