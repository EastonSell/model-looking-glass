"""Reserve fixed short-service science rows and masks; no forward is asserted."""
import copy
import hashlib
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v32 as prior

INVENTORY_VERSION = "distilgpt2-development-exposures-v33"
PREDECESSOR_SHA256 = "4a4ed0127d18a7c5a8f6472af619b82895cb4efa6c07314a48a463510b1878b2"
INVENTORY_SHA256 = "e586bae02145a62a7163eb030485fa85a33f9e94cd0949a9f88f3f74c7de194b"
FORBIDDEN_PREFIX_COUNT = 5926
CASE_COMMIT = "b7cfe08107a4b56f5c19ad3122c28578799d0a4a"
CASE_PATH = "experiments/short_cpu_service_reference_inputs_v1.json"
CASE_SHA256 = "47077a930963da6785d2208a21c782b045191068a835579cf295c1e990fb7f2c"
SUPPORT_SHA256 = "18b5b8f37406dd970fbad6ca18bd5b5a07545954b2c729b8bf7192a7ba2e58d4"

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
              'SYNTHETIC_SERVICE_CASES', 'TWO_ATTRACTOR_PATCH_CASES', 'TWO_ATTRACTOR_PATCH_ROWS'):
    globals()[_name] = getattr(prior, _name)
_canonical = prior._canonical
_prefixes = prior._prefixes

_GAS = (1890, 281, 7306, 3623, 11, 262, 3833, 5732, 618, 5951, 5732, 379, 6937, 6115, 780)
_LIGAND = (818, 257, 26106, 392, 12765, 6306, 11, 262, 6737, 16736, 355, 10368,
           5732, 290, 4191, 10581, 257, 38008, 26, 530, 1744, 7468, 318)
_GAS_TEXT = "For an ideal gas, the pressure increases when temperature increases at constant volume because"
_LIGAND_TEXT = "In a ligand binding experiment, the signal rises as concentration increases and eventually approaches a plateau; one possible explanation is"
SCIENCE_SERVICE_CASES = (
    ('gas-active', _GAS_TEXT, _GAS, (1,) * 15),
    ('gas-padded', _GAS_TEXT, _GAS + (50256,), (1,) * 15 + (0,)),
    ('ligand-active', _LIGAND_TEXT, _LIGAND, (1,) * 23),
    ('ligand-padded', _LIGAND_TEXT, _LIGAND + (50256,) * 9, (1,) * 23 + (0,) * 9),
)


def _related(left, right):
    return left[:len(right)] == right or right[:len(left)] == left


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    old_ids = [r['record_id'] for r in records]
    if len(set(old_ids)) != len(old_ids):
        raise ValueError('predecessor record identities repeat')
    expected = (
        ('gas-active', _GAS_TEXT, _GAS, (1,) * 15),
        ('gas-padded', _GAS_TEXT, _GAS + (50256,), (1,) * 15 + (0,)),
        ('ligand-active', _LIGAND_TEXT, _LIGAND, (1,) * 23),
        ('ligand-padded', _LIGAND_TEXT, _LIGAND + (50256,) * 9, (1,) * 23 + (0,) * 9),
    )
    cases = SCIENCE_SERVICE_CASES
    if (type(cases) is not tuple or cases != expected
            or any(type(case) is not tuple or len(case) != 4
                   or type(case[0]) is not str or type(case[1]) is not str
                   or type(case[2]) is not tuple or type(case[3]) is not tuple
                   or any(type(value) is not int for value in case[2] + case[3])
                   for case in cases)):
        raise ValueError('fixed input or mask differs from the public proposal')
    ids = ['planned-short-science-' + case[0] for case in cases]
    if len(set(ids)) != 4 or any(key in old_ids for key in ids):
        raise ValueError('reserved case identities repeat')
    rows = [case[2] for case in cases]
    old_rows = [tuple(r['token_ids']) for r in records]
    old_rows.extend(tuple(r['token_ids'] + [r['target_token_id']]) for r in records
                    if type(r.get('target_token_id')) is int)
    old_rows.extend(tuple(r['token_ids'] + [token]) for r in records
                    for token in r.get('candidate_token_ids', ()))
    old_rows.extend(tuple(r['token_ids'] + [token]) for r in records
                    for group in r.get('candidate_token_groups', {}).values() for token in group)
    if any(_related(row, old) for row in rows for old in old_rows):
        raise ValueError('science input has a known full-row prefix relation')
    relationships = {(i, j) for i in range(4) for j in range(i + 1, 4) if _related(rows[i], rows[j])}
    if relationships != {(0, 1), (2, 3)}:
        raise ValueError('only the fixed active-to-padded pairs may share a prefix')
    for case_id, text, row, mask in cases:
        records.append({'record_id': 'planned-short-science-' + case_id, 'category': 'planned',
                        'token_ids': list(row), 'attention_mask': list(mask), 'public_text': text,
                        'reference_commit': CASE_COMMIT,
                        'references': [CASE_PATH, 'docs/SHORT_SERVICE_SCIENCE_PUBLIC_SUPPORT_V1.json'],
                        'provenance': 'Public short-service reference proposal; no model forward asserted. '
                                      f'Case SHA-256 {CASE_SHA256}; point-in-time support SHA-256 {SUPPORT_SHA256}. '
                                      'Active and padded versions intentionally share the same prefix.'})
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'planned_short_science_rows': 4}
    if len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('science input prefix count changed')
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
