"""Reconcile saved copying-screen and incomplete invariant-discovery exposures."""
import copy
import hashlib
import re
from . import pretrained_exposures as v1
from . import pretrained_exposures_v2 as v2
from . import pretrained_exposures_v41 as prior

INVENTORY_VERSION = 'distilgpt2-development-exposures-v42'
PREDECESSOR_SHA256 = '110ebcf674168e794544c257d158b3c3242f98bcf108375cdc073bb145e2bfad'
INVENTORY_SHA256 = 'edd409c3200f69819bb55f91d188229cc268bcbd50bc08870a3e875469a6d656'
FORBIDDEN_PREFIX_COUNT = 9011
COPY_COMMIT = 'c2b88d39e2826ec82253818d3778a59c0b4c7350'
COPY_CASE_SHA256 = '96ffdac66a540cef8ff2a64e08649184a7f6fc148b4ef44eb5eedf96b4a18734'
COPY_JOURNAL_SHA256 = 'f5a7857dec21cdc21d855ed4c07686c8b976162e41beae8256c6ab4f0877ade0'
COPY_RESULT_SHA256 = 'fcee5b994d9f8d70c0d9c35eae62da8827f5cb92008c9db709c8beb9a733aa99'
COPY_RAW_SHA256 = '7d54f4f40b9bcdf56dd6a34823ceb4da61098c0f2c6ed84c90396272bc7328aa'
COPY_ROOT = 'results/distilgpt2-copy-behavior-screen-v1-20261029/screen/'
INVARIANT_COMMIT = 'c0cfb101901db185336555f229c6b4d8782f7943'
INVARIANT_ROOT = 'results/distilgpt2-agreement-invariant-noun-v1/'
INVARIANT_REFERENCES = ('events.jsonl','terminal.json','owner-exit-observation.json',
                        'independent-prefix-review.json','independent-inventory-review.json')
INVARIANT_EVENTS_SHA256 = '0be814ca696b4da0b90c99609098e926db3d489f2010da70c0bc9fd0adc3874d'
INVARIANT_TERMINAL_SHA256 = '32326e95b19b9632ed459f06eb781206427a3961a6f6f93df1a442a9277cc26b'
INVARIANT_EXIT_SHA256 = 'e401742df188267944a5c25fb7ff9fbb419413e43be97b7b3981da0dff8ea9f4'
INVARIANT_CASE_SHA256 = '50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827'
INVARIANT_REPLAY_SHA256 = '9bf1540c6b9466a16171c0242553db7b3df3d3c38aca5a58d2e47a40552bbd35'
INVARIANT_AUDIT_SHA256 = 'd81fb0d0add26e09974e3f5c5e68cca3279e34bc54593ccc745af68ecb7b137a'
DISCOVERY_IDS = ('planned-discovery_target_f0_c2_n0_this', 'planned-discovery_target_f0_c2_n0_these', 'planned-discovery_target_f1_c0_n0_these', 'planned-discovery_target_f1_c0_n0_this', 'planned-discovery_fragment_f0_c0', 'planned-discovery_target_f1_c2_n0_these', 'planned-discovery_target_f1_c2_n0_this', 'planned-discovery_demonstrative_f1_these', 'planned-discovery_demonstrative_f1_this', 'planned-discovery_fragment_f0_c1', 'planned-discovery_target_f0_c1_n1_these', 'planned-discovery_target_f0_c1_n1_this', 'planned-discovery_target_f0_c1_n0_this', 'planned-discovery_target_f0_c1_n0_these', 'planned-discovery_fragment_f1_c1', 'planned-discovery_target_f0_c0_n1_these', 'planned-discovery_target_f0_c0_n1_this', 'planned-discovery_target_f0_c0_n0_this', 'planned-discovery_target_f0_c0_n0_these', 'planned-discovery_fragment_f1_c2', 'planned-discovery_target_f1_c0_n1_this', 'planned-discovery_target_f1_c0_n1_these', 'planned-discovery_target_f0_c2_n1_these', 'planned-discovery_target_f0_c2_n1_this', 'planned-discovery_fragment_f1_c0', 'planned-discovery_demonstrative_f0_this', 'planned-discovery_demonstrative_f0_these', 'planned-discovery_target_f1_c1_n1_this', 'planned-discovery_target_f1_c1_n1_these', 'planned-discovery_fragment_f0_c2', 'planned-discovery_target_f1_c1_n0_these', 'planned-discovery_target_f1_c1_n0_this', 'planned-discovery_target_f1_c2_n1_this', 'planned-discovery_target_f1_c2_n1_these')
EVALUATION_IDS = ('planned-evaluation_target_f1_c2_n1_these', 'planned-evaluation_target_f1_c2_n1_this', 'planned-evaluation_target_f1_c2_n0_this', 'planned-evaluation_target_f1_c2_n0_these', 'planned-evaluation_fragment_f0_c2', 'planned-evaluation_target_f0_c1_n0_these', 'planned-evaluation_target_f0_c1_n0_this', 'planned-evaluation_demonstrative_f0_these', 'planned-evaluation_demonstrative_f0_this', 'planned-evaluation_fragment_f1_c1', 'planned-evaluation_target_f1_c1_n1_these', 'planned-evaluation_target_f1_c1_n1_this', 'planned-evaluation_target_f0_c2_n1_this', 'planned-evaluation_target_f0_c2_n1_these', 'planned-evaluation_fragment_f1_c2', 'planned-evaluation_target_f0_c0_n0_these', 'planned-evaluation_target_f0_c0_n0_this', 'planned-evaluation_target_f1_c0_n1_these', 'planned-evaluation_target_f1_c0_n1_this', 'planned-evaluation_fragment_f1_c0', 'planned-evaluation_target_f1_c0_n0_this', 'planned-evaluation_target_f1_c0_n0_these', 'planned-evaluation_target_f0_c2_n0_these', 'planned-evaluation_target_f0_c2_n0_this', 'planned-evaluation_fragment_f0_c0', 'planned-evaluation_target_f1_c1_n0_this', 'planned-evaluation_target_f1_c1_n0_these', 'planned-evaluation_target_f0_c1_n1_this', 'planned-evaluation_target_f0_c1_n1_these', 'planned-evaluation_fragment_f0_c1', 'planned-evaluation_demonstrative_f1_this', 'planned-evaluation_demonstrative_f1_these', 'planned-evaluation_target_f0_c0_n1_this', 'planned-evaluation_target_f0_c0_n1_these')
COPY_IDS = tuple(f'planned-copy-behavior-{g:02d}-{v}' for g in range(1, 25) for v in range(8))
for _name in dir(prior):
    if _name.isupper() and _name not in globals():
        globals()[_name] = getattr(prior, _name)
_canonical, _prefixes = prior._canonical, prior._prefixes


def _build_payload(predecessor):
    payload = copy.deepcopy(predecessor)
    records = payload['records']
    old_ids = [r['record_id'] for r in records]
    if len(old_ids) != len(set(old_ids)):
        raise ValueError('predecessor identities repeat')
    wanted = set(DISCOVERY_IDS) | set(COPY_IDS)
    if len(wanted) != 226 or len(set(EVALUATION_IDS)) != 34 or wanted & set(EVALUATION_IDS):
        raise ValueError('fixed status-set identities differ')
    seen, seen_evaluation = set(), set()
    for record in records:
        record_id = record['record_id']
        if record_id in EVALUATION_IDS:
            if record.get('category') != 'planned':
                raise ValueError('evaluation status unexpectedly differs')
            seen_evaluation.add(record_id)
        if record_id not in wanted:
            continue
        if record_id in seen or record.get('category') != 'planned':
            raise ValueError('status predecessor differs')
        new_id = 'observed-' + record_id[len('planned-'):]
        if new_id in old_ids:
            raise ValueError('observed identity already exists')
        seen.add(record_id)
        record['record_id'] = new_id
        if record_id in DISCOVERY_IDS:
            record['category'] = 'observed-reported'
            record['reference_commit'] = INVARIANT_COMMIT
            record['references'] = list(record['references']) + [INVARIANT_ROOT + x for x in INVARIANT_REFERENCES]
            record['provenance'] = (
                'Public nonblind invariant-noun discovery input in an incomplete saved attempt; '
                'each of the 34 discovery rows appears in the durable prefix. '
                'The attempt has 301 durable forward records, 302 reported returns, and incomplete tensor accounting; '
                'the 302nd action is inferred and three orphan tensor files are unbound. '
                'No evaluation input or completed study outcome is asserted. '
                'Process and execution records are cooperative, not authenticated attestation. '
                f'Case SHA-256 {INVARIANT_CASE_SHA256}; events {INVARIANT_EVENTS_SHA256}; '
                f'terminal {INVARIANT_TERMINAL_SHA256}; exit observation {INVARIANT_EXIT_SHA256}; '
                f'prefix replay {INVARIANT_REPLAY_SHA256}; independent inventory audit {INVARIANT_AUDIT_SHA256}.')
        else:
            record['category'] = 'observed-saved'
            record['reference_commit'] = COPY_COMMIT
            record['references'] = list(record['references']) + [COPY_ROOT + x for x in
                ('run-manifest.json','attempt.jsonl.gz','result.json.gz','independent-replay.json.gz',
                 'process-observation.json','raw/index.json')]
            record['provenance'] = (
                'Public nonblind copying-behavior baseline screen saved one full-vocabulary observation per input; '
                'its recorded direct child exited 0 and was reaped. '
                'Process and execution records are cooperative, not authenticated attestation. '
                f'Case SHA-256 {COPY_CASE_SHA256}; decompressed journal {COPY_JOURNAL_SHA256}; '
                f'decompressed result {COPY_RESULT_SHA256}; raw matrix {COPY_RAW_SHA256}.')
    if seen != wanted or seen_evaluation != set(EVALUATION_IDS):
        raise ValueError('fixed status records missing')
    if _prefixes(records) != _prefixes(predecessor['records']) or len(_prefixes(records)) != FORBIDDEN_PREFIX_COUNT:
        raise ValueError('status-only transition changed input prefixes')
    payload['version'] = INVENTORY_VERSION
    payload['predecessor'] = {'version': prior.INVENTORY_VERSION, 'sha256': PREDECESSOR_SHA256}
    payload['update'] = {'reported_invariant_discovery_rows': 34, 'observed_copy_screen_rows': 192}
    return payload


def inventory_payload():
    predecessor = prior.inventory_payload()
    if (prior.INVENTORY_SHA256 != PREDECESSOR_SHA256
            or hashlib.sha256(_canonical(predecessor)).hexdigest() != PREDECESSOR_SHA256):
        raise ValueError('predecessor inventory differs')
    return _build_payload(predecessor)


def canonical_inventory_bytes():
    return _canonical(inventory_payload())


def _checked_inventory():
    payload = inventory_payload()
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError('development inventory differs')
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
