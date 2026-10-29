"""Private, procedural bridge from committed number cases to the natural runner.

This is not an access-control, publication-time, model-identity or tokenizer
attestation mechanism. The trusted custodian owns those checks and must keep
the record, nonces, diagnostics, tracebacks and journal private until reveal.
"""

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math

from . import pretrained_number_commitments as commitments
from .agreement_patch_runner import (AgreementCase, NaturalAgreementRunner, StudyFailure,
                                     _CANDIDATES, _discovery_selection)


_PREDICTION_SCHEMA = 'mlg-number-natural-predictions-v2'
_MAX_PREDICTION_BYTES = 8 * 1024 * 1024
_SITES = tuple(site['id'] for site in _CANDIDATES)
_CELLS = ('same_lemma_same_number', 'same_lemma_opposite_number',
          'different_lemma_same_number', 'different_lemma_opposite_number')


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise StudyFailure('prediction record is not finite JSON') from error


def _finite(value):
    # Site contrasts arise from finite float32 logits and bounded sums; the
    # generous range prevents overflow in mean/RMSE arithmetic on bad files.
    return type(value) is float and math.isfinite(value) and abs(value) <= 32 * 3.4028234663852886e38


def _discovery_events(events, kind, contrast):
    """Validate the frozen per-site summaries used to reconstruct selection.

    The caller must independently audit all forward/pair evidence; this bounded
    structural and arithmetic check does not authenticate that execution.
    """
    if type(events) is not list or len(events) != len(_SITES):
        raise StudyFailure('prediction record has an incomplete site event set')
    summaries, family_names, reference_cases, reference_relations = [], None, None, None
    for site, event in zip(_SITES, events):
        if (type(event) is not dict or set(event) != {'kind', 'site', 'contrast', 'metrics'}
                or event['kind'] != kind or event['site'] != site
                or _canonical(event['contrast']) != _canonical(contrast)):
            raise StudyFailure('prediction site event does not match its frozen schema')
        metrics = event['metrics']
        if type(metrics) is not dict or set(metrics) != {'baseline', 'site_summary', 'quartets', 'pairs'}:
            raise StudyFailure('prediction site metrics have invalid fields')
        summary, quartets, pairs = metrics['site_summary'], metrics['quartets'], metrics['pairs']
        if (type(summary) is not dict or set(summary) != {'site', 'n_quartets',
                'mean_number_main_contrast', 'mean_lemma_number_interaction'}
                or summary['site'] != site or type(summary['n_quartets']) is not int
                or summary['n_quartets'] != 4
                or not _finite(summary['mean_number_main_contrast'])
                or not _finite(summary['mean_lemma_number_interaction'])
                or type(quartets) is not list or len(quartets) != 4
                or type(pairs) is not list or len(pairs) != 64):
            raise StudyFailure('prediction summary or factorial is incomplete')
        names = []
        for row in quartets:
            if (type(row) is not dict or set(row) != {'site', 'quartet', 'cell_means', 'cell_counts',
                                                       'number_main_contrast', 'lemma_number_interaction'}
                    or row.get('site') != site
                    or type(row.get('quartet')) is not str or not row['quartet']
                    or len(row['quartet']) > 128
                    or type(row['cell_means']) is not dict or set(row['cell_means']) != set(_CELLS)
                    or any(not _finite(row['cell_means'][cell]) for cell in _CELLS)
                    or type(row['cell_counts']) is not dict or set(row['cell_counts']) != set(_CELLS)
                    or any(type(row['cell_counts'][cell]) is not int or row['cell_counts'][cell] != 4
                           for cell in _CELLS)
                    or not _finite(row.get('number_main_contrast'))
                    or not _finite(row.get('lemma_number_interaction'))):
                raise StudyFailure('prediction family summary is invalid')
            means = row['cell_means']
            ss, so, ds, do = (means[cell] for cell in _CELLS)
            if (row['number_main_contrast'] != math.fsum((so, -ss, do, -ds)) / 2
                    or row['lemma_number_interaction'] != math.fsum((so, -ss, -do, ds))):
                raise StudyFailure('prediction contrast does not match its cell means')
            names.append(row['quartet'])
        if len(set(names)) != 4 or (family_names is not None and set(names) != family_names):
            raise StudyFailure('prediction sites do not use the same four families')
        family_names = set(names)
        for field, mean in (('number_main_contrast', 'mean_number_main_contrast'),
                            ('lemma_number_interaction', 'mean_lemma_number_interaction')):
            if math.fsum(row[field] for row in quartets) / 4 != summary[mean]:
                raise StudyFailure('prediction site mean does not match its families')
        expected_fields = {'site', 'quartet', 'target', 'donor', 'cell', 'weight',
                           'clean_margin', 'patched_margin', 'signed_target_shift',
                           'clean_margin_sign', 'patched_margin_sign', 'strict_sign_reversal',
                           'zero_transition'}
        if kind == 'present_site_metrics':
            expected_fields |= {'clean_is', 'clean_are', 'patched_is', 'patched_are'}
        else:
            expected_fields |= {'clean_was', 'clean_were', 'patched_was', 'patched_were'}
        seen, targets, cells = set(), {}, {name: {cell: [] for cell in _CELLS} for name in family_names}
        logit_keys = (('clean_is', 'clean_are', 'patched_is', 'patched_are')
                      if kind == 'present_site_metrics' else
                      ('clean_was', 'clean_were', 'patched_was', 'patched_were'))
        for row in pairs:
            if (type(row) is not dict or set(row) != expected_fields or row['site'] != site
                    or type(row['quartet']) is not str or row['quartet'] not in family_names
                    or any(type(row[key]) is not str or not row[key] or len(row[key]) > 128
                           for key in ('target', 'donor'))):
                raise StudyFailure('prediction pair metadata is invalid')
            key = (row['quartet'], row['target'], row['donor'])
            if key in seen:
                raise StudyFailure('prediction pair is duplicated')
            seen.add(key)
            if (type(row['weight']) is not int or row['weight'] not in (-1, 1)
                    or row['cell'] not in _CELLS
                    or any(not _finite(row[field]) for field in (*logit_keys, 'clean_margin',
                                                                  'patched_margin', 'signed_target_shift'))
                    or any(type(row[field]) is not int or row[field] not in (-1, 0, 1)
                           for field in ('clean_margin_sign', 'patched_margin_sign'))
                    or any(type(row[field]) is not bool for field in ('strict_sign_reversal', 'zero_transition'))):
                raise StudyFailure('prediction pair values are invalid')
            ci, ca, pi, pa = (row[field] for field in logit_keys)
            clean, patched = math.fsum((ci, -ca)), math.fsum((pi, -pa))
            cs, ps = int(clean > 0) - int(clean < 0), int(patched > 0) - int(patched < 0)
            if (row['clean_margin'] != clean or row['patched_margin'] != patched
                    or row['signed_target_shift'] != row['weight'] * math.fsum((clean, -patched))
                    or row['clean_margin_sign'] != cs or row['patched_margin_sign'] != ps
                    or row['strict_sign_reversal'] != (cs * ps == -1)
                    or row['zero_transition'] != (cs != ps and (cs == 0 or ps == 0))):
                raise StudyFailure('prediction pair arithmetic is inconsistent')
            identity = (row['quartet'], row['weight'], ci, ca)
            old = targets.setdefault(row['target'], identity)
            if old != identity:
                raise StudyFailure('prediction clean target differs across donors')
            cells[row['quartet']][row['cell']].append(row['signed_target_shift'])
        if any(sum(1 for row in pairs if row['quartet'] == name) != 16 for name in family_names):
            raise StudyFailure('prediction pair counts are incomplete')
        site_relations = {}
        for name in family_names:
            family = [row for row in pairs if row['quartet'] == name]
            members = {row['target'] for row in family}
            if (len(members) != 4 or {row['donor'] for row in family} != members
                    or sorted(targets[member][1] for member in members) != [-1, -1, 1, 1]):
                raise StudyFailure('prediction family does not contain four balanced cases')
            relations = {(row['target'], row['donor']): row['cell'] for row in family}
            site_relations.update({(name, target, donor): cell for (target, donor), cell in relations.items()})
            for target in members:
                if {donor for first, donor in relations if first == target} != members:
                    raise StudyFailure('prediction target does not have every family donor')
                same_lemma = []
                for donor in members:
                    cell = relations[(target, donor)]
                    number = 'same_number' if targets[target][1] == targets[donor][1] else 'opposite_number'
                    if not cell.endswith(number):
                        raise StudyFailure('prediction donor number cell is inconsistent')
                    if cell.startswith('same_lemma_'):
                        same_lemma.append(donor)
                    if cell.startswith('same_lemma_') != relations[(donor, target)].startswith('same_lemma_'):
                        raise StudyFailure('prediction lemma cells are not reciprocal')
                if len(same_lemma) != 2 or target not in same_lemma or len({targets[x][1] for x in same_lemma}) != 2:
                    raise StudyFailure('prediction lemma cells do not form paired numbers')
        for row in quartets:
            for cell in _CELLS:
                values = cells[row['quartet']][cell]
                if len(values) != 4 or math.fsum(values) / 4 != row['cell_means'][cell]:
                    raise StudyFailure('prediction cell mean does not match its four pairs')
        if (len(targets) != 16 or any(sum(1 for value in targets.values() if value[0] == name) != 4
                                      for name in family_names)):
            raise StudyFailure('prediction clean cases are incomplete')
        if reference_cases is not None and targets != reference_cases:
            raise StudyFailure('prediction sites do not share clean target values')
        if reference_relations is not None and site_relations != reference_relations:
            raise StudyFailure('prediction sites do not share target-donor cells')
        reference_cases = targets
        reference_relations = site_relations
        clean_margins = [weight * math.fsum((ci, -ca)) for _, weight, ci, ca in targets.values()]
        baseline = {'n': 16, 'correct': sum(value > 0 for value in clean_margins),
                    'incorrect': sum(value <= 0 for value in clean_margins),
                    'ties': sum(value == 0 for value in clean_margins)}
        if _canonical(metrics['baseline']) != _canonical(baseline):
            raise StudyFailure('prediction baseline does not match its clean cases')
        summaries.append(summary)
    return summaries, reference_cases, reference_relations


@dataclass(frozen=True)
class FrozenNaturalPredictions:
    """Owned exact bytes, not a proof of publication or execution."""
    prediction_bytes: bytes = field(repr=False)
    freeze: commitments.PredictionFreeze = field(repr=False)


@dataclass(frozen=True)
class HeldoutBinding:
    """Public commitment and checked inventory digest for journal linkage."""
    commitment_sha256: str
    inventory_sha256: str


def validate_heldout_binding(binding):
    if type(binding) is not HeldoutBinding:
        raise StudyFailure('held-out execution needs its commitment binding')
    try:
        commitments._hex(binding.commitment_sha256, 64)
        commitments._hex(binding.inventory_sha256, 64)
    except ValueError as error:
        raise StudyFailure('held-out commitment binding is invalid') from error
    return binding


def load_frozen_natural_predictions(prediction_bytes, freeze):
    """Verify the canonical published natural file against its aggregate freeze."""
    if type(prediction_bytes) is not bytes or len(prediction_bytes) > _MAX_PREDICTION_BYTES:
        raise StudyFailure('a bounded immutable prediction file is required')
    if type(freeze) is not commitments.PredictionFreeze:
        raise StudyFailure('a validated prediction aggregate is required')
    checked = commitments.load_prediction_freeze(freeze.record_bytes, freeze.sha256)
    aggregate = json.loads(checked.record_bytes)
    track = aggregate['tracks'].get('natural')
    if track is None or not hmac.compare_digest(hashlib.sha256(prediction_bytes).hexdigest(),
                                                track['prediction_sha256']):
        raise StudyFailure('natural prediction bytes do not match the aggregate')
    try:
        record = json.loads(prediction_bytes)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StudyFailure('invalid natural prediction JSON') from error
    if (type(record) is not dict or _canonical(record) != prediction_bytes
            or set(record) != {'schema', 'protocol_sha256', 'source_commit', 'checkpoint_revision',
                               'discovery_event_sha256', 'independent_audit_sha256',
                               'selection', 'present_site_metrics', 'secondary_past_site_metrics'}
            or record['schema'] != _PREDICTION_SCHEMA
            or record['protocol_sha256'] != commitments.PROTOCOL_SHA256
            or record['checkpoint_revision'] != commitments.CHECKPOINT_REVISION
            or record['source_commit'] != track['source_commit']):
        raise StudyFailure('natural prediction file identity or schema differs')
    try:
        commitments._hex(record['discovery_event_sha256'], 64)
        commitments._hex(record['independent_audit_sha256'], 64)
    except ValueError as error:
        raise StudyFailure('natural discovery evidence digests are invalid') from error
    summaries, cases, relations = _discovery_events(
        record['present_site_metrics'], 'present_site_metrics',
        {'first_token': ' is', 'first_id': 318, 'second_token': ' are', 'second_id': 389})
    _, past_cases, past_relations = _discovery_events(
        record['secondary_past_site_metrics'], 'secondary_past_site_metrics',
        {'first_token': ' was', 'first_id': 373, 'second_token': ' were', 'second_id': 547})
    if ({key: value[:2] for key, value in cases.items()} !=
            {key: value[:2] for key, value in past_cases.items()} or relations != past_relations):
        raise StudyFailure('present and past contrasts use different discovery cases')
    selection = record['selection']
    if (type(selection) is not dict or type(selection.get('magnitude_pairs')) is not list
            or len(selection['magnitude_pairs']) != 64):
        raise StudyFailure('frozen discovery selection is missing')
    families = {row['quartet'] for row in record['present_site_metrics'][0]['metrics']['quartets']}
    if any(type(row) is not dict or type(row.get('quartet')) is not str or row['quartet'] not in families
           or type(row.get('l2')) is not float or not math.isfinite(row['l2'])
           or not 0 <= row['l2'] <= 128 * 3.4028234663852886e38 for row in selection['magnitude_pairs']):
        raise StudyFailure('frozen magnitudes do not match the discovery families')
    covered = {}
    for row in selection['magnitude_pairs']:
        quartet, singular, plural = row.get('quartet'), row.get('singular'), row.get('plural')
        site = row.get('site')
        if (type(site) is not str or site not in _SITES
                or type(singular) is not str or type(plural) is not str or singular not in cases or plural not in cases
                or cases[singular][:2] != (quartet, 1) or cases[plural][:2] != (quartet, -1)
                or relations.get((quartet, singular, plural)) != 'same_lemma_opposite_number'
                or relations.get((quartet, plural, singular)) != 'same_lemma_opposite_number'):
            raise StudyFailure('frozen magnitude pair does not name discovery cases')
        covered.setdefault((site, quartet), []).extend((singular, plural))
    for site in _SITES:
        for quartet in families:
            members = covered.get((site, quartet), [])
            expected_cases = {key for key, value in cases.items() if value[0] == quartet}
            if len(members) != 4 or set(members) != expected_cases:
                raise StudyFailure('frozen magnitudes do not cover both discovery lemma pairs')
    expected = _discovery_selection({'site_summaries': summaries}, selection['magnitude_pairs'], 4)
    if _canonical(selection) != _canonical(expected):
        raise StudyFailure('frozen discovery selection does not match its summaries')
    return FrozenNaturalPredictions(prediction_bytes, checked)


def frozen_selection(frozen):
    """Revalidate owned bytes before execution and return private copied data."""
    if type(frozen) is not FrozenNaturalPredictions:
        raise StudyFailure('held-out execution needs frozen natural predictions')
    checked = load_frozen_natural_predictions(frozen.prediction_bytes, frozen.freeze)
    record = json.loads(checked.prediction_bytes)
    return record['selection'], checked


def score_heldout(metrics, selection):
    """Score each frozen site against exactly four held-out family contrasts."""
    if type(metrics) is not dict or type(metrics.get('quartets')) is not list:
        raise StudyFailure('held-out family effects are missing')
    if (type(selection) is not dict or type(selection.get('predictions')) is not dict
            or set(selection['predictions']) != set(_SITES)
            or any(not _finite(selection['predictions'][site]) for site in _SITES)
            or selection.get('selected_site') not in (None, *_SITES)
            or any(selection.get(name) not in _SITES
                   for name in ('earliest_site', 'random_site', 'magnitude_site'))):
        raise StudyFailure('frozen site predictions are invalid')
    rows, family_names = [], None
    for site in _SITES:
        family = [row for row in metrics['quartets'] if type(row) is dict and row.get('site') == site]
        if (len(family) != 4 or any(type(row.get('quartet')) is not str or not row['quartet']
                                    or not _finite(row.get('number_main_contrast')) for row in family)):
            raise StudyFailure('held-out site lacks four finite family effects')
        names = {row['quartet'] for row in family}
        if len(names) != 4 or (family_names is not None and names != family_names):
            raise StudyFailure('held-out sites do not use the same four families')
        family_names = names
        prediction = selection['predictions'][site]
        effects = [row['number_main_contrast'] for row in family]
        errors = [math.fsum((effect, -prediction)) for effect in effects]
        def error_summary(values):
            return {'mae': math.fsum(abs(value) for value in values) / 4,
                    'rmse': math.sqrt(math.fsum(value * value for value in values) / 4)}
        rows.append({'site': site, 'frozen_prediction': prediction,
                     'families': [{'quartet': row['quartet'], 'effect': row['number_main_contrast'],
                                   'prediction_error': error}
                                  for row, error in zip(family, errors)],
                     'mean_effect': math.fsum(effects) / 4,
                     'frozen_predictor': error_summary(errors), 'zero_predictor': error_summary(effects)})
    if len(metrics['quartets']) != 4 * len(_SITES):
        raise StudyFailure('held-out family effects contain unexpected rows')
    return {'selected_site': selection['selected_site'], 'earliest_site': selection['earliest_site'],
            'random_site': selection['random_site'], 'magnitude_site': selection['magnitude_site'],
            'sites': rows}


def run_committed_heldout(runner, *, freeze, prediction_bytes, split_nonce, record_bytes,
                          heldout_commitment, inventory_snapshot, current_inventory_sha256,
                          site_nonce=None, site_record_bytes=None, site_commitment=None):
    """Validate private commitments and run the exact committed family rows.

    The caller must ensure the source/runtime/model, tokenizer, publication
    order and current inventory are independently established before calling.
    """
    if type(runner) is not NaturalAgreementRunner or runner.expected_quartets != 4:
        raise StudyFailure('held-out execution requires the four-family natural runner')
    frozen = load_frozen_natural_predictions(prediction_bytes, freeze)
    commitments._inventory_prefixes(inventory_snapshot, current_inventory_sha256)
    commitments.verify_heldout_reveal(freeze, split_nonce, record_bytes, heldout_commitment,
                                      inventory_snapshot=inventory_snapshot, site_nonce=site_nonce,
                                      site_record_bytes=site_record_bytes, site_commitment=site_commitment)
    record = json.loads(record_bytes)
    cases = []
    for family in record['families']:
        index = family['family_index']
        for item_index, item in enumerate(family['cases']):
            cases.append(AgreementCase(case_id=f'heldout-{index}-{item_index}', quartet=f'heldout-{index}',
                                       lemma=f'lemma-{item_index // 2}',
                                       number=('singular' if item_index % 2 == 0 else 'plural'),
                                       input_ids=tuple(item['input_ids']), subject_index=item['subject_index']))
    binding = HeldoutBinding(heldout_commitment, inventory_snapshot.sha256)
    return runner.run(tuple(cases), mode='heldout', frozen_predictions=frozen, heldout_binding=binding)
