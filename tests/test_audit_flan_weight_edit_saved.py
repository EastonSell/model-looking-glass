"""Fabricated-only, tiny saved-record tests; no model or checkpoint is loaded."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/audit_flan_weight_edit_saved.py'
spec = importlib.util.spec_from_file_location('independent_flan_saved_reader', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_score_and_distribution_use_all_vocabulary_entries():
    before = (0.0, 0.0, 0.0, 0.0)
    after = (math.log(3.0), 0.0, 0.0, 0.0)
    terms = audit.score_terms(before, 0)
    assert terms == [0.0, 0.0, math.log(4.0)]
    measurement = audit.drift(before, after)
    assert measurement['tv'] == pytest.approx(0.25, abs=1e-15)
    assert measurement['kl'] == pytest.approx(-math.log(4) - 0.25 * (math.log(0.5) + 3*math.log(1/6)), abs=1e-15)
    assert audit.drift(before, before) == {'kl': 0.0, 'tv': 0.0}


def test_control_selection_checks_each_case_metric_and_ties_by_dose():
    primary = [{'margin_gain': 1.0, 'target_gain': 0.5, 'negative': True} for _ in range(2)]
    points = [(dose, [dict(primary[0]), dict(primary[1])]) for dose in audit.DOSES]
    points[0][1][1]['target_gain'] = 0.61
    selected = audit.selection(primary, points)
    assert selected['dose'] == 0.5 and selected['matched'] is True
    assert selected['grid'][0]['matched'] is False
    primary[1]['target_gain'] = 0.49
    selected = audit.selection(primary, points)
    assert selected['primary_eligible'] is False and selected['matched'] is False


@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="historical candidate reconstruction requires Python 3.11")
def test_candidate_rows_reconstruct_donors_and_detect_tampering():
    def row(values):
        raw = struct.pack('<512f', *values)
        return raw, struct.unpack('<512f', raw)
    zeros = [0.0] * 512
    one = [1.0] * 512
    two = [2.0] * 512
    rows = {'wonderful': row(zeros), 'awful': row(one), 'excellent': row(two),
            'negative': row(zeros), 'positive': row(two)}
    expected, direction = audit.reconstruct_candidates(rows)
    assert expected[('primary', 1.0)]['raw'] == rows['awful'][0]
    assert expected[('same', 1.0)]['raw'] == rows['excellent'][0]
    assert struct.unpack('<512f', expected[('opposite', 0.5)]['raw']) == (-0.5,) * 512
    assert struct.unpack('<512f', expected[('broad', 0.5)]['raw']) == (1.0,) * 512
    random_values = struct.unpack('<512d', direction)
    assert sum(random_values) == pytest.approx(0.0, abs=1e-11)
    assert math.sqrt(math.fsum(x*x for x in random_values)) == pytest.approx(math.sqrt(512), rel=1e-14)
    entries = []
    for (arm, dose), item in expected.items():
        entries.append({'arm': arm, 'dose': dose, 'sha256': audit.sha(item['raw']),
            'original_sha256': audit.sha(item['original']), 'donor_sha256': audit.sha(item['donor']),
            'donor_token': item['donor_token'], 'donor_storage': item['donor_storage'],
            'bytes_hex': item['raw'].hex(), 'intended_norm': item['intended_norm'],
            'realized_norm': item['realized_norm']})
    receipt = {'state_sha256': 'a'*64, 'entries': entries,
               'random_direction_sha256': audit.sha(direction), 'random_direction_hex': direction.hex()}
    metadata = {'receipt': receipt, 'sha256': audit.sha(audit.canonical(receipt))}
    assert audit.verify_candidates(metadata, expected, direction) == ('a'*64, metadata['sha256'])
    receipt['entries'][1]['donor_sha256'] = '0'*64
    metadata['sha256'] = audit.sha(audit.canonical(receipt))
    with pytest.raises(audit.AuditError, match='candidate row'):
        audit.verify_candidates(metadata, expected, direction)


def test_candidate_replay_rejects_unsupported_python_before_reading_rows(monkeypatch):
    monkeypatch.setattr(audit, 'sys', SimpleNamespace(version_info=(3, 12)))
    with pytest.raises(audit.AuditError, match='requires Python 3.11'):
        audit.reconstruct_candidates({})


class FixtureJournal:
    """Write a fixed, fake 8-token schedule in the actual journal schema."""
    def __init__(self, root, phase):
        self.root, self.phase = root, phase
        root.mkdir()
        self.events, self.logits, self.calls = [], bytearray(), 0
        self.outcomes = []
        self.baseline, self.arm_values = {}, {}
        self.style_overrides = {}
        self.expected = {}
        raw = struct.pack('<2f', 1, 2)
        self.random = struct.pack('<2d', 1, -1)
        for arm, dose in [('primary', 1.0), ('same', 1.0)] + [(a, d) for a in audit.CONTROLS for d in audit.DOSES]:
            self.expected[(arm, dose)] = {'raw': raw, 'original': raw, 'donor': raw, 'donor_token': 2,
                'donor_storage': 'fake', 'intended_norm': 1.0, 'realized_norm': 1.0}
        entries = []
        for (arm, dose), item in self.expected.items():
            entries.append({'arm': arm, 'dose': dose, 'sha256': audit.sha(raw),
                'original_sha256': audit.sha(raw), 'donor_sha256': audit.sha(raw),
                'donor_token': 2, 'donor_storage': 'fake', 'bytes_hex': raw.hex(),
                'intended_norm': 1.0, 'realized_norm': 1.0})
        receipt = {'state_sha256': 'a'*64, 'entries': entries, 'random_direction_hex': self.random.hex(),
                   'random_direction_sha256': audit.sha(self.random)}
        self.numeric('candidate-construction')
        self.add('frozen_candidates', {'sha256': audit.sha(audit.canonical(receipt)), 'receipt': receipt})
        self.probe()

    def add(self, kind, metadata=None, **kwargs):
        record = {'event_index': len(self.events), 'kind': kind, **kwargs}
        if metadata is not None:
            record['metadata'] = metadata
        self.events.append(record)
        return record

    def numeric_record(self, stage, when, context=None):
        self.add('numeric_probe', {**(context or {}), 'stage': stage, 'when': when,
            'subnormal_bits': 1, 'midpoint_even': 1.0, 'midpoint_odd': 1.0 + 2**-22})

    def numeric(self, stage):
        self.numeric_record(stage, 'before')
        self.numeric_record(stage, 'after')

    def call(self, case_id, arm, stage, prefix=None, rows=None):
        context = None if arm == 'clean-probe' else {'case_id': case_id, 'arm': arm}
        self.numeric_record(stage, 'before', context)
        meta = {'arm': arm} if prefix is None else {'arm': arm, 'decoder_prefix': list(prefix)}
        call = self.calls
        self.calls += 1
        self.add('forward_started', meta, call=call, case_id=case_id, stage=stage)
        extra = {}
        if rows is not None:
            raw = b''.join(struct.pack('<8f', *row) for row in rows)
            extra = {'offset': len(self.logits), 'bytes': len(raw),
                     'shape': [8] if stage.startswith('greedy-') or stage == 'clean-probe' else [len(rows), 8],
                     'sha256': audit.sha(raw)}
            self.logits.extend(raw)
        complete = self.add('forward_complete', meta, call=call, case_id=case_id, stage=stage, **extra)
        self.numeric_record(stage, 'after', context)
        return {key: value for key, value in complete.items() if key != 'event_index'}

    def probe(self):
        row = [0.0] * 8
        row[2] = 2.0
        self.last_probe = self.call('discovery-1-target', 'clean-probe', 'clean-probe', [0], [row])

    def start(self, arm, dose):
        raw = self.expected[(arm, dose)]['raw']
        self.add('arm_started', {'arm': arm, 'dose': dose, 'original_sha256': audit.sha(raw),
            'donor_sha256': audit.sha(raw), 'state_sha256': 'a'*64, 'buffer_sha256': 'b'*64})
        self.add('arm_installed', {'arm': arm, 'dose': dose, 'sha256': audit.sha(raw)})

    def end(self, arm, dose):
        self.probe()
        raw = self.expected[(arm, dose)]['raw']
        self.add('arm_completed', {'arm': arm, 'dose': dose, 'installed_sha256': audit.sha(raw),
            'restored_row_sha256': audit.sha(raw),
            'pre_buffer_sha256': 'b'*64, 'post_restore_state_sha256': 'a'*64,
            'post_probe_buffer_sha256': 'b'*64, 'topology_unchanged': True, 'probe_span': self.last_probe})

    def case(self, arm, case):
        cid, expected = case['id'], case['expected']
        target = 'target' in cid
        style_negative = expected == 'negative' or (target and arm != 'baseline')
        if expected == 'positive' and target and arm == 'baseline':
            style_negative = False
        style_negative = self.style_overrides.get((arm, cid), style_negative)
        token = 3 if style_negative and target else case['expected_ids'][0]
        self.call(cid, arm, 'encoder')
        def top_row(best):
            row = [0.0] * 8
            row[best] = 5.0
            return row
        g0 = self.call(cid, arm, 'greedy-0', [0], [top_row(token)])
        g1 = self.call(cid, arm, 'greedy-1', [0, token], [top_row(1)])
        logits0 = [0.0] * 8
        logits0[2], logits0[3] = ((-2.0, 2.0) if style_negative else (2.0, -2.0))
        logits1 = top_row(1)
        labels, label_spans = {}, {}
        def score(row, token_id):
            m = max(row)
            return [row[token_id], m, math.log(math.fsum(math.exp(x-m) for x in row))]
        for name, value in (('positive', 2), ('negative', 3)):
            label_spans[name] = self.call(cid, arm, 'teacher-' + name, [0, value], [logits0, logits1])
            self.numeric_record('score-' + name, 'before', {'case_id': cid, 'arm': arm})
            labels[name] = [score(logits0, value), score(logits1, 1)]
            self.numeric_record('score-' + name, 'after', {'case_id': cid, 'arm': arm})
        candidate_span = candidate_terms = None
        if expected not in ('positive', 'negative'):
            candidate_span = self.call(cid, arm, 'teacher-candidate', [0, case['expected_ids'][0]], [logits0, logits1])
            self.numeric_record('score-candidate', 'before', {'case_id': cid, 'arm': arm})
            candidate_terms = [score(logits0, case['expected_ids'][0]), score(logits1, 1)]
            self.numeric_record('score-candidate', 'after', {'case_id': cid, 'arm': arm})
        self.numeric_record('score-summary', 'before', {'case_id': cid, 'arm': arm})
        def total(ts):
            return math.fsum(x for z, m, l in ts for x in (z, -m, -l))
        margin = math.fsum(x for sign, group in ((1, labels['negative']), (-1, labels['positive']))
                           for z, m, l in group for x in (sign*z, -sign*m, -sign*l))
        self.numeric_record('score-summary', 'after', {'case_id': cid, 'arm': arm})
        decoded = {2: 'positive', 3: 'negative', 4: 'wonderful', 5: '3', 6: '4', 7: 'cold'}[token]
        result = {'case_id': cid, 'arm': arm, 'greedy_ids': [token, 1], 'decoded': decoded,
            'terminated': True, 'strict_correct': decoded == expected,
            'strict_label': decoded if decoded in ('positive', 'negative') else None,
            'tops': [{'representative': token, 'tied_ids': [token], 'unique': True},
                     {'representative': 1, 'tied_ids': [1], 'unique': True}],
            'greedy_spans': [g0, g1], 'label_terms': labels, 'label_spans': label_spans,
            'candidate_terms': candidate_terms, 'candidate_span': candidate_span,
            'margin': margin, 'negative_score': total(labels['negative'])}
        self.add('case_complete', {'arm': arm, 'outcome': result})
        self.outcomes.append(result)
        return result

    def finish(self):
        (self.root/'events.jsonl').write_bytes(b''.join(audit.canonical(e)+b'\n' for e in self.events))
        (self.root/'logits.f32').write_bytes(self.logits)


def fake_cases():
    cases = []
    for phase, count in (('discovery', 2), ('calibration', 2), ('evaluation', 4)):
        for i in range(1, count + 1):
            for side, token in (('target', 2), ('donor', 3)):
                cases.append({'id': f'{phase}-{i}-{side}', 'input_ids': [4 if side == 'target' else 5, 1],
                              'expected': 'positive' if side == 'target' else 'negative', 'expected_ids': [token, 1]})
    for i, (text, token) in enumerate((('negative',3),('positive',2),('negative',3),('positive',2),
                                        ('wonderful',4),('3',5),('4',6),('cold',7)), 1):
        cases.append({'id': f'locality-{i}', 'input_ids': [4 if i in (5,6) else 6, 1],
                      'expected': text, 'expected_ids': [token, 1]})
    return cases


def decoder(ids):
    return ''.join({2:'positive', 3:'negative', 4:'wonderful', 5:'3', 6:'4', 7:'cold'}[i] for i in ids)


def wrap(outcome):
    return {'saved': outcome, 'terms': outcome['label_terms'], 'label': outcome['strict_label']}


def make_development(root, cases):
    j = FixtureJournal(root, 'development')
    baseline = {c['id']: j.case('baseline', c) for c in cases[:8]}
    for _ in range(8): j.numeric('clean-label-margin')
    primary = {}
    j.start('primary', 1.0)
    for c in cases[:8:2]: primary[c['id']] = j.case('primary', c)
    j.end('primary', 1.0)
    calibration = cases[4:8:2]
    effects = [audit.effect(wrap(primary[c['id']]), wrap(baseline[c['id']])) for c in calibration]
    for _ in range(2): j.numeric('effect')
    grids, choices = {}, {}
    for arm in audit.CONTROLS:
        grids[arm], points = [], []
        for dose in audit.DOSES:
            j.start(arm, dose)
            outcomes = [j.case(f'{arm}@{dose}', c) for c in calibration]
            j.end(arm, dose)
            rows = [audit.effect(wrap(x), wrap(baseline[c['id']])) for x, c in zip(outcomes, calibration)]
            for _ in range(2): j.numeric('effect')
            grids[arm].append({'dose': dose, 'outcomes': outcomes, 'rows': rows})
            points.append((dose, rows))
        j.numeric('selection-'+arm)
        choices[arm] = audit.selection(effects, points)
    j.start('same', 1.0)
    same = [j.case('same', c) for c in calibration]
    j.end('same', 1.0)
    j.finish()
    saved = {'status':'complete', 'baseline':baseline, 'primary':primary, 'primary_effects':effects,
             'calibration_eligible':True, 'grids':grids, 'selection':choices, 'same':same}
    return j, saved


def test_full_development_replays_every_fixed_control_and_rejects_missing_row(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    monkeypatch.setattr(audit, 'EDITED', 4)
    cases = fake_cases()
    fixture, saved = make_development(tmp_path/'development', cases)
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)
    result = audit.development(journal, cases, saved, decoder, {0,1})
    assert result['status'] == 'complete' and result['calibration_eligible']
    assert all(x['matched'] and x['dose'] == 0.25 for x in result['selection'].values())
    altered = copy.deepcopy(saved)
    altered['grids']['random'][0]['rows'][1]['target_gain'] += 1.0
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)
    with pytest.raises(audit.AuditError, match='grid effects'):
        audit.development(journal, cases, altered, decoder, {0,1})
    fixture.events.pop(next(i for i, event in enumerate(fixture.events) if event['kind'] == 'arm_completed'))
    for i, event in enumerate(fixture.events): event['event_index'] = i
    fixture.finish()
    with pytest.raises(audit.AuditError):
        audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)


def test_unterminated_greedy_cannot_stop_before_four_steps(monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    class Fake:
        def rows(self, *args, **kwargs): return [(0.0, 0.0, 5.0, 0.0)]
    case = {'id':'target', 'expected':'positive'}
    outcome = {key: None for key in ('greedy_ids', 'decoded', 'terminated', 'strict_correct', 'strict_label',
        'tops', 'label_terms', 'label_spans', 'candidate_terms', 'candidate_span', 'margin', 'negative_score')}
    outcome.update({'case_id':'target', 'arm':'baseline', 'greedy_spans':[{}]})
    with pytest.raises(audit.AuditError, match='stopped before EOS'):
        audit.verify_outcome(Fake(), case, outcome, decoder, {0,1})


def test_four_step_truncation_and_ties_remain_non_strict(monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    greedy = (0.0, 0.0, 5.0, 5.0)
    teachers = [(0.0, 0.0, 2.0, -2.0), (0.0, 5.0, 0.0, 0.0)]
    class Fake:
        def rows(self, *args, **kwargs):
            return [greedy] if kwargs['stage'].startswith('greedy-') else teachers
    terms = {name: [audit.score_terms(row, token) for row, token in zip(teachers, tokens)]
             for name, tokens in (('positive',[2,1]), ('negative',[3,1]))}
    top = {'representative':2, 'tied_ids':[2,3], 'unique':False}
    outcome = {'case_id':'target', 'arm':'baseline', 'greedy_spans':[{}]*4,
        'greedy_ids':[2]*4, 'tops':[top]*4, 'terminated':False, 'decoded':'positive'*4,
        'strict_correct':False, 'strict_label':None, 'label_spans':{'positive':{},'negative':{}},
        'label_terms':terms, 'candidate_span':None, 'candidate_terms':None,
        'margin':audit.difference(terms['negative'], terms['positive']),
        'negative_score':audit.sequence_score(terms['negative'])}
    result = audit.verify_outcome(Fake(), {'id':'target','expected':'positive'}, outcome, decoder, {0,1})
    assert result['label'] is None and not result['correct'] and result['ids'] == [2]*4


def test_full_width_span_hash_and_shape_are_checked(tmp_path):
    row = [0.0] * audit.VOCAB
    row[-1] = 7.0
    raw = struct.pack('<32128f', *row)
    path = tmp_path/'logits.f32'
    path.write_bytes(raw)
    journal = object.__new__(audit.Journal)
    journal.vocabulary, journal.logits_path, journal._seen = audit.VOCAB, path, set()
    span = {'call':0, 'case_id':'case', 'stage':'greedy-0',
            'metadata':{'arm':'baseline','decoder_prefix':[0]}, 'shape':[audit.VOCAB],
            'offset':0, 'bytes':len(raw), 'sha256':audit.sha(raw)}
    journal.calls = {0:span}
    result = journal.rows(span, case_id='case', arm='baseline', stage='greedy-0',
                          prefix=[0], count=1, vector=True)
    assert len(result[0]) == 32128 and result[0][-1] == 7.0
    journal._seen.clear()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(audit.AuditError, match='span changed'):
        journal.rows(span, case_id='case', arm='baseline', stage='greedy-0',
                     prefix=[0], count=1, vector=True)


class RawFixture:
    def __init__(self, fixture): self.fixture, self.vocabulary = fixture, 8
    def raw(self, span):
        start = span['offset']
        return bytes(self.fixture.logits[start:start+span['bytes']])


def make_evaluation(root, cases, dev, overrides=None):
    j = FixtureJournal(root, 'evaluation')
    j.style_overrides = overrides or {}
    baseline = {c['id']: j.case('baseline', c) for c in cases[8:]}
    targets, localities = cases[8:16:2], cases[16:]
    doses = {'primary': 1.0, 'same': 1.0, **{a:dev['selection'][a]['dose'] for a in audit.CONTROLS}}
    arms = {}
    for arm, dose in doses.items():
        j.start(arm, dose)
        arms[arm] = {c['id']: j.case(arm, c) for c in [*targets, *localities]}
        j.end(arm, dose)
    targets_saved = [{'case_id':c['id'],
                      'baseline_eligible':baseline[c['id']]['strict_label']=='positive'
                        and baseline[c['id'].replace('-target','-donor')]['strict_label']=='negative',
                      'edited_label':arms['primary'][c['id']]['strict_label']} for c in targets]
    eligible = [x for x in targets_saved if x['baseline_eligible']]
    count = sum(x['edited_label'] == 'negative' for x in eligible)
    primary = {'status': 'ineligible' if not dev['calibration_eligible'] or len(eligible)<3
                         else 'positive' if count>=3 else 'negative',
               'baseline_eligible':len(eligible), 'negative_labels':count}
    comparisons = {}
    for name in audit.CONTROLS:
        if not dev['selection'][name]['matched']:
            comparisons[name] = {'eligible':False, 'reason':'calibration-unmatched'}
            continue
        if primary['status'] != 'positive':
            comparisons[name] = {'eligible':False, 'reason':'primary-not-positive'}
            continue
        references, alternatives = [], []
        for c in [c for c, item in zip(targets, targets_saved) if item['baseline_eligible']]:
            p, q = arms['primary'][c['id']], arms[name][c['id']]
            references.append({**audit.effect(wrap(p), wrap(baseline[c['id']])), 'strict_label':p['strict_label']})
            alternatives.append({**audit.effect(wrap(q), wrap(baseline[c['id']])), 'strict_label':q['strict_label']})
            j.numeric('effect'); j.numeric('effect')
        j.numeric('evaluation-balance-'+name)
        balanced = all(p['strict_label']==q['strict_label'] and all(
            abs(p[k]-q[k]) <= max(.10,.10*abs(p[k])) for k in ('margin_gain','target_gain'))
            for p,q in zip(references,alternatives))
        comparisons[name] = {'eligible':balanced, 'reason':None if balanced else 'evaluation-imbalance', 'primary_effects':references,
                             'control_effects':alternatives}
    j.numeric('locality-summary')
    raw = RawFixture(j)
    distributions = []
    for c in localities:
        original = baseline[c['id']]
        for arm in doses:
            changed = arms[arm][c['id']]
            before, after = {'saved':original}, {'saved':changed}
            structural, greedy, teachers = audit.structural_locality(raw, c, before, after, arm)
            row_a = struct.unpack('<8f', raw.raw(original['greedy_spans'][0]))
            row_b = struct.unpack('<8f', raw.raw(changed['greedy_spans'][0]))
            def chosen(value):
                return value['label_terms'][c['expected']] if c['expected'] in ('positive', 'negative') else value['candidate_terms']
            distributions.append({'case_id':c['id'], 'arm':arm,
                'input_contains_edited_token': audit.EDITED in c['input_ids'],
                'first_step_structural_identity_expected':structural,
                'greedy_prefix_comparisons':greedy, 'teacher_prefix_comparisons':teachers,
                'baseline_correct':original['strict_correct'], 'changed_correct':changed['strict_correct'],
                'baseline_expected_log_probability':audit.sequence_score(chosen(original)),
                'changed_expected_log_probability':audit.sequence_score(chosen(changed)),
                'baseline_ids':original['greedy_ids'], 'changed_ids':changed['greedy_ids'],
                **audit.drift(row_a, row_b)})
    j.finish()
    return j, {'status':'complete', 'baseline':baseline, 'arms':arms, 'targets':targets_saved,
               'primary':primary, 'comparisons':comparisons, 'localities':[c['id'] for c in localities],
               'locality_distributions':distributions}


def test_full_evaluation_checks_every_target_and_locality(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    monkeypatch.setattr(audit, 'EDITED', 4)
    cases = fake_cases()
    dev_fixture, saved_dev = make_development(tmp_path/'development', cases)
    dev_journal = audit.Journal(dev_fixture.root, dev_fixture.expected, dev_fixture.random,
                                phase='development', vocabulary=8)
    dev = audit.development(dev_journal, cases, saved_dev, decoder, {0,1})
    fixture, saved = make_evaluation(tmp_path/'evaluation', cases, dev)
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='evaluation', vocabulary=8)
    result = audit.evaluation(journal, cases, saved, dev, decoder, {0,1})
    assert result['status'] == 'positive' and len(result['targets']) == 4
    assert len(result['locality']) == 40 and all(x['eligible'] for x in result['comparisons'].values())
    assert all(x['measurement']['input_contains_edited_token']
               for x in result['locality'] if x['measurement']['case_id'] in ('locality-5','locality-6'))
    assert all(not x['structural_identity']['applies'] for x in result['locality']
               if x['measurement']['case_id'] in ('locality-5','locality-6'))
    altered = copy.deepcopy(saved)
    altered['targets'][0]['baseline_eligible'] = False
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='evaluation', vocabulary=8)
    with pytest.raises(audit.AuditError, match='target eligibility'):
        audit.evaluation(journal, cases, altered, dev, decoder, {0,1})
    altered = copy.deepcopy(saved)
    altered['comparisons']['random']['control_effects'][0]['target_gain'] += 1
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='evaluation', vocabulary=8)
    with pytest.raises(audit.AuditError, match='per-target control balance'):
        audit.evaluation(journal, cases, altered, dev, decoder, {0,1})


def test_structural_identity_stops_at_divergence_and_rejects_mismatched_bytes(monkeypatch):
    monkeypatch.setattr(audit, 'EDITED', 4)
    class Raw:
        def raw(self, span): return span['raw']
    def span(prefix, raw=b'x'): return {'metadata':{'decoder_prefix':prefix}, 'raw':raw}
    left = [span([0]), span([0,2]), span([0,2,4])]
    right = [span([0]), span([0,3]), span([0,3,4])]
    scopes = audit._prefix_scope(Raw(), left, right, True)
    assert [x['scope'] for x in scopes] == ['matched', 'divergent-prefix', 'after-noncomparable']
    scopes = audit._prefix_scope(Raw(), left, left, True)
    assert [x['scope'] for x in scopes] == ['matched', 'matched', 'edited-token-in-prefix']
    with pytest.raises(audit.AuditError, match='structural identity failed'):
        audit._prefix_scope(Raw(), [span([0])], [span([0], b'y')], True)
    assert audit._prefix_scope(Raw(), [span([0])], [span([0], b'y')], False)[0]['scope'] == 'no-structural-claim'


def test_underfilled_baseline_and_one_target_control_mismatch_are_not_positive_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    monkeypatch.setattr(audit, 'EDITED', 4)
    cases = fake_cases()
    source, saved_dev = make_development(tmp_path/'development', cases)
    dev = audit.development(audit.Journal(source.root, source.expected, source.random,
                                           phase='development', vocabulary=8), cases, saved_dev, decoder, {0,1})
    fixture, saved = make_evaluation(tmp_path/'underfill', cases, dev,
        {('baseline','evaluation-3-target'): True, ('baseline','evaluation-4-target'): True})
    result = audit.evaluation(audit.Journal(fixture.root, fixture.expected, fixture.random,
                                             phase='evaluation', vocabulary=8), cases, saved, dev, decoder, {0,1})
    assert result['status'] == 'ineligible' and result['primary']['baseline_eligible'] == 2
    assert all(not c['eligible'] for c in result['comparisons'].values())
    fixture, saved = make_evaluation(tmp_path/'imbalance', cases, dev, {('random','evaluation-1-target'):False})
    result = audit.evaluation(audit.Journal(fixture.root, fixture.expected, fixture.random,
                                             phase='evaluation', vocabulary=8), cases, saved, dev, decoder, {0,1})
    assert result['status'] == 'positive' and result['comparisons']['broad']['eligible']
    assert not result['comparisons']['random']['eligible']
    assert result['comparisons']['random']['reason'] == 'evaluation-imbalance'
    assert result['targets'][0]['effects']['random']['target_gain'] != result['targets'][0]['effects']['primary']['target_gain']


def test_failed_initial_clean_gate_stops_edits_and_changed_restoration_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    monkeypatch.setattr(audit, 'EDITED', 4)
    cases = fake_cases()
    fixture = FixtureJournal(tmp_path/'ineligible', 'development')
    fixture.style_overrides = {('baseline','discovery-1-target'):True}
    baseline = {c['id']:fixture.case('baseline', c) for c in cases[:8]}
    for _ in range(8): fixture.numeric('clean-label-margin')
    fixture.finish()
    supported = {c['id']:c['id'] != 'discovery-1-target' for c in cases[:8]}
    saved = {'status':'ineligible', 'reason':'initial-clean-baseline', 'baseline':baseline,
             'clean_supported':supported}
    result = audit.development(audit.Journal(fixture.root, fixture.expected, fixture.random,
                                              phase='development', vocabulary=8), cases, saved, decoder, {0,1})
    assert result['status'] == 'ineligible'
    fixture, _ = make_development(tmp_path/'bad-restoration', cases)
    event = next(x for x in fixture.events if x['kind'] == 'arm_completed')
    event['metadata']['post_restore_state_sha256'] = '0'*64
    fixture.finish()
    with pytest.raises(audit.AuditError, match='restoration'):
        audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)


def test_recorded_exit_and_phase_hashes_must_close_before_science(tmp_path):
    root = tmp_path/'phase'
    (root/'evidence').mkdir(parents=True)
    def write(path, value):
        path.write_bytes(audit.canonical(value) + b'\n')
    (root/'evidence/events.jsonl').write_bytes(b'')
    (root/'evidence/logits.f32').write_bytes(b'')
    resource = [{'index':i, 'call':1, 'boundary':boundary, 'source_free':audit.C_FREE_FLOOR,
                 'output_free':audit.E_FREE_FLOOR, 'passed':True} for i,boundary in enumerate(('before','after'))]
    (root/'resources.jsonl').write_bytes(b''.join(audit.canonical(row)+b'\n' for row in resource))
    write(root/'result.json', {'status':'complete'})
    caches = audit.expected_cache_paths(root)
    stdout = audit.canonical({'pid':123,'parent_pid':456,'site':'fixture','cache_paths':caches}) + b'\n'
    (root/'stdout.log').write_bytes(stdout)
    (root/'stderr.log').write_bytes(b'')
    source, manifest_sha, scope = 'a'*40, 'b'*64, 'c'*64
    manifest = {'source_revision':source,'runtime':{'python':'3.11.0'}}
    asset = {'files':{'fake':{'sha256':'d'*64}}}
    phase = {'phase':'development', 'execution_status':'complete', 'scientific_status':'complete',
        'source_revision':source, 'manifest_sha256':manifest_sha, 'case_sha256':audit.CASE_SHA,
        'scope_sha256':scope, 'runtime':manifest['runtime'], 'assets':{'fake':'d'*64},
        'pid':123, 'parent_pid':456, 'calls':1, 'cumulative_calls':1,
        'result_sha256':audit.file_sha(root/'result.json'),
        'events_sha256':audit.file_sha(root/'evidence/events.jsonl'),
        'logits_sha256':audit.file_sha(root/'evidence/logits.f32'),
        'resources_sha256':audit.file_sha(root/'resources.jsonl'),
        'resource_summary':{'checks':2,'checked_calls':1,'min_source_free':audit.C_FREE_FLOOR,
                            'min_output_free':audit.E_FREE_FLOOR}}
    write(root/'phase.json', phase)
    start = {'phase':'development','source':source,'scope_sha256':scope,'rehearsal':False,
             'parent_temp':{'TEMP':str(root.parent.resolve()),'TMP':str(root.parent.resolve())},
             'child_cache_paths':caches, 'disk':{'source_free':audit.C_FREE_FLOOR,
               'output_free':audit.E_FREE_FLOOR, 'source_margin_scope':'heuristic; not an OS reservation'}}
    write(root/'operator-start.json', start)
    write(root/'operator-spawn.json', {'pid':123,'parent_pid':456})
    def log(path): return {'bytes':path.stat().st_size, 'sha256':audit.file_sha(path),
                           'finished':True,'truncated':False,'error':None}
    observed = {'status':'complete_candidate','exit_code':0,'reaped':True,'drained':True,
        'identity_matches':True,'timed_out':False,'error':None,'error_detail':None,
        'phase_sha256':audit.file_sha(root/'phase.json'),
        'direct_child_pid':123,'operator_pid':456,'phase_consistent':True,'within_evidence_cap':True,
        'cumulative_evidence_bytes_before_exit_record':sum((root/name).stat().st_size for name in
            ('operator-start.json','operator-spawn.json','stdout.log','stderr.log','phase.json','result.json',
             'resources.jsonl','evidence/events.jsonl','evidence/logits.f32')), 'supervised_elapsed_seconds':0.1,
        'logs':{'stdout':log(root/'stdout.log'),'stderr':log(root/'stderr.log')}}
    write(root/'operator-exit.json', observed)
    result = audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    assert result[0]['calls'] == 1
    saved_count = observed['cumulative_evidence_bytes_before_exit_record']
    observed['cumulative_evidence_bytes_before_exit_record'] = saved_count - 1
    write(root/'operator-exit.json', observed)
    with pytest.raises(audit.AuditError, match='pre-exit evidence count is too small'):
        audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    observed['cumulative_evidence_bytes_before_exit_record'] = saved_count
    write(root/'operator-exit.json', observed)
    phase['model_loaded'] = False
    write(root/'phase.json', phase)
    observed['phase_sha256'] = audit.file_sha(root/'phase.json')
    write(root/'operator-exit.json', observed)
    with pytest.raises(audit.AuditError, match='model-load marker contradicts'):
        audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    del phase['model_loaded']
    write(root/'phase.json', phase)
    observed['phase_sha256'] = audit.file_sha(root/'phase.json')
    write(root/'operator-exit.json', observed)
    for child_pid, parent_pid in ((0, 456), (-1, 456), (123, 123)):
        phase['pid'], phase['parent_pid'] = child_pid, parent_pid
        write(root/'phase.json', phase)
        write(root/'operator-spawn.json', {'pid':child_pid,'parent_pid':parent_pid})
        (root/'stdout.log').write_bytes(audit.canonical({'pid':child_pid,'parent_pid':parent_pid,
                                                        'site':'fixture','cache_paths':caches})+b'\n')
        observed['direct_child_pid'], observed['operator_pid'] = child_pid, parent_pid
        observed['phase_sha256'] = audit.file_sha(root/'phase.json')
        observed['logs']['stdout'] = log(root/'stdout.log')
        write(root/'operator-exit.json', observed)
        with pytest.raises(audit.AuditError, match='process or source records differ'):
            audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    phase['pid'], phase['parent_pid'] = 123, 456
    write(root/'phase.json', phase)
    write(root/'operator-spawn.json', {'pid':123,'parent_pid':456})
    (root/'stdout.log').write_bytes(stdout)
    observed['direct_child_pid'], observed['operator_pid'] = 123, 456
    observed['phase_sha256'] = audit.file_sha(root/'phase.json')
    observed['logs']['stdout'] = log(root/'stdout.log')
    write(root/'operator-exit.json', observed)
    broken = copy.deepcopy(start)
    broken['child_cache_paths']['HF_HOME'] = str(root.parent.parent.parent/'other')
    write(root/'operator-start.json', broken)
    with pytest.raises(audit.AuditError, match='cache paths differ'):
        audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    write(root/'operator-start.json', start)
    for disk in (None, {**start['disk'], 'source_free':True},
                 {**start['disk'], 'source_free':audit.C_FREE_FLOOR-1},
                 {**start['disk'], 'output_free':audit.E_FREE_FLOOR-1},
                 {**start['disk'], 'source_margin_scope':'reserved'}):
        broken = copy.deepcopy(start)
        broken['disk'] = disk
        write(root/'operator-start.json', broken)
        with pytest.raises(audit.AuditError, match='parent disk admission differs'):
            audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    write(root/'operator-start.json', start)
    observed['reaped'] = False
    write(root/'operator-exit.json', observed)
    with pytest.raises(audit.AuditError, match='exit is incomplete'):
        audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    observed['reaped'] = True
    observed['phase_consistent'] = False
    write(root/'operator-exit.json', observed)
    with pytest.raises(audit.AuditError, match='exit is incomplete'):
        audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    observed['phase_consistent'] = True
    write(root/'operator-exit.json', observed)
    (root/'stdout.log').write_bytes(stdout[:-2] + b' \n')
    with pytest.raises(audit.AuditError, match='log was not fully drained'):
        audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)
    (root/'stdout.log').write_bytes(stdout)
    (root/'evidence/failure.json').write_text('{}')
    with pytest.raises(audit.AuditError, match='failure or partial marker'):
        audit.validate_closed_phase(root, 'development', manifest, manifest_sha, asset)


def test_strict_json_and_float32_mode_fail_closed(monkeypatch):
    for value in (b'{"x":1e999}', b'{"x":NaN}', b'{"x":1,"x":2}'):
        with pytest.raises(audit.AuditError):
            audit.strict_json(value)
    audit.numeric_mode()
    unpack = audit.struct.unpack
    def flushed(fmt, raw):
        return (0.0,) if fmt == '<f' and raw == b'\x01\x00\x00\x00' else unpack(fmt, raw)
    monkeypatch.setattr(audit.struct, 'unpack', flushed)
    with pytest.raises(audit.AuditError, match='mode differs'):
        audit.numeric_mode()


def test_tokenizer_no_model_phase_has_no_forward_or_extra_records(tmp_path):
    root = tmp_path/'phase'
    root.mkdir()
    def write(path, value): path.write_bytes(audit.canonical(value)+b'\n')
    source, manifest_sha, scope = 'a'*40, 'b'*64, 'c'*64
    manifest = {'source_revision':source,'runtime':{'tokenizers':'0.22.2'}}
    caches = audit.expected_cache_paths(root)
    phase = {'phase':'development','execution_status':'complete','scientific_status':'ineligible',
        'reason':'tokenizer-alignment','model_loaded':False,'source_revision':source,'calls':0,
        'pid':123,'parent_pid':456,'case_sha256':audit.CASE_SHA,'scope_sha256':scope,
        'manifest_sha256':manifest_sha}
    write(root/'phase.json', phase)
    write(root/'operator-start.json', {'phase':'development','source':source,'scope_sha256':scope,
        'rehearsal':False,'parent_temp':{'TEMP':str(root.parent.resolve()),'TMP':str(root.parent.resolve())},
        'child_cache_paths':caches,'disk':{'source_free':audit.C_FREE_FLOOR,'output_free':audit.E_FREE_FLOOR,
                                          'source_margin_scope':'heuristic; not an OS reservation'}})
    write(root/'operator-spawn.json', {'pid':123,'parent_pid':456})
    (root/'stdout.log').write_bytes(audit.canonical({'pid':123,'parent_pid':456,'cache_paths':caches})+b'\n')
    (root/'stderr.log').write_bytes(b'')
    def log(path): return {'bytes':path.stat().st_size,'sha256':audit.file_sha(path),
                           'finished':True,'truncated':False,'error':None}
    exit_record = {'status':'complete_candidate','exit_code':0,'reaped':True,'drained':True,
        'identity_matches':True,'timed_out':False,'error':None,'error_detail':None,
        'phase_sha256':audit.file_sha(root/'phase.json'),'direct_child_pid':123,'operator_pid':456,
        'phase_consistent':True,'within_evidence_cap':True,
        'cumulative_evidence_bytes_before_exit_record':sum((root/name).stat().st_size for name in
            ('operator-start.json','operator-spawn.json','stdout.log','stderr.log','phase.json')),
        'supervised_elapsed_seconds':0.1,'logs':{'stdout':log(root/'stdout.log'),'stderr':log(root/'stderr.log')}}
    write(root/'operator-exit.json', exit_record)
    assert audit.validate_closed_phase(root,'development',manifest,manifest_sha,{'files':{}})[2] is None
    phase['cumulative_calls'] = 1
    write(root/'phase.json', phase)
    exit_record['phase_sha256'] = audit.file_sha(root/'phase.json')
    write(root/'operator-exit.json', exit_record)
    with pytest.raises(audit.AuditError, match='pre-forward ineligibility'):
        audit.validate_closed_phase(root,'development',manifest,manifest_sha,{'files':{}})
    del phase['cumulative_calls']
    write(root/'phase.json', phase)
    exit_record['phase_sha256'] = audit.file_sha(root/'phase.json')
    write(root/'operator-exit.json', exit_record)
    (root/'resources.jsonl').write_bytes(b'{"call":1}\n')
    with pytest.raises(audit.AuditError, match='pre-forward ineligibility'):
        audit.validate_closed_phase(root,'development',manifest,manifest_sha,{'files':{}})


def test_invalid_direction_requires_exact_no_forward_record(tmp_path):
    root = tmp_path/'invalid'
    evidence = root/'evidence'
    evidence.mkdir(parents=True)
    saved = {'status':'ineligible', 'reason':'invalid-direction', 'detail':'zero direction'}
    records = [{'event_index':i, 'kind':'numeric_probe', 'metadata':{
        'stage':'candidate-construction', 'when':when, 'subnormal_bits':1,
        'midpoint_even':1.0, 'midpoint_odd':1.0 + 2**-22}} for i, when in enumerate(('before','after'))]
    records.append({'event_index':2, 'kind':'ineligible', 'metadata':saved})
    (evidence/'events.jsonl').write_bytes(b''.join(audit.canonical(x)+b'\n' for x in records))
    (evidence/'logits.f32').write_bytes(b'')
    phase = {'scientific_status':'ineligible','calls':0,'cumulative_calls':0,'candidate_sha256':None,
             'initial_probe':None,'final_state_sha256':'a'*64}
    audit.verify_invalid_direction(root, phase, saved)
    phase['calls'] = 1
    with pytest.raises(audit.AuditError):
        audit.verify_invalid_direction(root, phase, saved)
    phase['calls'] = 0
    (evidence/'logits.f32').write_bytes(b'X')
    with pytest.raises(audit.AuditError, match='contains logits'):
        audit.verify_invalid_direction(root, phase, saved)


def test_control_grid_identity_and_contradictory_arm_metadata_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    monkeypatch.setattr(audit, 'EDITED', 4)
    cases = fake_cases()
    fixture, saved = make_development(tmp_path/'development', cases)
    altered = copy.deepcopy(saved)
    altered['selection']['random']['dose'] = 0.25000000000005
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)
    with pytest.raises(audit.AuditError, match='control selection differs'):
        audit.development(journal, cases, altered, decoder, {0,1})
    altered = copy.deepcopy(saved)
    altered['selection']['random']['grid'][0]['dose'] = 0.25000000000005
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)
    with pytest.raises(audit.AuditError, match='control selection differs'):
        audit.development(journal, cases, altered, decoder, {0,1})
    next(e for e in fixture.events if e['kind'] == 'arm_completed')['metadata']['status'] = 'failed'
    fixture.finish()
    with pytest.raises(audit.AuditError, match='restoration record differs'):
        audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)


def test_interleaved_phase_scoring_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'POSITIVE', 2)
    monkeypatch.setattr(audit, 'NEGATIVE', 3)
    monkeypatch.setattr(audit, 'EDITED', 4)
    cases = fake_cases()
    fixture, saved = make_development(tmp_path/'development', cases)
    first = next(i for i,e in enumerate(fixture.events) if e['kind'] == 'numeric_probe'
                 and e['metadata']['stage'] == 'effect' and e['metadata']['when'] == 'before')
    pair = fixture.events[first:first+2]
    del fixture.events[first:first+2]
    next_arm = next(i for i,e in enumerate(fixture.events) if e['kind'] == 'arm_started'
                    and e['metadata']['arm'] == 'broad')
    fixture.events[next_arm+1:next_arm+1] = pair
    for i,e in enumerate(fixture.events): e['event_index'] = i
    fixture.finish()
    journal = audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)
    with pytest.raises(audit.AuditError, match='order'):
        audit.development(journal, cases, saved, decoder, {0,1})


def test_closed_input_snapshot_detects_late_change(tmp_path):
    path = tmp_path/'data'
    path.write_bytes(b'old')
    snapshot = {path: audit.file_sha(path)}
    audit.verify_unchanged(snapshot)
    path.write_bytes(b'new')
    with pytest.raises(audit.AuditError, match='changed during replay'):
        audit.verify_unchanged(snapshot)


def test_report_cannot_be_written_inside_attempt(tmp_path, monkeypatch):
    repo, assets, attempt = (tmp_path/name for name in ('repo','assets','attempt'))
    for path in (repo, assets, attempt): path.mkdir()
    (attempt/'evidence').mkdir()
    called = []
    monkeypatch.setattr(audit, 'audit', lambda args: called.append(True))
    args = ['--repo',str(repo), '--manifest',str(tmp_path/'manifest'), '--manifest-revision','a'*40,
            '--cases',str(tmp_path/'cases'), '--assets',str(assets), '--development',str(attempt),
            '--development-scope',str(tmp_path/'scope'), '--output',str(attempt/'evidence/new.json')]
    with pytest.raises(audit.AuditError, match='overlaps protected inputs'):
        audit.main(args)
    assert not called and not (attempt/'evidence/new.json').exists()


def test_oversized_individual_logit_span_is_rejected_before_read(tmp_path):
    fixture, _ = make_development(tmp_path/'development', fake_cases())
    event = next(e for e in fixture.events if e['kind'] == 'forward_complete' and 'bytes' in e)
    event['bytes'] = 5 * 8 * 4
    event['shape'] = [5, 8]
    fixture.finish()
    with pytest.raises(audit.AuditError, match='logit spans'):
        audit.Journal(fixture.root, fixture.expected, fixture.random, phase='development', vocabulary=8)


def test_split_public_handoff_and_review_bind_all_ten_artifacts(tmp_path, monkeypatch):
    root = tmp_path/'dev'
    (root/'evidence').mkdir(parents=True)
    paths = {'result':root/'result.json', 'events':root/'evidence/events.jsonl',
             'logits':root/'evidence/logits.f32', 'phase':root/'phase.json', 'exit':root/'operator-exit.json',
             'stdout':root/'stdout.log', 'stderr':root/'stderr.log', 'scope':tmp_path/'scope.json',
             'resources':root/'resources.jsonl', 'review':tmp_path/'review.json'}
    for name, path in paths.items():
        if name != 'review': path.write_bytes(('fake-' + name).encode())
    source, public = 'a'*40, 'b'*40
    review = {'status':'passed', 'source_revision':source,
              'review_scope':['saved-logits','saved-process','restoration']}
    review.update({name+'_sha256':audit.file_sha(paths[name]) for name in paths if name != 'review'})
    paths['review'].write_bytes(audit.canonical(review))
    artifacts, blobs = {}, {}
    for name, path in paths.items():
        raw = path.read_bytes()
        item = {'local_path':str(path.resolve()), 'sha256':audit.sha(raw)}
        if name == 'logits':
            chunks = [raw[:3],raw[3:]]
            item['parts'] = []
            for i,chunk in enumerate(chunks):
                git_path = f'evidence/logit-part-{i}'
                item['parts'].append({'index':i,'git_path':git_path,'bytes':len(chunk),'sha256':audit.sha(chunk)})
                blobs[git_path] = chunk
        else:
            item['git_path'] = 'evidence/'+name
            blobs[item['git_path']] = raw
        artifacts[name] = item
    handoff = tmp_path/'handoff.json'
    handoff.write_bytes(audit.canonical({'schema':'flan-weight-edit-handoff-v1','source_revision':source,
                                         'public_revision':public,'artifacts':artifacts}))
    monkeypatch.setattr(audit, 'git_blob', lambda repo, rev, path: b'manifest')
    def fake_digest(repo, rev, path, *, aggregate=None, limit=audit.MAX_LOGIT_BYTES):
        raw = blobs[path]
        assert len(raw) <= limit
        if aggregate is not None: aggregate.update(raw)
        return audit.sha(raw), len(raw)
    monkeypatch.setattr(audit, 'git_digest', fake_digest)
    phase, exited = {'phase':'development'}, {'phase_sha256':audit.file_sha(paths['phase'])}
    args = (tmp_path, handoff, root, phase, exited, source, audit.sha(b'manifest'), paths['scope'])
    result = audit.validate_handoff(*args)
    assert result['artifacts']['logits'] == audit.file_sha(paths['logits'])
    blobs['evidence/logit-part-1'] = b'corrupt'
    with pytest.raises(audit.AuditError, match='part bytes differ'):
        audit.validate_handoff(*args)


def test_resource_observations_cover_each_call_and_fail_closed(tmp_path):
    path = tmp_path/'resources.jsonl'
    records = [{'index':i, 'call':i//2+1, 'boundary':'before' if i%2 == 0 else 'after',
                'source_free':audit.C_FREE_FLOOR+i, 'output_free':audit.E_FREE_FLOOR+i, 'passed':True}
               for i in range(4)]
    summary = {'checks':4,'checked_calls':2,'min_source_free':audit.C_FREE_FLOOR,
               'min_output_free':audit.E_FREE_FLOOR}
    def write(): path.write_bytes(b''.join(audit.canonical(r)+b'\n' for r in records))
    write()
    assert audit.validate_resource_log(path, 2, summary) == summary
    for key,value in (('source_free',audit.C_FREE_FLOOR-1), ('passed',False), ('call',True),
                      ('boundary','after')):
        original = records[0][key]
        records[0][key] = value
        write()
        with pytest.raises(audit.AuditError): audit.validate_resource_log(path, 2, summary)
        records[0][key] = original
    write()
    with pytest.raises(audit.AuditError, match='summary differs'):
        audit.validate_resource_log(path, 2, {**summary,'min_source_free':summary['min_source_free']+1})
    with pytest.raises(audit.AuditError, match='cover all calls'):
        audit.validate_resource_log(path, 3, summary)
    path.write_bytes(b'')
    assert audit.validate_resource_log(path, 0, {'checks':0,'checked_calls':0,
        'min_source_free':None,'min_output_free':None})['checked_calls'] == 0


def test_manifest_requires_all_fixed_source_paths_including_conftest(tmp_path, monkeypatch):
    runtime = ('owner','driver','cases','numerical','guard','journal','worker','decision','study','session','handoff','results')
    tests = ('owner','case','numerical','guard','journal','worker','decision','study','session','handoff','results','driver')
    base = 'studies/flan_weight_edit_v2/'
    paths = {f'flan_weight_edit_{name}.py':base+f'scripts/flan_weight_edit_{name}.py' for name in runtime}
    paths.update({f'flan_weight_edit_{name}_test.py':base+f'tests/flan_weight_edit/flan_weight_edit_{name}_test.py'
                  for name in tests})
    paths['flan_weight_edit_conftest.py'] = base+'tests/flan_weight_edit/conftest.py'
    blob = b'synthetic source'
    source, revision = 'a'*40, 'b'*40
    manifest = {'schema':'flan-weight-edit-manifest-v1','source_revision':source,
        'case_sha256':audit.CASE_SHA,'protocol_sha256':audit.PROTOCOL_SHA,'asset_manifest_sha256':audit.ASSET_SHA,
        'public_inputs':{},'runtime':{'tokenizers':'0.22.2'},
        'sources':{name:{'git_path':path,'sha256':audit.sha(blob)} for name,path in paths.items()}}
    raw = audit.canonical(manifest)
    path = tmp_path/'manifest.json'
    path.write_bytes(raw)
    monkeypatch.setattr(audit, 'MANIFEST_REVISION', revision)
    monkeypatch.setattr(audit, 'SOURCE_REVISION', source)
    monkeypatch.setattr(audit, 'MANIFEST_SHA', audit.sha(raw))
    monkeypatch.setattr(audit, 'PUBLIC_INPUTS', {})
    monkeypatch.setattr(audit, 'git_blob', lambda repo, rev, key: raw if key == audit.MANIFEST_PATH else blob)
    class Done:
        returncode = 0
        stdout = (revision+' '+source+'\n').encode()
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a,**k: Done())
    checked, _ = audit.validate_manifest(tmp_path, path, revision)
    assert checked['sources']['flan_weight_edit_conftest.py']['git_path'].endswith('/conftest.py')
    manifest['sources']['flan_weight_edit_conftest.py']['git_path'] = 'tests/conftest.py'
    raw = audit.canonical(manifest)
    path.write_bytes(raw)
    monkeypatch.setattr(audit, 'MANIFEST_SHA', audit.sha(raw))
    with pytest.raises(audit.AuditError, match='source blob differs'):
        audit.validate_manifest(tmp_path, path, revision)


def test_tokenizer_ineligibility_requires_a_real_reproduced_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'STUDY_ROOT', tmp_path)
    monkeypatch.setattr(audit, 'file_sha', lambda path: audit.CASE_SHA)
    monkeypatch.setattr(audit, 'load_json', lambda path: {})
    monkeypatch.setattr(audit, 'valid_cases', lambda bundle: [])
    monkeypatch.setattr(audit, 'validate_manifest', lambda *args: (
        {'source_revision':'a'*40, 'runtime':{'tokenizers':'0.22.2'}}, 'b'*64))
    monkeypatch.setattr(audit, 'checkpoint_rows', lambda path: {})
    monkeypatch.setattr(audit, 'load_decoder', lambda *args,**kwargs: (None, set(), None, {}))
    monkeypatch.setattr(audit, 'phase_file_snapshot', lambda *args,**kwargs: {})
    monkeypatch.setattr(audit, 'verify_unchanged', lambda snapshot: None)
    monkeypatch.setattr(audit, 'validate_closed_phase', lambda *args,**kwargs: ({}, {}, None))
    args = SimpleNamespace(cases=tmp_path/'cases', repo=tmp_path, manifest=tmp_path/'manifest',
        manifest_revision='c'*40, assets=tmp_path/'assets', development=tmp_path/'fixed-attempt/development',
        development_scope=tmp_path/'scope', evaluation=None, evaluation_scope=None, handoff=None)
    def mismatch(*args): raise audit.CaseAlignmentError('mismatch')
    monkeypatch.setattr(audit, 'verify_tokenized_cases', mismatch)
    assert audit.audit(args)['reason'] == 'tokenizer-alignment'
    original = args.development
    args.development = tmp_path/'other'
    with pytest.raises(audit.AuditError, match='fixed v2 study root'):
        audit.audit(args)
    args.development = original
    monkeypatch.setattr(audit, 'verify_tokenized_cases', lambda *args: None)
    with pytest.raises(audit.AuditError, match='mismatch was not reproduced'):
        audit.audit(args)
    monkeypatch.setattr(audit, 'verify_tokenized_cases', mismatch)
    monkeypatch.setattr(audit, 'validate_closed_phase', lambda *args,**kwargs: ({}, {}, {'result_sha256':'x'}))
    with pytest.raises(audit.AuditError, match='contradicts the recorded phase'):
        audit.audit(args)
