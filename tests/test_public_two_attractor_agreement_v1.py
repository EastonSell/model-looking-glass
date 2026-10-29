import importlib.util
import json
from pathlib import Path


def test_prespecified_factorial_contrasts():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('public_two_attractor', repo / 'tools/public_two_attractor_agreement_v1.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = json.loads((repo / 'experiments/distilgpt2_two_attractor_cases_v1.json').read_bytes())['cases']
    rows = []
    for c in cases:
        sign = 1 if c['subject_number'] == 'singular' else -1
        a = int(c['near_number'] != c['subject_number'])
        b = int(c['beside_number'] != c['subject_number'])
        m = sign * (3 - a - 2*b + 7*a*b)
        rows.append({'case_id': c['case_id'], 'logits': {'is': m, 'are': 0.0, 'was': m, 'were': 0.0}})
    out = mod.aggregate(cases, rows)
    assert len(out['case_results']) == 128
    assert len(out['factorial_units']) == 32
    for unit in out['factorial_units']:
        assert unit['contrasts']['present']['interaction'] == 7
        assert unit['contrasts']['present']['near_shift'] == 2.5
        assert unit['contrasts']['present']['beside_shift'] == 1.5
    assert out['unit_summary']['partition=evaluation']['units'] == 8
    assert out['unit_summary']['all']['present']['interaction'] == 7
    assert all(item['cases'] == 32 for item in out['condition_summary'].values())
    assert len(out['subject_controls']) == 64
    assert len(out['subject_control_summary']) == 16
    assert out['subject_control_summary']['evaluation/singular/singular']['present'] == 10
    assert out['subject_control_summary']['evaluation/singular/plural']['present'] == 3
    assert all(x['present'] == {'correct': 4, 'ties': 0, 'wrong_sign': 0,
                                'incorrect_including_ties': 0}
               for x in out['baseline_counts'].values())
    baseline = [c['case_id'] for c in cases if c['subject_pair'] in (3, 4)
                and c['frame_pair'] == 2 and c['subject_number'] == 'singular'
                and c['near_number'] == 'singular' and c['beside_number'] == 'singular']
    by_id = {row['case_id']: row for row in rows}
    by_id[baseline[0]]['logits']['is'] = 0.0
    by_id[baseline[1]]['logits']['is'] = -1.0
    counts = mod.aggregate(cases, rows)['baseline_counts']['evaluation/singular']['present']
    assert counts == {'correct': 2, 'ties': 1, 'wrong_sign': 1, 'incorrect_including_ties': 2}
