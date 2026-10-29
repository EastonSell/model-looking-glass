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
