"""Fabricated public-inventory scans; no model or checkpoint is involved."""
import hashlib
import json

import pytest

from looking_glass import distilbert_color_preference_public_v2 as public


def test_scan_rejects_prefixes_and_ignores_only_exact_own_files(monkeypatch):
    planned = [{'input_ids': [101, i, 103, 102], 'attention_mask': [1, 1, 1, 1]}
               for i in range(1, 38)]
    case_raw, exposure_raw = b'own cases', b'own exposure'
    other = {'rows': [{'input_ids': [101, 90, 103, 102], 'attention_mask': [1, 1, 1, 1]}]}
    other_raw = json.dumps(other).encode()
    paths = {public.CASE_PATH: 'a' * 40, public.EXPOSURE_PATH: 'b' * 40,
             'experiments/bert_extra_cases_v1.json': 'c' * 40}
    contents = {public.CASE_PATH: case_raw, public.EXPOSURE_PATH: exposure_raw,
                'experiments/bert_extra_cases_v1.json': other_raw}
    monkeypatch.setattr(public, 'CASE_SHA256', hashlib.sha256(case_raw).hexdigest())
    monkeypatch.setattr(public, 'EXPOSURE_SHA256', hashlib.sha256(exposure_raw).hexdigest())
    monkeypatch.setattr(public, '_tree', lambda *_: paths)
    monkeypatch.setattr(public.publication, 'git_blob', lambda _repo, _commit, path: contents[path])
    result = public.scan('.', 'd' * 40, planned)
    assert result['row_references'] == result['unique_rows'] == 1
    assert result['pairs_compared'] == 37 and result['prefix_relations'] == 0
    other['rows'][0] = {'input_ids': [101, 1, 103, 45, 102], 'attention_mask': [1] * 5}
    contents['experiments/bert_extra_cases_v1.json'] = json.dumps(other).encode()
    with pytest.raises(ValueError, match='prefix relation'):
        public.scan('.', 'd' * 40, planned)
    contents[public.CASE_PATH] = b'altered own cases'
    with pytest.raises(ValueError, match='own input identity'):
        public.scan('.', 'd' * 40, planned)


def test_extra_context_and_suffix_variants_all_get_checked(monkeypatch):
    planned = [{'input_ids': [101, 1, 103, 102], 'attention_mask': [1] * 4}]
    clean = {'input_ids': [101, 90, 103, 102], 'attention_mask': [1] * 4}
    collision = {'input_ids': [101, 1, 103, 45, 102], 'attention_mask': [1] * 5}
    docs = ({'pairs': [{'contexts': [{'rows': {'safe': clean, 'bad': collision}}]}]},
            {'cases': [{'variants': {'terminal': clean, 'continued': collision}}]})
    for document in docs:
        raw = json.dumps(document).encode()
        monkeypatch.setattr(public.publication, 'git_blob', lambda *_: raw)
        with pytest.raises(ValueError, match='prefix relation'):
            public.additional('.', 'a' * 40, 'experiments/bert_new_cases.json', planned)


def test_unknown_format_fails_closed_and_coverage_ignores_duplicate_versions():
    with pytest.raises(ValueError, match='unknown'):
        public.rows({'unrecognized': []}, 'experiments/bert_new_cases.json')
    a = [101, 1, 103, 102]
    b = [101, 2, 103, 102]
    assert public._coverage_digest([a, b, a]) == public._coverage_digest([b, a])
    assert public._coverage_digest([a]) != public._coverage_digest([a, b])
