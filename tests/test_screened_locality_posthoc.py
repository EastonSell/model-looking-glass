"""Model-free checks for the screened-edit post-hoc analysis."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools import screened_locality_posthoc as analysis


def test_probability_metrics_have_the_expected_direction_and_units():
    p = np.array([[0.5, 0.25, 0.25]])
    q = np.array([[0.25, 0.5, 0.25]])
    kl, tv, js = analysis.distribution_changes(np.log(p), np.log(q))
    assert kl[0] == pytest.approx(math.log(2) / 4)
    assert tv[0] == pytest.approx(0.25)
    assert 0 < js[0] < kl[0]
    logp = analysis.log_probabilities(np.log(p) + 1000)
    assert np.exp(logp) == pytest.approx(p)


def test_mean_matched_logit_interpolation_does_not_match_each_target():
    shape = (8, analysis.PARIS + 1)
    original = np.zeros(shape, dtype=np.float64)
    matched = original.copy()
    matched[:, analysis.LONDON] = np.arange(1, 9)
    projected = original.copy()
    projected[-1, analysis.LONDON] = 18
    alpha, interpolated = analysis.matched_logit_interpolation(original, projected, matched)
    assert alpha == pytest.approx(0.5)
    assert np.mean(interpolated[:, analysis.LONDON]) == pytest.approx(
        np.mean(projected[:, analysis.LONDON]))
    assert not np.array_equal(interpolated[:, analysis.LONDON], projected[:, analysis.LONDON])
    with pytest.raises(ValueError, match='positive'):
        analysis.matched_logit_interpolation(original, projected, original)


def test_pinned_read_rejects_changed_bytes(tmp_path):
    path = tmp_path / 'data'
    raw = b'original'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert analysis.pinned(path, digest, len(raw)) == raw
    path.write_bytes(b'modified')
    with pytest.raises(ValueError, match='digest'):
        analysis.pinned(path, digest, len(raw))


def test_bytelevel_vocabulary_keeps_spaces_and_ids():
    vocab = {f'a{i}': i for i in range(analysis.VOCABULARY)}
    del vocab['a100']
    vocab['\u0120London'] = 100
    tokens = analysis.bytelevel_tokens(json.dumps({'model': {'vocab': vocab}}).encode())
    assert tokens[100] == ' London'
    assert tokens[99] == 'a99'
    assert len(tokens) == analysis.VOCABULARY


def test_published_rows_reproduce_the_known_locality_anchor():
    root = Path(__file__).resolve().parents[1]
    cases, matrices, provenance = analysis.load_closed_data(root)
    assert len(cases) == 20 and len(matrices) == 7
    assert provenance['phase_sha256'] == analysis.PHASE_SHA
    logp = analysis.log_probabilities(matrices['original'])
    logq = analysis.log_probabilities(matrices['projected'])
    kl, tv, _ = analysis.distribution_changes(logp, logq)
    assert np.mean(kl[8:]) == pytest.approx(0.0002767190668740136, abs=1e-14)
    assert np.mean(tv[8:]) == pytest.approx(0.005869987059834576, abs=1e-14)


def test_locality_table_agrees_with_unrounded_data():
    directory = (Path(__file__).resolve().parents[1] / analysis.RESULT / 'posthoc-locality')
    with (directory / 'locality.csv').open(encoding='utf-8', newline='') as stream:
        projected = {row['prompt']: row for row in csv.DictReader(stream) if row['condition'] == 'projected'}
    with (directory / 'interpolated-locality.csv').open(encoding='utf-8', newline='') as stream:
        sensitivity = {row['prompt']: row for row in csv.DictReader(stream)}
    table = []
    for line in (directory / 'README.md').read_text(encoding='utf-8').splitlines():
        if line.startswith('| ') and line.endswith(' |'):
            cells = [item.strip() for item in line.strip('|').split('|')]
            if len(cells) == 8 and cells[1] in projected:
                table.append(cells)
    assert len(table) == 12
    for expected, prompt, baseline, delta, tv, projected_kl, matched_kl, interpolated_kl in table:
        row, contrast = projected[prompt], sensitivity[prompt]
        assert expected == row['expected_text'].strip()
        assert float(baseline.rstrip('%')) == pytest.approx(100 * float(row['expected_probability_before']), abs=0.000051)
        assert float(delta) == pytest.approx(float(row['expected_probability_change_points']), abs=0.00000051)
        assert float(tv.rstrip('%')) == pytest.approx(100 * float(row['total_variation']), abs=0.00051)
        for displayed, key in ((projected_kl, 'projected_kl'), (matched_kl, 'matched_kl'),
                               (interpolated_kl, 'interpolated_logit_kl')):
            assert float(displayed) == pytest.approx(1000 * float(contrast[key]), abs=0.000051)
