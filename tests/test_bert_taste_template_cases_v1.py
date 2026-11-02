"""Fabricated-only checks of the fixed case-selection rule."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

source = Path(__file__).resolve().parents[1] / 'scripts/bert_taste_template_cases_v1.py'
spec = importlib.util.spec_from_file_location('taste_cases_v1', source)
cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases)


def encode(text):
    words = text.rstrip('.').split()
    stem, ingredient = words[1], words[-1]
    stem_id = 20000 + cases.POOL.index(stem)
    template = [2044, 5815] if words[4:6] == ['after', 'adding'] else [2007, 2070]
    return [101, 1996, stem_id, 12595, 103, *template, cases.TOKENS[ingredient], 1012, 102], [1] * 10


def test_fixed_selection_order_and_crossed_pairs():
    selected = cases.select(encode, [[], [], [], []])
    assert len(selected['selected_stems']) == 6 and len(selected['rows']) == 24
    assert selected['selected_stems'] == selected['digest_order'][:6]
    for offset in range(0, 24, 4):
        rows = selected['rows'][offset:offset+4]
        assert [r['template'] for r in rows] == ['after', 'after', 'with', 'with']
        assert [r['ingredient'] for r in rows] == ['sugar', 'lemon', 'sugar', 'lemon']
        for i in (0, 2):
            assert [j for j, (a, b) in enumerate(zip(rows[i]['input_ids'], rows[i+1]['input_ids'])) if a != b] == [7]


def test_complete_prefix_collision_skips_without_reordering():
    initial = cases.select(encode, [[], [], [], []])
    old = dict(initial['rows'][0])
    old['input_ids'], old['attention_mask'] = old['input_ids'][:8], old['attention_mask'][:8]
    selected = cases.select(encode, [[old], [], [], []])
    assert selected['decisions'][0]['reason'] == 'public_prefix_collision'
    assert selected['selected_stems'] == initial['digest_order'][1:7]


def test_bad_mask_or_too_few_valid_candidates_fail():
    def invalid(text):
        ids, mask = encode(text)
        return ids, [True] * len(mask)
    with pytest.raises(ValueError, match='attention mask'):
        cases.select(invalid, [[], [], [], []])
    def misaligned(text):
        ids, mask = encode(text)
        ids.insert(2, 888)
        mask.insert(2, 1)
        return ids, mask
    with pytest.raises(ValueError, match='fewer than six'):
        cases.select(misaligned, [[], [], [], []])


def test_cross_template_stem_and_fixed_template_tokens_must_match():
    def different_stem(text):
        ids, mask = encode(text)
        if 'with some' in text:
            ids[2] += 1
        return ids, mask
    with pytest.raises(ValueError, match='fewer than six'):
        cases.select(different_stem, [[], [], [], []])
    def wrong_template(text):
        ids, mask = encode(text)
        if 'with some' in text:
            ids[6] += 1
        return ids, mask
    with pytest.raises(ValueError, match='fewer than six'):
        cases.select(wrong_template, [[], [], [], []])


def test_actual_cited_inventory_identity_and_reservation_binding():
    # This reads already-public manifests only, with no tokenizer or model.
    assert [len(rows) for rows in cases.load_priors()] == [160, 64, 8, 12]


def test_selected_source_commit_and_bytes_are_bound(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    git('init', '-q')
    git('config', 'user.name', 'Synthetic Test')
    git('config', 'user.email', 'synthetic@example.invalid')
    (tmp_path / 'selector.py').write_text('selector\n', encoding='utf-8')
    (tmp_path / 'protocol.md').write_text('protocol\n', encoding='utf-8')
    git('add', 'selector.py', 'protocol.md')
    git('commit', '-q', '-m', 'fabricated source')
    commit = git('rev-parse', 'HEAD').decode().strip()
    monkeypatch.setattr(cases, 'ROOT', tmp_path)
    monkeypatch.setattr(cases, 'SELECTION_SOURCE_PATHS', ('selector.py', 'protocol.md'))
    assert cases.selection_source(commit)['commit'] == commit
    with pytest.raises(ValueError, match='commit differs'):
        cases.selection_source('0' * 40)
    (tmp_path / 'selector.py').write_text('changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checkout or commit differs'):
        cases.selection_source(commit)
