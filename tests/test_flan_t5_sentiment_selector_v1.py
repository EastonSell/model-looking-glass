"""Fabricated-only tests for the prospective FLAN sentiment selector."""
import copy
import hashlib
import json
import re
import subprocess

import pytest

from looking_glass import flan_t5_sentiment_selector_v1 as selector


class FakeTokenizer:
    all_special_ids = [0, 1, 2]
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r'[A-Za-z]+|[^\w\s]', text))
        ids, offsets = [], []
        for match in matches:
            word = match.group()
            ids.append(3 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], 'big') % 30000)
            start = match.start()
            if start and text[start - 1] == ' ':
                start -= 1
            offsets.append([start, match.end()])
        return {'input_ids': ids + [1], 'attention_mask': [1] * (len(ids) + 1),
                'special_tokens_mask': [0] * len(ids) + [1], 'offset_mapping': offsets + [[0, 0]]}


def pool():
    units = []
    for i, letter in enumerate('abcdefghijkl'):
        units.append({'id': f'u{i:02d}', 'subject': f'subject{letter}',
                      'adjectives': {'P1': f'brava{letter}', 'N1': f'grima{letter}',
                                     'P2': f'cheera{letter}', 'N2': f'dreada{letter}'},
                      'semantic_ok': i != 1, 'semantic_reason': 'ambiguous' if i == 1 else ''})
    return {'schema': 'flan-sentiment-private-pool-v1',
            'protocol_sha256': selector.PROTOCOL_SHA256,
            'runner_revision': 'a' * 40,
            'asset_manifest_sha256': selector.ASSET_MANIFEST_SHA256,
            'exposure_sources': [{'path': 'known/input.json', 'revision': 'b' * 40, 'sha256': 'c' * 64}],
            'units': units}


def source(rows=(), words=()):
    return [{'descriptor': pool()['exposure_sources'][0], 'rows': [list(x) for x in rows],
             'words': list(words)}]


def test_pool_commitment_canonicalization_and_exclusions():
    value = pool()
    selector.validate_pool(value, source(words=['oldword']))
    digest = selector.pool_commitment(value, bytes(range(32)))
    assert digest == selector.pool_commitment(copy.deepcopy(value), bytes(range(32)))
    assert digest != selector.pool_commitment(value, bytes(range(1, 33)))
    duplicate = copy.deepcopy(value)
    duplicate['units'][2]['subject'] = duplicate['units'][0]['subject']
    with pytest.raises(ValueError, match='distinct'):
        selector.validate_pool(duplicate, source())
    with pytest.raises(ValueError, match='exposed'):
        selector.validate_pool(value, source(words=[value['units'][0]['subject'].upper()]))
    with pytest.raises(ValueError, match='salt'):
        selector.pool_commitment(value, b'bad')


def test_bidirectional_prefix_ignores_exactly_one_final_eos():
    assert selector.prefix_conflict([3, 4, 1], [3, 4, 5, 1])
    assert selector.prefix_conflict([3, 4, 5, 1], [3, 4, 1])
    assert selector.prefix_conflict([3, 4, 1], [3, 4, 1])
    assert not selector.prefix_conflict([3, 4, 1], [3, 5, 1])
    for bad in ([3, 4], [3, 1, 1], [1], [True, 1]):
        with pytest.raises(ValueError):
            selector.prefix_conflict(bad, [3, 4, 1])


def test_all_twelve_are_screened_and_first_four_eligible_are_frozen():
    value, tok = pool(), FakeTokenizer()
    first = selector.case_text(value['units'][0], 0, 'P1')
    exposed = tok(first)['input_ids']
    result = selector.select(value, tok, source(rows=[exposed]), 'f' * 64)
    assert len(result['unit_screens']) == 12
    assert result['selected_units'] == ['u02', 'u03', 'u04', 'u05']
    assert not result['unit_screens'][0]['instructions'][0]['eligible']
    assert result['unit_screens'][0]['instructions'][0]['source_comparisons'][0]['conflict']
    assert result['unit_screens'][1]['semantic_ok'] is False
    instruction = result['unit_screens'][2]['instructions'][0]
    assert instruction['eligible'] and len(instruction['cases']) == 4
    positions = [case['adjective_index'] for case in instruction['cases']]
    assert len(set(positions)) == 1 and positions[0] is not None
    for case in instruction['cases']:
        start, end = case['offset_mapping'][case['adjective_index']]
        assert case['text'][start] == ' ' and case['text'][end - 1].isalpha()
    assert result['donor_mapping'][0]['opposite'] == {'P1': 'N1', 'N1': 'P1', 'P2': 'N2', 'N2': 'P2'}
    assert result['donor_mapping'][0]['same'] == {'P1': 'P2', 'P2': 'P1', 'N1': 'N2', 'N2': 'N1'}


def test_selection_commitment_changes_for_any_edited_case():
    result = selector.select(pool(), FakeTokenizer(), source(), 'f' * 64)
    first = selector.selection_commitment(result, bytes(range(32)))
    changed = copy.deepcopy(result)
    changed['unit_screens'][0]['instructions'][0]['cases'][0]['input_ids'][0] += 1
    assert selector.selection_commitment(changed, bytes(range(32))) != first


def test_prefix_check_executes_every_recorded_comparison_after_a_conflict(monkeypatch):
    value, tok = pool(), FakeTokenizer()
    first = tok(selector.case_text(value['units'][0], 0, 'P1'))['input_ids']
    original = selector.prefix_conflict
    counter = []
    def counted(left, right):
        counter.append((left, right))
        return original(left, right)
    monkeypatch.setattr(selector, 'prefix_conflict', counted)
    group = selector._group(value['units'][0], 0, tok, source(rows=[first, [999, 1]]))
    assert group['source_comparisons'][0] == {'source_index': 0, 'comparisons': 8, 'conflict': True}
    assert len(counter) == 8


def test_subject_position_is_based_on_the_fixed_frame_not_the_last_substring():
    value = pool()
    value['units'][0]['subject'] = 'bra'
    group = selector._group(value['units'][0], 0, FakeTokenizer(), source())
    assert group['eligible']
    for case in group['cases']:
        start, end = case['offset_mapping'][case['subject_index']]
        assert case['text'][start:end].strip() == 'bra'


def test_offsets_and_token_differences_must_be_complete_and_at_one_position():
    class BadOffsets(FakeTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            for i, (start, end) in enumerate(result['offset_mapping']):
                if text[start:end].strip().startswith('brava'):
                    result['offset_mapping'][i] = [start, end - 1]
            return result
    result = selector.select(pool(), BadOffsets(), source(), 'f' * 64)
    assert result['selected_units'] == []
    assert all(not group['eligible'] for unit in result['unit_screens'] for group in unit['instructions'])


def test_fake_git_bound_two_stage_selection_and_readback(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo).strip().decode('ascii')
    git('init', '-q')
    (repo / 'known').mkdir()
    first = {'text': 'A legacy example', 'input_ids': [500, 1], 'source_words': ['für', 'old']}
    second = {'case': {'text': 'A past example', 'input_ids': [501, 1]}}
    raw_json = json.dumps(first, ensure_ascii=False).encode()
    raw_jsonl = (json.dumps(second) + '\n' + json.dumps({'text': 'other', 'input_ids': [502, 1]}) + '\n').encode()
    (repo / 'known/input.json').write_bytes(raw_json)
    (repo / 'known/events.jsonl').write_bytes(raw_jsonl)
    git('add', 'known')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'known sources')
    revision = git('rev-parse', 'HEAD')
    descriptors = [{'path': path, 'revision': revision, 'sha256': hashlib.sha256(data).hexdigest()}
                   for path, data in (('known/input.json', raw_json), ('known/events.jsonl', raw_jsonl))]
    monkeypatch.setattr(selector, 'REQUIRED_SOURCE_PATHS', frozenset(d['path'] for d in descriptors))
    monkeypatch.setattr(selector, 'REQUIRED_SOURCE_HASHES', {d['path']: d['sha256'] for d in descriptors})
    exposed = selector.load_exposures(repo, descriptors)
    assert len(exposed[0]['rows']) == 1 and len(exposed[1]['rows']) == 2
    assert 'für' in exposed[0]['words'] and 'r' not in exposed[0]['words']
    value = pool()
    value['exposure_sources'] = descriptors
    pool_path = tmp_path / 'pool.json'
    pool_path.write_bytes(selector.canonical(value))
    pool_salt, card = tmp_path / 'pool.salt', tmp_path / 'pool-card.json'
    selector.main(['commit-pool', '--repo', str(repo), '--pool', str(pool_path),
                   '--pool-salt', str(pool_salt), '--pool-card', str(card)])
    assert len(pool_salt.read_bytes()) == 32
    (repo / 'pool-card.json').write_bytes(card.read_bytes())
    git('add', 'pool-card.json')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'publish fake card')
    card_revision = git('rev-parse', 'HEAD')
    observation = tmp_path / 'observed.json'
    observation.write_bytes(selector.canonical({'schema': 'flan-sentiment-pool-observation-v1',
                            'path': 'pool-card.json', 'revision': card_revision,
                            'sha256': hashlib.sha256(card.read_bytes()).hexdigest()}))
    from looking_glass import flan_t5_feasibility
    import transformers
    monkeypatch.setattr(flan_t5_feasibility, 'verify_assets', lambda path: None)
    class T5TokenizerFast(FakeTokenizer):
        pass
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: T5TokenizerFast())
    selected, salt, selection_card = tmp_path / 'selected.json', tmp_path / 'selected.salt', tmp_path / 'selected-card.json'
    args = ['select', '--repo', str(repo), '--pool', str(pool_path), '--pool-salt', str(pool_salt),
            '--pool-card', str(card), '--pool-observation', str(observation), '--asset-dir', str(tmp_path),
            '--selection', str(selected), '--selection-salt', str(salt), '--selection-card', str(selection_card)]
    selector.main(args)
    saved = json.loads(selected.read_bytes())
    assert saved['selected_units'] == ['u00', 'u02', 'u03', 'u04']
    assert json.loads(selection_card.read_bytes())['commitment'] == selector.selection_commitment(saved, salt.read_bytes())

    bad_observation = json.loads(observation.read_bytes())
    bad_observation['sha256'] = '0' * 64
    observation.write_bytes(selector.canonical(bad_observation))
    with pytest.raises(ValueError, match='observation digest'):
        selector.main(args)
