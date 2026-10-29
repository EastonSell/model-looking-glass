"""Behavioral tests using only fabricated names and tokenizer inputs."""
import hashlib
import importlib
import importlib.util
import os
import re

import pytest


def subject():
    name = 'looking_glass.flan_t5_heldout_custody'
    assert importlib.util.find_spec(name) is not None, 'custody implementation is missing'
    return importlib.import_module(name)


def test_canonical_pool_commitment_binds_domain_salt_and_utf8_json():
    custody = subject()
    payload = {'z': 'ä', 'pairs': [3, 1]}
    expected = b'{"pairs":[3,1],"z":"\\u00e4"}'
    salt = bytes(range(32))
    assert custody.canonical_json(payload) == expected
    assert custody.commitment('pool', payload, salt) == hashlib.sha256(
        b'mlg-flant5-heldout-pool-v1\x00' + salt + expected).hexdigest()
    assert custody.commitment('cases', payload, salt) != custody.commitment('pool', payload, salt)
    with pytest.raises(ValueError):
        custody.commitment('pool', payload, salt[:-1])
    with pytest.raises(ValueError):
        custody.canonical_json({'value': float('nan')})


def test_card_contains_only_salted_commitment_and_public_identities():
    custody = subject()
    payload = {'schema': 'mlg-flant5-heldout-pool-v1', 'source_commit': 'a' * 40,
               'protocol_sha256': 'b' * 64, 'asset_manifest_sha256': 'c' * 64,
               'tokenizer_revision': 'd' * 40, 'private': 'fabricated-secret'}
    card = custody.commitment_card('pool', payload, bytes(range(32)))
    assert set(card) == {'schema', 'kind', 'commitment', 'source_commit', 'protocol_sha256',
                         'asset_manifest_sha256', 'tokenizer_revision'}
    assert card['kind'] == 'pool' and card['source_commit'] == 'a' * 40
    assert 'fabricated-secret' not in custody.canonical_json(card).decode()
    assert card['commitment'] == custody.commitment('pool', payload, bytes(range(32)))
    cases = dict(payload, schema='mlg-flant5-heldout-cases-v1', pool_commitment=card['commitment'])
    case_card = custody.commitment_card('cases', cases, b'x' * 32)
    assert case_card['pool_commitment'] == card['commitment']


def test_noncanonical_or_duplicate_key_record_is_not_accepted():
    custody = subject()
    assert custody.parse_canonical(b'{"a":1}') == {'a': 1}
    for raw in (b'{"a": 1}', b'{"a":1,"a":1}', b'{"a":NaN}', b'[]', b'\xef\xbb\xbf{"a":1}'):
        with pytest.raises(ValueError):
            custody.parse_canonical(raw)


def test_exposure_includes_prior_multitoken_and_normalized_words():
    custody = subject()
    first = {'pairs': [{'sources': [{'text': 'Say Straße.', 'input_ids': [40, 51, 1],
                                    'offsets': [[0, 3], [3, 10], [0, 0]], 'word_position': 1}],
                        'targets': [{'text': 'MÜTZE', 'input_ids': [81, 82, 1]}]}]}
    screen = {'candidates': [{'cases': [{'source': [{'word': 'cafe\u0301',
                                                    'word_positions': [1, 2],
                                                    'input_ids': [40, 61, 62, 1]}],
                                        'targets': [{'text': 'Baum', 'input_ids': [91, 1],
                                                     'special_tokens_mask': [0, 1]}]}]}]}
    exposure = custody.exposure_from_documents([first, screen], special_ids={0, 1, 2})
    assert exposure['source_words'] == {'strasse', 'café'}
    assert exposure['target_words'] == {'mütze', 'baum'}
    assert exposure['word_token_ids'] == {51, 61, 62}
    assert exposure['target_token_ids'] == {81, 82, 91}


def fabricated_pool():
    custody = subject()
    refs = [{'path': name, 'commit': 'a' * 40, 'sha256': 'f' * 64} for name in custody.EXPOSURE_PATHS]
    return {'schema': 'mlg-flant5-heldout-pool-v1', 'source_commit': 'a' * 40,
            'protocol_sha256': 'b' * 64, 'asset_manifest_sha256': 'c' * 64,
            'tokenizer_revision': 'd' * 40, 'exposure_inputs': refs,
            'templates': [{'template_id': 'template0', 'prefix': 'Say ', 'suffix': '.'},
                          {'template_id': 'template1', 'prefix': 'Name ', 'suffix': '?'}],
            'pairs': [{'candidate_index': i, 'source_words': [f'a{i}', f'b{i}'],
                       'target_words': [f'x{i}', f'y{i}']} for i in range(24)]}


def empty_exposure(pool):
    return {'inputs': pool['exposure_inputs'], 'source_words': set(), 'target_words': set(),
            'word_token_ids': set(), 'target_token_ids': set()}


def test_pool_rejects_normalized_duplicates_and_known_text_without_leaking_values():
    custody = subject()
    pool = fabricated_pool()
    exposure = empty_exposure(pool)
    custody.validate_pool(pool, exposure)
    pool['pairs'][0]['source_words'][0] = 'café'
    pool['pairs'][1]['source_words'][0] = 'cafe\u0301'
    with pytest.raises(ValueError) as duplicate:
        custody.validate_pool(pool, exposure)
    assert 'café' not in str(duplicate.value)
    pool = fabricated_pool()
    exposure = empty_exposure(pool)
    exposure['target_words'].add('x3')
    with pytest.raises(ValueError):
        custody.validate_pool(pool, exposure)
    pool = fabricated_pool()
    exposure = empty_exposure(pool)
    exposure['target_words'].add('a3')
    with pytest.raises(ValueError):
        custody.validate_pool(pool, exposure)


def test_pool_rejects_prior_template_and_identity_drift():
    custody = subject()
    pool = fabricated_pool()
    exposure = empty_exposure(pool)
    pool['templates'][0] = {'template_id': 'template0', 'prefix': 'Translate English to German: ', 'suffix': ''}
    with pytest.raises(ValueError):
        custody.validate_pool(pool, exposure)
    pool = fabricated_pool()
    exposure = empty_exposure(pool)
    pool['exposure_inputs'][0]['sha256'] = 'e' * 64
    exposure['inputs'] = [{'path': name, 'commit': 'a' * 40, 'sha256': 'f' * 64}
                          for name in custody.EXPOSURE_PATHS]
    with pytest.raises(ValueError):
        custody.validate_pool(pool, exposure)


def test_inventory_includes_private_logs_observations_and_all_nested_files(tmp_path):
    custody = subject()
    (tmp_path / 'observations').mkdir()
    (tmp_path / 'observations' / 'four.json').write_bytes(b'1234')
    (tmp_path / 'stderr.bin').write_bytes(b'abc')
    (tmp_path / 'stdout.bin').write_bytes(b'')
    result = custody.inventory_custody(tmp_path)
    assert result == {'bytes': 7, 'files': 3, 'entries': [
        {'path': 'observations/four.json', 'size': 4},
        {'path': 'stderr.bin', 'size': 3}, {'path': 'stdout.bin', 'size': 0}]}


def test_inventory_rejects_oversized_or_linked_files_without_private_path(tmp_path):
    custody = subject()
    secret = tmp_path / 'fabricated-private-name'
    secret.write_bytes(b'x' * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError) as error:
        custody.inventory_custody(tmp_path)
    assert 'fabricated-private-name' not in str(error.value)
    secret.unlink()
    try:
        os.symlink(tmp_path.parent, tmp_path / 'link', target_is_directory=True)
    except OSError:
        pytest.skip('Creating symbolic links is unavailable to this account')
    with pytest.raises(ValueError):
        custody.inventory_custody(tmp_path)


class T5TokenizerFast:
    """Deterministic tokenizer double; no pretrained data or weights."""
    pad_token_id, eos_token_id, unk_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2]

    def __init__(self, split=None):
        self.split = split
        self.ids = {}

    def __call__(self, text, **kwargs):
        chunks = []
        for match in re.finditer(r'[A-Za-z0-9]+|[^\w\s]', text):
            start, end = match.span()
            if match.group() == self.split:
                middle = start + 1
                chunks.extend([(start, middle), (middle, end)])
            else:
                chunks.append((start, end))
        ids, offsets = [], []
        for start, end in chunks:
            token = text[start:end]
            if token not in self.ids:
                self.ids[token] = len(self.ids) + 10
            ids.append(self.ids[token])
            offsets.append([start, end])
        return {'input_ids': ids + [1], 'attention_mask': [1] * (len(ids) + 1),
                'special_tokens_mask': [0] * len(ids) + [1],
                'offset_mapping': offsets + [[0, 0]]}

    def convert_ids_to_tokens(self, ids):
        reverse = {value: key for key, value in self.ids.items()}
        return ['</s>' if value == 1 else reverse[value] for value in ids]


def test_screen_checks_all_candidates_and_selects_first_six_without_replacement():
    custody = subject()
    pool = fabricated_pool()
    screen = custody.screen_pool(pool, T5TokenizerFast(split='a0'), empty_exposure(pool), '9' * 64)
    assert screen['status'] == 'complete'
    assert len(screen['candidates']) == 24
    assert all(len(candidate['cases']) == 2 for candidate in screen['candidates'])
    assert not screen['candidates'][0]['eligible']
    assert screen['selected_candidate_indices'] == [1, 2, 3, 4, 5, 6]
    selected = custody.selected_cases(screen)
    assert [case['case_id'] for case in selected] == [
        f'pair{pair}-template{template}' for pair in range(6) for template in range(2)]
    assert selected[0]['candidate_index'] == 1
    assert selected[0]['word_position'] == 1
    assert selected[0]['instruction_position'] == 0
    assert selected[0]['context_position'] == 2
    assert selected[0]['neutral']['text'] == 'Say .'


def test_screen_retains_token_exclusions_and_insufficient_selection():
    custody = subject()
    pool = fabricated_pool()
    tokenizer = T5TokenizerFast()
    exposure = empty_exposure(pool)
    # Populate IDs without consulting a model or computing any scores.
    for pair in pool['pairs'][:19]:
        tokenizer(pair['source_words'][0])
    exposure['word_token_ids'] = {tokenizer.ids[pair['source_words'][0]] for pair in pool['pairs'][:19]}
    screen = custody.screen_pool(pool, tokenizer, exposure, '8' * 64)
    assert screen['status'] == 'selection-incomplete'
    assert screen['selected_candidate_indices'] == [19, 20, 21, 22, 23]
    assert 'word-token-exposed' in screen['candidates'][0]['cases'][0]['eligibility_reasons']
    with pytest.raises(ValueError):
        custody.selected_cases(screen)


def test_screen_finds_word_at_insertion_not_its_earlier_occurrence():
    custody = subject()
    pool = fabricated_pool()
    pool['templates'][0]['prefix'] = 'German word: '
    pool['pairs'][0]['source_words'] = ['man', 'cat']
    screen = custody.screen_pool(pool, T5TokenizerFast(), empty_exposure(pool), '7' * 64)
    case = screen['candidates'][0]['cases'][0]
    assert case['alignment_eligible']
    assert case['word_position'] == 3
    assert case['source'][0]['word_positions'] == [3]
    assert case['instruction_position'] == 0 and case['context_position'] == 4


def pool_observation(custody, card):
    return {'schema': 'mlg-flant5-heldout-card-observation-v1', 'kind': 'pool',
            'commit': 'e' * 40,
            'commit_url': 'https://github.com/EastonSell/model-looking-glass/commit/' + 'e' * 40,
            'card_path': 'experiments/fabricated/pool-commitment.json',
            'card_sha256': hashlib.sha256(custody.canonical_json(card)).hexdigest(),
            'source_commit': card['source_commit'], 'observer': 'fabricated reviewer',
            'observed_utc': 'fabricated observation'}


def verdicts(screen):
    custody = subject()
    return [{'case_id': case['case_id'], 'semantic_eligible': i != 1,
             'semantic_reasons': [] if i != 1 else ['fabricated grammar issue']}
            for i, case in enumerate(custody.selected_cases(screen))]


def test_sealed_bundle_replays_all_candidates_and_private_semantic_exclusions(tmp_path, monkeypatch):
    custody = subject()
    pool = fabricated_pool()
    exposure, tokenizer = empty_exposure(pool), T5TokenizerFast(split='a0')
    salts = iter((b'p' * 32, b'c' * 32))
    monkeypatch.setattr(custody, '_new_salt', lambda: next(salts))
    root = tmp_path / 'custody'
    pool_card = custody.create_pool_attempt(root, pool, exposure)
    with pytest.raises(FileExistsError):
        custody.create_pool_attempt(root, pool, exposure)
    screen = custody.screen_attempt(root, pool_card, pool_observation(custody, pool_card), tokenizer, exposure)
    case_card = custody.seal_cases_attempt(root, pool_card, screen, verdicts(screen))
    result = custody.verify_bundle(root, pool_card, case_card, tokenizer, exposure)
    assert len(result['cases']) == 12
    assert result['cases'][1]['semantic_eligible'] is False
    assert result['cases'][1]['semantic_reasons'] == ['fabricated grammar issue']
    assert result['cases'][0]['candidate_index'] == 1
    assert result['custody_bytes'] == custody.inventory_custody(root)['bytes']
    assert b'fabricated grammar issue' not in (root / 'case-commitment.json').read_bytes()
    with pytest.raises(FileExistsError):
        custody.seal_cases_attempt(root, pool_card, screen, verdicts(screen))
    raw = (root / 'cases-payload.json').read_bytes()
    (root / 'cases-payload.json').write_bytes(raw.replace(b'fabricated grammar issue', b'fabricated altered issue'))
    with pytest.raises(ValueError):
        custody.verify_bundle(root, pool_card, case_card, tokenizer, exposure)


def test_screen_failure_is_retained_and_does_not_authorize_retry(tmp_path, monkeypatch):
    custody = subject()
    pool = fabricated_pool()
    exposure = empty_exposure(pool)
    monkeypatch.setattr(custody, '_new_salt', lambda: b'p' * 32)
    root = tmp_path / 'custody'
    card = custody.create_pool_attempt(root, pool, exposure)
    monkeypatch.setattr(custody, 'screen_pool', lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        custody.screen_attempt(root, card, pool_observation(custody, card), T5TokenizerFast(), exposure)
    failure = custody.parse_canonical((root / 'screen-failed.json').read_bytes())
    assert failure == {'schema': 'mlg-flant5-heldout-custody-failure-v1',
                       'phase': 'screen', 'status': 'incomplete', 'error_type': 'KeyboardInterrupt'}
    with pytest.raises(FileExistsError):
        custody.screen_attempt(root, card, pool_observation(custody, card), T5TokenizerFast(), exposure)


def test_resigned_tampered_screen_is_rejected_before_model_work(tmp_path, monkeypatch):
    custody = subject()
    pool, tokenizer = fabricated_pool(), T5TokenizerFast()
    exposure = empty_exposure(pool)
    salts = iter((b'p' * 32, b'c' * 32))
    monkeypatch.setattr(custody, '_new_salt', lambda: next(salts))
    root = tmp_path / 'custody'
    pool_card = custody.create_pool_attempt(root, pool, exposure)
    screen = custody.screen_attempt(root, pool_card, pool_observation(custody, pool_card), tokenizer, exposure)
    case_card = custody.seal_cases_attempt(root, pool_card, screen, verdicts(screen))
    payload = custody.parse_canonical((root / 'cases-payload.json').read_bytes())
    payload['screen']['candidates'][0]['cases'][0]['source'][0]['input_ids'][0] += 1
    (root / 'cases-payload.json').write_bytes(custody.canonical_json(payload))
    forged = custody.commitment_card('cases', payload, b'c' * 32)
    (root / 'case-commitment.json').write_bytes(custody.canonical_json(forged))
    assert forged != case_card
    with pytest.raises(ValueError, match='screen differs'):
        custody.verify_bundle(root, pool_card, forged, tokenizer, exposure)


def test_semantic_review_failure_is_retained_without_case_salt_or_retry(tmp_path, monkeypatch):
    custody = subject()
    pool, tokenizer = fabricated_pool(), T5TokenizerFast()
    exposure = empty_exposure(pool)
    monkeypatch.setattr(custody, '_new_salt', lambda: b'p' * 32)
    root = tmp_path / 'custody'
    card = custody.create_pool_attempt(root, pool, exposure)
    screen = custody.screen_attempt(root, card, pool_observation(custody, card), tokenizer, exposure)
    with pytest.raises(ValueError, match='semantic review is incomplete'):
        custody.seal_cases_attempt(root, card, screen, verdicts(screen)[:-1])
    assert not (root / 'case-salt.bin').exists()
    assert (root / 'cases-failed.json').exists()
    with pytest.raises(FileExistsError):
        custody.seal_cases_attempt(root, card, screen, verdicts(screen))


def test_private_write_preserves_cancellation_across_close_failure(tmp_path, monkeypatch):
    custody = subject()
    real_close = os.close

    def close_then_error(fd):
        real_close(fd)
        raise OSError('fabricated close error')

    with monkeypatch.context() as patch:
        patch.setattr(custody.os, 'write', lambda fd, raw: (_ for _ in ()).throw(KeyboardInterrupt()))
        patch.setattr(custody.os, 'close', close_then_error)
        with pytest.raises(KeyboardInterrupt):
            custody._write_new(tmp_path, 'interrupted.bin', b'synthetic')


def test_private_write_promotes_new_cancellation_during_close(tmp_path, monkeypatch):
    custody = subject()
    real_close = os.close

    def close_then_cancel(fd):
        real_close(fd)
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(custody.os, 'write', lambda fd, raw: (_ for _ in ()).throw(OSError('fabricated')))
        patch.setattr(custody.os, 'close', close_then_cancel)
        with pytest.raises(KeyboardInterrupt):
            custody._write_new(tmp_path, 'cancelled.bin', b'synthetic')
