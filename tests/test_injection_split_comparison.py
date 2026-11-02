"""File-bound split comparison tests use fabricated commitments and rows only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from looking_glass import injection_split_custody as custody


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


here = Path(__file__).parent
comparison = _load(here.parent / 'tools' / 'injection_split_comparison.py',
                   '_fabricated_split_comparison')
fixtures = _load(here / 'test_injection_split_custody.py', '_fabricated_split_fixtures')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def pair(tmp_path):
    fixture = fixtures.InjectionCustodyTests('test_claim_precedes_one_draw_and_private_commitment_verifies')
    fixture.setUp()
    try:
        injection = fixture.injection_split(fixture.prepare())
        natural_collision = fixture.natural_split(fixtures.SPLIT_NONCE)
        comparator_sha = digest(Path(custody.__file__).read_bytes())
        for counter in range(1, 256):
            natural_clear = fixture.natural_split(bytes([counter]) * 32)
            trial = custody.compare_committed_splits(
                natural_clear, injection, comparator_source_sha256=comparator_sha,
                publication_check=lambda name, raw: True)
            if json.loads(trial)['status'] == 'clear':
                break
        else:
            raise AssertionError('fabricated pool did not produce a disjoint split')

        def files(track, split):
            root = tmp_path / track
            root.mkdir()
            result = {}
            for key, raw in (('freeze', split.freeze_bytes), ('commitment', split.commitment_bytes),
                             ('inventory', split.inventory_bytes), ('record', split.record_bytes),
                             ('nonce', split.nonce)):
                path = root / (key + '.bin')
                path.write_bytes(raw)
                result[key] = path
            return result

        def build(natural):
            n = files('natural', natural)
            i = files('injection', injection)
            site_nonce = tmp_path / 'site-nonce.bin'
            site_record = tmp_path / 'site-record.bin'
            site_nonce.write_bytes(injection.site_nonce)
            site_record.write_bytes(injection.site_record_bytes)
            expected = {
                'natural_freeze_sha256': natural.freeze_sha256,
                'injection_freeze_sha256': injection.freeze_sha256,
                'natural_commitment_sha256': digest(natural.commitment_bytes),
                'injection_commitment_sha256': digest(injection.commitment_bytes),
                'comparator_source_sha256': comparator_sha,
            }
            return dict(natural=n, injection=i, site_nonce=site_nonce, site_record=site_record,
                        site_commitment=injection.site_commitment, expected=expected,
                        output=tmp_path / 'receipt.json')

        yield SimpleNamespace(build=build, clear=natural_clear, collision=natural_collision,
                              injection=injection)
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize('status', ['clear', 'collision'])
def test_retained_receipt_reports_only_committed_comparison(pair, status):
    args = pair.build(getattr(pair, status))
    observed = []
    result = comparison.compare_files(custody, **args,
        publication_check=lambda name, raw: observed.append(name) or True)
    raw = args['output'].read_bytes()
    receipt = json.loads(raw)
    assert observed == ['natural_freeze', 'natural_commitment', 'injection_freeze', 'injection_commitment']
    assert receipt['status'] == status
    assert receipt['natural_case_count'] == receipt['injection_case_count'] == 16
    assert result == {'status': status, 'receipt_sha256': digest(raw)}
    for private in (b'input_ids', b'nonce', b'alpha', pair.injection.site_nonce):
        assert private not in raw


def test_unobserved_publication_stops_before_private_files_are_opened(pair):
    args = pair.build(pair.clear)
    args['natural']['nonce'].unlink()
    with pytest.raises(comparison.ComparisonError, match='publication'):
        comparison.compare_files(custody, **args, publication_check=lambda name, raw: False)
    assert not args['output'].exists()


def test_existing_receipt_is_never_overwritten_or_used_to_open_private_files(pair):
    args = pair.build(pair.clear)
    args['output'].write_bytes(b'existing evidence')
    args['natural']['nonce'].unlink()
    with pytest.raises(comparison.ComparisonError, match='receipt'):
        comparison.compare_files(custody, **args, publication_check=lambda name, raw: True)
    assert args['output'].read_bytes() == b'existing evidence'


def test_corrupt_committed_private_record_cannot_produce_a_clear_receipt(pair):
    args = pair.build(pair.clear)
    args['natural']['nonce'].write_bytes(b'x' * 32)
    with pytest.raises(custody.CustodyError):
        comparison.compare_files(custody, **args, publication_check=lambda name, raw: True)
    assert args['output'].read_bytes() == b''


def test_partial_receipt_write_is_retained_but_not_reported_as_success(pair, monkeypatch):
    args = pair.build(pair.clear)
    original_open = Path.open

    class ShortWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self.stream.__exit__(*exc)

        def write(self, raw):
            return self.stream.write(raw[:len(raw) // 2])

        def flush(self):
            return self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    def opened(path, mode='r', *pos, **kw):
        stream = original_open(path, mode, *pos, **kw)
        return ShortWriter(stream) if path == args['output'] and mode == 'xb' else stream

    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(comparison.ComparisonError, match='receipt'):
        comparison.compare_files(custody, **args, publication_check=lambda name, raw: True)
    assert args['output'].is_file()
    assert 0 < args['output'].stat().st_size < 8192


def test_unexpected_public_commitment_field_stops_before_private_read(pair):
    args = pair.build(pair.clear)
    value = json.loads(args['natural']['commitment'].read_bytes())
    value['site'] = 'fabricated private field'
    raw = fixtures.canonical(value)
    args['natural']['commitment'].write_bytes(raw)
    args['expected']['natural_commitment_sha256'] = digest(raw)
    args['natural']['nonce'].unlink()
    with pytest.raises(comparison.ComparisonError, match='commitment'):
        comparison.compare_files(custody, **args, publication_check=lambda name, raw: True)
    assert not args['output'].exists()


def test_all_source_readbacks_precede_any_project_execution(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    (root / 'looking_glass').mkdir(parents=True)
    marker = tmp_path / 'executed'
    first = ('from pathlib import Path\nPath(' + repr(str(marker)) + ').write_text("ran")\n').encode()
    second = b'VALUE = 2\n'
    (root / 'looking_glass' / 'first.py').write_bytes(first)
    (root / 'looking_glass' / 'second.py').write_bytes(second)
    monkeypatch.setattr(comparison, '_SOURCE_SPECS', (
        ('first', digest(first), 'a' * 40), ('second', digest(second), 'a' * 40)))
    with pytest.raises(comparison.ComparisonError, match='source'):
        comparison.load_sources(root, root / 'looking_glass' / 'second.py',
                                reader=lambda name, commit: first if name == 'first' else b'wrong')
    assert not marker.exists()


def test_reviewed_source_stack_loads_in_a_fresh_source_only_interpreter():
    root = here / 'fixtures' / 'agreement-split-gate-02e1c5d'
    if not root.exists():
        root = here.parent
    source = here / 'fixtures' / 'injection-split-comparison-b3cf11f' / 'injection_split_custody.py'
    script = '''
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('_split_caller', sys.argv[1])
caller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(caller)
root, custody = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
def read(name, commit):
    path = custody if name == 'injection_split_custody' else root / 'looking_glass' / (name + '.py')
    return path.read_bytes()
module = caller.load_sources(root, custody, reader=read)
assert module.__name__ == 'looking_glass.injection_split_custody'
assert 'torch' not in sys.modules
print('source-only stack loaded')
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', script,
                             str(here.parent / 'tools' / 'injection_split_comparison.py'),
                             str(root), str(source)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'source-only stack loaded'


def test_study_config_rejects_unpinned_publication_and_extra_fields(tmp_path):
    fields = {key: str(tmp_path / key) for key in
              ('source_root', 'custody_source', 'pycache_prefix', 'output', 'site_nonce', 'site_record')}
    fields.update({'schema': 'mlg-injection-split-comparison-config-v1',
                   'natural': {key: str(tmp_path / ('natural-' + key)) for key in comparison._SPLIT_KEYS},
                   'injection': {key: str(tmp_path / ('injection-' + key)) for key in comparison._SPLIT_KEYS},
                   'injection_commitment_sha256': 'a' * 64,
                   'injection_commitment_url': comparison._RAW + 'a' * 40 + '/results/commitment.json'})
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    assert comparison.parse_config(canonical(fields)) == fields
    for bad_url in (comparison._RAW + 'main/results/commitment.json',
                    comparison._RAW + 'a' * 40 + '/../private.json',
                    'https://example.com/' + 'a' * 40 + '/commitment.json'):
        changed = dict(fields, injection_commitment_url=bad_url)
        with pytest.raises(comparison.ComparisonError):
            comparison.parse_config(canonical(changed))
    with pytest.raises(comparison.ComparisonError):
        comparison.parse_config(canonical(dict(fields, unreviewed_field='value')))


def test_study_entry_pins_original_freezes_and_site_before_comparison(tmp_path, monkeypatch):
    fields = {key: str(tmp_path / key) for key in
              ('source_root', 'custody_source', 'pycache_prefix', 'output', 'site_nonce', 'site_record')}
    fields.update({'schema': 'mlg-injection-split-comparison-config-v1',
                   'natural': {key: str(tmp_path / ('natural-' + key)) for key in comparison._SPLIT_KEYS},
                   'injection': {key: str(tmp_path / ('injection-' + key)) for key in comparison._SPLIT_KEYS},
                   'injection_commitment_sha256': 'a' * 64,
                   'injection_commitment_url': comparison._RAW + 'b' * 40 + '/results/commitment.json'})
    captured = {}
    monkeypatch.setattr(comparison, 'compare_files', lambda module, **kwargs: captured.update(kwargs) or 'done')
    assert comparison.compare_study(object(), fields, reader=lambda url: b'public') == 'done'
    assert captured['expected']['natural_freeze_sha256'] == comparison._NATURAL_FREEZE
    assert captured['expected']['natural_commitment_sha256'] == comparison._NATURAL_COMMITMENT
    assert captured['expected']['injection_freeze_sha256'] == comparison._INJECTION_FREEZE
    assert captured['expected']['injection_commitment_sha256'] == 'a' * 64
    assert captured['site_commitment'] == comparison._SITE_COMMITMENT
