"""Fabricated-only interoperability and corruption probes; no model or private pool."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path, expected_sha=None):
    raw = path.read_bytes()
    if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError('test reference source bytes differ')
    obj = types.ModuleType(name)
    obj.__file__ = str(path)
    obj.__package__ = name.rpartition('.')[0]
    sys.modules[name] = obj
    exec(compile(raw, str(path), 'exec', dont_inherit=True), obj.__dict__)
    return obj


reader = module('fabricated_independent_reader', ROOT / 'tools/flan_sentiment_independent_selection_v1.py')
package = types.ModuleType('fabricated_flan')
package.__path__ = []
sys.modules[package.__name__] = package
screen = types.ModuleType('fabricated_flan.flan_t5_sentiment_screen')
screen.INSTRUCTIONS = reader.INSTRUCTIONS
sys.modules[screen.__name__] = screen
source_path = ROOT / 'looking_glass/flan_t5_sentiment_selector_v1.py'
selector = module('fabricated_flan.flan_t5_sentiment_selector_v1', source_path,
                  '4f0dc5b4e875a6e2f725639b3d2f336b0d17a882ac9b601ea4ab3ab93ee4da17')


class FakeTokenizer:
    all_special_ids = [0, 1, 2]
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        for match in re.finditer(r'[A-Za-z]+|[^\w\s]', text):
            ids.append(3 + int.from_bytes(hashlib.sha256(match.group().encode()).digest()[:4], 'big') % 30000)
            start = match.start()
            if start and text[start - 1] == ' ':
                start -= 1
            offsets.append([start, match.end()])
        return {'input_ids': ids + [1], 'attention_mask': [1] * (len(ids) + 1),
                'special_tokens_mask': [0] * len(ids) + [1], 'offset_mapping': offsets + [[0, 0]]}


def fixture():
    units = []
    for number, letter in enumerate('abcdefghijkl'):
        units.append({'id': f'u{number:02}', 'subject': f'subject{letter}',
                      'adjectives': {'P1': f'brava{letter}', 'N1': f'grima{letter}',
                                     'P2': f'cheera{letter}', 'N2': f'dreada{letter}'},
                      'semantic_ok': number != 1, 'semantic_reason': 'ambiguous' if number == 1 else ''})
    descriptor = {'path': 'known/fabricated.json', 'revision': 'b' * 40, 'sha256': 'c' * 64}
    pool = {'schema': 'flan-sentiment-private-pool-v1', 'protocol_sha256': reader.PROTOCOL,
            'runner_revision': 'a' * 40, 'asset_manifest_sha256': reader.ASSET_MANIFEST,
            'exposure_sources': [descriptor], 'units': units}
    tokenizer = FakeTokenizer()
    conflict = tokenizer(selector.case_text(units[0], 0, 'P1'))['input_ids']
    exposures = [{'descriptor': descriptor, 'rows': [conflict, [999, 1]], 'words': ['prior']}]
    pool_raw = selector.canonical(pool)
    pool_salt = bytes(range(32))
    pool_hash = selector.pool_commitment(pool, pool_salt)
    pool_card = {'schema': 'flan-sentiment-pool-card-v1', 'commitment': pool_hash,
                 'protocol_sha256': reader.PROTOCOL, 'runner_revision': 'a' * 40,
                 'asset_manifest_sha256': reader.ASSET_MANIFEST}
    selection = selector.select(pool, tokenizer, exposures, pool_hash)
    return pool_raw, pool_salt, pool_card, selection, bytes(range(1, 33)), exposures, tokenizer


def encode_selection(selection, salt, pool_hash):
    record = selector.canonical(selection)
    card = {'schema': 'flan-sentiment-selection-card-v1',
            'commitment': selector.selection_commitment(selection, salt), 'pool_commitment': pool_hash,
            'protocol_sha256': reader.PROTOCOL, 'runner_revision': 'a' * 40,
            'asset_manifest_sha256': reader.ASSET_MANIFEST}
    return record, selector.canonical(card) + b'\n'


def check(data, *, selection=None, pool_raw=None, tokenizer=None):
    original_pool, pool_salt, pool_card, original_selection, selection_salt, exposures, original_tok = data
    record, card = encode_selection(original_selection if selection is None else selection,
                                    selection_salt, pool_card['commitment'])
    return reader.check_saved(original_pool if pool_raw is None else pool_raw, pool_salt,
        selector.canonical(pool_card) + b'\n', record, selection_salt, card, exposures,
        original_tok if tokenizer is None else tokenizer, 'a' * 40)


class IndependentReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()

    def test_real_selector_fake_tokenizer_interoperability_and_safe_receipt(self):
        report = check(self.data)
        self.assertEqual(report['status'], 'consistent')
        self.assertEqual((report['unit_count'], report['token_rows'], report['selection_count']), (12, 96, 4))
        self.assertTrue(report['enough_selected'])
        self.assertEqual(self.data[3]['selected_units'], ['u02', 'u03', 'u04', 'u05'])
        dumped = json.dumps(report)
        self.assertNotIn('subjecta', dumped)
        self.assertNotIn('bravaa', dumped)
        self.assertNotIn('u02', dumped)

    def test_rehashed_selection_corruptions_fail(self):
        def row(s): return s['unit_screens'][2]['instructions'][0]['cases'][0]
        def group(s): return s['unit_screens'][0]['instructions'][0]
        mutations = {
            'token': lambda s: row(s)['input_ids'].__setitem__(0, 1200),
            'boolean-mask': lambda s: row(s)['attention_mask'].__setitem__(0, True),
            'offset': lambda s: row(s)['offset_mapping'][1].__setitem__(0, 0),
            'collision': lambda s: group(s)['source_comparisons'][0].__setitem__('conflict', False),
            'comparison-count': lambda s: group(s)['source_comparisons'][0].__setitem__('comparisons', 7),
            'missing-screen': lambda s: s['unit_screens'].pop(),
            'semantic': lambda s: s['unit_screens'][1].__setitem__('semantic_ok', True),
            'selected-order': lambda s: s['selected_units'].reverse(),
            'donor': lambda s: s['donor_mapping'][0]['opposite'].__setitem__('P1', 'P2'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                altered = copy.deepcopy(self.data[3])
                mutate(altered)
                with self.assertRaises(ValueError):
                    check(self.data, selection=altered)

    def test_noncanonical_pool_fails_even_with_canonicalized_commitment(self):
        with self.assertRaisesRegex(ValueError, 'noncanonical'):
            check(self.data, pool_raw=self.data[0] + b'\n')
        with self.assertRaisesRegex(ValueError, 'noncanonical'):
            pretty = json.dumps(json.loads(self.data[0]), indent=2).encode()
            check(self.data, pool_raw=pretty)

    def test_wrong_salt_or_card_fails(self):
        pool_raw, pool_salt, pool_card, selection, salt, exposures, tok = self.data
        raw, card = encode_selection(selection, salt, pool_card['commitment'])
        with self.assertRaisesRegex(ValueError, 'pool commitment'):
            reader.check_saved(pool_raw, bytes([0] * 32), selector.canonical(pool_card) + b'\n',
                               raw, salt, card, exposures, tok, 'a' * 40)
        with self.assertRaisesRegex(ValueError, 'selection commitment'):
            reader.check_saved(pool_raw, pool_salt, selector.canonical(pool_card) + b'\n',
                               raw, bytes([0] * 32), card, exposures, tok, 'a' * 40)

    def test_prefix_both_directions_and_exact_eos_types(self):
        self.assertTrue(reader.conflicts([3, 4, 1], [3, 4, 5, 1]))
        self.assertTrue(reader.conflicts([3, 4, 5, 1], [3, 4, 1]))
        self.assertTrue(reader.conflicts([3, 4, 1], [3, 4, 1]))
        self.assertFalse(reader.conflicts([3, 4, 1], [3, 5, 1]))
        for row in ([3, 4], [3, 1, 1], [1], [True, 1]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                reader.conflicts(row, [3, 4, 1])

    def test_duplicate_nonfinite_and_boolean_tokenizer_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e309}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reader.strict_json(raw)
        tok = FakeTokenizer()
        tok.eos_token_id = True
        with self.assertRaisesRegex(ValueError, 'special IDs'):
            check(self.data, tokenizer=tok)

    def test_public_sources_are_pinned_and_parse_unicode(self):
        with tempfile.TemporaryDirectory(prefix='fake-flan-source-') as name:
            root = Path(name)
            (root / 'a').mkdir()
            raw = '{"text":"A FÜR-old item", "input_ids":[900,1]}'.encode()
            (root / 'a/prior.json').write_bytes(raw)
            def git(*args):
                return subprocess.check_output(['git.exe' if sys.platform == 'win32' else 'git', '-C', str(root), *args],
                                               stderr=subprocess.DEVNULL).decode().strip()
            git('init', '-q')
            git('add', 'a/prior.json')
            git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fabricated source')
            revision = git('rev-parse','HEAD')
            reader.checked_git_commit(root, revision)
            with self.assertRaisesRegex(ValueError, 'revision unavailable'):
                reader.checked_git_commit(root, '0' * 40)
            blob = git('rev-parse', 'HEAD:a/prior.json')
            with self.assertRaisesRegex(ValueError, 'not a commit'):
                reader.checked_git_commit(root, blob)
            descriptor = {'path':'a/prior.json', 'revision':revision, 'sha256':hashlib.sha256(raw).hexdigest()}
            old = reader.REQUIRED
            try:
                reader.REQUIRED = {descriptor['path']:descriptor['sha256']}
                loaded = reader.exposures_from_files([descriptor], root, root)
                self.assertEqual(loaded[0]['rows'], [[900, 1]])
                self.assertIn('für', loaded[0]['words'])
                (root / 'a/prior.json').write_bytes(raw + b' ')
                with self.assertRaisesRegex(ValueError, 'Git blob mismatch'):
                    reader.exposures_from_files([descriptor], root, root)
                changed = dict(descriptor, sha256=hashlib.sha256(raw + b' ').hexdigest())
                reader.REQUIRED = {changed['path']:changed['sha256']}
                with self.assertRaisesRegex(ValueError, 'Git blob mismatch'):
                    reader.exposures_from_files([changed], root, root)
            finally:
                reader.REQUIRED = old

    def test_fake_reference_compiles_captured_bytes_even_with_stale_bytecode(self):
        with tempfile.TemporaryDirectory(prefix='fake-flan-pyc-') as name:
            path = Path(name) / 'example.py'
            path.write_bytes(b"value = 'a'\n")
            stamp = path.stat()
            py_compile.compile(str(path), doraise=True)
            path.write_bytes(b"value = 'b'\n")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            spec = importlib.util.spec_from_file_location('fake_loader_positive_control', path)
            positive = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(positive)
            self.assertEqual(positive.value, 'a')
            checked = module('fake_loader_verified_source', path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(checked.value, 'b')

    def test_stage_receipt_binds_exact_source_tree_and_commitments(self):
        with tempfile.TemporaryDirectory(prefix='fake-flan-stage-') as name:
            repo = Path(name)
            frozen = ROOT
            fixed = {'docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md',
                     'docs/FLAN_T5_SENTIMENT_SELECTOR_V1.md',
                     'looking_glass/flan_t5_sentiment_selector_v1.py'}
            contents = {}
            for path in sorted(reader.STAGE_SOURCES):
                raw = ((frozen / path).read_bytes() if path in fixed else ('fabricated ' + path).encode())
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                contents[path] = raw
            def git(*args):
                return subprocess.check_output(['git.exe' if sys.platform == 'win32' else 'git', '-C', str(repo), *args],
                                               stderr=subprocess.DEVNULL).decode().strip()
            git('init','-q')
            git('add','.')
            git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fabricated stage')
            stage = {'schema':'flan-t5-sentiment-selector-stage-runtime-v1', 'stage':'select', 'status':'completed',
                     'selector_revision':git('rev-parse','HEAD'),
                     'source_sha256':{path:hashlib.sha256(raw).hexdigest() for path,raw in contents.items()},
                     'versions':{'python':'3.11.0','transformers':'4.0','tokenizers':'0.1'},
                     'pool_commitment':'a' * 64, 'selection_commitment':'b' * 64, 'scope':reader.STAGE_SCOPE}
            encoded = reader.canonical(stage) + b'\n'
            self.assertEqual(reader.checked_stage_receipt(encoded, repo, stage['selector_revision']), stage)
            with self.assertRaisesRegex(ValueError, 'stage identity'):
                reader.checked_stage_receipt(encoded, repo, '0' * 40)
            reader.bind_stage_commitments({'pool_commitment':'a' * 64,'selection_commitment':'b' * 64}, stage)
            with self.assertRaisesRegex(ValueError, 'commitments'):
                reader.bind_stage_commitments({'pool_commitment':'c' * 64,'selection_commitment':'b' * 64}, stage)
            with self.assertRaisesRegex(ValueError, 'stage receipt'):
                reader.checked_stage_receipt(encoded.rstrip(b'\n'), repo, stage['selector_revision'])
            malformed = copy.deepcopy(stage)
            malformed['versions']['python'] = True
            with self.assertRaisesRegex(ValueError, 'stage identity'):
                reader.checked_stage_receipt(reader.canonical(malformed) + b'\n', repo, stage['selector_revision'])
            altered = copy.deepcopy(stage)
            altered['source_sha256']['tools/run_flan_sentiment_selector_stage_v1.py'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'stage Git source'):
                reader.checked_stage_receipt(reader.canonical(altered) + b'\n', repo, stage['selector_revision'])

    def test_direct_exit_receipt_checks_types_logs_and_runtime_hash(self):
        stage = {'selector_revision':'f' * 40}
        stage_raw = b'fabricated-stage-bytes'
        stdout, stderr = b'completed\n', b''
        def log(raw):
            return {'retained_bytes':len(raw), 'truncated':False, 'error_type':None,
                    'finished':True, 'sha256':hashlib.sha256(raw).hexdigest()}
        record = {'schema':'flan-t5-sentiment-selector-stage-exit-v1','stage':'select','status':'completed',
                  'stage_receipt_sha256':hashlib.sha256(stage_raw).hexdigest(), 'reaped':True, 'exit_code':0,
                  'timed_out':False, 'exception_type':None, 'selector_revision':stage['selector_revision'],
                  'child_prefix_empty_after':True, 'logs':{'stdout':log(stdout),'stderr':log(stderr)},
                  'scope':reader.EXIT_SCOPE}
        self.assertEqual(reader.checked_stage_exit(reader.canonical(record) + b'\n', stage_raw, stage, stdout, stderr), record)
        for key, value in (('exit_code',False), ('reaped',1), ('timed_out',0),
                           ('child_prefix_empty_after',1), ('stage_receipt_sha256','0' * 64),
                           ('selector_revision','b' * 40)):
            with self.subTest(key=key):
                altered = copy.deepcopy(record)
                altered[key] = value
                with self.assertRaises(ValueError):
                    reader.checked_stage_exit(reader.canonical(altered) + b'\n', stage_raw, stage, stdout, stderr)
        for key, value in (('truncated',True),('finished',False),('retained_bytes',True),('sha256','0' * 64)):
            with self.subTest(log_key=key):
                altered = copy.deepcopy(record)
                altered['logs']['stdout'][key] = value
                with self.assertRaises(ValueError):
                    reader.checked_stage_exit(reader.canonical(altered) + b'\n', stage_raw, stage, stdout, stderr)
        with self.assertRaisesRegex(ValueError, 'direct-exit'):
            reader.checked_stage_exit(reader.canonical(record), stage_raw, stage, stdout, stderr)


if __name__ == '__main__':
    unittest.main()
