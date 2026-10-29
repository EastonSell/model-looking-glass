"""The real custody API is exercised only with fabricated pools and a stub model."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import natural_agreement_heldout_child as child


@pytest.mark.parametrize('reject', [False, True])
def test_child_requires_real_custody_preparation_receipt_before_fake_model(reject):
    source = Path(__file__).with_name('test_natural_heldout_custody.py')
    if not source.is_file():
        pytest.skip('the separately published custody fixture is not in this checkout')
    pytest.importorskip('tools.natural_heldout_custody')
    spec = importlib.util.spec_from_file_location('fabricated_custody_fixture', source)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    fixture = fixtures.CustodyTests('test_publication_precedes_one_durable_claim_and_nonce')
    fixture.setUp()
    steps = []
    try:
        attempt, _ = fixture.prepare()
        supervisor = fixture.supervised(attempt)
        (attempt / 'owner').mkdir()
        url = ('https://raw.githubusercontent.com/EastonSell/model-looking-glass/' + 'c' * 40 +
               '/results/distilgpt2-natural-agreement-v1-20261028/heldout-commitment-v1.json')
        fixture.remote[url] = (attempt / 'commitment.json').read_bytes()
        if reject:
            receipt_path = supervisor / 'supervisor.json'
            receipt = json.loads(receipt_path.read_bytes())
            receipt['exit_code'] = 7
            receipt_path.write_bytes(fixtures.custody._json_bytes(receipt))
        def verify(*args, **kwargs):
            steps.append('verify')
            return fixtures.custody.verify_prepared(*args, reader=fixture.reader, **kwargs)
        def start(*args):
            steps.append('start')
            return fixtures.custody.start_owner_once(*args)
        def tokenize(*args):
            steps.append('retokenize')
            return fixtures.custody.tokenize_assignment(*args)
        custody = SimpleNamespace(verify_prepared=verify, start_owner_once=start,
                                  tokenize_assignment=tokenize, observe_current_inventory=fixture.observe_inventory)
        class Preflight:
            def _tokenizer(self, cache):
                return fixtures.FakeTokenizer()
            def tokenize_inputs(self, tokenizer):
                return [], []
        def run(*args, **kwargs):
            steps.append('fabricated-runner')
        heldout = SimpleNamespace(load_frozen_natural_predictions=lambda *args: None, run_committed_heldout=run)
        metadata = {'model_revision':'revision', 'asset_sha256':{}, 'device':'cpu', 'dtype':'float32',
                    'attention_implementation':'eager', 'use_cache':False}
        def model(**kwargs):
            steps.append('fake-model')
            return SimpleNamespace(model=object(), tokenizer=fixtures.FakeTokenizer(), metadata=metadata)
        dependencies = {'preflight':Preflight(), 'commitments':fixtures.commitments, 'heldout':heldout,
                        'load_model':model, 'state_hash':lambda model: child.MODEL_STATE_SHA256,
                        'journal':lambda *a, **k: object(), 'runner':lambda *a, **k: object(),
                        'control':SimpleNamespace(bind_public_cases=lambda *a: []), 'manifest':b'fake'}
        binding = {'checkpoint_revision':'revision', 'assets_sha256':{}, 'cases':[]}
        config = {'cache_dir':str(fixture.root), 'phase_dir':str(attempt),
                  'preparation_supervisor':str(supervisor), 'public_urls':{'commitment_url':url}}
        frozen = {'_start_record':lambda *args: {}}
        if reject:
            with pytest.raises(fixtures.custody.CustodyError):
                child._run_pipeline(frozen, config, binding, fixture.public, custody, fixture.wrapper, dependencies, {})
            assert steps == ['verify'] and not (attempt / 'owner-started.json').exists()
        else:
            child._run_pipeline(frozen, config, binding, fixture.public, custody, fixture.wrapper, dependencies, {})
            assert steps == ['verify', 'start', 'fake-model', 'retokenize', 'fabricated-runner']
            assert (attempt / 'owner-started.json').is_file()
    finally:
        fixture.doCleanups()
