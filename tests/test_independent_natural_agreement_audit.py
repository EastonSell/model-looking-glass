"""Synthetic-only regression checks for the independent saved-evidence auditor."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from looking_glass.agreement_journal import AgreementJournal
from looking_glass.agreement_patch_runner import AgreementCase, NaturalAgreementRunner


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module('independent_agreement_audit', ROOT / 'tools' / 'audit_natural_agreement_discovery.py')
fixture = load_module('toy_agreement_fixture', ROOT / 'tests' / 'test_agreement_patch_runner.py')


@pytest.fixture(scope='module')
def synthetic_journal(tmp_path_factory):
    """Generate a public, fabricated four-family journal; no pretrained weights."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(123)
            model = fixture.ToyModel().eval()
        cases = []
        for index in range(4):
            for member in fixture.cases():
                cases.append(AgreementCase(
                    case_id=f'family-{index}-{member.case_id}', quartet=f'family-{index}',
                    lemma=member.lemma, number=member.number,
                    input_ids=(1, member.input_ids[1], 30 + index, 40 + index), subject_index=1))
        path = tmp_path_factory.mktemp('agreement-audit') / 'attempt'
        NaturalAgreementRunner(model, AgreementJournal(path, {'protocol': 'fabricated-only'}),
                               expected_quartets=4).run(cases)
        yield path
    finally:
        torch.set_num_threads(old_threads)


def test_audits_complete_fabricated_journal(synthetic_journal):
    report = audit.inspect(synthetic_journal, None, synthetic=True)
    assert report['status'] == 'independent_saved_evidence_pass'
    assert report['scope'] == 'Fabricated synthetic data only.'
    assert report['forwards'] == 608


@pytest.mark.parametrize('tamper', [
    'boolean_invocation', 'duplicate_pair', 'duplicate_quartet',
    'false_landing', 'false_untouched', 'boolean_index', 'float_forward_count',
    'false_accounting', 'false_control_check', 'site_summary_site',
    'site_summary_count', 'quartet_site', 'summary_seed', 'summary_rng_hash',
])
def test_rejects_contradictory_saved_evidence(synthetic_journal, monkeypatch, tamper):
    event_path = synthetic_journal / 'events.jsonl'
    terminal_path = synthetic_journal / 'terminal.json'
    read = Path.read_bytes
    rows = [json.loads(line) for line in read(event_path).splitlines()]
    terminal = json.loads(read(terminal_path))

    def event(kind):
        return next(row['event'] for row in rows if row['event']['kind'] == kind)

    if tamper == 'boolean_invocation':
        event('clean_capture')['plan_metadata']['steps'][0]['invocation'] = False
    elif tamper == 'duplicate_pair':
        pairs = event('present_site_metrics')['metrics']['pairs']
        pairs.append(copy.deepcopy(pairs[0]))
    elif tamper == 'duplicate_quartet':
        quartets = event('present_site_metrics')['metrics']['quartets']
        quartets.append(copy.deepcopy(quartets[0]))
    elif tamper == 'false_landing':
        event('patch')['post_action_matches_donor_bits'] = False
    elif tamper == 'false_untouched':
        event('patch')['untouched_output_bits_equal'] = False
    elif tamper == 'boolean_index':
        rows[0]['index'] = False
    elif tamper == 'float_forward_count':
        event('summary')['result']['forwards'] = 608.0
    elif tamper == 'false_accounting':
        terminal['storage_accounting_complete'] = False
    elif tamper == 'false_control_check':
        event('clean_repeat')['checks']['last_logits']['within_tolerance'] = False
    elif tamper == 'site_summary_site':
        event('present_site_metrics')['metrics']['site_summary']['site'] = '4:mlp:last'
    elif tamper == 'site_summary_count':
        event('present_site_metrics')['metrics']['site_summary']['n_quartets'] = 3
    elif tamper == 'quartet_site':
        event('present_site_metrics')['metrics']['quartets'][0]['site'] = '4:mlp:last'
    elif tamper == 'summary_seed':
        event('summary')['result']['cpu_rng_seed'] = 20261029
    elif tamper == 'summary_rng_hash':
        event('summary')['result']['seeded_cpu_rng_sha256'] = '0' * 64
    raw_events = b''.join(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(',', ':')).encode() + b'\n' for row in rows)
    terminal['event_log_bytes'] = len(raw_events)
    raw_terminal = json.dumps(terminal, sort_keys=True, separators=(',', ':')).encode()

    def override(path):
        if path == event_path:
            return raw_events
        if path == terminal_path:
            return raw_terminal
        return read(path)

    monkeypatch.setattr(Path, 'read_bytes', override)
    with pytest.raises(AssertionError):
        audit.inspect(synthetic_journal, None, synthetic=True)
