import hashlib
import json
import os
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from looking_glass import year_label_control as study
from scripts import year_label_control_v1 as runner


def test_empty_bytecode_prefix_rejects_existing_entries(tmp_path, monkeypatch):
    prefix = tmp_path/'pycache'
    prefix.mkdir()
    monkeypatch.setattr(sys, 'pycache_prefix', str(prefix))
    assert runner.checked_empty_pycache_prefix() == str(prefix.resolve())
    (prefix/'stale.pyc').write_bytes(b'stale')
    with pytest.raises(RuntimeError, match='empty'):
        runner.checked_empty_pycache_prefix()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    child = subprocess.run([sys.executable, '-B', '-X', f'pycache_prefix={prefix}', runner.__file__, '--help'],
                           env=env, text=True, capture_output=True)
    assert child.returncode != 0 and 'pycache prefix must be empty' in child.stderr
    monkeypatch.setattr(sys, 'pycache_prefix', None)
    with pytest.raises(RuntimeError, match='pycache'):
        runner.checked_empty_pycache_prefix()


def test_empty_prefix_bypasses_stale_timestamp_valid_bytecode(tmp_path):
    module = tmp_path/'freshness_example.py'
    module.write_text("VALUE='cached'\n", encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'py_compile', str(module)], check=True)
    stat = module.stat()
    module.write_text("VALUE='source'\n", encoding='utf-8')
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    code = "import freshness_example; print(freshness_example.VALUE)"
    old = subprocess.check_output([sys.executable, '-B', '-c', code], cwd=tmp_path, text=True).strip()
    prefix = tmp_path/'fresh-cache'
    prefix.mkdir()
    isolated = subprocess.check_output([sys.executable, '-B', '-X', f'pycache_prefix={prefix}', '-c', code],
                                       cwd=tmp_path, text=True).strip()
    assert old == 'cached' and isolated == 'source'
    assert not list(prefix.iterdir())


def test_runtime_rejects_switch_to_a_different_empty_cache_prefix(tmp_path, monkeypatch):
    first, second = tmp_path/'first', tmp_path/'second'
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(runner, 'PYCACHE_AT_IMPORT', str(first.resolve()))
    monkeypatch.setattr(sys, 'pycache_prefix', str(second))
    with pytest.raises(RuntimeError, match='pycache prefix changed'):
        runner.environment()


def test_unbound_reservation_fails_before_model_work(monkeypatch):
    monkeypatch.setattr(runner, 'RESERVATION_VERSION', None)
    monkeypatch.setattr(runner, 'RESERVATION_SHA256', None)
    monkeypatch.setattr(runner, 'RESERVATION_MODULE', None)
    with pytest.raises(RuntimeError, match='reservation is not yet bound'):
        runner.check_exposure(study.make_inventory()['cases'])


def test_reservation_requires_exact_unique_planned_records(monkeypatch):
    import importlib
    rows = study.make_inventory()['cases']
    baseline = SimpleNamespace(INVENTORY_VERSION=study.BASELINE_VERSION, INVENTORY_SHA256=study.BASELINE_SHA256)
    payload = {'predecessor': {'version': study.BASELINE_VERSION, 'sha256': study.BASELINE_SHA256},
               'records': [{'record_id': 'planned-' + row['case_id'], 'category': 'planned',
                            'token_ids': row['input_ids']} for row in rows]}
    reservation = SimpleNamespace(INVENTORY_VERSION='test-reservation',
                                  YEAR_LABEL_CONTROL_ROWS=tuple(tuple(row['input_ids']) for row in rows),
                                  _checked_inventory=lambda: (payload, 'a' * 64))
    monkeypatch.setattr(runner, 'RESERVATION_VERSION', 'test-reservation')
    monkeypatch.setattr(runner, 'RESERVATION_SHA256', 'a' * 64)
    monkeypatch.setattr(runner, 'RESERVATION_MODULE', 'test-reservation-module')
    modules = {'looking_glass.pretrained_exposures_v24': baseline, 'test-reservation-module': reservation}
    monkeypatch.setattr(importlib, 'import_module', lambda name: modules[name])
    assert runner.check_exposure(rows)['reservation_sha256'] == 'a' * 64
    payload['records'][0]['category'] = 'observed-saved'
    with pytest.raises(RuntimeError, match='planned'):
        runner.check_exposure(rows)
    payload['records'][0]['category'] = 'planned'
    payload['records'].append(dict(payload['records'][0]))
    with pytest.raises(RuntimeError, match='planned'):
        runner.check_exposure(rows)
    payload['records'].pop()
    payload['records'][0]['token_ids'] = [0]
    with pytest.raises(RuntimeError, match='planned'):
        runner.check_exposure(rows)
    payload['records'][0]['token_ids'] = rows[0]['input_ids']
    reservation._checked_inventory = lambda: (payload, 'b' * 64)
    with pytest.raises(RuntimeError, match='exposure inventory'):
        runner.check_exposure(rows)


def test_nonfinite_returned_logits_are_saved_before_rejection(tmp_path):
    row = study.make_inventory()['cases'][0]
    payload = bytearray(runner.LOGITS_BYTES)
    struct.pack_into('<I', payload, 0, 0x7f800000)
    marker = {'recorded': False}
    with (tmp_path/'data').open('xb') as raw, (tmp_path/'log').open('xb') as log:
        with pytest.raises(ValueError):
            runner.persist_action(log, raw, row, 'clean', {'logits': bytes(payload)},
                                  {}, {}, lambda: None, marker)
    assert marker['recorded'] and (tmp_path/'data').read_bytes() == bytes(payload)


def _prepare(monkeypatch, tmp_path, model):
    monkeypatch.setattr(runner, 'source_identity', lambda: {'revision': 'fabricated'})
    monkeypatch.setattr(runner, 'check_exposure', lambda cases: {'fabricated': True})
    monkeypatch.setattr(runner, 'environment', lambda: {'fabricated': True})
    monkeypatch.setattr(runner, 'checked_empty_pycache_prefix', lambda: str(tmp_path))
    monkeypatch.setattr(runner, 'runtime', lambda model: {'fabricated': True})
    monkeypatch.setattr(runner, 'assert_no_forward_hooks', lambda model: None)
    monkeypatch.setattr(runner.base, 'dependency_identity', lambda: {'fabricated': True})
    monkeypatch.setattr(runner.base, 'numeric_policy', lambda **kwargs: {'fabricated': True})
    monkeypatch.setattr(runner.base, 'state_hash', lambda model: 'state')
    monkeypatch.setattr(runner.base, 'buffer_hash', lambda model: 'buffers')
    monkeypatch.setattr(runner.base, 'load_distilgpt2', lambda **kwargs:
                        SimpleNamespace(model=model, tokenizer=None,
                                        metadata={'weights_sha256': runner.base.WEIGHTS_SHA256}))
    monkeypatch.setattr(study, 'validate_tokenizer', lambda tokenizer, cases: None)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    output = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['year_label_control_v1.py', '--cache-dir', str(tmp_path),
                                     '--output-dir', str(output)])
    return output


@pytest.mark.parametrize('fault', ['post_return', 'return_count', 'helper_count', 'partial_write'])
def test_post_return_failure_preserves_logits_and_counts(tmp_path, monkeypatch, fault):
    import inspect
    calls = []
    class Model:
        def __call__(self, input_ids, **kwargs):
            calls.append(1)
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], runner.VOCAB)),
                                   past_key_values=None, attentions=None, hidden_states=None)
    output = _prepare(monkeypatch, tmp_path, Model())
    lines, first = inspect.getsourcelines(runner.main)
    marker_lines = {'post_return': "phase = 'persist_and_validate'", 'return_count': 'returned = returned_before + 1',
                    'helper_count': 'payload_returned = True'}
    target = (first + next(i for i, line in enumerate(lines) if line.strip() == marker_lines[fault])
              if fault != 'partial_write' else None)
    fired = []
    def trace(frame, event, argument):
        if (event == 'line' and frame.f_code.co_name == 'perform' and frame.f_code.co_filename == runner.__file__
                and frame.f_lineno == target and not fired):
            fired.append(True)
            raise MemoryError('synthetic post-return failure')
        return trace
    if fault != 'partial_write':
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            with pytest.raises(MemoryError, match='synthetic post-return failure'):
                runner.main()
        finally:
            sys.settrace(previous)
    else:
        original = runner.base._write_payload
        def fail_once(stream, payload):
            if not fired:
                fired.append(True)
                stream.write(payload['logits'][:128])
                stream.flush()
                raise OSError('synthetic partial write')
            return original(stream, payload)
        monkeypatch.setattr(runner.base, '_write_payload', fail_once)
        with pytest.raises(OSError, match='synthetic partial write'):
            runner.main()
    failure = json.loads((output/'failure.json').read_bytes())
    assert fired and len(calls) == 1
    assert failure['model_call_returns_observed'] == 1 and failure['returned_forwards'] == 1
    assert failure['model_call_return_observed'] is True and failure['payload_returned'] is True
    assert failure['partial_payload_recorded'] is True
    assert next(output.glob('*.partial.f32')).read_bytes() == bytes(runner.LOGITS_BYTES)
    if fault == 'partial_write':
        assert len(failure['raw_files']) == 1 and failure['raw_files'][0]['bytes'] == 128


def test_complete_fabricated_schedule_retains_every_control(tmp_path, monkeypatch):
    rows = study.make_inventory()['cases']
    by_input = {tuple(row['input_ids']): row for row in rows}
    torch.set_num_threads(1)
    class Block(torch.nn.Module):
        def forward(self, value):
            return (value,)
    class Model:
        def __init__(self):
            self.block = Block()
            self.transformer = SimpleNamespace(h=[None, None, None, self.block])
        def __call__(self, input_ids, **kwargs):
            row = by_input[tuple(input_ids[0].tolist())]
            length = input_ids.shape[1]
            hidden = torch.zeros((1, length, 768), dtype=torch.float32)
            hidden[0, -1, 5] = 10.0 * row['anchor_index']
            hidden[0, -1, 9] = 5.0 * row['label_index']
            observed = self.block(hidden)[0]
            logits = torch.zeros((1, length, runner.VOCAB), dtype=torch.float32)
            logits[0, -1, row['middle_ids']] = observed[0, -1, 5] * 0.03 + observed[0, -1, 9] * 0.05
            return SimpleNamespace(logits=logits, past_key_values=None, attentions=None, hidden_states=None)
    model = Model()
    output = _prepare(monkeypatch, tmp_path, model)
    runner.main()
    result = json.loads((output/'result.json').read_bytes())
    manifest_bytes = (output/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_prefix_sha = hashlib.sha256(os.path.normcase(str(tmp_path.resolve())).encode()).hexdigest()
    assert manifest['empty_pycache_prefix_sha256'] == expected_prefix_sha
    assert 'empty_pycache_prefix' not in manifest and str(tmp_path) not in manifest.values()
    assert result['status'] == 'complete' and result['submitted_forwards'] == 384
    assert result['raw_payload_bytes'] == 103_736_832 and len(result['records']) == 384
    assert len(result['family_doses']) == 4
    assert len(result['intervention_summary']['families']) == 4
    first = result['intervention_summary']['families'][0]['targets'][0]
    assert first['effects']['year_final'] == pytest.approx(0.075, abs=1e-6)
    assert first['effects']['label_final'] == pytest.approx(0.125, abs=1e-6)
    assert first['effects']['self'] == first['effects']['year_at_zero'] == first['effects']['rotated_year_final'] == 0.0
    assert first['year_donorward'] == pytest.approx(-0.075, abs=1e-6) and first['pass'] is False
    expected = [(row['case_id'], action) for row in rows for action in ('clean', 'noop')]
    for row in rows:
        shift = row['rotation']
        expected.extend((row['case_id'], action)
                        for action in study.ACTIONS[shift:] + study.ACTIONS[:shift] + ('clean_after',))
    assert [(item['case_id'], item['action']) for item in result['records']] == expected
    events = [json.loads(line)['event'] for line in (output/'attempt.jsonl').read_bytes().splitlines()]
    assert all(events.count(event) == 384 for event in ('start', 'raw_recorded', 'complete'))
    assert events.count('family_dose') == 4 and not model.block._forward_hooks
