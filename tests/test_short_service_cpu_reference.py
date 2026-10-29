"""Fabricated CPU checks; no pretrained model or GPU is used."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest
import torch

from looking_glass import short_service_cpu_reference as collector


def fixed_manifest():
    path = Path(__file__).resolve().parents[1] / 'experiments/short_cpu_service_reference_inputs_v1.json'
    return json.loads(path.read_bytes())


def synthetic_inventory(manifest):
    mapping = {'synthetic-16': ('planned-service-synthetic-mask-16',) * 2,
               'synthetic-32': ('planned-service-synthetic-mask-32',) * 2,
               'gas': ('planned-short-science-gas-active', 'planned-short-science-gas-padded'),
               'ligand': ('planned-short-science-ligand-active', 'planned-short-science-ligand-padded')}
    records = {}
    for case in manifest['comparisons']:
        for role, identity in zip(('reference', 'service'), mapping[case['id']], strict=True):
            value = case[role]
            records[identity] = {'record_id': identity, 'category': 'planned',
                                 'token_ids': value['input_ids'], 'attention_mask': value['attention_mask']}
    return {'records': list(records.values())}


def test_reservation_validation_rejects_missing_or_changed_mask(tmp_path, monkeypatch):
    manifest = fixed_manifest()
    inventory = synthetic_inventory(manifest)
    manifest_raw = json.dumps(manifest).encode()
    inventory_raw = json.dumps(inventory).encode()
    (tmp_path / 'manifest.json').write_bytes(manifest_raw)
    (tmp_path / 'inventory.json').write_bytes(inventory_raw)
    monkeypatch.setattr(collector, 'MANIFEST_SHA256', hashlib.sha256(manifest_raw).hexdigest())
    monkeypatch.setattr(collector, 'INVENTORY_SHA256', hashlib.sha256(inventory_raw).hexdigest())
    collector._read_inputs(tmp_path / 'manifest.json', tmp_path / 'inventory.json')
    inventory['records'][-1]['attention_mask'][-1] = 1
    changed = json.dumps(inventory).encode()
    (tmp_path / 'inventory.json').write_bytes(changed)
    monkeypatch.setattr(collector, 'INVENTORY_SHA256', hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match='reservation'):
        collector._read_inputs(tmp_path / 'manifest.json', tmp_path / 'inventory.json')


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False, vocab_size=50257, n_positions=32,
                                      _attn_implementation='eager')
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.calls += 1
        return SimpleNamespace(logits=input_ids.unsqueeze(-1).to(torch.float32) +
                               torch.arange(50257, dtype=torch.float32).view(1, 1, -1) / 1000)


def test_full_reference_evidence_and_padding_comparisons_are_reproducible():
    model = FakeModel().eval()
    with tempfile.TemporaryDirectory(prefix='short-cpu-reference-test-') as directory:
        path = Path(directory)
        entries, comparisons = collector._sequence(SimpleNamespace(model=model),
                                                    fixed_manifest()['comparisons'], path)
        assert model.calls == 6 and len(entries) == 6
        assert sum(entry['bytes'] for entry in entries) == (16 + 15 + 16 + 32 + 23 + 32) * 50257 * 4
        assert all(item['comparison']['violations'] == 0 for item in comparisons)
        for entry in entries:
            raw = (path / entry['filename']).read_bytes()
            assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']


def test_cpu_only_environment_gate_precedes_model_loading(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError, match='CPU-only'):
        collector.main(['--cache-dir', 'unused', '--inventory', 'unused', '--output-dir', 'unused',
                        '--git', 'unused', '--expected-commit', '0' * 40])


def test_low_disk_gate_precedes_output_and_model_load(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setattr(collector, '_source_identity', lambda *_args: {})
    monkeypatch.setattr(collector, '_read_inputs', lambda *_args: (fixed_manifest(), {}))
    monkeypatch.setattr(collector.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=0))
    output = tmp_path / 'new-evidence'
    with pytest.raises(RuntimeError, match='disk'):
        collector.main(['--cache-dir', str(tmp_path), '--inventory', 'unused', '--output-dir', str(output),
                        '--git', 'unused', '--expected-commit', '0' * 40])
    assert not output.exists()
