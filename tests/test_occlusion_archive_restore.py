"""The published archive helpers reject incomplete output writes."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest


@pytest.mark.parametrize('phase', ['discovery', 'evaluation'])
def test_restore_detects_short_write(tmp_path, monkeypatch, phase):
    root = Path(__file__).resolve().parents[1] / 'results/vision-coco-occlusion-v1-20261029'
    part = b'fabricated archive bytes'
    name = f'{phase}-evidence.part-000'
    (tmp_path / name).write_bytes(part)
    index = {'schema': f'vision-occlusion-v1-{phase}-archive-index',
             'parts': [{'file': name, 'bytes': len(part), 'sha256': hashlib.sha256(part).hexdigest()}],
             'archive_bytes': len(part), 'archive_sha256': hashlib.sha256(part).hexdigest()}
    index_path, output = tmp_path / 'index.json', tmp_path / 'restored.zip'
    index_path.write_text(json.dumps(index), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['restore', '--index', str(index_path), '--output', str(output)])
    original_open = Path.open

    class ShortWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, raw):
            return self.stream.write(raw[:len(raw) // 2])

    def patched_open(path, mode='r', *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return ShortWriter(stream) if path == output and mode == 'xb' else stream

    monkeypatch.setattr(Path, 'open', patched_open)
    with pytest.raises(ValueError, match='short output write'):
        runpy.run_path(str(root / f'restore-{phase}.py'), run_name='__main__')
