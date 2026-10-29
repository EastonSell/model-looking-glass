"""Read-only Git blob reconciliation on a synthetic local source tree."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from looking_glass import agreement_calibration_saved_audit as audit


def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


@pytest.fixture
def committed_tree(tmp_path):
    git = shutil.which('git')
    if git is None: pytest.skip('Git is unavailable')
    root = tmp_path / 'source'; root.mkdir()
    paths = ('looking_glass/__init__.py', 'looking_glass/empty.py', 'docs/TRUSTED_SERVICE_PROFILE_V1.md',
             'docs/AGREEMENT_GPU_CALIBRATION_V1.md', 'docs/AGREEMENT_CUDA_ADAPTER.md',
             'docs/AGREEMENT_CALIBRATION_SAMPLING_DRAFT.md', 'scripts/agreement_calibration_bootstrap.py',
             'scripts/v2_source_guards.py')
    files = {}
    for name in paths:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        data = b'' if name.endswith('empty.py') else name.encode()
        path.write_bytes(data); files[name] = sha(data)
    def run(*args):
        return subprocess.run([git, *args], cwd=root, capture_output=True, check=True).stdout
    run('init', '-q'); run('add', '--all')
    run('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    commit = run('rev-parse', 'HEAD').decode().strip()
    manifest = {'schema': 'agreement-controller-source-v1', 'commit': commit, 'files': files}
    def item(name): return {'path': name, 'size': (root/name).stat().st_size, 'sha256': files[name]}
    source = {'schema': 'agreement-calibration-source-v1',
              'base': {'repository_files': [item(n) for n in paths[:3]]},
              'protocols': [item(n) for n in paths[3:6]]}
    return root.resolve(), git, commit, manifest, source


def test_rehashes_committed_blobs_checkout_and_recorded_source_sizes(committed_tree):
    root, git, commit, manifest, source = committed_tree
    assert hasattr(audit, 'verify_committed_sources'), 'committed source verification is missing'
    assert audit.verify_committed_sources(root, git, commit, manifest, canonical(source)) == 8


@pytest.mark.parametrize('mutation', ['changed_checkout', 'wrong_blob_digest', 'omitted_tracked_file',
                                      'wrong_recorded_size', 'untracked_python'])
def test_rejects_source_inventory_or_checkout_mismatch(committed_tree, mutation):
    root, git, commit, manifest, source = deepcopy(committed_tree)
    if mutation == 'changed_checkout': (root/'looking_glass/__init__.py').write_bytes(b'changed')
    elif mutation == 'wrong_blob_digest': manifest['files']['looking_glass/__init__.py'] = 'f' * 64
    elif mutation == 'omitted_tracked_file': manifest['files'].pop('looking_glass/empty.py')
    elif mutation == 'wrong_recorded_size': source['base']['repository_files'][0]['size'] += 1
    elif mutation == 'untracked_python': (root/'looking_glass/untracked.py').write_bytes(b'new')
    assert hasattr(audit, 'verify_committed_sources'), 'committed source verification is missing'
    with pytest.raises(ValueError):
        audit.verify_committed_sources(root, git, commit, manifest, canonical(source))
