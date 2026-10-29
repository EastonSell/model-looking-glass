"""Optional fake-telemetry replay against the frozen sampler source.

The normal auditor suite has no dependency on this unmerged producer. Set
MLG_AGREEMENT_PRODUCER_CHECKOUT to a trusted checkout of the exact commit to
run the compatibility check. It archives tracked source into a temporary
directory; it never runs or writes inside the supplied checkout.
"""

from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import sysconfig
import tarfile

import pytest

from test_agreement_calibration_saved_audit import frozen_fixture


PRODUCER_COMMIT = 'ff03d3c9739c894b33839e907b633109dba2a7b3'
PRODUCER_CHECKOUT = os.environ.get('MLG_AGREEMENT_PRODUCER_CHECKOUT')

_REPLAY = r'''
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

root, site = (Path(value) for value in sys.argv[1:])
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.append(str(site))

def load(name, path, package=False):
    options = {'submodule_search_locations': [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load('looking_glass', root / 'looking_glass' / '__init__.py', True)
load('test_agreement_cuda_runner', root / 'tests' / 'test_agreement_cuda_runner.py')
fixture = load('test_agreement_calibration_collector',
               root / 'tests' / 'test_agreement_calibration_collector.py')
from looking_glass import agreement_calibration_collector as collector
from looking_glass.agreement_calibration_owner import assemble_profiles
import torch
assert not torch.cuda.is_initialized()
owner, key, _ = fixture.AgreementCalibrationSamplingTests().runner()
records = []
with patch.object(collector, 'CALIBRATION_EXECUTION_READY', True), \
     patch.object(collector, '_SAMPLING_CUSTODY', [None] * 4):
    cold, warm = collector.measure_calibration(owner, key, records.append)
source = b'synthetic source manifest'
profiles, evidence = assemble_profiles(key, cold, warm, source, '2026-10-29T00:00:00Z')
assert not torch.cuda.is_initialized()
print(json.dumps({'records': records, 'warm_key': asdict(key),
                  'profile_documents': [p.to_json() for p in profiles],
                  'evidence_documents': [item.decode() for item in evidence],
                  'source': source.decode(), 'profile_ids': [p.profile_id for p in profiles]},
                 sort_keys=True, separators=(',', ':'), allow_nan=False))
'''


@pytest.mark.skipif(not PRODUCER_CHECKOUT, reason='requires an explicitly selected frozen producer checkout')
def test_frozen_producer_replays_the_synthetic_fixture(tmp_path):
    checkout = Path(PRODUCER_CHECKOUT).resolve(strict=True)
    git = shutil.which('git')
    assert git is not None, 'Git is required for the optional producer replay'
    env = {name: value for name, value in os.environ.items() if not name.startswith('GIT_')}
    env['GIT_NO_REPLACE_OBJECTS'] = '1'
    head = subprocess.run([git, '-C', str(checkout), 'rev-parse', 'HEAD'], env=env,
                          capture_output=True, check=True, timeout=30).stdout.decode().strip()
    assert head == PRODUCER_COMMIT
    archive = subprocess.run([git, '-C', str(checkout), 'archive', PRODUCER_COMMIT, '--',
                              'looking_glass', 'tests/test_agreement_cuda_runner.py',
                              'tests/test_agreement_calibration_collector.py'], env=env,
                             capture_output=True, check=True, timeout=30).stdout
    root = tmp_path / 'producer'
    root.mkdir()
    with tarfile.open(fileobj=BytesIO(archive), mode='r:') as source:
        members = source.getmembers()
        for item in members:
            path = PurePosixPath(item.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert item.isfile() or item.isdir(), 'unexpected non-regular archive member'
        source.extractall(root, members=members)
    prefix = tmp_path / 'empty-bytecode-prefix'
    prefix.mkdir()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    replay = subprocess.run([sys.executable, '-I', '-S', '-B', '-X', 'pycache_prefix=' + str(prefix),
                             '-c', _REPLAY, str(root), sysconfig.get_path('purelib')],
                            cwd=tmp_path, env=env, capture_output=True, check=True, timeout=90)
    actual = json.loads(replay.stdout)
    expected = frozen_fixture()
    assert expected['producer_source_commit'] == PRODUCER_COMMIT
    fields = ('records', 'warm_key', 'profile_documents', 'evidence_documents', 'source', 'profile_ids')
    assert actual == {name: expected[name] for name in fields}
    assert not any(prefix.iterdir())
