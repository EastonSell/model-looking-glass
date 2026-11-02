"""Direct-child observation is distinct from a child's completion marker."""
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.observe_cross_model_agreement_v1 import observe_child


CHILD = r'''
import hashlib, json, os, pathlib, sys
attempt = pathlib.Path(sys.argv[1]); attempt.mkdir()
metadata = {'pid': os.getpid(), 'source_commit': 'a' * 40}
metadata_bytes = json.dumps(metadata).encode('utf-8')
(attempt / 'metadata.json').write_bytes(metadata_bytes)
complete = {'status': 'complete', 'events': 192, 'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest()}
(attempt / 'complete.json').write_text(json.dumps(complete))
if sys.argv[2] == 'fail':
    (attempt / 'failure.txt').write_text('failure after completion')
sys.exit(int(sys.argv[3]))
'''


@pytest.mark.parametrize('failure,code,status', [('none', 0, 'complete'), ('fail', 0, 'incomplete'),
                                                 ('none', 1, 'incomplete')])
def test_child_exit_and_failure_sentinel_are_independent(tmp_path, failure, code, status):
    controller = tmp_path / 'controller'
    controller.mkdir()
    attempt = tmp_path / 'attempt'
    command = [str(Path(sys._base_executable).resolve()), '-c', CHILD, str(attempt), failure, str(code)]
    receipt = observe_child(command, controller=controller, attempt=attempt, source_commit='a' * 40,
                            environment=os.environ.copy(), timeout=15)
    assert receipt['status'] == status
    assert receipt['exit_code'] == code
    assert receipt['child_pid'] == json.loads((attempt / 'metadata.json').read_text())['pid']
    assert json.loads((controller / 'exit-observation.json').read_text()) == receipt
