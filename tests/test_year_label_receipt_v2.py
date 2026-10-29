import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_receipt_records_closed_files_and_nonempty_cache(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/year_label_receipt_v2.py'
    attempt = tmp_path/'attempt'
    run = attempt/'run'
    cache = tmp_path/'pycache'
    run.mkdir(parents=True)
    cache.mkdir()
    (run/'result.json').write_bytes(b'{"status":"complete"}')
    (attempt/'stdout.txt').write_bytes(b'done')
    (attempt/'stderr.txt').write_bytes(b'')
    prefix_sha = hashlib.sha256(os.path.normcase(str(cache.resolve())).encode('utf-8')).hexdigest()
    launch = {'schema': 'distilgpt2-year-label-control-launch-v2', 'source_revision': 'revision',
              'source_tree': 'tree', 'intended_execution': {'pycache_prefix_sha256': prefix_sha}}
    raw = json.dumps(launch, sort_keys=True).encode('utf-8')
    (attempt/'pre-run.json').write_bytes(raw)
    pre_sha = hashlib.sha256(raw).hexdigest()
    (cache/'unexpected.pyc').write_bytes(b'not empty')
    subprocess.run([sys.executable, '-B', str(script), str(attempt), '--exit-code', '0',
                    '--pre-run-sha256', pre_sha, '--source-revision', 'revision', '--source-tree', 'tree',
                    '--pycache-prefix', str(cache)],
                   check=True, capture_output=True)
    receipt = json.loads((attempt/'supervisor.json').read_bytes())
    assert receipt['process_exit_code'] == 0 and receipt['interrupted'] is False
    assert receipt['pycache_prefix_empty_after'] is False
    assert receipt['pycache_prefix_sha256'] == prefix_sha and receipt['pycache_prefix_matches_pre_run'] is True
    assert 'pycache_prefix' not in receipt and str(cache) not in receipt.values()
    assert receipt['run_sha256']['result.json'] == hashlib.sha256(b'{"status":"complete"}').hexdigest()
    assert receipt['stderr_bytes'] == 0 and receipt['failure_file_present'] is False


def test_receipt_retains_prefix_mismatch_instead_of_claiming_a_match(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/year_label_receipt_v2.py'
    attempt, cache = tmp_path/'attempt', tmp_path/'pycache'
    (attempt/'run').mkdir(parents=True)
    cache.mkdir()
    (attempt/'stdout.txt').write_bytes(b'')
    (attempt/'stderr.txt').write_bytes(b'')
    launch = {'schema': 'distilgpt2-year-label-control-launch-v2', 'source_revision': 'revision',
              'source_tree': 'tree', 'intended_execution': {'pycache_prefix_sha256': '0' * 64}}
    raw = json.dumps(launch, sort_keys=True).encode('utf-8')
    (attempt/'pre-run.json').write_bytes(raw)
    subprocess.run([sys.executable, '-B', str(script), str(attempt), '--exit-code', '1',
                    '--pre-run-sha256', hashlib.sha256(raw).hexdigest(), '--source-revision', 'revision',
                    '--source-tree', 'tree', '--pycache-prefix', str(cache)], check=True, capture_output=True)
    receipt = json.loads((attempt/'supervisor.json').read_bytes())
    assert receipt['pycache_prefix_empty_after'] is True
    assert receipt['pycache_prefix_matches_pre_run'] is False
    assert receipt['process_exit_code'] == 1
