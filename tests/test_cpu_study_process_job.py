"""CPU-only integration of study custody with the reviewed Windows job."""

import os
from pathlib import Path
import sys
import textwrap

import pytest

from looking_glass import cpu_study_supervision as cpu
from looking_glass import supervisor


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows process job required")
ROOT = Path(__file__).resolve().parents[1]


def launch(study_root, output, body, job):
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "TEMP", "TMP")
           if key in os.environ}
    env["PYTHONPATH"] = str(ROOT)
    command = [sys._base_executable, "-c", textwrap.dedent(body), str(study_root), str(output)]
    return cpu.run_cpu_study(command, study_root=study_root, timeout=10, lease_timeout=3,
                             terminate_grace=3, cwd=study_root, env=env, process_job=job)


def recover_job(job):
    for record in supervisor.pending_supervised_children():
        if record._process_job is job:
            record.recover(terminate_grace=3)
    job.close()


def test_clean_cpu_owner_reaps_and_drains_supplied_job(tmp_path):
    from scripts.v2_source_guards import WindowsJob
    job = WindowsJob()
    output = tmp_path / "ran"
    body = """
        from pathlib import Path
        import sys
        from looking_glass.cpu_study_supervision import claim_cpu_study
        session, lease = claim_cpu_study(Path(sys.argv[1]))
        Path(sys.argv[2]).touch()
        session.attest_clean(lease)
    """
    try:
        result = launch(tmp_path, output, body, job)
        assert result.clean and result.exit_code == 0 and output.is_file()
        assert not (tmp_path / "cpu-study.unsafe.json").exists()
        assert job._confirmed and job._handle is None
    finally:
        recover_job(job)


def test_live_descendant_drains_but_keeps_study_marker(tmp_path):
    from scripts.v2_source_guards import WindowsJob
    job = WindowsJob()
    output = tmp_path / "descendant.pid"
    body = """
        from pathlib import Path
        import subprocess, sys
        from looking_glass.cpu_study_supervision import claim_cpu_study
        session, lease = claim_cpu_study(Path(sys.argv[1]))
        child = subprocess.Popen([sys._base_executable, '-c', 'import time; time.sleep(60)'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Path(sys.argv[2]).write_text(str(child.pid))
        session.attest_clean(lease)
    """
    try:
        result = launch(tmp_path, output, body, job)
        assert output.is_file() and not result.clean
        assert (tmp_path / "cpu-study.unsafe.json").is_file()
        assert job._confirmed and job._handle is None
    finally:
        recover_job(job)


def test_invalid_job_is_rejected_before_study_start(tmp_path):
    output = tmp_path / "must-not-run"
    with pytest.raises(ValueError):
        launch(tmp_path, output, "raise AssertionError('must not run')", object())
    assert not output.exists()
    assert not (tmp_path / "cpu-study.unsafe.json").exists()
