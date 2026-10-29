"""No-checkpoint, Windows-only source and CPU custody rehearsal."""

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading
import textwrap
from types import SimpleNamespace
import uuid

import pytest
import torch

from scripts import targeted_edit_v2_source_rehearsal as rehearsal

if __package__:
    from .test_targeted_edit_v2_cpu_probe import _matches_recorded_host
else:
    from test_targeted_edit_v2_cpu_probe import _matches_recorded_host


ROOT = Path(__file__).resolve().parents[1]
PARENT = "scripts/targeted_edit_v2_source_rehearsal.py"
CHILD = "scripts/targeted_edit_v2_source_child.py"
REFERENCE = "results/targeted-edit-cpu-v1-20261028/reference/precommit.json"


def test_published_quarantine_cannot_be_released_during_admission():
    attempt = uuid.uuid4().hex
    quarantine = rehearsal.register_quarantine(attempt)
    try:
        with pytest.raises(rehearsal.RehearsalError, match="running"):
            quarantine.release()
        assert rehearsal.get_quarantine(attempt) is quarantine
    finally:
        quarantine.running = False
        if rehearsal.get_quarantine(attempt) is quarantine:
            quarantine.release()


def test_interrupted_registration_return_keeps_the_exact_record_rooted():
    attempt = uuid.uuid4().hex
    previous = sys.gettrace()
    def trace(frame, event, _arg):
        if frame.f_code is rehearsal.register_quarantine.__code__ and event == "return":
            sys.settrace(None)
            raise KeyboardInterrupt("registration return lost")
        return trace
    try:
        sys.settrace(trace)
        with pytest.raises(KeyboardInterrupt, match="registration return lost"):
            rehearsal.register_quarantine(attempt)
    finally:
        sys.settrace(previous)
    quarantine = rehearsal.get_quarantine(attempt)
    assert quarantine is not None and quarantine.running
    try:
        with pytest.raises(rehearsal.RehearsalError):
            quarantine.release()
    finally:
        # The injected return occurred before this test acquired any resource.
        quarantine.running = False
        quarantine.release()


def test_simultaneous_release_cannot_close_the_same_guard_twice():
    attempt = uuid.uuid4().hex
    quarantine = rehearsal.register_quarantine(attempt)
    entered, resume = threading.Event(), threading.Event()
    outcomes = []

    class BlockedCache:
        _used = True
        def close(self):
            entered.set()
            assert resume.wait(5)

    quarantine.running = False
    quarantine.complete = True
    quarantine.cache = BlockedCache()
    def release():
        try:
            quarantine.release()
            outcomes.append("released")
        except BaseException as error:
            outcomes.append(error)
    first = threading.Thread(target=release)
    second = threading.Thread(target=release)
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        second.join(timeout=1)
        assert not second.is_alive(), "a concurrent release entered guarded cleanup"
    finally:
        resume.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert outcomes.count("released") == 1
    assert sum(isinstance(item, rehearsal.RehearsalError) for item in outcomes) == 1


def test_uncertain_supervisor_return_retains_exact_cache_guard():
    attempt = uuid.uuid4().hex
    quarantine = rehearsal.register_quarantine(attempt)
    released = []
    quarantine.cache = SimpleNamespace(_used=True, close=lambda: released.append(True))
    quarantine.launch_started = True
    quarantine.running = False
    try:
        with pytest.raises(rehearsal.RehearsalError, match="uncertain"):
            quarantine.release()
        assert released == [] and rehearsal.get_quarantine(attempt) is quarantine
    finally:
        # No process was started in this unit test; clear only its synthetic flag.
        quarantine.launch_started = False
        quarantine.release()


def test_failed_disposal_remains_rooted_and_cannot_be_retried_blindly():
    attempt = uuid.uuid4().hex
    quarantine = rehearsal.register_quarantine(attempt)
    def fail_close():
        raise OSError("unknown close outcome")
    cache = SimpleNamespace(_used=True, close=fail_close)
    quarantine.cache = cache
    quarantine.complete = True
    quarantine.running = False
    try:
        with pytest.raises(OSError, match="unknown close"):
            quarantine.release()
        assert rehearsal.get_quarantine(attempt) is quarantine and quarantine.cache is cache
        with pytest.raises(rehearsal.RehearsalError):
            quarantine.release()
    finally:
        # Only a fake cache was used; the test can remove this synthetic record.
        quarantine.cache = None
        quarantine.disposing = False
        quarantine.release()


def test_lookup_rejects_non_string_before_hashing():
    class HostileIdentity:
        def __hash__(self):
            raise AssertionError("unexpected hashing under the registry lock")
    with pytest.raises(rehearsal.RehearsalError):
        rehearsal.get_quarantine(HostileIdentity())


def test_stale_quarantine_cannot_release_a_reused_identifier():
    attempt = uuid.uuid4().hex
    old = rehearsal.register_quarantine(attempt)
    old.running = False
    old.release()
    replacement = rehearsal.register_quarantine(attempt)
    try:
        with pytest.raises(rehearsal.RehearsalError):
            old.release()
        assert rehearsal.get_quarantine(attempt) is replacement
    finally:
        replacement.running = False
        replacement.release()


def test_second_attempt_is_rejected_before_acquisition(monkeypatch):
    calls = []
    monkeypatch.setattr(rehearsal, "_PARENT_USED", True, raising=False)
    monkeypatch.setattr(rehearsal, "_execute", lambda *_args: calls.append(True))
    with pytest.raises(rehearsal.RehearsalError, match="single-use"):
        rehearsal.run(None, None, None, None, None, attempt_id=uuid.uuid4().hex)
    assert calls == []


def test_cli_failure_reports_exact_retained_quarantine(monkeypatch, capsys, tmp_path):
    held = tmp_path / "retained"
    held.mkdir()
    def fail_before_launch(quarantine, *_args):
        quarantine.temp = held
        quarantine.temp_owned = True
        raise OSError("synthetic prelaunch failure")
    monkeypatch.setattr(rehearsal, "_PARENT_USED", False, raising=False)
    monkeypatch.setattr(rehearsal, "_execute", fail_before_launch)
    args = ["--checkout", "x", "--venv-root", "x", "--dependency", "x", "--work-dir", "x",
            "--study-root", "x"]
    assert rehearsal.main(args) == 1
    report = json.loads(capsys.readouterr().err)
    assert report["quarantine_rooted_at_report"] is True
    assert report["parent_exit_releases_handles"] is True
    assert report["quarantine_location"] == str(held)
    quarantine = rehearsal.get_quarantine(report["attempt_id"])
    assert quarantine is not None and not quarantine.running
    quarantine.release()


def test_interrupted_directory_creation_retains_its_known_path(monkeypatch, tmp_path):
    quarantine = rehearsal.register_quarantine(uuid.uuid4().hex)
    original = Path.mkdir
    def create_then_interrupt(path, *args, **kwargs):
        original(path, *args, **kwargs)
        raise KeyboardInterrupt("directory return lost")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "mkdir", create_then_interrupt)
            with pytest.raises(KeyboardInterrupt):
                rehearsal._create_attempt_dir(quarantine, tmp_path)
        assert quarantine.temp is not None and quarantine.temp.is_dir()
        quarantine.running = False
        with pytest.raises(rehearsal.RehearsalError, match="directory"):
            quarantine.release()
        assert quarantine.temp.is_dir()
    finally:
        # The controlled test knows this mkdir succeeded and owns the path.
        quarantine.running = False
        quarantine.temp_owned = True
        if rehearsal.get_quarantine(quarantine.attempt_id) is quarantine:
            quarantine.release()


def test_cleanup_path_inspection_does_not_hold_registry_lock(monkeypatch, tmp_path):
    attempt = uuid.uuid4().hex
    quarantine = rehearsal.register_quarantine(attempt)
    directory = tmp_path / "owned"
    directory.mkdir()
    quarantine.temp, quarantine.temp_owned, quarantine.running = directory, True, False
    entered, resume = threading.Event(), threading.Event()
    seen = []
    real_stat, real_lstat = Path.stat, Path.lstat
    def gate(function):
        def gated(path, *args, **kwargs):
            if path == directory and not entered.is_set():
                entered.set()
                assert resume.wait(5)
            return function(path, *args, **kwargs)
        return gated
    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", gate(real_stat))
        patch.setattr(Path, "lstat", gate(real_lstat))
        first = threading.Thread(target=quarantine.release)
        first.start()
        try:
            assert entered.wait(5)
            second = threading.Thread(target=lambda: seen.append(rehearsal.get_quarantine(attempt)))
            second.start()
            second.join(timeout=1)
            assert not second.is_alive(), "filesystem inspection held the registry lock"
        finally:
            resume.set()
            first.join(timeout=5)
            if 'second' in locals():
                second.join(timeout=5)
    assert seen == [quarantine]
    assert not first.is_alive() and rehearsal.get_quarantine(attempt) is None


def test_report_reader_rejects_a_link_before_opening(monkeypatch, tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"value":1}', encoding="utf-8")
    original = Path.lstat
    def forged_lstat(candidate):
        return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0) if candidate == path else original(candidate)
    monkeypatch.setattr(Path, "lstat", forged_lstat)
    with pytest.raises(rehearsal.RehearsalError, match="regular"):
        rehearsal._bounded_json(path, 100)


def test_report_validation_rejects_boolean_integer_substitution():
    reference = {"environment": {name: False for name in rehearsal._ENVIRONMENT_KEYS}}
    runtime = dict(reference["environment"])
    run_id = "a" * 32
    result = SimpleNamespace(pid=123, run_id=run_id)
    sources = {"looking_glass/" + name + ".py": "c" * 64 for name in
               ("__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
                "targeted_edit_cpu_v1", "targeted_edit_v2_edits")}
    child = {"status": "matching_v1_runtime", "source_revision": "b" * 40,
             "owner_pid": 123, "owner_run_id": run_id, "claim_mask": "",
             "model_import_mask": "-1", "runtime": runtime, "denormal_probe_before": False,
             "denormal_probe_after": False, "project_sources": sources,
             "checkpoint_loaded": False, "model_forwarded": False}
    rehearsal._check_child_report(child, result, "b" * 40, sources, reference)
    runtime["deterministic_algorithms"] = 0
    with pytest.raises(rehearsal.RehearsalError):
        rehearsal._check_child_report(child, result, "b" * 40, sources, reference)


@pytest.mark.parametrize("mutation", ["pid", "run_id", "revision", "mask", "runtime", "probe", "source",
                                       "missing_source", "checkpoint", "forward", "extra"])
def test_report_rejects_rehashed_identity_and_preflight_mismatches(mutation):
    reference = json.loads((ROOT / REFERENCE).read_text(encoding="utf-8"))
    runtime = {name: reference["environment"][name] for name in rehearsal._ENVIRONMENT_KEYS}
    sources = {"looking_glass/" + name + ".py": "c" * 64 for name in
               ("__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
                "targeted_edit_cpu_v1", "targeted_edit_v2_edits")}
    child = {"status": "matching_v1_runtime", "source_revision": "b" * 40,
             "owner_pid": 123, "owner_run_id": "a" * 32, "claim_mask": "", "model_import_mask": "-1",
             "runtime": runtime, "denormal_probe_before": dict(runtime["denormal_probe"]),
             "denormal_probe_after": dict(runtime["denormal_probe"]), "project_sources": dict(sources),
             "checkpoint_loaded": False, "model_forwarded": False}
    result = SimpleNamespace(pid=123, run_id="a" * 32)
    rehearsal._check_child_report(child, result, "b" * 40, sources, reference)
    if mutation == "pid":
        child["owner_pid"] = 124
    elif mutation == "run_id":
        child["owner_run_id"] = "d" * 32
    elif mutation == "revision":
        child["source_revision"] = "d" * 40
    elif mutation == "mask":
        child["model_import_mask"] = ""
    elif mutation == "runtime":
        child["runtime"]["torch_build_sha256"] = "d" * 64
    elif mutation == "probe":
        child["denormal_probe_after"]["subnormal_times_one_bits"] = 0
    elif mutation == "source":
        child["project_sources"]["looking_glass/__init__.py"] = "d" * 64
    elif mutation == "missing_source":
        del child["project_sources"]["looking_glass/supervisor.py"]
    elif mutation == "checkpoint":
        child["checkpoint_loaded"] = 0
    elif mutation == "forward":
        child["model_forwarded"] = True
    else:
        child["unexpected"] = True
    with pytest.raises(rehearsal.RehearsalError):
        rehearsal._check_child_report(child, result, "b" * 40, sources, reference)


def checkout_for_rehearsal(root):
    checkout = root / "checkout"
    paths = [PARENT, CHILD, "scripts/v2_source_launcher.py", "scripts/v2_source_bootstrap.py",
             "scripts/v2_source_guards.py", REFERENCE]
    modules = ("__init__", "cpu_study_supervision", "supervisor", "device_lease", "device_identity",
               "pretrained", "pretrained_exposures", "pretrained_exposures_v2", "targeted_edit_cpu_v1",
               "targeted_edit_cpu_v2", "targeted_edit_v2_edits")
    paths += ["looking_glass/" + module + ".py" for module in modules]
    for relative in paths:
        source, target = ROOT / relative, checkout / relative
        assert source.is_file(), f"missing rehearsal source: {relative}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()
    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("config", "user.name", "No-forward test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-qm", "source rehearsal fixture")
    return checkout, git("rev-parse", "HEAD")


@pytest.mark.skipif(not _matches_recorded_host() or not shutil.which("git"),
                    reason="requires the recorded Windows CPU runtime and Git")
def test_real_source_bound_supervised_child_has_no_model_forward(tmp_path):
    checkout, revision = checkout_for_rehearsal(tmp_path)
    work, study = tmp_path / "work", tmp_path / "study"
    work.mkdir()
    study.mkdir()
    shadow = tmp_path / "shadow-executed"
    (checkout / "platform.py").write_text("open(" + repr(str(shadow)) + ", 'w').close()\n")
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    site = Path(torch.__file__).resolve().parent.parent
    command = [sys._base_executable, "-I", "-S", "-B", str(checkout / PARENT),
               "--checkout", str(checkout), "--venv-root", str(Path(sys.prefix).resolve()),
               "--dependency", str(site), "--work-dir", str(work), "--study-root", str(study),
               "--timeout", "120"]
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "matching_v1_runtime" and report["source_revision"] == revision
    assert report["owner_pid"] == report["reaped_child_pid"] and report["owner_run_id"] == report["supervisor_run_id"]
    assert report["clean"] is True and report["job_drained"] is True
    assert report["claim_mask"] == "" and report["model_import_mask"] == "-1"
    assert report["checkpoint_loaded"] is False and report["model_forwarded"] is False
    assert report["parent_death_source_protection"] is False and report["study_ready"] is False
    assert not shadow.exists()
    assert not (study / "cpu-study.unsafe.json").exists()


@pytest.mark.skipif(not _matches_recorded_host() or not shutil.which("git"),
                    reason="requires the recorded Windows CPU runtime and Git")
def test_lost_real_supervisor_return_retains_source_and_cache(tmp_path):
    checkout, revision = checkout_for_rehearsal(tmp_path)
    work, study = tmp_path / "work", tmp_path / "study"
    work.mkdir()
    study.mkdir()
    site = Path(torch.__file__).resolve().parent.parent
    attempt = uuid.uuid4().hex
    driver = textwrap.dedent("""
        import importlib.util, json, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('test_rehearsal', sys.argv[1])
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        observed = []
        def trace(frame, event, arg):
            if (event == 'return' and frame.f_code.co_name == 'run_cpu_study'
                    and frame.f_code.co_filename.endswith('cpu_study_supervision.py') and arg is not None):
                observed.append(arg)
                sys.settrace(None)
                raise KeyboardInterrupt('lost supervisor return')
            return trace
        try:
            sys.settrace(trace)
            owner.run(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]),
                      Path(sys.argv[6]), attempt_id=sys.argv[7], timeout=120)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('supervisor return was not interrupted')
        finally:
            sys.settrace(None)
        q = owner.get_quarantine(sys.argv[7])
        assert q and not q.running and q.launch_started and not q.complete
        assert q.source_lease._active and q.manifest_lease._active and q.cache._active
        try:
            q.release()
        except owner.RehearsalError:
            pass
        else:
            raise AssertionError('uncertain quarantine was released')
        assert len(observed) == 1 and observed[0].clean and observed[0].exit_code == 0
        assert q.job._confirmed and q.job._handle is None
        hashes = owner._bounded_json(q.manifest, 2 * 1024 * 1024)
        reference = owner._bounded_json(Path(sys.argv[2]) / owner._REFERENCE, 65536)
        child = owner._bounded_json(q.report, 1024 * 1024)
        owner._check_child_report(child, observed[0], sys.argv[8], hashes, reference)
        # Test-only cleanup: this driver captured and independently checked the
        # otherwise-lost exact return, so it can dispose its scratch resources.
        q.complete = True
        q.release()
        print(json.dumps({'retained': True, 'released_after_test_verification': True}))
    """)
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    command = [sys._base_executable, "-I", "-S", "-B", "-c", driver, str(checkout / PARENT),
               str(checkout), str(Path(sys.prefix).resolve()), str(site), str(work), str(study),
               attempt, revision]
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"retained": True, "released_after_test_verification": True}
