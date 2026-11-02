"""Direct-child launch/grant tests using stdlib-only fabricated processes."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/run_flan_multitoken_sentiment_v2.py"
spec = importlib.util.spec_from_file_location("multitoken_operator_test", SCRIPT)
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


def fake_child_code(sleep=False):
    return """
import hashlib, json, os, pathlib, sys, time
directory, revision, binding, subject = sys.argv[1:5]
grant = sys.stdin.buffer.read(33)
if len(grant) != 32 or hashlib.sha256(grant).hexdigest() != sys.argv[-1]:
    raise SystemExit(9)
print(json.dumps({'pid': os.getpid(), 'parent_pid': os.getppid()}), flush=True)
if %r:
    time.sleep(2.5)
    pathlib.Path(directory, 'late-marker').write_text('unexpected')
else:
    record = {'schema':'flan-multisubtoken-owner-receipt-v1', 'subject':subject,
              'status':'no-selected-groups', 'model_loaded':False,
              'source_revision':revision, 'binding_sha256':binding}
    pathlib.Path(directory, 'owner-receipt.json').write_text(json.dumps(record))
""" % sleep


def test_direct_base_child_reports_the_observed_pid_and_is_reaped(tmp_path):
    directory = tmp_path / "story"
    directory.mkdir()
    prefix = directory / "prefix"
    prefix.mkdir()
    revision, binding = "a" * 40, "b" * 64
    args = SimpleNamespace(timeout_seconds=10, revision=revision, binding_sha256=binding, venv=tmp_path)
    source = {"revision": revision}
    owner = SimpleNamespace(source_snapshot=lambda *a: source)
    def identity(path, venv, pid):
        record = json.loads(path.read_text().splitlines()[0])
        return record == {"pid": pid, "parent_pid": os.getpid()}
    runtime = SimpleNamespace(_check_child_identity=identity)
    command = [sys._base_executable, "-I", "-S", "-B", "-c", fake_child_code(),
               str(directory), revision, binding, "story"]
    result = operator._supervise(command, args=args, root=tmp_path, owner=owner, runtime=runtime,
                                 source=source, subject="story", directory=directory, prefix=prefix)
    receipt = json.loads((directory / "operator-exit.json").read_text())
    assert result["status"] == "no-selected-groups"
    assert receipt["direct_child_exit_code"] == 0 and receipt["child_reaped"] is True
    assert receipt["terminal_consistent"] is True and receipt["log_drain_complete"] is True


def test_timeout_kills_the_actual_direct_child_before_a_delayed_action(tmp_path):
    directory = tmp_path / "story"
    directory.mkdir()
    prefix = directory / "prefix"
    prefix.mkdir()
    args = SimpleNamespace(timeout_seconds=1, revision="a" * 40, binding_sha256="b" * 64, venv=tmp_path)
    owner = SimpleNamespace(source_snapshot=lambda *a: {})
    runtime = SimpleNamespace(_check_child_identity=lambda *a: True)
    command = [sys._base_executable, "-I", "-S", "-B", "-c", fake_child_code(True),
               str(directory), args.revision, args.binding_sha256, "story"]
    with pytest.raises(RuntimeError, match="incomplete"):
        operator._supervise(command, args=args, root=tmp_path, owner=owner, runtime=runtime,
                            source={}, subject="story", directory=directory, prefix=prefix)
    receipt = json.loads((directory / "operator-exit.json").read_text())
    assert receipt["timed_out"] is True and receipt["child_reaped"] is True
    time.sleep(2)
    assert not (directory / "late-marker").exists()


def test_direct_child_without_a_complete_grant_does_no_work(tmp_path):
    marker = tmp_path / "after-grant"
    code = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('isolated_operator', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module._receive_grant('0' * 64)
pathlib.Path(sys.argv[2]).write_text('unexpected')
"""
    child = subprocess.Popen([sys._base_executable, "-I", "-S", "-B", "-c", code,
                              str(SCRIPT), str(marker)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, _ = child.communicate(input=b"short", timeout=10)
    assert child.returncode != 0 and not marker.exists()


def test_cancellation_priority_is_preserved_during_finalization():
    first = KeyboardInterrupt("first")
    with pytest.raises(KeyboardInterrupt) as caught:
        operator._raise_with_priority(first, OSError("later"))
    assert caught.value is first and isinstance(caught.value.__cause__, OSError)
    second = KeyboardInterrupt("later")
    with pytest.raises(KeyboardInterrupt) as caught:
        operator._raise_with_priority(OSError("first"), second)
    assert caught.value is second and isinstance(caught.value.__cause__, OSError)


def test_parent_rejects_bad_output_volume_before_creating_attempt(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    observed = []
    def reject(path):
        observed.append(path)
        raise ValueError("resolved output volume differs")
    fake_owner = SimpleNamespace(source_snapshot=lambda *a: {}, ensure_output_directory=reject)
    fake_runtime = SimpleNamespace(_dependency_site=lambda *a: (tmp_path, None))
    monkeypatch.setattr(operator, "_empty_prefix", lambda: None)
    monkeypatch.setattr(operator, "_clear_git_environment", lambda: None)
    monkeypatch.setattr(operator, "_load_owner", lambda *a: fake_owner)
    monkeypatch.setattr(operator, "_load_runtime_helper", lambda *a: fake_runtime)
    command = ["--revision", "a" * 40, "--binding-sha256", "b" * 64]
    options = {"repo": SCRIPT.parents[1], "venv": tmp_path, "asset-dir": tmp_path,
               "attempt": attempt, "binding": tmp_path / "binding", "discovery": tmp_path / "discovery",
               "painting": tmp_path / "painting", "selection-review": tmp_path / "review"}
    for name, value in options.items():
        command.extend(["--" + name, str(value)])
    with pytest.raises(ValueError, match="output volume"):
        operator.main(command)
    assert observed == [tmp_path] and not attempt.exists()


def test_isolated_source_bound_operator_runs_a_no_model_child(tmp_path):
    repo = tmp_path / "checkout"
    (repo / "tools").mkdir(parents=True)
    (repo / "looking_glass").mkdir()
    (repo / "tools/run_flan_multitoken_sentiment_v2.py").write_bytes(SCRIPT.read_bytes())
    helper = subprocess.check_output(["git", "show", "HEAD:tools/run_flan_sentiment_selector_stage_v1.py"],
                                     cwd=SCRIPT.parents[1])
    (repo / "tools/run_flan_sentiment_selector_stage_v1.py").write_bytes(helper)
    (repo / "looking_glass/__init__.py").write_bytes(b"")
    (repo / "looking_glass/flan_multitoken_owner_v2.py").write_text("""
import json
from pathlib import Path
def source_snapshot(root, revision):
    return {'revision': revision}
def ensure_output_directory(path):
    return Path(path).resolve(strict=True)
def run(args):
    result = {'schema': 'flan-multisubtoken-owner-receipt-v1', 'subject': args.subject,
              'status': 'no-selected-groups', 'model_loaded': False,
              'source_revision': args.revision, 'binding_sha256': args.binding_sha256}
    Path(args.attempt, 'owner-receipt.json').write_text(json.dumps(result))
    return result
""")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Synthetic reviewer"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "synthetic@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tools", "looking_glass"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fake bootstrap"], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
    prefix = tmp_path / "parent-prefix"
    prefix.mkdir()
    attempt = tmp_path / "attempt"
    command = [sys._base_executable, "-I", "-S", "-B", "-X", "pycache_prefix=" + str(prefix),
               str(repo / "tools/run_flan_multitoken_sentiment_v2.py"), "--revision", revision,
               "--binding-sha256", "b" * 64, "--timeout-seconds", "20"]
    options = {"repo": repo, "venv": Path(sys.prefix), "asset-dir": tmp_path, "attempt": attempt,
               "binding": tmp_path / "binding", "discovery": tmp_path / "discovery",
               "painting": tmp_path / "painting", "selection-review": tmp_path / "review"}
    for name, value in options.items():
        command.extend(["--" + name, str(value)])
    completed = subprocess.run(command, cwd=repo, capture_output=True, timeout=40)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    summary = json.loads((attempt / "operator-summary.json").read_text())
    assert summary["status"] == "no-selected-groups"
    assert [item["subject"] for item in summary["passes"]] == ["story"]
    exit_record = json.loads((attempt / "story/operator-exit.json").read_text())
    assert exit_record["direct_child_exit_code"] == 0 and exit_record["child_reaped"] is True
