"""Fabricated durable-outcome checks; no study inputs or model work."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from contextlib import ExitStack
import shutil
from types import SimpleNamespace

import pytest

from looking_glass import registry


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2_owner_receipt_test", ROOT / "scripts" / "run_synthetic_v2.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def _make_saved(output, *, omit=None):
    output.mkdir()
    revision = "1" * 40
    sources = {"looking_glass/model.py": "2" * 64,
               "looking_glass/causal_v2_study_entry.py": "3" * 64,
               "docs/CAUSAL_CONFIRMATION_V2.md": "4" * 64}
    states = [{"seed": seed, "sha256": "3" * 64} for seed in range(31201, 31207)]
    pre = registry._json_bytes({"schema": "synthetic-v2-pre-manifest-v1", "mode": "study",
        "context": {"protocol_id": "synthetic-causal-v2", "protocol_sha256": "4" * 64,
                    "source_revision": revision,
                    "model_state_hashes": states}, "source_sha256": sources})
    pre_sha = hashlib.sha256(pre).hexdigest()
    identity = {"protocol": {"id": "synthetic-causal-v2", "pre_manifest_sha256": pre_sha},
                "inputs": {"id": "fabricated"}, "model": {"states": states},
                "code": {"source_sha256": sources}, "metadata": {"scope": "test"}}
    private_names = sorted(owner._PRIVATE_ARTIFACTS - ({omit} if omit else set()))
    chunks = sorted(name for name in private_names if name.startswith("cell-"))
    private_identity = dict(identity)
    private_identity["protocol"] = {**identity["protocol"], "v2_evidence": {
        "schema": "causal-v2-evidence-layout-v1", "chunks": chunks,
        "external_reserve_bytes": 128 * 1024**2, "disk_limit_bytes": 1024**3}}
    limits = {name: 1024 for name in private_names}
    private = registry.create_run(output / "private", **private_identity, artifact_limits=limits,
                                  required_artifacts=private_names, max_total_bytes=sum(limits.values()))
    for name in private_names:
        private.write_artifact(name, b"{}")  # Structurally named placeholders, not study results.
    private.finalize("succeeded", {"scope": "private evidence", "submissions": 57408, "root_forwards": 57408})
    private_raw = (output / "private" / "outcome.json").read_bytes()
    public_names = sorted(owner._PUBLIC_ARTIFACTS)
    limits = {name: 4096 for name in public_names}
    public = registry.create_run(output / "public", **identity, artifact_limits=limits,
                                 required_artifacts=public_names, max_total_bytes=sum(limits.values()))
    for name in public_names:
        if name == "pre_manifest.json":
            data = pre
        elif name.startswith("source_inventory_"):
            stage = name.removeprefix("source_inventory_").removesuffix(".json")
            data = registry._json_bytes({"schema": "synthetic-v2-source-inventory-v1", "stage": stage,
                                         "pre_manifest_sha256": pre_sha,
                                         "source_sha256": {key: value for key, value in sources.items()
                                                           if key.startswith("looking_glass/")}})
        else:
            data = b"{}"  # The parent does not claim semantic/scientific replay.
        public.write_artifact(name, data)
    public.finalize("succeeded", {"scope": "complete study", "submissions": 57408, "root_forwards": 57408,
        "private_outcome_sha256": hashlib.sha256(private_raw).hexdigest()})
    return output, revision, sources


@pytest.fixture(scope="module")
def saved_template(tmp_path_factory):
    return _make_saved(tmp_path_factory.mktemp("owner-receipts") / "study")


@pytest.fixture
def saved(tmp_path, saved_template):
    template, revision, sources = saved_template
    output = tmp_path / "study"
    shutil.copytree(template, output)
    return output, revision, sources


def test_readback_uses_registry_inventory_and_saved_counts(saved):
    output, revision, sources = saved
    class Launcher:
        LaunchError = RuntimeError
        _checked_path = staticmethod(lambda path: path.resolve(strict=True))
    source = ROOT / "looking_glass" / "registry.py"
    verifier = owner._registry(ROOT, Launcher, {"looking_glass/registry.py": hashlib.sha256(source.read_bytes()).hexdigest()})
    assert owner._verified_success(verifier, output, revision, sources) == (57408, 57408)


def test_readback_rejects_changed_artifact(saved):
    output, revision, sources = saved
    (output / "private" / "artifacts" / "cell-00-d.json").write_bytes(b"tampered")
    with pytest.raises(registry.RegistryIntegrityError):
        owner._verified_success(registry, output, revision, sources)


def test_readback_rejects_wrong_source_revision(saved):
    output, _, sources = saved
    with pytest.raises(ValueError, match="source identity"):
        owner._verified_success(registry, output, "3" * 40, sources)


def test_readback_rejects_self_consistent_but_truncated_private_registry(tmp_path):
    output, revision, sources = _make_saved(tmp_path / "study", omit="cell-53-h.json")
    assert registry.verify_run(output / "private")["state"] == "succeeded"
    with pytest.raises(ValueError, match="artifact inventory"):
        owner._verified_success(registry, output, revision, sources)


@pytest.mark.parametrize("field,value", [("root_forwards", True), ("submissions", 57407),
                                       ("private_outcome_sha256", "0" * 64)])
def test_readback_rejects_incorrect_or_noninteger_terminal_fields(saved, field, value):
    output, revision, sources = saved
    path = output / "public" / "outcome.json"
    outcome = json.loads(path.read_bytes())
    outcome["details"][field] = value
    path.write_bytes(registry._json_bytes(outcome))
    with pytest.raises(ValueError, match="terminal counts or private outcome"):
        owner._verified_success(registry, output, revision, sources)


@pytest.fixture
def fake_owner(tmp_path, monkeypatch):
    """Replace the child and job with inert fakes; never launch study work."""
    checkout, work = tmp_path / "checkout", tmp_path / "work"
    checkout.mkdir()
    work.mkdir()
    events, leases = [], []

    class FakeError(RuntimeError):
        def __init__(self, message, child_reaped=None, *, status="failed", quarantine=None):
            super().__init__(message)
            self.child_reaped, self.status, self.quarantine = child_reaped, status, quarantine

    class Lease:
        def __init__(self, *args):
            self.active = False
            leases.append(self)
        def __enter__(self):
            self.active = True
            return self
        def __exit__(self, *args):
            self.active = False
        def handles(self):
            return [1]

    class Cache(Lease):
        def __init__(self, temporary):
            super().__init__()
            self.prefix = temporary / "cache"
        def __enter__(self):
            super().__enter__()
            self.prefix.mkdir()
            return self.prefix

    class Job:
        def assign(self, child):
            pass

    class Quarantine:
        def __init__(self, temporary, manifest):
            self.stack = ExitStack()
            self.source_lease = self.manifest_lease = self.cache = self.job = self.child = None
            self.child_reaped = None
            self.fail_release = False
        def retry_cleanup(self):
            events.append("release")
            if self.fail_release:
                raise OSError("cleanup unconfirmed")
            self.stack.close()

    class Child:
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
        def wait(self, timeout):
            return 0

    guards = SimpleNamespace(SourceFileLease=Lease, PrivateEmptyCache=Cache,
                             WindowsJob=Job, duplicate_to_process=lambda handles, child: [7])
    revision, sources = "1" * 40, {"looking_glass/registry.py": "2" * 64}

    def checked(path):
        if Path(path) == Path(owner.__file__):
            return checkout / owner._EXTRA[0]
        return Path(path).resolve(strict=True)

    def drain(child, job):
        assert all(lease.active for lease in leases)
        events.append("drain")
        return True

    launcher = SimpleNamespace(LaunchError=FakeError, _check_startup=lambda *args: Path("python.exe"),
        _checked_path=checked, _parent_platform_observation=lambda: ["Windows", "host", "", "", "AMD64"],
        _revision=lambda checkout: revision, _committed_sources=lambda *args, **kwargs: sources,
        _verify_sources=lambda *args: sources, _load_guards=lambda *args: guards,
        LaunchQuarantine=Quarantine, _check_single_threaded=lambda: None,
        _SCRIPTS=("scripts/v2_source_launcher.py", "scripts/v2_source_bootstrap.py"),
        _TOKEN=b"start\n", _stop_and_reap=drain)
    def audit(*args):
        assert events == ["drain"] and all(lease.active for lease in leases)
        events.append("audit")
        return 57408, 57408

    monkeypatch.setattr(owner, "_STUDY_LAUNCH_ENABLED", True)
    monkeypatch.setattr(owner, "_launcher", lambda: launcher)
    monkeypatch.setattr(owner, "_registry", lambda *args: object())
    monkeypatch.setattr(owner, "_verified_success", audit)
    # This fixture simulates a Windows owner; no native process is created.
    monkeypatch.setattr(owner.subprocess, "DETACHED_PROCESS", 0x00000008, raising=False)
    monkeypatch.setattr(owner.subprocess, "Popen", Child)
    return SimpleNamespace(checkout=checkout, work=work, output=tmp_path / "output", launcher=launcher,
                           events=events, leases=leases, quarantine_type=Quarantine)


def _run_fake(fake):
    return owner.run(fake.checkout, fake.work, (fake.work,), fake.work, fake.output,
                     rehearsal=False, timeout=10)


def test_owner_audits_only_after_job_drain_and_before_lease_release(fake_owner):
    report = _run_fake(fake_owner)
    assert fake_owner.events == ["drain", "audit", "release"]
    assert report["submissions"] == report["root_forwards"] == 57408
    assert report["status"] == "terminal_verified"
    assert report["scientific_replay_verified"] is False
    assert "study_ready" not in report
    assert all(not lease.active for lease in fake_owner.leases)


def test_failed_drain_does_not_audit_or_claim_saved_success(fake_owner):
    def fail_drain(child, job):
        fake_owner.events.append("drain_failed")
        raise OSError("unconfirmed job")
    fake_owner.launcher._stop_and_reap = fail_drain
    with pytest.raises(fake_owner.launcher.LaunchError) as error:
        _run_fake(fake_owner)
    assert fake_owner.events == ["drain_failed", "release"]
    assert error.value.recorded_success is None


def test_post_success_cleanup_failure_preserves_recorded_success(fake_owner):
    base = fake_owner.quarantine_type
    class Uncertain(base):
        def __init__(self, temporary, manifest):
            super().__init__(temporary, manifest)
            self.fail_release = True
    fake_owner.launcher.LaunchQuarantine = Uncertain
    with pytest.raises(fake_owner.launcher.LaunchError) as error:
        _run_fake(fake_owner)
    assert fake_owner.events == ["drain", "audit", "release", "release"]
    assert error.value.recorded_success is True
    assert error.value.quarantine is not None


@pytest.mark.parametrize("outcome", ["nonzero", "timeout", "cancel"])
def test_nonzero_or_interrupted_child_reconciles_success_only_after_drain(fake_owner, monkeypatch, outcome):
    class ChildAfterTerminal:
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
        def wait(self, timeout):
            fake_owner.output.mkdir()
            if outcome == "timeout":
                raise owner.subprocess.TimeoutExpired("fabricated-child", timeout)
            if outcome == "cancel":
                raise KeyboardInterrupt()
            return 1
    monkeypatch.setattr(owner.subprocess, "Popen", ChildAfterTerminal)
    with pytest.raises(fake_owner.launcher.LaunchError) as error:
        _run_fake(fake_owner)
    assert fake_owner.events == ["drain", "audit", "release"]
    assert error.value.recorded_success is True
    assert error.value.status == ("cancelled" if outcome == "cancel" else "failed")


def test_nonzero_child_with_invalid_terminal_does_not_claim_success(fake_owner, monkeypatch):
    class ChildAfterFailure:
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
        def wait(self, timeout):
            fake_owner.output.mkdir()
            return 1
    def bad_terminal(*args):
        assert fake_owner.events == ["drain"] and all(lease.active for lease in fake_owner.leases)
        fake_owner.events.append("audit_failed")
        raise ValueError("invalid saved terminal")
    monkeypatch.setattr(owner.subprocess, "Popen", ChildAfterFailure)
    monkeypatch.setattr(owner, "_verified_success", bad_terminal)
    with pytest.raises(fake_owner.launcher.LaunchError) as error:
        _run_fake(fake_owner)
    assert fake_owner.events == ["drain", "audit_failed", "release"]
    assert error.value.recorded_success is None
    assert isinstance(error.value.__cause__.__cause__, ValueError)


def test_fresh_system_exit_in_drain_outranks_earlier_timeout(fake_owner, monkeypatch):
    class TimedOutChild:
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
        def wait(self, timeout):
            raise owner.subprocess.TimeoutExpired("fabricated-child", timeout)
    def exit_during_drain(child, job):
        fake_owner.events.append("drain_cancelled")
        raise SystemExit(7)
    monkeypatch.setattr(owner.subprocess, "Popen", TimedOutChild)
    fake_owner.launcher._stop_and_reap = exit_during_drain
    with pytest.raises(fake_owner.launcher.LaunchError) as error:
        _run_fake(fake_owner)
    assert fake_owner.events == ["drain_cancelled", "release"]
    assert error.value.status == "cancelled"
    assert error.value.recorded_success is None
    assert isinstance(error.value.__cause__, SystemExit)
    assert error.value.__cause__.code == 7
    assert isinstance(error.value.__cause__.__cause__, owner.subprocess.TimeoutExpired)


@pytest.mark.parametrize("stage", ["write", "close"])
def test_interrupted_start_delivery_reconciles_after_drain(fake_owner, monkeypatch, stage):
    class StartStream:
        def write(self, frame):
            if stage == "write":
                fake_owner.output.mkdir()
                raise OSError("acknowledgment lost")
            return len(frame)
        def close(self):
            if stage == "close":
                fake_owner.output.mkdir()
                raise KeyboardInterrupt()
    class StartedChild:
        def __init__(self, *args, **kwargs):
            self.stdin = StartStream()
        def wait(self, timeout):
            raise AssertionError("wait must not run after interrupted handoff")
    monkeypatch.setattr(owner.subprocess, "Popen", StartedChild)
    with pytest.raises(fake_owner.launcher.LaunchError) as error:
        _run_fake(fake_owner)
    assert fake_owner.events == ["drain", "audit", "release"]
    assert error.value.recorded_success is True
    assert error.value.status == ("cancelled" if stage == "close" else "failed")
