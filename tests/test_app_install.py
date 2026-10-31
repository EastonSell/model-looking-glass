"""Installer planning is side-effect free under --dry-run."""
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_is_explicit_and_no_environment_created(tmp_path):
    script = tmp_path / "install.py"
    script.write_bytes((ROOT / "install.py").read_bytes())
    out = subprocess.run([sys.executable, str(script), "--dry-run"], capture_output=True, text=True, check=True)
    flavor = "cu132" if sys.platform in ("linux", "win32") else "cpu"
    assert "https://download.pytorch.org/whl/" + flavor in out.stdout
    assert "https://pypi.org/simple" in out.stdout
    assert "[app]" in out.stdout and "looking_glass.app" in out.stdout
    assert not (tmp_path / ".venv").exists()


def test_existing_environment_requires_explicit_reuse(tmp_path):
    script = tmp_path / "install.py"; script.write_bytes((ROOT / "install.py").read_bytes())
    (tmp_path / ".venv").mkdir()
    out = subprocess.run([sys.executable, str(script), "--no-launch"], capture_output=True, text=True)
    assert out.returncode != 0 and "--reuse-venv" in out.stderr
    assert list((tmp_path / ".venv").iterdir()) == []


def test_application_assets_declared_for_package():
    project = (ROOT / "pyproject.toml").read_text()
    assert 'model-looking-glass = "looking_glass.app:main"' in project
    patterns = ["app_static/*.html", "app_static/*.js", "app_static/*.css"]
    assert all(pattern in project for pattern in patterns)
    for filename in ("index.html", "app.js", "app.css"):
        file = ROOT / "looking_glass" / "app_static" / filename
        assert file.is_file()
        assert any(file.match(pattern) for pattern in patterns)


def test_installer_rejects_python_310_before_creating_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    spec = importlib.util.spec_from_file_location("mlg_install_test", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "sys", SimpleNamespace(version_info=(3, 10, 15), platform=sys.platform))
    with pytest.raises(SystemExit): module.main(["--no-launch"])
    assert not (tmp_path / ".venv").exists()


def test_cuda_plan_uses_explicit_official_build(tmp_path):
    spec = importlib.util.spec_from_file_location("mlg_cuda_install_test", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    commands = module.plan(tmp_path, tmp_path / "python", "cuda", platform="linux")
    assert "https://download.pytorch.org/whl/cu132" in commands[0]
    assert "torch==2.15.0+cu132" in commands[0]
    assert "torch==2.15.0+cpu" in module.plan(tmp_path, tmp_path / "python", "cpu", platform="linux")[0]
    assert "torch==2.15.0" in module.plan(tmp_path, tmp_path / "python", "cpu", platform="darwin")[0]


def test_cuda_installer_rejects_unsupported_platform_before_write(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import pytest
    spec = importlib.util.spec_from_file_location("mlg_cuda_platform_test", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "sys", SimpleNamespace(version_info=(3, 12), platform="darwin"))
    with pytest.raises(SystemExit): module.main(["--backend", "cuda", "--no-launch"])
    assert "CUDA installer requires Linux or Windows" in capsys.readouterr().err
    assert not (tmp_path / ".venv").exists()


def test_cpu_backend_dry_run_and_unknown_backend(tmp_path):
    script = tmp_path / "install.py"; script.write_bytes((ROOT / "install.py").read_bytes())
    result = subprocess.run([sys.executable, str(script), "--dry-run", "--backend", "cpu"], capture_output=True, text=True)
    assert result.returncode == 0 and "/whl/cpu" in result.stdout
    bad = subprocess.run([sys.executable, str(script), "--backend", "arbitrary-url"], capture_output=True, text=True)
    assert bad.returncode != 0 and not (tmp_path / ".venv").exists()


def test_gpu_check_entrypoint_and_installer_hint():
    project = (ROOT / "pyproject.toml").read_text()
    assert 'model-looking-glass-gpu-check = "looking_glass.app_selftest:main"' in project
    out = subprocess.run([sys.executable, str(ROOT / "install.py"), "--dry-run", "--backend", "cpu"], capture_output=True, text=True, check=True)
    assert "looking_glass.app_selftest" in out.stdout


def installer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mlg_reuse_test", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    env = tmp_path / ".venv"; env.mkdir()
    (env / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    executable = env / ("Scripts/python.exe" if module.os.name == "nt" else "bin/python")
    executable.parent.mkdir(); executable.touch()
    return module, env


def test_child_validation_rejects_old_wrong_prefix_and_missing_pip(tmp_path, monkeypatch):
    import pytest
    module, env = installer(tmp_path, monkeypatch)
    valid = {"version": [3, 12, 0], "prefix": str(env), "base_prefix": "/base", "pip": True}
    for update, message in [({"version": [3, 10, 9]}, "3.11"), ({"prefix": "/wrong"}, "prefix"), ({"pip": False}, "pip"), ({"base_prefix": str(env)}, "isolated")]:
        monkeypatch.setattr(module, "_probe_child", lambda *args, data={**valid, **update}: data)
        with pytest.raises(module.InstallError, match=message): module.probe_environment(env, env / ("Scripts/python.exe" if module.os.name == "nt" else "bin/python"))


def test_system_site_rejected_before_child_or_pip(tmp_path, monkeypatch):
    import pytest
    module, env = installer(tmp_path, monkeypatch)
    (env / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    monkeypatch.setattr(module, "_probe_child", lambda *args: pytest.fail("must reject before child"))
    with pytest.raises(module.InstallError, match="system-site"): module.probe_environment(env, env / ("Scripts/python.exe" if module.os.name == "nt" else "bin/python"))


def test_pip_failure_stops_next_install_and_launch_with_resume(tmp_path, monkeypatch, capsys):
    module, env = installer(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "probe_environment", lambda *args: {})
    calls = []
    def fail(command, **kwargs):
        calls.append(command); raise subprocess.CalledProcessError(17, command)
    monkeypatch.setattr(module.subprocess, "run", fail)
    monkeypatch.setattr(module.subprocess, "call", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not launch")))
    assert module.main(["--reuse-venv", "--backend", "cpu"]) == 17
    output = capsys.readouterr()
    assert len(calls) == 1 and "--reuse-venv --backend cpu" in output.err
    assert "Installed" not in output.out and env.exists()


def test_interruption_preserves_environment_and_returns_130(tmp_path, monkeypatch, capsys):
    module, env = installer(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "probe_environment", lambda *args: {})
    def interrupt(*args, **kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(module.subprocess, "run", interrupt)
    assert module.main(["--reuse-venv", "--backend", "cpu", "--no-launch"]) == 130
    assert env.exists() and "130" in capsys.readouterr().err


def test_dry_run_does_not_probe_create_or_install(tmp_path, monkeypatch):
    import pytest
    module, _ = installer(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "probe_environment", lambda *args: pytest.fail("no probe"))
    monkeypatch.setattr(module.venv.EnvBuilder, "create", lambda *args: pytest.fail("no creation"))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("no installation"))
    assert module.main(["--dry-run", "--backend", "cpu"]) == 0


def test_actual_existing_venv_probe_no_install():
    spec = importlib.util.spec_from_file_location("mlg_actual_probe", ROOT / "install.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module._probe_child(Path(sys.executable))
    assert report["version"] == list(sys.version_info[:3])
    assert report["prefix"] == sys.prefix and report["base_prefix"] == sys.base_prefix
    assert report["pip"] == (importlib.util.find_spec("pip") is not None)


def test_creation_error_preserves_partial_env_without_reuse_advice(tmp_path, monkeypatch, capsys):
    module, env = installer(tmp_path, monkeypatch)
    import shutil
    shutil.rmtree(env)
    def fail(builder, target):
        target.mkdir(); (target / "partial").write_text("keep")
        raise OSError("synthetic ensurepip failure")
    monkeypatch.setattr(module.venv.EnvBuilder, "create", fail)
    assert module.main(["--backend", "cpu", "--no-launch"]) == 1
    output = capsys.readouterr()
    assert "environment creation" in output.err and "resume explicitly" not in output.err
    assert (env / "partial").read_text() == "keep"


def test_probe_failure_never_calls_pip(tmp_path, monkeypatch, capsys):
    import pytest
    module, env = installer(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_probe_child", lambda *args: {"version": [3, 10, 9], "prefix": str(env), "base_prefix": "/base", "pip": True})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("no pip"))
    assert module.main(["--reuse-venv", "--backend", "cpu"]) == 1
    output = capsys.readouterr()
    assert "3.11" in output.err and "resume explicitly" not in output.err


def test_app_failure_distinguished_from_install_failure(tmp_path, monkeypatch, capsys):
    module, _ = installer(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "probe_environment", lambda *args: {})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.subprocess, "call", lambda *args, **kwargs: 9)
    assert module.main(["--reuse-venv", "--backend", "cpu"]) == 9
    output = capsys.readouterr()
    assert "Installation finished" in output.err and "exited with code 9" in output.err
    assert "resume explicitly" not in output.err


def test_probe_output_and_schema_are_bounded(tmp_path, monkeypatch):
    import pytest
    module, _ = installer(tmp_path, monkeypatch)
    actual = subprocess.Popen
    def child(command, **kwargs):
        return actual([sys.executable, "-I", "-c", "print('x' * 70000)"], **kwargs)
    monkeypatch.setattr(module.subprocess, "Popen", child)
    with pytest.raises(module.InstallError, match="output limit"):
        module._probe_child(Path(sys.executable))


def test_posix_command_quoting_matches_shell_tokens(tmp_path, monkeypatch):
    import shlex
    module, _ = installer(tmp_path, monkeypatch)
    if module.os.name != "nt":
        args = ["/path with space/python", "a'b", "$HOME", "semi;colon"]
        assert shlex.split(module.command_text(args)) == args


def test_probe_timeout_kills_child_and_reports_failure(tmp_path, monkeypatch):
    import io
    import pytest
    module, _ = installer(tmp_path, monkeypatch)
    class Process:
        stdout = io.BytesIO(b'{}')
        stopped = False
        def wait(self, timeout):
            if not self.stopped: raise subprocess.TimeoutExpired("synthetic probe", timeout)
            return -9
        def poll(self): return -9 if self.stopped else None
        def kill(self): self.stopped = True
    child = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(module.InstallError, match="15 seconds"):
        module._probe_child(Path(sys.executable))
    assert child.stopped


def test_probe_reader_failure_is_not_accepted_as_json(tmp_path, monkeypatch):
    import pytest
    module, _ = installer(tmp_path, monkeypatch)
    class Pipe:
        def read1(self, size): raise OSError("synthetic pipe failure")
        def close(self): pass
    class Process:
        stdout = Pipe()
        def wait(self, timeout): return 0
        def poll(self): return 0
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(module.InstallError, match="could not be read"):
        module._probe_child(Path(sys.executable))


def test_install_commands_receive_sanitized_environment_without_parent_mutation(tmp_path, monkeypatch):
    module, _ = installer(tmp_path, monkeypatch)
    monkeypatch.setenv("PIP_TARGET", "/outside")
    monkeypatch.setenv("PIP_PREFIX", "/different")
    monkeypatch.setenv("PYTHONPATH", "/injection")
    monkeypatch.setenv("EXAMPLE_PROXY_SETTING", "retained")
    before = dict(module.os.environ)
    monkeypatch.setattr(module, "probe_environment", lambda *args: {})
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert module.main(["--reuse-venv", "--backend", "cpu", "--no-launch"]) == 0
    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[1:6] == ["-I", "-m", "pip", "--isolated", "--require-virtualenv"]
        environment = kwargs["env"]
        assert environment["PIP_CONFIG_FILE"] == module.os.devnull
        assert "PIP_TARGET" not in environment and "PIP_PREFIX" not in environment and "PYTHONPATH" not in environment
        assert environment["EXAMPLE_PROXY_SETTING"] == "retained"
    assert dict(module.os.environ) == before


def test_actual_pip_config_inspection_ignores_conflicting_settings(tmp_path, monkeypatch):
    import json
    import pytest
    if importlib.util.find_spec("pip") is None: pytest.skip("actual pip config inspection requires pip; no installation performed")
    module, _ = installer(tmp_path, monkeypatch)
    conflict = tmp_path / "pip.conf"
    conflict.write_text("[global]\nindex-url = https://invalid.example/simple\n[install]\ntarget = /outside\nprefix = /outside-prefix\n")
    monkeypatch.setenv("PIP_CONFIG_FILE", str(conflict))
    monkeypatch.setenv("PIP_TARGET", "/environment-target")
    monkeypatch.setenv("PIP_INDEX_URL", "https://invalid.example/environment")
    # Inject conflicting global AND site file locations in this read-only
    # child inspector, so the test proves os.devnull disables both categories.
    code = ("import json,sys; import pip._internal.configuration as c; "
            "c.get_configuration_files=lambda:{c.kinds.GLOBAL:[sys.argv[1]],c.kinds.USER:[sys.argv[1]],c.kinds.SITE:[sys.argv[1]]}; "
            "cfg=c.Configuration(isolated=True); cfg.load(); print(json.dumps(dict(cfg.items())))")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(conflict)], env=module.pip_environment(),
                         capture_output=True, text=True, check=True, timeout=15)
    assert json.loads(out.stdout) == {}
    assert "target = /outside" in conflict.read_text()
