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
    assert "https://download.pytorch.org/whl/cpu" in out.stdout
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
    monkeypatch.setattr(module, "sys", SimpleNamespace(version_info=(3, 10, 15)))
    with pytest.raises(SystemExit): module.main(["--no-launch"])
    assert not (tmp_path / ".venv").exists()
