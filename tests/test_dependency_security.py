"""Regression checks for the supported dependency security baseline.

These exercise trust gating without fetching or executing remote code; the
version checks do not substitute for upstream vulnerability reproductions.
"""
import importlib.metadata
from pathlib import Path
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


@pytest.mark.parametrize("name,floor", [
    ("transformers", "5.22.1"), ("Pillow", "13.0.0"), ("setuptools", "84.2.1"), ("pytest", "9.0.3"),
])
def test_installed_security_baseline(name, floor):
    assert Version(importlib.metadata.version(name)) >= Version(floor)


def test_active_manifest_excludes_vulnerable_dependency_baselines():
    manifest = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    build = {Requirement(s).name: Requirement(s) for s in manifest["build-system"]["requires"]}
    assert "78.1.0" not in build["setuptools"].specifier
    assert "84.2.1" in build["setuptools"].specifier
    runtime = {Requirement(s).name: Requirement(s) for s in manifest["project"]["dependencies"]}
    assert "78.1.0" not in runtime["setuptools"].specifier
    assert "84.2.1" in runtime["setuptools"].specifier
    for extra in ("app", "transformer", "test"):
        requirements = {Requirement(s).name: Requirement(s) for s in manifest["project"]["optional-dependencies"][extra]}
        if "transformers" in requirements:
            assert "4.57.3" not in requirements["transformers"].specifier
            assert "5.22.1" in requirements["transformers"].specifier
        if "Pillow" in requirements:
            assert "11.3.0" not in requirements["Pillow"].specifier
            assert "13.0.0" in requirements["Pillow"].specifier


@pytest.mark.parametrize("local", [False, True])
def test_custom_generation_declined_before_code_download_or_execution(monkeypatch, tmp_path, local):
    from transformers.generation import utils
    from transformers.generation.utils import GenerationMixin
    calls = []
    monkeypatch.setattr(utils, "has_file", lambda *a, **kw: True)
    def forbidden(*args, **kwargs):
        calls.append(args)
        pytest.fail("custom code must not be fetched or executed before consent")
    monkeypatch.setattr(utils, "get_cached_module_file", forbidden)
    monkeypatch.setattr(utils, "get_class_in_module", forbidden)
    path = str(tmp_path) if local else "untrusted/example"
    with pytest.raises(ValueError, match="trust_remote_code"):
        GenerationMixin.load_custom_generate(object(), path, trust_remote_code=False)
    assert calls == []
