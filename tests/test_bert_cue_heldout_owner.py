import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
OWNER = REPO / "scripts" / "bert_cue_heldout_frames_v1.py"


def load_owner():
    if not OWNER.is_file():
        pytest.fail("held-out-frame owner is not implemented")
    spec = importlib.util.spec_from_file_location("heldout_owner", OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owner_import_does_not_import_torch_or_transformers():
    if not OWNER.is_file():
        pytest.fail("held-out-frame owner is not implemented")
    code = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('owner', {str(OWNER)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_owner_schedule_counts_dose_ineligible_contexts():
    owner = load_owner()
    cases = [{"dose_eligible": True}] * 5 + [{"dose_eligible": False}]
    assert owner.expected_forwards(cases) == 368


def test_owner_source_identity_includes_reused_and_new_scientific_code():
    owner = load_owner()
    paths = owner.prior_owner(REPO).SOURCE_PATHS
    assert "looking_glass/bert_cue_class.py" in paths
    assert "looking_glass/bert_cue_heldout_frames.py" in paths
    assert "scripts/bert_cue_heldout_frames_v1.py" in paths
    assert "experiments/bert_cue_heldout_frames_cases_v1.json" in paths
    assert len(paths) == len(set(paths))
