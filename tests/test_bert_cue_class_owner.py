import importlib.util
import pathlib
import subprocess
import sys

OWNER = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "bert_cue_class_v1.py"


def test_owner_import_does_not_import_torch_or_transformers():
    code = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('owner', {str(OWNER)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_schedule_total_accounts_for_ineligible_doses():
    spec = importlib.util.spec_from_file_location("owner", OWNER)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    assert owner.expected_forwards([{"dose_eligible": True}, {"dose_eligible": False}]) == 80
