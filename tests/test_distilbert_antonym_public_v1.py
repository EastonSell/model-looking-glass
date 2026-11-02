"""Public complete-row and terminal-SEP prefix comparison contracts."""
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys


def test_final_sep_removal_detects_a_prefix_in_either_direction():
    name = 'looking_glass.distilbert_antonym_public_v1'
    assert importlib.util.find_spec(name) is not None, 'antonym public comparator is not implemented'
    module = importlib.import_module(name)
    a, b = [101, 103, 102], [101, 103, 200, 102]
    assert module.related_any(a, b)
    assert module.related_any(b, a)
    assert not module.related_any(a, [101, 104, 102])


def test_public_comparison_cli_imports_from_an_isolated_checkout():
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, '-I', '-B', str(repo / 'scripts/check_distilbert_antonym_public_v1.py'), '--help'],
                          cwd=repo, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert '--cases' in proc.stdout
