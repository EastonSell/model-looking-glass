"""Make the frozen study's scripts importable in its published test layout."""
from pathlib import Path
import sys

scripts = Path(__file__).resolve().parents[2] / "scripts"
if scripts.is_dir():
    sys.path.insert(0, str(scripts))
