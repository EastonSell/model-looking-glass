"""Start the study from an isolated, source-bound Python interpreter."""

import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys

if (not sys.flags.isolated or not sys.dont_write_bytecode or not sys.pycache_prefix
        or not os.path.isabs(sys.pycache_prefix) or os.path.lexists(sys.pycache_prefix)):
    raise SystemExit("use -I -B -X pycache_prefix=<fresh absent absolute path>")

root = Path(__file__).resolve().parents[1]
files = (
    "docs/VIT_SYNTHETIC_INJECTION_V1.md", "examples/vit_synthetic_injection_bootstrap.py",
    "examples/vit_synthetic_injection_cpu.py", "looking_glass/__init__.py",
    "looking_glass/vit_synthetic_injection_v1.py", "looking_glass/vit_trace.py",
    "looking_glass/vit_trace_comparison.py", "looking_glass/intervention_plan.py",
    "looking_glass/interventions.py", "looking_glass/model_ownership.py",
    "looking_glass/localization.py", "looking_glass/trace_schema.py", "looking_glass/agreement_numeric.py",
    "looking_glass/bert_matched_numeric.py", "looking_glass/vision_coco_cases.py",
    "tests/test_vit_synthetic_injection_v1.py",
)
if sys.argv[1:] != ["--help"]:
    if sys.argv.count("--published-source") != 1 or sys.argv.count("--git-executable") != 1:
        raise SystemExit("exactly one --published-source and --git-executable are required")
    index = sys.argv.index("--published-source")
    if index + 1 >= len(sys.argv):
        raise SystemExit("missing published source")
    commit = sys.argv[index + 1]
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SystemExit("published source must be a full lowercase commit ID")
    git_index = sys.argv.index("--git-executable")
    if git_index + 1 >= len(sys.argv):
        raise SystemExit("missing Git executable")
    git = Path(sys.argv[git_index + 1])
    if not git.is_absolute() or not git.is_file():
        raise SystemExit("Git executable must be an existing absolute path")
    git = git.resolve()
    if git.name.lower() not in ("git", "git.exe") or git.is_relative_to(root):
        raise SystemExit("Git executable must be outside the checkout")
    git_env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    if subprocess.check_output([str(git), "--no-replace-objects", "cat-file", "-t", commit],
                               cwd=root, env=git_env).strip() != b"commit":
        raise SystemExit("published source must name a commit object")
    for name in files:
        committed = subprocess.check_output([str(git), "--no-replace-objects", "show", f"{commit}:{name}"],
                                            cwd=root, env=git_env)
        if (root / name).read_bytes() != committed:
            raise SystemExit(f"source differs from the selected commit: {name}")

# Keep the checkout root off sys.path: ignored root files cannot shadow
# unrelated standard-library or dependency imports.
package = root / "looking_glass" / "__init__.py"
spec = importlib.util.spec_from_file_location("looking_glass", package,
                                             submodule_search_locations=[str(package.parent)])
module = importlib.util.module_from_spec(spec)
sys.modules["looking_glass"] = module
spec.loader.exec_module(module)
runpy.run_path(str(root / "examples" / "vit_synthetic_injection_cpu.py"), run_name="__main__")
