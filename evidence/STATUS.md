# Verification status — 2026-10-28

## Current evidence

- Python 3.12.14; PyTorch 2.15.0+cpu; CPU eager float32.
- `python -m compileall -q looking_glass tests`: passed.
- `.venv/bin/python -m unittest discover -s tests -v`: **26 passed**, no skips; `tests-final.txt`.
- `.venv/bin/python -O -m unittest discover -s tests -v`: **26 passed**, no skips; `tests-final-optimized.txt`.
- Final full protocol: **48 transparent training comparisons, 16 equal channel comparisons, 192 perturbation trials, 56 timing cells / 5,600 raw samples**; `verified-full-run.txt` and `../results/verified-cpu-v1/`.
- Independent reviewer executed tests and verified a built wheel's bundled protocol and smoke execution outside the source checkout. See `../REVIEW.md` for exact scope and remaining limitations.
- PyTorch warns that optional NumPy is unavailable. The implementation and all recorded evaluations do not use NumPy; this warning was preserved, not hidden.
- Public publication is established by the remotely verified commit, not by this local status document.

## Historical evidence

Design and protocol were committed before the first full experiment. Initial tests were blocked at import (`ModuleNotFoundError: No module named torch`), retained in `initial-test-blocker.txt`. After the dependency became available, runtime tests exposed the reviewer's hook-cleanup regression; subsequent RED-to-GREEN records cover hook cleanup, ambient dtype, raw CSV, autocast and failure-record writing. Historical failure logs are not current test failures.

Tests were authored before implementation, but executable initial RED-to-GREEN development was impossible while PyTorch was absent. Later regression cycles were executed and retained. This is not represented as uninterrupted completed TDD.

Log paths are normalized to `<repo>` and `<python>` for portable public evidence; test outcomes, timing values and warnings are unchanged. Results use only synthetic inputs and do not contain activation tensors or model weights.
