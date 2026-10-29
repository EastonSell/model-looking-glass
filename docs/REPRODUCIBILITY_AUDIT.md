# Audit of the preserved CPU v1 records

This is a read-only audit of saved evidence, not an independent execution of the model. The protocol's synthetic residual network ran on CPU; none of these observations establishes a causal explanation or a result for a pretrained transformer. The results and historical logs remain unchanged.

## Checks that can be repeated

From a source checkout, using Python 3.10 or later and only its standard library:

```sh
python -m looking_glass.reproduce results/verified-cpu-v1
python -m looking_glass.reproduce results/initial-cpu-v1
python -m unittest discover -s tests -p test_reproduce.py -v
```

The verifier accepts the frozen **full protocol v1**, not the quick smoke run or a later protocol. It verifies the expected widths, seeds, placements, modes, warmups, repeat counts and case identities; rejects missing or duplicate cases; and compares each indexed raw CSV timing with JSON. It recomputes each cell's median and p90 from the nanosecond samples, checks the summary CSV against those results, and computes the table's medians and ratios from the four seed-specific cell medians. It checks saved case counts, boolean consistency, localization intervals and expected recorded-value sizes. It also compares each recorded source hash with bytes under `looking_glass/`; `--source-dir` selects another package source directory.

The source comparison is separate from record consistency: an earlier run can have internally consistent data when the current source has since changed. A missing or mismatching source file is displayed in `source_hashes`; it is not represented as a successful source match and does not by itself make the command fail. Malformed or inconsistent records produce an error and nonzero exit. The source hashes and the results are held in the same untrusted record; matching them proves byte agreement with the supplied files, not that those files generated the results, or that the results were not altered. The verifier never executes the model, verifies the machine was idle, or establishes the truth of saved booleans.

## Findings

Both saved full runs contain 56 unique timing cells and 5,600 indexed samples. Each raw CSV sample agrees with JSON, and every per-cell median, p90 and summary CSV entry agrees with recomputation. The verified run's table can be reconstructed: the median of the four per-seed gradient/direct ratios is approximately 2.7919, 3.8088, 2.5999 and 3.6395 for width/placement 16/sparse, 16/dense, 64/sparse and 64/dense, respectively. These pairings are **by model/input seed**, not temporally paired observations. Each timing cell ran as a block of 100 calls, and the host was not exclusively reserved. The values are descriptive and do not establish a general speed ratio or a confidence interval.

The final record reports 48/48 transparent training cases and 16/16 equal-capture cases containing 51,200 elements; both runs report 96/96 detections per placement, with 96/96 dense first observations at the injected tap and 96/96 sparse first observations later than the injected tap. The saved booleans and intervals support reproducing these *reported counts*, but no activation tensors, gradients, or model weights were stored, so their equality cannot be independently recomputed from the artifacts. Branch-control values are similarly saved only as booleans. The fixed perturbations were deliberately placed immediately before known observation points; localization is an instrumentation sanity check rather than a discovered mechanism.

At audit time the five recorded source/protocol SHA-256 values for `verified-cpu-v1` match the published snapshot, and both root and packaged protocol files have SHA-256 `07586071c05174315466f90c5d7552e40a6e46a22d3b4b588e1bcb64b5d77fd7`. For `initial-cpu-v1`, the four other hashes match, while `experiment.py` records `a4edcb211d82a4aa72256b97b29cf056fd308147e376b89699a7fda05b4fce59` instead of the published snapshot's `7d6f0a9591cf808358dad3fc94672c5e04868ef273ea4cded2a4c938dd3d0d5d`. The recorded local measurement commits `38ff944e46f05ec06a58e07b2905f8b6bff6c4a3` and `68d17069a67afc2a23df28d72c9b3e9b64811b09` are unavailable in this checkout; the published history is a content snapshot. Their absence here does not prove they cannot be recovered elsewhere.

The five final-run files are recoverable from published snapshot commit `e94e20660cff32ddcd2643307d60e8d9a3c74f2f`, even if newer work has changed the current package. To compare against that snapshot without changing the current checkout, create a separate worktree and point the verifier at its package directory:

```sh
git -c core.autocrlf=false worktree add --detach ../historical-cpu-v1 e94e20660cff32ddcd2643307d60e8d9a3c74f2f
python -m looking_glass.reproduce results/verified-cpu-v1 --source-dir ../historical-cpu-v1/looking_glass
```

The destination must not already exist. The verifier hashes actual checkout bytes; line-ending conversion can produce a mismatch even when the underlying Git blob matches. The command disables automatic CRLF conversion for this checkout. This is a byte comparison to the published snapshot; it does not recreate or independently rerun the historical local commit.

## Limits of replay

The recorded environment identifies Linux, Python 3.12.14 and PyTorch 2.15.0+cpu, with CPU float32 and one intra-op thread; it does not record the CPU model, full OS image, complete dependency artifacts, scheduler state or concurrent host activity. The present Windows environment is different. Exact historical timing replay is therefore not established, and even a fresh same-version run on another host would not reproduce the old wall-clock samples. The original initial-run `experiment.py` is not available in the checked-out snapshot, so its original source cannot currently be independently replayed from this checkout. Matching final-run source bytes are useful for reconstruction, but no independent full run was performed in this audit.

For a future independent replication, obtain and verify the original initial-run source if available; capture exact source revision and hashes, immutable dependencies and environment/hardware metadata; prespecify a new protocol and holdout plan; then save a new run and its failures separately. Never overwrite or pool these historical results into a fresh run. Timing comparisons require their own controlled measurement environment and appropriate paired design.
