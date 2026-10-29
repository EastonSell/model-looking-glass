# Auditing the DistilGPT-2 CPU batch evidence

This read-only, standard-library checker validates the fixed v1 manifest,
committed source and manifest blobs, recorded model-entry rows, terminal
records, report links and all six raw float32 ZIP members. It does not load a
model, initialize CUDA, or execute numerical tolerance comparisons. The
synthetic test fixture is not a model result.

For the saved v1 result, use a checkout containing the recorded Git history:

```sh
python -m looking_glass.distilgpt2_batch_cpu_audit \
  results/distilgpt2-batch-cpu-v1-20261028/run \
  --manifest results/distilgpt2-batch-cpu-v1-20261028/frozen.json \
  --expected-manifest-sha256 4f8151f6d946d4395fa49080d699a4eccb6a6416b6683810a0dd5fe6718c140d \
  --manifest-commit a1d1e960e9d4cb55e66ab5460d3e8f8ecc127794 \
  --manifest-repo-path results/distilgpt2-batch-cpu-v1-20261028/frozen.json \
  --repository .
```

The manifest digest, commit and path are supplied separately; they must not be
inferred solely from the run being checked. A shallow checkout may need the
recorded commits fetched first. `--verify-worktree` also compares the current
source files with the recorded source blobs; it is expected to fail after those
files have changed, so use a frozen checkout for that check. Optional
`--asset-directory` hashes the seven pinned local checkpoint assets, and
`--transformers-source` hashes the local installed GPT-2 implementation.
Omitted local checks are labeled `not_checked`.

Add `--replay-directory /path/to/another-run` to check another completed run
against the same frozen manifest. It compares the six uncompressed member
hashes, primary report, controls and model-state fingerprints. ZIP container
hashes are reported separately because container timestamps may differ. This
checks saved records; it does not itself launch a fresh process or establish
that either record came from an independent execution.

Exit 0 means structurally consistent passing evidence, and matching saved
checks when a replay is supplied. Exit 2 means an incomplete or nonpassing
reference, or different saved replay checks. Exit 1 means invalid evidence or
an unavailable required verification; an interrupted check exits 130. The CLI
emits a generic validation error or cancellation without local paths or a
traceback; retain API exceptions privately. Partial/failure
records are never promoted to a passing result.

The checker verifies source-to-manifest ancestry and committed bytes, but Git
history and self-reported times do not prove when the model actually ran or
when a remote first exposed a commit. The producer writes its passing terminal
only after the worker close call returns; that control flow is an attestation,
not independent proof of all cleanup. Source and asset hashes do not prove
what a process executed, or that the loaded model exactly matched the declared
checkpoint. These are four fixed synthetic cases on one CPU runtime, not a
general batch-separability, privacy, throughput or causal claim.
