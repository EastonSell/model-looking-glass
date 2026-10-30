# Historical result payload archive

This cleanup removes **2,106 large tensor/archive files (2,811,857,194 bytes)** from the active tree. It does **not** erase or rewrite Git history. Their original bytes remain in ancestor commit [`3cbdec78134855532811a58a64e8dd5d4d457185`](https://github.com/EastonSell/model-looking-glass/tree/3cbdec78134855532811a58a64e8dd5d4d457185).

[`results-manifest.json`](results-manifest.json) records every removed path, Git blob SHA-1, and byte size. Selection was by tensor/archive format (`.f32`, `.f64`, `.f64le`, `.bin`, `.zip`, `.gz`, `.gzchunk`, `.bundle`, and numbered multipart suffixes), not scientific outcome. These are **not necessarily redundant files**. Some compressed files contain decisions, logs or source bundles; those too remain available in the pinned archive. Uncompressed reports, JSON/CSV records, protocols, controls and failure records stay in the active tree. The initial CPU benchmark, denormal counterexample, and [eight compact records required by active CPU tests](CPU_TEST_FIXTURES.md) remain directly accessible.

The removed payload is 89.1% of pre-cleanup `results/` bytes and about 88.3% of all pre-cleanup tracked bytes. This is a checkout-size reduction, not a promise that a full clone or server storage becomes smaller. No force push or history rewrite is needed.

## List and selectively restore

The tool uses only Python's standard library and Git. It runs no model code, installs no packages, and configures no credentials. Run from the repository root:

```bash
python scripts/archive_results.py list --prefix results/targeted-edit-cpu-v1-20261028
python scripts/archive_results.py restore \
  --path results/targeted-edit-screened-matched-v1-20261029/screen/screen/screen.f32 \
  --output /tmp/model-looking-glass-restored
python scripts/archive_results.py verify \
  --path results/targeted-edit-screened-matched-v1-20261029/screen/screen/screen.f32 \
  --output /tmp/model-looking-glass-restored
```

`--output` for restoration must be a **new directory whose parent already exists**. The tool never merges with or overwrites an existing run. Select `--path` (repeatable), `--prefix` (repeatable), or explicitly `--all`; listing defaults to all. Restoration checks the pinned commit tree and verifies each file's size and Git blob hash before publishing its final filename. `verify` checks local bytes only and invokes no Git/network operation. Atomic publication requires same-volume hard-link support; an unsupported filesystem fails rather than weakening the no-overwrite guarantee.

A completion/incomplete status file is written in the new output directory. A failed run can retain already verified files, but must not be presented as a complete restoration. The tool is not a sandbox against hostile same-account processes.

## Shallow/partial clone requirements

A shallow clone may not contain the historical commit. If it is missing, explicitly fetch it before restoration:

```bash
git fetch --filter=blob:none origin 3cbdec78134855532811a58a64e8dd5d4d457185
```

The restore command itself does not call `git fetch`. In a partial clone, ordinary Git object reads may lazily download the selected missing blobs through your configured remote. Account/login failures are not bypassed. `verify` is offline.

## Replaying a historical study

Restoration returns **only the selected archived payloads**, under their original `results/...` paths inside the new directory. It does not copy current reports, install dependencies, or recreate the historical source environment. Source-hashed auditors may require the entire pinned historical snapshot rather than current code. To inspect one without changing your working tree:

```bash
git worktree add --detach ../model-looking-glass-historical 3cbdec78134855532811a58a64e8dd5d4d457185
```

That command can materialize the full historical payload; use a separate filtered checkout or a study-specific sparse checkout if you need a small subset. Follow that study's original instructions and dependency versions. Do not rewrite old source hashes or replace missing observations with invented values. Multipart files must be restored together before their documented reassembly/checksum checks.

Markdown links that directly targeted archived payloads now point to the immutable historical object. Literal paths inside preserved source, manifests and transcripts are deliberately unchanged; use the archive manifest when their old files are absent.
