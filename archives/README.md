# Historical result payload archive

This cleanup removes **2,099 large tensor/archive files (2,783,713,274 bytes)** from the active tree. It does **not** erase or rewrite Git history. Their original bytes remain in ancestor commit [`3cbdec78134855532811a58a64e8dd5d4d457185`](https://github.com/EastonSell/model-looking-glass/tree/3cbdec78134855532811a58a64e8dd5d4d457185).

[`results-manifest.json`](results-manifest.json) records every removed path, Git blob SHA-1, and byte size. Selection was by tensor/archive format (`.f32`, `.f64`, `.f64le`, `.bin`, `.zip`, `.gz`, `.gzchunk`, `.bundle`, and numbered multipart suffixes), not scientific outcome. These are **not necessarily redundant files**. Some compressed files contain decisions, logs or source bundles; those too remain available in the pinned archive. Uncompressed result reports, JSON/CSV records, source-bound protocols, controls and failure records stay in the active tree. Explanatory documentation has a separate archive below. The initial CPU benchmark, denormal counterexample, and [15 original records required by active CPU tests](CPU_TEST_FIXTURES.md) remain directly accessible.

The removed payload is 88.2% of pre-cleanup `results/` bytes and about 87.4% of all pre-cleanup tracked bytes. This is a checkout-size reduction, not a promise that a full clone or server storage becomes smaller. No force push or history rewrite is needed.

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

## Historical documentation

The documentation consolidation removes **203 historical explanatory Markdown files** from the active tree, not from history. Two overlapping plans are now one [living roadmap](../docs/RESEARCH_ROADMAP.md); recurring service/ownership/budget limits are collected in one [experimental infrastructure reference](../docs/EXPERIMENTAL_INFRASTRUCTURE.md). The [documentation guide](../docs/README.md) separates the tutorial and API shelf from frozen research contracts, and the [findings map](../docs/PROJECT_STATUS.md#research-evidence) links positive, negative and inconclusive results.

[`documentation-manifest.json`](documentation-manifest.json) records each removed document's original path, title, category, Git blob hash, SHA-256, size and immutable reading URL at commit **`f3d3e28980379640f7548541225ef40b017e5557`**. This is a different archive from the large result payloads above. These were incremental exposure summaries, historical runner/component manuals and superseded planning—not interchangeable versions of a current user guide. Their unique detail remains available; archival does not declare the associated code, findings or negative evidence useless.

Selection excluded every document referenced by path, basename or content hash in tracked non-Markdown source/records, plus destinations needed by frozen Markdown. Core scientific/numerical/API guides were retained explicitly. Code, tests, case manifests, result records, packaged protocols and source-bound document bytes are unchanged. Remaining duplicate protocol copies are intentional where packaging or historical hash contracts require them. No published source or evidence hashes were rewritten.

To read an archived document, follow its manifest URL. In a checkout already containing the pinned commit:

```bash
git show f3d3e28980379640f7548541225ef40b017e5557:docs/GPU_BOOTSTRAP.md
```

A shallow clone may need an explicit `git fetch --filter=blob:none origin f3d3e28980379640f7548541225ef40b017e5557` first; object reads in a partial clone may obtain missing blobs. To reproduce an old study, use the matching historical source/runtime as explained above, not a current summary in place of its protocol. **`archive_results.py` is result-only and does not accept this documentation manifest.**

The active `docs/` Markdown count falls from **447 to 246**, including the two consolidated entry points; total repository Markdown falls from **634 to 433**. This reduces competing documents, not research scope. Frozen protocol statuses describe their original stage and should not be read as current setup claims.

Validation of this documentation-only change: all 203 archive entries matched the original Git blobs and SHA-256/size records; a tracked-text dependency scan and independent review found no retained source/manifest dependency or newly broken local Markdown destination. Offline pytest collection still finds **8,430 tests**, and **67 focused archive/probe/intervention tests passed**. No runtime code, tests or result records changed. The earlier complete CPU result remains historical validation of that implementation; the full suite was not rerun merely to change documentation.
