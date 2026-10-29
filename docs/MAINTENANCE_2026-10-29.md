# Maintenance audit — 2026-10-29

Base snapshot: `3cbdec78134855532811a58a64e8dd5d4d457185`. This is a bounded cleanup/validation record, not a certification that every historical study or declared dependency version works.

## Changes

- Replaced the sprawling onboarding page with a direct-capture-first workflow, explicit component limits, and an immutable link to the complete historical research index.
- Archived 2,114 historical tensor/archive payloads totaling 2,830,512,412 bytes through a path/blob-hash/size manifest. The manifest exactly matches the removed Git paths and the pinned original tree. No history rewrite, scientific outcome filtering, or destruction of unique historical bytes.
- Added a standard-library selective restore/verify tool. It requires a new output directory, checks pinned tree membership and byte integrity, publishes without overwrite, rejects unsafe paths, and records incomplete failures honestly. Local verification is offline.
- Updated 49 direct Markdown links in 31 files to immutable historical payload URLs, preserving report prose. Compressed decision records are explicitly identified as archived, not misrepresented as remaining readable in the current tree.
- Distinguished reusable observation/intervention components from experimental service infrastructure and historical reproduction code; source retirement requires dependency-aware review rather than indiscriminate deletion.

## Executed checks in this environment

| Check | Outcome |
|---|---|
| `python -m unittest discover -s tests -p 'test_archive_results.py' -v` | 18 passed, no skips; independently rerun. The same 18 also passed under `python -O`. |
| `test_trace_schema.py` via unittest discovery | 14 passed, 1 explicit missing-dependency skip. |
| `test_registry.py` via unittest discovery | 21 passed, 1 explicit missing-dependency skip. |
| `test_causal_metrics.py` via unittest discovery | 21 passed, no skips. |
| AST parsing of package/tests/scripts/tools/experiments/examples | 1,389 Python files parsed successfully; syntax evidence only. |
| Manifest/tree/deletion correspondence | All 2,114 paths and blob identities match the pinned source; exact deletion-set equality. |
| Real archive restoration | Float32 payload, gzip JSON and ZIP restored and verified against hashes/sizes; gzip parsed as JSON and ZIP member integrity passed. A separate 2,613,364-byte float32 artifact also restored and verified. |

## Blocked or not established

The current executor has no PyTorch, pytest or Transformers. No packages were installed during this cleanup. The attempted `test_probes.py` run fails at import with `ModuleNotFoundError: torch`; `test_behavioral_edit_evaluation.py` is blocked by missing pytest. These are **not passing tests**. The two dependency skips above likewise do not validate their skipped behavior.

The complete historical suite, GPU behavior, pretrained models, saved-study auditors requiring full archived datasets, and cross-version/platform support have **not** been revalidated by this pass. Restoration integrity is not a replay of the science. Existing study reports keep their original scope and dates.

Source consolidation is a separate audit step: retained historical launchers and test fixtures are not thereby supported public APIs. Do not interpret this first archive pass as the completion of every possible source-level repair.
