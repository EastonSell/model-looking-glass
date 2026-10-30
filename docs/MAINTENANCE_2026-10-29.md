# Maintenance audit — 2026-10-29

Base snapshot: `3cbdec78134855532811a58a64e8dd5d4d457185`. This is a bounded cleanup/validation record, not a certification that every historical study or declared dependency version works.

## Changes

- Replaced the sprawling onboarding page with a direct-capture-first workflow, explicit component limits, and an immutable link to the complete historical research index.
- Archived 2,114 historical tensor/archive payloads totaling 2,830,512,412 bytes through a path/blob-hash/size manifest. The manifest exactly matches the removed Git paths and the pinned original tree. No history rewrite, scientific outcome filtering, or destruction of unique historical bytes.
- Added a standard-library selective restore/verify tool. It requires a new output directory, checks pinned tree membership and byte integrity, publishes without overwrite, rejects unsafe paths, and records incomplete failures honestly. Local verification is offline.
- Updated 49 direct Markdown links in 31 files to immutable historical payload URLs, preserving report prose. Compressed decision records are explicitly identified as archived, not misrepresented as remaining readable in the current tree.
- Consolidated 10 byte-identical standalone historical auditors into their retained result-local copies (228,814 duplicate source bytes), after reverse-reference and blob-identity checks. See [canonical mapping](../archives/SOURCE_CONSOLIDATION.md).
- Retired the three-script original v4 case-preparation/comparison workflow, explicitly superseded before forwarding by corrected v4.1. Preserved original protocol/receipts and immutable source links; retained v4.1 successors.
- Fixed `TraceSession` capacity/sequence bookkeeping to use a private cumulative event count: removing public-list entries cannot reset session sequence IDs or replenish capture allowance. Caller-inserted entries are explicitly caller-owned. Added focused regressions and qualified the auxiliary-gradient fidelity docstring. The original dependency blocker and later actual PyTorch validation are recorded below.
- Distinguished reusable observation/intervention components from experimental service infrastructure and historical reproduction code; source retirement requires dependency-aware review rather than indiscriminate deletion.

## Initial checks (2026-10-29; before approved dependency installation)

| Check | Outcome |
|---|---|
| `python -m unittest discover -s tests -p 'test_archive_results.py' -v` | 18 passed, no skips; independently rerun. The same 18 also passed under `python -O`. |
| `test_trace_schema.py` via unittest discovery | 14 passed, 1 explicit missing-dependency skip. |
| `test_registry.py` via unittest discovery | 21 passed, 1 explicit missing-dependency skip. |
| `test_causal_metrics.py` via unittest discovery | 21 passed, no skips. |
| AST parsing of package/tests/scripts/tools/experiments/examples | 1,389 files parsed before consolidation; 1,379 files parsed after duplicate consolidation; 1,376 remaining files parsed after superseded-workflow retirement. Syntax evidence only. |
| Manifest/tree/deletion correspondence | All 2,114 paths and blob identities match the pinned source; exact deletion-set equality. |
| Event-count regression | The actual ID-only bookkeeping method, isolated with a tensor-metadata stub, reproduced the old allowance bypass/duplicate sequence and rejected it after the patch. This is not a PyTorch hook/integration test. |
| Real archive restoration | Float32 payload, gzip JSON and ZIP restored and verified against hashes/sizes; gzip parsed as JSON and ZIP member integrity passed. A separate 2,613,364-byte float32 artifact also restored and verified. |

## Initial limitations (2026-10-29)

At the initial cleanup checkpoint, the executor had no PyTorch, pytest or Transformers. No packages were installed during that initial pass. The attempted `test_probes.py` run fails at import with `ModuleNotFoundError: torch`; `test_behavioral_edit_evaluation.py` is blocked by missing pytest. These are **not passing tests**. The two dependency skips above likewise do not validate their skipped behavior.

The complete historical suite, GPU behavior, pretrained models, saved-study auditors requiring full archived datasets, and cross-version/platform support have **not** been revalidated by this pass. Restoration integrity is not a replay of the science. Existing study reports keep their original scope and dates.

Source consolidation removed verified duplicates, not entire coupled study families: retained historical launchers and test fixtures are not thereby supported public APIs. Do not interpret this audit as a claim that all remaining components have passed runtime validation.

## CPU validation continuation — 2026-10-30

After explicit approval, installed official CPU PyTorch and project testing extras in an isolated environment. This checkpoint uses Python 3.12.14, Torch 2.15.0+cpu, Transformers 4.57.3 and pytest 8.4.2. The dependency environment is outside the checkout so source-isolation checks can enforce separate dependency and source roots. `pip check` passed. Tests run offline, with CUDA hidden and CPU threads bounded; no pretrained model downloads or historical GPU runs were performed.

- Actual probe/ownership/archive focused run: **78 passed**.
- Portable isolated-entry, invariant collection, historical-metadata unit guards and journal-before-tensor validation: **65 passed** in one combined run.
- Causal compatibility, BERT workspace/ledger/parser guards, archive tool and historical-auditor boundary: **188 passed, 4 skipped** in one combined run. The skips explicitly require the removed named-tensor API, Python 3.11 historical reconstruction, Windows junction behavior, or historical Torch 2.8.0+cu128. They are not validated behavior.
- Native dependency-site and fabricated-owner bootstrap fixtures: **50 passed, 4 explicit Windows integration skips** in one combined run. Cross-platform shadow/hash/no-subprocess guards remain exercised.
- Fixed ordinary-checkout test fixtures rather than weakening workspace, version, disk-space, isolation or scientific source pins. A Torch API compatibility fix preserves named-tensor rejection on builds with that API.
- Restored **six original compact test evidence fixtures (13,428,490 bytes)** with their exact historical blob hashes; see [retained fixtures](../archives/CPU_TEST_FIXTURES.md). The current archive is **2,108 files / 2,817,083,922 bytes**. The original 2,114-file archival checkpoint above is historical.

The first two broad-suite diagnostics were interrupted after identifying environment and fixture prerequisites; neither is a completed passing run. A complete fixed-snapshot CPU-suite result remains pending at this checkpoint. Mocked runtime metadata tests are guard/state-machine coverage, not evidence that the old CUDA/Python/Windows runtime was reproduced. No hosted CI run is configured at this checkpoint.

### Additional compatibility findings

A later fixed-snapshot diagnostic was externally interrupted at approximately 45%; its log recorded 46 failed cases and 37 skips, without a terminal suite result. It is not a completed or passing run. Targeted fixes subsequently validated:

- Further Torch named-API compatibility: 114 passed / 1 API prerequisite skip.
- Actual return-opcode fault injection on Python 3.12 and synthetic GPU workspace fixtures: 175 passed / 8 pre-existing platform/cache/opt-in skips. No production GPU guards changed.
- Strict dependency-site admission now recognizes both virtualenv and standard-library venv metadata, preserving exact base/version/implementation and no-system-site checks. Complete sentiment-owner module: 74 passed; committed-source multitoken-operator module: 6 passed.
- Synthetic FLAN controller/runtime and output-directory fixtures: 38 passed / 2 existing skips. Synthetic empty-selection/runtime rejection: 11 passed / 1 Windows junction skip.
- Restored the original 1,115,136-byte attention fixture needed by two saved-head-selection tests; both related modules: 14 passed.

These scoped results do not substitute for the still-pending complete CPU-suite result or a real historical GPU run.

### Test-isolation continuation

The remaining-case diagnostic uncovered a shared synthetic-fixture problem: Windows-only lease admission failed on Linux, correctly retained quarantine, and therefore retained the source-bound module namespace. Downstream tests then held different class identities. Production quarantine and exact-type checks were not weakened. Synthetic tests now label simulated OS/Job records explicitly, verify lease closure and module-object restoration, and keep actual Windows integrations separately prerequisite-gated.

Fresh scoped checks (overlapping groups, **not additive suite totals**): injection observer group 35 passed / 4 Windows skips with empty post-suite quarantine; controller/receipt group 49 passed / 11 Windows skips; custody/query plus downstream journal/supervisor group 151 passed / 10 Windows skips; query followed by owner-completion group 83 passed / 1 Windows skip. Fresh number-injection and multiprocessing checks passed all 51 cases without changing their production code. Owner-completion alone passed 64 cases after nonvacuous return/UNPACK fault-injection setup was corrected for Python 3.12.

An unguarded local diagnostic launcher also broke multiprocessing spawn; this was external test tooling, not repository behavior. It was corrected and its spawn-import guard verified. These interrupted diagnostics are not a final all-suite pass. The final source-pinned CPU suite remains pending, with terminal-case checkpoints to distinguish completed cases from interrupted ones.

### Single-snapshot exposure validation

A later all-tests attempt reached 5,526 passing cases and 153 prerequisite skips before historical Git access was interrupted; it produced no terminal suite result. Five exposure-evidence tests require historical commits absent from the shallow checkout. Missing Git history is not a failed scientific hash, and the incomplete run is not an all-suite pass. Subsequent offline runs explicitly forbid Git transports as well as lazy fetching.

The v3–v14 exposure inventory constructors redundantly reconstructed predecessors for digest, payload and prefix checks. They now build each immediate predecessor once, verify that exact object's canonical digest and expected pin, and derive the copied payload/prefix sets from it. There is no shared mutable cache. Historical inventory constants, scientific records and source-bound evidence hashes are unchanged; changed implementation source hashes do not establish historical runtime replay equivalence.

New regressions cover all 45 canonical inventory versions, one predecessor construction, deep-copy isolation, payload tampering despite a separate valid digest claim, and wrong predecessor constants. Synthetic downstream-prefix/identity fixtures now coherently pin their intentionally modified payload, so those independent guards still execute rather than failing at the earlier digest check. Independent review found no blocking issue.

An illustrative single local CPU measurement of v14 construction was 30.585 seconds before and 0.0244 seconds after, with the same 279 records and SHA-256 `236e4ab86b006998cbf39096edf78222cc8506c3f2b81d5e7ae70b2a32b30d86`. This is one timing per revision, not a statistical or model-inference benchmark. Complete post-change validation remains pending below.

Integrated constructor, canonical-pin and modified guard tests: **103 passed, zero skips**, with Git protocols forbidden. This focused result does not validate the missing historical commits or substitute for a final complete CPU-suite result.

### Completed exposure checks and later validation limits

After the explicitly approved historical-file download, all **255 exposure/inventory tests passed offline**, including every canonical digest and the previously missing historical provenance checks. The tested snapshot was unchanged. Only the required commit ancestry and evidence objects were hydrated; no scientific bytes or pins were changed.

The subsequent full CPU attempt on tree `2d2ffd77d85417773f2ee14cba242c69150ff0c8` terminated with native exit 139 during an opcode-exception injection test: 7,452 passed, 187 skipped and 43 failed cases had reached terminal status; there was no terminal suite summary. A separate remaining-files diagnostic completed with 678 passed, 42 skipped and 8 failed. These are diagnostic results, not a complete passing suite. A later combined fault-injection group also crashed natively; individual passing groups do not resolve that same-process interaction.

Python's [documented opcode-tracing contract](https://docs.python.org/3.12/reference/datamodel.html#frame.f_trace_opcodes) warns that an exception escaping an opcode trace callback can cause undefined interpreter behavior. Several historical tests intentionally use this mechanism. Correcting missing opcode events does not establish that arbitrary bytecode interruption is supported, and isolating a native failure does not prove its cause fixed. These experiments must be distinguished explicitly from supported call-boundary failure controls; final selection and validation results remain pending.

Separately, six corrected ordinary test modules passed **47 tests with 2 explicit Windows prerequisite skips** in one integrated run. Their fixes isolate a no-model import assertion from earlier tests, label fake CUDA/runtime metadata as synthetic, verify genuine CPU-build CUDA rejection, preserve Windows-volume admission guards, and compare a recorded float32 intervention norm with its realized value rather than assuming exact real-number addition. No production runtime or custody guards were relaxed.

Two additional original reference-logit fixtures (gradient and random edits, 5,226,728 bytes total) were restored with unchanged Git hashes after their saved-normalization test exposed the dependency. That module plus archive-tool tests passed **44 cases** offline. The current retained fixture set is eight files / 18,655,218 bytes; the current archive manifest is 2,106 files / 2,811,857,194 bytes. Seven separate screened-evaluation matrices remain an unresolved prerequisite at this checkpoint.
