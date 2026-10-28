# Independent implementation review

Review date: 2026-10-28 UTC.

**Reviewed commit:** `493bdcb4218bb9c465d7bf3d597e3441f6ce49a5`.
The implementation in this commit is identical to `f3e3563934208cb7a7e0f02e65d5f1510041094f`; the later commit adds documentation and status evidence. Findings below refer to this immutable baseline, not to later fixes.

**Latest correction reviewed:** `68d17069a67afc2a23df28d72c9b3e9b64811b09`. R1, R2, R3 and R5 are corrected; R4's guarantee is explicitly narrowed to preserve import failures on stderr rather than in a run directory, and failure-record write errors preserve the original exception. Within the stated CPU eager/source-and-wheel scope, no remaining material blocker was identified. Runtime and untested-scope qualifications are recorded in the addenda.

**Baseline disposition:** the mathematical construction and bounded research framing are reasonable, but correct the material implementation/reporting issues below and obtain executable verification before presenting numerical or performance results. See the dated correction-validation addendum at the end for subsequent evidence.

## Scope and evidence

Read the entire tracked Python source and tests, `pyproject.toml`, `PROTOCOL.md`, `docs/MATHEMATICS.md`, the design and implementation plan under `docs/superpowers/`, README, and existing verification evidence. The requested protocol is at repository root, not `design/PROTOCOL.md`.

Independently parsed all four package Python files and all three test files with Python's `ast.parse`: successful. This is syntax evidence only. The initial recorded unittest attempt was blocked by `ModuleNotFoundError: No module named torch`. No PyTorch numerical tests, timing cells, model execution, wheel installation, or full protocol were executed by this reviewer during the initial static-review phase. Initial dependency installation was denied; the parent subsequently reported explicit approval and an installation retry. This reviewer performed no installation. Subsequent independently executed validation is recorded separately below.

## Material findings

### R1 — Partial hook registration is not cleaned up

**Location:** `looking_glass/probes.py:107–111`. **Priority:** high for cleanup correctness.

`self._handles = [m.register_forward_hook(...) for m in points]` assigns only after the complete comprehension succeeds. If registration at a later tap raises, handles already returned for earlier taps are not present in `self._handles`; the exception handler's `close()` cannot remove them. The escaped hooks retain the session and can continue capturing after a failed context entry. Existing exception/overflow tests cover failures after entry, not during partial entry.

**Correction/verification:** append each handle immediately inside the protected loop. Inject failure on a second tap, assert the original exception propagates, assert the first tap has no remaining hook, and confirm a later forward neither records into the failed session nor encounters its capacity limit. Retain a genuine failing-then-passing regression result.

### R2 — Environment labels are not enforced

**Location:** `looking_glass/experiment.py:104–113, 140–153`. **Priority:** medium; measurement validity.

Model and input constructors inherit PyTorch's ambient default dtype/device, but the result always declares CPU/float32. An embedding caller that changes the default dtype to float64 can obtain float64 measurements labeled float32. A non-CPU default can violate the fixed CPU protocol and the timer's synchronous-CPU assumption.

**Correction/verification:** either reject unsupported ambient defaults before any measurement, or construct every model parameter and random input explicitly as CPU float32. Verify actual parameters/inputs, not just reported strings. Add a regression that changes the default dtype and restores it in `finally`. Reject or explicitly support ambient inference/autocast modes rather than silently broadening the protocol. Record any changed evaluation policy in the versioned protocol before a fresh full run.

### R3 — Non-editable package execution cannot locate the protocol

**Location:** `pyproject.toml`; `looking_glass/experiment.py:20, 99–101`. **Priority:** medium; reproducibility/packaging.

The package configuration includes the `looking_glass` Python package but not root-level `PROTOCOL.md`. In a conventional wheel installation, `ROOT` resolves to the package's parent in site-packages and `_hash_sources()` reads a nonexistent `ROOT/PROTOCOL.md`. This occurs at the end of evaluation, after measurement work. Source-checkout usage documented in README is not affected; a normal wheel was not built or installed in this review.

**Correction/verification:** ship the protocol as package data and resolve it through package resources, or explicitly define and enforce a source-checkout-only CLI and remove misleading package-install support. Verify in an isolated non-editable installation, not merely the source checkout or an editable install. Check packaged and canonical protocol content agree.

### R4 — Missing dependencies bypass failure evidence

**Location:** `looking_glass/experiment.py:14–17, 165–179`; README failure-record statement. **Priority:** medium; failure evidence.

The top-level `import torch` fails before argument parsing, output directory creation, or the `run_experiment` exception handler. Thus a missing-dependency CLI invocation produces no `failure.json`, despite the protocol's missing-dependency/failure-evidence requirements and README's general promise that a failed run writes a failure record. The separately retained initial unittest blocker is valuable but does not fix the CLI path.

**Correction/verification:** add an import-independent CLI wrapper/deferred imports that can preserve the dependency failure, or explicitly scope the failure-record guarantee to errors after successful import and require captured stderr for import failures. Test the documented behavior in an environment without the dependency. Failure logging must not overwrite existing evidence or mask the original exception if writing the failure record itself fails.

### R5 — CSV does not contain raw timing rows

**Location:** `PROTOCOL.md:23–24`; `looking_glass/experiment.py:170–175`. **Priority:** medium; protocol alignment.

JSON retains each `raw_ns` sample, but `timing.csv` includes only per-cell aggregates. README correctly calls these CSV summaries, whereas the frozen protocol calls for raw JSON/CSV timing rows. This is an output-format ambiguity/gap, not loss of the JSON raw samples.

**Correction/verification:** preserve the summary CSV and add one raw CSV row per measured iteration, including cell identifiers, sample index and elapsed nanoseconds, or document a pre-run protocol clarification. Verify full/quick row counts and equality with the JSON samples.

## Mathematical and measurement assessment

- For finite activation entries, `t(q) = sum(stop_gradient(a) * q)` is linear in the independent leaf `q`, so its derivative with respect to `q` is the copied activation. This is a standard autodiff identity, not a new mechanism for extracting otherwise unavailable information.
- The implementation detaches and clones the activation before building the reporting graph, never adds `q` to model parameters, and uses `autograd.grad` only for `q`. It returns `None` from the hook and does not mutate the main output. Static graph inspection therefore supports isolation from task gradients; equality through a complete SGD update still requires the supplied numerical tests. PyTorch documents that `autograd.grad` returns derivatives without accumulating them into `.grad`. [Official API](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)
- The local `inference_mode(False)` plus `enable_grad()` encloses the clone, leaf creation and derivative calculation. Cloning after disabling inference mode is the important distinction from feeding an inference tensor directly into a recorded backward graph. This structure is appropriate, but inference-mode numerical execution and context restoration were not established by static review. [Official grad-mode documentation](https://docs.pytorch.org/docs/stable/notes/autograd.html#grad-modes)
- Hooks returning `None` leave the ordinary output in place, consistent with the API's observer contract. Exceptions deliberately abort model calls rather than report successful telemetry. Existing tests cover output/gradient comparisons, detached nonaliasing snapshots, repeated-module occurrences, duplicate labels, nonfinite gradient inputs, overflow and ordinary context-body exceptions. [Official hook API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
- Equal-valued branches cannot be disambiguated from their common vector alone. The negative control compares copied branch parameters, equal input and equal values; distinct tap labels are explicit additional information, not recovered provenance or causal attribution.
- The known intervention is immediately before a named tap and acts on a clone, preserving the original input. The first differing observation brackets where a numerical difference becomes visible; it does not establish a unique cause or reveal model reasoning. Dense tap sites give finer observation granularity, not an information advantage per saved byte. The code records nondetection rather than silently asserting successful localization.
- Timing covers an entire **forward-only, no-grad** call, session setup, hooks, validation, metadata, recorded-byte calculation, cleanup and value disposal. It does not measure task backward/optimizer overhead; those are separate untimed transparency checks. Fixed warmups, per-cell raw timings and shuffled cell order support descriptive comparisons on this workload, not broad performance or statistical significance claims.
- `recorded_tensor_bytes` correctly sums logical storage size of retained tensor snapshots, counting the snapshots that are actually saved. It excludes Python metadata, the isolated gradient computation's temporary tensors, allocator overhead and process/GPU peak memory. ID-only mode has zero recorded tensor bytes, not zero memory use.
- The event limit bounds count, not a fixed number of tensor bytes. Concurrent tracing, compilation, distributed execution, checkpoint recomputation, complex values and broad dtype/device support remain outside the validated scope. The simple first-difference result records node IDs but omits occurrence indices, so it should not be promoted as unambiguous repeated-node localization without extending that output.

## Privacy and local-data assessment

The reviewed runtime source contains no network client, telemetry endpoint, credential handling, external dataset/model download, or subprocess execution. The experiment creates synthetic random tensors and parameters. Saved experiment JSON contains scalar checks, timing samples, labels, source hashes and environment information, not raw activation tensors or model weights. The public observation API does retain complete detached activations in process memory until the session/events are released; it is not anonymization or privacy protection if applied to real private inputs. Platform/version strings may reveal host environment details if a user later publishes results.

No external source project or model payload was executed in this review. Locality and synthetic inputs support the current privacy claim; they are not a general security audit or certification for arbitrary user models or third-party PyTorch installations.

## Acceptance evidence still required

1. Run the complete unit suite with recorded Python/PyTorch versions and no skips hiding unsupported channels.
2. Demonstrate R1's registration-failure regression, plus model-level dense gradient capture under `torch.inference_mode()` and preservation/restoration of surrounding grad modes.
3. Verify R2's actual CPU/float32 configuration, the dependency-failure evidence policy, raw CSV/JSON consistency, and isolated package execution if supported.
4. Run the frozen or explicitly versioned full protocol only after fixes; preserve any old failure directories. Confirm 56 timing cells, 48 transparency cases, 16 reconstruction cases and 192 localization rows for the specified full configuration, not the quick smoke configuration.
5. Inspect the persisted raw results, code/protocol hashes, successful nonoverwrite regression and independent branch-control result before publishing claims. Review any code changes after the commit named above separately.

This review does not certify unexecuted tests as passing. Later test evidence and fixes should be appended with their exact commit IDs and commands rather than silently rewriting the historical baseline.

## Correction-validation addendum — 2026-10-28 UTC

**Correction commit independently inspected:** `38ff944e46f05ec06a58e07b2905f8b6bff6c4a3`.

After the parent reported explicit user approval and installation completion, this reviewer independently ran `.venv/bin/python -m unittest discover -s tests -v`. All **24 tests passed**, with no skipped tests. The environment is Python 3.12.14 and PyTorch 2.15.0+cpu. PyTorch emitted a warning that optional NumPy is unavailable; these tests did not use NumPy. This establishes these test results, not support for every version permitted by `torch>=2.6`.

The diff changes registration to incremental handle append (R1), makes all benchmark parameters and random inputs explicitly CPU float32 (the ambient-default portion of R2), bundles an identical package-local protocol and hashes it (R3 source correction), adds raw timing CSV output (R5), and scopes README's failure-record promise to errors after evaluation starts (R4 documentation correction). Added tests cover registration failure, ambient default dtype, started failure logging, raw CSV row count and full-model inference-mode output preservation. The retained RED evidence shows the original hook leak and dtype regression failures plus missing raw CSV; the independent GREEN suite confirms the updated tests pass.

Independently inspected `results/initial-cpu-v1`: 56 timing cells with 100 samples each, 48 passing transparency checks, 16 equal reconstruction cases and 192 localization rows. Every one of the **5,600 raw CSV samples exactly matches JSON**, and the summary CSV has 56 rows. All saved package source hashes match the correction commit's current files; root/package protocols are byte-identical. Dense captures detect and first-localize exactly at all 96 injected tap locations; sparse captures detect all 96 but, as expected, do not name the untapped intra-block injection location as their first point. These are checks of the parent's persisted experiment, not a second independently executed full protocol.

Timing-process independence was checked: the complete result and run-log filesystem modification times are approximately 19:14:27 UTC, before this reviewer's unit suite at approximately 19:15:06–19:15:09 UTC. Thus the recorded initial run preceded that independent suite. The results do not themselves include per-cell absolute wall-clock timestamps. The parent plans a new timing directory for the final autocast-guard source revision; the earlier evidence should remain preserved.

At this intermediate point, isolated wheel-content execution for R3 and evaluation under ambient autocast for R2 remained pending. Both were subsequently checked as described below. Missing-dependency import failure still intentionally reports stderr before creating a directory; users must preserve that stderr as failure evidence. Failures in the failure-record write can still replace the original exception and are not covered by the injected evaluation-failure test. This is a residual error-reporting limitation for unwritable/full filesystems, not a claim that evidence can always be persisted. No general adversarial/custom-Tensor, GPU, compiled, concurrent or distributed validation has been performed.

### Final context and packaging checks

Independently reviewed `2f06d8407e6749b4fc9592eb404a01f946ed87d5`, which rejects active CPU autocast before evaluation. In an independent context check, a forward under CPU autocast produced bfloat16 output/capture as expected, and `run_experiment` correctly rejected that context with `RuntimeError: CPU autocast must be disabled for the float32 protocol`. The parent's saved autocast regression RED shows the old implementation accepted it; saved normal and optimized-Python suite results each report **25 tests passed**. Those 25-test suite runs were inspected, not rerun by this reviewer while a new timing experiment might start.

Additional independent PyTorch checks passed: full-model direct and gradient traces match inside `torch.inference_mode()`; inference mode and disabled-grad state remain active after capture inside that context; ordinary grad state is restored after exit; and a pre-existing activation `.grad` tensor is unchanged by isolated gradient readout.

Built a wheel with `.venv/bin/python -m pip wheel . --no-deps --no-build-isolation --wheel-dir <temporary-directory>` without installing or downloading dependencies. Confirmed the wheel contains `looking_glass/PROTOCOL.md`, extracted it to a separate temporary directory, changed working directory to `/tmp`, and ran the quick experiment with only that extracted package added to `PYTHONPATH`. Asserted the imported package path points to the extracted wheel contents rather than the source checkout. Compared every packaged Python source byte-for-byte against Git: the tested wheel contains `38ff944e46f05ec06a58e07b2905f8b6bff6c4a3` source. The isolated smoke passed all reported transparency/channel checks and produced seven timing cells. This establishes executable wheel contents; it is not a test of pip's install/uninstall machinery or a rebuilt wheel of later source revisions.

Independent runtime checks ended by 19:17:49 UTC; the parent was notified that measurement could proceed without this reviewer's test workload. At this point remaining limitations were the explicitly scoped import-failure evidence policy, untested error-record write failure, optional NumPy absence, and lack of broad PyTorch-version/device/compiler coverage. The write-failure limitation was subsequently addressed below. No generalized performance, causality, safety, privacy, or novelty claim follows from this review.

### Failure-record correction — final source review

Independently inspected commit `68d17069a67afc2a23df28d72c9b3e9b64811b09`. If evaluation fails and writing `failure.json` also raises an ordinary exception, the original evaluation exception is now re-raised with the recording error chained as its cause. The added regression injects both failures and checks their types and messages. The retained RED evidence shows the original OSError masked the evaluation failure; the final normal and optimized-Python logs each report **26 tests passed**. These logs were inspected; no further reviewer runtime test was launched during the parent's final measurement window.

At this intermediate stage, the reviewer's complete-suite run was the earlier 24-test version; the final full protocol and current 26-test suite still awaited independent checking. These checks are now recorded below. The tested isolated wheel predates the two small guard/error-path corrections. The broad backend/version/privacy/causality limits above continue to apply. Import-time missing-dependency failures deliberately require retaining stderr, and no software can guarantee durable local failure evidence when its filesystem is unwritable.

### Final independent acceptance — 2026-10-28 19:20 UTC

After receiving confirmation that measurement had finished, independently reran `.venv/bin/python -m unittest discover -s tests -v` at `68d17069a67afc2a23df28d72c9b3e9b64811b09`: **26 tests passed, no skips**, with the unchanged optional-NumPy warning. Retained this reviewer's output in `evidence/reviewer-final-tests.txt`.

Independently checked `results/verified-cpu-v1` against the final source and frozen configuration:

- All package/protocol hashes match; canonical and bundled protocols match byte-for-byte.
- Exactly 56 timing cells, 100 samples per cell and 5,600 raw CSV rows, each identical to the corresponding JSON sample; 56 summary rows. Medians and p90 values recompute exactly using the implementation's documented index convention.
- Every event count and recorded-byte count matches the number, width, batch and dtype of retained tensors; ID-only and no-capture cells record zero tensor bytes.
- All 48 training-transparency checks and all 16 direct/gradient reconstruction checks pass. Both identical-branch-control booleans are true.
- All 192 perturbations are detected. The 96 dense cases first differ exactly at the injected tap; the 96 sparse cases first differ at a block-output tap. On the common ordered list of available taps, last-equal-to-first-unequal span is 1 for dense and 4 for sparse in every case. This is observation spacing, not a causal-distance measure.
- Median across the four seed-paired gradient/direct median-latency ratios is 2.7919 (width 16, sparse), 3.8088 (16, dense), 2.5999 (64, sparse) and 3.6395 (64, dense). Thus the reported **2.60–3.81×** range describes these four grouped median ratios, not the range of every individual sample/cell; individual seed-paired cell ratios range approximately 2.41–4.10×. These are descriptive forward-call overhead results for this synthetic CPU workload.

**Final disposition:** no unresolved material implementation or measurement blocker found within the stated scope. Historical runs are separate evidence and should not be pooled without an explicit analysis plan. This reviewer inspected rather than independently repeated the final full timed protocol, tested wheel contents from the earlier packaging-fix commit, and did not validate minimum-supported or alternative PyTorch versions, accelerators, arbitrary model integrations, compilation, concurrency, distributed execution, or general model-security/privacy properties.
