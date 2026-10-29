# Development-only CPU resource pilot for causal v2

This pilot estimates whether the proposed
[v2 schedule](CAUSAL_CONFIRMATION_V2.md) is feasible. It is not a v2 discovery
or held-out evaluation and makes no efficacy claim. It does not generate the
v2 discovery or held-out cases, private roots, commitments, or site schedule.

Use one distinct untrained `TinyResidual(width=8, depth=4)` model seed 49351,
route 0, CPU float32, evaluation mode and `torch.no_grad()`. Record the actual
ambient intra-op thread count during model initialization; the measured
workload itself uses one intra-op thread. Initialization under this ambient
policy does not model the proposed v2 runner's one-thread initialization.
The standalone process must successfully call `torch.set_flush_denormal(False)`
and demonstrate that the smallest positive float32 subnormal does not compare
equal to zero on the execution thread. It must also verify that halving the
smallest normal float32 produces raw bits `0x00400000`, checking output
preservation separately. Record this policy and probes. The
programmatic API checks the caller's existing policy and fails rather than
changing its state. Do not infer behavior of other threads or devices from
this single-thread CPU check.
Generate two deterministic pilot vectors with opposite first-coordinate signs
on the same bounded 1/16 grid as v2. Use the domain-separated byte prefix
`MLG-v2-resource-pilot\0`, the model seed, input index and collision counter to
derive them. Reject exact float32 duplicates, signed-zero equivalents, and v1
vectors. The published `pilot_inputs.json` is the canonical v2 exclusion record
`{"inputs": [{"case_id": ..., "values": [...]}, ...]}` sorted by case ID,
encoded as UTF-8 canonical JSON with sorted keys and compact separators. Its
SHA-256 and exact input tuples must be excluded from **both** future v2 splits.

Exercise three synthetic change configurations, `block.1.branch.0`,
`block.2.activation`, and `block.3.merge`, each with both input signs and
strengths 0.25 and 0.75. Reuse
two clean sparse traces per input and one clean target capture per input/site.
For each active condition run one changed sparse trace, one changed target
capture, three forward patches, three reverse patches, three null interventions,
three same-run identity patches, one upstream patch, and one fresh clean pass.
This is 4 clean trace forwards + 6 clean target captures + 12 × 16 condition
forwards = **202**. Count every root forward and fail at a configured limit,
which may not exceed 60,000. Verify full two-logit controls, clean trace
repetition, private target delta, and the known before/after-block patch
equalities using finite float32 bit patterns with signed zeros treated as equal.
Each sparse trace is capped at four events and 128 logical snapshot
bytes; each capture or intervention is capped at one invocation and 32 bytes.
The pilot uses separate, exclusive hook contexts, never nested sessions.

Before the first forward, save the exact input exclusion record and counted
plan in a new registry run. Save wall and process CPU seconds from function
entry to workload start and to workload end, plus workload-only timings. These
spans exclude Python/PyTorch import and final result/terminal publication;
the workload-only span also excludes model construction, registry setup and
input publication. Save
forward counts by phase, Python/PyTorch versions, CPU-only device information,
source revision and hashes, and peak process working set since process start.
After successful registry creation, attempt to preserve input and plan
artifacts, partial counts, and a failed/cancelled terminal outcome on error or
cancellation. Failures before registry creation and filesystem write failures
may leave no complete record. If finalization raises after a terminal outcome
was already published, verify and report the actual outcome; if verification
also fails, treat it as unknown. An existing directory is never
overwritten. The peak is process resident working set, including interpreter
and library startup, not incremental tensor memory or a GPU-memory estimate.

For scale only, report 57,408 / 202 times the measured **workload-only wall duration**
as a simple linear extrapolation of the proposed schedule. This is not a
benchmark or upper bound: it excludes initial model construction, publication,
some six-model setup and final-run record costs, and depends on host load and
other activity. Arithmetic shows only that 57,408 fits the 60,000 forward cap;
this tiny pilot does not establish that the full schedule fits a wall-time or
host-memory budget. Coordinate with isolated GPU timing windows before execution.
If feasibility is uncertain, collect a separately labeled larger development
pilot or revise the v2 plan; never reduce the frozen v2 workload silently.
