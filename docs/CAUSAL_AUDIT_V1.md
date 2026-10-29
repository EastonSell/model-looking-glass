# Auditing the saved synthetic causal v1 record

The v1 auditor checks the **already published** exploratory artifact without
loading PyTorch, running a model or evaluating any input. From a checkout
containing the result:

```bash
python -m looking_glass.causal_verify results/synthetic-causal-v1-20261028
```

The auditor requires a successful registry outcome and the three pinned
published manifest, precommit and result SHA-256 values. It verifies their
internal bindings, fixed protocol and runtime settings, source/model identity
claims, fixture hashes and ordered split IDs, intervention definitions and
frozen discovery-effect means. It independently recomputes the saved
float32 logit differences, per-case intervention effects, target corrections,
scalar control effects and descriptive prediction/residual summaries. It
also computes the recorded reverse-opposition and duplicate-site discrepancies,
checks the random site's seeded draw and the absence of bootstrap intervals,
and checks the terminal outcome's held-out count. For this pinned artifact it
checks the published report's equality observations; the in-memory semantic
checker reports discrepancies rather than assuming those equalities hold in
other data. Comparisons of derived numerical summaries use relative tolerance
`1e-10` and absolute tolerance `1e-12` for cross-implementation arithmetic.

The digest constants identify this particular published artifact. This command
intentionally rejects another run or replay with a different manifest, even
if its results are otherwise equivalent. `audit_payload` is available for
checking decoded in-memory payloads, but it does not authenticate their source
or establish a registry outcome. A pinned digest is a consistency anchor, not
a signature, secure timestamp or independent proof of an execution. Protect
the repository and compare digests against an independently trusted source if
authenticity matters.

Important limits of this audit:

- The discovery trace vectors and ranking summaries were not saved. The audit
  checks that the selected sites are consistently recorded; it cannot
  independently reconstruct the first-difference or magnitude selections.
- Full control logits were not saved. Scalar control deltas and the recorded
  clean-trace assertion are checked; full-output control equality cannot be
  reconstructed from these files.
- The files do not independently establish that their reported model forwards
  occurred, that predictions preceded held-out evaluation, or that the public
  held-out data were previously unseen. The existing development exposure is
  described in [the pre-run clarification](CAUSAL_PROTOCOL_V1_NOTES.md).
- Consistent arithmetic and a matching injected tap do not prove a unique
  mechanism, general necessity or sufficiency, or a pretrained-model result.

See the [results report](CAUSAL_RESULTS_V1.md) for the study's actual values
and the distinction between the candidate and final-state control.
