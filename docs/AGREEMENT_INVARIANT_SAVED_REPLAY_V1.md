# Invariant-noun saved-byte replay

`tools/audit_agreement_invariant_v1.py` is a model-free reader for the fixed
public invariant-noun protocol and case package. It independently reconstructs
the 612-action ordering, checks the case coordinates and internal token-prefix
relationships, reads every saved last-token logit and hook witness, verifies
the no-op and realized direction deliveries, and recalculates all four margin
families. It retains all 68 case responses, twelve factorial cells, the
fragment and demonstrative comparators, baseline counts, grouped statistics,
and both fixed criteria. Baseline failures are labelled baseline-limited; cases
and negative contrasts are never removed. The full-vocabulary KL calculation
uses nonnegative log-domain contributions; a positive total below binary64's
representable range is explicitly marked as underflowed to zero. The replay
checks calling-thread binary64 subnormal arithmetic before and after the audit
and around KL calculations; it stops if that arithmetic is flushed to zero.

The reader requires the independently observed owner source revision, prelaunch,
reservation, current-exposure and event-log identities as command arguments.
It checks the published case, direction and numerical-helper hashes, the
case manifest's recorded protocol digest, the complete event/terminal schemas,
and the exact content-addressed tensor inventory. The caller independently
reads and verifies the published protocol bytes. A missing, failed, partial,
contradictory or altered attempt is
rejected rather than reported as a completed result. The output must be a fresh
absolute file outside the saved attempt; the reader does not modify the attempt.

The entry hashes and compiles the exact reader bytes before execution, and the
reader similarly loads its numerical helper from verified bytes, avoiding stale
bytecode. The caller must separately verify and compile the exact published entry
bytes before invoking it, retain and observe the replay process exit, and verify
the owner/child process, public exposure and applicable custody observations.
An ordinary saved-file check cannot authenticate execution, establish unknown
exposure, prove descendant cleanup, or verify the finiteness of unsaved sequence
positions. Those limitations remain distinct from successful saved-byte replay.

The focused tests use only fabricated values and journals. From the repository
root they are `python -B -m pytest -q -p no:cacheprovider
tests/test_agreement_invariant_science_v1.py
tests/test_agreement_invariant_saved_v1.py
tests/test_agreement_invariant_entry_v1.py`. A separate fabricated executor
journal can be replayed with `audit_records` in synthetic mode for interop;
the command-line entry has no synthetic-mode option.
