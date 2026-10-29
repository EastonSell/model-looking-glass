# Saved-byte replay for agreement noun substitution

`looking_glass/agreement_substitution_science_v1.py` replays the fixed public
54-case, 486-forward candidate without importing a model or executing a
forward. It independently derives the order in the published protocol,
checks the run contract and every recorded case/action, and decodes every
saved last-position vocabulary vector. For every hooked action it verifies
the whole-output and selected-slice witnesses, independently recomputes the
rounded direction delivery, and checks realized dose and signed projection.
It also checks exact clean/no-op/clean-after logits, the closed terminal and
the complete content-addressed tensor inventory.

For all four fixed margins the report retains all case-level odd/even
responses and every one of the 12 morphology-by-frame-by-context factorial
cells. It gives signed cross-lemma differences, opposite and same absolute
distances, their difference, and the number, lemma and interaction contrasts.
The report includes all six subjectless fragments, realized doses, vocabulary
top-20, target ranks, top-one changes and `KL(clean||arm)` diagnostics. The
baseline uses the fixed 48-case denominator, counts ties incorrect, applies
all predeclared group and four-correct-cell requirements, and labels inadequate
coverage `baseline_limited` without excluding any response. The descriptive
criterion uses the mean of individual absolute control differences and strict
comparisons where specified. Neither the criterion nor a failed one proves a
unique mechanism or an absence of number representation.

## Required external observations

The caller must independently observe and retain the exact owner/worker
source, direct parent exit, prelaunch, reservation and current exposure and
custody observations. The caller must pin the closed `events.jsonl` digest
from that independent observation, before invoking the reader. A complete
looking terminal or a consistent saved journal does not itself establish
execution, process cleanup, blind inputs or absence of unrecorded exposures.
The event flag for full-sequence finiteness is checked as a recorded assertion;
unsaved sequence positions cannot be replayed. An incomplete, failed,
ambiguous or inconsistent attempt is not a scientific result.

Verify the reader's exact published source bytes **before executing them**.
The runtime must execute those verified bytes, rather than trusting a later
hash of an imported module; `-B` by itself does not prevent reading existing
bytecode. The reader separately hashes and compiles the exact bytes of
`agreement_science_v1.py` with SHA-256
`917694d448b796359db0002a630877e5226a2583d273092871892fade7a38ac4`.
It independently pins the published case file and three binary64 directions.
The following argument names are required; paths must name the independently
verified public inputs and a fresh output outside the attempt directory:

```text
--attempt --cases --directions --numeric-source --output --source-revision
--prelaunch-sha256 --reservation-sha256 --latest-exposure-sha256
--events-sha256 --start-scope
```

For a real run `--start-scope` must be exactly
`public prospective agreement noun-substitution`. The report is a candidate
saved-byte consistency check, and promotion requires the separate observations
above. Output creation is exclusive; retain an incomplete output if writing or
readback fails.

## Fabricated checks

From the project root, run:

```text
python -B -m pytest -q tests/test_agreement_substitution_science_v1.py tests/test_agreement_substitution_saved_v1.py
```

The tests use invented cases and small CPU-only tensors; they include a
complete synthetic journal, a separate scalar oracle, baseline and threshold
counterexamples, and corrupt ordering, input, finiteness, landing, dose,
binding, terminal and log-digest records. They do not load a pretrained model.
The test fixture's small width and vocabulary are accepted only through the
explicit synthetic API, never through the real-command entry point.
