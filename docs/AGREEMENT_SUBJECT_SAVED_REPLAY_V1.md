# Agreement subject-dependence: independent saved-byte replay

`looking_glass/agreement_subject_science_v1.py` replays the fixed, public
subject-dependence protocol from a closed candidate journal. It loads no model
and makes no forward. It independently derives the 80-case, 720-action order;
checks the recorded inputs, site, capture/patch plan, complete no-op and clean
logits, exact f32 landing, unchanged outside positions and both dose tests; and
hashes every content-addressed saved tensor. It checks the terminal's event
and retained-byte totals and rejects missing, partial, failed or extra evidence.

The case, three direction and output-reference bytes are fixed by their
published SHA-256 identities. The numerical primitives are the previously
reviewed `looking_glass/agreement_science_v1.py` at commit
`514459c52ccf50b0b3d07250c4ac9b028afe2da5`, SHA-256
`917694d448b796359db0002a630877e5226a2583d273092871892fade7a38ac4`.
The supplied source-first entry reads and compiles the exact pinned replay
source bytes; the reader then reads, hashes and compiles the numerical helper
bytes directly. This avoids executing stale cached bytecode. A trusted caller
must verify the entry source itself against its published bytes before running
it, supply the independently pinned reader SHA-256, and independently verify
the owner, its observed clean exit, prelaunch identity and current exposure.
The audit is not proof of execution or of complete historical exposure. A
recorded all-sequence-finite flag cannot be reconstructed from a successful
last-logit-only witness.

The report includes all 80 rows' four margin families, odd/even responses,
signed changes, realized dose and projection, top-20 values, target ranks and
full-vocabulary KL. It includes the 32 paired K/G values, within-pair
attractor/mass contrasts, condition and crossed frame/subject aggregates,
clean baseline counts, and all 16 fragment results. It scores the general
constant-additive K criterion separately from the particular prior-reference
fit, and reports reference residuals and centered full-vocabulary cosine,
including undefined counts. Baseline failure labels the sensitivity
interpretation `baseline_limited` without removing cases or changing the raw
threshold result. The prespecified criteria are descriptive, not statistical
tests or proof of a grammatical mechanism.

With the exact prelaunch/source identities independently established, run the
source-verified module from a separate CPU-only process, for example:

```text
python -I -S -B ABSOLUTE_SOURCE_VERIFIED_ENTRY_FILE \
  ABSOLUTE_READER_SOURCE_FILE INDEPENDENT_READER_SHA256 \
  --attempt ABSOLUTE_CLOSED_ATTEMPT_DIRECTORY \
  --cases ABSOLUTE_PUBLISHED_CASE_FILE \
  --directions ABSOLUTE_PUBLISHED_DIRECTION_DIRECTORY \
  --reference ABSOLUTE_PUBLISHED_OUTPUT_SLOPE_FILE \
  --source-revision OWNER_SOURCE_COMMIT \
  --prelaunch-sha256 PRELAUNCH_DIGEST \
  --reservation-sha256 RESERVATION_DIGEST \
  --latest-exposure-sha256 EXPOSURE_DIGEST \
  --output ABSOLUTE_NEW_REPORT_PATH
```

The output path must be fresh and outside the attempt. The caller should reserve
and account for the report and any partial write. A successful return gives a
saved-byte candidate, not a separate observed process-exit receipt. A failed or
incomplete attempt remains incomplete; retain it without reporting a scientific
pass or retrying an uncertain model attempt.
