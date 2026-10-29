# Agreement-direction saved-byte scientific replay v1

This reader independently reconstructs the fixed numerical comparisons in
`AGREEMENT_DIRECTION_STUDY_V1.md` from a closed candidate journal. It uses only
the Python standard library and saved data; it does not load a model, issue a
forward, use a GPU, or access the network.

The reader pins the published 40-case file, direction manifest and three
binary64 direction files. It checks the protocol digest recorded in the pinned
case file. It checks the complete ordered 784-forward schedule and the eight
family-geometry and sixteen centering records, input identities, content
addresses and tensor dimensions. It independently reconstructs doses, selected
float32 landing bits and recorded geometry, verifies the unchanged portions of
each captured activation, and checks full saved last-logit equality for clean
repeats, no-ops and zero-dose interventions. The exact journal and tensor file
inventory and terminal byte accounting must agree.
For the real-width path it also rejects a recorded environment, numeric runtime,
or root cache/attention/layer-count/dimension setting that contradicts the fixed owner
configuration. Other module settings and dependency/asset evidence require a
separate source and receipt review.

The report retains every target and context, including zero-dose cases. It
computes the fixed margins, signed and even responses, both descriptive
threshold decisions and their full denominators, per-pair projection losses,
donor comparisons, vocabulary ranks, top-20 entries, and
`KL(clean || intervention)`. Top entries are GPT-2 vocabulary IDs and logits;
obtain token spellings separately with the pinned tokenizer if needed. The
cutoffs are exploratory, not statistical tests, and a passing comparison does
not by itself establish transfer of an abstract grammatical feature. Output
bias, surface morphology, and asymmetric disruption remain alternatives.

Run from a checkout containing this reader, after independently establishing
the exact input and external binding values. This example uses POSIX shell line
continuations; in PowerShell, put the arguments on one line or use its own
continuation syntax:

```text
python -B -m tools.agreement_direction_science_replay_v1 \
  --case-file PATH_TO_CASE_JSON \
  --direction-directory PATH_TO_DIRECTION_DIRECTORY \
  --journal-directory PATH_TO_EXACT_ATTEMPT_DIRECTORY \
  --source-revision 40_HEX_SOURCE_REVISION \
  --prelaunch-sha256 64_HEX_DIGEST \
  --reservation-sha256 64_HEX_DIGEST \
  --latest-exposure-sha256 64_HEX_DIGEST \
  --output NEW_REPORT_JSON
```

The output path must not already exist. The reader serializes finite JSON
before creating it, but an I/O failure after creation can leave a partial file.
The attempt directory must contain exactly `started.json`, `events.jsonl`,
`terminal.json`, and `tensors/`; use a separate exact copy if a publication
folder also contains reports or other files. The attempt and tensor directories
themselves must be ordinary directories, including on Windows; their ancestor
paths and the source location are trusted inputs, not a security boundary. The
CLI requires the real 768-wide,
50,257-token dimensions; reduced synthetic dimensions are available only to
the Python test API, whose reports identify themselves as `synthetic_test`.

`consistent_candidate` means that the supplied saved records satisfy these
checks. The recorded full-sequence-finiteness flag is checked, but only the
saved last-position logits can be independently inspected. The supplied source
revision and three external digests are compared with the recorded fields;
this tool does not authenticate those pins, the source tree, a process exit,
custody or exposure chronology, or private-split disjointness. Independently
verify those facts and the direct child exit before interpreting a result as a
completed study. A failed or incomplete attempt is not a negative result.

The calculations reproduce the protocol's prescribed Python binary64
operation order. Transcendental and square-root results depend on the local
runtime and math library; a mismatch fails the replay and requires investigation
rather than establishing which environment or record is wrong.
