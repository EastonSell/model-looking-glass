# DistilGPT-2 public agreement preflight, CPU v1

The fixed public development set met its preregistered criterion: all 16
present-tense ` is` minus ` are` margins had the grammatical sign, including
8/8 singular, 8/8 plural, 8/8 congruent and 8/8 incongruent cases. The criterion
required at least 12/16 overall and 5/8 in each of those overlapping strata.
The past-tense ` was` minus ` were` margins are descriptive secondary data.
This is one small exploratory discovery set, not a population accuracy,
held-out evaluation, causal explanation or GPU performance measurement.

The operator observed the protocol and the complete inputs published before
the model run. Git commits and local event timestamps are not an independent
attestation of that sequence. The recorded references are:
source commit `e39fafa0a84843cf6413946a77c19ef09cbf025e` and frozen-manifest
commit `0e5e3ae68d8dc4fba51e4aadf378e4c3dac27e96`. The manifest SHA-256 is
`1b1adf4d72163bfa3e0399ed8ee92eba6b5c0eb82a47edf28cecde5ffb1deafe`.
The protocol records the original baker/bakers proposal and its tokenizer-only
writer/writers amendment before any model output, as well as the fixed
negative-result criterion. Exposure inventory v2 digest
`e64bc0b875fb295f03d7ad4d5180f6978d1d632487a98ea2824df01844ea4fb1`
reserved all 16 rows and their nonempty prefixes before the run.

The offline checkpoint was `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, with float32 eager attention,
evaluation mode, no cache, one CPU thread, seed `2026102817`, and CUDA
uninitialized. The registered-model-state hashes match before and after.
The execution-thread subnormal input/output probes passed both before and
after the work. Runtime, asset hashes, exact inputs, per-row target bits,
logits, margins, token coordinates and subgroup counts are in `run/result.json`.

`run/last_logits.f32` is the unmodified archive: 16 consecutive little-endian
float32 vectors, each 50,257 values, for 3,216,448 bytes. Its SHA-256 is
`7c262f3fe81236888a8f5fdb6d31f7835666674be9ffbf36f8ce2ca6a375eb6c`.
`run/attempt.jsonl` records the 16 ordered forward starts/completions and a
terminal result digest; `run/result.json` has SHA-256
`96bcb687dbdd9bb4017a8506b6acaffeb5a3da43478cf2da551a4a5cb0c723c8`.
The supervisor recorded exit code zero; its public record omits the original
command because it contains local filesystem paths, identifies that omission,
and retains a SHA-256 of the unredacted original kept in the local work area.
The result, attempt log, raw archive, stdout and stderr are unmodified copies.

Independent CPU-only reviewers checked the frozen tokenizer coordinates and
all continuation IDs before execution, then checked the archive and per-row
hashes, finiteness, target bits, margins, classifications, strata and terminal
digest without repeating a model forward. The study's later intervention
controls must capture their own activations on their bound device and runtime;
this preflight saved no activation donors.
