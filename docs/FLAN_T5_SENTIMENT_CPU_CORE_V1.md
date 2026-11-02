# FLAN-T5 sentiment CPU execution core

This is the execution and saved-journal component of the prospective protocol
in `FLAN_T5_SENTIMENT_INTERVENTION_V1.md`. It does not select cases, load a
checkpoint, launch a process, authorize an attempt, or publish a scientific
result. All tests in this slice use fabricated inputs and models. The separately
reviewed owner must establish the protocol's source, asset, selection, exposure
and custody gates before the first forward, and establish actual exit and
cleanup independently afterward.

## Interface and admission

`selection_to_groups(selection, expected_runner_revision=..., vocab_size=...,
special_ids=...)` checks the selector record's structure, first-four eligible
order, reciprocal donor mapping and selected token alignment. It returns an
empty tuple if fewer than four units qualify. It does not authenticate the
selection, verify tokenization, check publication chronology or establish the
custodians' observations; those are owner duties. The owner must not call the
execution core with an empty or unverified selection.

Create `EvidenceJournal` in a new trusted attempt directory before model
loading, with the default `Limits`, after independently verifying that the
specified free-space and source/asset preconditions hold. The owner passes its
already loaded model, tokenizer and selected groups to
`run_cpu_core(model, tokenizer, groups, journal, custody_check=...)`. The
zero-argument callback must raise if the supplied custody binding is no longer
valid; its return value is not an attestation. The core calls it at setup,
immediately around each encoder and decoder call, and before its provisional
terminal. A callback error stops the attempt. Full committed-source and asset
verification before model loading and after the run remain the owner's duties.

The production entry requires the exact Transformers `T5ForConditionalGeneration`
and `T5TokenizerFast` classes, the specified start/EOS/pad IDs, a 512-wide,
32,128-token CPU float32 model, evaluation mode for every child module, eager
attention, disabled config cache, deterministic algorithms without warn-only,
one intra-op and one inter-op thread, highest float32 matmul precision, and no
CPU autocast. `CUDA_VISIBLE_DEVICES=-1`, both thread environment variables set
to `1`, and both Hugging Face offline variables set to `1` are also required.
CUDA must remain uninitialized. These checks are observations in a trusted,
exclusively controlled process; they do not authenticate the Python runtime,
native libraries, arbitrary model attributes, or checkpoint bytes. The caller
must not mutate model, tokenizer, groups, journal or custody while executing.

The fixed full schedule contains 32 clean baselines, 32 passthrough, 96
self-site, 32 full-state donor, up to 256 matched, up to 32 natural donor, and
four repeat outputs. All baselines and controls are retained even for an
unsupported group. An unsupported or dose-ineligible group is unmeasured, not
a measured negative; the four-unit denominator remains fixed. The core retains
all proposed rounded landings and checks dose and direction before a matched
forward. The generated sequence, EOS-inclusive teacher scores, full teacher
logits, greedy final-position logits and first-step distribution comparisons
are recorded as specified by the protocol.

## Evidence and failures

The single-threaded journal stores `started.json`, an append-only
`events.jsonl`, sequential little-endian float32 tensor files, and a
`terminal.json` when terminal writing succeeds. Tensor references include
shape, byte count and SHA-256. Component starts are flushed before the model
call; returned tensors are persisted before the component is accepted. A
failure after a component return attempts to retain a bounded prefix and records
whether a return was observed. An interruption, rejected component, exhausted
capacity, changed runtime or failed custody check stops subsequent work. A lost
return or partial write is ambiguous; preserve the attempt and do not replay it
automatically. An incomplete terminal can itself be absent or partial after an
I/O failure, and must not be treated as successful cleanup.

The maximum retained attempt is 1.25 GiB, including 64 MiB metadata and a
separate 4 MiB failure reserve; at least 2 GiB free space is checked when the
journal is created. Its worst-case allocation covers 2,940 component calls,
the full specified outputs, state witnesses and candidate vectors. The owner
must account for external custody records and private logs within the protocol's
8 MiB share of metadata. These limits cover journaled file bytes, not process
RSS, allocations made by model libraries, or arbitrary caller-owned objects.
The attempt directory and model process are trusted and quiescent; the file
checks are not a filesystem sandbox or a guarantee against a malicious writer.

`complete_candidate` means only that the core reached its provisional end and
saved its own observations. It does not establish source/asset authenticity,
process-tree teardown, model disposal, or correctness of the scientific
reduction. A separate reader must validate the saved bytes and recompute the
decision, and the parent must verify its own cleanup and exit evidence. The
test-only `_run_fabricated` interface and small journal limits cannot be used
as a live admission path.
