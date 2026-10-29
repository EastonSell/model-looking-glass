# Replay of pretrained trace smoke v1

The replay command independently reruns the fixed pretrained correctness smoke
and compares its saved hashes and event metadata with a previous successful
record. It is an execution check, not proof of the original run's provenance,
a performance benchmark, or a causal explanation. The exploratory heredoc
smoke predating this protocol has no durable v1 result and is not a reference.

Use the same committed source revision recorded in the reference's
`source_hashes.git_commit`. If main has advanced, check out that revision in a
separate worktree that contains both `pretrained_smoke.py` and
`pretrained_replay.py`. Disabling Git's automatic CRLF conversion when creating
that worktree avoids changing hashed source bytes. Use the same runtime,
backend settings and device model, and an offline cache containing
the pinned checkpoint assets. The reference and replay must use the same
device; CPU and CUDA hashes cannot establish a tolerance-based comparison.
Recorded environment identity includes the PyTorch build-configuration hash,
CPU capability, thread settings and one descriptive subnormal-multiplication
probe, but does not capture every native library or floating-point mode
or machine state; a match is not proof of exhaustive environment equivalence.
Interpreter and package-origin path fingerprints can reveal guessed paths and
do not hash package contents; they are not anonymous identifiers.
The smoke runner checks committed source bytes but does not attest which Python
bytecode was executed. A fresh empty bytecode cache prefix and disabled bytecode
writes can reduce ordinary stale-cache reuse; this remains an operator setup.
Physical GPU identifiers and raw device-visibility values are omitted. Thus,
matching public metadata cannot distinguish two otherwise identical GPUs.

From the source checkout, with paths adjusted for the local cache and saved
reference:

```sh
python -m looking_glass.pretrained_replay \
  --reference results/pretrained-cpu-v1/result.json \
  --cache-dir /path/to/offline-huggingface-cache \
  --output results/pretrained-cpu-v1-replay
```

The CLI starts a new Python process, defaults to CPU, and requires a new output
directory. The Python `replay_to_directory` API runs in its caller's process; it
does not itself create a fresh process. CUDA replay additionally requires
`--device cuda --lease-id <coordinated-window-id>`. The ID is an operator
attestation, **not** a device lock: drain the broker and coordinate an exclusive
window with other GPU work before using it. Do not use this command to overlap a
separate GPU model load with shared workloads.

The checker validates the v1 configuration and pinned model/tokenizer assets,
runtime identity, two baseline and reference output hashes, all 18 events per
mode, selectors, positions, invocations, retained byte counts, and the internal
relationships among direct, gradient and independent-reference value hashes.
The reference's source-hash fields are structurally checked and compared with
the fresh runner's locally verified committed blobs; the old record alone
cannot prove its provenance. The replay tool verifies its own file against the
same Git commit and saves its source hash. It compares complete ordered checks while
normalizing only each newly generated session's run ID; original IDs remain in
the saved records. IDs must be consistent within a mode, distinct across modes,
and disjoint from the reference's IDs. Unique IDs help detect reusing a record;
they are not proof that a new execution occurred. Timing, memory observations
and lease labels are not equality criteria. The original reference file's
SHA-256 is recorded in the replay result.

- `reproduced`: the compared identity and exact saved checks match.
- `incompatible`: source, model, configuration, device or runtime identity
  differs; this is not a numerical comparison.
- `different`: identity matches, but one or more output hashes or ordered
  checks differ. This does not diagnose why.

The latter two statuses are saved in `results.json` and cause a nonzero CLI exit.
A malformed reference or execution failure saves `failure.json` instead; a
pre-existing directory is never overwritten. `started.json` marks a begun run,
and `identity.json` captures available source/reference identity. The runner's
`prepared.json` records the checked protocol, source and expected assets before
model execution; failure records retain available runner phase and memory
diagnostics. A run without either terminal is incomplete. If the result was
already published when cleanup fails, the command raises and keeps the result
without adding a contradictory failure terminal; inspect the files before
deciding what happened. A failure record cannot be guaranteed if the filesystem
itself refuses writes.

Failure records retain exception text and diagnostic details for investigation.
They may contain local paths or other identifiers: keep the original privately
and review any copy for sensitive content before sharing or publishing it.

Exact hashes cannot measure the size of a difference, and the saved records do
not contain the numerical tensor values needed to evaluate a tolerance after
the fact. A meaningful cross-device comparison needs a separate prespecified
protocol and retained bounded values or numerical summaries.
