# Grouped broker CPU development replay, v3

This is a CPU-only integration check of an experimental grouped broker. It
reuses the four synthetic rows already published in the
[sampled DistilGPT2 CPU batch check](../distilgpt2-batch-cpu-v1-20261028/RESULTS.md).
Those same rows build the sampled policy and exercise the broker, so this is
not independent held-out evidence, a general batch-separability result, a GPU
experiment, a throughput benchmark, or an approved memory profile.

The run loaded `distilbert/distilgpt2` at revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa` once on CPU. Two teams submitted
one `[1, 4]` token row each per group, with an all-ones attention mask and
seed `2026102808`. The sample policy first compared four serial forwards with
two mixed forwards. The broker check then observed two serial model entries
and one `[2, 4]` grouped entry per group, and checked each returned team's
request identity and logits against its serial reference. For all four cases,
the elementwise rule was `abs(grouped - serial) <= 1e-6 + 1e-5 * abs(serial)`.

| Group | Case | Maximum absolute difference | Maximum error / tolerance | Violations |
| --- | --- | ---: | ---: | ---: |
| 1 | case-1-1 | 0.00003814697265625 | 0.0643481372185932 | 0 |
| 1 | case-1-2 | 0.000030517578125 | 0.05235598873388127 | 0 |
| 2 | case-2-1 | 0.000030517578125 | 0.0464215905332079 | 0 |
| 2 | case-2-2 | 0.000030517578125 | 0.046536598734667214 | 0 |

After receipt confirmation, release and disposal, the cooperative host ledger
reported zero used bytes and the **CPU-simulated** device ledger reported zero
active leases; the broker and worker closed. These are ledger observations,
not GPU memory measurements or proof of an operating-system memory bound.
Cancellation is covered separately by synthetic CPU tests. No raw logits were saved, so the
original numerical summaries cannot be independently recomputed from this
record alone. A new run can test whether they reproduce in its environment.
The raw record contains no wall-clock run timestamp.

The saved runtime reports Python 3.11.15, PyTorch 2.8.0+cu128 with CUDA hidden
and uninitialized, Transformers 4.57.3, one CPU thread and deterministic
algorithms. The recorded versions are not a complete dependency lock. The
`batch_separable` and certification fields in the sampled binding are local
development declarations, not evidence for untested inputs or devices.

## Preserved evidence

The original v3 protocol, runner and JSONL result are unchanged. Their SHA-256
digests are:

| File | SHA-256 |
| --- | --- |
| `protocol.json` | `3cb5dafe577fc7ad673e3550c31ef35c6ce11f3d6cd27f500b97648d60c46b2c` |
| `runner.py` | `0cfe29d4f66a2bdbac2782ee7779e88e351998f49275ea4aad1d8824a81e6a10` |
| `result.jsonl` | `b5538fe57c286eb6b5de0ae2280bf2f133576ddb7810cdab823dcf5d3f3ec5a6` |
| `grouped_broker.py` | `b4d4f4ada71be86309c69dbcf00f5b7fe0b72f4db14f1bf2975bfe19b6e0baf2` |
| `source.bundle` | `6787c35585749f5738725e4b28d21b029ae2f03c0ca02cbbe07706c259f2bb12` |

The execution used the pre-merge commit
`c9a3db4f7a0d5e61a82a6cc3a812bf40905f0abf`. It was not a published GitHub
commit. `source.bundle` preserves that exact commit and its unpublished parent
on top of public prerequisite `00f77ccc1c71afca186f6965437ead8749d6a2b4`.
The plain `grouped_broker.py` is the historical source snapshot, not the
current importable module. The result's first record additionally stores
SHA-256 digests for eight source files. The later merged broker contains
additional health and clock fixes; this replay must not be attributed to it.

`v2-archive/` preserves the earlier completed development record, its runner,
protocol, and the then-uncommitted broker source. That prior source hash is
`b9b1643e81fda8e55a4300454def48a8de6b95b36be5da3ca5b2c4a9d756a218`;
its result hash is
`fe6a9f0001ce6c75ad8ffa623cf74f0301bc6414e4d6f3fc2a1035b360c29964`.
It predates subsequent custody and liveness fixes and is retained as history,
not substituted for v3.

## Reproduce on CPU

From a full repository clone that contains the public prerequisite, recover
the historical source without modifying the current checkout:

```sh
git bundle verify results/grouped-broker-cpu-dev-v3/source.bundle
git fetch results/grouped-broker-cpu-dev-v3/source.bundle \
  refs/heads/archive/grouped-broker-cpu-dev-source-v3:refs/reviews/grouped-broker-cpu-v3
git worktree add --detach ../grouped-broker-cpu-v3-checkout \
  refs/reviews/grouped-broker-cpu-v3
```

Use a fresh Python process, an environment compatible with the recorded
versions, and an already populated Hugging Face cache containing the pinned
checkpoint. The loader verifies its pinned assets and performs no download.
Choose a new output path; the runner refuses to overwrite it.

```sh
CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 python -B \
  results/grouped-broker-cpu-dev-v3/runner.py \
  ../grouped-broker-cpu-v3-checkout /tmp/grouped-broker-cpu-new.jsonl \
  --protocol results/grouped-broker-cpu-dev-v3/protocol.json \
  --cache-dir /path/to/huggingface/hub
```

Check the new record's commit, script and source hashes before comparing its
results. Different CPU kernels or dependencies can change floating-point
values; the protocol's frozen tolerance, rather than byte identity, determines
whether this sampled comparison passes.
