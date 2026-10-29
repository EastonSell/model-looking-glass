# Short-input DistilGPT2 CPU reference

This development run retained full float32 vocabulary logits for six fixed,
publicly reserved inputs. For both right-padded scientific inputs, every saved
active-position logit was bit-identical to the corresponding unpadded CPU
reference. The last positions added by padding were not compared.

| Input | Active / padded tokens | Compared logits | Maximum absolute difference | Tolerance violations |
| --- | ---: | ---: | ---: | ---: |
| Gas | 15 / 16 | 753,855 | 0 | 0 |
| Ligand | 23 / 32 | 1,155,911 | 0 | 0 |

The planned tolerance was `1e-6 + 1e-5 * abs(reference)` for each logit; exact
saved-byte equality is stronger for these two observations. One 16-token and
one 32-token all-active synthetic input were also forwarded once each and their
full logits retained; those rows have no padded/unpadded comparison. These are
small exposed development inputs, not a held-out accuracy result or a general
claim about padding, GPU equivalence, service throughput or capacity.

The recorded model is `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, weights SHA-256
`e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`.
The saved model metadata reports CPU float32, eager attention,
`use_cache=False`, Torch `2.8.0+cu128` and Transformers `4.57.3`. The pinned
owner sets seed 29 and checks one Torch CPU thread. Model pretraining exposure
to these inputs is unknown.

Before this reference, a separate CPU teardown probe forwarded the already
reserved synthetic-16 and padded gas rows twice each, in the order synthetic,
gas, gas, synthetic. Its four entry records are included. That probe used fake
CUDA/owner telemetry and a fake output recorder, which retained no logits; its
operator report is not a GPU or native-cleanup observation. The six reference
rows were then each run once in the separately observed CPU controller. Public
inventory entries may still carry their historical planned labels.

The actual execution checkout recorded local commit
`80f82a81c62403045c9c33295e6a61b4f410d446`; its complete tree
`a15f3ebac5c129f03d9cb9e0d52eabac990958e2` is identical to public commit
[`75e3ed56`](https://github.com/EastonSell/model-looking-glass/commit/75e3ed56cad3f0586b8ea4f4010f0d116e1ca83b).
The direct controller was observed reaped with exit 0 and no timeout; its
private exit receipt SHA-256 is
`4f2711b9064c38ec08936341179cc06151c1a848c97d97ba62b5f95903649e6b`.
The receipt digest is a commitment to private PID-bearing bytes, not publicly
replayable process identity or independent model-execution attestation. The
source and saved-tensor checks are cooperative consistency evidence, not proof
of process-tree custody.

`archive-index.json` pins all 26 parts and 28 archive members. From this
directory, reconstruct into a new absolute destination:

```text
python -I -S -B rebuild_evidence.py --destination <new-absolute-directory>
```

The reconstruction verifies the pinned index, part lengths and SHA-256/Git
blob hashes, whole archive, exact member inventory and each extracted file.
It leaves the rebuilt ZIP beside the extracted `data`, `inventory`,
`currentness` and `rehearsal` directories. A failure leaves partial files in
the destination; choose a separate fresh directory for a later attempt.

Run the independent public saved-byte reader with a Git checkout containing
the public source commit:

```text
python -I -S -B audit_short_cpu_reference_public_v2.py \
  --run <new-absolute-directory>/data --repo <git-checkout> \
  --git <absolute-git-executable> --output <new-absolute-report-path>
```

The expected public replay report is `short-cpu-reference-public-replay-v2.json`.
The public reader checks the saved source and tensors; the original inventory
and private process receipt are digest commitments in that reader. The
archive separately includes the pinned public v33 inventory and scoped v37
and v38 public-currentness observations. These point-in-time comparisons do
not establish complete historical exposure or private-split membership. The
separately included private-audit reader and report document an independent
check of the original local commit and PID-bearing receipts; reproducing that
check requires those private originals.
