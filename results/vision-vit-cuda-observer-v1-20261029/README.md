# ViT CUDA exposed-pair observation

The supervised CUDA observer completed six forwards on one previously exposed
cat/dog image pair. On the cat image, the dog-minus-cat group log-odds moved
from **−2.984785** to **−2.730793** when four dog activation vectors captured
on the GPU were inserted; using the saved CPU donor vectors gave **−2.730794**.
Both patched scores remained negative. This is a descriptive comparison on
one exposed pair, not a held-out replication or a general localization result.

The score is `logsumexp(dog logits) − logsumexp(cat logits)` over 118 dog
classes and five cat classes. Those unequal group sizes matter. The selected
tokens have already mixed information through self-attention, and the image
and model may have been encountered during pretraining. The observer does not
establish that the patch transferred animal-specific information. The earlier
[CPU study](https://github.com/EastonSell/model-looking-glass/blob/ecf6ab79164a8068fb4b42348efe440e342402e2/results/vision-coco-vit-v1-20261029/README.md) reports the held-out
result and the spatial, control, and provenance limits in detail.

## Observation and controls

The pinned ViT-tiny checkpoint was run in float32 eager mode on an NVIDIA
GeForce RTX 2070 SUPER with PyTorch 2.8.0+cu128, one CPU thread, deterministic
settings, TF32 disabled and autocast disabled. The fixed pair is COCO mirror
cat 520531 and dog 377575. The four token coordinates were 161, 162, 175 and
176 after encoder block index 2 (the third block). The protocol and exact source
are at [commit 7df264b](https://github.com/EastonSell/model-looking-glass/tree/7df264b4c7d76ab6f9b360cba9ecaa17867d92da)
and in [the frozen protocol](../../docs/VISION_VIT_CUDA_OBSERVER_V1.md).

The six operations were cat clean, dog clean, cat self-patch, cat with GPU dog
donor, cat with the previously saved CPU dog donor, and a cat clean repeat.
The saved GPU self and repeat logits and early states are bit-identical to the
GPU cat baseline. Same-forward witnesses show that all four requested vectors
landed exactly and that the other tokens were unchanged. The independent
saved-byte replay rechecks these controls and all 1,000 logits per forward.

| Comparison | Maximum absolute difference | Mean absolute difference |
| --- | ---: | ---: |
| CPU vs GPU cat clean logits | 1.9133e−5 | 3.9548e−6 |
| CPU vs GPU dog clean logits | 1.5736e−5 | 3.4630e−6 |
| CPU vs GPU cat logits with the same CPU donor bytes | 1.5616e−5 | 3.5980e−6 |
| GPU cat logits with GPU vs CPU donor bytes | 5.9605e−6 | 1.4556e−6 |
| CPU vs GPU cat early state | 1.8388e−5 | 8.5513e−7 |

These are descriptive numerical differences, with no prespecified agreement
threshold. The CPU-donor comparison also differs in receiver state and
downstream kernels. Comparing GPU and CPU donors on the same GPU receiver
isolates a donor-state contrast, but is not a complete causal decomposition.

## Memory and custody

Across eleven synchronized stage snapshots, the cumulative PyTorch
process-local peak since the pre-model reset was **57.734 MiB allocated** and
**60 MiB reserved**. The minimum sampled device-wide free memory was **7,021
MiB**; these are point samples and can miss transients. At the worker's final
cleanup snapshot, its PyTorch allocated and reserved counters were both zero
and its recorded model and input weak references were dead. These observations
exclude CUDA context and other non-PyTorch allocations and do not prove native
CUDA teardown, full process-tree drainage, or global device idleness. They are
not a latency or throughput benchmark.

An independently observed controller exited zero and was reaped; the saved
supervisor records a clean, zero-exit direct child with no timeout. A later
unlocked operator observation found the marker and attestation absent and
sampled 7,063 MiB free. The physical lease is cooperative and assumes no
CUDA-using descendants; the operator's later observation is not an admission
or future-headroom guarantee. The controller, child, and raw operator receipts
are retained privately because they include local process identifiers or device
details. The [public projection](vision-vit-post-exit-public-projection.json)
and [coordinator reconciliation](coordinator-reconciliation.json) record their
hash commitments and scoped observations. Hash commitments are not public
execution attestation. Source, interpreter, dependencies and native libraries
remain trusted.

Before this attempt, an unrelated earlier GPU incident was released through a
separately qualified operator risk decision without a clean CUDA attestation.
This observer's successful exit does not establish that earlier native cleanup.

An earlier separately recorded attempt failed in the isolated bootstrap's
first Git lookup before importing the project, starting the supervisor, or
accessing the device; no output directory was created. A corrected Git
executable passed a CPU-only bootstrap probe before the one successful
attempt. The failed stderr and process receipts remain private; this is not
treated as a CUDA cleanup event.

## Evidence and independent replay

The three `evidence.partNN` files concatenate to a ZIP whose SHA-256 is
`5ab3cbae287ad3dc93aba9fa3b7a6be3f211fe862881f9d3fd0c434accfda418`.
The [index](evidence-index.json) binds each part and each of the 44 raw files:
six forward JSON/bin pairs, six returned-logit receipt pairs, eleven memory
stage records, and nine CPU reference files. These are byte-preserving copies;
the private PID-bearing process records are not included. The independently
written [saved-data audit](vision-vit-cuda-independent-saved-audit-v1.json)
also documents a private receipt-linked review. Its source is included, but its
full private-receipt invocation cannot be reproduced from the public bundle.
The independent [provenance audit](vision-vit-cuda-provenance-audit-v2.json)
and [reader](audit_vision_vit_cuda_provenance.py) similarly record a private
receipt-linked check and its scope.

To verify and extract the public files into a new or empty directory, then
replay the numerical and source checks without a model forward:

```text
python -I -S -B unpack_evidence.py <empty-evidence-directory>
python -I -S -B audit_vision_vit_public.py --repo <checkout-with-7df264b> \
  --run-dir <empty-evidence-directory>/data \
  --reference <empty-evidence-directory>/reference \
  --closure vision-vit-cuda-public-closure-v1.json \
  --coordinator coordinator-reconciliation.json \
  --independent-report vision-vit-cuda-independent-saved-audit-v1.json \
  --assets <pinned-vit-asset-directory> --git <absolute-git-executable>
```

The asset directory must contain `config.json`, `preprocessor_config.json`, and
`model.safetensors` from [WinKawaks/vit-tiny-patch16-224 at revision
77d1485](https://huggingface.co/WinKawaks/vit-tiny-patch16-224/tree/77d1485af66b34d4ed0fe95dbb0c60c7496f950b).
Their exact hashes and the 17 source hashes are checked by the public replay.
The saved [public replay result](vision-vit-cuda-public-replay-v4.json) can be
compared with the replay output. This public replay
checks released tensor and metadata bytes, the reference, source and checkpoint,
not the private process identity or actual model execution.
