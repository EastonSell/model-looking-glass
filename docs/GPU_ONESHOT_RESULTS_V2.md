# DistilGPT2 GPU memory pilot results

On 2026-10-29 UTC, a short-lived supervised process loaded the pinned
`distilbert/distilgpt2` revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`
on an NVIDIA GeForce RTX 2070 SUPER. With batch size 1 and input IDs
`[15496, 995]`, the float32 eager model in evaluation/inference mode returned
finite logits of shape `[1, 2, 50257]`. CUDA runtime identity was checked
against the physical device under the cooperative lease. The process used
PyTorch 2.8.0+cu128 and a 25% PyTorch allocator cap. This is one direct forward,
not a performance benchmark or a capacity profile for the shared service.

The [v2 result](../results/gpu-oneshot-v2-20261029.json) records synchronized
snapshots in bytes. The table converts them to MiB:

| Stage | Allocated | Reserved | Device-wide free |
| --- | ---: | ---: | ---: |
| Before model copy | 0.00 | 0.00 | 7,131.00 |
| After model copy | 319.24 | 356.00 | 6,775.00 |
| After forward | 351.62 | 388.00 | 6,693.00 |
| After disposal, before workspace clear | 32.00 | 32.00 | 7,049.00 |
| After workspace clear | 0.00 | 0.00 | 7,081.00 |

The observed peak was 352.30 MiB allocated and 388.00 MiB reserved. The
model, input and output weakrefs were dead before workspace cleanup. After
the version-bound PyTorch private cuBLAS workspace-clear call and another
synchronization, current allocated and reserved values were both zero. The
child attested clean and was reaped with exit code 0. Device-wide free memory
can vary with unrelated desktop activity; the allocator ceiling does not cap
all CUDA allocations. The saved code records a successful live UUID check,
but the two raw UUID values were not retained for an independent post-hoc
comparison. The finite-output flag and SHA-256 are runner observations: the
logits were not saved, so the digest cannot be independently recomputed.
The result hashes five project files, not the package/native dependencies or
all code executed by the process. Windows WDDM limits process-visibility claims.

The original [v1 attempt](../results/gpu-oneshot-v1-20261029.json) completed
its forward but failed its cleanup assertion and left the unsafe marker. It did
not save its intermediate allocator counters, so the cause of that failure is
unknown. An independent assessment and explicit operator recovery preserved the
marker and failure evidence, reacquired both cooperative locks, and removed the
marker without manufacturing a clean attestation. The
[redacted recovery record](../results/gpu-oneshot-recovery-v1.json) states the
evidence and its limits. The 32 MiB observed in v2 is consistent with the
workspace explanation but does not establish the cause of the v1 failure.

The frozen v1 and v2 execution commits are
`23cf8ec9a10e168cc6e6fd486908287697f6c751` and
`df9d66336d9375c45660df2f1c62b9f0fc2808b0`; the latter's result has SHA-256
`9c9af77278053954ed8ef0a80d163d16fa4d78243c76ead205df0909b2f4ede4`.
The [v1](GPU_ONESHOT_V1.md) and [v2](GPU_ONESHOT_V2.md) protocols describe
the corresponding execution and cleanup paths. The independently reviewed
CPU fake tests exercise failures and lifecycle checks; they do not simulate a
real CUDA allocator or validate behavior on other devices.
