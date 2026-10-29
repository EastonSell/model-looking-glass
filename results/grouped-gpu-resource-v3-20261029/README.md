# Grouped DistilGPT2 GPU development run

On this fixed workload, the supervised owner reported successful serial and
two-row comparisons and the controller exited successfully. The run used one
resident `distilbert/distilgpt2` model and executed 12 model entries: eight
single-row `[1, 4]` entries and four two-row `[2, 4]` entries. The first two
single-row controls completed before the first two-row entry. This is a
provisional functional and resource observation, not a throughput benchmark
or a general capacity result.

The recorded source revision was [05ca62b9](https://github.com/EastonSell/model-looking-glass/tree/05ca62b99aa3930eaa91654b2d312024885e20e7);
its checked source bytes matched the retained manifest. It includes the
[v3 protocol](https://github.com/EastonSell/model-looking-glass/blob/05ca62b99aa3930eaa91654b2d312024885e20e7/experiments/grouped_gpu_dev_v3.json)
and [plan](https://github.com/EastonSell/model-looking-glass/blob/05ca62b99aa3930eaa91654b2d312024885e20e7/docs/GROUPED_GPU_DEV_PLAN.md).
The protocol SHA-256 is `011e214dbdc57e29869f41e38082cd1fa4baaf15747169d0d778e6f27b33a6f1`.
The model revision was `2290a62682d06624634c1f46a6ad5be0f47f38aa`; the
independently checked source and checkpoint hashes are in
[source-audit-05ca.json](source-audit-05ca.json).

## Comparisons and routing

The owner reported that all four sampled comparisons passed the elementwise
rule `abs(grouped - serial) <= 1e-6 + 1e-5 * abs(serial)`. Its largest reported
sample absolute difference was `5.340576171875e-5`: an absolute difference above
`1e-6` can still pass because the rule also includes the relative term. The
maximum summaries alone do not establish the per-element result.

For each of four later deliveries, the saved record reports a match to its
own serial reference and a mismatch to the other row's reference. The saved
records associate each delivery with its intended team, row digest and one
of four distinct job slots. The later rows were excluded from the policy
construction sample for this run; they are not an independent behavioral
evaluation. Full logits were not retained, so an independent reader cannot
recalculate the reported elementwise comparisons from this bundle.
[sample-evidence.json](sample-evidence.json), [frozen-policy.json](frozen-policy.json)
and the projected [result](result-public.json) retain the recorded summaries.

## Observed memory and cleanup

The saved result contains 54 ordered memory observations, including 28
admissions. The largest recorded PyTorch allocated and reserved peaks were
`370,421,760` bytes (353.26 MiB) and `406,847,488` bytes (388 MiB), and the
minimum sampled device-free memory was `7,018,119,168` bytes (6,693 MiB).
Peaks were reset before each forward: these maxima are across recorded stages,
not a continuously accumulated run-wide peak. Counter reads are sequential
point observations and do not bound future external memory pressure.

After guarded cleanup, the owner recorded zero PyTorch allocated and reserved
bytes and zero accounting leases. The source fixes a 25% allocator cap and
separate logical device and host commitments; those are admission/custody
limits, not a measurement or cap of process RSS. This workload uses zero trace
allowance and does not test intervention tracing. Saved cleanup and a reaped
owner do not independently attest native CUDA cleanup. WDDM limits process
visibility, and point-in-time device observations do not prove global idle.
The 54 original [memory records](memory/) and [resource profile](resource-profile.json)
are included.

## Exposure and chronology

The [v18 exposure inventory](https://github.com/EastonSell/model-looking-glass/blob/94a55846d438cf2fa8bc8fbbde9c5609de5021d0/looking_glass/pretrained_exposures_v18.py)
recorded rows 103–106 as observed and rows 107–110 as planned before this run;
the latter were used by this successful owner. Its canonical inventory
SHA-256 is `3fa3eda9508ca314505aa658b1c02d04dbc98681bfc776337cd9ca1c9c384f5a`.
Prior or outside exposure is not known to be complete.

An earlier controller at source `f112060375e9b107277ac41bbfec6779abe4ae2a`
exited 1 before owner supervision: the protected run directory blocked its own
create-only startup publication with Windows error 32. The retained partial,
exit observation and source order were independently checked in
[startup-failure-audit.json](startup-failure-audit.json); it is a separate
pre-owner failure and contains no GPU profile. The corrected source protected
the manifest in its own directory and used a fresh run and result path.

After that failure and before the successful controller launch, a trusted
operator recorded a clear bidirectional complete-row-prefix comparison of the
eight public rows against the natural study's 16-case private split. The
[comparison receipt](natural-split-comparison.json) was recorded at
`2026-10-29T07:55:53.161648+00:00` and binds the included
[public inputs](natural-split-public-inputs.json). This is a recorded
comparison under that scope, not proof of general distributional independence,
complete exposure history or clearance of a different private study.

## Evidence and reproducibility

[result-public.json](result-public.json) is a documented projection of the
closed controller result, whose original SHA-256 was
`218e5e7144667f309e82532699c6bac05ecb31da6616c5319da099dd64eefbba`.
[project_result.py](project_result.py) removes the private run identity and
the already-redacted physical-device field, and replaces the four distinct
broker job identities with verified ordered slots. The original result and
private inventories are withheld. Their hashes in the projected inventory and
[operational review](operational-review.json) are commitments, not publicly
reconstructible records. The terminal's `partial_memory_observations` and
`partial_sample_evidence` are duplicated fallback fields, not a second run.

With Python's standard library, run from this directory:

```text
python -B audit_saved_result.py result-public.json source-audit-05ca.json
```

The checker validates saved schedule, row identities, routing statuses,
admission arithmetic, copied evidence hashes and cleanup consistency; it does
not replay unsaved logits or attest model execution. Its output is retained in
[public-replay.json](public-replay.json). An independent resource readback is
in [resource-audit.json](resource-audit.json). The controller exit of zero was
observed by the coordinator; the saved supervisor reports an owner exit of zero,
clean and without timeout. The post-exit marker/attestation check is a point
observation. Source and dependency checks assume cooperative local state and
do not independently attest all bytes executed or native teardown.

[inventory.json](inventory.json) lists the other files in this directory with
their byte lengths and SHA-256 values; this README and the inventory omit
themselves from that list.
