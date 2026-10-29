# Five-token agreement GPU resource pilot: incomplete attempt

On 29 October 2026 (UTC), a supervised DistilGPT2 resource pilot reached its
900-second child-wait timeout. The controller exited with code 1 and reported
`quarantined`; its supervisor reported `timed_out=true`, `clean=false`, and child
exit code 1. **This is an incomplete attempt, not an approved resource profile.**
The retained observations do not establish a safe admission budget, a
worst-case memory bound, or a behavioral or held-out result.

The fixed schedule was one cold warmup and 13 ordered warm actions. The saved
prefix contains ten completed samples: the warmup, clean capture, and eight
cross-patch actions. The next `null` action has only `started`, `baseline`, and
`baseline_check` records; its callback outcome is unknown. There is no completed
`null` sample or subsequent self-patch and block-control sample. The independent
partial audits matched the ten journal events to the completed sample IDs and
checked the 63 contiguous stage records. No candidate, final profile, cleanup
record, or clean owner attestation was saved.

## Workload and memory observations

The run used pinned `distilbert/distilgpt2` revision
`2290a62682d06624634c1f46a6ad5be0f47f38aa`, one batch of five tokens, the
fixed public development inputs, eager attention, float32 evaluation, and
no KV cache or autocast. The one supervised owner used an NVIDIA GeForce RTX
2070 SUPER, PyTorch `2.8.0+cu128`, and the native PyTorch allocator with a
25% per-process fraction. The fraction limits that allocator rather than all
CUDA context or driver memory. The [published source revision](https://github.com/EastonSell/model-looking-glass/tree/ff03d3c9739c894b33839e907b633109dba2a7b3)
defines the exact workload and measurement protocol.

The collector synchronized and reset its allocator peak counters for each
action. These are **per-action PyTorch allocator peaks**; the maximum in this
partial prefix is not an uninterrupted run-wide or device-wide peak.

| Completed sample | Peak allocated, bytes | Peak reserved, bytes |
| --- | ---: | ---: |
| Cold warmup | 369,544,704 | 406,847,488 |
| Clean capture | 369,575,424 | 406,847,488 |
| Layer 1 attention cross-patch | 369,563,136 | 406,847,488 |
| Layer 1 MLP cross-patch | 369,600,000 | 406,847,488 |
| Layer 2 attention cross-patch | 369,636,864 | 406,847,488 |
| Layer 2 MLP cross-patch | 369,673,728 | 406,847,488 |
| Layer 3 attention cross-patch | 369,710,592 | 406,847,488 |
| Layer 3 MLP cross-patch | 369,747,456 | 406,847,488 |
| Layer 4 attention cross-patch | 369,563,136 | 406,847,488 |
| Layer 4 MLP cross-patch | 369,600,000 | 406,847,488 |

The largest recorded peaks were 369,747,456 bytes allocated (about 352.62 MiB)
and 406,847,488 bytes reserved (388 MiB). The lowest saved device-free
observation among the stage and budget snapshots was 7,018,119,168 bytes
(6,693 MiB); the CUDA-reported total was 8,589,606,912 bytes. The separately
queried NVIDIA inventory rounded the total to 8,192 MiB, a 327,680-byte
difference within the protocol's 1 MiB tolerance. These are sampled readings,
not a continuous record of device-wide usage.

The run held a 2 GiB resident claim and a 1.5 GiB job claim, with 2 MiB each
for GPU and host trace claims. The saved preload and 21 available stage
headroom checks satisfied their recorded 3.5 GiB commitment and
1,288,441,037-byte reserve inequalities. These cooperative claims describe
the attempted admission policy; they are not measurements of actual physical
memory consumption or a validated profile for the unfinished schedule.

## Termination and limits

The operator observed the unsafe marker still present after timeout. The
controller and exact owner process had exited, and a post-exit NVIDIA query
did not list that owner. Windows WDDM process visibility is limited, and the
absence of a listed PID does not attest CUDA cleanup. The exact source's
known-exit timeout path returns after its child stop/wait and Windows Job
finish calls; the saved artifacts do not include raw Job counts or a member
inventory. That ordering is an inference from the source and terminal result,
not independently recorded proof of process-tree or CUDA cleanup. The marker
remained retained at the terminal observation; later device recovery requires
a separate assessment.

The operator recorded reports of concurrent CUDA-hidden CPU work, including
synthetic hook tests and a public factorial study; the exact process intervals
were not independently witnessed. This pilot did not attempt CPU or GPU timing
isolation, and its callback durations are not a latency or throughput result.
Other desktop activity, device state, and native allocations were not controlled.

## Public subset and verification

The original private run and terminal evidence remain separately retained.
This directory publishes only a selected subset:

- `evidence/stages.jsonl` concatenates the 63 original single-line stage files
  in order without changing their bytes. `inventory.json` gives the original
  filename, offset, length, and SHA-256 of each slice.
- `evidence/events.jsonl` is the original ten-record event-journal prefix;
  `preload-admission.json` is the original saved preload observation.
- `terminal-projection.json` is a derived, limited projection of the private
  controller result and contemporaneous operator observations. It omits
  physical-device identifiers, process IDs, private paths, and the marker
  contents. It does not independently attest cleanup or marker state.
- `inventory.json` inventories this public subset only. The public subset
  cannot independently replay the private source/owner/device identity
  bindings or reconstruct the omitted private terminal evidence.

Run `python verify_subset.py` from any directory to check the subset hashes,
stage slices and order, journal/sample IDs, recorded headroom arithmetic, and
the peak/free values above. This check does not attest execution, device
cleanup, or profile approval. Independent reviewers also compared the public
stage slices and journal bytes with the retained private originals.
