# BERT agreement observer: CPU-tested GPU adapter draft

This component prepares the existing `BERT_AGREEMENT_V1.md` family schedule
for a separately supervised device owner. It does not run the pretrained model,
create a GPU owner, profile the full family, or authorize evaluation. Both the
CUDA entry and non-CPU execution in the private prototype fail immediately.
The existing public `run_family` remains CPU-only and keeps its existing
observer and CPU audit defaults.

The private schedule seam accepts a trusted observer but retains the same 140
forward order, four donor cells, five comparison sites, controls, selected
logits, float32-bit arithmetic and failure events. The prototype accepts one
CPU int64 row of ten tokens, builds an all-ones mask, and uses the same six
block paths and batch/token/selector/invocation coordinates as the CPU runner.
Its before/patch/after plan verifies exact action metadata, finite complete
logits, self/donor landing bits, the eager-attention configuration and actual
per-block attention classes, and disabled CPU and execution-device autocast
before and after the forward. These checks describe observed
state at their boundaries; they do not detect transient same-process changes.
The returned mask logits and captures are owned CPU tensors copied through
integer views, so their saved bit representation does not depend on float32
arithmetic or conversion. The emitted action device remains the execution
device; it must never be rewritten to `cpu` merely because evidence was
offloaded. A synthetic CPU run produced exactly the existing runner's full
family events and results. This is not evidence of numerical agreement between
CPU and CUDA, which may use different kernels. Comparisons and controls in a
future experiment must be within the same bound runtime.

`audit_family_events` accepts `expected_device='cpu'` by default and, for a
future bound owner, the explicit value `cuda:0`. It checks recorded metadata;
it does not attest actual device execution. A future GPU controller must bind
the expected value to the verified physical-device/runtime manifest and pass
it through the closed-journal audit. The present CPU journal wrapper retains
its original CPU default. The journal's selected logits, vector bytes, and
hash relationships can be audited, but unsaved complete mask-logit values
cannot be reconstructed from their digests.

The prototype requires trusted ownership and synchronization callbacks. It
keeps the live plan, input, mask, external source, exact plan-owned source
clones, logits and captures in preallocated custody slots through the relevant
synchronization boundary. It retains captures before plan exit can clear them
on an error. It checks ownership between transfer operations, synchronizes
while output and captures remain live, then disposes its device references and
requests a second synchronization. A failed check, transfer, hook cleanup, or
drain is not clean completion. A real owner must retain the custody object and
first exception graph, quarantine the worker, and terminate/reap the dedicated
process when cleanup cannot be verified. Python cannot promise to root a native
return interrupted before it is assigned or a custody update interrupted by a
new failure; those cases require process termination rather than a clean-drain
claim. Transient device allocations and asynchronous behavior have not been
measured by the synthetic CPU tests.

Before replacing the CUDA gate, publish and independently review an owner that
holds the common physical-device lease, verifies the pinned model/runtime and
resource identity, preclaims the complete family and host/serialization
lifetimes, checks native allocator/device headroom, synchronizes and records
the actual workload, and fails closed on cancellation, OOM, or uncertain
cleanup. Schedule and measure the entire 140-forward *development* family,
including overlapping donor, reference, capture, logit, transfer and evidence
buffers. The earlier four-forward calibration is useful development evidence,
but does not establish this family's maximum or justify an evaluation run.
Keep all failed/partial attempts and do not use evaluation outcomes to tune the
site, protocol, or resource policy. A later owner and its fixed development
calibration need their own source identities and independent reviews.
