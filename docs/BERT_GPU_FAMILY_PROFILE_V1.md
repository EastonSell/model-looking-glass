# DistilBERT full development-family GPU profile, v1

This is a proposed supervised **resource and control profile**, not a latency
benchmark, held-out experiment, or general memory bound. It repeats only the
eight publicly exposed development-family-1 rows from the pinned case manifest
(SHA-256 `20a8ce96bab80c1f3b19ca86ac43e5d32f62c4b4a9bbd3c23e1a3fc9abfd91e5`).
The four-forward GPU calibration is preserved separately and is not an
admission profile for this larger workload. Publish and independently review
the exact owner, launch source, and workload before coordinating the device
window.

Use the pinned offline `distilbert/distilbert-base-uncased` revision
`12040accade4e8a0f71eabdb258fecc2e7e948be` and its verified five-file
checkpoint/tokenizer snapshot. Tokenize and verify the eight family rows,
including mask ID 103 and output IDs 2003/2024, before any model forward. Use
one eager, evaluation-mode float32 masked model, one 1×10 int64 input with an
all-ones mask, six encoder blocks, and one visible physical GPU under its
common lease. Fix CPython 3.11.15, PyTorch 2.8.0+cu128 (CUDA build 12.8),
Transformers 4.57.3, tokenizers 0.22.2, safetensors 0.8.0 and NumPy 2.4.6. Use
seed 17 for the CPU and visible CUDA generators, one intra-op and one inter-op
thread, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disabled
matmul/cuDNN TF32, disabled cuDNN benchmark, and enabled cuDNN deterministic
mode. Require eager attention, evaluation mode and disabled CPU/device
autocast at each observation. Record runtime readbacks and the actual
physical-device identity check. Identity and setup checks are not evidence of
a population claim.

Run exactly the existing family order: five sites each receive eight donor
captures followed by four targets × four same/different-lemma and
same/opposite-number source actions. Then run two final-block delivery
controls per target, each with a donor capture and a patch, followed by a
fresh clean target observation. This totals **140 root forwards**: 48 donor
captures, 88 patches, and four final clean checks. The mask is at token 6;
subject and distractor sites, output tuple selector, invocation zero and
action order match the fixed CPU study. Each returned full mask-position
vocabulary and selected before/after activation is CPU-owned and preserves
float32 bits; each event retains its actual `cuda:0` action metadata.
Within this runtime, require the fixed input, donor landing, self, final-mask,
final-subject and clean-repeat controls. Reconstruct the complete closed event
journal using an independently bound `cuda:0` expectation and compare its
case effects with the owner's live family summary. Numerical agreement with
CPU outputs is not a prerequisite and is not implied by this profile.
The live owner checks full mask-position vocabulary controls, but the event
journal retains only each full-vector digest and the two selected logit bits.
A post-close reader can compare those recorded digests and selected values;
it cannot reconstruct the unsaved full vectors or independently repeat the
owner's bytewise comparison from this evidence alone.

Before CUDA-capable imports, the supervisor establishes child/marker
coordination and the direct child acquires the common physical-device lease.
After verifying UUID/device count and native allocator,
set the per-process allocator cap to 25% of the reported device total. Hold a
whole-owner GPU commitment of the allocator cap plus 256 MiB, alongside a
separate logical device-trace claim and a **64 MiB logical host-evidence
claim**, from before model/input construction until the journal, candidate,
candidate serialization buffers and device references are disposed. The small
final cleanup record is separately capped by its writer and is outside the
released host-evidence claim. Require matching
device totals, at least 3 GiB free initially, the existing `max(1 GiB, 15% of
total)` reserve and 256 MiB allocator headroom at every synchronized
checkpoint. These are conservative cooperative checks and cannot prevent OOM
or outside use between checks. CUDA context/driver allocations precede the
software commitment under the physical lease; the allocator cap excludes those
and other non-PyTorch use.

The 64 MiB claim is a logical evidence allowance, **not a total host-memory or
RSS cap**. The family loop holds four mask-vocabulary references (488,352
logical bytes), up to eight 3,072-byte donor vectors, prior observation objects,
and final-control source/before byte dictionaries; temporary current and prior
mask outputs, integer-view copies, base64/JSON records, and the later journal
read/audit can overlap. The owner limits the complete journal to 4 MiB on disk
and individual records to 12 KiB. The journal reader checks the record count
before splitting and bounds its raw, split, parsed-string and per-record
decoded/re-encoded logical byte payloads; the host claim remains held through
candidate serialization. Python object overhead, checkpoint and
model storage, native workspaces and interpreter overhead remain outside a
physical-memory guarantee. Parent-side validation after the child is reaped is
also outside the child's host claim. Measure or explicitly report available host
observations without calling this claim an enforceable process-RAM cap.

Reset CUDA peak counters once before model loading. Record synchronized memory
before and after model transfer, before and after every forward, and during
final cleanup. The post-forward checkpoint follows the observer's synchronized
disposal of that forward's device objects; earlier family state can still be
live. Peaks are cumulative since the one reset and include transient allocator
use, not individual per-forward maxima. Hold the live output, plan-owned source clones and captures through
their synchronization, dispose them only after successful drain, then
synchronize again. On a failure, cancellation, OOM, lost ownership or
unverified drain, preserve the bounded journal and the first exception/custody
roots, do not attest clean, and terminate/reap the owner under the supervisor's
quarantine marker. Any native return interrupted before it can be rooted is an
unverified process-termination case. Do not retry a failed drain.
The owner rechecks its supervised session and the exact held commitment/claims
before each observation device action and checkpoint, before candidate
publication, and between individual releases. Obtained memory observations
are recorded before policy validation, including a failing observation. These are
sequential checks, not an atomic guarantee against outside changes; if a later
check fails, retain remaining claims and quarantine the process.
The private observer accepts trusted callbacks; the source-bound owner and
supervisor establish the physical lease and process-local claims. This is a
cooperative trust boundary, not access control against arbitrary code in the
same process.

The model checkpoint and source/launcher bytes must be checked and recorded,
as must model state and registered buffers before/after the family, hook
cleanup, event ordering, owner exit and final CUDA allocator cleanup. Record
every started/completed or explicit failed event and preserve partial records;
a failure may prevent later planned forwards. On complete success, publish the journal, memory
observations and control outcomes with the source/runtime identity and clearly
scope the result to this one public development family and allocator history.
Create-only publication can be interrupted after its final link is created;
accept the controller terminal only with the corresponding successful
controller exit or acknowledgement. Readback does not prove power-loss
durability.
A later evaluation remains a separate frozen, source-bound run; this profile
does not establish an explanation, enable evaluation, or support a throughput
or latency claim.
