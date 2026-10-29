# Supervised five-token calibration controller draft

This controller is enabled for the reviewed, supervised five-token resource
pilot. It does not approve a profile or establish a shared service. A device
window and the frozen owner, launcher, source and identity remain specific to
the run; a profile requires separate evidence review after cleanup. Each
controller process permits one attempt, including a failed preflight, and its
owner process is single-use as well.

The isolated bootstrap requires `-I -S -B` and a fresh empty bytecode prefix.
It verifies the complete package Python inventory, four protocol documents and
the bootstrap itself against a clean Git commit. Project modules are compiled
from the captured committed bytes, not imported from a Python path inherited
from the environment. The parent captures them using sanitized Git reads with
replacement objects disabled; after the launch grant, the child checks the
protected checkout against the parent's hashed inventory without running Git.
It appends one explicit site-packages
directory after the standard library and checks the top-level origins of
Torch, Transformers, Hugging Face Hub, NumPy, Safetensors and Tokenizers before
their import. It rejects other meta-path finders, and the owner repeats that
check immediately before importing Torch. These checks rely on a trusted,
otherwise idle interpreter rather than isolating against hostile code already
running in the process. The controller also compares the owner's live source inventory
to that frozen commit and checks the checkout again after the child exits. On
Windows, the parent holds read-only share protections on every frozen package,
protocol and bootstrap file and on all seven lexical checkpoint assets through
the child outcome. This pilot rejects symlinked or reparse-point asset paths;
an uncertain child outcome retains the guards until controller-process exit.
These checks assume a trusted, stable local filesystem and dependency/native
environment; they do not authenticate all native libraries or prevent a
privileged actor from changing and restoring other files during a run.

The parent inventories the physical device and starts one direct child through
the common `run_supervised` marker and physical-UUID lease. It pre-roots one
single-run Windows Job and passes it to the supervisor, which assigns the
child before granting its start token. The child waits for supervisor
authorization before source capture or package imports, acquires and
checks that exact lease before importing CUDA-capable packages, and retains the
lease until process exit. It runs the
single trusted owner callback only after that check. The callback retains its
model, memory commitments and private error on uncertain work; neither the
child nor its parent retries CUDA or attests a failed callback. A clean result
requires the callback's explicit runtime-identity, synchronization, model
weakref, PyTorch allocator, cuBLAS-cache and claim observations, matching
private evidence hashes, and a matching supervisor attestation and reaped
direct-child identity. When the Job is supplied, supervisor cleanliness also
requires its complete observable lifetime inventory, confirmed drain, and
handle close and a zero owner exit before removing the marker. An abnormal or
unknown exit retains the marker even if a clean attestation was written earlier.
Uncertain or failed cleanup keeps the
Job and source/asset protections rooted until process replacement. These are
cooperative observations, not proof that all CUDA context or native allocations
have disappeared. The Job cannot certify escaped, breakaway, or unrelated
processes, and external applications remain outside the device lease. A
successful post-exit
device-process query must explicitly show that the owner PID is absent before
the controller offers a candidate for audit; that query does not prove that a
process tree was fully reaped. The private controller evidence preserves the
queried UUID with each before/after device inventory so this comparison can be
reconstructed. Any public summary must omit the physical UUID.
Before device inventory or acquisition, the controller requires the checkout
to have exactly one case-insensitive `work` ancestor, spelled `work`, and
requires the supplied device workspace to be its parent. The child repeats
that check against its supervised identity. A nested or differently cased
workspace is rejected rather than selecting another lock namespace.
Before launch, the controller durably freezes the exact source commit and hash,
source-preflight digest, pinned asset identities, runtime versions, device
inventory and UUID, seed and fixed public target/donor coordinates. The child
verifies those saved bytes before device ownership and compares actual runtime
and assets to the freeze before the model workload begins.

A fresh private run directory is checked as a canonical ordinary non-reparse
directory outside source and cache. Its filesystem identity is passed to the
child and checked again after callback and reaping. The checks detect observed
replacement; they are not a security boundary against a privileged actor that
can replace and restore filesystem objects between observations.

The owner writes 84 ordered private stage records and provisional cold/warm
evidence. The controller verifies the run, PID, supervisor identity, source,
cleanup and presence of those records, then reports `awaiting_independent_audit`.
The supervision sidecar records `calibration_run_id` separately from the nested
supervisor's own `run_id`.
That status is deliberately **not** an approved profile or a successful causal
study. A separate audit must recompute stage order and metadata, exact case and
runtime provenance, headroom and both profile estimates before promotion.
The controller never publishes raw behavioral logits or donor activations.
