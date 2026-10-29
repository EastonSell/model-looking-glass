# V4 CPU runtime and screen handoff

This clarification is fixed before the first v4 model forward. The v4 child
uses the pinned ViT weights and processor, CPU only, one intra-op and one
interop thread, deterministic algorithms, highest float32 matmul precision,
enabled MKLDNN and preserved subnormals. It additionally sets
`MKL_CBWR=COMPATIBLE` for both screening and intervention. Earlier v1/v3
records did not establish the historical value of that environment variable;
this new setting is not evidence of an identical historical runtime.
The v4 owner also checks binary64 round-to-nearest using arithmetic sentinels
before and after forwards and before the relevant scoring and dose decisions.

The independently audited 64-image manifest is checked before screening,
and the screen saves each 1,000-logit float32 vector before scoring. It
reconstructs all 64 recognition results and the deterministic selected pairs
from those bytes. An archive of the exact 64 full logit files and 64 forward
records, the start/terminal records, and the selected-pair result is published
alongside the selected-pair result, a path-free supervisor receipt projection,
and the separate observer's record of the controller's zero exit.
The entire public handoff must be read back
and independently observed before intervention. The owner compares each archive
member with the retained closed screen. Before a selected case can be
patched, the intervention owner replays the saved screen, validates every
selected baseline and matches all repeated clean vectors bit for bit against
the saved screen. It records the same model state, source files and CPU runtime
across phases. These checks rely on trusted quiescent execution and do not
independently prove chronology, execution, upstream image authenticity, visual
near-duplicate absence or pretraining independence.

An unsuccessful attempt is retained as incomplete. A successful supervisor
receipt checks the direct child exit, file identities and schedule; an
independent saved-byte scientific audit is still required before interpreting
intervention results. The one-shot screen and any later interventions have
separate evidence roots; neither is a latency benchmark.

An external trusted launcher creates a fresh private bytecode prefix and starts
the controller with `-I -S -B -X pycache_prefix=<prefix>`, adding only the
selected dependency and source paths before invoking the owner. The controller
itself rejects an unisolated startup before importing non-builtin modules. Its
child receives a separate freshly created prefix with the same isolation flags.
The controller receives the absolute site-packages directory explicitly and
passes it to the child; it does not infer a virtual-environment path from the
base interpreter's prefix under `-S`.
The same dedicated CPU custody root is used for both actual phases; the owner
creates an exclusive phase-specific attempt marker before launching either
child, and preflight consumes neither marker. Reusing a different root or
deleting a marker remains a trusted-operator responsibility, not a security
guarantee against another same-account process.
If controller launch, termination or reaping is unknown, preserve the attempt
and do not retry on the assumption that it was clean. A forced stop of the
direct controller does not establish that its worker or descendants stopped;
failure-receipt writing is best effort.
