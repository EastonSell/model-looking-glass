# Development-only injected-change custodian

`looking_glass/pretrained_custodian.py` assembles and checks the ordered
interventions used by the v1-shaped analysis on a trusted, exclusively owned
CPU model. It is a development helper, **not an approved recorded-study runner**.
The caller must keep that model exclusive for the entire 28-forward case.
`ActivationPlan` owns it only for one forward at a time, and locks on two
separate custodians do not prevent those custodians from interleaving cases.
Its tests use a small untrained arithmetic model and artificial token IDs. They
do not load DistilGPT2, choose a real hidden site or split, run a discovery
study, or touch reserved held-out inputs. Successful synthetic controls do not
identify a natural model computation or establish pretrained-model behavior.

The helper requires a v1-shaped six-block, 768-wide, 50,257-token-vocabulary
model in eval mode with float32 tensors, eager attention, no cache, no existing
module hooks, CPU inputs and outputs, and one invocation of each target per
forward. It checks configuration, tensor identity and version, registered
module identities, local hooks, and PyTorch's global forward, forward-pre,
backward and backward-pre hook registries before and after each forward. It
requires a **trusted CPU-only forward implementation**: examining parameters,
buffers and outputs cannot prevent arbitrary model code from launching CUDA or
performing unrelated work. It does not load or verify a checkpoint, tokenizer,
runtime, actual source checkout, GPU admission, or exclusive broker lease.
The caller must keep its input tensor quiescent while the helper copies it.

For each synthetic case, the helper runs 28 sequential forwards: clean and
repeat, injected and its fresh clean check, then null, same-run self-patch,
clean upstream patch, clean late-block patch, four clean-to-injected candidate
patches, four injected-to-clean reverse patches, each followed by a fresh
clean check. It captures six last-token MLP outputs and the tuple-first final
block output. The fixed hidden-site order is private pre-add capture, add,
private post-add capture, public observation, then a same-site patch. An
earlier candidate patch precedes the later injection in actual model order.
Each action also has a private post-action capture compared bit-for-bit with
its prescribed source, including signed zero. For self-patch, the source is
also bit-compared with the immediately preceding public same-site observation;
its private source hash therefore identifies that saved public vector. The
ordinary clean, null, self, upstream and late-output comparisons use the
analysis tolerance on all recorded vectors and last-token logits. Full
last-token vocabulary logits are retained for every public condition.

`private_evidence()` is custodian-only. Its copied records retain immutable
float32 byte snapshots and hashes, input hash and shape, case and forward
identities, selected site, ordered plan metadata and patch-source identities.
Public observations omit those plan operations, sources and private vectors.
Evidence is appended before later metadata and integrity checks, with false
verification flags until checks pass. If a capture, metadata validation,
allocation or storage operation fails, a private record can be partial and
`private_failures()` is only a bounded, best-effort diagnostic. Its
`captures_available` flag says the plan exposed captures to this helper, not
that every capture was retained. Busy/quarantined rejections before the
protected attempt are not recorded there. A separate trusted registry must
retain all attempt outcomes and private evidence; this helper has no durable
custody or publication mechanism.

The helper charges fixed, site-independent tensor payload counts before each
forward: seven public vectors and logits, private injection pre/post/realized
vectors when injected, plus the nominal injection source, patch or self source,
and post-action snapshot where applicable. The caps are 1 MiB per forward and
at most 256 MiB for a helper instance; private plan metadata additionally has
64 KiB per forward and 16 MiB total limits. These logical payload charges are
neither a cap on peak process memory, temporary clones, JSON/base64 buffers,
model memory, allocator usage, nor VRAM. Caller-owned public cases and copied
objects may persist after execution. The trusted caller must separately budget
all copies and model execution.

An optional `threading.Event` is checked before and after each forward and
before returning the case; it cannot interrupt a forward already running. An
event set immediately after the last check can race a successful return, so a
future broker needs its own atomic cancellation/publication decision. Any
caught failure or cancellation quarantines the helper and produces a sanitized
public exception; private diagnostics can contain site-bearing details. A
`KeyboardInterrupt` within a forward or a cleanup failure may leave hooks and
the model lease retained. Discard the entire model, helper and owning process
unless an independently proven recovery path exists. An arbitrary asynchronous
interrupt at lock acquisition or release can likewise strand ownership; this
helper does not claim recovery from every Python interruption boundary.

The helper and its evidence are in the same OS account and Python process as
other code. Hook checks detect persistent cooperative changes; they cannot
exclude transient hooks, races, arbitrary introspection, changes to Python
objects, timing leakage or malicious code. They do not provide security
isolation or proven blinding. Do not pass its private records to an
investigator before reveal.

Before any recorded v1 run, the project still needs an independently reviewed,
separately bound **pre-seal clarification** specifying the bit-exact float32
injection and post-action checks, their signed-zero rule, the extra snapshot
accounting, and the hook-free trusted-process requirement. That clarification
must be bound to the public manifest before discovery; it must not rewrite
prior evidence. A real runner must additionally bind the pinned assets and
source/runtime, use a reviewed exclusive shared-model lease and durable private
custody, gate held-out execution on frozen predictions, retain failed attempts,
and satisfy the remaining requirements of `PRETRAINED_CAUSAL_PROTOCOL_V1.md`.
