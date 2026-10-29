# Known-injection number-study primitives

`looking_glass.pretrained_number_injection` provides CPU-only single-forward
instrumentation for the known-injection track of
[the number-study protocol](PRETRAINED_NUMBER_PATCH_V1.md), alongside the fixed
direction and numerical checks. It is a building block, not a complete study
runner: it does not verify the pinned checkpoint or source, commit a hidden
site, create a held-out split, freeze predictions, persist evidence, compare
all controls, or authorize discovery or GPU execution. The higher-level
custodian must perform those steps against the final reviewed protocol bytes.

The reference addition rounds finite binary32 operands to nearest, ties to
even, with explicit signed-zero, subnormal and overflow handling. It does not
use a host float32 cast to decode subnormal operands. The selected runtime
sentinels check subnormal inputs and outputs and halfway rounding; they are
not a proof of all arithmetic behavior. The caller must run them on the actual
CPU execution thread and reject mismatches before an injection. Injection
verification consumes immutable little-endian snapshots of the whole selected
vector, rejects nonfinite or incorrect results and requires a changed element.
The caller is responsible for preserving those snapshots privately and
checking site, invocation, action order and cleanup.

CPU unit tests compare 4,096 finite operand pairs with the local PyTorch
runtime, cover fixed edge cases, and check that the sentinel rejects an enabled
flush-to-zero mode when the runtime supports changing it. An independent
reviewer also checked 274,480 operand pairs on this machine; that operator
observation is not a cross-platform certification.

`NumberInjectionExecutor` holds a cooperative model-tree lease across its
CPU forwards and requires evaluation mode, eager attention, no cache, no
existing hooks, one intra-op thread and verified addition sentinels before and
after each forward. A caller supplies a fresh opaque request ID and a private
site. The site alone is hidden: the direction and nominal 0.5 strength are
public in the protocol. The executor freezes input IDs, binds each same-input
donor to an earlier capture from the same executor, and checks the actual
module, invocation, batch/token, tuple selector, shape, dtype and device.
Selector-facing traces contain only the 13 boundary captures in execution
order, with contiguous public capture-order indices; private action metadata
retains the original plan ordinals, which could reveal the site.
The separate public `action` descriptor records only the requested patch or
null operation's actual coordinates and, for patches, the typed donor identity.
Clean and injected baseline observations have no descriptor. The selector must
check the descriptor against the intended condition, candidate layer and source
request; the descriptor does not contain the hidden injection or plan ordinals.

Each add, patch, null or self action has private full-output captures
immediately before and after. The executor checks every unselected token's raw
bytes, exact selected-vector landing, adjacent same-site witnesses, and the
public post-action capture. A patch donor is an immutable prior observation;
its source request and boundary metadata are retained privately. For an add,
`realized_delta_f64_le` stores 768 little-endian binary64 values formed by
`math.fsum` from decoded pre/post binary32 values. The original raw witnesses
are also retained so an auditor can recompute the delta independently.

The 1 MiB forward admission calculation includes every planned capture,
last-token logits, external action sources including the fixed direction, and
the binary64 realized-delta record. The 512 MiB study ledger counts unique
retained capture, logit, donor and realized-delta byte objects. Metadata,
Python object overhead, transient full-sequence logits, plan source clones,
and other working memory are not covered by the retained-payload ledger; it
is not a host-memory or GPU reservation. A validation, overflow, cancellation
or forward failure poisons the executor. Completed observations and any
available post-plan partial captures remain in memory for the custodian;
uncommitted captures discarded by the plan are recorded as unavailable.
Uncertain or failed ownership is quarantined until the dedicated process is
discarded. If in-memory failure recording itself fails, its original and
secondary exceptions are chained where possible; a supervisor must mark the
attempt incomplete when no durable terminal record can be written.

This in-process API is procedural custody, not isolation from code in the same
process. Private evidence, action metadata, exceptions and exception chains
must stay with the custodian. Publish only freshly constructed sanitized status
records. A successful primitive call does not by itself establish clean-repeat,
upstream, downstream, null, self, model-drift or held-out controls; the future
study coordinator must enforce and save them.
