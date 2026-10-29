# Local CPU injection-discovery composition

`looking_glass.pretrained_number_supervisor.run_discovery` connects the
family-final 128-request schedule, `NumberInjectionExecutor`, private pair
journal, receipt sidecar, public observation archive and selector. It uses
the eight already-public discovery inputs; it does not draw a site or a
held-out split, publish anything, load a checkpoint, or establish a study
result. A `complete-local` result means the local checks and archive finished;
it is not proof of publication, source/runtime identity, independent entropy,
execution on the pinned model, or secrecy from same-account code.

An external custodian must first publish and independently verify the exact
reviewed source, protocol, runtime and discovery inputs. It then publishes
the v2 request plan from independently drawn public entropy, draws and
privately retains one separate site nonce and record, and publishes the site
commitment before the first injection-track forward. The real caller verifies
the pinned model/tokenizer assets, source, environment, CPU-only execution,
and model before calling this component. The same source and runtime must be
used throughout this track. Local clocks and hashes document operator
observations; they do not prove chronology or faithful execution.

The attempt directory must be a fresh absolute path under a trusted parent
with appropriate Windows ACLs. Receipts and all files in `private/` may reveal
the hidden site and must stay private until protocol reveal. Each completed
pair is staged, semantically audited, and then its exact receipt is appended
and fsynced before another model forward starts. After the executor exits,
the sidecar is reread and the journal independently reaudits the complete
acknowledged prefix. A receipt marker by itself is not proof of a returned
receipt or of stable-media durability.

A journal inspection reporting `complete-prefix` is not sufficient to publish:
a terminal write can itself fail after all receipts were saved. The custodian
must independently confirm successful child exit, the launcher's postcheck and
success record, and the expected complete artifacts. Missing or ambiguous
terminal evidence remains an incomplete attempt.

Available completed observations, private action witnesses and executor
failure captures are serialized into the private incomplete-forward format
if a forward or serialization fails. The serializer reports any omissions;
captures already discarded before exposure are unavailable. If serialization
or writing fails, status records that the partial evidence is unavailable
when it can. If the journal advanced before its separate receipt failed, the
current evidence is saved as `private/incomplete-forward.bin` instead of being
mislabelled as the next journal slot; inspection still treats the pair as
unacknowledged. Every prior frame remains; failed or uncertain attempts are not
resumed or converted into successes. The caller must retain a sanitized
failure status even if creating the attempt or writing its terminal file fails.
Exception objects, tracebacks, paths, timing and private hashes remain with
the custodian.

The phase takes a caller-reserved share of the protocol's 512 MiB whole-study
logical tensor cap and a separate share of the 4 GiB physical evidence
allowance in `experiments/agreement_physical_budget_v2.json`. It checks free disk for its physical share and reserves at least
1 MiB of logical capacity before each forward. The executor counts unique
retained byte objects; its public and private serialized representations of
those same tensors are not charged again to the logical cap. The journal's
encoded bytes, a conservative full public-archive copy, the receipt sidecar,
and bounded final/failure allowance count against the physical share. A
custodian must allocate and reconcile all phases, earlier failures and
orphaned records against the global caps; a fresh local process is not a
global ledger. Python serialization copies, process RSS, model parameters,
transient outputs and filesystem overhead require separate headroom.

Only after successful executor cleanup, per-family final controls, model and
buffer hash equality, all 128 receipt checks and complete archive reanalysis
are the local public files written. The separate custodian publishes and
verifies their exact bytes, the runtime manifest, prediction and aggregate
freeze before drawing any held-out nonce or constructing held-out inputs.
