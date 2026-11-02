# Experimental fair ingress owner binding

The CPU-tested fair adapter, bridge and broker handoff can share the exact
`OwnerCompletionAuthority` already bound to their broker. This is a trusted
in-process composition; it does not establish a ready GPU service or prove
that a supervisor can attest process quiescence. The existing unbound mode
remains cooperative and does not gain this authority guarantee.

The adapter rejects an authority different from its concrete broker's. An
authority-bound bridge must be constructed through the adapter's single
pre-rooted construction guard; the trusted owner can adopt the same completed
bridge if a return was interrupted. The bridge's server is created by its
guarded `make_server` path, and IPC verifies the exact bridge, server, target,
broker and authority graph before accepting work. Team credentials must still
be privately distributed and mapped to trusted principals. Round-robin
dispatch describes admitted pending jobs, not admission, GPU-time or byte-level
fairness.

The handoff pins the original broker, worker and authority, and the adapter and
bridge pin their ingress graph. Before their admission or retirement decisions,
they check those identities and mark the original authority unsafe and retain
custody if the graph has changed. These are cooperative integrity checks: the
owner must not concurrently mutate private fields, and a check cannot make such
mutation atomic with a later decision. Target, adapter and bridge status reads
share 256 preallocated graph-observation slots; each failure during an inspection
while its reserved slot remains locally attached retains its own private exception,
and exhaustion fails closed before inspecting the graph. The pool is an exact
owner-created native queue; a substituted producer is rejected without calling
it. An interruption before a slot is obtained or after it is detached for return
to the pool latches the owner but has only one fallback diagnostic root. Multiple
simultaneous, independently resource-bearing exceptions at those boundaries
remain outside this cooperative guarantee. The trusted owner must bound
concurrent status readers within that limit and retain any independently shared
exception resources until process replacement.

An indeterminate adapter transition or bridge custody failure marks the shared
authority unsafe at the fault source. Admission and receipt or custody
retirement check that authority at their decision points. An unsafe signal
that wins a decision prevents that transition and keeps its known claims and
roots; a decision that linearizes first can become visible after a later
signal. Cleanup uncertainty retains the relevant owner graph for process
replacement. The bounded host commitments cover only the documented
cooperative inputs and receipts, not all process memory, external aliases,
model intermediates or serialized client copies.

A nonblocking result poll reports an ordinary timeout only after its direct
handle and outcome aliases have been cleared. Once a terminal result is seen,
any exception from the underlying result observation, including a
timeout-shaped interruption, quarantines the bridge and retains its known
receipt and connection claims. Cancellation checks terminal state under the
broker condition instead of calling the blocking wait path. These rules do not
make arbitrary interpreter interruptions at every call/return boundary safe;
the trusted handler must retain custody and report uncertainty on failure.

The owner must retain the adapter, bridge, target, server and their guards and
must provide separately budgeted result custody before acknowledging a
receipt. It must also wait for exact target retirement and actual handler and
dispatcher exits before considering disposal. Those observations alone do not
prove every wrapper/bootstrap fault was reported, nor do they replace the
independent supervisor and device-lease checks required for a GPU service.
