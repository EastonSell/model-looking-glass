# Cooperative agreement split ordering gate

The natural and injection tracks retain separate frozen predictions, nonce
draws and held-out records. This small gate serializes only their allocation,
public-commitment and first-forward announcements; it does not reserve study
capacity, make a split private, authenticate same-account participants, or run
a model. A track can proceed independently when its peer is absent or has an
allocation in progress but no committed split or forward announcement. Those
states are explicitly **unchecked**, not evidence of disjointness.

Both owners must independently bind the same protected canonical gate root in
their reviewed private configurations. For this study the intended root is
`work/agreement-study-ledger-v5/cross-track-gate-v1`, within the protected
ledger parent. `acquire(root, expected_root=..., timeout=...)` rejects a
different directory and retains the root's filesystem identity, but cannot
prove that separately supplied expected roots came from the same trusted
configuration. Owners must check that identity and the exact published gate,
comparator and lease-source hashes before use. The fixed `DeviceLease` source
from 924983043cb2006206866de1593e7c5b036ed642, or a byte-identical descendant,
is required; the gate does not use CUDA or a GPU service.

Under the common OS lease, a custodian writes an exclusive allocation-start
marker before drawing its nonce. After the public commitment is independently
observed, it writes a commitment marker binding the track freeze, inventory,
domain commitment and exact public commitment-file digest. Before the first
model forward, the owner reacquires the lease and writes a durable forward
announcement binding that track's freeze, inventory, domain commitment and
exact commitment-file digest. That announcement is the ordering point and
conservatively marks the track as **possibly exposed** even if execution or
cleanup is later uncertain. Later decisions must match all four identities
against the committed marker. Each operation checks the lease and gate-root
identity after writing.
The caller must preserve a marker when an operation is interrupted or ambiguous;
an existing marker never authorizes another draw or another first-forward
announcement.

If the peer has no allocation marker, the announcement records
`other_not_allocated_unchecked`. If the peer has started allocation but has no
commitment or forward marker, it records `other_allocation_pending_unchecked`.
If the peer commitment exists, the owner must supply the exact, independently
verified `clear` receipt for both committed splits before its announcement.
The gate checks every receipt identity against the two commitment markers and
rechecks the markers after the caller's short, local receipt verifier returns.
The caller must verify the published comparator/source and exact commitment
bytes before calling. Perform network publication/readback before acquiring the
shared lease: the callbacks invoked while holding it must be bounded, local
checks against retained bytes/receipts. The helper cannot preempt an unbounded
callback. If the peer has a malformed marker, a forward without a
matching committed split, or an uncertain prior execution whose rows cannot be
verified and compared, the affected later evaluation fails closed. A completed
independent earlier result remains intact; a collision never permits redraw.

Markers are bounded canonical JSON with public digests only. The directory
accepts only the lock and six named markers, each at most 8 KiB and at most
64 KiB in total. The operator must include the gate, lock, database, sidecars
and reconciliation receipt in the reviewed ledger-associated allowance and
check the actual parent size before and after. A partial or unexpected file is
retained and makes the gate uncertain; it is not deleted to enable a retry.

The lease has an important process-lifecycle limit. If interruption occurs
after OS acquisition but before the returned lease is safely owned by the
caller, that process must terminate rather than retry or infer the lock was not
held. An interrupted or ambiguous release invalidates the lease; the process
must not continue or fork until it has terminated and cleanup is established.
Neither marker ordering nor this cooperative lock proves physical time,
confidentiality, or compliance by a nonparticipating process.
