# Fair IPC service lifecycle (development seam)

The private `_FairOwnerLifecycle` adapts an already constructed worker, actual
broker, fair bridge and its paired IPC server to the existing owner watchdog.
The caller must retain this graph, including partially constructed components,
until the supervised owner process exits. It must bind `owner_quarantined` to
the actual process-wide sticky GPU-owner latch. A missing callback is rejected.
This seam neither constructs a service nor authorizes a GPU run.

The watchdog independently observes the worker, broker, bridge, IPC custody
and owner quarantine signals. The broker's nonblocking `health_snapshot` is
observational, not an admission grant, and can be non-atomic. The bridge's
`dispatcher_alive` observation detects an exited thread; a live thread does not
prove progress. A separate supervisor must bound a blocked handler, validator,
broker condition or shutdown thread and reap the owner.

Health reads use four preallocated, nonblocking probe slots with separate error
roots, so normal overlap does not serialize the watchdog. Exhausting all four
slots causes a conservative quarantine; a stalled probe still requires the
external supervisor to bound shutdown and replace the process.
Server control operations are serialized and check quarantine before starting.
Those checks are not atomic with a concurrent owner-latch change: a cleanup
call can begin after the latch flips. Likewise, the existing worker can release
a job lease and publish a result after an independently latched owner fault.
The fair GPU service must remain disabled until those admission, completion and
retirement transitions share a reviewed ownership/quarantine contract.
Separate preallocated error slots cover the bounded in-flight lifecycle and
health categories; repeated arbitrary interruption still requires process
replacement rather than a retention claim.

Startup starts the bridge before the IPC listener, checks dispatcher liveness
and quarantine after bridge startup and after listener startup, and announces
only afterward. These checks do not atomically order an asynchronous owner
fault against the following startup call; the future shared contract must do so.
On an ordinary shutdown, the server must close and retire its handlers before
the bridge can close, and the bridge must stop and retire its claims before the
actual broker can close. Only then can the enclosing GPU owner check job leases,
synchronize and consider a clean attestation. A failed, timed-out, interrupted
or unverified step must retain the graph and leave the durable device marker for
supervisor recovery. A workload failure and a verified device drain are separate
facts. Process termination remains necessary for unknown CUDA internals and
arbitrary Python interruption windows.

The intended initial fixed two-token service uses at most two connections,
4 KiB request and codec limits, a 64-byte decoded-input limit and a 4 MiB
response limit. The bridge's current conservative accounting requires
1,075,838,976 declared bytes per connection. Together with two handoff
preparation claims, two 512 KiB receipt claims and an 8 KiB waiting queue,
the separate cooperative ledgers total 2,155,233,408 bytes. These commitments
are not allocated RAM, a measured peak or a physical memory guarantee; model
staging, the Python runtime and other processes need additional headroom.
The GPU load/cold/warm profile does not measure the fair bridge's host-memory
envelope. Recheck resources and independently validate the composed service
before any GPU launch.

This is a CPU-tested lifecycle component, not yet a wired or measured fair GPU
service. The bridge provides per-team scheduling only after global pre-auth
connection admission; blocking validators and other shared critical sections
can still delay peers. It does not imply concurrent GPU kernels.
