# Supervised service control building blocks

`looking_glass.gpu_service_control` provides a **private control-plane seam**;
it is not an enabled GPU model service or a resource profile. The caller must
construct a fixed, independently reviewed owner command with a canonical
physical `DeviceIdentity`, exact measured workload profiles, and a sanitized
replacement environment. Clients cannot supply commands or backend callbacks.
The CPU tests also exercise the private seam with an identity-less supervisor;
that mode is not a GPU launch path.

The caller prepares and retains a controller before starting the supervised
owner. This matters if startup returns but the caller is interrupted before the
call completes: the already-held handle can still request shutdown or inspect
the supervisor result. An interrupted or failed startup is not evidence that
no child was created. A locked launch gate revokes a thread that has not yet
entered the supervisor; once it has entered, the supervisor remains rooted
until it reports an outcome. If the outcome remains uncertain, the device
marker is authoritative and no caller should reuse the device blindly.

After the IPC server is ready, the owner sends one bounded JSON frame through
the inherited private startup pipe. It contains a loopback address, a broker
lifetime ID, the pinned model identity, and distinct random bearer tokens for
the configured teams. The controller checks the exact schema and team set and
does not put the frame or credentials into logs, environment variables,
command arguments, or a persistent file. Recipients receive only their own
team-scoped client from the trusted controller. Python and OS copies of the
credentials cannot be securely erased by this interface.

Shutdown is signalled with a nonsecret, single-use marker in a caller-owned
private directory. The owner monitors its directory identity, records any
monitor failure separately from best-effort diagnostics, and reports that
failure even if device teardown succeeds. The controller waits for the actual
supervisor result: a workload failure, an unverified teardown, a timeout, and a
clean teardown remain distinct. On timeout or rejected readiness, it aborts the
startup channel, requests termination through the supervisor, and retains the
handle while the child state is uncertain. The supervisor clears the unsafe
marker only after a valid clean attestation, exact child exit and device-lease
reacquisition. A failed or absent result is not a clean teardown.

The controller discards its own credentials on shutdown or supervisor exit and
will not issue new clients afterward. Previously issued `IPCClient` instances
can retain their tokens; callers must discard them when the owner stops. The
current IPC handshake does not authenticate a server before a client sends its
bearer token, so this is a cooperative local-process interface, not protection
against a hostile same-user process that takes over a released port. A future
challenge/response wire protocol requires separate review.

The accompanying CPU tests use a real supervisor and a tiny CPU model to check
one owner, concurrent submission by two teams, cross-team rejection, shutdown,
startup cancellation and sticky-marker behavior. They do not measure GPU
memory, establish CUDA cleanup, authorize a profile, or prove that model
forwards run concurrently.
