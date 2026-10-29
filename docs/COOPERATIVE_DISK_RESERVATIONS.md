# Cooperative disk reservations

`looking_glass.disk_reservation` coordinates estimated disk use among trusted
local study owners. It is optional: existing study-specific disk checks remain
independent. A reservation is accounting, not an allocation of filesystem
blocks, a quota, or a guarantee against another process filling the volume.

All participating owners for one volume must use the **same trusted ledger
directory** and the same fixed `cushion_bytes`. The directory and every supplied
target directory must exist on that local volume. On Windows the ledger binds a
volume GUID and serial; on POSIX it binds the device number. A different volume,
changed ledger/owner-directory identity, mismatched cushion, unsupported volume
identity, unavailable free-space reading, database failure, or unknown owner
fails closed. The owner-lock directory and database must be ordinary paths, not
links or Windows reparse points; the owner's resolved volume is rechecked. Use a
local filesystem with working SQLite and advisory file locks;
do not treat separate ledger directories or nonparticipating writers as
coordinated. The implementation does not defend against another program replacing
the database or deliberately evading the protocol. SQLite uses a full-sync
rollback journal, while probe markers rely on ordinary filesystem operations;
storage failure, power loss or filesystem corruption are outside this protocol's
durability guarantee and require separate reconciliation.
On POSIX, the at-fork hook fences owner-lock probes and closes inherited claim
descriptors in the child. Applications must quiesce all ledger operations before
forking and must not call `fork()` from a signal handler or reentrant callback.
The current independent runtime checks were performed on Windows; POSIX `flock`
and fork behavior need verification on the intended filesystem before use.

One claim covers bytes actually written to its one volume; its output and staging
amounts cannot cover writes elsewhere. If output and staging use different
volumes, obtain and check a separate claim against each destination before
starting any writer. Coordinate the ledgers in a consistent order and retain any
uncertain partial claims for reconciliation. This API cannot discover paths the
caller omits and does not provide an atomic multi-volume reservation.

```python
from looking_glass.disk_reservation import DiskLedger

ledger = DiskLedger(shared_dir, target_dir=output_dir, cushion_bytes=2 * 1024**3)
lease = ledger.reserve(output_bytes=500 * 1024**2, staging_bytes=400 * 1024**2,
                       target_dir=output_dir, staging_dir=staging_dir)
# Bound output and staging writes independently to the declared estimates.
# Before another phase, observe the actual target volume again:
lease.check()
run_study()
stop_and_reap_all_writers()
lease.release()
```

Release only after cleanup has actually stopped **every** writer, including
children, and the result is not uncertain. If that cannot be
established, keep the claim and owner process alive for reconciliation instead of
calling `release()`. Starting `release()` permanently revokes further use of the
lease, even if its return is interrupted. A failed release can be retried only
when the same process still holds its verified lock; otherwise treat it as
uncertain. No context manager silently releases on an exception.

Admission takes a SQLite write transaction, checks the owner locks, and reads
current available bytes on the actual target volume before committing the claim.
The threshold is:

```
observed free >= all active output + staging claims + new claim + cushion
```

The full declared amount remains charged until release, even as files consume
real free space. This can double-count written bytes and reject an otherwise
possible run; it avoids pretending to measure remaining output. `lease.check()`
rechecks both directories and repeats the conservative check for all active claims.
If either directory changes, supply and check the new directory before writing.
These are point
observations: another writer can consume space immediately afterward. Callers
must enforce their own output/staging caps, account for physical copies and
temporary files, and handle write failures. Ledger metadata and lock-file use
also consume real space; include adequate room for them in the cushion.

`ledger.snapshot()` reports each active claim as `held`, `stale`, or `unknown`,
plus observed free space and the full active total. `held` means the expected
owner lock is currently held; it is not proof that a process tree is healthy.
A stale or unknown claim remains charged and blocks new admissions and checks.
No claim is reclaimed by PID, age, or process death. The claim history and owner
lock files remain in the ledger directory; this version permits at most 1,024
active claims in one ledger. History and lock files grow with use, so plan for
their disk and scan overhead; any maintenance requires a trusted, quiescent
ledger. An interrupted owner-lock probe leaves a marker and
the claim becomes `unknown`; this prevents a leaked probe lock from looking like
the original owner. Normal probes remove their markers after closing the probe.

After independently establishing that all associated writers, including any
descendants, have stopped and that their output is settled, a trusted operator
may call `ledger.reconcile_stale(claim_id, reason="...")`. The method only
accepts an active claim whose original lock file is identifiable and currently
unlocked; it cannot verify the operator's assertion about descendants. A missing,
replaced or otherwise unknown lock cannot be reconciled by this method. Do not
remove a ledger, lock file or probe marker to clear a claim. An existing probe
marker can be reconciled only after the original lock is obtainable and the
operator has also verified any probing process and associated writers stopped.
Interrupted or ambiguous
reservation commits retain a process-rooted handle where possible; a lost return
may require process exit and subsequent explicit reconciliation.
