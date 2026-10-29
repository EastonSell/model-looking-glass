# Two-team supervised service smoke

`run_two_team_smoke` is a bounded operator check for a prepared fixed
two-token service. It is disabled until the actual fair owner exports the
reviewed `two-team-fair-service-v1` compatibility marker. That marker is not
proof of a measured profile or of the source's integrity. The operator must
first obtain independently reviewed profiles for the exact worker path and
frozen source/runtime, and arrange the exclusive supervised GPU window.

The caller creates and retains a `prepare_two_token_service` handle configured
for two distinct teams, and keeps exclusive controller access during the smoke.
The smoke obtains their private clients from that
handle, submits `[15496, 995]` with seed 29 to both before waiting, and checks
team and job identities, `[1, 2, 50257]` float32 finite logits and matching
little-endian float32 hashes. It then closes its handles and requires the
supervisor to report a reaped, clean, zero-exit owner with no remaining unsafe
marker. Only a small result summary is returned; credentials, device UUID,
paths and logits are not included. Failures do not produce a success record.

The caller must keep the prepared handle if startup or shutdown is interrupted
and reconcile its supervisor and marker before another owner uses the device.
This check demonstrates two scoped requests through one owner; it does not
establish fairness, throughput, simultaneous GPU execution or a host-memory
bound. The fair bridge's host envelope and admission are separate obligations.
