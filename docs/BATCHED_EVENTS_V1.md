# Observed events for a batched load window

`looking_glass.batched_events.assemble_window` is a CPU-only, offline adapter
from an explicit event set to the window schema in
[`BATCHED_THROUGHPUT_V1.md`](BATCHED_THROUGHPUT_V1.md). It is not a recorder,
broker hook, runner, device supervisor or evidence that a GPU forward happened.
An experimental grouped-broker path can supply some observations; see
[GROUPED_OBSERVATION.md](GROUPED_OBSERVATION.md). A trusted driver must capture
the remaining events, retain the original raw observations, and establish
model/profile identity, event completeness and device safety.

Pass the frozen protocol and independently retained protocol digest, a context
and a list of ordinary JSON event dictionaries. The context contains `mode`
(`serial` or `grouped`), `window_id`, `owner_session_id`, `model_sha256`,
`runtime_sha256`, the mode's `policy_sha256`, `clock` (`perf_counter_ns`),
positive integer `clock_resolution_ns`, and `events_complete` (exactly true).
The latter is a producer attestation, not proof that no event was lost. The
adapter snapshots bounded metadata, accepts at most 4,096 events and one MiB
of JSON, and rejects duplicates, missing observations and inconsistent event
times. JSON serialization occurs before the byte-length check, so the limit
bounds accepted metadata rather than peak memory from an oversized input; the
trusted producer must independently bound buffering and serialization memory.
Input records must remain exclusively owned during the snapshot.

All `time_ns` values are captured at the actual event source in the same clock
domain, not interpolated from queue delay or client response. Events may be
provided out of list order. The caller must retain the original event list:
the derived window omits owner-drain time, broker-terminal times, the time of
an uncertain-call observation and the separately observed team IDs. Rechecking
their chronology and attribution requires the raw events and this adapter;
the derived window alone is insufficient.
Every event has `kind` and `time_ns`, plus exactly these fields:

| Kind | Additional fields | Required source |
| --- | --- | --- |
| `window_start` | none | Common barrier release after clients are ready. |
| `window_end` | none | End of timed terminal handling and output work. |
| `owner_drain` | `drained` | Independently observed owner drain state; not inferred from a client timeout. |
| `attempt` | `job_id`, `team_id`, `input_sha256`, `request_sha256` | Caller attempt and actual frozen-request identities. |
| `ready` | `job_id`, `team_id`, `broker_id`, `broker_job_id` | Actual validated snapshot ready for dispatch. |
| `selected` | `job_id` | Actual scheduler selection of this job. |
| `broker_terminal` | `job_id`, `state` | Actual broker outcome: `succeeded`, `failed`, or `cancelled`. |
| `returned` | `job_id`, `state`, `error`, `output_sha256`, `cancellation_requested`, `cancellation_requested_ns`, `cancellation_accepted`, `cancellation_decision_ns` | Client-visible terminal handling or explicitly unresolved return; success fingerprint covers the split tensor itself. |
| `forward_entry` | `call_id`, `group_id`, `profile_sha256`, `input_shape`, `members` | Actual model entry; shape and ordered membership must come from the launched input. |
| `forward_finished` | `call_id`, `state`, `synchronized`, `error` | Actual completed model call and its synchronization outcome. |
| `forward_uncertain` | `call_id`, `error` | Explicitly observed uncertain call, not an invented finish timestamp. |

Each `members` item is an exact dictionary with `job_id`, `team_id`,
`broker_id`, `broker_job_id`, and `row_offset`. The observed attempt, ready and
member team IDs must match the frozen job attribution. Broker job IDs retain
their actual bounded string or positive integer type. A forward entry needs exactly one finished
or uncertain observation; uncertain calls retain `finish_ns = null` in the
output. Every observed attempt needs one client-return observation. A delivered
broker outcome needs its own terminal observation; an unresolved client may
have a later known broker outcome or none. A selected job that never reached
model entry does not acquire a fabricated call ID. A real model call may include
members whose cancellation is recorded separately from their eventual broker
and client states. The adapter validates the resulting window with the offline
analyzer's single-window checks and rejects a failed mapping; it does not drop
the original event list or publish a partial success.

The driver should buffer observations with a measured, bounded policy and save
them outside the timed window. Buffer overflow, observer failure or missing
events must invalidate the recording; a caller must never set
`events_complete` to true merely because no exception was visible. Report
instrumentation and hashing overhead symmetrically in both modes. Device-wide
free/total endpoints, allocator peaks, warmup, held-out numerical evidence,
profile approval and supervisor cleanup are separate records and are not
validated by this adapter. Passing timestamps, hashes or `events_complete`
cannot authenticate a producer or prove CUDA synchronization.
