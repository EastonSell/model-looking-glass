# Client-side cancellation acknowledgement observation

An IPC client may opt in with `IPCClient(..., observe_cancel_ack=True)`.
The returned handle then retains at most one immutable
`CancelAckObservation(accepted, received_ns)`. The timestamp is a client
`time.perf_counter_ns()` reading taken after a complete acknowledgement frame
has been received, parsed and checked against the handle's request, team,
broker and job IDs. It is not the broker's cancellation decision timestamp,
its linearization point, a model completion, device synchronization or
memory release. A driver must record the clock implementation and nominal
resolution and independently establish whether client and owner clocks are
comparable. This receipt cannot supply the source decision timestamp required
by the current batched-event protocol.

The observation is bounded to one acknowledgement. An unsolicited frame, a
second acknowledgement, a failed or invalid clock reading, or an accepted
acknowledgement followed by a successful terminal result marks
`handle.cancel_ack_observation_failed` true. An ordinary observation failure
does not replace the actual terminal result. A control-flow interruption
propagates; a pending capture also makes the failure property true. The caller
must independently reject an interrupted or unjoined wait, an incomplete
producer or an uncertain owner even when this property is false. The stored
receipt is provisional until the client wait and all relevant producers have
finished successfully. A driver must retain the request attempt and the
outcome of its own `cancel()` call separately.

The cancellation attempt is marked before sending the frame, because the
peer can acknowledge while the send call is still returning. The receipt
therefore does not prove that the caller's send returned successfully. If the
server delivers a terminal outcome directly, a timeout or disconnect occurs,
or no acknowledgement is read, the receipt is `None`: absence does not prove
that cancellation was rejected or never processed. Client `wait()` retains its
existing socket-close behavior on both success and failure. The legacy
`cancel_acknowledged` boolean remains unchanged; its default false value alone
does not distinguish a rejected acknowledgement from an absent one.

The default client has this observation disabled and does not read the extra
clock on acknowledgements. Enabling it adds client-side timing and allocation
work, so performance studies must account for instrumentation separately in
each mode. The existing loopback authentication and custody boundaries are
unchanged; a validated frame and timestamp are observations, not independent
proof of the server's internal state.
