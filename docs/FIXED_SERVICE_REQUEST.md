# Fixed service request pre-validation

`FixedServiceRequestFactory` wraps a trusted tensor-producing factory for the
fixed, untraced two-token service. It permits only a bounded ASCII request ID,
the exact built-in JSON shape `{"input_ids": [[15496, 995]]}`, and warm seed
29, 43 or 71. All checks occur before the downstream factory is called. An
invalid request raises the IPC layer's exact `IPCInputRejected` signal without
creating a tensor or retaining an input alias. For accepted requests, the
wrapper constructs a fresh canonical value for the tensor factory, so changes
to the caller's mutable input cannot change the admitted token IDs.

If the downstream factory raises that signal, the wrapper converts it to an
ordinary uncertain failure: the wrapper cannot prove that a tensor factory
had no side effects or retained allocations. Other factory exceptions remain
uncertain as well. This is trusted in-process behavior, not authentication or
a security proof for an arbitrary factory. The IPC clean-rejection seam and
owner wiring require separate review; this module does not enable a service.
