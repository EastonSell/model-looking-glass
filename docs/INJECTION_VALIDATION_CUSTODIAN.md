# No-model validation custodian

This one-shot caller holds the captured source, public input, configuration,
offline asset and protected empty-bytecode-prefix handles before it grants the
reviewed no-model host. It roots the host process object before initialization,
duplicates the protections into that process, retains the exact process handle
through zero exit and reap, and checks the nested host terminal. An exception
after launch keeps the rooted process and leases in the existing cooperative
quarantine. Forced process death and arbitrary native return boundaries remain
outside that guarantee.

The caller must hash-verify this source before execution, release its start
token, and separately retain and reap this custodian. Its terminal is marked
`awaiting_custodian_exit`; `check_custodian_terminal` checks saved consistency
against an externally retained PID and creation time but does not itself
observe that exit. Reconcile the caller's final receipt and every diagnostic
file before the one-use phase claim. This component makes no claim, reads no
site or split, loads no model and performs no forward.
