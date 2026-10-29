# Supervised GPU ownership integration tests

`tests/test_gpu_ownership_integration.py` exercises ownership coordination
without initializing CUDA or using a GPU. It creates temporary workspace roots,
uses synthetic full GPU UUIDs, and starts real operating-system processes and
cooperative file locks. Each child is launched directly with the base Python
interpreter; the test propagates the test environment's site-packages through
`PYTHONPATH` for Windows environments whose virtual-environment executable is a
launcher process.

Run from the repository root in an environment with the project's test
dependencies:

```shell
python -m unittest discover -s tests -p test_gpu_ownership_integration.py -v
```

The suite covers:

- A common UUID-derived workspace and lock/marker bundle, validation of the
  active lease before a simulated runtime query, and rejection of a wrong
  runtime UUID before a simulated device callback.
- A running supervised owner excluding a second simulated owner or isolated
  benchmark, followed by clean teardown and a permitted successor.
- Owner crash, supervisor death, and timeout while work is active. The OS
  releases a dead process's lock, but the sticky marker blocks a successor.
- A successful child exit without an attestation, malformed attestation,
  attestation for a different run, and an attempt to switch leases. All retain
  the marker and block another supervised owner.
- The actual `serve_owner`, authenticated loopback IPC, and `Broker` lifecycle
  with two concurrent team requests to one broker, a fake worker, and a fake
  drain verifier. A failed request with verified simulated teardown can clear
  the marker while retaining a nonzero workload exit code. An unverified
  simulated drain retains quarantine even after successful requests and
  closure of the worker and transport.

In these tests, `SupervisionResult.clean` means the supervisor accepted the
matching teardown attestation; it does not mean the scientific workload
succeeded. Participating owners retain their leases until process exit; one
negative test deliberately releases and reacquires a different lease. The
tests use bounded waits; a failed wait is a test failure, not evidence that a
device has drained.

These tests do **not** verify CUDA device properties, CUDA synchronization,
VRAM limits, a real model or GPU, external GPU users, or real benchmark and
profiler entry-point wiring. The synthetic binding and drain callbacks model
trusted inputs only; real owners must obtain and validate the actual runtime
UUID and prove teardown before attesting. The cooperative lock coordinates only
participants using the same physical workspace and canonical identity bundle.
