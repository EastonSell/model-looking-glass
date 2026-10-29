# Preparing the fixed two-token supervised service

`prepare_two_token_service` prepares a caller-held control object for the
separately profiled, fixed DistilGPT2 workload. Keep that object **before**
calling `start()`: an interrupted startup can have created a child, and the
same object is needed to request shutdown or wait for its supervisor.

This pilot accepts only `input_ids=[[15496, 995]]`, batch one, and warm seeds
29, 43 or 71. It returns bounded full logits. It does not implement the longer
agreement workloads, tracing, interventions, batching, or guaranteed fairness.
The current owner uses a serial ready queue; transport connection slots are
global and are reserved before team authentication. The separate fair bridge
requires integration and review before a stronger scheduling claim is made.

Before use, obtain three independently reviewed **worker-path** profiles from
the same frozen checkout path, interpreter, environment and selected physical
device: model load, cold forward and warm forward, in that order. The direct
model one-shot is not a worker profile. Pass each as a `ProfileFiles` containing
its absolute document, evidence and source-manifest paths plus the independently
approved full profile ID. Current profiles are checked during preparation and
again immediately before startup; the child independently reconstructs live
runtime identity, the fixed token case and ordered stage seeds under its lease.
A fixed-commitment owner, a supervisor with the reviewed child-reaping contract,
and collector source must also be present in the selected checkout before
launch. This compatibility check does not prove
that evidence was collected as claimed or independently reviewed; source custody
and approved pins are trusted operator inputs.
A source or runtime change requires new matching profiles. Measured estimates
and the 2 GiB resident / 512 MiB job commitments are not OOM guarantees.

Call `prepare_two_token_service` with absolute repository, offline cache,
workspace, private launch parent, environment and site-packages directories;
a controlled executable search path; distinct team IDs; the selected full GPU
UUID; and a **current trusted physical UUID inventory tuple**. The inventory
can be obtained from a trusted `nvidia-smi --query-gpu=uuid --format=csv,noheader`
and parsed with `parse_nvidia_smi_inventory`. Keep UUIDs and private paths out
of published results. Preparation hashes the selected source and pinned offline
assets, verifies profile files against their approved pins, prepares a private
config and the fixed child command, and derives the common UUID/workspace
lease and marker. It does not initialize CUDA or start the service.

After the scheduled exclusive window begins, call `service.start()`, obtain a
team client through `service.client(team)`, and submit the fixed request using
the documented IPC interface. Signal and wait with `service.shutdown()`;
`service.wait()` can recheck a supervisor after a timeout. A shutdown before
the supervisor is claimed closes the start gate and returns `None`. Otherwise
consult the returned supervisor result and the local
`service.unsafe_marker_path`; a failed call or missing result is not proof that
the owner exited cleanly. The marker path contains the physical UUID and is
private. Previously issued clients may retain credentials; discard them when
the owner stops.

The controller sends no credentials in command arguments or the config file.
The current IPC client does not authenticate a server before sending its bearer
token, so use this only with trusted cooperative local processes and protect
the endpoint against port reuse. The source, interpreter, native dependencies,
private directory and bytecode prefix also require trusted exclusive custody;
the launcher records hashes but does not attest executed bytecode or isolate
hostile same-user processes. See `TRUSTED_SERVICE_LAUNCH.md`, `GPU_SERVICE.md`
and `IPC.md` for these boundaries and the supervisor's quarantine behavior.

CPU tests exercise the prepared command, identity and configuration, repeated
preflight and cancellation gate. They do not establish a measured profile,
successful CUDA execution, or a GPU service lifecycle. Actual execution stays
gated until the matching collector, fixed-commitment owner and their independent
reviews are complete.
