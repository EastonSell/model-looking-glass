# Qualified recovery after the incomplete GPU resource pilot

After the [five-token resource pilot](README.md) timed out on 29 October 2026,
the operator retained the original unsafe marker and partial evidence for a
separate device-safety assessment. The saved supervisor result and reviewed
source support an inference that the exact child was stopped and waited for and
the Windows Job finish path returned. The saved result does not include the
Job's raw process counts or independently prove the absence of escaped or
unrelated processes. **The owner did not produce a clean attestation, and its
CUDA or native cleanup remains unverified.**

The operator performed a qualified manual recovery under the shared controller
lock and device lease, in that order. A fresh, read-only inventory matched the
recorded physical device and driver; it observed 7,026 MiB free of 8,192 MiB.
The original marker was preserved, then retired while an explicitly non-clean
recovery hold continued to block cooperative launchers. Independent reviewers
checked the first transition's receipt, process exit, and hold identity before
a second, separately reviewed operation removed the hold under both locks.
Its final point sample observed 6,959 MiB free. Both recovery processes reported
exit code 0. The coordinator released its scheduling hold after independent
readback of the saved records. Each subsequent workload still requires its own
fresh device identity, ownership, and memory-admission checks.

This was an explicit operator risk decision, not the supervisor's automatic
clean-attestation path. It assumed a trusted, quiescent coordination directory
and cooperating launchers; the scheduler hold was not an OS-enforced or
crash-durable admission gate. Windows directory-entry durability after power
loss was not established. NVIDIA's WDDM mode did not supply per-process GPU
memory values in these observations, and the compute-process listing was not a
complete inventory of all possible GPU users. Device-wide free memory is a
point-in-time reading, not proof of cleanup, global idleness, or future
headroom. The ten completed samples remain an incomplete attempt, **not an
approved profile, a memory bound, or a behavioral result**.

## Evidence bindings

The [published evidence inventory](evidence/inventory.json) has SHA-256
`8dbade4a125188e942314a2c259d94a856fd17d4cd76b919a33c29b58b1d4277`;
the [timeout projection](evidence/terminal-projection.json) has SHA-256
`e6d89bd75708dfb602e9f80cb8a591e7d4cbab8086f108c59c3c219efad2c918`.
The original private incident, marker, and recovery records remain retained.
The following SHA-256 values identify exact private originals; they do not let
readers independently inspect or authenticate withheld contents:

| Private original | SHA-256 |
| --- | --- |
| Preserved original marker | `41d57fb3c997cf7ddd53fc77dfd66e3fe2a53a78def6b6b2100563bd06eee2a5` |
| Read-only device assessment | `34143a64f865be7e8258d7c811fad3f1f755b52ed9fdefd84dcaa38aaba281e1` |
| Qualified recovery decision | `6a4a6fb3a114910bbcab144f71d72a526d4487395167e8073b262bcbd6cebcff` |
| Marker-retirement receipt | `bb77876197ef43d9f4c2a768db371daff3bec2430e735ff106cb3e081354b4bd` |
| Independently reviewed hold-release decision | `aea7957abd08ca5146b97c83d676e94e23a76e75a1d0a1e7194f9cf6506f9479` |
| Pre-release record | `5f21ca779cfe9a801a74212f9d59fde1eb7d2030a5349cb00be2bc365866f858` |
| Release process record | `8c5facee478762f73afd1a44f629763dc3d39b363e2e93a37cad6a48f10c7134` |
| Qualified recovery record | `f9a8bbe21d2aec78e7046670a3c3011c63c6f88205bc9548a74e7b2a048fd9ff` |

The recovery assessment and operator transition were independently reviewed
against the private records and CPU-only interruption tests. The public timeout
subset can be checked with the verifier described in the pilot README; neither
that verifier nor these hashes independently proves device cleanup.
