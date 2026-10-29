# Correction to the copying-head launch-boundary observation

The launch-boundary record with SHA-256
`0dcbe4b18db1d58b10e05f90f450adb0743d72fea7317078d1db93997f85ea5c`
incorrectly says that no exposure-inventory module changed between v33 and
observed main `57a481d477e3d983b092f963d8d3e24002cc7961`. The filtered diff
used for that observation returned no names, but an unfiltered comparison
shows that `looking_glass/pretrained_exposures_v34.py` had been added. The
incorrect record is retained alongside this correction. The earlier public
prelaunch observation was at main `49dfa2a65912fd41a3d07fbdc4003919e7aa8649`,
where the unfiltered comparison shows no v34 module.

Before the launch, an independent reviewer had compared the v34 candidate's
16 inputs and 32 candidate continuations against all 32 copying inputs and
their target continuations, finding zero prefix relations in 3,072 pairs.
After launch, the official v34 source at `57a481d4` was compared byte for byte
with that already-reviewed candidate: both have SHA-256
`7d3147ee6d474ae560a65b9d1f41effb20b6f1330714cb00ef1aa76d4a0bf5e3`.
The candidate's canonical inventory SHA-256 is
`e5ae202c1e3449a67e92788a83cd0dc62d404a3fa1e052716a11770ee92cc00c`;
independent reconstruction of the official canonical payload is recorded
by the reviewer after launch: all 985 v33 records were unchanged, all 32 own
reservations matched exactly, and 135,552 input plus 135,552 continuation
comparisons found zero public prefix relations. The official-byte and full
inventory comparisons were made after the process started and must not be
described as prelaunch verification or a protected-pool check.

The direct-process start record reports 2026-10-29 11:27:09.994068 UTC by
the local clock. These are cooperative observations, not authenticated time
or independent process attestations. This correction makes no outcome claim.
