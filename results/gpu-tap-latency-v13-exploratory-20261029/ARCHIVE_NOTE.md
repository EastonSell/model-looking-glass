# Archived public replay

This directory preserves the public evidence from PR #562. It describes one
completed **contaminated, exploratory** v1.3 session, not an isolated or general
tap-overhead estimate. The original files and their SHA256SUMS are unchanged.

The exact historical `looking_glass/tap_repeated_protocol.py` from source
commit `0c14778447ed6dcbc4890984641370659a09ffef` is retained in
`pinned-source/` because current main has a different protocol. Run
`python3 replay_archived.py` for the CPU-only public arithmetic replay. The
original replay verifies the pinned protocol and input hashes. It does not
authenticate private screening, device custody, or process closure.

This archive does not enable or authorize a new GPU run.

## Source branch movement

On November 1, 2026 at 9:24:57 p.m. Pacific, GitHub records PR #699 merged
into `research/tap-v13-bounded-grant-enabled`, despite that PR's frozen-base
constraint. The branch moved from the recorded source
`0c14778447ed6dcbc4890984641370659a09ffef` (tree
`0ff3b037d19a5bc839e634524d480b6eace5d7c2`) to merge commit
`a23fb638a259b2744bc4f4c3e260f85faf1e1d5c` (tree
`cf197b5923397f96997beb45e05241b28a8e872d`). The merge has the original
source and #699 head `1b7cc475c46bb623c0ec87a6f8f026454756270d` as parents;
its tree equals the #699 head tree. It changes the storage amendment,
controller placement and controller tests. It is not the source of this saved
observation.

Annotated archival tags preserve both histories without resetting the shared
branch: `archive/tap-v13-original-0c147784` and
`archive/tap-v13-storage-merge-a23fb638`. Use the full commit and tree IDs in
`public-binding.json` for historical identity; branch and tag names are only
navigation aids. Neither archival tag authorizes execution. The historical
controller, owner and parent execution gates are enabled, unlike the disabled
successor integrated on main through #782.

GitHub identifies the readiness and merge actor as `EastonSell`, with GitHub
`web-flow` as merge committer. Those records do not identify the initiating
person, thread, browser or API client. See `source-provenance.json` for the
API evidence and exact relationships. This correction preserves all original
evidence bytes and checksums. The public arithmetic replay checks its pinned
protocol and evidence; it cannot prove that the moved branch has the same
controller or full source tree. Native/Torch workloads remain paused.
