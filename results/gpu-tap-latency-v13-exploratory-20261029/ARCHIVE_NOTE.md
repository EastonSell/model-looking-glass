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
