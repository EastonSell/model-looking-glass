# Experimental development host bootstrap

The fixed two-client development bootstrap requires explicit opt-in and the
three independently approved profiles for the exact final source and live owner
path. Before startup it charges a fixed 4 GiB logical host allowance and
requires a separate 2 GiB reserve. The existing 8 GiB minimum applies to each
fresh physical, system-commit and calling-process-commit headroom value. The
claim and original run graph stay rooted until controller process exit, even
when startup, collection or shutdown is uncertain. The result and retained
claim identify `development-bootstrap`; inspect client outcomes and supervision.

This allowance is experimental: it is neither an OS reservation nor an RSS cap.
The earlier [two-client observation](../results/serial-two-client-gpu-bootstrap-v1-20261029/summary.json)
recorded separate owner and controller lifetime private-commit peaks totaling
2,636,173,312 bytes, leaving 1,658,793,984 bytes below the 4 GiB allowance.
Those peaks need not be simultaneous, exclude some process and system
allocations, and came from one successful run on an earlier source and cache
path. They do not bound this source, cancellation, out-of-memory failures or
future runs. The 4 GiB bootstrap has not been measured; changed source and
runtime require matching profiles and fresh admission.
