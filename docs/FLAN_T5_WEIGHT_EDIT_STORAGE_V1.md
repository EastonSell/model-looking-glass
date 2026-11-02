# FLAN weight-change study: storage plan

Before either supervised CPU phase, independently inspect the resolved output
path and current free space on both volumes. The attempt directory, its raw
logits, process logs, Python bytecode prefix, and `TEMP`/`TMP` directories use
`E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v1` on E:.
Create each phase's attempt and temporary directories exclusively; do not
reuse or delete prior evidence. E: must have at least 2 GiB plus the 256 MiB
total evidence cap plus a 1 MiB failure reserve free at each launch.

The existing seven-file checkpoint cache, source object store, Python base
interpreter and validated venv dependencies remain on C:. Verify the original
cache in place before and after each phase; do not copy the 307,867,048-byte
weights or download new ones. Before either launch C: must have at least
2 GiB plus 128 MiB free for small source, interpreter and staging writes.
This C: margin is a conservative operational heuristic, not an operating
system reservation or a guarantee about paging, native libraries or other
processes. Treat a path resolving to the wrong volume, a missing volume,
insufficient free space or uncertain output accounting as a no-launch/incomplete
condition as appropriate; recheck immediately before each direct-child grant.

An operator observation before this note found E: to be NTFS with approximately
400 GB free and C: approximately 2.35 GB free. Those are transient observations,
not launch-time attestations. Git publication, including any large object
staging, needs its own fresh space check; it must not silently duplicate the
bounded evidence on C:. `TEMP`/`TMP` redirection and an empty, disabled
bytecode prefix do not claim to control every library write or OS paging.
