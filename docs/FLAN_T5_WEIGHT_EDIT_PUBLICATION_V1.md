# Development evidence publication on E:

The fixed weight-edit study has not forwarded a case when this plan was
prepared. Its frozen protocol requires the development evidence and an
independent saved-data and process review to be public before evaluation.
The original C: reserve and its prospective amendment remain in force.

Keep the closed development files on E: in the fixed attempt directory.
Upload binary logits from E: as ordered Git blobs of at most 1 MiB each;
publish the ordered part paths, lengths and SHA-256 digests alongside the
complete-file SHA-256. Publish the other evidence and independent review as
ordinary Git blobs. No large evidence file or model checkpoint is copied to C:.
The published part paths and the aggregate local and public byte hashes must
agree before evaluation; missing, duplicated, reordered or altered parts stop
the attempt without an evaluation forward.

The publication and handoff Git object store is a small, blob-filtered bare
repository at `E:\Codex\model-looking-glass\work\flan-t5-weight-edit-v1\fixed-attempt\publication.git`.
Fetch the exact immutable published revisions there and let Git obtain only
the required blobs. The output-volume check is repeated before publication and
before each phase; C: is rechecked under the amended heuristic. The object
store and the original phase files both count toward the cumulative 256 MiB
evidence budget, with the additional 1 MiB failure reserve. If they exceed
that bound, preserve the closed evidence and do not start evaluation. The
filesystem observations are point-in-time measurements, not reservations or
limits on unrelated operating-system writes.

The independent review must bind the exact result, journal, logits, phase,
exit receipt, child logs and scoped custody receipt, and explicitly cover
saved-process and restoration observations as well as the saved-logit replay.
These are cooperative records; their agreement does not authenticate execution
or chronology. No private custody rows are included.
