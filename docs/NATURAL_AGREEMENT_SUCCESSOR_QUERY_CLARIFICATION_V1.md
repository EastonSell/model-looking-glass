# Natural successor: cumulative disclosure and query output

This prospective clarification supplements the [frozen successor protocol](NATURAL_AGREEMENT_SUCCESSOR_V1.md)
without changing its public pool, prediction, sampler, one-query rule, or
no-redraw conditions. It must be bound before the successor's pool query.

Before this query, seven distinct public-input predicates had been compared
with the original committed natural split: 240 rows across those inputs, 200
unique rows, and seven recorded clear results. The year-transfer predicate was
deliberately repeated on the same fixed rows and sealed split to obtain a
source-bound process observation, also recording clear. That makes eight
invocations and 288 submitted rows in total. The proposed fixed 256-by-16 pool
comparison is a further aggregate disclosure; it is not redundant confirmation
of those earlier checks. No private row, nonce, or per-row finding is needed for this
account.

The intended public result is only `clear` or `collision`, the fixed public
input and comparator digests, the original salted commitment, the counts, and
the comparison method. A collision, failed check, or uncertain process outcome
ends the fixed attempt without revising the pool or making another private
query. Do not publish the internal comparator receipt's observation time,
process timing, private paths, error details, or per-row diagnostics.

This is a cooperative-custodian disclosure boundary, not a formal one-bit
privacy guarantee. The published comparator stops on its first collision, so
the trusted operator can observe timing or failure channels even when the
public result is aggregate. Publication timing and the earlier clear results
can also reduce uncertainty about the protected split. Preserve internal
process and receipt evidence for the reviewed custodian without releasing its
private details.
