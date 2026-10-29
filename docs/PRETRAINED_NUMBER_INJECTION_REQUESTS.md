# Site-independent discovery request plan

`looking_glass.pretrained_number_requests` deterministically assigns unique
32-hex request IDs to the 16 fixed roles for each of the eight public discovery
prompts. The canonical plan binds those roles and input digests to the protocol,
checkpoint revision, full source commit and 32 bytes of public entropy. Each
ID is the first 16 bytes of SHA-256 over the documented domain, public entropy
and four-byte big-endian logical ordinal (case index times 16 plus role
index). The helper checks for collisions and
recomputes the complete plan when validating a request. Loading requires the
caller to provide the separately established published source commit and plan
digest, which are checked on both loading and each request verification. A
`PublicRequestPlan` object by itself is not an authority to accept a request.

`create_family_request_plan` produces the versioned v2 plan: for each family,
both target cases execute their first 15 roles before either `verify_clean`
role. This implements the protocol's checks after the family. Case/role IDs
are unchanged from v1 for the same public entropy; execution order is recorded
in the new plan bytes and digest. The original `create_request_plan` and its
v1 case-major bytes remain available for reproducibility, but its first clean
check is mid-family and it must not launch this study. The discovery journal
requires v2 explicitly. No real study request plan or discovery forward had
been published or executed when this correction was made.

This module generates no randomness and publishes nothing. Before drawing the
separate private site nonce, the trusted custodian must obtain fresh public
entropy independently, durably publish the request plan, and bind every actual
request to it. A failed entropy, collision, publication or validation check is
terminal for that attempt; do not reroll based on a site or outcome. The plan
and its digest do not prove entropy quality, timing, publication, source
execution or security against a custodian deliberately coordinating its
choices. Logs, filenames, exceptions and other side channels also require
separate control. Tests use fixed fabricated entropy and no model or held-out
cases.
