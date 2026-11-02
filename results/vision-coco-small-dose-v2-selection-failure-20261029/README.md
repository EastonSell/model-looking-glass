# Small-dose v2 annotation-only selection failure

The fixed selector and protocol were published at commit
`95b9f9172db503bb30c01f44bbef46f74d0fbf6b` before this attempt. The selector
could not produce eight globally disjoint target/donor quadruplets; an
independent annotation-only reconstruction also found no qualifying single
quadruplet under that geometry and candidate rule. After the prior v1 image
exclusions, 14 cats and seven dogs qualified. No image was downloaded and no
model was loaded or forwarded; no v2 case manifest was produced.

The original [supervisor receipt](supervisor.json) has SHA-256
`570a0fdb7f231d32bcee7d8a3a0ef5461516f89b9435eed80758b6c683f5d0bd`.
It records exit 1, zero forwards and the original stderr SHA-256
`c1cadc4125ccf06a2e9ec001a1a46f41b1bc99cafadf2fa87cd665a208858fec`.
The path-bearing stderr is retained locally rather than republished. These
records establish the saved outcome within the trusted execution scope; they
do not independently attest execution or timing. The explicitly versioned
same-site v3 protocol is a new annotation-only selection rule, not a successful
completion of this v2 attempt.
