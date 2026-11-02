# Public-input comparisons before the cross-model agreement study

These are point-in-time, model-free comparisons of the fixed case manifest
(SHA-256 `b1d6dbad76240f07be3a55196ace982767400b60e497d4e7fc75aaae82c8c503`),
not evidence that a model was run or a complete record of all prior exposure.

* [GPT-2 receipt](gpt-public-exposure-review.json) compares 160 inputs and
  declared continuations to the v35 inventory and two published natural
  agreement supports in both full-row-prefix directions. The original support
  covers its earlier protected preparation only under that preparation's
  recorded binding. The 32 inputs are now reserved in v36, canonical SHA-256
  `6f05f07a0fe6ed61ca00978ef4b225e43c9c71a4a8be05df33e8c7ed778d06b7`.
* [BERT receipt](bert-public-exposure-review.json) compares all 32 masked
  inputs to 1,061 named public row references, including the 37 planned
  color cases, both as complete rows and without final SEP. It reports zero
  matches among 33,952 proposal/reference pairs and 496 proposal pairs.
  Repeated references count repeatedly. After this check, the color screen
  recorded nine observed cases and 28 planned cases in v2; its row set did
  not change. Neither receipt covers unknown or pretraining exposures.

The adjacent review scripts preserve the independent comparison logic with
path arguments substituted for the original local workspace paths. The GPT-2
script requires the pinned tokenizer JSON and a checkout containing the cited
Git objects and v35 source modules:

```text
python review_gpt_public_exposure.py REPO CASE_MANIFEST TOKENIZER_JSON OUTPUT
python review_bert_public_exposure.py REPO CASE_MANIFEST > OUTPUT
```

The path-adapted scripts were rerun without a model; the GPT receipt matched
byte-for-byte and the BERT receipt matched as parsed JSON. The original
independent script SHA-256s were
`c95ba6ac730bed5889af70025be0c317c6babe7b46129efc72e4a30bcf73a0bf`
for GPT-2 and
`65291cf66af726e978f172cc5915eea69a9c1cb70a95a9fe52ed44c38952758f`
for BERT. Refresh applicable public inventories and pending cases at launch;
these receipts do not certify later changes.
