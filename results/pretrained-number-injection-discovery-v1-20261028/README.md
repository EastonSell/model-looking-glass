# DistilGPT2 number-injection discovery: public evidence

This directory contains the closed, site-blind public observations and frozen discovery prediction from one CPU run of the preregistered number-injection track. The 128 observations and `public/prediction.json` are the original retained bytes. The public archive digest below covers the 128 observation files; the prediction has its own digest. The concealed injection layer and site nonce were not included in this publication.

The frozen prediction selected candidate layer 4 (zero-based): all eight public discovery cases voted for it as the first changed candidate. The mean clean-to-injected patch effect on the specified logit margin at that candidate was -0.0059299468994140625; clean accuracy on these eight cases was 8/8. These are discovery results on a small selected public set. The predicted candidate is not presented as the true concealed site, and there are no held-out results or claims about general reasoning here.

The `audit/` directory contains two independent checks of the public bytes: the prebound public-only scorer and a separate bit-level headline check, plus their saved results and the provenance review. They did not inspect the site nonce, private witnesses, or held-out inputs. The public-data audits establish consistency of retained bytes and calculations; by themselves they do not prove execution, secrecy, publication timing, or a complete absence of sensitive metadata. The owner records report a zero child exit, reaping, no remaining job members, and matching closed-output hashes. Those are retained operator observations, not externally authenticated timing or execution attestations.

## Bindings

- Frozen source: `e84f66ae63b6f20e6df81789f2e4741b13598fb1`; protocol SHA-256: `95a8914ca1b7412cfe931a9ccb5341a388ec391159c934095c63a966beb051ee`.
- Model and tokenizer: `distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
- Published runtime SHA-256: `b6860ec21fe6f0ab626164b86b494d783f3c18c92a697744b118be4cdb33754e`; public plan SHA-256: `e59d98638e474ed9f915d24fa527edd6357caa0e62c8bdabdda0433c20c683f8`.
- Observation archive SHA-256: `082fee918cffe437c5fe49732c1cf9045fba9ace39608ceca5c68efabe61c523`; prediction SHA-256: `11376d8aae517db76d6a33c1ae5dc1f4b0c747a9d5216c9b02a17b1b0b300107`.
- The public auditor was prebound at source `82cc1b87dce824306650ac776dc509dca003500d`, `tools/review_injection_discovery.py` SHA-256 `19ae9cb66a42e010ca66eacde2315420e04fb5eeff5c0b609e9360d65fea9e17`.

The runtime, plan, site commitment, and public-auditor binding were published as successive commits before this discovery evidence. The plan and site commitment do not themselves embed both runtime and plan digests: retained markers and the later auditor binding associate those artifacts, subject to the protocol's cooperative chronology limitation.

The owner ran CPU-only with CUDA hidden while a separate GPU resource-profile sampler was active; this is not an isolated timing measurement. The initial physical inventory omitted four later-discovered duplicate checkout roots. Their known public copies were accounted for in the [initial-inventory gap record](https://github.com/EastonSell/model-looking-glass/blob/00edc1ba30abacf6d159490d4c7c881d8a80b603/results/agreement-physical-accounting-v1-20261029/initial-inventory-gap.json). That record is a lower-bound observation, not a complete current inventory; claim-preserving reconciliation remains required before later phases. The original discovery claim remains consumed and the site was not redrawn.
