# Tokenizer-only selection plan for the FLAN-T5 length-matched successor

This plan was fixed before tokenizing any of these new candidate pairs and before any pretrained forward on them. It defines a separate prospective study; the previous six-cell dose-response run remains an inconclusive 36-forward record with six baseline-unsupported cells and no interventions. The prior observed outcome motivates examining equal target lengths but does not establish that length caused its baseline preferences.

Use the pinned local `google/flan-t5-small` revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, the same seven-file asset manifest SHA-256 `d369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083`, and `T5TokenizerFast` with pad 0 and EOS 1. No model weights are loaded and no pretrained model is run during this selection.

Inspect every row in this exact order, with German targets as written. Retain all source/target texts, IDs, token strings, attention masks, special-token masks, offset mappings, candidate eligibility and reasons, including candidates after the selection is full.

| Order | First English / German | Second English / German |
|---:|---|---|
| 0 | man / Mann | woman / Frau |
| 1 | father / Vater | mother / Mutter |
| 2 | day / Tag | night / Nacht |
| 3 | car / Auto | train / Zug |
| 4 | river / Fluss | mountain / Berg |
| 5 | chair / Stuhl | table / Tisch |
| 6 | red / rot | blue / blau |
| 7 | coffee / Kaffee | tea / Tee |
| 8 | king / König | queen / Königin |
| 9 | brother / Bruder | sister / Schwester |
| 10 | star / Stern | cloud / Wolke |
| 11 | rain / Regen | snow / Schnee |
| 12 | horse / Pferd | rabbit / Kaninchen |
| 13 | salt / Salz | sugar / Zucker |
| 14 | road / Straße | bridge / Brücke |
| 15 | summer / Sommer | winter / Winter |

For each candidate tokenize both `Translate English to German: {word}` and `What is the German word for {word}?`, with their context-only prompts. A candidate is tokenizer-eligible only if **both** templates satisfy all of the following:

1. The source pair has equal lengths and identical all-ones attention and special-token masks; its token IDs differ only at position 6. Both word-offset slices, stripped of surrounding whitespace, equal the intended full English word. Word position 6, instruction position 0 and punctuation position 5 for translate or 7 for question are distinct and non-special; the instruction and punctuation token IDs are unchanged between sources. Source lengths do not exceed 32, and every source/context has exactly one final EOS.
2. Each fixed target has exactly two IDs: one non-special, non-unknown lexical token followed by EOS. The first token's offset slice, stripped, equals the complete target text. Targets are distinct and masks are all ones.

Select the first six eligible candidates, in the stated order, without using model scores, previous baseline margins, or downstream intervention outcomes. Give their two templates pair IDs 0–5 in selection order, preserve the original candidate number in every case, and write the fixed 12-cell manifest and the complete candidate-screen record. If fewer than six candidates satisfy the rule, record `selection-incomplete` and do not run the model; do not add or change candidates. After the manifest is frozen, any selected case that later has an unsupported baseline or unmatchable dose remains reported as such and is never replaced.

The successor keeps the preceding dose-response cell hypothesis and controls: at the same encoder site, compare the word, instruction and punctuation at common absolute doses q/3, 2q/3 and q in both directions; the baseline gate, q/3 eligibility, rounding rules, ordered 0.0001-nat tolerance, 0.05-gap endpoint threshold and dose-scaled 0.02-gap control comparisons are unchanged. Both templates must be evaluable for a lexical pair; both positive make the pair positive, at least one negative makes it negative, and otherwise it is unevaluable. For six selected pairs, aggregate positive requires at least three positive pairs and more positive than negative pairs; aggregate negative requires at least three negative pairs and more negative than positive pairs; ties or too few evaluable pairs are inconclusive. This is an operational rule on a small public sample, not a statistical test.

Twelve cells imply at most 936 root forwards under the prior 78-forward-per-supported-cell schedule. Preserve the CPU-only numerical environment, bounded evidence, raw witnesses, no-op/full-swap controls, final capture repeat, exact source/asset/publication binding and independently observed process exit. The implementation must state and enforce its evidence cap and disk preflight for these fixed shapes before running. Equal target length controls one source of score-length asymmetry; it does not equalize target prior probabilities, guarantee clean baseline support, or establish a unique word-localized mechanism. Some English forms, including `train`, can be ambiguous; the baseline gate and explicit public case scope remain essential.
