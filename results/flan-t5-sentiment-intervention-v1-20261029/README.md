# FLAN-T5 sentiment intervention: selection-incomplete

The planned held-out intervention did not proceed. Only **1 of 12** fixed lexical units qualified in both instruction frames, below the preregistered minimum of four. The custodian reports that no pretrained model was loaded or forwarded in this attempt; no replacement units were drawn. This is a tokenizer-screen outcome, not a model-performance or causal-intervention result.

The earlier [development screen](../flan-t5-sentiment-screen-cpu-v1-20261029/README.md) produced the expected strict greedy label on 64/64 constructed inputs. Its frozen exact-span alignment rule yielded 0/32 opposite pairs; a separately reported post-screen rule allowing one leading ASCII space yielded 32/32. The [prospective intervention protocol](../../docs/FLAN_T5_SENTIMENT_INTERVENTION_V1.md) adopted that whitespace-aware rule for a new fixed pool. The 1/12 result here is a separate, prospective screen and does not alter either development count.

## Screen result

The custodian fixed 12 ordered units, each with four sentiment words and one subject, and a separate reviewer recorded semantic dispositions before tokenization. The two fixed instruction frames produced **24 groups and 96 input rows**. Ten units were semantically accepted; two were rejected. All 12 passed the recorded lexical exclusions against the ten bound public sources.

| Recorded condition | Units or rows |
| --- | ---: |
| Eligible in both instruction frames | 1 of 12 units (`u01`) |
| Adjective alignment failed | 11 of 12 units, in both frames |
| Within-group token-difference check failed | The same 11 units, in both frames |
| Semantic review rejected | 2 units, both among those 11 |
| Rows with unaligned adjective spans | 34 of 96 rows |

For those 34 rows, the saved adjective spans overlap two tokens in 18 rows, three in 4, four in 10 and five in 2. Thus the observed alignment failures involve multi-token adjective segmentation; they are not merely the development screen's leading-space offset issue. These are overlapping failure counts, not additive exclusions. The constructed units are a small local sample and do not establish the general frequency of single-token sentiment words.

The screen recorded **60,000** comparisons: each of the 96 candidate rows against 625 recorded row occurrences from ten public sources. No bidirectional, final-EOS-stripped complete-row-prefix collision was recorded. The source list includes previously rejected and exposed cases as well as the fixed weight-edit and known-injection proposals. This is a check against those recorded sources, not proof of no unknown exposure, pretraining overlap or semantic independence.

## Evidence and chronology

The complete [pool](custody/pool.json), [semantic draft](custody/semantic-draft.json), [semantic verdicts](custody/semantic-verdicts.json), [lexical candidates](custody/lexical-candidates.json), [source descriptors](custody/exposure-descriptors.json) and [96-row screen](custody/selection.json) are released here. The [pool salt](custody/pool-salt.hex) and [selection salt](custody/selection-salt.hex) are the original 32-byte salts encoded as lowercase hexadecimal followed by a newline; decode the 64 hexadecimal characters to recover each original salt. The canonical pool and screen files retain their original no-newline bytes.

The [pool card](custody/pool-card.json) was published at commit `a4039ebc0262b2756f3885134017495f5a26d3a9` and independently read back before the tokenizer stage. The [selection card](custody/selection-card.json) was published at commit `332ea04e73ae977915ea152a8140a31373febb7d` and independently read back before any planned model forward. GitHub ancestry places the pool-card commit before the selection-card commit. The [pool](custody/pool-observation.json) and [selection](custody/selection-observation.json) card observations record the exact paths, commits and raw SHA-256s. The private screening timeline, independent salt generation and secrecy are cooperative custody observations, not authenticated by Git ancestry alone.

The tokenizer-only pool and selection stages each recorded a completed direct child, exit code zero and drained bounded logs, and each foreground observer also recorded exit zero. Their [stage records](process/) and the [closure receipt](custody/selection-incomplete-closure.json) are included. Original PID-bearing logs and spawn records remain private; the public stage records preserve their original log hashes, but do not by themselves establish execution or descendant termination. The no-model/no-forward statement is the custodian's report, consistent with the reviewed tokenizer-only source and retained records.

The [independent tokenizer replay](custody/independent-selection-replay.json) reconstructed all 96 saved rows and the fixed first-four rule from the pinned assets and ten public source inventories. A [second independent audit](audit/second-independent-audit.json) separately retokenized the rows through the fast tokenizer wrapper and its underlying tokenizer, and reproduced the failure and comparison counts. Its [source](audit/second-independent-audit.py) is included; its full process-log checks require the retained private originals. Neither audit authenticates model execution, the independent semantic judgment, pretraining history or physical secrecy. A separate [custody review](audit/custody-review.json) reconciled the staged files and their retained originals. Its recorded README hash is the prior version; the only later wording change adds these custody-review and manifest links. The [package manifest](manifest.json) lists file lengths and SHA-256s, excluding itself.

## Replaying the released screen

With a trusted checkout containing the Git objects named by the source descriptors and both card observations, the exact [independent selector reader](../../tools/flan_sentiment_independent_selection_v1.py), and the seven pinned FLAN-T5 asset files plus their included [manifest](custody/asset-manifest.json), run with Python 3.11.15, Transformers 4.57.3 and tokenizers 0.22.2:

```text
python -I -B results/flan-t5-sentiment-intervention-v1-20261029/replay_released_screen.py \
  --repo CHECKOUT_WITH_RECORDED_GIT_OBJECTS --asset-dir PINNED_ASSET_DIRECTORY
```

The [public replay wrapper](replay_released_screen.py) checks the reader and public cards against their recorded Git blobs, reconstructs the lexical draft and recorded semantic dispositions, verifies the asset hashes, retokenizes the 96 rows and checks both salted commitments and the fixed selection. It reads the checkpoint weight file for hash verification but does not load or run a model. It does not replay the private process logs or prove publication timing, native-library identity, semantic truth or absence of unrecorded exposure. The original full independent reader's process-log validation likewise requires those private logs.
