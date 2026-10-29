"""Compare a closed public 640-row pilot to two local tokenizer interfaces.

The Hugging Face fast tokenizer and direct Rust Tokenizer share tokenizers'
backend; agreement is an interface/configuration check, not independent
implementation validation. No model, checkpoint weights or private pool read.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
import tokenizers
import transformers
from transformers import T5TokenizerFast

PILOT_SHA = 'c1426f1ea4690f7c5185580167001d946425dcf0e3d08664a1f993fc7ac0a62f'
CANDIDATE_SHA = '2b32da31aab398d19ef80e013c377d2ed3d1a06912a431ee918bfeccb64107d1'
ASSET_SHA = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
TOKENIZER_SHA = 'fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, expected):
    raw = path.read_bytes()
    require(sha(raw) == expected, f'fixed input differs: {path.name}')
    return json.loads(raw)


def compare(pilot_path, candidates_path, asset_dir):
    pilot = load(pilot_path, PILOT_SHA)
    candidates = load(candidates_path, CANDIDATE_SHA)
    assets = load(asset_dir / 'asset-manifest.json', ASSET_SHA)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model'):
        expected = assets['files'][name]['sha256']
        require(sha((asset_dir / name).read_bytes()) == expected, f'fixed tokenizer asset differs: {name}')
    require(assets['files']['tokenizer.json']['sha256'] == TOKENIZER_SHA and
            pilot['tokenizer_sha256'] == TOKENIZER_SHA and pilot['asset_manifest_sha256'] == ASSET_SHA and
            pilot['candidate_sha256'] == CANDIDATE_SHA, 'pilot tokenizer binding differs')
    rust = tokenizers.Tokenizer.from_file(str(asset_dir / 'tokenizer.json'))
    fast = T5TokenizerFast.from_pretrained(str(asset_dir), local_files_only=True)
    require(type(fast).__name__ == 'T5TokenizerFast' and fast.eos_token_id == 1 and fast.unk_token_id == 2,
            'fast tokenizer identity differs')
    require(len(candidates['positive']) == len(candidates['negative']) == 80 and
            len(candidates['templates']) == 4 and len(pilot['all_candidates']) == 160 and
            pilot['tokenized_rows'] == 640, 'pilot inventory differs')
    lexical_counts, whole_counts, rows = Counter(), Counter(), 0
    signature_hashes = Counter()
    for index, recorded in enumerate(pilot['all_candidates']):
        polarity = 'positive' if index < 80 else 'negative'
        order = index if index < 80 else index - 80
        word = candidates[polarity][order]
        require(recorded['polarity_proposal'] == polarity and type(recorded['order']) is int and
                recorded['order'] == order and recorded['word'] == word and len(recorded['templates']) == 4,
                'candidate order or word differs')
        signature = []
        for template_index, (template, row) in enumerate(zip(candidates['templates'], recorded['templates'])):
            require(type(template) is str and template.count('{word}') == 1, 'template differs')
            text = template.replace('{word}', word)
            begin, end = template.index('{word}'), template.index('{word}') + len(word)
            require(type(row['template_index']) is int and row['template_index'] == template_index and
                    row['text'] == text and row['adjective_char_span'] == [begin, end] and
                    text[begin:end] == word and begin > 0 and text[begin-1] == ' ', 'text or word span differs')
            encoded = rust.encode(text, add_special_tokens=True)
            via_rust = {'input_ids': list(encoded.ids), 'attention_mask': list(encoded.attention_mask),
                        'special_tokens_mask': list(encoded.special_tokens_mask),
                        'offset_mapping': [list(pair) for pair in encoded.offsets]}
            hf = fast(text, add_special_tokens=True, truncation=False, padding=False,
                      return_attention_mask=True, return_special_tokens_mask=True, return_offsets_mapping=True)
            via_hf = {key: ([list(pair) for pair in hf[key]] if key == 'offset_mapping' else list(hf[key]))
                      for key in via_rust}
            for key in via_rust:
                require(via_rust[key] == row[key], f'Rust {key} differs at row {index}/{template_index}')
                require(via_hf[key] == row[key], f'Hugging Face {key} differs at row {index}/{template_index}')
            offsets, ids = via_hf['offset_mapping'], via_hf['input_ids']
            whole = [i for i, (a, b) in enumerate(offsets) if a < end and b > begin - 1]
            lexical = [i for i, (a, b) in enumerate(offsets) if a < end and b > begin]
            require(whole and lexical and whole == list(range(whole[0], whole[-1]+1)) and
                    lexical[-1] == whole[-1] and row['word_plus_space_token_indices'] == whole and
                    row['lexical_token_indices'] == lexical and row['last_lexical_index'] == lexical[-1] and
                    row['standalone_leading_space'] == (offsets[whole[0]] == [begin-1, begin]),
                    'word overlap or final adjective site differs')
            require(all(begin-1 <= offsets[i][0] < offsets[i][1] <= end and
                        via_hf['attention_mask'][i] == 1 and via_hf['special_tokens_mask'][i] == 0 and
                        ids[i] != fast.unk_token_id for i in whole) and
                    {j for i in whole for j in range(*offsets[i])} == set(range(begin-1, end)) and
                    ids[-1] == fast.eos_token_id and ids.count(fast.eos_token_id) == 1 and
                    len(ids) <= 64 and row['structural_rejections'] == [], 'structural alignment differs')
            left, right = whole[0], whole[-1] + 1
            signature.append({'lexical_count': len(lexical), 'word_plus_space_count': len(whole),
                              'last_lexical_index': lexical[-1], 'prefix_ids': ids[:left], 'suffix_ids': ids[right:],
                              'prefix_attention': via_hf['attention_mask'][:left],
                              'suffix_attention': via_hf['attention_mask'][right:],
                              'prefix_special': via_hf['special_tokens_mask'][:left],
                              'suffix_special': via_hf['special_tokens_mask'][right:]})
            lexical_counts[str(len(lexical))] += 1
            whole_counts[str(len(whole))] += 1
            rows += 1
        raw_signature = json.dumps(signature, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        require(recorded['signature'] == signature and recorded['signature_sha256'] == sha(raw_signature) and
                recorded['structural_rejections'] == [], 'saved alignment signature differs')
        signature_hashes[sha(raw_signature)] += 1
    require(rows == 640 and pilot['structural_failure_count'] == 0, 'row/failure count differs')
    return {'schema': 'flan-multitoken-tokenizer-independent-check-v1',
            'pilot_sha256': PILOT_SHA, 'candidate_sha256': CANDIDATE_SHA,
            'asset_manifest_sha256': ASSET_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
            'runtime': {'python': sys.version.split()[0], 'transformers': transformers.__version__,
                        'tokenizers': tokenizers.__version__, 'huggingface_class': type(fast).__name__},
            'candidate_count': 160, 'row_count': rows, 'rust_row_mismatches': 0,
            'huggingface_row_mismatches': 0, 'word_span_or_signature_mismatches': 0,
            'lexical_token_count_across_rows': dict(sorted(lexical_counts.items())),
            'word_and_space_token_count_across_rows': dict(sorted(whole_counts.items())),
            'distinct_alignment_signatures': len(signature_hashes),
            'scope': 'Outcome-informed, development-exposed tokenizer comparison. Both interfaces use the '
                     'same Rust fast-tokenizer backend; no model, weights, private pool or behavioral inference.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.pilot, args.candidates, args.asset_dir)
    raw = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.report.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'short report write')
    print(json.dumps({'report_sha256': sha(raw), 'rows': result['row_count'],
                      'mismatches': result['rust_row_mismatches'] + result['huggingface_row_mismatches']}))


if __name__ == '__main__':
    main()
