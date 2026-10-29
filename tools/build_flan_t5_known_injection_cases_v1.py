"""Tokenize the fixed public FLAN-T5 injection study; never load a model."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update({'CUDA_VISIBLE_DEVICES': '-1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1'})

CASES = (
    ('discovery-01', 'discovery', 'Archive label: basalt compass.', ('basalt', 'compass')),
    ('discovery-02', 'discovery', 'Archive label: cobalt compass.', ('cobalt', 'compass')),
    ('discovery-03', 'discovery', 'Archive label: amber lantern.', ('amber', 'lantern')),
    ('discovery-04', 'discovery', 'Archive label: cedar lantern.', ('cedar', 'lantern')),
    ('heldout-01', 'heldout', 'Catalog label: quartz turbine.', ('quartz', 'turbine')),
    ('heldout-02', 'heldout', 'Catalog label: silver turbine.', ('silver', 'turbine')),
    ('heldout-03', 'heldout', 'Catalog label: nickel pulley.', ('nickel', 'pulley')),
    ('heldout-04', 'heldout', 'Catalog label: ivory pulley.', ('ivory', 'pulley')),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve(strict=True)))
    from looking_glass.flan_t5_feasibility import MANIFEST_HASH, REVISION, verify_assets
    from transformers import AutoTokenizer
    import torch
    assets = args.assets.resolve(strict=True)
    verify_assets(assets)
    if MANIFEST_HASH != 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083':
        raise ValueError('unexpected asset-manifest pin')
    if REVISION != '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab':
        raise ValueError('unexpected checkpoint revision')
    tok = AutoTokenizer.from_pretrained(str(assets), local_files_only=True,
                                        trust_remote_code=False, use_fast=True)
    if not tok.is_fast or tok.pad_token_id != 0 or tok.eos_token_id != 1:
        raise ValueError('unexpected tokenizer or special tokens')
    decoder_piece = tok('record', add_special_tokens=False)['input_ids']
    if not decoder_piece or len(decoder_piece) > 3 or any(x in tok.all_special_ids for x in decoder_piece):
        raise ValueError('unsupported teacher-forced prefix')
    decoder_ids = [tok.pad_token_id] + decoder_piece
    cases = []
    for case_id, split, text, words in CASES:
        encoded = tok(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        row = list(encoded['input_ids'])
        if (not 3 <= len(row) <= 24 or row[-1] != 1 or row.count(1) != 1
                or encoded['attention_mask'] != [1] * len(row)
                or encoded['special_tokens_mask'][1] != 0):
            raise ValueError('unsupported source row: ' + case_id)
        cases.append({'case_id': case_id, 'split': split, 'text': text,
                      'words': list(words), 'input_ids': row,
                      'attention_mask': list(encoded['attention_mask']),
                      'special_tokens_mask': list(encoded['special_tokens_mask']),
                      'offset_mapping': [list(x) for x in encoded['offset_mapping']]})
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA initialized in tokenizer-only preparation')
    result = {'schema': 'flan-t5-known-injection-public-cases-v1',
              'model_id': 'google/flan-t5-small', 'model_revision': REVISION,
              'asset_manifest_sha256': MANIFEST_HASH,
              'tokenizer_json_sha256': 'fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a',
              'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'selection': 'all eight fixed proposed cases; no rejected candidates and no model call',
              'decoder_prefix': {'text': 'record', 'prefix_token_ids': decoder_piece,
                                 'decoder_input_ids': decoder_ids,
                                 'decoder_attention_mask': [1] * len(decoder_ids),
                                 'text_token_ids_with_eos': list(tok('record')['input_ids'])},
              'cases': cases}
    data = (json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(data)
    print('case_sha256=' + hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
