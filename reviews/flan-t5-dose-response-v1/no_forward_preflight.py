"""Verify committed source, pinned assets and all fixed tokenizer rows without a model forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkout', required=True, type=Path)
parser.add_argument('--assets', required=True, type=Path)
parser.add_argument('--commit', required=True)
args = parser.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false')
sys.path.insert(0, str(args.checkout.resolve()))
from looking_glass import flan_t5_dose_response as study
from transformers import AutoTokenizer

before = study.source_identity()
if before['commit'] != args.commit:
    raise ValueError('checkout commit differs')
assets = study.verify_assets(args.assets)
raw = study.CASES.read_bytes()
if hashlib.sha256(raw).hexdigest() != study.CASE_HASH:
    raise ValueError('case manifest hash differs')
manifest = json.loads(raw)
tokenizer = AutoTokenizer.from_pretrained(str(args.assets), local_files_only=True,
                                          use_fast=True, trust_remote_code=False)
eligibility = study.check_cases(tokenizer, manifest)
config = json.loads((args.assets / 'config.json').read_bytes())
if any(config.get(k) != v for k, v in {'decoder_start_token_id': 0, 'pad_token_id': 0,
                                        'eos_token_id': 1, 'd_model': 512}.items()):
    raise ValueError('pinned model config differs')
if study.source_identity() != before:
    raise ValueError('source changed during preflight')
print(json.dumps({'status': 'passed', 'scope': 'no model loaded and no model forward',
                  'source_commit': before['commit'], 'source_hashes': before['source_sha256'],
                  'asset_manifest_sha256': study.MANIFEST_HASH, 'verified_asset_count': len(assets['files']),
                  'case_sha256': study.CASE_HASH, 'cell_eligibility_reasons': eligibility,
                  'tokenizer_class': type(tokenizer).__name__}))
