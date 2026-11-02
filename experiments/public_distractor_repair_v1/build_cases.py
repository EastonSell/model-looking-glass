"""Construct fixed public cases using the pinned tokenizer only."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path
os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

PROTOCOL_SHA = '6d87a579eb27f214db3e8769232b5940d263da59490480765118174ef99b5d70'
PROTOCOL_COMMIT = '29698dc5319e8c578c3e9f87ca1bb5377b9fa270'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'


def main(protocol_path, cache_dir, destination):
    raw = Path(protocol_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA:
        raise ValueError('frozen protocol bytes differ')
    texts = re.findall(r'^\d+\. (.+)$', raw.decode('utf-8'), re.MULTILINE)
    if len(texts) != 16 or len(set(texts)) != 16:
        raise ValueError('expected sixteen unique fixed texts')
    cache = str(Path(cache_dir).resolve(strict=True))
    path = Path(hf_hub_download(repo_id='distilbert/distilgpt2', filename='tokenizer.json',
        revision='2290a62682d06624634c1f46a6ad5be0f47f38aa', cache_dir=cache,
        local_files_only=True, token=False))
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA:
        raise ValueError('pinned tokenizer checksum differs')
    rust = Tokenizer.from_file(str(path))
    fast = PreTrainedTokenizerFast(tokenizer_file=str(path))
    cases = []
    for index, text in enumerate(texts):
        full = rust.encode(text, add_special_tokens=False).ids
        prefix = rust.encode(text[:-1], add_special_tokens=False).ids
        if (not text.endswith('.') or not 3 <= len(full) <= 32 or full != prefix + [13]
                or fast.encode(text, add_special_tokens=False) != full
                or fast.encode(text[:-1], add_special_tokens=False) != prefix
                or rust.decode(full, skip_special_tokens=False) != text):
            raise ValueError('input boundary, round trip or tokenizer agreement failed')
        for token, suffix in ((318, ' is'), (389, ' are')):
            expected = prefix + [token]
            if (rust.encode(text[:-1] + suffix, add_special_tokens=False).ids != expected
                    or fast.encode(text[:-1] + suffix, add_special_tokens=False) != expected):
                raise ValueError('scored-prefix continuation is not aligned')
        cases.append({'case_id': f'distractor-repair-{index + 1:02d}',
                      'split': 'discovery' if index < 8 else 'held-out',
                      'public_text': text, 'token_ids': full,
                      'scored_prefix_token_ids': prefix, 'candidate_token_ids': [318, 389],
                      'scored_position': len(full) - 2, 'future_position': len(full) - 1})
    if 'torch' in sys.modules:
        raise RuntimeError('tokenizer-only construction unexpectedly imported torch')
    data = (json.dumps({'schema':'public-distractor-repair-cases-v1',
                       'protocol_sha256': PROTOCOL_SHA, 'protocol_commit': PROTOCOL_COMMIT,
                       'tokenizer_sha256': TOKENIZER_SHA, 'model_forward': False,
                       'cases': cases}, indent=2, ensure_ascii=True) + '\n').encode('utf-8')
    with Path(destination).open('xb') as out:
        out.write(data)
    print(json.dumps({'sha256':hashlib.sha256(data).hexdigest(), 'cases':len(cases),
                      'lengths':[len(x['token_ids']) for x in cases], 'torch_imported':False},sort_keys=True))


if __name__ == '__main__':
    main(*sys.argv[1:])
