"""Build the fixed public rows with the pinned tokenizer only; no model call."""
import hashlib
import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


PROTOCOL_SHA = '9eeca112d06f998037bea9d5a7a6b7097307f76634565441c62b24e5f2855cfd'
TOKENIZER_SHA = '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'


def main(protocol_path, cache_dir, destination):
    raw = Path(protocol_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA:
        raise ValueError('protocol bytes do not match the independently reviewed revision')
    texts = re.findall(r'^\d+\. `([^`]+)`$', raw.decode('utf-8'), re.MULTILINE)
    if len(texts) != 16 or len(set(texts)) != 16:
        raise ValueError('protocol did not contain exactly the fixed 16 prompts')
    tokenizer_path = Path(hf_hub_download(repo_id='distilbert/distilgpt2',
        filename='tokenizer.json', revision='2290a62682d06624634c1f46a6ad5be0f47f38aa',
        cache_dir=cache_dir, local_files_only=True, token=False))
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA:
        raise ValueError('pinned tokenizer checksum mismatch')
    rust = Tokenizer.from_file(str(tokenizer_path))
    fast = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    suffixes = {318: ' is', 389: ' are'}
    cases = []
    for index, text in enumerate(texts):
        ids = rust.encode(text, add_special_tokens=False).ids
        if not 1 <= len(ids) <= 32 or ids != fast.encode(text, add_special_tokens=False):
            raise ValueError('tokenizer disagreement or input outside fixed length cap')
        if rust.decode(ids, skip_special_tokens=False) != text:
            raise ValueError('tokenizer round-trip mismatch')
        for token, suffix in suffixes.items():
            if (rust.encode(text + suffix, add_special_tokens=False).ids != ids + [token]
                    or fast.encode(text + suffix, add_special_tokens=False) != ids + [token]):
                raise ValueError('fixed score token does not align at the prompt boundary')
        cases.append({'case_id': f'known-injection-{index + 1:02d}',
                      'split': 'discovery' if index < 8 else 'held-out',
                      'public_text': text, 'token_ids': ids,
                      'candidate_token_ids': [318, 389]})
    payload = {'schema': 'public-known-injection-cases-v1',
               'protocol_sha256': PROTOCOL_SHA,
               'protocol_commit': 'c353969d02d767371cfafb3dc9b1327c8ea8e4b0',
               'tokenizer_sha256': TOKENIZER_SHA,
               'model_forward': False, 'cases': cases}
    data = (json.dumps(payload, indent=2, ensure_ascii=True) + '\n').encode('utf-8')
    path = Path(destination)
    with path.open('xb') as out:
        out.write(data)
    print(json.dumps({'case_sha256': hashlib.sha256(data).hexdigest(),
                      'count': len(cases), 'lengths': [len(case['token_ids']) for case in cases],
                      'model_forward': False}, sort_keys=True))


if __name__ == '__main__':
    main(*sys.argv[1:])
