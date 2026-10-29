"""Exercise the equal-length full path with a fabricated CPU model and tokenizer."""
import hashlib
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--commit', required=True)
parser.add_argument('--source-sha256', required=True)
parser.add_argument('--checkout', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--mixed', action='store_true')
args = parser.parse_args()
EXPECTED_COMMIT, EXPECTED_SOURCE = args.commit, args.source_sha256
CHECKOUT = args.checkout.resolve()
source = CHECKOUT / 'looking_glass/flan_t5_equal_length_response.py'
def checked_source():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CHECKOUT, text=True).strip() == EXPECTED_COMMIT
    identity = study.source_identity()
    assert identity['commit'] == EXPECTED_COMMIT
    hashes = {}
    for name, declared_hash in identity['source_sha256'].items():
        raw = (CHECKOUT / name).read_bytes()
        assert raw == subprocess.check_output(['git', 'show', f'{EXPECTED_COMMIT}:{name}'], cwd=CHECKOUT)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        assert hashes[name] == declared_hash
    assert hashes['looking_glass/flan_t5_equal_length_response.py'] == EXPECTED_SOURCE
    return hashes
sys.path.insert(0, str(CHECKOUT))
os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  MKL_CBWR='COMPATIBLE', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import torch
import transformers
from looking_glass import flan_t5_equal_length_response as study
source_hashes = checked_source()


def main():
    cases = json.loads(study.CASES.read_bytes())
    rows = [row for c in cases['cases'] for row in [c['neutral'], *c['source'], *c['targets']]]
    by_text = {row['text']: row for row in rows}
    by_ids = {tuple(row['input_ids']): row['token_strings'] for row in rows}
    signs = {c['source'][side]['input_ids'][6]:
             (1 if side == 0 or (args.mixed and c['pair_id'] in (0, 5)) else -1)
             for c in cases['cases'] for side in (0, 1)}
    instruction_sensitive = {c['source'][side]['input_ids'][6]
                             for c in cases['cases'] if c['pair_id'] in (1, 3) for side in (0, 1)}
    target_a = {c['targets'][0]['input_ids'][0] for c in cases['cases']}
    target_b = {c['targets'][1]['input_ids'][0] for c in cases['cases']}
    assert not target_a.intersection(target_b)
    assert len(signs) == 12

    class T5TokenizerFast:
        pad_token_id = 0
        eos_token_id = 1
        unk_token_id = 2
        def __call__(self, text, **kwargs):
            row = by_text[text]
            return {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
                    'special_tokens_mask': row['special_tokens_mask'],
                    'offset_mapping': [tuple(x) for x in row['offsets']]}
        def convert_ids_to_tokens(self, ids):
            return by_ids[tuple(ids)]

    class T5ForConditionalGeneration(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.encoder = torch.nn.Module()
            self.encoder.final_layer_norm = torch.nn.Identity()
            self.config = SimpleNamespace(d_model=512, vocab_size=32128, decoder_start_token_id=0,
                                          eos_token_id=1, pad_token_id=0, use_cache=False,
                                          _attn_implementation='eager')
        def forward(self, input_ids, attention_mask, decoder_input_ids, **kwargs):
            state = torch.zeros((1, input_ids.shape[1], 512), dtype=torch.float32)
            sign = signs.get(int(input_ids[0, 6]), 0)
            state[0, 6, 0] = sign
            state[0, 0, 0] = sign * .5
            if input_ids.shape[1] == 8:
                state[0, 5, 0] = sign * .25
            elif input_ids.shape[1] == 9:
                state[0, 7, 0] = sign * .25
            state = self.encoder.final_layer_norm(state)
            token = int(input_ids[0, 6])
            coefficient = 2.0 if args.mixed and token in instruction_sensitive else 0.0
            signal = state[0, 6, 0] + coefficient * state[0, 0, 0]
            logits = torch.full((1, decoder_input_ids.shape[1], 32128), -2.0)
            for token in target_a:
                logits[0, 0, token] = signal * 20
            for token in target_b:
                logits[0, 0, token] = -signal * 20
            return SimpleNamespace(logits=logits, encoder_last_hidden_state=state, past_key_values=None)

    real_t5 = transformers.T5ForConditionalGeneration
    output = args.output.resolve()
    assert output.parent.is_dir() and not output.exists()
    with patch.object(study, 'source_identity', return_value={'commit': 'synthetic', 'source_sha256': {}}), \
         patch.object(study, 'check_publication_observation', return_value=b'synthetic-publication'), \
         patch.object(study, 'verify_assets', return_value={'synthetic': True}), \
         patch.object(study, 'sys', SimpleNamespace(modules={})), \
         patch.object(transformers.AutoTokenizer, 'from_pretrained', return_value=T5TokenizerFast()), \
         patch.object(real_t5, 'from_pretrained', return_value=(T5ForConditionalGeneration(), {})):
        result = study.run(output.parent, output, output.parent / 'synthetic-publication.json')
    assert result['status'] == 'complete' and result['decision'] == ('inconclusive' if args.mixed else 'positive')
    assert result['forwards'] == (648 if args.mixed else 936) and len(result['cases']) == 12
    offset = 0
    for index, case in enumerate(result['cases']):
        if args.mixed and case['pair_id'] in (0, 5):
            assert case['status'] == 'baseline-unsupported'
            offset += 6
            continue
        expected_status = 'negative' if args.mixed and case['pair_id'] in (1, 3) else 'positive'
        assert case['status'] == expected_status and case['dose']['status'] == 'matchable'
        if expected_status == 'negative':
            assert any(x.startswith('control_comparison') for x in case['reasons'])
        assert case['dose']['q'] == 0.5
        for level, fraction in zip(case['dose']['levels'], (1/3, 2/3, 1.0)):
            assert level['status'] == 'matchable' and level['q'] == .5*fraction
            assert all(abs(value-level['q']) <= level['tolerance'] for value in level['achieved'].values())
        for site in (('punctuation',) if expected_status == 'negative' else ('instruction', 'punctuation')):
            assert all(abs(v) < 1e-12 for pair in case['response_curves'][site] for v in pair)
        for side in range(2):
            curve = [pair[side] for pair in case['response_curves']['word']]
            assert len(curve) == 4 and curve[0] == 0 and 0 < curve[1] < curve[2] < curve[3]
        end = offset+78
        for pos in range(end-4, end):
            row = json.loads((output / f'forward-{pos:03d}.json').read_bytes())
            assert row['name'].startswith(case['case_id']+'-repeat-') and row['action'] == 'capture'
        offset = end
    assert offset == result['forwards']
    assert checked_source() == source_hashes
    total = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    print(json.dumps({'status': 'passed', 'scope': 'fabricated model and tokenizer; not pretrained execution',
                      'forwards': result['forwards'], 'bytes': total,
                      'completed_sha256': hashlib.sha256((output/'completed.json').read_bytes()).hexdigest(),
                      'implementation_commit': EXPECTED_COMMIT}))


if __name__ == '__main__':
    main()
