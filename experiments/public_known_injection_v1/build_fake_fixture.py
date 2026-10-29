"""Generate fabricated local test evidence with a tiny random GPT-2 only."""
from pathlib import Path
import json
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from executor import execute_phase
from analyze import discovery, heldout, load_phase


def main(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    config = GPT2Config(vocab_size=41, n_positions=16, n_embd=8, n_layer=6, n_head=2,
                        use_cache=False, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
    config._attn_implementation = 'eager'
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = GPT2LMHeadModel(config).eval()
    cases = [{'case_id': f'known-injection-{index+1:02d}',
              'split': 'discovery' if index < 8 else 'held-out',
              'token_ids': [1, 7, 11, 21, 17 + index]} for index in range(16)]
    (root / 'cases.json').write_text(json.dumps(cases), encoding='utf-8')
    execute_phase(model, cases[:8], root / 'discovery', phase='discovery', vocab_size=41)
    records, evidence = load_phase(root / 'discovery', cases[:8], vocab_size=41, hidden=8)
    frozen = discovery(records, cases[:8], score_ids=(3, 4))
    (root / 'fake-discovery-freeze.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    execute_phase(model, cases[8:], root / 'held-out', phase='held-out', vocab_size=41)
    heldout_records, heldout_evidence = load_phase(root / 'held-out', cases[8:], vocab_size=41, hidden=8)
    result = {'discovery': frozen, 'held-out': heldout(heldout_records, cases[8:], frozen, score_ids=(3, 4)),
              'evidence': {'discovery': evidence, 'held-out': heldout_evidence},
              'scope': 'fabricated random model, no pretrained evidence'}
    (root / 'fake-result.json').write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'actions': evidence['actions'] + heldout_evidence['actions'],
                      'raw_bytes': evidence['raw_bytes'] + heldout_evidence['raw_bytes'],
                      'scope': result['scope']}, sort_keys=True))


if __name__ == '__main__':
    main(sys.argv[1])
