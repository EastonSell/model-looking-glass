"""Recompute the recorded float32 log-softmax on saved logits; no model is loaded."""
import hashlib
import json
import math
from pathlib import Path
import torch

work = Path(__file__).resolve().parent
attempt = work/'t5-results/flan-t5-dose-response-cpu-v1-20261029'
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
token_count = 0
for index in range(36):
    record = json.loads((attempt/f'forward-{index:03}.json').read_bytes())
    data = (attempt/record['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['sha256']
    spec = record['tensor_layout']['logits']
    logits = torch.frombuffer(bytearray(data[spec['offset']:spec['offset']+spec['bytes']]),
                              dtype=torch.float32).reshape(spec['shape'])
    selected = torch.log_softmax(logits[0], dim=-1)[torch.arange(len(record['target_ids'])),
                                                    torch.tensor(record['target_ids'], dtype=torch.long)]
    values = [float(value) for value in selected]
    assert values == record['selected_logprobs']
    assert math.fsum(values) == record['score']
    token_count += len(values)
assert not torch.cuda.is_initialized()
print(json.dumps({'forward_records': 36, 'token_scores': token_count, 'bit_exact': True,
                  'model_loaded': False, 'torch': torch.__version__}, sort_keys=True))
