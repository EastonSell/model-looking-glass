"""One tokenizer-only screen following observed publication of the pool card."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = HERE / 'custody'
CHECKOUT = WORK / 'teams' / 'flan_t5_heldout_execution'
ASSETS = WORK / 't5-cache' / 'flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
SOURCE = '5342de9e2df4c0c2ecef731154a369b5f8feae62'
OBS_HASH = 'ab66b905c82e335588bed7aebe25b224beb2ce3cce9d447d3f9f493dcde2e5ee'
sys.path.insert(0, str(CHECKOUT))


def main():
    from transformers import AutoTokenizer
    from looking_glass import flan_t5_heldout_custody as custody
    from looking_glass.flan_t5_heldout_response import source_identity
    from looking_glass.flan_t5_feasibility import verify_assets
    source = source_identity()
    if source['commit'] != SOURCE:
        raise ValueError('source identity mismatch')
    observation_path = WORK / 'flan-t5-heldout-pool-observation-6a002698.json'
    raw = observation_path.read_bytes()
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != OBS_HASH:
        raise ValueError('pool readback observation mismatch')
    observation = json.loads(raw)
    verify_assets(ASSETS)
    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS), local_files_only=True,
                                               use_fast=True, trust_remote_code=False)
    exposure = custody.load_exposure(CHECKOUT, SOURCE, tokenizer)
    card = custody.parse_canonical((ROOT / 'pool-commitment.json').read_bytes())
    os.replace(Path(__file__).resolve(), ROOT / 'screen-once.py')
    screen = custody.screen_attempt(ROOT, card, observation, tokenizer, exposure)
    if source_identity() != source:
        raise ValueError('source changed during screening')
    inventory = custody.inventory_custody(ROOT)
    print(json.dumps({'status': screen['status'], 'candidate_count': len(screen['candidates']),
                      'selected_count': len(screen['selected_candidate_indices']),
                      'custody_bytes': inventory['bytes']}, sort_keys=True), flush=True)


try:
    main()
except BaseException:
    try:
        path = ROOT / 'screen-supervisor-failed.json'
        with path.open('xb') as stream:
            stream.write(b'{"phase":"screen-supervisor","status":"incomplete"}')
    except BaseException:
        pass
    print('Screening is incomplete; retain the protected directory.', file=sys.stderr)
    raise SystemExit(1) from None
