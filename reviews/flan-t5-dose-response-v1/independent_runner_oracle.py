"""Independent synthetic decision and scalar dose audit for a frozen runner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkout', required=True, type=Path)
parser.add_argument('--commit', required=True)
parser.add_argument('--source-sha256', required=True)
parser.add_argument('--helper-sha256', required=True)
parser.add_argument('--fixtures', type=Path, default=Path(__file__).with_name('decision-fixtures.json'))
args = parser.parse_args()
source = args.checkout / 'looking_glass/flan_t5_dose_response.py'
helper = args.checkout / 'looking_glass/flan_t5_equal_dose.py'
before = hashlib.sha256(source.read_bytes()).hexdigest()
helper_before = hashlib.sha256(helper.read_bytes()).hexdigest()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.checkout, text=True).strip()
if before != args.source_sha256 or helper_before != args.helper_sha256 or head != args.commit:
    raise ValueError('frozen runner source or commit mismatch')
for path in (source, helper):
    relative = path.relative_to(args.checkout).as_posix()
    if path.read_bytes() != subprocess.check_output(['git', 'show', f'{head}:{relative}'], cwd=args.checkout):
        raise ValueError('working source differs from Git blob')
sys.path.insert(0, str(args.checkout.resolve()))
from looking_glass.flan_t5_dose_response import aggregate_decision, match_response_levels, response_decision
import torch

torch.set_num_threads(1)
assert torch.set_flush_denormal(False)
data = json.loads(args.fixtures.read_bytes())

def category(reason):
    return reason.split('-side')[0]

for case in data['cells']:
    movement = {site: [[case[site][side][level] for side in range(2)] for level in range(3)]
                for site in ('word', 'instruction', 'punctuation')}
    status, reasons = response_decision(case['delta'], movement)
    assert status == case['expected_status'], (case['name'], status, case['expected_status'])
    assert sorted({category(x) for x in reasons}) == case['expected_flags'], (case['name'], reasons)

# A binary64 subtraction boundary where moving the tolerance to the other side
# of an inequality changes the answer. The exact protocol comparison must win.
previous = -4.2070601430730603e-13
word = -0.00010000000042070602
assert not word < previous - 1e-4 and word + 1e-4 < previous
curve = {'word': [[previous, 0.3], [word, 0.6], [0.5, 1.0]],
         'instruction': [[0.0, 0.0] for _ in range(3)],
         'punctuation': [[0.0, 0.0] for _ in range(3)]}
status, reasons = response_decision(10.0, curve)
assert status == 'negative' and not any(x.startswith('nonmonotone-side0-level2') for x in reasons)

for case in data['aggregates']:
    rows = [{'pair_id': pair, 'template_id': ('translate', 'question')[idx], 'status': status}
            for pair, statuses in enumerate(case['pairs']) for idx, status in enumerate(statuses)]
    assert aggregate_decision(rows) == case['expected'], case['name']

positions = {'word': 6, 'instruction': 0, 'punctuation': 5}
randomizer = random.Random(102904)
for trial in range(100):
    states = [torch.zeros((1, 9, 512), dtype=torch.float32) for _ in range(2)]
    for site, pos in positions.items():
        for side in range(2):
            states[side][0, pos] = torch.tensor([randomizer.uniform(-5.0, 5.0) for _ in range(512)])
    dose, delivered = match_response_levels(states, positions)
    assert dose['status'] == 'matchable', trial
    distances = {}
    for site, pos in positions.items():
        a = states[0][0, pos].tolist()
        b = states[1][0, pos].tolist()
        distances[site] = math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b)))
        assert math.isclose(distances[site], dose['distances'][site], rel_tol=1e-12, abs_tol=1e-12)
    q = min(distances.values())
    assert math.isclose(q, dose['q'], rel_tol=1e-12, abs_tol=1e-12)
    for (num, den), level in zip(((1, 3), (2, 3), (1, 1)), dose['levels']):
        amount = q*(num/den)
        assert math.isclose(level['q'], amount, rel_tol=1e-12, abs_tol=1e-12) and level['status'] == 'matchable'
        for site, pos in positions.items():
            for side in (0, 1):
                own = states[side][0, pos].tolist()
                donor = states[1-side][0, pos].tolist()
                expected = b''.join(struct.pack('<f', a+(amount/distances[site])*(b-a))
                                    for a, b in zip(own, donor))
                actual = delivered[num, den, site, side].numpy().astype('<f4', copy=False).tobytes()
                assert actual == expected, (trial, num, den, site, side)
                actual_values = struct.unpack('<512f', actual)
                norm = math.sqrt(math.fsum((x-y)**2 for x, y in zip(actual_values, own)))
                assert math.isclose(norm, level['achieved'][f'{site}-{side}'], rel_tol=1e-12, abs_tol=1e-12)

empty = [torch.zeros((1, 9, 512), dtype=torch.float32) for _ in range(2)]
assert match_response_levels(empty, positions)[0]['status'] == 'dose-ineligible'
rounding = [torch.zeros((1, 9, 512), dtype=torch.float32) for _ in range(2)]
rounding[1][0, 6, 0] = 2e-5
for pos, amount in ((0, 0.01), (5, 0.02)):
    rounding[0][0, pos, 0] = 100.0
    rounding[1][0, pos, 0] = 100.0 + amount
assert match_response_levels(rounding, positions)[0]['status'] == 'dose-unmatchable'

after = hashlib.sha256(source.read_bytes()).hexdigest()
head_after = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.checkout, text=True).strip()
assert after == before and head_after == head
assert hashlib.sha256(helper.read_bytes()).hexdigest() == helper_before
print(json.dumps({'status': 'passed', 'source_sha256': before, 'commit': head,
                  'helper_sha256': helper_before,
                  'cell_fixtures': len(data['cells']), 'aggregate_fixtures': len(data['aggregates']),
                  'additional_binary64_boundary': True, 'scalar_dose_trials': 100,
                  'ineligible_and_rounding_unmatchable': True,
                  'scope': 'fabricated CPU inputs; no pretrained model or study output'}))
