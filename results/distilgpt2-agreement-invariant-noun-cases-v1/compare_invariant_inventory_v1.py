"""Compare the fixed public invariant cases to an independently pinned inventory."""

import argparse
import hashlib
import json
from pathlib import Path
import types

PREPARER_SHA = '983eb7cac32f8792fc61cabae3539b63d05e3907768c51c5bac89fb6cc23335e'
CASES_SHA = '50ef3aea53a21a76ecf1bc4910882f78939a4d2fcca12528013ea56eaa586827'


def read(path, limit, expected):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('input size or SHA-256 differs')
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparer', 'cases', 'inventory', 'inventory-sha256', 'inventory-version',
                 'inventory-revision', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    raw = read(args.preparer, 256 * 1024, PREPARER_SHA)
    prep = types.ModuleType('_verified_invariant_preparer')
    exec(compile(raw, args.preparer, 'exec', dont_inherit=True), prep.__dict__)
    cases = json.loads(read(args.cases, 4 * 1024 * 1024, CASES_SHA))
    raw = read(args.inventory, 32 * 1024 * 1024, args.inventory_sha256)
    inventory = prep.verify_inventory(raw, args.inventory_version)
    groups = prep.make_groups(cases['cases'])
    proposed = [x for group in groups for x in group]
    other = prep.inventory_sequences(inventory)
    collisions = sum(prep.related(a, b) for a in proposed for b in other)
    receipt = {'schema': 'agreement-invariant-public-inventory-comparison-v1',
               'case_sha256': CASES_SHA, 'inventory_sha256': args.inventory_sha256,
               'inventory_revision': args.inventory_revision, 'inventory_version': args.inventory_version,
               'proposed_cases': len(groups), 'proposed_sequences': len(proposed),
               'inventory_records': len(inventory['records']), 'distinct_inventory_sequences': len(other),
               'bidirectional_sequence_pairs': len(proposed) * len(other), 'relations': collisions,
               'own_base_answer_prefix_exempt': True,
               'scope': 'Public inventory comparison; not a private or unknown-exposure attestation.'}
    if collisions:
        raise ValueError('public input prefix collision')
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with Path(args.output).open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), **receipt}, sort_keys=True))


if __name__ == '__main__':
    main()
