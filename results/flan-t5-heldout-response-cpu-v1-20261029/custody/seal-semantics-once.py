"""Bind one completed private linguistic review; never select replacement rows."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = HERE / 'custody'
CHECKOUT = WORK / 'teams' / 'flan_t5_heldout_execution'
SOURCE = '5342de9e2df4c0c2ecef731154a369b5f8feae62'
sys.path.insert(0, str(CHECKOUT))


def main():
    from looking_glass import flan_t5_heldout_custody as custody
    from looking_glass.flan_t5_heldout_response import source_identity
    source = source_identity()
    if source['commit'] != SOURCE:
        raise ValueError('source identity mismatch')
    pool_card = custody.parse_canonical((ROOT / 'pool-commitment.json').read_bytes())
    screen = custody.parse_canonical((ROOT / 'screen.json').read_bytes())
    cases = custody.selected_cases(screen)
    verdicts = []
    for case in cases:
        # The two bare prompts do not disambiguate the distinct ordinary
        # senses of "room" as a physical interior versus available space.
        ambiguous = case['candidate_index'] == 2
        verdicts.append({'case_id': case['case_id'], 'semantic_eligible': not ambiguous,
                         'semantic_reasons': (['An English noun has distinct common interior and space senses; '
                                               'the bare prompt does not disambiguate them.'] if ambiguous else [])})
    os.replace(Path(__file__).resolve(), ROOT / 'seal-semantics-once.py')
    custody._write_new(ROOT, 'semantic-review.json', custody.canonical_json(
        {'schema': 'mlg-flant5-heldout-semantic-review-v1', 'verdicts': verdicts}))
    card = custody.seal_cases_attempt(ROOT, pool_card, screen, verdicts)
    if source_identity() != source:
        raise ValueError('source changed during semantic review')
    inventory = custody.inventory_custody(ROOT)
    print(json.dumps({'status': 'case-committed-locally',
                      'case_commitment': card['commitment'],
                      'public_card_sha256': hashlib.sha256(custody.canonical_json(card)).hexdigest(),
                      'semantic_eligible_cells': sum(v['semantic_eligible'] for v in verdicts),
                      'semantic_ineligible_cells': sum(not v['semantic_eligible'] for v in verdicts),
                      'custody_bytes': inventory['bytes']}, sort_keys=True), flush=True)


try:
    main()
except BaseException:
    try:
        with (ROOT / 'semantic-supervisor-failed.json').open('xb') as stream:
            stream.write(b'{"phase":"semantic-supervisor","status":"incomplete"}')
    except BaseException:
        pass
    print('Semantic review is incomplete; retain the protected directory.', file=sys.stderr)
    raise SystemExit(1) from None
