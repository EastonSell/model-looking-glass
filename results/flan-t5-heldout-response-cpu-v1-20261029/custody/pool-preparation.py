"""Single protected custody preparation; no held-out tokenization or model."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
CHECKOUT = WORK / 'teams' / 'flan_t5_heldout_execution'
ASSETS = WORK / 't5-cache' / 'flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
SOURCE = '5342de9e2df4c0c2ecef731154a369b5f8feae62'
PROTOCOL = '29bb1d80c6e7d81197489b00fdeb942ff2fe95315110c7d0a04838fcebc2773b'
MANIFEST = 'd369e3095dc35c52608f40fbd2715d7a2ff80a02fd71f0692c24af2cf3402083'
REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
OBS_HASH = 'd6173be0f4d532df090929dc922e325e6d69c1004cf2c1bbe6dc5fcdfaf1ce3b'
sys.path.insert(0, str(CHECKOUT))


def main():
    from transformers import AutoTokenizer
    from looking_glass import flan_t5_heldout_custody as custody
    from looking_glass.flan_t5_heldout_response import source_identity
    from looking_glass.flan_t5_equal_dose import check_publication_observation
    from looking_glass.flan_t5_feasibility import verify_assets
    source = source_identity()
    if source['commit'] != SOURCE or source['source_sha256']['docs/FLAN_T5_HELDOUT_RESPONSE_V1.md'] != PROTOCOL:
        raise ValueError('source identity mismatch')
    observation_path = WORK / 'flan-t5-heldout-public-observation-5342de9e.json'
    observation = check_publication_observation(observation_path, source)
    if hashlib.sha256(observation).hexdigest() != OBS_HASH:
        raise ValueError('source observation mismatch')
    verify_assets(ASSETS)
    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS), local_files_only=True,
                                               use_fast=True, trust_remote_code=False)
    exposure = custody.load_exposure(CHECKOUT, SOURCE, tokenizer)
    # These choices are fixed before tokenizing any of the withheld rows.
    pairs = [
        (('pen', 'pencil'), ('Stift', 'Bleistift')),
        (('door', 'window'), ('Tür', 'Fenster')),
        (('bed', 'room'), ('Bett', 'Zimmer')),
        (('hand', 'foot'), ('Hand', 'Fuß')),
        (('head', 'eye'), ('Kopf', 'Auge')),
        (('ear', 'nose'), ('Ohr', 'Nase')),
        (('mouth', 'heart'), ('Mund', 'Herz')),
        (('school', 'church'), ('Schule', 'Kirche')),
        (('city', 'village'), ('Stadt', 'Dorf')),
        (('garden', 'kitchen'), ('Garten', 'Küche')),
        (('wall', 'floor'), ('Wand', 'Boden')),
        (('paper', 'stone'), ('Papier', 'Stein')),
        (('silver', 'air'), ('Silber', 'Luft')),
        (('earth', 'sea'), ('Erde', 'Meer')),
        (('lake', 'boat'), ('See', 'Boot')),
        (('ship', 'truck'), ('Schiff', 'Lastwagen')),
        (('wheel', 'ball'), ('Rad', 'Ball')),
        (('game', 'music'), ('Spiel', 'Musik')),
        (('world', 'time'), ('Welt', 'Zeit')),
        (('uncle', 'aunt'), ('Onkel', 'Tante')),
        (('hat', 'coat'), ('Hut', 'Mantel')),
        (('fork', 'knife'), ('Gabel', 'Messer')),
        (('spoon', 'plate'), ('Löffel', 'Teller')),
        (('bottle', 'cup'), ('Flasche', 'Tasse')),
    ]
    pool = {'schema': 'mlg-flant5-heldout-pool-v1', 'source_commit': SOURCE,
            'protocol_sha256': PROTOCOL, 'asset_manifest_sha256': MANIFEST,
            'tokenizer_revision': REVISION, 'exposure_inputs': exposure['inputs'],
            'templates': [{'template_id': 'template0', 'prefix': 'Provide the German noun for ', 'suffix': '.'},
                          {'template_id': 'template1', 'prefix': 'How do you say ', 'suffix': ' in German?'}],
            'pairs': [{'candidate_index': i, 'source_words': list(words), 'target_words': list(targets)}
                      for i, (words, targets) in enumerate(pairs)]}
    root = HERE / 'custody'
    card = custody.create_pool_attempt(root, pool, exposure)
    custody._write_new(root, 'source-observation.json', observation)
    os.replace(Path(__file__).resolve(), root / 'pool-preparation.py')
    if source_identity() != source:
        raise ValueError('source changed during preparation')
    inventory = custody.inventory_custody(root)
    print(json.dumps({'status': 'pool-committed-locally', 'pool_commitment': card['commitment'],
                      'public_card_sha256': hashlib.sha256(custody.canonical_json(card)).hexdigest(),
                      'custody_bytes': inventory['bytes']}, sort_keys=True), flush=True)


try:
    main()
except BaseException as error:
    try:
        failure = {'status': 'incomplete', 'phase': 'pool-preparation',
                   'error_type': type(error).__name__}
        with (HERE / 'pool-preparation-failed.json').open('xb') as stream:
            stream.write(json.dumps(failure, sort_keys=True).encode('utf-8'))
    except BaseException:
        pass
    print('Pool preparation is incomplete; retain the protected directory.', file=sys.stderr)
    raise SystemExit(1) from None
