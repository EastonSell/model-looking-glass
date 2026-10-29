"""Reconcile closed v4 intervention custody and exact saved file inventory.

CPU-only saved-byte review; this is not a model replay, a publication-time
attestation, or proof of physical process-tree/native teardown.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work'
REPO = WORK / 'frozen/vision-recognized-v4-runner'
RUN = WORK / 'vision-recognized-v4-intervention-01'
DATA = RUN / 'data'
OUTER = WORK / 'vision-recognized-v4-intervention-outer-01'
SCREEN = WORK / 'vision-recognized-v4-screen-stage-01'
CUSTODY = WORK / 'vision-recognized-v4-cpu-custody'
REV = '9d90d90176a0903c47bce584c13c839038a88bb8'
MANIFEST = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'
STATE = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
TERMINAL = '479f1ea67efc001457427c7337313df31fa13f095ea2250ff6bfc9d99331fd1b'
SUPERVISOR = '3f96b518e38d95f9836505feea2acdf5ce2b7c7b6b72158c306f47d77754fbc3'
OBSERVER = '480978937e18aa4417e2e343fa270532a57ab8a9ac840e324d758911b0459c95'
HANDOFF = {'screen-custody.json': '74ea046ed1820b2e1fe0dd2e03ee7c0072c96b960e64f245b0bb8b8ee5ee2e24',
           'screen-evidence.zip': 'fdd48436398a9705e39b15fa410599a0d806e562e1770e54cdecd4c0babb9e81',
           'screen-observer.json': 'f51ffea425f3b25cca1052f8f6a77c40a0e7f8e250d25ee85a6067be17c59e04',
           'screen-result.json': '711894680eac6bed8e60b8acb4413775cce51f5630115e561ecbfe0504efa648'}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return path.read_bytes()


def unique_pairs(rows):
    value = {}
    for key, item in rows:
        require(key not in value, 'duplicate JSON key')
        value[key] = item
    return value


def bad_constant(value):
    raise ValueError('nonfinite JSON constant ' + value)


def finite_float(value):
    value = float(value)
    require(math.isfinite(value), 'nonfinite JSON value')
    return value


def obj(path):
    return json.loads(read(path), object_pairs_hook=unique_pairs, parse_constant=bad_constant, parse_float=finite_float)


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


source, start, terminal = obj(RUN / 'source.json'), obj(DATA / 'started.json'), obj(DATA / 'terminal.json')
screen_source = obj(WORK / 'vision-recognized-v4-screen-01/source.json')
screen_start = obj(WORK / 'vision-recognized-v4-screen-01/data/started.json')
screen_result = obj(SCREEN / 'screen-result.json')
require(source['revision'] == REV and canon(start['source']) == canon(source)
        and canon(source['sources']) == canon(screen_source['sources']) and len(source['sources']) == 23,
        'source binding')
for name, digest in source['sources'].items():
    require(sha(read(REPO / name)) == digest, 'source digest ' + name)
expected_handoff = {'results/vision-coco-recognized-v4-20261029/' + name: digest for name, digest in HANDOFF.items()}
require(canon(source['public_handoff']) == canon(expected_handoff), 'public screen bindings')
for name, digest in HANDOFF.items():
    require(sha(read(SCREEN / name)) == digest and sha(read(REPO / 'results/vision-coco-recognized-v4-20261029' / name)) == digest,
            'public handoff bytes ' + name)
manifest = REPO / 'experiments/vision_coco_recognized_v4.json'
require(sha(read(manifest)) == MANIFEST, 'manifest digest')
require(start['schema'] == 'vision-recognized-v4-start' and start['mode'] == 'intervention'
        and start['manifest_sha256'] == MANIFEST and start['model_state_sha256'] == STATE
        and canon(start['runtime']) == canon(screen_start['runtime']) and start['runtime']['device'] == 'cpu'
        and start['runtime']['cuda_initialized'] is False, 'intervention start/runtime')
require(sha(read(DATA / 'terminal.json')) == TERMINAL and terminal['schema'] == 'vision-recognized-v4-terminal'
        and terminal['status'] == 'complete' and terminal['mode'] == 'intervention' and terminal['source_revision'] == REV
        and terminal['manifest_sha256'] == MANIFEST and terminal['model_state_sha256_before_after'] == STATE
        and canon(terminal['runtime']) == canon(start['runtime'])
        and terminal['screen_result_sha256'] == HANDOFF['screen-result.json']
        and terminal['screen_archive_sha256'] == HANDOFF['screen-evidence.zip']
        and terminal['screen_terminal_sha256'] == '5aa69a7399434e775ee5b40838bba2d843c3b4d8f9b9b335fae8a2da97c28cb1'
        and type(terminal['submissions']) is int and terminal['submissions'] == 102
        and type(terminal['saved_forwards']) is int and terminal['saved_forwards'] == 102, 'intervention terminal')

screen_pairs = screen_result['pairs']
require(len(screen_pairs) == len(terminal['pairs']) == 4, 'pair count')
conditions = ('repeat', 'null_inside', 'self_inside', 'full_self', 'full_swap', 'natural_inside',
              'matched_opposite', 'matched_same_species', 'matched_random', 'final_inside', 'post_clean')
seen_images, labels = set(), []
for index, pair in enumerate(screen_pairs):
    name = f'pair-{index:02}'
    require(pair['pair_id'] == name, 'pair schedule')
    for role in ('cat', 'dog', 'cat_donor', 'dog_donor'):
        image_id = pair[role]['image_id']
        require(type(image_id) is int, 'image identity type')
        if image_id not in seen_images:
            seen_images.add(image_id)
            labels.append(f'{name}:{role}:initial')
for index in range(4):
    for side in ('cat', 'dog'):
        labels.extend(f'pair-{index:02}:{side}:{condition}' for condition in conditions)
require(len(seen_images) == 14 and len(labels) == 102 and len(set(labels)) == 102, 'reuse/forward schedule')
expected_names = {'started.json', 'terminal.json'} | {f'pair-{i:02}.json' for i in range(4)}
expected_names |= {f'forward-{i:03}.{ext}' for i in range(1, 103) for ext in ('json', 'bin')}
require(all(path.is_file() for path in DATA.iterdir()) and {path.name for path in DATA.iterdir()} == expected_names,
        'exact saved inventory')
for index, expected_label in enumerate(labels, 1):
    meta = obj(DATA / f'forward-{index:03}.json')
    raw = read(DATA / f'forward-{index:03}.bin')
    require(type(meta['submission']) is int and meta['submission'] == index and meta['label'] == expected_label
            and meta['file'] == f'forward-{index:03}.bin' and meta['dtype'] == 'little-endian-float32'
            and sha(raw) == meta['sha256'], 'forward identity/digest ' + str(index))
    intervals = []
    for segment in meta['layout'].values():
        offset, size, shape = segment['offset'], segment['bytes'], segment['shape']
        require(type(offset) is int and type(size) is int and offset >= 0 and size >= 0
                and type(shape) is list and all(type(d) is int and d >= 0 for d in shape)
                and size == 4 * math.prod(shape), 'tensor layout ' + str(index))
        intervals.append((offset, offset + size))
    intervals.sort()
    cursor = 0
    for begin, end in intervals:
        require(begin == cursor and end <= len(raw), 'tensor layout gap/overlap ' + str(index))
        cursor = end
    require(cursor == len(raw), 'tensor layout trailing bytes ' + str(index))
for index, (record, screen_pair) in enumerate(zip(terminal['pairs'], screen_pairs)):
    require(canon(obj(DATA / f'pair-{index:02}.json')) == canon(record)
            and record['pair_id'] == screen_pair['pair_id'], 'pair saved result')
    for side in ('cat', 'dog'):
        require(type(record['sides'][side]['image_id']) is int
                and record['sides'][side]['image_id'] == screen_pair[side]['image_id']
                and record['sides'][side]['same_species_donor_id'] == screen_pair[side + '_donor']['image_id'],
                'target/donor identity')
require(canon(terminal['aggregate']) == canon({'status': 'negative', 'selected': 4, 'evaluable': 4, 'positive': 0})
        and all(row['status'] == 'negative' for row in terminal['pairs']), 'retained results')
supervisor, observer = obj(RUN / 'supervisor.json'), obj(OUTER / 'receipt.json')
require(sha(read(RUN / 'supervisor.json')) == SUPERVISOR and supervisor['status'] == 'complete'
        and type(supervisor['exit_code']) is int and supervisor['exit_code'] == 0
        and supervisor['clean'] is True and supervisor['timed_out'] is False
        and supervisor['terminal_sha256'] == TERMINAL, 'supervisor receipt')
require(sha(read(OUTER / 'receipt.json')) == OBSERVER and observer['status'] == 'complete'
        and observer['schema'] == 'vision-recognized-v4-external-observer' and observer['mode'] == 'intervention'
        and observer['source_revision'] == REV and type(observer['controller_exit_code']) is int
        and observer['controller_exit_code'] == 0 and observer['controller_reaped'] is True
        and observer['timed_out'] is False and observer['forced_stop'] is False
        and observer['supervisor_sha256'] == SUPERVISOR and observer['terminal_sha256'] == TERMINAL
        and observer['stdout_sha256'] == sha(read(OUTER / 'stdout.txt'))
        and observer['stderr_sha256'] == sha(read(OUTER / 'stderr.txt')) and not read(OUTER / 'stderr.txt'),
        'external controller receipt')
claim = obj(CUSTODY / 'vision-recognized-v4-intervention.claim.json')
require(canon(claim) == canon({'schema': 'vision-recognized-v4-attempt-claim', 'mode': 'intervention',
                               'manifest_sha256': MANIFEST, 'source_revision': REV})
        and not (CUSTODY / 'cpu-study.unsafe.json').exists(), 'custody claim')
report = {'status': 'saved-intervention-custody-consistent', 'source_revision': REV,
          'manifest_sha256': MANIFEST, 'model_state_recorded_sha256': STATE, 'source_files': 23,
          'screen_handoff_sha256': HANDOFF, 'unique_images': 14, 'saved_forwards': 102,
          'saved_files': len(expected_names), 'selected_pairs': 4, 'evaluable_pairs': 4, 'positive_pairs': 0,
          'recorded_aggregate_status': 'negative', 'terminal_sha256': TERMINAL,
          'supervisor_sha256': SUPERVISOR, 'external_observer_sha256': OBSERVER,
          'limitations': ['saved-byte consistency, not model execution or publication-time attestation',
                          'cooperative direct-child waits do not prove native or process-tree cleanup',
                          'a negative aggregate is not proof of zero effect; donors were reused',
                          'separate scientific replay verifies numerical controls and decisions']}
payload = (canon(report) + '\n').encode()
output = WORK / 'vision-recognized-v4-intervention-independent-custody.json'
with output.open('xb') as stream:
    stream.write(payload)
print('intervention audit SHA-256:', sha(payload))
