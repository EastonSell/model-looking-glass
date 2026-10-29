"""Independent, CPU-only reconciliation of the closed public v4 screen.

This verifies saved-byte consistency, not that a model executed, publication
chronology, process-tree cleanup, image origin, or pretraining independence.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work'
REPO = WORK / 'frozen/vision-recognized-v4-runner'
RUN = WORK / 'vision-recognized-v4-screen-01'
DATA = RUN / 'data'
OUTER = WORK / 'vision-recognized-v4-screen-outer-01'
STAGE = WORK / 'vision-recognized-v4-screen-stage-01'
CUSTODY = WORK / 'vision-recognized-v4-cpu-custody'
PREFLIGHT = WORK / 'vision-recognized-v4-preflight-01'
ASSETS = WORK / 'vision-coco-assets'
REV = 'f5769fb7fa6f4672bb3eb8edeacfc55917704655'
MANIFEST_SHA = '21140888136a4cee670e89271c993ff2ce484b348c6f474e1456739bfd5355bb'
STATE_SHA = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
RESULT_SHA = '711894680eac6bed8e60b8acb4413775cce51f5630115e561ecbfe0504efa648'
ARCHIVE_SHA = 'fdd48436398a9705e39b15fa410599a0d806e562e1770e54cdecd4c0babb9e81'
TERMINAL_SHA = '5aa69a7399434e775ee5b40838bba2d843c3b4d8f9b9b335fae8a2da97c28cb1'
SUPERVISOR_SHA = 'd68948683062e035d8c6f76f1f7dad6f35f62e0af5718fcef2dc20da3d6919f1'
OBSERVER_SHA = 'f51ffea425f3b25cca1052f8f6a77c40a0e7f8e250d25ee85a6067be17c59e04'
REMOTE_BLOBS = {
    'screen-custody.json': '2d8ed24e1346d6a056ca1e94b16cf93940235e1e',
    'screen-evidence.zip': 'a18e78a402c7b2b6c48816343fdbc96ce4b65685',
    'screen-independent-audit.json': '233cf62cad8b36d68160fa20564e6afe7230f99c',
    'screen-observer.json': '82948ae6573db5f47a158695a873f59364fd83ae',
    'screen-result.json': '4a4a56cdd03238f108ebbfed3664b8b301f20450',
}


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()


def read(path):
    return path.read_bytes()


def pairs(rows):
    result = {}
    for key, value in rows:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError('nonfinite JSON constant ' + value)


def obj(path):
    def finite_float(value):
        value = float(value)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(read(path), object_pairs_hook=pairs, parse_constant=bad_constant, parse_float=finite_float)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


source = obj(RUN / 'source.json')
start = obj(DATA / 'started.json')
terminal = obj(DATA / 'terminal.json')
prior_source = obj(PREFLIGHT / 'source.json')
prior_start = obj(PREFLIGHT / 'data/started.json')
manifest_raw = read(REPO / 'experiments/vision_coco_recognized_v4.json')
manifest = obj(REPO / 'experiments/vision_coco_recognized_v4.json')
require(sha(manifest_raw) == MANIFEST_SHA, 'manifest hash')
require(canonical(source) == canonical(prior_source) == canonical(start['source']), 'source/preflight binding')
require(source['revision'] == REV and len(source['sources']) == 23, 'source inventory')
for name, digest in source['sources'].items():
    require(sha(read(REPO / name)) == digest, 'source file hash: ' + name)
for name, digest in manifest['model_files'].items():
    require(sha(read(ASSETS / name)) == digest, 'model asset hash: ' + name)
require(start['schema'] == 'vision-recognized-v4-start' and start['mode'] == 'screen'
        and start['manifest_sha256'] == MANIFEST_SHA and start['model_state_sha256'] == STATE_SHA
        and canonical(start['runtime']) == canonical(prior_start['runtime']), 'start identity/runtime')
require(start['runtime']['device'] == 'cpu' and start['runtime']['cuda_initialized'] is False, 'CPU runtime')
require(sha(read(DATA / 'terminal.json')) == TERMINAL_SHA and terminal['schema'] == 'vision-recognized-v4-terminal'
        and terminal['status'] == 'complete' and terminal['mode'] == 'screen'
        and terminal['source_revision'] == REV and terminal['manifest_sha256'] == MANIFEST_SHA
        and terminal['model_state_sha256_before_after'] == STATE_SHA
        and canonical(terminal['runtime']) == canonical(start['runtime'])
        and type(terminal['submissions']) is int and terminal['submissions'] == 64
        and type(terminal['saved_forwards']) is int and terminal['saved_forwards'] == 64
        and terminal['screen_result_sha256'] == RESULT_SHA, 'terminal identity/counts')

result_path = DATA / 'screen-result.json'
result = obj(result_path)
require(sha(read(result_path)) == RESULT_SHA and result['schema'] == 'vision-recognized-v4-screen-result'
        and result['manifest_sha256'] == MANIFEST_SHA and len(result['screen']) == len(manifest['images']) == 64,
        'screen result binding')
expected_names = {'started.json', 'terminal.json', 'screen-result.json'}
for number in range(1, 65):
    expected_names |= {f'forward-{number:03}.json', f'forward-{number:03}.bin'}
require({path.name for path in DATA.iterdir() if path.is_file()} == expected_names
        and all(path.is_file() for path in DATA.iterdir()), 'exact screen file inventory')

recognized = 0
for number, (image, row) in enumerate(zip(manifest['images'], result['screen']), 1):
    record = obj(DATA / f'forward-{number:03}.json')
    raw = read(DATA / f'forward-{number:03}.bin')
    identity = image['image_id']
    require(type(identity) is int and type(row['image_id']) is int and row['image_id'] == identity
            and type(record['submission']) is int and record['submission'] == number
            and record['label'] == f'screen:{identity}'
            and canonical(record['metadata']) == canonical({'kind': 'screen', 'image_id': identity})
            and record['dtype'] == 'little-endian-float32' and record['file'] == f'forward-{number:03}.bin'
            and canonical(record['layout']) == canonical({'logits': {'offset': 0, 'bytes': 4000, 'shape': [1, 1000]}})
            and len(raw) == 4000 and record['sha256'] == sha(raw), 'forward binding ' + str(number))
    score = record['score']
    require(canonical(row) == canonical({'image_id': identity, **score}) and score['logit_sha256'] == sha(raw),
            'screen row binding ' + str(number))
    values = struct.unpack('<1000f', raw)
    require(all(math.isfinite(v) for v in values), 'nonfinite logits ' + str(number))
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = math.fsum(weights)
    dog_sum, cat_sum = math.fsum(weights[151:269]), math.fsum(weights[281:286])
    off_sum = math.fsum(weights[i] for i in range(1000) if not (151 <= i <= 268 or 281 <= i <= 285))
    expected = {'dog_mass': dog_sum / total, 'cat_mass': cat_sum / total, 'off_mass': off_sum / total,
                'margin': math.log(dog_sum) - math.log(cat_sum)}
    for key, value in expected.items():
        require(type(score[key]) in (int, float) and math.isclose(score[key], value, rel_tol=1e-12, abs_tol=1e-12),
                f'float score {number} {key}')
    top = max(range(1000), key=lambda i: values[i])
    require(type(score['top1_id']) is int and score['top1_id'] == top, 'top1 ' + str(number))
    category = image['category_id']
    require(type(category) is int and category in (17, 18), 'category type')
    matches = 281 <= top <= 285 if category == 17 else 151 <= top <= 268
    mass = expected['cat_mass'] if category == 17 else expected['dog_mass']
    reasons = ([] if matches else ['top1-outside-annotated-group']) + ([] if mass >= .2 else ['group-mass-below-0.2'])
    require(type(score['recognized']) is bool and score['recognized'] == (not reasons)
            and score['reasons'] == reasons, 'classification ' + str(number))
    recognized += int(score['recognized'])

archive_path = STAGE / 'screen-evidence.zip'
require(sha(read(archive_path)) == ARCHIVE_SHA, 'archive hash')
with zipfile.ZipFile(archive_path) as archive:
    names = archive.namelist()
    require(len(names) == len(expected_names) and len(set(names)) == len(names)
            and set(names) == expected_names, 'archive member inventory')
    for name in names:
        require(archive.read(name) == read(DATA / name), 'archive member bytes ' + name)
require(read(STAGE / 'screen-result.json') == read(result_path), 'staged result bytes')
require(read(STAGE / 'screen-observer.json') == read(OUTER / 'receipt.json'), 'staged observer bytes')
for name, expected_blob in REMOTE_BLOBS.items():
    local = (WORK / 'vision-recognized-v4-screen-independent-audit-v1.json'
             if name == 'screen-independent-audit.json' else STAGE / name)
    require(git_blob(read(local)) == expected_blob, 'published Git blob ' + name)

supervisor, observer = obj(RUN / 'supervisor.json'), obj(OUTER / 'receipt.json')
projection = obj(STAGE / 'screen-custody.json')
require(sha(read(RUN / 'supervisor.json')) == SUPERVISOR_SHA and supervisor['status'] == 'complete'
        and type(supervisor['exit_code']) is int and supervisor['exit_code'] == 0
        and supervisor['clean'] is True and supervisor['timed_out'] is False
        and supervisor['terminal_sha256'] == TERMINAL_SHA, 'supervisor receipt')
require(sha(read(OUTER / 'receipt.json')) == OBSERVER_SHA and observer['status'] == 'complete'
        and observer['schema'] == 'vision-recognized-v4-external-observer' and observer['mode'] == 'screen'
        and observer['source_revision'] == REV and type(observer['controller_exit_code']) is int
        and observer['controller_exit_code'] == 0 and observer['controller_reaped'] is True
        and observer['timed_out'] is False and observer['forced_stop'] is False
        and observer['terminal_sha256'] == TERMINAL_SHA and observer['supervisor_sha256'] == SUPERVISOR_SHA
        and observer['stdout_sha256'] == sha(read(OUTER / 'stdout.txt'))
        and observer['stderr_sha256'] == sha(read(OUTER / 'stderr.txt')) and not read(OUTER / 'stderr.txt'),
        'external controller receipt')
require(projection['schema'] == 'vision-recognized-v4-screen-custody' and projection['status'] == 'complete'
        and projection['source_revision'] == REV and projection['manifest_sha256'] == MANIFEST_SHA
        and projection['screen_result_sha256'] == RESULT_SHA and projection['screen_archive_sha256'] == ARCHIVE_SHA
        and projection['screen_terminal_sha256'] == TERMINAL_SHA and projection['private_supervisor_sha256'] == SUPERVISOR_SHA
        and projection['external_observer_sha256'] == OBSERVER_SHA
        and type(projection['controller_exit_code']) is int and projection['controller_exit_code'] == 0
        and type(projection['owner_exit_code']) is int and projection['owner_exit_code'] == 0
        and projection['owner_clean'] is True and projection['owner_timed_out'] is False, 'public custody projection')
claim = obj(CUSTODY / 'vision-recognized-v4-screen.claim.json')
require(claim == {'schema': 'vision-recognized-v4-attempt-claim', 'mode': 'screen',
                  'manifest_sha256': MANIFEST_SHA, 'source_revision': REV}, 'screen claim')
intervention_claim = CUSTODY / 'vision-recognized-v4-intervention.claim.json'
require(not (CUSTODY / 'cpu-study.unsafe.json').exists(), 'CPU unsafe marker present')
summary = result['summary']
require(type(summary['selected_pairs']) is int and summary['selected_pairs'] == len(result['pairs']) == 4
        and type(summary['distinct_selected_targets']) is int and summary['distinct_selected_targets'] == 8
        and type(summary['distinct_selected_donors']) is int and summary['distinct_selected_donors'] == 6,
        'selected inventory summary')
report = {
    'status': 'saved-screen-consistency-verified', 'source_revision': REV,
    'public_screen_commit': '9d90d90176a0903c47bce584c13c839038a88bb8',
    'manifest_sha256': MANIFEST_SHA, 'model_state_recorded_sha256': STATE_SHA,
    'source_files': len(source['sources']), 'model_asset_files': len(manifest['model_files']),
    'screened_images': 64, 'recognized': recognized, 'not_recognized': 64 - recognized,
    'screen_archive_members': len(expected_names), 'selected_pairs': 4, 'distinct_targets': 8,
    'distinct_donors': 6, 'screen_result_sha256': RESULT_SHA, 'screen_archive_sha256': ARCHIVE_SHA,
    'screen_terminal_sha256': TERMINAL_SHA, 'supervisor_sha256': SUPERVISOR_SHA,
    'external_observer_sha256': OBSERVER_SHA,
    'intervention_claim_present_at_check': intervention_claim.exists(),
    'public_git_blobs': REMOTE_BLOBS,
    'limitations': ['saved-byte consistency, not execution or publication-order attestation',
                    'cooperative direct-child receipts do not prove native or descendant cleanup',
                    'screen selection is model-output-based; reused donors and negative screens are retained',
                    'input-byte equality does not independently authenticate image origin or pretraining independence']}
payload = (canonical(report) + '\n').encode('utf-8')
output = WORK / 'vision-recognized-v4-screen-independent-custody.json'
with output.open('xb') as stream:
    stream.write(payload)
print('saved screen audit SHA-256:', sha(payload))
print('recognized/not-recognized:', recognized, 64 - recognized)
