"""Reconcile saved subject-study boundary records, without inspecting private data."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/teams/agreement_subject_executor'
HOME = ROOT / 'work/agreement_subject_followup'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError('pinned record differs')
    return json.loads(raw)


def git(*args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='NUL', GIT_CONFIG_SYSTEM='NUL',
               GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    return subprocess.check_output(['git', '--no-replace-objects', '-C', str(REPO), *args],
                                   env=env, stderr=subprocess.DEVNULL, timeout=60)


def modules(revision):
    entries = git('ls-tree', '-rz', '--full-tree', revision).split(b'\0')
    return {name.decode(): header.decode() for item in entries if item
            for header, name in [item.split(b'\t', 1)]
            if name.startswith(b'looking_glass/pretrained_exposures') and name.endswith(b'.py')}


def time(value):
    result = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timestamp lacks timezone')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pre_dir, outer = HOME / 'study-prelaunch-v1', HOME / 'study-attempt-v1'
    exit_record = read(pre_dir / 'parent-exit.json', '8a83d9d16c3d8e18d2ccf36bc61a730e5405ed065319a348a3bf13d657367884')
    boundary = read(pre_dir / 'launch-boundary.json', exit_record['launch_boundary_sha256'])
    post = read(pre_dir / 'post-start-v28-observation.json', exit_record['post_launch_v28_observation_sha256'])
    pre = read(pre_dir / 'prelaunch.json', exit_record['prelaunch_sha256'])
    supervisor = read(outer / 'supervisor.json', exit_record['supervisor_sha256'])
    if sha((pre_dir / 'argv-live.json').read_bytes()) != supervisor['argument_vector_sha256']:
        raise ValueError('saved argument-vector binding differs')
    argv = json.loads((pre_dir / 'argv-live.json').read_bytes())
    if type(argv) is not list or '--validate-only' in argv:
        raise ValueError('study argument-vector differs')
    if not (time(pre['recorded_utc']) <= time(boundary['recorded_utc']) <= time(supervisor['start_utc'])
            <= time(post['recorded_utc']) <= time(supervisor['end_utc']) <= time(exit_record['recorded_utc'])):
        raise ValueError('recorded chronology differs')
    v27 = 'baff8872b35193bcaf767c9e260ad4820e36332e'
    observed = boundary['official_main_observed_revision']
    git('merge-base', '--is-ancestor', v27, observed)
    if modules(v27) != modules(observed):
        raise ValueError('exposure modules changed at observed revision')
    blob = git('rev-parse', observed + ':looking_glass/pretrained_exposures_v27.py').strip().decode()
    if (blob != boundary['v27_source_blob'] or boundary['owner_revision'] != pre['owner']['source_revision']
            or boundary['owner_tree'] != pre['owner']['source_tree']
            or boundary['prelaunch_sha256'] != exit_record['prelaunch_sha256']
            or boundary['latest_reviewed_exposure_sha256'] != pre['exposure']['operator_sha256']
            or boundary['latest_reviewed_exposure_version'] != pre['exposure']['operator_version']
            or post['predecessor_sha256'] != pre['exposure']['latest_sha256']
            or post['saved_forward_observed_at_check'] is not True):
        raise ValueError('recorded launch/source binding differs')
    storage = boundary['storage']
    if (type(storage['same_volume']) is not bool or type(storage['cache_free_bytes']) is not int
            or type(storage['output_free_bytes']) is not int
            or storage['cache_free_bytes'] < storage['required_cache_free_bytes']
            or storage['output_free_bytes'] < storage['required_output_free_bytes']
            or (storage['same_volume'] and storage['output_free_bytes'] <
                storage['required_output_free_bytes'] + storage['required_cache_free_bytes'])):
        raise ValueError('recorded storage observation differs')
    replay = ROOT / 'work/reviews/review_agreement_subject_v28.py'
    replay_sha = sha(replay.read_bytes())
    raw = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(replay)], timeout=180)
    v28 = json.loads(raw)
    if (v28['v28_commit'] != post['v28_revision'] or v28['v28_sha256'] != post['v28_sha256']
            or v28['v27_records_preserved'] != post['prior_records_preserved']
            or v28['new_records'] != post['added_records']
            or v28['subject_planned_records'] != post['own_planned_records']
            or v28['subject_sequences'] != post['subject_sequences']
            or v28['other_known_entries'] != post['other_known_sequences']
            or v28['comparisons_including_duplicates'] != post['bidirectional_comparisons']
            or v28['relations'] != post['relations']):
        raise ValueError('post-start public inventory comparison differs')
    result = {'status': 'recorded_consistency_pass',
              'prelaunch_sha256': exit_record['prelaunch_sha256'],
              'parent_exit_sha256': '8a83d9d16c3d8e18d2ccf36bc61a730e5405ed065319a348a3bf13d657367884',
              'supervisor_sha256': exit_record['supervisor_sha256'],
              'launch_boundary_sha256': exit_record['launch_boundary_sha256'],
              'post_start_observation_sha256': exit_record['post_launch_v28_observation_sha256'],
              'argv_sha256': supervisor['argument_vector_sha256'],
              'recorded_times_ordered': True,
              'operator_reported_prelaunch_main_revision': observed,
              'v27_module_blob_at_observed_revision': blob,
              'v27_modules_match_reservation_commit': True,
              'recorded_storage': storage,
              'post_start_public_inventory_replay': v28,
              'v28_replay_script_sha256': replay_sha,
              'scope': ('Record consistency and Git/public-byte replay; the reported main observation, custody, '
                        'clock times and free space are cooperative point-in-time observations, not authenticated '
                        'execution, storage reservations, private-membership checks or complete exposure history. '
                        'The child remained v27-bound; v28 was inspected after start.')}
    data = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as stream:
        stream.write(data)
    print(json.dumps({'sha256': sha(data), 'bytes': len(data)}, sort_keys=True))


if __name__ == '__main__':
    main()
