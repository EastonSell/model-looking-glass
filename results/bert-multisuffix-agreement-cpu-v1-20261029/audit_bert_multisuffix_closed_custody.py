"""Reconcile the closed multisuffix CPU attempt and cooperative observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

COMMIT = '42519c13a5b33ce3601b89ba65d67b07ebd2f78d'
RECEIPT = '4f85c2794bb45a6e598069c97be1a089d0203c80c5c625ac325eeceb25d42347'
COMPLETE = 'eb8d5d0af89363046512936e62f26e024659c6a39a0b192eff7b5991e804d460'
OUTER = '86897fe8a359ee97c22dc84e93ba85a9958051a467c07befc9f6880cfb5c48a4'
STAGE_RECEIPT = 'd8f0d8415492fba043204499d66694f5d0a109401967c61245baf7943d1461f8'
SOURCE_PATHS = {
    'scripts/bert_multisuffix_agreement_v1.py', 'scripts/read_bert_multisuffix_agreement_v1.py',
    'scripts/observe_bert_multisuffix_agreement_v1.py', 'scripts/cross_model_agreement_v1.py',
    'scripts/read_cross_model_agreement_v1.py', 'docs/BERT_MULTISUFFIX_AGREEMENT_V1.md',
    'docs/BERT_MULTISUFFIX_EXECUTION_V1.md', 'experiments/bert_multisuffix_agreement_cases_v1.json',
}
ASSETS = {
    'config.json': '69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83',
    'model.safetensors': '5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063',
    'tokenizer.json': 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98',
    'tokenizer_config.json': 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
    'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def inventory(root):
    out = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            info = path.lstat()
            require(not path.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
                    'reparse point in saved inventory')
            out.append({'path': path.relative_to(root).as_posix(), 'bytes': info.st_size, 'sha256': sha(path)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('attempt', 'controller', 'outer', 'repo', 'stage', 'stage-receipt', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    output = a.output.resolve()
    for root in (a.attempt, a.controller, a.outer, a.repo, a.stage):
        resolved = root.resolve(strict=True)
        require(output != resolved and not output.is_relative_to(resolved), 'output inside an input root')
    require(not output.exists(), 'fresh output required')
    receipt_path, complete_path, outer_path = (a.controller / 'exit-observation.json',
                                               a.attempt / 'complete.json', a.outer / 'observer-exit.json')
    require((sha(receipt_path), sha(complete_path), sha(outer_path), sha(a.stage_receipt))
            == (RECEIPT, COMPLETE, OUTER, STAGE_RECEIPT), 'observation digest differs')
    receipt, complete, outer = map(load, (receipt_path, complete_path, outer_path))
    meta = load(a.attempt / 'metadata.json')
    child = load(a.controller / 'spawn.json')
    observed = load(a.outer / 'observer-spawn.json')
    admission, outer_admission, staged = map(load, (a.controller / 'admission.json',
                                                     a.outer / 'admission.json', a.stage_receipt))
    require(inventory(a.attempt) == outer['attempt_inventory']
            and inventory(a.controller) == outer['controller_inventory'], 'closed inventory differs')
    require(not (a.controller / 'observer-pending.json').exists()
            and not (a.attempt / 'failure.txt').exists(), 'pending/failure marker present')
    require(outer['source_commit'] == receipt['source_commit'] == meta['source_commit']
            == child['source_commit'] == observed['source_commit'] == COMMIT, 'source identity differs')
    require(type(receipt['child_pid']) is int and receipt['child_pid'] > 0
            and type(child['pid']) is int and type(meta['pid']) is int
            and receipt['child_pid'] == child['pid'] == meta['pid']
            and type(outer['observer_pid']) is int and outer['observer_pid'] > 0
            and type(observed['pid']) is int and outer['observer_pid'] == observed['pid']
            and outer['observer_pid'] != receipt['child_pid'], 'reported process IDs differ')
    require(receipt['status'] == 'complete' and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
            and receipt['interrupted'] is None and receipt['complete_sha256'] == COMPLETE
            and receipt['metadata_sha256'] == sha(a.attempt / 'metadata.json')
            and type(outer['exit_code']) is int and outer['exit_code'] == 0 and outer['interrupted'] is None
            and outer['direct_receipt_valid'] is True and outer['observer_source_unchanged'] is True
            and outer['pending_marker_present'] is False and outer['receipt_sha256'] == RECEIPT,
            'reported completion/exit differs')
    require(complete['status'] == 'complete' and type(complete['events']) is int and complete['events'] == 1088
            and type(complete['eligible_cases']) is int and complete['eligible_cases'] == 32
            and complete['model_weakref_dead'] is True, 'recorded completion differs')
    for item in outer['attempt_inventory']:
        if item['path'] != 'complete.json':
            require(complete[item['path'].replace('.', '_') + '_sha256'] == item['sha256'],
                    'completion hash differs from saved bytes')
    require(observed['start_unix'] == outer['start_unix'] <= child['start_unix']
            == receipt['start_unix'] <= meta['start_unix'] <= complete['end_unix']
            <= receipt['end_unix'] <= outer['end_unix'], 'reported chronology differs')
    require(meta['device'] == 'cpu' and meta['deterministic_algorithms'] is True
            and meta['isolated'] is True and meta['base_interpreter_direct'] is True
            and type(meta['intra_threads']) is int and meta['intra_threads'] == 1
            and type(meta['inter_threads']) is int and meta['inter_threads'] == 1, 'recorded CPU policy differs')
    require(set(meta['sources_sha256']) == SOURCE_PATHS, 'source inventory differs')
    git = Path(shutil.which('git')).resolve()
    require(sha(git) == meta['git_sha256'], 'Git executable differs')
    env = {k: os.environ[k] for k in ('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'PATH') if k in os.environ}
    env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    for name in sorted(SOURCE_PATHS):
        committed = subprocess.run([str(git), '-C', str(a.repo), '--no-replace-objects', 'cat-file',
                                    'blob', COMMIT + ':' + name], check=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env, timeout=30).stdout
        require(hashlib.sha256(committed).hexdigest() == meta['sources_sha256'][name]
                and (a.repo / name).read_bytes() == committed, 'source blob differs')
    require(outer['observer_source_sha256'] == meta['sources_sha256']['scripts/observe_bert_multisuffix_agreement_v1.py'],
            'outer observer source digest differs')
    require(meta['assets_sha256'] == ASSETS and set(staged['files']) == set(ASSETS)
            and {p.name for p in a.stage.iterdir()} == set(ASSETS), 'asset inventory differs')
    for name, digest in ASSETS.items():
        require(sha(a.stage / name) == staged['files'][name]['sha256'] == digest
                and (a.stage / name).stat().st_size == staged['files'][name]['bytes'], 'staged asset differs')
    require(outer_admission['source_commit'] == COMMIT and outer_admission['stage_receipt_sha256'] == STAGE_RECEIPT
            and outer_admission['available_physical_bytes'] >= outer_admission['host_minimum_bytes'] == 2 * 1024**3
            and outer_admission['output_free_bytes'] >= outer_admission['output_minimum_bytes'] == 1024**3
            and outer_admission['system_free_bytes'] >= outer_admission['system_minimum_bytes'] == 256 * 1024**2
            and admission['available_physical_bytes'] >= admission['host_minimum_bytes'] == 2 * 1024**3
            and min(admission['attempt_free_bytes'], admission['controller_free_bytes'])
            >= admission['output_volume_minimum_bytes'] == 1024**3, 'reported admission differs')
    report = {'schema': 'independent-multisuffix-closed-custody-v1', 'status': 'record_consistency',
              'source_commit': COMMIT, 'receipt_sha256': RECEIPT, 'complete_sha256': COMPLETE,
              'outer_observation_sha256': OUTER, 'stage_receipt_sha256': STAGE_RECEIPT,
              'saved_attempt_files': len(outer['attempt_inventory']),
              'saved_controller_files': len(outer['controller_inventory']), 'source_files': len(SOURCE_PATHS),
              'checkpoint_asset_files': len(ASSETS), 'reported_events': 1088, 'reported_eligible_cases': 32,
              'reported_child_pid': receipt['child_pid'], 'reported_observer_pid': outer['observer_pid'],
              'outer_admission': outer_admission, 'child_admission': admission,
              'scope': ['Hashes and observations reconcile the saved cooperative records and current staged bytes.',
                        'They do not independently attest execution, all live memory, public exposure, source chronology or process-tree drain.',
                        'The outer observer process itself requires a separate operator exit observation.']}
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with output.open('xb') as stream:
        require(stream.write(data) == len(data), 'short report write')
    print('independent_custody', hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
