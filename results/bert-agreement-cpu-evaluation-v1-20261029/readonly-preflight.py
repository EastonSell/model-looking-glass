"""Read-only preflight for the published BERT CPU evaluation source.

This script does not launch a study, reserve rows, import a model dependency,
tokenize an evaluation input, or run a forward. It checks only source, frozen
bindings and cached checkpoint bytes using the study's existing verifiers.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def main():
    if len(sys.argv) != 8:
        raise ValueError('expected checkout, site, cache, three roots and output')
    repo, site, cache, cpu_root, ledger_root, controller_root, out = map(Path, sys.argv[1:])
    source = repo / 'scripts' / 'bert_agreement_bootstrap.py'
    spec = importlib.util.spec_from_file_location('bert_readonly_bootstrap', source)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    repo = bootstrap._checked_path(repo)
    site = bootstrap._checked_path(site)
    bootstrap._check_startup(repo)
    if not site.is_dir() or site == repo or site.is_relative_to(repo) or repo.is_relative_to(site):
        raise RuntimeError('dependency directory must be separate from checkout')
    sys.path.append(str(site))
    bootstrap._install_bytecode_guard()
    bootstrap._load_package(repo / 'looking_glass')
    os.environ['LOOKING_GLASS_BERT_BOOTSTRAP_PID'] = str(os.getpid())
    os.environ['LOOKING_GLASS_BERT_ROOT'] = str(repo)
    os.environ['LOOKING_GLASS_BERT_SITE'] = str(site)
    from looking_glass import bert_agreement_study as study

    commit = '4ee50069464228798071218b58b64221ac5960f4'
    study._require_bootstrap()
    study._require_checkout(repo)
    git = shutil.which('git')
    if not git:
        raise RuntimeError('Git is required')
    hashes, manifest, manifest_sha = study._checked_manifest(repo, commit, git, 'evaluation')
    binding = study._evaluation_binding(repo, commit, hashes, manifest, manifest_sha,
                                        cpu_root, ledger_root, controller_root)
    if not study.EVALUATION_ENABLED or len(binding['rows']) != 64:
        raise RuntimeError('unexpected evaluation gate or inventory')
    snapshot = (cache / 'models--distilbert--distilbert-base-uncased' /
                'snapshots' / study.REVISION)
    assets = {}
    for name, expected in study.ASSETS.items():
        observed = sha256_file(snapshot / name)
        if observed != expected:
            raise RuntimeError('checkpoint asset mismatch: ' + name)
        assets[name] = {'size': observed[0], 'sha256': observed[1]}
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise RuntimeError('a model dependency was imported during preflight')
    if any(Path(sys.pycache_prefix).iterdir()):
        raise RuntimeError('the isolated bytecode prefix is no longer empty')
    for root in (cpu_root, controller_root):
        for name in ('cpu-study.unsafe.json', 'cpu-study.unsafe.json.attestation'):
            if os.path.lexists(root / name):
                raise RuntimeError('a study custody marker is present')
    if any(ledger_root.iterdir()):
        raise RuntimeError('evaluation ledger is not empty')
    result = {
        'status': 'passed', 'scope': 'read-only; no reservation or model forward',
        'source_commit': commit, 'source_file_count': len(hashes),
        'protocol_sha256': binding['protocol_sha256'], 'manifest_sha256': manifest_sha,
        'exposure_sha256': binding['exposure_sha256'], 'evaluation_rows': len(binding['rows']),
        'ledger_root_sha256': binding['ledger_root_sha256'],
        'cpu_custody_root_sha256': binding['cpu_study']['custody_root_sha256'],
        'controller_custody_root_sha256': binding['controller_cpu_study']['custody_root_sha256'],
        'model': study.MODEL_ID, 'model_revision': study.REVISION, 'assets': assets,
        'torch_imported': False, 'transformers_imported': False,
        'bytecode_prefix_empty': True, 'evaluation_ledger_empty': True,
    }
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode('utf-8')
    with out.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
