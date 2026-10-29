"""Prepare a point-in-time public-input check without loading a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from looking_glass import distilbert_color_edit_publication_v1 as publication
from looking_glass import distilbert_color_edit_runner_v1 as runner
from looking_glass.bert_agreement_study import verify_git_sources
from scripts import check_distilbert_color_edit_exposures_v1 as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--phase', choices=('screen', 'development', 'evaluation'), required=True)
    parser.add_argument('--additional', nargs=3, action='append', default=[], metavar=('COMMIT', 'PATH', 'FORMAT'),
                        help='extra public manifest in case-rows, cue-variants or bert-cases format')
    parser.add_argument('--scope', required=True,
                        help='scoped owner/custody observation; do not claim complete or pretraining history')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    verify_git_sources(repo, args.source_commit, runner.SOURCE_PATHS)
    remote = subprocess.run(['git', '-C', str(repo), 'ls-remote', '--heads', 'origin', 'refs/heads/main'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60).stdout.split()
    if len(remote) != 2 or remote[1] != b'refs/heads/main':
        raise RuntimeError('unable to observe official main')
    sources = []
    for commit, path, format_ in args.additional:
        raw = publication.git_blob(repo, commit, path)
        document = publication.strict_json(raw)
        if format_ == 'case-rows':
            rows = [comparison.valid_input(row) for row in document['cases']]
        elif format_ == 'bert-cases':
            rows = [comparison.valid_input(case['bert']) for case in document['cases']]
        elif format_ == 'cue-variants':
            rows = [comparison.valid_input(row) for case in document['cases'] for row in case['rows'].values()]
        else:
            raise ValueError('unsupported public input format')
        sources.append({'commit': commit, 'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'format': format_, 'rows': len(rows)})
    observation = {'schema': 'distilbert-color-prelaunch-v1', 'phase': args.phase,
                   'source_commit': args.source_commit, 'case_sha256': runner.CASE_SHA256,
                   'public_check_sha256': runner.CHECK_SHA256, 'official_main': remote[0].decode('ascii'),
                   'additional_public': sources, 'scope': args.scope, 'prepared_utc': runner._now()}
    raw = runner._json_bytes(observation)
    runner._verify_prelaunch(repo, args.source_commit, args.phase, raw)
    runner._write(args.output, raw)
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'official_main': observation['official_main'],
                      'additional_public_sources': len(sources)}, sort_keys=True))


if __name__ == '__main__':
    main()
