"""Prepare the current named-public scan for the antonym screen; no model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from looking_glass import distilbert_antonym_public_v1 as public
from looking_glass import distilbert_antonym_runner_v1 as runner
from looking_glass import distilbert_color_edit_publication_v1 as publication
from looking_glass.bert_agreement_study import verify_git_sources


def make_document(source, scan, additional, scope):
    return {'schema': 'distilbert-antonym-prelaunch-v1', 'phase': 'screen',
            'source_commit': source, 'case_sha256': runner.CASE_SHA256,
            'public_check_sha256': runner.CHECK_SHA256, 'public_scan': scan,
            'additional_public': additional, 'tiny_tail_check': runner.check_tiny_tail(),
            'scope': scope, 'prepared_utc': runner.base._now()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--additional', nargs=2, action='append', default=[], metavar=('COMMIT', 'PATH'))
    parser.add_argument('--scope', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    verify_git_sources(repo, args.source_commit, runner.SOURCE_PATHS)
    remote = subprocess.run(['git', '-C', str(repo), 'ls-remote', '--heads', 'origin', 'refs/heads/main'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60).stdout.split()
    if len(remote) != 2 or remote[1] != b'refs/heads/main':
        raise RuntimeError('unable to observe official main')
    cases = publication.strict_json((repo / public.CASE_PATH).read_bytes())['cases']
    observed = runner.CurrentPublic.scan(repo, remote[0].decode('ascii'), cases)
    extra = []
    for commit, path in args.additional:
        runner.ensure_commit(repo, commit)
        extra.append(public.additional(repo, commit, path, cases))
    document = make_document(args.source_commit, observed, extra, args.scope)
    raw = runner.base._json_bytes(document)
    runner.verify_prelaunch(repo, args.source_commit, 'screen', raw, check_main=False)
    runner.base._write(args.output, raw)
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'official_main': observed['commit'],
                      'public_sources': len(observed['sources']), 'additional_sources': len(extra)}, sort_keys=True))


if __name__ == '__main__':
    main()
