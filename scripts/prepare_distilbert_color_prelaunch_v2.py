"""Prepare a named-public BERT input scan for the v2 screen; no model is loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from looking_glass import distilbert_color_edit_publication_v1 as publication
from looking_glass import distilbert_color_preference_public_v2 as public
from looking_glass import distilbert_color_preference_runner_v2 as runner
from looking_glass.bert_agreement_study import verify_git_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--additional', nargs=2, action='append', default=[], metavar=('COMMIT', 'PATH'))
    parser.add_argument('--scope', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    verify_git_sources(repo, args.source_commit, runner.SOURCE_PATHS)
    remote = subprocess.run(['git', '-C', str(repo), 'ls-remote', '--heads', 'origin', 'refs/heads/main'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60).stdout.split()
    if len(remote) != 2 or remote[1] != b'refs/heads/main':
        raise RuntimeError('unable to observe official main')
    manifest = publication.strict_json((repo / public.CASE_PATH).read_bytes())
    proposed = manifest['cases']
    observed = public.scan(repo, remote[0].decode('ascii'), proposed)
    sources = [public.additional(repo, commit, path, proposed) for commit, path in args.additional]
    document = {'schema': 'distilbert-color-preference-prelaunch-v2', 'phase': 'screen',
                'source_commit': args.source_commit, 'case_sha256': runner.CASE_SHA256,
                'public_check_sha256': runner.CHECK_SHA256, 'public_scan': observed,
                'additional_public': sources, 'scope': args.scope, 'prepared_utc': runner._now()}
    raw = runner._json_bytes(document)
    runner._verify_prelaunch(repo, args.source_commit, 'screen', raw)
    runner._write(args.output, raw)
    print(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(), 'official_main': observed['commit'],
                      'public_sources': len(observed['sources']), 'additional_sources': len(sources)},
                     sort_keys=True))


if __name__ == '__main__':
    main()
