"""Release packaging has a narrow trigger, scoped token and no overwrite action."""
from pathlib import Path
import subprocess

import pytest

WORKFLOW = Path(__file__).parents[1] / '.github/workflows/release-app.yml'


def test_release_only_scope_and_quoted_context():
    yaml = pytest.importorskip('yaml')
    document = yaml.safe_load(WORKFLOW.read_text())
    trigger = document.get('on', document.get(True))
    assert trigger == {'release': {'types': ['published']}}
    assert document['permissions'] == {'contents': 'read'}
    jobs = document['jobs']
    assert list(jobs) == ['app-zip']
    job = jobs['app-zip']
    assert job['permissions'] == {'contents': 'write'}
    assert job['timeout-minutes'] == 10
    assert job['runs-on'] == 'ubuntu-24.04'
    assert all('uses' not in step for step in job['steps'])
    steps = job['steps']
    assert all('GH_TOKEN' not in step.get('env', {}) for step in steps[:-1])
    assert steps[-1]['env']['GH_TOKEN'] == '${{ secrets.GITHUB_TOKEN }}'
    for step in steps:
        assert '${{' not in step['run']  # Context values enter via env, not shell source.
        subprocess.run(['bash', '-n'], input=step['run'], text=True, check=True)
    all_commands = '\n'.join(step['run'] for step in steps)
    assert '--clobber' not in all_commands
    assert 'pip install' not in all_commands and 'curl' not in all_commands
    assert 'git fetch --depth=1 --filter=blob:none origin "$GITHUB_REF"' in all_commands
    assert 'git sparse-checkout set --no-cone' in all_commands
    assert 'test "$actual" = "$GITHUB_SHA"' in all_commands
    assert '--revision "$GITHUB_SHA"' in all_commands
    assert 'gh release upload "$RELEASE_TAG"' in all_commands
    assert 'gh release download "$RELEASE_TAG"' in all_commands
    assert 'cmp dist/model-looking-glass-app.zip verified/model-looking-glass-app.zip' in all_commands
    assert 'sha256sum -c SHA256SUMS' in all_commands
