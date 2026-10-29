"""Adapt the reviewed bounded CPU screen supervisor to the fixed antonym cases."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from . import distilbert_antonym_public_v1 as public
from . import distilbert_antonym_screen_v1 as screen
from . import distilbert_color_preference_runner_v2 as base
from . import distilbert_color_edit_publication_v1 as publication

CASE_SHA256 = public.CASE_SHA256
CHECK_SHA256 = 'f030cf522295013708e98a7e5b83dc9bebf94d4cec5b492f970c3d4380c314fa'
EXPOSURE_SHA256 = '74caf31b3dc9021a316057386a611217153f18379628938486bfdd96a355320d'
CHECK_PATH = 'experiments/bert_antonym_preference_public_comparison_v1.json'
SOURCE_PATHS = base.SOURCE_PATHS + (
    'docs/DISTILBERT_ANTONYM_PREFERENCE_V1.md', 'docs/DISTILBERT_ANTONYM_CASES_V1.md',
    'docs/DISTILBERT_ANTONYM_SCREEN_V1.md', public.CASE_PATH, public.EXPOSURE_PATH, CHECK_PATH,
    'scripts/build_distilbert_antonym_cases_v1.py', 'scripts/build_distilbert_antonym_exposures_v1.py',
    'scripts/check_distilbert_antonym_public_v1.py', 'scripts/prepare_distilbert_antonym_prelaunch_v1.py',
    'scripts/observe_distilbert_antonym_v1.py', 'scripts/review_distilbert_antonym_screen.py',
    'scripts/test_review_distilbert_antonym_screen.py', 'looking_glass/distilbert_antonym_public_v1.py',
    'looking_glass/distilbert_antonym_screen_v1.py', 'looking_glass/distilbert_antonym_runner_v1.py',
)
BOOTSTRAP = ("import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
             "sys.path.insert(1,sys.argv.pop(1));"
             "runpy.run_module('looking_glass.distilbert_antonym_runner_v1',run_name='__main__')")
_ORIGINAL_LOAD = base._load_model


class TokenizerAdapter:
    """Give the pinned fast tokenizer the small encoding API used by the case builder."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, add_special_tokens=True):
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return SimpleNamespace(ids=ids, tokens=self.tokenizer.convert_ids_to_tokens(ids))

    def id_to_token(self, index):
        return self.tokenizer.convert_ids_to_tokens([index])[0]


def load_model(cache, manifest, repo, scratch):
    """Reuse the frozen loader while independently reconstructing all 55 inputs."""
    from scripts import build_distilbert_color_edit_cases_v1 as legacy_builder
    from scripts import build_distilbert_color_preference_cases_v2 as legacy_specs
    from scripts import build_distilbert_antonym_cases_v1 as antonym
    old_encode, old_specs = legacy_builder.encode_inventory, legacy_specs.specifications

    def encode(tokenizer, specs):
        if specs != antonym.specifications():
            raise ValueError('fixed antonym specifications differ')
        return antonym.encode_inventory(TokenizerAdapter(tokenizer), specs)

    legacy_builder.encode_inventory = encode
    legacy_specs.specifications = antonym.specifications
    try:
        return _ORIGINAL_LOAD(cache, manifest, repo, scratch)
    finally:
        legacy_builder.encode_inventory, legacy_specs.specifications = old_encode, old_specs


def ensure_commit(repo, commit):
    if type(commit) is not str or re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('full immutable main commit is required')
    command = ['git', '-C', str(repo), 'cat-file', '-t', commit]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        subprocess.run(['git', '-C', str(repo), 'fetch', '--no-tags', '--no-write-fetch-head',
                        '--filter=blob:none', 'origin', commit], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode or result.stdout.strip() != b'commit':
        raise ValueError('requested object is not a commit')


def check_tiny_tail():
    from scripts import review_distilbert_antonym_screen as independent
    independent.require_denormal_support()
    result = independent.decimal_summary((0., -300.), {'factual': 0, 'counterfactual': 1})
    correction = Decimal(result['shifted_log_normalizer_decimal'])
    omitted = Decimal(result['log_normalizer_omission_upper_bound_decimal'])
    if not (Decimal('5.1481e-131') < correction < Decimal('5.1483e-131') and omitted == 0 and
            result['decimal_precision_digits'] >= 100):
        raise ValueError('independent adaptive tiny-tail check failed')
    return {'precision_digits': result['decimal_precision_digits'],
            'shifted_log_normalizer_decimal': result['shifted_log_normalizer_decimal'],
            'omission_bound_decimal': result['log_normalizer_omission_upper_bound_decimal'],
            'positive_correction': True}


class CurrentPublic:
    CASE_PATH = public.CASE_PATH
    EXPOSURE_PATH = public.EXPOSURE_PATH
    EXPOSURE_SHA256 = EXPOSURE_SHA256

    @staticmethod
    def scan(repo, commit, cases):
        ensure_commit(repo, commit)
        result = public.scan(repo, commit, cases, exposure_sha256=EXPOSURE_SHA256, require_own=True)
        if result['related_pairs']:
            raise ValueError('current public input prefix relation')
        result['commit'] = result['public_commit']
        return result

    additional = staticmethod(public.additional)


def verify_prelaunch(repo, source, phase, raw, *, check_main=True):
    """Bind prepared and current named-public comparisons to the fixed cases."""
    document = publication.strict_json(raw)
    if (type(document) is not dict or document.get('schema') != 'distilbert-antonym-prelaunch-v1' or
            document.get('phase') != phase or phase != 'screen' or document.get('source_commit') != source or
            document.get('case_sha256') != CASE_SHA256 or document.get('public_check_sha256') != CHECK_SHA256 or
            type(document.get('public_scan')) is not dict or type(document.get('additional_public')) is not list or
            len(document['additional_public']) > 20 or type(document.get('scope')) is not str or not document['scope']):
        raise ValueError('invalid scoped antonym prelaunch record')
    if base._json_bytes(document.get('tiny_tail_check')) != base._json_bytes(check_tiny_tail()):
        raise ValueError('independent adaptive tiny-tail check differs')
    repo = Path(repo)
    case_path, check_path, exposure_path = repo / public.CASE_PATH, repo / CHECK_PATH, repo / public.EXPOSURE_PATH
    if (base._file_sha(case_path)[1] != CASE_SHA256 or base._file_sha(check_path)[1] != CHECK_SHA256 or
            base._file_sha(exposure_path)[1] != EXPOSURE_SHA256):
        raise ValueError('planned case, comparison or exposure bytes differ')
    check = publication.strict_json(check_path.read_bytes())
    if (check.get('candidate_sha256') != CASE_SHA256 or check.get('candidate_rows') != 55 or
            check.get('related_pairs') != []):
        raise ValueError('fixed public comparison does not clear the cases')
    cases = publication.strict_json(case_path.read_bytes())['cases']
    observed = CurrentPublic.scan(repo, document['public_scan'].get('public_commit'), cases)
    if base._json_bytes(observed) != base._json_bytes(document['public_scan']):
        raise ValueError('prepared public scan does not reconstruct')
    seen = set()
    for item in document['additional_public']:
        if type(item) is not dict or set(item) != {'commit', 'path', 'sha256', 'rows'}:
            raise ValueError('invalid additional public source')
        key = (item['commit'], item['path'])
        if key in seen:
            raise ValueError('duplicate additional public source')
        seen.add(key)
        ensure_commit(repo, item['commit'])
        if base._json_bytes(public.additional(repo, item['commit'], item['path'], cases)) != base._json_bytes(item):
            raise ValueError('additional public source does not reconstruct')
    current = observed
    if check_main:
        remote = subprocess.run(['git', '-C', str(repo), 'ls-remote', '--heads', 'origin', 'refs/heads/main'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60).stdout.split()
        if len(remote) != 2 or remote[1] != b'refs/heads/main':
            raise RuntimeError('unable to observe official main')
        current = CurrentPublic.scan(repo, remote[0].decode('ascii'), cases)
    return {'document': document, 'current_scan': current}


def configure():
    base.CASE_SHA256, base.CHECK_SHA256 = CASE_SHA256, CHECK_SHA256
    base.SOURCE_PATHS, base.MAX_SCREEN_STARTS, base.BOOTSTRAP = SOURCE_PATHS, 12, BOOTSTRAP
    base.preference, base.stages, base.public = screen, screen, CurrentPublic
    base._load_model, base._verify_prelaunch = load_model, verify_prelaunch
    return base


def main():
    configure().main()


if __name__ == '__main__':
    main()
