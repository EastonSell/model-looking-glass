"""Compare antonym input rows with a named public BERT Git snapshot."""
import hashlib
from pathlib import Path

from . import distilbert_color_edit_publication_v1 as publication
from . import distilbert_color_preference_public_v2 as public
from scripts import check_distilbert_color_preference_exposures_v2 as comparison

CASE_PATH = 'experiments/bert_antonym_preference_public_cases_v1.json'
EXPOSURE_PATH = 'experiments/bert_antonym_preference_public_exposures_v1.json'
CASE_SHA256 = '284fc9adc0cc3caa52ea5d122c1f08db50c8dda2d881297e2dcd1f2229cb6386'


def related_any(left, right):
    return any(comparison.related(a, b) for a in (left, left[:-1]) for b in (right, right[:-1]))


def _planned(cases):
    rows = [comparison.valid_input(row) for row in cases]
    if len(rows) != 55 or len({tuple(row) for row in rows}) != 55:
        raise ValueError('fixed 55-row planned inventory differs')
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if related_any(left, right):
                raise ValueError('planned inputs contain a prefix relation')
    return rows


def scan(repo, commit, cases, *, exposure_sha256=None, require_own=False):
    planned = _planned(cases)
    labels = [row['id'] for row in cases]
    tree = public._tree(repo, commit)
    if require_own and (CASE_PATH not in tree or EXPOSURE_PATH not in tree):
        raise ValueError('current public planned case or exposure is missing')
    sources, hits = [], []
    for path, blob in sorted(tree.items()):
        raw = publication.git_blob(Path(repo), commit, path)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('public input file exceeds scan bound')
        sha = hashlib.sha256(raw).hexdigest()
        if path == CASE_PATH or path == EXPOSURE_PATH:
            expected = CASE_SHA256 if path == CASE_PATH else exposure_sha256
            if expected is None or sha != expected:
                raise ValueError('own public planned input identity changed')
            continue
        prior = public.rows(publication.strict_json(raw), path)
        if not 1 <= len(prior) <= 1000 or sum(item['rows'] for item in sources) + len(prior) > 10000:
            raise ValueError('public input row count exceeds scan bound')
        sources.append({'path': path, 'git_blob': blob, 'sha256': sha, 'rows': len(prior)})
        for index, left in enumerate(planned):
            for prior_index, right in enumerate(prior):
                if related_any(left, right):
                    hits.append({'case_id': labels[index], 'source': path, 'source_row': prior_index})
    total = sum(item['rows'] for item in sources)
    return {'candidate_sha256': CASE_SHA256, 'candidate_rows': len(planned), 'public_commit': commit,
            'sources': sources, 'source_row_references': total, 'row_pairs_compared': len(planned) * total,
            'related_pairs': hits,
            'scope': ('Named public BERT and cross-model case/exposure JSON rows; both prefix directions '
                      'with each final SEP present or removed. No private, unknown-history or pretraining claim.')}


def additional(repo, commit, path, cases):
    planned = _planned(cases)
    raw = publication.git_blob(Path(repo), commit, path)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('additional public input file exceeds scan bound')
    rows = public.rows(publication.strict_json(raw), path)
    if not 1 <= len(rows) <= 1000:
        raise ValueError('additional source row count exceeds bound')
    if any(related_any(left, right) for left in planned for right in rows):
        raise ValueError(f'additional public input prefix relation: {path}')
    return {'commit': commit, 'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows)}
