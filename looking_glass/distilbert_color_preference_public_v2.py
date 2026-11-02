"""Recheck current named public BERT inputs before the preference screen."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import distilbert_color_edit_publication_v1 as publication
from scripts import check_distilbert_color_preference_exposures_v2 as comparison

CASE_PATH = 'experiments/bert_color_preference_public_cases_v2.json'
CASE_SHA256 = 'f938cee4acab09d70f3d827e6172a62d273a5c621c35b6e3e47bd46303aa5a10'
EXPOSURE_PATH = 'experiments/bert_color_preference_public_exposures_v1.json'
EXPOSURE_SHA256 = '069516b19b575a13daa140afeca421bf8d8d99184dfc3cf833b715436ce4d6db'


def relevant(path):
    if not path.startswith('experiments/') or not path.endswith('.json'):
        return False
    name = path.rsplit('/', 1)[-1]
    return ((name.startswith('bert_') and ('cases' in name or 'exposures' in name)) or
            (name.startswith('cross_model_') and 'cases' in name))


def _tree(repo, commit):
    if type(commit) is not str or re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('full immutable commit required')
    raw = subprocess.run(['git', '-C', str(repo), 'ls-tree', '-r', '-z', '--full-tree',
                          commit, '--', 'experiments'], check=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=60).stdout
    paths = {}
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        metadata, path_bytes = entry.split(b'\t', 1)
        mode, kind, sha = metadata.decode('ascii').split(' ')
        path = path_bytes.decode('utf-8')
        if relevant(path):
            if kind != 'blob' or mode != '100644' or path in paths:
                raise ValueError('invalid public inventory tree')
            paths[path] = sha
    if not 1 <= len(paths) <= 128:
        raise ValueError('unexpected public inventory count')
    return paths


def rows(document, path):
    if path.endswith('/bert_negation_public_cases_v1.json'):
        return comparison.source_rows(document, 'negation')
    if type(document) is not dict:
        raise ValueError('public input document must be an object')
    if 'records' in document:
        return comparison.source_rows(document, 'records')
    if 'rows' in document:
        return comparison.source_rows(document, 'rows')
    if 'pairs' in document:
        return [comparison.valid_input(row) for pair in document['pairs'] for context in pair['contexts']
                for _, row in sorted(context['rows'].items())]
    if 'cases' not in document or type(document['cases']) is not list or not document['cases']:
        raise ValueError('unknown public input schema')
    first = document['cases'][0]
    if 'bert' in first:
        return comparison.source_rows(document, 'nested-bert')
    if 'rows' in first:
        return comparison.source_rows(document, 'variants')
    if 'variants' in first:
        return [comparison.valid_input(row) for case in document['cases']
                for _, row in sorted(case['variants'].items())]
    return [comparison.valid_input(row) for row in document['cases']]


def _coverage_digest(rows_):
    # Duplicated historical versions do not change the known-input set.
    unique = sorted({tuple(row) for row in rows_})
    raw = json.dumps(unique, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return len(unique), hashlib.sha256(raw).hexdigest()


def additional(repo, commit, path, planned):
    """Check one separately identified public proposal, including its variants."""
    raw = publication.git_blob(Path(repo), commit, path)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('additional public input file exceeds the scan bound')
    prior = rows(publication.strict_json(raw), path)
    if not 1 <= len(prior) <= 1000:
        raise ValueError('invalid additional public row count')
    for left in planned:
        comparison.valid_input(left)
        for right in prior:
            if comparison.related(left['input_ids'], right) or comparison.related(left['input_ids'][:-1], right[:-1]):
                raise ValueError(f'additional public input prefix relation: {path}')
    return {'commit': commit, 'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(prior)}


def scan(repo, commit, planned):
    """Read all matching public Git blobs, refusing unknown formats or overlap."""
    planned = [comparison.valid_input(row) for row in planned]
    if len(planned) != 37 or len({tuple(x) for x in planned}) != 37:
        raise ValueError('fixed planned inputs differ')
    tree = _tree(repo, commit)
    if CASE_PATH not in tree or EXPOSURE_PATH not in tree:
        raise ValueError('planned case and exposure files are missing from observed main')
    all_rows, sources = [], []
    for path, blob in sorted(tree.items()):
        raw = publication.git_blob(Path(repo), commit, path)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('public input file exceeds the scan bound')
        sha = hashlib.sha256(raw).hexdigest()
        if path in (CASE_PATH, EXPOSURE_PATH):
            expected = CASE_SHA256 if path == CASE_PATH else EXPOSURE_SHA256
            if sha != expected:
                raise ValueError('planned own input identity changed')
            continue
        prior = rows(publication.strict_json(raw), path)
        if not 1 <= len(prior) <= 1000 or len(all_rows) + len(prior) > 10000:
            raise ValueError('public input row count exceeds the scan bound')
        for left in planned:
            for right in prior:
                if comparison.related(left, right) or comparison.related(left[:-1], right[:-1]):
                    raise ValueError(f'public input prefix relation: {path}')
        all_rows.extend(prior)
        sources.append({'path': path, 'git_blob': blob, 'sha256': sha, 'rows': len(prior)})
    unique_count, coverage_sha = _coverage_digest(all_rows)
    return {'commit': commit, 'sources': sources, 'row_references': len(all_rows),
            'unique_rows': unique_count, 'coverage_sha256': coverage_sha,
            'pairs_compared': len(planned) * len(all_rows), 'prefix_relations': 0,
            'scope': ('Named bert/cross-model case and exposure JSON paths in this public tree; '
                      'not an unknown, private or pretraining-history claim.')}
