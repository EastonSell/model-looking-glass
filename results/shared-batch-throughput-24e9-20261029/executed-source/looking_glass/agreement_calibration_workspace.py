"""Require one unambiguous shared workspace for the agreement pilot."""

from pathlib import Path


def shared_workspace_root(repo_root):
    """Resolve the single canonical `work` ancestor, rejecting aliases."""
    repo = Path(repo_root)
    if not repo.is_absolute():
        raise ValueError('agreement checkout must be an absolute path')
    repo = repo.resolve(strict=True)
    if not repo.is_dir():
        raise ValueError('agreement checkout must be a directory')
    matches = [parent for parent in repo.parents if parent.name.casefold() == 'work']
    if len(matches) != 1 or matches[0].name != 'work':
        raise ValueError('agreement checkout must have one canonical work ancestor')
    work = matches[0]
    if (work.parent / 'work').resolve(strict=True) != work:
        raise ValueError('agreement work ancestor is not canonical')
    return work.parent
