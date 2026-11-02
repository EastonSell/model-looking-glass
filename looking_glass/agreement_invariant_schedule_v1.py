"""Model-free, deterministic ordering for the fixed invariant-noun study."""

import hashlib

SEED = 'agreement-invariant-noun-v1:20261029'
SPLITS = ('discovery', 'evaluation')
BASE = ('u_plus', 'u_minus', 'v1_plus', 'v1_minus', 'v2_plus', 'v2_minus')
PAIR_COUNTS = ((2, 2, 3, 2, 2, 3), (2, 2, 2, 3, 3, 2))
FRAGMENT_COUNTS = ((2, 1, 0, 1, 1, 1), (1, 2, 2, 0, 0, 1))


def sort_key(split, kind, identity):
    key = f'{SEED}|{split}|{kind}|{identity}'.encode('utf-8')
    return hashlib.sha256(key).digest(), identity


def arms(pattern):
    if type(pattern) is not int or pattern not in range(6):
        raise ValueError('invalid pattern')
    words = BASE
    if pattern % 2:
        words = tuple(x[:-5] + '_minus' if x.endswith('_plus') else x[:-6] + '_plus' for x in words)
    rotate = 2 * (pattern // 2)
    return words[rotate:] + words[:rotate]


def _pairs(split, split_index):
    pairs = []
    for f in range(2):
        for c in range(3):
            for n in range(2):
                identity = f'{split}_target_f{f}_c{c}_n{n}'
                pairs.append((identity, (f + c + n) % 2,
                              (f + n + split_index) % 2))
    for f in range(2):
        identity = f'{split}_demonstrative_f{f}'
        pairs.append((identity, f, (f + split_index) % 2))
    return pairs


def _fragments(split):
    return [f'{split}_fragment_f{f}_c{c}' for f in range(2) for c in range(3)]


def _assign_pairs(split, pairs, counts):
    result = {}
    for parity in (0, 1):
        selected = sorted((x for x in pairs if x[1] == parity),
                          key=lambda x: sort_key(split, 'pair-pattern', x[0]))
        patterns = [j for j in range(6) if j % 2 == parity for _ in range(counts[j])]
        if len(selected) != len(patterns):
            raise ValueError('pair pattern count mismatch')
        result.update((item[0], pattern) for item, pattern in zip(selected, patterns))
    return result


def _assign_fragments(split, fragments, counts):
    ordered = sorted(fragments, key=lambda x: sort_key(split, 'fragment-pattern', x))
    patterns = [j for j in range(6) for _ in range(counts[j])]
    if len(ordered) != len(patterns):
        raise ValueError('fragment pattern count mismatch')
    return dict(zip(ordered, patterns))


def cases():
    """Return case IDs and arm orders, with discovery wholly before evaluation."""
    out = []
    for s, split in enumerate(SPLITS):
        pairs = _pairs(split, s)
        fragments = _fragments(split)
        pair_patterns = _assign_pairs(split, pairs, PAIR_COUNTS[s])
        fragment_patterns = _assign_fragments(split, fragments, FRAGMENT_COUNTS[s])
        pairs.sort(key=lambda x: sort_key(split, 'pair-order', x[0]))
        fragments.sort(key=lambda x: sort_key(split, 'fragment-order', x))
        for index, (identity, _flip, reverse) in enumerate(pairs, 1):
            order = ('these', 'this') if reverse else ('this', 'these')
            for determiner in order:
                out.append((f'{identity}_{determiner}', arms(pair_patterns[identity])))
            if index in (2, 4, 6, 8, 10, 12):
                fragment = fragments[index // 2 - 1]
                out.append((fragment, arms(fragment_patterns[fragment])))
    return out


def actions():
    """Expand each case into the nine fixed forward actions."""
    return [(case_id, action) for case_id, ordered in cases()
            for action in ('clean', 'noop', *ordered, 'clean_after')]
