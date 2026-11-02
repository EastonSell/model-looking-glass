"""Frozen, metadata-free discovery selector for the public injection study."""

import hashlib
import math


def fixed_random_site(cell):
    if cell not in ('A', 'B'):
        raise ValueError('cell must be A or B')
    value = hashlib.sha256(b'public-known-injection-v1-random\0' + cell.encode('ascii')).digest()[0]
    return 1 + value % 4


def discovery_selection(rows):
    """Select from exactly eight (clean margin, injection margin, four patch margins).

    This function does not accept any site truth, text, cell label or held-out data.
    """
    if type(rows) not in (tuple, list) or len(rows) != 8:
        raise ValueError('exactly eight numeric rows required')
    reductions = [[], [], [], []]
    signed = [[], [], [], []]
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 3:
            raise ValueError('row must contain exactly three numeric fields')
        clean, injected, patched = row
        if type(patched) not in (tuple, list) or len(patched) != 4:
            raise ValueError('exactly four candidate margins required')
        values = (clean, injected, *patched)
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ValueError('all margins must be finite binary64 values')
        baseline_error = abs(injected - clean)
        for index, value in enumerate(patched):
            reductions[index].append(baseline_error - abs(value - clean))
            signed[index].append(value - injected)
    scores = [math.fsum(values) / 8.0 for values in reductions]
    predictions = [math.fsum(values) / 8.0 for values in signed]
    if not all(math.isfinite(value) for value in (*scores, *predictions)):
        raise ValueError('nonfinite discovery aggregate')
    ranking = sorted(range(1, 5), key=lambda site: (-scores[site - 1], site))
    best = scores[ranking[0] - 1]
    return {
        'scores': scores,
        'ranking': ranking,
        'tied_max_sites': [site for site in range(1, 5) if scores[site - 1] == best],
        'selected_site': ranking[0] if best > 0.0 else None,
        'signed_predictions': predictions,
    }
