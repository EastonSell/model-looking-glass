"""Model-free scoring and deterministic pairing for the v4 clean screen."""
from __future__ import annotations

import hashlib
import math
import struct


def check_rounding():
    """Detect persistent non-nearest host binary64 rounding without changing it."""
    half = 2.0 ** -53
    unit = 2.0 ** -52
    if 1.0 + half != 1.0 or -1.0 - half != -1.0 or 1.0 + 3.0 * half != 1.0 + 2.0 * unit:
        raise ValueError('binary64 round-to-nearest is unavailable')


def _binary32(bits):
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    if exponent == 255:
        raise ValueError('nonfinite screen logit')
    magnitude = math.ldexp(fraction, -149) if exponent == 0 else math.ldexp((1 << 23) + fraction, exponent - 150)
    return -magnitude if bits >> 31 else magnitude


def screen_score(raw, category):
    """Derive the complete fixed screen score from exactly 1,000 float32 logits."""
    check_rounding()
    if type(raw) is not bytes or len(raw) != 4000 or category not in (17, 18):
        raise ValueError('screen logit vector or category differs')
    values = [_binary32(bits) for bits in struct.unpack('<1000I', raw)]
    def shifted(group):
        maximum = max(group)
        return maximum, math.fsum(math.exp(value - maximum) for value in group)
    dog_max, dog_sum = shifted(values[151:269])
    cat_max, cat_sum = shifted(values[281:286])
    off_max, off_sum = shifted([values[i] for i in range(1000) if not (151 <= i <= 268 or 281 <= i <= 285)])
    total_max, total_sum = shifted(values)
    top = max(range(1000), key=lambda index: values[index])
    result = {'margin': ((dog_max - cat_max) + math.log(dog_sum) - math.log(cat_sum)),
              'dog_mass': math.exp(dog_max - total_max) * dog_sum / total_sum,
              'cat_mass': math.exp(cat_max - total_max) * cat_sum / total_sum,
              'off_mass': math.exp(off_max - total_max) * off_sum / total_sum,
              'top1_id': top}
    matches = 281 <= top <= 285 if category == 17 else 151 <= top <= 268
    mass = result['cat_mass'] if category == 17 else result['dog_mass']
    reasons = ([] if matches else ['top1-outside-annotated-group']) + ([] if mass >= .2 else ['group-mass-below-0.2'])
    result.update(recognized=not reasons, reasons=reasons, logit_sha256=hashlib.sha256(raw).hexdigest())
    return result


def select_pairs(images, screen, *, limit=4):
    """Use only fixed roles and saved scores; preserve every examined rejection."""
    check_rounding()
    if type(limit) is not int or not 1 <= limit <= 4 or len(images) != len(screen):
        raise ValueError('screen inventory or pair limit differs')
    seen, ranks = set(), {17: [], 18: []}
    groups = {17: {'target': [], 'donor': []}, 18: {'target': [], 'donor': []}}
    for image, observed in zip(images, screen):
        identity, category, role = image['image_id'], image['category_id'], image['role']
        if (type(identity) is not int or identity in seen or category not in groups or role not in groups[category] or
                type(image.get('rank')) is not int or image['rank'] < 0 or
                role != ('donor' if image['rank'] % 4 == 0 else 'target') or
                observed.get('image_id') != identity or type(observed.get('recognized')) is not bool or
                type(observed.get('margin')) not in (int, float) or not math.isfinite(observed['margin'])):
            raise ValueError('screen identities or roles differ')
        seen.add(identity)
        ranks[category].append(image['rank'])
        groups[category][role].append((image, observed))
    if any(sorted(ranks[category]) != list(range(len(ranks[category]))) for category in (17, 18)):
        raise ValueError('candidate ranks differ')
    for species in groups.values():
        for role in species:
            species[role].sort(key=lambda item: item[0]['rank'])
    use, used_targets, pairs, rejected = {}, set(), [], []
    for cat, cs in groups[17]['target']:
        if not cs['recognized']:
            continue
        for dog, ds in groups[18]['target']:
            if dog['image_id'] in used_targets or not ds['recognized']:
                continue
            identity = {'cat_image_id': cat['image_id'], 'dog_image_id': dog['image_id']}
            if not (cs['margin'] < 0 and ds['margin'] > 0 and ds['margin'] - cs['margin'] >= 1):
                rejected.append({**identity, 'reason': 'baseline-margin-or-gap'})
                continue
            common = {tuple(site['row_col']) for site in cat['sites']} & {tuple(site['row_col']) for site in dog['sites']}
            if not common:
                rejected.append({**identity, 'reason': 'no-common-qualified-site'})
                continue
            chosen = None
            for site in sorted(common, key=lambda p: ((p[0] - 6) ** 2 + (p[1] - 6) ** 2, p)):
                donors = []
                for category in (17, 18):
                    available = [(row, sr) for row, sr in groups[category]['donor']
                                 if sr['recognized'] and site in {tuple(s['row_col']) for s in row['sites']}]
                    if not available:
                        break
                    donors.append(min(available, key=lambda item: (use.get(item[0]['image_id'], 0), item[0]['rank']))[0])
                if len(donors) == 2:
                    chosen = site, donors
                    break
            if chosen is None:
                rejected.append({**identity, 'reason': 'no-recognized-same-species-donors-at-common-site'})
                continue
            site, (cat_donor, dog_donor) = chosen
            for donor in (cat_donor, dog_donor):
                use[donor['image_id']] = use.get(donor['image_id'], 0) + 1
            used_targets.update((cat['image_id'], dog['image_id']))
            row, col = site
            pairs.append({'pair_id': f'pair-{len(pairs):02d}', 'split': 'heldout',
                          'cat': cat, 'dog': dog, 'cat_donor': cat_donor, 'dog_donor': dog_donor,
                          'box_block_row_col': [row, col],
                          'box_tokens': [1 + 14 * (row + dr) + col + dc for dr in range(2) for dc in range(2)],
                          'squared_center_radius': (row - 6) ** 2 + (col - 6) ** 2})
            break
        if len(pairs) == limit:
            break
    summary = {'screened_images': len(images), 'target_role_images': sum(r['role'] == 'target' for r in images),
               'donor_role_images': sum(r['role'] == 'donor' for r in images), 'selected_pairs': len(pairs),
               'distinct_selected_targets': len(used_targets), 'distinct_selected_donors': len(use),
               'donor_side_counts': {str(key): count for key, count in sorted(use.items())},
               'planned_decision': 'inconclusive' if len(pairs) < 2 else 'pending-intervention',
               'interventions_planned': bool(pairs),
               'examined_rejections': rejected}
    return pairs, summary


def aggregate_recognized(rows):
    # All selected baselines were screened and must match the repeated clean
    # logits. A later baseline-ineligible row is a contradiction, not an
    # ordinary denominator exclusion.
    allowed = {'positive', 'negative', 'dose-ineligible', 'dose-unmatchable'}
    if (len(rows) > 4 or [r.get('pair_id') for r in rows] != [f'pair-{i:02d}' for i in range(len(rows))] or
            any(r.get('split') != 'heldout' or r.get('status') not in allowed for r in rows)):
        raise ValueError('selected-pair results differ')
    evaluable = [r for r in rows if r['status'] in ('positive', 'negative')]
    positive = sum(r['status'] == 'positive' for r in evaluable)
    status = ('inconclusive' if len(evaluable) < 2 else
              'positive' if 4 * positive >= 3 * len(evaluable) else 'negative')
    return {'status': status, 'selected': len(rows), 'evaluable': len(evaluable), 'positive': positive}
