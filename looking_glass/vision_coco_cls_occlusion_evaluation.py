"""Evaluate all fixed response-screen selections with a common-dose comparison."""
from looking_glass.vision_coco_cls_occlusion_study import capture_initials, run_case
from looking_glass.vision_coco_occlusion_study import aggregate_images, log_group_probability


def baseline_contrast(image, initial):
    category = image['category_id']
    clean = log_group_probability(initial['clean']['logits'], category)
    animal = clean - log_group_probability(initial['animal']['logits'], category)
    background = clean - log_group_probability(initial['background']['logits'], category)
    return animal - background


def run_evaluation(model, screen, screen_rows, pixels, evidence, *, numeric_guard=None):
    """Authenticate every selected baseline before the first intervention."""
    selected = screen['selected']
    ids = [row['image_id'] for row in selected]
    if len(ids) > 8 or len(set(ids)) != len(ids) or not set(ids).issubset(pixels):
        raise ValueError('selected input inventory differs')
    screen_by_id = {row['image_id']: row for row in screen_rows}
    if len(screen_by_id) != len(screen_rows) or not set(ids).issubset(screen_by_id):
        raise ValueError('selected screen-row inventory differs')
    # One call for the whole list ensures even the last clean/masked mismatch
    # stops the phase before any run_case can submit a patch.
    initial = capture_initials(model, selected, {ident: pixels[ident] for ident in ids},
                               {ident: screen['inputs'][ident] for ident in ids},
                               {ident: screen['logits'][ident] for ident in ids}, evidence,
                               numeric_guard=numeric_guard)
    if set(initial) != set(ids):
        raise ValueError('initial capture inventory differs')
    for image in selected:
        row = screen_by_id[image['image_id']]
        value = baseline_contrast(image, initial[image['image_id']])
        if (row.get('category_id') != image['category_id'] or type(row.get('occlusion_contrast')) is not float or
                row['occlusion_contrast'] < .05 or value != row['occlusion_contrast']):
            raise ValueError('selected response baseline differs')
    cases = [run_case(model, image, initial[image['image_id']], evidence, numeric_guard=numeric_guard)
             for image in selected]
    if [row.get('image_id') for row in cases] != ids:
        raise ValueError('evaluated case inventory differs')
    evaluable = []
    counts = {name: {'selected': 0, 'matchable': 0, 'limited-dose': 0,
                     'dose-ineligible': 0, 'dose-unmatchable': 0, 'evaluable': 0, 'positive': 0}
              for name in ('cat', 'dog')}
    for image, case in zip(selected, cases):
        category = image['category_id']
        if category not in (17, 18) or case.get('category_id') != category:
            raise ValueError('evaluated category differs')
        status = case['dose']['status']
        if status not in ('matchable', 'limited-dose', 'dose-ineligible', 'dose-unmatchable'):
            raise ValueError('evaluated dose status differs')
        count = counts['cat' if category == 17 else 'dog']
        count['selected'] += 1
        count[status] += 1
        if status == 'matchable':
            positive = case.get('comparisons', {}).get('positive')
            if type(positive) is not bool:
                raise ValueError('evaluable decision differs')
            evaluable.append((category, positive))
            count['evaluable'] += 1
            count['positive'] += int(positive)
    return {'cases': cases, 'selected': len(cases), 'by_species': counts, 'aggregate': aggregate_images(evaluable)}
