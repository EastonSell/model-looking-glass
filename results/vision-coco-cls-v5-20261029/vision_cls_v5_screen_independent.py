"""Independent stdlib v5 screen and cross-phase clean-logit replay.

Inputs must remain quiescent. Verifies saved-byte and selection consistency,
not source/model authenticity, execution, publication timing, supervision or
pretraining independence. Screened pairs are model-conditioned. Numerical
rounding is checked at boundaries, not each intermediate operation. Optional
human-readable class labels are not independently checked.
"""
import argparse
import hashlib
import io
import re
import zipfile
import json
import math
from pathlib import Path
import struct

STATE_SHAPE = [1, 197, 192]
STATE_BYTES = 197 * 192 * 4
TOKEN_BYTES = 192 * 4
EARLY = 'vit.encoder.layer.2'
LATE = 'vit.encoder.layer.11'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_nearest_rounding():
    # Nonconstant expressions distinguish all four standard rounding modes.
    # This samples current process state, not concurrent transient changes.
    one = float.fromhex('0x1p0')
    delta = float.fromhex('0x1.8p-52')
    expected = float.fromhex('0x1.0000000000002p0')
    require(one + delta == expected and -one - delta == -expected,
            'binary64 rounding mode is not nearest-even')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bounded_read(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    require(len(raw) <= limit, f'oversized evidence file: {path.name}')
    return raw


def read_json(path, limit=4_000_000):
    raw = bounded_read(path,limit)
    require(0 < len(raw) <= limit, f'invalid JSON length: {path.name}')
    def unique_pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    return raw, value


def typed_equal(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == \
           json.dumps(right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def close(a, b, message):
    require(type(a) in (int, float) and type(b) in (int, float) and not isinstance(a, bool)
            and not isinstance(b, bool) and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), message)


def decode_word(word):
    exponent, fraction = (word >> 23) & 255, word & 0x7fffff
    require(exponent != 255, 'nonfinite binary32')
    mantissa = fraction if exponent == 0 else (1 << 23) | fraction
    value = math.ldexp(float(mantissa), -149 if exponent == 0 else exponent - 150)
    return -value if word & 0x80000000 else value


def values(raw):
    require(len(raw) % 4 == 0, 'unaligned binary32 payload')
    return [decode_word(word[0]) for word in struct.iter_unpack('<I', raw)]



def summarize(raw, reference=None):
    z = values(raw)
    require(len(z) == 1000, 'invalid logit vector')
    dog, cat = z[151:269], z[281:286]
    off = z[:151] + z[269:281] + z[286:]
    def shifted(v):
        m = max(v)
        return m, math.fsum(math.exp(x - m) for x in v)
    dm, ds = shifted(dog)
    cm, cs = shifted(cat)
    om, os = shifted(off)
    tm, ts = shifted(z)
    conditional = [math.exp(x - om) / os for x in off]
    score = {'margin': (dm - cm) + math.log(ds) - math.log(cs),
             'dog_mass': math.exp(dm - tm) * ds / ts,
             'cat_mass': math.exp(cm - tm) * cs / ts,
             'off_mass': math.exp(om - tm) * os / ts,
             'top1_id': max(range(1000), key=lambda i: z[i])}
    if reference is not None:
        require(len(reference) == len(conditional), 'off-group reference length mismatch')
        score['off_conditional_tv'] = .5 * math.fsum(abs(x - y) for x, y in zip(conditional, reference))
    return score, conditional


def check_score(stored, computed):
    for key, expected in computed.items():
        if key == 'top1_id':
            require(type(stored.get(key)) is int and stored[key] == expected, 'top-1 mismatch')
        else:
            close(stored.get(key), expected, f'score mismatch: {key}')
    for key in ('dog_mass','cat_mass','off_mass','off_conditional_tv'):
        if key in computed:
            require(0 <= stored[key] <= 1, f'probability outside [0,1]: {key}')
    if all(key in computed for key in ('dog_mass','cat_mass','off_mass')):
        close(math.fsum(stored[key] for key in ('dog_mass','cat_mass','off_mass')),1.,
              'reported probability masses do not sum to one')


def metadata_expected(row, path, tokens, action):
    events = row['metadata']['steps']
    if path is None:
        desired = [('early', 'capture', EARLY, None), ('late', 'capture', LATE, None)]
    else:
        desired = [('before_all', 'capture', path, None)]
        if tokens is None:
            desired.append(('action', action, path, None))
        else:
            for i, t in enumerate(tokens):
                desired.extend([(f'before_{i}', 'capture', path, t), (f'action_{i}', action, path, t),
                                (f'after_{i}', 'capture', path, t)])
        desired.append(('after_all', 'capture', path, None))
    require(type(events) is list and len(events) == len(desired), 'metadata action count mismatch')
    for order, (event, (name, operation, module, index)) in enumerate(zip(events, desired)):
        require(event.get('name') == name and event.get('operation') == operation and
                event.get('module_path') == module and typed_equal(event.get('order'), order) and
                typed_equal(event.get('invocation'), 0) and typed_equal(event.get('forward_index'), 0) and
                typed_equal(event.get('batch_index'), 0 if index is not None else None) and
                typed_equal(event.get('token_index'), index) and typed_equal(event.get('output_shape'), STATE_SHAPE) and
                typed_equal(event.get('selected_shape'), [192] if index is not None else STATE_SHAPE) and
                event.get('dtype') == 'torch.float32' and event.get('output_selector') == [] and
                module in event.get('module_aliases', []), 'metadata action identity/shape mismatch')


class Records:
    def __init__(self, root, count):
        self.root, self.count, self.next = root, count, 1
        self.hashes = []

    def take(self, label, path=None, tokens=None, action='capture'):
        index = self.next
        require(index <= self.count, 'missing forward')
        raw_json, row = read_json(self.root / f'forward-{index:03d}.json', 200_000)
        require(typed_equal(row.get('submission'), index) and row.get('label') == label and
                row.get('file') == f'forward-{index:03d}.bin' and
                row.get('dtype') == 'little-endian-float32', 'forward order/identity mismatch')
        packed = bounded_read(self.root / row['file'],1_000_000)
        require(len(packed) <= 1_000_000 and sha(packed) == row.get('sha256'), 'forward bytes/hash mismatch')
        layout = row.get('layout')
        require(type(layout) is dict, 'missing tensor layout')
        ordered = sorted(layout.items(), key=lambda item: item[1]['offset'])
        offset, tensors = 0, {}
        for name, entry in ordered:
            shape, start, size = entry.get('shape'), entry.get('offset'), entry.get('bytes')
            require(type(name) is str and type(shape) is list and all(type(x) is int and x > 0 for x in shape)
                    and type(start) is int and start == offset and type(size) is int
                    and size == 4 * math.prod(shape) and start + size <= len(packed), 'invalid tensor layout')
            tensors[name] = packed[start:start + size]
            offset += size
        require(offset == len(packed) and layout.get('logits', {}).get('shape') == [1, 1000], 'payload coverage mismatch')
        metadata_expected(row, path, tokens, action)
        self.hashes.append({'submission': index, 'json_sha256': sha(raw_json), 'binary_sha256': sha(packed)})
        self.next += 1
        return row, tensors


def expect_shapes(row, tensors, shapes):
    require(set(tensors) == set(shapes), 'unexpected saved tensor set')
    for name, shape in shapes.items():
        require(row['layout'][name]['shape'] == shape, f'tensor shape mismatch: {name}')


def checked_score(row, tensors, reference=None):
    computed, conditional = summarize(tensors['logits'], reference)
    check_score(row['score'], computed)
    return computed, conditional


def initial(records, pair_id, side):
    row, tensors = records.take(f'{pair_id}:{side}:initial')
    expect_shapes(row, tensors, {'logits': [1, 1000], 'early': STATE_SHAPE, 'late': STATE_SHAPE})
    for name in ('early', 'late'):
        require(all((word[0] >> 23) & 255 != 255 for word in struct.iter_unpack('<I', tensors[name])),
                f'nonfinite initial {name} state')
    score_, conditional = checked_score(row, tensors)
    return {'logits': tensors['logits'], 'early': tensors['early'], 'late': tensors['late'],
            'score': score_, 'conditional': conditional}



MODEL_STATE = 'd1bfb26b30f847366de6388385d547fc29cc3c20b330c4faad993891e165eccd'
MANIFEST_SHA = 'd2fa15c4287f15f78ff5f1a1cc1eab2551d73ee2ac876b5f7bb956f31b1d2931'


def validate_images(images, synthetic=False):
    require(type(images) is list and len(images) == 64, 'candidate count differs')
    ids, jpeg_hashes, pixel_hashes = set(), set(), set()
    for index, row in enumerate(images):
        cat, rank = (17, index) if index < 32 else (18, index - 32)
        image_id = row.get('image_id')
        require(type(image_id) is int and image_id >= 0 and image_id not in ids and
                type(row.get('category_id')) is int and row['category_id'] == cat and
                type(row.get('rank')) is int and row['rank'] == rank and
                row.get('role') == ('donor' if rank % 4 == 0 else 'target') and
                row.get('file_name') == f'{image_id:012d}.jpg', 'candidate identity or role differs')
        ids.add(image_id)
        for key, previous in (('jpeg_sha256', jpeg_hashes), ('pixel_sha256', pixel_hashes)):
            value = row.get(key)
            require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None and
                    value not in previous, 'candidate image hash differs or is duplicated')
            previous.add(value)
        require(type(row.get('jpeg_bytes')) is int and 0 < row['jpeg_bytes'] <= 5_000_000,
                'candidate byte count differs')
        if not synthetic:
            rank_hash = sha(f'vision-coco-cls-v5:{cat}:{image_id}'.encode('ascii'))
            require(row.get('selection_rank_sha256') == rank_hash, 'candidate rank hash differs')
            if rank:
                previous = images[index - 1]
                require((previous['selection_rank_sha256'], previous['image_id']) < (rank_hash, image_id),
                        'candidate order differs')
        sites = row.get('sites')
        require(type(sites) is list and sites, 'candidate has no qualified site')
        coords = []
        for site in sites:
            coord, counts = site.get('row_col'), site.get('token_mask_pixels')
            require(type(coord) is list and len(coord) == 2 and
                    all(type(x) is int and 0 <= x <= 12 for x in coord) and
                    type(counts) is list and len(counts) == 4 and
                    all(type(x) is int and 0 <= x <= 256 for x in counts), 'invalid site or occupancy')
            r, c = coord
            require(type(site.get('mask_pixels')) is int and site['mask_pixels'] == sum(counts) and
                    site['mask_pixels'] >= 512 and
                    typed_equal(site.get('token_ids'), [1 + 14*(r+dr) + c+dc for dr in range(2) for dc in range(2)]) and
                    type(site.get('squared_center_radius')) is int and
                    site['squared_center_radius'] == (r-6)**2 + (c-6)**2, 'site geometry differs')
            coords.append((r, c))
        require(len(coords) == len(set(coords)) and
                coords == sorted(coords, key=lambda p: ((p[0]-6)**2 + (p[1]-6)**2, p)),
                'site order differs')


def screen_scoring(raw, category):
    require(len(raw) == 4000 and category in (17, 18), 'screen vector or category differs')
    score, _ = summarize(raw)
    top = score['top1_id']
    top_matches = 281 <= top <= 285 if category == 17 else 151 <= top <= 268
    mass = score['cat_mass'] if category == 17 else score['dog_mass']
    reasons = ([] if top_matches else ['top1-outside-annotated-group']) + \
              ([] if mass >= .2 else ['group-mass-below-0.2'])
    return {**score, 'recognized': not reasons, 'reasons': reasons, 'logit_sha256': sha(raw)}


def reported_screen_gate(score, category):
    top = score['top1_id']
    matches = 281 <= top <= 285 if category == 17 else 151 <= top <= 268
    mass = score['cat_mass'] if category == 17 else score['dog_mass']
    reasons = ([] if matches else ['top1-outside-annotated-group']) + \
              ([] if mass >= .2 else ['group-mass-below-0.2'])
    return not reasons, reasons


def selection(images, observed):
    groups = {(category, role): [] for category in (17, 18) for role in ('target', 'donor')}
    require(len(images) == len(observed) == 64, 'screen inventory differs')
    for image, score in zip(images, observed):
        require(image['image_id'] == score['image_id'] and type(score['recognized']) is bool,
                'screen observation identity differs')
        groups[image['category_id'], image['role']].append((image, score))
    pairs, used, uses, rejects = [], set(), {}, []
    for cat, cat_score in groups[17, 'target']:
        if not cat_score['recognized']:
            continue
        for dog, dog_score in groups[18, 'target']:
            if dog['image_id'] in used or not dog_score['recognized']:
                continue
            attempted = {'cat_image_id': cat['image_id'], 'dog_image_id': dog['image_id']}
            if not (cat_score['margin'] < 0 and dog_score['margin'] > 0 and
                    dog_score['margin'] - cat_score['margin'] >= 1):
                rejects.append({**attempted, 'reason': 'baseline-margin-or-gap'})
                continue
            common = {tuple(s['row_col']) for s in cat['sites']} & {tuple(s['row_col']) for s in dog['sites']}
            if not common:
                rejects.append({**attempted, 'reason': 'no-common-qualified-site'})
                continue
            selected = None
            for coord in sorted(common, key=lambda p: ((p[0]-6)**2 + (p[1]-6)**2, p)):
                donors = []
                for category in (17, 18):
                    choices = [row for row, score in groups[category, 'donor']
                               if score['recognized'] and coord in {tuple(s['row_col']) for s in row['sites']}]
                    if not choices:
                        break
                    donors.append(min(choices, key=lambda x: (uses.get(x['image_id'], 0), x['rank'])))
                if len(donors) == 2:
                    selected = coord, donors
                    break
            if selected is None:
                rejects.append({**attempted, 'reason': 'no-recognized-same-species-donors-at-common-site'})
                continue
            (r, c), (cat_donor, dog_donor) = selected
            for donor in (cat_donor, dog_donor):
                uses[donor['image_id']] = uses.get(donor['image_id'], 0) + 1
            used.update((cat['image_id'], dog['image_id']))
            pairs.append({'pair_id': f'pair-{len(pairs):02d}', 'split': 'heldout',
                          'cat': cat, 'dog': dog, 'cat_donor': cat_donor, 'dog_donor': dog_donor,
                          'box_block_row_col': [r, c],
                          'box_tokens': [1 + 14*(r+dr) + c+dc for dr in range(2) for dc in range(2)],
                          'squared_center_radius': (r-6)**2 + (c-6)**2})
            break
        if len(pairs) == 4:
            break
    summary = {'screened_images': 64, 'target_role_images': 48, 'donor_role_images': 16,
               'selected_pairs': len(pairs), 'distinct_selected_targets': len(used),
               'distinct_selected_donors': len(uses),
               'donor_side_counts': {str(k): value for k, value in sorted(uses.items())},
               'planned_decision': 'inconclusive' if len(pairs) < 2 else 'pending-intervention',
               'interventions_planned': bool(pairs), 'examined_rejections': rejects}
    return pairs, summary


def validate_start_terminal(data, mode, manifest_hash, synthetic=False):
    start_raw, start = read_json(data / 'started.json')
    terminal_raw, terminal = read_json(data / 'terminal.json', 5_000_000)
    require(not (data / 'failure.json').exists() and
            start.get('schema') == 'vision-recognized-v4-start' and start.get('mode') == mode and
            start.get('manifest_sha256') == manifest_hash and
            terminal.get('schema') == 'vision-recognized-v4-terminal' and terminal.get('status') == 'complete' and
            terminal.get('mode') == mode and terminal.get('manifest_sha256') == manifest_hash and
            terminal.get('source_revision') == start.get('source', {}).get('revision') and
            typed_equal(terminal.get('runtime'), start.get('runtime')) and
            terminal.get('model_state_sha256_before_after') == start.get('model_state_sha256'),
            'start/terminal identity differs')
    require(type(start.get('source')) is dict and type(start['source'].get('revision')) is str and
            type(start['source'].get('sources')) is dict and type(start.get('runtime')) is dict and
            start['runtime'].get('binary64_rounding') == 'nearest-even',
            'source or runtime missing')
    if not synthetic:
        require(start.get('model_state_sha256') == MODEL_STATE, 'model state digest differs')
    return start_raw, start, terminal_raw, terminal


def validate_archive(raw, data, expected):
    require(0 < len(raw) <= 5_000_000, 'screen archive size differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and {item.filename for item in entries} == expected,
                'screen archive inventory differs')
        for item in entries:
            require(item.file_size <= 1_000_000 and archive.read(item) == bounded_read(data / item.filename,1_000_000),
                    'screen archive member differs')


def validate_custody(path, observer_path, start, manifest_hash, result_hash, terminal_hash, archive_hash, supervisor):
    custody_raw, receipt = read_json(path)
    observer_raw, observer = read_json(observer_path)
    require(receipt.get('schema') == 'vision-recognized-v4-screen-custody' and
            receipt.get('status') == 'complete' and receipt.get('manifest_sha256') == manifest_hash and
            receipt.get('source_revision') == start['source']['revision'] and
            receipt.get('screen_result_sha256') == result_hash and
            receipt.get('screen_terminal_sha256') == terminal_hash and
            receipt.get('screen_archive_sha256') == archive_hash and
            type(receipt.get('owner_exit_code')) is int and receipt['owner_exit_code'] == 0 and
            receipt.get('owner_clean') is True and receipt.get('owner_timed_out') is False and
            type(receipt.get('controller_exit_code')) is int and receipt['controller_exit_code'] == 0 and
            receipt.get('external_observer_sha256') == sha(observer_raw) and
            type(receipt.get('private_supervisor_sha256')) is str and
            re.fullmatch('[0-9a-f]{64}', receipt['private_supervisor_sha256']) is not None,
            'screen custody projection differs')
    require(observer.get('schema') == 'vision-recognized-v4-external-observer' and
            observer.get('status') == 'complete' and observer.get('mode') == 'screen' and
            type(observer.get('controller_exit_code')) is int and observer['controller_exit_code'] == 0 and
            observer.get('controller_reaped') is True and observer.get('timed_out') is False and
            observer.get('forced_stop') is False and
            observer.get('source_revision') == start['source']['revision'] and
            observer.get('supervisor_sha256') == receipt['private_supervisor_sha256'] and
            observer.get('terminal_sha256') == terminal_hash, 'screen observer differs')
    if supervisor is not None:
        supervisor_raw, value = read_json(supervisor)
        require(sha(supervisor_raw) == receipt['private_supervisor_sha256'] and
                value.get('status') == 'complete' and type(value.get('exit_code')) is int and
                value['exit_code'] == 0 and value.get('clean') is True and value.get('timed_out') is False and
                value.get('terminal_sha256') == terminal_hash, 'screen supervisor differs')
    return sha(custody_raw), sha(observer_raw)


def screen_audit(manifest_path, data, public_result, archive_path, custody_path, observer_path,
                 supervisor=None, synthetic=False):
    require_nearest_rounding()
    manifest_raw, manifest = read_json(manifest_path, 5_000_000)
    require(manifest.get('schema') == 'vision-coco-cls-compare-cases-v5', 'manifest schema differs')
    require(manifest.get('fabricated') is True if synthetic else manifest.get('fabricated') is not True,
            'synthetic flag and manifest differ')
    images = manifest.get('images')
    validate_images(images, synthetic=synthetic)
    manifest_hash = sha(manifest_raw)
    require(synthetic or manifest_hash == MANIFEST_SHA, 'fixed public manifest digest differs')
    start_raw, start, terminal_raw, terminal = validate_start_terminal(data, 'screen', manifest_hash, synthetic)
    require(type(terminal.get('submissions')) is int and terminal['submissions'] == 64 and
            type(terminal.get('saved_forwards')) is int and terminal['saved_forwards'] == 64,
            'screen forward count differs')
    expected = {'started.json', 'terminal.json', 'screen-result.json'} | {
        f'forward-{i:03d}.{ext}' for i in range(1, 65) for ext in ('json', 'bin')}
    require({p.name for p in data.iterdir()} == expected, 'screen inventory differs')
    result_raw, result = read_json(data / 'screen-result.json', 5_000_000)
    require(bounded_read(public_result,5_000_000) == result_raw and result.get('schema') == 'vision-recognized-v4-screen-result' and
            result.get('manifest_sha256') == manifest_hash and terminal.get('screen_result_sha256') == sha(result_raw),
            'public screen-result binding differs')
    observations, raw_by_id, hashes = [], {}, []
    require(type(result.get('screen')) is list and len(result['screen']) == 64, 'screen result count differs')
    for index, image in enumerate(images, 1):
        row_raw, row = read_json(data / f'forward-{index:03d}.json', 200_000)
        raw = bounded_read(data / f'forward-{index:03d}.bin',4000)
        score = screen_scoring(raw, image['category_id'])
        require(type(row.get('submission')) is int and row['submission'] == index and
                row.get('label') == f"screen:{image['image_id']}" and
                row.get('file') == f'forward-{index:03d}.bin' and row.get('sha256') == sha(raw) and
                row.get('dtype') == 'little-endian-float32' and
                typed_equal(row.get('layout'), {'logits': {'shape': [1,1000], 'offset': 0, 'bytes': 4000}}) and
                typed_equal(row.get('metadata'), {'kind':'screen', 'image_id':image['image_id']}),
                'screen forward metadata differs')
        saved = row.get('score')
        result_score = result['screen'][index-1]
        require(type(saved) is dict and set(saved) == set(score) and
                type(result_score) is dict and set(result_score) == set(score) | {'image_id'} and
                type(result_score['image_id']) is int and result_score['image_id'] == image['image_id'],
                'screen score fields differ')
        check_score(saved, {k:score[k] for k in ('margin','dog_mass','cat_mass','off_mass','top1_id')})
        check_score(result_score, {k:score[k] for k in ('margin','dog_mass','cat_mass','off_mass','top1_id')})
        for key in ('recognized','reasons','logit_sha256'):
            require(typed_equal(saved[key],score[key]) and typed_equal(result_score[key],score[key]),
                    'screen gate or saved logit hash differs')
        require(typed_equal(saved,{key:result_score[key] for key in score}) and
                reported_screen_gate(saved,image['category_id']) == (score['recognized'],score['reasons']),
                'reported screen scores contradict the gate')
        observations.append({'image_id':image['image_id'], **score})
        raw_by_id[image['image_id']] = raw
        hashes.append({'submission':index,'json_sha256':sha(row_raw),'binary_sha256':sha(raw)})
    pairs, summary = selection(images, observations)
    reported_pairs, reported_summary = selection(images,result['screen'])
    require(typed_equal(result.get('pairs'),pairs) and typed_equal(result.get('summary'),summary),
            'screen selection or summary differs')
    require(typed_equal(reported_pairs,pairs) and typed_equal(reported_summary,summary),
            'reported screen margins contradict the selection')
    archive_raw = bounded_read(archive_path,5_000_000)
    validate_archive(archive_raw, data, expected)
    custody_hash, observer_hash = validate_custody(custody_path,observer_path,start,manifest_hash,sha(result_raw),
                                                   sha(terminal_raw),sha(archive_raw),supervisor)
    require_nearest_rounding()
    return {'manifest_sha256':manifest_hash, 'start':start, 'start_sha256':sha(start_raw),
            'terminal_sha256':sha(terminal_raw), 'screen_result_sha256':sha(result_raw),
            'screen_archive_sha256':sha(archive_raw), 'screen_custody_sha256':custody_hash,
            'screen_observer_sha256':observer_hash,
            'raw_by_id':raw_by_id, 'observations':observations, 'pairs':pairs, 'summary':summary,
            'forward_hashes':hashes, 'supervisor_checked':supervisor is not None}


def cross_phase_audit(data, screen, synthetic=False):
    """Check each unique selected initial forward before any patch forward."""
    require_nearest_rounding()
    pairs = screen['pairs']
    require(1 <= len(pairs) <= 4, 'intervention requires one to four selected pairs')
    start_raw, started, terminal_raw, terminal = validate_start_terminal(
        data, 'intervention', screen['manifest_sha256'], synthetic)
    require(typed_equal(started['source'].get('sources'), screen['start']['source'].get('sources')) and
            typed_equal(started['runtime'], screen['start']['runtime']) and
            started.get('model_state_sha256') == screen['start'].get('model_state_sha256') and
            terminal.get('screen_result_sha256') == screen['screen_result_sha256'] and
            terminal.get('screen_terminal_sha256') == screen['terminal_sha256'] and
            terminal.get('screen_archive_sha256') == screen['screen_archive_sha256'],
            'screen/intervention linkage differs')
    if not synthetic:
        prefix = 'results/vision-coco-cls-v5-20261029/'
        expected = {prefix + 'screen-result.json': screen['screen_result_sha256'],
                    prefix + 'screen-evidence.zip': screen['screen_archive_sha256'],
                    prefix + 'screen-custody.json': screen['screen_custody_sha256'],
                    prefix + 'screen-observer.json': screen['screen_observer_sha256']}
        require(typed_equal(started['source'].get('public_handoff'), expected),
                'parent-bound public handoff differs')
    count = terminal.get('submissions')
    require(type(count) is int and 0 < count <= 256 and
            type(terminal.get('saved_forwards')) is int and terminal['saved_forwards'] == count,
            'intervention forward count differs')
    unique = {}
    records = Records(data, count)
    for pair in pairs:
        for role in ('cat', 'dog', 'cat_donor', 'dog_donor'):
            ident = pair[role]['image_id']
            if ident in unique:
                continue
            clean = initial(records, pair['pair_id'], role)
            require(clean['logits'] == screen['raw_by_id'][ident], 'cross-phase clean logits differ')
            unique[ident] = {'image_id': ident, 'first_pair_id': pair['pair_id'], 'first_role': role,
                             'logit_sha256': sha(clean['logits'])}
    require_nearest_rounding()
    return {'start_sha256': sha(start_raw), 'terminal_sha256': sha(terminal_raw),
            'initial_forward_count': len(unique), 'recorded_intervention_forward_count': count,
            'initial_records': list(unique.values()), 'initial_forward_hashes': records.hashes,
            'scope': 'only the initial clean forwards and screen linkage were checked; '
                     'later interventions, doses and decisions require a separate audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--screen-data', type=Path, required=True)
    parser.add_argument('--public-screen-result', type=Path, required=True)
    parser.add_argument('--public-screen-archive', type=Path, required=True)
    parser.add_argument('--public-screen-custody', type=Path, required=True)
    parser.add_argument('--public-screen-observer', type=Path, required=True)
    parser.add_argument('--screen-supervisor', type=Path)
    parser.add_argument('--intervention-data', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    screen = screen_audit(args.manifest, args.screen_data, args.public_screen_result,
                          args.public_screen_archive, args.public_screen_custody,
                          args.public_screen_observer, args.screen_supervisor, args.synthetic)
    cross = None if args.intervention_data is None else cross_phase_audit(args.intervention_data, screen, args.synthetic)
    report = {'schema': 'independent-vision-cls-v5-screen-audit-v1',
              'scope': 'Saved-byte screen, selection and initial cross-phase consistency only; no source, '
                       'model, execution, supervision, chronology or pretraining attestation. Inputs must remain '
                       'quiescent. Top-one IDs are checked but optional label strings are not. '
                       'Selected pairs are model-conditioned.',
              'synthetic': args.synthetic, 'manifest_sha256': screen['manifest_sha256'],
              'screen_start_sha256': screen['start_sha256'], 'screen_terminal_sha256': screen['terminal_sha256'],
              'screen_result_sha256': screen['screen_result_sha256'],
              'screen_archive_sha256': screen['screen_archive_sha256'],
              'screen_custody_sha256': screen['screen_custody_sha256'],
              'screen_observer_sha256': screen['screen_observer_sha256'],
              'screen_supervisor_checked': screen['supervisor_checked'],
              'screen_forwards': 64, 'screen_forward_hashes': screen['forward_hashes'],
              'screen_summary': screen['summary'],
              'screen_gate_counts': {'recognized': sum(r['recognized'] for r in screen['observations']),
                                     'unrecognized': sum(not r['recognized'] for r in screen['observations'])},
              'selected_pairs': [{'pair_id': p['pair_id'], 'cat_id': p['cat']['image_id'],
                                  'dog_id': p['dog']['image_id'], 'cat_donor_id': p['cat_donor']['image_id'],
                                  'dog_donor_id': p['dog_donor']['image_id'], 'site': p['box_block_row_col']}
                                 for p in screen['pairs']],
              'initial_cross_phase': cross}
    raw = (json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.report.open('xb') as stream:
        if stream.write(raw) != len(raw):
            raise OSError('short report write')
    print(json.dumps({'report_sha256': sha(raw), 'screened': 64, 'selected': len(screen['pairs']),
                      'cross_phase_initials': None if cross is None else cross['initial_forward_count']}))


if __name__ == '__main__':
    main()


