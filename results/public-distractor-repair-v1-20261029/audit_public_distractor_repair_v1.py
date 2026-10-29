"""Independent model-free saved-byte review of the public distractor study.

This reader checks saved observations, not process execution, publication
chronology, model provenance, or unknown input exposure. It imports neither the
producer nor its analyzer. The positional fixture interface accepts tiny fake
models; the same checks apply to the fixed public model dimensions.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import struct
from pathlib import Path


MAP = {
    1: ((2, 't', 'downstream'), (0, 'f', 'distractor'), (1, 't', 'source'), (0, 't', 'upstream')),
    2: ((2, 't', 'source'), (1, 't', 'upstream'), (0, 'f', 'distractor'), (3, 't', 'downstream')),
    3: ((2, 't', 'upstream'), (4, 't', 'downstream'), (3, 't', 'source'), (0, 'f', 'distractor')),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nearest():
    if os.name == 'nt':
        get = ctypes.CDLL('msvcrt')._controlfp
        get.argtypes, get.restype = [ctypes.c_uint, ctypes.c_uint], ctypes.c_uint
        require((get(0, 0) & 0x300) == 0, 'rounding mode is not nearest-even')
    else:
        get = ctypes.CDLL(None).fegetround
        get.argtypes, get.restype = [], ctypes.c_int
        require(get() == 0, 'rounding mode is not nearest-even')
    require(1.0 + float.fromhex('0x1p-53') == 1.0, 'binary64 rounding probe')


def parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def floats(raw):
    require(len(raw) % 4 == 0, 'non-float-sized tensor')
    values = [x[0] for x in struct.iter_unpack('<f', raw)]
    require(all(math.isfinite(x) for x in values), 'nonfinite float32')
    return values


def vector(full, token, hidden):
    start = token * hidden * 4
    return full[start:start + hidden * 4]


def replace(full, token, value, hidden):
    start = token * hidden * 4
    require(len(value) == hidden * 4, 'replacement dimension')
    return full[:start] + value + full[start + len(value):]


def assert_add(before, after, token, hidden):
    start = token * hidden * 4
    old = before[start:start+4]
    value = struct.unpack('<f', old)[0]
    new = struct.pack('<f', value + 4.0)
    require(new != old and math.isfinite(struct.unpack('<f', new)[0]), 'undelivered add')
    require(after == before[:start] + new + before[start+4:], 'add landing or untouched bit')


def distance(left, right):
    require(len(left) == len(right) and len(left) % 4 == 0, 'distance shape')
    return math.sqrt(math.fsum((a[0]-b[0])**2 for a,b in
                               zip(struct.iter_unpack('<f', left), struct.iter_unpack('<f', right))))


def score(raw, vocab):
    require(len(raw) == vocab * 4, 'score dimension')
    return struct.unpack_from('<f', raw, 318*4)[0] - struct.unpack_from('<f', raw, 389*4)[0]


def schedule(cases):
    out = []
    for case in cases:
        cid = case['case_id']
        out.extend([(cid, 'clean', None, None), (cid, 'null-self', None, None),
                    (cid, 'decoy', None, None)])
        for layer in (1, 2, 3):
            out.extend([(cid, 'target', layer, None), (cid, 'combined', layer, None)])
            out.extend((cid, 'patch', layer, q) for q in range(1, 5))
            out.extend([(cid, 'rotation', layer, None), (cid, 'self', layer, None)])
        out.append((cid, 'clean-repeat', None, None))
    return out


def plan(kind, layer, q):
    if kind == 'null-self':
        return [('before', 'capture', 0, None), ('null', 'null', 0, 't'),
                ('self_source', 'capture', 0, 't'), ('self_patch', 'patch', 0, 't'),
                ('after', 'capture', 0, None)]
    candidate = MAP[layer][q-1] if kind == 'patch' else None
    out = []
    for block in range(5):
        if block == 0 and kind != 'target':
            out += [('decoy_before', 'capture', block, None), ('decoy_add', 'add', block, 'f'),
                    ('decoy_after', 'capture', block, None)]
        if block == layer and kind != 'decoy':
            out += [('target_before', 'capture', block, None), ('target_add', 'add', block, 't'),
                    ('target_after', 'capture', block, None)]
        if candidate and block == candidate[0]:
            out += [('patch_before', 'capture', block, None), ('patch', 'patch', block, candidate[1]),
                    ('patch_after', 'capture', block, None)]
        if kind == 'rotation' and block == layer + 1:
            out += [('patch_before', 'capture', block, None), ('patch', 'patch', block, 't'),
                    ('patch_after', 'capture', block, None)]
        if kind == 'self' and block == layer + 1:
            out += [('self_source', 'capture', block, 't'), ('self_before', 'capture', block, None),
                    ('self_patch', 'patch', block, 't'), ('self_after', 'capture', block, None)]
        out += [(f'tap_{block}_t', 'capture', block, 't'),
                (f'tap_{block}_f', 'capture', block, 'f')]
    return out


def check_hooks(hooks, action, count, hidden, phase):
    cid, kind, layer, q = action
    require(type(hooks) is list, 'hook list')
    shapes = {}
    if kind in ('clean', 'clean-repeat'):
        require(len(hooks) == 5, 'trace count')
        run_id = f"{phase}-{cid}-{'clean' if kind == 'clean' else 'repeat'}"
        for i, h in enumerate(hooks):
            require(h.get('sequence') == i and h.get('layer_index') == i and h.get('invocation') == 0
                    and h.get('forward_index') == 0 and h.get('module_kind') == 'mlp'
                    and h.get('module_path') == f'transformer.h.{i}.mlp'
                    and h.get('batch_positions') == [0] and h.get('token_positions') == [count-2, count-1]
                    and h.get('shape') == [1, count, hidden] and h.get('saved_shape') == [1, 2, hidden]
                    and h.get('dtype') == 'torch.float32' and h.get('device') == 'cpu'
                    and h.get('generation', {}).get('use_cache') is False
                    and h.get('identity', {}).get('run_id') == run_id, 'trace metadata')
            shapes[f'tap_{i}_t'] = [hidden]
            shapes[f'tap_{i}_f'] = [hidden]
        return shapes
    expected = plan(kind, layer, q)
    require(len(hooks) == len(expected), 'plan length')
    previous = {}
    for i, (name, op, block, pos) in enumerate(expected):
        h = hooks[i]
        token = {'t': count-2, 'f': count-1, None: None}[pos]
        shape = [hidden] if pos else [1, count, hidden]
        prior = previous.setdefault(block, [])
        require(h.get('order') == i and h.get('name') == name and h.get('operation') == op
                and h.get('module_path') == f'transformer.h.{block}.mlp'
                and h.get('invocation') == 0 and h.get('forward_index') == 0
                and h.get('device') == 'cpu' and h.get('dtype') == 'torch.float32'
                and h.get('output_shape') == [1, count, hidden] and h.get('selected_shape') == shape
                and h.get('batch_index') == (0 if pos else None) and h.get('token_index') == token
                and h.get('preceding_same_site_actions') == prior
                and h.get('source_capture') == ('self_source' if name == 'self_patch' else None),
                'hook ordering or metadata')
        prior.append(name)
        if op == 'capture':
            shapes[name] = shape
    if kind == 'rotation':
        shapes['rotation_planned'] = [hidden]
    if kind == 'patch' and MAP[layer][q-1][2] == 'downstream':
        shapes['realized_displacement'] = [hidden]
    return shapes


def expected_checks(action):
    _, kind, layer, q = action
    if kind == 'clean': return {'trace_complete': True}
    if kind == 'clean-repeat': return {'logits_equal_clean': True, 'vectors_equal_clean': True}
    if kind == 'null-self': return {'before_after_equal': True, 'self_source_matches': True, 'logits_equal_clean': True}
    if kind == 'decoy': return {'landing_valid': True, 'scored_logits_equal_clean': True}
    if kind == 'combined': return {'landing_valid': True, 'scored_logits_equal_target': True}
    if kind == 'self': return {'landing_valid': True, 'self_source_matches': True, 'logits_equal_combined': True}
    checks = {'landing_valid': True}
    if kind == 'patch':
        role = MAP[layer][q-1][2]
        if role == 'source': checks['scored_logits_equal_clean'] = True
        if role in ('upstream', 'distractor'): checks['scored_logits_equal_combined'] = True
        if role == 'upstream': checks['upstream_donor_equal'] = True
        if role == 'downstream': checks['downstream_base_equal_combined'] = True
    return checks


def read_phase(root, cases, hidden, vocab, phase):
    expected = schedule(cases)
    by_id = {c['case_id']: c for c in cases}
    require(len(by_id) == len(cases), 'duplicate case')
    expected_files = {'events.jsonl', 'phase-summary.json'} | {cid + ext for cid in by_id for ext in ('.f32','.f64')}
    require({x.name for x in root.iterdir()} == expected_files, 'phase inventory')
    events = root/'events.jsonl'
    require(events.stat().st_size <= 16*1024*1024, 'metadata cap')
    positions = {cid+ext: 0 for cid in by_id for ext in ('.f32','.f64')}
    result = {}
    active = None
    descriptors = []
    tensors = {}
    shapes = {}
    hooks = None
    recorded = None
    displacement_norm = None
    with events.open('rb') as stream:
        for raw in stream:
            require(len(raw) <= 48*1024, 'metadata event bound')
            row = parse(raw)
            typ = row.get('type')
            ordinal = len(result)
            if typ == 'start':
                require(active is None and ordinal < len(expected), 'unexpected start')
                details = row.get('details')
                require(type(details) is dict, 'details')
                candidate = details.get('candidate')
                require(candidate is None or candidate in ('Q1','Q2','Q3','Q4'), 'candidate ID')
                active = (row.get('case_id'), row.get('action'), details.get('layer'),
                          int(candidate[1]) if candidate else None)
                require(active == expected[ordinal] and row.get('ordinal') == ordinal, 'schedule')
                wanted_details = ({} if active[2] is None else
                                  {'layer': active[2], **({'candidate': candidate} if candidate else {})})
                require(details == wanted_details, 'action details')
                tensors, shapes, descriptors, hooks, recorded, displacement_norm = {}, {}, [], None, None, None
            elif typ == 'tensor':
                require(active is not None and hooks is None and row.get('ordinal') == ordinal
                        and (row.get('case_id'), row.get('action')) == active[:2], 'orphan tensor')
                d = row['tensor']; cid = active[0]
                shape, length, name = d['shape'], d['length'], d['name']
                is64 = name == 'realized_displacement'
                size, ext, dtype = (8,'.f64','float64-le') if is64 else (4,'.f32','float32-le')
                filename = cid+ext
                require(d['file'] == filename and d['offset'] == positions[filename] and d['dtype'] == dtype
                        and type(shape) is list and shape and all(type(x) is int and x > 0 for x in shape)
                        and type(length) is int and length == size*math.prod(shape) and name not in tensors,
                        'tensor descriptor')
                with (root/d['file']).open('rb') as source:
                    source.seek(d['offset'])
                    payload = source.read(length)
                require(len(payload) == length and hashlib.sha256(payload).hexdigest() == d['sha256'], 'tensor hash')
                require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<d' if is64 else '<f', payload)),
                        'nonfinite tensor')
                tensors[name], shapes[name] = payload, shape
                descriptors.append(d)
                positions[filename] += length
            elif typ == 'raw-recorded':
                require(active is not None and hooks is None and row.get('ordinal') == ordinal
                        and (row.get('case_id'), row.get('action')) == active[:2]
                        and row.get('tensors') == descriptors, 'recorded metadata')
                hooks, recorded = row['hook_events'], row.get('rotation_info')
                displacement_norm = row.get('displacement_norm')
            elif typ == 'complete':
                require(active is not None and hooks is not None and row.get('ordinal') == ordinal
                        and (row.get('case_id'), row.get('action')) == active[:2], 'completion identity')
                checks = expected_checks(active)
                require(row.get('checks') == checks and all(type(x) is bool for x in row['checks'].values()),
                        'mandatory controls')
                wanted = {'logits': [2, vocab]}
                wanted.update(check_hooks(hooks, active, len(by_id[active[0]]['token_ids']), hidden, phase))
                require(shapes == wanted, 'tensor inventory')
                require((type(recorded) is dict) if active[1] == 'rotation' else recorded is None, 'rotation record')
                is_downstream = active[1] == 'patch' and MAP[active[2]][active[3]-1][2] == 'downstream'
                require((type(displacement_norm) is float and math.isfinite(displacement_norm))
                        if is_downstream else displacement_norm is None, 'displacement norm record')
                result[active] = (tensors, {'rotation_info': recorded, 'displacement_norm': displacement_norm})
                active = None
            else:
                require(False, 'unexpected or incomplete journal event')
    require(active is None and len(result) == len(expected), 'incomplete action schedule')
    require(sum(positions.values()) <= 256*1024*1024, 'raw cap')
    require(all((root/name).stat().st_size == size for name, size in positions.items()), 'unaccounted raw bytes')
    summary = parse((root/'phase-summary.json').read_bytes())
    require(summary == {'phase': phase, 'cases': len(cases), 'completed_actions': len(expected),
                        'raw_bytes': sum(positions.values()), 'metadata_bytes': events.stat().st_size,
                        'status': 'complete'}, 'phase summary')
    return result


def audit_case(data, case, hidden, vocab):
    cid = case['case_id']; t, f = len(case['token_ids'])-2, len(case['token_ids'])-1
    def get(kind, layer=None, q=None):
        return data[(cid, kind, layer, q)]
    clean, _ = get('clean'); repeat, _ = get('clean-repeat'); null, _ = get('null-self')
    require(clean['logits'] == repeat['logits'] == null['logits'], 'clean/null/repeat logits')
    require(null['before'] == null['after'] and vector(null['before'], t, hidden) == null['self_source'], 'null/self landing')
    require(all(vector(null['before'], index, hidden) == clean[f'tap_0_{pos}']
                for index,pos in ((t,'t'), (f,'f'))), 'null clean source')
    require(all(clean[f'tap_{b}_{p}'] == repeat[f'tap_{b}_{p}'] for b in range(5) for p in ('t','f')),
            'clean-repeat taps')
    decoy, _ = get('decoy')
    assert_add(decoy['decoy_before'], decoy['decoy_after'], f, hidden)
    require(all(vector(decoy['decoy_before'], index, hidden) == clean[f'tap_0_{pos}']
                for index,pos in ((t,'t'), (f,'f'))), 'decoy clean source')
    require(all(vector(decoy['decoy_after'], index, hidden) == decoy[f'tap_0_{pos}']
                for index,pos in ((t,'t'), (f,'f'))), 'decoy post-add taps')
    require(decoy['logits'][:4*vocab] == clean['logits'][:4*vocab], 'future causal equality')
    result = {}
    for layer in (1,2,3):
        target, _ = get('target', layer); combined, _ = get('combined', layer)
        base = combined[f'tap_{layer+1}_t']; donor = clean[f'tap_{layer+1}_t']
        for x, kind in ((target, 'target'), (combined, 'combined')):
            if kind == 'combined': assert_add(x['decoy_before'], x['decoy_after'], f, hidden)
            assert_add(x['target_before'], x['target_after'], t, hidden)
            require(vector(x['target_after'], t, hidden) == x[f'tap_{layer}_t'], 'target post-add tap')
        require(target['logits'][:4*vocab] == combined['logits'][:4*vocab], 'combined causal equality')
        by_q = []
        for q, (block, pos, role) in enumerate(MAP[layer], 1):
            patch, patch_metadata = get('patch', layer, q)
            assert_add(patch['decoy_before'], patch['decoy_after'], f, hidden)
            assert_add(patch['target_before'], patch['target_after'], t, hidden)
            token = t if pos == 't' else f
            replacement = clean[f'tap_{block}_{pos}']
            require(patch['patch_after'] == replace(patch['patch_before'], token, replacement, hidden), 'candidate landing')
            require(patch[f'tap_{block}_{pos}'] == replacement, 'candidate tap')
            if role == 'downstream':
                before = vector(patch['patch_before'], t, hidden)
                require(before == base, 'downstream clean patch base differs from combined')
                realized = [a-b for a,b in zip(floats(replacement), floats(before))]
                require(patch['realized_displacement'] == b''.join(struct.pack('<d', x) for x in realized),
                        'saved binary64 displacement')
                norm = math.sqrt(math.fsum(x*x for x in realized))
                require(math.isclose(patch_metadata['displacement_norm'], norm, rel_tol=1e-14, abs_tol=1e-15),
                        'clean-patch displacement norm')
            if role == 'upstream':
                require(vector(patch['patch_before'], t, hidden) == replacement, 'upstream no-op donor')
            if role == 'source':
                require(patch['logits'][:4*vocab] == clean['logits'][:4*vocab], 'source scored restoration')
            if role in ('upstream','distractor'):
                require(patch['logits'][:4*vocab] == combined['logits'][:4*vocab], 'upstream/distractor scored equality')
            by_q.append(patch)
        rotated, metadata = get('rotation', layer)
        info = metadata['rotation_info']
        assert_add(rotated['decoy_before'], rotated['decoy_after'], f, hidden)
        assert_add(rotated['target_before'], rotated['target_after'], t, hidden)
        require(vector(rotated['patch_before'], t, hidden) == base, 'rotation base')
        b, c = floats(base), floats(donor)
        d = [x-y for x,y in zip(c,b)]
        planned = b''.join(struct.pack('<f', b[i]+d[(i-1)%hidden]) for i in range(hidden))
        require(rotated['rotation_planned'] == planned, 'rotation planned vector')
        require(rotated['patch_after'] == replace(rotated['patch_before'], t, planned, hidden), 'rotation landing')
        require(rotated[f'tap_{layer+1}_t'] == planned, 'rotation tap')
        actual = [z-y for z,y in zip(floats(planned), b)]
        norm = math.sqrt(math.fsum(x*x for x in d)); norm_rot = math.sqrt(math.fsum(x*x for x in actual))
        cosine = math.fsum(x*y for x,y in zip(d,actual))/(norm*norm_rot) if norm > 0 and norm_rot > 0 else None
        matched = bool(norm > 0 and norm_rot > 0 and abs(norm_rot-norm)/norm <= .001)
        require(set(info) == {'clean_norm','rotated_norm','cosine','matched'} and type(info['matched']) is bool
                and info['matched'] == matched, 'rotation report fields')
        for key, value in [('clean_norm',norm), ('rotated_norm',norm_rot), ('cosine',cosine)]:
            recorded = info[key]
            require((recorded is None and value is None) or
                    (type(recorded) is float and value is not None and math.isclose(recorded, value, rel_tol=1e-14, abs_tol=1e-15)),
                    'rotation numeric report')
        self_control, _ = get('self', layer)
        assert_add(self_control['decoy_before'], self_control['decoy_after'], f, hidden)
        assert_add(self_control['target_before'], self_control['target_after'], t, hidden)
        require(self_control['self_before'] == self_control['self_after']
                and self_control['self_source'] == vector(self_control['self_before'], t, hidden)
                and self_control['logits'] == combined['logits'], 'downstream self control')
        logits = clean['logits'][:4*vocab]
        combined_logits = combined['logits'][:4*vocab]
        d_combined = distance(logits, combined_logits)
        downstream_q = next(i for i,item in enumerate(MAP[layer]) if item[2] == 'downstream')
        effects = [score(x['logits'][:4*vocab], vocab)-score(combined_logits,vocab) for x in by_q]
        magnitude = [distance(clean[f'tap_{b}_{pos}'], combined[f'tap_{b}_{pos}']) for b,pos,_ in MAP[layer]]
        changed = [(block,pos) for block in range(5) for pos in ('t','f')
                   if clean[f'tap_{block}_{pos}'] != combined[f'tap_{block}_{pos}']]
        result[layer] = {'effects': effects, 'magnitude': magnitude,
                         'first_changed': list(changed[0]) if changed else None,
                         'D_combined': d_combined,
                         'D_target': distance(logits, target['logits'][:4*vocab]),
                         'target_margin_effect': score(target['logits'][:4*vocab],vocab)-score(logits,vocab),
                         'Q_distances': [distance(logits, item['logits'][:4*vocab]) for item in by_q],
                         'decoy_final_distance': distance(clean['logits'][4*vocab:], decoy['logits'][4*vocab:]),
                         'combined_final_vs_target': distance(combined['logits'][4*vocab:], target['logits'][4*vocab:]),
                         'D_downstream': distance(logits, by_q[downstream_q]['logits'][:4*vocab]),
                         'D_rotated': distance(logits, rotated['logits'][:4*vocab]),
                         'rotation': {'clean_norm': norm, 'rotated_norm': norm_rot, 'cosine': cosine, 'matched': matched}}
    return result


def run(root, cases_path, hidden, vocab, phase_prefix=''):
    nearest()
    require(phase_prefix in ('', 'run-'), 'unsupported phase directory prefix')
    cases = parse(cases_path.read_bytes())['cases']
    require(len(cases) == 16 and all(len(c['token_ids']) >= 3 for c in cases), 'case inventory')
    all_results, evidence = {}, {}
    for phase in ('discovery','held-out'):
        selected = [c for c in cases if c['split'] == phase]
        require(len(selected) == 8, 'phase case count')
        directory = root/(phase_prefix+phase)
        saved = read_phase(directory, selected, hidden, vocab, phase)
        all_results[phase] = {c['case_id']: audit_case(saved,c,hidden,vocab) for c in selected}
        del saved
        raw_files = {}
        for case in selected:
            cid = case['case_id']
            for ext in ('.f32','.f64'):
                name = cid+ext; raw = (directory/name).read_bytes()
                raw_files[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        events = (directory/'events.jsonl').read_bytes()
        evidence[phase] = {'events_sha256': hashlib.sha256(events).hexdigest(),
                           'actions': len(selected)*28, 'raw_files': raw_files,
                           'raw_bytes': sum(x['bytes'] for x in raw_files.values())}
    summary = {}
    for layer in (1,2,3):
        discovery = [x[layer] for x in all_results['discovery'].values()]
        held = [x[layer] for x in all_results['held-out'].values()]
        means_abs = [math.fsum(abs(row['effects'][i]) for row in discovery)/8 for i in range(4)]
        signed = [math.fsum(row['effects'][i] for row in discovery)/8 for i in range(4)]
        selection = (means_abs.index(max(means_abs))+1) if max(means_abs) > 0 else None
        magnitude = [math.fsum(row['magnitude'][i] for row in discovery)/8 for i in range(4)]
        mag_selection = magnitude.index(max(magnitude))+1 if max(magnitude) > 0 else None
        votes = [sum(row['first_changed'] == [block,pos] for row in discovery)
                 for block,pos,_ in MAP[layer]]
        first = votes.index(max(votes))+1 if max(votes) else None
        order = sorted(range(4), key=lambda i: (-means_abs[i], i))
        random_q = int.from_bytes(hashlib.sha256(f'public-distractor-repair-v1|{layer}'.encode()).digest()[:8], 'big')%4+1
        means = {key: math.fsum(row[key] for row in held)/8 for key in ('D_combined','D_downstream','D_rotated')}
        denom = means['D_combined']
        summary[layer] = {'mean_absolute_patch_effect': means_abs, 'signed_predictions': signed,
                          'selected_Q': selection, 'magnitude_Q': mag_selection,
                          'magnitude_scores': magnitude, 'first_changed_votes': votes,
                          'first_changed_Q': first, 'ranking': [i+1 for i in order],
                          'max_tie': [i+1 for i in order if means_abs[i] == means_abs[order[0]]],
                          'constant_Q': 3, 'random_Q': random_q,
                          'causal_mask_reference': [i+1 for i,(_,pos,_) in enumerate(MAP[layer]) if pos == 't'],
                          'first_changed_discovery': [row['first_changed'] for row in discovery],
                          'mean_distances': means,
                          'repair_fraction': (1-means['D_downstream']/denom) if denom > 1e-12 else None,
                          'rotated_fraction': (1-means['D_rotated']/denom) if denom > 1e-12 else None,
                          'heldout_prediction_mae': [math.fsum(abs(row['effects'][i]-signed[i]) for row in held)/8 for i in range(4)],
                          'heldout_zero_mae': [math.fsum(abs(row['effects'][i]) for row in held)/8 for i in range(4)]}
    fractions = [summary[layer]['repair_fraction'] for layer in (1,2,3)]
    primary = None if any(x is None for x in fractions) else sum(x > .25 for x in fractions) >= 2
    nearest()
    return {'scope': 'saved-byte numerical and structural checks; not execution or publication attestation',
            'actions_checked': 448, 'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            'phase_results': all_results, 'evidence': evidence, 'summary': summary, 'primary_pattern_supported': primary}


def compare_reports(ours, freeze, produced):
    """Compare the producer's saved reports with the independent calculations."""
    def eq(actual, expected, field):
        if isinstance(expected, float):
            require(type(actual) is float and math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-15), field)
        elif type(expected) is dict:
            require(type(actual) is dict and set(actual) == set(expected), field + ' fields')
            for key in expected: eq(actual[key], expected[key], field + '.' + str(key))
        elif type(expected) is list:
            require(type(actual) is list and len(actual) == len(expected), field + ' length')
            for index, value in enumerate(expected): eq(actual[index], value, field + f'[{index}]')
        else:
            require(type(actual) is type(expected) and actual == expected, field)
    require(freeze['schema'] == 'public-distractor-discovery-freeze-v1', 'freeze schema')
    require(produced['discovery'] == freeze['discovery'], 'frozen discovery preserved')
    eq(freeze['evidence'], ours['evidence']['discovery'], 'discovery evidence')
    eq(produced['heldout_evidence'], ours['evidence']['held-out'], 'held-out evidence')
    held = list(ours['phase_results']['held-out'].items())
    for layer in (1,2,3):
        s = ours['summary'][layer]; actual = freeze['discovery'][str(layer)]
        q = lambda i: f'Q{i}' if i is not None else None
        expected = {'selector': {'selected': q(s['selected_Q']), 'ranking': list(map(q,s['ranking'])),
                      'mean_absolute_effects': s['mean_absolute_patch_effect'], 'signed_predictions': s['signed_predictions'],
                      'max_tie': list(map(q,s['max_tie'])), 'abstained': s['selected_Q'] is None},
                    'first_changed': q(s['first_changed_Q']), 'first_changed_votes': s['first_changed_votes'],
                    'magnitude': q(s['magnitude_Q']), 'magnitude_scores': s['magnitude_scores'],
                    'constant': q(s['constant_Q']), 'random': q(s['random_Q']), 'uniform_site_reference': .25,
                    'causal_mask_reference': list(map(q,s['causal_mask_reference']))}
        eq(actual, expected, f'discovery.{layer}')
        for recorded, computed in zip(actual['selector']['signed_predictions'], s['signed_predictions']):
            require(struct.pack('<d', recorded) == struct.pack('<d', computed), 'exact frozen signed prediction')
        cell = produced['heldout']['cells'][str(layer)]
        eq(cell['selected_source'], None if s['selected_Q'] is None else
           MAP[layer][s['selected_Q']-1][2] == 'source', 'selected source')
        eq(cell['target_effect_count'], sum(row[layer]['D_target'] > 1e-12 for _,row in held), 'target count')
        eq(cell['rotation_reduction'], s['rotated_fraction'], 'rotation aggregate')
        require(len(cell['cases']) == 8, 'held-out case count')
        for actual_case,(cid,rows) in zip(cell['cases'], held):
            case = rows[layer]; den = case['D_combined']
            expected_case = {'case_id': cid, 'target_full_logit_distance': case['D_target'],
                'target_contrast_effect': case['target_margin_effect'], 'combined_distance': den,
                'downstream_distance': case['D_downstream'], 'rotation_distance': case['D_rotated'],
                'downstream_reduction': 1-case['D_downstream']/den if den > 1e-12 else None,
                'rotation_reduction': 1-case['D_rotated']/den if den > 1e-12 else None,
                'decoy_final_distance': case['decoy_final_distance'],
                'combined_final_vs_target': case['combined_final_vs_target'], 'rotation': case['rotation']}
            eq(actual_case, expected_case, f'heldout.{layer}.{cid}')
        for i in range(4):
            candidate = cell['candidates'][f'Q{i+1}']
            expected_candidate = {'prediction': s['signed_predictions'][i],
                'accuracy': {'mae': s['heldout_prediction_mae'][i], 'zero_mae': s['heldout_zero_mae'][i],
                             'improved': s['heldout_prediction_mae'][i] < s['heldout_zero_mae'][i]},
                'cases': [{'case_id': cid, 'effect': rows[layer]['effects'][i],
                           'distance_to_clean': rows[layer]['Q_distances'][i]} for cid,rows in held]}
            eq(candidate, expected_candidate, f'candidate.{layer}.Q{i+1}')
        primary = produced['heldout']['primary']['cells'][layer-1]
        mean = s['mean_distances']; fraction = s['repair_fraction']
        expected_primary = {'mean_combined_distance': mean['D_combined'], 'mean_patched_distance': mean['D_downstream'],
                            'reduction': fraction, 'passes': None if fraction is None else fraction > .25,
                            'per_case_reductions': [1-row[layer]['D_downstream']/row[layer]['D_combined']
                                 if row[layer]['D_combined'] > 1e-12 else None for _,row in held]}
        eq(primary, expected_primary, f'primary.{layer}')
    eq(produced['heldout']['primary']['supported'], ours['primary_pattern_supported'], 'primary decision')


def verify_receipts(root, cases_path, pre_path, freeze_path, result_path):
    pre_bytes = pre_path.read_bytes(); freeze_bytes = freeze_path.read_bytes(); result_bytes = result_path.read_bytes()
    pre, freeze, result = map(parse, (pre_bytes, freeze_bytes, result_bytes))
    digest = lambda value: hashlib.sha256(value).hexdigest()
    pre_sha, freeze_sha, cases_sha = digest(pre_bytes), digest(freeze_bytes), digest(cases_path.read_bytes())
    require(pre['schema'] == 'public-distractor-repair-pre-run-v1'
            and cases_sha == '811a1f0251ff21ddaa1869ee96c80dfa47beb87a6b6c2f1e3b6505728fee1825'
            and pre['case_sha256'] == cases_sha
            and pre['protocol_sha256'] == '6d87a579eb27f214db3e8769232b5940d263da59490480765118174ef99b5d70'
            and pre['reservation_sha256'] == 'f3a500dc02d85b8f004275ba2edd74febf9cec138e83755568e5be31045898a9',
            'pre-run identities')
    receipt_hashes = {}
    for phase in ('discovery','held-out'):
        receipt_path = root/(phase+'.receipt.json'); receipt_bytes = receipt_path.read_bytes()
        receipt = parse(receipt_bytes); receipt_hashes[phase] = digest(receipt_bytes)
        require(receipt['schema'] == 'public-distractor-repair-direct-receipt-v1'
                and receipt['phase'] == phase and receipt['pre_run_sha256'] == pre_sha
                and receipt['reservation_sha256'] == pre['reservation_sha256']
                and receipt['freeze_sha256'] == (freeze_sha if phase == 'held-out' else None)
                and receipt['accepted_complete_phase'] is True and receipt['exit_code'] == 0
                and receipt['interrupted'] is False and receipt['prefix_empty_after'] is True
                and receipt['stage_empty_after'] is True and receipt['failure_file_present'] is False
                and receipt['stderr']['bytes'] == 0 and receipt['source_revision'] == pre['source_revision']
                and receipt['source_tree'] == pre['source_tree'], 'direct-child receipt')
        directory = root/('run-'+phase); files = receipt['run_files']
        require(set(files) == {p.name for p in directory.iterdir()}, 'receipt file inventory')
        for name,item in files.items():
            file = directory/name; raw = file.read_bytes()
            require(not file.is_symlink() and len(raw) == item['bytes'] and digest(raw) == item['sha256'],
                    'receipt file hash')
        for stream in ('stdout','stderr'):
            raw = (root/(phase+'.'+stream+'.txt')).read_bytes()
            require(len(raw) == receipt[stream]['bytes'] and digest(raw) == receipt[stream]['sha256'],
                    'receipt log hash')
    require(freeze['case_sha256'] == cases_sha and freeze['pre_run_sha256'] == pre_sha
            and freeze['reservation_sha256'] == pre['reservation_sha256']
            and freeze['discovery_receipt_sha256'] == receipt_hashes['discovery'], 'freeze binding')
    require(result['schema'] == 'public-distractor-repair-result-v1'
            and result['case_sha256'] == cases_sha and result['pre_run_sha256'] == pre_sha
            and result['discovery_freeze_sha256'] == freeze_sha
            and result['discovery_receipt_sha256'] == receipt_hashes['discovery']
            and result['heldout_receipt_sha256'] == receipt_hashes['held-out'], 'result binding')
    return {'pre_run_sha256':pre_sha, 'freeze_sha256':freeze_sha, 'receipt_sha256':receipt_hashes,
            'result_sha256':digest(result_bytes),
            'scope':'saved cooperative direct-child receipts; not independent process attestation'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--hidden', type=int, default=768)
    parser.add_argument('--vocab', type=int, default=50257)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--phase-prefix', choices=('', 'run-'), default='')
    parser.add_argument('--pre-run', type=Path)
    args = parser.parse_args()
    report = run(args.root, args.cases, args.hidden, args.vocab, args.phase_prefix)
    require((args.freeze is None) == (args.result is None), 'supply freeze and result together')
    if args.freeze:
        compare_reports(report, parse(args.freeze.read_bytes()), parse(args.result.read_bytes()))
        report['producer_report_comparison'] = 'matched independently computed fields'
    if args.pre_run:
        require(args.phase_prefix == 'run-' and args.freeze is not None, 'pre-run requires final result and run- prefix')
        report['binding'] = verify_receipts(args.root, args.cases, args.pre_run, args.freeze, args.result)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    if args.output:
        with args.output.open('xb') as out: out.write(raw)
    print(json.dumps({'actions_checked': report['actions_checked'], 'summary': report['summary'],
                      'primary_pattern_supported': report['primary_pattern_supported']}, sort_keys=True))


if __name__ == '__main__':
    main()
