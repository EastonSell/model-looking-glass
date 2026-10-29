"""Independent, model-free audit of public known-injection saved records.

The original positional interface checks both phases of a fabricated fixture.
The discovery-only interface checks one saved phase and its frozen selections
before a held-out phase exists. This reader does not attest execution or
publication chronology. It imports neither the producer nor its analyzer.
"""
import hashlib
import argparse
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path


SITES = (1, 2, 3, 4)


def fail(message):
    raise ValueError(message)


def close(a, b):
    return a == b or (type(a) is float and type(b) is float
                      and math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def margin(raw, ids):
    return struct.unpack_from('<f', raw, 4 * ids[0])[0] - struct.unpack_from('<f', raw, 4 * ids[1])[0]


def distance(left, right):
    if len(left) != len(right) or len(left) % 4:
        fail('incompatible vectors')
    return math.sqrt(math.fsum((a[0] - b[0]) ** 2 for a, b in
                     zip(struct.iter_unpack('<f', left), struct.iter_unpack('<f', right))))


def expected_actions(cases):
    actions = []
    for case in cases:
        cid = case['case_id']
        actions += [(cid, 'clean', None, None), (cid, 'null-self', None, None)]
        for cell in ('A', 'B'):
            actions.append((cid, 'injected', cell, None))
            actions.extend((cid, 'patch', cell, site) for site in SITES)
            actions.append((cid, 'injected-self', cell, None))
        actions.append((cid, 'clean-repeat', None, None))
    return actions


def expected_plan(kind, cell, candidate):
    if kind == 'null-self':
        return [('before', 'capture', 1, False), ('null', 'null', 1, True),
                ('self_source', 'capture', 1, True), ('self_patch', 'patch', 1, True),
                ('after', 'capture', 1, False)]
    injection = {'A': 1, 'B': 4}[cell]
    result = []
    for site in SITES:
        if site == injection:
            result += [('inject_before', 'capture', site, False), ('inject', 'add', site, True),
                       ('inject_after', 'capture', site, False)]
        if kind == 'patch' and site == candidate:
            result += [('patch_before', 'capture', site, False), ('patch', 'patch', site, True),
                       ('patch_after', 'capture', site, False)]
        if kind == 'injected-self' and site == injection:
            result += [('self_source', 'capture', site, True), ('self_before', 'capture', site, False),
                       ('self_patch', 'patch', site, True), ('self_after', 'capture', site, False)]
        result.append((f'tap_{site}', 'capture', site, True))
    return result


def check_hooks(hooks, action, token_count, hidden):
    _, kind, cell, candidate = action
    if kind in ('clean', 'clean-repeat'):
        if type(hooks) is not list or len(hooks) != 4:
            fail('trace event count')
        for index, site in enumerate(SITES):
            h = hooks[index]
            if (h.get('sequence') != index or h.get('layer_index') != site or h.get('invocation') != 0
                or h.get('module_path') != f'transformer.h.{site}.mlp' or h.get('module_kind') != 'mlp'
                or h.get('forward_index') != 0 or h.get('batch_positions') != [0]
                or h.get('token_positions') != [token_count - 1] or h.get('shape') != [1, token_count, hidden]
                or h.get('saved_shape') != [1, 1, hidden] or h.get('dtype') != 'torch.float32'
                or h.get('generation', {}).get('use_cache') is not False):
                fail('trace metadata')
        return {f'tap_{site}': [hidden] for site in SITES}
    plan = expected_plan(kind, cell, candidate)
    if type(hooks) is not list or len(hooks) != len(plan):
        fail('plan event count')
    shapes = {}
    for index, (name, operation, site, selected) in enumerate(plan):
        h = hooks[index]
        if (h.get('order') != index or h.get('name') != name or h.get('operation') != operation
            or h.get('module_path') != f'transformer.h.{site}.mlp' or h.get('invocation') != 0
            or h.get('forward_index') != 0 or h.get('dtype') != 'torch.float32'
            or h.get('output_shape') != [1, token_count, hidden]
            or h.get('batch_index') != (0 if selected else None)
            or h.get('token_index') != (token_count - 1 if selected else None)
            or h.get('selected_shape') != ([hidden] if selected else [1, token_count, hidden])):
            fail('plan metadata or ordering')
        if name == 'self_patch' and h.get('source_capture') != 'self_source':
            fail('self-patch source')
        if operation == 'capture':
            shapes[name] = [hidden] if selected else [1, token_count, hidden]
    return shapes


def expected_checks(action):
    _, kind, cell, candidate = action
    if kind == 'clean': return {'trace_valid': True}
    if kind == 'clean-repeat': return {'logits_equal_clean': True, 'vectors_equal_clean': True}
    if kind == 'null-self': return {'before_after_equal': True, 'logits_equal_clean': True}
    if kind == 'injected': return {'landing_valid': True}
    if kind == 'injected-self': return {'landing_valid': True, 'logits_equal_injected': True}
    site = {'A': 1, 'B': 4}[cell]
    relation = 'upstream' if candidate < site else 'same' if candidate == site else 'downstream'
    out = {'landing_valid': True, 'relationship': relation}
    if relation == 'upstream':
        out.update(upstream_donor_equal=True, upstream_logits_equal_injected=True)
    if relation == 'same': out['same_site_logits_equal_clean'] = True
    return out


def read_phase(root, cases, hidden, vocab):
    files = {x['case_id']: (root / (x['case_id'] + '.f32')).read_bytes() for x in cases}
    positions = {key: 0 for key in files}
    cases_by_id = {x['case_id']: x for x in cases}
    expected = expected_actions(cases)
    rows = (root / 'events.jsonl').read_bytes().splitlines()
    complete, active, tensors, shapes, descriptors, hooks = {}, None, None, None, None, None
    for line in rows:
        row = json.loads(line)
        kind = row['type']
        if kind == 'start':
            if active is not None or len(complete) >= len(expected): fail('unexpected start')
            detail = row['details']
            active = (row['case_id'], row['action'], detail.get('cell'), detail.get('candidate'))
            if active != expected[len(complete)] or row['ordinal'] != len(complete): fail('schedule')
            if active[2] in ('A', 'B') and detail.get('site') != {'A': 1, 'B': 4}[active[2]]:
                fail('injection site metadata')
            tensors, shapes, descriptors, hooks = {}, {}, [], None
        elif kind == 'tensor':
            if active is None or row['ordinal'] != len(complete) or (row['case_id'], row['action']) != active[:2]:
                fail('orphan tensor')
            item = row['tensor']; cid = active[0]; offset = positions[cid]
            if item['file'] != cid + '.f32' or item['offset'] != offset or item['dtype'] != 'float32-le':
                fail('tensor identity')
            shape, length, name = item['shape'], item['length'], item['name']
            if (type(shape) is not list or not shape or any(type(x) is not int or x < 1 for x in shape)
                or type(length) is not int or length != 4 * math.prod(shape) or name in tensors):
                fail('tensor shape')
            payload = files[cid][offset:offset + length]
            if len(payload) != length or sha(payload) != item['sha256']:
                fail('tensor bytes')
            if any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f', payload)):
                fail('nonfinite tensor')
            tensors[name], shapes[name] = payload, shape
            descriptors.append(item); positions[cid] += length
        elif kind == 'raw-recorded':
            if (active is None or hooks is not None or row['ordinal'] != len(complete)
                or (row['case_id'], row['action']) != active[:2] or row['tensors'] != descriptors):
                fail('recorded metadata')
            hooks = row['hook_events']
        elif kind == 'complete':
            if (active is None or hooks is None or row['ordinal'] != len(complete)
                or (row['case_id'], row['action']) != active[:2] or row['checks'] != expected_checks(active)):
                fail('completion or control check')
            for key, value in expected_checks(active).items():
                if type(row['checks'][key]) is not type(value): fail('control type')
            expected_shape = {'logits': [vocab]}
            expected_shape.update(check_hooks(hooks, active, len(cases_by_id[active[0]]['token_ids']), hidden))
            if shapes != expected_shape: fail('saved tensor inventory')
            complete[active] = tensors; active = None
        else:
            fail('unexpected journal event')
    if active is not None or len(complete) != len(expected) or any(positions[k] != len(v) for k, v in files.items()):
        fail('incomplete phase or unaccounted bytes')
    return complete


def get(data, cid, kind, cell=None, candidate=None):
    return data[(cid, kind, cell, candidate)]


def audit_controls(data, cases, hidden):
    for case in cases:
        cid = case['case_id']; offset = (len(case['token_ids']) - 1) * hidden * 4
        clean = get(data, cid, 'clean'); repeat = get(data, cid, 'clean-repeat'); null = get(data, cid, 'null-self')
        if clean['logits'] != repeat['logits'] or clean['logits'] != null['logits'] or null['before'] != null['after']:
            fail('clean/null/repeat')
        if any(clean[f'tap_{s}'] != repeat[f'tap_{s}'] for s in SITES): fail('repeat vectors')
        for cell, site in (('A', 1), ('B', 4)):
            injected = get(data, cid, 'injected', cell)
            for kind, candidate in [('injected', None)] + [('patch', s) for s in SITES] + [('injected-self', None)]:
                item = get(data, cid, kind, cell, candidate)
                before, after = item['inject_before'], item['inject_after']
                intended = struct.pack('<f', struct.unpack_from('<f', before, offset)[0] + 4.0)
                want = bytearray(before); want[offset:offset + 4] = intended
                if bytes(want) != after or intended == before[offset:offset + 4]: fail('injection landing')
                if kind == 'injected' and item[f'tap_{site}'] != after[offset:offset + hidden * 4]:
                    fail('post-add tap')
                if kind == 'patch':
                    donor = clean[f'tap_{candidate}']; want = bytearray(item['patch_before'])
                    want[offset:offset + hidden * 4] = donor
                    if bytes(want) != item['patch_after']: fail('patch landing')
                    if candidate < site and (item['patch_before'][offset:offset + hidden * 4] != donor
                                             or item['logits'] != injected['logits']): fail('upstream control')
                    if candidate == site and item['logits'] != clean['logits']: fail('same-site control')
                if kind == 'injected-self' and (item['self_before'] != item['self_after']
                                                 or item['logits'] != injected['logits']): fail('self control')


def audit_science(discovery, held, cases, frozen, result, ids):
    for cell in ('A', 'B'):
        reductions = [[] for _ in SITES]; signed = [[] for _ in SITES]
        magnitudes = [[] for _ in SITES]; votes = Counter()
        for case in cases[:8]:
            cid = case['case_id']; clean = get(discovery, cid, 'clean'); inj = get(discovery, cid, 'injected', cell)
            mc, mi = margin(clean['logits'], ids), margin(inj['logits'], ids)
            changed = [s for s in SITES if clean[f'tap_{s}'] != inj[f'tap_{s}']]
            if changed: votes[min(changed)] += 1
            for s in SITES:
                patched = margin(get(discovery, cid, 'patch', cell, s)['logits'], ids)
                reductions[s - 1].append(abs(mi - mc) - abs(patched - mc)); signed[s - 1].append(patched - mi)
                magnitudes[s - 1].append(distance(clean[f'tap_{s}'], inj[f'tap_{s}']))
        scores = [math.fsum(x) / 8 for x in reductions]; preds = [math.fsum(x) / 8 for x in signed]
        mags = [math.fsum(x) / 8 for x in magnitudes]
        rank = sorted(SITES, key=lambda s: (-scores[s - 1], s)); best = scores[rank[0] - 1]
        selected = rank[0] if best > 0 else None
        mag = sorted(SITES, key=lambda s: (-mags[s - 1], s))[0]; mag = mag if mags[mag - 1] > 0 else None
        first = min(s for s in SITES if votes[s] == max(votes.values())) if votes else None
        random = 1 + hashlib.sha256(b'public-known-injection-v1-random\0' + cell.encode('ascii')).digest()[0] % 4
        f, selection = frozen[cell], frozen[cell]['selector']
        if (selection['scores'] != scores or selection['signed_predictions'] != preds or selection['ranking'] != rank
            or selection['selected_site'] != selected or selection['tied_max_sites'] != [s for s in SITES if scores[s-1] == best]
            or f['magnitude_scores'] != mags or f['magnitude_site'] != mag or f['first_changed_site'] != first
            or f['first_changed_votes'] != [votes[s] for s in SITES] or f['constant_site'] != 1
            or f['random_site'] != random or f['uniform_site_reference'] != 0.25): fail('discovery computation')
        if held is None:
            continue
        choices = {'selector': selected, 'first_changed': first, 'magnitude': mag, 'constant': 1, 'random': random}
        if result[cell]['choices'] != choices: fail('frozen choices changed')
        for s in SITES:
            observed = result[cell]['by_site'][str(s)]; effects=[]; errors=[]; zero=[]; di=[]; dp=[]; ratios=[]
            saved_cases = observed['cases']
            if type(saved_cases) is not list or len(saved_cases) != len(cases[8:]):
                fail('held-out case count')
            for case, saved in zip(cases[8:], saved_cases):
                cid=case['case_id']; clean=get(held,cid,'clean')['logits']; inj=get(held,cid,'injected',cell)['logits']
                patched=get(held,cid,'patch',cell,s)['logits']; effect=margin(patched,ids)-margin(inj,ids)
                a,b=distance(inj,clean),distance(patched,clean)
                effects.append(effect);errors.append(abs(preds[s-1]-effect));zero.append(abs(effect))
                di.append(a);dp.append(b);ratios.append(b/a if a>1e-12 else None)
                expected = {'actual_effect': effect, 'prediction': preds[s-1],
                            'absolute_error': errors[-1], 'zero_absolute_error': zero[-1],
                            'distance_injected': a, 'distance_patch': b}
                if (type(saved) is not dict or set(saved) != set(expected) | {'case_id', 'distance_ratio'}
                    or saved['case_id'] != cid or any(type(saved[k]) is not float or not close(saved[k], v)
                                                       for k, v in expected.items())
                    or (ratios[-1] is None and saved['distance_ratio'] is not None)
                    or (ratios[-1] is not None and (type(saved['distance_ratio']) is not float
                                                   or not close(saved['distance_ratio'], ratios[-1])))):
                    fail('held-out per-case computation')
            mae,zmae,avgi,avgp=(math.fsum(x)/8 for x in (errors,zero,di,dp))
            if (not close(observed['prediction_mae'],mae) or not close(observed['zero_mae'],zmae)
                or type(observed['prediction_success']) is not bool
                or observed['prediction_success'] != (mae<zmae) or not close(observed['mean_distance_injected'],avgi)
                or not close(observed['mean_distance_patch'],avgp)
                or (observed['restoration_success'] is not None and type(observed['restoration_success']) is not bool)
                or observed['restoration_success'] != (None if avgi<=1e-12 else avgp<avgi)
                or type(observed['undefined_ratio_count']) is not int
                or observed['undefined_ratio_count'] != sum(x is None for x in ratios)
                or [x['actual_effect'] for x in observed['cases']] != effects
                or [x['distance_ratio'] for x in observed['cases']] != ratios): fail('held-out computation')


def evidence_inventory(root, cases):
    """Derive the freeze's phase inventory directly from the retained files."""
    root = Path(root)
    raw = {}
    for case in cases:
        cid = case['case_id']
        digest = hashlib.sha256()
        length = 0
        with (root / (cid + '.f32')).open('rb') as stream:
            while True:
                part = stream.read(1024 * 1024)
                if not part:
                    break
                digest.update(part)
                length += len(part)
        raw[cid] = {'sha256': digest.hexdigest(), 'bytes': length}
    return {'events_sha256': sha((root / 'events.jsonl').read_bytes()),
            'actions': len(expected_actions(cases)), 'raw_files': raw,
            'raw_bytes': sum(item['bytes'] for item in raw.values())}


def discovery_only(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--hidden', type=int, default=768)
    parser.add_argument('--vocab', type=int, default=50257)
    parser.add_argument('--score-ids', type=int, nargs=2, default=(318, 389))
    parser.add_argument('--expected-case-sha256')
    parser.add_argument('--expected-freeze-sha256')
    parser.add_argument('--expected-pre-run-sha256')
    parser.add_argument('--expected-reservation-sha256')
    args = parser.parse_args(argv)
    if args.hidden < 1 or args.vocab < 1 or any(i < 0 or i >= args.vocab for i in args.score_ids):
        fail('invalid model dimensions or score IDs')
    case_bytes = args.cases.read_bytes()
    if args.expected_case_sha256 and sha(case_bytes) != args.expected_case_sha256:
        fail('independently pinned case digest')
    data = json.loads(case_bytes)
    cases = data['cases'] if type(data) is dict else data
    if len(cases) != 16 or any(x['split'] != ('discovery' if i < 8 else 'held-out')
                               for i, x in enumerate(cases)):
        fail('case split')
    freeze_bytes = args.freeze.read_bytes()
    if args.expected_freeze_sha256 and sha(freeze_bytes) != args.expected_freeze_sha256:
        fail('independently pinned freeze digest')
    frozen = json.loads(freeze_bytes)
    inventory = evidence_inventory(args.discovery, cases[:8])
    summary = json.loads((args.discovery / 'phase-summary.json').read_bytes())
    if ((args.discovery / 'failure.json').exists() or (args.discovery / 'partial-return.f32').exists()
        or summary.get('phase') != 'discovery' or summary.get('status') != 'complete'
        or summary.get('cases') != 8 or summary.get('completed_actions') != 120
        or summary.get('raw_bytes') != inventory['raw_bytes']
        or summary.get('metadata_bytes') != (args.discovery / 'events.jsonl').stat().st_size):
        fail('discovery summary or failure evidence')
    if frozen.get('schema') == 'public-known-injection-discovery-freeze-v1':
        if frozen.get('case_sha256') != sha(case_bytes):
            fail('freeze case digest')
        if args.expected_pre_run_sha256 and frozen.get('pre_run_sha256') != args.expected_pre_run_sha256:
            fail('independently pinned pre-run digest')
        if args.expected_reservation_sha256 and frozen.get('reservation_sha256') != args.expected_reservation_sha256:
            fail('independently pinned reservation digest')
        if frozen.get('evidence') != inventory:
            fail('freeze evidence inventory')
        frozen = frozen['discovery']
    elif args.expected_pre_run_sha256 or args.expected_reservation_sha256:
        fail('expected a public discovery freeze')
    phase = read_phase(args.discovery, cases[:8], args.hidden, args.vocab)
    audit_controls(phase, cases[:8], args.hidden)
    audit_science(phase, None, cases, frozen, None, tuple(args.score_ids))
    print(json.dumps({'status': 'passed', 'scope': 'saved discovery bytes and arithmetic only; '
                      'no execution or publication attestation', 'actions': len(phase),
                      'cases': 8, 'hidden': args.hidden, 'vocab': args.vocab,
                      'case_sha256': sha(case_bytes), 'freeze_sha256': sha(freeze_bytes)}, sort_keys=True))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--discovery-only':
        discovery_only(sys.argv[2:])
        return
    if len(sys.argv) != 6: fail('usage: root hidden vocab score-id-1 score-id-2')
    root=Path(sys.argv[1]); hidden,vocab,one,two=map(int,sys.argv[2:])
    data=json.loads((root/'cases.json').read_bytes()); cases=data['cases'] if type(data) is dict else data
    frozen=json.loads((root/'fake-discovery-freeze.json').read_bytes())
    result=json.loads((root/'fake-result.json').read_bytes())
    if len(cases)!=16 or any(x['split'] != ('discovery' if i<8 else 'held-out') for i,x in enumerate(cases)):
        fail('case split')
    if result['discovery']!=frozen: fail('result differs from freeze')
    discovery=read_phase(root/'discovery',cases[:8],hidden,vocab)
    held=read_phase(root/'held-out',cases[8:],hidden,vocab)
    audit_controls(discovery,cases[:8],hidden);audit_controls(held,cases[8:],hidden)
    audit_science(discovery,held,cases,frozen,result['held-out'],(one,two))
    print(json.dumps({'status':'passed','scope':'saved bytes and arithmetic only; no execution or publication attestation',
                      'actions':len(discovery)+len(held),'cases':16,'hidden':hidden,'vocab':vocab},sort_keys=True))


if __name__ == '__main__':
    main()
